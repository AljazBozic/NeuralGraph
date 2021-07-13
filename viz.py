import sys,os

import numpy as np
import torch
import cv2 as cv
import matplotlib.pyplot as plt
import shutil
import copy
import time
import glob
from skimage import io
import json, codecs
from timeit import default_timer as timer
import math
from tqdm import tqdm
import argparse
import math

import marching_cubes as mcubes
from dataset.dataset import MeshDataset
import config as cfg
from utils.pcd_utils import (BBox,
                            transform_pointcloud_to_opengl_coords,
                            load_and_transform_mesh_to_unit_cube,
                            rotate_around_axis,
                            origin, normalize_transformation)
from utils.viz_utils import (visualize_grid, visualize_mesh, Colors, merge_line_sets, merge_meshes)
from utils.sdf_utils import (sample_grid_points, scale_grid)
from utils.line_mesh import LineMesh
from nnutils.node_proc import convert_embedding_to_explicit_params, compute_inverse_occupancy, sample_rbf_surface, sample_rbf_weights
from utils.parser_utils import check_non_negative, check_positive
from nnutils.geometry import augment_grid

import open3d as o3d

from node_sampler.model import NodeSampler
from multi_sdf.model import MultiSDF



class Viewer:
    def __init__(self, checkpoint_path, time_inc=1, gt_data_dir=None, \
                grid_dim=128, grid_num_chunks=256, num_neighbors=1, edge_weight_threshold=0.0, viz_only_graph=False, viz_dense_tracking=True):
        self.time = 0
        self.time_inc = time_inc
        self.obj_mesh = None
        self.gt_mesh = None
        self.sphere_mesh = None
        self.edge_mesh = None
        self.show_gt = False
        self.show_spheres = False
        self.grid_dim = grid_dim
        self.grid_num_chunks = grid_num_chunks
        self.num_neighbors = num_neighbors
        self.viz_edges = num_neighbors > 0
        self.edge_weight_threshold = edge_weight_threshold
        self.viz_only_graph = viz_only_graph
        self.viz_dense_tracking = viz_dense_tracking

        self.initialize(checkpoint_path, gt_data_dir)

    def initialize(self, checkpoint_path, gt_data_dir):
        ###############################################################################################
        # Paths.
        ###############################################################################################
        gt_data_paths = [os.path.join(gt_data_dir, gt_mesh) for gt_mesh in sorted(os.listdir(gt_data_dir)) if os.path.isdir(os.path.join(gt_data_dir, gt_mesh))]

        num_gt_meshes = len(gt_data_paths)
        num_time_steps = math.ceil(float(num_gt_meshes) / self.time_inc)

        if num_time_steps > 1:
            time_steps = np.linspace(-1.0, 1.0, num_time_steps).tolist()
        else:
            time_steps = [0.0]

        time_idxs = []
        for t in range(len(time_steps)):
            time_idxs.append(t * self.time_inc)
            
        print("Time steps:", time_steps)
        print("Time idxs:", time_idxs)

        ###############################################################################################
        # Load model.
        ###############################################################################################
        assert os.path.isfile(checkpoint_path), "\nModel {} does not exist. Please train a model from scratch or specify a valid path to a model.".format(model_path)
        pretrained_dict = torch.load(checkpoint_path)

        # Check if the provided checkpoint is graph-only model, or complete multi-sdf model.
        only_graph_model = True
        for k in pretrained_dict:
            if "node_sampler" in k:
                only_graph_model = False
                break

        if only_graph_model:
            model = NodeSampler().cuda()
        else:
            model = MultiSDF().cuda()

        # Initialize weights from checkpoint.
        model.load_state_dict(pretrained_dict)

        # Put model into evaluation mode.
        model.eval()

        # Count parameters.
        n_all_model_params = int(sum([np.prod(p.size()) for p in model.parameters()]))
        n_trainable_model_params = int(sum([np.prod(p.size()) for p in filter(lambda p: p.requires_grad, model.parameters())]))
        print("Number of parameters: {0} / {1}".format(n_trainable_model_params, n_all_model_params))
        print()

        if only_graph_model:
            node_sampler = model
        else:
            node_sampler = model.node_sampler

        ###############################################################################################
        # Load groundtruth data (meshes and bounding boxes).
        ###############################################################################################
        print("Loading groundtruth data ...")

        self.gt_meshes = []
        self.grid_array = []
        self.rotated2gaps_array = []
        self.world2grid_array = []

        for time_idx in tqdm(time_idxs):
            gt_data_path = gt_data_paths[time_idx]

            # Load groundtruth mesh.
            orig2world = np.reshape(np.loadtxt(f'{gt_data_path}/orig_to_gaps.txt'), [4, 4]) 
            world2orig = np.linalg.inv(orig2world)

            gt_mesh = o3d.io.read_triangle_mesh(f'{gt_data_path}/mesh_orig.ply')
            gt_mesh.transform(orig2world)
            gt_mesh.compute_vertex_normals()
            gt_mesh.paint_uniform_color([0.5, 0.0, 0.0])

            self.gt_meshes.append(gt_mesh)

            # Load data from directory.
            grid, world2grid = MeshDataset.load_grid(f'{gt_data_path}/coarse_grid.grd')
            
            # Store loaded data.
            self.rotated2gaps_array.append(np.eye(4).astype(np.float32))
            self.world2grid_array.append(world2grid)
            self.grid_array.append(grid)

        ###############################################################################################
        # Evaluate node sampler.
        ###############################################################################################
        embeddings = []
        affinity_matrix_array = []

        print("Predicting embedding vectors and descriptors ...")
        
        for i in tqdm(range(len(self.grid_array))):
            t = time_idxs[i]
            with torch.no_grad():
                # Move to device
                grid = torch.from_numpy(self.grid_array[i]).cuda().unsqueeze(0)
                rotated2gaps = torch.from_numpy(self.rotated2gaps_array[i]).cuda().unsqueeze(0)
                world2grid = torch.from_numpy(self.world2grid_array[i]).cuda().unsqueeze(0)

                # Compute augmented sdfs.
                sdfs = augment_grid(grid, world2grid, rotated2gaps)
                
                # Forward pass.
                embedding_pred, source_idxs, target_idxs, pair_distances, pair_weights, affinity_matrix = node_sampler(sdfs)
                embeddings.append(embedding_pred)

                affinity_matrix_union = torch.sum(affinity_matrix, dim=0) / float(cfg.num_neighbors)
                affinity_matrix_array.append(affinity_matrix_union)

                # Compute explicit parameters.
                constants, scales, rotations, centers = convert_embedding_to_explicit_params(embeddings[i], rotated2gaps, cfg.num_nodes, cfg.scaling_type)

                # Transform centers to grid cs.
                centers = centers.view(cfg.num_nodes, 3, 1)
                A_world2grid = world2grid[:, :3, :3].view(1, 3, 3).expand(cfg.num_nodes, -1, -1)
                t_world2grid = world2grid[:, :3, 3].view(1, 3, 1).expand(cfg.num_nodes, -1, -1)
                
        ###############################################################################################
        # Generate node spheres.
        ###############################################################################################
        print("Generating node spheres ...")

        constant_threshold = 0.0 #-0.07
        
        const_colors = np.asarray([
            [3.50792014e-01, 6.09477877e-01, 9.18623692e-02],
            [2.79068176e-01, 5.94389778e-01, 5.58948203e-01],
            [6.27522062e-01, 5.70644842e-01, 1.72806306e-02],
            [8.92691048e-01, 3.53497705e-01, 8.90946981e-01],
            [6.99646307e-01, 2.81853718e-01, 2.42462125e-01],
            [6.79624789e-01, 3.80735752e-01, 4.75500547e-01],
            [5.46483570e-01, 7.43261411e-01, 6.90361556e-01],
            [7.14046942e-01, 6.17673144e-01, 5.99036692e-02],
            [3.25295745e-01, 7.84217895e-01, 5.91601475e-01],
            [1.08086754e-01, 9.52495735e-01, 1.87504136e-01],
            [6.24282339e-01, 7.60901968e-01, 8.81341701e-01],
            [3.63955128e-01, 2.88742423e-02, 5.64829338e-01],
            [2.02798925e-02, 6.80357014e-01, 3.85774930e-01],
            [6.58386583e-02, 9.76293689e-01, 5.88616401e-01],
            [5.38961060e-01, 9.79709667e-01, 1.66701485e-01],
            [1.81211371e-01, 4.21112349e-01, 3.78951883e-01],
            [6.86638449e-01, 2.33668792e-01, 1.97007038e-01],
            [4.57075650e-02, 7.58674440e-01, 7.37527133e-01],
            [6.43171711e-02, 8.88758551e-01, 6.16603030e-01],
            [9.31488993e-01, 3.04792580e-01, 1.17191193e-01],
            [3.33673947e-01, 8.98382130e-02, 8.76965764e-01],
            [8.49970634e-01, 8.12028903e-01, 5.40814056e-01],
            [3.50971791e-01, 9.68156604e-01, 1.59864796e-01],
            [8.32253211e-01, 9.73765873e-01, 9.70596322e-01],
            [1.27919313e-01, 6.88889382e-01, 5.76930563e-02],
            [7.79185181e-01, 6.93173726e-01, 9.56890943e-01],
            [2.81626499e-01, 9.74879032e-01, 3.58350755e-01],
            [9.93950080e-01, 1.23948422e-01, 2.51924664e-01],
            [5.27512858e-01, 7.07643230e-01, 6.27539915e-01],
            [4.97324086e-01, 2.61647738e-01, 1.36418717e-01],
            [8.11152291e-02, 9.17237087e-01, 5.18565854e-01],
            [6.73714658e-01, 3.49781380e-01, 5.53389767e-01],
            [9.92012783e-01, 8.39381388e-01, 1.01922416e-01],
            [4.84007631e-01, 8.48385618e-01, 6.04498506e-02],
            [1.54256087e-01, 9.55727215e-01, 5.90924427e-01],
            [9.54361060e-01, 5.79361344e-01, 2.33875403e-01],
            [1.60999869e-01, 8.00560055e-01, 3.03767333e-01],
            [2.70643815e-01, 4.03547230e-01, 6.21161496e-01],
            [3.82533844e-02, 6.37586825e-01, 5.64930127e-01],
            [2.06585934e-01, 2.63728265e-02, 7.17424207e-01],
            [6.48253350e-01, 8.95915454e-01, 1.52278293e-01],
            [6.23859768e-01, 1.05515979e-01, 2.01621219e-01],
            [1.37128531e-01, 2.69236500e-02, 8.62754467e-02],
            [3.49167738e-01, 8.53994404e-01, 5.27581943e-01],
            [1.39917765e-01, 1.53769276e-01, 7.63822042e-01],
            [6.95345004e-01, 4.08147872e-01, 3.28909925e-02],
            [4.16815963e-01, 8.21005129e-01, 5.50930167e-01],
            [6.84387747e-01, 8.08732107e-01, 5.44412880e-01],
            [8.44283751e-01, 9.93672095e-01, 4.62634098e-01],
            [3.18664582e-01, 3.31959103e-01, 4.47889995e-01],
            [7.44297062e-01, 8.38381449e-01, 2.12874971e-01],
            [5.88025993e-01, 4.05135756e-01, 1.54364116e-01],
            [1.75223301e-02, 5.24205472e-01, 2.77206305e-01],
            [3.93847656e-01, 5.01281715e-01, 5.57155760e-01],
            [7.16408592e-01, 7.01498740e-01, 2.51220176e-01],
            [8.54081063e-01, 8.13440402e-01, 2.22408542e-02],
            [2.61096323e-01, 2.93266567e-01, 5.06796141e-01],
            [6.42760896e-01, 4.98096300e-01, 3.01218380e-01],
            [4.25094097e-01, 8.26193033e-01, 2.90128837e-01],
            [1.36329017e-01, 6.38528704e-01, 5.08623776e-01],
            [8.48359573e-01, 4.29848371e-01, 3.46253741e-01],
            [7.03699847e-01, 6.64829287e-01, 6.05347704e-01],
            [7.25077479e-02, 9.61059635e-01, 4.05956193e-01],
            [6.77370262e-01, 3.68489497e-01, 2.84454499e-01],
            [7.88994524e-01, 3.35815684e-02, 9.75602360e-01],
            [8.56994586e-01, 4.16028506e-01, 1.91315948e-02],
            [2.05596301e-01, 4.87622110e-01, 7.74492500e-01],
            [2.58930304e-02, 2.68993674e-01, 4.68945841e-01],
            [7.98756373e-01, 2.55635793e-01, 2.73059274e-01],
            [4.95439935e-01, 1.18126704e-01, 6.84599864e-01],
            [2.41817636e-01, 3.48070370e-02, 9.24605009e-01],
            [8.00118629e-01, 4.05223193e-01, 8.03749315e-01],
            [9.38699061e-01, 4.51710260e-01, 2.97656241e-02],
            [1.30773198e-01, 2.88007156e-01, 2.33667793e-01],
            [8.10373480e-01, 9.57549650e-01, 6.31906778e-01],
            [7.50106754e-01, 1.58937016e-01, 7.55600873e-01],
            [4.63147410e-01, 4.26552025e-01, 2.19361689e-01],
            [4.67242693e-02, 9.82408668e-01, 3.82955602e-02],
            [8.87171391e-02, 9.43413398e-01, 3.46403080e-01],
            [3.75409348e-01, 4.44265040e-01, 3.28233114e-01],
            [7.03871883e-01, 7.33258655e-01, 1.68267124e-02],
            [1.08410317e-01, 8.50328327e-01, 3.53926247e-02],
            [3.72468303e-01, 2.74178511e-01, 1.35933711e-02],
            [5.31439095e-01, 6.80433855e-01, 2.86019526e-01],
            [5.00900365e-01, 8.15920185e-01, 7.51165436e-01],
            [5.58376907e-01, 2.57388920e-01, 2.45789841e-01],
            [8.86619715e-01, 9.97335246e-01, 7.44252708e-01],
            [2.91643858e-01, 8.94111569e-01, 3.94427853e-01],
            [7.43123944e-01, 8.52062055e-01, 1.73246815e-02],
            [6.12157990e-01, 9.56356491e-02, 8.76692366e-01],
            [3.15755191e-01, 3.83060006e-02, 9.97474535e-01],
            [5.45427319e-01, 9.03467828e-01, 1.29310988e-01],
            [8.50168595e-01, 2.58222006e-01, 6.27015683e-01],
            [2.50854605e-01, 5.75063379e-01, 9.67714830e-01],
            [8.54280280e-01, 7.01158319e-01, 8.40013634e-02],
            [3.32071891e-01, 7.11038791e-01, 7.65968008e-01],
            [6.45127968e-01, 4.97076984e-01, 1.06592161e-01],
            [3.82236386e-01, 5.95559497e-02, 8.87104325e-01],
            [7.55241975e-01, 3.92046556e-01, 9.47614393e-01],
            [4.67903252e-01, 3.45381922e-01, 4.37716856e-01],
            [3.98241433e-01, 9.06286855e-01, 7.43686041e-01],
            [7.66690071e-01, 6.23392204e-01, 1.27189513e-02],
            [8.96998241e-01, 4.81304566e-01, 3.57954898e-01],
            [4.82691058e-01, 9.58656561e-01, 9.50728725e-01],
            [3.99090382e-01, 5.76457999e-01, 2.03014409e-01],
            [8.10361392e-01, 5.54272856e-01, 6.38357715e-01],
            [7.62928305e-01, 1.67030077e-01, 1.53994063e-01],
            [8.65736567e-01, 3.04297391e-02, 3.27750958e-01],
            [2.72967497e-02, 7.33548593e-01, 8.48753415e-01],
            [9.73147198e-01, 3.21626275e-01, 4.38045373e-01],
            [2.61606092e-01, 8.24933115e-01, 6.86654126e-01],
            [3.71328733e-01, 8.32764260e-01, 4.23040496e-01],
            [2.87341959e-01, 1.03533846e-01, 4.66692702e-01],
            [4.88638350e-02, 3.32529754e-01, 9.14546413e-01],
            [2.21447923e-01, 8.60659943e-02, 2.82257585e-01],
            [6.47381800e-02, 2.45633569e-01, 2.96419413e-02],
            [8.73050009e-01, 3.95819997e-01, 9.22366273e-01],
            [4.86760231e-01, 6.56189668e-01, 2.20876896e-01],
            [2.71018689e-01, 6.40972134e-01, 2.83961484e-01],
            [5.24937026e-02, 6.59090930e-01, 2.29470319e-01],
            [7.87635190e-01, 3.70769598e-01, 7.00663168e-01],
            [2.38445190e-01, 9.29091995e-01, 4.24473727e-01],
            [3.26522853e-01, 4.87460288e-01, 3.06745724e-01],
            [4.72089251e-01, 1.90689039e-02, 1.79343609e-01],
            [9.89288945e-01, 6.60907642e-01, 3.29235147e-01],
            [9.50884852e-01, 3.27848297e-01, 1.84433740e-01],
            [8.13695209e-01, 6.91812695e-01, 6.03638784e-01],
            [6.89160606e-01, 8.31642157e-01, 3.38249301e-01],
            [8.08866405e-01, 2.95494442e-01, 3.99583567e-01],
            [8.34562746e-01, 4.62335081e-01, 2.28252471e-01],
            [1.92407425e-01, 2.48617784e-01, 1.21468809e-01],
            [8.89129736e-01, 1.18470274e-01, 8.25274453e-02],
            [7.82823534e-02, 1.12937426e-01, 1.17604658e-01],
            [6.89284600e-01, 1.94066650e-01, 6.24052771e-01],
            [7.35866607e-01, 3.54281871e-01, 9.10410473e-01],
            [1.32680217e-01, 4.81338149e-01, 8.88847950e-01],
            [3.22362826e-01, 4.64024694e-01, 8.49987306e-01],
            [6.73312431e-02, 5.83563767e-01, 3.07485547e-01],
            [2.27570085e-01, 4.66877826e-01, 3.19679722e-01],
            [9.40000771e-01, 4.88487013e-01, 7.91264995e-01],
            [5.91739070e-01, 7.76597078e-02, 6.97716287e-01],
            [9.09124331e-01, 4.65388401e-01, 5.11858736e-01],
            [2.57003614e-01, 8.15289484e-01, 2.44486922e-01],
            [8.02053890e-01, 5.94931647e-01, 2.19941265e-01],
            [1.85701158e-01, 9.17371775e-01, 5.76441391e-01],
            [3.90116713e-01, 9.07963132e-01, 6.88300491e-01],
            [1.55828357e-01, 3.89036568e-01, 9.09327290e-01],
            [3.21531182e-01, 8.24452812e-01, 8.11864264e-01],
            [1.84324521e-01, 8.00131365e-02, 8.68693676e-01],
            [3.39946390e-01, 8.29309598e-01, 4.96760466e-01],
            [8.73197559e-01, 7.68457615e-01, 1.64133112e-01],
            [2.71687181e-01, 2.77911836e-01, 8.12761280e-01],
            [9.33389880e-01, 5.97116155e-01, 3.70621826e-01],
            [5.35032313e-01, 1.18414675e-01, 2.56625330e-01],
            [9.10976789e-01, 5.50431575e-01, 4.60804479e-01],
            [8.66369532e-01, 2.19963221e-01, 9.22649474e-01],
            [9.31113079e-01, 7.40364787e-01, 4.41741868e-01],
            [2.10669258e-01, 7.20529184e-02, 4.78947719e-01],
            [6.02665361e-01, 3.41906620e-01, 6.40111884e-01],
            [9.81400148e-01, 5.33550075e-01, 3.62542864e-01],
            [4.02233637e-01, 7.68091333e-01, 1.57028661e-01],
            [5.79178159e-01, 7.15662499e-01, 8.07596709e-01],
            [4.99970057e-01, 6.83703604e-01, 5.22246315e-01],
            [6.88272437e-01, 4.79104455e-02, 9.61179491e-01],
            [1.86333802e-01, 3.73489633e-01, 3.54471019e-01],
            [8.72942502e-01, 2.92591862e-01, 9.16145641e-01],
            [8.12290056e-02, 1.90602881e-01, 4.18012265e-02],
            [8.83239031e-01, 1.07011880e-04, 9.98090558e-01],
            [5.80973543e-01, 8.88195686e-01, 3.64240684e-01],
            [1.96649784e-01, 5.55530040e-01, 8.77188085e-01],
            [6.71717670e-01, 1.06156404e-01, 5.45697354e-04],
            [3.23460062e-01, 2.76504984e-01, 9.84278097e-01],
            [6.31777781e-01, 7.93690710e-01, 7.05451964e-01],
            [4.07040187e-01, 9.78708545e-01, 6.06938254e-01],
            [8.66111249e-01, 5.29340657e-01, 1.12528947e-01],
            [8.71645698e-01, 5.03522030e-01, 7.93218080e-01],
            [5.27951984e-02, 8.06657437e-01, 6.83237991e-01],
            [6.67541443e-01, 2.35395120e-01, 5.11600144e-01],
            [6.47444770e-01, 7.29164453e-01, 6.86209582e-01],
            [4.35335233e-01, 1.06468034e-01, 8.01930739e-01],
            [9.09516250e-01, 5.67047548e-01, 3.56997149e-02],
            [3.06913154e-02, 7.39222325e-01, 9.13483204e-01],
            [1.11863623e-02, 7.60723164e-01, 8.50404166e-01],
            [6.20462528e-01, 5.28570847e-01, 8.85034411e-01],
            [6.75866279e-01, 9.21647540e-01, 4.32277832e-01],
            [3.49200686e-01, 5.75550998e-01, 4.51718975e-01],
            [8.68167168e-01, 4.55780719e-01, 8.66757538e-01],
            [2.58722805e-01, 6.66566670e-01, 2.87467169e-01],
            [5.37755153e-01, 6.27057251e-01, 2.19439832e-01],
            [8.99289047e-02, 1.67293873e-01, 6.69520778e-01],
            [4.28156001e-01, 9.44515665e-01, 6.93096826e-01],
            [8.13660832e-01, 9.52835071e-01, 6.23293584e-01],
            [3.46837222e-01, 1.61399248e-01, 1.72654286e-01],
            [3.26977280e-01, 6.64102753e-01, 2.29723294e-01],
            [7.53751557e-01, 6.39692238e-01, 3.46404916e-01],
            [1.22487933e-01, 5.99360275e-01, 5.25031986e-01],
            [5.01663920e-01, 7.53287055e-01, 7.67306891e-01],
            [7.85549286e-01, 9.79270654e-01, 6.12221081e-01],
            [5.03505481e-01, 2.54080662e-02, 5.26944595e-01],
            [1.46355641e-01, 7.24169692e-01, 6.88465058e-01],
            [7.24428744e-01, 2.16337220e-01, 2.53959701e-01],
            [7.78353589e-01, 9.79820243e-01, 7.65592725e-01],
            [8.17313099e-01, 7.42336700e-01, 6.71980515e-01],
            [4.89901133e-01, 3.01452755e-01, 2.52487760e-01],
            [8.42504588e-01, 8.14758696e-02, 2.08627630e-01],
            [4.10913721e-01, 9.36568755e-01, 7.28202494e-01],
            [3.45518339e-01, 3.20965792e-01, 2.55943136e-02],
            [6.72175102e-01, 8.86368708e-01, 2.56445419e-01],
            [5.09249467e-01, 2.31873357e-01, 2.22732974e-01],
            [7.28462758e-01, 4.26688071e-02, 2.93284582e-01],
            [4.70983502e-01, 8.66300317e-02, 4.16890860e-01],
            [4.59127219e-01, 7.90275868e-01, 6.45938070e-01],
            [1.16087607e-01, 8.03820098e-01, 7.37217348e-01],
            [1.64771903e-01, 5.32436143e-01, 5.01306573e-01],
            [1.54581010e-01, 4.65351091e-01, 7.91736641e-01],
            [1.79421844e-01, 9.72654548e-01, 8.97424633e-01],
            [8.88279214e-01, 1.53734507e-01, 5.50361986e-02],
            [3.81791252e-01, 1.06544569e-01, 7.10149831e-02],
            [4.85495980e-01, 3.75825379e-01, 1.59128017e-01],
            [2.41387055e-01, 9.44246872e-02, 2.63485761e-01],
            [6.29200010e-01, 8.29783948e-01, 3.47657069e-01],
            [4.45243502e-01, 8.03300738e-01, 6.24876850e-01],
            [8.75023021e-01, 4.01896536e-01, 9.69829731e-01],
            [6.94298374e-01, 8.70741435e-01, 6.60044493e-01],
            [1.19385065e-01, 3.10806911e-01, 7.48933520e-02],
            [9.88254003e-01, 1.79600290e-01, 6.28242083e-01],
            [9.18924053e-01, 7.86231076e-01, 2.15115754e-01],
            [3.14942988e-01, 3.26422802e-01, 5.45723890e-01],
            [7.71109219e-01, 2.39561496e-01, 5.80650396e-01],
            [1.72781466e-01, 9.15671601e-01, 2.35114992e-01],
            [5.86080178e-01, 6.56983059e-01, 9.29742029e-01],
            [3.51928430e-01, 2.63152335e-01, 4.73863945e-02],
            [5.29427354e-01, 2.10905001e-01, 7.75019248e-01],
            [9.30107340e-01, 7.34533257e-01, 1.79268492e-01],
            [8.17823489e-01, 7.47770660e-01, 4.29203953e-01],
            [4.19373803e-01, 2.24740527e-01, 8.12064594e-01],
            [4.90612027e-02, 9.01289928e-01, 3.13369830e-01],
            [3.64457609e-01, 8.00905784e-01, 8.75230698e-01],
            [9.34454896e-01, 3.39273998e-01, 7.82753532e-02],
            [1.98005771e-01, 9.51814456e-02, 6.38950723e-01],
            [5.91984571e-01, 7.73614876e-01, 5.48393459e-01],
            [5.17929193e-01, 6.82177032e-01, 3.26602163e-01],
            [3.75807023e-01, 6.54474944e-01, 4.03939130e-01],
            [6.31510691e-01, 4.72125619e-01, 5.20578442e-01],
            [1.46768716e-02, 5.31508080e-01, 6.59205205e-01],
            [2.79612391e-01, 4.23656778e-01, 5.40825939e-01],
            [3.11868869e-01, 3.58720242e-01, 4.58388128e-01],
            [4.60507415e-01, 4.66786535e-01, 1.78046518e-01],
            [5.45540658e-02, 7.97725758e-01, 7.71974791e-01],
            [5.86482212e-02, 8.79910934e-01, 1.09915130e-01]
        ])

        self.node_meshes = []
        self.edge_meshes = []
        graph_nodes = []

        for i in tqdm(range(len(embeddings))):
            # Compute explicit parameters.
            rotated2gaps_i = torch.from_numpy(self.rotated2gaps_array[i]).cuda().unsqueeze(0)
            constants, scales, rotations, centers = convert_embedding_to_explicit_params(embeddings[i], rotated2gaps_i, cfg.num_nodes, cfg.scaling_type)

            if affinity_matrix_array[i] is not None:
                affinity_matrix = affinity_matrix_array[i].cpu().detach().numpy()
                self.viz_edges = True
            else:
                self.viz_edges = False

            # Store graph nodes for dense tracking visualization later.
            graph_nodes.append({
                "constants": constants.cpu().numpy(),
                "scales": scales.cpu().numpy(),
                "rotations": rotations.cpu().numpy(),
                "centers": centers.cpu().numpy()
            })

            # Generate sphere meshes.
            sphere_meshes = []
            for node_id in range(cfg.num_nodes):
                constant = constants[0, node_id].cpu().numpy()
                center = centers[0, node_id].cpu().numpy()
                rotation = rotations[0, node_id].cpu().numpy()
                scale = scales[0, node_id].cpu().numpy()

                if constant > constant_threshold:
                    continue

                # Create a sphere mesh.
                mesh_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=1.0, resolution=10)

                scale = scale + 1e-8

                T_t = np.eye(4)
                T_t[0:3, 3] = center

                T_s = np.eye(4)
                T_s[0, 0] = scale[0]
                T_s[1, 1] = scale[1]
                T_s[2, 2] = scale[2]

                T_R = np.eye(4)

                T = np.matmul(T_t, np.matmul(T_R, T_s))
                mesh_sphere.transform(T)

                # We view spheres as wireframe.
                node_sphere = o3d.geometry.LineSet.create_from_triangle_mesh(mesh_sphere)
                node_sphere.paint_uniform_color(const_colors[node_id])
                sphere_meshes.append(node_sphere)

            # Merge sphere meshes.
            merged_spheres = merge_line_sets(sphere_meshes)

            self.node_meshes.append(merged_spheres)

            # Generate edge meshes.
            K = self.num_neighbors
            min_neighbor_weight = self.edge_weight_threshold
            if self.viz_edges:
                points = np.zeros((cfg.num_nodes, 3))
                edge_coords = []
                for node_id in range(cfg.num_nodes):   
                    # Compute nearest K neighbors.
                    max_idxs = np.argpartition(affinity_matrix[node_id], -K)[-K:]

                    for max_idx in max_idxs:
                        max_val = affinity_matrix[node_id, max_idx]

                        source_idx = node_id
                        target_idx = max_idx

                        if math.isfinite(max_val) and max_val >= min_neighbor_weight:
                            edge_coords.append([source_idx, target_idx])

                    # Store center position.
                    center = centers[0, node_id].cpu().numpy()
                    points[node_id] = center

                if len(edge_coords) > 0: 
                    line_mesh = LineMesh(points, edge_coords, radius=0.005)
                    line_meshes = merge_meshes(line_mesh.get_line_meshes())

                    self.edge_meshes.append(line_meshes)
                
                else:
                    self.edge_meshes.append(None)

        ###############################################################################################
        # Generate reconstructed meshes.
        ###############################################################################################
        self.obj_meshes = []

        if not only_graph_model and not self.viz_only_graph:
            print("Sampling SDF values and extracting meshes ...")
            
            dim                 = self.grid_dim
            num_chunks_mlp      = self.grid_num_chunks
            num_chunks_weights  = 128
            grid_size           = 0.7

            influence_threshold = 0.02
            print("Influence threshold: {}".format(influence_threshold))

            # Sample grid points
            points = sample_grid_points(dim, grid_size)
            points = torch.from_numpy(np.transpose(points.reshape(3, -1), (1, 0))).cuda()

            num_points = points.shape[0]
            assert num_points % num_chunks_mlp == 0, "The number of points in the grid must be divisible by the number of chunks"
            points_per_chunk_mlp = int(num_points / num_chunks_mlp)
            print("Num. points per chunk: {}".format(points_per_chunk_mlp))
            
            for t in tqdm(range(len(time_steps))):
                torch.cuda.empty_cache()

                with torch.no_grad():
                    # Move to device
                    grid            = torch.from_numpy(self.grid_array[t]).cuda().unsqueeze(0)
                    rotated2gaps    = torch.from_numpy(self.rotated2gaps_array[t]).cuda().unsqueeze(0)
                    world2grid      = torch.from_numpy(self.world2grid_array[t]).cuda().unsqueeze(0)

                    # Compute augmented inputs.
                    sdfs = augment_grid(grid, world2grid, rotated2gaps)

                    # Predict reconstruction.
                    sdf_pred = np.empty((num_points), dtype=np.float32)

                    for i in range(num_chunks_mlp):
                        points_i = points[i*points_per_chunk_mlp:(i+1)*points_per_chunk_mlp, :].unsqueeze(0)

                        sdf_pred_i = model(points_i, sdfs, rotated2gaps)

                        sdf_pred[i*points_per_chunk_mlp:(i+1)*points_per_chunk_mlp] = sdf_pred_i.cpu().numpy()

                    # Extract mesh with Marching cubes.
                    sdf_pred = sdf_pred.reshape(dim, dim, dim)
                    vertices, triangles = mcubes.marching_cubes(sdf_pred, 0)

                    if vertices.shape[0] > 0 and triangles.shape[0] > 0:
                        # Normalize vertices to be in [-grid_size, grid_size]
                        vertices = 2.0 * grid_size * (vertices / (dim - 1)) - grid_size

                        # Convert extracted surface to o3d mesh.
                        mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(vertices), o3d.utility.Vector3iVector(triangles))
                        mesh.compute_vertex_normals()
                    else:
                        print("No mesh vertices are extracted!")
                        mesh = None #o3d.geometry.TriangleMesh.create_sphere(radius=0.5)

                self.obj_meshes.append(mesh)

        else:
            print("Only graph model was provided, so no reconstruction is executed!")
            print("Graph will be visualized together with ground truth meshes.")

            for gt_mesh in self.gt_meshes:
                self.obj_meshes.append(copy.deepcopy(gt_mesh))

        ###############################################################################################
        # Generate vertex colors.
        ###############################################################################################
        if self.viz_dense_tracking:
            print("Visualizing dense tracking by computing colors in canonical space ...")

            num_chunks = 5

            constants_canonical = torch.from_numpy(graph_nodes[0]["constants"]).cuda()
            scales_canonical = torch.from_numpy(graph_nodes[0]["scales"]).cuda()
            rotations_canonical = torch.from_numpy(graph_nodes[0]["rotations"]).cuda()
            centers_canonical = torch.from_numpy(graph_nodes[0]["centers"]).cuda()
            
            for k in tqdm(range(len(time_steps))):
                mesh_k = copy.deepcopy(self.obj_meshes[k])
                
                vertices_k = np.asarray(mesh_k.vertices)
                faces_k = np.asarray(mesh_k.triangles)

                num_vertices = vertices_k.shape[0]
                max_points_per_chunk = int(math.ceil(float(num_vertices) / num_chunks))

                constants_k = torch.from_numpy(graph_nodes[k]["constants"]).cuda()
                scales_k = torch.from_numpy(graph_nodes[k]["scales"]).cuda()
                rotations_k = torch.from_numpy(graph_nodes[k]["rotations"]).cuda()
                centers_k = torch.from_numpy(graph_nodes[k]["centers"]).cuda()

                # Compute vertices transformed to from canonical to current space.
                transformed_vertices = np.empty((num_vertices, 3), dtype=np.float32)

                with torch.no_grad():
                    # Compute skinning weights.
                    skinning_weights_k = np.empty((num_vertices, cfg.num_nodes), dtype=np.float32)

                    idx_offset = 0
                    for i in range(num_chunks):
                        points_per_chunk = min(max_points_per_chunk, num_vertices - idx_offset)
                        points_i = torch.from_numpy(vertices_k[idx_offset:idx_offset + points_per_chunk, :]).cuda().float()
                        points_i = points_i.view(1, points_per_chunk, 3)
                        
                        # Compute skinning weights.
                        weights_i = sample_rbf_weights(points_i, constants_k, scales_k, centers_k, cfg.use_constants) # (1, num_points, num_nodes)

                        ### WEIGHT NORMALIZATION
                        # We normalize them to sum up to 1.
                        weights_sum = weights_i.sum(dim=2, keepdim=True)
                        weights_i = weights_i.div(weights_sum)

                        skinning_weights_k[idx_offset:idx_offset+points_per_chunk, :] = weights_i.cpu().numpy()

                        idx_offset += points_per_chunk
                        
                    # Use current pose estimates to transform the points to the canonical space.
                    R_canonical = rotations_canonical.view(cfg.num_nodes, 3, 3)
                    R_current = rotations_k.view(cfg.num_nodes, 3, 3)

                    # Compute relative frame-to-frame rotation and translation estimates.
                    t_canonical = centers_canonical
                    t_current = centers_k

                    R_current_inv = R_current.permute(0, 2, 1)
                    R_rel = torch.matmul(R_canonical, R_current_inv)    # (num_nodes, 3, 3)

                    idx_offset = 0
                    for i in range(num_chunks):
                        points_per_chunk = min(max_points_per_chunk, num_vertices - idx_offset)
                        points_i = torch.from_numpy(vertices_k[idx_offset:idx_offset + points_per_chunk, :]).cuda().float()
                        points_i = points_i.view(1, points_per_chunk, 3)
                        
                        # Get corresponding skinning weights.
                        weights_i = torch.from_numpy(skinning_weights_k[idx_offset:idx_offset+points_per_chunk, :]).cuda()

                        # Apply deformation to canonical vertices.
                        t_canonical_exp = t_canonical.view(1, cfg.num_nodes, 3, 1).expand(points_per_chunk, -1, -1, -1)         # (points_per_chunk, num_nodes, 3, 1)
                        t_current_exp = t_current.view(1, cfg.num_nodes, 3, 1).expand(points_per_chunk, -1, -1, -1)             # (points_per_chunk, num_nodes, 3, 1)
                        R_rel_exp = R_rel.view(1, cfg.num_nodes, 3, 3).expand(points_per_chunk, -1, -1, -1)                     # (points_per_chunk, num_nodes, 3, 3)
                        points_i = points_i.view(points_per_chunk, 1, 3, 1).expand(-1, cfg.num_nodes, -1, -1)                   # (points_per_chunk, num_nodes, 3, 1)
                        weights_i = weights_i.view(points_per_chunk, cfg.num_nodes, 1, 1).expand(-1, -1, 3, -1)                 # (points_per_chunk, num_nodes, 3, 1)

                        transformed_points_i = torch.matmul(R_rel_exp, (points_i - t_current_exp)) + t_canonical_exp            # (points_per_chunk, num_nodes, 3, 1)
                        transformed_points_i = torch.sum(weights_i * transformed_points_i, dim=1).view(points_per_chunk, 3) 
            
                        # Store canonical vertex positions.
                        transformed_vertices[idx_offset:idx_offset+points_per_chunk, :] = transformed_points_i.cpu().numpy()

                        idx_offset += points_per_chunk
                
                # Color the vertices depending on their positions in canonical space.
                grid_size = 0.7
                scale = 2.0
                vertex_colors = (scale * transformed_vertices + grid_size) / (2 * grid_size)
                vertex_colors = np.clip(vertex_colors, 0.0, 1.0)
                
                self.obj_meshes[k].vertex_colors = o3d.utility.Vector3dVector(vertex_colors)


    def _update_obj(self, vis):
        param = vis.get_view_control().convert_to_pinhole_camera_parameters()

        prev_obj_mesh = self.obj_mesh
        self.obj_mesh = self.obj_meshes[self.time]

        if self.show_spheres and self.obj_mesh is not None:
            # Convert to wireframe.
            self.obj_mesh = o3d.geometry.LineSet.create_from_triangle_mesh(self.obj_mesh)

        if self.obj_mesh is not None:
            vis.add_geometry(self.obj_mesh)

        if prev_obj_mesh is not None:
            vis.remove_geometry(prev_obj_mesh)

        ctr = vis.get_view_control()
        ctr.convert_from_pinhole_camera_parameters(param)

    def _update_gt(self, vis):
        param = vis.get_view_control().convert_to_pinhole_camera_parameters()

        prev_gt_mesh = self.gt_mesh

        # If requested, we show a (new) mesh.
        if self.show_gt and len(self.gt_meshes) > 0:
            self.gt_mesh = self.gt_meshes[self.time]
            vis.add_geometry(self.gt_mesh)
        else:
            self.gt_mesh = None

        if prev_gt_mesh is not None:
            vis.remove_geometry(prev_gt_mesh)

        ctr = vis.get_view_control()
        ctr.convert_from_pinhole_camera_parameters(param)

    def _update_spheres(self, vis):
        param = vis.get_view_control().convert_to_pinhole_camera_parameters()

        if self.sphere_mesh is not None:
            vis.remove_geometry(self.sphere_mesh)
            self.sphere_mesh = None
        
        if self.edge_mesh is not None:
            vis.remove_geometry(self.edge_mesh)
            self.edge_mesh = None

        if self.show_spheres:
            gt_mesh_idx = self.time

            self.sphere_mesh = self.node_meshes[self.time]
            vis.add_geometry(self.sphere_mesh)

            if self.viz_edges:
                self.edge_mesh = self.edge_meshes[self.time]
                
                if self.edge_mesh is not None: vis.add_geometry(self.edge_mesh)

        ctr = vis.get_view_control()
        ctr.convert_from_pinhole_camera_parameters(param)

    def run(self):
        # Define callbacks.
        def toggle_next(vis):
            # print("::toggle_next")
            self.time += 1
            if self.time >= len(self.obj_meshes):
                self.time = 0

            self._update_obj(vis)
            self._update_gt(vis)
            self._update_spheres(vis)
            
            # print(f"frame: {self.time * self.time_inc}")
            
            return False
        
        def toggle_previous(vis):
            # print("::toggle_previous")
            self.time -= 1
            if self.time < 0:
                self.time = len(self.obj_meshes) - 1

            self._update_obj(vis)
            self._update_gt(vis)
            self._update_spheres(vis)
            
            # print(f"frame: {self.time * self.time_inc}")

            return False

        def toggle_groundtruth(vis):
            # print("::toggle_groundtruth")
            self.show_gt = not self.show_gt

            self._update_gt(vis)

            return False

        def toggle_spheres(vis):
            # print("::toggle_spheres")
            self.show_spheres = not self.show_spheres

            self._update_obj(vis)
            self._update_spheres(vis)

            return False

        key_to_callback = {}
        key_to_callback[ord("D")] = toggle_next
        key_to_callback[ord("A")] = toggle_previous
        key_to_callback[ord("G")] = toggle_groundtruth
        key_to_callback[ord("N")] = toggle_spheres

        # Add mesh at initial time step.
        assert self.time < len(self.obj_meshes)
        self.obj_mesh = self.obj_meshes[self.time]

        if not self.obj_mesh:
            print("Object mesh doesn't exist. Exiting ...")
            exit(1)

        # Print instructions.
        print()
        print("#" * 100)
        print("VISUALIZATION CONTROLS")
        print("#" * 100)
        print()
        print("N: toggle graph nodes and edges")
        print("G: toggle ground truth")
        print("D: show next")
        print("A: show previous")
        print("S: toggle smooth shading")
        print()

        # Run visualization.
        o3d.visualization.draw_geometries_with_key_callbacks([self.obj_mesh], key_to_callback)


if __name__ == "__main__":
    # Parse command line arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_path', '--checkpoint_path', action='store', dest='checkpoint_path')
    parser.add_argument('--time_inc', type=check_non_negative)
    parser.add_argument('--gt_data_dir', action='store', dest='gt_data_dir')
    parser.add_argument('--grid_dim', choices=[32, 64, 128, 256], type=int, help='Grid dimension')
    parser.add_argument('--grid_num_chunks', type=check_positive, help='Number of grid chunks')
    parser.add_argument('--num_neighbors', type=check_non_negative, help='Number of visualized graph neighbors')
    parser.add_argument('--edge_weight_threshold', type=float, help='Graph edge weight threshold')
    parser.add_argument('--viz_only_graph', action='store_true', help='Specify if you want to visualize only graph (much faster)')
    parser.add_argument('--viz_dense_tracking', action='store_true', help='Specify if you want to color reconstructed meshes by estimated dense motion')
 
    args = parser.parse_args()
    
    viewer = Viewer(
        args.checkpoint_path,
        time_inc=args.time_inc, gt_data_dir=args.gt_data_dir, 
        grid_dim=args.grid_dim, grid_num_chunks=args.grid_num_chunks,
        num_neighbors=args.num_neighbors, edge_weight_threshold=args.edge_weight_threshold,
        viz_only_graph=args.viz_only_graph, viz_dense_tracking=args.viz_dense_tracking
    )
    viewer.run()
    