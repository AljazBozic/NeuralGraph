import os
import numpy as np
import torch
import torch.nn as nn
from timeit import default_timer as timer

import config as cfg

from nnutils.mlp import MultiMLP
from node_sampler.model import NodeSampler
from nnutils.node_proc import convert_embedding_to_explicit_params, sample_rbf_weights


class MultiSDF(nn.Module):
    def __init__(self):
        super(MultiSDF, self).__init__()

        ################################################################################################
        # Node sampler
        ################################################################################################
        self.node_sampler = NodeSampler()

        # We freeze the parameters of node sampler.
        for param in self.node_sampler.parameters():
            param.requires_grad = False

        ################################################################################################
        # Local MLPs
        ################################################################################################
        self.surface_mlp = MultiMLP(
            point_dim=3, time_dim=cfg.descriptor_dim, num_groups=cfg.num_nodes, output_dim=1, n_layers=8, n_pos_freq=10, n_time_freq=0, ngf=cfg.num_features
        )
        
        if cfg.freeze_surface_mlp:
            for param in self.surface_mlp.parameters():
                param.requires_grad = False

        ################################################################################################
        # Descriptors
        ################################################################################################
        node_length = cfg.position_length + cfg.scale_length + cfg.rotation_length
        self.descriptor_projector = nn.Conv1d(cfg.num_nodes * node_length, cfg.num_nodes * cfg.descriptor_dim, kernel_size=1)
        
        if cfg.freeze_surface_mlp:
            for param in self.descriptor_projector.parameters():
                param.requires_grad = False

    def forward(self, points, grid, rotated2gaps):
        batch_size = points.shape[0]
        num_points = points.shape[1]

        ################################################################################################
        # Evaluate node sampler.
        ################################################################################################
        embedding, _, _, _, _, _ = self.node_sampler(grid)

        # Convert embedding to rotations and translations.
        constants, scales, rotations, centers = convert_embedding_to_explicit_params(embedding, rotated2gaps, cfg.num_nodes, cfg.scaling_type)

        ################################################################################################
        # Convert points to local coordinate systems.
        ################################################################################################
        points_local = points.view(batch_size, 1, num_points, 3, 1).expand(-1, cfg.num_nodes, -1, -1, -1)           # (bs, num_nodes, num_points, 3, 1)
        centers_exp  = centers.view(batch_size, cfg.num_nodes, 1, 3, 1).expand(-1, -1, num_points, -1, -1)          # (bs, num_nodes, num_points, 3, 1)

        rotations_inv = rotations.view(batch_size, cfg.num_nodes, 3, 3).permute(0, 1, 3, 2)
        rotations_inv = rotations_inv.view(batch_size, cfg.num_nodes, 1, 3, 3).expand(-1, -1, num_points, -1, -1)   # (bs, num_nodes, num_points, 3, 3)

        points_local = torch.matmul(rotations_inv, points_local - centers_exp)                                      # (bs, num_nodes, num_points, 3, 1)
        points_local = points_local.view(batch_size, cfg.num_nodes, num_points, 3).permute(0, 1, 3, 2)              # (bs, num_nodes, 3, num_points)
        
        ################################################################################################
        # Apply MLPs.
        ################################################################################################
        # We use linear layer to project current embedding to the pose.        
        descriptors = self.descriptor_projector(embedding.view(batch_size, -1, 1))
        time_vec = descriptors.view(batch_size, cfg.num_nodes, cfg.descriptor_dim, 1).expand(-1, -1, -1, num_points)
        
        time_vec = time_vec.contiguous()    # (bs, num_nodes, desc_dim, num_points)

        sdfs = self.surface_mlp(points_local, time_vec) 

        if cfg.use_tanh:
            sdfs = torch.tanh(sdfs)
        
        ################################################################################################
        # Compute skinning weights.
        ################################################################################################
        points = points.view(batch_size, num_points, 3)

        weights = sample_rbf_weights(points, constants, scales, centers, cfg.use_constants) # (bs, num_points, num_nodes)

        # Normalize the weights.
        weights_sum = weights.sum(dim=2, keepdim=True)
        weights = weights.div(weights_sum)

        # For points that are far away from any node all weights will be zero, therefore the 
        # division will produce NaN. For now we assume these points move with average pose,
        # so we just set all weights to 1/cfg.num_nodes.
        weights[~torch.isfinite(weights)] = 1.0 / cfg.num_nodes

        weights = weights.permute(0, 2, 1) # (bs, num_nodes, num_points)

        ################################################################################################
        # Execute SDF blending.
        ################################################################################################
        sdf_merged = torch.sum(weights * sdfs, axis=1).view(batch_size, 1, num_points)

        return sdf_merged