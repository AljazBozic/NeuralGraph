import sys,os

import math
import torch
import numpy as np
from skimage import io, transform
from torch.utils.data import Dataset
import json
import open3d as o3d
import pickle
import random
import struct
from timeit import default_timer as timer
from tqdm import tqdm


class MeshDataset(Dataset):
    def __init__(
        self, 
        dataset_dir, num_point_samples,
        cache_data=False, use_augmentation=False
    ):
        self.dataset_dir = dataset_dir
        self.num_point_samples = num_point_samples
        self.cache_data = cache_data
        self.use_augmentation = use_augmentation

        # Load labels.
        self.labels = []
        self._load()

        # Preload data.
        if self.cache_data:
            print("Preloading cached data ...")
            loading_start = timer()

            num_samples = len(self.labels)
            assert num_samples > 0

            data_dir = self.labels[0]["data_dir"]
            sample_data = self._load_data(data_dir)

            self.cache = {
                'uniform_samples':      np.zeros((num_samples,) + sample_data['uniform_samples'].shape, dtype=np.float32),
                'surface_samples':      np.zeros((num_samples,) + sample_data['surface_samples'].shape, dtype=np.float32),
                'near_surface_samples': np.zeros((num_samples,) + sample_data['near_surface_samples'].shape, dtype=np.float32),
                'grid':                 np.zeros((num_samples,) + sample_data['grid'].shape, dtype=np.float32),
                'world2grid':           np.zeros((num_samples,) + sample_data['world2grid'].shape, dtype=np.float32),
                'world2orig':           np.zeros((num_samples,) + sample_data['world2orig'].shape, dtype=np.float32),
                'bbox_lower':           np.zeros((num_samples,) + sample_data['bbox_lower'].shape, dtype=np.float32),
                'bbox_upper':           np.zeros((num_samples,) + sample_data['bbox_upper'].shape, dtype=np.float32)
            }

            for index in tqdm(range(len(self.labels))):
                data_dir = self.labels[index]["data_dir"]
                data = self._load_data(data_dir)

                self.cache['uniform_samples'][index]        = data['uniform_samples']
                self.cache['surface_samples'][index]        = data['surface_samples']
                self.cache['near_surface_samples'][index]   = data['near_surface_samples']
                self.cache['grid'][index]                   = data['grid']
                self.cache['world2grid'][index]             = data['world2grid']
                self.cache['world2orig'][index]             = data['world2orig']
                self.cache['bbox_lower'][index]             = data['bbox_lower']
                self.cache['bbox_upper'][index]             = data['bbox_upper']

            print("Done: {} s".format(timer() - loading_start))
            print()

    def _load(self):
        sample_dirs = sorted([os.path.join(self.dataset_dir, f) for f in os.listdir(self.dataset_dir) if os.path.isdir(os.path.join(self.dataset_dir, f))])

        for sample_dir in sample_dirs:
            self.labels.append({
                "data_dir": sample_dir
            })

    @staticmethod
    def load_pts_file(path):
        _, ext = os.path.splitext(path)
        assert ext in ['.sdf', '.pts']
        l = 4 if ext == '.sdf' else 6
        with open(path, 'rb') as f:
            points = np.fromfile(f, dtype=np.float32)
        points = np.reshape(points, [-1, l])
        return points

    @staticmethod
    def load_grid(path):
        with open(path, 'rb') as f:
            content = f.read()
        res = struct.unpack('iii', content[:4 * 3])
        vcount = res[0] * res[1] * res[2]
        content = content[4 * 3:]
        tx = struct.unpack('f' * 16, content[:4 * 16])
        tx = np.array(tx).reshape([4, 4]).astype(np.float32)
        content = content[4 * 16:]
        grd = struct.unpack('f' * vcount, content[:4 * vcount])
        grd = np.array(grd).reshape(res).astype(np.float32)
        return grd, tx

    def __len__(self):
        return len(self.labels)

    def unpack(self, x):
        # We concatenate the first two dimensions, corresponding to 
        # batch size and sample size.
        n_dims = len(x.shape)
        
        new_shape = (x.shape[0] * x.shape[1], )
        for i in range(2, n_dims):
            new_shape += (x.shape[i],)
        
        return x.view(new_shape)

    def _load_data(self, data_dir):
        # Load data from directory.
        uniform_samples = MeshDataset.load_pts_file(f'{data_dir}/uniform_points.sdf')
        near_surface_samples = MeshDataset.load_pts_file(f'{data_dir}/nss_points.sdf')
        surface_samples = MeshDataset.load_pts_file(f'{data_dir}/surface_points.pts')
        grid, world2grid = MeshDataset.load_grid(f'{data_dir}/coarse_grid.grd')
        orig2world = np.reshape(np.loadtxt(f'{data_dir}/orig_to_gaps.txt'), [4, 4]) 
        world2orig = np.linalg.inv(orig2world)

        # Compute bounding box of interior uniform samples.
        interior_sample_points = uniform_samples[uniform_samples[:, 3] < 0.0, :3]
        bbox_upper = interior_sample_points.max(axis=0)
        bbox_lower = interior_sample_points.min(axis=0)

        return {
            'uniform_samples':          uniform_samples,
            'surface_samples':          surface_samples,
            'near_surface_samples':     near_surface_samples,
            'grid':                     grid,
            'world2grid':               world2grid,
            'world2orig':               world2orig,
            'bbox_lower':               bbox_lower,
            'bbox_upper':               bbox_upper
        }

    def _get_cached_data(self, index):
        return {
            'uniform_samples':          self.cache['uniform_samples'][index],
            'surface_samples':          self.cache['surface_samples'][index],
            'near_surface_samples':     self.cache['near_surface_samples'][index],
            'grid':                     self.cache['grid'][index],
            'world2grid':               self.cache['world2grid'][index],
            'world2orig':               self.cache['world2orig'][index],
            'bbox_lower':               self.cache['bbox_lower'][index],
            'bbox_upper':               self.cache['bbox_upper'][index]
        }

    def __getitem__(self, index):
        data_dir = self.labels[index]["data_dir"]

        # Check if data is already cached.
        if self.cache_data:
            data = self._get_cached_data(index)
        else:
            data = self._load_data(data_dir)

        # Subsample points.
        uniform_samples = data['uniform_samples']
        if uniform_samples.shape[0] > self.num_point_samples:
            uniform_samples_idxs = np.random.permutation(uniform_samples.shape[0])[:self.num_point_samples]
            uniform_samples = uniform_samples[uniform_samples_idxs, :]

        near_surface_samples = data['near_surface_samples']
        if near_surface_samples.shape[0] > self.num_point_samples:
            near_surface_samples_idxs = np.random.permutation(near_surface_samples.shape[0])[:self.num_point_samples]
            near_surface_samples = near_surface_samples[near_surface_samples_idxs, :]

        surface_samples = data['surface_samples']
        if surface_samples.shape[0] > self.num_point_samples:
            surface_samples_idxs = np.random.permutation(surface_samples.shape[0])[:self.num_point_samples]
            surface_samples = surface_samples[surface_samples_idxs, :]

        if self.use_augmentation:
            # We rotate randomly around y axis.        
            axis = np.array([0, 1, 0])[:, None]

            T_array = []
            for i in range(2):
                angle = random.uniform(0, 2*math.pi)
                axis_angle = axis * angle
                R = o3d.geometry.get_rotation_matrix_from_axis_angle(axis_angle)

                T = np.eye(4)
                T[:3, :3] = R
                T_array.append(T)

            rotated2gaps0 = T_array[0].astype(np.float32)
            rotated2gaps1 = T_array[1].astype(np.float32)

            return np.stack([uniform_samples, uniform_samples], axis=0), np.stack([near_surface_samples, near_surface_samples], axis=0), \
                np.stack([surface_samples, surface_samples], axis=0), np.stack([data['grid'], data['grid']], axis=0),  \
                    np.stack([data['world2grid'], data['world2grid']], axis=0), np.stack([data['world2orig'], data['world2orig']], axis=0), \
                        np.stack([rotated2gaps0, rotated2gaps1], axis=0), \
                            np.stack([data['bbox_lower'], data['bbox_lower']], axis=0), np.stack([data['bbox_upper'], data['bbox_upper']], axis=0), \
                                np.stack([index, index], axis=0)

        else:
            rotated2gaps = np.eye(4).astype(np.float32)

            return uniform_samples[np.newaxis, ...], near_surface_samples[np.newaxis, ...], \
                surface_samples[np.newaxis, ...], data['grid'][np.newaxis, ...],  \
                    data['world2grid'][np.newaxis, ...], data['world2orig'][np.newaxis, ...], \
                        rotated2gaps[np.newaxis, ...], \
                            data['bbox_lower'][np.newaxis, ...], data['bbox_upper'][np.newaxis, ...], \
                                [[index]]

        

    def get_metadata(self, index):
        return self.labels[index]