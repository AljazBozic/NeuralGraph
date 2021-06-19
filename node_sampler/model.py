import os
import numpy as np
import torch
import torch.nn as nn

import config as cfg
from nnutils.blocks import ResBlock, make_downscale


class Encoder3D(nn.Module):
    def __init__(self, encoding_dim):
        super().__init__()

        nf1 = 24
        nf2 = 32
        nf3 = 40
        nf4 = 64

        # input: 64x64x64
        self.encoder = nn.Sequential(
            make_downscale(1, nf1, kernel=8, normalization=torch.nn.BatchNorm3d),   # 32x32x32
            ResBlock(nf1, normalization=torch.nn.BatchNorm3d),
            ResBlock(nf1, normalization=torch.nn.BatchNorm3d),
            make_downscale(nf1, nf2, normalization=torch.nn.BatchNorm3d),           # 16x16x16
            ResBlock(nf2, normalization=torch.nn.BatchNorm3d),
            ResBlock(nf2, normalization=torch.nn.BatchNorm3d),
            make_downscale(nf2, nf3, normalization=torch.nn.BatchNorm3d),           # 8x8x8
            ResBlock(nf3, normalization=torch.nn.BatchNorm3d),
            ResBlock(nf3, normalization=torch.nn.BatchNorm3d),
            make_downscale(nf3, nf4, normalization=torch.nn.BatchNorm3d),           # 4x4x4
            ResBlock(nf4, normalization=torch.nn.BatchNorm3d),
            ResBlock(nf4, normalization=torch.nn.BatchNorm3d)
        )
        # output: 4x4x4

        # input: 1
        self.classifier = nn.Sequential(
            nn.Conv1d(nf4 * 4 * 4 * 4, encoding_dim, kernel_size=1)
        )
        # output: 1
        
    def forward(self, x):
        batch_size = x.shape[0]
        grid_dim = x.shape[-1]
        x = x.view(batch_size, 1, grid_dim, grid_dim, grid_dim)

        x = self.encoder(x)

        x = x.view(batch_size, -1, 1)
        x = self.classifier(x)

        return x


class NodeSampler(nn.Module):
    def __init__(self):
        super(NodeSampler, self).__init__()

        encoding_dim = 2048
        std = 0.01

        ################################################################################################
        # Nodes' scale params
        ################################################################################################
        self.scale_params = torch.nn.Parameter(std * torch.randn([cfg.num_nodes, cfg.scale_length], dtype=torch.float32).cuda())
        self.node_length = cfg.position_length

        if cfg.freeze_scale_estimator:
            self.scale_params.requires_grad = False

        ################################################################################################
        # 3D encoder (64x64x64 SDF grid) 
        ################################################################################################
        self.encoder = Encoder3D(encoding_dim)
        
        if cfg.freeze_node_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

        ################################################################################################
        # POSITION MLP
        ################################################################################################
        # Important: We SHOULD NOT have any batch norm in refinement MLP, it
        # doesn't play well with regression.
        self.position_mlp = nn.Sequential(
            nn.Conv1d(encoding_dim, encoding_dim, kernel_size=1),
            torch.nn.LeakyReLU(inplace=True),
            nn.Conv1d(encoding_dim, encoding_dim, kernel_size=1),
            torch.nn.LeakyReLU(inplace=True),
            nn.Conv1d(encoding_dim, cfg.num_nodes * self.node_length, kernel_size=1)
        )

        if cfg.freeze_position_estimator:
            for param in self.position_mlp.parameters():
                param.requires_grad = False

        ################################################################################################
        # ROTATION MLP
        ################################################################################################
        # Important: We SHOULD NOT have any batch norm in refinement MLP, it
        # doesn't play well with regression.
        self.rotation_mlp = nn.Sequential(
            nn.Conv1d(encoding_dim, encoding_dim, kernel_size=1),
            torch.nn.LeakyReLU(inplace=True),
            nn.Conv1d(encoding_dim, encoding_dim, kernel_size=1),
            torch.nn.LeakyReLU(inplace=True),
            nn.Conv1d(encoding_dim, cfg.num_nodes * cfg.rotation_length, kernel_size=1)
        )

        if cfg.freeze_rotation_estimator:
            for param in self.rotation_mlp.parameters():
                param.requires_grad = False

        ################################################################################################
        # Affinity estimation
        ################################################################################################
        # When we use multiple sequences, we share the affinity matrix, but have a separate distance
        # matrix for every sequence.
        self.affinity_matrix = torch.nn.Parameter(std * torch.randn([cfg.num_neighbors, cfg.num_nodes, cfg.num_nodes], dtype=torch.float32).cuda())
        self.distance_matrix = torch.nn.Parameter(std * torch.randn([cfg.num_nodes, cfg.num_nodes], dtype=torch.float32).cuda())

        source_idxs_np  = np.zeros((cfg.num_nodes * (cfg.num_nodes - 1)), dtype=np.int32)
        target_idxs_np  = np.zeros((cfg.num_nodes * (cfg.num_nodes - 1)), dtype=np.int32)
        pair_idxs_np    = np.zeros((cfg.num_nodes * (cfg.num_nodes - 1)), dtype=np.int32)

        mask_matrix_np  = np.ones((cfg.num_nodes, cfg.num_nodes), dtype=np.float32)

        range_idxs = np.arange(cfg.num_nodes)

        idx_offset = 0
        for i in range(cfg.num_nodes):
            # All pairs are valid except the edge to itself.
            source_idxs_np[idx_offset:idx_offset + cfg.num_nodes - 1] = i 
            target_idxs_np[idx_offset:idx_offset + i] = range_idxs[:i]
            target_idxs_np[idx_offset + i:idx_offset + cfg.num_nodes - 1] = range_idxs[i+1:]
            pair_idxs_np[idx_offset:idx_offset + i] = i*cfg.num_nodes + range_idxs[:i]
            pair_idxs_np[idx_offset + i:idx_offset + cfg.num_nodes - 1] = i*cfg.num_nodes + range_idxs[i+1:]

            idx_offset += (cfg.num_nodes - 1)

            # Set the invalid distance in the mask.
            mask_matrix_np[i, i] = 0.0

        self.source_idxs    = torch.from_numpy(source_idxs_np).cuda()
        self.target_idxs    = torch.from_numpy(target_idxs_np).cuda()
        self.pair_idxs      = torch.from_numpy(pair_idxs_np).cuda()
        self.mask_matrix    = torch.from_numpy(mask_matrix_np).cuda().view(1, cfg.num_nodes, cfg.num_nodes).repeat(cfg.num_neighbors, 1, 1)

        self.source_idxs    = self.source_idxs.long()
        self.target_idxs    = self.target_idxs.long()
        self.pair_idxs      = self.pair_idxs.long()

        if cfg.freeze_affinity:
            self.affinity_matrix.requires_grad = False
            self.distance_matrix.requires_grad = False

    def get_affinity_matrix(self):
        # We subtract the max row value for numerical stability.
        # You can derive that this doesn't influence result.
        maxes = torch.max(self.affinity_matrix + torch.log(self.mask_matrix), dim=2, keepdim=True)[0]
        affinity_values = self.mask_matrix * torch.exp(self.affinity_matrix - maxes)
        affinity_values = affinity_values / torch.sum(affinity_values, dim=2, keepdim=True)
        return affinity_values

    def get_distance_matrix(self):
        return torch.abs(self.distance_matrix)

    def forward(self, grid):
        batch_size = grid.shape[0]

        ################################################################################################
        # Compute embedding.
        ################################################################################################
        encoding = self.encoder(grid).view(batch_size, -1, 1)

        # Normalize encoding.
        norm = encoding.norm(p=2, dim=1, keepdim=True)
        encoding = encoding.div(norm)

        # Convert encoding to embedding.
        embedding_position = self.position_mlp(encoding).view(batch_size, cfg.num_nodes, self.node_length)
        embedding_scale = self.scale_params.view(1, cfg.num_nodes, cfg.scale_length).expand(batch_size, -1, -1)
        embedding_rotation = self.rotation_mlp(encoding).view(batch_size, cfg.num_nodes, cfg.rotation_length)

        embedding = torch.cat([embedding_position, embedding_scale, embedding_rotation], dim=2)

        # Compute affinity and distance values for node pairs.
        affinity_matrix = self.get_affinity_matrix()
        distance_matrix = self.get_distance_matrix()

        affinity_matrix_union = torch.sum(affinity_matrix, dim=0) / float(cfg.num_neighbors)

        pair_distances = distance_matrix.view(1, -1)[:, self.pair_idxs].expand(batch_size, -1)
        pair_weights = affinity_matrix_union.view(1, -1)[:, self.pair_idxs].expand(batch_size, -1)

        return embedding, self.source_idxs, self.target_idxs, pair_distances, pair_weights, affinity_matrix
