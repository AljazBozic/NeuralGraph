import os

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from utils import embedder


class MLP(nn.Module):
    def __init__(self, point_dim, time_dim, output_dim=1, n_layers=8, n_pos_freq=10, n_time_freq=0, ngf=256):
        super(MLP, self).__init__()

        self.skips = [4]

        self.pos_embedder, pos_embedder_out_dim = embedder.get_embedder_nerf(n_pos_freq, input_dims=point_dim, i=0)
        self.time_embedder, time_embedder_out_dim = embedder.get_embedder_nerf(n_time_freq, input_dims=time_dim, i=0)

        # print('point_dim: ', point_dim)
        # print('embedder_out_dim: ', pos_embedder_out_dim + time_embedder_out_dim)

        self.pts_linears = nn.ModuleList(
            [nn.Conv1d(pos_embedder_out_dim + time_embedder_out_dim, ngf, kernel_size=1)] +
            [nn.Conv1d(ngf, ngf, kernel_size=1) if i not in self.skips else nn.Conv1d(ngf + pos_embedder_out_dim + time_embedder_out_dim, ngf, kernel_size=1) for i in range(n_layers-1)]
        )
        self.output_linear = nn.Conv1d(ngf, output_dim, kernel_size=1)

        # for the "w/o positional encoding" case
        self.simple_lin = nn.Conv1d(point_dim + time_dim, output_dim, kernel_size=1)

    def forward(self, input_pts, timestamps, use_simple_net=False):
        if use_simple_net:
            # w/o positional encoding
            input_4d = torch.cat([input_pts, timestamps], dim=1)
            return self.simple_lin(input_4d)
        else:
            # w/ positional encoding
            # Note that input_pts must be in [-1, 1]
            
            input_pos_embed = self.pos_embedder(input_pts)
            input_time_embed = self.time_embedder(timestamps)

            input_embed = torch.cat([input_pos_embed, input_time_embed], dim=1)
            h = input_embed
            for i, layer in enumerate(self.pts_linears):
                h = layer(h)
                h = F.leaky_relu(h) # TODO: look into SIREN paper?
                if i in self.skips:
                    h = torch.cat([input_embed, h], 1)
            
            return self.output_linear(h)


class MultiMLP(nn.Module):
    def __init__(self, point_dim, time_dim, num_groups, output_dim=1, n_layers=8, n_pos_freq=10, n_time_freq=0, ngf=256):
        super(MultiMLP, self).__init__()

        self.point_dim = point_dim
        self.time_dim = time_dim
        self.num_groups = num_groups
        self.skips = [4]

        self.pos_embedder, pos_embedder_out_dim = embedder.get_embedder_nerf(n_pos_freq, input_dims=point_dim, i=0)
        self.time_embedder, time_embedder_out_dim = embedder.get_embedder_nerf(n_time_freq, input_dims=time_dim, i=0)

        self.pts_linears = nn.ModuleList(
            [nn.Conv1d(self.num_groups * (pos_embedder_out_dim + time_embedder_out_dim), self.num_groups * ngf, kernel_size=1, groups=self.num_groups)] +
            [nn.Conv1d(self.num_groups * ngf, self.num_groups * ngf, kernel_size=1, groups=self.num_groups) if i not in self.skips else \
                nn.Conv1d(self.num_groups * (ngf + pos_embedder_out_dim + time_embedder_out_dim), self.num_groups * ngf, kernel_size=1, groups=self.num_groups) \
                    for i in range(n_layers-1)]
        )
        self.output_linear = nn.Conv1d(self.num_groups * ngf, self.num_groups * output_dim, kernel_size=1, groups=self.num_groups)

    def forward(self, points, timestamps):
        batch_size = points.shape[0]
        assert points.shape[1] == self.num_groups
        assert points.shape[2] == self.point_dim
        assert timestamps.shape[1] == self.num_groups
        assert timestamps.shape[2] == self.time_dim
        
        num_points = points.shape[3]

        # Embedded points and time.
        points_embedded = self.pos_embedder(points.view(batch_size * self.num_groups, self.point_dim, num_points))
        time_embedded = self.time_embedder(timestamps.view(batch_size * self.num_groups, self.time_dim, num_points))

        # Realign inputs to fit the multi MLP network.
        points_embedded = points_embedded.view(batch_size, self.num_groups, -1, num_points)
        time_embedded = time_embedded.view(batch_size, self.num_groups, -1, num_points)

        input_embedded = torch.cat([points_embedded, time_embedded], dim=2)
        input_embedded = input_embedded.view(batch_size, -1, num_points)

        # Run multi MLP.
        h = input_embedded
        for i, layer in enumerate(self.pts_linears):
            h = layer(h)
            h = F.leaky_relu(h) # TODO: look into SIREN paper?
            if i in self.skips:
                h = torch.cat([input_embedded.view(batch_size, self.num_groups, -1, num_points), h.view(batch_size, self.num_groups, -1, num_points)], 2)
                h = h.view(batch_size, -1, num_points)
        
        return self.output_linear(h)