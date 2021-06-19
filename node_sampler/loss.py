import sys,os

import torch
import torch.nn as nn
import numpy as np
import math

import config as cfg

from utils.pcd_utils import *
from nnutils.node_proc import convert_embedding_to_explicit_params, compute_inverse_occupancy, \
    sample_rbf_surface, sample_rbf_weights, bounding_box_error, extract_view_omegas_from_embedding


class SamplerLoss(torch.nn.Module):
    def __init__(self):
        super(SamplerLoss, self).__init__()

        self.point_loss = PointLoss()
        self.node_center_loss = NodeCenterLoss()
        self.affinity_loss = AffinityLoss()
        self.unique_neighbor_loss = UniqueNeighborLoss()
        self.viewpoint_consistency_loss = ViewpointConsistencyLoss()
        self.surface_consistency_loss = SurfaceConsistencyLoss()
        
    def forward(self, embedding, uniform_samples, near_surface_samples, surface_samples, \
        grid, world2grid, world2orig, rotated2gaps, bbox_lower, bbox_upper, \
        source_idxs, target_idxs, pred_distances, pair_weights, affinity_matrix, evaluate=False
    ):
        loss_total = torch.zeros((1), dtype=embedding.dtype, device=embedding.device)

        view_omegas = extract_view_omegas_from_embedding(embedding, cfg.num_nodes)
        constants, scales, rotations, centers = convert_embedding_to_explicit_params(embedding, rotated2gaps, cfg.num_nodes, cfg.scaling_type)

        # Uniform sampling loss.
        loss_uniform = None
        if cfg.lambda_sampling_uniform is not None:
            loss_uniform = self.point_loss(uniform_samples, constants, scales, centers)
            loss_total += cfg.lambda_sampling_uniform * loss_uniform

        # Near surface sampling loss.
        loss_near_surface = None
        if cfg.lambda_sampling_near_surface is not None:
            loss_near_surface = self.point_loss(near_surface_samples, constants, scales, centers)
            loss_total += cfg.lambda_sampling_near_surface * loss_near_surface
        
        # Node center loss.
        loss_node_center = None
        if cfg.lambda_sampling_node_center is not None:
            loss_node_center = self.node_center_loss(constants, scales, centers, grid, world2grid, bbox_lower, bbox_upper)
            loss_total += cfg.lambda_sampling_node_center * loss_node_center

        # Affinity loss.
        loss_affinity_rel = None
        loss_affinity_abs = None

        if (cfg.lambda_affinity_rel_dist is not None) or (cfg.lambda_affinity_abs_dist is not None):
            loss_affinity_rel, loss_affinity_abs = self.affinity_loss(centers, source_idxs, target_idxs, pred_distances, pair_weights)
            
            if cfg.lambda_affinity_rel_dist is not None: loss_total += cfg.lambda_affinity_rel_dist * loss_affinity_rel
            if cfg.lambda_affinity_abs_dist is not None: loss_total += cfg.lambda_affinity_abs_dist * loss_affinity_abs

        # Unique neighbor loss.
        loss_unique_neighbor = None
        if cfg.lambda_unique_neighbor is not None and affinity_matrix is not None:
            loss_unique_neighbor = self.unique_neighbor_loss(affinity_matrix)
            loss_total += cfg.lambda_unique_neighbor * loss_unique_neighbor

        # Viewpoint consistency loss.
        loss_viewpoint_position = None
        loss_viewpoint_scale = None
        loss_viewpoint_constant = None
        loss_viewpoint_rotation = None

        if (cfg.lambda_viewpoint_position is not None) or (cfg.lambda_viewpoint_scale is not None) or \
            (cfg.lambda_viewpoint_constant is not None) or (cfg.lambda_viewpoint_rotation is not None):
            loss_viewpoint_position, loss_viewpoint_scale, loss_viewpoint_constant, loss_viewpoint_rotation = \
                self.viewpoint_consistency_loss(constants, scales, rotations, centers)
            
            if cfg.lambda_viewpoint_position is not None: 
                loss_total += cfg.lambda_viewpoint_position * loss_viewpoint_position
            if cfg.lambda_viewpoint_scale is not None:    
                loss_total += cfg.lambda_viewpoint_scale * loss_viewpoint_scale
            if cfg.lambda_viewpoint_constant is not None:
                loss_total += cfg.lambda_viewpoint_constant * loss_viewpoint_constant
            if cfg.lambda_viewpoint_rotation is not None:
                loss_total += cfg.lambda_viewpoint_rotation * loss_viewpoint_rotation

        # Surface consistency loss.
        loss_surface_consistency = None
        if cfg.lambda_surface_consistency is not None:
            loss_surface_consistency = self.surface_consistency_loss(constants, scales, rotations, centers, surface_samples, grid, world2grid)
            loss_total += cfg.lambda_surface_consistency * loss_surface_consistency

        if evaluate:
            return loss_total, {
                "loss_uniform": loss_uniform, 
                "loss_near_surface": loss_near_surface, 
                "loss_node_center": loss_node_center, 
                "loss_affinity_rel": loss_affinity_rel, 
                "loss_affinity_abs": loss_affinity_abs, 
                "loss_unique_neighbor": loss_unique_neighbor,
                "loss_viewpoint_position": loss_viewpoint_position, 
                "loss_viewpoint_scale": loss_viewpoint_scale, 
                "loss_viewpoint_constant": loss_viewpoint_constant, 
                "loss_viewpoint_rotation": loss_viewpoint_rotation, 
                "loss_surface_consistency": loss_surface_consistency
            }
        else:
            return loss_total
        

class PointLoss(nn.Module):
    def __init__(self):
        super(PointLoss, self).__init__()

    def forward(self, points_with_sdf, constants, scales, centers):
        batch_size = points_with_sdf.shape[0]

        points = points_with_sdf[:, :, :3]
        is_outside = (points_with_sdf[:, :, 3] > 0.0)
        class_gt = is_outside.float() # outside: 1, inside: 0

        # Evaluate predicted class at given points.
        sdf_pred = sample_rbf_surface(points, constants, scales, centers, cfg.use_constants, cfg.aggregate_coverage_with_max)
        class_pred = compute_inverse_occupancy(sdf_pred, cfg.soft_transfer_scale, cfg.level_set)

        # We apply weight scaling to interior points.
        weights = is_outside.float() + cfg.interior_point_weight * (~is_outside).float()

        # Compute weighted L2 loss.
        diff = class_gt - class_pred
        diff2 = diff * diff
        weighted_diff2 = weights * diff2

        loss = weighted_diff2.mean()
        return loss


class NodeCenterLoss(nn.Module):
    def __init__(self):
        super(NodeCenterLoss, self).__init__()

    def forward(self, constants, scales, centers, grid, world2grid, bbox_lower, bbox_upper):
        batch_size = constants.shape[0]

        # Check if centers are inside the bounding box.
        # If not, we penalize them by using L2 distance to nearest bbox corner, 
        # since there would be no SDF gradients there.
        bbox_error = bounding_box_error(centers, bbox_lower, bbox_upper)    # (bs, num_nodes)

        # Query SDF grid, to encourage centers to be inside the shape.
        # Convert center positions to grid CS.
        centers_grid_cs = centers.view(batch_size, cfg.num_nodes, 3, 1)
        A_world2grid = world2grid[:, :3, :3].view(batch_size, 1, 3, 3).expand(-1, cfg.num_nodes, -1, -1)
        t_world2grid = world2grid[:, :3, 3].view(batch_size, 1, 3, 1).expand(-1, cfg.num_nodes, -1, -1)
        
        centers_grid_cs = torch.matmul(A_world2grid, centers_grid_cs) + t_world2grid
        centers_grid_cs = centers_grid_cs.view(batch_size, -1, 3)

        # Sample signed distance field.
        dim_z = grid.shape[1]
        dim_y = grid.shape[2]
        dim_x = grid.shape[3]
        grid = grid.view(batch_size, 1, dim_z, dim_y, dim_x)

        centers_grid_cs[..., 0] /= float(dim_x - 1)
        centers_grid_cs[..., 1] /= float(dim_y - 1)
        centers_grid_cs[..., 2] /= float(dim_z - 1)
        centers_grid_cs = 2.0 * centers_grid_cs - 1.0   
        centers_grid_cs = centers_grid_cs.view(batch_size, -1, 1, 1, 3)

        # We use border values for out-of-the-box queries, to have gradient zero at boundaries.
        centers_sdf_gt = torch.nn.functional.grid_sample(grid, centers_grid_cs, align_corners=True, padding_mode="border")

        # If SDF value is higher than 0, we penalize it.
        centers_sdf_gt = centers_sdf_gt.view(batch_size, cfg.num_nodes)
        center_distance_error = torch.max(centers_sdf_gt, torch.zeros_like(centers_sdf_gt))     # (bs, num_nodes)

        # Final loss is just a sum of both losses.
        node_center_loss = bbox_error + center_distance_error
        return torch.mean(node_center_loss)


class AffinityLoss(nn.Module):
    def __init__(self):
        super(AffinityLoss, self).__init__()

    def forward(self, centers, source_idxs, target_idxs, pred_distances, pair_weights):
        batch_size = centers.shape[0]
        num_pairs = pred_distances.shape[1]

        loss_rel = 0.0
        loss_abs = 0.0

        if num_pairs > 0:
            source_positions = centers[:, source_idxs]  
            target_positions = centers[:, target_idxs] 

            diff = (source_positions - target_positions)
            dist2 = torch.sum(diff*diff, 2)                                         # (bs, num_pairs)

            abs_distance2 = pair_weights * dist2
            loss_abs = abs_distance2.mean()

            pred_distances2 = pred_distances * pred_distances
            pred_distances2 = pred_distances2                                   # (bs, num_pairs)

            weights_dist = pair_weights * torch.abs(pred_distances2 - dist2)    # (bs, num_pairs)
            loss_rel = weights_dist.mean()

        return loss_rel, loss_abs


class UniqueNeighborLoss(nn.Module):
    def __init__(self):
        super(UniqueNeighborLoss, self).__init__()

    def forward(self, affinity_matrix):
        assert affinity_matrix.shape[0] == cfg.num_neighbors and affinity_matrix.shape[1] == cfg.num_nodes and affinity_matrix.shape[2] == cfg.num_nodes
        
        loss = 0.0

        for source_idx in range(cfg.num_neighbors):
            for target_idx in range(source_idx + 1, cfg.num_neighbors):
                affinity_source = affinity_matrix[source_idx].view(cfg.num_nodes, cfg.num_nodes)
                affinity_target = affinity_matrix[target_idx].view(cfg.num_nodes, cfg.num_nodes)

                # We want rows of different neighbors to be unique.
                affinity_dot = affinity_source * affinity_target
                affinity_dist = torch.sum(affinity_dot, dim=1)

                loss += affinity_dist.sum()

        # Normalize the loss by dividing with the number of pairs.
        num_pairs = (cfg.num_neighbors * (cfg.num_neighbors - 1)) / 2
        loss = loss / float(num_pairs)
        
        return loss


class ViewpointConsistencyLoss(nn.Module):
    def __init__(self):
        super(ViewpointConsistencyLoss, self).__init__()

    def forward(self, constants, scales, rotations, centers):
        batch_size = constants.shape[0]
        assert batch_size % 2 == 0

        # We expect every two consecutive samples are different viewpoints at same time step.
        loss_viewpoint_position = 0.0
        if cfg.lambda_viewpoint_position is not None:
            centers_pairs = centers.view(batch_size // 2, 2, cfg.num_nodes, -1)
            centers_diff = centers_pairs[:, 0, :, :] - centers_pairs[:, 1, :, :]
            centers_dist2 = centers_diff * centers_diff
            loss_viewpoint_position += centers_dist2.mean()

        loss_viewpoint_scale = 0.0
        if cfg.lambda_viewpoint_scale is not None:
            scales_pairs = scales.view(batch_size // 2, 2, cfg.num_nodes, -1)
            scales_diff = scales_pairs[:, 0, :, :] - scales_pairs[:, 1, :, :]
            scales_dist2 = scales_diff * scales_diff
            loss_viewpoint_scale += scales_dist2.mean()

        loss_viewpoint_constant = 0.0
        if cfg.lambda_viewpoint_constant is not None:
            constants_pairs = constants.view(batch_size // 2, 2, cfg.num_nodes, -1)
            constants_diff = constants_pairs[:, 0, :, :] - constants_pairs[:, 1, :, :]
            constants_dist2 = constants_diff * constants_diff
            loss_viewpoint_constant += constants_dist2.mean()

        loss_viewpoint_rotation = 0.0
        if cfg.lambda_viewpoint_rotation is not None:
            rotations_pairs = rotations.view(batch_size // 2, 2, cfg.num_nodes, 3, 3)
            rotations_diff = rotations_pairs[:, 0, :, :, :] - rotations_pairs[:, 1, :, :, :]
            rotations_dist2 = rotations_diff * rotations_diff
            loss_viewpoint_rotation +=  rotations_dist2.mean()

        return loss_viewpoint_position, loss_viewpoint_scale, loss_viewpoint_constant, loss_viewpoint_rotation


class SurfaceConsistencyLoss(nn.Module):
    def __init__(self):
        super(SurfaceConsistencyLoss, self).__init__()

    def forward(self, constants, scales, rotations, centers, surface_samples, grid, world2grid):
        batch_size = constants.shape[0]
        num_points = surface_samples.shape[1]

        loss = 0.0

        surface_points = surface_samples[:, :, :3]

        # Compute skinning weights for sampled points.
        skinning_weights = sample_rbf_weights(surface_points, constants, scales, centers, cfg.use_constants) # (bs, num_points, num_nodes)

        # Compute loss for pairs of frames.
        for source_idx in range(batch_size):
            target_idx = source_idx + 1 if source_idx < batch_size - 1 else 0

            # Get source points and target grid.
            source_points = surface_points[source_idx]  # (num_points, 3)
            target_grid = grid[target_idx]              # (grid_dim, grid_dim, grid_dim)

            # Get source and target rotations.
            R_source = rotations[source_idx]            # (num_nodes, 3, 3)
            R_target = rotations[target_idx]            # (num_nodes, 3, 3)

            # Compute relative frame-to-frame rotation and translation estimates.
            t_source = centers[source_idx]
            t_target = centers[target_idx]

            R_source_inv = R_source.permute(0, 2, 1)
            R_rel = torch.matmul(R_target, R_source_inv)    # (num_nodes, 3, 3)

            # Get correspondending skinning weights and normalize them to sum up to 1.
            weights = skinning_weights[source_idx].view(num_points, cfg.num_nodes)
            weights_sum = weights.sum(dim=1, keepdim=True)
            weights = weights.div(weights_sum)

            # Apply deformation to sampled points.
            t_source = t_source.view(1, cfg.num_nodes, 3, 1).expand(num_points, -1, -1, -1)             # (num_points, num_nodes, 3, 1)
            t_target = t_target.view(1, cfg.num_nodes, 3, 1).expand(num_points, -1, -1, -1)             # (num_points, num_nodes, 3, 1)
            R_rel = R_rel.view(1, cfg.num_nodes, 3, 3).expand(num_points, -1, -1, -1)                   # (num_points, num_nodes, 3, 3)
            source_points = source_points.view(num_points, 1, 3, 1).expand(-1, cfg.num_nodes, -1, -1)   # (num_points, num_nodes, 3, 1)
            weights = weights.view(num_points, cfg.num_nodes, 1, 1).expand(-1, -1, 3, -1)               # (num_points, num_nodes, 3, 1)

            transformed_points = torch.matmul(R_rel, (source_points - t_source)) + t_target             # (num_points, num_nodes, 3, 1)
            transformed_points = torch.sum(weights * transformed_points, dim=1).view(num_points, 3) 

            # Convert transformed points to grid CS.
            transformed_points = transformed_points.view(num_points, 3, 1)
            A_world2grid = world2grid[target_idx, :3, :3].view(1, 3, 3).expand(num_points, -1, -1)
            t_world2grid = world2grid[target_idx, :3, 3].view(1, 3, 1).expand(num_points, -1, -1)
            
            transformed_points_grid_cs = torch.matmul(A_world2grid, transformed_points) + t_world2grid
            transformed_points_grid_cs = transformed_points_grid_cs.view(num_points, 3)

            # Sample signed distance field.
            dim_z = target_grid.shape[0]
            dim_y = target_grid.shape[1]
            dim_x = target_grid.shape[2]
            target_grid = target_grid.view(1, 1, dim_z, dim_y, dim_x)

            transformed_points_grid_cs[..., 0] /= float(dim_x - 1)
            transformed_points_grid_cs[..., 1] /= float(dim_y - 1)
            transformed_points_grid_cs[..., 2] /= float(dim_z - 1)
            transformed_points_grid_cs = 2.0 * transformed_points_grid_cs - 1.0   
            transformed_points_grid_cs = transformed_points_grid_cs.view(1, -1, 1, 1, 3)

            # We use border values for out-of-the-box queries, to have gradient zero at boundaries.
            transformed_points_sdf_gt = torch.nn.functional.grid_sample(target_grid, transformed_points_grid_cs, align_corners=True, padding_mode="border")

            # If SDF value is different than 0, we penalize it.
            transformed_points_sdf_gt = transformed_points_sdf_gt.view(num_points)
            df_error = torch.mean(transformed_points_sdf_gt * transformed_points_sdf_gt)

            loss += df_error

        return loss