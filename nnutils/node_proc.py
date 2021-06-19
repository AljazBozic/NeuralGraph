import torch.nn as nn
import torch
import math
import kornia


NORMALIZATION_EPS = 1e-8


def extract_view_omegas_from_embedding(embedding, num_nodes):
    batch_size = embedding.shape[0]

    embedding = embedding.view(batch_size, num_nodes, 10)
    omegas = embedding[:, :, 7:10]

    return omegas


def convert_embedding_to_explicit_params(embedding, rotated2gaps, num_nodes, scaling_type, max_blob_radius=0.15, center_scale=0.5, max_constant=1.0):
# def convert_embedding_to_explicit_params(embedding, rotated2gaps, num_nodes, scaling_type, max_blob_radius=0.05, center_scale=0.5, max_constant=1.0):
    batch_size = embedding.shape[0]

    embedding = embedding.view(batch_size, num_nodes, 10)

    center = embedding[:, :, 0:3]
    constant = embedding[:, :, 3]
    scale = embedding[:, :, 4:7]
    rotation = embedding[:, :, 7:10]

    if scaling_type == "anisotropic":
        scale = torch.sigmoid(scale) * max_blob_radius
        # TODO: Doesn't support general augmentation, not reliable!
    elif scaling_type == "isotropic":
        scale = torch.sigmoid(scale[:, :, 0].view(batch_size, num_nodes, 1).expand(-1, -1, 3)) * max_blob_radius
    else:
        scale = torch.ones_like(scale) * max_blob_radius

    constant = -torch.abs(constant)
    # constant = -torch.min(torch.abs(constant), max_constant * torch.ones_like(constant))
    # constant = -torch.sigmoid(constant) * max_constant
    center = center * center_scale
    center = center.view(batch_size, num_nodes, 3, 1)

    # We represent rotations in axis-angle notation.
    rotation = kornia.angle_axis_to_rotation_matrix(rotation.view(batch_size * num_nodes, 3))
    rotation = rotation.view(batch_size, num_nodes, 3, 3)

    # Since we augment views at train time, we apply extrinsics of the current view.
    # Both center and rotation need to be corrected.
    R_rotated2gaps = rotated2gaps[:, :3, :3].view(batch_size, 1, 3, 3).expand(-1, num_nodes, -1, -1)    # (bs, num_nodes, 3, 3)
    t_rotated2gaps = rotated2gaps[:, :3, 3].view(batch_size, 1, 3, 1).expand(-1, num_nodes, -1, -1)     # (bs, num_nodes, 3, 1)

    center = torch.matmul(R_rotated2gaps, center) + t_rotated2gaps
    
    rotation = torch.matmul(R_rotated2gaps, rotation)

    rotation = rotation.view(batch_size, num_nodes, 3, 3)
    center = center.view(batch_size, num_nodes, 3)

    return constant, scale, rotation, center


def compute_inverse_occupancy(vals, soft_transfer_scale, level_set):
    # vals are negative everywhere, with higher negative values inside the object,
    # slowly decaying to zero outside.
    return torch.sigmoid(soft_transfer_scale * (vals - level_set))


def sample_rbf_weights(points, constants, scales, centers, use_constants):
    batch_size = points.shape[0]
    num_points = points.shape[1]
    assert points.shape[2] == 3

    num_nodes = centers.shape[1]

    # Compute centered points.
    points = points.view(batch_size, num_points, 1, 3).expand(-1, -1, num_nodes, -1)    # (bs, num_points, num_nodes, 3)
    centers = centers.view(batch_size, 1, num_nodes, 3).expand(-1, num_points, -1, -1)  # (bs, num_points, num_nodes, 3)

    delta = points - centers    # (bs, num_points, num_nodes, 3)
    delta2 = delta * delta      # (bs, num_points, num_nodes, 3)
    
    # Add anisotropic scaling.
    # For numerical stability we use at least eps. 
    scales2 = torch.max(scales * scales, NORMALIZATION_EPS*torch.ones_like(scales))  # (bs, num_nodes, 3)

    inv_scales2 = 1.0 / scales2               
    inv_scales2 = inv_scales2.view(batch_size, 1, num_nodes, 3).expand(-1, num_points, -1, -1)  # (bs, num_points, num_nodes, 3)
    
    delta_length_scaled = torch.sum(inv_scales2 * delta2, axis=3)   # (bs, num_points, num_nodes)

    # Apply scaled exponential kernel.
    if use_constants:
        constants = constants.view(batch_size, 1, num_nodes).expand(-1, num_points, -1)     # (bs, num_points, num_nodes)
        weight_vals = -constants * torch.exp(-0.5 * delta_length_scaled)
    else:
        weight_vals = torch.exp(-0.5 * delta_length_scaled)

    return weight_vals.view(batch_size, num_points, num_nodes)


def sample_rbf_surface(points, constants, scales, centers, use_constants, aggregate_coverage_with_max):
    batch_size = points.shape[0]
    num_points = points.shape[1]
    
    # Sum the contributions of all kernels.
    weights_vals = sample_rbf_weights(points, constants, scales, centers, use_constants) # (bs, num_points, num_nodes)

    if aggregate_coverage_with_max:
        sdf_vals, _ = torch.min(-weights_vals, 2)       # (bs, num_points)
    else:
        sdf_vals = torch.sum(-weights_vals, axis=2)     # (bs, num_points)

    return sdf_vals.view(batch_size, num_points)


def bounding_box_error(points, bbox_lower, bbox_upper):
  batch_size = points.shape[0]
  num_points = points.shape[1]
  assert points.shape[2] == 3

  bbox_lower_vec = bbox_lower.view(batch_size, 1, 3).expand(-1, num_points, -1)     # (bs, num_points, 3)
  bbox_upper_vec = bbox_upper.view(batch_size, 1, 3).expand(-1, num_points, -1)     # (bs, num_points, 3)
    
  lower_error = torch.max(bbox_lower_vec - points, torch.zeros_like(points))
  upper_error = torch.max(points - bbox_upper_vec, torch.zeros_like(points))
  
  constraint_error = torch.sum(lower_error * lower_error + upper_error * upper_error, axis=2)   # (bs, num_points)
  return constraint_error