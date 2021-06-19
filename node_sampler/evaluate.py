import sys,os
import math
import torch
from timeit import default_timer as timer
from nnutils.geometry import augment_grid


def evaluate(model, criterion, dataloader, batch_num):
    dataset_obj = dataloader.dataset
    dataset_batch_size = dataloader.batch_size
    total_size = len(dataset_obj)

    # Losses
    loss_sum = 0.0
    loss_uniform_sum = 0.0
    loss_near_surface_sum = 0.0
    loss_node_center_sum = 0.0
    loss_affinity_rel_sum = 0.0
    loss_affinity_abs_sum = 0.0
    loss_unique_neighbor_sum = 0.0
    loss_viewpoint_position_sum = 0.0
    loss_viewpoint_scale_sum = 0.0
    loss_viewpoint_constant_sum = 0.0
    loss_viewpoint_rotation_sum = 0.0
    loss_surface_consistency_sum = 0.0

    max_num_batches = int(math.ceil(total_size / dataset_batch_size))
    total_num_batches = batch_num if batch_num != -1 else max_num_batches
    total_num_batches = min(max_num_batches, total_num_batches)

    print()

    for i, data in enumerate(dataloader):
        if i >= total_num_batches: 
            break

        sys.stdout.write("\r############# Eval iteration: {0} / {1}".format(i + 1, total_num_batches))
        sys.stdout.flush()

        # Data loading.
        uniform_samples, near_surface_samples, surface_samples, grid, world2grid, world2orig, rotated2gaps, bbox_lower, bbox_upper, sample_idx = data

        uniform_samples         = dataset_obj.unpack(uniform_samples).cuda()
        near_surface_samples    = dataset_obj.unpack(near_surface_samples).cuda()
        surface_samples         = dataset_obj.unpack(surface_samples).cuda()
        grid                    = dataset_obj.unpack(grid).cuda()
        world2grid              = dataset_obj.unpack(world2grid).cuda()
        world2orig              = dataset_obj.unpack(world2orig).cuda()
        rotated2gaps            = dataset_obj.unpack(rotated2gaps).cuda()
        bbox_lower              = dataset_obj.unpack(bbox_lower).cuda()
        bbox_upper              = dataset_obj.unpack(bbox_upper).cuda()

        with torch.no_grad():
            # Compute augmented sdfs.
            sdfs = augment_grid(grid, world2grid, rotated2gaps)

            # Forward pass.
            embedding_pred, source_idxs, target_idxs, pair_distances, pair_weights, affinity_matrix = model(sdfs)

            # Loss.
            loss, all_losses = criterion(
                embedding_pred, uniform_samples, near_surface_samples, surface_samples,
                grid, world2grid, world2orig, rotated2gaps, bbox_lower, bbox_upper, 
                source_idxs, target_idxs, pair_distances, pair_weights, affinity_matrix,
                evaluate=True
            )

            loss_sum += loss.item()
            if all_losses["loss_uniform"]:              loss_uniform_sum += all_losses["loss_uniform"].item()
            if all_losses["loss_near_surface"]:         loss_near_surface_sum += all_losses["loss_near_surface"].item()
            if all_losses["loss_node_center"]:          loss_node_center_sum += all_losses["loss_node_center"].item()
            if all_losses["loss_affinity_rel"]:         loss_affinity_rel_sum += all_losses["loss_affinity_rel"].item()
            if all_losses["loss_affinity_rel"]:         loss_affinity_abs_sum += all_losses["loss_affinity_abs"].item()
            if all_losses["loss_unique_neighbor"]:      loss_unique_neighbor_sum += all_losses["loss_unique_neighbor"].item()
            if all_losses["loss_viewpoint_position"]:   loss_viewpoint_position_sum += all_losses["loss_viewpoint_position"].item()
            if all_losses["loss_viewpoint_scale"]:      loss_viewpoint_scale_sum += all_losses["loss_viewpoint_scale"].item()
            if all_losses["loss_viewpoint_constant"]:   loss_viewpoint_constant_sum += all_losses["loss_viewpoint_constant"].item()
            if all_losses["loss_viewpoint_rotation"]:   loss_viewpoint_rotation_sum += all_losses["loss_viewpoint_rotation"].item()
            if all_losses["loss_surface_consistency"]:  loss_surface_consistency_sum += all_losses["loss_surface_consistency"].item()
          
            # Metrics.
        
    # Losses.
    loss_avg = loss_sum / total_num_batches
    loss_uniform_avg = loss_uniform_sum / total_num_batches
    loss_near_surface_avg = loss_near_surface_sum / total_num_batches
    loss_node_center_avg = loss_node_center_sum / total_num_batches
    loss_affinity_rel_avg = loss_affinity_rel_sum / total_num_batches
    loss_affinity_abs_avg = loss_affinity_abs_sum / total_num_batches
    loss_unique_neighbor_avg = loss_unique_neighbor_sum / total_num_batches
    loss_viewpoint_position_avg = loss_viewpoint_position_sum / total_num_batches
    loss_viewpoint_scale_avg = loss_viewpoint_scale_sum / total_num_batches
    loss_viewpoint_constant_avg = loss_viewpoint_constant_sum / total_num_batches
    loss_viewpoint_rotation_avg = loss_viewpoint_rotation_sum / total_num_batches
    loss_surface_consistency_avg = loss_surface_consistency_sum / total_num_batches

    losses = {
        "total": loss_avg,
        "uniform": loss_uniform_avg,
        "near_surface": loss_near_surface_avg,
        "node_center": loss_node_center_avg,
        "affinity_rel": loss_affinity_rel_avg,
        "affinity_abs": loss_affinity_abs_avg,
        "unique_neighbor": loss_unique_neighbor_avg,
        "viewpoint_position": loss_viewpoint_position_avg,
        "viewpoint_scale": loss_viewpoint_scale_avg,
        "viewpoint_constant": loss_viewpoint_constant_avg,
        "viewpoint_rotation": loss_viewpoint_rotation_avg,
        "surface_consistency": loss_surface_consistency_avg
    }

    # Metrics.
    metrics = {
    }

    return losses, metrics


if __name__ == "__main__":
    pass