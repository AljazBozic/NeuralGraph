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
    loss_geometry_uniform_sum = 0.0
    loss_geometry_near_surface_sum = 0.0

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
        grid                    = dataset_obj.unpack(grid).cuda()
        world2grid              = dataset_obj.unpack(world2grid).cuda()
        rotated2gaps            = dataset_obj.unpack(rotated2gaps).cuda()

        # Merge uniform and near surface samples.
        batch_size = uniform_samples.shape[0]
        
        num_uniform_points = uniform_samples.shape[1]
        num_near_surfaces_points = near_surface_samples.shape[1]
        num_points = num_uniform_points + num_near_surfaces_points

        points = torch.zeros((batch_size, num_points, 3), dtype=uniform_samples.dtype, device=uniform_samples.device)
        points[:, :num_uniform_points, :] = uniform_samples[:, :, :3]
        points[:, num_uniform_points:, :] = near_surface_samples[:, :, :3]

        with torch.no_grad():
            # Compute augmented sdfs.
            sdfs = augment_grid(grid, world2grid, rotated2gaps)

            # Forward pass.
            sdf_pred = model(points, sdfs, rotated2gaps)

            # Loss.
            uniform_sdf_pred = sdf_pred[:, :, :num_uniform_points].view(batch_size, 1, -1)
            near_surface_sdf_pred = sdf_pred[:, :, num_uniform_points:].view(batch_size, 1, -1)

            loss, loss_geometry_uniform, loss_geometry_near_surface = \
                criterion(uniform_samples, near_surface_samples, uniform_sdf_pred, near_surface_sdf_pred, eval=True)

            loss_sum += loss.item()
            if loss_geometry_uniform:       loss_geometry_uniform_sum += loss_geometry_uniform.item()
            if loss_geometry_near_surface:  loss_geometry_near_surface_sum += loss_geometry_near_surface.item()
          
            # Metrics.
        
    # Losses
    loss_avg = loss_sum / total_num_batches
    loss_geometry_uniform_avg = loss_geometry_uniform_sum / total_num_batches
    loss_geometry_near_surface_avg = loss_geometry_near_surface_sum / total_num_batches

    losses = {
        "total": loss_avg,
        "geometry_uniform": loss_geometry_uniform_avg,
        "geometry_near_surface": loss_geometry_near_surface_avg
    }

    # Metrics.
    metrics = {
    }

    return losses, metrics


if __name__ == "__main__":
    pass