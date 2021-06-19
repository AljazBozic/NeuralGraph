import numpy as np
import torch


def augment_grid(grid, world2grid, aug2world):
    batch_size = grid.shape[0]
    dim_z = grid.shape[1]
    dim_y = grid.shape[2]
    dim_x = grid.shape[3]

    # Compute grid coordinates.
    Z, Y, X = torch.meshgrid([
        torch.arange(dim_z, dtype=torch.float32, device=grid.device),
        torch.arange(dim_y, dtype=torch.float32, device=grid.device),
        torch.arange(dim_x, dtype=torch.float32, device=grid.device)
    ])
    X = X.view(dim_z, dim_y, dim_x, 1)
    Y = Y.view(dim_z, dim_y, dim_x, 1)
    Z = Z.view(dim_z, dim_y, dim_x, 1)
    
    positions_grid_cs = torch.cat([X, Y, Z], dim=3)
    positions_grid_cs = positions_grid_cs.view(1, dim_z, dim_y, dim_x, 3).expand(batch_size, -1, -1, -1, -1)
    positions_grid_cs = positions_grid_cs.view(batch_size, -1, 3, 1)

    # Apply transformation: grid space -> world aug space -> world orig space -> grid space
    grid2world = torch.inverse(world2grid)
    num_points = positions_grid_cs.shape[1]

    T = torch.matmul(world2grid, torch.matmul(aug2world, grid2world))

    R = T[:, :3, :3] 
    t = T[:, :3, 3] 

    positions_aug_grid_cs = torch.matmul(R.view(batch_size, 1, 3, 3).expand(-1, num_points, -1, -1), positions_grid_cs) + \
        t.reshape(batch_size, 1, 3, 1).expand(-1, num_points, -1, -1)
    positions_aug_grid_cs = positions_aug_grid_cs.view(-1 ,3)

    # Query grid values at transformed positions.
    grid = grid.view(batch_size, 1, dim_z, dim_y, dim_x)

    positions_aug_grid_cs[..., 0] /= float(dim_x - 1)
    positions_aug_grid_cs[..., 1] /= float(dim_y - 1)
    positions_aug_grid_cs[..., 2] /= float(dim_z - 1)
    positions_aug_grid_cs = 2.0 * positions_aug_grid_cs - 1.0   
    positions_aug_grid_cs = positions_aug_grid_cs.view(batch_size, -1, 1, 1, 3)

    # We use border values for out-of-the-box queries.
    grid_aug = torch.nn.functional.grid_sample(grid, positions_aug_grid_cs, align_corners=True, padding_mode="border")
    grid_aug = grid_aug.view(batch_size, dim_z, dim_y, dim_x)

    return grid_aug
