import numpy as np


def scale_grid(xyz, x_scale, y_scale, z_scale, disp=1.0):
    X, Y, Z = xyz
    X = X*x_scale
    Y = Y*y_scale
    Z = Z*z_scale
    points = np.concatenate((X[np.newaxis, ...], Y[np.newaxis, ...], Z[np.newaxis, ...]), axis=0)
    points = points.reshape(3, -1).astype(np.float32)
    return 2.0 * points - disp


def sample_grid_points(dim, disp=1.0, use_supersampling=False, num_supersamples=2):
    # Generate regular input.
    if use_supersampling:
        edges = dim-1
        supersamples = dim + edges * num_supersamples
        coords_x = scale_grid(np.mgrid[:supersamples, :dim, :dim], disp/(supersamples-1.0), disp/(dim-1.0), disp/(dim-1.0), disp)
        coords_y = scale_grid(np.mgrid[:dim, :supersamples, :dim], disp/(dim-1.0), disp/(supersamples-1.0), disp/(dim-1.0), disp)
        coords_z = scale_grid(np.mgrid[:dim, :dim, :supersamples], disp/(dim-1.0), disp/(dim-1.0), disp/(supersamples-1.0), disp)

        points_x = coords_x.reshape(3, -1).astype(np.float32)
        points_y = coords_y.reshape(3, -1).astype(np.float32)
        points_z = coords_z.reshape(3, -1).astype(np.float32)

        points = np.concatenate((points_x, points_y, points_z), axis=1)
    else:
        coords = scale_grid(np.mgrid[:dim, :dim, :dim], disp/(dim-1.0), disp/(dim-1.0), disp/(dim-1.0), disp)
        points = coords.reshape(3, -1).astype(np.float32)

    return points
    