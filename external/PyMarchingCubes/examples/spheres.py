
import numpy as np
import marching_cubes as mcubes

print("Example 1: Isosurface in NumPy volume...")
#print(mcubes.__dir__())

# Create a data volume (100 x 100 x 100)
X, Y, Z = np.mgrid[:100, :100, :100]
sdf = (X-50)**2 + (Y-50)**2 + (Z-50)**2 - 25**2

# Extract the 0-isosurface
vertices, triangles = mcubes.marching_cubes(sdf, 0)
mcubes.export_obj(vertices, triangles, "sphere.obj")




print("Example 2: Isosurface and color in NumPy volume...")

# Extract isosurface and color
#color = 0.01 * np.concatenate((X[:,:,:,None],X[:,:,:,None],X[:,:,:,None]), axis=3) # color array (grayscale gradient in this example)
color = 0.01 * np.concatenate((X[:,:,:,None],Y[:,:,:,None],Z[:,:,:,None]), axis=3) # color array (positions as color)
vertices_color, triangles_color = mcubes.marching_cubes_color(sdf, color, 0)
mcubes.export_obj(vertices_color, triangles_color, "sphere_color.obj")
mcubes.export_off(vertices_color, triangles_color, "sphere_color.off")




print("Example 3: TSDF isosurface with super resolution...")

dim = 100

# Create a data volume (100 x 100 x 100)
def sphere_tsdf(xyz, x_scale, y_scale, z_scale):
    X, Y, Z = xyz
    X = X*x_scale
    Y = Y*y_scale
    Z = Z*z_scale
    sdf = (X-0.5)**2 + (Y-0.5)**2 + (Z-0.5)**2 - 0.25**2
    truncation = 0.001 # relatively small truncation
    return np.clip(sdf, -truncation, truncation)

# Extract the 0-isosurface
sdf = sphere_tsdf(np.mgrid[:dim, :dim, :dim], 1.0/(dim-1.0),1.0/(dim-1.0),1.0/(dim-1.0))
vertices, triangles = mcubes.marching_cubes(sdf, 0)
mcubes.export_off(vertices, triangles, "sphere_tsdf_without_super_res.off")

# Extract the 0-isosurface with super res
edges = dim-1
n_edge_samples = 10
supersamples = dim + edges * n_edge_samples
sdf_x = sphere_tsdf(np.mgrid[:supersamples, :dim, :dim], 1.0/(supersamples-1.0), 1.0/(dim-1.0), 1.0/(dim-1.0)) ## generates 10x more samples in x
sdf_y = sphere_tsdf(np.mgrid[:dim, :supersamples, :dim], 1.0/(dim-1.0), 1.0/(supersamples-1.0), 1.0/(dim-1.0)) ## generates 10x more samples in y
sdf_z = sphere_tsdf(np.mgrid[:dim, :dim, :supersamples], 1.0/(dim-1.0), 1.0/(dim-1.0), 1.0/(supersamples-1.0)) ## generates 10x more samples in z
vertices, triangles = mcubes.marching_cubes_super_sampling(sdf_x, sdf_y, sdf_z, 0)
mcubes.export_off(vertices, triangles, "sphere_tsdf_super_res.off")




# old examples

# Export the result to sphere.dae
#mcubes.export_mesh(vertices1, triangles1, "sphere1.dae", "MySphere")

# print("Done. Result saved in 'sphere1.dae'.")

# print("Example 2: Isosurface in Python function...")
# print("(this might take a while...)")

# # Create the volume
# def f(x, y, z):
#     return x**2 + y**2 + z**2

# # Extract the 16-isosurface
# vertices2, triangles2 = mcubes.marching_cubes_func(
#         (-10,-10,-10), (10,10,10),  # Bounds
#         100, 100, 100,              # Number of samples in each dimension
#         f,                          # Implicit function
#         16)                         # Isosurface value

# # Export the result to sphere2.dae
# mcubes.export_mesh(vertices2, triangles2, "sphere2.dae", "MySphere")
# print("Done. Result saved in 'sphere2.dae'.")

# try:
#     print("Plotting mesh...")
#     from mayavi import mlab
#     mlab.triangular_mesh(
#         vertices1[:, 0], vertices1[:, 1], vertices1[:, 2],
#         triangles1)
#     print("Done.")
#     mlab.show()
# except ImportError:
#     print("Could not import mayavi. Interactive demo not available.")
