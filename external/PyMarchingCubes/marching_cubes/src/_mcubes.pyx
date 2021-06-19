
# distutils: language = c++
# cython: embedsignature = True

# from libcpp.vector cimport vector
import numpy as np

# Define PY_ARRAY_UNIQUE_SYMBOL
cdef extern from "pyarray_symbol.h":
    pass

cimport numpy as np

np.import_array()

cdef extern from "pywrapper.h":
    cdef object c_marching_cubes "marching_cubes"(np.ndarray, double) except +
    cdef object c_marching_cubes_func "marching_cubes_func"(tuple, tuple, int, int, int, object, double) except +
    cdef object c_marching_cubes_color "marching_cubes_color"(np.ndarray, np.ndarray, double) except +
    cdef object c_marching_cubes_color_func "marching_cubes_color_func"(tuple, tuple, int, int, int, object, object, object, object, double) except +
    cdef object c_marching_cubes_super_sampling "marching_cubes_super_sampling"(np.ndarray, np.ndarray, np.ndarray, double) except +

def marching_cubes(np.ndarray volume, float isovalue):
    
    verts, faces = c_marching_cubes(volume, isovalue)
    verts.shape = (-1, 3)
    faces.shape = (-1, 3)
    return verts, faces
    
def marching_cubes_super_sampling(np.ndarray volumeX, np.ndarray volumeY, np.ndarray volumeZ, float isovalue):
    
    verts, faces = c_marching_cubes_super_sampling(volumeX, volumeY, volumeZ, isovalue)
    verts.shape = (-1, 3)
    faces.shape = (-1, 3)
    return verts, faces

def marching_cubes_func(tuple lower, tuple upper, int numx, int numy, int numz, object f, double isovalue):
    
    if any(l_i >= u_i for l_i, u_i in zip(lower, upper)):
        raise ValueError("lower coordinates cannot be larger than upper coordinates")
    
    if numx < 2 or numy < 2 or numz < 2:
        raise ValueError("numx, numy, numz cannot be smaller than 2")

    verts, faces = c_marching_cubes_func(lower, upper, numx, numy, numz, f, isovalue)
    verts.shape = (-1, 3)
    faces.shape = (-1, 3)
    return verts, faces



def marching_cubes_color(np.ndarray volume_sdf, np.ndarray volume_color, float isovalue):
    
    verts, faces = c_marching_cubes_color(volume_sdf, volume_color, isovalue)
    verts.shape = (-1, 6)
    faces.shape = (-1, 3)
    return verts, faces

def marching_cubes_color_func(tuple lower, tuple upper, int numx, int numy, int numz, object f_sdf, object f_color_r, object f_color_g, object f_color_b, double isovalue):
    
    if any(l_i >= u_i for l_i, u_i in zip(lower, upper)):
        raise ValueError("lower coordinates cannot be larger than upper coordinates")
    
    if numx < 2 or numy < 2 or numz < 2:
        raise ValueError("numx, numy, numz cannot be smaller than 2")

    verts, faces = c_marching_cubes_color_func(lower, upper, numx, numy, numz, f_sdf, f_color_r, f_color_g, f_color_b, isovalue)
    verts.shape = (-1, 6)
    faces.shape = (-1, 3)
    return verts, faces


