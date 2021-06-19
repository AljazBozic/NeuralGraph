import os, sys

import numpy as np
import torch

import config as cfg
import open3d as o3d


T_opengl_cv = np.array([
    [1.0,  0.0, 0.0],
    [0.0, -1.0, 0.0],
    [0.0, 0.0, -1.0]
])

T_opengl_cv_homogeneous = np.array([
    [1.0,  0.0,  0.0, 0.0],
    [0.0, -1.0,  0.0, 0.0],
    [0.0,  0.0, -1.0, 0.0],
    [0.0,  0.0,  0.0, 1.0],
])

origin = np.array([0, 0, 0])
z_axis = np.array([0, 0, 1])

unit_p_min = np.array([-1, -1, -1])
unit_p_max = np.array([ 1,  1,  1])


def transform_pointcloud_to_opengl_coords(points_cv):
    assert len(points_cv.shape) == 2 and points_cv.shape[1] == 3, points_cv.shape

    # apply 180deg rotation around 'x' axis to transform the mesh into OpenGL coordinates
    point_opengl = np.matmul(points_cv, T_opengl_cv.transpose())

    return point_opengl


class UnitBBox():
    unit_bbox_points = np.array([
            [0, 0, 0],
            [1, 0, 0],
            [1, 1, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 1],
            [1, 1, 1],
            [0, 1, 1]
        ], dtype=np.float32)

    bbox_edges = [
            [0, 1],
            [1, 2],
            [2, 3],
            [3, 0],
            [4, 5],
            [5, 6],
            [6, 7],
            [7, 4],
            [0, 4],
            [1, 5],
            [2, 6],
            [3, 7]
        ]


class BBox:
    def __init__(self, points, percentage_of_diagonal_to_add=None):

        points_copy = np.copy(points)
        len_points_shape = len(points_copy.shape) 
        if len_points_shape == 3:
            if points_copy.shape[0] == 3:
                points_copy = np.moveaxis(points_copy, 0, -1)
            assert points_copy.shape[2] == 3
            points_copy = points_copy.reshape(-1, 3)
        elif len_points_shape == 2:
            if points_copy.shape[0] == 3:
                points_copy = np.moveaxis(points_copy, 0, -1)
            assert points_copy.shape[1] == 3
        else:
            raise Exception("Input shape not valid: {}".format(points.shape))
            
        self.min_point = np.min(points_copy, axis=0)[:, np.newaxis]
        self.max_point = np.max(points_copy, axis=0)[:, np.newaxis]
        self._enlarge_bbox(percentage_of_diagonal_to_add)
        self._compute_extent()

    def _compute_extent(self):
        self.extent = self.max_point - self.min_point
        
    def _enlarge_bbox(self, percentage_of_diagonal_to_add):
        if percentage_of_diagonal_to_add is None:
            return 

        diagonal = self.max_point - self.min_point
        diagonal_length = np.linalg.norm(diagonal)
    
        self.min_point = self.min_point - percentage_of_diagonal_to_add * diagonal_length
        self.max_point = self.max_point + percentage_of_diagonal_to_add * diagonal_length

    def get_bbox_as_array(self):
        return np.concatenate((self.min_point, self.extent), axis=1)
    
    def get_bbox_as_line_set(self):
        return BBox.compute_bbox_from_min_point_and_extent(self.min_point, self.extent)

    @staticmethod
    def compute_extent(p_min, p_max):
        return p_max - p_min

    @staticmethod
    def compute_bbox_from_min_point_and_extent(p_min, extent, color=[1, 0, 0]):
        bbox_points = p_min.squeeze() + UnitBBox.unit_bbox_points * extent.squeeze()

        colors = [color for i in range(len(UnitBBox.bbox_edges))]
        bbox = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(bbox_points),
            lines=o3d.utility.Vector2iVector(UnitBBox.bbox_edges),
        )
        bbox.colors = o3d.utility.Vector3dVector(colors)

        return bbox

    @staticmethod
    def compute_bbox_from_min_point_and_max_point(p_min, p_max, color=[1, 0, 0]):
        extent = BBox.compute_extent(p_min, p_max)
        bbox = BBox.compute_bbox_from_min_point_and_extent(p_min, extent, color)
        return bbox

    @staticmethod
    def enlarge_bbox(p_min, p_max, displacement):
        p_min = p_min - displacement
        p_max = p_max + displacement
        return p_min, p_max

    @staticmethod
    def convert_bbox_to_cube(p_min, p_max):
        current_extent = BBox.compute_extent(p_min, p_max)
        cube_extent = np.max(current_extent) * np.ones_like(current_extent)

        delta = cube_extent - current_extent
        half_delta = delta / 2.0

        p_min = p_min - half_delta
        p_max = p_max + half_delta

        return p_min, p_max


def transform_to_noc_space(points, p_min, extent):
    translated_points = np.copy(points) - p_min
    scaled_points = translated_points / extent
    nocs = 2.0 * scaled_points - 1.0
    return nocs


def compute_world_to_noc_transform(p_min, extent):
    T_offset = np.eye(4)
    T_offset[:3, 3] = -p_min.reshape(3)

    T_scale = np.eye(4)
    T_scale[0, 0] = 1.0 / extent[0]
    T_scale[1, 1] = 1.0 / extent[1]
    T_scale[2, 2] = 1.0 / extent[2]

    T_noc = np.eye(4)
    T_noc[0, 0] = 2.0
    T_noc[1, 1] = 2.0
    T_noc[2, 2] = 2.0
    T_noc[:3, 3] = -1.0

    return np.matmul(T_noc, np.matmul(T_scale, T_offset))


def normalize_transformation(R, t, p_min, scale):
    R_normalized = R
    t_normalized = scale * (np.matmul(R, p_min) + t - p_min)
    return R_normalized, t_normalized



def align_vector_to_another(a=np.array([0, 0, 1]), b=np.array([1, 0, 0])):
    """
    Aligns vector a to vector b with axis angle rotation
    """
    if np.array_equal(a, b):
        return None, None
    axis_ = np.cross(a, b)
    axis_ = axis_ / np.linalg.norm(axis_)
    angle = np.arccos(np.dot(a, b))

    return axis_, angle


def normalize(a, axis=-1, order=2):
    """Normalizes a numpy array of points"""
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2 == 0] = 1
    return a / np.expand_dims(l2, axis), l2


def transform_mesh_to_unit_cube(
        mesh, p_min, extent, return_wireframe=False, to_opengl=False, return_only_vertices=False
    ):
    vertices = np.asarray(mesh.vertices)

    # The mesh is given in opengl coordinates.
    # So we'll transform first to vision coordinates and work there.
    vertices_cv = transform_pointcloud_to_opengl_coords(vertices)

    # Move point dimensions to the front
    vertices_cv = np.moveaxis(vertices_cv, -1, 0)
    assert vertices_cv.shape[0] == 3

    # The mesh's coordinates are already in world space
    # So we only need to transform them to noc space
    vertices_noc = transform_to_noc_space(vertices_cv, p_min=p_min, extent=extent)

    # Move point dimensions again to the last axis
    vertices_noc = np.moveaxis(vertices_noc, -1, 0)

    if return_only_vertices:
        return vertices_noc

    # Update mesh vertices
    mesh.vertices = o3d.utility.Vector3dVector(vertices_noc)

    if to_opengl:
        mesh.transform(T_opengl_cv_homogeneous)
    
    # Get a wireframe from the mesh
    if return_wireframe:
        mesh_wireframe = o3d.geometry.LineSet.create_from_triangle_mesh(mesh)
        mesh_wireframe.colors = o3d.utility.Vector3dVector(
            np.array([0.5, 0.1, 0.0]) * np.ones_like(vertices_noc)
        )
        # mesh_wireframe.paint_uniform_color([0.5, 0.1, 0.0])
        return mesh_wireframe

    mesh.compute_vertex_normals()

    return mesh


def load_and_transform_mesh_to_unit_cube(
        mesh_path, p_min, extent, return_wireframe=False, to_opengl=False
    ):
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    return transform_mesh_to_unit_cube(mesh, p_min, extent, return_wireframe, to_opengl)
    

def rotate_around_axis(obj, axis_name, angle):
    if axis_name == "x":
        axis = np.array([1, 0, 0])[:, None]
    if axis_name == "y":
        axis = np.array([0, 1, 0])[:, None]
    if axis_name == "z":
        axis = np.array([0, 0, 1])[:, None]

    axis_angle = axis * angle
    R = o3d.geometry.get_rotation_matrix_from_axis_angle(axis_angle)
    obj.rotate(R, center=origin)
    return obj
