import sys, os

import struct
import json
import numpy as np
import torch
import math
import scipy
import scipy.ndimage
import torchvision.transforms.functional as TF


def generate_icosahedron():
    t = (1.0 + math.sqrt(5.0)) / 2.0

    # Vertices
    num_vertices = 12
    vertices = np.zeros((num_vertices, 3), dtype=np.float32)
    vertices[ 0] = np.array([-1.0,  t, 0.0])
    vertices[ 1] = np.array([ 1.0,  t, 0.0])
    vertices[ 2] = np.array([-1.0, -t, 0.0])
    vertices[ 3] = np.array([ 1.0, -t, 0.0])
    vertices[ 4] = np.array([0.0, -1.0,  t])
    vertices[ 5] = np.array([0.0,  1.0,  t])
    vertices[ 6] = np.array([0.0, -1.0, -t])
    vertices[ 7] = np.array([0.0,  1.0, -t])
    vertices[ 8] = np.array([ t, 0.0, -1.0])
    vertices[ 9] = np.array([ t, 0.0,  1.0])
    vertices[10] = np.array([-t, 0.0, -1.0])
    vertices[11] = np.array([-t, 0.0,  1.0])

    for i in range(num_vertices):
        vertices[i] = vertices[i] / np.linalg.norm(vertices[i])

    # Faces
    num_faces = 20
    faces = np.zeros((num_faces, 3), dtype=np.int32)
    faces[ 0] = np.array([0, 11, 5])
    faces[ 1] = np.array([0, 5, 1])
    faces[ 2] = np.array([0, 1, 7])
    faces[ 3] = np.array([0, 7, 10])
    faces[ 4] = np.array([0, 10, 11])
    faces[ 5] = np.array([1, 5, 9])
    faces[ 6] = np.array([5, 11, 4])
    faces[ 7] = np.array([11, 10, 2])
    faces[ 8] = np.array([10, 7, 6])
    faces[ 9] = np.array([7, 1, 8])
    faces[10] = np.array([3, 9, 4])
    faces[11] = np.array([3, 4, 2])
    faces[12] = np.array([3, 2, 6])
    faces[13] = np.array([3, 6, 8])
    faces[14] = np.array([3, 8, 9])
    faces[15] = np.array([4, 9, 5])
    faces[16] = np.array([2, 4, 11])
    faces[17] = np.array([6, 2, 10])
    faces[18] = np.array([8, 6, 7])
    faces[19] = np.array([9, 8, 1])

    return vertices, faces


def subdivide_edge(idx0, idx1, v0, v1, vertex_idx, vertices_map, edge_divisions):
	if (idx0, idx1) in edge_divisions:
		idx01 = edge_divisions[(idx0, idx1)]
		v01 = vertices_map[idx01]
	else:
		v01 = 0.5 * (v0 + v1)
		v01 /= np.linalg.norm(v01)
		idx01 = vertex_idx
		edge_divisions[(idx0, idx1)] = idx01
		vertex_idx += 1
	
	return idx01, v01, vertex_idx


def subdivide_mesh(vertices_in, faces_in):
	num_vertices = vertices_in.shape[0]
	num_faces = faces_in.shape[0]

	vertices_map = {}
	faces_list = []
	edge_divisions = {}

	vertex_idx = num_vertices
	for i in range(num_faces):
		idx0 = faces_in[i, 0]
		idx1 = faces_in[i, 1]
		idx2 = faces_in[i, 2]

		v0 = vertices_in[idx0]
		v1 = vertices_in[idx1]
		v2 = vertices_in[idx2]

		idx01, v01, vertex_idx = subdivide_edge(idx0, idx1, v0, v1, vertex_idx, vertices_map, edge_divisions)
		idx02, v02, vertex_idx = subdivide_edge(idx0, idx2, v0, v2, vertex_idx, vertices_map, edge_divisions)
		idx12, v12, vertex_idx = subdivide_edge(idx1, idx2, v1, v2, vertex_idx, vertices_map, edge_divisions)

		vertices_map[idx0] = v0
		vertices_map[idx1] = v1
		vertices_map[idx2] = v2
		vertices_map[idx01] = v01
		vertices_map[idx12] = v12
		vertices_map[idx02] = v02

		faces_list.append([idx0, idx01, idx02])
		faces_list.append([idx01, idx1, idx12])
		faces_list.append([idx01, idx12, idx02])
		faces_list.append([idx02, idx12, idx2])

	vertices_out = np.zeros((vertex_idx, 3), dtype=np.float32)
	for idx, v in vertices_map.items():
		vertices_out[idx] = v

	faces_out = np.array(faces_list, dtype=np.int32)

	return vertices_out, faces_out


if __name__ == "__main__":
    pass
