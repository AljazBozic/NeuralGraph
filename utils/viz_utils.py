import os, sys
import numpy as np
import torch
import open3d as o3d

from . import pcd_utils


class Colors():
    red   = [0.8, 0.2, 0]
    green = [0, 0.7, 0.2]
    blue  = [0, 0, 1]
    gold  = [1, 0.706, 0]
    greenish  = [0, 0.8, 0.506]


def visualize_point_tensor(
        points_list, R, t,
        colors_list=None, 
        compute_bbox_list=None, 
        additional_pcds=[],
        exit_after=False,
        convert_to_opengl_coords=True
    ):

    assert len(points_list) == len(colors_list) == len(compute_bbox_list)

    # World frame
    referece_frame = create_frame(size=1.0)
    additional_pcds.append(referece_frame)

    # camera frame
    camera_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=1.0, origin=[0, 0, 0]
    )
    camera_frame.rotate(R, pcd_utils.origin)
    camera_frame.translate(t, relative=True)
    additional_pcds.append(camera_frame)
    
    # Unit bbox
    unit_bbox = create_unit_bbox()
    additional_pcds.append(unit_bbox)

    # Go over list of numpy arrays and convert them to o3d.geometry.PointClouds 
    # (maybe also create bboxes around them)
    pcds = []
    bboxes = []
    for i, points in enumerate(points_list):
        if torch.is_tensor(points):
            points_np = points.cpu().numpy()
        elif isinstance(points, type(np.empty(0))):
            points_np = points
        
        if len(points_np.shape) == 3:
            # we then assume the first dimension is the batch_size
            points_np = points_np.squeeze(axis=0)
        
        if points_np.shape[1] > points_np.shape[0] and points_np.shape[0] == 3:
            points_np = np.moveaxis(points_np, 0, -1) # [N, 3]

        # transform to opengl coordinates
        if convert_to_opengl_coords:
            points_np = pcd_utils.transform_pointcloud_to_opengl_coords(points_np)
    
        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points_np))
        if colors_list is not None:
            if colors_list[i] is not None:
                color_np = colors_list[i] * np.ones_like(points_np)
                pcd.colors = o3d.utility.Vector3dVector(color_np)
        pcds.append(pcd)

        if compute_bbox_list is not None:
            if compute_bbox_list[i]:
                bbox = pcd_utils.BBox(points_np)
                bboxes.append(bbox.get_bbox_as_line_set())

    # sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.05)
    # sphere = sphere.translate(np.array([0, -1, 0]), relative=True)
    # sphere.paint_uniform_color([1.0, 0.0, 0.0])
    # additional_pcds.append(sphere)

    # sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.05)
    # sphere = sphere.translate(np.array([0, 0, 1]), relative=True)
    # sphere.paint_uniform_color([1.0, 0.0, 0.0])
    # additional_pcds.append(sphere)

    # transform also additional_pcds if necessary
    if convert_to_opengl_coords:
        for additional_pcd in additional_pcds:
            additional_pcd.transform(pcd_utils.T_opengl_cv_homogeneous)

    o3d.visualization.draw_geometries([*additional_pcds, *pcds, *bboxes])

    if exit_after:
        exit()


def create_unit_bbox():
    # unit bbox
    unit_bbox = pcd_utils.BBox.compute_bbox_from_min_point_and_max_point(
        np.array([-1, -1, -1]), np.array([1, 1, 1])
    )
    return unit_bbox


def create_frame(size=1.0, origin=[0, 0, 0]):
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=size, origin=origin
    )
    return frame


def create_lines_from_start_and_end_points(start_points, end_points, color=[201/255, 177/255, 14/255]):

    if start_points.shape[1] > start_points.shape[0] and start_points.shape[0] == 3:
        start_points = start_points.transpose()
        end_points = end_points.transpose()
        
    num_pairs = start_points.shape[0]
    all_points = np.concatenate((start_points, end_points), axis=0)

    lines       = [[i, i + num_pairs] for i in range(0, num_pairs, 1)]
    line_colors = [color for i in range(num_pairs)]
    line_set   = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(all_points),
        lines=o3d.utility.Vector2iVector(lines),
    )
    line_set.colors = o3d.utility.Vector3dVector(line_colors)
    
    return line_set


def create_lines_from_view_vectors(
        view_vectors_original, 
        offsets_original, 
        dist_original, 
        R, t,
        return_geoms=False,
        convert_to_opengl_coords=False
    ):
    view_vectors = np.copy(view_vectors_original)
    offsets      = np.copy(offsets_original)
    dist         = np.copy(dist_original)

    # Move coordinates to the last axis
    view_vectors = np.moveaxis(view_vectors, 0, -1) # [N, 3]
    offsets      = np.moveaxis(offsets, 0, -1)      # [N, 3]

    len_dist_shape = len(dist.shape)
    if len_dist_shape == 1:
        dist = dist[:, np.newaxis]
    else:
        dist = np.moveaxis(dist, 0, -1)             # [N, 1]

    N = offsets.shape[0] # number of points (and lines)

    # Advance along the view_vectors by a distance of "dist"
    end_points = offsets + view_vectors * dist

    # Concatenate offsets and end_points into one array
    points = np.concatenate((offsets, end_points), axis=0)

    # Compute list of edges between offsets and end_points
    lines       = [[i, i + N] for i in range(0, N, 1)]
    line_colors = [[201/255, 177/255, 14/255] for i in range(N)]
    line_set   = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points),
        lines=o3d.utility.Vector2iVector(lines),
    )
    line_set.colors = o3d.utility.Vector3dVector(line_colors)

    # Offsets PointCloud
    offsets_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(offsets))
    offsets_pcd.paint_uniform_color(Colors.red)
    
    # End points PointCloud
    end_points_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(end_points))
    end_points_pcd.paint_uniform_color(Colors.green)
    
    # Concatenate PointClouds
    pcds = [offsets_pcd, end_points_pcd]

    # Convert to opengl coordinates if necessary
    if not return_geoms or convert_to_opengl_coords:
        offsets_pcd.transform(pcd_utils.T_opengl_cv_homogeneous)
        end_points_pcd.transform(pcd_utils.T_opengl_cv_homogeneous)
        line_set.transform(pcd_utils.T_opengl_cv_homogeneous)

    if return_geoms:
        return line_set, pcds 
    else:
        # camera frame
        camera_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
                size=1.0, origin=[0, 0, 0]
        )
        camera_frame.rotate(R, pcd_utils.origin)
        camera_frame.translate(t, relative=True)
        camera_frame.rotate(pcd_utils.T_opengl_cv, pcd_utils.origin) # convert to opengl coordinates for visualization
        
        o3d.visualization.draw_geometries([camera_frame, *pcds, line_set])
        exit()
      

def viz_and_exit(pcd_list):
    o3d.visualization.draw_geometries(pcd_list)
    exit()


def visualize_mesh(mesh_path):
    # world frame
    world_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=1.0, origin=[0, 0, 0]
    )

    mesh = o3d.io.read_triangle_mesh(mesh_path)
    o3d.visualization.draw_geometries([world_frame, mesh])


def visualize_grid(points_list, colors=None, exit_after=True):
    # world frame
    world_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=1.5, origin=[0, 0, 0]
    )
    world_frame = pcd_utils.rotate_around_axis(world_frame, axis_name="x", angle=-np.pi) 
        
    pcds = []
    for i, points in enumerate(points_list):
        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.moveaxis(points, 0, -1)))
        pcd = pcd_utils.rotate_around_axis(pcd, "x", np.pi)
        if colors:
            pcd.paint_uniform_color(colors[i])
        pcds.append(pcd)
    o3d.visualization.draw_geometries([world_frame, *pcds])
    if exit_after: exit()


def visualize_sphere():
    import marching_cubes as mcubes
    from utils.sdf_utils import sphere_tsdf

    # Extract sphere with Marching cubes.
    dim = 20

    # Extract the 0-isosurface.
    X, Y, Z = np.meshgrid(np.arange(-1, 1, 2.0 / dim), np.arange(-1, 1, 2.0 / dim), np.arange(-1, 1, 2.0 / dim))
    sdf = sphere_tsdf(X, Y, Z)

    vertices, triangles = mcubes.marching_cubes(sdf, 0)

    # Convert extracted surface to o3d mesh.
    mesh_sphere = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(vertices), o3d.utility.Vector3iVector(triangles))
    mesh_sphere.compute_vertex_normals()

    o3d.visualization.draw_geometries([mesh_sphere])


def merge_line_sets(line_sets):
  # Compute total number of vertices and faces.
  num_points = 0
  num_lines = 0
  num_line_colors = 0
  for i in range(len(line_sets)):
    num_points += np.asarray(line_sets[i].points).shape[0]
    num_lines += np.asarray(line_sets[i].lines).shape[0]
    num_line_colors += np.asarray(line_sets[i].colors).shape[0]

  # Merge points and faces.
  points = np.zeros((num_points, 3), dtype=np.float64)
  lines = np.zeros((num_lines, 2), dtype=np.int32)
  line_colors = np.zeros((num_line_colors, 3), dtype=np.float64)

  vertex_offset = 0
  line_offset = 0
  vertex_color_offset = 0
  for i in range(len(line_sets)):
    current_points = np.asarray(line_sets[i].points)
    current_lines = np.asarray(line_sets[i].lines)
    current_line_colors = np.asarray(line_sets[i].colors)

    points[vertex_offset:vertex_offset + current_points.shape[0]] = current_points
    lines[line_offset:line_offset + current_lines.shape[0]] = current_lines + vertex_offset
    line_colors[vertex_color_offset:vertex_color_offset + current_line_colors.shape[0]] = current_line_colors

    vertex_offset += current_points.shape[0]
    line_offset += current_lines.shape[0]
    vertex_color_offset += current_line_colors.shape[0]

  # Create a merged line set object.
  line_set = o3d.geometry.LineSet(o3d.utility.Vector3dVector(points), o3d.utility.Vector2iVector(lines))
  line_set.colors = o3d.utility.Vector3dVector(line_colors)
  return line_set


def merge_meshes(meshes):
  # Compute total number of vertices and faces.
  num_vertices = 0
  num_triangles = 0
  num_vertex_colors = 0
  for i in range(len(meshes)):
    num_vertices += np.asarray(meshes[i].vertices).shape[0]
    num_triangles += np.asarray(meshes[i].triangles).shape[0]
    num_vertex_colors += np.asarray(meshes[i].vertex_colors).shape[0]

  # Merge vertices and faces.
  vertices = np.zeros((num_vertices, 3), dtype=np.float64)
  triangles = np.zeros((num_triangles, 3), dtype=np.int32)
  vertex_colors = np.zeros((num_vertex_colors, 3), dtype=np.float64)

  vertex_offset = 0
  triangle_offset = 0
  vertex_color_offset = 0
  for i in range(len(meshes)):
    current_vertices = np.asarray(meshes[i].vertices)
    current_triangles = np.asarray(meshes[i].triangles)
    current_vertex_colors = np.asarray(meshes[i].vertex_colors)

    vertices[vertex_offset:vertex_offset + current_vertices.shape[0]] = current_vertices
    triangles[triangle_offset:triangle_offset + current_triangles.shape[0]] = current_triangles + vertex_offset
    vertex_colors[vertex_color_offset:vertex_color_offset + current_vertex_colors.shape[0]] = current_vertex_colors

    vertex_offset += current_vertices.shape[0]
    triangle_offset += current_triangles.shape[0]
    vertex_color_offset += current_vertex_colors.shape[0]

  # Create a merged mesh object.
  mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(vertices), o3d.utility.Vector3iVector(triangles))
  mesh.paint_uniform_color([1, 0, 0])
  mesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)
  return mesh