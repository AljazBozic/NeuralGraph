import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import tqdm
from joblib import Parallel, delayed
from absl import flags
from absl import app
import glob
import argparse
import open3d as o3d
import subprocess as sp


def process_one(mesh_path, mesh_directory, dataset_directory, skip_existing):
    """Processes a single mesh, adding it to the dataset."""
    name = os.path.basename(mesh_path)
    name, extension = os.path.splitext(name)
    valid_extensions = ['.ply']
    if extension not in valid_extensions:
        raise ValueError(f'File with unsupported extension {extension} found: {f}.'
                         f' Only {valid_extensions} are supported.')
    
    output_dir = f'{dataset_directory}/{name}/'
    scripts_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts")
    external_root_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "external")

    # This is the last file the processing writes, if it already exists the
    # example has already been processed.
    if not skip_existing or not os.path.isfile(f'{output_dir}/uniform_points.sdf'):
        # print(f'{codebase_root_dir}/scripts/process_mesh_local.sh {mesh_path} {dirpath} {external_root_dir}')
        sp.check_output(
            f'{scripts_dir}/process_mesh_local.sh {mesh_path} {output_dir} {external_root_dir}',
            shell=True)
    else:
        print(f'Skipping shell script processing for {output_dir},'
              ' the output already exists.')

    return output_dir


def generate_meshes(mesh_directory, dataset_directory, mesh_format, skip_existing=True, max_threads=-1):
    raw_files = glob.glob(f'{mesh_directory}/*.{mesh_format}')

    if not raw_files:
        raise ValueError(f"Didn't find any {mesh_format} files in {mesh_directory}")

    if mesh_format != "ply":
        print(f"Started mesh conversion from '{mesh_format}' to 'ply'")

        for input_mesh_path in raw_files:
            base_name = os.path.splitext(input_mesh_path)[0]
            output_mesh_path = os.path.join(mesh_directory, base_name + ".ply")
            mesh = o3d.io.read_triangle_mesh(input_mesh_path)
            o3d.io.write_triangle_mesh(output_mesh_path, mesh)

        files = glob.glob(f'{mesh_directory}/*.ply')

    else:
        files = raw_files

    # Make the directories first because it's not threadsafe and also might fail.
    print('Creating directories...')
    if not os.path.isdir(f'{dataset_directory}'):
        os.makedirs(f'{dataset_directory}')

    print('Making dataset...')
    n_jobs = os.cpu_count()
    assert max_threads != 0
    if max_threads > 0:
        n_jobs = max_threads
    output_dirs = Parallel(n_jobs=n_jobs)(
        delayed(process_one)(f, mesh_directory, dataset_directory,
                             skip_existing) for f in tqdm.tqdm(files))
    
    print('Done!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_mesh_dir', action='store', dest='input_mesh_dir', required=True, help='Provide input watertight mesh directory')
    parser.add_argument('--output_data_dir', action='store', dest='output_data_dir', required=True, help='Provide output directory')
    parser.add_argument('--mesh_format', action='store', dest='mesh_format', help='Provide mesh format')
    parser.add_argument('--max_threads', action='store', type=int, dest='max_threads', help='Maximum number of threads to be used (uses all available threads by default)')

    args = parser.parse_args()

    input_mesh_dir = args.input_mesh_dir
    output_data_dir = args.output_data_dir
    mesh_format = args.mesh_format
    max_threads = args.max_threads
    
    if not mesh_format:
        mesh_format = "ply"
    if not max_threads:
        max_threads = -1

    generate_meshes(input_mesh_dir, output_data_dir, mesh_format, max_threads=max_threads)
