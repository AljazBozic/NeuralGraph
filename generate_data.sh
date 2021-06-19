#!/bin/sh
input_mesh_dir="out/meshes/doozy"
output_data_dir="out/dataset/doozy"
max_threads=12

python dataset/generate_dataset.py --input_mesh_dir="${input_mesh_dir}" --output_data_dir="${output_data_dir}" --max_threads=${max_threads}