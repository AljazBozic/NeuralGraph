#!/bin/bash
input_dir="SPECIFY INPUT DIR"
output_dir="SPECIFY OUTPUT DIR"

echo "Input dir:  ${input_dir}"
echo "Output dir: ${output_dir}"

mkdir -p ${output_dir}

for filename in $input_dir/*.ply; do
# for filename in $input_dir/*.obj; do
    basename=$(basename $filename)

    # ../../external/gaps/bin/x86_64/msh2df "${input_dir}/${basename}" tmp.grd -output_mesh "${output_dir}/${basename}" \
    # -estimate_sign -spacing 0.005 -v -estimate_sign_using_normals
    # rm tmp.grd

    meshlabserver -i "${input_dir}/${basename}" -o "${output_dir}/${basename}" -s screened_poisson.mlx
done
