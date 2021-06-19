#!/bin/bash
# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -e

mesh_in=$1
outdir=$2
external_root=$3

make_watertight=false

gaps=${external_root}/gaps/bin/x86_64/

# On macos osmesa is not used, on linux it is:
if [[ $(uname -s) == Darwin* ]]
then
  mesa=""
else
  mesa="-mesa"
fi

mkdir -p $outdir || true

mesh_orig=${outdir}/mesh_orig.${mesh_in##*.}
cp $mesh_in $mesh_orig

# Step -1) Compute watertight mesh.
if [ "$make_watertight" = true ] ; then
  mesh_watertight=${outdir}/model_watertight.ply
  tmp_grid="${outdir}/tmp.grd"

  # ${gaps}/msh2df $mesh_orig $tmp_grid -output_mesh $mesh_watertight \
  #   -estimate_sign -spacing 0.005 -v

  ${gaps}/msh2df $mesh_orig $tmp_grid -output_mesh $mesh_watertight \
    -estimate_sign -spacing 0.005 -v -estimate_sign_using_normals

  rm $tmp_grid

else
  mesh_watertight=$mesh_orig
fi

# Step 0) Normalize the mesh before applying all other operations.
mesh=${outdir}/model_normalized.obj

# ${gaps}/msh2msh $mesh_watertight $mesh -scale_by_pca -translate_by_centroid \
#   -scale 0\.25 -debug_matrix ${outdir}/orig_to_gaps.txt

${gaps}/msh2msh $mesh_watertight $mesh -scale_by_pca -translate_by_centroid \
  -scale 0\.35 -debug_matrix ${outdir}/orig_to_gaps.txt

# Step 1) Generate the coarse inside/outside grid:
${gaps}/msh2df $mesh ${outdir}/coarse_grid.grd -bbox -0\.7 -0\.7 -0\.7 0\.7 \
  0\.7 0\.7 -border 0 -spacing 0\.022 -estimate_sign -v

# Step 2) Generate the near surface points:
${gaps}/msh2pts $mesh ${outdir}/nss_points.sdf -near_surface -max_distance \
  0\.04 -num_points 100000 -v # -curvature_exponent 0

# Step 3) Generate the uniform points:
${gaps}/msh2pts $mesh ${outdir}/uniform_points.sdf -uniform_in_bbox -bbox \
  -0\.7 -0\.7 -0\.7 0\.7 0\.7 0\.7 -npoints 100000

# Step 4) Generate the random surface points:
${gaps}/msh2pts $mesh ${outdir}/surface_points.pts -random_surface_points -num_points 100000

# The normalized mesh is no longer needed on disk; we have the transformation,
# so if we need it we can load the original mesh and transform it to the 
# normalized frame.
rm $mesh
