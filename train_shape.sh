#!/bin/sh
GPU=${1:-0}

# Dataset to train on and experiment name.
# We also need to additionally provide the pre-trained graph model.
data="doozy"
experiment="doozy_shape"
graph_model_path="out/experiments/doozy_graph/checkpoints/00510000_model.pt"

CUDA_VISIBLE_DEVICES=${GPU} python train_shape.py --data=${data} --experiment="${experiment}" --graph_model_path="${graph_model_path}"