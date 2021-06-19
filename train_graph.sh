#!/bin/sh
GPU=${1:-0}

# Dataset to train on and experiment name.
data="doozy"
experiment="doozy_graph"

CUDA_VISIBLE_DEVICES=${GPU} python train_graph.py --data=${data} --experiment="${experiment}"