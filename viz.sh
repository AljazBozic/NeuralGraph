###############################################################
# PARAMS 
###############################################################
# Time step increment
TIME_INC=5 #1

# Grid params
GRID_DIM=128
GRID_NUM_CHUNKS=64

# Graph params
NUM_NEIGHBORS=4
EDGE_WEIGHT_THRESHOLD=0.05

# Dataset
GT_DATA_DIR="out/dataset/doozy"

# Model
# CHECKPOINT_PATH="out/experiments/doozy_graph/checkpoints/00530000_model.pt"
# CHECKPOINT_PATH="out/experiments/doozy_shape/checkpoints/00430000_model.pt"
CHECKPOINT_PATH="out/models/doozy_checkpoint.pt"

###############################################################
# RUN VISUALIZATION 
###############################################################
python viz.py   --checkpoint_path ${CHECKPOINT_PATH} \
                --time_inc ${TIME_INC} \
                --gt_data_dir ${GT_DATA_DIR} \
                --grid_dim ${GRID_DIM} \
                --grid_num_chunks ${GRID_NUM_CHUNKS} \
                --num_neighbors ${NUM_NEIGHBORS} \
                --edge_weight_threshold ${EDGE_WEIGHT_THRESHOLD} \
                --viz_dense_tracking \
                # --viz_only_graph \
                          