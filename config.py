import os

#####################################################################################################################
# DATA OPTIONS
#####################################################################################################################
data_root_dir    = "./out/dataset"
experiments_dir  = "./out/experiments"

#####################################################################################################################
# DATA LOADING OPTIONS
#####################################################################################################################
num_worker_threads = 0 # 0 means that the base thread does all the job (that makes sense when hdf5 is already loaded into memory).
num_threads = 4

num_samples_eval = 100
cache_data = True

#####################################################################################################################
# MODEL INFO
#####################################################################################################################
# Pretrained model
initialize_from_other  = False
saved_model_path = ""
saved_model_iteration = 0

# Freeze model
freeze_node_encoder = False
freeze_scale_estimator = False
freeze_position_estimator = False
freeze_rotation_estimator = False
freeze_affinity = False
freeze_surface_mlp = False

#####################################################################################################################
# GENERAL OPTIONS
#####################################################################################################################
# Do evaluation
do_evaluation = True

# Shuffle batch
shuffle = True

# Detect anomalies, such as when gradients explode
detect_anomaly = False

#####################################################################################################################
# NODE OPTIONS
#####################################################################################################################
num_nodes = 100
position_length = 3 + 1
scale_length = 3
rotation_length = 3

graph_num_point_samples = 3000
shape_num_point_samples = 1500
interior_point_weight = 10.0
soft_transfer_scale = 100.0
level_set = -0.07
num_neighbors = 2
use_constants = True
scaling_type = "isotropic" # ["isotropic", "anisotropic", "none"]
aggregate_coverage_with_max = False

#####################################################################################################################
# MULTI-SDF OPTIONS
#####################################################################################################################
# SDF settings
truncation = 0.1

# Model settings
num_features = 32
use_tanh = True

# Descriptors
descriptor_dim = 32

#####################################################################################################################
# TRAINING OPTIONS
#####################################################################################################################
graph_batch_size = 16
shape_batch_size = 4
evaluation_frequency = 5000
epochs = 1000000
graph_learning_rate = 5e-5
shape_learning_rate = 5e-4
weight_decay = 0
interval_step = 50000

# Losses
lambda_geometry = 1.0
lambda_sampling_uniform = 1.0
lambda_sampling_near_surface = 0.1
lambda_sampling_node_center = 1.0
lambda_viewpoint_position = 10.0
lambda_viewpoint_scale = 1.0
lambda_viewpoint_constant = 1.0
lambda_viewpoint_rotation = 1e-4
lambda_surface_consistency = 1e-6
lambda_surface_consistency_f = 10.0
lambda_surface_consistency_max = 1000.0
lambda_affinity_rel_dist = 0.1
lambda_affinity_rel_dist_f = 10.0
lambda_affinity_rel_dist_max = 10000.0
lambda_affinity_abs_dist = 0.1
lambda_affinity_abs_dist_f = 10.0
lambda_affinity_abs_dist_max = 1.0
lambda_unique_neighbor = 1e-8
lambda_unique_neighbor_f = 10.0
lambda_unique_neighbor_max = 1e-3

#####################################################################################################################
# PRINT HYPERPARAMS
#####################################################################################################################
def print_hyperparams(data, experiment_name):
    print()
    print("HYPERPARAMETERS:")
    print()

    print("\tDATA:                        ", data)
    print("\tEXPERIMENT:                  ", experiment_name)

    print()
    print("\tnum_worker_threads           ", num_worker_threads)
    print("\tnum_threads                  ", num_threads)

    print()
    print("################# NODE OPTIONS #######################")
    print("\tgraph_num_point_samples      ", graph_num_point_samples)
    print("\tshape_num_point_samples      ", shape_num_point_samples)
    print("\tuse_constants                ", use_constants)
    print("\tscaling_type                 ", scaling_type)

    print()
    print("############### TRAINING OPTIONS #####################")
    print("\tgraph_batch_size             ", graph_batch_size)
    print("\tshape_batch_size             ", shape_batch_size)
    print("\tevaluation_frequency         ", evaluation_frequency)
    print("\tgraph_learning_rate          ", graph_learning_rate)
    print("\tshape_learning_rate          ", shape_learning_rate)
    print()
    print("\tlambda_geometry              ", lambda_geometry)
    print("\tlambda_sampling_uniform      ", lambda_sampling_uniform)
    print("\tlambda_sampling_near_surface ", lambda_sampling_near_surface)
    print("\tlambda_sampling_node_center  ", lambda_sampling_node_center)
    print("\tlambda_viewpoint_position    ", lambda_viewpoint_position)
    print("\tlambda_viewpoint_scale       ", lambda_viewpoint_scale)
    print("\tlambda_viewpoint_constant    ", lambda_viewpoint_constant)
    print("\tlambda_viewpoint_rotation    ", lambda_viewpoint_rotation)
    print("\tlambda_surface_cons          ", lambda_surface_consistency)
    print("\tlambda_surface_cons_f        ", lambda_surface_consistency_f)
    print("\tlambda_surface_cons_max      ", lambda_surface_consistency_max)
    print("\tlambda_affinity_rel_dist     ", lambda_affinity_rel_dist)
    print("\tlambda_affinity_rel_dist_f   ", lambda_affinity_rel_dist_f)
    print("\tlambda_affinity_rel_dist_max ", lambda_affinity_rel_dist_max)
    print("\tlambda_affinity_abs_dist     ", lambda_affinity_abs_dist)
    print("\tlambda_affinity_abs_dist_f   ", lambda_affinity_abs_dist_f)
    print("\tlambda_affinity_abs_dist_max ", lambda_affinity_abs_dist_max)
    print("\tlambda_unique_neighbor       ", lambda_unique_neighbor)
    print("\tlambda_unique_neighbor_f     ", lambda_unique_neighbor_f)
    print("\tlambda_unique_neighbor_max   ", lambda_unique_neighbor_max)
