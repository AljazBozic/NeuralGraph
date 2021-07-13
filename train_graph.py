import sys, os

import time
import argparse
from datetime import datetime
import torch
from tensorboardX import SummaryWriter
import random
import numpy as np
import signal
import math
import json
from timeit import default_timer as timer
import numpy as np

import config as cfg
from dataset.dataset import MeshDataset as Dataset
from utils import gradient_utils
from utils.time_statistics import TimeStatistics
from nnutils.geometry import augment_grid

from node_sampler.model import NodeSampler
from node_sampler.loss import SamplerLoss
from node_sampler.evaluate import evaluate


def main():
    torch.set_num_threads(cfg.num_threads)
    
    # Make execution deterministic.
    seed = 42
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    # Parse command line arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', action='store', dest='data', help='Provide a subfolder with training data')
    parser.add_argument('--experiment', action='store', dest='experiment', help='Provide an experiment name')

    args = parser.parse_args()

    # Train set on which to actually train.
    data = args.data

    # Experiment.
    experiment_name = args.experiment

    if cfg.initialize_from_other:
        print("Will initialize from provided checkpoint")
    else:
        print("Will train from scratch")
    print()

    # Print hyperparameters.
    cfg.print_hyperparams(data, experiment_name)

    print()

    #####################################################################################
    # Creating tf writer and folders 
    #####################################################################################
    data_dir = os.path.join(cfg.data_root_dir, data)
    experiment_dir = os.path.join(cfg.experiments_dir, experiment_name)
    checkpoints_dir = None

    # Writer initialization.
    log_dir = os.path.join(experiment_dir, "tf_run")

    train_log_dir = log_dir + "/" + data
    if not os.path.exists(train_log_dir): os.makedirs(train_log_dir, exist_ok=True)
    train_writer = SummaryWriter(train_log_dir)

    # Creation of model output directory.
    checkpoints_dir = os.path.join(experiment_dir, "checkpoints")   
    if not os.path.exists(checkpoints_dir): os.mkdir(checkpoints_dir)

    # We count the execution time between evaluations.
    time_statistics = TimeStatistics()

    #####################################################################################
    # Create datasets and dataloaders
    #####################################################################################
    train_dataset = Dataset(
        data_dir, cfg.graph_num_point_samples, 
        cache_data=cfg.cache_data, use_augmentation=True
    )

    effective_batch_size = cfg.graph_batch_size // 2

    train_dataloader = torch.utils.data.DataLoader(
        dataset=train_dataset, batch_size=effective_batch_size, shuffle=cfg.shuffle, num_workers=cfg.num_worker_threads, pin_memory=False
    )

    print("Num. training samples: {0}".format(len(train_dataset)))
    print()

    if len(train_dataset) < effective_batch_size:
        print()
        print("Reduce the batch_size, since we only have {} training samples but you indicated a batch_size of {}".format(
            len(train_dataset), effective_batch_size)
        )
        exit()

    #####################################################################################
    # Initializing: model, criterion, optimizer...
    #####################################################################################
    # Set the iteration number
    iteration_number = 0

    # Model
    model = NodeSampler().cuda()
    
    # Maybe load pretrained model
    if cfg.initialize_from_other:
        # Initialize with other model
        print(f"Initializing from model: {cfg.saved_model_path}")
        print()

        iteration_number = cfg.saved_model_iteration + 1

        # Load pretrained dict
        pretrained_dict = torch.load(cfg.saved_model_path)
        model.load_state_dict(pretrained_dict)

    # Criterion.
    criterion = SamplerLoss()

    # Count parameters.
    print()

    n_all_model_params = int(sum([np.prod(p.size()) for p in model.parameters()]))
    n_trainable_model_params = int(sum([np.prod(p.size()) for p in filter(lambda p: p.requires_grad, model.parameters())]))
    print("Number of parameters: {0} / {1}".format(n_trainable_model_params, n_all_model_params))

    n_all_model_params_encoder = int(sum([np.prod(p.size()) for p in model.encoder.parameters()]))
    n_trainable_model_params_encoder = int(sum([np.prod(p.size()) for p in filter(lambda p: p.requires_grad, model.encoder.parameters())]))
    print("\tEncoder: {0} / {1}".format(n_trainable_model_params_encoder, n_all_model_params_encoder))

    n_all_model_params_position_mlp = int(sum([np.prod(p.size()) for p in model.position_mlp.parameters()]))
    n_trainable_model_params_position_mlp = int(sum([np.prod(p.size()) for p in filter(lambda p: p.requires_grad, model.position_mlp.parameters())]))
    print("\tPosition MLP: {0} / {1}".format(n_trainable_model_params_position_mlp, n_all_model_params_position_mlp))

    n_all_model_params_rotation_mlp = int(sum([np.prod(p.size()) for p in model.rotation_mlp.parameters()]))
    n_trainable_model_params_rotation_mlp = int(sum([np.prod(p.size()) for p in filter(lambda p: p.requires_grad, model.rotation_mlp.parameters())]))
    print("\tRotation MLP: {0} / {1}".format(n_trainable_model_params_rotation_mlp, n_all_model_params_rotation_mlp))

    print()

    # Set up optimizer.
    optimizer = torch.optim.Adam(model.parameters(), cfg.graph_learning_rate, weight_decay=cfg.weight_decay)
    
    # Initialize training.
    train_writer.add_text("hyperparams",
                    "Training data: " + data
                    + ",\nBatch size: " + str(cfg.graph_batch_size)
                    + ",\nLearning rate:" + str(cfg.graph_learning_rate)
                    + ",\nEpochs: " + str(cfg.epochs))

    # Initialize non-constant lambdas.
    lambda_affinity_rel_dist_init   = cfg.lambda_affinity_rel_dist
    lambda_affinity_abs_dist_init   = cfg.lambda_affinity_abs_dist
    lambda_unique_neighbor_init     = cfg.lambda_unique_neighbor
    lambda_surface_consistency_init = cfg.lambda_surface_consistency
    
    if cfg.interval_step is not None:
        if cfg.lambda_affinity_rel_dist is not None:
            cfg.lambda_affinity_rel_dist    = min(lambda_affinity_rel_dist_init * cfg.lambda_affinity_rel_dist_f**(iteration_number // cfg.interval_step), cfg.lambda_affinity_rel_dist_max)
        if cfg.lambda_affinity_abs_dist is not None:
            cfg.lambda_affinity_abs_dist    = min(lambda_affinity_abs_dist_init * cfg.lambda_affinity_abs_dist_f**(iteration_number // cfg.interval_step), cfg.lambda_affinity_abs_dist_max)
        if cfg.lambda_unique_neighbor is not None:
            cfg.lambda_unique_neighbor          = min(lambda_unique_neighbor_init * cfg.lambda_unique_neighbor_f**(iteration_number // cfg.interval_step), cfg.lambda_unique_neighbor_max)
        if cfg.lambda_surface_consistency is not None:
            cfg.lambda_surface_consistency  = min(lambda_surface_consistency_init * cfg.lambda_surface_consistency_f**(iteration_number // cfg.interval_step), cfg.lambda_surface_consistency_max)

    # Execute training.
    complete_cycle_start = timer()

    try:
        for epoch in range(0, cfg.epochs):
            model.train()
            for i, data in enumerate(train_dataloader):
                # We increase affinity weights after certain iteration.
                if cfg.interval_step is not None and iteration_number % cfg.interval_step == 0:
                    if cfg.lambda_affinity_rel_dist is not None:
                        cfg.lambda_affinity_rel_dist    = min(lambda_affinity_rel_dist_init * cfg.lambda_affinity_rel_dist_f**(iteration_number // cfg.interval_step), cfg.lambda_affinity_rel_dist_max)
                    if cfg.lambda_affinity_abs_dist is not None:
                        cfg.lambda_affinity_abs_dist    = min(lambda_affinity_abs_dist_init * cfg.lambda_affinity_abs_dist_f**(iteration_number // cfg.interval_step), cfg.lambda_affinity_abs_dist_max)
                    if cfg.lambda_unique_neighbor is not None:
                        cfg.lambda_unique_neighbor      = min(lambda_unique_neighbor_init * cfg.lambda_unique_neighbor_f**(iteration_number // cfg.interval_step), cfg.lambda_unique_neighbor_max)
                    if cfg.lambda_surface_consistency is not None:
                        cfg.lambda_surface_consistency  = min(lambda_surface_consistency_init * cfg.lambda_surface_consistency_f**(iteration_number // cfg.interval_step), cfg.lambda_surface_consistency_max)

                    print()
                    print("########################## LAMBDA UPDATE ##########################")
                    print(f"lambda_affinity_rel_dist:   {cfg.lambda_affinity_rel_dist}")
                    print(f"lambda_affinity_abs_dist:   {cfg.lambda_affinity_abs_dist}")
                    print(f"lambda_unique_neighbor:     {cfg.lambda_unique_neighbor}")
                    print(f"lambda_surface_consistency: {cfg.lambda_surface_consistency}")
                    print("###################################################################")

                #####################################################################################
                #################################### Evaluation #####################################
                #####################################################################################

                if cfg.do_evaluation and iteration_number % cfg.evaluation_frequency == 0:
                    model.eval()

                    eval_start = timer()

                    # Compute train metrics.
                    num_samples = cfg.num_samples_eval
                    num_eval_batches = math.ceil(num_samples / cfg.graph_batch_size) # We evaluate on approximately 1000 samples.

                    train_losses, train_metrics = evaluate(model, criterion, train_dataloader, num_eval_batches)

                    # Save current model checkpoint.
                    output_checkpoint_path = os.path.join(checkpoints_dir, "{0:08d}_model.pt".format(iteration_number))
                    torch.save(model.state_dict(), output_checkpoint_path)

                    train_writer.add_scalar('Loss/Loss',                    train_losses["total"],                  iteration_number)
                    train_writer.add_scalar('Loss/Uniform',                 train_losses["uniform"],                iteration_number)
                    train_writer.add_scalar('Loss/NearSurface',             train_losses["near_surface"],           iteration_number)
                    train_writer.add_scalar('Loss/NodeCenter',              train_losses["node_center"],            iteration_number)
                    train_writer.add_scalar('Loss/AffinityRelative',        train_losses["affinity_rel"],           iteration_number)
                    train_writer.add_scalar('Loss/AffinityAbsolute',        train_losses["affinity_abs"],           iteration_number)
                    train_writer.add_scalar('Loss/UniqueNeighbor',          train_losses["unique_neighbor"],        iteration_number)
                    train_writer.add_scalar('Loss/ViewpointPosition',       train_losses["viewpoint_position"],     iteration_number)
                    train_writer.add_scalar('Loss/ViewpointScale',          train_losses["viewpoint_scale"],        iteration_number)
                    train_writer.add_scalar('Loss/ViewpointConstant',       train_losses["viewpoint_constant"],     iteration_number)
                    train_writer.add_scalar('Loss/ViewpointRotation',       train_losses["viewpoint_rotation"],     iteration_number)
                    train_writer.add_scalar('Loss/SurfaceConsistency',      train_losses["surface_consistency"],    iteration_number)

                    print()
                    print()
                    print("Epoch number {0}, Iteration number {1}".format(epoch, iteration_number))
                    print("{:<50} {}".format("Current Train Loss   TOTAL",                  train_losses["total"]))
                    print("{:<50} {}".format("Current Train Loss   UNIFORM",                train_losses["uniform"]))
                    print("{:<50} {}".format("Current Train Loss   NEAR_SURFACE",           train_losses["near_surface"]))
                    print("{:<50} {}".format("Current Train Loss   NODE_CENTER",            train_losses["node_center"]))
                    print("{:<50} {}".format("Current Train Loss   AFFINITY_RELATIVE",      train_losses["affinity_rel"]))
                    print("{:<50} {}".format("Current Train Loss   AFFINITY_ABSOLUTE",      train_losses["affinity_abs"]))
                    print("{:<50} {}".format("Current Train Loss   UNIQUE_NEIGHBOR",        train_losses["unique_neighbor"]))
                    print("{:<50} {}".format("Current Train Loss   VIEWPOINT_POSITION",     train_losses["viewpoint_position"]))
                    print("{:<50} {}".format("Current Train Loss   VIEWPOINT_SCALE",        train_losses["viewpoint_scale"]))
                    print("{:<50} {}".format("Current Train Loss   VIEWPOINT_CONSTANT",     train_losses["viewpoint_constant"]))
                    print("{:<50} {}".format("Current Train Loss   VIEWPOINT_ROTATION",     train_losses["viewpoint_rotation"]))
                    print("{:<50} {}".format("Current Train Loss   SURFACE_CONSISTENCY",    train_losses["surface_consistency"]))
                    print()

                    time_statistics.eval_duration = timer() - eval_start

                    # We compute the time of IO as the complete time, subtracted by all processing time.
                    time_statistics.io_duration += (timer() - complete_cycle_start - time_statistics.train_duration - time_statistics.eval_duration)
                    
                    # Set CUDA_LAUNCH_BLOCKING=1 environmental variable for reliable timings. 
                    print("Cycle duration (s): {0:3f} (IO: {1:3f}, TRAIN: {2:3f}, EVAL: {3:3f})".format(
                        timer() - time_statistics.start_time, time_statistics.io_duration, time_statistics.train_duration, time_statistics.eval_duration
                    ))
                    print("FORWARD: {0:3f}, LOSS: {1:3f}, BACKWARD: {2:3f}".format(
                        time_statistics.forward_duration, time_statistics.loss_eval_duration, time_statistics.backward_duration
                    ))                       

                    print()

                    time_statistics = TimeStatistics()
                    complete_cycle_start = timer()

                    sys.stdout.flush()

                    model.train()
                
                sys.stdout.write("\r############# Train iteration: {0} (of Epoch {1}) || Experiment: {2}".format(
                    iteration_number, epoch, experiment_name)
                )
                sys.stdout.flush()

                #####################################################################################
                ####################################### Train #######################################
                #####################################################################################
                
                #####################################################################################
                # Data loading
                #####################################################################################
                uniform_samples, near_surface_samples, surface_samples, grid, world2grid, world2orig, rotated2gaps, bbox_lower, bbox_upper, sample_idx = data

                uniform_samples         = train_dataset.unpack(uniform_samples).cuda()
                near_surface_samples    = train_dataset.unpack(near_surface_samples).cuda()
                surface_samples         = train_dataset.unpack(surface_samples).cuda()
                grid                    = train_dataset.unpack(grid).cuda()
                world2grid              = train_dataset.unpack(world2grid).cuda()
                world2orig              = train_dataset.unpack(world2orig).cuda()
                rotated2gaps            = train_dataset.unpack(rotated2gaps).cuda()
                bbox_lower              = train_dataset.unpack(bbox_lower).cuda()
                bbox_upper              = train_dataset.unpack(bbox_upper).cuda()

                # Compute augmented sdfs.
                with torch.no_grad():
                    sdfs = augment_grid(grid, world2grid, rotated2gaps)

                train_batch_start = timer()

                #####################################################################################
                # Forward pass.
                #####################################################################################
                train_batch_forward_pass = timer()

                try:
                    embedding_pred, source_idxs, target_idxs, pair_distances, pair_weights, affinity_matrix = model(sdfs)
                except RuntimeError as e:
                    print("Runtime error!", e)
                    print("Exiting...")
                    exit()
                
                time_statistics.forward_duration += (timer() - train_batch_forward_pass)

                #####################################################################################
                # Loss.
                #####################################################################################
                train_batch_loss_eval = timer()

                # Compute Loss
                loss = criterion(
                    embedding_pred, uniform_samples, near_surface_samples, surface_samples,
                    grid, world2grid, world2orig, rotated2gaps, bbox_lower, bbox_upper,  
                    source_idxs, target_idxs, pair_distances, pair_weights, affinity_matrix
                )
            
                if cfg.detect_anomaly:
                    if not np.isfinite(loss.item()):
                        print("Non-finite loss: {}".format(loss.item()))
                        exit()

                time_statistics.loss_eval_duration += (timer() - train_batch_loss_eval)

                #####################################################################################
                # Backprop.
                #####################################################################################
                train_batch_backprop = timer()

                optimizer.zero_grad()
                loss.backward()

                # Check gradients.
                if cfg.detect_anomaly:
                    weight_sum, grad_sum = gradient_utils.compute_forward_pass_info(model)

                    if torch.isfinite(grad_sum):
                        optimizer.step()
                    else:
                        print("Invalid gradient: {0}".format(grad_sum))
                        exit()
                else:
                    optimizer.step()

                time_statistics.backward_duration += (timer() - train_batch_backprop)
                time_statistics.train_duration += (timer() - train_batch_start)

                iteration_number = iteration_number + 1

    except (KeyboardInterrupt, TypeError, ConnectionResetError) as err:
        train_writer.close()
        raise err

    train_writer.close()

    print()
    print("I'm done")


if __name__=="__main__":
    main()