import sys, os

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

import config as cfg
from dataset.dataset import MeshDataset as Dataset
from utils import gradient_utils
from utils.time_statistics import TimeStatistics
from nnutils.geometry import augment_grid

from multi_sdf.model import MultiSDF
from multi_sdf.loss import LossSDF
from multi_sdf.evaluate import evaluate


def main():
    torch.set_num_threads(cfg.num_threads)
    torch.backends.cudnn.benchmark = False

    # Parse command line arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', action='store', dest='data', help='Provide a subfolder with training data')
    parser.add_argument('--experiment', action='store', dest='experiment', help='Provide an experiment name')
    parser.add_argument('--graph_model_path', action='store', dest='graph_model_path', help='Provide a path to pre-trained graph model')

    args = parser.parse_args()

    # Train set on which to actually train
    data = args.data

    # Experiment
    experiment_name = args.experiment

    # Path to graph model.
    graph_model_path = args.graph_model_path

    if cfg.initialize_from_other:
        print("Will initialize from provided checkpoint")
    else:
        print("Will train from scratch")
    print()

    # Print hyperparameters
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

    # Creation of model output directories.
    checkpoints_dir = os.path.join(experiment_dir, "checkpoints")   
    if not os.path.exists(checkpoints_dir): os.mkdir(checkpoints_dir)

    # We count the execution time between evaluations.
    time_statistics = TimeStatistics()

    #####################################################################################
    # Create datasets and dataloaders
    #####################################################################################
    # Augmentation is currently not supported for shape training.
    train_dataset = Dataset(
        data_dir, cfg.shape_num_point_samples, 
        cache_data=cfg.cache_data, use_augmentation=False
    )

    train_dataloader = torch.utils.data.DataLoader(
        dataset=train_dataset, batch_size=cfg.shape_batch_size, shuffle=cfg.shuffle, num_workers=cfg.num_worker_threads, pin_memory=False
    )

    print("Num. training samples: {0}".format(len(train_dataset)))
    print()

    if len(train_dataset) < cfg.shape_batch_size:
        print()
        print("Reduce the batch_size, since we only have {} training samples but you indicated a batch_size of {}".format(
            len(train_dataset), cfg.shape_batch_size)
        )
        exit()

    #####################################################################################
    # Initializing: model, criterion, optimizer...
    #####################################################################################
    # Set the iteration number
    iteration_number = 0

    # Canonical model
    model = MultiSDF().cuda()
    
    # Maybe load pretrained model
    if cfg.initialize_from_other:
        # Initialize with other model
        print(f"Initializing from model: {cfg.saved_model_path}")
        print()

        iteration_number = cfg.saved_model_iteration + 1

        # Load pretrained dict
        pretrained_dict = torch.load(cfg.saved_model_path)
        model.load_state_dict(pretrained_dict)

    else:
        ## Always load pretrained sampler
        # First, get model path
        # If provided experiment is already the absolute path, use it:
        assert os.path.exists(graph_model_path)

        print(f"Using graph model from: {graph_model_path}")
        print()

        sampler_pretrained_dict = torch.load(graph_model_path)
        model.node_sampler.load_state_dict(sampler_pretrained_dict)
        
    # Criterion.
    criterion = LossSDF()

    # Count parameters.
    n_all_model_params = int(sum([np.prod(p.size()) for p in model.parameters()]))
    n_trainable_model_params = int(sum([np.prod(p.size()) for p in filter(lambda p: p.requires_grad, model.parameters())]))
    print("Number of parameters: {0} / {1}".format(n_trainable_model_params, n_all_model_params))
    print()

    # Set up optimizer.
    optimizer = torch.optim.Adam(model.parameters(), cfg.shape_learning_rate, weight_decay=cfg.weight_decay)
    
    # Initialize training.
    train_writer.add_text("hyperparams",
                    "Training data: " + data
                    + ",\nBatch size: " + str(cfg.shape_batch_size)
                    + ",\nLearning rate:" + str(cfg.shape_learning_rate)
                    + ",\nEpochs: " + str(cfg.epochs))

    # Execute training.
    complete_cycle_start = timer()

    try:
        for epoch in range(0, cfg.epochs):
            model.train()
            for i, data in enumerate(train_dataloader):
                #####################################################################################
                #################################### Evaluation #####################################
                #####################################################################################
                if cfg.do_evaluation and iteration_number % cfg.evaluation_frequency == 0:
                    model.eval()

                    eval_start = timer()

                    # Compute train metrics.
                    num_samples = cfg.num_samples_eval
                    num_eval_batches = math.ceil(num_samples / cfg.shape_batch_size) # We evaluate on approximately 1000 samples.

                    train_losses, train_metrics = evaluate(model, criterion, train_dataloader, num_eval_batches)

                    # Save current model checkpoint.
                    output_checkpoint_path = os.path.join(checkpoints_dir, "{0:08d}_model.pt".format(iteration_number))
                    torch.save(model.state_dict(), output_checkpoint_path)

                    train_writer.add_scalar('Loss/Loss',                train_losses["total"],                  iteration_number)
                    train_writer.add_scalar('Loss/GeometryUniform',     train_losses["geometry_uniform"],       iteration_number)
                    train_writer.add_scalar('Loss/GeometryNearSurface', train_losses["geometry_near_surface"],  iteration_number)

                    print()
                    print()
                    print("Epoch number {0}, Iteration number {1}".format(epoch, iteration_number))
                    print("{:<50} {}".format("Current Train Loss   TOTAL",                  train_losses["total"]))
                    print("{:<50} {}".format("Current Train Loss   GEOMETRY_UNIFORM",       train_losses["geometry_uniform"]))
                    print("{:<50} {}".format("Current Train Loss   GEOMETRY_NEAR_SURFACE",  train_losses["geometry_near_surface"]))
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
                grid                    = train_dataset.unpack(grid).cuda()
                world2grid              = train_dataset.unpack(world2grid).cuda()
                rotated2gaps            = train_dataset.unpack(rotated2gaps).cuda()

                # Merge uniform and near surface samples.
                batch_size = uniform_samples.shape[0]

                num_uniform_points = uniform_samples.shape[1]
                num_near_surfaces_points = near_surface_samples.shape[1]
                num_points = num_uniform_points + num_near_surfaces_points

                points = torch.zeros((batch_size, num_points, 3), dtype=uniform_samples.dtype, device=uniform_samples.device)
                points[:, :num_uniform_points, :] = uniform_samples[:, :, :3]
                points[:, num_uniform_points:, :] = near_surface_samples[:, :, :3]

                # Compute augmented sdfs.
                with torch.no_grad():
                    sdfs = augment_grid(grid, world2grid, rotated2gaps)

                train_batch_start = timer()

                #####################################################################################
                # Forward pass.
                #####################################################################################
                train_batch_forward_pass = timer()

                try:
                    sdf_pred = model(points, sdfs, rotated2gaps)
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
                uniform_sdf_pred = sdf_pred[:, :, :num_uniform_points].view(batch_size, 1, -1)
                near_surface_sdf_pred = sdf_pred[:, :, num_uniform_points:].view(batch_size, 1, -1)

                loss = criterion(uniform_samples, near_surface_samples, uniform_sdf_pred, near_surface_sdf_pred)
            
                if cfg.detect_anomaly:
                    if not np.isfinite(loss.item()):
                        print("Non-finite loss: {}".format(loss.item()))
                        exit()

                time_statistics.loss_eval_duration += (timer() - train_batch_loss_eval)

                #####################################################################################
                # Backprop.
                #####################################################################################
                train_batch_backprop = timer()

                # Backprop
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