import sys,os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import torch

from external.pyTorchChamferDistance.chamfer_distance import ChamferDistance 


def compute_chamfer_distance(gt_points, pred_points):
    chamfer_dist = ChamferDistance()

    dist1, dist2 = chamfer_dist(gt_points, pred_points)
    loss = (torch.mean(dist1)) + (torch.mean(dist2))

    return loss


def precision(tp_count, fp_count):
    return tp_count / (tp_count + fp_count)


def recall(tp_count, fn_count):
    return tp_count / (tp_count + fn_count)


def accuracy(tp_count, tn_count, fp_count, fn_count):
    return (tp_count + tn_count) / (tp_count + tn_count + fp_count + fn_count)


def f1_score(precision, recall):
    return (2.0 * precision * recall) / (precision + recall)