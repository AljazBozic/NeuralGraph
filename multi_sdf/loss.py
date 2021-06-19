import sys,os
import torch
import torch.nn as nn
import numpy as np
import math

import config as cfg
from utils.pcd_utils import *


class LossSDF(torch.nn.Module):
    def __init__(self):
        super(LossSDF, self).__init__()

        self.geometry_loss = GeometryLoss()

    def forward(self, uniform_samples, near_surface_samples, uniform_sdf_pred, near_surface_sdf_pred, eval=False):
        batch_size = uniform_samples.shape[0]
        num_uniform_samples = uniform_samples.shape[1]
        num_near_surface_samples = near_surface_samples.shape[1]

        loss_total = torch.zeros((1), dtype=uniform_sdf_pred.dtype, device=uniform_sdf_pred.device)

        # Uniform error.
        loss_uniform = None
        if cfg.lambda_geometry:
            loss_uniform = self.geometry_loss(uniform_sdf_pred.view(batch_size, num_uniform_samples), uniform_samples[:, :, 3].view(batch_size, num_uniform_samples))
            loss_total += cfg.lambda_geometry * loss_uniform

        # Near surface error.
        loss_near_surface = None
        if cfg.lambda_geometry:
            loss_near_surface = self.geometry_loss(near_surface_sdf_pred.view(batch_size, num_near_surface_samples), near_surface_samples[:, :, 3].view(batch_size, num_near_surface_samples))
            loss_total += cfg.lambda_geometry * loss_near_surface

        if eval:
            return loss_total, loss_uniform, loss_near_surface
        else:
            return loss_total
        

class GeometryLoss(nn.Module):
    def __init__(self):
        super(GeometryLoss, self).__init__()

        self.criterion_L1 = nn.L1Loss(reduction='none')
    
    def forward(self, sdf_pred, sdf_gt):
        sdf_gt_truncated = torch.max(torch.min(sdf_gt, cfg.truncation * torch.ones_like(sdf_gt)), \
            -cfg.truncation * torch.ones_like(sdf_gt))

        loss = self.criterion_L1(sdf_pred, sdf_gt_truncated)

        return loss.mean()