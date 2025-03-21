import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from fast_soft_sort.pytorch_ops import soft_sort

class DistLoss(nn.Module):
    def __init__(self, size_average=True, loss_fn='L1', loss_weight=1.0, regularization_strength=0.1, require_loss_values=False):
        """
        Initialize the DistLoss module.

        Parameters:
            size_average (bool): If True, average the loss; if False, sum the loss.
            loss_fn (str or nn.Module): The type of loss function to use:
                                        - 'L1': Use nn.L1Loss.
                                        - 'L2': Use nn.MSELoss.
                                        - Otherwise, should be a custom nn.Module for a specific loss.
            loss_weight (float): The weight to apply to the distribution loss.
            regularization_strength (float): Strength of regularization in soft_sort algorithm.
            require_loss_values (bool): Whether to return the individual loss values along with the total loss.
        """
        super(DistLoss, self).__init__()
        self.size_average = size_average
        self.loss_weight = loss_weight
        self.regularization_strength = regularization_strength
        self.require_loss_values = require_loss_values

        # Determine the loss function based on the loss_fn parameter
        if loss_fn in ['l1', 'L1', 'MAE', 'mae']:
            self.loss_fn = nn.L1Loss(reduction='none')
        elif loss_fn in ['l2', 'L2', 'mse', 'MSE']:
            self.loss_fn = nn.MSELoss(reduction='none')
        elif loss_fn == 'focal_l1':
            self.loss_fn = FocalL1Loss()
        elif loss_fn == 'huber':
            self.loss_fn = HuberLoss()
        elif isinstance(loss_fn, nn.Module):
            self.loss_fn = loss_fn
        else:
            raise ValueError("Invalid loss function. Choose 'L1', 'L2', or provide a custom nn.Module.")

    def forward(self, inp, tar, theoretical_labels, weights=None):
        """
        Compute the loss between the input and theoretical labels using soft_sort.

        Parameters:
            inp (torch.Tensor): The input tensor.
            tar (torch.Tensor): The target tensor.
            theoretical_labels (torch.Tensor): Theoretical labels tensor computed from kernel density estimation.

        Returns:
            torch.Tensor: The computed loss, and optionally the individual loss values.
        """
        # Perform soft sorting on the input tensor
        sorted_inp = soft_sort(inp.reshape(1, -1).cpu(), regularization_strength=self.regularization_strength)
        sorted_inp = sorted_inp.reshape(-1, 1).cuda()

        # Compute the distribution loss using the specified loss function
        dist_loss = self.loss_fn(sorted_inp, theoretical_labels)

        if weights is not None:
            dist_loss *= weights.expand_as(dist_loss)

        # Compute the plain loss
        plain_loss = self.loss_fn(inp, tar)
        if weights is not None:
            plain_loss *= weights.expand_as(plain_loss)
        
        # Compute the total loss
        total_loss = self.loss_weight * dist_loss + plain_loss
        
        # Return the average or sum of the loss based on size_average
        if self.size_average:
            if self.require_loss_values:
                return total_loss.mean(), dist_loss.mean(), plain_loss.mean()
            else:
                return total_loss.mean()
        else:
            if self.require_loss_values:
                return total_loss.sum(), dist_loss.sum(), plain_loss.sum()
            else:
                return total_loss.sum()

class FocalL1Loss(torch.nn.Module):
    def __init__(self, beta=0.2, gamma=1, activate='sigmoid'):
        super(FocalL1Loss, self).__init__()
        self.beta = beta
        self.gamma = gamma
        self.activate = activate

    def forward(self, inputs, targets):
        loss = F.l1_loss(inputs, targets, reduction='none')
        
        if self.activate == 'tanh':
            loss *= (torch.tanh(self.beta * torch.abs(inputs - targets))) ** self.gamma
        else:
            loss *= (2 * torch.sigmoid(self.beta * torch.abs(inputs - targets)) - 1) ** self.gamma
        
        # loss = torch.mean(loss)
        return loss

class HuberLoss(torch.nn.Module):
    def __init__(self, beta=1.0):
        super(HuberLoss, self).__init__()
        self.beta = beta

    def forward(self, inputs, targets):
        l1_loss = torch.abs(inputs - targets)
        cond = l1_loss < self.beta
        loss = torch.where(cond, 0.5 * l1_loss ** 2 / self.beta, l1_loss - 0.5 * self.beta)
        
        # loss = torch.mean(loss)
        return loss

# For more details on the fast and differentiable sorting algorithm, visit:
# https://github.com/google-research/fast-soft-sort