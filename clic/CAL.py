# clic/CAL.py
# !/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
Complexity-Aware Loss (CAL) implementation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def compute_entropy(tensor, normalize=True):
    """
    Compute entropy of a tensor
    """
    # Flatten spatial dimensions
    batch_size = tensor.size(0)
    tensor_flat = tensor.view(batch_size, -1)

    # Compute probability distribution
    tensor_prob = F.softmax(tensor_flat, dim=1)

    # Compute entropy
    log_prob = torch.log(tensor_prob + 1e-8)
    entropy = -torch.sum(tensor_prob * log_prob, dim=1)

    if normalize:
        # Normalize by log of number of elements
        max_entropy = np.log(tensor_flat.size(1))
        entropy = entropy / max_entropy

    return entropy


def compute_spatial_attention(feature_maps):
    """
    Compute spatial attention from feature maps
    """
    # Average across channel dimension
    attention = torch.mean(feature_maps, dim=1, keepdim=True)

    # Normalize to [0, 1]
    batch_size = attention.size(0)
    attention = attention.view(batch_size, -1)
    attention_min = attention.min(dim=1, keepdim=True)[0]
    attention_max = attention.max(dim=1, keepdim=True)[0]
    attention = (attention - attention_min) / (attention_max - attention_min + 1e-8)

    # Reshape back
    attention = attention.view(batch_size, 1, feature_maps.size(2), feature_maps.size(3))

    return attention


def compute_batch_ge(stage_maps, temperature=1.0):
    """
    Compute Global Entropy (GE) for batch of feature maps

    Args:
        stage_maps: Dictionary of feature maps from different stages
        temperature: Temperature for entropy computation

    Returns:
        ge_scores: Global entropy scores for the batch
    """
    ge_scores = []

    for stage_name, features in stage_maps.items():
        if features is not None:
            # Apply temperature scaling
            features_scaled = features / temperature

            # Compute entropy for this stage
            entropy = compute_entropy(features_scaled)
            ge_scores.append(entropy)

    # Average across stages if multiple stages
    if len(ge_scores) > 1:
        ge_scores = torch.stack(ge_scores).mean(dim=0)
    else:
        ge_scores = ge_scores[0]

    return ge_scores


def compute_ge_fae_error(ge_q, ge_k, weight=0.25):
    """
    Compute complexity-aware loss term

    Args:
        ge_q: Global entropy of query features
        ge_k: Global entropy of key features
        weight: Weight for the complexity loss

    Returns:
        cal_loss: Complexity-aware loss
    """
    # Compute L2 distance between entropy scores
    cal_loss = F.mse_loss(ge_q, ge_k)

    # Apply weight
    cal_loss = weight * cal_loss

    return cal_loss


class ComplexityAwareLoss(nn.Module):
    """
    Full Complexity-Aware Loss module
    """

    def __init__(self, weight=0.25, temperature=1.0):
        super(ComplexityAwareLoss, self).__init__()
        self.weight = weight
        self.temperature = temperature

    def forward(self, features_q, features_k):
        """
        Args:
            features_q: Query features (dict of stage outputs)
            features_k: Key features (dict of stage outputs)
        """
        # Compute global entropy
        ge_q = compute_batch_ge(features_q, self.temperature)
        ge_k = compute_batch_ge(features_k, self.temperature)

        # Compute complexity-aware loss
        cal_loss = compute_ge_fae_error(ge_q, ge_k, self.weight)

        return cal_loss