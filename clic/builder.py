# clic/builder.py
# !/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
CLIC model builder
"""

import torch
import torch.nn as nn


class CLIC(nn.Module):
    """
    Build a CLIC model with: a query encoder, a key encoder, and a queue
    Modified from MoCo
    """

    def __init__(self, base_encoder, dim=128, K=65536, m=0.999, T=0.07):
        """
        Args:
            base_encoder: backbone encoder (e.g., ResNet)
            dim: feature dimension (default: 128)
            K: queue size; number of negative keys (default: 65536)
            m: momentum of updating key encoder (default: 0.999)
            T: softmax temperature (default: 0.07)
        """
        super(CLIC, self).__init__()

        self.K = K
        self.m = m
        self.T = T

        # Create the encoders
        self.encoder_q = base_encoder(num_classes=dim)
        self.encoder_k = base_encoder(num_classes=dim)

        # Initialize key encoder with query encoder weights
        for param_q, param_k in zip(
                self.encoder_q.parameters(), self.encoder_k.parameters()
        ):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False  # not update by gradient

        # Create the queue
        self.register_buffer("queue", torch.randn(dim, K))
        self.queue = nn.functional.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(
                self.encoder_q.parameters(), self.encoder_k.parameters()
        ):
            param_k.data = param_k.data * self.m + param_q.data * (1.0 - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # Replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr: ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def _batch_shuffle_ddp(self, x):
        """
        Batch shuffle, for making use of BatchNorm.
        """
        # Random shuffle index
        batch_size = x.shape[0]
        idx_shuffle = torch.randperm(batch_size).cuda()

        # Shuffle
        x_shuffled = x[idx_shuffle]

        return x_shuffled, idx_shuffle

    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x, idx_unshuffle):
        """
        Undo batch shuffle
        """
        return x[idx_unshuffle]

    def forward(self, im_q, im_k):
        """
        Args:
            im_q: a batch of query images
            im_k: a batch of key images
        Returns:
            logits, targets
        """
        # Compute query features
        q = self.encoder_q(im_q)  # queries: NxC
        q = nn.functional.normalize(q, dim=1)

        # Compute key features
        with torch.no_grad():
            self._momentum_update_key_encoder()  # update the key encoder

            # Shuffle for making use of BN
            im_k_shuffled, idx_unshuffle = self._batch_shuffle_ddp(im_k)

            k = self.encoder_k(im_k_shuffled)  # keys: NxC
            k = nn.functional.normalize(k, dim=1)

            # Undo shuffle
            k = self._batch_unshuffle_ddp(k, idx_unshuffle)

        # Compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        l_pos = torch.einsum("nc,nc->n", [q, k]).unsqueeze(-1)
        # negative logits: NxK
        l_neg = torch.einsum("nc,ck->nk", [q, self.queue.clone().detach()])

        # Logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)

        # Apply temperature
        logits /= self.T

        # Labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

        # Dequeue and enqueue
        self._dequeue_and_enqueue(k)

        return logits, labels


def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    """
    tensors_gather = [tensor.clone()]
    output = torch.cat(tensors_gather, dim=0)
    return output