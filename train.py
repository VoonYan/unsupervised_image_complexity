# train.py
# !/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
CLIC training script
"""

import sys
import os

# Add current directory to path to import local clic module
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse
import logging
import time
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.models as models

from clic.builder import CLIC
from clic.loader import CLICDataset, TwoCropsTransform
from clic.CAL import compute_batch_ge, compute_ge_fae_error

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description='CLIC Training')

    # Data parameters
    parser.add_argument('--data', default='./data/clic_dataset/images',
                        help='path to dataset')
    parser.add_argument('-j', '--workers', default=4, type=int,
                        help='number of data loading workers')

    # Model parameters
    parser.add_argument('-a', '--arch', default='resnet50',
                        help='model architecture')
    parser.add_argument('--dim', default=128, type=int,
                        help='feature dimension')
    parser.add_argument('--k', default=65536, type=int,
                        help='queue size')
    parser.add_argument('--m', default=0.999, type=float,
                        help='momentum of updating key encoder')
    parser.add_argument('--t', default=0.07, type=float,
                        help='softmax temperature')

    # Training parameters
    parser.add_argument('--epochs', default=200, type=int,
                        help='number of total epochs to run')
    parser.add_argument('-b', '--batch-size', default=32, type=int,
                        help='mini-batch size')
    parser.add_argument('--lr', default=0.03, type=float,
                        help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float,
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        dest='weight_decay', help='weight decay')
    parser.add_argument('--ca_lambda', default=0.25, type=float,
                        help='complexity aware loss coefficient')

    # Other parameters
    parser.add_argument('--gpu', default=0, type=int,
                        help='GPU id to use')
    parser.add_argument('--seed', default=None, type=int,
                        help='seed for initializing training')
    parser.add_argument('--resume', default='', type=str,
                        help='path to latest checkpoint')
    parser.add_argument('--save-dir', default='./checkpoints', type=str,
                        help='directory to save checkpoints')

    args = parser.parse_args()

    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)

    # Set random seed
    if args.seed is not None:
        torch.manual_seed(args.seed)
        cudnn.deterministic = True

    # Set GPU
    if args.gpu is not None and torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        device = torch.device(f'cuda:{args.gpu}')
        logger.info(f'Using GPU: {args.gpu}')
    else:
        device = torch.device('cpu')
        logger.info('Using CPU')

    # Create model
    logger.info(f"Creating model '{args.arch}'")
    base_encoder = getattr(models, args.arch)
    model = CLIC(
        base_encoder,
        dim=args.dim,
        K=args.k,
        m=args.m,
        T=args.t
    )
    model = model.to(device)

    # Create optimizer
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )

    # Load checkpoint if resume
    if args.resume:
        if os.path.isfile(args.resume):
            logger.info(f"Loading checkpoint '{args.resume}'")
            checkpoint = torch.load(args.resume, map_location=device)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            start_epoch = checkpoint['epoch']
            logger.info(f"Loaded checkpoint (epoch {checkpoint['epoch']})")
        else:
            logger.error(f"No checkpoint found at '{args.resume}'")
            return
    else:
        start_epoch = 0

    # Create data loader
    train_dataset = CLICDataset(
        root_dir=args.data,
        transform=TwoCropsTransform()
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True
    )

    logger.info(f'Dataset size: {len(train_dataset)}')

    # Training loop
    for epoch in range(start_epoch, args.epochs):
        # Adjust learning rate
        adjust_learning_rate(optimizer, epoch, args)

        # Train one epoch
        train_one_epoch(
            train_loader, model, criterion, optimizer,
            epoch, args, device
        )

        # Save checkpoint
        if (epoch + 1) % 10 == 0:
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, args.save_dir, epoch + 1)

    logger.info('Training completed!')


def train_one_epoch(train_loader, model, criterion, optimizer, epoch, args, device):
    """Train for one epoch"""
    model.train()

    for i, (images, _) in enumerate(train_loader):
        # Move to device
        images[0] = images[0].to(device)
        images[1] = images[1].to(device)

        # Forward pass
        output, target = model(im_q=images[0], im_k=images[1])

        # Compute loss
        loss = criterion(output, target)

        # Add complexity-aware loss if needed
        # (simplified version without feature maps for now)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Log progress
        if i % 50 == 0:
            logger.info(
                f'Epoch: [{epoch}][{i}/{len(train_loader)}]\t'
                f'Loss: {loss.item():.4f}'
            )


def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate based on schedule"""
    lr = args.lr
    if epoch >= 120:
        lr *= 0.01
    elif epoch >= 80:
        lr *= 0.1

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def save_checkpoint(state, save_dir, epoch):
    """Save checkpoint"""
    filename = os.path.join(save_dir, f'checkpoint_{epoch:04d}.pth.tar')
    torch.save(state, filename)
    logger.info(f'Saved checkpoint to {filename}')


if __name__ == '__main__':
    main()