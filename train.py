# train.py
# !/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
CLIC training script - CPU compatible version
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
    parser.add_argument('-j', '--workers', default=2, type=int,
                        help='number of data loading workers (default: 2 for CPU)')

    # Model parameters
    parser.add_argument('-a', '--arch', default='resnet50',
                        help='model architecture')
    parser.add_argument('--dim', default=128, type=int,
                        help='feature dimension')
    parser.add_argument('--k', default=4096, type=int,
                        help='queue size (reduced for CPU)')
    parser.add_argument('--m', default=0.999, type=float,
                        help='momentum of updating key encoder')
    parser.add_argument('--t', default=0.07, type=float,
                        help='softmax temperature')

    # Training parameters
    parser.add_argument('--epochs', default=10, type=int,
                        help='number of total epochs to run (reduced for CPU)')
    parser.add_argument('-b', '--batch-size', default=8, type=int,
                        help='mini-batch size (reduced for CPU)')
    parser.add_argument('--lr', default=0.03, type=float,
                        help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float,
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        dest='weight_decay', help='weight decay')
    parser.add_argument('--ca_lambda', default=0.25, type=float,
                        help='complexity aware loss coefficient')

    # Other parameters
    parser.add_argument('--use-gpu', action='store_true',
                        help='use GPU if available')
    parser.add_argument('--gpu', default=0, type=int,
                        help='GPU id to use (if --use-gpu is set)')
    parser.add_argument('--seed', default=42, type=int,
                        help='seed for initializing training')
    parser.add_argument('--resume', default='', type=str,
                        help='path to latest checkpoint')
    parser.add_argument('--save-dir', default='./checkpoints', type=str,
                        help='directory to save checkpoints')
    parser.add_argument('--print-freq', default=10, type=int,
                        help='print frequency')

    args = parser.parse_args()

    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)

    # Set random seed
    if args.seed is not None:
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            cudnn.deterministic = True

    # Set device
    if args.use_gpu and torch.cuda.is_available():
        device = torch.device(f'cuda:{args.gpu}')
        logger.info(f'Using GPU: {args.gpu}')
    else:
        device = torch.device('cpu')
        logger.info('Using CPU - This will be slower than GPU training')
        logger.info('Reduced batch size and queue size for CPU training')
        # Adjust parameters for CPU
        if args.batch_size > 8:
            logger.warning(f'Batch size {args.batch_size} might be too large for CPU. Consider using 8 or less.')
        if args.k > 4096:
            logger.warning(f'Queue size {args.k} might be too large for CPU. Consider using 4096 or less.')

    # Create model
    logger.info(f"Creating model '{args.arch}'")
    base_encoder = getattr(models, args.arch)
    model = CLIC(
        base_encoder,
        dim=args.dim,
        K=args.k,
        m=args.m,
        T=args.t,
        device=device
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
    start_epoch = 0
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
        pin_memory=(device.type == 'cuda'),  # Only pin memory if using GPU
        drop_last=True
    )

    logger.info(f'Dataset size: {len(train_dataset)}')
    logger.info(f'Batch size: {args.batch_size}')
    logger.info(f'Number of batches: {len(train_loader)}')

    # Training loop
    for epoch in range(start_epoch, args.epochs):
        # Adjust learning rate
        adjust_learning_rate(optimizer, epoch, args)

        # Train one epoch
        loss = train_one_epoch(
            train_loader, model, criterion, optimizer,
            epoch, args, device
        )

        # Save checkpoint
        if (epoch + 1) % 5 == 0 or epoch == args.epochs - 1:
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'loss': loss,
            }, args.save_dir, epoch + 1)

    logger.info('Training completed!')


def train_one_epoch(train_loader, model, criterion, optimizer, epoch, args, device):
    """Train for one epoch"""
    model.train()

    losses = AverageMeter('Loss', ':.4e')
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')

    end = time.time()

    for i, (images, _) in enumerate(train_loader):
        # Measure data loading time
        data_time.update(time.time() - end)

        # Move to device
        images[0] = images[0].to(device, non_blocking=True)
        images[1] = images[1].to(device, non_blocking=True)

        # Forward pass
        output, target = model(im_q=images[0], im_k=images[1])

        # Compute loss
        loss = criterion(output, target)

        # Record loss
        losses.update(loss.item(), images[0].size(0))

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # Log progress
        if i % args.print_freq == 0:
            logger.info(
                f'Epoch: [{epoch}][{i}/{len(train_loader)}]\t'
                f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                f'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                f'Loss {losses.val:.4f} ({losses.avg:.4f})'
            )

    return losses.avg


def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate based on schedule"""
    lr = args.lr

    # Cosine annealing
    lr *= 0.5 * (1. + torch.cos(torch.tensor(epoch / args.epochs * 3.14159)))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return lr


def save_checkpoint(state, save_dir, epoch):
    """Save checkpoint"""
    filename = os.path.join(save_dir, f'checkpoint_{epoch:04d}.pth.tar')
    torch.save(state, filename)
    logger.info(f'Saved checkpoint to {filename}')


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


if __name__ == '__main__':
    main()