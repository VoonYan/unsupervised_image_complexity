# fine_tuning.py
# !/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
Fine-tuning CLIC on IC9600 dataset - CPU compatible
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
from torch import optim
from torchvision import transforms
from torch.utils.data import DataLoader
from scipy.stats import pearsonr, spearmanr
import numpy as np
import logging

from clic.loader import ICDataset
from clic.icnet import ICNet_ft

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def download_ic9600():
    """Instructions to download IC9600"""
    print("\n" + "=" * 60)
    print("IC9600 Dataset Required")
    print("=" * 60)
    print("Please download IC9600 from:")
    print("https://github.com/tinglyfeng/IC9600")
    print("\nExtract it to ./IC9600/ with structure:")
    print("  ./IC9600/")
    print("    ├── images/")
    print("    ├── train.txt")
    print("    └── test.txt")
    print("=" * 60 + "\n")


def finetune_ic9600():
    # Parameters optimized for CPU
    args = {
        'batch_size': 4,
        'lr': 0.001,
        'epochs': 10,
        'image_size': 224,  # Reduced from 512 for CPU
        'num_workers': 0,
        'device': 'cpu',
        'checkpoint': './checkpoints/cpu_checkpoint_epoch_5.pth'  # From pre-training
    }

    device = torch.device(args['device'])

    # Check if IC9600 exists
    if not os.path.exists('./IC9600'):
        download_ic9600()
        return

    # Transforms
    train_transform = transforms.Compose([
        transforms.Resize((args['image_size'], args['image_size'])),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_transform = transforms.Compose([
        transforms.Resize((args['image_size'], args['image_size'])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Datasets
    train_dataset = ICDataset(
        txt_path="./IC9600/train.txt",
        img_path="./IC9600/images/",
        transform=train_transform
    )

    test_dataset = ICDataset(
        txt_path="./IC9600/test.txt",
        img_path="./IC9600/images/",
        transform=test_transform
    )

    # Dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args['batch_size'],
        shuffle=True,
        num_workers=args['num_workers']
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args['batch_size'],
        shuffle=False,
        num_workers=args['num_workers']
    )

    # Model
    model = ICNet_ft(pretrained_path=args['checkpoint'])
    model = model.to(device)

    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args['lr'])

    logger.info(f"Starting fine-tuning on IC9600...")
    logger.info(f"Train samples: {len(train_dataset)}")
    logger.info(f"Test samples: {len(test_dataset)}")

    # Training loop
    best_corr = 0
    for epoch in range(args['epochs']):
        # Train
        model.train()
        train_loss = 0
        for batch_idx, (images, labels, _) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device).float().unsqueeze(1)

            scores, _ = model(images)
            loss = criterion(scores, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            if batch_idx % 10 == 0:
                logger.info(f'Epoch [{epoch + 1}/{args["epochs"]}] '
                            f'Batch [{batch_idx}/{len(train_loader)}] '
                            f'Loss: {loss.item():.4f}')

        # Evaluate
        model.eval()
        all_scores = []
        all_labels = []

        with torch.no_grad():
            for images, labels, _ in test_loader:
                images = images.to(device)
                scores, _ = model(images)
                all_scores.extend(scores.cpu().numpy())
                all_labels.extend(labels.numpy())

        # Calculate correlations
        pearson_corr, _ = pearsonr(all_scores, all_labels)
        spearman_corr, _ = spearmanr(all_scores, all_labels)

        logger.info(f'Epoch [{epoch + 1}/{args["epochs"]}] '
                    f'Pearson: {pearson_corr:.4f} '
                    f'Spearman: {spearman_corr:.4f}')

        # Save best model
        if pearson_corr > best_corr:
            best_corr = pearson_corr
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'pearson': pearson_corr,
                'spearman': spearman_corr
            }, './checkpoints/best_finetuned_model.pth')
            logger.info(f'Saved best model with Pearson correlation: {best_corr:.4f}')

    logger.info(f'Fine-tuning completed! Best Pearson correlation: {best_corr:.4f}')
    return model


if __name__ == "__main__":
    finetune_ic9600()