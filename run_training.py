# run_training.py
import os
import sys
import torch
import argparse


def train_clic():
    """Run CLIC unsupervised training"""

    # Set up arguments
    args = [
        '--data', './clic_data/',
        '--arch', 'resnet50',
        '--workers', '8',
        '--epochs', '200',
        '--batch-size', '32',
        '--lr', '0.03',
        '--momentum', '0.9',
        '--weight-decay', '1e-4',
        '--print-freq', '10',
        '--gpu', '0',
        '--dim', '128',
        '--k', '65536',
        '--m', '0.999',
        '--t', '0.07',
        '--ca_lambda', '0.25',
        '--cos',  # Use cosine learning rate schedule
    ]

    # Run training
    sys.argv = ['train.py'] + args

    # Import and run the training script
    import train
    train.main()


if __name__ == "__main__":
    # Check if GPU is available
    if not torch.cuda.is_available():
        print("Warning: CUDA is not available. Training will be slow on CPU.")

    # Create necessary directories
    os.makedirs("./clic_data/images", exist_ok=True)
    os.makedirs("./ckpts", exist_ok=True)

    print("Starting CLIC training...")
    train_clic()