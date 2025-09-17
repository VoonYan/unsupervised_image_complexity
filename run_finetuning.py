# run_finetuning.py
import os
import sys
import torch
import argparse


def download_ic9600():
    """Download IC9600 dataset"""
    # You need to download IC9600 from: https://github.com/tinglyfeng/IC9600
    print("Please download IC9600 dataset from: https://github.com/tinglyfeng/IC9600")
    print("Extract it to ./IC9600/ directory")


def finetune_on_ic9600():
    """Fine-tune CLIC on IC9600 dataset"""

    args = [
        '--warm', '1',
        '--lr', '0.05',
        '--weight_decay', '1e-3',
        '--lr_decay_rate', '0.2',
        '--batch_size', '64',
        '--num_workers', '8',
        '--epoch', '30',
        '--image_size', '512',
        '--gpu_id', '0',
        '--ck_path', './ckpts/checkpoint_0200.pth.tar',  # Path to pre-trained model
    ]

    sys.argv = ['fine_tuning.py'] + args

    # Import and run fine-tuning
    import fine_tuning

    # Note: You'll need to complete the fine_tuning.py script's main execution


if __name__ == "__main__":
    if not os.path.exists("./IC9600"):
        download_ic9600()
    else:
        print("Starting fine-tuning on IC9600...")
        finetune_on_ic9600()