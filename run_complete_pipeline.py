# run_complete_pipeline.py
# !/usr/bin/env python
"""
Complete pipeline to run the CLIC project
"""

import os
import subprocess
import sys


def check_requirements():
    """Check if all requirements are installed"""
    required_packages = ['torch', 'torchvision', 'numpy', 'pandas', 'scipy', 'PIL', 'tqdm']

    for package in required_packages:
        try:
            __import__(package)
            print(f"✓ {package} is installed")
        except ImportError:
            print(f"✗ {package} is not installed")
            return False
    return True


def setup_directories():
    """Create necessary directory structure"""
    directories = [
        './clic_data/images',
        './Flickr/parquet',
        './Flickr/train',
        './ImageNet/train',
        './IC9600/images',
        './ckpts',
        './logs'
    ]

    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Created directory: {directory}")


def main():
    print("=" * 60)
    print("CLIC: Contrastive Learning for Image Complexity")
    print("=" * 60)

    # Step 1: Check requirements
    print("\n1. Checking requirements...")
    if not check_requirements():
        print("Please install missing requirements:")
        print("pip install torch torchvision numpy pandas scipy pillow tqdm")
        return

    # Step 2: Setup directories
    print("\n2. Setting up directories...")
    setup_directories()

    # Step 3: Data preparation
    print("\n3. Data Preparation")
    print("Please ensure you have:")
    print("  - ImageNet dataset in ./ImageNet/")
    print("  - Run download_flickr.py to get Flickr data")
    print("  - Run uniform_sample.py to create CLIC dataset")

    response = input("\nHave you prepared the data? (y/n): ")
    if response.lower() != 'y':
        print("Please prepare the data first.")
        return

    # Step 4: Training
    print("\n4. Starting unsupervised training...")
    print("This will train for 200 epochs on the CLIC dataset")
    response = input("Start training? (y/n): ")
    if response.lower() == 'y':
        subprocess.run([sys.executable, "train.py"])

    # Step 5: Fine-tuning
    print("\n5. Fine-tuning on IC9600")
    response = input("Start fine-tuning? (y/n): ")
    if response.lower() == 'y':
        subprocess.run([sys.executable, "fine_tuning.py"])

    print("\n" + "=" * 60)
    print("Pipeline completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()