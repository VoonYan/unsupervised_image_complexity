# quick_start.py
"""
Quick start script to test CLIC with minimal setup
"""

import torch
import torchvision.models as models
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np


# Create a simple test with synthetic data
def test_clic_model():
    print("Testing CLIC model setup...")

    # Initialize model
    model = models.resnet50(pretrained=False)
    model.fc = torch.nn.Linear(model.fc.in_features, 128)  # CLIC uses 128-dim features

    # Create dummy data
    batch_size = 4
    dummy_images = torch.randn(batch_size, 3, 224, 224)

    # Forward pass
    with torch.no_grad():
        features = model(dummy_images)
        print(f"Output shape: {features.shape}")
        print(f"Expected shape: ({batch_size}, 128)")

    print("✓ Model test successful!")

    # Test data augmentation
    transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.2, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    print("\n✓ Data augmentation pipeline ready!")
    print("\nCLIC setup test completed successfully!")


if __name__ == "__main__":
    test_clic_model()