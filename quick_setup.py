# quick_setup.py
"""
Quick setup script to get CLIC running with minimal effort
"""

import subprocess
import sys
import os


def install_requirements():
    """Install required packages"""
    print("Installing required packages...")

    packages = [
        "torch",
        "torchvision",
        "datasets",
        "pillow",
        "numpy",
        "pandas",
        "scipy",
        "tqdm",
        "requests"
    ]

    for package in packages:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])

    print("✓ All packages installed!")


def download_data():
    """Download and prepare data"""
    print("\nPreparing data...")

    # Run the data preparation script
    code = """
from prepare_data import CLICDataPreparer

preparer = CLICDataPreparer('./data')
preparer.prepare_all(
    flickr_images=1000,  # Start with smaller dataset for testing
    imagenet_images=0,   # Skip ImageNet for quick test
    final_samples=1000
)
"""

    with open("temp_prepare.py", "w") as f:
        f.write(code)

    subprocess.run([sys.executable, "temp_prepare.py"])
    os.remove("temp_prepare.py")


def test_setup():
    """Test if everything is set up correctly"""
    print("\nTesting setup...")

    test_code = """
import torch
from datasets import load_dataset

# Test PyTorch
print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

# Test dataset loading
print("\\nTesting Flickr dataset loading...")
ds = load_dataset("jxie/flickr8k", split="train", streaming=True)
sample = next(iter(ds))
print("✓ Dataset loading successful!")

# Test image processing
from PIL import Image
print("✓ PIL/Pillow working!")

print("\\n✅ All tests passed! Ready to train CLIC.")
"""

    exec(test_code)


if __name__ == "__main__":
    print("=" * 60)
    print("CLIC Quick Setup")
    print("=" * 60)

    # Step 1: Install requirements
    print("\n1. Installing requirements...")
    response = input("Install required packages? (y/n): ")
    if response.lower() == 'y':
        install_requirements()

    # Step 2: Download data
    print("\n2. Downloading data...")
    response = input("Download Flickr dataset? (y/n): ")
    if response.lower() == 'y':
        download_data()

    # Step 3: Test setup
    print("\n3. Testing setup...")
    test_setup()

    print("\n" + "=" * 60)
    print("Setup complete! You can now run:")
    print("  python train.py        # For unsupervised training")
    print("  python fine_tuning.py  # For fine-tuning on IC9600")
    print("=" * 60)