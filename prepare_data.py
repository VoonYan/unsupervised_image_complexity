# prepare_data.py
"""
Complete data preparation pipeline for CLIC
"""

import os
import shutil
import random
from pathlib import Path
from datasets import load_dataset
from PIL import Image
import requests
import io
from tqdm import tqdm


class CLICDataPreparer:
    def __init__(self, data_root="./data"):
        self.data_root = Path(data_root)
        self.flickr_dir = self.data_root / "flickr"
        self.imagenet_dir = self.data_root / "imagenet"
        self.clic_dir = self.data_root / "clic_dataset"

        # Create directories
        for dir_path in [self.flickr_dir, self.imagenet_dir, self.clic_dir / "images"]:
            dir_path.mkdir(parents=True, exist_ok=True)

    def download_flickr(self, num_images=5000):
        """Download Flickr images using Hugging Face datasets"""
        print("Downloading Flickr dataset...")

        # Load Flickr8k dataset
        try:
            ds = load_dataset("jxie/flickr8k")
            dataset = ds['train']  # Use training split
        except:
            print("Primary dataset failed, trying alternative...")
            ds = load_dataset("embedding-data/flickr8k-captions")
            dataset = ds['train']

        count = 0
        for idx, item in enumerate(tqdm(dataset, desc="Downloading Flickr images")):
            if count >= num_images:
                break

            try:
                # Get image
                if 'image' in item:
                    img = item['image']

                    # Convert to RGB if needed
                    if img.mode != 'RGB':
                        img = img.convert('RGB')

                    # Save image
                    img_path = self.flickr_dir / f"flickr_{idx:06d}.jpg"
                    img.save(img_path)
                    count += 1

            except Exception as e:
                print(f"Error processing image {idx}: {e}")
                continue

        print(f"Downloaded {count} Flickr images to {self.flickr_dir}")
        return count

    def prepare_imagenet_samples(self, imagenet_path=None, num_samples=5000):
        """
        Prepare ImageNet samples (assuming you have ImageNet downloaded)
        If not, create dummy images for testing
        """
        if imagenet_path and Path(imagenet_path).exists():
            print("Sampling from existing ImageNet dataset...")

            # Get all ImageNet images
            imagenet_images = list(Path(imagenet_path).rglob("*.JPEG"))

            # Sample random images
            sampled = random.sample(imagenet_images, min(num_samples, len(imagenet_images)))

            for idx, img_path in enumerate(tqdm(sampled, desc="Copying ImageNet images")):
                dst_path = self.imagenet_dir / f"imagenet_{idx:06d}.jpg"
                shutil.copy2(img_path, dst_path)

            return len(sampled)
        else:
            print("ImageNet not found. Creating synthetic samples for testing...")

            # Create synthetic images for testing
            for i in tqdm(range(min(100, num_samples)), desc="Creating test images"):
                # Create a random colored image
                img = Image.new('RGB', (224, 224),
                                color=(random.randint(0, 255),
                                       random.randint(0, 255),
                                       random.randint(0, 255)))
                img_path = self.imagenet_dir / f"synthetic_{i:06d}.jpg"
                img.save(img_path)

            print(f"Created {min(100, num_samples)} synthetic test images")
            return min(100, num_samples)

    def create_clic_dataset(self, num_samples=10000):
        """
        Create the final CLIC dataset by combining Flickr and ImageNet samples
        """
        print("\nCreating CLIC dataset...")

        # Get all available images
        flickr_images = list(self.flickr_dir.glob("*.jpg"))
        imagenet_images = list(self.imagenet_dir.glob("*.jpg"))

        print(f"Found {len(flickr_images)} Flickr images")
        print(f"Found {len(imagenet_images)} ImageNet images")

        # Calculate samples per dataset
        total_available = len(flickr_images) + len(imagenet_images)
        if total_available < num_samples:
            print(f"Warning: Only {total_available} images available, using all")
            num_samples = total_available

        # Sample proportionally
        flickr_ratio = len(flickr_images) / total_available
        num_flickr = int(num_samples * flickr_ratio)
        num_imagenet = num_samples - num_flickr

        # Sample images
        sampled_flickr = random.sample(flickr_images,
                                       min(num_flickr, len(flickr_images)))
        sampled_imagenet = random.sample(imagenet_images,
                                         min(num_imagenet, len(imagenet_images)))

        # Copy to CLIC dataset
        all_samples = sampled_flickr + sampled_imagenet
        random.shuffle(all_samples)  # Shuffle to mix datasets

        for idx, src_path in enumerate(tqdm(all_samples, desc="Creating CLIC dataset")):
            dst_path = self.clic_dir / "images" / f"{idx + 1:06d}.jpg"
            shutil.copy2(src_path, dst_path)

        print(f"\nCLIC dataset created with {len(all_samples)} images")
        print(f"Dataset location: {self.clic_dir / 'images'}")

        return len(all_samples)

    def prepare_all(self, flickr_images=5000, imagenet_images=5000,
                    imagenet_path=None, final_samples=10000):
        """
        Run the complete data preparation pipeline
        """
        print("=" * 60)
        print("CLIC Data Preparation Pipeline")
        print("=" * 60)

        # Step 1: Download Flickr
        print("\nStep 1: Downloading Flickr images...")
        self.download_flickr(flickr_images)

        # Step 2: Prepare ImageNet
        print("\nStep 2: Preparing ImageNet samples...")
        self.prepare_imagenet_samples(imagenet_path, imagenet_images)

        # Step 3: Create CLIC dataset
        print("\nStep 3: Creating final CLIC dataset...")
        total = self.create_clic_dataset(final_samples)

        print("\n" + "=" * 60)
        print("Data preparation complete!")
        print(f"Total images in CLIC dataset: {total}")
        print(f"Ready for training at: {self.clic_dir / 'images'}")
        print("=" * 60)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Prepare CLIC dataset")
    parser.add_argument('--data_root', type=str, default='./data',
                        help='Root directory for datasets')
    parser.add_argument('--flickr_images', type=int, default=5000,
                        help='Number of Flickr images to download')
    parser.add_argument('--imagenet_images', type=int, default=5000,
                        help='Number of ImageNet images to sample')
    parser.add_argument('--imagenet_path', type=str, default=None,
                        help='Path to existing ImageNet dataset')
    parser.add_argument('--final_samples', type=int, default=10000,
                        help='Final number of samples in CLIC dataset')

    args = parser.parse_args()

    preparer = CLICDataPreparer(args.data_root)
    preparer.prepare_all(
        flickr_images=args.flickr_images,
        imagenet_images=args.imagenet_images,
        imagenet_path=args.imagenet_path,
        final_samples=args.final_samples
    )