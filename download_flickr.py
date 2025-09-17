# download_flickr.py
import os
import requests
from tqdm import tqdm
from datasets import load_dataset
from PIL import Image
import io


def download_flickr_dataset(save_dir="./Flickr/train/", max_images=None):
    """
    Download Flickr dataset using Hugging Face datasets

    Args:
        save_dir: Directory to save downloaded images
        max_images: Maximum number of images to download (None for all)
    """

    # Create save directory
    os.makedirs(save_dir, exist_ok=True)

    print("Loading Flickr8k dataset from Hugging Face...")

    # Load the dataset
    try:
        ds = load_dataset("jxie/flickr8k")
        print(f"Dataset loaded successfully!")
        print(f"Available splits: {list(ds.keys())}")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Trying alternative Flickr dataset...")
        # Try alternative datasets if needed
        try:
            ds = load_dataset("nlphuji/flickr30k")
        except:
            ds = load_dataset("embedding-data/flickr8k-captions")

    # Process the dataset
    count = 0
    failed_count = 0

    # Iterate through the dataset splits
    for split in ds.keys():
        print(f"\nProcessing split: {split}")
        dataset = ds[split]

        # Create split-specific directory
        split_dir = os.path.join(save_dir, split)
        os.makedirs(split_dir, exist_ok=True)

        # Determine the image field name (different datasets may use different names)
        if 'image' in dataset.column_names:
            image_field = 'image'
        elif 'image_url' in dataset.column_names:
            image_field = 'image_url'
        else:
            print(f"Available columns: {dataset.column_names}")
            image_field = input("Enter the image field name: ")

        # Download images
        for idx, item in enumerate(tqdm(dataset, desc=f"Downloading {split} images")):
            if max_images and count >= max_images:
                break

            try:
                # Get image (could be PIL Image or URL)
                image_data = item[image_field]

                # Handle different image data types
                if isinstance(image_data, str):  # URL
                    response = requests.get(image_data, timeout=10)
                    if response.status_code == 200:
                        img = Image.open(io.BytesIO(response.content))
                    else:
                        failed_count += 1
                        continue
                else:  # PIL Image or similar
                    img = image_data

                # Save image
                image_path = os.path.join(split_dir, f"flickr_{split}_{idx:06d}.jpg")

                # Convert to RGB if necessary
                if img.mode != 'RGB':
                    img = img.convert('RGB')

                img.save(image_path)
                count += 1

                if count % 100 == 0:
                    print(f"Downloaded {count} images so far...")

            except Exception as e:
                failed_count += 1
                if failed_count % 10 == 0:
                    print(f"Failed to download {failed_count} images. Latest error: {e}")
                continue

    print(f"\nDownload complete!")
    print(f"Successfully downloaded: {count} images")
    print(f"Failed downloads: {failed_count}")
    print(f"Images saved to: {save_dir}")

    return count


def download_flickr_alternative(save_dir="./Flickr/train/", num_images=10000):
    """
    Alternative method using different Flickr datasets from Hugging Face
    """
    print("Attempting to download from alternative Flickr datasets...")

    # List of alternative Flickr datasets on Hugging Face
    alternative_datasets = [
        "jxie/flickr8k",
        "nlphuji/flickr30k",
        "embedding-data/flickr8k-captions",
        "yerevann/flickr8k-train",
    ]

    os.makedirs(save_dir, exist_ok=True)
    total_downloaded = 0

    for dataset_name in alternative_datasets:
        if total_downloaded >= num_images:
            break

        print(f"\nTrying dataset: {dataset_name}")

        try:
            ds = load_dataset(dataset_name, split="train", streaming=True)

            # Download images
            for idx, item in enumerate(tqdm(ds.take(num_images - total_downloaded),
                                            desc=f"Downloading from {dataset_name}")):
                try:
                    # Try to get image
                    if 'image' in item:
                        img = item['image']
                    elif 'image_url' in item:
                        response = requests.get(item['image_url'], timeout=10)
                        img = Image.open(io.BytesIO(response.content))
                    else:
                        continue

                    # Save image
                    if hasattr(img, 'save'):
                        if img.mode != 'RGB':
                            img = img.convert('RGB')

                        image_path = os.path.join(save_dir, f"flickr_{total_downloaded:06d}.jpg")
                        img.save(image_path)
                        total_downloaded += 1

                except Exception as e:
                    continue

        except Exception as e:
            print(f"Could not load {dataset_name}: {e}")
            continue

    print(f"\nTotal images downloaded: {total_downloaded}")
    return total_downloaded


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download Flickr dataset")
    parser.add_argument('--save_dir', type=str, default='./Flickr/train/',
                        help='Directory to save images')
    parser.add_argument('--max_images', type=int, default=None,
                        help='Maximum number of images to download')
    parser.add_argument('--alternative', action='store_true',
                        help='Use alternative download method')

    args = parser.parse_args()

    if args.alternative:
        download_flickr_alternative(args.save_dir, args.max_images or 10000)
    else:
        download_flickr_dataset(args.save_dir, args.max_images)