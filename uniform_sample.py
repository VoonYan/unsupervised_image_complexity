# uniform_sample.py
import os
import shutil
import random


def create_clic_dataset(imagenet_dir, flickr_dir, output_dir, num_samples=10000):
    """
    Create CLIC dataset by uniformly sampling from ImageNet and Flickr
    """
    os.makedirs(output_dir, exist_ok=True)

    # Get all image paths
    imagenet_images = []
    flickr_images = []

    # Collect ImageNet images
    for root, dirs, files in os.walk(imagenet_dir):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                imagenet_images.append(os.path.join(root, file))

    # Collect Flickr images
    for root, dirs, files in os.walk(flickr_dir):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                flickr_images.append(os.path.join(root, file))

    # Sample uniformly
    num_imagenet = num_samples // 2
    num_flickr = num_samples - num_imagenet

    sampled_imagenet = random.sample(imagenet_images, min(num_imagenet, len(imagenet_images)))
    sampled_flickr = random.sample(flickr_images, min(num_flickr, len(flickr_images)))

    # Copy to output directory
    count = 1
    for img_path in sampled_imagenet + sampled_flickr:
        ext = os.path.splitext(img_path)[1]
        output_path = os.path.join(output_dir, f"{count:06d}{ext}")
        shutil.copy2(img_path, output_path)
        count += 1
        if count % 100 == 0:
            print(f"Processed {count} images...")

    print(f"Created CLIC dataset with {count - 1} images in {output_dir}")


if __name__ == "__main__":
    create_clic_dataset(
        imagenet_dir="./ImageNet/train",
        flickr_dir="./Flickr/train",
        output_dir="./clic_data/images",
        num_samples=10000
    )