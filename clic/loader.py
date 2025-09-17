# clic/loader.py
# !/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
CLIC datasets and transforms
"""

import os
import random
import numpy as np
import torch
from PIL import Image, ImageFilter
from torch.utils.data import Dataset
from torchvision import transforms


class GaussianBlur(object):
    """Gaussian blur augmentation"""

    def __init__(self, sigma=[0.1, 2.0]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


class TwoCropsTransform:
    """Take two random crops of one image as the query and key."""

    def __init__(self):
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.2, 1.0)),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur([0.1, 2.0])], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def __call__(self, x):
        return [self.transform(x), self.transform(x)]


class CLICDataset(Dataset):
    """
    CLIC Dataset for unsupervised training
    """

    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir: Directory with all the images
            transform: Transform to be applied on a sample
        """
        self.root_dir = root_dir
        self.transform = transform if transform else TwoCropsTransform()

        # Get all image files
        self.image_files = []
        valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')

        for file in os.listdir(root_dir):
            if file.lower().endswith(valid_extensions):
                self.image_files.append(file)

        print(f"Found {len(self.image_files)} images in {root_dir}")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.image_files[idx])

        # Load image
        image = Image.open(img_path).convert('RGB')

        # Apply transform
        if self.transform:
            image = self.transform(image)

        return image, 0  # Return dummy label for compatibility


class ICDataset(Dataset):
    """ICDataset for IC9600 fine-tuning"""

    def __init__(self, txt_path, img_path, transform=None):
        super(ICDataset, self).__init__()
        self.txt_lines = self.readlines(txt_path)
        self.img_path = img_path
        self.transform = transform
        self.img_info_list = self.parse_lines(self.txt_lines)

    def parse_lines(self, lines):
        image_info_list = []
        for line in lines:
            line_split = line.strip().split("  ")
            if len(line_split) >= 2:
                img_name = line_split[0]
                img_label = line_split[1]
                image_info_list.append((img_name, img_label))
        return image_info_list

    def readlines(self, txt_path):
        with open(txt_path, 'r') as f:
            lines = f.readlines()
        return lines

    def __getitem__(self, index):
        imgName, imgLabel = self.img_info_list[index]
        oriImgPath = os.path.join(self.img_path, imgName)
        img = Image.open(oriImgPath).convert("RGB")

        if self.transform:
            img = self.transform(img)

        label = torch.tensor(float(imgLabel))
        return img, label, imgName

    def __len__(self):
        return len(self.img_info_list)