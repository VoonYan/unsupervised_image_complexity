# evaluate_model.py
"""
Evaluate the trained CLIC model
"""

import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from clic.icnet import ICNet_ft


class CLICEvaluator:
    def __init__(self, checkpoint_path='./checkpoints/best_finetuned_model.pth'):
        """Initialize evaluator with trained model"""

        self.device = torch.device('cpu')

        # Load model
        self.model = ICNet_ft()
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

        # Image transform
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        print(f"Loaded model from {checkpoint_path}")
        print(f"Model performance - Pearson: {checkpoint.get('pearson', 'N/A'):.4f}")

    def predict_complexity(self, image_path):
        """Predict complexity score for a single image"""

        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)

        # Predict
        with torch.no_grad():
            score, features = self.model(image_tensor)
            complexity_score = score.item()

        return complexity_score, features

    def evaluate_directory(self, image_dir):
        """Evaluate all images in a directory"""

        results = []

        for img_file in os.listdir(image_dir):
            if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(image_dir, img_file)
                score, _ = self.predict_complexity(img_path)
                results.append({
                    'image': img_file,
                    'complexity_score': score
                })
                print(f"{img_file}: {score:.4f}")

        # Sort by complexity
        results.sort(key=lambda x: x['complexity_score'])

        return results

    def visualize_complexity_ranking(self, image_dir, top_k=10):
        """Visualize images ranked by complexity"""

        results = self.evaluate_directory(image_dir)

        # Get top and bottom k images
        simplest = results[:top_k]
        most_complex = results[-top_k:]

        # Create visualization
        fig, axes = plt.subplots(2, top_k, figsize=(20, 8))
        fig.suptitle('Image Complexity Ranking', fontsize=16)

        # Plot simplest images
        for i, item in enumerate(simplest):
            img_path = os.path.join(image_dir, item['image'])
            img = Image.open(img_path)
            axes[0, i].imshow(img)
            axes[0, i].set_title(f"Score: {item['complexity_score']:.3f}")
            axes[0, i].axis('off')

        axes[0, 0].set_ylabel('Simplest', fontsize=12)

        # Plot most complex images
        for i, item in enumerate(most_complex):
            img_path = os.path.join(image_dir, item['image'])
            img = Image.open(img_path)
            axes[1, i].imshow(img)
            axes[1, i].set_title(f"Score: {item['complexity_score']:.3f}")
            axes[1, i].axis('off')

        axes[1, 0].set_ylabel('Most Complex', fontsize=12)

        plt.tight_layout()
        plt.savefig('./complexity_ranking.png')
        plt.show()

        return results


if __name__ == "__main__":
    # Initialize evaluator
    evaluator = CLICEvaluator()

    # Evaluate single image
    print("\n" + "=" * 60)
    print("Single Image Evaluation")
    print("=" * 60)

    # You can test with any image
    test_image = "./data/clic_dataset/images/000001.jpg"
    if os.path.exists(test_image):
        score, _ = evaluator.predict_complexity(test_image)
        print(f"Complexity score: {score:.4f}")

    # Evaluate directory
    print("\n" + "=" * 60)
    print("Directory Evaluation")
    print("=" * 60)

    results = evaluator.evaluate_directory("./data/clic_dataset/images/")

    # Visualize ranking
    print("\n" + "=" * 60)
    print("Creating Visualization")
    print("=" * 60)

    evaluator.visualize_complexity_ranking("./data/clic_dataset/images/", top_k=5)