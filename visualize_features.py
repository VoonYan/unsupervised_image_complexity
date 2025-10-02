# visualize_features.py
"""
Visualize learned features and attention maps
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns
from PIL import Image
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from evaluate_model import CLICEvaluator


def extract_features(evaluator, image_dir, max_images=100):
    """Extract features from multiple images"""

    features_list = []
    scores_list = []

    for i, img_file in enumerate(os.listdir(image_dir)):
        if i >= max_images:
            break

        if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
            img_path = os.path.join(image_dir, img_file)
            score, features = evaluator.predict_complexity(img_path)

            # Get the last layer features
            feat = features['layer4']
            feat_avg = torch.mean(feat, dim=[2, 3]).squeeze().numpy()

            features_list.append(feat_avg)
            scores_list.append(score)

    return np.array(features_list), np.array(scores_list)


def visualize_tsne(features, scores):
    """Create t-SNE visualization"""

    print("Computing t-SNE...")
    tsne = TSNE(n_components=2, random_state=42)
    features_2d = tsne.fit_transform(features)

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1],
                          c=scores, cmap='viridis', s=50, alpha=0.7)
    plt.colorbar(scatter, label='Complexity Score')
    plt.title('t-SNE Visualization of Image Features')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.savefig('./tsne_visualization.png')
    plt.show()


def visualize_attention_maps(evaluator, image_path):
    """Visualize attention maps from different layers"""

    # Get features
    score, features = evaluator.predict_complexity(image_path)

    # Create attention maps
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))

    # Original image
    img = Image.open(image_path)
    axes[0].imshow(img)
    axes[0].set_title(f'Original\nComplexity: {score:.3f}')
    axes[0].axis('off')

    # Layer attention maps
    for i, (layer_name, layer_features) in enumerate(features.items()):
        if i < 4:
            # Average across channels
            attention = torch.mean(layer_features, dim=1).squeeze().numpy()

            # Normalize
            attention = (attention - attention.min()) / (attention.max() - attention.min())

            axes[i + 1].imshow(attention, cmap='hot')
            axes[i + 1].set_title(f'{layer_name}')
            axes[i + 1].axis('off')

    plt.suptitle('Feature Activation Maps Across Layers')
    plt.tight_layout()
    plt.savefig('./attention_maps.png')
    plt.show()


if __name__ == "__main__":
    # Initialize evaluator
    evaluator = CLICEvaluator()

    # Extract features
    print("Extracting features...")
    features, scores = extract_features(evaluator, "./data/clic_dataset/images/", max_images=50)

    # Create t-SNE visualization
    visualize_tsne(features, scores)

    # Visualize attention maps
    test_image = "./data/clic_dataset/images/000001.jpg"
    if os.path.exists(test_image):
        visualize_attention_maps(evaluator, test_image)