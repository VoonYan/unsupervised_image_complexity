# clic/icnet.py
# !/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
ICNet for fine-tuning on IC9600
"""

import torch
import torch.nn as nn
import torchvision.models as models


class ICNet_ft(nn.Module):
    """
    Image Complexity Network for fine-tuning
    """

    def __init__(self, pretrained_path=None, feature_dim=128):
        super(ICNet_ft, self).__init__()

        # Load ResNet50 backbone
        self.backbone = models.resnet50(pretrained=False)

        # Modify the final FC layer for regression
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Linear(num_features, feature_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(feature_dim, 1)  # Single output for complexity score
        )

        # Load pretrained weights if provided
        if pretrained_path:
            self.load_pretrained(pretrained_path)

    def load_pretrained(self, checkpoint_path):
        """Load pretrained CLIC weights"""
        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        # Load encoder_q weights from CLIC
        state_dict = checkpoint['state_dict']

        # Filter and rename keys
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.encoder_q.'):
                # Remove 'module.encoder_q.' prefix
                new_key = k.replace('module.encoder_q.', '')
                if not new_key.startswith('fc'):  # Skip the FC layer
                    new_state_dict[new_key] = v

        # Load weights
        self.backbone.load_state_dict(new_state_dict, strict=False)
        print(f"Loaded pretrained weights from {checkpoint_path}")

    def forward(self, x):
        """
        Forward pass
        Returns: (complexity_score, features)
        """
        # Get features from different stages
        features = {}

        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        features['layer1'] = x

        x = self.backbone.layer2(x)
        features['layer2'] = x

        x = self.backbone.layer3(x)
        features['layer3'] = x

        x = self.backbone.layer4(x)
        features['layer4'] = x

        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)

        # Get complexity score
        score = self.backbone.fc(x)

        return score, features