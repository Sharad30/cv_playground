# classification/model.py
import torch
import torch.nn as nn
from torchvision import models

class BinaryClassifier(nn.Module):
    def __init__(self, model_name='resnet18', pretrained=True, freeze_backbone=True):
        super(BinaryClassifier, self).__init__()
        self.model_name = model_name.lower()
        
        # Dictionary of available models and their configurations
        self.model_configs = {
            'resnet18': (models.resnet18, 512),
            'resnet34': (models.resnet34, 512),
            'resnet50': (models.resnet50, 2048),
            'efficientnet_b0': (models.efficientnet_b0, 1280),
            'efficientnet_b1': (models.efficientnet_b1, 1280),
            'vit_b_16': (models.vit_b_16, 768),
            'mobilenet_v3_small': (models.mobilenet_v3_small, 576),
            'mobilenet_v3_large': (models.mobilenet_v3_large, 960),
        }

        if model_name not in self.model_configs:
            raise ValueError(f"Model {model_name} not supported. Available models: {list(self.model_configs.keys())}")

        # Get model configuration
        model_fn, num_features = self.model_configs[model_name]

        # Load pre-trained model
        if pretrained:
            self.backbone = model_fn(weights='IMAGENET1K_V1')
        else:
            self.backbone = model_fn(weights=None)

        # Freeze backbone if specified
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Modify the final layer based on the model architecture
        if 'resnet' in model_name:
            self.backbone.fc = nn.Sequential(
                nn.Linear(num_features, 256),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, 1),
                nn.Sigmoid()
            )
        elif 'efficientnet' in model_name:
            self.backbone.classifier = nn.Sequential(
                nn.Linear(num_features, 256),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, 1),
                nn.Sigmoid()
            )
        elif 'vit' in model_name:
            self.backbone.heads = nn.Sequential(
                nn.Linear(num_features, 256),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, 1),
                nn.Sigmoid()
            )
        elif 'mobilenet' in model_name:
            self.backbone.classifier = nn.Sequential(
                nn.Linear(num_features, 256),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, 1),
                nn.Sigmoid()
            )

    def forward(self, x):
        return self.backbone(x)

    def unfreeze_layers(self, num_layers=None):
        """Unfreeze the last n layers of the backbone"""
        if num_layers is None:
            # Unfreeze all layers
            for param in self.backbone.parameters():
                param.requires_grad = True
        else:
            # Get all parameters
            parameters = list(self.backbone.parameters())
            # Unfreeze the last n layers
            for param in parameters[-num_layers:]:
                param.requires_grad = True

    def get_trainable_params(self):
        """Return the number of trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)