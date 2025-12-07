# src/models.py
import torch
import torch.nn as nn
from torchvision import models

class ResNet50Melanoma(nn.Module):
    def __init__(self, pretrained: bool = True, num_classes: int = 2):
        super().__init__()
        self.backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.backbone(x)

def create_model(pretrained: bool = True, num_classes: int = 2) -> nn.Module:
    model = ResNet50Melanoma(pretrained=pretrained, num_classes=num_classes)
    return model
