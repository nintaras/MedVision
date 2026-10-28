import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

class MedicalResNet50(nn.Module):
    """
    ResNet-50 προσαρμοσμένο για ταξινόμηση ιατρικών εικόνων (Chest X-Rays).
    Χρησιμοποιεί ImageNet weights ως feature extractor και custom head
    με LayerNorm και Dropout για αποφυγή overfitting.
    """
    def __init__(self, num_classes: int = 2, pretrained: bool = True):
        super().__init__()
        weights = ResNet50_Weights.DEFAULT if pretrained else None
        self.backbone = resnet50(weights=weights)

        # Αντικατάσταση του τελικού Fully Connected layer
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)
