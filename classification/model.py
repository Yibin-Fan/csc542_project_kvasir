import torch.nn as nn
from torchvision import models


class ResNetClassifier(nn.Module):
    """
    ResNet-18 fine-tuned for 8-class Kvasir image classification.

    Uses ImageNet-pretrained weights. The original FC head is replaced
    with a single linear layer mapping 512 -> num_classes.

    Architecture:
        ResNet-18 backbone (pretrained)
        └── fc: Linear(512 -> num_classes)
    """

    def __init__(self, num_classes: int = 8):
        super().__init__()
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.backbone(x)   # raw logits


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
