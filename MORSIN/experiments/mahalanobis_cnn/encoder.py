"""ResNet18 encoder on resized multi-channel log-mel -> embedding."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class MelResNetEncoder(nn.Module):
    def __init__(self, embed_dim: int = 128, *, pretrained: bool = True) -> None:
        super().__init__()
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.resnet18(weights=weights)
        self.stem = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
        )
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        self.pool = backbone.avgpool
        self.fc = nn.Linear(512, embed_dim)

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        """
        mel: [B, 3, F, T] log-mel stack (3 resolutions as channels).
        Returns L2-normalized embeddings [B, embed_dim].
        """
        x = F.interpolate(mel, size=(224, 224), mode="bilinear", align_corners=False)
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        z = self.fc(x)
        return F.normalize(z, dim=1)
