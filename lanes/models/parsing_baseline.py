import torch
import torch.nn as nn
from torchvision import models

from lanes.models.heads import ClassificationHead


class ParsingBaseline(nn.Module):
    """
    Minimal single-frame baseline.
    Output shape: [B, griding_num + 1, num_rows, num_lanes]
    """

    def __init__(self, griding_num: int, num_rows: int, num_lanes: int):
        super().__init__()
        backbone = models.resnet18(weights=None)
        self.encoder = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,
            backbone.layer2,
            backbone.layer3,
            backbone.layer4,
        )
        self.resize = nn.AdaptiveAvgPool2d((9, 25))
        self.channel_pool = nn.Conv2d(512, 8, kernel_size=1)

        self.cls_dim = (griding_num + 1, num_rows, num_lanes)
        total_dim = self.cls_dim[0] * self.cls_dim[1] * self.cls_dim[2]
        self.head = ClassificationHead(in_features=1800, hidden_features=1024, out_features=total_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.encoder(x)
        feat = self.resize(feat)
        feat = self.channel_pool(feat)
        feat = feat.flatten(start_dim=1)
        logits = self.head(feat).view(-1, *self.cls_dim)
        return logits
