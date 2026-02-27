import torch
import torch.nn as nn
from torchvision import models

from lanes.models.heads import ClassificationHead


class ParsingTemporalLSTM(nn.Module):
    """
    Temporal lane parser.
    Input:
      - sequence: [B, T, 3, H, W]
      - single frame fallback: [B, 3, H, W]
    Output:
      - logits for the last time step: [B, griding_num + 1, num_rows, num_lanes]
    """

    def __init__(
        self,
        griding_num: int,
        num_rows: int,
        num_lanes: int,
        lstm_hidden_size: int = 512,
        lstm_num_layers: int = 1,
        lstm_dropout: float = 0.0,
        head_hidden_features: int = 1024,
        pretrained_backbone: bool = True,
    ):
        super().__init__()
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained_backbone else None
        backbone = models.resnet18(weights=weights)
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
        self.feature_dim = 1800

        effective_dropout = lstm_dropout if lstm_num_layers > 1 else 0.0
        self.temporal = nn.LSTM(
            input_size=self.feature_dim,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            dropout=effective_dropout,
            batch_first=True,
        )

        self.cls_dim = (griding_num + 1, num_rows, num_lanes)
        total_dim = self.cls_dim[0] * self.cls_dim[1] * self.cls_dim[2]
        self.head = ClassificationHead(
            in_features=lstm_hidden_size,
            hidden_features=head_hidden_features,
            out_features=total_dim,
        )

    def _extract_frame_features(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.encoder(x)
        feat = self.resize(feat)
        feat = self.channel_pool(feat)
        return feat.flatten(start_dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 4:
            x = x.unsqueeze(1)
        if x.ndim != 5:
            raise ValueError(f"expected input dim 4 or 5, got shape={tuple(x.shape)}")

        b, t, c, h, w = x.shape
        if c != 3:
            raise ValueError(f"expected channel=3, got channel={c}")

        x_bt = x.reshape(b * t, c, h, w)
        frame_feat = self._extract_frame_features(x_bt)
        frame_feat = frame_feat.view(b, t, self.feature_dim)

        temporal_out, _ = self.temporal(frame_feat)
        last_feat = temporal_out[:, -1, :]
        logits = self.head(last_feat).view(-1, *self.cls_dim)
        return logits
