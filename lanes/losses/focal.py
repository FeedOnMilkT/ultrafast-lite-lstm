import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftmaxFocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0):
        super().__init__()
        self.gamma = gamma
        self.nll = nn.NLLLoss()

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        scores = F.softmax(logits, dim=1)
        factor = torch.pow(1.0 - scores, self.gamma)
        log_prob = F.log_softmax(logits, dim=1)
        return self.nll(factor * log_prob, labels)
