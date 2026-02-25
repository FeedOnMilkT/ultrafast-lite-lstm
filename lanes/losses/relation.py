import torch
import torch.nn as nn
import torch.nn.functional as F


class ParsingRelationLoss(nn.Module):
    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        _, _, num_rows, _ = logits.shape
        if num_rows < 2:
            return logits.new_tensor(0.0)
        diffs = [logits[:, :, i, :] - logits[:, :, i + 1, :] for i in range(num_rows - 1)]
        stacked = torch.cat(diffs, dim=0)
        return F.smooth_l1_loss(stacked, torch.zeros_like(stacked))


class ParsingRelationDis(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.L1Loss()

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        n, dim, num_rows, _ = logits.shape
        if num_rows < 3 or dim <= 1:
            return logits.new_tensor(0.0)
        prob = torch.softmax(logits[:, : dim - 1, :, :], dim=1)
        embedding = torch.arange(dim - 1, device=logits.device, dtype=prob.dtype).view(1, -1, 1, 1)
        pos = torch.sum(prob * embedding, dim=1)
        diffs = [pos[:, i, :] - pos[:, i + 1, :] for i in range(num_rows // 2)]
        if len(diffs) < 2:
            return logits.new_tensor(0.0)
        loss = sum(self.l1(diffs[i], diffs[i + 1]) for i in range(len(diffs) - 1))
        return loss / (len(diffs) - 1)
