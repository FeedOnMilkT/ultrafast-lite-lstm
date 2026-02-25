import torch


def top1_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    pred = torch.argmax(logits, dim=1)
    correct = (pred == labels).float().mean()
    return float(correct.detach().cpu())
