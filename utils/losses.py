from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class BCEDiceLoss(nn.Module):
    def __init__(self, bce_weight: float = 0.5, dice_weight: float = 0.5, eps: float = 1e-7):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.eps = eps
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = self.bce(logits, targets)
        probs = torch.sigmoid(logits)
        probs = probs.flatten(1)
        targets = targets.flatten(1)
        intersection = (probs * targets).sum(dim=1)
        dice = (2 * intersection + self.eps) / (probs.sum(dim=1) + targets.sum(dim=1) + self.eps)
        dice_loss = 1 - dice.mean()
        return self.bce_weight * bce + self.dice_weight * dice_loss


class MulticlassDiceCrossEntropyLoss(nn.Module):
    def __init__(self, ce_weight: float = 0.5, dice_weight: float = 0.5, eps: float = 1e-7):
        super().__init__()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, targets)
        probs = torch.softmax(logits, dim=1)
        one_hot = F.one_hot(targets.long(), num_classes=logits.shape[1]).permute(0, 3, 1, 2).float()

        dims = (0, 2, 3)
        intersection = (probs * one_hot).sum(dim=dims)
        cardinality = probs.sum(dim=dims) + one_hot.sum(dim=dims)
        dice_per_class = (2 * intersection + self.eps) / (cardinality + self.eps)
        foreground = dice_per_class[1:] if dice_per_class.numel() > 1 else dice_per_class
        dice_loss = 1 - foreground.mean()
        return self.ce_weight * ce + self.dice_weight * dice_loss


def build_segmentation_loss(task_mode: str) -> nn.Module:
    if task_mode == "binary":
        return BCEDiceLoss()
    return MulticlassDiceCrossEntropyLoss()
