from __future__ import annotations

from typing import Dict, List

import numpy as np
import torch


def binary_segmentation_metrics(
    logits: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
    eps: float = 1e-7,
) -> Dict[str, float]:
    probs = torch.sigmoid(logits)
    preds = (probs >= threshold).float()
    targets = (targets >= 0.5).float()

    preds = preds.view(preds.size(0), -1)
    targets = targets.view(targets.size(0), -1)

    intersection = (preds * targets).sum(dim=1)
    union = preds.sum(dim=1) + targets.sum(dim=1) - intersection

    dice = ((2 * intersection + eps) / (preds.sum(dim=1) + targets.sum(dim=1) + eps)).mean()
    iou = ((intersection + eps) / (union + eps)).mean()
    pixel_acc = (preds.eq(targets).float().mean(dim=1)).mean()

    tp = intersection
    fp = (preds * (1 - targets)).sum(dim=1)
    fn = ((1 - preds) * targets).sum(dim=1)
    precision = ((tp + eps) / (tp + fp + eps)).mean()
    recall = ((tp + eps) / (tp + fn + eps)).mean()

    return {
        "dice": float(dice.item()),
        "iou": float(iou.item()),
        "pixel_acc": float(pixel_acc.item()),
        "precision": float(precision.item()),
        "recall": float(recall.item()),
    }


def multiclass_segmentation_metrics(
    logits: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
    ignore_background: bool = True,
    eps: float = 1e-7,
) -> Dict[str, float]:
    preds = torch.argmax(logits, dim=1)
    targets = targets.long()

    pixel_acc = preds.eq(targets).float().mean().item()
    class_range = range(1, num_classes) if ignore_background and num_classes > 1 else range(num_classes)

    ious: List[float] = []
    dices: List[float] = []
    precisions: List[float] = []
    recalls: List[float] = []

    for cls_idx in class_range:
        pred_mask = preds == cls_idx
        target_mask = targets == cls_idx

        pred_sum = pred_mask.sum().float()
        target_sum = target_mask.sum().float()
        intersection = (pred_mask & target_mask).sum().float()
        union = pred_sum + target_sum - intersection

        if target_sum == 0 and pred_sum == 0:
            continue

        iou = (intersection + eps) / (union + eps)
        dice = (2 * intersection + eps) / (pred_sum + target_sum + eps)
        precision = (intersection + eps) / (pred_sum + eps)
        recall = (intersection + eps) / (target_sum + eps)

        ious.append(float(iou.item()))
        dices.append(float(dice.item()))
        precisions.append(float(precision.item()))
        recalls.append(float(recall.item()))

    return {
        "dice": float(np.mean(dices) if dices else 0.0),
        "iou": float(np.mean(ious) if ious else 0.0),
        "pixel_acc": float(pixel_acc),
        "precision": float(np.mean(precisions) if precisions else 0.0),
        "recall": float(np.mean(recalls) if recalls else 0.0),
    }


def segmentation_metrics(
    logits: torch.Tensor,
    targets: torch.Tensor,
    task_mode: str,
    threshold: float = 0.5,
    num_classes: int = 2,
) -> Dict[str, float]:
    if task_mode == "binary":
        return binary_segmentation_metrics(logits=logits, targets=targets, threshold=threshold)
    return multiclass_segmentation_metrics(logits=logits, targets=targets, num_classes=num_classes)


def summarize_metric_list(metric_list: List[Dict[str, float]]) -> Dict[str, float]:
    if not metric_list:
        return {}
    keys = metric_list[0].keys()
    return {k: float(np.mean([m[k] for m in metric_list])) for k in keys}
