from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, Tuple

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.roboflow_coco import COCOSemanticSegmentationDataset
from models.unet_model import build_unet
from utils.common import count_trainable_parameters, ensure_dir, get_device, load_yaml, save_checkpoint, set_seed
from utils.losses import build_segmentation_loss
from utils.metrics import segmentation_metrics, summarize_metric_list
from utils.transforms import get_unet_transforms


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train U-Net from one Roboflow COCO segmentation dataset")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    return parser.parse_args()


def _build_loader_kwargs(data_cfg: Dict, device: torch.device, shuffle: bool) -> Dict:
    num_workers = int(data_cfg.get("num_workers", 0))
    kwargs = {
        "batch_size": data_cfg["batch_size"],
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": device.type == "cuda",
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = bool(data_cfg.get("persistent_workers", False))
        kwargs["prefetch_factor"] = int(data_cfg.get("prefetch_factor", 1))
    return kwargs


def build_dataloaders(cfg: Dict, device: torch.device):
    data_cfg = cfg["data"]
    img_size = int(data_cfg["img_size"])
    train_ds = COCOSemanticSegmentationDataset(
        root_dir=data_cfg["root_dir"],
        split=data_cfg.get("train_split", "train"),
        task_mode=data_cfg.get("task_mode", "multiclass"),
        transform=get_unet_transforms(img_size, train=True),
    )
    val_ds = COCOSemanticSegmentationDataset(
        root_dir=data_cfg["root_dir"],
        split=data_cfg.get("val_split", "valid"),
        task_mode=data_cfg.get("task_mode", "multiclass"),
        transform=get_unet_transforms(img_size, train=False),
    )

    train_loader = DataLoader(train_ds, **_build_loader_kwargs(data_cfg, device, shuffle=True))
    val_loader = DataLoader(val_ds, **_build_loader_kwargs(data_cfg, device, shuffle=False))
    return train_ds, val_ds, train_loader, val_loader


def train_one_epoch(model, loader, criterion, optimizer, device, scaler, amp_enabled: bool, task_mode: str, threshold: float, num_classes: int, channels_last: bool = False, grad_accum_steps: int = 1):
    model.train()
    running_loss = 0.0
    metric_list = []
    optimizer.zero_grad(set_to_none=True)

    for step, (images, masks) in enumerate(tqdm(loader, desc="[U-Net] Train", leave=False), start=1):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        if channels_last and images.ndim == 4:
            images = images.to(memory_format=torch.channels_last)

        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
            logits = model(images)
            loss = criterion(logits, masks)
            loss_for_backward = loss / grad_accum_steps

        scaler.scale(loss_for_backward).backward()

        if step % grad_accum_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        running_loss += float(loss.item())
        metric_list.append(
            segmentation_metrics(
                logits=logits.detach(),
                targets=masks.detach(),
                task_mode=task_mode,
                threshold=threshold,
                num_classes=num_classes,
            )
        )

    if len(loader) % grad_accum_steps != 0:
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

    return running_loss / max(1, len(loader)), summarize_metric_list(metric_list)


def validate(model, loader, criterion, device, task_mode: str, threshold: float, num_classes: int, amp_enabled: bool = False, channels_last: bool = False):
    model.eval()
    running_loss = 0.0
    metric_list = []

    with torch.no_grad():
        for images, masks in tqdm(loader, desc="[U-Net] Val", leave=False):
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            if channels_last and images.ndim == 4:
                images = images.to(memory_format=torch.channels_last)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
                logits = model(images)
                loss = criterion(logits, masks)
            running_loss += float(loss.item())
            metric_list.append(
                segmentation_metrics(
                    logits=logits,
                    targets=masks,
                    task_mode=task_mode,
                    threshold=threshold,
                    num_classes=num_classes,
                )
            )

    return running_loss / max(1, len(loader)), summarize_metric_list(metric_list)


def main():
    args = parse_args()
    cfg = load_yaml(args.config)
    set_seed(cfg.get("seed", 42))

    device = get_device(cfg.get("device", "cuda"))
    output_dir = Path(cfg["output_dir"])
    ensure_dir(output_dir)

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = bool(cfg.get("train", {}).get("cudnn_benchmark", True))

    train_ds, val_ds, train_loader, val_loader = build_dataloaders(cfg, device)
    task_mode = cfg["data"].get("task_mode", "multiclass")
    num_classes = train_ds.num_segmentation_classes
    class_names = train_ds.class_names
    cfg.setdefault("data", {})["class_names"] = class_names

    model_cfg = dict(cfg["model"])
    model_cfg["classes"] = num_classes
    model = build_unet(**model_cfg).to(device)

    channels_last = bool(cfg["train"].get("channels_last", False) and device.type == "cuda")
    if channels_last:
        model = model.to(memory_format=torch.channels_last)

    use_compile = bool(cfg["train"].get("compile", False) and device.type == "cuda" and hasattr(torch, "compile"))
    if use_compile:
        model = torch.compile(model, mode=cfg["train"].get("compile_mode", "reduce-overhead"))

    print(f"[INFO] Device: {device}")
    print(f"[INFO] Task mode: {task_mode}")
    print(f"[INFO] Segmentation classes: {num_classes}")
    print(f"[INFO] Class names: {class_names}")
    print(f"[INFO] Trainable params: {count_trainable_parameters(model):,}")
    print(f"[INFO] Train size: {len(train_ds)} | Val size: {len(val_ds)}")
    print(f"[INFO] AMP: {bool(cfg['train'].get('amp', True) and device.type == 'cuda')} | channels_last: {channels_last} | compile: {use_compile}")

    criterion = build_segmentation_loss(task_mode)
    optimizer = AdamW(
        model.parameters(),
        lr=cfg["train"]["learning_rate"],
        weight_decay=cfg["train"]["weight_decay"],
    )
    amp_enabled = bool(cfg["train"].get("amp", True) and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    grad_accum_steps = max(1, int(cfg["train"].get("grad_accum_steps", 1)))

    history_path = output_dir / "history.csv"
    with open(history_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "epoch",
            "train_loss",
            "val_loss",
            "train_dice",
            "val_dice",
            "train_iou",
            "val_iou",
            "val_precision",
            "val_recall",
            "val_pixel_acc",
        ])

    monitor_metric = str(cfg["train"].get("monitor_metric", "val_dice")).strip().lower()
    maximize_metrics = {"val_dice", "val_iou", "val_precision", "val_recall", "val_pixel_acc"}
    valid_monitor_metrics = maximize_metrics | {"val_loss"}
    if monitor_metric not in valid_monitor_metrics:
        raise ValueError(f"Unsupported monitor_metric: {monitor_metric}. Supported: {sorted(valid_monitor_metrics)}")

    if monitor_metric == "val_loss":
        best_monitor_value = float("inf")
    else:
        best_monitor_value = float("-inf")

    epochs = int(cfg["train"]["epochs"])
    threshold = float(cfg["train"].get("threshold", 0.5))
    patience = int(cfg["train"].get("early_stopping_patience", 0))
    epochs_without_improvement = 0

    for epoch in range(1, epochs + 1):
        train_loss, train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            scaler=scaler,
            amp_enabled=amp_enabled,
            task_mode=task_mode,
            threshold=threshold,
            num_classes=num_classes,
            channels_last=channels_last,
            grad_accum_steps=grad_accum_steps,
        )
        val_loss, val_metrics = validate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            task_mode=task_mode,
            threshold=threshold,
            num_classes=num_classes,
            amp_enabled=amp_enabled,
            channels_last=channels_last,
        )

        print(
            f"Epoch {epoch:03d}/{epochs} | "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} | "
            f"train_dice={train_metrics.get('dice', 0):.4f} val_dice={val_metrics.get('dice', 0):.4f} | "
            f"val_iou={val_metrics.get('iou', 0):.4f} val_pixel_acc={val_metrics.get('pixel_acc', 0):.4f}"
        )

        with open(history_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch,
                train_loss,
                val_loss,
                train_metrics.get("dice", 0.0),
                val_metrics.get("dice", 0.0),
                train_metrics.get("iou", 0.0),
                val_metrics.get("iou", 0.0),
                val_metrics.get("precision", 0.0),
                val_metrics.get("recall", 0.0),
                val_metrics.get("pixel_acc", 0.0),
            ])

        monitor_value_map = {
            "val_loss": val_loss,
            "val_dice": float(val_metrics.get("dice", 0.0)),
            "val_iou": float(val_metrics.get("iou", 0.0)),
            "val_precision": float(val_metrics.get("precision", 0.0)),
            "val_recall": float(val_metrics.get("recall", 0.0)),
            "val_pixel_acc": float(val_metrics.get("pixel_acc", 0.0)),
        }
        current_monitor_value = monitor_value_map[monitor_metric]

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": cfg,
            "val_loss": val_loss,
            "val_metrics": val_metrics,
            "monitor_metric": monitor_metric,
            "monitor_value": current_monitor_value,
            "class_names": class_names,
            "task_mode": task_mode,
        }
        latest_path = output_dir / "last_unet.pth"
        save_checkpoint(checkpoint, str(latest_path))

        improved = (
            current_monitor_value < best_monitor_value
            if monitor_metric == "val_loss"
            else current_monitor_value > best_monitor_value
        )

        if improved:
            best_monitor_value = current_monitor_value
            epochs_without_improvement = 0
            best_path = output_dir / "best_unet.pth"
            save_checkpoint(checkpoint, str(best_path))
            print(
                f"[INFO] Saved best checkpoint to: {best_path} "
                f"({monitor_metric}={current_monitor_value:.4f})"
            )
        else:
            epochs_without_improvement += 1
            if patience > 0 and epochs_without_improvement >= patience:
                print(
                    f"[INFO] Early stopping triggered after {patience} epochs without improvement "
                    f"on {monitor_metric}."
                )
                break

    print(f"[DONE] Training U-Net finished. History saved to {history_path}")


if __name__ == "__main__":
    main()
