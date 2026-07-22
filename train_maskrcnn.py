from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.roboflow_coco import COCOMaskRCNNDataset, detection_collate_fn
from models.maskrcnn_model import build_maskrcnn
from utils.common import count_trainable_parameters, ensure_dir, get_device, load_yaml, save_checkpoint, set_seed
from utils.transforms import get_maskrcnn_image_transform


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Mask R-CNN from one Roboflow COCO segmentation dataset")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    return parser.parse_args()


def _build_loader_kwargs(data_cfg: Dict, device: torch.device, shuffle: bool) -> Dict:
    num_workers = int(data_cfg.get("num_workers", 0))
    kwargs = {
        "batch_size": data_cfg["batch_size"],
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": device.type == "cuda",
        "collate_fn": detection_collate_fn,
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = bool(data_cfg.get("persistent_workers", False))
        kwargs["prefetch_factor"] = int(data_cfg.get("prefetch_factor", 1))
    return kwargs


def build_dataloaders(cfg: Dict, device: torch.device):
    data_cfg = cfg["data"]
    img_size = int(data_cfg.get("img_size", 512))
    train_ds = COCOMaskRCNNDataset(
        root_dir=data_cfg["root_dir"],
        split=data_cfg.get("train_split", "train"),
        transforms=get_maskrcnn_image_transform(img_size=img_size, train=True),
    )
    val_ds = COCOMaskRCNNDataset(
        root_dir=data_cfg["root_dir"],
        split=data_cfg.get("val_split", "valid"),
        transforms=get_maskrcnn_image_transform(img_size=img_size, train=False),
    )

    train_loader = DataLoader(train_ds, **_build_loader_kwargs(data_cfg, device, shuffle=True))
    val_loader = DataLoader(val_ds, **_build_loader_kwargs(data_cfg, device, shuffle=False))
    return train_ds, val_ds, train_loader, val_loader


def set_backbone_trainable(model: torch.nn.Module, trainable: bool) -> None:
    for parameter in model.backbone.parameters():
        parameter.requires_grad = trainable


def train_one_epoch(model, loader, optimizer, device, scaler, amp_enabled: bool):
    model.train()
    running_loss = 0.0
    components = {"loss_classifier": 0.0, "loss_box_reg": 0.0, "loss_mask": 0.0, "loss_objectness": 0.0, "loss_rpn_box_reg": 0.0}

    for images, targets in tqdm(loader, desc="[Mask R-CNN] Train", leave=False):
        images = [img.to(device, non_blocking=True) for img in images]
        targets = [{k: v.to(device, non_blocking=True) for k, v in t.items()} for t in targets]

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
        scaler.scale(losses).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += float(losses.item())
        for key in components:
            if key in loss_dict:
                components[key] += float(loss_dict[key].item())

    num_batches = max(1, len(loader))
    return running_loss / num_batches, {k: v / num_batches for k, v in components.items()}


def validate_loss(model, loader, device, amp_enabled: bool):
    model.train()
    running_loss = 0.0
    components = {"loss_classifier": 0.0, "loss_box_reg": 0.0, "loss_mask": 0.0, "loss_objectness": 0.0, "loss_rpn_box_reg": 0.0}

    with torch.no_grad():
        for images, targets in tqdm(loader, desc="[Mask R-CNN] ValLoss", leave=False):
            images = [img.to(device, non_blocking=True) for img in images]
            targets = [{k: v.to(device, non_blocking=True) for k, v in t.items()} for t in targets]
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
            running_loss += float(losses.item())
            for key in components:
                if key in loss_dict:
                    components[key] += float(loss_dict[key].item())

    num_batches = max(1, len(loader))
    return running_loss / num_batches, {k: v / num_batches for k, v in components.items()}


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
    class_names = train_ds.class_names
    cfg.setdefault("data", {})["class_names"] = class_names

    model = build_maskrcnn(
        num_classes=train_ds.num_detection_classes,
        backbone=cfg["model"].get("backbone", "maskrcnn_resnet50_fpn"),
        pretrained=cfg["model"].get("pretrained", True),
    ).to(device)

    freeze_backbone = bool(cfg["model"].get("freeze_backbone", False))
    if freeze_backbone:
        set_backbone_trainable(model, trainable=False)

    print(f"[INFO] Device: {device}")
    print(f"[INFO] Detection classes: {train_ds.num_detection_classes}")
    print(f"[INFO] Class names: {class_names}")
    print(f"[INFO] Train size: {len(train_ds)} | Val size: {len(val_ds)}")
    print(f"[INFO] Trainable params: {count_trainable_parameters(model):,}")
    print(f"[INFO] Freeze backbone: {freeze_backbone}")

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(
        trainable_params,
        lr=cfg["train"]["learning_rate"],
        weight_decay=cfg["train"]["weight_decay"],
    )
    scheduler = StepLR(
        optimizer,
        step_size=cfg["train"].get("step_size", 7),
        gamma=cfg["train"].get("gamma", 0.1),
    )

    amp_enabled = bool(cfg["train"].get("amp", True) and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    history_path = output_dir / "history.csv"
    with open(history_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "epoch",
            "train_loss",
            "val_loss",
            "loss_classifier",
            "loss_box_reg",
            "loss_mask",
            "loss_objectness",
            "loss_rpn_box_reg",
        ])

    best_val_loss = float("inf")
    epochs = int(cfg["train"]["epochs"])
    patience = int(cfg["train"].get("early_stopping_patience", 0))
    epochs_without_improvement = 0

    for epoch in range(1, epochs + 1):
        train_loss, train_components = train_one_epoch(model, train_loader, optimizer, device, scaler, amp_enabled)
        val_loss, val_components = validate_loss(model, val_loader, device, amp_enabled)
        scheduler.step()

        print(
            f"Epoch {epoch:03d}/{epochs} | "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} | "
            f"cls={val_components.get('loss_classifier', 0):.4f} "
            f"box={val_components.get('loss_box_reg', 0):.4f} "
            f"mask={val_components.get('loss_mask', 0):.4f}"
        )

        with open(history_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch,
                train_loss,
                val_loss,
                val_components.get("loss_classifier", 0.0),
                val_components.get("loss_box_reg", 0.0),
                val_components.get("loss_mask", 0.0),
                val_components.get("loss_objectness", 0.0),
                val_components.get("loss_rpn_box_reg", 0.0),
            ])

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": cfg,
            "val_loss": val_loss,
            "val_components": val_components,
            "class_names": class_names,
        }
        latest_path = output_dir / "last_maskrcnn.pth"
        save_checkpoint(checkpoint, str(latest_path))

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            best_path = output_dir / "best_maskrcnn.pth"
            save_checkpoint(checkpoint, str(best_path))
            print(f"[INFO] Saved best checkpoint to: {best_path}")
        else:
            epochs_without_improvement += 1
            if patience > 0 and epochs_without_improvement >= patience:
                print(f"[INFO] Early stopping triggered after {patience} epochs without improvement.")
                break

    print(f"[DONE] Training Mask R-CNN finished. History saved to {history_path}")


if __name__ == "__main__":
    main()
