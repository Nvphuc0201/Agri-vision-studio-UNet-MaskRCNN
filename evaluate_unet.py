from __future__ import annotations

import argparse

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.roboflow_coco import COCOSemanticSegmentationDataset
from models.unet_model import build_unet
from utils.common import get_device, load_yaml
from utils.metrics import segmentation_metrics, summarize_metric_list
from utils.transforms import get_unet_transforms


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate U-Net on Roboflow COCO split")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--split", type=str, default="valid")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_yaml(args.config)
    device = get_device(cfg.get("device", "cuda"))
    task_mode = cfg["data"].get("task_mode", "multiclass")

    dataset = COCOSemanticSegmentationDataset(
        root_dir=cfg["data"]["root_dir"],
        split=args.split,
        task_mode=task_mode,
        transform=get_unet_transforms(cfg["data"]["img_size"], train=False),
    )
    loader = DataLoader(
        dataset,
        batch_size=cfg["data"]["batch_size"],
        shuffle=False,
        num_workers=cfg["data"].get("num_workers", 4),
    )

    ckpt = torch.load(args.checkpoint, map_location=device)
    model_cfg = dict(ckpt.get("config", {}).get("model", cfg.get("model", {})))
    model_cfg["classes"] = dataset.num_segmentation_classes
    model = build_unet(
        encoder_name=model_cfg.get("encoder_name", "resnet34"),
        encoder_weights=None,
        in_channels=model_cfg.get("in_channels", 3),
        classes=model_cfg.get("classes", dataset.num_segmentation_classes),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()

    metrics = []
    with torch.no_grad():
        for images, masks in tqdm(loader, desc="Evaluating U-Net"):
            images = images.to(device)
            masks = masks.to(device)
            logits = model(images)
            metrics.append(
                segmentation_metrics(
                    logits=logits,
                    targets=masks,
                    task_mode=task_mode,
                    threshold=cfg["train"].get("threshold", 0.5),
                    num_classes=dataset.num_segmentation_classes,
                )
            )

    summary = summarize_metric_list(metrics)
    print("[RESULT]", summary)


if __name__ == "__main__":
    main()
