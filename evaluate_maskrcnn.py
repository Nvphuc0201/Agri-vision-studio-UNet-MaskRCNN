from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from pycocotools import mask as mask_utils
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.roboflow_coco import COCOMaskRCNNDataset, detection_collate_fn
from models.maskrcnn_model import build_maskrcnn
from utils.common import get_device, load_yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Mask R-CNN with COCO bbox and segm metrics")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--split", type=str, default="valid")
    parser.add_argument("--score_thr", type=float, default=0.05)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output_json", type=str, default="outputs/maskrcnn/coco_predictions.json")
    return parser.parse_args()


def load_model(checkpoint_path: str, cfg: Dict, device: torch.device, num_classes: int):
    model = build_maskrcnn(
        num_classes=num_classes,
        backbone=cfg["model"].get("backbone", "maskrcnn_resnet50_fpn"),
        pretrained=False,
    )
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    return model


def xyxy_to_xywh(box: np.ndarray) -> List[float]:
    x1, y1, x2, y2 = box.tolist()
    return [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]


def encode_mask(mask: np.ndarray) -> Dict[str, str]:
    rle = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
    if isinstance(rle["counts"], bytes):
        rle["counts"] = rle["counts"].decode("utf-8")
    return rle


def main():
    args = parse_args()
    cfg = load_yaml(args.config)
    device = get_device(args.device or cfg.get("device", "cuda"))

    dataset = COCOMaskRCNNDataset(
        root_dir=cfg["data"]["root_dir"],
        split=args.split,
        transforms=None,
    )
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=cfg["data"].get("num_workers", 4),
        collate_fn=detection_collate_fn,
    )
    model = load_model(args.checkpoint, cfg, device, dataset.num_detection_classes)

    detections = []
    for images, targets in tqdm(loader, desc="Evaluating"):
        image = images[0].to(device)
        target = targets[0]
        image_id = int(target["image_id"].item())
        with torch.no_grad():
            pred = model([image])[0]

        boxes = pred["boxes"].detach().cpu().numpy() if "boxes" in pred else np.zeros((0, 4))
        scores = pred["scores"].detach().cpu().numpy() if "scores" in pred else np.zeros((0,))
        labels = pred["labels"].detach().cpu().numpy() if "labels" in pred else np.zeros((0,))
        masks = pred["masks"].detach().cpu().numpy()[:, 0] if len(pred.get("masks", [])) else np.zeros((0,))

        for idx in range(len(scores)):
            if scores[idx] < args.score_thr:
                continue
            label = int(labels[idx])
            category_id = dataset.label_to_cat_id.get(label)
            if category_id is None:
                continue
            pred_mask = (masks[idx] >= 0.5).astype(np.uint8) if len(masks) else None
            detections.append(
                {
                    "image_id": image_id,
                    "category_id": int(category_id),
                    "bbox": xyxy_to_xywh(boxes[idx]),
                    "score": float(scores[idx]),
                    "segmentation": encode_mask(pred_mask) if pred_mask is not None else None,
                }
            )

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(detections, f)

    coco_gt = COCO(str(dataset.annotation_json))
    coco_dt = coco_gt.loadRes(str(output_json)) if detections else coco_gt.loadRes([])

    print("\n[COCO bbox]")
    coco_eval_bbox = COCOeval(coco_gt, coco_dt, iouType="bbox")
    coco_eval_bbox.evaluate()
    coco_eval_bbox.accumulate()
    coco_eval_bbox.summarize()

    if detections and any(det.get("segmentation") is not None for det in detections):
        print("\n[COCO segm]")
        coco_eval_segm = COCOeval(coco_gt, coco_dt, iouType="segm")
        coco_eval_segm.evaluate()
        coco_eval_segm.accumulate()
        coco_eval_segm.summarize()

    print(f"[DONE] Saved predictions to {output_json}")


if __name__ == "__main__":
    main()
