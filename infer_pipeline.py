from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import torch

from models.maskrcnn_model import build_maskrcnn
from models.unet_model import build_unet
from utils.common import get_class_names_from_checkpoint, get_device
from utils.visualization import colorize_semantic_mask, draw_instance_predictions, overlay_semantic_mask

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inference pipeline: U-Net semantic segmentation + Mask R-CNN instance segmentation")
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--unet_checkpoint", type=str, required=True)
    parser.add_argument("--maskrcnn_checkpoint", type=str, required=True)
    parser.add_argument("--output_path", type=str, default="outputs/infer/result.jpg")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--score_thr", type=float, default=0.5)
    parser.add_argument("--mask_thr", type=float, default=0.5)
    parser.add_argument("--use_unet_preprocess", action="store_true")
    parser.add_argument("--class_names", nargs="*", default=None)
    return parser.parse_args()


def load_unet(checkpoint_path: str, device: torch.device):
    ckpt = torch.load(checkpoint_path, map_location=device)
    cfg_model = dict(ckpt.get("config", {}).get("model", {}))
    cfg_data = ckpt.get("config", {}).get("data", {})
    task_mode = ckpt.get("task_mode", cfg_data.get("task_mode", "multiclass"))
    class_names = ckpt.get("class_names") or get_class_names_from_checkpoint(ckpt) or ["background", "object"]
    num_classes = 1 if task_mode == "binary" else len(class_names)
    model = build_unet(
        encoder_name=cfg_model.get("encoder_name", "resnet34"),
        encoder_weights=None,
        in_channels=cfg_model.get("in_channels", 3),
        classes=num_classes,
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    img_size = int(cfg_data.get("img_size", 512))
    return model, task_mode, class_names, img_size


def load_maskrcnn(checkpoint_path: str, device: torch.device):
    ckpt = torch.load(checkpoint_path, map_location=device)
    cfg_model = ckpt.get("config", {}).get("model", {})
    class_names = ckpt.get("class_names") or get_class_names_from_checkpoint(ckpt) or ["background", "object"]
    model = build_maskrcnn(
        num_classes=len(class_names),
        backbone=cfg_model.get("backbone", "maskrcnn_resnet50_fpn"),
        pretrained=False,
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    return model, class_names


def preprocess_for_unet(image_rgb: np.ndarray, img_size: int) -> Tuple[torch.Tensor, Tuple[int, int]]:
    original_h, original_w = image_rgb.shape[:2]
    resized = cv2.resize(image_rgb, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
    tensor = torch.from_numpy(np.ascontiguousarray(resized)).permute(2, 0, 1).float() / 255.0
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)
    tensor = (tensor - mean) / std
    return tensor.unsqueeze(0), (original_h, original_w)


def run_unet_mask(model, image_rgb: np.ndarray, img_size: int, device: torch.device, task_mode: str, mask_thr: float) -> np.ndarray:
    tensor, original_size = preprocess_for_unet(image_rgb, img_size)
    tensor = tensor.to(device)
    with torch.no_grad():
        logits = model(tensor)
        if task_mode == "binary":
            probs = torch.sigmoid(logits)[0, 0].cpu().numpy()
            pred_mask = (probs >= mask_thr).astype(np.uint8)
        else:
            pred_mask = torch.argmax(logits, dim=1)[0].cpu().numpy().astype(np.uint8)
    pred_mask = cv2.resize(pred_mask, (original_size[1], original_size[0]), interpolation=cv2.INTER_NEAREST)
    return pred_mask


def apply_foreground_mask(image_rgb: np.ndarray, semantic_mask: np.ndarray) -> np.ndarray:
    filtered = image_rgb.copy()
    filtered[semantic_mask == 0] = 0
    return filtered


def run_maskrcnn(model, image_rgb: np.ndarray, device: torch.device):
    tensor = torch.from_numpy(np.ascontiguousarray(image_rgb)).permute(2, 0, 1).float() / 255.0
    with torch.no_grad():
        preds = model([tensor.to(device)])[0]
    boxes = preds["boxes"].detach().cpu().numpy() if "boxes" in preds else np.zeros((0, 4))
    labels = preds["labels"].detach().cpu().numpy() if "labels" in preds else np.zeros((0,))
    scores = preds["scores"].detach().cpu().numpy() if "scores" in preds else np.zeros((0,))
    masks = preds["masks"].detach().cpu().numpy()[:, 0] if "masks" in preds and len(preds["masks"]) else None
    return boxes, labels, scores, masks


def resolve_class_names(args_class_names: List[str] | None, unet_class_names: List[str], maskrcnn_class_names: List[str]) -> List[str]:
    if args_class_names:
        return args_class_names
    if len(maskrcnn_class_names) > 1:
        return maskrcnn_class_names
    return unet_class_names


def main():
    args = parse_args()
    device = get_device(args.device)

    image_path = Path(args.image)
    if image_path.suffix.lower() not in ALLOWED_EXTENSIONS:
        raise ValueError("Dinh dang anh khong ho tro")

    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError(f"Khong doc duoc anh: {image_path}")
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    unet, task_mode, unet_class_names, unet_img_size = load_unet(args.unet_checkpoint, device)
    maskrcnn, maskrcnn_class_names = load_maskrcnn(args.maskrcnn_checkpoint, device)
    class_names = resolve_class_names(args.class_names, unet_class_names, maskrcnn_class_names)

    semantic_mask = run_unet_mask(unet, image_rgb, unet_img_size, device, task_mode, args.mask_thr)
    processed_image = apply_foreground_mask(image_rgb, semantic_mask) if args.use_unet_preprocess else image_rgb
    boxes, labels, scores, masks = run_maskrcnn(maskrcnn, processed_image, device)

    overlay = overlay_semantic_mask(image_rgb, semantic_mask)
    result = draw_instance_predictions(
        overlay,
        boxes=boxes,
        labels=labels,
        scores=scores,
        masks=masks,
        class_names=class_names,
        score_thr=args.score_thr,
    )

    semantic_color = colorize_semantic_mask(semantic_mask)
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    semantic_path = output_path.with_name(output_path.stem + "_semantic.png")
    cv2.imwrite(str(output_path), cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(semantic_path), cv2.cvtColor(semantic_color, cv2.COLOR_RGB2BGR))

    kept = int((scores >= args.score_thr).sum()) if scores.size > 0 else 0
    print(f"[DONE] Saved result to {output_path}")
    print(f"[DONE] Saved semantic mask preview to {semantic_path}")
    print(f"[INFO] Detected objects above threshold: {kept}")

    for i in range(len(scores)):
        if scores[i] < args.score_thr:
            continue
        cls_id = int(labels[i])
        cls_name = class_names[cls_id] if cls_id < len(class_names) else str(cls_id)
        print(f" - {cls_name}: score={scores[i]:.4f}, box={boxes[i].tolist()}")


if __name__ == "__main__":
    main()
