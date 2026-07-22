from __future__ import annotations

from typing import List, Optional, Sequence

import cv2
import numpy as np


PALETTE = np.array(
    [
        [0, 0, 0],
        [0, 255, 0],
        [255, 0, 0],
        [0, 0, 255],
        [255, 255, 0],
        [255, 0, 255],
        [0, 255, 255],
        [255, 128, 0],
        [128, 0, 255],
        [0, 128, 255],
    ],
    dtype=np.uint8,
)


def get_color(class_id: int) -> np.ndarray:
    return PALETTE[class_id % len(PALETTE)]


def colorize_semantic_mask(mask: np.ndarray) -> np.ndarray:
    canvas = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    for cls_idx in np.unique(mask):
        canvas[mask == cls_idx] = get_color(int(cls_idx))
    return canvas


def overlay_semantic_mask(image: np.ndarray, mask: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    colored = colorize_semantic_mask(mask)
    return cv2.addWeighted(image, 1.0, colored, alpha, 0)


def draw_instance_predictions(
    image: np.ndarray,
    boxes: np.ndarray,
    labels: np.ndarray,
    scores: np.ndarray,
    masks: Optional[np.ndarray] = None,
    class_names: Optional[Sequence[str]] = None,
    score_thr: float = 0.5,
) -> np.ndarray:
    canvas = image.copy()

    if masks is not None:
        for idx in range(len(masks)):
            if scores[idx] < score_thr:
                continue
            class_id = int(labels[idx])
            color = get_color(class_id)
            mask = (masks[idx] > 0.5).astype(np.uint8)
            overlay = np.zeros_like(canvas)
            overlay[mask > 0] = color
            canvas = cv2.addWeighted(canvas, 1.0, overlay, 0.28, 0)

    for idx, box in enumerate(boxes):
        if scores[idx] < score_thr:
            continue
        x1, y1, x2, y2 = map(int, box)
        class_id = int(labels[idx])
        color = tuple(int(v) for v in get_color(class_id).tolist())
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        label_name = class_names[class_id] if class_names and class_id < len(class_names) else str(class_id)
        text = f"{label_name}: {scores[idx]:.2f}"
        cv2.putText(canvas, text, (x1, max(22, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
    return canvas
