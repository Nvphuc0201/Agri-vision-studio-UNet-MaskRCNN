from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import albumentations as A
import cv2
import numpy as np
import torch
from pycocotools.coco import COCO
from torch.utils.data import Dataset

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
IMAGE_DIR_CANDIDATES = ("image", "images")


class RoboflowCOCOBase(Dataset):
    def __init__(self, root_dir: str, split: str = "train"):
        self.root_dir = Path(root_dir)
        self.split = split
        self.split_dir = self.root_dir / split
        self.annotation_json = self.split_dir / "_annotations.coco.json"
        if not self.split_dir.exists():
            raise FileNotFoundError(f"Khong tim thay split directory: {self.split_dir}")
        if not self.annotation_json.exists():
            raise FileNotFoundError(f"Khong tim thay annotation file: {self.annotation_json}")

        self.image_dir = self._resolve_image_dir()
        self.coco = COCO(str(self.annotation_json))
        self.category_ids = sorted(self.coco.getCatIds())
        self.cat_id_to_label = {cat_id: idx + 1 for idx, cat_id in enumerate(self.category_ids)}
        self.label_to_cat_id = {label: cat_id for cat_id, label in self.cat_id_to_label.items()}
        categories = sorted(self.coco.loadCats(self.category_ids), key=lambda item: item["id"])
        self.class_names = ["background"] + [cat["name"] for cat in categories]
        self.image_ids = self._collect_valid_image_ids()

    def _resolve_image_dir(self) -> Path:
        for dirname in IMAGE_DIR_CANDIDATES:
            candidate = self.split_dir / dirname
            if candidate.exists() and candidate.is_dir():
                return candidate
        return self.split_dir

    def _resolve_image_path(self, file_name: str) -> Path:
        raw = Path(file_name)
        candidates = []

        if raw.is_absolute():
            candidates.append(raw)
        else:
            candidates.extend(
                [
                    self.split_dir / raw,
                    self.image_dir / raw,
                    self.image_dir / raw.name,
                    self.split_dir / raw.name,
                ]
            )

        for candidate in candidates:
            if candidate.exists() and candidate.is_file() and candidate.suffix.lower() in ALLOWED_EXTENSIONS:
                return candidate

        raise FileNotFoundError(
            f"Khong tim thay anh '{file_name}' trong {self.image_dir} hoac {self.split_dir}."
        )

    def _collect_valid_image_ids(self) -> List[int]:
        image_ids: List[int] = []
        for image_id in sorted(self.coco.getImgIds()):
            image_info = self.coco.loadImgs(image_id)[0]
            try:
                _ = self._resolve_image_path(image_info["file_name"])
                image_ids.append(image_id)
            except FileNotFoundError:
                continue
        if not image_ids:
            raise FileNotFoundError(
                f"Khong tim thay anh hop le trong {self.image_dir}. Kiem tra lai cau truc dataset va file_name trong COCO JSON."
            )
        return image_ids

    def __len__(self) -> int:
        return len(self.image_ids)

    def get_image_info(self, image_id: int) -> Dict[str, Any]:
        return self.coco.loadImgs(image_id)[0]

    def load_image(self, image_id: int) -> np.ndarray:
        image_info = self.get_image_info(image_id)
        image_path = self._resolve_image_path(image_info["file_name"])
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Khong doc duoc anh: {image_path}")
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    def load_annotations(self, image_id: int) -> List[Dict[str, Any]]:
        ann_ids = self.coco.getAnnIds(imgIds=[image_id], iscrowd=None)
        anns = self.coco.loadAnns(ann_ids)
        valid_anns: List[Dict[str, Any]] = []
        for ann in anns:
            if ann.get("category_id") not in self.cat_id_to_label:
                continue
            x, y, w, h = ann.get("bbox", [0, 0, 0, 0])
            if w <= 1 or h <= 1:
                continue
            valid_anns.append(ann)
        return valid_anns

    @property
    def num_foreground_classes(self) -> int:
        return len(self.category_ids)

    @property
    def num_detection_classes(self) -> int:
        return len(self.class_names)


class COCOSemanticSegmentationDataset(RoboflowCOCOBase):
    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        task_mode: str = "multiclass",
        transform: A.Compose | None = None,
    ):
        super().__init__(root_dir=root_dir, split=split)
        if task_mode not in {"binary", "multiclass"}:
            raise ValueError("task_mode phai la 'binary' hoac 'multiclass'")
        self.task_mode = task_mode
        self.transform = transform

    @property
    def num_segmentation_classes(self) -> int:
        return 1 if self.task_mode == "binary" else len(self.class_names)

    def build_semantic_mask(self, image_id: int, height: int, width: int) -> np.ndarray:
        anns = sorted(self.load_annotations(image_id), key=lambda ann: float(ann.get("area", 0.0)), reverse=True)
        semantic_mask = np.zeros((height, width), dtype=np.uint8)

        for ann in anns:
            category_label = self.cat_id_to_label[ann["category_id"]]
            ann_mask = self.coco.annToMask(ann).astype(bool)
            if self.task_mode == "binary":
                semantic_mask[ann_mask] = 1
            else:
                semantic_mask[ann_mask] = np.uint8(category_label)
        return semantic_mask

    def __getitem__(self, index: int):
        image_id = self.image_ids[index]
        image = self.load_image(image_id)
        height, width = image.shape[:2]
        mask = self.build_semantic_mask(image_id=image_id, height=height, width=width)

        if self.transform is not None:
            transformed = self.transform(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]
        else:
            image = torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1).float() / 255.0
            mask = torch.from_numpy(mask)

        if not torch.is_tensor(mask):
            mask = torch.as_tensor(mask)

        if self.task_mode == "binary":
            mask = (mask > 0).float()
            if mask.ndim == 2:
                mask = mask.unsqueeze(0)
            elif mask.ndim == 3 and mask.shape[0] != 1:
                mask = mask[:1].float()
        else:
            if mask.ndim == 3:
                mask = mask.squeeze(0)
            mask = mask.long()

        return image, mask


class COCOMaskRCNNDataset(RoboflowCOCOBase):
    def __init__(self, root_dir: str, split: str = "train", transforms: A.Compose | None = None):
        super().__init__(root_dir=root_dir, split=split)
        self.transforms = transforms

    def __getitem__(self, index: int):
        image_id = self.image_ids[index]
        image = self.load_image(image_id)
        anns = self.load_annotations(image_id)

        boxes: List[List[float]] = []
        labels: List[int] = []
        masks: List[np.ndarray] = []

        for ann in anns:
            category_id = ann["category_id"]
            if category_id not in self.cat_id_to_label:
                continue
            ann_mask = self.coco.annToMask(ann).astype(np.uint8)
            if ann_mask.sum() == 0:
                continue
            ys, xs = np.where(ann_mask > 0)
            if len(xs) == 0 or len(ys) == 0:
                continue
            x1, y1, x2, y2 = float(xs.min()), float(ys.min()), float(xs.max()) + 1.0, float(ys.max()) + 1.0
            if x2 <= x1 or y2 <= y1:
                continue
            boxes.append([x1, y1, x2, y2])
            labels.append(int(self.cat_id_to_label[category_id]))
            masks.append(ann_mask)

        if self.transforms is not None and len(boxes) > 0:
            transformed = self.transforms(
                image=image,
                masks=masks,
                bboxes=boxes,
                class_labels=labels,
            )
            image = transformed["image"]
            masks = [np.asarray(mask, dtype=np.uint8) for mask in transformed["masks"]]
            boxes = [list(box) for box in transformed["bboxes"]]
            labels = [int(label) for label in transformed["class_labels"]]

        if not torch.is_tensor(image):
            image = torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1).float() / 255.0

        if len(boxes) == 0:
            boxes_tensor = torch.zeros((0, 4), dtype=torch.float32)
            labels_tensor = torch.zeros((0,), dtype=torch.int64)
            masks_tensor = torch.zeros((0, image.shape[1], image.shape[2]), dtype=torch.uint8)
            areas_tensor = torch.zeros((0,), dtype=torch.float32)
            iscrowd_tensor = torch.zeros((0,), dtype=torch.int64)
        else:
            masks_np = np.stack(masks, axis=0).astype(np.uint8)
            if masks_np.ndim != 3:
                raise ValueError("Masks sau augmentation phai co shape [N, H, W]")
            boxes_tensor = torch.as_tensor(boxes, dtype=torch.float32)
            labels_tensor = torch.as_tensor(labels, dtype=torch.int64)
            masks_tensor = torch.as_tensor(masks_np, dtype=torch.uint8)
            areas_tensor = masks_tensor.flatten(1).sum(dim=1).to(dtype=torch.float32)
            iscrowd_tensor = torch.zeros((labels_tensor.numel(),), dtype=torch.int64)

        target: Dict[str, Any] = {
            "boxes": boxes_tensor,
            "labels": labels_tensor,
            "masks": masks_tensor,
            "image_id": torch.tensor([image_id], dtype=torch.int64),
            "area": areas_tensor,
            "iscrowd": iscrowd_tensor,
        }
        return image, target



def detection_collate_fn(batch: Sequence[Tuple[torch.Tensor, Dict[str, Any]]]):
    images, targets = list(zip(*batch))
    return list(images), list(targets)
