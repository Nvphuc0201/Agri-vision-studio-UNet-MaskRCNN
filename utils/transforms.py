from __future__ import annotations

import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_unet_transforms(img_size: int, train: bool = True) -> A.Compose:
    if train:
        return A.Compose(
            [
                A.Resize(img_size, img_size),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.1),
                A.RandomRotate90(p=0.2),
                A.Affine(
                    scale=(0.95, 1.05),
                    translate_percent={"x": (-0.03, 0.03), "y": (-0.03, 0.03)},
                    rotate=(-12, 12),
                    border_mode=0,
                    p=0.35,
                ),
                A.RandomBrightnessContrast(p=0.2),
                A.Normalize(),
                ToTensorV2(),
            ]
        )
    return A.Compose([A.Resize(img_size, img_size), A.Normalize(), ToTensorV2()])


def get_maskrcnn_image_transform(img_size: int = 512, train: bool = True) -> A.Compose:
    if train:
        return A.Compose(
            [
                A.Resize(img_size, img_size),
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.2),
            ],
            bbox_params=A.BboxParams(
                format="pascal_voc",
                label_fields=["class_labels"],
                min_area=1.0,
                min_visibility=0.01,
            ),
        )
    return A.Compose(
        [A.Resize(img_size, img_size)],
        bbox_params=A.BboxParams(format="pascal_voc", label_fields=["class_labels"]),
    )
