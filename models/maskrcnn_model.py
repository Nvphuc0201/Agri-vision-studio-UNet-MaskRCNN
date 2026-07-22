from __future__ import annotations
import torchvision
from torchvision.models import ResNet50_Weights
from torchvision.models.detection import (
    MaskRCNN_ResNet50_FPN_V2_Weights,
    MaskRCNN_ResNet50_FPN_Weights,
    maskrcnn_resnet50_fpn,
    maskrcnn_resnet50_fpn_v2,
)

def build_maskrcnn(num_classes: int, backbone: str = "maskrcnn_resnet50_fpn", pretrained: bool = True):
    if backbone == "maskrcnn_resnet50_fpn_v2":
        weights = MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT if pretrained else None
        model = maskrcnn_resnet50_fpn_v2(weights=weights)
    else:
        weights = MaskRCNN_ResNet50_FPN_Weights.DEFAULT if pretrained else None
        model = maskrcnn_resnet50_fpn(weights=weights, weights_backbone=ResNet50_Weights.IMAGENET1K_V1 if not pretrained else None)

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
        in_features, num_classes
    )

    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = torchvision.models.detection.mask_rcnn.MaskRCNNPredictor(
        in_features_mask,
        hidden_layer,
        num_classes,
    )
    return model