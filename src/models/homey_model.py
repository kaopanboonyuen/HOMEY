import torch
import torch.nn as nn
from ultralytics import YOLO
from ..utils.mask_utils import apply_heuristic_mask

class HOMEYModel(nn.Module):
    """
    HOMEY: YOLO backbone + Heuristic Object Masking
    """

    def __init__(self, backbone='yolov8n.pt', num_classes=17):
        super(HOMEYModel, self).__init__()
        self.num_classes = num_classes
        self.backbone = YOLO(backbone)
        self.backbone.model[-1].nc = num_classes  # Update YOLO output classes

    def forward(self, images):
        # Apply heuristic masking first
        masked_images = apply_heuristic_mask(images)
        # Forward through YOLO
        preds = self.backbone(masked_images)
        return preds