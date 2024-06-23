import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from detectron2.layers import ShapeSpec
from detectron2.modeling import META_ARCH_REGISTRY
from detectron2.modeling.backbone.resnet import build_resnet_backbone


@META_ARCH_REGISTRY.register()
class Model(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.register_buffer("pixel_mean", torch.tensor(cfg.MODEL.PIXEL_MEAN).reshape(1, -1, 1, 1), False)
        self.register_buffer("pixel_std", torch.tensor(cfg.MODEL.PIXEL_STD).reshape(1, 3, 1, 1), False)

        self.backbone = build_resnet_backbone(cfg, ShapeSpec(channels=3))
        
    @property
    def device(self):
        return self.pixel_mean.device

    def forward(self, samples, *args, **argw):
        y = self.backbone(torch.rand(2,3,224,224).to(self.device))
        # print("inputs:", len(samples), samples[0].keys())
        # print("y:", type(y), y.keys())
        # print("y shapes:", [(k, v.shape) for k, v in y.items()])
        return {
            "loss": y["res5"].mean()
        }