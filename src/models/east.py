import math
from typing import cast, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.nn import functional as F

from torchvision.models import vgg16_bn, VGG16_BN_Weights

# -------------------------------------------------
# VGG configurations
# -------------------------------------------------

cfgs: Dict[str, List[Union[str, int]]] = {
    "A": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "B": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "D": [64, 64, "M", 128, 128, "M",
          256, 256, 256, "M",
          512, 512, 512, "M",
          512, 512, 512, "M"],
    "E": [64, 64, "M", 128, 128, "M",
          256, 256, 256, 256, "M",
          512, 512, 512, 512, "M",
          512, 512, 512, 512, "M"],
}


# -------------------------------------------------
# VGG backbone (custom)
# -------------------------------------------------

class VGG(nn.Module):
    def __init__(
        self,
        features: nn.Module,
        num_classes: int = 1000,
        init_weights: bool = True,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, num_classes),
        )

        if init_weights:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, 0, 0.01)
                    nn.init.constant_(m.bias, 0)


def make_layers(cfg: List[Union[str, int]], batch_norm: bool = False) -> nn.Sequential:
    layers: List[nn.Module] = []
    in_channels = 3

    for v in cfg:
        if v == "M":
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        else:
            v = cast(int, v)
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers.extend([conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)])
            else:
                layers.extend([conv2d, nn.ReLU(inplace=True)])
            in_channels = v

    return nn.Sequential(*layers)


# -------------------------------------------------
# Basic conv block
# -------------------------------------------------

class Conv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: Optional[int] = None,
        inplace: bool = True,
        bias: bool = True,
    ) -> None:
        super().__init__()

        if padding is None:
            padding = kernel_size // 2

        self.conv = nn.Conv2d(
            in_channels, out_channels,
            kernel_size, stride, padding, bias=bias
        )
        self.norm = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=inplace)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x)))


# -------------------------------------------------
# Feature extractor (VGG16_bn)
# -------------------------------------------------

class FeatureExtractor(nn.Module):
    """
    VGG16_bn feature extractor
    Supports:
      - weights=None        → scratch
      - weights="imagenet"  → ImageNet pretrained
    """

    def __init__(self, cfg: str = "D", weights: Optional[str] = None):
        super().__init__()

        # Build custom VGG backbone (no random init if pretrained)
        model = VGG(
            make_layers(cfg=cfgs[cfg], batch_norm=True),
            init_weights=False,
        )

        if weights == "imagenet":
            print("[INFO] Loading ImageNet pretrained VGG16_bn backbone")
            tv_model = vgg16_bn(weights=VGG16_BN_Weights.IMAGENET1K_V1)
            model.features.load_state_dict(tv_model.features.state_dict())
            print("[INFO] ImageNet weights loaded successfully")

        elif weights is not None:
            print("[INFO] Loading custom pretrained weights")
            model.load_state_dict(torch.load(weights, map_location="cpu"))

        self.features = model.features

    def forward(self, x: torch.Tensor) -> tuple:
        outputs = []
        for m in self.features:
            x = m(x)
            if isinstance(m, nn.MaxPool2d):
                outputs.append(x)
        # return feature maps at 1/4, 1/8, 1/16, 1/32
        return outputs[1:]


# -------------------------------------------------
# Feature merging branch
# -------------------------------------------------

class FeatureMerge(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.h2 = nn.Sequential(
            Conv(1024, 128, kernel_size=1),
            Conv(128, 128, kernel_size=3),
        )
        self.h3 = nn.Sequential(
            Conv(384, 64, kernel_size=1),
            Conv(64, 64, kernel_size=3),
        )
        self.h4 = nn.Sequential(
            Conv(192, 32, kernel_size=1),
            Conv(32, 32, kernel_size=3),
        )
        self.h5 = Conv(32, 32, kernel_size=3)

    def forward(self, x: tuple) -> torch.Tensor:
        y = F.interpolate(x[3], scale_factor=2, mode="bilinear", align_corners=True)
        y = self.h2(torch.cat([y, x[2]], dim=1))

        y = F.interpolate(y, scale_factor=2, mode="bilinear", align_corners=True)
        y = self.h3(torch.cat([y, x[1]], dim=1))

        y = F.interpolate(y, scale_factor=2, mode="bilinear", align_corners=True)
        y = self.h4(torch.cat([y, x[0]], dim=1))

        return self.h5(y)


# -------------------------------------------------
# Output head
# -------------------------------------------------

class Output(nn.Module):
    def __init__(self, scope: int = 512) -> None:
        super().__init__()

        self.score = nn.Sequential(
            nn.Conv2d(32, 1, kernel_size=1),
            nn.Sigmoid()
        )
        self.loc = nn.Sequential(
            nn.Conv2d(32, 4, kernel_size=1),
            nn.Sigmoid()
        )
        self.angle = nn.Sequential(
            nn.Conv2d(32, 1, kernel_size=1),
            nn.Sigmoid()
        )

        self.scope = scope

    def forward(self, x: torch.Tensor) -> tuple:
        score_map = self.score(x)
        location = self.loc(x) * self.scope
        angle = (self.angle(x) - 0.5) * math.pi
        geometry = torch.cat([location, angle], dim=1)
        return score_map, geometry


# -------------------------------------------------
# EAST model
# -------------------------------------------------

class EAST(nn.Module):
    """
    EAST: Efficient and Accurate Scene Text Detector
    """

    def __init__(
        self,
        cfg: str = "D",
        weights: Optional[str] = None,
        scope: int = 512,
    ) -> None:
        super().__init__()

        self.extract = FeatureExtractor(cfg=cfg, weights=weights)
        self.merge = FeatureMerge()
        self.detect = Output(scope=scope)

    def forward(self, x: torch.Tensor) -> tuple:
        feats = self.extract(x)
        merged = self.merge(feats)
        return self.detect(merged)
