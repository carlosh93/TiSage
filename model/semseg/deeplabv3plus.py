"""DeepLabV3+ with a torchvision ResNet-50 encoder.

This is a self-contained implementation of the DeepLabV3+ decoder: ASPP on
the stride-16 encoder feature map, a 48-channel low-level projection, and two
3x3 decoder convolutions before the pixel classifier.
"""

from pathlib import Path
from typing import Mapping, Optional, Sequence

import torch
from torch import nn
import torch.nn.functional as F
from torchvision.models import ResNet50_Weights, resnet50


class ASPPConv(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, dilation: int):
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=dilation,
                dilation=dilation,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        spatial_size = x.shape[-2:]
        x = super().forward(x)
        return F.interpolate(x, size=spatial_size, mode="bilinear", align_corners=False)


class ASPP(nn.Module):
    def __init__(
        self,
        in_channels: int,
        atrous_rates: Sequence[int],
        out_channels: int = 256,
    ):
        super().__init__()
        branches = [
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )
        ]
        branches.extend(
            ASPPConv(in_channels, out_channels, rate) for rate in atrous_rates
        )
        branches.append(ASPPPooling(in_channels, out_channels))
        self.branches = nn.ModuleList(branches)
        self.project = nn.Sequential(
            nn.Conv2d(len(branches) * out_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.project(torch.cat([branch(x) for branch in self.branches], dim=1))


def _extract_state_dict(checkpoint: object) -> Mapping[str, torch.Tensor]:
    if not isinstance(checkpoint, Mapping):
        raise TypeError("The pretrained checkpoint must contain a state-dict mapping.")

    for key in ("state_dict", "model"):
        candidate = checkpoint.get(key)
        if isinstance(candidate, Mapping):
            checkpoint = candidate
            break

    state_dict = {}
    for key, value in checkpoint.items():
        if not isinstance(value, torch.Tensor):
            continue
        for prefix in ("module.", "backbone.", "encoder."):
            if key.startswith(prefix):
                key = key[len(prefix) :]
        state_dict[key] = value
    return state_dict


class ResNet50Encoder(nn.Module):
    """ResNet-50 feature extractor returning stride-4 and high-level features."""

    def __init__(
        self,
        output_stride: int = 16,
        pretrained_path: Optional[str] = None,
        allow_weight_download: bool = False,
    ):
        super().__init__()
        if output_stride not in (8, 16):
            raise ValueError(f"output_stride must be 8 or 16, got {output_stride}")

        replace_stride_with_dilation = (
            [False, True, True] if output_stride == 8 else [False, False, True]
        )

        weights = None
        if pretrained_path is None and allow_weight_download:
            weights = ResNet50_Weights.IMAGENET1K_V2

        backbone = resnet50(
            weights=weights,
            replace_stride_with_dilation=replace_stride_with_dilation,
        )

        if pretrained_path is not None:
            path = Path(pretrained_path)
            if not path.is_file():
                raise FileNotFoundError(
                    f"ResNet-50 pretrained weights not found: {path}. "
                    "Copy resnet50-11ad3fa6.pth into pretrained/ or explicitly "
                    "enable weight downloading."
                )
            checkpoint = torch.load(path, map_location="cpu", weights_only=True)
            state_dict = _extract_state_dict(checkpoint)
            missing, unexpected = backbone.load_state_dict(state_dict, strict=False)
            allowed_missing = {"fc.weight", "fc.bias"}
            real_missing = set(missing) - allowed_missing
            if real_missing or unexpected:
                raise RuntimeError(
                    "Incompatible ResNet-50 pretrained checkpoint. "
                    f"Missing keys: {sorted(real_missing)}; "
                    f"unexpected keys: {sorted(unexpected)}"
                )

        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        low_level = self.layer1(x)
        x = self.layer2(low_level)
        x = self.layer3(x)
        high_level = self.layer4(x)
        return low_level, high_level


class DeepLabV3Plus(nn.Module):
    """DeepLabV3+ using a ResNet-50 encoder and a six-class-compatible head."""

    def __init__(
        self,
        nclass: int,
        output_stride: int = 16,
        pretrained_path: Optional[str] = None,
        allow_weight_download: bool = False,
    ):
        super().__init__()
        self.encoder = ResNet50Encoder(
            output_stride=output_stride,
            pretrained_path=pretrained_path,
            allow_weight_download=allow_weight_download,
        )
        atrous_rates = (12, 24, 36) if output_stride == 8 else (6, 12, 18)
        self.aspp = ASPP(2048, atrous_rates, out_channels=256)
        self.low_level_projection = nn.Sequential(
            nn.Conv2d(256, 48, kernel_size=1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(304, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Conv2d(256, nclass, kernel_size=1),
        )
        self._initialize_decoder()

    def _initialize_decoder(self) -> None:
        for module in (self.aspp, self.low_level_projection, self.decoder):
            for layer in module.modules():
                if isinstance(layer, nn.Conv2d):
                    nn.init.kaiming_normal_(layer.weight, mode="fan_out", nonlinearity="relu")
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)
                elif isinstance(layer, nn.BatchNorm2d):
                    nn.init.ones_(layer.weight)
                    nn.init.zeros_(layer.bias)

    def encoder_parameters(self):
        return self.encoder.parameters()

    def decoder_parameters(self):
        for module in (self.aspp, self.low_level_projection, self.decoder):
            yield from module.parameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_size = x.shape[-2:]
        low_level, high_level = self.encoder(x)
        high_level = self.aspp(high_level)
        low_level = self.low_level_projection(low_level)
        high_level = F.interpolate(
            high_level,
            size=low_level.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        logits = self.decoder(torch.cat([low_level, high_level], dim=1))
        return F.interpolate(logits, size=input_size, mode="bilinear", align_corners=False)
