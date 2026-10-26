"""From-scratch multi-scale skip-fusion pose network.

The model keeps a 1/4-resolution localization stream while a deeper 1/16
context stream is decoded through two lateral skip connections.  It has no
external weights and preserves the existing 192x256 -> 48x64 contract.
"""

from __future__ import annotations

import torch
from torch import nn

from .scratch_cnn import HEATMAP_HEIGHT, HEATMAP_WIDTH, NUM_KEYPOINTS


class ConvNormAct(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        *,
        stride: int = 1,
        padding: int = 0,
    ) -> None:
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=padding,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )


class ResidualUnit(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.activation = nn.SiLU(inplace=True)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.activation(inputs + self.body(inputs))


class UpsampleFusion(nn.Module):
    """Upsample the context stream and fuse it with an equal-scale skip."""

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )
        self.fuse = nn.Sequential(
            ConvNormAct(out_channels + skip_channels, out_channels, 3, padding=1),
            ResidualUnit(out_channels),
        )

    def forward(self, context: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        upsampled = self.upsample(context)
        return self.fuse(torch.cat((upsampled, skip), dim=1))


class ScratchSkipPoseNetV4(nn.Module):
    """Higher-capacity encoder-decoder with 1/8 and 1/4 skip fusion."""

    def __init__(self) -> None:
        super().__init__()
        self.stem = ConvNormAct(3, 48, 7, stride=2, padding=3)
        self.quarter = nn.Sequential(
            ConvNormAct(48, 96, 3, stride=2, padding=1),
            ResidualUnit(96),
            ResidualUnit(96),
        )
        self.eighth = nn.Sequential(
            ConvNormAct(96, 192, 3, stride=2, padding=1),
            ResidualUnit(192),
            ResidualUnit(192),
            ResidualUnit(192),
        )
        self.sixteenth = nn.Sequential(
            ConvNormAct(192, 384, 3, stride=2, padding=1),
            ResidualUnit(384),
            ResidualUnit(384),
            ResidualUnit(384),
            ResidualUnit(384),
        )
        self.decode_eighth = UpsampleFusion(384, 192, 192)
        self.decode_quarter = UpsampleFusion(192, 96, 96)
        self.head = nn.Conv2d(96, NUM_KEYPOINTS, 1)
        self._initialize()

    def _initialize(self) -> None:
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
        nn.init.normal_(self.head.weight, std=0.001)
        nn.init.zeros_(self.head.bias)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        stem = self.stem(inputs)
        quarter = self.quarter(stem)
        eighth = self.eighth(quarter)
        sixteenth = self.sixteenth(eighth)
        decoded_eighth = self.decode_eighth(sixteenth, eighth)
        output = self.head(self.decode_quarter(decoded_eighth, quarter))
        # Explicit invariants make accidental contract drift visible in eager mode.
        if not torch.jit.is_tracing() and output.shape[-2:] != (HEATMAP_HEIGHT, HEATMAP_WIDTH):
            raise RuntimeError(f"unexpected output size: {tuple(output.shape[-2:])}")
        return output


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())
