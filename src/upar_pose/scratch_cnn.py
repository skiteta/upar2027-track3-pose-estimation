"""Compact from-scratch heatmap pose model and shared geometry."""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np
import torch
from torch import nn


INPUT_WIDTH = 192
INPUT_HEIGHT = 256
HEATMAP_WIDTH = 48
HEATMAP_HEIGHT = 64
NUM_KEYPOINTS = 17
PADDING_FACTOR = 1.25
FLIP_PAIRS = ((3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16))
SCORED_KEYPOINTS = tuple(index for index in range(NUM_KEYPOINTS) if index not in (3, 4))
IMAGE_MEAN = np.asarray([0.5, 0.5, 0.5], dtype=np.float32).reshape(3, 1, 1)
IMAGE_STD = np.asarray([0.25, 0.25, 0.25], dtype=np.float32).reshape(3, 1, 1)


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.activation = nn.SiLU(inplace=True)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.activation(inputs + self.block(inputs))


class ScratchPoseNet(nn.Module):
    """Small all-convolutional top-down pose network with a 1/4 heatmap head."""

    def __init__(self, residual_blocks: int = 4) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(32),
            nn.SiLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.SiLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.SiLU(inplace=True),
            *[ResidualBlock(128) for _ in range(residual_blocks)],
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.SiLU(inplace=True),
            nn.Conv2d(64, NUM_KEYPOINTS, 1),
        )
        self._initialize()

    def _initialize(self) -> None:
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
        final = self.decoder[-1]
        assert isinstance(final, nn.Conv2d)
        nn.init.normal_(final.weight, std=0.001)
        nn.init.zeros_(final.bias)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(inputs))


def crop_geometry(box: Sequence[float]) -> tuple[float, float, float, float]:
    x_coord, y_coord, width, height = (float(value) for value in box[:4])
    if not all(math.isfinite(value) for value in (x_coord, y_coord, width, height)):
        raise ValueError("box values must be finite")
    if width <= 0.0 or height <= 0.0:
        raise ValueError("box width and height must be positive")
    center_x = x_coord + width * 0.5
    center_y = y_coord + height * 0.5
    target_aspect = INPUT_WIDTH / INPUT_HEIGHT
    if width > target_aspect * height:
        height = width / target_aspect
    elif width < target_aspect * height:
        width = height * target_aspect
    width *= PADDING_FACTOR
    height *= PADDING_FACTOR
    return center_x - width * 0.5, center_y - height * 0.5, width, height


def normalize_image(rgb: np.ndarray) -> np.ndarray:
    tensor = rgb.astype(np.float32).transpose(2, 0, 1) / 255.0
    return ((tensor - IMAGE_MEAN) / IMAGE_STD).astype(np.float32)


def transform_keypoints(
    keypoints: np.ndarray,
    geometry: tuple[float, float, float, float],
    *,
    flipped: bool,
) -> np.ndarray:
    left, top, width, height = geometry
    transformed = keypoints.astype(np.float32, copy=True)
    transformed[:, 0] = (transformed[:, 0] - left) / width * (INPUT_WIDTH - 1.0)
    transformed[:, 1] = (transformed[:, 1] - top) / height * (INPUT_HEIGHT - 1.0)
    if flipped:
        transformed[:, 0] = INPUT_WIDTH - 1.0 - transformed[:, 0]
        for left_index, right_index in FLIP_PAIRS:
            transformed[[left_index, right_index]] = transformed[[right_index, left_index]]
    return transformed


def make_heatmaps(keypoints: np.ndarray, sigma: float = 1.8) -> tuple[np.ndarray, np.ndarray]:
    heatmaps = np.zeros((NUM_KEYPOINTS, HEATMAP_HEIGHT, HEATMAP_WIDTH), dtype=np.float32)
    weights = np.zeros(NUM_KEYPOINTS, dtype=np.float32)
    radius = int(math.ceil(3.0 * sigma))
    for index, (x_coord, y_coord, visibility) in enumerate(keypoints):
        if visibility <= 0 or not math.isfinite(float(x_coord)) or not math.isfinite(float(y_coord)):
            continue
        heat_x = float(x_coord) / (INPUT_WIDTH - 1.0) * (HEATMAP_WIDTH - 1.0)
        heat_y = float(y_coord) / (INPUT_HEIGHT - 1.0) * (HEATMAP_HEIGHT - 1.0)
        if heat_x < -radius or heat_x >= HEATMAP_WIDTH + radius or heat_y < -radius or heat_y >= HEATMAP_HEIGHT + radius:
            continue
        x_min = max(0, int(math.floor(heat_x)) - radius)
        x_max = min(HEATMAP_WIDTH, int(math.floor(heat_x)) + radius + 1)
        y_min = max(0, int(math.floor(heat_y)) - radius)
        y_max = min(HEATMAP_HEIGHT, int(math.floor(heat_y)) + radius + 1)
        xs = np.arange(x_min, x_max, dtype=np.float32)
        ys = np.arange(y_min, y_max, dtype=np.float32)[:, None]
        heatmaps[index, y_min:y_max, x_min:x_max] = np.exp(
            -((xs - heat_x) ** 2 + (ys - heat_y) ** 2) / (2.0 * sigma * sigma)
        )
        weights[index] = 1.0
    weights[3] = 0.0
    weights[4] = 0.0
    return heatmaps, weights


def decode_heatmaps(
    heatmaps: np.ndarray,
    geometries: Sequence[tuple[float, float, float, float]],
) -> tuple[np.ndarray, np.ndarray]:
    batch, joints, height, width = heatmaps.shape
    flattened = heatmaps.reshape(batch, joints, -1)
    indices = np.argmax(flattened, axis=2)
    maxima = np.max(flattened, axis=2)
    coords = np.stack((indices % width, indices // width), axis=2).astype(np.float32)
    for batch_index in range(batch):
        for joint_index in range(joints):
            x_coord = int(coords[batch_index, joint_index, 0])
            y_coord = int(coords[batch_index, joint_index, 1])
            if 1 <= x_coord < width - 1 and 1 <= y_coord < height - 1:
                dx = heatmaps[batch_index, joint_index, y_coord, x_coord + 1] - heatmaps[batch_index, joint_index, y_coord, x_coord - 1]
                dy = heatmaps[batch_index, joint_index, y_coord + 1, x_coord] - heatmaps[batch_index, joint_index, y_coord - 1, x_coord]
                coords[batch_index, joint_index, 0] += np.sign(dx) * 0.25
                coords[batch_index, joint_index, 1] += np.sign(dy) * 0.25
        left, top, crop_width, crop_height = geometries[batch_index]
        coords[batch_index, :, 0] = coords[batch_index, :, 0] / (width - 1.0) * crop_width + left
        coords[batch_index, :, 1] = coords[batch_index, :, 1] / (height - 1.0) * crop_height + top
    return coords, maxima
