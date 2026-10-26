"""Bounded crop-scale TTA helpers for the frozen scratch pose model.

The model was trained with a 1.25 padding factor.  This module deliberately
changes only inference geometry; it does not expose a general hyperparameter
search surface.
"""

from __future__ import annotations

import math
from typing import Mapping, Sequence

import torch
import torch.nn.functional as functional

from .scratch_cnn import INPUT_HEIGHT, INPUT_WIDTH


def scaled_crop_geometry(
    box: Sequence[float], padding: float
) -> tuple[float, float, float, float]:
    """Return the aspect-correct, centered crop for an explicit padding."""

    x_coord, y_coord, width, height = (float(value) for value in box[:4])
    if not all(math.isfinite(value) for value in (x_coord, y_coord, width, height, padding)):
        raise ValueError("box and padding values must be finite")
    if width <= 0.0 or height <= 0.0 or padding <= 0.0:
        raise ValueError("box dimensions and padding must be positive")
    center_x = x_coord + width * 0.5
    center_y = y_coord + height * 0.5
    target_aspect = INPUT_WIDTH / INPUT_HEIGHT
    if width > target_aspect * height:
        height = width / target_aspect
    elif width < target_aspect * height:
        width = height * target_aspect
    width *= padding
    height *= padding
    return center_x - width * 0.5, center_y - height * 0.5, width, height


def align_heatmaps_between_paddings(
    heatmaps: torch.Tensor,
    *,
    source_padding: float,
    target_padding: float,
) -> torch.Tensor:
    """Resample centered source heatmaps into the target crop coordinate frame.

    Both crops share their center and aspect ratio.  With ``align_corners=True``
    (matching the decoder's ``size - 1`` convention), the source sampling
    coordinate is simply target/source times the target normalized coordinate.
    Values outside the source crop are zero.
    """

    if heatmaps.ndim != 4:
        raise ValueError("expected [batch,joints,height,width] heatmaps")
    if source_padding <= 0.0 or target_padding <= 0.0:
        raise ValueError("padding factors must be positive")
    if source_padding == target_padding:
        return heatmaps
    ratio = float(target_padding / source_padding)
    theta = heatmaps.new_zeros((heatmaps.shape[0], 2, 3))
    theta[:, 0, 0] = ratio
    theta[:, 1, 1] = ratio
    grid = functional.affine_grid(theta, heatmaps.shape, align_corners=True)
    return functional.grid_sample(
        heatmaps,
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )


def aligned_weighted_mean(
    heatmaps_by_padding: Mapping[float, torch.Tensor],
    members: Sequence[tuple[float, float]],
    *,
    target_padding: float,
) -> torch.Tensor:
    """Align a predeclared set of scales and compute its normalized mean."""

    if not members:
        raise ValueError("members must not be empty")
    total_weight = float(sum(weight for _, weight in members))
    if total_weight <= 0.0 or any(weight < 0.0 for _, weight in members):
        raise ValueError("weights must be nonnegative with positive sum")
    combined: torch.Tensor | None = None
    for padding, weight in members:
        if padding not in heatmaps_by_padding:
            raise KeyError(f"missing padding {padding}")
        aligned = align_heatmaps_between_paddings(
            heatmaps_by_padding[padding],
            source_padding=padding,
            target_padding=target_padding,
        )
        contribution = aligned * (float(weight) / total_weight)
        combined = contribution if combined is None else combined + contribution
    assert combined is not None
    return combined
