"""Horizontal-flip heatmap ensembling and deterministic coordinate decoding."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from .dark_decode import dark_kernel, dark_offsets

from .scratch_cnn import FLIP_PAIRS, NUM_KEYPOINTS


FLIP_CHANNELS = np.arange(NUM_KEYPOINTS, dtype=np.int64)
for _left, _right in FLIP_PAIRS:
    FLIP_CHANNELS[_left] = _right
    FLIP_CHANNELS[_right] = _left


def flip_back_heatmaps(heatmaps: np.ndarray, *, shift: str = "none") -> np.ndarray:
    """Map heatmaps predicted from horizontally flipped inputs to original view."""

    if heatmaps.ndim != 4 or heatmaps.shape[1] != NUM_KEYPOINTS:
        raise ValueError("expected [batch,17,height,width] heatmaps")
    mapped = heatmaps[:, FLIP_CHANNELS, :, ::-1].copy()
    if shift == "right":
        mapped[:, :, :, 1:] = mapped[:, :, :, :-1].copy()
    elif shift == "left":
        mapped[:, :, :, :-1] = mapped[:, :, :, 1:].copy()
    elif shift != "none":
        raise ValueError(f"unknown flip alignment: {shift}")
    return mapped


def decode_heatmaps_refined(
    heatmaps: np.ndarray,
    geometries: Sequence[tuple[float, float, float, float]],
    *,
    refinement: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Decode heatmap peaks with quarter-pixel, log-quadratic or DARK refinement.

    ``dark`` / ``dark7`` blur each map (kernel 11 / 7) before a Hessian Taylor step; the
    returned maxima are always the raw map peaks, so pose scores do not depend on the mode.
    """

    batch, joints, height, width = heatmaps.shape
    if refinement in ("dark", "dark7"):
        if len(geometries) != batch:
            raise ValueError("geometry count differs from heatmap batch")
        flattened = heatmaps.reshape(batch, joints, -1)
        indices = np.argmax(flattened, axis=2)
        maxima = np.max(flattened, axis=2)
        coords = np.stack((indices % width, indices // width), axis=2).astype(np.float32)
        kernel = dark_kernel(width) if refinement == "dark" else 7
        coords = coords + dark_offsets(heatmaps, coords, kernel).astype(np.float32)
        geometry = np.asarray(geometries, dtype=np.float64)
        coords[:, :, 0] = coords[:, :, 0] / (width - 1.0) * geometry[:, None, 2] + geometry[:, None, 0]
        coords[:, :, 1] = coords[:, :, 1] / (height - 1.0) * geometry[:, None, 3] + geometry[:, None, 1]
        return coords, maxima
    if len(geometries) != batch:
        raise ValueError("geometry count differs from heatmap batch")
    flattened = heatmaps.reshape(batch, joints, -1)
    indices = np.argmax(flattened, axis=2)
    maxima = np.max(flattened, axis=2)
    coords = np.stack((indices % width, indices // width), axis=2).astype(np.float32)
    for batch_index in range(batch):
        for joint_index in range(joints):
            x_coord = int(coords[batch_index, joint_index, 0])
            y_coord = int(coords[batch_index, joint_index, 1])
            if not (1 <= x_coord < width - 1 and 1 <= y_coord < height - 1):
                continue
            plane = heatmaps[batch_index, joint_index]
            if refinement == "quarter":
                dx = plane[y_coord, x_coord + 1] - plane[y_coord, x_coord - 1]
                dy = plane[y_coord + 1, x_coord] - plane[y_coord - 1, x_coord]
                coords[batch_index, joint_index, 0] += np.sign(dx) * 0.25
                coords[batch_index, joint_index, 1] += np.sign(dy) * 0.25
            elif refinement == "quadratic":
                log_plane = np.log(np.maximum(plane, 1e-10))
                center = float(log_plane[y_coord, x_coord])
                left = float(log_plane[y_coord, x_coord - 1])
                right = float(log_plane[y_coord, x_coord + 1])
                top = float(log_plane[y_coord - 1, x_coord])
                bottom = float(log_plane[y_coord + 1, x_coord])
                denom_x = left - 2.0 * center + right
                denom_y = top - 2.0 * center + bottom
                if abs(denom_x) > 1e-8:
                    offset_x = 0.5 * (left - right) / denom_x
                    coords[batch_index, joint_index, 0] += float(np.clip(offset_x, -0.5, 0.5))
                if abs(denom_y) > 1e-8:
                    offset_y = 0.5 * (top - bottom) / denom_y
                    coords[batch_index, joint_index, 1] += float(np.clip(offset_y, -0.5, 0.5))
            else:
                raise ValueError(f"unknown refinement: {refinement}")
        crop_left, crop_top, crop_width, crop_height = geometries[batch_index]
        coords[batch_index, :, 0] = (
            coords[batch_index, :, 0] / (width - 1.0) * crop_width + crop_left
        )
        coords[batch_index, :, 1] = (
            coords[batch_index, :, 1] / (height - 1.0) * crop_height + crop_top
        )
    return coords, maxima
