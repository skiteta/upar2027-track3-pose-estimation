"""DARK coordinate refinement (Zhang et al., CVPR 2020) shared by evaluation and the packaged run.py.

Self-contained (numpy only): the packager embeds this file verbatim into run.py.
"""
import numpy as np


def dark_kernel(width):
    """Blur kernel for a heatmap grid: 11 on the 64 grid (sigma-2 targets), scaled with the grid, odd."""
    return 2 * int(round(5.0 * (width - 1) / 63.0)) + 1


def gaussian_blur_preserving_peak(heatmaps: np.ndarray, kernel: int) -> np.ndarray:
    """OpenCV-equivalent separable Gaussian blur (sigma from the kernel size, reflect-101
    borders) rescaled so every map keeps its original maximum, as in DARK's modulation."""

    radius = (kernel - 1) // 2
    sigma = 0.3 * ((kernel - 1) * 0.5 - 1) + 0.8
    taps = np.exp(-0.5 * (np.arange(-radius, radius + 1) / sigma) ** 2)
    taps = (taps / taps.sum()).astype(np.float64)
    original_max = heatmaps.max(axis=(2, 3), keepdims=True)
    padded = np.pad(heatmaps.astype(np.float64), ((0, 0), (0, 0), (radius, radius), (radius, radius)), mode="reflect")
    rows = sum(taps[i] * padded[:, :, :, i:i + heatmaps.shape[3]] for i in range(kernel))
    blurred = sum(taps[i] * rows[:, :, i:i + heatmaps.shape[2], :] for i in range(kernel))
    blurred_max = blurred.max(axis=(2, 3), keepdims=True)
    return blurred * np.where(blurred_max > 0, original_max / np.where(blurred_max > 0, blurred_max, 1), 1.0)


_DARK_PATCH_CACHE = {}


def _get_dark_patch_tables(kernel: int, height: int, width: int):
    key = (int(kernel), int(height), int(width))
    cached = _DARK_PATCH_CACHE.get(key)
    if cached is None:
        radius = (kernel - 1) // 2
        sigma = 0.3 * ((kernel - 1) * 0.5 - 1) + 0.8
        taps = np.exp(-0.5 * (np.arange(-radius, radius + 1) / sigma) ** 2)
        taps = (taps / taps.sum()).astype(np.float64)
        win = 5 + 2 * radius
        b_mat = np.zeros((5, win), dtype=np.float64)
        for r in range(5):
            b_mat[r, r : r + kernel] = taps
        b_mat_t = b_mat.T.copy()
        refl_y = np.pad(np.arange(height, dtype=np.int64), (radius, radius), mode="reflect")
        refl_x = np.pad(np.arange(width, dtype=np.int64), (radius, radius), mode="reflect")
        cached = (win, b_mat, b_mat_t, refl_y, refl_x)
        _DARK_PATCH_CACHE[key] = cached
    return cached


def dark_offsets(heatmaps: np.ndarray, coords: np.ndarray, kernel: int) -> np.ndarray:
    """DARK (Zhang et al., CVPR 2020) second-order Taylor offsets on the blurred log map."""

    batch, joints, height, width = heatmaps.shape
    win, b_mat, b_mat_t, refl_y, refl_x = _get_dark_patch_tables(kernel, height, width)
    offsets = np.zeros_like(coords)
    for b in range(batch):
        for j in range(joints):
            px, py = int(coords[b, j, 0]), int(coords[b, j, 1])
            if not (2 <= px < width - 2 and 2 <= py < height - 2):
                continue
            plane = heatmaps[b, j]
            orig_max = float(plane[py, px])
            ys = refl_y[py - 2 : py - 2 + win]
            xs = refl_x[px - 2 : px - 2 + win]
            patch = plane[np.ix_(ys, xs)].astype(np.float64)
            blurred_5x5 = b_mat @ patch @ b_mat_t
            b_max = float(blurred_5x5.max())
            if b_max > 0.0 and orig_max > 0.0:
                blurred_5x5 = blurred_5x5 * (orig_max / b_max)
            m = np.log(np.maximum(blurred_5x5, 1e-10))
            dx = 0.5 * (m[2, 3] - m[2, 1])
            dy = 0.5 * (m[3, 2] - m[1, 2])
            dxx = 0.25 * (m[2, 4] - 2 * m[2, 2] + m[2, 0])
            dxy = 0.25 * (m[3, 3] - m[1, 3] - m[3, 1] + m[1, 1])
            dyy = 0.25 * (m[4, 2] - 2 * m[2, 2] + m[0, 2])
            det = dxx * dyy - dxy * dxy
            if abs(det) > 1e-12:
                offsets[b, j, 0] = -(dyy * dx - dxy * dy) / det
                offsets[b, j, 1] = -(dxx * dy - dxy * dx) / det
    return np.clip(offsets, -1.0, 1.0)
