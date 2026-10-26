"""Expected-OKS decoding blur and left/right ambiguity score, shared by evaluation and the packaged run.py.

A heatmap regressed to Gaussian targets (std ``t`` cells) approximates the joint
posterior blurred by that Gaussian.  The expected OKS term of a guess is the
posterior convolved with the OKS kernel, whose std is ``2 sigma_j sqrt(area)``
pixels, so blurring the map by ``sqrt(max((alpha k_j)^2 - t^2, 0))`` cells before
the argmax picks the point with the highest expected OKS term (``alpha`` = 1 in
theory).  ``area`` is the input box ``w*h``.

The ambiguity of a person is the mean over limb joints of the strongest local
maximum farther than ``0.1 sqrt(area)`` from the peak, relative to the peak: a
bimodal map is an uncertain left/right or limb choice.  The torso ambiguity is the
maximum of the same ratio over shoulders and hips: a second torso peak marks an
uncertain front/back orientation, the cause of most whole-body left/right swaps.

Self-contained (numpy + torch only): the packager embeds this file verbatim into run.py.
"""
import numpy as np
import torch
import torch.nn.functional as F

EOKS_SIGMAS = np.asarray((0.026, 0.025, 0.025, 0.035, 0.035, 0.079, 0.079, 0.072, 0.072,
                          0.062, 0.062, 0.107, 0.107, 0.087, 0.087, 0.089, 0.089))
AMBIGUITY_JOINTS = tuple(range(5, 17))
TORSO_JOINTS = (5, 6, 11, 12)


def expected_oks_stds(boxes, geometry, width, alpha, target_std):
    """(B, 17) extra blur in cells; boxes are input xywh boxes, geometry the target crops (left, top, side, side)."""
    boxes = np.asarray(boxes, dtype=np.float64)
    geometry = np.asarray(geometry, dtype=np.float64)
    k = 2 * EOKS_SIGMAS[None, :] * np.sqrt(boxes[:, 2] * boxes[:, 3])[:, None] * (width - 1) / geometry[:, 2:3]
    return np.sqrt(np.maximum((alpha * k) ** 2 - target_std ** 2, 0.0))


def expected_oks_blur(maps, stds):
    """Blur every (person, joint) map by its std (quantised to 0.25 cells; zero borders; taps truncated at ceil(3 std)).

    One depthwise convolution per axis for all maps: each map has its own kernel, zero beyond its radius.
    """
    batch, joints, height, width = maps.shape
    q = np.round(np.asarray(stds, dtype=np.float64).reshape(-1) * 4) / 4
    if not (q > 0).any():
        return maps
    rmax = max(1, int(np.ceil(3 * q.max())))
    offsets = np.arange(-rmax, rmax + 1, dtype=np.float64)
    radius = np.maximum(1, np.ceil(3 * q)).astype(np.int64)
    with np.errstate(divide='ignore', invalid='ignore'):
        taps = np.exp(-0.5 * (offsets[None, :] / np.where(q > 0, q, 1.0)[:, None]) ** 2)
    taps = np.where(np.abs(offsets)[None, :] <= radius[:, None], taps, 0.0)
    taps = np.where((q > 0)[:, None], taps, (offsets == 0)[None, :].astype(np.float64))
    taps = taps / taps.sum(axis=1, keepdims=True)
    kernel = torch.as_tensor(taps, device=maps.device, dtype=maps.dtype)
    x = maps.reshape(1, batch * joints, height, width)
    x = F.conv2d(F.pad(x, (rmax, rmax, 0, 0)), kernel.view(-1, 1, 1, 2 * rmax + 1), groups=batch * joints)
    x = F.conv2d(F.pad(x, (0, 0, rmax, rmax)), kernel.view(-1, 1, 2 * rmax + 1, 1), groups=batch * joints)
    return x.reshape(batch, joints, height, width)


def secondary_peak_ratio(maps, boxes, geometry, min_sep=0.1, k=4, pool=5):
    """(B, J) strongest local maximum farther than ``min_sep * sqrt(area)`` from the peak, relative to the peak."""
    arr = maps.detach().cpu().numpy().astype(np.float32, copy=False) if isinstance(maps, torch.Tensor) else np.asarray(maps, dtype=np.float32)
    batch, joints, height, width = arr.shape
    r = pool // 2
    pad_w = np.pad(arr, ((0, 0), (0, 0), (0, 0), (r, r)), mode='constant', constant_values=-np.inf)
    mx = pad_w[..., 0:width]
    for i in range(1, pool):
        mx = np.maximum(mx, pad_w[..., i:width + i])
    pad_h = np.pad(mx, ((0, 0), (0, 0), (r, r), (0, 0)), mode='constant', constant_values=-np.inf)
    my = pad_h[..., 0:height, :]
    for i in range(1, pool):
        my = np.maximum(my, pad_h[..., i:height + i, :])
    local = np.where(arr == my, arr, np.float32(0.0)).reshape(batch, joints, -1)
    part_idx = np.argpartition(local, -k, axis=2)[..., -k:]
    part_vals = np.take_along_axis(local, part_idx, axis=2)
    order = np.lexsort((part_idx, -part_vals), axis=2)
    idx = np.take_along_axis(part_idx, order, axis=2)
    vals = np.take_along_axis(part_vals, order, axis=2)
    xy = np.stack((idx % width, idx // width), axis=-1).astype(np.float32)
    boxes = np.asarray(boxes, dtype=np.float64)
    geometry = np.asarray(geometry, dtype=np.float64)
    sep = (min_sep * np.sqrt(boxes[:, 2] * boxes[:, 3]) * (width - 1) / geometry[:, 2])[:, None, None]
    far = np.hypot(*(xy - xy[:, :, :1]).transpose(3, 0, 1, 2)) >= sep
    second = np.where(far, vals, 0).max(axis=2)
    return second / np.maximum(vals[:, :, 0], 1e-8)


def ambiguity(maps, boxes, geometry):
    """(B,) mean far-secondary-peak ratio over the limb joints."""
    return secondary_peak_ratio(maps, boxes, geometry)[:, list(AMBIGUITY_JOINTS)].mean(axis=1)


def torso_ambiguity(maps, boxes, geometry):
    """(B,) maximum far-secondary-peak ratio over shoulders and hips (front/back orientation doubt)."""
    return secondary_peak_ratio(maps, boxes, geometry)[:, list(TORSO_JOINTS)].max(axis=1)
