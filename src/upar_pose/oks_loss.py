"""Optional OKS-aware training terms for the ViT pose heatmap loss.

Motivation (artifacts/runs/20261012T0015_head_error_mechanism): the head joints
(nose, head_bottom, head_top) carry the largest OKS deficit, and the error is a
fit limit: the peak sits ~0.7 cell off inside the right blob.  The plain heatmap
MSE charges a 0.75-cell shift the same for a head and a hip, while OKS charges it
~4x (sigma ratio 0.107 / 0.025) more for the head.  Two optional terms:

``joint_loss_weights``  17 per-joint multipliers on the heatmap MSE numerator.
    The denominator stays the visible-joint count, so a head multiplier of 2.5
    raises the head gradient 2.5x without diluting the other joints.

``coord_loss``  {"weight", "type", "temperature", "oks_scaled", ...}
    A differentiable soft-argmax readout of every predicted map (fp32, autocast
    off) compared with the target coordinate of each weighted joint whose target
    lies on the map, in units of the joint's OKS kernel width kappa_k * s =
    2 * sigma_k * s, where s = sqrt(GT area) in output cells.  The batch carries
    no box, but crops are box-normalised (square side = 1.25 x max(w, h) x
    U(0.9, 1.1) jitter; GT area == w*h on all three sources), so
    s = scale_fraction * (S - 1) with scale_fraction = sqrt(w h) / crop side.
    Its sampling-weighted median on the 40,422-person fit population is 0.502
    (5-95%: 0.45-0.57) -> default 0.5 (64 grid: 31.5 cells, as quality_targets'
    32; 96 grid: 47.5 cells).  The per-joint ratio (heads vs hips) is exact; the
    per-person scale is approximate (+-12% aspect, +-10% jitter), which only
    reweights persons, never joints.

    Readout (``soft_argmax_2d``): p = softmax(beta * h / max(h)) over a
    (2r+1)^2 window centred on the detached argmax of each channel (peak
    normalised by the detached peak, floor ``peak_floor``); x = sum p * x.
    ``temperature`` and ``window`` are given in 64-grid units and rescaled to the
    output grid: beta = temperature * ((S-1)/63)^2, r = ceil(window * (S-1)/63),
    so the readout width (sigma_target / sqrt(beta) = 0.63 cell at the default
    10) and its sub-cell linearity are the same on 64/96/192 grids.  Measured on
    ideal target blobs (test b): bias <= 0.005 cell, d readout / d position in
    [0.98, 1.06]; without the window the background mass biases the 64-grid
    readout up to 0.09 cell at beta 10 (1.9 cells at beta 10 on 96 without the
    rescale), and beta >= 16 quantises (slope 0.86-1.16).  beta is FIXED: a
    learnable beta would tune a training-only readout (deployment decodes the
    peak), could lower the loss by reshaping the readout rather than moving the
    peak, and would need a new optimizer parameter (a larger trainer change).

    Residual: u = (prediction - target) / (kappa_k s) per axis.  ``norm``
    "euclidean" (default) uses |u| = sqrt(ux^2 + uy^2 + d^2) - d (OKS is
    isotropic; the gradient always points at the target), "per_axis" sums the
    same per axis (classic integral-regression L1; its kink on an axis that is
    already right pushes that axis off by a step - test b).  d = max(``smooth``,
    1e-3) OKS widths: ``smooth`` 0 (default) is L1 (constant pull, head/hip ratio
    = sigma ratio at every distance); ``smooth`` > 0 is pseudo-Huber, quadratic
    below d like OKS itself (1 - exp(-u^2/2) ~ u^2/2), so the pull vanishes as
    the error vanishes instead of dithering around the target under Adam, at the
    cost of a ~sigma^2 head/hip ratio for errors below d.  CPU fit tests (README):
    16-crop memorisation, L1 dithers (head 0.055 vs MSE 0.020 cell) while smooth
    0.25 matches MSE (0.023); 512 crops, 200 steps, head error 0.456 (MSE) ->
    0.414 (L1) -> 0.383 cell (smooth 0.25) with the body unchanged.
      "l1"        the residual itself (OKS-scaled L1).
      "rle_lite"  NLL under a FIXED heavy-tailed density in OKS units:
                  -log(rho e^-u + (1 - rho) e^(-u / tau) / tau), shifted to 0 at
                  u = 0 (rho 0.9, tau 10): L1 slope near 0, slope 1/tau beyond
                  ~5 OKS widths (gross errors / flips, whose OKS term is ~0).
                  RLE without the flow and without a predicted sigma (both
                  would need new parameters, i.e. model/checkpoint changes).
    ``oks_scaled: false`` divides by the mean scored-joint kappa instead of the
    per-joint one (same overall magnitude, uniform joint weighting) for ablation.

With both keys absent (or null, or coord weight 0 without joint weights)
``oks_aware_vit_pose_loss`` returns ``vit_pose_loss(output, targets, weights)``
itself: bit-identical loss and gradients for every existing config.
"""

from __future__ import annotations

import math

import torch

from .constants import COCO_KEYPOINT_SIGMAS, IGNORED_KEYPOINT_INDICES
from .vit_pose import vit_pose_loss

DEFAULT_SCALE_FRACTION = 0.5
COORD_DEFAULTS = {'type': 'l1', 'temperature': 10.0, 'window': 6.0, 'norm': 'euclidean', 'smooth': 0.0, 'oks_scaled': True,
                  'scale_fraction': DEFAULT_SCALE_FRACTION, 'peak_floor': 0.1, 'rle_mix': 0.9, 'rle_tail': 10.0}
COORD_KEYS = set(COORD_DEFAULTS) | {'weight'}
EPS = 1e-3   # OKS-width units; keeps the norm differentiable at a zero residual when smooth == 0


def resolve_oks_loss_config(config: dict) -> tuple[list[float] | None, dict | None]:
    """Validated (joint_loss_weights, coord_loss) from a trainer config; (None, None) = feature off."""
    joint = config.get('joint_loss_weights')
    if joint is not None:
        joint = [float(v) for v in joint]
        if len(joint) != len(COCO_KEYPOINT_SIGMAS) or not all(math.isfinite(v) and v >= 0 for v in joint):
            raise ValueError('joint_loss_weights needs 17 finite non-negative multipliers')
    coord = config.get('coord_loss')
    if coord is not None:
        unknown = set(coord) - COORD_KEYS
        if unknown or 'weight' not in coord:
            raise ValueError(f'coord_loss needs "weight"; unknown keys {sorted(unknown)}')
        coord = {**COORD_DEFAULTS, **coord}
        coord['weight'] = float(coord['weight'])
        if coord['type'] not in ('l1', 'rle_lite') or coord['norm'] not in ('euclidean', 'per_axis'):
            raise ValueError(f"coord_loss type {coord['type']!r} / norm {coord['norm']!r} not supported")
        if not (coord['weight'] >= 0 and coord['temperature'] > 0 and (coord['window'] is None or coord['window'] > 0)
                and coord['smooth'] >= 0 and coord['scale_fraction'] > 0 and coord['peak_floor'] > 0 and 0 < coord['rle_mix'] < 1
                and coord['rle_tail'] > 1):
            raise ValueError(f'coord_loss values out of range: {coord}')
        if coord['weight'] == 0:
            coord = None
    return joint, coord


def soft_argmax_2d(maps: torch.Tensor, beta: float, peak_floor: float = 0.1, radius: int | None = None) -> torch.Tensor:
    """Differentiable [B, K, 2] (x, y) readout in output-cell units, fp32 with autocast off.

    p = softmax(beta * h / max(peak.detach(), peak_floor)) over each channel, restricted to the cells within
    ``radius`` (Chebyshev) of the detached argmax when radius is given.  The largest logit is <= beta and softmax
    subtracts the max, so nothing overflows for any input range; masked cells get -1e4 (exp underflows to 0).
    """
    with torch.autocast(device_type=maps.device.type, enabled=False):
        h = maps.float()
        b, k, height, width = h.shape
        flat = h.flatten(2)
        peak = flat.detach().amax(-1, keepdim=True).clamp_min(peak_floor)
        logits = flat * (beta / peak)
        xs = torch.arange(width, device=h.device, dtype=torch.float32)
        ys = torch.arange(height, device=h.device, dtype=torch.float32)
        if radius is not None:
            index = flat.detach().argmax(-1)
            ax, ay = (index % width).float(), torch.div(index, width, rounding_mode='floor').float()
            inside = (((xs.view(1, 1, 1, -1) - ax[..., None, None]).abs() <= radius)
                      & ((ys.view(1, 1, -1, 1) - ay[..., None, None]).abs() <= radius))
            logits = logits.masked_fill(~inside.flatten(2), -1e4)
        prob = torch.softmax(logits, dim=-1).view(b, k, height, width)
        return torch.stack(((prob.sum(2) * xs).sum(-1), (prob.sum(3) * ys).sum(-1)), -1)


def oks_widths(size: int, scale_fraction: float, oks_scaled: bool, device) -> torch.Tensor:
    """[17] OKS kernel widths kappa_k * s = 2 sigma_k * s in cells of a size x size map."""
    sigmas = torch.tensor(COCO_KEYPOINT_SIGMAS, dtype=torch.float32, device=device)
    if not oks_scaled:
        scored = [i for i in range(len(COCO_KEYPOINT_SIGMAS)) if i not in IGNORED_KEYPOINT_INDICES]
        sigmas = sigmas[scored].mean().expand(len(COCO_KEYPOINT_SIGMAS))
    return 2.0 * sigmas * (scale_fraction * (size - 1))


def readout_parameters(size: int, coord: dict) -> tuple[float, int | None]:
    """(beta, radius) on a size x size grid from the 64-grid-unit temperature / window."""
    factor = (size - 1) / 63.0
    radius = None if coord['window'] is None else int(math.ceil(coord['window'] * factor - 1e-9))
    return coord['temperature'] * factor * factor, radius


def coordinate_loss(output, weights, coords64, coord: dict):
    """(scalar, per-joint [B, K] loss, [B, K] mask, [B, K, 2] readout) of the OKS-shaped coordinate term."""
    with torch.autocast(device_type=output.device.type, enabled=False):
        if output.shape[-1] != output.shape[-2]:
            raise ValueError('coord_loss assumes square heatmaps')
        size = output.shape[-1]
        beta, radius = readout_parameters(size, coord)
        pred = soft_argmax_2d(output, beta, coord['peak_floor'], radius)
        # Same corner-aligned convention as gaussian_targets / make_heatmaps_highres (64-unit coords).
        target = coords64.float() * ((size - 1) / 63.0)
        inside = ((target >= 0) & (target <= size - 1)).all(-1)      # False for NaN / off-map targets
        keep = (weights.float() > 0) & inside
        mask = keep.float()
        # Masked joints read their own (detached) prediction: zero residual, so no inf/NaN can reach the gradient.
        target = torch.where(keep.unsqueeze(-1), target, pred.detach())
        u = (pred - target) / oks_widths(size, coord['scale_fraction'], coord['oks_scaled'], output.device).view(1, -1, 1)
        delta = max(float(coord['smooth']), EPS)   # pseudo-Huber: quadratic below delta, L1 slope above
        if coord['norm'] == 'euclidean':
            r = (u.square().sum(-1) + delta * delta).sqrt() - delta
        else:
            r = (u.square() + delta * delta).sqrt() - delta
        if coord['type'] == 'l1':
            per = r
        else:
            rho, tau = coord['rle_mix'], coord['rle_tail']
            per = math.log(rho + (1.0 - rho) / tau) - torch.log(rho * torch.exp(-r) + (1.0 - rho) / tau * torch.exp(-r / tau))
        per_joint = (per if per.dim() == 2 else per.sum(-1)) * mask
        value = per_joint.sum() / mask.sum().clamp_min(1.0)
    return value, per_joint, mask, pred


def oks_aware_vit_pose_loss(output, targets, weights, coords64, config: dict):
    """Drop-in for ``vit_pose_loss`` at the trainer call site; config keys absent -> exactly vit_pose_loss."""
    joint, coord = resolve_oks_loss_config(config)
    if joint is None and coord is None:
        return vit_pose_loss(output, targets, weights)
    joint_w = weights if joint is None else weights * torch.tensor(joint, dtype=weights.dtype, device=weights.device)
    heat = ((output.float() - targets.float()).square().mean((2, 3)) * joint_w).sum() / weights.sum().clamp_min(1.0)
    parts = {'heatmap': heat}
    loss = heat
    if coord is not None:
        value, _, _, _ = coordinate_loss(output, weights, coords64, coord)
        parts['coord'] = value
        loss = heat + coord['weight'] * value
    return loss, parts
