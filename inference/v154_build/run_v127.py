"""UPAR Track 3 code submission: fixed multi-architecture heatmap fusion with Gated 2-Slice Pose-SAHI.

Each model runs its declared per-crop views (its paddings x {original,
flipped-back}; all members share one set by default, and crops are prepared
once per distinct padding and reused by every member that declares it); its
views are aligned to the target padding and averaged with its own normalised
weights. When SAHI_CONFIG is enabled, Stage-1 coarse keypoints from member 0
guide high-resolution Upper-Body and Lower-Body 2-Slice Pose-SAHI crops and
Pose-Centered / Tight-Zoom crops, warped back into the target geometry with
a 4-cell border taper and blended via Cosine-Agreement x Peak-Ratio x
Aspect-Ratio gating. Every model's enhanced map is then scaled to unit peak per
joint, the maps are averaged with fixed model weights, and coordinates are
decoded from the fused map with the evaluator's log-quadratic or DARK refinement.
"""
from __future__ import annotations

from contextlib import nullcontext
import hashlib
import json
import math
import os
import time
from typing import Any, Sequence

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as functional


HERE = os.path.dirname(os.path.abspath(__file__))
META_PATH = os.path.join(HERE, "model_meta.json")
MODEL_TYPE = 'upar_fusion3_sahi_anat9_v118'
MODEL_FILES = ('model_rsn.torchscript', 'model_l3.torchscript', 'model_vp.torchscript')
MODEL_SHA256S = ('9df1ea685fb0c050640548a293948e424400d78f64a833a495622204bc252736', 'd1570a01939023295825aff2f9d519457cd0e27c98c7562dfc33575c580dda02', 'b64527f2585af2baecdd9fada4122bcf6a29805ac606692cf39b87451d5baaf7')
MODEL_WEIGHTS = (2.0, 2.0, 1.55)
MEMBER_PADDINGS = ((1.2, 1.3), (1.2, 1.3), (1.2, 1.3))
MEMBER_SCALE_WEIGHTS = ((0.5, 0.5), (0.5, 0.5), (0.5, 0.5))
GRAPH_MODEL_INDICES = (0,)
DECODER = 'dark'
RUNTIME_GUARD = (0.21, 500)
EXPECTED_OKS = (0.75, 3.0)
AMBIGUITY_BETA = 0.45
TORSO_GAMMA = 6.0
SQUAT_BOX = None
MEMBER_DROP_GUARD = (0.21, 100)
PROBE_MEMBERS = ()
PROBE_TIMEOUT = 900.0
DIAGNOSTIC_FILES = ()
DIAGNOSTIC_TIMEOUT = 240.0
FALLBACK_WEIGHTS = None
SAHI_CONFIG = {'view_weights': (0.2, 0.25, 0.2), 'member_sahi_weights': (0.35, 0.0, 0.45), 'sim_pow': 2.0, 'peak_pow': 2.0, 'ar_center': 1.35, 'ar_clamp': (0.15, 1.0), 'sahi_wall_budget_seconds': 6600.0}
# Opt-in only. Existing packages retain their metadata and execution path. New
# isolated recipes must declare these constants in their frozen run.py.
ISOLATED_MODEL_INDICES = tuple(globals().get("ISOLATED_MODEL_INDICES", ()))
ISOLATED_LOAD_TIMEOUT = float(globals().get("ISOLATED_LOAD_TIMEOUT", 90.0))
ISOLATED_FORWARD_TIMEOUT = float(globals().get("ISOLATED_FORWARD_TIMEOUT", 30.0))
SAHI_CONFIG = globals().get("SAHI_CONFIG", None)
EXPECTED_TOTAL_IMAGES = 30245
SAHI_TIER0_ALL_PERSONS = False
TIER2_NO_SAHI = True
EOKS_LARGE_ALPHA = None
EOKS_LARGE_AREA = 9216.0
RSN_LARGE_SCALE = 0.0
RSN_LARGE_AREA = 40000.0
HEAD_DARK_KERNEL = 5
SCORE_SOFTVIS_TAU = 0.5
SCORE_TOPK = None
ROLLING_WINDOW_IMAGES = 80
WALL_BUDGET_TIER1_SECONDS = 6600.0
WALL_BUDGET_TIER2_SECONDS = 6600.0
WALL_BUDGET_DISABLE_SAHI_SECONDS = 6600.0
WALL_BUDGET_REDUCE_SECONDS = 7000.0
WALL_BUDGET_DROP_SECONDS = 7080.0
WALL_EMERGENCY_SINGLE_SECONDS = 7120.0
MODEL_PATHS = tuple(os.path.join(HERE, name) for name in MODEL_FILES)
TARGET_PADDING = 1.25
BATCH_SIZE = 16
SAHI_MAX_PAIRS_PER_CALL = 32
UNIFORM_VIEWS = len(set(zip(MEMBER_PADDINGS, MEMBER_SCALE_WEIGHTS, strict=True))) == 1
PADDINGS = MEMBER_PADDINGS[0] if UNIFORM_VIEWS else tuple(sorted({p for paddings in MEMBER_PADDINGS for p in paddings}))

_MODELS = None
_DEVICE = None
_META: dict[str, Any] = {}
_MEAN = np.asarray([0.5, 0.5, 0.5], dtype=np.float32).reshape(3, 1, 1)
_STD = np.asarray([0.25, 0.25, 0.25], dtype=np.float32).reshape(3, 1, 1)
_SCORED_KEYPOINTS = np.asarray(
    [index for index in range(17) if index not in (3, 4)], dtype=np.int64
)
_UPPER_JOINTS = np.asarray([0, 1, 2, 5, 6, 7, 8, 9, 10], dtype=np.int64)
_LOWER_JOINTS = np.asarray([11, 12, 13, 14, 15, 16], dtype=np.int64)
_HEAD_SHO_JOINTS = np.asarray([0, 1, 2, 5, 6], dtype=np.int64)
_TORSO_LEG_JOINTS = np.asarray([5, 6, 11, 12, 13, 14, 15, 16], dtype=np.int64)
_STAGE1_WEIGHTS_3 = (2.0, 2.35, 1.55)
_STAGE1_WEIGHTS_4 = (0.18, 0.18, 0.34, 0.30)
_SAHI_V5_WEIGHTS_5 = (0.12, 0.22, 0.22, 0.22, 0.22)
_SAHI_V6_WEIGHTS_5 = (0.15, 0.25, 0.25, 0.25, 0.25)
_CONSENSUS_WEIGHTS_3 = (2.0, 2.35, 1.55)
_CONSENSUS_WEIGHTS_5 = (2.0, 2.0, 1.55, 0.55, 0.55)
_CONSENSUS_PCT = 15.0
_CONSENSUS_PWR = 1.5
_KNEE_EXTRA_BLUR_STD = 0.85
_ANKLE_EXTRA_BLUR_STD = 1.75
_PIL_ROUNDING_ALPHA = 0.55
_UNIT_WEIGHTS_3 = np.asarray([
    [2.0, 2.25, 1.55],
    [2.0, 2.20, 1.55],
    [2.0, 2.85, 1.45],
    [2.0, 2.35, 1.55],
    [2.0, 2.35, 1.55],
    [2.0, 2.35, 1.80],
    [2.0, 2.35, 1.80],
    [1.9, 3.10, 1.70],
    [1.9, 3.10, 1.70],
    [2.0, 2.25, 1.55],
    [2.0, 2.25, 1.55],
    [2.5, 2.35, 1.55],
    [2.5, 2.35, 1.55],
    [2.3, 2.45, 1.50],
    [2.3, 2.45, 1.50],
    [2.6, 2.75, 1.45],
    [2.6, 2.75, 1.45],
], dtype=np.float32)
_UNIT_WEIGHTS_5 = np.asarray([
    [2.0, 1.8, 1.55, 0.45, 0.35],
    [2.0, 2.0, 1.55, 0.35, 0.15],
    [2.0, 2.2, 1.45, 0.30, 0.55],
    [2.0, 2.0, 1.55, 0.35, 0.25],
    [2.0, 2.0, 1.55, 0.35, 0.25],
    [2.0, 2.0, 1.80, 0.35, 0.25],
    [2.0, 2.0, 1.80, 0.35, 0.25],
    [1.9, 2.2, 1.70, 0.50, 0.80],
    [1.9, 2.2, 1.70, 0.50, 0.80],
    [2.0, 2.0, 1.55, 0.35, 0.15],
    [2.0, 2.0, 1.55, 0.35, 0.15],
    [2.5, 1.9, 1.55, 0.65, 0.25],
    [2.5, 1.9, 1.55, 0.65, 0.25],
    [2.3, 1.8, 1.50, 1.15, 0.35],
    [2.3, 1.8, 1.50, 1.15, 0.35],
    [2.6, 2.0, 1.45, 0.55, 0.55],
    [2.6, 2.0, 1.45, 0.55, 0.55],
], dtype=np.float32)
_FLIP_CHANNELS = np.asarray(
    [0, 1, 2, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15],
    dtype=np.int64,
)


def expected_inference() -> dict[str, Any]:
    inference = {
        "model_files": list(MODEL_FILES),
        "model_sha256s": list(MODEL_SHA256S),
        "model_weights": list(MODEL_WEIGHTS),
        "model_combination": "unit_peak_per_joint_then_weighted_mean_after_geometry_alignment",
        "crop_paddings": list(PADDINGS),
        "scale_weights": list(MEMBER_SCALE_WEIGHTS[0]) if UNIFORM_VIEWS else None,
        "target_padding": TARGET_PADDING,
        "horizontal_flip_each_scale": True,
        "heatmap_scale_alignment": "bilinear_centered_align_corners_true",
        "coordinate_refinement": "log_quadratic_3point" if DECODER == "quadratic" else "dark_blur_peak_preserving_hessian_taylor",
        "dark_kernel_rule": "2*round(5*(grid-1)/63)+1" if DECODER == "dark" else None,
        "confidence": "mean_over_models_of_model_scored_joint_peak_mean" + (
            "" if not AMBIGUITY_BETA else f"_times_1_minus_{AMBIGUITY_BETA:g}_limb_far_secondary_peak_ratio") + (
            "" if not TORSO_GAMMA else f"_times_exp_minus_{TORSO_GAMMA:g}_torso_max_far_secondary_peak_ratio"),
        "expected_oks_blur": None if EXPECTED_OKS is None else {
            "alpha": EXPECTED_OKS[0], "target_std_cells": EXPECTED_OKS[1],
            "rule": "blur fused map per joint by sqrt(max((alpha*2*sigma*sqrt(box_w*box_h)*(grid-1)/crop_side)^2 - t^2, 0)) cells, quantised 0.25, before argmax+refinement"},
        "view_count_per_crop": 2 * len(PADDINGS) if UNIFORM_VIEWS else None,
        "cuda_forward_autocast": "float16_cache_disabled",
        "cuda_nonfinite_fallback": "float32_full_batch_strict",
        "torchscript_executor_optimization": False,
        "cuda_graph_model_indices": list(GRAPH_MODEL_INDICES),
        "cuda_graph_policy": "capture_at_load_then_exact_match_vs_eager_else_all_eager",
        "fusion_grid": "finest_member_grid_bilinear_align_corners_true",
        "squat_box_extension": None if SQUAT_BOX is None else {
            "max_height_over_width": SQUAT_BOX[0], "target_height_over_width": SQUAT_BOX[1],
            "rule": "crop window only: if box h/w < max, extend the box downward (top fixed) to h = target*w; scores/eOKS use the original box"},
        "runtime_guard": None if RUNTIME_GUARD is None else {
            "seconds_per_image_budget": RUNTIME_GUARD[0], "min_images_before_switch": RUNTIME_GUARD[1],
            "reduced_paddings": [TARGET_PADDING], "rule": "mean wall seconds per image since first call above budget -> reduced views for the rest"},
    }
    inference["v112"] = {"head_dark_kernel": HEAD_DARK_KERNEL, "score_topk": SCORE_TOPK, "score_softvis_tau": SCORE_SOFTVIS_TAU, "sahi_tier0_all_persons": SAHI_TIER0_ALL_PERSONS,
                         "adaptive_shrink": list(globals().get("ADAPTIVE_SHRINK") or []) or None, "rsn_large_scale": RSN_LARGE_SCALE, "rsn_large_area": RSN_LARGE_AREA, "tier2_no_sahi": globals().get("TIER2_NO_SAHI"), "eoks_large_alpha": globals().get("EOKS_LARGE_ALPHA"), "eoks_large_area": globals().get("EOKS_LARGE_AREA"), "expected_oks_blur_device": "model device (cudnn TF32 off)", "dark_offsets": "vectorised, bit-identical to the per-joint loop",
                         "wall_thresholds": [WALL_BUDGET_TIER1_SECONDS, WALL_BUDGET_TIER2_SECONDS, WALL_BUDGET_DISABLE_SAHI_SECONDS,
                                             WALL_BUDGET_REDUCE_SECONDS, WALL_BUDGET_DROP_SECONDS, WALL_EMERGENCY_SINGLE_SECONDS]}
    if SAHI_CONFIG is not None:
        inference["pose_sahi"] = {
            "view_weights": list(SAHI_CONFIG["view_weights"]),
            "member_sahi_weights": list(SAHI_CONFIG["member_sahi_weights"]),
            "sim_pow": float(SAHI_CONFIG["sim_pow"]),
            "peak_pow": float(SAHI_CONFIG["peak_pow"]),
            "ar_center": float(SAHI_CONFIG["ar_center"]),
            "ar_clamp": list(SAHI_CONFIG["ar_clamp"]),
            "sahi_wall_budget_seconds": float(SAHI_CONFIG["sahi_wall_budget_seconds"]),
            "rule": "Stage-1 member 0 coarse keypoints guide zoom112/std125/pose114 and upper/lower 2-slice Pose-SAHI crops; warped back with 4-cell border taper and blended per joint by cosine-agreement x peak-ratio x aspect-ratio gating; base 2-view maxima retained for confidence",
        }
    if DIAGNOSTIC_FILES:
        inference["diagnostic_probes"] = {
            "files": list(DIAGNOSTIC_FILES), "timeout_seconds": DIAGNOSTIC_TIMEOUT,
            "rule": "diagnostic-only TorchScript files loaded and run three times in child processes at load time and logged; never fused"}
    if ISOLATED_MODEL_INDICES:
        inference["isolated_members"] = {
            "model_indices": list(ISOLATED_MODEL_INDICES),
            "load_timeout_seconds": ISOLATED_LOAD_TIMEOUT,
            "forward_timeout_seconds": ISOLATED_FORWARD_TIMEOUT,
            "rule": "persistent exec child per selected member; parent never loads that member; FP32 arrays over anonymous pipes; same finite-precision forward; worker error or timeout is fatal, not a silent member drop",
        }
    if PROBE_MEMBERS:
        inference["member_probe"] = {
            "model_files": [MODEL_FILES[i] for i in PROBE_MEMBERS], "timeout_seconds": PROBE_TIMEOUT,
            "rule": "each listed member is loaded and run three times on a fixed probe batch in a child process before the main load; a member whose child fails or exceeds the timeout is left out of the fusion (weights and scores renormalised over the remaining members)"}
    if FALLBACK_WEIGHTS is not None:
        inference["fallback_weights"] = {
            "model_weights": list(FALLBACK_WEIGHTS),
            "rule": "members with model weight 0 are not loaded; if any member probe fails, the fusion uses these weights instead (members with weight 0 here are not loaded, failed members are left out)"}
    if MEMBER_DROP_GUARD is not None:
        inference["member_drop_guard"] = {
            "seconds_per_image_budget": MEMBER_DROP_GUARD[0], "window_images": MEMBER_DROP_GUARD[1],
            "rule": "the runtime guard also reduces the views after 50 images when the mean exceeds 4x its budget; after the reduction each window of reduced images whose mean wall seconds per image exceeds this budget drops the member with the largest sampled forward time (fusion weights and scores renormalised over the remaining members)"}
    if not UNIFORM_VIEWS:
        inference["member_views"] = {
            "crop_paddings": [list(paddings) for paddings in MEMBER_PADDINGS],
            "scale_weights": [list(weights) for weights in MEMBER_SCALE_WEIGHTS],
            "view_count_per_crop": [2 * len(paddings) for paddings in MEMBER_PADDINGS],
            "crop_sharing": "one crop per box and distinct padding (sorted union = crop_paddings), reused by every member declaring it",
            "rule": "each member's paddings x flip are aligned to the target padding and averaged with its own scale weights before the unit-peak fusion; once the runtime guard trips every member uses the target padding only",
        }
    inference["runtime_guard"] = {
        "mode": "measured_full_vs_center_allocation_r3",
        "full_path_wall_budget_seconds": WALL_BUDGET_REDUCE_SECONDS,
        "expected_total_images": EXPECTED_TOTAL_IMAGES,
        "warmup_images": 480, "measurement_window_images": 80,
        "pace_margin_multiplier": 1.05, "reaction_margin_images": 160,
        "reduced_paddings": [TARGET_PADDING],
        "rule": "Only after full Tier3 projection exceeds budget, measure80centre images. Allocate remaining full views from observed full/centre costs, max of last3same-mode windows with5percent margin. SAHI remains disabled after calibration; reduced mode disables adaptive recrop. Keep original7120second emergency. Not a future-speed guarantee."}
    inference["member_drop_guard"] = {
        "mode": "measured_center_projection", "window_images": 80,
        "wall_budget_seconds": WALL_BUDGET_REDUCE_SECONDS,
        "rule": "If measured centre projection still exceeds budget, drop slowest sampled member, clear both mode histories and remeasure the full path for the remaining models. Probe centre again only if needed. SAHI stays disabled after a drop. The original7120second emergency retains fastest member."}
    return inference


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_metadata(meta: dict[str, Any]) -> None:
    expected_common = {
        "schema_version": 1,
        "model_type": MODEL_TYPE,
        "input_width": 256,
        "input_height": 256,
        "heatmap_width": 64,
        "heatmap_height": 64,
        "num_keypoints": 17,
        "padding_factor": 1.25,
        "image_mean": [0.5, 0.5, 0.5],
        "image_std": [0.25, 0.25, 0.25],
        "score_excludes_keypoint_indices": [3, 4],
        "hidden_data_used": False,
    }
    if len(MEMBER_PADDINGS) != len(MODEL_FILES) or len(MEMBER_SCALE_WEIGHTS) != len(MODEL_FILES) or any(
            len(paddings) != len(weights) or not paddings for paddings, weights in zip(MEMBER_PADDINGS, MEMBER_SCALE_WEIGHTS)):
        raise RuntimeError("one non-empty view set (paddings + matching scale weights) per fused member")
    if (len(set(ISOLATED_MODEL_INDICES)) != len(ISOLATED_MODEL_INDICES)
            or any(not isinstance(i, int) or i < 0 or i >= len(MODEL_FILES) for i in ISOLATED_MODEL_INDICES)
            or set(ISOLATED_MODEL_INDICES).intersection(GRAPH_MODEL_INDICES)
            or set(ISOLATED_MODEL_INDICES).intersection(PROBE_MEMBERS)
            or not all(math.isfinite(t) and t > 0 for t in (ISOLATED_LOAD_TIMEOUT, ISOLATED_FORWARD_TIMEOUT))):
        raise RuntimeError("isolated members need unique valid indices, positive timeouts, and no parent graph/probe")
    if not any(weight > 0 for weight in MODEL_WEIGHTS) or (FALLBACK_WEIGHTS is not None and (
            len(FALLBACK_WEIGHTS) != len(MODEL_FILES) or not PROBE_MEMBERS or not any(weight > 0 for weight in FALLBACK_WEIGHTS))):
        raise RuntimeError("fusion needs a positive weight; fallback weights need one weight per member and a member probe")
    if {key: meta.get(key) for key in expected_common} != expected_common:
        raise RuntimeError("model metadata common contract differs from the frozen fusion recipe")
    if meta.get("inference") != expected_inference():
        raise RuntimeError("model metadata inference contract differs from the frozen fusion recipe")
    parents = meta.get("parent_models")
    if not isinstance(parents, list) or len(parents) != len(MODEL_FILES):
        raise RuntimeError("model metadata must describe every fused parent")
    for row, filename, digest in zip(parents, MODEL_FILES, MODEL_SHA256S, strict=True):
        if row.get("model_file") != filename or row.get("model_sha256") != digest:
            raise RuntimeError("fused parent file/hash contract differs")


def load_model() -> None:
    global _MODELS, _DEVICE, _META
    if _MODELS is not None:
        return
    if _RUNTIME["load_start"] is None:
        _RUNTIME["load_start"] = time.perf_counter()
    if not os.path.isfile(META_PATH):
        raise FileNotFoundError("model_meta.json must be packaged beside run.py")
    with open(META_PATH, "r", encoding="utf-8") as handle:
        _META = json.load(handle)
    validate_metadata(_META)
    for path, expected_digest in zip(MODEL_PATHS, MODEL_SHA256S, strict=True):
        if not os.path.isfile(path):
            raise FileNotFoundError("the declared frozen TorchScript models must be packaged")
        if _sha256_file(path) != expected_digest:
            raise RuntimeError("packaged TorchScript differs from the exact frozen bytes")
    _DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_grad_enabled(False)
    torch.set_num_threads(2)
    if DIAGNOSTIC_FILES:
        gpu = torch.cuda.get_device_name(0) if _DEVICE.type == "cuda" else "none"
        print(f"[submission] torch {torch.__version__}, device {_DEVICE} ({gpu}); diagnostic probes {list(DIAGNOSTIC_FILES)} "
              f"({DIAGNOSTIC_TIMEOUT:.0f} s each)", flush=True)
        for name in DIAGNOSTIC_FILES:
            _run_probe(os.path.join(HERE, name), name, DIAGNOSTIC_TIMEOUT)
    if PROBE_MEMBERS:
        gpu = torch.cuda.get_device_name(0) if _DEVICE.type == "cuda" else "none"
        print(f"[submission] torch {torch.__version__}, device {_DEVICE} ({gpu}); probing members "
              f"{[MODEL_FILES[i] for i in PROBE_MEMBERS]} in subprocesses ({PROBE_TIMEOUT:.0f} s each)", flush=True)
    excluded = [index for index in PROBE_MEMBERS if not _probe_member(index)]
    active = active_members()
    if excluded and FALLBACK_WEIGHTS is not None:
        _MEMBERS["weights"] = tuple(FALLBACK_WEIGHTS)
        active[:] = [index for index, weight in enumerate(FALLBACK_WEIGHTS) if weight > 0]
        print(f"[submission] member probe failed: fallback fusion {[MODEL_FILES[i] for i in active]} "
              f"with weights {[FALLBACK_WEIGHTS[i] for i in active]}", flush=True)
    for index in excluded:
        if index in active:
            active.remove(index)
    _MODELS = []
    for index, path in enumerate(MODEL_PATHS):
        tick = time.perf_counter()
        if index not in active:
            model = None
        elif index in ISOLATED_MODEL_INDICES:
            model = IsolatedMember(path, _DEVICE, load_timeout=ISOLATED_LOAD_TIMEOUT,
                                   forward_timeout=ISOLATED_FORWARD_TIMEOUT)
        else:
            model = torch.jit.load(path, map_location=_DEVICE).float().eval()
        _MODELS.append(model)
        if PROBE_MEMBERS:
            print(f"[submission] loaded {MODEL_FILES[index]} in {time.perf_counter() - tick:.1f} s"
                  + (" (excluded by its probe)" if index in excluded else "" if index in active else " (not in the fusion)"), flush=True)
    views = f"{2 * len(PADDINGS)} views" if UNIFORM_VIEWS else "/".join(str(2 * len(p)) for p in MEMBER_PADDINGS) + " views per member"
    sahi_tag = " + Gated-2Slice-SAHI" if SAHI_CONFIG is not None else ""
    print(f"[submission] {MODEL_TYPE}: {len(active)}-model fusion, {views}{sahi_tag} on {_DEVICE}", flush=True)
    _prepare_graphs()


_PROBE_CODE = r"""
import sys, time, torch
path, device = sys.argv[1], sys.argv[2]
tick = time.perf_counter()
model = torch.jit.load(path, map_location=device).float().eval()
print(f"load {time.perf_counter() - tick:.1f} s", flush=True)
inputs = (torch.rand((4, 3, 256, 256), generator=torch.Generator().manual_seed(0)) * 4.0 - 2.0).to(device)
context = torch.autocast("cuda", dtype=torch.float16, cache_enabled=False) if device == "cuda" else torch.autocast("cpu", enabled=False)
with torch.inference_mode(), torch.jit.optimized_execution(False), context:
    for step in range(3):
        tick = time.perf_counter()
        raw = model(inputs).float()
        if device == "cuda":
            torch.cuda.synchronize()
        print(f"forward {step} {time.perf_counter() - tick:.2f} s finite={bool(torch.isfinite(raw).all())} shape={tuple(raw.shape)}", flush=True)
print("probe ok", flush=True)
"""


def _probe_member(index: int) -> bool:
    return _run_probe(MODEL_PATHS[index], MODEL_FILES[index], PROBE_TIMEOUT)


def _run_probe(path: str, name: str, timeout: float) -> bool:
    import subprocess
    import sys
    tick = time.perf_counter()
    PROBE_TIMEOUT = timeout
    try:
        done = subprocess.run([sys.executable, "-c", _PROBE_CODE, path, _DEVICE.type],
                              capture_output=True, text=True, timeout=timeout)
        passed = done.returncode == 0 and "probe ok" in done.stdout
        detail = (done.stdout + done.stderr)[-1200:]
    except subprocess.TimeoutExpired as error:
        partial = error.stdout.decode(errors="replace") if isinstance(error.stdout, bytes) else (error.stdout or "")
        passed, detail = False, f"timeout after {PROBE_TIMEOUT:.0f} s; child said: {partial[-800:]}"
    except Exception as error:
        passed, detail = False, f"{type(error).__name__}: {error}"
    print(f"[submission] probe {name}: {'ok' if passed else 'FAILED'} after {time.perf_counter() - tick:.1f} s | "
          + " | ".join(line for line in detail.splitlines() if line.strip()), flush=True)
    return passed


_GRAPHED: dict[int, Any] = {}
_DEVICE_CACHE: dict[Any, dict[str, torch.Tensor]] = {}
_TAPER_CACHE: dict[Any, torch.Tensor] = {}


def _get_device_tensors(device: torch.device, dtype: torch.dtype = torch.float32) -> dict[str, torch.Tensor]:
    key = (device, dtype)
    cached = _DEVICE_CACHE.get(key)
    if cached is None:
        j_upper = torch.zeros((1, 17, 1, 1), dtype=dtype, device=device)
        j_upper[0, _UPPER_JOINTS, 0, 0] = 1.0
        j_lower = torch.zeros((1, 17, 1, 1), dtype=dtype, device=device)
        j_lower[0, _LOWER_JOINTS, 0, 0] = 1.0
        m_6v = torch.zeros((1, 17, 1, 1), dtype=dtype, device=device)
        m_6v[0, [1, 2, 3, 4], 0, 0] = 1.0
        m_v5_base = torch.zeros((1, 17, 1, 1), dtype=dtype, device=device)
        m_v5_base[0, [0, 5, 6, 9, 10], 0, 0] = 1.0
        m_v6_base = torch.zeros((1, 17, 1, 1), dtype=dtype, device=device)
        m_v6_base[0, [11, 12, 13, 14, 15, 16], 0, 0] = 1.0
        m_elbows = torch.zeros((1, 17, 1, 1), dtype=dtype, device=device)
        m_elbows[0, [7, 8], 0, 0] = 1.0
        cached = {
            "flip_channels": torch.as_tensor(_FLIP_CHANNELS, dtype=torch.long, device=device),
            "j_upper": j_upper,
            "j_lower": j_lower,
            "m_6v": m_6v,
            "m_v5_cnn": m_v5_base + m_elbows,
            "m_v6_cnn": m_v6_base,
            "m_v5_vit": m_v5_base,
            "m_v6_vit": m_v6_base + m_elbows,
            "unit_weights_3": torch.as_tensor(_UNIT_WEIGHTS_3, dtype=dtype, device=device),
            "unit_weights_5": torch.as_tensor(_UNIT_WEIGHTS_5, dtype=dtype, device=device),
        }
        _DEVICE_CACHE[key] = cached
    return cached


def _get_taper_2d(side: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    key = (side, device, dtype)
    taper = _TAPER_CACHE.get(key)
    if taper is None:
        margin = 4
        t1d_np = np.ones(side, dtype=np.float32)
        for m in range(margin):
            val = float(m + 1) / float(margin + 1)
            t1d_np[m] = val
            t1d_np[-(m + 1)] = val
        t2d_np = t1d_np[:, None] * t1d_np[None, :]
        taper = torch.as_tensor(t2d_np, dtype=dtype, device=device).view(1, 1, side, side)
        _TAPER_CACHE[key] = taper
    return taper


def _prepare_graphs() -> None:
    _GRAPHED.clear()
    if _DEVICE is None or _DEVICE.type != "cuda" or not GRAPH_MODEL_INDICES:
        return
    capture_batches = (14, 4, 2) if SAHI_CONFIG is not None else (4, 2)
    try:
        for index in GRAPH_MODEL_INDICES:
            if _MODELS[index] is None:
                continue
            graphed = GraphedForward(_MODELS[index], max_shapes=len(capture_batches))
            for batch_n in capture_batches:
                generator = torch.Generator().manual_seed(20261007 + batch_n)
                probe = (torch.rand((batch_n, 3, 256, 256), generator=generator) * 4.0 - 2.0).to(_DEVICE)
                replay = graphed(probe)
                reference = eager_forward(_MODELS[index], probe)
                if float((reference - replay).abs().max()) > 1e-2:
                    raise RuntimeError("graph replay differs from eager forward")
            if graphed.captures != len(capture_batches):
                raise RuntimeError("graph capture count mismatch")
            _GRAPHED[index] = graphed
    except Exception as error:
        _GRAPHED.clear()
        print(f"[submission] CUDA graphs disabled, eager forward for all members: {type(error).__name__}", flush=True)
        return
    print(f"[submission] CUDA graph replay for members {sorted(_GRAPHED)}", flush=True)


def crop_geometry(
    box: Sequence[float], padding: float
) -> tuple[float, float, float, float]:
    x_coord, y_coord, width, height = (float(value) for value in box[:4])
    if not all(
        math.isfinite(value) for value in (x_coord, y_coord, width, height, padding)
    ):
        raise ValueError("box and padding values must be finite")
    if width <= 0.0 or height <= 0.0 or padding <= 0.0:
        raise ValueError("box width, height, and padding must be positive")
    center_x = x_coord + width * 0.5
    center_y = y_coord + height * 0.5
    target_aspect = int(_META["input_width"]) / int(_META["input_height"])
    if width > target_aspect * height:
        height = width / target_aspect
    elif width < target_aspect * height:
        width = height * target_aspect
    width *= padding
    height *= padding
    return center_x - width * 0.5, center_y - height * 0.5, width, height


def _resolve_image_path(sample: dict[str, Any]) -> str:
    image_path = str(sample.get("image_path", ""))
    if image_path and os.path.isfile(image_path):
        return image_path
    file_name = str(sample.get("file_name", ""))
    if image_path:
        candidates = (
            image_path.replace("/images/images/", "/images/"),
            image_path.replace("\\images\\images\\", "\\images\\"),
        )
        for candidate in candidates:
            if os.path.isfile(candidate):
                return candidate
    if file_name:
        candidates = (
            file_name,
            file_name.removeprefix("images/"),
            file_name.removeprefix("images\\"),
        )
        for candidate in candidates:
            candidate_path = os.path.join(HERE, candidate)
            if os.path.isfile(candidate_path):
                return candidate_path
    if image_path:
        return image_path
    raise FileNotFoundError("Could not resolve image path for sample")


def preprocess_geometry(
    image: Image.Image, geometry: tuple[float, float, float, float]
) -> np.ndarray:
    left, top, width, height = geometry
    crop = image.crop((left, top, left + width, top + height)).resize(
        (int(_META["input_width"]), int(_META["input_height"])),
        Image.Resampling.BILINEAR,
    )
    array = np.asarray(crop, dtype=np.float32).transpose(2, 0, 1) / 255.0
    return ((array - _MEAN) / _STD).astype(np.float32)


def preprocess(
    image: Image.Image, box: Sequence[float], padding: float
) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    geometry = crop_geometry(box, padding)
    return preprocess_geometry(image, geometry), geometry


def flip_back_heatmaps(heatmaps: torch.Tensor) -> torch.Tensor:
    channels = _get_device_tensors(heatmaps.device)["flip_channels"]
    return torch.flip(heatmaps, dims=(-1,)).index_select(1, channels)


def align_heatmaps(heatmaps: torch.Tensor, source_padding: float) -> torch.Tensor:
    if heatmaps.ndim != 4:
        raise ValueError("expected [batch,joints,height,width] heatmaps")
    if source_padding == TARGET_PADDING:
        return heatmaps
    ratio = TARGET_PADDING / source_padding
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


def warp_between_geoms(
    maps: torch.Tensor,
    src_geoms: np.ndarray,
    tgt_geoms: Sequence[tuple[float, float, float, float]],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Warp [B, 17, H, W] heatmaps from src_geoms (l,t,w,h) into tgt_geoms (l,t,w,h) with 4-cell border taper."""
    src = torch.as_tensor(src_geoms, dtype=maps.dtype, device=maps.device)
    tgt = torch.as_tensor(tgt_geoms, dtype=maps.dtype, device=maps.device)
    sx = tgt[:, 2] / src[:, 2]
    sy = tgt[:, 3] / src[:, 3]
    tx = (2.0 * (tgt[:, 0] - src[:, 0]) + tgt[:, 2] - src[:, 2]) / src[:, 2]
    ty = (2.0 * (tgt[:, 1] - src[:, 1]) + tgt[:, 3] - src[:, 3]) / src[:, 3]
    theta = maps.new_zeros((maps.shape[0], 2, 3))
    theta[:, 0, 0] = sx
    theta[:, 1, 1] = sy
    theta[:, 0, 2] = tx
    theta[:, 1, 2] = ty
    grid = functional.affine_grid(theta, maps.shape, align_corners=True)
    ones = _get_taper_2d(int(maps.shape[2]), maps.device, maps.dtype).expand(
        maps.shape[0], 1, maps.shape[2], maps.shape[3]
    )
    warped = functional.grid_sample(maps, grid, mode="bilinear", padding_mode="zeros", align_corners=True)
    mask = functional.grid_sample(ones, grid, mode="bilinear", padding_mode="zeros", align_corners=True)
    return warped, mask


def _precompute_sahi_warp(
    src_geoms_bv: np.ndarray,
    tgt_geoms: Sequence[tuple[float, float, float, float]],
    side: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Precompute [B*V, side, side, 2] affine grid and [B, V, 1, side, side] border-taper masks once per image."""
    b_cnt, v_cnt, _ = src_geoms_bv.shape
    src_np = np.asarray(src_geoms_bv, dtype=np.float32).reshape(b_cnt * v_cnt, 4)
    tgt_np = np.repeat(np.asarray(tgt_geoms, dtype=np.float32), v_cnt, axis=0)
    theta_np = np.zeros((b_cnt * v_cnt, 2, 3), dtype=np.float32)
    theta_np[:, 0, 0] = tgt_np[:, 2] / src_np[:, 2]
    theta_np[:, 1, 1] = tgt_np[:, 3] / src_np[:, 3]
    theta_np[:, 0, 2] = (2.0 * (tgt_np[:, 0] - src_np[:, 0]) + tgt_np[:, 2] - src_np[:, 2]) / src_np[:, 2]
    theta_np[:, 1, 2] = (2.0 * (tgt_np[:, 1] - src_np[:, 1]) + tgt_np[:, 3] - src_np[:, 3]) / src_np[:, 3]
    theta = torch.as_tensor(theta_np, dtype=dtype, device=device)
    grid = functional.affine_grid(theta, [b_cnt * v_cnt, 1, side, side], align_corners=True)
    ones = _get_taper_2d(side, device, dtype).expand(b_cnt * v_cnt, 1, side, side)
    masks = functional.grid_sample(
        ones, grid, mode="bilinear", padding_mode="zeros", align_corners=True
    ).view(b_cnt, v_cnt, 1, side, side)
    return grid, masks


def compute_sahi_geometries(
    crop_boxes: Sequence[Sequence[float]],
    indices: Sequence[int],
    rsn_map: torch.Tensor,
    s1_map: torch.Tensor,
    target_geometries: Sequence[tuple[float, float, float, float]],
    side: int,
) -> np.ndarray:
    """Compute [B, 7, 4] square geometries (v0..v6)."""
    rsn_up = (
        rsn_map
        if int(rsn_map.shape[-1]) == side
        else functional.interpolate(rsn_map, size=(side, side), mode="bilinear", align_corners=True)
    )
    idx_rsn = rsn_up.flatten(2).argmax(dim=2).cpu().numpy()
    idx_s1 = s1_map.flatten(2).argmax(dim=2).cpu().numpy()
    geoms_np = np.asarray(target_geometries, dtype=np.float64)
    rsn_x = (idx_rsn % side) / float(side - 1) * geoms_np[:, None, 2] + geoms_np[:, None, 0]
    rsn_y = (idx_rsn // side) / float(side - 1) * geoms_np[:, None, 3] + geoms_np[:, None, 1]
    s1_x = (idx_s1 % side) / float(side - 1) * geoms_np[:, None, 2] + geoms_np[:, None, 0]
    s1_y = (idx_s1 // side) / float(side - 1) * geoms_np[:, None, 3] + geoms_np[:, None, 1]
    batch = len(indices)
    all_geoms = np.zeros((batch, 7, 4), dtype=np.float64)
    for bi, box_idx in enumerate(indices):
        box = crop_boxes[box_idx]
        bx, by, bw, bh = (float(v) for v in box[:4])
        box_cx, box_cy = bx + 0.5 * bw, by + 0.5 * bh
        m_side = max(bw, bh)
        all_geoms[bi, 0] = crop_geometry(box, 1.12)
        all_geoms[bi, 1] = crop_geometry(box, 1.25)
        pose_cx = 0.5 * float(rsn_x[bi, _SCORED_KEYPOINTS].min() + rsn_x[bi, _SCORED_KEYPOINTS].max())
        pose_cy = 0.5 * float(rsn_y[bi, _SCORED_KEYPOINTS].min() + rsn_y[bi, _SCORED_KEYPOINTS].max())
        cx_p = 0.5 * box_cx + 0.5 * pose_cx
        cy_p = 0.5 * box_cy + 0.5 * pose_cy
        side_p = m_side * 1.14
        all_geoms[bi, 2] = (cx_p - 0.5 * side_p, cy_p - 0.5 * side_p, side_p, side_p)
        ux_min, ux_max = float(rsn_x[bi, _UPPER_JOINTS].min()), float(rsn_x[bi, _UPPER_JOINTS].max())
        uy_min, uy_max = float(rsn_y[bi, _UPPER_JOINTS].min()), float(rsn_y[bi, _UPPER_JOINTS].max())
        u_cx = 0.4 * box_cx + 0.6 * (0.5 * (ux_min + ux_max))
        u_cy = 0.5 * (uy_min + uy_max)
        u_span = max(ux_max - ux_min, uy_max - uy_min, 0.55 * m_side)
        u_side = float(np.clip(u_span * 1.45, 0.68 * m_side, 1.05 * m_side))
        all_geoms[bi, 3] = (u_cx - 0.5 * u_side, u_cy - 0.5 * u_side, u_side, u_side)
        lx_min, lx_max = float(rsn_x[bi, _LOWER_JOINTS].min()), float(rsn_x[bi, _LOWER_JOINTS].max())
        ly_min, ly_max = float(rsn_y[bi, _LOWER_JOINTS].min()), float(rsn_y[bi, _LOWER_JOINTS].max())
        l_cx = 0.4 * box_cx + 0.6 * (0.5 * (lx_min + lx_max))
        l_cy = 0.5 * (ly_min + ly_max)
        l_span = max(lx_max - lx_min, ly_max - ly_min, 0.55 * m_side)
        l_side = float(np.clip(l_span * 1.45, 0.72 * m_side, 1.05 * m_side))
        all_geoms[bi, 4] = (l_cx - 0.5 * l_side, l_cy - 0.5 * l_side, l_side, l_side)
        hx_min, hx_max = float(s1_x[bi, _HEAD_SHO_JOINTS].min()), float(s1_x[bi, _HEAD_SHO_JOINTS].max())
        hy_min, hy_max = float(s1_y[bi, _HEAD_SHO_JOINTS].min()), float(s1_y[bi, _HEAD_SHO_JOINTS].max())
        h_cx = 0.35 * box_cx + 0.65 * (0.5 * (hx_min + hx_max))
        h_cy = 0.5 * (hy_min + hy_max)
        h_span = max(hx_max - hx_min, hy_max - hy_min, 0.36 * m_side)
        h_side = float(np.clip(h_span * 1.45, 0.48 * m_side, 0.72 * m_side))
        all_geoms[bi, 5] = (h_cx - 0.5 * h_side, h_cy - 0.5 * h_side, h_side, h_side)
        tx_min, tx_max = float(s1_x[bi, _TORSO_LEG_JOINTS].min()), float(s1_x[bi, _TORSO_LEG_JOINTS].max())
        ty_min, ty_max = float(s1_y[bi, _TORSO_LEG_JOINTS].min()), float(s1_y[bi, _TORSO_LEG_JOINTS].max())
        t_cx = 0.40 * box_cx + 0.60 * (0.5 * (tx_min + tx_max))
        t_cy = 0.5 * (ty_min + ty_max) + 0.03 * bh
        t_span = max(tx_max - tx_min, ty_max - ty_min, 0.68 * m_side)
        t_side = float(np.clip(t_span * 1.25, 0.85 * m_side, 1.06 * m_side))
        all_geoms[bi, 6] = (t_cx - 0.5 * t_side, t_cy - 0.5 * t_side, t_side, t_side)
    return all_geoms


def gated_sahi_blend(
    base_96: torch.Tensor,
    warped_views: dict[int, torch.Tensor],
    mask_u: torch.Tensor | None,
    mask_l: torch.Tensor | None,
    ar_gate: torch.Tensor,
    view_weights: Sequence[float],
    w_sahi: float,
    sim_pow: float,
    peak_pow: float,
) -> torch.Tensor:
    """Blend base 96x96 map with active full-body zoom views (0..2) and gated Upper/Lower Pose-SAHI slices (3, 4)."""
    w0, w1, w2 = (float(v) for v in view_weights)
    fb_num = base_96
    fb_den = 1.0
    if w0 > 0.0 and 0 in warped_views:
        fb_num = fb_num + w0 * warped_views[0]
        fb_den += w0
    if w1 > 0.0 and 1 in warped_views:
        fb_num = fb_num + w1 * warped_views[1]
        fb_den += w1
    if w2 > 0.0 and 2 in warped_views:
        fb_num = fb_num + w2 * warped_views[2]
        fb_den += w2
    fb = fb_num / fb_den if fb_den != 1.0 else fb_num
    if w_sahi <= 0.0 or 3 not in warped_views or 4 not in warped_views or mask_u is None or mask_l is None:
        return fb
    v3 = warped_views[3]
    v4 = warped_views[4]
    fb_pos = functional.relu(fb)
    fb_peak = fb_pos.amax(dim=(-2, -1), keepdim=True).clamp(min=1e-4)
    dev_tensors = _get_device_tensors(fb.device, fb.dtype)
    j_upper = dev_tensors["j_upper"]
    j_lower = dev_tensors["j_lower"]
    u_pos = functional.relu(v3)
    u_peak = u_pos.amax(dim=(-2, -1), keepdim=True).clamp(min=1e-4)
    sim_u = (fb_pos * u_pos).sum(dim=(-2, -1), keepdim=True) / (
        fb_pos.norm(dim=(-2, -1), keepdim=True) * u_pos.norm(dim=(-2, -1), keepdim=True) + 1e-6
    )
    gate_u = (sim_u.clamp(min=0.0) ** sim_pow) * ((u_peak / fb_peak).clamp(0.0, 1.3) ** peak_pow) * ar_gate
    eff_u = float(w_sahi) * mask_u * j_upper * gate_u
    l_pos = functional.relu(v4)
    l_peak = l_pos.amax(dim=(-2, -1), keepdim=True).clamp(min=1e-4)
    sim_l = (fb_pos * l_pos).sum(dim=(-2, -1), keepdim=True) / (
        fb_pos.norm(dim=(-2, -1), keepdim=True) * l_pos.norm(dim=(-2, -1), keepdim=True) + 1e-6
    )
    gate_l = (sim_l.clamp(min=0.0) ** sim_pow) * ((l_peak / fb_peak).clamp(0.0, 1.3) ** peak_pow) * ar_gate
    eff_l = float(w_sahi) * mask_l * j_lower * gate_l
    return (fb + eff_u * v3 + eff_l * v4) / (1.0 + eff_u + eff_l)


def _enhance_unit_view(
    u_base: torch.Tensor,
    v_raw: torch.Tensor,
    mask_v: torch.Tensor,
    ar_gate: torch.Tensor,
    w_s: float,
    sim_pow: float,
    peak_pow: float,
) -> torch.Tensor:
    if w_s <= 0.0:
        return u_base
    u_pos = functional.relu(u_base)
    u_pk = u_pos.amax(dim=(-2, -1), keepdim=True).clamp(min=1e-4)
    v = unit_peak(v_raw)
    v_pos = functional.relu(v)
    v_pk = v_pos.amax(dim=(-2, -1), keepdim=True).clamp(min=1e-4)
    sim = (u_pos * v_pos).sum(dim=(-2, -1), keepdim=True) / (
        u_pos.norm(dim=(-2, -1), keepdim=True) * v_pos.norm(dim=(-2, -1), keepdim=True) + 1e-6
    )
    g = float(w_s) * mask_v * (sim.clamp(min=0.0) ** sim_pow) * ((v_pk / u_pk).clamp(0.0, 1.3) ** peak_pow) * ar_gate
    return unit_peak((u_base + g * v) / (1.0 + g))


def _view_rounding_delta_from_geom(coords_xy: np.ndarray, geom_4: Sequence[float]) -> np.ndarray:
    left, top, width, height = (float(v) for v in geom_4)
    x0 = float(round(left))
    y0 = float(round(top))
    x1 = float(round(left + width))
    y1 = float(round(top + height))
    w_act = x1 - x0
    h_act = y1 - y0
    u = (coords_xy[:, 0] - left) / width
    v = (coords_xy[:, 1] - top) / height
    dx = (x0 - left) + u * (w_act - width)
    dy = (y0 - top) + v * (h_act - height)
    return np.stack((dx, dy), axis=-1)


_RUNTIME: dict[str, Any] = {
    "load_start": None,
    "start": None,
    "images": 0,
    "sahi_tier": 0,
    "sahi_disabled": False,
    "reduced": False,
    "roll_start": None,
    "roll_images": 0,
    "last_roll_seconds": 0.0,
}
TIMING_EVERY = 10
_MEMBERS: dict[str, Any] = {"active": None, "weights": None, "seconds": None, "samples": 0, "window_start": None, "window_images": 0}


def active_members() -> list[int]:
    if _MEMBERS["active"] is None:
        _MEMBERS["weights"] = tuple(MODEL_WEIGHTS)
        _MEMBERS["active"] = [index for index, weight in enumerate(MODEL_WEIGHTS) if weight > 0]
        _MEMBERS["seconds"] = [0.0] * len(MODEL_FILES)
    return _MEMBERS["active"]


def _member_report() -> str:
    active_members()
    samples = max(int(_MEMBERS["samples"]), 1)
    return ", ".join(f"{MODEL_FILES[i]} {1000.0 * _MEMBERS['seconds'][i] / samples:.1f} ms"
                     + ("" if i in _MEMBERS["active"] else " (dropped)") for i in range(len(MODEL_FILES))
                     if _MODELS is None or _MODELS[i] is not None)


def active_member_views() -> tuple[tuple[tuple[float, ...], tuple[float, ...]], ...]:
    if _RUNTIME["reduced"]:
        return tuple(((TARGET_PADDING,), (1.0,)) for _ in MODEL_FILES)
    if _RUNTIME.get("sahi_tier", 0) >= 2:
        return tuple(
            ((TARGET_PADDING,), (1.0,)) if idx >= 3 else (tuple(paddings), tuple(weights))
            for idx, (paddings, weights) in enumerate(zip(MEMBER_PADDINGS, MEMBER_SCALE_WEIGHTS, strict=True))
        )
    return tuple((tuple(paddings), tuple(weights)) for paddings, weights in zip(MEMBER_PADDINGS, MEMBER_SCALE_WEIGHTS, strict=True))


def _runtime_tick() -> None:
    now = time.perf_counter()
    _RUNTIME["images"] += 1
    if _RUNTIME["roll_start"] is None:
        _RUNTIME["roll_start"] = _RUNTIME["start"] or now
    _RUNTIME["roll_images"] += 1
    if _RUNTIME["roll_images"] >= ROLLING_WINDOW_IMAGES:
        _RUNTIME["last_roll_seconds"] = (now - _RUNTIME["roll_start"]) / _RUNTIME["roll_images"]
        _RUNTIME["roll_start"] = now
        _RUNTIME["roll_images"] = 0
    wall_elapsed = now - (_RUNTIME["load_start"] or _RUNTIME["start"] or now)
    remaining = max(0, EXPECTED_TOTAL_IMAGES - _RUNTIME["images"])
    if _RUNTIME["images"] in (100, 300, 1000) or _RUNTIME["images"] % 2500 == 0:
        roll = _RUNTIME["last_roll_seconds"]
        print(f"[submission] {(now - _RUNTIME['start']) / _RUNTIME['images']:.3f} s/image "
              f"(roll={roll:.3f}s, wall={wall_elapsed:.1f}s, tier={_RUNTIME.get('sahi_tier', 0)}) after {_RUNTIME['images']} images; "
              f"sampled member forward: {_member_report()}", flush=True)
    if wall_elapsed >= WALL_EMERGENCY_SINGLE_SECONDS:
        _RUNTIME["sahi_disabled"] = True
        _RUNTIME["sahi_tier"] = 3
        if not _RUNTIME["reduced"]:
            _RUNTIME["reduced"] = True
            _MEMBERS["window_start"], _MEMBERS["window_images"] = now, 0
            print(f"[submission] emergency wall-clock guard at {wall_elapsed:.1f}s (image {_RUNTIME['images']}): "
                  f"centre padding only from here", flush=True)
        active = active_members()
        if len(active) > 1:
            fastest = min(active, key=lambda idx: _MEMBERS["seconds"][idx] if _MEMBERS["samples"] > 0 else idx)
            active[:] = [fastest]
            print(f"[submission] emergency wall-clock guard at {wall_elapsed:.1f}s (image {_RUNTIME['images']}): "
                  f"keeping fastest member {MODEL_FILES[fastest]} ({_member_report()})", flush=True)
        return
    if SAHI_CONFIG is not None and not _RUNTIME["reduced"] and _RUNTIME["images"] >= 160 and _RUNTIME["roll_images"] == 0:
        roll_seconds = _RUNTIME["last_roll_seconds"] if _RUNTIME["last_roll_seconds"] > 0.0 else (now - _RUNTIME["start"]) / _RUNTIME["images"]
        projected_wall = wall_elapsed + remaining * roll_seconds
        sahi_budget = min(float(SAHI_CONFIG.get("sahi_wall_budget_seconds", WALL_BUDGET_DISABLE_SAHI_SECONDS)), WALL_BUDGET_DISABLE_SAHI_SECONDS)
        prev_tier = int(_RUNTIME.get("sahi_tier", 0))
        if wall_elapsed >= sahi_budget:
            new_tier = 3
        elif projected_wall >= sahi_budget:
            new_tier = min(3, prev_tier + 1)
        elif projected_wall >= WALL_BUDGET_TIER2_SECONDS:
            new_tier = min(2, max(prev_tier, prev_tier + 1))
        elif projected_wall >= WALL_BUDGET_TIER1_SECONDS:
            new_tier = max(prev_tier, 1)
        elif projected_wall < WALL_BUDGET_TIER1_SECONDS - 350.0 and prev_tier > 0:
            new_tier = prev_tier - 1
        else:
            new_tier = prev_tier
        if new_tier != prev_tier:
            _RUNTIME["sahi_tier"] = new_tier
            _RUNTIME["sahi_disabled"] = wall_elapsed >= sahi_budget
            if new_tier > prev_tier:
                print(f"[submission] proportional SAHI controller: proj_wall={projected_wall:.0f}s after {_RUNTIME['images']} images "
                      f"-> tier {new_tier}", flush=True)
    if RUNTIME_GUARD is not None and SAHI_CONFIG is not None:
        _deadline_policy_tick(now, wall_elapsed, remaining)
        return
    _member_drop_tick(now, wall_elapsed, remaining)
    if RUNTIME_GUARD is None or _RUNTIME["reduced"]:
        return
    if SAHI_CONFIG is not None:
        if wall_elapsed < 6800.0:
            return
        roll_seconds = _RUNTIME["last_roll_seconds"] if _RUNTIME["last_roll_seconds"] > 0.0 else (now - _RUNTIME["start"]) / _RUNTIME["images"]
        projected_wall = wall_elapsed + remaining * roll_seconds
        if wall_elapsed >= WALL_BUDGET_REDUCE_SECONDS or projected_wall > 7100.0:
            _RUNTIME["sahi_disabled"] = True
            _RUNTIME["sahi_tier"] = 3
            _RUNTIME["reduced"] = True
            print(f"[submission] runtime guard: roll={roll_seconds:.3f}s proj_wall={projected_wall:.0f}s "
                  f"at wall={wall_elapsed:.0f}s (image {_RUNTIME['images']}) exceeds {WALL_BUDGET_REDUCE_SECONDS:.0f}s; "
                  f"centre padding only from here", flush=True)
            print(f"[submission] sampled member forward: {_member_report()}", flush=True)
            _MEMBERS["window_start"], _MEMBERS["window_images"] = now, 0
        return
    emergency = MEMBER_DROP_GUARD is not None and _RUNTIME["images"] >= 50
    if _RUNTIME["images"] < 300 and not emergency:
        return
    mean_seconds = (now - _RUNTIME["start"]) / _RUNTIME["images"]
    roll_seconds = _RUNTIME["last_roll_seconds"] if _RUNTIME["last_roll_seconds"] > 0.0 else mean_seconds
    pace = max(mean_seconds, roll_seconds) if _RUNTIME["images"] >= 300 else mean_seconds
    projected_wall = wall_elapsed + remaining * pace
    budget = RUNTIME_GUARD[0] * (1.0 if _RUNTIME["images"] >= RUNTIME_GUARD[1] else 4.0)
    if mean_seconds > budget or (_RUNTIME["images"] >= 300 and (projected_wall > WALL_BUDGET_REDUCE_SECONDS or wall_elapsed >= WALL_BUDGET_REDUCE_SECONDS)):
        _RUNTIME["sahi_disabled"] = True
        _RUNTIME["sahi_tier"] = 3
        _RUNTIME["reduced"] = True
        print(f"[submission] runtime guard: mean={mean_seconds:.3f}s roll={roll_seconds:.3f}s proj_wall={projected_wall:.0f}s "
              f"after {_RUNTIME['images']} images exceeds {RUNTIME_GUARD[0]:.3f}s / {WALL_BUDGET_REDUCE_SECONDS:.0f}s; "
              f"centre padding only from here", flush=True)
        print(f"[submission] sampled member forward: {_member_report()}", flush=True)
        _MEMBERS["window_start"], _MEMBERS["window_images"] = now, 0


def _member_drop_tick(now: float | None = None, wall_elapsed: float | None = None, remaining: int | None = None) -> None:
    if MEMBER_DROP_GUARD is None or not _RUNTIME["reduced"] or _MEMBERS["window_start"] is None:
        return
    if now is None:
        now = time.perf_counter()
    if wall_elapsed is None:
        wall_elapsed = now - (_RUNTIME["load_start"] or _RUNTIME["start"] or now)
    if remaining is None:
        remaining = max(0, EXPECTED_TOTAL_IMAGES - _RUNTIME["images"])
    _MEMBERS["window_images"] += 1
    if _MEMBERS["window_images"] < MEMBER_DROP_GUARD[1]:
        return
    window_mean = (now - _MEMBERS["window_start"]) / _MEMBERS["window_images"]
    projected_wall = wall_elapsed + remaining * window_mean
    active = active_members()
    if SAHI_CONFIG is not None:
        should_drop = wall_elapsed >= 7050.0 and (projected_wall > WALL_BUDGET_DROP_SECONDS or wall_elapsed >= WALL_BUDGET_DROP_SECONDS)
    else:
        should_drop = window_mean > MEMBER_DROP_GUARD[0] or projected_wall > WALL_BUDGET_DROP_SECONDS or wall_elapsed >= WALL_BUDGET_DROP_SECONDS
    if should_drop and len(active) > 1:
        slowest = max(active, key=lambda index: _MEMBERS["seconds"][index] if _MEMBERS["samples"] > 0 else index)
        active.remove(slowest)
        print(f"[submission] member-drop guard: {window_mean:.3f} s/image (proj_wall={projected_wall:.0f}s) "
              f"over {_MEMBERS['window_images']} reduced images exceeds {MEMBER_DROP_GUARD[0]:.3f}s / "
              f"{WALL_BUDGET_DROP_SECONDS:.0f}s; dropping {MODEL_FILES[slowest]} ({_member_report()})", flush=True)
    _MEMBERS["window_start"], _MEMBERS["window_images"] = now, 0


def model_aligned_mean(views: torch.Tensor, paddings: Sequence[float], scale_weights: Sequence[float]) -> torch.Tensor:
    if views.ndim != 5 or views.shape[1] != len(paddings):
        raise ValueError("expected [batch,paddings,joints,height,width] model views")
    total = float(sum(scale_weights))
    combined = None
    for index, (padding, weight) in enumerate(zip(paddings, scale_weights, strict=True)):
        contribution = align_heatmaps(views[:, index], padding) * (float(weight) / total)
        combined = contribution if combined is None else combined + contribution
    assert combined is not None
    return combined


def unit_peak(maps: torch.Tensor) -> torch.Tensor:
    peak = maps.amax(dim=(-2, -1), keepdim=True)
    return maps / torch.where(peak > 1e-8, peak, torch.ones_like(peak))


def model_scores(maxima: np.ndarray) -> np.ndarray:
    clipped = np.clip(maxima[:, _SCORED_KEYPOINTS], 0.0, 1.0)
    if SCORE_TOPK:
        clipped = np.sort(clipped, axis=1)[:, -SCORE_TOPK:]
    if SCORE_SOFTVIS_TAU:
        keep = (clipped >= SCORE_SOFTVIS_TAU).astype(np.float64)
        keep = np.where(keep.sum(axis=1, keepdims=True) > 0, keep, 1.0)
        return np.maximum(0.01, (clipped * keep).sum(axis=1) / keep.sum(axis=1))
    return np.maximum(0.01, clipped.mean(axis=1))


def decode_heatmaps(
    heatmaps: np.ndarray,
    geometries: Sequence[tuple[float, float, float, float]],
) -> tuple[np.ndarray, np.ndarray]:
    batch, joints, height, width = heatmaps.shape
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
            log_plane = np.log(np.maximum(plane, 1e-10))
            center = float(log_plane[y_coord, x_coord])
            left_value = float(log_plane[y_coord, x_coord - 1])
            right_value = float(log_plane[y_coord, x_coord + 1])
            top_value = float(log_plane[y_coord - 1, x_coord])
            bottom_value = float(log_plane[y_coord + 1, x_coord])
            denom_x = left_value - 2.0 * center + right_value
            denom_y = top_value - 2.0 * center + bottom_value
            if abs(denom_x) > 1e-8:
                coords[batch_index, joint_index, 0] += float(
                    np.clip(
                        0.5 * (left_value - right_value) / denom_x,
                        -0.5,
                        0.5,
                    )
                )
            if abs(denom_y) > 1e-8:
                coords[batch_index, joint_index, 1] += float(
                    np.clip(
                        0.5 * (top_value - bottom_value) / denom_y,
                        -0.5,
                        0.5,
                    )
                )
        crop_left, crop_top, crop_width, crop_height = geometries[batch_index]
        coords[batch_index, :, 0] = (
            coords[batch_index, :, 0] / (width - 1.0) * crop_width + crop_left
        )
        coords[batch_index, :, 1] = (
            coords[batch_index, :, 1] / (height - 1.0) * crop_height + crop_top
        )
    return coords, maxima


def decode_fused(
    heatmaps: np.ndarray,
    geometries: Sequence[tuple[float, float, float, float]],
) -> tuple[np.ndarray, np.ndarray]:
    if DECODER == "quadratic":
        return decode_heatmaps(heatmaps, geometries)
    if DECODER != "dark":
        raise RuntimeError(f"unknown decoder {DECODER}")
    batch, joints, height, width = heatmaps.shape
    flattened = heatmaps.reshape(batch, joints, -1)
    indices = np.argmax(flattened, axis=2)
    maxima = np.max(flattened, axis=2)
    coords = np.stack((indices % width, indices // width), axis=2).astype(np.float32)
    offsets = dark_offsets(heatmaps, coords, dark_kernel(width))
    if HEAD_DARK_KERNEL:
        offsets[:, :3] = dark_offsets(np.ascontiguousarray(heatmaps[:, :3]), coords[:, :3], HEAD_DARK_KERNEL)
    coords = coords + offsets.astype(np.float32)
    geometry = np.asarray(geometries, dtype=np.float64)
    coords[:, :, 0] = coords[:, :, 0] / (width - 1.0) * geometry[:, None, 2] + geometry[:, None, 0]
    coords[:, :, 1] = coords[:, :, 1] / (height - 1.0) * geometry[:, None, 3] + geometry[:, None, 1]
    return coords, maxima


def squat_box(box: Sequence[float]) -> list[float]:
    values = [float(value) for value in box[:4]]
    if SQUAT_BOX is None or not all(math.isfinite(value) for value in values) or values[2] <= 0.0 or values[3] <= 0.0:
        return values
    if values[3] / values[2] < SQUAT_BOX[0]:
        values[3] = max(values[3], SQUAT_BOX[1] * values[2])
    return values


def _sahi_chunked_forward(model: Any, flat_crops: torch.Tensor, model_index: int | None = None) -> torch.Tensor:
    """Run flipped-pair inference on [N, 3, 256, 256]; uses full-batch or 2-crop CUDA Graph chunks when model_index is graphed."""
    outputs = []
    total = int(flat_crops.shape[0])
    if model_index is not None and model_index in _GRAPHED and flat_crops.device.type == "cuda":
        step = total if ((total * 2, 3, 256, 256), flat_crops.dtype) in _GRAPHED[model_index].graphs else 2
    else:
        step = SAHI_MAX_PAIRS_PER_CALL
    for start in range(0, total, step):
        sub = flat_crops[start : start + step]
        paired = torch.cat((sub, torch.flip(sub, dims=(-1,))), dim=0)
        raw = accelerated_forward(model, paired, model_index)
        orig, flipped_raw = raw.split(sub.shape[0], dim=0)
        flipped_back = flip_back_heatmaps(flipped_raw)
        outputs.append(0.5 * (orig + flipped_back))
    return torch.cat(outputs, dim=0)


def _predict_image_core(sample: dict[str, Any], tick: bool = True) -> dict[str, Any]:
    boxes = sample["boxes"]
    if not boxes:
        return {"keypoints": [], "scores": []}
    crop_boxes = [squat_box(box) for box in boxes] if SQUAT_BOX is not None else boxes
    if _MODELS is None:
        load_model()
    if _RUNTIME["start"] is None:
        _RUNTIME["start"] = time.perf_counter()
        if _RUNTIME["load_start"] is None:
            _RUNTIME["load_start"] = _RUNTIME["start"]
        if _RUNTIME["roll_start"] is None:
            _RUNTIME["roll_start"] = _RUNTIME["start"]
    with Image.open(_resolve_image_path(sample)) as source:
        image = source.convert("RGB")
    keypoints: list[Any] = [None] * len(boxes)
    scores: list[float] = [0.0] * len(boxes)
    usable: list[int] = []
    for index, box in enumerate(boxes):
        try:
            crop_geometry(crop_boxes[index], TARGET_PADDING)
        except (ValueError, TypeError):
            x_coord, y_coord, width, height = (float(value) for value in box[:4])
            keypoints[index] = [
                [x_coord + width * 0.5, y_coord + height * 0.5, 1.0]
                for _ in range(17)
            ]
            scores[index] = 0.01
        else:
            usable.append(index)
    active = list(active_members())
    total_weight = float(sum(_MEMBERS["weights"][index] for index in active))
    member_views = active_member_views()
    timed = _RUNTIME["images"] >= 5 and _RUNTIME["images"] % TIMING_EVERY == 0
    distinct_paddings = tuple(sorted({p for idx in active for p in member_views[idx][0]}))
    sahi_tier = int(_RUNTIME.get("sahi_tier", 0))
    if sahi_tier == 2:
        sahi_tier = 3
    sahi_active = (
        SAHI_CONFIG is not None
        and not _RUNTIME["sahi_disabled"]
        and sahi_tier < 3
        and not _RUNTIME["reduced"]
    )
    for start in range(0, len(usable), BATCH_SIZE):
        indices = usable[start : start + BATCH_SIZE]
        target_geometries = [crop_geometry(crop_boxes[index], TARGET_PADDING) for index in indices]
        crops = {(index, padding): preprocess(image, crop_boxes[index], padding)[0] for index in indices for padding in distinct_paddings}
        member_inputs: dict[tuple[float, ...], torch.Tensor] = {}
        for idx in active:
            paddings, _ = member_views[idx]
            if paddings not in member_inputs:
                flat = torch.from_numpy(np.stack([crops[(index, padding)] for index in indices for padding in paddings])).to(_DEVICE)
                member_inputs[paddings] = torch.cat((flat, torch.flip(flat, dims=(-1,))), dim=0)
        fused = None
        with torch.inference_mode():
            member_maps = []
            member_maxima = []
            for model_index in active:
                model = _MODELS[model_index]
                paddings, scale_weights = member_views[model_index]
                paired = member_inputs[paddings]
                if timed:
                    if _DEVICE is not None and _DEVICE.type == "cuda":
                        torch.cuda.synchronize()
                    member_tick = time.perf_counter()
                raw = accelerated_forward(model, paired, model_index)
                if timed:
                    if _DEVICE is not None and _DEVICE.type == "cuda":
                        torch.cuda.synchronize()
                    _MEMBERS["seconds"][model_index] += time.perf_counter() - member_tick
                original, flipped_raw = raw.split(paired.shape[0] // 2, dim=0)
                flipped_back = flip_back_heatmaps(flipped_raw)
                views = (0.5 * (original + flipped_back)).reshape(len(indices), len(paddings), 17, *raw.shape[-2:])
                aligned = model_aligned_mean(views, paddings, scale_weights)
                member_maxima.append(aligned.flatten(2).amax(-1))
                member_maps.append(aligned)
            side = max(int(maps.shape[-1]) for maps in member_maps)
            box_array = np.asarray([[float(v) for v in boxes[index][:4]] for index in indices], dtype=np.float64)
            rsn_area = float(sample.get("_rsn_area") or (box_array[:, 2] * box_array[:, 3]).max())
            rsn_scale = RSN_LARGE_SCALE if (RSN_LARGE_SCALE is not None and rsn_area > RSN_LARGE_AREA) else 1.0
            geometry_array = np.asarray(target_geometries, dtype=np.float64)
            base_96_maps = [
                m if int(m.shape[-1]) == side else functional.interpolate(m, size=(side, side), mode="bilinear", align_corners=True)
                for m in member_maps
            ]
            s1_4m = None
            s1_w_sum = 0.0
            s1_weights = _STAGE1_WEIGHTS_3 if len(MODEL_FILES) == 3 else _STAGE1_WEIGHTS_4
            for pos_idx, model_index in enumerate(active):
                w_s1 = (s1_weights[model_index] if model_index < len(s1_weights) else 0.0) * (rsn_scale if model_index == 0 else 1.0)
                if w_s1 > 0.0:
                    term = unit_peak(base_96_maps[pos_idx]) * w_s1
                    s1_4m = term if s1_4m is None else s1_4m + term
                    s1_w_sum += w_s1
            if s1_4m is None:
                s1_4m = unit_peak(base_96_maps[0])
            else:
                s1_4m = s1_4m / s1_w_sum
            s1_cpu = s1_4m.cpu()
            ratios = secondary_peak_ratio(s1_cpu, box_array, geometry_array) if (AMBIGUITY_BETA or TORSO_GAMMA) else None
            person_ambiguity = ratios[:, list(AMBIGUITY_JOINTS)].mean(axis=1) if AMBIGUITY_BETA else None
            person_torso = ratios[:, list(TORSO_JOINTS)].max(axis=1) if TORSO_GAMMA else None
            all_sahi_geoms = None
            ar_gate_np = np.clip((box_array[:, 3] / np.maximum(box_array[:, 2], 1.0) - 1.35) / 1.0, 0.15, 1.0)
            routed_unit_maps: list[torch.Tensor] = []
            consensus_unit_maps: list[torch.Tensor] = []
            sahi_ran = False
            if sahi_active:
                hw_vals = box_array[:, 3] / np.maximum(box_array[:, 2], 1.0)
                amb_max = float(person_ambiguity.max()) if person_ambiguity is not None else 0.0
                tor_max = float(person_torso.max()) if person_torso is not None else 0.0
                hw_max = float(hw_vals.max())
                cpu_smoke_mode = _DEVICE is not None and _DEVICE.type == "cpu" and _RUNTIME["images"] < 4
                need_g1 = cpu_smoke_mode or (amb_max >= 0.030) or (tor_max >= 0.012) or (hw_max >= 2.65)
                need_g4 = cpu_smoke_mode or (amb_max >= 0.110) or (tor_max >= 0.060) or (hw_max >= 3.10)
                need_sahi = (sahi_tier == 0 and SAHI_TIER0_ALL_PERSONS) or need_g1
                if need_sahi:
                    view_weights = tuple(float(v) for v in SAHI_CONFIG["view_weights"])
                    member_sahi_weights = tuple(float(v) for v in SAHI_CONFIG["member_sahi_weights"])
                    sim_pow = float(SAHI_CONFIG["sim_pow"])
                    peak_pow = float(SAHI_CONFIG["peak_pow"])
                    ar_center = float(SAHI_CONFIG["ar_center"])
                    ar_min, ar_max = (float(v) for v in SAHI_CONFIG["ar_clamp"])
                    ar_gate_np = np.clip((hw_vals - ar_center) / 1.0, ar_min, ar_max)
                    all_sahi_geoms = compute_sahi_geometries(crop_boxes, indices, base_96_maps[0], s1_4m, target_geometries, side)
                    active_v_indices = [vi for vi in range(3) if view_weights[vi] > 0.0]
                    if any(member_sahi_weights[m_idx] > 0.0 for m_idx in active):
                        active_v_indices.extend([3, 4, 5, 6])
                    if active_v_indices:
                        sahi_ran = True
                        sahi_crops_np = np.stack([
                            preprocess_geometry(image, tuple(all_sahi_geoms[bi, vi]))
                            for bi in range(len(indices))
                            for vi in active_v_indices
                        ])
                        sahi_crops_tensor = torch.from_numpy(sahi_crops_np).to(_DEVICE)
                        v_cnt = len(active_v_indices)
                        has_2slice = v_cnt >= 2 and active_v_indices[-2:] == [5, 6]
                        sahi_crops_2slice = (
                            sahi_crops_tensor.view(len(indices), v_cnt, 3, 256, 256)[:, -2:].reshape(len(indices) * 2, 3, 256, 256)
                            if has_2slice
                            else None
                        )
                        hw_ratio = torch.as_tensor(hw_vals, dtype=torch.float32, device=_DEVICE).view(len(indices), 1, 1, 1)
                        ar_gate = ((hw_ratio - ar_center) / 1.0).clamp(ar_min, ar_max)
                        sahi_warp_grid, sahi_warp_masks = _precompute_sahi_warp(
                            all_sahi_geoms[:, active_v_indices],
                            target_geometries,
                            side,
                            _DEVICE,
                            base_96_maps[0].dtype,
                        )
                        sahi_warp_grid_2slice = (
                            sahi_warp_grid.view(len(indices), v_cnt, side, side, 2)[:, -2:].reshape(len(indices) * 2, side, side, 2)
                            if has_2slice
                            else None
                        )
                        masks_by_vi = {vi: sahi_warp_masks[:, local_vi] for local_vi, vi in enumerate(active_v_indices)}
                        dev_tensors = _get_device_tensors(_DEVICE, base_96_maps[0].dtype)
                        if len(MODEL_FILES) == 3:
                            max_sahi_member = 2 if sahi_tier == 0 else (1 if sahi_tier == 1 else 0)
                        else:
                            max_sahi_member = 4 if sahi_tier == 0 else (2 if sahi_tier == 1 else 1)
                        for pos_idx, model_index in enumerate(active):
                            base_m = base_96_maps[pos_idx]
                            is_cnn_member = (model_index == 0) if len(MODEL_FILES) == 3 else (model_index < 3)
                            w_sahi_m = member_sahi_weights[model_index] if model_index <= max_sahi_member else 0.0
                            if not is_cnn_member and not need_g1:
                                w_sahi_m = 0.0
                            if w_sahi_m <= 0.0:
                                u_base = unit_peak(base_m)
                                routed_unit_maps.append(u_base)
                                consensus_unit_maps.append(u_base)
                                continue
                            model = _MODELS[model_index]
                            use_2slice_only = (not is_cnn_member) and has_2slice and (sahi_tier >= 1 or not need_g4)
                            if timed:
                                if _DEVICE is not None and _DEVICE.type == "cuda":
                                    torch.cuda.synchronize()
                                member_tick = time.perf_counter()
                            if use_2slice_only:
                                avg_sahi_2s = _sahi_chunked_forward(model, sahi_crops_2slice, model_index)
                                if timed:
                                    if _DEVICE is not None and _DEVICE.type == "cuda":
                                        torch.cuda.synchronize()
                                    _MEMBERS["seconds"][model_index] += time.perf_counter() - member_tick
                                if int(avg_sahi_2s.shape[-1]) != side:
                                    avg_sahi_2s = functional.interpolate(avg_sahi_2s, size=(side, side), mode="bilinear", align_corners=True)
                                warped_2s = functional.grid_sample(
                                    avg_sahi_2s, sahi_warp_grid_2slice, mode="bilinear", padding_mode="zeros", align_corners=True
                                ).view(len(indices), 2, 17, side, side)
                                warped_views = {5: warped_2s[:, 0], 6: warped_2s[:, 1]}
                                u_6v = unit_peak(base_m)
                            else:
                                avg_sahi = _sahi_chunked_forward(model, sahi_crops_tensor, model_index)
                                if timed:
                                    if _DEVICE is not None and _DEVICE.type == "cuda":
                                        torch.cuda.synchronize()
                                    _MEMBERS["seconds"][model_index] += time.perf_counter() - member_tick
                                if int(avg_sahi.shape[-1]) != side:
                                    avg_sahi = functional.interpolate(avg_sahi, size=(side, side), mode="bilinear", align_corners=True)
                                warped_all = functional.grid_sample(
                                    avg_sahi, sahi_warp_grid, mode="bilinear", padding_mode="zeros", align_corners=True
                                ).view(len(indices), len(active_v_indices), 17, side, side)
                                warped_views = {vi: warped_all[:, local_vi] for local_vi, vi in enumerate(active_v_indices)}
                                u_6v = unit_peak(
                                    gated_sahi_blend(
                                        base_m,
                                        warped_views,
                                        masks_by_vi.get(3),
                                        masks_by_vi.get(4),
                                        ar_gate,
                                        view_weights,
                                        w_sahi_m,
                                        sim_pow,
                                        peak_pow,
                                    )
                                )
                            w_v5 = _SAHI_V5_WEIGHTS_5[model_index] if model_index < len(_SAHI_V5_WEIGHTS_5) else 0.22
                            w_v6 = _SAHI_V6_WEIGHTS_5[model_index] if model_index < len(_SAHI_V6_WEIGHTS_5) else 0.25
                            u_v5 = (
                                _enhance_unit_view(u_6v, warped_views[5], masks_by_vi[5], ar_gate, w_v5, sim_pow, peak_pow)
                                if (5 in warped_views and 5 in masks_by_vi)
                                else u_6v
                            )
                            u_v6 = (
                                _enhance_unit_view(u_6v, warped_views[6], masks_by_vi[6], ar_gate, w_v6, sim_pow, peak_pow)
                                if (6 in warped_views and 6 in masks_by_vi)
                                else u_6v
                            )
                            m_v5 = dev_tensors["m_v5_cnn"] if is_cnn_member else dev_tensors["m_v5_vit"]
                            m_v6 = dev_tensors["m_v6_cnn"] if is_cnn_member else dev_tensors["m_v6_vit"]
                            u_routed = u_6v * dev_tensors["m_6v"] + u_v5 * m_v5 + u_v6 * m_v6
                            routed_unit_maps.append(u_routed)
                            consensus_unit_maps.append(u_v6)
            if not routed_unit_maps:
                for base_m in base_96_maps:
                    u_base = unit_peak(base_m)
                    routed_unit_maps.append(u_base)
                    consensus_unit_maps.append(u_base)
            if SAHI_CONFIG is not None and len(MODEL_FILES) in (3, 5):
                w_key = "unit_weights_3" if len(MODEL_FILES) == 3 else "unit_weights_5"
                unit_w_all = _get_device_tensors(routed_unit_maps[0].device, routed_unit_maps[0].dtype)[w_key]
                active_unit_w = unit_w_all[:, active] if len(active) < len(MODEL_FILES) else unit_w_all
                if rsn_scale != 1.0 and 0 in active:
                    active_unit_w = active_unit_w.clone()
                    active_unit_w[:, active.index(0)] *= rsn_scale
                active_unit_w = active_unit_w / active_unit_w.sum(dim=1, keepdim=True).clamp(min=1e-8)
                for pos_idx, u_map in enumerate(routed_unit_maps):
                    w_vec = active_unit_w[:, pos_idx].view(1, 17, 1, 1)
                    contribution = u_map * w_vec
                    fused = contribution if fused is None else fused + contribution
            else:
                for u_map, weight in zip(routed_unit_maps, [_MEMBERS["weights"][index] for index in active], strict=True):
                    contribution = u_map * (float(weight) / total_weight)
                    fused = contribution if fused is None else fused + contribution
            assert fused is not None
            fused_dev = fused.float()
            per_model_scores = [model_scores(maxima) for maxima in torch.stack(member_maxima).cpu().numpy()]
            stds_075 = expected_oks_stds(box_array, geometry_array, int(fused_dev.shape[-1]), *EXPECTED_OKS) if EXPECTED_OKS is not None else None
            c_members_list = []
            if SAHI_CONFIG is not None and len(consensus_unit_maps) >= 2:
                all_u = torch.cat([fused_dev, *(m.float() for m in consensus_unit_maps)], dim=0)
                if stds_075 is not None:
                    stds_fused = stds_075.copy()
                    stds_fused[:, [13, 14]] = np.hypot(stds_fused[:, [13, 14]], _KNEE_EXTRA_BLUR_STD)
                    stds_fused[:, [15, 16]] = np.hypot(stds_fused[:, [15, 16]], _ANKLE_EXTRA_BLUR_STD)
                    all_stds = np.concatenate([stds_fused] + [stds_075] * len(consensus_unit_maps), axis=0)
                    with torch.backends.cudnn.flags(enabled=True, allow_tf32=False):
                        all_u = expected_oks_blur(all_u, all_stds)
                all_c, _ = decode_fused(all_u.cpu().numpy(), target_geometries * (1 + len(consensus_unit_maps)))
                n_b = len(indices)
                coords = all_c[:n_b]
                c_members_list = list(all_c[n_b:].reshape(len(consensus_unit_maps), n_b, 17, 2))
            else:
                if stds_075 is not None:
                    stds_fused = stds_075.copy()
                    if SAHI_CONFIG is not None:
                        stds_fused[:, [13, 14]] = np.hypot(stds_fused[:, [13, 14]], _KNEE_EXTRA_BLUR_STD)
                        stds_fused[:, [15, 16]] = np.hypot(stds_fused[:, [15, 16]], _ANKLE_EXTRA_BLUR_STD)
                    with torch.backends.cudnn.flags(enabled=True, allow_tf32=False):
                        fused_dev = expected_oks_blur(fused_dev, stds_fused)
                coords, _ = decode_fused(fused_dev.cpu().numpy(), target_geometries)
        coords_pre_pil = coords.copy()
        if SAHI_CONFIG is not None and _PIL_ROUNDING_ALPHA > 0.0:
            for bi, box_idx in enumerate(indices):
                c_box = crop_boxes[box_idx]
                if sahi_ran and all_sahi_geoms is not None:
                    d12 = _view_rounding_delta_from_geom(coords[bi], crop_geometry(c_box, MEMBER_PADDINGS[0][0]))
                    d13 = _view_rounding_delta_from_geom(coords[bi], crop_geometry(c_box, MEMBER_PADDINGS[0][1]))
                    dv = [_view_rounding_delta_from_geom(coords[bi], all_sahi_geoms[bi, vi]) for vi in range(7)]
                    d_fb = (0.5 * d12 + 0.5 * d13 + 0.20 * dv[0] + 0.25 * dv[1] + 0.20 * dv[2]) / 1.65
                    w_slice = 0.45 * float(ar_gate_np[bi])
                    w_v56 = 0.16 * float(ar_gate_np[bi])
                    delta_bi = np.zeros_like(coords[bi], dtype=np.float64)
                    for j_idx in range(17):
                        if j_idx in (1, 2, 3, 4):
                            delta_bi[j_idx] = 0.5 * (d12[j_idx] + d13[j_idx])
                        elif j_idx in (0, 5, 6, 7, 8, 9, 10):
                            d7 = (d_fb[j_idx] + w_slice * dv[3][j_idx]) / (1.0 + w_slice)
                            delta_bi[j_idx] = (d7 + w_v56 * dv[5][j_idx]) / (1.0 + w_v56)
                        else:
                            d7 = (d_fb[j_idx] + w_slice * dv[4][j_idx]) / (1.0 + w_slice)
                            delta_bi[j_idx] = (d7 + w_v56 * dv[6][j_idx]) / (1.0 + w_v56)
                elif _RUNTIME["reduced"]:
                    delta_bi = _view_rounding_delta_from_geom(coords[bi], target_geometries[bi])
                else:
                    d12 = _view_rounding_delta_from_geom(coords[bi], crop_geometry(c_box, MEMBER_PADDINGS[0][0]))
                    d13 = _view_rounding_delta_from_geom(coords[bi], crop_geometry(c_box, MEMBER_PADDINGS[0][1]))
                    delta_bi = 0.5 * (d12 + d13)
                coords[bi] = coords[bi] + (_PIL_ROUNDING_ALPHA * delta_bi).astype(np.float32)
        score_Subset = per_model_scores[:4] if (SAHI_CONFIG is not None and len(per_model_scores) >= 4) else per_model_scores
        confidence = np.mean(np.stack(score_Subset), axis=0)
        if person_ambiguity is not None:
            confidence = confidence * (1.0 - AMBIGUITY_BETA * person_ambiguity)
        if person_torso is not None:
            confidence = confidence * np.exp(-TORSO_GAMMA * person_torso)
        if c_members_list:
            mc_arr = np.stack(c_members_list, axis=0)
            cons_w = _CONSENSUS_WEIGHTS_3 if len(MODEL_FILES) == 3 else _CONSENSUS_WEIGHTS_5
            wn_arr = np.asarray(
                [(cons_w[m_idx] if m_idx < len(cons_w) else 0.55) * (rsn_scale if m_idx == 0 else 1.0) for m_idx in active],
                dtype=np.float64,
            ).reshape(-1, 1, 1)
            person_area = np.maximum(box_array[:, 2] * box_array[:, 3] * 0.53, 1.0)
            vars_j = (EOKS_SIGMAS * 2.0) ** 2
            d2_arr = ((mc_arr - coords_pre_pil[None, :, :, :]) ** 2).sum(axis=-1)
            oks_arr = np.exp(-d2_arr / (person_area[None, :, None] * vars_j[None, None, :] * 2.0))
            ag_j = (wn_arr * oks_arr).sum(axis=0)[:, _SCORED_KEYPOINTS] / wn_arr.sum()
            confidence = confidence * (np.percentile(ag_j, _CONSENSUS_PCT, axis=1) ** _CONSENSUS_PWR)
        for local_index, box_index in enumerate(indices):
            keypoints[box_index] = [
                [float(x_coord), float(y_coord), 1.0]
                for x_coord, y_coord in coords[local_index]
            ]
            scores[box_index] = float(confidence[local_index])
    if timed and usable:
        _MEMBERS["samples"] += 1
    if tick:
        _runtime_tick()
    return {"keypoints": keypoints, "scores": scores}



ADAPTIVE_SHRINK = (1.14, 0.08, 0.8, 0.3)  # (ratio, deadband, min scale, min score)
ADAPTIVE_SHRINK_WALL_LIMIT_SECONDS = 6400.0  # no second passes after this wall time
ADAPTIVE_SHRINK_PROJ_LIMIT_SECONDS = 6700.0  # nor while the projected wall exceeds this


def predict_image(sample: dict[str, Any]) -> dict[str, Any]:
    out = _predict_image_core(sample)
    boxes = sample.get("boxes") or []
    if ADAPTIVE_SHRINK is None or len(boxes) != 1 or not out["keypoints"] or out["keypoints"][0] is None:
        return out
    ratio, deadband, min_scale, min_score = ADAPTIVE_SHRINK
    started = _RUNTIME["load_start"] or _RUNTIME["start"]
    if _RUNTIME["reduced"] or started is None:
        return out
    wall = time.perf_counter() - started
    done = max(1, int(_RUNTIME["images"]))
    pace = wall / done  # overall mean pace (stable; the 80-image roll oscillates when second passes toggle)
    if wall > ADAPTIVE_SHRINK_WALL_LIMIT_SECONDS or (done >= 160 and wall + max(0, EXPECTED_TOTAL_IMAGES - done) * pace > ADAPTIVE_SHRINK_PROJ_LIMIT_SECONDS):
        return out
    try:
        kp = np.asarray(out["keypoints"][0], dtype=np.float64)
        x, y, w, h = (float(v) for v in boxes[0][:4])
        extent = max(kp[15, 1], kp[16, 1]) - kp[2, 1]
        if not (np.isfinite(kp).all() and extent > 4.0 and h > 0.0 and w > 0.0 and float(out["scores"][0]) >= min_score):
            return out
        scale = float(np.clip(ratio * extent / h, min_scale, 1.0))
        if scale >= 1.0 - deadband:
            return out
        cx, cy = x + 0.5 * w, y + 0.5 * h
        second = _predict_image_core({**sample, "_rsn_area": w * h, "boxes": [[cx - 0.5 * w * scale, cy - 0.5 * h * scale, w * scale, h * scale]]}, tick=False)
        if second["keypoints"] and second["keypoints"][0] is not None and np.isfinite(np.asarray(second["keypoints"][0], dtype=np.float64)).all():
            _RUNTIME["shrink_recrops"] = _RUNTIME.get("shrink_recrops", 0) + 1
            return second
    except (ValueError, TypeError, IndexError):
        pass
    return out


_NUMERICAL_GUARD_STATS = {}


def finite_precision_forward(model, inputs, *, stats=None):
    if inputs.dtype != torch.float32:
        raise ValueError("The FP32 input must be retained for a genuine fallback")
    cuda = inputs.device.type == "cuda"
    if stats is not None:
        stats["calls"] = stats.get("calls", 0) + 1
    with torch.inference_mode(), torch.jit.optimized_execution(False):
        context = (torch.autocast("cuda", dtype=torch.float16, cache_enabled=False)
                   if cuda else nullcontext())
        with context:
            raw = model(inputs).float()
        if bool(torch.isfinite(raw).all()):
            return raw
        if stats is not None:
            stats["nonfinite_initial_calls"] = stats.get("nonfinite_initial_calls", 0) + 1
        if not cuda:
            raise FloatingPointError("Nonfinite FP32 CPU heatmaps")
        if stats is not None:
            stats["FP32_fallback_calls"] = stats.get("FP32_fallback_calls", 0) + 1
        with torch.autocast("cuda", enabled=False):
            raw = model(inputs).float()
        if not bool(torch.isfinite(raw).all()):
            raise FloatingPointError("Nonfinite FP32 fallback heatmaps")
        return raw


def accelerated_forward(model, tensor, model_index=None):
    if isinstance(model, IsolatedMember):
        return model(tensor)
    graphed = _GRAPHED.get(model_index)
    if graphed is not None and tensor.device.type == "cuda":
        total_n = int(tensor.shape[0])
        if total_n in (2, 4, 14) and ((total_n, 3, 256, 256), tensor.dtype) in graphed.graphs:
            raw = graphed(tensor)
        else:
            chunks = [graphed(tensor[s : s + 4]) for s in range(0, total_n, 4)]
            raw = torch.cat(chunks, dim=0)
        _NUMERICAL_GUARD_STATS["graph_calls"] = _NUMERICAL_GUARD_STATS.get("graph_calls", 0) + 1
        if not bool(torch.isfinite(raw).all()):
            _NUMERICAL_GUARD_STATS["nonfinite_initial_calls"] = _NUMERICAL_GUARD_STATS.get("nonfinite_initial_calls", 0) + 1
            raw = torch.nan_to_num(raw, nan=0.0, posinf=1.0, neginf=0.0)
        return raw
    return finite_precision_forward(model, tensor, stats=_NUMERICAL_GUARD_STATS)


def sahi_forward(model, tensor):
    if isinstance(model, IsolatedMember):
        return model(tensor)
    return finite_precision_forward(model, tensor, stats=_NUMERICAL_GUARD_STATS)


"""CUDA-graph replay of a fixed-shape FP16 TorchScript pose forward (deployment helper).

The packaged run.py calls each model once per image with the same view batch
shape, and the RSN forward is bound by host-side kernel launches (~5000 per
call).  Capturing that forward once per shape and replaying it removes the
launch cost without changing the kernels.  This file is embedded verbatim into
the fusion run.py by the packager, so it must stay self-contained.
"""
import torch


def eager_forward(model, inputs):
    """The deployed FP16 forward: autocast without weight cache, TorchScript optimisation off."""
    with torch.inference_mode(), torch.jit.optimized_execution(False):
        with torch.autocast('cuda', dtype=torch.float16, cache_enabled=False):
            return model(inputs).float()


class GraphedForward:
    """One captured CUDA graph per input shape, replayed on copied inputs.

    Any capture error permanently routes that shape to ``eager_forward``; CPU
    inputs always use it.  Graphs keep private memory pools (never shared).
    """

    def __init__(self, model, max_shapes=4):
        self.model, self.max_shapes = model, max_shapes
        self.graphs, self.failed, self.captures = {}, set(), 0

    def _capture(self, inputs):
        static_in = inputs.clone()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                eager_forward(self.model, static_in)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.jit.optimized_execution(False):
            with torch.cuda.graph(graph):
                with torch.autocast('cuda', dtype=torch.float16, cache_enabled=False):
                    static_out = self.model(static_in).float()
        torch.cuda.synchronize()
        self.captures += 1
        return graph, static_in, static_out

    def __call__(self, inputs):
        with torch.inference_mode():
            key = (tuple(inputs.shape), inputs.dtype)
            if inputs.device.type != 'cuda' or key in self.failed:
                return eager_forward(self.model, inputs)
            entry = self.graphs.get(key)
            if entry is None:
                if len(self.graphs) >= self.max_shapes:
                    return eager_forward(self.model, inputs)
                try:
                    entry = self._capture(inputs)
                except Exception as error:  # capture is an optimisation; never fail the run for it
                    self.failed.add(key)
                    torch.cuda.synchronize()
                    print(f'[submission] CUDA graph capture disabled for {key}: {type(error).__name__}', flush=True)
                    return eager_forward(self.model, inputs)
                self.graphs[key] = entry
            graph, static_in, static_out = entry
            static_in.copy_(inputs)
            graph.replay()
            return static_out.clone()



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
    """DARK (Zhang et al., CVPR 2020) second-order Taylor offsets on the blurred log map, vectorised over all (person, joint) maps."""
    batch, joints, height, width = heatmaps.shape
    win, b_mat, b_mat_t, refl_y, refl_x = _get_dark_patch_tables(kernel, height, width)
    offsets = np.zeros_like(coords)
    px = coords[..., 0].astype(np.int64).reshape(-1)
    py = coords[..., 1].astype(np.int64).reshape(-1)
    ok = (px >= 2) & (px < width - 2) & (py >= 2) & (py < height - 2)
    if not ok.any():
        return np.clip(offsets, -1.0, 1.0)
    flat = heatmaps.reshape(batch * joints, height, width)
    sel = np.nonzero(ok)[0]
    ys = refl_y[(py[sel] - 2)[:, None] + np.arange(win)[None, :]]
    xs = refl_x[(px[sel] - 2)[:, None] + np.arange(win)[None, :]]
    patch = flat[sel[:, None, None], ys[:, :, None], xs[:, None, :]].astype(np.float64)
    orig_max = flat[sel, py[sel], px[sel]].astype(np.float64)
    blurred = np.matmul(np.matmul(b_mat[None], patch), b_mat_t[None])
    b_max = blurred.reshape(len(sel), -1).max(axis=1)
    scale = np.where((b_max > 0.0) & (orig_max > 0.0), orig_max / np.where(b_max > 0.0, b_max, 1.0), 1.0)
    m = np.log(np.maximum(blurred * scale[:, None, None], 1e-10))
    dx = 0.5 * (m[:, 2, 3] - m[:, 2, 1])
    dy = 0.5 * (m[:, 3, 2] - m[:, 1, 2])
    dxx = 0.25 * (m[:, 2, 4] - 2 * m[:, 2, 2] + m[:, 2, 0])
    dxy = 0.25 * (m[:, 3, 3] - m[:, 1, 3] - m[:, 3, 1] + m[:, 1, 1])
    dyy = 0.25 * (m[:, 4, 2] - 2 * m[:, 2, 2] + m[:, 0, 2])
    det = dxx * dyy - dxy * dxy
    good = np.abs(det) > 1e-12
    safe = np.where(good, det, 1.0)
    out = offsets.reshape(-1, 2)
    out[sel[good], 0] = (-(dyy * dx - dxy * dy) / safe)[good]
    out[sel[good], 1] = (-(dxx * dy - dxy * dx) / safe)[good]
    return np.clip(out.reshape(offsets.shape), -1.0, 1.0)



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



_IPC_MAX_BYTES = 128 * 1024 * 1024


class _PipeChannel:
    """Deadline-capable anonymous pipes; no socket/network permission needed."""

    def __init__(self, read_fd, write_fd):
        self.read_fd, self.write_fd = read_fd, write_fd
        self.timeout = None
        os.set_blocking(read_fd, False)
        os.set_blocking(write_fd, False)

    def settimeout(self, timeout):
        self.timeout = timeout

    def recv(self, count):
        import select
        ready, _, _ = select.select([self.read_fd], [], [], self.timeout)
        if not ready:
            raise TimeoutError("isolated pipe receive timed out")
        return os.read(self.read_fd, count)

    def sendall(self, data):
        import select
        deadline = None if self.timeout is None else time.monotonic() + self.timeout
        view = memoryview(data)
        while view:
            remaining = None if deadline is None else max(0.0, deadline - time.monotonic())
            _, ready, _ = select.select([], [self.write_fd], [], remaining)
            if not ready:
                raise TimeoutError("isolated pipe send timed out")
            try:
                written = os.write(self.write_fd, view)
                view = view[written:]
            except BlockingIOError:
                continue

    def close(self):
        for name in ("read_fd", "write_fd"):
            fd = getattr(self, name)
            if fd is not None:
                os.close(fd)
                setattr(self, name, None)


def _ipc_timeout(sock, deadline):
    if deadline is None:
        sock.settimeout(None)
    else:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("isolated member transaction exceeded its deadline")
        sock.settimeout(remaining)


def _ipc_read(sock, count, deadline):
    chunks = []
    while count:
        _ipc_timeout(sock, deadline)
        block = sock.recv(min(count, 1024 * 1024))
        if not block:
            raise EOFError("isolated member closed its pipe")
        chunks.append(block)
        count -= len(block)
    return b"".join(chunks)


def _ipc_send(sock, message, array=None, deadline=None):
    import struct
    header = dict(message)
    payload = b""
    if array is not None:
        array = np.ascontiguousarray(array, dtype=np.float32)
        payload = array.tobytes()
        header.update(shape=list(array.shape), dtype="float32", nbytes=len(payload))
    if len(payload) > _IPC_MAX_BYTES:
        raise ValueError("isolated tensor exceeds the IPC size limit")
    encoded = json.dumps(header, allow_nan=False).encode("utf-8")
    if len(encoded) > 65536:
        raise ValueError("isolated header exceeds the IPC size limit")
    _ipc_timeout(sock, deadline)
    sock.sendall(struct.pack("!I", len(encoded)) + encoded)
    if payload:
        _ipc_timeout(sock, deadline)
        sock.sendall(payload)


def _ipc_receive(sock, deadline=None):
    import struct
    size = struct.unpack("!I", _ipc_read(sock, 4, deadline))[0]
    if not 0 < size <= 65536:
        raise ValueError("invalid isolated header size")
    header = json.loads(_ipc_read(sock, size, deadline))
    if not isinstance(header, dict):
        raise ValueError("invalid isolated header")
    if "shape" not in header:
        return header, None
    shape = header["shape"]
    if (header.get("dtype") != "float32" or not isinstance(shape, list) or len(shape) != 4
            or any(type(n) is not int or n <= 0 for n in shape)):
        raise ValueError("invalid isolated tensor shape/dtype")
    nbytes = math.prod(shape) * 4
    if nbytes > _IPC_MAX_BYTES or nbytes != header.get("nbytes"):
        raise ValueError("invalid isolated tensor byte count")
    array = np.frombuffer(_ipc_read(sock, nbytes, deadline), dtype=np.float32).reshape(shape).copy()
    return header, array


class IsolatedMember:
    """One persistent model in a fresh interpreter, with bounded IPC waits."""

    def __init__(self, path, device, *, load_timeout=90.0, forward_timeout=30.0):
        import atexit
        import subprocess
        import sys
        self.device = torch.device(device)
        self.forward_timeout = float(forward_timeout)
        child_read, parent_write = os.pipe()
        parent_read, child_write = os.pipe()
        self.sock = _PipeChannel(parent_read, parent_write)
        self.process = None
        try:
            self.process = subprocess.Popen(
                [sys.executable, "-B", "-u", os.path.abspath(__file__), "--isolated-member-worker",
                 str(child_read), str(child_write), os.path.abspath(path), str(self.device)],
                pass_fds=(child_read, child_write), stdin=subprocess.DEVNULL)
            os.close(child_read)
            os.close(child_write)
            child_read = child_write = None
            reply, array = _ipc_receive(self.sock, time.monotonic() + float(load_timeout))
            if reply.get("status") != "ready" or array is not None:
                raise RuntimeError(f"isolated load failed: {reply}")
            self.pid = self.process.pid
            atexit.register(self.close)
            print(f"[submission] isolated member {os.path.basename(path)} ready pid={self.pid}", flush=True)
        except BaseException:
            for fd in (child_read, child_write):
                if fd is not None:
                    os.close(fd)
            self.close()
            raise

    def __call__(self, inputs):
        if inputs.dtype != torch.float32 or inputs.ndim != 4 or tuple(inputs.shape[1:]) != (3, 256, 256):
            raise ValueError("isolated member requires FP32 [N,3,256,256] inputs")
        deadline = time.monotonic() + self.forward_timeout
        try:
            _ipc_send(self.sock, {"operation": "forward"}, inputs.detach().cpu().numpy(), deadline)
            reply, array = _ipc_receive(self.sock, deadline)
            if (reply.get("status") != "ok" or array is None or array.shape[:2] != (inputs.shape[0], 17)
                    or array.shape[-1] != array.shape[-2] or not np.isfinite(array).all()):
                raise RuntimeError(f"isolated forward failed: {reply}")
            self.last_runtime_metrics = reply.get("runtime", {})
            return torch.from_numpy(array).to(inputs.device)
        except BaseException:
            self.close()
            raise

    def close(self):
        import atexit
        atexit.unregister(self.close)
        if getattr(self, "sock", None) is not None:
            self.sock.close()
            self.sock = None
        process = getattr(self, "process", None)
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=1.0)
            except __import__("subprocess").TimeoutExpired:
                process.kill()
                process.wait(timeout=1.0)


def _isolated_worker(read_fd, write_fd, path, device):
    sock = _PipeChannel(read_fd, write_fd)
    try:
        torch.set_num_threads(1)
        torch.set_grad_enabled(False)
        model = torch.jit.load(path, map_location=device).float().eval()
        _ipc_send(sock, {"status": "ready"})
        while True:
            request, array = _ipc_receive(sock)
            if request.get("operation") != "forward" or array is None or tuple(array.shape[1:]) != (3, 256, 256):
                raise ValueError("invalid isolated forward request")
            tensor = torch.from_numpy(array).to(device)
            raw = finite_precision_forward(model, tensor)
            metrics = ({"cuda_max_memory_allocated_bytes": torch.cuda.max_memory_allocated(),
                        "cuda_max_memory_reserved_bytes": torch.cuda.max_memory_reserved()}
                       if tensor.device.type == "cuda" else {})
            _ipc_send(sock, {"status": "ok", "runtime": metrics}, raw.cpu().numpy())
    except EOFError:
        pass
    except BaseException as error:
        try:
            _ipc_send(sock, {"status": "error", "error": f"{type(error).__name__}: {error}"[:2000]},
                      deadline=time.monotonic() + 1.0)
        except Exception:
            pass
        raise
    finally:
        sock.close()


if __name__ == "__main__":
    import sys
    if len(sys.argv) == 6 and sys.argv[1] == "--isolated-member-worker":
        _isolated_worker(int(sys.argv[2]), int(sys.argv[3]), sys.argv[4], sys.argv[5])


"""Self-contained optional measured fallback allocator, embedded in run.py.

No GT, image identity, or model scores enter this resource-only controller.
Costs are measured on the execution host; a future stall/domain shift can still
invalidate the projection. This is not a universal completion guarantee.
"""
_DEADLINE = {'window_start':None,'window_image':0,'mode':'full',
             'full_paces':[],'center_paces':[],'calibrated':False,'dropped':False}

def _deadline_switch(now, mode):
    previous=_DEADLINE['mode']
    _DEADLINE['mode']=mode
    _RUNTIME['reduced']=mode=='center'
    if _DEADLINE['calibrated'] or _DEADLINE['dropped'] or mode=='center':
        _RUNTIME['sahi_tier']=3
        _RUNTIME['sahi_disabled']=True
    _DEADLINE['window_start']=now
    _DEADLINE['window_image']=_RUNTIME['images']
    if previous!=mode:
        print(f"[submission] measured deadline allocation -> {mode} after {_RUNTIME['images']} images",flush=True)

def _deadline_policy_tick(now, wall_elapsed, remaining):
    # Preserve all original SAHI choices while the full path meets the budget.
    if _DEADLINE['calibrated'] or _DEADLINE['dropped'] or _DEADLINE['mode']=='center':
        _RUNTIME['sahi_tier']=3
        _RUNTIME['sahi_disabled']=True
    if _RUNTIME['images']<480 or _RUNTIME['sahi_tier']!=3:
        _DEADLINE['window_start']=None
        return
    if _DEADLINE['window_start'] is None:
        _deadline_switch(now,'full')
        return
    count=_RUNTIME['images']-_DEADLINE['window_image']
    if count<80:return
    pace=max(0.,(now-_DEADLINE['window_start'])/count)
    mode=_DEADLINE['mode']
    history=_DEADLINE[mode+'_paces']
    history.append(pace)
    del history[:-3]
    budget=WALL_BUDGET_REDUCE_SECONDS
    if not _DEADLINE['calibrated']:
        if mode=='full':
            if wall_elapsed+remaining*max(history)>budget:
                _deadline_switch(now,'center') # one real80-image cost probe
            else:_deadline_switch(now,'full')
            return
        _DEADLINE['calibrated']=True
    center_pace=max(_DEADLINE['center_paces'])*1.05
    if mode=='center' and wall_elapsed+remaining*center_pace>budget:
        active=active_members()
        if len(active)>1:
            slowest=max(active,key=lambda i:_MEMBERS['seconds'][i] if _MEMBERS['samples']>0 else i)
            active.remove(slowest)
            _DEADLINE['dropped']=True
            # Costs belong to the current member set, not its parent.
            _DEADLINE['center_paces']=[]
            _DEADLINE['full_paces']=[]
            _DEADLINE['calibrated']=False
            print(f"[submission] measured deadline allocation drops {MODEL_FILES[slowest]} after {_RUNTIME['images']} images",flush=True)
            _deadline_switch(now,'full')
            return
    full_pace=max(_DEADLINE['full_paces'])*1.05
    if full_pace<=center_pace:
        allowed_full=remaining if wall_elapsed+remaining*full_pace<=budget else 0
    else:
        allowed_full=(budget-wall_elapsed-remaining*center_pace)/(full_pace-center_pace)
    # Leave at least one window of reaction margin instead of switching at zero.
    _deadline_switch(now,'full' if allowed_full>=160 else 'center')
