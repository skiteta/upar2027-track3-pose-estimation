"""UPAR Track 3 code submission: fixed multi-architecture heatmap fusion.

Each model runs its declared per-crop views (its paddings x {original,
flipped-back}; all members share one set by default, and crops are prepared
once per distinct padding and reused by every member that declares it); its
views are aligned to the target padding and averaged with its own normalised
weights.  Every model's aligned map is then scaled to unit peak per
joint, the maps are averaged with fixed model weights, and coordinates are
decoded from the fused map with the evaluator's log-quadratic refinement.  The
pose score is the mean of the models' own scores (mean clipped scored-joint
peak of each model's aligned map).  This mirrors
scripts/evaluate_multimodel_map_fusion.py of the development evaluation.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import time
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as functional
from PIL import Image


HERE = os.path.dirname(os.path.abspath(__file__))
META_PATH = os.path.join(HERE, "model_meta.json")
#__FUSION_CONSTANTS__
# Opt-in only. Existing packages retain their metadata and execution path. New
# isolated recipes must declare these constants in their frozen run.py.
ISOLATED_MODEL_INDICES = tuple(globals().get("ISOLATED_MODEL_INDICES", ()))
ISOLATED_LOAD_TIMEOUT = float(globals().get("ISOLATED_LOAD_TIMEOUT", 90.0))
ISOLATED_FORWARD_TIMEOUT = float(globals().get("ISOLATED_FORWARD_TIMEOUT", 30.0))
EXPECTED_TOTAL_IMAGES = 30245
ROLLING_WINDOW_IMAGES = 100
WALL_BUDGET_REDUCE_SECONDS = 5400.0
WALL_BUDGET_DROP_SECONDS = 5800.0
WALL_EMERGENCY_SINGLE_SECONDS = 6100.0
MODEL_PATHS = tuple(os.path.join(HERE, name) for name in MODEL_FILES)
TARGET_PADDING = 1.25
BATCH_SIZE = 16
# One (paddings, scale weights) view set per member.  When every member shares it, the package is the
# single-view-set recipe (same crops, batches, maths and metadata); otherwise crops are prepared once for
# the sorted union of paddings and each member selects its own.
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
        # Per-member view sets (index-aligned with model_files); the key is absent for single-view-set packages.
        inference["member_views"] = {
            "crop_paddings": [list(paddings) for paddings in MEMBER_PADDINGS],
            "scale_weights": [list(weights) for weights in MEMBER_SCALE_WEIGHTS],
            "view_count_per_crop": [2 * len(paddings) for paddings in MEMBER_PADDINGS],
            "crop_sharing": "one crop per box and distinct padding (sorted union = crop_paddings), reused by every member declaring it",
            "rule": "each member's paddings x flip are aligned to the target padding and averaged with its own scale weights before the unit-peak fusion; once the runtime guard trips every member uses the target padding only",
        }
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
    if DIAGNOSTIC_FILES:
        # Diagnostic-only TorchScript files: loaded and run in child processes and logged; they never enter the fusion.
        gpu = torch.cuda.get_device_name(0) if _DEVICE.type == "cuda" else "none"
        print(f"[submission] torch {torch.__version__}, device {_DEVICE} ({gpu}); diagnostic probes {list(DIAGNOSTIC_FILES)} "
              f"({DIAGNOSTIC_TIMEOUT:.0f} s each)", flush=True)
        for name in DIAGNOSTIC_FILES:
            _run_probe(os.path.join(HERE, name), name, DIAGNOSTIC_TIMEOUT)
    if PROBE_MEMBERS:
        gpu = torch.cuda.get_device_name(0) if _DEVICE.type == "cuda" else "none"
        print(f"[submission] torch {torch.__version__}, device {_DEVICE} ({gpu}); probing members "
              f"{[MODEL_FILES[i] for i in PROBE_MEMBERS]} in subprocesses ({PROBE_TIMEOUT:.0f} s each)", flush=True)
    # Members declared in PROBE_MEMBERS are first loaded and run in a child process; a member whose probe fails or
    # times out is left out of the fusion (weights renormalised) instead of stalling the whole run.
    excluded = [index for index in PROBE_MEMBERS if not _probe_member(index)]
    active = active_members()
    if excluded and FALLBACK_WEIGHTS is not None:
        # A failed probe switches the whole fusion to the declared fallback member set and weights.
        _MEMBERS["weights"] = tuple(FALLBACK_WEIGHTS)
        active[:] = [index for index, weight in enumerate(FALLBACK_WEIGHTS) if weight > 0]
        print(f"[submission] member probe failed: fallback fusion {[MODEL_FILES[i] for i in active]} "
              f"with weights {[FALLBACK_WEIGHTS[i] for i in active]}", flush=True)
    for index in excluded:
        if index in active:
            active.remove(index)
    # FP16-stored parameters are restored to FP32 once; a no-op for FP32 files.  Members outside the active
    # fusion (failed probe, or weight 0) are not loaded.
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
    print(f"[submission] {MODEL_TYPE}: {len(active)}-model fusion, {views} on {_DEVICE}", flush=True)
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
    """Load and run one member in a child process (same interpreter and device); False on error or timeout."""
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
    except Exception as error:  # the probe must never stop the run
        passed, detail = False, f"{type(error).__name__}: {error}"
    print(f"[submission] probe {name}: {'ok' if passed else 'FAILED'} after {time.perf_counter() - tick:.1f} s | "
          + " | ".join(line for line in detail.splitlines() if line.strip()), flush=True)
    return passed


_GRAPHED: dict[int, Any] = {}


def _prepare_graphs() -> None:
    """Capture declared members once at their own deployed view batch; keep them only if the replay
    equals the eager FP16 forward exactly on a fixed probe, otherwise run every member eagerly."""
    _GRAPHED.clear()
    if _DEVICE is None or _DEVICE.type != "cuda" or not GRAPH_MODEL_INDICES:
        return
    try:
        for index in GRAPH_MODEL_INDICES:
            if _MODELS[index] is None:   # not loaded (failed probe or outside the active fusion)
                continue
            # Same seed per member: a member with the shared view set gets the same probe as before.
            generator = torch.Generator().manual_seed(20261007)
            probe = (torch.rand((2 * len(MEMBER_PADDINGS[index]), 3, 256, 256), generator=generator) * 4.0 - 2.0).to(_DEVICE)
            graphed = GraphedForward(_MODELS[index], max_shapes=1)
            reference = eager_forward(_MODELS[index], probe)
            replay = graphed(probe)
            if graphed.captures != 1 or not torch.equal(reference, replay):
                raise RuntimeError("graph replay differs from eager forward")
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


def preprocess(
    image: Image.Image, box: Sequence[float], padding: float
) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    geometry = crop_geometry(box, padding)
    left, top, width, height = geometry
    crop = image.crop((left, top, left + width, top + height)).resize(
        (int(_META["input_width"]), int(_META["input_height"])),
        Image.Resampling.BILINEAR,
    )
    array = np.asarray(crop, dtype=np.float32).transpose(2, 0, 1) / 255.0
    return ((array - _MEAN) / _STD).astype(np.float32), geometry


def flip_back_heatmaps(heatmaps: torch.Tensor) -> torch.Tensor:
    channels = torch.as_tensor(_FLIP_CHANNELS, dtype=torch.long, device=heatmaps.device)
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


_RUNTIME: dict[str, Any] = {
    "load_start": None,
    "start": None,
    "images": 0,
    "reduced": False,
    "roll_start": None,
    "roll_images": 0,
    "last_roll_seconds": 0.0,
}
# Member forward times are sampled (device-synchronised) on every TIMING_EVERY-th image only and logged; with
# MEMBER_DROP_GUARD set, a member is dropped from the fusion when the reduced-view run is still too slow, so the
# run always ends inside the time limit.  With every member active the fused maths is unchanged.
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
    # Members never loaded (failed probe, or outside the active fusion) are not reported.
    return ", ".join(f"{MODEL_FILES[i]} {1000.0 * _MEMBERS['seconds'][i] / samples:.1f} ms"
                     + ("" if i in _MEMBERS["active"] else " (dropped)") for i in range(len(MODEL_FILES))
                     if _MODELS is None or _MODELS[i] is not None)


def active_member_views() -> tuple[tuple[tuple[float, ...], tuple[float, ...]], ...]:
    """Per-member declared (paddings, weights), or the centre padding only for every member once the runtime guard has tripped."""
    if _RUNTIME["reduced"]:
        return tuple(((TARGET_PADDING,), (1.0,)) for _ in MODEL_FILES)
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
              f"(roll={roll:.3f}s, wall={wall_elapsed:.1f}s) after {_RUNTIME['images']} images; "
              f"sampled member forward: {_member_report()}", flush=True)
    # Hard wall-clock emergency guard: guarantee completion well inside 7,200s
    if wall_elapsed >= WALL_EMERGENCY_SINGLE_SECONDS:
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
    _member_drop_tick(now, wall_elapsed, remaining)
    if RUNTIME_GUARD is None or _RUNTIME["reduced"]:
        return
    # With the member-drop guard, a run far over budget (4x after 50 images) reduces its views at once.
    emergency = MEMBER_DROP_GUARD is not None and _RUNTIME["images"] >= 50
    if _RUNTIME["images"] < 300 and not emergency:
        return
    mean_seconds = (now - _RUNTIME["start"]) / _RUNTIME["images"]
    roll_seconds = _RUNTIME["last_roll_seconds"] if _RUNTIME["last_roll_seconds"] > 0.0 else mean_seconds
    pace = max(mean_seconds, roll_seconds) if _RUNTIME["images"] >= 300 else mean_seconds
    projected_wall = wall_elapsed + remaining * pace
    budget = RUNTIME_GUARD[0] * (1.0 if _RUNTIME["images"] >= RUNTIME_GUARD[1] else 4.0)
    if mean_seconds > budget or (_RUNTIME["images"] >= 300 and projected_wall > WALL_BUDGET_REDUCE_SECONDS):
        _RUNTIME["reduced"] = True
        print(f"[submission] runtime guard: mean={mean_seconds:.3f}s roll={roll_seconds:.3f}s proj_wall={projected_wall:.0f}s "
              f"after {_RUNTIME['images']} images exceeds {RUNTIME_GUARD[0]:.3f}s / {WALL_BUDGET_REDUCE_SECONDS:.0f}s; "
              f"centre padding only from here", flush=True)
        print(f"[submission] sampled member forward: {_member_report()}", flush=True)
        _MEMBERS["window_start"], _MEMBERS["window_images"] = now, 0


def _member_drop_tick(now: float | None = None, wall_elapsed: float | None = None, remaining: int | None = None) -> None:
    """After the view reduction: per window of reduced images, drop the slowest member while the window is over budget."""
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
    if (window_mean > MEMBER_DROP_GUARD[0] or projected_wall > WALL_BUDGET_DROP_SECONDS) and len(active) > 1:
        slowest = max(active, key=lambda index: _MEMBERS["seconds"][index] if _MEMBERS["samples"] > 0 else index)
        active.remove(slowest)
        print(f"[submission] member-drop guard: {window_mean:.3f} s/image (proj_wall={projected_wall:.0f}s) "
              f"over {_MEMBERS['window_images']} reduced images exceeds {MEMBER_DROP_GUARD[0]:.3f}s / "
              f"{WALL_BUDGET_DROP_SECONDS:.0f}s; dropping {MODEL_FILES[slowest]} ({_member_report()})", flush=True)
    _MEMBERS["window_start"], _MEMBERS["window_images"] = now, 0


def model_aligned_mean(views: torch.Tensor, paddings: Sequence[float], scale_weights: Sequence[float]) -> torch.Tensor:
    """[batch, paddings, joints, h, w] of one member -> its normalised weighted mean aligned to the target padding."""
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
    """[batch, joints] per-joint peaks of one member's aligned map -> its mean clipped scored-joint peak."""
    return np.maximum(0.01, np.clip(maxima[:, _SCORED_KEYPOINTS], 0.0, 1.0).mean(axis=1))


def decode_heatmaps(
    heatmaps: np.ndarray,
    geometries: Sequence[tuple[float, float, float, float]],
) -> tuple[np.ndarray, np.ndarray]:
    """Copy the evaluator's log-quadratic decoder, including all 17 slots."""

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
    """Decode the fused map with the declared refinement (evaluator-identical code paths)."""

    if DECODER == "quadratic":
        return decode_heatmaps(heatmaps, geometries)
    if DECODER != "dark":
        raise RuntimeError(f"unknown decoder {DECODER}")
    batch, joints, height, width = heatmaps.shape
    flattened = heatmaps.reshape(batch, joints, -1)
    indices = np.argmax(flattened, axis=2)
    maxima = np.max(flattened, axis=2)
    coords = np.stack((indices % width, indices // width), axis=2).astype(np.float32)
    coords = coords + dark_offsets(heatmaps, coords, dark_kernel(width)).astype(np.float32)
    geometry = np.asarray(geometries, dtype=np.float64)
    coords[:, :, 0] = coords[:, :, 0] / (width - 1.0) * geometry[:, None, 2] + geometry[:, None, 0]
    coords[:, :, 1] = coords[:, :, 1] / (height - 1.0) * geometry[:, None, 3] + geometry[:, None, 1]
    return coords, maxima


def squat_box(box: Sequence[float]) -> list[float]:
    """Crop window for a squat (likely bottom-truncated) box: extend downward to the target aspect; others unchanged."""
    values = [float(value) for value in box[:4]]
    if SQUAT_BOX is None or not all(math.isfinite(value) for value in values) or values[2] <= 0.0 or values[3] <= 0.0:
        return values
    if values[3] / values[2] < SQUAT_BOX[0]:
        values[3] = max(values[3], SQUAT_BOX[1] * values[2])
    return values


def predict_image(sample: dict[str, Any]) -> dict[str, Any]:
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
    crop_paddings = (TARGET_PADDING,) if _RUNTIME["reduced"] else PADDINGS
    for start in range(0, len(usable), BATCH_SIZE):
        indices = usable[start : start + BATCH_SIZE]
        target_geometries = [crop_geometry(crop_boxes[index], TARGET_PADDING) for index in indices]
        # One crop per box and distinct padding, reused by every member that declares the padding.
        crops = {(index, padding): preprocess(image, crop_boxes[index], padding)[0] for index in indices for padding in crop_paddings}
        # One input batch per distinct view set, stacked (box-major) exactly as a single-view-set package stacks it.
        member_inputs: dict[tuple[float, ...], torch.Tensor] = {}
        for paddings, _ in member_views:
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
                # Per-joint peaks stay on the device: one host copy after the fusion instead of a sync per member.
                member_maxima.append(aligned.flatten(2).amax(-1))
                member_maps.append(aligned)
            side = max(int(maps.shape[-1]) for maps in member_maps)
            for aligned, weight in zip(member_maps, [_MEMBERS["weights"][index] for index in active], strict=True):
                if aligned.shape[-1] != side:
                    aligned = functional.interpolate(aligned, size=(side, side), mode="bilinear", align_corners=True)
                contribution = unit_peak(aligned) * (float(weight) / total_weight)
                fused = contribution if fused is None else fused + contribution
            assert fused is not None
            # Expected-OKS blur and ambiguity run on the CPU copy: a few ms per image, no extra GPU synchronisation.
            fused_cpu = fused.cpu()
            per_model_scores = [model_scores(maxima) for maxima in torch.stack(member_maxima).cpu().numpy()]
            box_array = np.asarray([[float(v) for v in boxes[index][:4]] for index in indices], dtype=np.float64)
            geometry_array = np.asarray(target_geometries, dtype=np.float64)
            # One far-secondary-peak pass serves both scores (same values as ambiguity() / torso_ambiguity(), computed once).
            ratios = secondary_peak_ratio(fused_cpu, box_array, geometry_array) if (AMBIGUITY_BETA or TORSO_GAMMA) else None
            person_ambiguity = ratios[:, list(AMBIGUITY_JOINTS)].mean(axis=1) if AMBIGUITY_BETA else None
            person_torso = ratios[:, list(TORSO_JOINTS)].max(axis=1) if TORSO_GAMMA else None
            if EXPECTED_OKS is not None:
                fused_cpu = expected_oks_blur(fused_cpu.double(), expected_oks_stds(box_array, geometry_array, int(fused_cpu.shape[-1]), *EXPECTED_OKS))
            fused_maps = fused_cpu.numpy()
        coords, _ = decode_fused(fused_maps, target_geometries)
        confidence = np.mean(np.stack(per_model_scores), axis=0)
        if person_ambiguity is not None:
            confidence = confidence * (1.0 - AMBIGUITY_BETA * person_ambiguity)
        if person_torso is not None:
            confidence = confidence * np.exp(-TORSO_GAMMA * person_torso)
        for local_index, box_index in enumerate(indices):
            keypoints[box_index] = [
                [float(x_coord), float(y_coord), 1.0]
                for x_coord, y_coord in coords[local_index]
            ]
            scores[box_index] = float(confidence[local_index])
    if timed and usable:
        _MEMBERS["samples"] += 1
    _runtime_tick()
    return {"keypoints": keypoints, "scores": scores}


from contextlib import nullcontext

_NUMERICAL_GUARD_STATS = {}

def finite_precision_forward(model, inputs, *, stats=None):
    """Try CUDA FP16; recompute the complete batch in FP32 if nonfinite.

    This preserves a finite FP16 result bit-for-bit. It is a numerical safety
    mechanism, not a claim of FP16 accuracy. CPU executes only FP32. A nonfinite
    FP32 result is fatal rather than a fabricated pose or score-floor result.
    """
    if inputs.dtype != torch.float32:
        raise ValueError('The FP32 input must be retained for a genuine fallback')
    cuda = inputs.device.type == 'cuda'
    if stats is not None:
        stats['calls'] = stats.get('calls', 0) + 1
    with torch.inference_mode(), torch.jit.optimized_execution(False):
        context = (torch.autocast('cuda', dtype=torch.float16, cache_enabled=False)
                   if cuda else nullcontext())
        with context:
            raw = model(inputs).float()
        if bool(torch.isfinite(raw).all()):
            return raw
        if stats is not None:
            stats['nonfinite_initial_calls'] = stats.get('nonfinite_initial_calls', 0) + 1
        if not cuda:
            raise FloatingPointError('Nonfinite FP32 CPU heatmaps')
        if stats is not None:
            stats['FP32_fallback_calls'] = stats.get('FP32_fallback_calls', 0) + 1
        with torch.autocast('cuda', enabled=False):
            raw = model(inputs).float()
        if not bool(torch.isfinite(raw).all()):
            raise FloatingPointError('Nonfinite FP32 fallback heatmaps')
        return raw


def accelerated_forward(model, tensor, model_index=None):
    if isinstance(model, IsolatedMember):
        # The child's forward already applies the exact same precision guard.
        return model(tensor)
    graphed = _GRAPHED.get(model_index)
    if graphed is not None and not _RUNTIME["reduced"]:
        raw = graphed(tensor)
        if bool(torch.isfinite(raw).all()):
            _NUMERICAL_GUARD_STATS['graph_calls'] = _NUMERICAL_GUARD_STATS.get('graph_calls', 0) + 1
            return raw
        # Same numerical guard as the eager path: recompute the full batch in FP32.
    return finite_precision_forward(model, tensor, stats=_NUMERICAL_GUARD_STATS)


#__CUDA_GRAPH_FORWARD__


#__DARK_DECODE__


#__EXPECTED_OKS__


# Isolated workers use exec, never fork a CUDA-initialised Python interpreter.
# Only numeric FP32 arrays and bounded JSON headers cross these pipes (no pickle).
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
    """One persistent model in a fresh interpreter, with bounded IPC waits.

    Process isolation cannot recover a device-wide driver hang. Such a hang is
    still a failed package, not evidence that this path is server-validated.
    """

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
            # This is our own inference child, never an unrelated training job.
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
        pass  # Parent finished; no orphan child survives closed pipes.
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
