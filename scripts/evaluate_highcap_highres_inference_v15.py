#!/usr/bin/env python3
"""Evaluate one frozen robust model at a predeclared higher inference resolution."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys
import time
from typing import Any, Sequence

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = PROJECT_ROOT / "artifacts/runs/20261003T155000+0900_highres_inference_v15"
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from evaluate_highcap_robust_distal_loss_v13 import _joint_errors  # noqa: E402
from evaluate_robust_seed_geometry_ensemble_v7 import aligned_geometry_mean  # noqa: E402
from evaluate_seed_geometry_ensemble_v6 import PADDINGS, TARGET_PADDING, prediction  # noqa: E402
from train_scratch_cnn import dataset_items, image_path, load_domain  # noqa: E402
from upar_pose.evaluation import (  # noqa: E402
    assigned_prediction_oks_by_image,
    evaluate_coco_keypoints,
    paired_bootstrap_mean_difference,
)
from upar_pose.flip_tta import FLIP_CHANNELS, decode_heatmaps_refined  # noqa: E402
from upar_pose.geometry_tta_v4 import scaled_crop_geometry  # noqa: E402
from upar_pose.pipeline import load_json, sha256_file, write_json_atomic  # noqa: E402
from upar_pose.scratch_cnn import normalize_image  # noqa: E402

CANDIDATE = "upar_highcap_robust_highres_inference_v15"
MODEL = PROJECT_ROOT / "artifacts/runs/20261003T051700+0900_highcap_bbox_robust_v8/fit_market_pa/model.torchscript"
MODEL_SHA256 = "68778c5f3e8e2abe2b1cb443c133816437318e3257b46f44f0c39f44f8a3475e"
REFERENCE = PROJECT_ROOT / (
    "artifacts/runs/20261003T130000+0900_highcap_robust_multiseed_v11/"
    "holdout_peta_three_robust/robust_seed20261005_geometry_predictions.json"
)
REFERENCE_SHA256 = "3ca347a9f133a734142467e7e6fa6d3e37056bc843ce1923f4bf1266a747d7e1"
REFERENCE_AP = 0.8708209714455037
STANDARD_SIZE = (192, 256)
HIGHRES_SIZE = (240, 320)
STANDARD_HEATMAP_SIZE = (48, 64)
HIGHRES_HEATMAP_SIZE = (60, 80)
MATERIAL_AP_DELTA = 0.001
BOOTSTRAP_REPLICATES = 5000
BOOTSTRAP_SEED = 20261505
EVALUATION_CONTRACT = RUN_ROOT / "work_os/evaluation_contract.json"
FIT_AUDIT = RUN_ROOT / "work_os/fit_audit.json"
TRAIN_GROUPS = RUN_ROOT / "work_os/train_groups.json"
EVAL_GROUPS = RUN_ROOT / "work_os/eval_groups.json"
TUNING_GROUPS = RUN_ROOT / "work_os/tuning_groups.json"
COMMITMENT = RUN_ROOT / "work_os/commitment.json"


EXACT_CROP = False  # opt-in: sample the unrounded square (Image.crop rounds its corners; labels/inverse use the unrounded geometry)


def crop_resize(image, left, top, width, height, size):
    """Crop + bilinear resize to ``size``. Default = legacy Image.crop (rounded corners) then resize.

    With EXACT_CROP the same Pillow BILINEAR resample (including its downscale support) runs over the
    unrounded box; pixels outside the image are black exactly as Image.crop pads them.
    """
    if not EXACT_CROP:
        return image.crop((left, top, left + width, top + height)).resize(size, Image.Resampling.BILINEAR)
    margin = int(math.ceil(max(0.0, -left, -top, left + width - image.width, top + height - image.height))) + 2
    canvas = Image.new('RGB', (image.width + 2 * margin, image.height + 2 * margin))
    canvas.paste(image, (margin, margin))
    return canvas.resize(size, Image.Resampling.BILINEAR, box=(left + margin, top + margin, left + width + margin, top + height + margin))


class ResolutionInteractionDataset(Dataset):
    def __init__(self, items: Sequence[dict[str, Any]], size: tuple[int, int]) -> None:
        self.items = list(items)
        self.width, self.height = size
        if self.width / self.height != STANDARD_SIZE[0] / STANDARD_SIZE[1]:
            raise ValueError("resolution must retain the frozen crop aspect ratio")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int):
        item = self.items[index]
        with Image.open(image_path(item["file_name"])) as source:
            image = source.convert("RGB")
        geometries = [scaled_crop_geometry(item["box"], padding) for padding in PADDINGS]
        inputs = []
        for left, top, width, height in geometries:
            crop = crop_resize(image, left, top, width, height, (self.width, self.height))
            inputs.append(torch.from_numpy(normalize_image(np.asarray(crop, dtype=np.uint8))))
        return (
            torch.stack(inputs),
            torch.tensor(geometries, dtype=torch.float64),
            torch.tensor(item["image_id"], dtype=torch.int64),
            torch.tensor(item["annotation_id"], dtype=torch.int64),
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--event-path", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--reuse-predictions", action="store_true")
    return parser.parse_args()


def infer(
    model: torch.nn.Module,
    items: list[dict[str, Any]],
    *,
    size: tuple[int, int],
    expected_heatmap_size: tuple[int, int],
    device: torch.device,
    batch_size: int,
    workers: int,
) -> tuple[list[dict[str, Any]], float, int | None]:
    loader = DataLoader(
        ResolutionInteractionDataset(items, size),
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        persistent_workers=workers > 0,
        prefetch_factor=2 if workers > 0 else None,
    )
    flip_channels = torch.as_tensor(FLIP_CHANNELS, dtype=torch.long, device=device)
    predictions: list[dict[str, Any]] = []
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    started = time.monotonic()
    with torch.inference_mode():
        for inputs, geometries, image_ids, annotation_ids in loader:
            count, scale_count = inputs.shape[:2]
            flat = inputs.reshape(-1, *inputs.shape[2:]).to(device, non_blocking=True)
            paired = torch.cat((flat, torch.flip(flat, dims=(-1,))), dim=0)
            raw = model(paired).float()
            actual_size = (int(raw.shape[-1]), int(raw.shape[-2]))
            if actual_size != expected_heatmap_size:
                raise RuntimeError(f"expected heatmap {expected_heatmap_size}, got {actual_size}")
            original, flipped_raw = raw.split(flat.shape[0], dim=0)
            flipped_back = torch.flip(flipped_raw, dims=(-1,)).index_select(1, flip_channels)
            views = (0.5 * (original + flipped_back)).reshape(
                count, scale_count, *original.shape[1:]
            )
            heatmaps = aligned_geometry_mean(views)
            geometry_rows = [
                tuple(float(value) for value in row)
                for row in geometries[:, PADDINGS.index(TARGET_PADDING), :].numpy()
            ]
            coordinates, maxima = decode_heatmaps_refined(
                heatmaps.cpu().numpy(), geometry_rows, refinement="quadratic"
            )
            for index in range(count):
                predictions.append(
                    prediction(
                        coordinates[index],
                        maxima[index],
                        int(image_ids[index]),
                        int(annotation_ids[index]),
                    )
                )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.monotonic() - started
    peak = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
    return predictions, elapsed, peak


def benchmark_forward(
    model: torch.nn.Module,
    *,
    size: tuple[int, int],
    device: torch.device,
    batch_size: int = 32,
    iterations: int = 20,
) -> dict[str, Any]:
    """Measure a matched model-forward cost independent of data-loader timing."""

    inputs = torch.zeros(batch_size, 3, size[1], size[0], device=device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    with torch.inference_mode():
        for _ in range(3):
            model(inputs)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        started = time.monotonic()
        for _ in range(iterations):
            model(inputs)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
    elapsed = time.monotonic() - started
    return {
        "batch_size": batch_size,
        "iterations": iterations,
        "seconds_per_batch": elapsed / iterations,
        "images_per_second": batch_size * iterations / elapsed,
        "peak_cuda_allocated_bytes": (
            torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
        ),
    }


def main() -> int:
    args = parse_args()
    if args.batch_size <= 0 or args.workers < 0:
        raise SystemExit("batch size must be positive and workers nonnegative")
    output_dir = args.output_dir.resolve()
    event_path = args.event_path.resolve()
    if not output_dir.is_relative_to(RUN_ROOT.resolve()) or not event_path.is_relative_to(RUN_ROOT.resolve()):
        raise SystemExit("output and event paths must be inside the v15 run root")
    if event_path.is_relative_to(output_dir):
        raise SystemExit("event path must remain outside the new output directory")
    if sha256_file(MODEL) != MODEL_SHA256 or sha256_file(REFERENCE) != REFERENCE_SHA256:
        raise RuntimeError("frozen model/reference hash differs")
    for required in (EVALUATION_CONTRACT, FIT_AUDIT, TRAIN_GROUPS, EVAL_GROUPS, TUNING_GROUPS, COMMITMENT):
        if not required.is_file():
            raise RuntimeError(f"missing WorkOS evidence: {required}")
    if args.require_cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA was required but is not visible")
    if args.reuse_predictions:
        if not output_dir.is_dir():
            raise RuntimeError("reuse requested but output directory is missing")
    else:
        output_dir.mkdir(parents=True, exist_ok=False)
    event_path.parent.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.jit.load(str(MODEL), map_location=device).eval()
    dataset = load_domain("peta")
    items = dataset_items(dataset, require_labels=False)
    standard_path = output_dir / "standard_resolution_replay_predictions.json"
    highres_path = output_dir / "highres_240x320_predictions.json"
    if args.reuse_predictions:
        if not standard_path.is_file() or not highres_path.is_file():
            raise RuntimeError("reuse requested but saved prediction files are missing")
        standard_seconds = None
        highres_seconds = None
        standard_peak = None
        highres_peak = None
    else:
        standard_predictions, standard_seconds, standard_peak = infer(
            model,
            items,
            size=STANDARD_SIZE,
            expected_heatmap_size=STANDARD_HEATMAP_SIZE,
            device=device,
            batch_size=args.batch_size,
            workers=args.workers,
        )
        highres_predictions, highres_seconds, highres_peak = infer(
            model,
            items,
            size=HIGHRES_SIZE,
            expected_heatmap_size=HIGHRES_HEATMAP_SIZE,
            device=device,
            batch_size=args.batch_size,
            workers=args.workers,
        )
        write_json_atomic(standard_path, standard_predictions)
        write_json_atomic(highres_path, highres_predictions)
    standard_predictions = load_json(standard_path)
    highres_predictions = load_json(highres_path)
    reference_predictions = load_json(REFERENCE)
    standard_metrics = evaluate_coco_keypoints(dataset, standard_predictions, max_dets=20)
    reference_metrics = evaluate_coco_keypoints(dataset, reference_predictions, max_dets=20)
    if not math.isclose(reference_metrics["AP"], REFERENCE_AP, rel_tol=0.0, abs_tol=1e-12):
        raise RuntimeError("frozen reference AP replay differs")
    standard_replay_ap_delta = standard_metrics["AP"] - REFERENCE_AP
    if not math.isclose(standard_metrics["AP"], REFERENCE_AP, rel_tol=0.0, abs_tol=1e-4):
        raise RuntimeError("standard-resolution evaluator replay differs from frozen reference")
    candidate_metrics = evaluate_coco_keypoints(dataset, highres_predictions, max_dets=20)
    paired = paired_bootstrap_mean_difference(
        assigned_prediction_oks_by_image(dataset, highres_predictions),
        assigned_prediction_oks_by_image(dataset, reference_predictions),
        replicates=BOOTSTRAP_REPLICATES,
        confidence_level=0.95,
        seed=BOOTSTRAP_SEED,
    )
    candidate_joint = _joint_errors(dataset, highres_predictions)
    reference_joint = _joint_errors(dataset, reference_predictions)
    joint_comparison = {
        name: {
            "count": candidate_joint[name]["count"],
            "reference_mean_error_over_sqrt_area": reference_joint[name]["mean_error_over_sqrt_area"],
            "candidate_mean_error_over_sqrt_area": candidate_joint[name]["mean_error_over_sqrt_area"],
            "candidate_minus_reference": candidate_joint[name]["mean_error_over_sqrt_area"] - reference_joint[name]["mean_error_over_sqrt_area"],
        }
        for name in candidate_joint
    }
    delta = candidate_metrics["AP"] - reference_metrics["AP"]
    standard_benchmark = benchmark_forward(model, size=STANDARD_SIZE, device=device)
    highres_benchmark = benchmark_forward(model, size=HIGHRES_SIZE, device=device)
    runtime_ratio = (
        highres_benchmark["seconds_per_batch"] / standard_benchmark["seconds_per_batch"]
    )
    standard_benchmark_peak = standard_benchmark["peak_cuda_allocated_bytes"]
    highres_benchmark_peak = highres_benchmark["peak_cuda_allocated_bytes"]
    peak_ratio = (
        highres_benchmark_peak / standard_benchmark_peak
        if highres_benchmark_peak and standard_benchmark_peak
        else None
    )
    if delta > MATERIAL_AP_DELTA and paired["ci_low"] > 0.0:
        decision = "support_evaluate_three_robust_seeds_at_240x320"
    elif delta <= 0.0 or paired["ci_high"] < 0.0:
        decision = "refute_inference_only_scaling_train_matched_highres_model"
    else:
        decision = "ambiguous_inspect_runtime_and_joint_tradeoffs_without_submission"
    next_actions = {
        "support_evaluate_three_robust_seeds_at_240x320": "Evaluate the frozen three-robust-seed RRR ensemble at the same 240x320 resolution; package only if ensemble gain and runtime validation remain positive.",
        "refute_inference_only_scaling_train_matched_highres_model": "Reject inference-only scaling; if pursuing resolution, fit one matched robust seed natively at 240x320 before any submission.",
        "ambiguous_inspect_runtime_and_joint_tradeoffs_without_submission": "Do not submit; inspect per-joint effects and measured cost, then decide whether a native 240x320 fit is justified.",
    }
    generated_at = datetime.now(timezone.utc).isoformat()
    comparison = {
        "schema_version": 1,
        "candidate": CANDIDATE,
        "generated_at": generated_at,
        "candidate_count": 1,
        "model": {"path": str(MODEL.relative_to(PROJECT_ROOT)), "sha256": MODEL_SHA256},
        "isolated_factor": "inference_input_and_heatmap_resolution",
        "standard_resolution": {
            "input_width_height": list(STANDARD_SIZE),
            "heatmap_width_height": list(STANDARD_HEATMAP_SIZE),
            "metrics": standard_metrics,
            "seconds": standard_seconds,
            "peak_cuda_allocated_bytes": standard_peak,
            "predictions_path": standard_path.name,
            "predictions_sha256": sha256_file(standard_path),
        },
        "candidate_resolution": {
            "input_width_height": list(HIGHRES_SIZE),
            "heatmap_width_height": list(HIGHRES_HEATMAP_SIZE),
            "metrics": candidate_metrics,
            "seconds": highres_seconds,
            "peak_cuda_allocated_bytes": highres_peak,
            "predictions_path": highres_path.name,
            "predictions_sha256": sha256_file(highres_path),
        },
        "reference": {
            "path": str(REFERENCE.relative_to(PROJECT_ROOT)),
            "sha256": REFERENCE_SHA256,
            "metrics": reference_metrics,
        },
        "standard_replay_AP_delta_vs_frozen": standard_replay_ap_delta,
        "standard_replay_parity_tolerance": 0.0001,
        "inference_reused_after_parity_gate_stop": args.reuse_predictions,
        "forward_benchmark": {
            "standard": standard_benchmark,
            "highres": highres_benchmark,
            "runtime_ratio_highres_over_standard": runtime_ratio,
            "peak_vram_ratio_highres_over_standard": peak_ratio,
        },
        "result": {
            "metrics": candidate_metrics,
            "prediction_count": len(highres_predictions),
            "rescored_from_saved_predictions": True,
            "paired_assigned_oks_candidate_minus_reference": paired,
        },
        "AP_delta_candidate_minus_reference": delta,
        "material_threshold_AP_delta": MATERIAL_AP_DELTA,
        "joint_normalized_error_comparison": joint_comparison,
        "decision": decision,
        "next_action": next_actions[decision],
        "limitations": [
            "PETA is a reused development surface, not independent confirmation or hidden-test evidence.",
            "The model was trained at 192x256; this test isolates inference-only scaling and does not estimate a natively trained high-resolution model.",
        ],
        "source_files": {
            str(path.relative_to(PROJECT_ROOT)): sha256_file(path)
            for path in (
                Path(__file__).resolve(),
                PROJECT_ROOT / "scripts/evaluate_robust_seed_geometry_ensemble_v7.py",
                PROJECT_ROOT / "src/upar_pose/geometry_tta_v4.py",
                PROJECT_ROOT / "src/upar_pose/evaluation.py",
            )
        },
    }
    comparison_path = output_dir / "highres_inference_comparison.json"
    write_json_atomic(comparison_path, comparison)
    manifest_path = output_dir / "manifest.json"
    write_json_atomic(
        manifest_path,
        {
            "schema_version": 1,
            "candidate": CANDIDATE,
            "generated_at": generated_at,
            "artifacts": {
                path.name: {"path": path.name, "sha256": sha256_file(path)}
                for path in (standard_path, highres_path, comparison_path)
            },
        },
    )
    rel = lambda path: str(path.relative_to(PROJECT_ROOT))
    event = {
        "id": "upar_highcap_robust_highres_inference_v15_peta",
        "candidate": CANDIDATE,
        "kind": "performance",
        "summary": f"Frozen robust seed-20261005 at 240x320 reached PETA AP {candidate_metrics['AP']:.6f} versus 192x256 AP {reference_metrics['AP']:.6f} ({delta:+.6f}).",
        "issue": "PETA is reused development data and the model was trained at the lower resolution; this only tests inference-time scaling.",
        "next_action": next_actions[decision],
        "candidate_status": "in_progress" if not decision.startswith("refute") else "rejected",
        "valid": True,
        "guardrails_pass": True,
        "is_best": False,
        "available_at": generated_at,
        "run_id": "20261003T155000+0900_highres_inference_v15_peta",
        "depends_on": ["upar_highcap_robust_coordinate_consistency_v14_peta"],
        "metric": {
            "name": "COCO_OKS_AP_PETA_HIGHRES_INFERENCE_MATCHED_MODEL",
            "direction": "max",
            "evaluation_id": "local-peta-highcap-highres-inference-v15",
            "evaluation_ref": rel(EVALUATION_CONTRACT),
            "baseline_id": "robust_seed_20261005_192x256",
            "baseline": reference_metrics["AP"],
            "candidate": candidate_metrics["AP"],
            "material_delta": MATERIAL_AP_DELTA,
        },
        "scope": {
            "evaluation_purpose": "candidate_comparison",
            "evaluation_role": "development",
            "execution": "Exact frozen robust seed-20261005 at one predeclared 240x320 resolution versus 192x256, unchanged six-view geometry/decoder, all 10,314 PETA images.",
            "review": "Parent hash, dynamic input/output shape, standard replay parity, saved candidate rescoring, runtime/VRAM, joint errors, and paired uncertainty were checked.",
            "train_groups": rel(TRAIN_GROUPS),
            "eval_groups": rel(EVAL_GROUPS),
            "tuning_groups": rel(TUNING_GROUPS),
            "fit_boundary_ref": rel(COMMITMENT),
            "fit_audit_ref": rel(FIT_AUDIT),
            "group_namespace": "upar-source-domain-v1",
        },
        "critical_check": {
            "risk": "A denser heatmap could add compute while changing the trained feature scale unfavorably.",
            "finding": f"Decision={decision}; AP delta={delta:+.6f}; paired OKS 95% CI=[{paired['ci_low']:.6f}, {paired['ci_high']:.6f}]; matched forward runtime ratio={runtime_ratio:.3f}.",
            "evidence": rel(comparison_path),
        },
        "visual": False,
        "evidence": [rel(comparison_path), rel(manifest_path), rel(MODEL), rel(REFERENCE)],
        "artifacts": [rel(comparison_path)],
    }
    if decision.startswith("support"):
        event["package_stage"] = "local_validated"
    write_json_atomic(event_path, event)
    print(json.dumps(comparison, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
