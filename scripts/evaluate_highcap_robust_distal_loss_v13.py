#!/usr/bin/env python3
"""Evaluate the fixed distal-loss model against its matched robust seed control."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = PROJECT_ROOT / "artifacts/runs/20261003T150900+0900_distal_loss_v13"
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from evaluate_robust_seed_geometry_ensemble_v7 import aligned_geometry_mean  # noqa: E402
from evaluate_seed_geometry_ensemble_v6 import (  # noqa: E402
    InteractionDataset,
    PADDINGS,
    TARGET_PADDING,
    prediction,
)
from train_scratch_cnn import dataset_items, load_domain  # noqa: E402
from upar_pose.constants import IGNORED_KEYPOINT_INDICES, KEYPOINT_NAMES  # noqa: E402
from upar_pose.evaluation import (  # noqa: E402
    assigned_prediction_oks_by_image,
    evaluate_coco_keypoints,
    paired_bootstrap_mean_difference,
)
from upar_pose.flip_tta import FLIP_CHANNELS, decode_heatmaps_refined  # noqa: E402
from upar_pose.pipeline import load_json, sha256_file, write_json_atomic  # noqa: E402

CANDIDATE = "upar_highcap_robust_distal_loss_v13"
SEED = 20261005
REFERENCE = PROJECT_ROOT / (
    "artifacts/runs/20261003T130000+0900_highcap_robust_multiseed_v11/"
    "holdout_peta_three_robust/robust_seed20261005_geometry_predictions.json"
)
REFERENCE_SHA256 = "3ca347a9f133a734142467e7e6fa6d3e37056bc843ce1923f4bf1266a747d7e1"
REFERENCE_AP = 0.8708209714455037
MATERIAL_AP_DELTA = 0.001
BOOTSTRAP_REPLICATES = 5000
BOOTSTRAP_SEED = 20261305


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--event-path", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--require-cuda", action="store_true")
    return parser.parse_args()


def _joint_errors(
    dataset: dict[str, Any], predictions: list[dict[str, Any]]
) -> dict[str, dict[str, Any]]:
    gt_by_id = {int(row["id"]): row for row in dataset["annotations"]}
    values: dict[str, list[float]] = {name: [] for name in KEYPOINT_NAMES}
    ignored = set(IGNORED_KEYPOINT_INDICES)
    for row in predictions:
        annotation = gt_by_id[int(row["source_annotation_id"])]
        gt = np.asarray(annotation["keypoints"], dtype=np.float64).reshape(-1, 3)
        pred = np.asarray(row["keypoints"], dtype=np.float64).reshape(-1, 3)
        scale = math.sqrt(max(float(annotation["area"]), np.spacing(1)))
        for index, name in enumerate(KEYPOINT_NAMES):
            if index in ignored or gt[index, 2] <= 0:
                continue
            values[name].append(float(np.linalg.norm(pred[index, :2] - gt[index, :2]) / scale))
    return {
        name: {"count": len(errors), "mean_error_over_sqrt_area": float(np.mean(errors))}
        for name, errors in values.items()
        if errors
    }


def main() -> int:
    args = parse_args()
    if args.batch_size <= 0 or args.workers < 0:
        raise SystemExit("batch size must be positive and workers nonnegative")
    model_path = args.model.resolve()
    output_dir = args.output_dir.resolve()
    event_path = args.event_path.resolve()
    if not output_dir.is_relative_to(RUN_ROOT.resolve()) or not event_path.is_relative_to(RUN_ROOT.resolve()):
        raise SystemExit("output and event paths must be inside the v13 run root")
    if event_path.is_relative_to(output_dir):
        raise SystemExit("event path must remain outside the new output directory")
    fit_dir = RUN_ROOT / "holdout_peta_seed20261005"
    if model_path.parent != fit_dir.resolve():
        raise RuntimeError("candidate model is outside the frozen v13 fit directory")
    manifest_path = fit_dir / "manifest.json"
    manifest = load_json(manifest_path)
    config = manifest.get("config", {})
    model_sha256 = sha256_file(model_path)
    if (
        manifest.get("candidate") != CANDIDATE
        or manifest.get("component_role") != "distal_loss_seed_20261005"
        or config.get("candidate") != CANDIDATE
        or config.get("fit_scope") != "holdout_peta"
        or config.get("train_domains") != ["market", "pa100k"]
        or config.get("train_examples") != 32422
        or config.get("epochs") != 20
        or config.get("seed") != SEED
        or config.get("joint_loss_weights", {}).get("left_wrist") != 2.0
        or config.get("joint_loss_weights", {}).get("right_wrist") != 2.0
        or config.get("joint_loss_weights", {}).get("left_ankle") != 1.5
        or config.get("joint_loss_weights", {}).get("right_ankle") != 1.5
        or manifest.get("artifacts", {}).get("model.torchscript", {}).get("sha256") != model_sha256
    ):
        raise RuntimeError("v13 fit manifest does not match the predeclared candidate")
    if sha256_file(REFERENCE) != REFERENCE_SHA256:
        raise RuntimeError("matched robust reference prediction hash differs")
    if args.require_cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA was required but is not visible")
    output_dir.mkdir(parents=True, exist_ok=False)
    event_path.parent.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.jit.load(str(model_path), map_location=device).eval()
    dataset = load_domain("peta")
    items = dataset_items(dataset, require_labels=False)
    loader = DataLoader(
        InteractionDataset(items),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.workers > 0,
        prefetch_factor=2 if args.workers > 0 else None,
    )
    flip_channels = torch.as_tensor(FLIP_CHANNELS, dtype=torch.long, device=device)
    predictions: list[dict[str, Any]] = []
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    started = time.monotonic()
    with torch.inference_mode():
        for inputs, geometries, image_ids, annotation_ids in loader:
            batch_size, scale_count = inputs.shape[:2]
            flat = inputs.reshape(-1, *inputs.shape[2:]).to(device, non_blocking=True)
            paired = torch.cat((flat, torch.flip(flat, dims=(-1,))), dim=0)
            raw = model(paired).float()
            original, flipped_raw = raw.split(flat.shape[0], dim=0)
            flipped_back = torch.flip(flipped_raw, dims=(-1,)).index_select(1, flip_channels)
            views = (0.5 * (original + flipped_back)).reshape(
                batch_size, scale_count, *original.shape[1:]
            )
            heatmaps = aligned_geometry_mean(views)
            geometry_rows = [
                tuple(float(value) for value in row)
                for row in geometries[:, PADDINGS.index(TARGET_PADDING), :].numpy()
            ]
            coordinates, maxima = decode_heatmaps_refined(
                heatmaps.cpu().numpy(), geometry_rows, refinement="quadratic"
            )
            for index in range(batch_size):
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
    prediction_path = output_dir / "distal_loss_seed20261005_geometry_predictions.json"
    write_json_atomic(prediction_path, predictions)
    candidate_predictions = load_json(prediction_path)
    reference_predictions = load_json(REFERENCE)
    candidate_metrics = evaluate_coco_keypoints(dataset, candidate_predictions, max_dets=20)
    reference_metrics = evaluate_coco_keypoints(dataset, reference_predictions, max_dets=20)
    if not math.isclose(reference_metrics["AP"], REFERENCE_AP, rel_tol=0.0, abs_tol=1e-12):
        raise RuntimeError("matched robust reference AP replay differs")
    paired = paired_bootstrap_mean_difference(
        assigned_prediction_oks_by_image(dataset, candidate_predictions),
        assigned_prediction_oks_by_image(dataset, reference_predictions),
        replicates=BOOTSTRAP_REPLICATES,
        confidence_level=0.95,
        seed=BOOTSTRAP_SEED,
    )
    candidate_joint = _joint_errors(dataset, candidate_predictions)
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
    if delta > MATERIAL_AP_DELTA and paired["ci_low"] > 0.0:
        decision = "support_replicate_same_objective_seed20261006"
    elif delta <= 0.0 or paired["ci_high"] < 0.0:
        decision = "refute_reject_distal_weighting_test_coordinate_consistency"
    else:
        decision = "ambiguous_inspect_joint_tradeoffs_without_submission"
    next_actions = {
        "support_replicate_same_objective_seed20261006": "Replicate the unchanged distal-loss objective at seed 20261006; package only after cross-seed support and an informative ensemble comparison.",
        "refute_reject_distal_weighting_test_coordinate_consistency": "Reject distal weighting and keep v11; next test a structural coordinate-consistency auxiliary objective rather than another joint-weight value.",
        "ambiguous_inspect_joint_tradeoffs_without_submission": "Do not submit; inspect wrist/ankle gains against head/torso regressions, then either replicate for uncertainty or move to coordinate consistency.",
    }
    generated_at = datetime.now(timezone.utc).isoformat()
    comparison = {
        "schema_version": 1,
        "candidate": CANDIDATE,
        "generated_at": generated_at,
        "candidate_count": 1,
        "fit_boundary": {"train": ["market", "pa100k"], "development": "peta"},
        "model": {
            "path": str(model_path.relative_to(PROJECT_ROOT)),
            "sha256": model_sha256,
            "manifest_path": str(manifest_path.relative_to(PROJECT_ROOT)),
            "manifest_sha256": sha256_file(manifest_path),
            "seed": SEED,
        },
        "isolated_factor": config["isolated_factor_vs_robust_seed20261005"],
        "joint_loss_weights": config["joint_loss_weights"],
        "interaction_contract": {
            "paddings": list(PADDINGS),
            "scale_weights": [1.0 / 3.0] * 3,
            "target_padding": TARGET_PADDING,
            "horizontal_flip_each_scale": True,
            "coordinate_refinement": "log_quadratic_3point",
            "grid_searched": False,
        },
        "reference": {
            "candidate": "robust_seed_20261005",
            "path": str(REFERENCE.relative_to(PROJECT_ROOT)),
            "sha256": REFERENCE_SHA256,
            "metrics": reference_metrics,
        },
        "result": {
            "metrics": candidate_metrics,
            "prediction_count": len(candidate_predictions),
            "predictions_path": prediction_path.name,
            "predictions_sha256": sha256_file(prediction_path),
            "rescored_from_saved_predictions": True,
            "paired_assigned_oks_candidate_minus_reference": paired,
        },
        "AP_delta_candidate_minus_reference": delta,
        "material_threshold_AP_delta": MATERIAL_AP_DELTA,
        "joint_normalized_error_comparison": joint_comparison,
        "decision": decision,
        "next_action": next_actions[decision],
        "inference_seconds_six_views": elapsed,
        "predictions_per_second_six_views": len(candidate_predictions) / elapsed,
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
        "limitations": [
            "PETA is a reused development surface, not independent confirmation or hidden-test evidence.",
            "Joint residual weighting was motivated by v12 PETA diagnostics; the matched one-seed comparison can reject but not establish broad generalization.",
        ],
        "source_files": {
            str(path.relative_to(PROJECT_ROOT)): sha256_file(path)
            for path in (
                Path(__file__).resolve(),
                PROJECT_ROOT / "scripts/train_highcap_robust_distal_loss_v13.py",
                PROJECT_ROOT / "scripts/evaluate_robust_seed_geometry_ensemble_v7.py",
                PROJECT_ROOT / "src/upar_pose/evaluation.py",
            )
        },
    }
    comparison_path = output_dir / "distal_loss_comparison.json"
    write_json_atomic(comparison_path, comparison)
    manifest_out = output_dir / "manifest.json"
    write_json_atomic(
        manifest_out,
        {
            "schema_version": 1,
            "candidate": CANDIDATE,
            "generated_at": generated_at,
            "artifacts": {
                path.name: {"path": path.name, "sha256": sha256_file(path)}
                for path in (prediction_path, comparison_path)
            },
        },
    )
    event = {
        "id": "upar_highcap_robust_distal_loss_v13_peta",
        "candidate": CANDIDATE,
        "kind": "performance",
        "summary": f"The fixed distal-loss seed-20261005 model reached PETA AP {candidate_metrics['AP']:.6f} versus matched robust AP {reference_metrics['AP']:.6f} ({delta:+.6f}).",
        "issue": "PETA is reused development data and also motivated the fixed joint grouping; this is a matched mechanism screen, not independent confirmation.",
        "next_action": next_actions[decision],
        "candidate_status": "in_progress" if decision.startswith("support") else "rejected" if decision.startswith("refute") else "in_progress",
        "valid": True,
        "guardrails_pass": True,
        "is_best": False,
        "available_at": generated_at,
        "run_id": "20261003T150900+0900_distal_loss_v13_peta",
        "depends_on": ["upar_highcap_robust_factorial_v12_peta"],
        "metric": {
            "name": "COCO_OKS_AP_PETA_DISTAL_LOSS_MATCHED_SEED",
            "direction": "max",
            "evaluation_id": "local-peta-highcap-distal-loss-v13",
            "baseline_id": "robust_seed_20261005",
            "baseline": reference_metrics["AP"],
            "candidate": candidate_metrics["AP"],
            "material_delta": MATERIAL_AP_DELTA,
        },
        "scope": {
            "evaluation_purpose": "candidate_comparison",
            "evaluation_role": "development",
            "execution": "One fixed seed-20261005 distal-loss fit versus matched robust control, unchanged six-view geometry/decoder, all 10,314 PETA images.",
            "review": "Fit manifest/hash, objective weights, frozen reference hash/AP replay, saved candidate replay, joint errors, and paired image-unit uncertainty were checked.",
        },
        "critical_check": {
            "risk": "Objective reweighting could trade head/torso accuracy for distal accuracy without improving official AP.",
            "finding": f"Decision={decision}; AP delta={delta:+.6f}; paired assigned-OKS 95% CI=[{paired['ci_low']:.6f}, {paired['ci_high']:.6f}].",
            "evidence": str(comparison_path.relative_to(PROJECT_ROOT)),
        },
        "visual": False,
        "evidence": [
            str(comparison_path.relative_to(PROJECT_ROOT)),
            str(manifest_out.relative_to(PROJECT_ROOT)),
            str(manifest_path.relative_to(PROJECT_ROOT)),
            str(REFERENCE.relative_to(PROJECT_ROOT)),
        ],
        "artifacts": [str(comparison_path.relative_to(PROJECT_ROOT))],
    }
    if decision.startswith("support"):
        event["package_stage"] = "local_validated"
    write_json_atomic(event_path, event)
    print(json.dumps(comparison, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
