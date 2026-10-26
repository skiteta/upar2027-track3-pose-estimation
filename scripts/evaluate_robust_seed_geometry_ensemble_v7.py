#!/usr/bin/env python3
"""Evaluate the one fixed v7 interaction: robust + two-seed geometry mean.

Each of the three frozen ScratchPoseNet models receives the already selected
padding 1.20/1.25/1.30 and horizontal-flip views.  Heatmaps are flip-corrected,
aligned to padding 1.25 exactly as in v6, averaged within model over geometry,
then averaged across models with fixed 1/3 weights.  There is no weight, scale,
decoder, or model grid in this evaluator.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys
import time
from typing import Any, Sequence

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = (
    PROJECT_ROOT
    / "artifacts/runs/20261003T045300+0900_robust_seed_geometry_ensemble_v7"
)
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from evaluate_seed_geometry_ensemble_v6 import (  # noqa: E402
    InteractionDataset,
    PADDINGS,
    SCALE_MEMBERS,
    TARGET_PADDING,
    prediction,
)
from train_scratch_cnn import dataset_items, load_domain  # noqa: E402
from upar_pose.evaluation import (  # noqa: E402
    assigned_prediction_oks_by_image,
    evaluate_coco_keypoints,
    paired_bootstrap_mean_difference,
)
from upar_pose.flip_tta import FLIP_CHANNELS, decode_heatmaps_refined  # noqa: E402
from upar_pose.geometry_tta_v4 import aligned_weighted_mean  # noqa: E402
from upar_pose.pipeline import load_json, sha256_file, write_json_atomic  # noqa: E402


CANDIDATE = "robust_seed_geometry_ensemble_v7"
MODEL_ROLES = ("standard_seed_20261003", "standard_seed_20261004", "robust_v5")
MODEL_SHA256 = (
    "f1dede691b24d52f4f2d34ce5cf36a96f8a24f347cb1a3934ae364b271de165a",
    "758ed395a12f201e6426223a0b0485f0011fab781d199de98e42f92c5ee446ed",
    "e00823c68ddc103c4f2991943fea32e0d3ae5cca54174f15dc1681fb9c6afe97",
)
MODEL_WEIGHTS = (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)
V6_REFERENCE_SHA256 = (
    "1a4a0f24c51e02fd60c6b10c2b7f58b343805f9c37ba5e7c8ae1d94b2edfe8be"
)
V6_REFERENCE_AP = 0.7802475244838076
BOOTSTRAP_REPLICATES = 2000
BOOTSTRAP_SEED = 20267003


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-standard-a", type=Path, required=True)
    parser.add_argument("--model-standard-b", type=Path, required=True)
    parser.add_argument("--model-robust", type=Path, required=True)
    parser.add_argument("--reference-v6-predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--require-cuda", action="store_true")
    return parser.parse_args()


def aligned_geometry_mean(model_views: torch.Tensor) -> torch.Tensor:
    """Align and equally average one model's three fixed geometry views."""

    if model_views.ndim != 5:
        raise ValueError("expected [batch,paddings,joints,height,width]")
    if model_views.shape[1] != len(PADDINGS):
        raise ValueError(f"expected exactly {len(PADDINGS)} padding views")
    by_padding = {
        padding: model_views[:, index] for index, padding in enumerate(PADDINGS)
    }
    return aligned_weighted_mean(
        by_padding,
        SCALE_MEMBERS,
        target_padding=TARGET_PADDING,
    )


def equal_model_geometry_mean(model_views: Sequence[torch.Tensor]) -> torch.Tensor:
    """Apply frozen geometry alignment per model, then the fixed 1/3 mean."""

    if len(model_views) != len(MODEL_WEIGHTS):
        raise ValueError(f"expected exactly {len(MODEL_WEIGHTS)} model tensors")
    combined: torch.Tensor | None = None
    for views, weight in zip(model_views, MODEL_WEIGHTS, strict=True):
        contribution = aligned_geometry_mean(views) * weight
        combined = contribution if combined is None else combined + contribution
    assert combined is not None
    return combined


def main() -> int:
    args = parse_args()
    if args.batch_size <= 0 or args.workers < 0:
        raise SystemExit("batch size must be positive and workers nonnegative")
    output_dir = args.output_dir.resolve()
    if not output_dir.is_relative_to(RUN_ROOT.resolve()):
        raise SystemExit(f"--output-dir must be inside {RUN_ROOT}")
    output_dir.mkdir(parents=True, exist_ok=False)

    model_paths = tuple(
        path.resolve()
        for path in (args.model_standard_a, args.model_standard_b, args.model_robust)
    )
    actual_hashes = tuple(sha256_file(path) for path in model_paths)
    if actual_hashes != MODEL_SHA256:
        raise RuntimeError(
            f"model hashes do not match frozen standard-A/standard-B/robust parents: "
            f"{actual_hashes}"
        )
    reference_path = args.reference_v6_predictions.resolve()
    reference_hash = sha256_file(reference_path)
    if reference_hash != V6_REFERENCE_SHA256:
        raise RuntimeError("v6 reference prediction hash does not match frozen parent")
    if args.require_cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA was required but is not visible")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = [torch.jit.load(str(path), map_location=device).eval() for path in model_paths]

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
            if scale_count != len(PADDINGS):
                raise RuntimeError("interaction dataset returned an unexpected scale count")
            flat = inputs.reshape(-1, *inputs.shape[2:]).to(device, non_blocking=True)
            paired = torch.cat((flat, torch.flip(flat, dims=(-1,))), dim=0)
            model_views: list[torch.Tensor] = []
            for model in models:
                raw = model(paired).float()
                original, flipped_raw = raw.split(flat.shape[0], dim=0)
                flipped_back = torch.flip(flipped_raw, dims=(-1,)).index_select(
                    1, flip_channels
                )
                model_views.append(
                    (0.5 * (original + flipped_back)).reshape(
                        batch_size, scale_count, *original.shape[1:]
                    )
                )
            heatmaps = equal_model_geometry_mean(model_views)
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
    peak_cuda = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None

    prediction_path = output_dir / "robust_seed_geometry_interaction_predictions.json"
    write_json_atomic(prediction_path, predictions)
    replayed = load_json(prediction_path)
    candidate_metrics = evaluate_coco_keypoints(dataset, replayed, max_dets=20)
    candidate_assigned = assigned_prediction_oks_by_image(dataset, replayed)

    reference_predictions = load_json(reference_path)
    reference_metrics = evaluate_coco_keypoints(dataset, reference_predictions, max_dets=20)
    if not math.isclose(
        reference_metrics["AP"], V6_REFERENCE_AP, rel_tol=0.0, abs_tol=1e-12
    ):
        raise RuntimeError("replayed v6 AP does not match its frozen reference")
    reference_assigned = assigned_prediction_oks_by_image(dataset, reference_predictions)
    paired = paired_bootstrap_mean_difference(
        candidate_assigned,
        reference_assigned,
        replicates=BOOTSTRAP_REPLICATES,
        confidence_level=0.95,
        seed=BOOTSTRAP_SEED,
    )

    summary = {
        "schema_version": 1,
        "candidate": CANDIDATE,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "device": str(device),
        "torch": torch.__version__,
        "models": [
            {
                "role": role,
                "weight": weight,
                "path": str(path.relative_to(PROJECT_ROOT)),
                "sha256": digest,
            }
            for role, weight, path, digest in zip(
                MODEL_ROLES, MODEL_WEIGHTS, model_paths, actual_hashes, strict=True
            )
        ],
        "interaction_contract": {
            "candidate_count": 1,
            "model_weights": list(MODEL_WEIGHTS),
            "model_combination": "equal_heatmap_mean_after_per_model_geometry_alignment",
            "scale_members": [
                {"padding": padding, "weight": weight}
                for padding, weight in SCALE_MEMBERS
            ],
            "target_padding": TARGET_PADDING,
            "horizontal_flip": True,
            "flip_alignment": "none",
            "heatmap_scale_alignment": "bilinear_centered_align_corners_true",
            "coordinate_refinement": "log_quadratic_3point",
            "confidence": "combined_heatmap_mean_scored_joint_max",
        },
        "evaluation_domain": "peta",
        "annotation_sha256": sha256_file(
            PROJECT_ROOT / "data/annotations/specialization/train_peta.json"
        ),
        "reference_v6": {
            "path": str(reference_path.relative_to(PROJECT_ROOT)),
            "sha256": reference_hash,
            "metrics": reference_metrics,
        },
        "result": {
            "metrics": candidate_metrics,
            "paired_assigned_oks_vs_v6": paired,
            "prediction_count": len(replayed),
            "predictions_path": prediction_path.name,
            "predictions_sha256": sha256_file(prediction_path),
            "rescored_from_saved_predictions": True,
        },
        "candidate_beats_v6_AP": candidate_metrics["AP"] > reference_metrics["AP"],
        "candidate_AP_delta_vs_v6": candidate_metrics["AP"] - reference_metrics["AP"],
        "inference_seconds": elapsed,
        "predictions_per_second": len(replayed) / elapsed,
        "forward_views_per_image": 2 * len(PADDINGS) * len(models),
        "peak_cuda_allocated_bytes": peak_cuda,
        "source_files": {
            str(path.relative_to(PROJECT_ROOT)): sha256_file(path)
            for path in (
                Path(__file__).resolve(),
                PROJECT_ROOT / "scripts/evaluate_seed_geometry_ensemble_v6.py",
                PROJECT_ROOT / "src/upar_pose/geometry_tta_v4.py",
                PROJECT_ROOT / "src/upar_pose/flip_tta.py",
                PROJECT_ROOT / "src/upar_pose/evaluation.py",
            )
        },
    }
    write_json_atomic(output_dir / "robust_seed_geometry_interaction_comparison.json", summary)
    manifest = {
        "schema_version": 1,
        "candidate": CANDIDATE,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "artifacts": {
            name: {"path": name, "sha256": sha256_file(output_dir / name)}
            for name in (
                "robust_seed_geometry_interaction_predictions.json",
                "robust_seed_geometry_interaction_comparison.json",
            )
        },
    }
    write_json_atomic(output_dir / "manifest.json", manifest)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
