#!/usr/bin/env python3
"""Test the single precommitted seed-ensemble x geometry-TTA interaction."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
from typing import Any, Sequence

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from train_scratch_cnn import (  # noqa: E402
    SCORED_KEYPOINTS,
    crop_person,
    dataset_items,
    image_path,
    load_domain,
)
from upar_pose.evaluation import (  # noqa: E402
    assigned_prediction_oks_by_image,
    evaluate_coco_keypoints,
    paired_bootstrap_mean_difference,
)
from upar_pose.flip_tta import FLIP_CHANNELS, decode_heatmaps_refined  # noqa: E402
from upar_pose.geometry_tta_v4 import (  # noqa: E402
    aligned_weighted_mean,
    scaled_crop_geometry,
)
from upar_pose.pipeline import load_json, sha256_file, write_json_atomic  # noqa: E402
from upar_pose.scratch_cnn import normalize_image  # noqa: E402


# This is one fixed interaction, not a new weight/scale grid.  Equal seed
# weights come from v3's selected result; equal scale weights and these three
# paddings come from v4's selected result.  The center-only replay is solely a
# numerical parent sentinel and is not another candidate.
PADDINGS = (1.20, 1.25, 1.30)
SCALE_MEMBERS = ((1.20, 1.0), (1.25, 1.0), (1.30, 1.0))
TARGET_PADDING = 1.25
MODEL_SHA256 = (
    "f1dede691b24d52f4f2d34ce5cf36a96f8a24f347cb1a3934ae364b271de165a",
    "758ed395a12f201e6426223a0b0485f0011fab781d199de98e42f92c5ee446ed",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-a", type=Path, required=True)
    parser.add_argument("--model-b", type=Path, required=True)
    parser.add_argument("--reference-predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--require-cuda", action="store_true")
    return parser.parse_args()


class InteractionDataset(Dataset):
    def __init__(self, items: Sequence[dict[str, Any]]) -> None:
        self.items = list(items)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int):
        item = self.items[index]
        with Image.open(image_path(item["file_name"])) as source:
            image = source.convert("RGB")
        geometries = [scaled_crop_geometry(item["box"], padding) for padding in PADDINGS]
        inputs = torch.stack(
            [
                torch.from_numpy(
                    normalize_image(np.asarray(crop_person(image, geometry), dtype=np.uint8))
                )
                for geometry in geometries
            ]
        )
        return (
            inputs,
            torch.tensor(geometries, dtype=torch.float64),
            torch.tensor(item["image_id"], dtype=torch.int64),
            torch.tensor(item["annotation_id"], dtype=torch.int64),
        )


def prediction(
    coords: np.ndarray,
    maxima: np.ndarray,
    image_id: int,
    annotation_id: int,
) -> dict[str, Any]:
    return {
        "image_id": image_id,
        "category_id": 1,
        "keypoints": [
            value
            for x_coord, y_coord in coords
            for value in (float(x_coord), float(y_coord), 1.0)
        ],
        "score": max(
            0.01,
            float(np.mean(np.clip(maxima[list(SCORED_KEYPOINTS)], 0.0, 1.0))),
        ),
        "source_annotation_id": annotation_id,
    }


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    if not output_dir.is_relative_to(PROJECT_ROOT):
        raise SystemExit("--output-dir must be inside the project")
    output_dir.mkdir(parents=True, exist_ok=False)
    model_paths = (args.model_a.resolve(), args.model_b.resolve())
    actual_hashes = tuple(sha256_file(path) for path in model_paths)
    if actual_hashes != MODEL_SHA256:
        raise RuntimeError(
            f"model hashes do not match frozen seed A/B parents: {actual_hashes}"
        )
    if args.require_cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA was required but is not visible")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = [torch.jit.load(str(path), map_location=device).eval() for path in model_paths]
    dataset = load_domain("peta")
    loader = DataLoader(
        InteractionDataset(dataset_items(dataset, require_labels=False)),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.workers > 0,
        prefetch_factor=2 if args.workers > 0 else None,
    )
    outputs: dict[str, list[dict[str, Any]]] = {
        "seed_ensemble_center_replay": [],
        "seed_geometry_interaction": [],
    }
    flip_channels = torch.as_tensor(FLIP_CHANNELS, dtype=torch.long, device=device)
    started = time.monotonic()
    with torch.inference_mode():
        for inputs, geometries, image_ids, annotation_ids in loader:
            batch_size, scale_count = inputs.shape[:2]
            flat = inputs.reshape(-1, *inputs.shape[2:]).to(device, non_blocking=True)
            paired = torch.cat((flat, torch.flip(flat, dims=(-1,))), dim=0)
            combined_by_seed: list[torch.Tensor] = []
            for model in models:
                raw = model(paired).float()
                original, flipped_raw = raw.split(flat.shape[0], dim=0)
                flipped_back = torch.flip(flipped_raw, dims=(-1,)).index_select(
                    1, flip_channels
                )
                combined_by_seed.append(
                    (0.5 * (original + flipped_back)).reshape(
                        batch_size, scale_count, *original.shape[1:]
                    )
                )
            seed_mean = 0.5 * (combined_by_seed[0] + combined_by_seed[1])
            by_padding = {padding: seed_mean[:, i] for i, padding in enumerate(PADDINGS)}
            variant_heatmaps = {
                "seed_ensemble_center_replay": by_padding[TARGET_PADDING],
                "seed_geometry_interaction": aligned_weighted_mean(
                    by_padding,
                    SCALE_MEMBERS,
                    target_padding=TARGET_PADDING,
                ),
            }
            geometry_rows = [
                tuple(float(value) for value in row)
                for row in geometries[:, PADDINGS.index(TARGET_PADDING), :].numpy()
            ]
            for name, heatmaps in variant_heatmaps.items():
                coords, maxima = decode_heatmaps_refined(
                    heatmaps.cpu().numpy(), geometry_rows, refinement="quadratic"
                )
                for index in range(batch_size):
                    outputs[name].append(
                        prediction(
                            coords[index],
                            maxima[index],
                            int(image_ids[index]),
                            int(annotation_ids[index]),
                        )
                    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.monotonic() - started

    reference_path = args.reference_predictions.resolve()
    reference_predictions = load_json(reference_path)
    reference_metrics = evaluate_coco_keypoints(dataset, reference_predictions, max_dets=20)
    reference_assigned = assigned_prediction_oks_by_image(dataset, reference_predictions)
    results: dict[str, Any] = {}
    for name, rows in outputs.items():
        path = output_dir / f"{name}_predictions.json"
        write_json_atomic(path, rows)
        replayed = load_json(path)
        assigned = assigned_prediction_oks_by_image(dataset, replayed)
        results[name] = {
            "metrics": evaluate_coco_keypoints(dataset, replayed, max_dets=20),
            "paired_assigned_oks_vs_v3": paired_bootstrap_mean_difference(
                assigned,
                reference_assigned,
                replicates=2000,
                confidence_level=0.95,
                seed=20266003,
            ),
            "prediction_count": len(replayed),
            "predictions_path": path.name,
            "predictions_sha256": sha256_file(path),
            "rescored_from_saved_predictions": True,
        }
    candidate_metrics = results["seed_geometry_interaction"]["metrics"]
    summary = {
        "schema_version": 1,
        "candidate": "seed_geometry_ensemble_v6",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "device": str(device),
        "torch": torch.__version__,
        "models": [
            {
                "role": "seed_a_20261003" if index == 0 else "seed_b_20261004",
                "weight": 0.5,
                "path": str(path.relative_to(PROJECT_ROOT)),
                "sha256": actual_hashes[index],
            }
            for index, path in enumerate(model_paths)
        ],
        "interaction_contract": {
            "candidate_count": 1,
            "scale_members": [
                {"padding": padding, "weight": weight} for padding, weight in SCALE_MEMBERS
            ],
            "target_padding": TARGET_PADDING,
            "horizontal_flip": True,
            "flip_alignment": "none",
            "heatmap_scale_alignment": "bilinear_centered_align_corners_true",
            "coordinate_refinement": "log_quadratic_3point",
            "confidence": "combined_heatmap_mean_scored_joint_max",
        },
        "reference": {
            "role": "v3_seed_ensemble_w050",
            "path": str(reference_path.relative_to(PROJECT_ROOT)),
            "sha256": sha256_file(reference_path),
            "metrics": reference_metrics,
        },
        "evaluation_domain": "peta",
        "annotation_sha256": sha256_file(
            PROJECT_ROOT / "data/annotations/specialization/train_peta.json"
        ),
        "inference_seconds_shared_forwards": elapsed,
        "forward_views_per_image": 2 * len(models) * len(PADDINGS),
        "candidate_beats_v3_AP": candidate_metrics["AP"] > reference_metrics["AP"],
        "candidate_AP_delta_vs_v3": candidate_metrics["AP"] - reference_metrics["AP"],
        "results": results,
    }
    write_json_atomic(output_dir / "seed_geometry_interaction_comparison.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
