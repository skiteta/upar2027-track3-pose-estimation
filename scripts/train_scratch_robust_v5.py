#!/usr/bin/env python3
"""Train one isolated ScratchPoseNet robustness hypothesis: bbox jitter.

The architecture, heatmap target, optimizer, schedule, color augmentation, and
flip augmentation are held equal to scratch-cnn v1.  The only experimental
factor is mild random translation/scale of the padded person crop during fit.
Evaluation is fixed to PETA and reports both v1-style single-view decoding and
the selected v2 flip/quadratic decoder from saved predictions.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import random
import sys
import time
from typing import Any, Sequence

import numpy as np
from PIL import Image, ImageEnhance
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = PROJECT_ROOT / "artifacts/runs/20261003T043000+0900_training_objective_v5"
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from train_scratch_cnn import (  # noqa: E402
    PoseInferenceDataset,
    crop_person,
    dataset_items,
    image_path,
    load_domain,
    seed_everything,
)
from upar_pose.evaluation import (  # noqa: E402
    assigned_prediction_oks_by_image,
    evaluate_coco_keypoints,
    paired_bootstrap_mean_difference,
)
from upar_pose.flip_tta import decode_heatmaps_refined, flip_back_heatmaps  # noqa: E402
from upar_pose.pipeline import load_json, sha256_file, write_json_atomic  # noqa: E402
from upar_pose.scratch_cnn import (  # noqa: E402
    HEATMAP_HEIGHT,
    HEATMAP_WIDTH,
    INPUT_HEIGHT,
    INPUT_WIDTH,
    NUM_KEYPOINTS,
    PADDING_FACTOR,
    SCORED_KEYPOINTS,
    ScratchPoseNet,
    crop_geometry,
    make_heatmaps,
    normalize_image,
    transform_keypoints,
)


CANDIDATE = "upar_scratch_bbox_jitter_v5"
TRAIN_DOMAINS = ("market", "pa100k")
EVAL_DOMAIN = "peta"
CENTER_JITTER_FRACTION = 0.04
SCALE_JITTER_FRACTION = 0.10
V1_PREDICTIONS = (
    PROJECT_ROOT
    / "artifacts/runs/20261003T024500+0900_upar_scratch_cnn_v1/holdout_peta/peta_predictions.json"
)
V2_PREDICTIONS = (
    PROJECT_ROOT
    / "artifacts/runs/20261003T033500+0900_upar_scratch_cnn_flip_tta_v2/"
    "holdout_peta_variants/flip_quadratic_none_combined_predictions.json"
)
V1_AP = 0.7280693003452431
V2_AP = 0.7578222316708247


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=0.003)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--seed", type=int, default=20261003)
    parser.add_argument("--center-jitter", type=float, default=CENTER_JITTER_FRACTION)
    parser.add_argument("--scale-jitter", type=float, default=SCALE_JITTER_FRACTION)
    parser.add_argument("--max-train-examples", type=int)
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--runtime-iterations", type=int, default=200)
    parser.add_argument("--require-cuda", action="store_true")
    return parser.parse_args()


def jitter_crop_geometry(
    box: Sequence[float],
    *,
    center_dx: float,
    center_dy: float,
    scale: float,
) -> tuple[float, float, float, float]:
    """Apply normalized translation and isotropic scale to the v1 crop."""

    if not all(math.isfinite(value) for value in (center_dx, center_dy, scale)):
        raise ValueError("jitter parameters must be finite")
    if scale <= 0.0:
        raise ValueError("jitter scale must be positive")
    left, top, width, height = crop_geometry(box)
    center_x = left + 0.5 * width + center_dx * width
    center_y = top + 0.5 * height + center_dy * height
    width *= scale
    height *= scale
    return center_x - 0.5 * width, center_y - 0.5 * height, width, height


class BBoxJitterTrainingDataset(Dataset):
    """v1 fit pipeline with only padded-crop geometry jitter added."""

    def __init__(
        self,
        items: Sequence[dict[str, Any]],
        *,
        center_jitter: float,
        scale_jitter: float,
    ) -> None:
        if not 0.0 <= center_jitter <= 0.25:
            raise ValueError("center_jitter must be in [0, 0.25]")
        if not 0.0 <= scale_jitter < 0.5:
            raise ValueError("scale_jitter must be in [0, 0.5)")
        self.items = list(items)
        self.center_jitter = center_jitter
        self.scale_jitter = scale_jitter

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int):
        item = self.items[index]
        with Image.open(image_path(item["file_name"])) as source:
            image = source.convert("RGB")
        # Keep the Python RNG stream used by v1 flip/colour augmentation intact.
        # NumPy is independently seeded by both ``seed_everything`` and PyTorch
        # DataLoader workers, so these three new draws isolate the bbox factor.
        dx = float(np.random.uniform(-self.center_jitter, self.center_jitter))
        dy = float(np.random.uniform(-self.center_jitter, self.center_jitter))
        scale = float(np.random.uniform(1.0 - self.scale_jitter, 1.0 + self.scale_jitter))
        geometry = jitter_crop_geometry(
            item["box"], center_dx=dx, center_dy=dy, scale=scale
        )
        crop = crop_person(image, geometry)
        flipped = random.random() < 0.5
        if flipped:
            crop = crop.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        crop = ImageEnhance.Brightness(crop).enhance(random.uniform(0.8, 1.2))
        crop = ImageEnhance.Contrast(crop).enhance(random.uniform(0.8, 1.2))
        pixels = normalize_image(np.asarray(crop, dtype=np.uint8))
        transformed = transform_keypoints(item["keypoints"], geometry, flipped=flipped)
        heatmaps, weights = make_heatmaps(transformed)
        return torch.from_numpy(pixels), torch.from_numpy(heatmaps), torch.from_numpy(weights)


def bounded_domain_subset(
    domain_items: MappingLike, max_examples: int | None
) -> list[dict[str, Any]]:
    """Build the fixed two-domain fit set; smoke subsets keep both domains."""

    market = list(domain_items["market"])
    pa100k = list(domain_items["pa100k"])
    if max_examples is None:
        return market + pa100k
    if max_examples < 2:
        raise ValueError("max_train_examples must be at least 2")
    market_count = min(len(market), max_examples // 2)
    pa_count = min(len(pa100k), max_examples - market_count)
    if market_count + pa_count < max_examples:
        market_count = min(len(market), max_examples - pa_count)
    return market[:market_count] + pa100k[:pa_count]


# A tiny alias keeps type annotations readable without importing typing.Mapping
# into TorchScript-facing code paths.
MappingLike = dict[str, Sequence[dict[str, Any]]]


def train_model(
    model: nn.Module,
    items: Sequence[dict[str, Any]],
    *,
    device: torch.device,
    output_dir: Path,
    epochs: int,
    batch_size: int,
    workers: int,
    learning_rate: float,
    weight_decay: float,
    center_jitter: float,
    scale_jitter: float,
    seed: int,
) -> list[dict[str, Any]]:
    loader = DataLoader(
        BBoxJitterTrainingDataset(
            items,
            center_jitter=center_jitter,
            scale_jitter=scale_jitter,
        ),
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        persistent_workers=workers > 0,
        drop_last=False,
        prefetch_factor=2 if workers > 0 else None,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=learning_rate,
        epochs=epochs,
        steps_per_epoch=len(loader),
        pct_start=0.15,
        div_factor=10.0,
        final_div_factor=100.0,
    )
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    history: list[dict[str, Any]] = []
    history_path = output_dir / "training_history.jsonl"
    peak_allocated = 0

    for epoch in range(1, epochs + 1):
        model.train()
        started = time.monotonic()
        total_loss = 0.0
        seen = 0
        for inputs, targets, weights in loader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            weights = weights.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type, dtype=torch.float16, enabled=use_amp
            ):
                outputs = model(inputs)
                per_joint = torch.square(outputs - targets).mean(dim=(2, 3))
                loss = (per_joint * weights).sum() / weights.sum().clamp_min(1.0)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            count = int(inputs.shape[0])
            total_loss += float(loss.detach()) * count
            seen += count
            if device.type == "cuda":
                peak_allocated = max(peak_allocated, torch.cuda.max_memory_allocated(device))
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = time.monotonic() - started
        row = {
            "epoch": epoch,
            "train_loss": total_loss / seen,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "elapsed_seconds": elapsed,
            "examples": seen,
            "examples_per_second": seen / elapsed,
            "peak_cuda_allocated_bytes_so_far": peak_allocated if device.type == "cuda" else None,
            "observed_at": datetime.now(timezone.utc).isoformat(),
        }
        history.append(row)
        with history_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        torch.save(
            {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "candidate": CANDIDATE,
            },
            output_dir / "last_checkpoint.pt",
        )
        print(json.dumps(row), flush=True)
    return history


def _prediction(
    coordinates: np.ndarray,
    maxima: np.ndarray,
    image_id: int,
    annotation_id: int,
) -> dict[str, Any]:
    keypoints = [
        value
        for x_coord, y_coord in coordinates
        for value in (float(x_coord), float(y_coord), 1.0)
    ]
    score = float(np.mean(np.clip(maxima[list(SCORED_KEYPOINTS)], 0.0, 1.0)))
    return {
        "image_id": image_id,
        "category_id": 1,
        "keypoints": keypoints,
        "score": max(0.01, score),
        "source_annotation_id": annotation_id,
    }


def _score_saved(
    dataset: dict[str, Any],
    prediction_path: Path,
    *,
    v1_assigned: MappingLikeAssigned,
    v2_assigned: MappingLikeAssigned,
    bootstrap_seed: int,
) -> dict[str, Any]:
    replayed = load_json(prediction_path)
    assigned = assigned_prediction_oks_by_image(dataset, replayed)
    return {
        "metrics": evaluate_coco_keypoints(dataset, replayed, max_dets=20),
        "prediction_count": len(replayed),
        "predictions_path": prediction_path.name,
        "predictions_sha256": sha256_file(prediction_path),
        "rescored_from_saved_predictions": True,
        "paired_assigned_oks_vs_v1": paired_bootstrap_mean_difference(
            assigned,
            v1_assigned,
            replicates=2000,
            confidence_level=0.95,
            seed=bootstrap_seed,
        ),
        "paired_assigned_oks_vs_v2": paired_bootstrap_mean_difference(
            assigned,
            v2_assigned,
            replicates=2000,
            confidence_level=0.95,
            seed=bootstrap_seed + 1,
        ),
    }


MappingLikeAssigned = dict[int, float]


def evaluate_model(
    model: nn.Module,
    dataset: dict[str, Any],
    *,
    device: torch.device,
    output_dir: Path,
    batch_size: int,
    workers: int,
) -> dict[str, Any]:
    items = dataset_items(dataset, require_labels=False)
    loader = DataLoader(
        PoseInferenceDataset(items),
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        persistent_workers=workers > 0,
        prefetch_factor=2 if workers > 0 else None,
    )
    model.eval()
    single_predictions: list[dict[str, Any]] = []
    flip_predictions: list[dict[str, Any]] = []
    started = time.monotonic()
    with torch.inference_mode():
        for inputs, geometries, image_ids, annotation_ids in loader:
            inputs = inputs.to(device, non_blocking=True)
            original = model(inputs).float().cpu().numpy()
            flipped_raw = model(torch.flip(inputs, dims=(3,))).float().cpu().numpy()
            geometry_rows = [
                tuple(float(value) for value in row) for row in geometries.numpy()
            ]
            single_coords, single_maxima = decode_heatmaps_refined(
                original, geometry_rows, refinement="quarter"
            )
            combined = 0.5 * (original + flip_back_heatmaps(flipped_raw, shift="none"))
            flip_coords, flip_maxima = decode_heatmaps_refined(
                combined, geometry_rows, refinement="quadratic"
            )
            for index in range(len(image_ids)):
                identity = (int(image_ids[index]), int(annotation_ids[index]))
                single_predictions.append(
                    _prediction(single_coords[index], single_maxima[index], *identity)
                )
                flip_predictions.append(
                    _prediction(flip_coords[index], flip_maxima[index], *identity)
                )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    inference_seconds = time.monotonic() - started

    single_path = output_dir / "peta_single_predictions.json"
    flip_path = output_dir / "peta_flip_quadratic_predictions.json"
    write_json_atomic(single_path, single_predictions)
    write_json_atomic(flip_path, flip_predictions)
    v1_replayed = load_json(V1_PREDICTIONS)
    v2_replayed = load_json(V2_PREDICTIONS)
    v1_metrics = evaluate_coco_keypoints(dataset, v1_replayed, max_dets=20)
    v2_metrics = evaluate_coco_keypoints(dataset, v2_replayed, max_dets=20)
    if not math.isclose(v1_metrics["AP"], V1_AP, rel_tol=0.0, abs_tol=1e-12):
        raise RuntimeError("replayed v1 AP does not match its frozen reference")
    if not math.isclose(v2_metrics["AP"], V2_AP, rel_tol=0.0, abs_tol=1e-12):
        raise RuntimeError("replayed v2 AP does not match its frozen reference")
    v1_assigned = assigned_prediction_oks_by_image(dataset, v1_replayed)
    v2_assigned = assigned_prediction_oks_by_image(dataset, v2_replayed)
    variants = {
        "single_quarter": _score_saved(
            dataset,
            single_path,
            v1_assigned=v1_assigned,
            v2_assigned=v2_assigned,
            bootstrap_seed=20265003,
        ),
        "flip_quadratic_none_combined": _score_saved(
            dataset,
            flip_path,
            v1_assigned=v1_assigned,
            v2_assigned=v2_assigned,
            bootstrap_seed=20265013,
        ),
    }
    selected_ap = variants["flip_quadratic_none_combined"]["metrics"]["AP"]
    result = {
        "schema_version": 1,
        "candidate": CANDIDATE,
        "evaluation_domain": EVAL_DOMAIN,
        "inference_seconds_two_views": inference_seconds,
        "predictions_per_second_two_views": len(items) / inference_seconds,
        "reference": {
            "v1_AP": V1_AP,
            "v1_replayed_metrics": v1_metrics,
            "v1_predictions_sha256": sha256_file(V1_PREDICTIONS),
            "v2_AP": V2_AP,
            "v2_replayed_metrics": v2_metrics,
            "v2_predictions_sha256": sha256_file(V2_PREDICTIONS),
        },
        "variants": variants,
        "primary_variant": "flip_quadratic_none_combined",
        "primary_AP": selected_ap,
        "delta_AP_vs_v1": selected_ap - V1_AP,
        "delta_AP_vs_v2": selected_ap - V2_AP,
        "beats_v2_AP": selected_ap > V2_AP,
    }
    write_json_atomic(output_dir / "peta_comparison.json", result)
    return result


def export_model(model: nn.Module, output_dir: Path, config: dict[str, Any]) -> Path:
    cpu_model = model.eval().cpu()
    example = torch.zeros(1, 3, INPUT_HEIGHT, INPUT_WIDTH)
    traced = torch.jit.trace(cpu_model, example, strict=True)
    frozen = torch.jit.freeze(traced)
    path = output_dir / "model.torchscript"
    frozen.save(str(path))
    reloaded = torch.jit.load(str(path), map_location="cpu").eval()
    output_shape = tuple(reloaded(example).shape)
    if output_shape != (1, NUM_KEYPOINTS, HEATMAP_HEIGHT, HEATMAP_WIDTH):
        raise RuntimeError(f"unexpected TorchScript output shape: {output_shape}")
    metadata = {
        "schema_version": 1,
        "model_type": CANDIDATE,
        "architecture": "ScratchPoseNet(residual_blocks=4), unchanged from v1/v2",
        "input_width": INPUT_WIDTH,
        "input_height": INPUT_HEIGHT,
        "heatmap_width": HEATMAP_WIDTH,
        "heatmap_height": HEATMAP_HEIGHT,
        "num_keypoints": NUM_KEYPOINTS,
        "padding_factor": PADDING_FACTOR,
        "image_mean": [0.5, 0.5, 0.5],
        "image_std": [0.25, 0.25, 0.25],
        "score_excludes_keypoint_indices": [3, 4],
        "external_weights": False,
        "training_config": config,
        "torchscript_output_shape_verified": list(output_shape),
    }
    write_json_atomic(output_dir / "model_meta.json", metadata)
    return path


def benchmark_torchscript(
    model_path: Path,
    *,
    device: torch.device,
    iterations: int,
    output_dir: Path,
) -> dict[str, Any]:
    if iterations <= 0:
        raise ValueError("runtime iterations must be positive")
    model = torch.jit.load(str(model_path), map_location=device).eval()
    example = torch.zeros(1, 3, INPUT_HEIGHT, INPUT_WIDTH, device=device)
    warmup = min(20, iterations)
    with torch.inference_mode():
        for _ in range(warmup):
            model(example)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        started = time.monotonic()
        for _ in range(iterations):
            model(example)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = time.monotonic() - started
    result = {
        "device": str(device),
        "iterations": iterations,
        "elapsed_seconds": elapsed,
        "milliseconds_per_crop": elapsed / iterations * 1000.0,
        "crops_per_second": iterations / elapsed,
        "peak_cuda_allocated_bytes": (
            torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
        ),
        "concurrent_gpu_jobs_may_affect_timing": True,
        "model_sha256": sha256_file(model_path),
    }
    write_json_atomic(output_dir / "torchscript_runtime.json", result)
    return result


def jitter_distribution(seed: int, center_jitter: float, scale_jitter: float) -> dict[str, Any]:
    rng = random.Random(seed)
    rows = [
        (
            rng.uniform(-center_jitter, center_jitter),
            rng.uniform(-center_jitter, center_jitter),
            rng.uniform(1.0 - scale_jitter, 1.0 + scale_jitter),
        )
        for _ in range(10000)
    ]
    array = np.asarray(rows, dtype=np.float64)
    return {
        "draw_count": len(rows),
        "center_dx_fraction_min_max_mean": [
            float(array[:, 0].min()),
            float(array[:, 0].max()),
            float(array[:, 0].mean()),
        ],
        "center_dy_fraction_min_max_mean": [
            float(array[:, 1].min()),
            float(array[:, 1].max()),
            float(array[:, 1].mean()),
        ],
        "isotropic_scale_min_max_mean": [
            float(array[:, 2].min()),
            float(array[:, 2].max()),
            float(array[:, 2].mean()),
        ],
    }


def main() -> int:
    args = parse_args()
    if args.epochs <= 0 or args.batch_size <= 0 or args.eval_batch_size <= 0:
        raise SystemExit("epochs and batch sizes must be positive")
    output_dir = args.output_dir.resolve()
    if not output_dir.is_relative_to(RUN_ROOT.resolve()):
        raise SystemExit(f"--output-dir must be inside {RUN_ROOT}")
    output_dir.mkdir(parents=True, exist_ok=False)
    seed_everything(args.seed)
    if args.require_cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA was required but is not visible")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_float32_matmul_precision("high")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    datasets = {domain: load_domain(domain) for domain in (*TRAIN_DOMAINS, EVAL_DOMAIN)}
    by_domain: MappingLike = {
        domain: dataset_items(datasets[domain], require_labels=True)
        for domain in TRAIN_DOMAINS
    }
    train_items = bounded_domain_subset(by_domain, args.max_train_examples)
    model = ScratchPoseNet().to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    config = {
        "candidate": CANDIDATE,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "hypothesis": (
            "Mild padded-bbox translation/scale jitter improves cross-domain crop robustness "
            "without changing architecture, objective, data, optimizer, or decoder."
        ),
        "isolated_factor": "training_crop_bbox_jitter",
        "held_constant_vs_v1": [
            "ScratchPoseNet architecture and initialization",
            "heatmap MSE objective and targets",
            "AdamW + OneCycleLR schedule",
            "horizontal flip and brightness/contrast augmentation",
            "Market+PA100K fit boundary",
            "seed, epochs, batch size, learning rate, weight decay",
        ],
        "jitter": {
            "center_uniform_fraction": [-args.center_jitter, args.center_jitter],
            "isotropic_scale_uniform": [1.0 - args.scale_jitter, 1.0 + args.scale_jitter],
            "relative_to": "aspect-corrected 1.25x padded bbox crop",
            "distribution_check": jitter_distribution(
                args.seed, args.center_jitter, args.scale_jitter
            ),
        },
        "train_domains": list(TRAIN_DOMAINS),
        "eval_domain": None if args.skip_eval else EVAL_DOMAIN,
        "domain_train_examples": {domain: len(by_domain[domain]) for domain in TRAIN_DOMAINS},
        "train_examples_used": len(train_items),
        "max_train_examples": args.max_train_examples,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "eval_batch_size": args.eval_batch_size,
        "workers": args.workers,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "seed": args.seed,
        "parameter_count": parameter_count,
        "external_weights": False,
        "external_training_data": False,
        "source_files": {
            str(path.relative_to(PROJECT_ROOT)): sha256_file(path)
            for path in (
                Path(__file__).resolve(),
                PROJECT_ROOT / "scripts/train_scratch_cnn.py",
                PROJECT_ROOT / "src/upar_pose/scratch_cnn.py",
                PROJECT_ROOT / "src/upar_pose/flip_tta.py",
                PROJECT_ROOT / "src/upar_pose/evaluation.py",
            )
        },
        "annotation_files": {
            domain: {
                "path": f"data/annotations/specialization/train_{domain}.json",
                "sha256": sha256_file(
                    PROJECT_ROOT / f"data/annotations/specialization/train_{domain}.json"
                ),
            }
            for domain in (*TRAIN_DOMAINS, EVAL_DOMAIN)
        },
        "device": str(device),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "device_name": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
    }
    write_json_atomic(output_dir / "config.json", config)
    started = time.monotonic()
    history = train_model(
        model,
        train_items,
        device=device,
        output_dir=output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        workers=args.workers,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        center_jitter=args.center_jitter,
        scale_jitter=args.scale_jitter,
        seed=args.seed,
    )
    evaluation = None
    if not args.skip_eval:
        evaluation = evaluate_model(
            model,
            datasets[EVAL_DOMAIN],
            device=device,
            output_dir=output_dir,
            batch_size=args.eval_batch_size,
            workers=args.workers,
        )
        print(json.dumps({"evaluation": evaluation}, ensure_ascii=False), flush=True)
    model_path = export_model(model, output_dir, config)
    runtime = benchmark_torchscript(
        model_path,
        device=device,
        iterations=args.runtime_iterations,
        output_dir=output_dir,
    )
    elapsed = time.monotonic() - started
    curve = {
        "epoch": [row["epoch"] for row in history],
        "train_loss": [row["train_loss"] for row in history],
        "learning_rate": [row["learning_rate"] for row in history],
        "elapsed_seconds": [row["elapsed_seconds"] for row in history],
    }
    write_json_atomic(output_dir / "training_curve.json", curve)
    decision = None
    if evaluation is not None:
        decision = {
            "criterion": "PETA saved-prediction flip/quadratic AP strictly exceeds v2 AP",
            "candidate_AP": evaluation["primary_AP"],
            "v2_AP": V2_AP,
            "delta_AP": evaluation["delta_AP_vs_v2"],
            "adopt": evaluation["beats_v2_AP"],
            "interpretation": (
                "Adopt bbox-jitter fit as the stronger held-out candidate."
                if evaluation["beats_v2_AP"]
                else "Reject bbox-jitter preset; it did not exceed the frozen v2 held-out AP."
            ),
        }
        write_json_atomic(output_dir / "decision.json", decision)
    artifact_names = [
        "config.json",
        "last_checkpoint.pt",
        "model.torchscript",
        "model_meta.json",
        "training_history.jsonl",
        "training_curve.json",
        "torchscript_runtime.json",
    ]
    if evaluation is not None:
        artifact_names.extend(
            [
                "peta_single_predictions.json",
                "peta_flip_quadratic_predictions.json",
                "peta_comparison.json",
                "decision.json",
            ]
        )
    manifest = {
        "schema_version": 1,
        "candidate": CANDIDATE,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "config": config,
        "history": history,
        "evaluation": evaluation,
        "runtime": runtime,
        "decision": decision,
        "artifacts": {
            name: {"path": name, "sha256": sha256_file(output_dir / name)}
            for name in artifact_names
        },
        "elapsed_seconds": elapsed,
    }
    write_json_atomic(output_dir / "manifest.json", manifest)
    print(
        json.dumps(
            {
                "manifest": str(output_dir / "manifest.json"),
                "evaluation": evaluation,
                "elapsed_seconds": elapsed,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
