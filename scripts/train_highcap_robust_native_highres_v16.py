#!/usr/bin/env python3
"""Fit the robust high-capacity pose model natively at 240x320 resolution."""

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
from torch.utils.data import Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = PROJECT_ROOT / "artifacts/runs/20261003T160000+0900_native_highres_v16"
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import train_scratch_hrnet_v4 as highcap  # noqa: E402
from train_scratch_cnn import dataset_items, image_path, load_domain  # noqa: E402
from train_scratch_robust_v5 import (  # noqa: E402
    CENTER_JITTER_FRACTION,
    SCALE_JITTER_FRACTION,
    jitter_crop_geometry,
    jitter_distribution,
)
from upar_pose.pipeline import sha256_file, write_json_atomic  # noqa: E402
from upar_pose.scratch_cnn import FLIP_PAIRS, NUM_KEYPOINTS, normalize_image  # noqa: E402
from upar_pose.scratch_hrnet_v4 import ScratchSkipPoseNetV4, parameter_count  # noqa: E402

CANDIDATE = "upar_highcap_robust_native_highres_v16"
TRAIN_DOMAINS = ("market", "pa100k")
SEED = 20261005
INPUT_WIDTH = 240
INPUT_HEIGHT = 320
HEATMAP_WIDTH = 60
HEATMAP_HEIGHT = 80
HEATMAP_SIGMA = 2.25


class NativeHighResPoseNet(ScratchSkipPoseNetV4):
    """Identical topology without the parent's fixed eager output-size sentinel."""

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        stem = self.stem(inputs)
        quarter = self.quarter(stem)
        eighth = self.eighth(quarter)
        sixteenth = self.sixteenth(eighth)
        decoded_eighth = self.decode_eighth(sixteenth, eighth)
        return self.head(self.decode_quarter(decoded_eighth, quarter))


def transform_keypoints_highres(
    keypoints: np.ndarray,
    geometry: tuple[float, float, float, float],
    *,
    flipped: bool,
) -> np.ndarray:
    left, top, width, height = geometry
    transformed = keypoints.astype(np.float32, copy=True)
    transformed[:, 0] = (transformed[:, 0] - left) / width * (INPUT_WIDTH - 1.0)
    transformed[:, 1] = (transformed[:, 1] - top) / height * (INPUT_HEIGHT - 1.0)
    if flipped:
        transformed[:, 0] = INPUT_WIDTH - 1.0 - transformed[:, 0]
        for left_index, right_index in FLIP_PAIRS:
            transformed[[left_index, right_index]] = transformed[[right_index, left_index]]
    return transformed


def make_heatmaps_highres(
    keypoints: np.ndarray, *, sigma: float = HEATMAP_SIGMA
) -> tuple[np.ndarray, np.ndarray]:
    heatmaps = np.zeros((NUM_KEYPOINTS, HEATMAP_HEIGHT, HEATMAP_WIDTH), dtype=np.float32)
    weights = np.zeros(NUM_KEYPOINTS, dtype=np.float32)
    radius = int(math.ceil(3.0 * sigma))
    for index, (x_coord, y_coord, visibility) in enumerate(keypoints):
        if visibility <= 0 or not math.isfinite(float(x_coord)) or not math.isfinite(float(y_coord)):
            continue
        heat_x = float(x_coord) / (INPUT_WIDTH - 1.0) * (HEATMAP_WIDTH - 1.0)
        heat_y = float(y_coord) / (INPUT_HEIGHT - 1.0) * (HEATMAP_HEIGHT - 1.0)
        if heat_x < -radius or heat_x >= HEATMAP_WIDTH + radius or heat_y < -radius or heat_y >= HEATMAP_HEIGHT + radius:
            continue
        x_min = max(0, int(math.floor(heat_x)) - radius)
        x_max = min(HEATMAP_WIDTH, int(math.floor(heat_x)) + radius + 1)
        y_min = max(0, int(math.floor(heat_y)) - radius)
        y_max = min(HEATMAP_HEIGHT, int(math.floor(heat_y)) + radius + 1)
        xs = np.arange(x_min, x_max, dtype=np.float32)
        ys = np.arange(y_min, y_max, dtype=np.float32)[:, None]
        heatmaps[index, y_min:y_max, x_min:x_max] = np.exp(
            -((xs - heat_x) ** 2 + (ys - heat_y) ** 2) / (2.0 * sigma * sigma)
        )
        weights[index] = 1.0
    weights[3] = 0.0
    weights[4] = 0.0
    return heatmaps, weights


class NativeHighResRobustDataset(Dataset):
    def __init__(self, items: Sequence[dict[str, Any]], *, augment: bool) -> None:
        if not augment:
            raise ValueError("native high-resolution robust fit requires augmentation")
        self.items = list(items)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int):
        item = self.items[index]
        with Image.open(image_path(item["file_name"])) as source:
            image = source.convert("RGB")
        dx = float(np.random.uniform(-CENTER_JITTER_FRACTION, CENTER_JITTER_FRACTION))
        dy = float(np.random.uniform(-CENTER_JITTER_FRACTION, CENTER_JITTER_FRACTION))
        scale = float(np.random.uniform(1.0 - SCALE_JITTER_FRACTION, 1.0 + SCALE_JITTER_FRACTION))
        geometry = jitter_crop_geometry(item["box"], center_dx=dx, center_dy=dy, scale=scale)
        left, top, width, height = geometry
        crop = image.crop((left, top, left + width, top + height)).resize(
            (INPUT_WIDTH, INPUT_HEIGHT), Image.Resampling.BILINEAR
        )
        flipped = random.random() < 0.5
        if flipped:
            crop = crop.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        crop = ImageEnhance.Brightness(crop).enhance(random.uniform(0.8, 1.2))
        crop = ImageEnhance.Contrast(crop).enhance(random.uniform(0.8, 1.2))
        pixels = normalize_image(np.asarray(crop, dtype=np.uint8))
        transformed = transform_keypoints_highres(item["keypoints"], geometry, flipped=flipped)
        heatmaps, weights = make_heatmaps_highres(transformed)
        return torch.from_numpy(pixels), torch.from_numpy(heatmaps), torch.from_numpy(weights)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=160)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=0.003)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--max-train-items", type=int)
    parser.add_argument("--require-cuda", action="store_true")
    return parser.parse_args()


def export_model(
    model: torch.nn.Module, output_dir: Path, config: dict[str, Any]
) -> dict[str, Any]:
    cpu_model = model.eval().cpu()
    example = torch.zeros(1, 3, INPUT_HEIGHT, INPUT_WIDTH)
    traced = torch.jit.trace(cpu_model, example, strict=True)
    frozen = torch.jit.freeze(traced)
    model_path = output_dir / "model.torchscript"
    frozen.save(str(model_path))
    reloaded = torch.jit.load(str(model_path), map_location="cpu").eval()
    expected_shape = (1, NUM_KEYPOINTS, HEATMAP_HEIGHT, HEATMAP_WIDTH)
    if tuple(reloaded(example).shape) != expected_shape:
        raise RuntimeError(f"exported model violated native high-resolution contract {expected_shape}")
    torch.set_num_threads(max(1, min(8, torch.get_num_threads())))
    with torch.inference_mode():
        for _ in range(3):
            reloaded(example)
        started = time.monotonic()
        iterations = 10
        for _ in range(iterations):
            reloaded(example)
        cpu_seconds = (time.monotonic() - started) / iterations
    runtime = {
        "cpu_threads": torch.get_num_threads(),
        "batch_size": 1,
        "input_shape": [1, 3, INPUT_HEIGHT, INPUT_WIDTH],
        "output_shape": list(expected_shape),
        "iterations": iterations,
        "seconds_per_forward": cpu_seconds,
        "flip_tta_seconds_per_crop_estimate": 2.0 * cpu_seconds,
        "torchscript_bytes": model_path.stat().st_size,
        "parameter_count": parameter_count(cpu_model),
    }
    write_json_atomic(output_dir / "runtime_benchmark.json", runtime)
    write_json_atomic(
        output_dir / "model_meta.json",
        {
            "schema_version": 1,
            "model_type": CANDIDATE,
            "architecture": "ScratchSkipPoseNetV4 topology at native 240x320",
            "input_width": INPUT_WIDTH,
            "input_height": INPUT_HEIGHT,
            "heatmap_width": HEATMAP_WIDTH,
            "heatmap_height": HEATMAP_HEIGHT,
            "heatmap_sigma": HEATMAP_SIGMA,
            "num_keypoints": NUM_KEYPOINTS,
            "padding_factor": 1.25,
            "image_mean": [0.5, 0.5, 0.5],
            "image_std": [0.25, 0.25, 0.25],
            "score_excludes_keypoint_indices": [3, 4],
            "external_weights": False,
            "external_training_data": False,
            "training_config": config,
            "runtime": runtime,
        },
    )
    return runtime


def main() -> int:
    args = parse_args()
    if args.epochs <= 0 or args.batch_size <= 0 or args.workers < 0:
        raise SystemExit("epochs/batch size must be positive and workers nonnegative")
    if args.seed != SEED or args.batch_size != 160:
        raise SystemExit("predeclared v16 seed and batch size are fixed at 20261005 and 160")
    output_dir = args.output_dir.resolve()
    if not output_dir.is_relative_to(RUN_ROOT.resolve()):
        raise SystemExit(f"--output-dir must be inside {RUN_ROOT}")
    output_dir.mkdir(parents=True, exist_ok=False)
    highcap.seed_everything(args.seed)
    if args.require_cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA was required but is not visible")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_float32_matmul_precision("high")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    datasets = {domain: load_domain(domain) for domain in TRAIN_DOMAINS}
    by_domain = {
        domain: dataset_items(datasets[domain], require_labels=True) for domain in TRAIN_DOMAINS
    }
    train_items = [item for domain in TRAIN_DOMAINS for item in by_domain[domain]]
    if args.max_train_items is not None:
        if args.max_train_items < 2:
            raise SystemExit("--max-train-items must be at least 2")
        per_domain = args.max_train_items // 2
        train_items = by_domain["market"][:per_domain] + by_domain["pa100k"][: args.max_train_items - per_domain]
    model = NativeHighResPoseNet().to(device)
    config = {
        "candidate": CANDIDATE,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "architecture": "ScratchSkipPoseNetV4 topology unchanged",
        "isolated_factor_vs_robust_seed20261005": "native_input_heatmap_resolution",
        "input_width": INPUT_WIDTH,
        "input_height": INPUT_HEIGHT,
        "heatmap_width": HEATMAP_WIDTH,
        "heatmap_height": HEATMAP_HEIGHT,
        "heatmap_sigma": HEATMAP_SIGMA,
        "sigma_scaling": "1.8 * 1.25 to preserve normalized target width",
        "jitter": {
            "center_uniform_fraction": [-CENTER_JITTER_FRACTION, CENTER_JITTER_FRACTION],
            "isotropic_scale_uniform": [1.0 - SCALE_JITTER_FRACTION, 1.0 + SCALE_JITTER_FRACTION],
            "distribution_check": jitter_distribution(args.seed, CENTER_JITTER_FRACTION, SCALE_JITTER_FRACTION),
        },
        "held_constant_vs_robust_seed20261005": [
            "ScratchSkipPoseNetV4 topology and initialization seed",
            "uniform visibility-masked heatmap MSE",
            "bbox center/scale jitter draws and color/flip augmentation",
            "AdamW + OneCycleLR schedule",
            "Market+PA100K fit boundary",
            "epochs, batch size, learning rate, weight decay",
        ],
        "train_domains": list(TRAIN_DOMAINS),
        "eval_domain": None,
        "fit_scope": "holdout_peta",
        "domain_train_examples": {domain: len(by_domain[domain]) for domain in TRAIN_DOMAINS},
        "train_examples": len(train_items),
        "max_train_items": args.max_train_items,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "workers": args.workers,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "seed": args.seed,
        "parameter_count": parameter_count(model),
        "external_weights": False,
        "external_training_data": False,
        "hidden_data_used": False,
        "annotation_files": {
            domain: {
                "path": highcap.DOMAIN_ANNOTATIONS[domain],
                "sha256": sha256_file(PROJECT_ROOT / highcap.DOMAIN_ANNOTATIONS[domain]),
            }
            for domain in TRAIN_DOMAINS
        },
        "source_files": {
            str(path.relative_to(PROJECT_ROOT)): sha256_file(path)
            for path in (
                Path(__file__).resolve(),
                PROJECT_ROOT / "scripts/train_scratch_hrnet_v4.py",
                PROJECT_ROOT / "scripts/train_scratch_robust_v5.py",
                PROJECT_ROOT / "src/upar_pose/scratch_hrnet_v4.py",
            )
        },
        "device": str(device),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "device_name": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
    }
    write_json_atomic(output_dir / "config.json", config)
    started = time.monotonic()
    original_dataset = highcap.PoseTrainingDataset
    highcap.PoseTrainingDataset = NativeHighResRobustDataset
    try:
        history = highcap.train_model(
            model,
            train_items,
            device=device,
            output_dir=output_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            workers=args.workers,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
        )
    finally:
        highcap.PoseTrainingDataset = original_dataset
    peak_cuda = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
    runtime = export_model(model, output_dir, config)
    artifact_names = (
        "config.json", "last_checkpoint.pt", "model.torchscript", "model_meta.json",
        "training_curve.json", "training_history.jsonl", "runtime_benchmark.json",
    )
    manifest = {
        "schema_version": 1,
        "candidate": CANDIDATE,
        "component_role": "native_highres_seed_20261005",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "config": config,
        "history": history,
        "evaluation": None,
        "runtime": runtime,
        "peak_cuda_allocated_bytes": peak_cuda,
        "artifacts": {
            name: {"path": name, "sha256": sha256_file(output_dir / name)} for name in artifact_names
        },
        "elapsed_seconds": time.monotonic() - started,
    }
    write_json_atomic(output_dir / "manifest.json", manifest)
    print(json.dumps({"manifest": str(output_dir / "manifest.json")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
