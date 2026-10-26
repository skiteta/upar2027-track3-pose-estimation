#!/usr/bin/env python3
"""Train and evaluate the rules-compliant compact UPAR pose CNN from scratch."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
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
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from upar_pose.evaluation import (  # noqa: E402
    assigned_prediction_oks_by_image,
    evaluate_coco_keypoints,
    paired_bootstrap_mean_difference,
)
from upar_pose.pipeline import load_json, sha256_file, write_json_atomic  # noqa: E402
from upar_pose.scratch_cnn import (  # noqa: E402
    HEATMAP_HEIGHT,
    HEATMAP_WIDTH,
    INPUT_HEIGHT,
    INPUT_WIDTH,
    NUM_KEYPOINTS,
    SCORED_KEYPOINTS,
    ScratchPoseNet,
    crop_geometry,
    decode_heatmaps,
    make_heatmaps,
    normalize_image,
    transform_keypoints,
)

DOMAIN_ANNOTATIONS = {
    "market": "data/annotations/specialization/train_market.json",
    "pa100k": "data/annotations/specialization/train_pa100k.json",
    "peta": "data/annotations/specialization/train_peta.json",
}
BASELINE_RUN = PROJECT_ROOT / "artifacts/runs/20261002T161201054180Z_local_validation"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-domain", action="append", choices=tuple(DOMAIN_ANNOTATIONS), required=True)
    parser.add_argument("--eval-domain", choices=tuple(DOMAIN_ANNOTATIONS))
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=192)
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=0.003)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--seed", type=int, default=20261003)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--require-cuda", action="store_true")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_domain(domain: str) -> dict[str, Any]:
    return load_json(PROJECT_ROOT / DOMAIN_ANNOTATIONS[domain])


def image_path(file_name: str) -> Path:
    path = (PROJECT_ROOT / "data" / file_name).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def crop_person(image: Image.Image, geometry: tuple[float, float, float, float]) -> Image.Image:
    left, top, width, height = geometry
    return image.crop((left, top, left + width, top + height)).resize(
        (INPUT_WIDTH, INPUT_HEIGHT),
        Image.Resampling.BILINEAR,
    )


def dataset_items(dataset: dict[str, Any], *, require_labels: bool) -> list[dict[str, Any]]:
    images = {int(image["id"]): image for image in dataset["images"]}
    items: list[dict[str, Any]] = []
    for annotation in dataset["annotations"]:
        if int(annotation.get("category_id", 1)) != 1:
            continue
        box = annotation.get("bbox", [])
        if len(box) != 4 or float(box[2]) <= 0 or float(box[3]) <= 0:
            continue
        raw = annotation.get("keypoints", [])
        if len(raw) != NUM_KEYPOINTS * 3:
            raise ValueError(f"annotation {annotation.get('id')} has malformed keypoints")
        keypoints = np.asarray(raw, dtype=np.float32).reshape(NUM_KEYPOINTS, 3)
        if require_labels and not np.any(keypoints[:, 2] > 0):
            continue
        image = images[int(annotation["image_id"])]
        items.append(
            {
                "image_id": int(annotation["image_id"]),
                "annotation_id": int(annotation["id"]),
                "file_name": str(image["file_name"]),
                "box": [float(value) for value in box],
                "keypoints": keypoints,
            }
        )
    return items


class PoseTrainingDataset(Dataset):
    def __init__(self, items: Sequence[dict[str, Any]], *, augment: bool) -> None:
        self.items = list(items)
        self.augment = augment

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int):
        item = self.items[index]
        with Image.open(image_path(item["file_name"])) as source:
            image = source.convert("RGB")
        geometry = crop_geometry(item["box"])
        crop = crop_person(image, geometry)
        flipped = self.augment and random.random() < 0.5
        if flipped:
            crop = crop.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        if self.augment:
            crop = ImageEnhance.Brightness(crop).enhance(random.uniform(0.8, 1.2))
            crop = ImageEnhance.Contrast(crop).enhance(random.uniform(0.8, 1.2))
        pixels = normalize_image(np.asarray(crop, dtype=np.uint8))
        transformed = transform_keypoints(item["keypoints"], geometry, flipped=flipped)
        heatmaps, weights = make_heatmaps(transformed)
        return torch.from_numpy(pixels), torch.from_numpy(heatmaps), torch.from_numpy(weights)


class PoseInferenceDataset(Dataset):
    def __init__(self, items: Sequence[dict[str, Any]]) -> None:
        self.items = list(items)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int):
        item = self.items[index]
        with Image.open(image_path(item["file_name"])) as source:
            image = source.convert("RGB")
        geometry = crop_geometry(item["box"])
        crop = crop_person(image, geometry)
        pixels = normalize_image(np.asarray(crop, dtype=np.uint8))
        return (
            torch.from_numpy(pixels),
            torch.tensor(geometry, dtype=torch.float64),
            torch.tensor(item["image_id"], dtype=torch.int64),
            torch.tensor(item["annotation_id"], dtype=torch.int64),
        )


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
) -> list[dict[str, Any]]:
    loader = DataLoader(
        PoseTrainingDataset(items, augment=True),
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        persistent_workers=workers > 0,
        drop_last=False,
        prefetch_factor=2 if workers > 0 else None,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
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
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                outputs = model(inputs)
                per_joint = torch.square(outputs - targets).mean(dim=(2, 3))
                loss = (per_joint * weights).sum() / weights.sum().clamp_min(1.0)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            count = inputs.shape[0]
            total_loss += float(loss.detach()) * count
            seen += count
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
            "observed_at": datetime.now(timezone.utc).isoformat(),
        }
        history.append(row)
        with history_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        torch.save(
            {"epoch": epoch, "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict()},
            output_dir / "last_checkpoint.pt",
        )
        print(json.dumps(row), flush=True)
    return history


def evaluate_model(
    model: nn.Module,
    dataset: dict[str, Any],
    *,
    domain: str,
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
    predictions: list[dict[str, Any]] = []
    started = time.monotonic()
    with torch.inference_mode():
        for inputs, geometries, image_ids, annotation_ids in loader:
            heatmaps = model(inputs.to(device, non_blocking=True)).float().cpu().numpy()
            geometry_rows = [tuple(float(value) for value in row) for row in geometries.numpy()]
            coords, maxima = decode_heatmaps(heatmaps, geometry_rows)
            for index in range(len(image_ids)):
                keypoints = [
                    value
                    for x_coord, y_coord in coords[index]
                    for value in (float(x_coord), float(y_coord), 1.0)
                ]
                score = float(np.mean(np.clip(maxima[index, SCORED_KEYPOINTS], 0.0, 1.0)))
                predictions.append(
                    {
                        "image_id": int(image_ids[index]),
                        "category_id": 1,
                        "keypoints": keypoints,
                        "score": max(0.01, score),
                        "source_annotation_id": int(annotation_ids[index]),
                    }
                )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.monotonic() - started
    prediction_path = output_dir / f"{domain}_predictions.json"
    write_json_atomic(prediction_path, predictions)
    replayed = load_json(prediction_path)
    metrics = evaluate_coco_keypoints(dataset, replayed, max_dets=20)
    candidate_oks = assigned_prediction_oks_by_image(dataset, replayed)
    baseline_path = BASELINE_RUN / f"holdout_{domain}" / "predictions.json"
    paired = None
    if baseline_path.is_file():
        baseline_predictions = load_json(baseline_path)
        baseline_oks = assigned_prediction_oks_by_image(dataset, baseline_predictions)
        paired = paired_bootstrap_mean_difference(
            candidate_oks,
            baseline_oks,
            replicates=1000,
            confidence_level=0.95,
            seed=20263003,
        )
    result = {
        "domain": domain,
        "metrics": metrics,
        "prediction_count": len(replayed),
        "predictions_path": str(prediction_path.relative_to(output_dir)),
        "predictions_sha256": sha256_file(prediction_path),
        "rescored_from_saved_predictions": True,
        "paired_assigned_oks_vs_bbox_mean": paired,
        "elapsed_seconds": elapsed,
        "predictions_per_second": len(replayed) / elapsed,
    }
    write_json_atomic(output_dir / f"{domain}_metrics.json", result)
    return result


def export_model(model: nn.Module, output_dir: Path, config: dict[str, Any]) -> None:
    model = model.eval().cpu()
    example = torch.zeros(1, 3, INPUT_HEIGHT, INPUT_WIDTH)
    traced = torch.jit.trace(model, example, strict=True)
    frozen = torch.jit.freeze(traced)
    frozen.save(str(output_dir / "model.torchscript"))
    meta = {
        "schema_version": 1,
        "model_type": "upar_scratch_cnn_v1",
        "input_width": INPUT_WIDTH,
        "input_height": INPUT_HEIGHT,
        "heatmap_width": HEATMAP_WIDTH,
        "heatmap_height": HEATMAP_HEIGHT,
        "num_keypoints": NUM_KEYPOINTS,
        "padding_factor": 1.25,
        "image_mean": [0.5, 0.5, 0.5],
        "image_std": [0.25, 0.25, 0.25],
        "score_excludes_keypoint_indices": [3, 4],
        "external_weights": False,
        "training_config": config,
    }
    write_json_atomic(output_dir / "model_meta.json", meta)
    reloaded = torch.jit.load(str(output_dir / "model.torchscript"), map_location="cpu")
    shape = tuple(reloaded(example).shape)
    if shape != (1, NUM_KEYPOINTS, HEATMAP_HEIGHT, HEATMAP_WIDTH):
        raise RuntimeError(f"unexpected exported output shape: {shape}")


def main() -> int:
    args = parse_args()
    if args.epochs <= 0 or args.batch_size <= 0 or args.eval_batch_size <= 0:
        raise SystemExit("epochs and batch sizes must be positive")
    if len(set(args.train_domain)) != len(args.train_domain):
        raise SystemExit("train domains must be unique")
    if args.eval_domain in args.train_domain:
        raise SystemExit("eval domain must be disjoint from train domains")
    output_dir = args.output_dir.resolve()
    if not output_dir.is_relative_to(PROJECT_ROOT):
        raise SystemExit("--output-dir must be inside the project")
    output_dir.mkdir(parents=True, exist_ok=False)
    seed_everything(args.seed)
    if args.require_cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA was required but is not visible")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_float32_matmul_precision("high")

    datasets = {domain: load_domain(domain) for domain in set(args.train_domain + ([args.eval_domain] if args.eval_domain else []))}
    train_items = [
        item
        for domain in args.train_domain
        for item in dataset_items(datasets[domain], require_labels=True)
    ]
    model = ScratchPoseNet().to(device)
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model_state"])
    config = {
        "candidate": "upar_scratch_cnn_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "train_domains": args.train_domain,
        "eval_domain": args.eval_domain,
        "train_examples": len(train_items),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "eval_batch_size": args.eval_batch_size,
        "workers": args.workers,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "seed": args.seed,
        "external_weights": False,
        "external_training_data": False,
        "annotation_files": {
            domain: {
                "path": DOMAIN_ANNOTATIONS[domain],
                "sha256": sha256_file(PROJECT_ROOT / DOMAIN_ANNOTATIONS[domain]),
            }
            for domain in args.train_domain
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
    )
    evaluation = None
    if args.eval_domain:
        evaluation = evaluate_model(
            model,
            datasets[args.eval_domain],
            domain=args.eval_domain,
            device=device,
            output_dir=output_dir,
            batch_size=args.eval_batch_size,
            workers=args.workers,
        )
        print(json.dumps({"evaluation": evaluation}, ensure_ascii=False), flush=True)
    export_model(model, output_dir, config)
    elapsed = time.monotonic() - started
    manifest = {
        "schema_version": 1,
        "candidate": "upar_scratch_cnn_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "config": config,
        "history": history,
        "evaluation": evaluation,
        "artifacts": {
            "checkpoint": {
                "path": "last_checkpoint.pt",
                "sha256": sha256_file(output_dir / "last_checkpoint.pt"),
            },
            "torchscript": {
                "path": "model.torchscript",
                "sha256": sha256_file(output_dir / "model.torchscript"),
            },
            "model_meta": {
                "path": "model_meta.json",
                "sha256": sha256_file(output_dir / "model_meta.json"),
            },
        },
        "elapsed_seconds": elapsed,
    }
    write_json_atomic(output_dir / "manifest.json", manifest)
    print(json.dumps({"manifest": str(output_dir / "manifest.json"), "evaluation": evaluation, "elapsed_seconds": elapsed}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
