#!/usr/bin/env python3
"""Train and evaluate the organizer-only skip-fusion pose hypothesis."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import sys
import time
from typing import Any, Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = PROJECT_ROOT / "artifacts/runs/20261003T043000+0900_higher_capacity_v4"
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from train_scratch_cnn import (  # noqa: E402
    DOMAIN_ANNOTATIONS,
    PoseInferenceDataset,
    PoseTrainingDataset,
    dataset_items,
    load_domain,
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
    SCORED_KEYPOINTS,
)
from upar_pose.scratch_hrnet_v4 import ScratchSkipPoseNetV4, parameter_count  # noqa: E402
from upar_pose.training_recovery import restore_training, training_checkpoint  # noqa: E402

CANDIDATE = "upar_scratch_skip_pose_v4"
V1_PREDICTIONS = PROJECT_ROOT / (
    "artifacts/runs/20261003T024500+0900_upar_scratch_cnn_v1/"
    "holdout_peta/peta_predictions.json"
)
V2_PREDICTIONS = PROJECT_ROOT / (
    "artifacts/runs/20261003T033500+0900_upar_scratch_cnn_flip_tta_v2/"
    "holdout_peta_variants/flip_quadratic_none_combined_predictions.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-domain", action="append", choices=("market", "pa100k"), required=True)
    parser.add_argument("--eval-domain", choices=("peta",))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=160)
    parser.add_argument("--eval-batch-size", type=int, default=192)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=0.003)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--seed", type=int, default=20261005)
    parser.add_argument("--max-train-items", type=int)
    parser.add_argument("--max-eval-items", type=int)
    parser.add_argument("--require-cuda", action="store_true")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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
    resume_from: Path | None = None,
    amp_dtype: torch.dtype = torch.float16,
    require_finite: bool = False,
) -> list[dict[str, Any]]:
    loader = DataLoader(
        PoseTrainingDataset(items, augment=True),
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        persistent_workers=workers > 0,
        prefetch_factor=2 if workers > 0 else None,
        drop_last=False,
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
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and amp_dtype == torch.float16)
    history: list[dict[str, Any]] = []
    history_path = output_dir / "training_history.jsonl"
    completed_epoch = 0
    if resume_from is not None:
        if resume_from.resolve() == output_dir.resolve():
            raise ValueError("Resume into a new directory; preserve the original checkpoint")
        checkpoint = torch.load(resume_from / "last_checkpoint.pt", map_location="cpu", weights_only=True)
        completed_epoch = restore_training(checkpoint, model, optimizer, scheduler, scaler, steps_per_epoch=len(loader))
        history = [json.loads(line) for line in (resume_from / "training_history.jsonl").read_text().splitlines() if line.strip()]
        history = [row for row in history if row["epoch"] <= completed_epoch]
        if [row["epoch"] for row in history] != list(range(1, completed_epoch + 1)):
            raise ValueError("Checkpoint/history epoch mismatch")
        if not 0 < completed_epoch < epochs:
            raise ValueError("No unfinished training epochs to resume")
        with history_path.open("x", encoding="utf-8") as handle:
            for row in history:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
    for epoch in range(completed_epoch + 1, epochs + 1):
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
                device_type=device.type,
                dtype=amp_dtype,
                enabled=device.type == "cuda",
            ):
                outputs = model(inputs)
                per_joint = torch.square(outputs - targets).mean(dim=(2, 3))
                loss = (per_joint * weights).sum() / weights.sum().clamp_min(1.0)
            if require_finite and not torch.isfinite(loss):
                torch.save({"epoch":epoch,"examples_before_batch":seen,"inputs":inputs.cpu(),
                            "targets":targets.cpu(),"weights":weights.cpu(),"model_state":model.state_dict()},
                           output_dir / "nonfinite_failure.pt")
                raise FloatingPointError("Nonfinite loss; retained exact failed batch; no automatic restart")
            scaler.scale(loss).backward()
            if require_finite:
                # One device reduction, not a synchronize for every parameter.
                finite = torch.stack([torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None]).all()
                if not finite:
                    torch.save({"epoch":epoch,"examples_before_batch":seen,"inputs":inputs.cpu(),
                                "targets":targets.cpu(),"weights":weights.cpu(),"model_state":model.state_dict()},
                               output_dir / "nonfinite_failure.pt")
                    raise FloatingPointError("Nonfinite gradients; retained exact failed batch")
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
        checkpoint_path = output_dir / "last_checkpoint.pt"
        temporary_checkpoint = output_dir / "last_checkpoint.tmp"
        torch.save(training_checkpoint(epoch, model, optimizer, scheduler, scaler), temporary_checkpoint)
        temporary_checkpoint.replace(checkpoint_path)
        print(json.dumps(row), flush=True)
    write_json_atomic(
        output_dir / "training_curve.json",
        {
            "metric": "weighted_heatmap_mse",
            "direction": "min",
            "epochs": history,
            "minimum_loss": min(row["train_loss"] for row in history),
            "final_loss": history[-1]["train_loss"],
        },
    )
    return history


def _prediction(
    coords: np.ndarray,
    maxima: np.ndarray,
    image_id: int,
    annotation_id: int,
) -> dict[str, Any]:
    keypoints = [
        value
        for x_coord, y_coord in coords
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


def evaluate_peta(
    model: nn.Module,
    *,
    device: torch.device,
    output_dir: Path,
    batch_size: int,
    workers: int,
    seed: int,
    max_items: int | None,
) -> dict[str, Any]:
    dataset = load_domain("peta")
    items = dataset_items(dataset, require_labels=False)
    if max_items is not None:
        items = items[:max_items]
    loader = DataLoader(
        PoseInferenceDataset(items),
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        persistent_workers=workers > 0,
        prefetch_factor=2 if workers > 0 else None,
    )
    variants: dict[str, list[dict[str, Any]]] = {
        "single_quadratic": [],
        "flip_quadratic_none_combined": [],
    }
    model.eval()
    started = time.monotonic()
    with torch.inference_mode():
        for inputs, geometries, image_ids, annotation_ids in loader:
            inputs = inputs.to(device, non_blocking=True)
            original = model(inputs).float().cpu().numpy()
            flipped = model(torch.flip(inputs, dims=(3,))).float().cpu().numpy()
            combined = 0.5 * (original + flip_back_heatmaps(flipped, shift="none"))
            geometry_rows = [tuple(float(value) for value in row) for row in geometries.numpy()]
            for name, heatmaps in (
                ("single_quadratic", original),
                ("flip_quadratic_none_combined", combined),
            ):
                coords, maxima = decode_heatmaps_refined(
                    heatmaps,
                    geometry_rows,
                    refinement="quadratic",
                )
                for index in range(len(image_ids)):
                    variants[name].append(
                        _prediction(
                            coords[index],
                            maxima[index],
                            int(image_ids[index]),
                            int(annotation_ids[index]),
                        )
                    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.monotonic() - started

    references = {
        "single_quadratic": (V1_PREDICTIONS, "v1_single_quarter"),
        "flip_quadratic_none_combined": (V2_PREDICTIONS, "v2_flip_quadratic"),
    }
    results: dict[str, Any] = {}
    for name, predictions in variants.items():
        path = output_dir / f"peta_{name}_predictions.json"
        write_json_atomic(path, predictions)
        replayed = load_json(path)
        reference_path, reference_name = references[name]
        candidate_oks = assigned_prediction_oks_by_image(dataset, replayed)
        reference_predictions = load_json(reference_path)
        if max_items is not None:
            selected_ids = {int(item["annotation_id"]) for item in items}
            reference_predictions = [
                row
                for row in reference_predictions
                if int(row.get("source_annotation_id", -1)) in selected_ids
            ]
        reference_oks = assigned_prediction_oks_by_image(dataset, reference_predictions)
        results[name] = {
            "metrics": evaluate_coco_keypoints(dataset, replayed, max_dets=20),
            "prediction_count": len(replayed),
            "predictions_path": path.name,
            "predictions_sha256": sha256_file(path),
            "rescored_from_saved_predictions": True,
            "paired_assigned_oks_vs": reference_name,
            "paired_assigned_oks": paired_bootstrap_mean_difference(
                candidate_oks,
                reference_oks,
                replicates=2000,
                confidence_level=0.95,
                seed=seed + (1 if name == "single_quadratic" else 2),
            ),
        }
    summary = {
        "schema_version": 1,
        "candidate": CANDIDATE,
        "evaluation_domain": "peta",
        "annotation_sha256": sha256_file(PROJECT_ROOT / DOMAIN_ANNOTATIONS["peta"]),
        "inference_seconds_two_views": elapsed,
        "predictions_per_second_two_views": len(items) / elapsed,
        "references": {
            "v1_predictions": {
                "path": str(V1_PREDICTIONS.relative_to(PROJECT_ROOT)),
                "sha256": sha256_file(V1_PREDICTIONS),
            },
            "v2_predictions": {
                "path": str(V2_PREDICTIONS.relative_to(PROJECT_ROOT)),
                "sha256": sha256_file(V2_PREDICTIONS),
            },
        },
        "results": results,
    }
    write_json_atomic(output_dir / "peta_evaluation.json", summary)
    return summary


def export_model(model: nn.Module, output_dir: Path, config: dict[str, Any]) -> dict[str, Any]:
    cpu_model = model.eval().cpu()
    example = torch.zeros(1, 3, INPUT_HEIGHT, INPUT_WIDTH)
    traced = torch.jit.trace(cpu_model, example, strict=True)
    frozen = torch.jit.freeze(traced)
    model_path = output_dir / "model.torchscript"
    frozen.save(str(model_path))
    reloaded = torch.jit.load(str(model_path), map_location="cpu").eval()
    if tuple(reloaded(example).shape) != (1, NUM_KEYPOINTS, HEATMAP_HEIGHT, HEATMAP_WIDTH):
        raise RuntimeError("exported model violated the 192x256 -> 48x64 contract")
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
        "output_shape": [1, NUM_KEYPOINTS, HEATMAP_HEIGHT, HEATMAP_WIDTH],
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
            "architecture_hypothesis": "deep 1/16 context plus 1/8 and 1/4 skip fusion",
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
            "external_training_data": False,
            "training_config": config,
            "runtime": runtime,
        },
    )
    return runtime


def main() -> int:
    args = parse_args()
    if args.epochs <= 0 or args.batch_size <= 0 or args.eval_batch_size <= 0:
        raise SystemExit("epochs and batch sizes must be positive")
    if len(set(args.train_domain)) != len(args.train_domain):
        raise SystemExit("train domains must be unique")
    output_dir = args.output_dir.resolve()
    if not output_dir.is_relative_to(RUN_ROOT.resolve()):
        raise SystemExit(f"--output-dir must be inside {RUN_ROOT.relative_to(PROJECT_ROOT)}")
    output_dir.mkdir(parents=True, exist_ok=False)
    seed_everything(args.seed)
    if args.require_cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA was required but is not visible")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_float32_matmul_precision("high")
    datasets = {domain: load_domain(domain) for domain in args.train_domain}
    train_items = [
        item
        for domain in args.train_domain
        for item in dataset_items(datasets[domain], require_labels=True)
    ]
    if args.max_train_items is not None:
        if args.max_train_items <= 0:
            raise SystemExit("--max-train-items must be positive")
        train_items = train_items[: args.max_train_items]
    model = ScratchSkipPoseNetV4().to(device)
    config = {
        "candidate": CANDIDATE,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "architecture_hypothesis": "deep 1/16 context plus 1/8 and 1/4 skip fusion",
        "train_domains": args.train_domain,
        "eval_domain": args.eval_domain,
        "train_examples": len(train_items),
        "max_train_items": args.max_train_items,
        "max_eval_items": args.max_eval_items,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "eval_batch_size": args.eval_batch_size,
        "workers": args.workers,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "seed": args.seed,
        "parameter_count": parameter_count(model),
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
        evaluation = evaluate_peta(
            model,
            device=device,
            output_dir=output_dir,
            batch_size=args.eval_batch_size,
            workers=args.workers,
            seed=args.seed,
            max_items=args.max_eval_items,
        )
        print(json.dumps({"evaluation": evaluation}, ensure_ascii=False), flush=True)
    runtime = export_model(model, output_dir, config)
    manifest = {
        "schema_version": 1,
        "candidate": CANDIDATE,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "config": config,
        "history": history,
        "evaluation": evaluation,
        "runtime": runtime,
        "artifacts": {
            name: {"path": name, "sha256": sha256_file(output_dir / name)}
            for name in (
                "last_checkpoint.pt",
                "model.torchscript",
                "model_meta.json",
                "training_curve.json",
                "runtime_benchmark.json",
            )
        },
        "elapsed_seconds": time.monotonic() - started,
    }
    write_json_atomic(output_dir / "manifest.json", manifest)
    print(json.dumps({"manifest": str(output_dir / "manifest.json"), "evaluation": evaluation}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
