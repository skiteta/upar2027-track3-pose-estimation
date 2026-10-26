#!/usr/bin/env python3
"""Render fixed PETA examples comparing the two final high-capacity candidates."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any

from PIL import Image, ImageDraw, ImageOps

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from upar_pose.evaluation import assigned_prediction_oks_by_image  # noqa: E402
from upar_pose.pipeline import load_json, sha256_file, write_json_atomic  # noqa: E402
from upar_pose.visualization import (  # noqa: E402
    BBOX_COLOR,
    GT_COLOR,
    MUTED_COLOR,
    TEXT_COLOR,
    _draw_pose,
    _font,
    _scaled_keypoints,
    verify_png,
)

RUN_DIR = PROJECT_ROOT / "artifacts/runs/20261003T062000+0900_final_candidate_selection_v10"
REFERENCE_PREDICTIONS = PROJECT_ROOT / "artifacts/runs/20261003T043000+0900_higher_capacity_v4/three_seed_geometry_peta/equal_3_seed_geometry_predictions.json"
CANDIDATE_PREDICTIONS = PROJECT_ROOT / "artifacts/runs/20261003T060500+0900_highcap_robust_ensemble_v9/holdout_peta/equal_thirds_highcap_robust_geometry_predictions.json"
ANNOTATIONS = PROJECT_ROOT / "data/annotations/specialization/train_peta.json"
REFERENCE_COLOR = "#D55E00"
CANDIDATE_COLOR = "#0072B2"


def select_samples(
    dataset: dict[str, Any],
    reference_predictions: list[dict[str, Any]],
    candidate_predictions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    images = {int(row["id"]): row for row in dataset["images"]}
    annotations = {int(row["id"]): row for row in dataset["annotations"]}
    reference_by_annotation = {
        int(row["source_annotation_id"]): row for row in reference_predictions
    }
    candidate_by_annotation = {
        int(row["source_annotation_id"]): row for row in candidate_predictions
    }
    reference_oks = assigned_prediction_oks_by_image(dataset, reference_predictions)
    candidate_oks = assigned_prediction_oks_by_image(dataset, candidate_predictions)
    rows: list[dict[str, Any]] = []
    for annotation_id in sorted(set(reference_by_annotation) & set(candidate_by_annotation)):
        annotation = annotations[annotation_id]
        image_id = int(annotation["image_id"])
        image = images[image_id]
        image_path = PROJECT_ROOT / "data" / str(image["file_name"])
        if image_id not in reference_oks or image_id not in candidate_oks or not image_path.is_file():
            continue
        rows.append(
            {
                "annotation_id": annotation_id,
                "image_id": image_id,
                "image_path": image_path,
                "file_name": str(image["file_name"]),
                "bbox": annotation["bbox"],
                "ground_truth": annotation["keypoints"],
                "reference": reference_by_annotation[annotation_id]["keypoints"],
                "candidate": candidate_by_annotation[annotation_id]["keypoints"],
                "reference_oks": float(reference_oks[image_id]),
                "candidate_oks": float(candidate_oks[image_id]),
                "delta": float(candidate_oks[image_id] - reference_oks[image_id]),
            }
        )
    if len(rows) < 3:
        raise RuntimeError("need at least three readable paired samples")
    rows.sort(key=lambda row: (row["delta"], row["image_id"]))
    selected: list[dict[str, Any]] = []
    for label, quantile in (
        ("lower delta (5%)", 0.05),
        ("median delta (50%)", 0.50),
        ("upper delta (95%)", 0.95),
    ):
        index = round((len(rows) - 1) * quantile)
        row = dict(rows[index])
        row.update(selection_label=label, selection_quantile=quantile, rank_index=index)
        selected.append(row)
    return selected


def render(samples: list[dict[str, Any]], output: Path, *,
           title: str = "UPAR Track 3 — three standard seeds vs two standard + robust",
           reference_label: str = "3 standard", candidate_label: str = "2 standard + robust") -> tuple[int, int]:
    panel_width, panel_height, header_height = 400, 620, 110
    margin, image_top, image_bottom = 18, 68, 500
    canvas = Image.new("RGB", (panel_width * len(samples), header_height + panel_height), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (margin, 12),
        title,
        fill=TEXT_COLOR,
        font=_font(21, bold=True),
    )
    for color, label, left in (
        (GT_COLOR, "Ground truth", 18),
        (REFERENCE_COLOR, reference_label, 185),
        (CANDIDATE_COLOR, candidate_label, 350),
        (BBOX_COLOR, "Provided box", 580),
    ):
        draw.line((left, 64, left + 28, 64), fill=color, width=5)
        draw.text((left + 36, 54), label, fill=TEXT_COLOR, font=_font(15))
    draw.text((770, 54), "Fixed image-OKS-delta quantiles", fill=MUTED_COLOR, font=_font(15))

    for column, sample in enumerate(samples):
        panel_left = column * panel_width
        draw.rectangle(
            (panel_left, header_height, panel_left + panel_width - 1, header_height + panel_height - 1),
            outline="#DADCE0",
            width=1,
        )
        draw.text(
            (panel_left + margin, header_height + 16),
            sample["selection_label"].upper(),
            fill=TEXT_COLOR,
            font=_font(17, bold=True),
        )
        with Image.open(sample["image_path"]) as source:
            source_rgb = ImageOps.exif_transpose(source).convert("RGB")
            original_width, original_height = source_rgb.size
            fitted = ImageOps.contain(
                source_rgb,
                (panel_width - 2 * margin, image_bottom - image_top),
                method=Image.Resampling.NEAREST,
            )
        paste_left = panel_left + (panel_width - fitted.width) // 2
        paste_top = header_height + image_top
        canvas.paste(fitted, (paste_left, paste_top))
        scale_x, scale_y = fitted.width / original_width, fitted.height / original_height
        overlay = ImageDraw.Draw(canvas)
        x_coord, y_coord, width, height = [float(value) for value in sample["bbox"]]
        overlay.rectangle(
            (
                paste_left + x_coord * scale_x,
                paste_top + y_coord * scale_y,
                paste_left + (x_coord + width) * scale_x,
                paste_top + (y_coord + height) * scale_y,
            ),
            outline=BBOX_COLOR,
            width=3,
        )
        for key, color, line_width, radius in (
            ("reference", REFERENCE_COLOR, 2, 3),
            ("candidate", CANDIDATE_COLOR, 3, 4),
            ("ground_truth", GT_COLOR, 3, 4),
        ):
            _draw_pose(
                overlay,
                _scaled_keypoints(
                    sample[key],
                    left=paste_left,
                    top=paste_top,
                    scale_x=scale_x,
                    scale_y=scale_y,
                ),
                color=color,
                width=line_width,
                radius=radius,
            )
        details_y = header_height + image_bottom + 12
        details = (
            sample["file_name"],
            f"image={sample['image_id']}  annotation={sample['annotation_id']}",
            f"assigned OKS: old={sample['reference_oks']:.4f}  new={sample['candidate_oks']:.4f}",
            f"paired delta={sample['delta']:+.4f}",
        )
        for row_index, line in enumerate(details):
            overlay.text(
                (panel_left + margin, details_y + row_index * 20),
                line,
                fill=MUTED_COLOR if row_index == 0 else TEXT_COLOR,
                font=_font(13),
            )
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, format="PNG", optimize=False, compress_level=9)
    return canvas.size


def main() -> int:
    dataset = load_json(ANNOTATIONS)
    samples = select_samples(
        dataset,
        load_json(REFERENCE_PREDICTIONS),
        load_json(CANDIDATE_PREDICTIONS),
    )
    output = RUN_DIR / "visualizations/peta_three_standard_vs_standard_robust.png"
    expected = render(samples, output)
    width, height = verify_png(output, expected_size=expected)
    provenance = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "selection_rule": "Sort readable paired PETA images by assigned image-level OKS delta (two-standard-plus-robust minus three-standard); choose nearest ranks at 5%, 50%, and 95%.",
        "evaluation_role": "development; PETA was reused for parent and interaction selection",
        "hidden_data_used": False,
        "output": {
            "path": str(output.relative_to(PROJECT_ROOT)),
            "width": width,
            "height": height,
            "sha256": sha256_file(output),
            "decode_verified": True,
        },
        "sources": {
            "annotations": {
                "path": str(ANNOTATIONS.relative_to(PROJECT_ROOT)),
                "sha256": sha256_file(ANNOTATIONS),
            },
            "reference_predictions": {
                "path": str(REFERENCE_PREDICTIONS.relative_to(PROJECT_ROOT)),
                "sha256": sha256_file(REFERENCE_PREDICTIONS),
            },
            "candidate_predictions": {
                "path": str(CANDIDATE_PREDICTIONS.relative_to(PROJECT_ROOT)),
                "sha256": sha256_file(CANDIDATE_PREDICTIONS),
            },
        },
        "samples": [
            {
                key: value
                for key, value in sample.items()
                if key
                not in {
                    "image_path",
                    "bbox",
                    "ground_truth",
                    "reference",
                    "candidate",
                }
            }
            | {"source_image_sha256": sha256_file(sample["image_path"])}
            for sample in samples
        ],
        "source_sha256": {
            "scripts/render_highcap_robust_ensemble_comparison_v10.py": sha256_file(Path(__file__)),
            "src/upar_pose/visualization.py": sha256_file(
                PROJECT_ROOT / "src/upar_pose/visualization.py"
            ),
        },
    }
    write_json_atomic(output.with_suffix(".json"), provenance)
    print(json.dumps(provenance, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
