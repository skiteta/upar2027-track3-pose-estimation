"""Official-shape COCO OKS evaluation plus independent-unit diagnostics."""

from __future__ import annotations

from collections import defaultdict
from contextlib import redirect_stdout
import copy
import io
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from .constants import COCO_KEYPOINT_SIGMAS, KEYPOINT_NAMES


def _make_coco(dataset: Mapping[str, Any]) -> COCO:
    """Construct COCO in memory while tolerating starter JSON's missing info."""

    normalized = copy.deepcopy(dict(dataset))
    normalized.setdefault("info", {})
    normalized.setdefault("licenses", [])
    categories = normalized.setdefault("categories", [{"id": 1, "name": "person"}])
    for category in categories:
        if int(category.get("id", -1)) == 1:
            category.setdefault("keypoints", list(KEYPOINT_NAMES))
    coco = COCO()
    coco.dataset = normalized
    with redirect_stdout(io.StringIO()):
        coco.createIndex()
    return coco


def evaluate_coco_keypoints(
    ground_truth: Mapping[str, Any],
    predictions: Sequence[Mapping[str, Any]],
    *,
    sigmas: Sequence[float] = COCO_KEYPOINT_SIGMAS,
    max_dets: int = 20,
) -> dict[str, float]:
    """Run pycocotools keypoint AP with explicitly frozen public defaults."""

    if not predictions:
        raise ValueError("COCO keypoint evaluation requires at least one prediction")
    if len(sigmas) != len(KEYPOINT_NAMES):
        raise ValueError(f"Expected {len(KEYPOINT_NAMES)} OKS sigmas, got {len(sigmas)}")

    coco_gt = _make_coco(ground_truth)
    # COCO.loadRes mutates result dictionaries; do not alter the saved/replayed
    # prediction object supplied by the caller.
    with redirect_stdout(io.StringIO()):
        coco_dt = coco_gt.loadRes(copy.deepcopy(list(predictions)))
        evaluator = COCOeval(coco_gt, coco_dt, iouType="keypoints")
        evaluator.params.kpt_oks_sigmas = np.asarray(sigmas, dtype=np.float64)
        evaluator.params.maxDets = [int(max_dets)]
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()

    stats = evaluator.stats
    return {
        "AP": float(stats[0]),
        "AP50": float(stats[1]),
        "AP75": float(stats[2]),
        "AP_medium": float(stats[3]),
        "AP_large": float(stats[4]),
        "AR": float(stats[5]),
        "AR50": float(stats[6]),
        "AR75": float(stats[7]),
        "AR_medium": float(stats[8]),
        "AR_large": float(stats[9]),
    }


def assigned_prediction_oks_by_image(
    ground_truth: Mapping[str, Any],
    predictions: Sequence[Mapping[str, Any]],
    *,
    sigmas: Sequence[float] = COCO_KEYPOINT_SIGMAS,
) -> dict[int, float]:
    """Mean direct GT-box/prediction OKS per image for diagnostic bootstrap.

    This is intentionally labelled a diagnostic, not AP: local predictions are
    linked to their source box by ``source_annotation_id``, so the statistic
    avoids rematching and supports a natural image-level independent unit.
    """

    gt_by_id = {int(ann["id"]): ann for ann in ground_truth.get("annotations", [])}
    by_image: dict[int, list[float]] = defaultdict(list)
    sigma_array = np.asarray(sigmas, dtype=np.float64)
    variances = (2.0 * sigma_array) ** 2

    for prediction in predictions:
        source_id = prediction.get("source_annotation_id")
        if source_id is None or int(source_id) not in gt_by_id:
            continue
        gt = gt_by_id[int(source_id)]
        gt_keypoints = np.asarray(gt.get("keypoints", []), dtype=np.float64)
        pred_keypoints = np.asarray(prediction.get("keypoints", []), dtype=np.float64)
        if gt_keypoints.size != 3 * len(KEYPOINT_NAMES) or pred_keypoints.size != 3 * len(KEYPOINT_NAMES):
            continue
        gt_keypoints = gt_keypoints.reshape(-1, 3)
        pred_keypoints = pred_keypoints.reshape(-1, 3)
        visible = gt_keypoints[:, 2] > 0
        if not np.any(visible) or float(gt.get("area", 0.0)) <= 0:
            continue
        squared_distance = np.square(pred_keypoints[:, 0] - gt_keypoints[:, 0]) + np.square(
            pred_keypoints[:, 1] - gt_keypoints[:, 1]
        )
        exponent = squared_distance / variances / (float(gt["area"]) + np.spacing(1)) / 2.0
        oks = float(np.exp(-exponent[visible]).mean())
        if math.isfinite(oks):
            by_image[int(gt["image_id"])].append(oks)

    return {image_id: float(np.mean(values)) for image_id, values in by_image.items()}


def bootstrap_mean_ci(
    values_by_unit: Mapping[Any, float],
    *,
    replicates: int,
    confidence_level: float,
    seed: int,
) -> dict[str, Any]:
    """Percentile bootstrap over independent units (images here)."""

    if not values_by_unit:
        raise ValueError("Cannot bootstrap an empty set of independent units")
    if replicates <= 0:
        raise ValueError("replicates must be positive")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be between zero and one")

    values = np.asarray(list(values_by_unit.values()), dtype=np.float64)
    rng = np.random.default_rng(seed)
    means = np.empty(replicates, dtype=np.float64)
    for index in range(replicates):
        means[index] = float(rng.choice(values, size=len(values), replace=True).mean())
    alpha = (1.0 - confidence_level) / 2.0
    low, high = np.quantile(means, [alpha, 1.0 - alpha])
    return {
        "estimate": float(values.mean()),
        "ci_low": float(low),
        "ci_high": float(high),
        "confidence_level": confidence_level,
        "bootstrap_replicates": replicates,
        "independent_unit": "image",
        "unit_count": int(len(values)),
        "seed": int(seed),
    }


def paired_bootstrap_mean_difference(
    candidate_by_unit: Mapping[Any, float],
    reference_by_unit: Mapping[Any, float],
    *,
    replicates: int,
    confidence_level: float,
    seed: int,
) -> dict[str, Any]:
    """Paired image bootstrap for later candidate-versus-baseline comparisons."""

    common = sorted(set(candidate_by_unit) & set(reference_by_unit), key=str)
    if not common:
        raise ValueError("Candidate and reference have no shared independent units")
    differences = {
        unit: float(candidate_by_unit[unit]) - float(reference_by_unit[unit]) for unit in common
    }
    result = bootstrap_mean_ci(
        differences,
        replicates=replicates,
        confidence_level=confidence_level,
        seed=seed,
    )
    result["statistic"] = "paired_mean_assigned_oks_difference"
    return result
