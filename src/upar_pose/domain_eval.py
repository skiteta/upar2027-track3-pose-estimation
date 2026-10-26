"""Domain-structured evaluation that mirrors the official "mean over hidden subsets".

The official leaderboard reports COCO keypoint metrics per hidden subset and then
averages them across four subsets.  A single pooled PETA AP hides that structure,
so this module splits PETA into its constituent camera datasets (recovered from
``peta_file_mapping.txt``) and reports per-domain metrics, subset-mean
aggregations and official-format per-keypoint statistics.

Subsets that contain no persons of an area range make pycocotools return -1 for
that range.  How the organizer's scorer aggregates such -1 values is not public,
so both a zero-filled mean and a skip-missing mean are reported.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .constants import COCO_KEYPOINT_SIGMAS, IGNORED_KEYPOINT_INDICES, KEYPOINT_NAMES
from .evaluation import evaluate_coco_keypoints

METRIC_KEYS = ("AP", "AP50", "AP75", "AP_medium", "AP_large", "AR", "AR50", "AR75", "AR_medium", "AR_large")


def load_peta_domain_map(mapping_path: str | Path) -> dict[str, str]:
    """Map PETA png basenames (e.g. ``09001.png``) to their source camera dataset."""

    mapping: dict[str, str] = {}
    for line in Path(mapping_path).read_text().splitlines():
        if not line.strip():
            continue
        source, target = line.strip().split(",")
        mapping[target.rsplit("/", 1)[-1]] = source.split("/")[2]
    return mapping


def image_domains(ground_truth: Mapping[str, Any], domain_map: Mapping[str, str]) -> dict[int, str]:
    """Domain label for every GT image id; unknown basenames are an error."""

    domains: dict[int, str] = {}
    for image in ground_truth["images"]:
        basename = str(image["file_name"]).rsplit("/", 1)[-1]
        if basename not in domain_map:
            raise KeyError(f"No domain for {image['file_name']}")
        domains[int(image["id"])] = domain_map[basename]
    return domains


def subset_ground_truth(ground_truth: Mapping[str, Any], image_ids: set[int]) -> dict[str, Any]:
    return {
        **{k: v for k, v in ground_truth.items() if k not in ("images", "annotations")},
        "images": [img for img in ground_truth["images"] if int(img["id"]) in image_ids],
        "annotations": [ann for ann in ground_truth["annotations"] if int(ann["image_id"]) in image_ids],
    }


def per_keypoint_matched_stats(
    ground_truth: Mapping[str, Any],
    predictions: Sequence[Mapping[str, Any]],
    *,
    sigmas: Sequence[float] = COCO_KEYPOINT_SIGMAS,
    match_threshold: float = 0.5,
) -> dict[str, Any]:
    """Official-report style per-joint mean OKS term and accuracy over matched pairs.

    Every local image carries one GT box and one prediction, so the pair is the
    prediction of the same image (highest score if several).  Pairs whose
    person OKS is below ``match_threshold`` are excluded, as in the report.
    """

    sigma = np.asarray(sigmas, dtype=np.float64)
    variances = (2.0 * sigma) ** 2
    best_pred: dict[int, Mapping[str, Any]] = {}
    for pred in predictions:
        image_id = int(pred["image_id"])
        if image_id not in best_pred or float(pred.get("score", 0)) > float(best_pred[image_id].get("score", 0)):
            best_pred[image_id] = pred
    terms: dict[int, list[float]] = defaultdict(list)
    person_oks: list[float] = []
    matched = 0
    for ann in ground_truth["annotations"]:
        pred = best_pred.get(int(ann["image_id"]))
        gt = np.asarray(ann["keypoints"], dtype=np.float64).reshape(-1, 3)
        visible = gt[:, 2] > 0
        if pred is None or not visible.any() or float(ann.get("area", 0)) <= 0:
            continue
        pk = np.asarray(pred["keypoints"], dtype=np.float64).reshape(-1, 3)
        d2 = np.square(pk[:, 0] - gt[:, 0]) + np.square(pk[:, 1] - gt[:, 1])
        term = np.exp(-d2 / variances / (float(ann["area"]) + np.spacing(1)) / 2.0)
        oks = float(term[visible].mean())
        person_oks.append(oks)
        if oks < match_threshold:
            continue
        matched += 1
        for j in np.flatnonzero(visible):
            terms[int(j)].append(float(term[j]))
    joints = {}
    for j, name in enumerate(KEYPOINT_NAMES):
        if j in IGNORED_KEYPOINT_INDICES or not terms.get(j):
            continue
        values = np.asarray(terms[j])
        joints[name] = {"labelled": int(values.size), "mean_oks": float(values.mean()), "acc": float((values >= 0.5).mean())}
    return {"matched_pairs": matched, "persons": len(person_oks), "mean_person_oks": float(np.mean(person_oks)) if person_oks else None, "joints": joints}


def oracle_ranked_predictions(
    ground_truth: Mapping[str, Any], predictions: Sequence[Mapping[str, Any]], *, sigmas: Sequence[float] = COCO_KEYPOINT_SIGMAS
) -> list[dict[str, Any]]:
    """Diagnostic only: replace scores by the true OKS (uses GT; never deployable)."""

    sigma = np.asarray(sigmas, dtype=np.float64)
    variances = (2.0 * sigma) ** 2
    gt_by_image = {int(a["image_id"]): a for a in ground_truth["annotations"]}
    out = []
    for pred in predictions:
        ann = gt_by_image.get(int(pred["image_id"]))
        new = dict(pred)
        if ann is not None:
            gt = np.asarray(ann["keypoints"], dtype=np.float64).reshape(-1, 3)
            pk = np.asarray(pred["keypoints"], dtype=np.float64).reshape(-1, 3)
            visible = gt[:, 2] > 0
            d2 = np.square(pk[:, 0] - gt[:, 0]) + np.square(pk[:, 1] - gt[:, 1])
            term = np.exp(-d2 / variances / (float(ann["area"]) + np.spacing(1)) / 2.0)
            new["score"] = float(term[visible].mean()) if visible.any() else 0.0
        out.append(new)
    return out


def aggregate_subsets(per_domain: Mapping[str, Mapping[str, float]]) -> dict[str, dict[str, float | None]]:
    """Subset-mean of every metric under zero-fill and skip-missing (-1) policies."""

    result: dict[str, dict[str, float | None]] = {"zero_fill_mean": {}, "skip_missing_mean": {}}
    for key in METRIC_KEYS:
        values = [float(m[key]) for m in per_domain.values()]
        present = [v for v in values if v >= 0]
        result["zero_fill_mean"][key] = float(np.mean([max(v, 0.0) for v in values])) if values else None
        result["skip_missing_mean"][key] = float(np.mean(present)) if present else None
    return result


def evaluate_domain_structure(
    ground_truth: Mapping[str, Any],
    predictions: Sequence[Mapping[str, Any]],
    domains: Mapping[int, str],
    *,
    groups: Mapping[str, Sequence[str]] | None = None,
    include_oracle: bool = True,
) -> dict[str, Any]:
    """Pooled, per-domain and subset-mean metrics for one prediction set.

    ``groups`` optionally merges camera domains into pseudo-subsets (e.g. four
    groups to mirror the four hidden subsets); otherwise each domain is a subset.
    """

    preds_by_image: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for pred in predictions:
        preds_by_image[int(pred["image_id"])].append(pred)
    unit_of_domain = {d: d for d in set(domains.values())}
    if groups:
        unit_of_domain = {d: g for g, members in groups.items() for d in members}
        missing = set(domains.values()) - set(unit_of_domain)
        if missing:
            raise KeyError(f"Domains without a group: {sorted(missing)}")
    images_by_unit: dict[str, set[int]] = defaultdict(set)
    for image_id, domain in domains.items():
        images_by_unit[unit_of_domain[domain]].add(image_id)

    def run(gt: Mapping[str, Any], preds: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        metrics = evaluate_coco_keypoints(gt, preds)
        if include_oracle:
            metrics["AP_oracle_rank"] = evaluate_coco_keypoints(gt, oracle_ranked_predictions(gt, preds))["AP"]
        return metrics

    report: dict[str, Any] = {"pooled": run(ground_truth, predictions), "per_subset": {}}
    for unit, image_ids in sorted(images_by_unit.items()):
        gt = subset_ground_truth(ground_truth, image_ids)
        if not gt["annotations"]:
            continue
        preds = [p for i in image_ids for p in preds_by_image.get(i, [])]
        metrics = run(gt, preds)
        metrics["persons"] = len(gt["annotations"])
        report["per_subset"][unit] = metrics
    report["subset_mean"] = aggregate_subsets(report["per_subset"])
    if include_oracle:
        for policy in report["subset_mean"].values():
            policy["AP_oracle_rank"] = float(np.mean([m["AP_oracle_rank"] for m in report["per_subset"].values()]))
    report["per_keypoint_matched"] = per_keypoint_matched_stats(ground_truth, predictions)
    return report


def subset_bootstrap(per_subset_values: Mapping[str, float], *, replicates: int = 2000, seed: int = 20261006) -> dict[str, float]:
    """Percentile interval of the subset-mean when subsets are resampled as units."""

    values = np.asarray(list(per_subset_values.values()), dtype=np.float64)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, values.size, size=(replicates, values.size))
    means = values[draws].mean(axis=1)
    return {"mean": float(values.mean()), "ci_low": float(np.percentile(means, 2.5)), "ci_high": float(np.percentile(means, 97.5)), "units": int(values.size)}
