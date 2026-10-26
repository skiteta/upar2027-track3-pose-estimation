"""End-to-end leave-one-domain-out validation runner."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from .baseline import NormalizedMeanPose
from .constants import COCO_KEYPOINT_SIGMAS, COMPETITION_ID, IGNORED_KEYPOINT_INDICES, KEYPOINT_NAMES, PHASE_ID
from .evaluation import (
    assigned_prediction_oks_by_image,
    bootstrap_mean_ci,
    evaluate_coco_keypoints,
    paired_bootstrap_mean_difference,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def _validate_dataset(dataset: Mapping[str, Any], *, domain: str, path: Path) -> dict[str, Any]:
    images = dataset.get("images")
    annotations = dataset.get("annotations")
    if not isinstance(images, list) or not isinstance(annotations, list):
        raise ValueError(f"{path}: expected COCO images and annotations lists")
    image_ids = [int(image["id"]) for image in images]
    annotation_ids = [int(ann["id"]) for ann in annotations]
    if len(image_ids) != len(set(image_ids)):
        raise ValueError(f"{path}: image ids are not unique within domain {domain}")
    if len(annotation_ids) != len(set(annotation_ids)):
        raise ValueError(f"{path}: annotation ids are not unique within domain {domain}")
    image_id_set = set(image_ids)
    malformed_keypoints = 0
    invalid_references = 0
    visible_counts = [0] * len(KEYPOINT_NAMES)
    crowd_count = 0
    unlabelled_count = 0
    for ann in annotations:
        if int(ann["image_id"]) not in image_id_set:
            invalid_references += 1
        raw = ann.get("keypoints", [])
        if len(raw) != 3 * len(KEYPOINT_NAMES):
            malformed_keypoints += 1
            continue
        visibilities = raw[2::3]
        for index, visibility in enumerate(visibilities):
            if float(visibility) > 0:
                visible_counts[index] += 1
        if not any(float(value) > 0 for value in visibilities):
            unlabelled_count += 1
        if int(ann.get("iscrowd", 0)):
            crowd_count += 1
    if malformed_keypoints or invalid_references:
        raise ValueError(
            f"{path}: malformed_keypoints={malformed_keypoints}, invalid_image_references={invalid_references}"
        )
    ignored_visible = {KEYPOINT_NAMES[index]: visible_counts[index] for index in IGNORED_KEYPOINT_INDICES}
    if any(ignored_visible.values()):
        raise ValueError(f"{path}: officially ignored ear slots unexpectedly have labels: {ignored_visible}")
    return {
        "domain": domain,
        "image_count": len(images),
        "annotation_count": len(annotations),
        "crowd_annotation_count": crowd_count,
        "unlabelled_annotation_count": unlabelled_count,
        "visible_annotation_counts_by_keypoint": dict(zip(KEYPOINT_NAMES, visible_counts)),
        "ignored_slots_verified_unlabelled": ignored_visible,
    }


def _predict_dataset(
    model: NormalizedMeanPose, dataset: Mapping[str, Any]
) -> list[dict[str, Any]]:
    annotations_by_image: dict[int, list[Mapping[str, Any]]] = {}
    for ann in dataset["annotations"]:
        if int(ann.get("category_id", 1)) != 1:
            continue
        annotations_by_image.setdefault(int(ann["image_id"]), []).append(ann)
    predictions: list[dict[str, Any]] = []
    for image in dataset["images"]:
        image_id = int(image["id"])
        anns = annotations_by_image.get(image_id, [])
        if not anns:
            continue
        sample = {
            "image_id": image_id,
            "file_name": image.get("file_name"),
            "width": int(image.get("width", 0)),
            "height": int(image.get("height", 0)),
            "boxes": [ann["bbox"] for ann in anns],
            "box_ids": [int(ann["id"]) for ann in anns],
        }
        output = model.predict_sample(sample)
        if len(output["keypoints"]) != len(anns) or len(output["scores"]) != len(anns):
            raise RuntimeError("Baseline violated the one-prediction-per-box contract")
        for ann, keypoints, score in zip(anns, output["keypoints"], output["scores"]):
            predictions.append(
                {
                    "image_id": image_id,
                    "category_id": 1,
                    "keypoints": [value for triplet in keypoints for value in triplet],
                    "score": float(score),
                    # Extra local-only provenance; pycocotools preserves but does
                    # not score this field.
                    "source_annotation_id": int(ann["id"]),
                }
            )
    return predictions


def _predict_box_center_reference(
    confidence_model: NormalizedMeanPose, dataset: Mapping[str, Any]
) -> list[dict[str, Any]]:
    """Fixed-pose control sharing the candidate's train-only confidence rule."""

    predictions: list[dict[str, Any]] = []
    for ann in dataset["annotations"]:
        if int(ann.get("category_id", 1)) != 1:
            continue
        x0, y0, width, height = (float(value) for value in ann["bbox"])
        center_x = x0 + width / 2.0
        center_y = y0 + height / 2.0
        predictions.append(
            {
                "image_id": int(ann["image_id"]),
                "category_id": 1,
                "keypoints": [value for _ in KEYPOINT_NAMES for value in (center_x, center_y, 1.0)],
                "score": confidence_model._box_score(ann["bbox"]),
                "source_annotation_id": int(ann["id"]),
            }
        )
    return predictions


def _path_overlap(fit_datasets: Sequence[Mapping[str, Any]], eval_dataset: Mapping[str, Any]) -> list[str]:
    fit_paths = {
        str(image.get("file_name", ""))
        for dataset in fit_datasets
        for image in dataset.get("images", [])
    }
    eval_paths = {str(image.get("file_name", "")) for image in eval_dataset.get("images", [])}
    return sorted((fit_paths & eval_paths) - {""})


def _source_hashes(project_root: Path) -> dict[str, str]:
    paths = [
        project_root / "configs/local_validation.json",
        project_root / "scripts/run_local_validation.py",
        *sorted((project_root / "src/upar_pose").glob("*.py")),
    ]
    return {str(path.relative_to(project_root)): sha256_file(path) for path in paths}


def run_validation(
    config_path: Path,
    output_dir: Path,
    *,
    selected_folds: Sequence[str] | None = None,
) -> dict[str, Any]:
    config_path = config_path.resolve()
    project_root = config_path.parent.parent
    config = load_json(config_path)
    annotation_paths = {
        domain: (project_root / relative_path).resolve()
        for domain, relative_path in config["annotation_files"].items()
    }
    excluded_combined = config["excluded_combined_annotation"]
    excluded_combined_path = (project_root / excluded_combined["path"]).resolve()
    domains = tuple(annotation_paths)
    requested = tuple(selected_folds) if selected_folds else domains
    unknown = sorted(set(requested) - set(domains))
    if unknown:
        raise ValueError(f"Unknown held-out folds: {unknown}; expected one of {domains}")

    output_dir.mkdir(parents=True, exist_ok=False)
    generated_at = datetime.now(timezone.utc).isoformat()
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "run_type": "leave_one_domain_out_local_validation",
        "generated_at_utc": generated_at,
        "competition": {"id": COMPETITION_ID, "phase_id": PHASE_ID},
        "keypoint_order": list(KEYPOINT_NAMES),
        "ignored_keypoint_indices": list(IGNORED_KEYPOINT_INDICES),
        "split_protocol": {
            "type": "leave_one_domain_out",
            "domains": list(domains),
            "independent_unit": "image",
            "fit_rule": "fit normalized mean pose only on the two non-held-out domain JSON files",
            "eval_rule": "freeze model, then predict every provided GT box in the held-out domain",
            "upar_all_used": False,
            "reason_upar_all_excluded": (
                "Individual domain files are the authoritative split boundary and have overlapping "
                "numeric ids. The official raw combined file also contains the recorded PETA path-prefix issue."
            ),
        },
        "excluded_combined_annotation": {
            **excluded_combined,
            "sha256": sha256_file(excluded_combined_path),
            "preserved_unmodified": True,
        },
        "metric": {
            "implementation": "pycocotools.cocoeval.COCOeval(iouType='keypoints')",
            "max_dets": int(config["metric"]["max_dets"]),
            "oks_sigmas": list(COCO_KEYPOINT_SIGMAS),
            "ranking_metric": "AP @ OKS 0.50:0.95",
            "parity_status": "LIMITED_LOCAL",
            "parity_limitation": (
                "The public competition page specifies COCOeval and maxDets=20, but the private "
                "organizer scoring bundle/fallback implementation is unavailable. Local scoring "
                "therefore freezes pycocotools' standard 17-slot sigmas; hidden-score identity is not claimed."
            ),
        },
        "model": {
            "type": "normalized_mean_pose",
            "input": "one COCO xywh person box plus image width/height",
            "output": "17 absolute-pixel [x,y,v] keypoints and one confidence per box",
            "external_weights": False,
            "external_pretraining": False,
            "vitpose_plus_used": False,
        },
        "validation_scope": {
            "status": "LIMITED_LOCAL",
            "official_validation_split_provided": False,
            "hidden_test_score_predicted": False,
            "limitations": [
                "Leave-one-source-domain-out is a diagnostic stress test inferred from the released data, not an organizer-designated validation split.",
                "The normalized mean-pose baseline does not inspect pixels and is intentionally a weak end-to-end plumbing baseline.",
                "Bootstrap intervals cover direct source-box-assigned OKS over images, not nonlinear COCO AP or hidden-domain uncertainty.",
                "No external weights, foundation model, or ViTPose+ checkpoint is used.",
            ],
        },
        "source_sha256": _source_hashes(project_root),
        "annotation_sha256": {domain: sha256_file(path) for domain, path in annotation_paths.items()},
        "folds": {},
    }

    all_metrics: dict[str, Any] = {"schema_version": 1, "folds": {}}
    for fold_index, eval_domain in enumerate(requested):
        fit_domains = [domain for domain in domains if domain != eval_domain]
        # Fit data are read and the immutable model is materialized before the
        # held-out labels are loaded.  This ordering is part of the leak guard.
        fit_datasets = [load_json(annotation_paths[domain]) for domain in fit_domains]
        fit_summaries = [
            _validate_dataset(dataset, domain=domain, path=annotation_paths[domain])
            for domain, dataset in zip(fit_domains, fit_datasets)
        ]
        model = NormalizedMeanPose.fit(fit_datasets)
        fold_dir = output_dir / f"holdout_{eval_domain}"
        model_path = fold_dir / "model.json"
        write_json_atomic(model_path, model.to_dict())

        eval_dataset = load_json(annotation_paths[eval_domain])
        eval_summary = _validate_dataset(
            eval_dataset, domain=eval_domain, path=annotation_paths[eval_domain]
        )
        overlap = _path_overlap(fit_datasets, eval_dataset)
        if overlap:
            raise ValueError(
                f"Leak guard failed for {eval_domain}: {len(overlap)} file paths overlap fit/eval"
            )
        predictions = _predict_dataset(model, eval_dataset)
        prediction_path = fold_dir / "predictions.json"
        write_json_atomic(prediction_path, predictions)

        # Required replay path: score only the serialized predictions reloaded
        # from disk, not the in-memory generation object.
        replayed_predictions = load_json(prediction_path)
        official_shape_metrics = evaluate_coco_keypoints(
            eval_dataset,
            replayed_predictions,
            sigmas=COCO_KEYPOINT_SIGMAS,
            max_dets=int(config["metric"]["max_dets"]),
        )
        assigned_oks = assigned_prediction_oks_by_image(eval_dataset, replayed_predictions)
        reference_predictions = _predict_box_center_reference(model, eval_dataset)
        reference_prediction_path = fold_dir / "reference_box_center_predictions.json"
        write_json_atomic(reference_prediction_path, reference_predictions)
        replayed_reference_predictions = load_json(reference_prediction_path)
        reference_metrics = evaluate_coco_keypoints(
            eval_dataset,
            replayed_reference_predictions,
            sigmas=COCO_KEYPOINT_SIGMAS,
            max_dets=int(config["metric"]["max_dets"]),
        )
        reference_assigned_oks = assigned_prediction_oks_by_image(
            eval_dataset, replayed_reference_predictions
        )
        uncertainty_config = config["uncertainty"]
        diagnostic_ci = bootstrap_mean_ci(
            assigned_oks,
            replicates=int(uncertainty_config["bootstrap_replicates"]),
            confidence_level=float(uncertainty_config["confidence_level"]),
            seed=int(uncertainty_config["seed"]) + fold_index,
        )
        diagnostic_ci["statistic"] = "mean_direct_assigned_box_OKS_not_AP"
        paired_difference = paired_bootstrap_mean_difference(
            assigned_oks,
            reference_assigned_oks,
            replicates=int(uncertainty_config["bootstrap_replicates"]),
            confidence_level=float(uncertainty_config["confidence_level"]),
            seed=int(uncertainty_config["seed"]) + 1000 + fold_index,
        )
        fold_metrics = {
            "official_shape_coco_metrics": official_shape_metrics,
            "reference_box_center_official_shape_coco_metrics": reference_metrics,
            "AP_difference_vs_box_center": official_shape_metrics["AP"] - reference_metrics["AP"],
            "diagnostic_uncertainty": diagnostic_ci,
            "paired_diagnostic_vs_box_center": paired_difference,
            "prediction_count": len(replayed_predictions),
            "rescored_from_saved_predictions": True,
            "reference_rescored_from_saved_predictions": True,
        }
        metrics_path = fold_dir / "metrics.json"
        write_json_atomic(metrics_path, fold_metrics)
        all_metrics["folds"][eval_domain] = fold_metrics
        manifest["folds"][eval_domain] = {
            "fit_domains": fit_domains,
            "eval_domain": eval_domain,
            "fit_annotation_files": [
                str(annotation_paths[domain].relative_to(project_root)) for domain in fit_domains
            ],
            "eval_annotation_file": str(annotation_paths[eval_domain].relative_to(project_root)),
            "fit_data_summaries": fit_summaries,
            "eval_data_summary": eval_summary,
            "fit_eval_file_path_overlap_count": 0,
            "fit_completed_before_eval_annotations_parsed": True,
            "eval_file_sha256_was_computed_before_fit_but_not_exposed_to_model": True,
            "prediction_count": len(replayed_predictions),
            "artifacts": {
                "model": str(model_path.relative_to(output_dir)),
                "model_sha256": sha256_file(model_path),
                "predictions": str(prediction_path.relative_to(output_dir)),
                "predictions_sha256": sha256_file(prediction_path),
                "reference_box_center_predictions": str(
                    reference_prediction_path.relative_to(output_dir)
                ),
                "reference_box_center_predictions_sha256": sha256_file(
                    reference_prediction_path
                ),
                "metrics": str(metrics_path.relative_to(output_dir)),
                "metrics_sha256": sha256_file(metrics_path),
            },
        }

    fold_aps = [
        metrics["official_shape_coco_metrics"]["AP"] for metrics in all_metrics["folds"].values()
    ]
    all_metrics["macro_mean_AP_across_selected_domains"] = sum(fold_aps) / len(fold_aps)
    metrics_path = output_dir / "metrics.json"
    write_json_atomic(metrics_path, all_metrics)
    manifest["artifacts"] = {
        "metrics": "metrics.json",
        "metrics_sha256": sha256_file(metrics_path),
    }
    manifest_path = output_dir / "manifest.json"
    write_json_atomic(manifest_path, manifest)
    return {"manifest": manifest_path, "metrics": metrics_path, "results": all_metrics}
