#!/usr/bin/env python3
"""Config-driven native-resolution model/ensemble comparison on saved PETA controls."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys
import time

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from evaluate_highcap_highres_inference_v15 import ResolutionInteractionDataset
from evaluate_robust_seed_geometry_ensemble_v7 import aligned_geometry_mean
from evaluate_seed_geometry_ensemble_v6 import PADDINGS, TARGET_PADDING, prediction
from train_scratch_cnn import dataset_items, load_domain
from upar_pose.evaluation import assigned_prediction_oks_by_image, evaluate_coco_keypoints, paired_bootstrap_mean_difference
from upar_pose.flip_tta import FLIP_CHANNELS, decode_heatmaps_refined
from upar_pose.pipeline import load_json, sha256_file, write_json_atomic


def aligned_model_heatmaps(model, inputs, flip_channels, expected_hw, *, return_view_means=False):
    """Legacy six-view mean; opt-in also exposes already-computed flip means.

    Returning the three per-padding maps adds no model forward and does not
    change the six-view arithmetic. Callers must not mutate the returned views.
    """
    count, scales = inputs.shape[:2]
    flat = inputs.reshape(-1, *inputs.shape[2:])
    raw = model(torch.cat((flat, torch.flip(flat, dims=(-1,))), dim=0)).float()
    if tuple(raw.shape[-2:]) != tuple(expected_hw) or raw.shape[1] != 17:
        raise ValueError("Native heatmap shape mismatch")
    original, flipped = raw.split(len(flat), dim=0)
    flip_back = torch.flip(flipped, dims=(-1,)).index_select(1, flip_channels)
    views = (.5 * (original + flip_back)).reshape(count, scales, *original.shape[1:])
    aligned = aligned_geometry_mean(views)
    return (aligned, views) if return_view_means else aligned


def weighted_heatmaps(heatmaps, weights):
    if len(heatmaps) != len(weights) or not weights or any(w <= 0 for w in weights) or abs(sum(weights) - 1.) > 1e-12:
        raise ValueError("Explicit positive normalized model weights required")
    # Same ordered arithmetic as the production v11 ensemble; do not average coordinates.
    return sum(h * w for h, w in zip(heatmaps, weights, strict=True))


def check_ids(predictions, expected):
    ids = [(int(p["image_id"]), int(p["source_annotation_id"])) for p in predictions]
    if len(ids) != len(set(ids)) or set(ids) != expected:
        raise ValueError("Prediction identity mismatch or duplicate")


def inspect_model(spec, config):
    path = ROOT / spec["model"]
    manifest_path = path.parent / "manifest.json"
    manifest = load_json(manifest_path)
    fit = manifest["config"]
    expected = {"seed":spec["seed"], "train_domains":["market","pa100k"], "train_examples":32422,
                "fit_scope":"holdout_peta", "epochs":20, "batch_size":160,
                "input_width":config["size"][0], "input_height":config["size"][1],
                "heatmap_width":config["heatmap_size"][0], "heatmap_height":config["heatmap_size"][1],
                "heatmap_sigma":config["heatmap_sigma"], "external_weights":False,
                "external_training_data":False, "hidden_data_used":False}
    if any(fit.get(k) != v for k, v in expected.items()):
        raise ValueError(f"Fit-boundary/recipe mismatch: {path}")
    digest = sha256_file(path)
    if manifest["artifacts"]["model.torchscript"]["sha256"] != digest or len(manifest["history"]) != 20:
        raise ValueError("Model hash or completed training history mismatch")
    for annotation in fit["annotation_files"].values():
        if sha256_file(ROOT / annotation["path"]) != annotation["sha256"]:
            raise ValueError("Fit annotation changed")
    return {"path": spec["model"], "sha256":digest, "manifest_sha256":sha256_file(manifest_path),
            "seed":spec["seed"], "recovery":fit.get("recovery"), "training_seconds_this_segment":manifest["elapsed_seconds"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--require-cuda", action="store_true")
    args = parser.parse_args()
    config = load_json(args.config)
    output = ROOT / config["output_dir"]
    if output.exists():
        raise RuntimeError("Output exists; inspect saved predictions instead of blindly rerunning")
    bindings = [inspect_model(spec, config) for spec in config["models"]]
    for spec in [*config["models"], config["ensemble_reference"]]:
        if sha256_file(ROOT / spec["reference"]) != spec["reference_sha256"]:
            raise ValueError("Frozen low-resolution reference changed")
    if sha256_file(ROOT / config["annotation"]) != config["annotation_sha256"]:
        raise ValueError("Evaluation annotation changed")
    if args.require_cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = load_domain("peta")
    items = dataset_items(dataset, require_labels=False)
    expected = {(int(item["image_id"]), int(item["annotation_id"])) for item in items}
    if len(items) != config["expected_count"]:
        raise ValueError("Evaluation sample count changed")
    output.mkdir(parents=True)
    write_json_atomic(output / "config_snapshot.json", config)
    write_json_atomic(output / "model_bindings.json", bindings)
    models = [torch.jit.load(str(ROOT / spec["model"]), map_location=device).eval() for spec in config["models"]]
    names = [str(spec["seed"]) for spec in config["models"]] + ["ensemble"]
    predictions = {name: [] for name in names}
    loader = DataLoader(ResolutionInteractionDataset(items, tuple(config["size"])),
                        batch_size=config["batch_size"], num_workers=config["workers"], shuffle=False,
                        pin_memory=device.type == "cuda")
    channels = torch.tensor(FLIP_CHANNELS, device=device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    started = time.monotonic()
    with torch.inference_mode():
        for batch, geometries, image_ids, annotation_ids in loader:
            batch = batch.to(device)
            heatmaps = [aligned_model_heatmaps(model, batch, channels, config["heatmap_size"][::-1]) for model in models]
            heatmaps.append(weighted_heatmaps(heatmaps, config["weights"]))
            geometry_rows = [tuple(float(x) for x in row) for row in geometries[:, PADDINGS.index(TARGET_PADDING), :].numpy()]
            for name, values in zip(names, heatmaps, strict=True):
                coords, maxima = decode_heatmaps_refined(values.cpu().numpy(), geometry_rows, refinement="quadratic")
                for i in range(len(image_ids)):
                    predictions[name].append(prediction(coords[i], maxima[i], int(image_ids[i]), int(annotation_ids[i])))
    if device.type == "cuda":
        torch.cuda.synchronize()
    runtime = {"inference_seconds":time.monotonic()-started,
               "peak_cuda_allocated_bytes":torch.cuda.max_memory_allocated() if device.type == "cuda" else None}
    write_json_atomic(output / "runtime.json", runtime)
    for name in names:
        check_ids(predictions[name], expected)
        write_json_atomic(output / f"{name}_predictions.json", predictions[name])
    del predictions, models, heatmaps
    results = {}
    for name, spec in zip(names, [*config["models"],config["ensemble_reference"]], strict=True):
        path = output / f"{name}_predictions.json"
        candidate = load_json(path)
        reference = load_json(ROOT / spec["reference"])
        check_ids(candidate, expected)
        check_ids(reference, expected)
        metrics = evaluate_coco_keypoints(dataset, candidate, max_dets=20)
        control = evaluate_coco_keypoints(dataset, reference, max_dets=20)
        paired = paired_bootstrap_mean_difference(assigned_prediction_oks_by_image(dataset,candidate),
            assigned_prediction_oks_by_image(dataset,reference), replicates=5000, confidence_level=.95, seed=config["bootstrap_seed"])
        results[name] = {"metrics":metrics, "reference_metrics":control, "AP_delta":metrics["AP"]-control["AP"],
                         "paired_assigned_image_oks":paired, "prediction_sha256":sha256_file(path)}
        print(name, results[name]["AP_delta"], flush=True)
    ensemble = results["ensemble"]
    positive = sum(results[name]["AP_delta"] > 0 for name in names[:-1])
    decision = ("supports_development_candidate_with_recovery_confound" if positive>=2 and ensemble["AP_delta"]>config["material_delta"] and ensemble["paired_assigned_image_oks"]["ci_low"]>0
                else "refutes_fixed_ensemble" if positive<2 or ensemble["AP_delta"]<=0 else "ambiguous")
    next_action = {"supports_development_candidate_with_recovery_confound":"Fit production-all native counterparts and validate exact package/runtime for a resolution-transfer probe; preserve v11 fallback and disclose recovery confound.",
                   "refutes_fixed_ensemble":"Retain v11; stop expanding this fixed native-resolution family and use failure-slice evidence to choose a distinct representation/target-person-disambiguation experiment.",
                   "ambiguous":"Keep v11 and saved highres predictions. Inspect complementarity and runtime once; an informative external probe is not prohibited by a nonsignificant local interval."}[decision]
    comparison = {"candidate":config["candidate"], "generated_at":datetime.now(timezone.utc).isoformat(),
        "models":bindings, "results":results, "positive_seed_count":positive, "decision":decision,
        "next_action":next_action, "runtime":runtime, "config_sha256":sha256_file(args.config),
        "limitations":["PETA is repeatedly used development, not independent confirmation.",
                       "Seeds106/107 resumed with reset legacy RNG/scaler and reduced workers; not isolated resolution causality.",
                       "Bootstrap samples images, not unknown tracks/person identities. CI describes assigned OKS, not AP uncertainty.",
                       "Local scorer uses documented COCO defaults; hidden organizer aggregation is not locally observable."],
        "source_hashes":{str(Path(__file__).relative_to(ROOT)):sha256_file(Path(__file__)),
                         "src/upar_pose/evaluation.py":sha256_file(ROOT/"src/upar_pose/evaluation.py")}}
    comparison_path = output / "comparison.json"
    write_json_atomic(comparison_path, comparison)
    rel = lambda p:str(p.relative_to(ROOT))
    work = ROOT / config["work_os_dir"]
    fit_path = work / "recovery_fit_audit.json"
    write_json_atomic(fit_path, {"status":"clean", "pipeline":"matched", "target_mode":"unseen_group",
        "note":"Fit-domain isolation and inference interface verified against manifests/model hashes. This does NOT establish matched uninterrupted stochastic trajectories; two fits are explicitly recovered with RNG/scaler/worker changes.",
        "evidence":[rel(output/"model_bindings.json"),rel(output/"config_snapshot.json")]})
    event = {"id":config["candidate"]+"_peta", "candidate":config["candidate"], "kind":"performance",
        "summary":f"Recovered native-highres three-seed ensemble PETA AP {ensemble['metrics']['AP']:.6f} vs v11 {ensemble['reference_metrics']['AP']:.6f} ({ensemble['AP_delta']:+.6f}); {positive}/3 individual deltas positive.",
        "issue":"Repeated PETA development plus interrupted stochastic trajectories limit causal/generalization claims.",
        "next_action":next_action, "candidate_status":"rejected" if decision.startswith("refutes") else "in_progress",
        "valid":True, "guardrails_pass":True, "is_best":False,
        "metric":{"name":"COCO_OKS_AP_PETA_NATIVE_HIGHRES_RECOVERED_ENSEMBLE", "direction":"max",
            "evaluation_id":config["candidate"]+"_peta", "evaluation_ref":rel(args.config.resolve()),
            "baseline_id":"v11_RRR_three_robust_seeds", "baseline":ensemble["reference_metrics"]["AP"],
            "candidate":ensemble["metrics"]["AP"], "material_delta":config["material_delta"]},
        "scope":{"evaluation_purpose":"candidate_comparison", "evaluation_role":"development",
            "execution":"Fixed ordered equal-weight heatmap ensemble, saved prediction rescore, same person boxes and six-view geometry.",
            "review":"Hash-bound model/fit scope checks and exact prediction ID agreement; not an independent reviewer.",
            "train_groups":rel(work/"train_groups.json"), "eval_groups":rel(work/"eval_groups.json"),
            "tuning_groups":rel(work/"tuning_groups.json"), "group_namespace":"upar-source-domain-v1",
            "fit_boundary_ref":rel(work/"recovery_commitment.json"), "fit_audit_ref":rel(fit_path)},
        "critical_check":{"risk":"Recovery-induced trajectory differences and reused evaluation can masquerade as resolution benefit.",
            "finding":decision+"; recovery limitations retained; no hidden-score guarantee.", "evidence":rel(comparison_path)},
        "visual":False, "evidence":[rel(comparison_path),rel(output/"model_bindings.json")],
        "artifacts":[rel(comparison_path)]+[rel(output/f"{name}_predictions.json") for name in names]}
    write_json_atomic(ROOT/config["event_path"], event)


if __name__ == "__main__":
    main()
