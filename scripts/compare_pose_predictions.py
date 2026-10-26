#!/usr/bin/env python3
"""Rescore an explicitly fixed saved-prediction contrast; no weight search or fit."""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
sys.path.insert(0, str(ROOT/'scripts'))
from duplicate_cluster_uncertainty import cluster_interval
from evaluate_resolution_ensemble import check_ids
from train_scratch_cnn import load_domain, dataset_items
from upar_pose.evaluation import assigned_prediction_oks_by_image, evaluate_coco_keypoints
from upar_pose.pipeline import load_json, sha256_file, write_json_atomic
from render_highcap_robust_ensemble_comparison_v10 import select_samples, render


def coordinate_half(reference, other):
    """Order-independent identity join; fixed equal coordinates and pose scores."""
    key = lambda r: (r['image_id'], r['source_annotation_id'])
    mapped = {key(r): r for r in other}
    if len(mapped) != len(other) or len({key(r) for r in reference}) != len(reference):
        raise ValueError('Duplicate prediction identity')
    if set(mapped) != {key(r) for r in reference}:
        raise ValueError('Prediction identity mismatch')
    result = []
    for row in reference:
        rhs = mapped[key(row)]
        left = np.asarray(row['keypoints'], dtype=np.float64).reshape(17, 3)
        right = np.asarray(rhs['keypoints'], dtype=np.float64).reshape(17, 3)
        if row['category_id'] != rhs['category_id'] or not np.array_equal(left[:, 2], right[:, 2]):
            raise ValueError('Prediction category/visibility contract mismatch')
        values = .5*left + .5*right
        score = .5*row['score'] + .5*rhs['score']
        if not np.isfinite(values).all() or not np.isfinite(score):
            raise ValueError('Nonfinite prediction')
        result.append({**row, 'keypoints': values.flatten().tolist(), 'score': score})
    return result


def compare(config):
    out = ROOT/config['output_dir']
    out.mkdir(parents=True, exist_ok=False)
    for spec in config['frozen_evidence']:
        if sha256_file(ROOT/spec['path']) != spec['sha256']:
            raise ValueError('Frozen evidence changed: '+spec['path'])
    ds = load_domain('peta')
    expected = {(r['image_id'], r['annotation_id']) for r in dataset_items(ds, require_labels=False)}
    rows = {k: load_json(ROOT/config[k]) for k in ('reference', 'other')}
    for pred in rows.values():
        check_ids(pred, expected)
    if config['mode'] == 'coordinate_half':
        candidate = coordinate_half(rows['reference'], rows['other'])
    elif config['mode'] == 'direct':
        candidate = rows['other']
    else:
        raise ValueError('Unknown fixed comparison mode')
    write_json_atomic(out/'candidate_predictions.json', candidate)
    rows['candidate'] = load_json(out/'candidate_predictions.json')
    metrics = {k: evaluate_coco_keypoints(ds, pred, max_dets=20) for k, pred in rows.items()}
    oks = {k: assigned_prediction_oks_by_image(ds, pred) for k, pred in rows.items()}
    groups = load_json(ROOT/config['duplicate_groups'])
    intervals = {k: cluster_interval(oks['reference'], oks[k], groups) for k in ('candidate', 'other')}
    ids = sorted(oks['reference'])
    errors = np.array([[1-oks[k][i] for i in ids] for k in ('reference', 'other')])
    plot = out/'comparison.png'
    chosen = select_samples(ds, rows['reference'], rows['candidate'])
    render(chosen, plot, title=config['title'], reference_label=config['reference_label'], candidate_label=config['candidate_label'])
    delta = metrics['candidate']['AP'] - metrics['reference']['AP']
    report = {'generated_at': datetime.now(timezone.utc).isoformat(), 'config': config,
        'metrics': metrics, 'AP_delta': delta, 'paired_exact_image_cluster_OKS': intervals,
        'reference_other_assigned_OKS_error_correlation': float(np.corrcoef(errors)[0, 1]),
        'decision': 'supports_fixed_candidate_on_reused_development' if delta > config['minimum_AP_delta'] else 'do_not_adopt_fixed_candidate',
        'prediction_sha256': {k: sha256_file(ROOT/config[k]) for k in ('reference', 'other')},
        'candidate_prediction_sha256': sha256_file(out/'candidate_predictions.json'),
        'limits': ['Repeated PETA development; not an untouched confirmation or Public guarantee.',
                   'Exact-image-cluster OKS interval is not an AP interval; unknown person/track dependence remains.',
                   'Fixed coordinate mixture is not heatmap ensembling; result cannot rule out other fusion mechanisms.'],
        'source_sha256': sha256_file(Path(__file__))}
    write_json_atomic(out/'comparison.json', report)
    print({k: m['AP'] for k, m in metrics.items()}, flush=True)
    print(report['decision'], flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    compare(load_json(parser.parse_args().config))
