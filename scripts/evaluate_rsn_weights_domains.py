#!/usr/bin/env python3
"""Evaluate saved RSN weights (single file or uniform average of several) on chosen PETA cameras.

Uses the same six-view validator and domain-structured metric as
train_rsn_domain_continuation.py.  Averaging several epoch weights from one
run is a fixed-rule weight soup (BatchNorm buffers averaged as well); it is
reported as a separate inference variant, never as the source checkpoints.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]
from train_rsn_domain_continuation import evaluate_heldout  # noqa: E402
from train_scratch_cnn import load_domain, dataset_items  # noqa: E402
from upar_pose.rsn_adapter import RSNPose  # noqa: E402
from upar_pose.domain_eval import image_domains, load_peta_domain_map  # noqa: E402


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--weights', type=Path, nargs='+', required=True)
    p.add_argument('--cameras', nargs='+', required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--label', required=True)
    p.add_argument('--eval-batch-size', type=int, default=8)
    p.add_argument('--allocator-cap-gib', type=float, default=8.0)
    p.add_argument('--save-weights', type=Path, help='write the (averaged) state as a weights file for packaging')
    p.add_argument('--weights-only', action='store_true', help='write --save-weights and stop (CPU only, no evaluation)')
    a = p.parse_args()
    out = a.output_dir
    out.mkdir(parents=True, exist_ok=True)
    target = out / f'{a.label}_metrics.json'
    if target.exists():
        raise FileExistsError(target)
    torch.set_num_threads(1)
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    if a.weights_only and not a.save_weights:
        raise ValueError('--weights-only needs --save-weights')
    if not a.weights_only:
        torch.cuda.set_per_process_memory_fraction(a.allocator_cap_gib * 1024**3 / torch.cuda.get_device_properties(0).total_memory)
    states = [torch.load(w, map_location='cpu', weights_only=False)['model_state'] for w in a.weights]
    merged = {}
    for key, value in states[0].items():
        if value.is_floating_point():
            merged[key] = sum(s[key].double() for s in states).div(len(states)).to(value.dtype)
        else:
            merged[key] = states[-1][key]
    model = RSNPose()
    model.load_state_dict(merged, strict=True)
    if a.save_weights:
        if a.save_weights.exists():
            raise FileExistsError(a.save_weights)
        torch.save({'epoch': 'soup:' + ','.join(str(w) for w in a.weights), 'model_state': merged}, a.save_weights)
    if a.weights_only:
        print(json.dumps({'label': a.label, 'saved': str(a.save_weights), 'parents': [str(w) for w in a.weights]}))
        return
    model.cuda()
    peta = load_domain('peta')
    domains = image_domains(peta, load_peta_domain_map(ROOT / 'peta_file_mapping.txt'))
    unknown = set(a.cameras) - set(domains.values())
    if unknown:
        raise KeyError(f'Unknown cameras {unknown}')
    items = [r for r in dataset_items(peta, require_labels=False) if domains[r['image_id']] in set(a.cameras)]
    result = evaluate_heldout(model, peta, items, domains, out / f'{a.label}_predictions.json', batch_size=a.eval_batch_size)
    result.update(label=a.label, cameras=a.cameras, weights=[{'path': str(w), 'sha256': sha256(w)} for w in a.weights],
                  averaged=len(a.weights) > 1)
    target.write_text(json.dumps(result, ensure_ascii=False, indent=1) + '\n')
    print(json.dumps({k: result[k] for k in ('label', 'subset_mean', 'per_subset_AP', 'seconds')}))


if __name__ == '__main__':
    main()
