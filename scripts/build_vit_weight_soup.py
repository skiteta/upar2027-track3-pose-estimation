#!/usr/bin/env python3
"""Uniform weight soup of fine-tuned ViT pose models that share one initialisation (no training).

Floating tensors (weights and BN running statistics) are averaged; integer
counters must agree or the largest is kept.  The output has the trainer weights
format (``model_state`` + lineage), so ``load_vit_weights`` and the packager read
it unchanged.
"""
import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src')]
from upar_pose.pipeline import sha256_file  # noqa: E402
from upar_pose.vit_pose import load_vit_weights  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--weights', type=Path, nargs='+', required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if len(a.weights) < 2:
        raise ValueError('a soup needs at least two parents')
    if a.output.exists():
        raise FileExistsError(a.output)
    states = [torch.load(w, map_location='cpu', weights_only=False)['model_state'] for w in a.weights]
    keys = list(states[0])
    if any(list(s) != keys or any(s[k].shape != states[0][k].shape for k in keys) for s in states[1:]):
        raise ValueError('parents differ in architecture')
    soup = {}
    for k in keys:
        values = [s[k] for s in states]
        soup[k] = torch.stack([v.double() for v in values]).mean(0).to(values[0].dtype) if values[0].is_floating_point() else max(values, key=lambda v: int(v))
    lineage = [{'path': str(w), 'sha256': sha256_file(w), 'epoch': torch.load(w, map_location='cpu', weights_only=False).get('epoch')} for w in a.weights]
    a.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'model_state': soup, 'epoch': 'soup', 'init_epoch': 0, 'soup_parents': lineage}, a.output)
    load_vit_weights(a.output)  # strict reload check
    print(json.dumps({'output': str(a.output), 'sha256': sha256_file(a.output), 'parents': lineage}))


if __name__ == '__main__':
    main()
