#!/usr/bin/env python3
"""Weighted average ("soup") of trainer-format checkpoints that share one architecture.

The deployed soups record their coefficients in the ``epoch`` field of the saved file, e.g.
``soup_L3L4_v109_prod:0.82*soup_L3_v108_prod+0.18*soup_L4_v108_prod``. The original command
that built the weighted soups was not recorded; this script reproduces the same operation:
floating tensors (weights and BatchNorm running statistics) are averaged with the given
coefficients in float64 and cast back; integer tensors (e.g. BatchNorm counters) are taken
from the first parent.

Usage::

    python scripts/weighted_soup.py --out soup.pt --name soup_L3L4_v109_prod \
        0.82:soup_L3_v108_prod_weights.pt 0.18:soup_L4_v108_prod_weights.pt
"""

import argparse
import hashlib
from pathlib import Path

import torch


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 24), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("parents", nargs="+", help="COEFFICIENT:PATH pairs")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--name", default="soup")
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    pairs = [(float(c), Path(p)) for c, p in (item.split(":", 1) for item in args.parents)]
    if abs(sum(c for c, _ in pairs) - 1.0) > 1e-6:
        raise ValueError("coefficients must sum to 1")
    states = [torch.load(p, map_location="cpu", weights_only=False)["model_state"] for _, p in pairs]
    keys = list(states[0])
    if any(list(s) != keys or any(s[k].shape != states[0][k].shape for k in keys) for s in states[1:]):
        raise ValueError("parents differ in architecture")
    soup = {}
    for key in keys:
        first = states[0][key]
        if first.is_floating_point():
            soup[key] = sum(c * s[key].double() for (c, _), s in zip(pairs, states)).to(first.dtype)
        else:
            soup[key] = first.clone()
    epoch = args.name + ":" + "+".join(f"{c:g}*{p.stem}" for c, p in pairs)
    lineage = [{"coefficient": c, "path": str(p), "sha256": sha256(p)} for c, p in pairs]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": soup, "epoch": epoch, "lineage": lineage}, args.out)
    print(epoch, sha256(args.out))


if __name__ == "__main__":
    main()
