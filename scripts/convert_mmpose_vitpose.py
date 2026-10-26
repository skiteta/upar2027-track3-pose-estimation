"""Convert an MMPose ViTPose COCO checkpoint (ImageNet-1K MAE -> COCO train2017) to this project's ViT weights file.

MMPose layout: mmpretrain VisionTransformer (patch_cfg padding 2, cls token prepended for the position add and removed
before the encoder, final LayerNorm ``ln1``) + HeatmapHead (two stride-2 deconvs, 1x1 conv) on 256x192 crops.
The 16x12 position grid is bicubically resampled to the project's square grid; COCO eye channels (slots 1, 2) become
fresh head_bottom/head_top outputs because the competition layout uses those slots for head points.
"""

from __future__ import annotations

import argparse
import codecs
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from upar_pose.vit_pose import SPECS, ViTPose  # noqa: E402

FRESH_CHANNELS = (1, 2)   # COCO left/right eye -> competition head_bottom/head_top


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 24), b''):
            h.update(chunk)
    return h.hexdigest()


def load_mmpose(path: Path) -> dict:
    dtypes = [getattr(np.dtypes, n) for n in dir(np.dtypes) if n.endswith('DType')]
    torch.serialization.add_safe_globals([np.core.multiarray._reconstruct, np.ndarray, np.dtype, codecs.encode, *dtypes])
    return torch.load(path, map_location='cpu', weights_only=True)


def convert(src: dict, grid: int, seed: int) -> tuple[dict, dict]:
    sd = src['state_dict']
    dim = sd['backbone.cls_token'].shape[-1]
    depth = 1 + max(int(k.split('.')[2]) for k in sd if k.startswith('backbone.layers.'))
    arch = next(a for a, s in SPECS.items() if s['dim'] == dim and s['depth'] == depth)
    out = {'cls_token': sd['backbone.cls_token'],
           'patch_embed.proj.weight': sd['backbone.patch_embed.projection.weight'],
           'patch_embed.proj.bias': sd['backbone.patch_embed.projection.bias'],
           'norm.weight': sd['backbone.ln1.weight'], 'norm.bias': sd['backbone.ln1.bias']}
    for i in range(depth):
        p = f'backbone.layers.{i}.'
        for ours, theirs in (('norm1', 'ln1'), ('attn.qkv', 'attn.qkv'), ('attn.proj', 'attn.proj'), ('norm2', 'ln2'),
                             ('mlp.fc1', 'ffn.layers.0.0'), ('mlp.fc2', 'ffn.layers.1')):
            for t in ('weight', 'bias'):
                out[f'blocks.{i}.{ours}.{t}'] = sd[f'{p}{theirs}.{t}']
    pos = sd['backbone.pos_embed']
    gh, gw = 16, 12
    if pos.shape[1] != 1 + gh * gw:
        raise ValueError(f'unexpected pos_embed {tuple(pos.shape)}')
    patches = pos[:, 1:].reshape(1, gh, gw, dim).permute(0, 3, 1, 2).float()
    patches = F.interpolate(patches, size=(grid, grid), mode='bicubic', align_corners=False)
    out['pos_embed'] = torch.cat([pos[:, :1].float(), patches.permute(0, 2, 3, 1).reshape(1, grid * grid, dim)], 1)
    for ours, theirs in ((0, 0), (1, 1), (3, 3), (4, 4)):
        for k in [k for k in sd if k.startswith(f'head.deconv_layers.{theirs}.')]:
            out[f'decoder.{ours}.' + k.split('.', 3)[3]] = sd[k]
    weight, bias = sd['head.final_layer.weight'].clone(), sd['head.final_layer.bias'].clone()
    g = torch.Generator().manual_seed(seed)
    for ch in FRESH_CHANNELS:
        weight[ch] = torch.randn(weight[ch].shape, generator=g) * 0.001
        bias[ch] = 0.0
    out['final.weight'], out['final.bias'] = weight, bias
    unused = sorted(k for k in sd if not (k.startswith('backbone.') or k.startswith('head.deconv_layers.')
                                          or k.startswith('head.final_layer.')))
    model = ViTPose(arch, image_size=grid * 16, drop_path=0.0, patch_padding=2, with_cls_token=False)
    expected = model.state_dict()
    out['patch_padding_marker'] = expected['patch_padding_marker']
    out['no_cls_token_marker'] = expected['no_cls_token_marker']
    out = {k: v.float().contiguous() if v.is_floating_point() else v for k, v in out.items()}
    missing, unexpected = model.load_state_dict(out, strict=False)
    if missing or unexpected:
        raise ValueError(f'conversion mismatch missing={missing} unexpected={unexpected}')
    shapes = [k for k, v in out.items() if v.shape != expected[k].shape]
    if shapes:
        raise ValueError(f'shape mismatch {shapes}')
    info = {'arch': arch, 'dim': dim, 'depth': depth, 'grid': grid, 'pos_grid_from': [gh, gw], 'patch_padding': 2,
            'with_cls_token': False, 'fresh_final_channels': list(FRESH_CHANNELS), 'fresh_seed': seed,
            'unmapped_source_keys': unused, 'source_keypoints': list((src.get('meta', {}).get('dataset_meta') or {}).get('keypoint_id2name', {}).values())}
    return out, info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--grid', type=int, default=16, help='square encoder grid (vit_image_size / 16)')
    ap.add_argument('--seed', type=int, default=20261011)
    ap.add_argument('--provenance', default='', help='free-text provenance note stored with the weights')
    a = ap.parse_args()
    src_path = Path(a.source)
    src = load_mmpose(src_path)
    state, info = convert(src, a.grid, a.seed)
    info.update({'source': str(src_path), 'source_sha256': sha256_file(src_path), 'provenance': a.provenance})
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'epoch': 'mmpose_coco', 'init_epoch': 0, 'model_state': state, 'conversion': info}, out)
    info['out_sha256'] = sha256_file(out)
    out.with_suffix('.json').write_text(json.dumps(info, indent=1) + '\n')
    print(json.dumps({k: v for k, v in info.items() if k != 'unmapped_source_keys'}, indent=1))
    print('unmapped', len(info['unmapped_source_keys']), info['unmapped_source_keys'][:8])


if __name__ == '__main__':
    main()
