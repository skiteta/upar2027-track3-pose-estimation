"""ViTPose-style top-down pose model on a ViT encoder initialised from COCO-only MAE weights.

Input/output contract matches RSNPose in this project: input is the project's
standardised RGB crop (x - 0.5) / 0.25 of shape [B, 3, 256, 256]; output is
peak-1 heatmaps [B, 17, 64, 64] (with image_size > 256 the same crop is resampled
inside the model and the heatmaps are image_size/4 square).  The encoder accepts the facebookresearch
long_seq_mae checkpoints pretrained only on COCO train2017 + unlabeled2017
images (random init, no labels); the decoder is the ViTPose "classic" head
(two stride-2 deconvolutions + 1x1 conv).
"""

from __future__ import annotations

import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
SPECS = {'vit_b': dict(dim=768, depth=12, heads=12), 'vit_l': dict(dim=1024, depth=24, heads=16),
         'vit_h': dict(dim=1280, depth=32, heads=16)}


class DropPath(nn.Module):
    def __init__(self, p: float):
        super().__init__()
        self.p = p

    def forward(self, x):
        if not self.training or self.p == 0.0:
            return x
        keep = 1.0 - self.p
        mask = x.new_empty((x.shape[0],) + (1,) * (x.ndim - 1)).bernoulli_(keep)
        return x * mask / keep


class Attention(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.heads = heads
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        b, n, c = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.heads, c // self.heads).permute(2, 0, 3, 1, 4)
        x = F.scaled_dot_product_attention(qkv[0], qkv[1], qkv[2])
        return self.proj(x.transpose(1, 2).reshape(b, n, c))


class Mlp(nn.Module):
    def __init__(self, dim, hidden):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class Block(nn.Module):
    def __init__(self, dim, heads, drop_path):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention(dim, heads)
        self.drop_path = DropPath(drop_path)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = Mlp(dim, dim * 4)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        return x + self.drop_path(self.mlp(self.norm2(x)))


class PatchEmbed(nn.Module):
    def __init__(self, dim, padding=0):
        super().__init__()
        self.proj = nn.Conv2d(3, dim, kernel_size=16, stride=16, padding=padding)
        # Export only (not in the state dict): zero-pad the input explicitly and run an unpadded convolution.
        # Same arithmetic; the deployed graph then holds the k16/s16/p0 convolution of the other ViT members.
        self.explicit_pad = False

    def forward(self, x):
        pad = int(self.proj.padding[0])
        if self.explicit_pad and pad:
            return F.conv2d(F.pad(x, (pad, pad, pad, pad)), self.proj.weight, self.proj.bias, stride=self.proj.stride)
        return self.proj(x)


class ViTPose(nn.Module):
    def __init__(self, arch='vit_l', image_size=256, drop_path=0.3, num_keypoints=17, deconv_channels=256, quality_head=False,
                 upsample_layers=2, patch_padding=0, with_cls_token=True):
        super().__init__()
        spec = SPECS[arch]
        dim, depth = spec['dim'], spec['depth']
        self.arch, self.grid = arch, image_size // 16
        self.grad_checkpoint = False
        self.upsample_layers = upsample_layers
        self.patch_embed = PatchEmbed(dim, patch_padding)
        # ImageNet-1K MAE -> COCO ViTPose weights (MMPose) use a padded patch embedding and drop the cls token before the
        # encoder; both are recorded as state markers so load_vit_weights rebuilds the same arithmetic.
        self.with_cls_token = with_cls_token
        if patch_padding:
            self.register_buffer('patch_padding_marker', torch.tensor(int(patch_padding)))
        if not with_cls_token:
            self.register_buffer('no_cls_token_marker', torch.tensor(1))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + self.grid * self.grid, dim))
        rates = [drop_path * i / max(depth - 1, 1) for i in range(depth)]
        self.blocks = nn.ModuleList(Block(dim, spec['heads'], r) for r in rates)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        layers, channels = [], dim
        for _ in range(upsample_layers):   # 2: grid x4 (ViTPose classic); 3: grid x8 finer heatmaps
            layers += [nn.ConvTranspose2d(channels, deconv_channels, 4, stride=2, padding=1, bias=False),
                       nn.BatchNorm2d(deconv_channels), nn.ReLU(inplace=True)]
            channels = deconv_channels
        self.decoder = nn.Sequential(*layers)
        self.final = nn.Conv2d(deconv_channels, num_keypoints, 1)
        nn.init.normal_(self.final.weight, std=0.001)
        nn.init.zeros_(self.final.bias)
        # Optional per-joint OKS-quality readout (detached features): predicts how well each
        # decoded joint will match, to rank persons by expected OKS instead of heatmap peaks.
        self.quality = None
        if quality_head:
            self.quality = nn.Sequential(nn.Linear(dim + num_keypoints, 512), nn.GELU(), nn.Linear(512, num_keypoints))
        self.register_buffer('rgb_mean', torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer('rgb_std', torch.tensor(IMAGENET_STD).view(1, 3, 1, 1), persistent=False)

    def load_mae(self, path: str | Path) -> dict:
        """Load a COCO-only MAE encoder; resize its 2D position grid to this input."""
        payload = torch.load(path, map_location='cpu', weights_only=False)
        state = payload.get('model', payload)
        state = {k: v for k, v in state.items() if not k.startswith('decoder') and k != 'mask_token'}
        pos = state['pos_embed']
        old = int(math.isqrt(pos.shape[1] - 1))
        grid = pos[:, 1:].reshape(1, old, old, -1).permute(0, 3, 1, 2)
        grid = F.interpolate(grid, size=(self.grid, self.grid), mode='bicubic', align_corners=False)
        state['pos_embed'] = torch.cat([pos[:, :1], grid.permute(0, 2, 3, 1).reshape(1, self.grid * self.grid, -1)], 1)
        missing, unexpected = self.load_state_dict(state, strict=False)
        allowed_missing = {k for k in self.state_dict() if k.startswith(('decoder.', 'final.'))}
        if set(missing) - allowed_missing or unexpected:
            raise ValueError(f'MAE load mismatch missing={sorted(set(missing) - allowed_missing)} unexpected={unexpected}')
        args = payload.get('args')
        args = vars(args) if args is not None and not isinstance(args, dict) else (args or {})
        return {'source': str(path), 'pos_grid_from': old, 'pos_grid_to': self.grid, 'mae_model': args.get('model'),
                'mae_data_path': args.get('data_path'), 'mae_epochs': args.get('epochs'), 'loaded_keys': len(state)}

    def features(self, x):
        side = self.grid * 16
        if x.shape[-1] != side or x.shape[-2] != side:
            # Internal-resolution variant: the project's 256x256 crop is resampled to the encoder
            # grid with the same corner-aligned convention as the heatmap targets.
            x = F.interpolate(x, size=(side, side), mode='bilinear', align_corners=True)
        rgb = x * 0.25 + 0.5
        x = (rgb - self.rgb_mean) / self.rgb_std
        x = self.patch_embed(x).flatten(2).transpose(1, 2)
        if self.with_cls_token:
            x = torch.cat([self.cls_token.expand(x.shape[0], -1, -1), x], 1) + self.pos_embed
        else:
            x = x + self.pos_embed[:, 1:]
        for block in self.blocks:
            if self.grad_checkpoint and self.training:
                # Activation recomputation only: same arithmetic and RNG stream (preserve_rng_state).
                x = torch.utils.checkpoint.checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)
        x = self.norm(x)
        if self.with_cls_token:
            x = x[:, 1:]
        b, n, c = x.shape
        return x.transpose(1, 2).reshape(b, c, self.grid, self.grid)

    def forward(self, x):
        return self.final(self.decoder(self.features(x)))

    def forward_with_quality(self, x):
        feats = self.features(x)
        maps = self.final(self.decoder(feats))
        if self.quality is None:
            return maps, None
        pooled = feats.detach().mean((2, 3))
        peaks = maps.detach().flatten(2).amax(-1)
        return maps, self.quality(torch.cat([pooled, peaks], 1))

    def forward_training(self, x):
        maps, quality = self.forward_with_quality(x)
        return maps, None, quality


def resize_pos_embed(state: dict, grid: int) -> dict:
    """Bicubically resample a weights file's 2D position grid (cls token kept) to ``grid`` x ``grid``."""
    pos = state['pos_embed']
    old = int(math.isqrt(pos.shape[1] - 1))
    if old == grid:
        return state
    patches = pos[:, 1:].reshape(1, old, old, -1).permute(0, 3, 1, 2).float()
    patches = F.interpolate(patches, size=(grid, grid), mode='bicubic', align_corners=False)
    resized = torch.cat([pos[:, :1].float(), patches.permute(0, 2, 3, 1).reshape(1, grid * grid, -1)], 1)
    return {**state, 'pos_embed': resized.to(pos.dtype)}


def load_vit_weights(path: str | Path, map_location='cpu') -> 'ViTPose':
    """Rebuild a ViTPose from a trainer weights file (architecture inferred from the state)."""
    state = torch.load(path, map_location=map_location, weights_only=False)['model_state']
    depth = 1 + max(int(k.split('.')[1]) for k in state if k.startswith('blocks.'))
    arch = {12: 'vit_b', 24: 'vit_l', 32: 'vit_h'}[depth]
    grid = int(math.isqrt(state['pos_embed'].shape[1] - 1))
    upsample = sum(1 for k in state if k.startswith('decoder.') and k.endswith('.weight') and state[k].ndim == 4)
    model = ViTPose(arch, image_size=grid * 16, drop_path=0.0, quality_head=any(k.startswith('quality.') for k in state),
                    upsample_layers=upsample, patch_padding=int(state['patch_padding_marker']) if 'patch_padding_marker' in state else 0,
                    with_cls_token='no_cls_token_marker' not in state)
    model.load_state_dict(state, strict=True)
    return model.eval()


SIGMAS = (0.026, 0.025, 0.025, 0.035, 0.035, 0.079, 0.079, 0.072, 0.072, 0.062, 0.062, 0.107, 0.107, 0.087, 0.087, 0.089, 0.089)


def quality_targets(maps, coords, weights, person_scale=32.0):
    """Per-joint OKS-term proxy of the current (detached) argmax decode vs GT, in heatmap pixels.

    person_scale approximates sqrt(box area) in 64x64 heatmap units for a 1.25-padded
    pedestrian crop (height about 51 px, aspect about 0.4).
    """
    b, k, h, w = maps.shape
    idx = maps.detach().flatten(2).argmax(-1)
    pred = torch.stack(((idx % w).float(), (idx // w).float()), -1)
    d2 = (pred - coords.float()).square().sum(-1)
    kappa = torch.tensor(SIGMAS, device=maps.device) * 2.0
    return torch.exp(-d2 / (2.0 * (kappa * person_scale) ** 2)), weights > 0


def gaussian_targets(coords64, size: int, sigma):
    """Peak-1 Gaussian targets on a size x size grid from coordinates in 64x64 heatmap units.

    Same corner-aligned convention and 3-sigma square window as make_heatmaps_highres, so a
    size-64/sigma-2 call reproduces the dataset targets; joints are weighted by the caller.
    ``sigma`` is a scalar or one value per joint (each joint keeps its own 3-sigma window).
    """
    centre = coords64.float() * ((size - 1) / 63.0)
    axis = torch.arange(size, device=coords64.device, dtype=torch.float32)
    sig = torch.as_tensor(sigma, dtype=torch.float32, device=coords64.device)
    sig = (sig.expand(coords64.shape[1]) if sig.ndim == 0 else sig).view(1, -1, 1, 1)
    radius = torch.ceil(3.0 * sig)
    offsets = axis.view(1, 1, 1, -1) - centre.unsqueeze(-1)              # [B, K, 2, size]
    window = (axis.view(1, 1, 1, -1) - centre.floor().unsqueeze(-1)).abs() <= radius
    profile = torch.exp(-offsets.square() / (2.0 * sig * sig)) * window
    return profile[:, :, 1, :, None] * profile[:, :, 0, None, :]


def vit_pose_loss(output, targets, weights):
    """Joint-weighted heatmap MSE in peak-1 units (same normalisation as rsn_loss's final stage)."""
    value = ((output.float() - targets.float()).square().mean((2, 3)) * weights).sum() / weights.sum().clamp_min(1.0)
    return value, {'heatmap': value}


def layer_decay_param_groups(model: ViTPose, base_lr: float, weight_decay: float, layer_decay: float):
    depth = len(model.blocks)
    groups = {}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith(('cls_token', 'pos_embed', 'patch_embed')):
            layer = 0
        elif name.startswith('blocks.'):
            layer = int(name.split('.')[1]) + 1
        else:
            layer = depth + 1
        no_decay = param.ndim == 1 or name.endswith('.bias') or name in ('cls_token', 'pos_embed')
        key = (layer, no_decay)
        if key not in groups:
            scale = layer_decay ** (depth + 1 - layer)
            groups[key] = {'params': [], 'lr': base_lr * scale, 'lr_scale': scale, 'weight_decay': 0.0 if no_decay else weight_decay}
        groups[key]['params'].append(param)
    return list(groups.values())
