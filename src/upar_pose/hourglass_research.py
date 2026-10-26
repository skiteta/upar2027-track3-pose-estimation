"""Configurable Hourglass with inference-active position/occlusion/identity heads.

This is an experimental local design, not an implementation of MIPNet or a
claim that attention identifies the correct person. Outputs remain 17 heatmaps.
"""
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F

from .hourglass_adapter import HourglassPose, CHECKPOINT_SHA256
from .hourglass_upstream import HourglassNet


class JointRelationBlock(nn.Module):
    """Small explicit attention avoids version-dependent fused export operators."""
    def __init__(self, width=128, heads=4):
        super().__init__()
        self.heads, self.depth = heads, width // heads
        self.norm1, self.norm2 = nn.LayerNorm(width), nn.LayerNorm(width)
        self.qkv, self.projection = nn.Linear(width, 3 * width), nn.Linear(width, width)
        self.ffn = nn.Sequential(nn.Linear(width, width * 2), nn.GELU(), nn.Linear(width * 2, width))

    def forward(self, tokens):
        batch, joints, width = tokens.shape
        qkv = self.qkv(self.norm1(tokens)).reshape(batch, joints, 3, self.heads, self.depth)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        attention = ((q @ k.transpose(-1, -2)).float() / self.depth ** .5).softmax(-1).to(v.dtype)
        tokens = tokens + self.projection((attention @ v).transpose(1, 2).reshape(batch, joints, width))
        return tokens + self.ffn(self.norm2(tokens))


def translate_heatmaps(maps, offsets, grid=None):
    """Positive dx/dy moves a peak right/down; units are heatmap pixels."""
    batch, joints, height, width = maps.shape
    if grid is None:
        yy, xx = torch.meshgrid(torch.linspace(-1, 1, height, device=maps.device),
                                torch.linspace(-1, 1, width, device=maps.device), indexing='ij')
        grid = torch.stack((xx, yy), -1)[None]
    # Fixed square64 contract. The inference caller supplies a registered buffer;
    # no traced CPU device constants are used to construct a grid on GPU.
    grid = grid[:, None] - (offsets.float() * (2. / 63))[:, :, None, None, :]
    moved = F.grid_sample(maps.float().reshape(batch * joints, 1, height, width),
                          grid.reshape(batch * joints, height, width, 2),
                          mode='bilinear', padding_mode='zeros', align_corners=True)
    return moved.reshape(batch, joints, height, width)


class PoseErrorHeads(nn.Module):
    def __init__(self, enabled=('position', 'occlusion', 'identity')):
        super().__init__()
        if not set(enabled) <= {'position', 'occlusion', 'identity'}:
            raise ValueError('Unknown error head')
        self.position, self.occlusion, self.identity = [name in enabled for name in ('position', 'occlusion', 'identity')]
        self.embedding = nn.Parameter(torch.randn(1, 17, 128) * .01)
        self.token_projection = nn.Linear(259, 128)
        self.relations = nn.Sequential(JointRelationBlock(), JointRelationBlock())
        self.offset = nn.Linear(128, 2)
        self.visible = nn.Linear(128, 1)
        self.completion = nn.Sequential(nn.Conv2d(403, 64, 1), nn.ReLU(), nn.Conv2d(64, 17, 3, padding=1))
        self.instance = nn.Sequential(nn.Conv2d(386, 64, 1), nn.ReLU(), nn.Conv2d(64, 1, 3, padding=1))
        for layer in (self.offset, self.completion[-1], self.instance[-1]):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)
        yy, xx = torch.meshgrid(torch.linspace(-1, 1, 64), torch.linspace(-1, 1, 64), indexing='ij')
        self.register_buffer('xy', torch.stack((xx, yy))[None])

    def forward(self, features, raw):
        batch = raw.shape[0]
        # Joint tokens attend to their current appearances, then exchange context.
        # Detaching spatial pooling does not detach features or final heatmap loss.
        distribution = (raw.detach().float().flatten(2) / .1).softmax(-1)
        appearances = distribution.to(features.dtype) @ features.flatten(2).transpose(1, 2)
        positions = distribution @ self.xy.flatten(2).transpose(1, 2)
        confidence = raw.detach().float().flatten(2).amax(-1, keepdim=True)
        tokens = self.relations(self.token_projection(torch.cat((appearances.float(), positions, confidence), -1)) + self.embedding)
        offsets = 4 * torch.tanh(self.offset(tokens).float())
        visible_logits = self.visible(tokens).squeeze(-1).float()
        context = tokens.mean(1)[:, :, None, None].expand(-1, -1, 64, 64)
        shared = torch.cat((features, context, self.xy.expand(batch, -1, -1, -1)), 1)
        identity_logits = self.instance(shared).float()
        completion = .2 * torch.tanh(self.completion(torch.cat((shared, raw), 1)).float())
        output = translate_heatmaps(raw, offsets, self.xy.permute(0, 2, 3, 1)) if self.position else raw.float()
        if self.occlusion:
            output = output + (1 - visible_logits.sigmoid())[:, :, None, None] * completion
        if self.identity:
            output = output * (1 + .5 * identity_logits.tanh())
        return output, {'offsets': offsets, 'visibility': visible_logits, 'identity': identity_logits, 'raw': raw}


class HourglassResearchPose(nn.Module):
    def __init__(self, stacks=1, error_heads=()):
        super().__init__()
        if stacks not in (1, 2):
            raise ValueError('This audited experiment supports one or two stacks')
        self.backbone = HourglassNet(num_stacks=stacks)
        self.keypoint_head = nn.Module()
        self.keypoint_head.multi_final_layers = nn.ModuleList([nn.Conv2d(256, 17, 1) for _ in range(stacks)])
        self.error_heads = PoseErrorHeads(error_heads) if error_heads else None
        self.register_buffer('coco_mean', torch.tensor([.485, .456, .406]).reshape(1, 3, 1, 1), persistent=False)
        self.register_buffer('coco_std', torch.tensor([.229, .224, .225]).reshape(1, 3, 1, 1), persistent=False)

    def forward_training(self, inputs):
        features = self.backbone((inputs * .25 + .5 - self.coco_mean) / self.coco_std)
        stages = [head(feature) for head, feature in zip(self.keypoint_head.multi_final_layers, features)]
        if self.error_heads is None:
            return stages[-1], stages, {}
        refined, auxiliary = self.error_heads(features[-1], stages[-1])
        return refined, stages, auxiliary

    def forward(self, inputs):
        return self.forward_training(inputs)[0]

    def initialize(self, checkpoint: Path, seed=20261005):
        # Use EXACT same audited initialization, including reset head semantics.
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed)
            original = HourglassPose()
            initialization = original.initialize(checkpoint)
        origin = original.state_dict()
        destination = self.state_dict()
        copied, duplicated, fresh = [], [], []
        for key, value in destination.items():
            source = key
            for stem in ('backbone.hourglass_modules.', 'backbone.out_convs.', 'keypoint_head.multi_final_layers.'):
                if source.startswith(stem + '1.'):
                    source = source.replace(stem + '1.', stem + '0.', 1)
            if source in origin and origin[source].shape == value.shape:
                destination[key] = origin[source].clone()
                (copied if source == key else duplicated).append({'target': key, 'source': source})
            else:
                fresh.append(key)
        self.load_state_dict(destination, strict=True)
        # A two-stack model is NOT a publicly pretrained HG104. The second stage
        # copies audited HG52 weights; new bridges/heads keep recorded init.
        return {**initialization, 'checkpoint_sha256': CHECKPOINT_SHA256,
                'initialization_method': 'HG52 transfer; explicit stage0-to-stage1 copy where present; new bridges/heads initialized locally',
                'copied': copied, 'duplicated': duplicated, 'new_parameters_and_buffers': fresh,
                'all_destination_keys_accounted': len(copied) + len(duplicated) + len(fresh) == len(destination)}


def masked_mean(values, weights):
    return (values * weights).sum() / weights.sum().clamp_min(1.)


def quadratic_peak_coordinates(heatmaps):
    """Detached GPU equivalent of the actual log-quadratic deployment decoder."""
    flat = heatmaps.detach().float().flatten(2)
    indices = flat.argmax(-1)
    x, y = indices % 64, indices // 64
    logs = flat.clamp_min(1e-10).log()
    sample = lambda index: logs.gather(-1, index.clamp(0, 4095).unsqueeze(-1)).squeeze(-1)
    center = sample(indices)
    before = torch.stack((sample(indices - 1), sample(indices - 64)), -1)
    after = torch.stack((sample(indices + 1), sample(indices + 64)), -1)
    denominator = before - 2 * center[..., None] + after
    valid = (denominator.abs() > 1e-8) & ((x > 0) & (x < 63) & (y > 0) & (y < 63))[..., None]
    safe = torch.where(valid, denominator, torch.ones_like(denominator))
    delta = torch.where(valid, (.5 * (before - after) / safe).clamp(-.5, .5), torch.zeros_like(safe))
    return torch.stack((x, y), -1).float() + delta


def research_loss(model, output, stages, auxiliary, targets, weights, coordinates, visibility):
    mse = lambda maps: masked_mean((maps.float() - targets.float()).square().mean((2, 3)), weights)
    final = mse(output)
    if model.error_heads is None:
        loss = torch.stack([mse(stage) for stage in stages]).mean()
        return loss, {'heatmap': final}
    terms = {'heatmap': final, 'intermediate': torch.stack([mse(stage) for stage in stages]).mean()}
    loss = .75 * final + .25 * terms['intermediate']
    heads = model.error_heads
    if heads.position:
        peaks = quadratic_peak_coordinates(auxiliary['raw'])
        # Residual beyond the existing subpixel decoder, not GT minus integer
        # argmax (which would incorrectly teach a second fractional correction).
        coordinates = torch.where(weights[..., None] > 0, coordinates, torch.zeros_like(coordinates))
        desired = coordinates - peaks
        local = weights * (desired.abs().amax(-1) <= 4).float()
        terms['position'] = masked_mean(F.smooth_l1_loss(auxiliary['offsets'] / 4, desired / 4, reduction='none').mean(-1), local)
        loss = loss + 1e-4 * terms['position']
    if heads.occlusion:
        terms['visibility'] = masked_mean(F.binary_cross_entropy_with_logits(auxiliary['visibility'], (visibility == 2).float(), reduction='none'), weights)
        loss = loss + 1e-4 * terms['visibility']
    if heads.identity:
        support = ((targets * weights[:, :, None, None]).amax(1, keepdim=True) > .1).float()
        terms['identity'] = (F.binary_cross_entropy_with_logits(auxiliary['identity'], support, reduction='none') * (1 + 9 * support)).mean()
        loss = loss + 1e-4 * terms['identity']
    return loss, terms
