"""Official HRFormer-B COCO backbone/head with the project's normalization API.

Adapted MMPose v0.29.0 (Apache-2.0); see hrformer_LICENSE.txt. No download,
dynamic config execution, full MMPose import, or custom CUDA operator is used.
Native first validation is 384x288 -> 96x72; no implicit input resizing.
"""
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .hrformer_upstream import HRFormer, HRFormerBlock
from .pipeline import sha256_file


OFFICIAL_SHA256 = 'ecf0758d7d482b40f71ab909c11c63d94c2c8045de53300ad915796656001152'
OFFICIAL_URL = ('https://download.openmmlab.com/mmpose/top_down/hrformer/'
                'hrformer_base_coco_384x288-ecf0758d_20220316.pth')
IMAGENET_INITIALIZATION_URL = ('https://download.openmmlab.com/mmpose/pretrain_models/'
                               'hrformer_base-32815020_20220226.pth')


def hrformer_base_extra():
    """Fresh mutable config, identical to the official 384x288 HRFormer-B."""
    result = dict(drop_path_rate=0.3, with_rpe=True)
    result['stage1'] = dict(num_modules=1, num_branches=1, block='BOTTLENECK',
                            num_blocks=(2,), num_channels=(64,), num_heads=[2], mlp_ratios=[4])
    for stage, modules, widths, heads in (
            (2, 1, (78, 156), [2, 4]),
            (3, 4, (78, 156, 312), [2, 4, 8]),
            (4, 2, (78, 156, 312, 624), [2, 4, 8, 16])):
        branches = len(widths)
        result[f'stage{stage}'] = dict(num_modules=modules, num_branches=branches,
            block='HRFORMERBLOCK', num_blocks=(2,) * branches, num_channels=widths,
            num_heads=heads, mlp_ratios=[4] * branches, window_sizes=[7] * branches)
    return result


def load_official_checkpoint(checkpoint: Path):
    """Pinned bytes + restricted loader, allowing only numpy scalar metadata.

    The original checkpoint contains a numpy validation-score scalar. Do not
    fall back to unrestricted pickle or globally alter safe globals.
    """
    checkpoint = Path(checkpoint)
    if sha256_file(checkpoint) != OFFICIAL_SHA256:
        raise ValueError('HRFormer official checkpoint SHA256 mismatch')
    from numpy.core.multiarray import scalar
    with torch.serialization.safe_globals([
            scalar, np.dtype, type(np.dtype(np.float64)), type(np.dtype(np.float32))]):
        payload = torch.load(checkpoint, map_location='cpu', weights_only=True)
    meta = payload['meta']
    config = meta['config']
    required = ("type='HRFormer'", "pretrained='models/hrformer_base.pth'",
                "ann_file='data/coco/annotations/person_keypoints_train2017.json'",
                'image_size=[288, 384]', 'heatmap_size=[72, 96]', 'sigma=3',
                'num_channels=(78, 156, 312, 624)', 'num_deconv_layers=0')
    if meta['epoch'] != 210 or any(item not in config for item in required):
        raise ValueError('Unexpected HRFormer fit boundary or architecture')
    return payload


class HRFormerPose(nn.Module):
    native_input_height_width = (384, 288)
    native_output_height_width = (96, 72)

    def __init__(self, *, internal_size=None, grad_checkpoint=False):
        super().__init__()
        if type(grad_checkpoint) is not bool:
            raise ValueError('grad_checkpoint must be a bool')
        if internal_size is not None and (isinstance(internal_size, bool)
                or not isinstance(internal_size, int) or internal_size < 32 or internal_size % 32):
            raise ValueError('internal_size must be None (native) or a positive multiple-of-32 square side')
        self.internal_size = internal_size
        # SyncBN -> local BN is an explicit single-process training adaptation;
        # running stats and affine state names remain unchanged for strict load.
        self.backbone = HRFormer(extra=hrformer_base_extra(),
                                 norm_cfg=dict(type='BN', requires_grad=True))
        self.keypoint_head = nn.Module()
        self.keypoint_head.final_layer = nn.Conv2d(78, 17, kernel_size=1)
        self.error_heads = None
        self.register_buffer('coco_mean', torch.tensor([.485, .456, .406]).view(1, 3, 1, 1), persistent=False)
        self.register_buffer('coco_std', torch.tensor([.229, .224, .225]).view(1, 3, 1, 1), persistent=False)
        self.set_grad_checkpoint(grad_checkpoint)

    def set_grad_checkpoint(self, enabled=True):
        """Opt in to attention-only training recomputation, never FFN/BN.

        Evaluation/no-grad and the default path are unchanged. This plain bool
        must be retained in the training config, not the model state_dict.
        """
        if type(enabled) is not bool:
            raise ValueError('grad_checkpoint must be a bool')
        self.grad_checkpoint = enabled
        for block in self.backbone.modules():
            if isinstance(block, HRFormerBlock):
                block.attention_checkpoint = enabled
        return self

    def prepare_inputs(self, inputs):
        if self.internal_size is not None:
            if inputs.shape[-2] != inputs.shape[-1]:
                raise ValueError('Square HRFormer adaptation requires square input crops; no aspect squeezing')
            # Matches ViTPose.features and gaussian_targets endpoint coordinates.
            # This is a new fine-tuning geometry, not the pretrained native recipe.
            inputs = nn.functional.interpolate(inputs, size=(self.internal_size, self.internal_size),
                                                mode='bilinear', align_corners=True)
        elif inputs.shape[-2:] != self.native_input_height_width:
            raise ValueError('Native HRFormer requires 384x288; select explicit internal_size for square adaptation')
        normalized = (inputs * .25 + .5 - self.coco_mean) / self.coco_std
        return normalized

    def forward(self, inputs):
        return self.keypoint_head.final_layer(self.backbone(self.prepare_inputs(inputs))[0])

    def forward_training(self, inputs):
        maps = self(inputs)
        return maps, [maps], {}

    def initialize(self, checkpoint: Path, *, reset_head=True):
        payload = load_official_checkpoint(checkpoint)
        self.load_state_dict(payload['state_dict'], strict=True)
        if reset_head:
            with torch.no_grad():
                for joint in (1, 2):
                    nn.init.normal_(self.keypoint_head.final_layer.weight[joint], std=.001)
                    self.keypoint_head.final_layer.bias[joint].zero_()
        return {
            'checkpoint_sha256': OFFICIAL_SHA256, 'checkpoint_url': OFFICIAL_URL,
            'strict_state_load': True, 'upstream_epoch': payload['meta']['epoch'],
            'upstream_config': payload['meta']['config'],
            'pretraining_history': ['ImageNet-1K', 'COCO train2017'],
            'imagenet_initialization_url': IMAGENET_INITIALIZATION_URL,
            'pretraining_evidence_limit': 'Upstream config/documentation chain; COCO checkpoint records a local initialization filename, not the initialization binary SHA.',
            'upstream_input_height_width': [384, 288], 'upstream_heatmap_height_width': [96, 72],
            'input_resize': None if self.internal_size is None else {
                'height_width': [self.internal_size, self.internal_size],
                'interpolation': 'bilinear', 'align_corners': True, 'aspect_rule': 'square-only'},
            'adapted_heatmap_height_width': [96, 72] if self.internal_size is None else [self.internal_size // 4] * 2,
            'normalization': 'project RGB (.5,.25) -> ImageNet RGB mean/std',
            'bn_adaptation': 'SyncBatchNorm -> BatchNorm2d, identical evaluation state',
            'output_units': 'peak-one Gaussian heatmaps; no rescaling',
            'reinitialized_channels': [1, 2] if reset_head else [], 'loss_ignored_channels': [3, 4],
            'method_limit': 'Not yet organizer fine-tuned or evaluated; COCO AP is not a Public forecast.'}


def hrformer_loss(model, output, stages, auxiliary, targets, weights, coordinates, visibility):
    """Project peak-one weighted MSE with explicit ignored COCO-ear channels."""
    if output.shape != targets.shape or weights.shape != output.shape[:2]:
        raise ValueError('HRFormer output/target/weight shape mismatch')
    used_weights = weights.clone()
    used_weights[:, [3, 4]] = 0
    errors = (output.float() - targets.float()).square().mean((2, 3))
    loss = (errors * used_weights).sum() / used_weights.sum().clamp_min(1.)
    return loss, {'heatmap': loss}


def hrformer_optimizer_groups(model, weight_decay=0.01):
    """Explicit AdamW policy from upstream: relative position bias has no WD.

    Use a separately selected local learning rate, not ViT block-index decay.
    """
    decay, no_decay = [], []
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            (no_decay if 'relative_position_bias_table' in name else decay).append(parameter)
    return [{'params': decay, 'weight_decay': weight_decay}, {'params': no_decay, 'weight_decay': 0.0}]


def load_hrformer_weights(path, *, config_path=None):
    """Load organizer weights with their explicit geometry, never guess a grid.

    Unlike ViT absolute positions, HRFormer parameters cannot identify the fit
    resolution. Preserve the source config beside checkpoints or pass its path.
    """
    path = Path(path)
    config_path = Path(config_path) if config_path is not None else path.parent / 'config.json'
    if not config_path.is_file():
        raise ValueError('HRFormer weights need their source config.json to establish square geometry')
    config = json.loads(config_path.read_text())
    if config.get('model_type') != 'hrformer' or config.get('input_size') != [256, 256]:
        raise ValueError('HRFormer source config must declare model_type and square256 input')
    if config.get('hrformer_internal_size') is None:
        raise ValueError('HRFormer source config lacks explicit internal square size')
    payload = torch.load(path, map_location='cpu', weights_only=True)
    if not payload.get('config_sha256') or payload['config_sha256'] != config.get('config_sha256'):
        raise ValueError('HRFormer weights and source config identity disagree')
    model = HRFormerPose(internal_size=config['hrformer_internal_size'])
    model.load_state_dict(payload['model_state'], strict=True)
    return model.eval()
