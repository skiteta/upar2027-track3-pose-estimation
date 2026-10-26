#!/usr/bin/env python3
"""Continue a trained RSN on a declared domain population with matched compute.

Purpose: measure whether adding unseen camera domains (PETA sub-datasets) to the
fit improves accuracy on *other* unseen camera domains, and produce an
all-source candidate.  Every arm starts from the same checkpoint (model and
AdamW state), uses the same recipe/augmentation/LR and the same number of
optimizer steps per epoch; only the training population differs.

Evaluation uses the parent six-view validator on the held-out PETA camera
domains only and reports pooled and subset-mean (official-style) metrics.
Held-out images are refused from the training population by construction.
"""
import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]
from continue_pose_research import save_checkpoint, relative  # noqa: E402
from train_scratch_hrnet_v4 import seed_everything  # noqa: E402
from train_scratch_cnn import load_domain, dataset_items  # noqa: E402
from pose_factor_data import FactorDataset  # noqa: E402
from pose_square_adapter import SquareInteractionDataset  # noqa: E402
from evaluate_resolution_ensemble import aligned_model_heatmaps, check_ids  # noqa: E402
from evaluate_seed_geometry_ensemble_v6 import PADDINGS, TARGET_PADDING, prediction  # noqa: E402
from upar_pose.rsn_adapter import RSNPose, rsn_loss  # noqa: E402
from upar_pose.hrformer_adapter import HRFormerPose, hrformer_loss, hrformer_optimizer_groups, OFFICIAL_SHA256 as HRFORMER_COCO_SHA256  # noqa: E402
from upar_pose.vit_pose import ViTPose, vit_pose_loss, layer_decay_param_groups, quality_targets, gaussian_targets, resize_pos_embed  # noqa: E402
from upar_pose.flip_tta import FLIP_CHANNELS, decode_heatmaps_refined  # noqa: E402
from upar_pose.oks_loss import oks_aware_vit_pose_loss  # noqa: E402
from upar_pose.domain_eval import evaluate_domain_structure, image_domains, load_peta_domain_map, subset_ground_truth  # noqa: E402
from upar_pose.pipeline import load_json, sha256_file, write_json_atomic  # noqa: E402
from upar_pose.scratch_cnn import IMAGE_MEAN, IMAGE_STD, normalize_image  # noqa: E402
from upar_pose.evaluation_map_cache import EvaluationMapCache, aligned_v4_from_flip_means  # noqa: E402
from upar_pose.teacher_storage import convert_frozen_affine_parameters_to_bf16  # noqa: E402

SOURCE_PREFIXES = ('market/', 'pa100k/')


def validate_hrformer_recipe(config):
    """Explicit new fit geometry; never silently inherit ViT-only knobs."""
    if config.get('model_type') != 'hrformer':
        return
    side = config.get('hrformer_internal_size')
    if isinstance(side, bool) or not isinstance(side, int) or side < 32 or side % 32:
        raise ValueError('HRFormer requires explicit square hrformer_internal_size divisible by 32')
    if config.get('input_size') != [256, 256] or config.get('quality_head'):
        raise ValueError('HRFormer common trainer requires square256 inputs and no quality head')
    if config.get('layer_decay', 1.0) != 1.0 or config.get('coord_loss') or config.get('joint_loss_weights'):
        raise ValueError('HRFormer recipe does not silently reuse ViT layer decay or OKS loss knobs')
    if config.get('drop_path', .3) != .3:
        raise ValueError('HRFormer integration fixes official drop_path=.3')
    checkpointed = config.get('grad_checkpoint', False)
    scope = config.get('hrformer_checkpoint_scope')
    if type(checkpointed) is not bool or scope not in (None, 'attention_only_nonreentrant'):
        raise ValueError('HRFormer checkpoint needs a bool and supported explicit attention-only scope')
    if checkpointed and scope != 'attention_only_nonreentrant':
        raise ValueError('HRFormer grad_checkpoint requires explicit attention_only_nonreentrant scope')
    if config.get('init_mode') not in ('coco', 'weights', 'checkpoint'):
        raise ValueError('HRFormer init_mode must be coco, weights, or checkpoint')
    if config['init_mode'] == 'coco' and (config.get('init_checkpoint_sha256') != HRFORMER_COCO_SHA256
            or config.get('hrformer_reset_head') is not True):
        raise ValueError('HRFormer COCO initialization requires official SHA and explicit head1/2 reset')


def validate_hrformer_continuation_source(config, source):
    if source.get('model_type') != 'hrformer' or source.get('hrformer_internal_size') != config['hrformer_internal_size']:
        raise ValueError('HRFormer continuation architecture/geometry changed')
    if config['evaluation_role'] == 'development':
        heldout = set(config['heldout_subdomains'])
        if (source.get('evaluation_role') != 'development'
                or not heldout <= set(source.get('heldout_subdomains', []))
                or source.get('fit_includes_heldout')
                or heldout & set(source.get('peta_train_subdomains', []))):
            raise ValueError('HRFormer continuation source crosses or does not establish development fit boundary')


def model_heatmap_side(model):
    if type(model) is HRFormerPose:
        if model.internal_size is None:
            raise ValueError('Native rectangular HRFormer is not a square-training/evaluation model')
        return model.internal_size // 4
    # Unchanged RSN / ViT computation; do not pretend HRFormer has ViT blocks.
    return getattr(model, 'grid', 16) * 2 ** getattr(model, 'upsample_layers', 2)


def hrformer_training_loss(model, output, stages, auxiliary, targets, weights, coordinates, visibility, config):
    if output.shape[-2] != output.shape[-1]:
        raise ValueError('HRFormer common loss requires a square output grid')
    size = output.shape[-1]
    if targets.shape[-2:] != (size, size) or config.get('heatmap_sigma_joint_scale') is not None:
        sigma = config.get('heatmap_sigma_out', config['heatmap_sigma'] * (size - 1) / 63.0)
        if config.get('heatmap_sigma_joint_scale') is not None:
            sigma = [sigma * float(f) for f in config['heatmap_sigma_joint_scale']]
        targets = gaussian_targets(coordinates, size, sigma)
    # Explicit ear masking and masked-weight denominator. Dataset already sets
    # ears to zero, so usual batches match weighted peak-one MSE normalization.
    return hrformer_loss(model, output, stages, auxiliary, targets, weights, coordinates, visibility)


def scheduled_learning_rate(config, epoch, batch_index=0, batches_per_epoch=1):
    """Keep legacy recipes unchanged; step warmup is an explicit new recipe knob."""
    lr = float(config['learning_rate'])
    for start_epoch, value in config.get('lr_schedule', []):
        if epoch >= start_epoch:
            lr = float(value)
    warmup = float(config.get('warmup_epochs', 0))
    mode = config.get('warmup_mode', 'epoch')
    if mode not in ('epoch', 'linear_steps'):
        raise ValueError(f'Unknown warmup_mode: {mode}')
    if warmup < 0 or batches_per_epoch < 1:
        raise ValueError('Warmup must be nonnegative and batches_per_epoch positive')
    if warmup:
        progress = epoch if mode == 'epoch' else (epoch - 1) + (batch_index + 1) / batches_per_epoch
        lr *= min(1.0, progress / warmup)
    return lr


def validate_resume_config(config, saved_config):
    """A different recipe needs a new revision, never silently reuse --resume."""
    generated = {'config_sha256', 'train_examples', 'train_population', 'eval_persons', 'init_epoch', 'audit_only'}
    previous = {k: v for k, v in saved_config.items() if k not in generated}
    changed = sorted(k for k in set(previous) | set(config) if previous.get(k) != config.get(k))
    if changed:
        raise ValueError(f'Resume config changed ({", ".join(changed)}); use a new revision')


def improves_validation(score, validations):
    if not math.isfinite(score):
        raise ValueError('Nonfinite validation AP; refusing a best-model claim')
    return not validations or score > max(float(v['subset_mean']['AP']) for v in validations)


def preserved_validations(validations, root):
    """Observed high scores without weights are not deployable checkpoints."""
    return [v for v in validations if v.get('weights') and (root / v['weights']).is_file()]


def validation_selection(validations, root):
    observed = max(validations, key=lambda v: v['subset_mean']['AP']) if validations else None
    preserved = preserved_validations(validations, root)
    available = max(preserved, key=lambda v: v['subset_mean']['AP']) if preserved else None
    if available:
        actual_hash = sha256_file(root / available['weights'])
        if available.get('weights_sha256') and available['weights_sha256'] != actual_hash:
            raise ValueError('Best available validation weights changed; refusing selection')
        available = {**available, 'weights_sha256': actual_hash}
    return {'best_observed': observed, 'best_available_weights': available,
            'observed_best_has_weights': bool(observed and observed.get('weights')
                                              and (root / observed['weights']).is_file())}


def save_epoch_weights(path, payload):
    """Epoch artifacts may be consumed by soups/packages; never replace them on replay."""
    if path.exists():
        raise FileExistsError(f'Epoch weights already exist: {path}; preserve them and use a new revision')
    save_checkpoint(path, payload)


def restore_adamw_moments(optimizer, saved):
    """Restore same-architecture moments, retaining this revision's LR settings."""
    current_groups = optimizer.param_groups
    saved_groups = saved['param_groups']
    if len(current_groups) != len(saved_groups):
        raise ValueError('AdamW parameter-group count changed')
    if not saved.get('state'):
        raise ValueError('Checkpoint has no AdamW moments to continue')
    desired_lrs = []
    for current, previous in zip(current_groups, saved_groups):
        if len(current['params']) != len(previous['params']):
            raise ValueError('AdamW parameter-group membership changed')
        for key in ('lr_scale', 'weight_decay', 'betas', 'eps', 'amsgrad'):
            if current.get(key) != previous.get(key):
                raise ValueError(f'AdamW {key} changed; an optimizer-reset recipe is required')
        desired_lrs.append(current['lr'])
        for parameter, old_id in zip(current['params'], previous['params']):
            state = saved['state'].get(old_id)
            if state is None:  # Frozen/unused parameters may not yet own moments.
                continue
            if not {'exp_avg', 'exp_avg_sq', 'step'} <= set(state):
                raise ValueError('Checkpoint is not an AdamW moment state')
            if any(state[key].shape != parameter.shape for key in ('exp_avg', 'exp_avg_sq')):
                raise ValueError('AdamW moment shape changed')
    optimizer.load_state_dict(saved)
    for group, lr in zip(optimizer.param_groups, desired_lrs):
        group['lr'] = lr


def validate_vit_continuation_source(config, source):
    defaults = {'model_type': 'rsn', 'vit_arch': None, 'vit_image_size': 256, 'vit_upsample_layers': 2,
                'vit_patch_padding': 0, 'vit_with_cls_token': True, 'quality_head': False}
    for key, default in defaults.items():
        if config.get(key, default) != source.get(key, default):
            raise ValueError(f'ViT checkpoint architecture changed: {key}')
    if config['evaluation_role'] == 'development':
        heldout = set(config['heldout_subdomains'])
        if (source.get('evaluation_role') != 'development'
                or not heldout <= set(source.get('heldout_subdomains', []))
                or source.get('fit_includes_heldout')
                or heldout & set(source.get('peta_train_subdomains', []))):
            raise ValueError('ViT continuation source crosses or does not establish the development fit boundary')


def validate_teacher_first_models(teachers, allowed_classes):
    # Process RNG checks cannot detect arbitrary private generators in unknown
    # user-defined models. Opt-in execution supports only the reviewed classes.
    if any(type(teacher) not in allowed_classes for teacher, _ in teachers):
        raise ValueError('Teacher-first does not support this teacher architecture')


def distillation_supervision_definition(spec):
    if not spec:
        return None
    mode = spec.get('supervision', 'all_scored')
    if mode not in ('all_scored', 'gt_labelled', 'gt_supported'):
        raise ValueError(f'Unknown distill.supervision: {mode}')
    if mode == 'gt_supported':
        return {
            'mode': mode,
            'mask': '(original batch GT joint weight > 0) AND scored joint, multiplied by detached positive teacher mass inside exact existing GT Gaussian nonzero support / total positive teacher mass',
            'target': 'detached nonnegative teacher map times (actual training GT Gaussian > 0); no peak renormalization; zero teacher mass gives zero gate',
            'normalization': 'full spatial MSE per joint; gated-joint sum per sample, batch mean, divide by the same fixed 15 scored joints; no valid-count or supported-pixel renormalization',
            'interpretation': 'new D recipe; v1 and v2 both included through original GT weights > 0; not annotation correction; incorrect GT or broad correct modes can reject useful teacher signal'}
    return {
        'mode': mode,
        'mask': '15 scored joints, independent of GT weights' if mode == 'all_scored' else
                '(original batch GT joint weight > 0, after augmentation and before KD) AND scored joint; positive weight magnitude is not used',
        'normalization': 'spatial MSE per joint; scored-joint sum per sample, batch mean, divide by 15 (legacy arithmetic)' if mode == 'all_scored' else
                         'spatial MSE per joint; masked-joint sum per sample, batch mean, divide by the same fixed 15 scored joints; no valid-count renormalization; zero-valid samples contribute finite zero',
        'interpretation': 'pseudo-label supervision on unlabelled joints is intentional in all_scored; gt_labelled is a new comparison recipe, not a correction of old results'}


def gt_supported_training_targets(output, targets, coordinates, config):
    """Exact GT used by this trainer, not a resized/threshold-tuned proxy.

    Dataset coordinates and targets use the fixed 64-grid/sigma-2 convention.
    ViT/HRFormer redraw on their square output grid exactly as their GT loss does.
    Reject unsupported or ignored sigma/geometry settings rather than infer them.
    This helper is called only for the explicit gt_supported recipe.
    """
    if (output.ndim != 4 or output.shape[1] != 17 or output.shape[-2] != output.shape[-1]
            or output.shape[-1] < 2 or config.get('input_size') != [256, 256]
            or targets.shape != (*output.shape[:2], 64, 64)
            or coordinates.shape != (*output.shape[:2], 2)):
        raise ValueError('gt_supported requires square student maps, square256 inputs and original [B,17,64,64] GT / coords64')
    kind, size = config.get('model_type', 'rsn'), output.shape[-1]
    if kind not in ('rsn', 'vit', 'hrformer'):
        raise ValueError('gt_supported does not support this model geometry')
    scales = config.get('heatmap_sigma_joint_scale')
    redraw = size != 64 or scales is not None
    if kind == 'rsn' and (size != 64 or scales is not None or 'heatmap_sigma_out' in config):
        raise ValueError('gt_supported RSN requires its existing 64-grid dataset GT; output sigma overrides are unsupported')
    if not redraw:
        if config.get('heatmap_sigma') != 2 or config.get('heatmap_sigma_out', 2) != 2:
            raise ValueError('gt_supported refuses sigma settings ignored by the fixed sigma-2 dataset GT')
        return targets.detach()
    sigma = config.get('heatmap_sigma_out', config['heatmap_sigma'] * (size - 1) / 63.0)
    if isinstance(sigma, bool) or not isinstance(sigma, (int, float)) or not math.isfinite(sigma) or sigma <= 0:
        raise ValueError('gt_supported requires a positive finite scalar heatmap sigma')
    if scales is not None:
        if (not isinstance(scales, (list, tuple)) or len(scales) != 17
                or any(isinstance(f, bool) or not isinstance(f, (int, float))
                       or not math.isfinite(f) or f <= 0 for f in scales)):
            raise ValueError('gt_supported requires 17 positive finite joint sigma scales')
        sigma = [sigma * float(f) for f in scales]
    return gaussian_targets(coordinates.detach(), size, sigma).detach()


def gt_supported_teacher_target(guide, gt_targets, gt_joint_weights, scored_mask):
    """Detached GT-supported target/gate; keep acceptable non-argmax modes."""
    if gt_targets is None or guide.ndim != 4 or guide.shape != gt_targets.shape or guide.shape[1] != 17:
        raise ValueError('gt_supported teacher and actual GT must share [B,17,H,W] shape')
    if gt_joint_weights.shape != guide.shape[:2] or scored_mask.shape != (17,):
        raise ValueError('gt_supported GT joint weights/scored mask shape mismatch')
    expected = torch.ones_like(scored_mask)
    expected[[3, 4]] = 0
    if not torch.equal(scored_mask, expected):
        raise ValueError('gt_supported requires exactly the fixed 15 scored joints (ears excluded)')
    with torch.no_grad():
        positive = guide.detach().float().clamp_min(0)
        target = positive * (gt_targets.detach() > 0)
        total = positive.sum((2, 3))
        retained = target.sum((2, 3))
        ratio = torch.where(total > 0, retained / total.clamp_min(torch.finfo(positive.dtype).tiny),
                            torch.zeros_like(total)).clamp(0, 1)
        gate = ratio * (gt_joint_weights.detach() > 0).to(positive.dtype) * scored_mask.detach()
    return target, gate


def distillation_heatmap_loss(output, guide, gt_joint_weights, scored_mask, supervision='all_scored', *, gt_targets=None):
    if supervision == 'gt_supported':
        if output.shape != guide.shape:
            raise ValueError('gt_supported student and teacher maps must share shape')
        target, gate = gt_supported_teacher_target(guide, gt_targets, gt_joint_weights, scored_mask)
        return ((output.float() - target).square().mean((2, 3)) * gate).sum(1).mean() / scored_mask.sum()
    per_joint = (output.float() - guide).square().mean((2, 3))
    if supervision == 'all_scored':
        # Preserve the original operation order exactly, including the placement
        # of the batch mean and division; GT weights intentionally do not enter.
        return (per_joint * scored_mask).sum(1).mean() / scored_mask.sum()
    if supervision != 'gt_labelled':
        raise ValueError(f'Unknown distill.supervision: {supervision}')
    if gt_joint_weights.shape != per_joint.shape:
        raise ValueError('GT joint weights must match the [batch, joints] KD map shape')
    valid = (gt_joint_weights > 0).to(per_joint.dtype) * scored_mask
    # Keep lambda's scale fixed: this arm removes unlabelled-joint KD, without
    # simultaneously increasing the weight of each remaining labelled joint.
    return (per_joint * valid).sum(1).mean() / scored_mask.sum()


def positive_finite_gib(value):
    if isinstance(value, bool):
        raise ValueError('allocator cap must be positive finite GiB')
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError('allocator cap must be positive finite GiB')
    return value


def allocator_cap_execution(config, override=None, device_total_bytes=None):
    """Execution-only ceiling; never mutate the immutable training recipe."""
    cap = positive_finite_gib(config['allocator_cap_gib'] if override is None else override)
    fraction = None
    if device_total_bytes is not None:
        if not math.isfinite(device_total_bytes) or device_total_bytes <= 0:
            raise ValueError('allocator cap requires actual positive finite device total memory')
        fraction = cap * 1024**3 / device_total_bytes
        if fraction > 1:
            raise ValueError('allocator cap exceeds actual device total memory')
    return {'config_gib': config['allocator_cap_gib'], 'override_gib': override,
            'effective_gib': cap, 'device_total_bytes': device_total_bytes, 'fraction': fraction,
            'scope': 'PyTorch process allocator ceiling, not a free-memory reading or resource admission'}


def completion_event_target(config, root, event_path_override=None, event_id_override=None):
    relative_path = event_path_override or config.get('event_path')
    if not relative_path:
        if event_id_override:
            raise ValueError('--event-id requires an event path')
        return None, None
    target = (root / relative_path).resolve()
    if not target.is_relative_to(root.resolve()):
        raise ValueError('Completion event must stay inside this project')
    if target.exists():
        raise FileExistsError(f'Completion event already exists: {target}; preserve it and choose a new event path')
    event_id = event_id_override or (target.stem if event_path_override else config['candidate'] + '_fit_completed')
    return target, event_id


def checked_teacher_maps(teachers, inputs, size):
    """Teacher-first is valid only for frozen eval teachers that consume no RNG.

    This guards process-global RNGs on every call. It does not establish CUDA
    numerical parity or peak-memory savings; those require a separate GPU test.
    """
    for teacher, _ in teachers:
        if any(module.training for module in teacher.modules()) or any(p.requires_grad for p in teacher.parameters()):
            raise ValueError('Teacher-first requires fully eval, frozen teachers')
    cpu_rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state(inputs.device) if inputs.is_cuda else None
    python_rng, numpy_rng = random.getstate(), np.random.get_state()
    guide = teacher_maps(teachers, inputs, size)
    numpy_after = np.random.get_state()
    unchanged = (torch.equal(cpu_rng, torch.get_rng_state()) and python_rng == random.getstate()
                 and numpy_rng[0] == numpy_after[0] and np.array_equal(numpy_rng[1], numpy_after[1])
                 and numpy_rng[2:] == numpy_after[2:])
    if cuda_rng is not None:
        unchanged = unchanged and torch.equal(cuda_rng, torch.cuda.get_rng_state(inputs.device))
    if not unchanged:
        raise ValueError('Teacher-first rejected a stochastic teacher: process RNG changed')
    if guide.requires_grad:
        raise ValueError('Teacher-first guide must be detached')
    return guide


class DomainFactorDataset(FactorDataset):
    """FactorDataset whose Market/PA-only guard is replaced by an explicit population check.

    The parent guard protects PETA-holdout fits.  Here the caller has already
    removed every held-out image (verified in ``build_population``), so PETA
    training-group items may enter the fit and the donor pool.
    """

    def __init__(self, items, recipe, **kwargs):
        super().__init__([i for i in items if str(i['file_name']).startswith(SOURCE_PREFIXES)], recipe, **kwargs)
        self.items = list(items)


def degrade_camera_quality(pixels, rng, settings):
    """Surveillance-camera quality degradation of a square crop; geometry unchanged.

    Colour cast/gamma -> downscale to a tiny effective side (random filter) ->
    sensor noise -> JPEG at that resolution -> bilinear upscale back to 256.
    Covers the low-information regime (effective side 40-200 px for the whole
    square) that the existing information axis (125-190 px, mild blur) does not.
    """
    import io
    from PIL import Image
    image = pixels.astype(np.float32)
    if rng.random() < settings['color_probability']:
        gains = rng.uniform(*settings['channel_gain'], size=3)
        gamma = float(np.exp(rng.uniform(np.log(settings['gamma'][0]), np.log(settings['gamma'][1]))))
        image = 255.0 * np.clip(image / 255.0, 0, 1) ** gamma * gains
    image = Image.fromarray(np.clip(image, 0, 255).astype(np.uint8))
    side = int(round(np.exp(rng.uniform(np.log(settings['effective_side'][0]), np.log(settings['effective_side'][1])))))
    resample = [Image.Resampling.NEAREST, Image.Resampling.BILINEAR, Image.Resampling.BOX][int(rng.integers(3))]
    small = image.resize((side, side), resample)
    if rng.random() < settings['noise_probability']:
        array = np.asarray(small, dtype=np.float32) + rng.normal(0, rng.uniform(*settings['noise_sigma']), (side, side, 3))
        small = Image.fromarray(np.clip(array, 0, 255).astype(np.uint8))
    if rng.random() < settings['jpeg_probability']:
        buffer = io.BytesIO()
        small.save(buffer, format='JPEG', quality=int(rng.integers(*settings['jpeg_quality'])))
        buffer.seek(0)
        small = Image.open(buffer).convert('RGB')
    return np.asarray(small.resize((256, 256), Image.Resampling.BILINEAR), dtype=np.uint8)


class CameraQualityDataset(torch.utils.data.Dataset):
    """Applies degrade_camera_quality to the already augmented input with probability p."""

    def __init__(self, base, settings):
        self.base, self.settings = base, settings
        self.rng = np.random.default_rng(settings['seed'])
        self.applied = 0

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        sample = self.base[index]
        if self.rng.random() >= self.settings['probability']:
            return sample
        inputs = sample[0].numpy()
        pixels = np.rint(np.clip((inputs * IMAGE_STD + IMAGE_MEAN).transpose(1, 2, 0) * 255, 0, 255)).astype(np.uint8)
        degraded = degrade_camera_quality(pixels, self.rng, self.settings)
        self.applied += 1
        return (torch.from_numpy(normalize_image(degraded)),) + tuple(sample[1:])


def posetrack_items(c):
    """PoseTrack18 labelled persons for pre-training (Terms 3 permit PoseTrack18 pre-training)."""
    items = []
    for rel_path in c['posetrack18_annotation_files']:
        data = json.loads((ROOT / rel_path).read_text())
        sizes = {int(i['id']): (float(i['width']), float(i['height'])) for i in data['images']}
        for r in dataset_items(data, require_labels=True):
            x, y, w, h = r['box']
            width, height = sizes[r['image_id']]
            # A few official boxes are corrupt (e.g. -1.15e6 offsets); keep only boxes
            # that overlap the frame with a plausible size.
            if not (np.isfinite([x, y, w, h]).all() and 0 < w <= 1.2 * width and 0 < h <= 1.2 * height
                    and -0.5 * width < x < width and -0.5 * height < y < height):
                continue
            labelled = int((r['keypoints'][:, 2] > 0).sum())
            if labelled < c.get('posetrack18_min_labelled', 6) or r['box'][3] < c.get('posetrack18_min_box_height', 32):
                continue
            # Extra roots (e.g. official PoseTrack 2017 frames mapped to PT18 names) are read only when a config lists them.
            roots = c.get('posetrack18_image_roots', ['external/posetrack18'])
            rel = next((f'{root}/' + r['file_name'] for root in roots if (ROOT / 'data' / root / r['file_name']).is_file()),
                       f'{roots[0]}/' + r['file_name'])
            if c.get('posetrack18_require_image', True) and not (ROOT / 'data' / rel).is_file():
                continue
            items.append({**r, 'file_name': rel})
    return items


def coco_items(c):
    """COCO person keypoints for pre-training (Terms 3 permit COCO pre-training).

    COCO and the UPAR/PoseTrack17 layout share nose, ears and the 12 body joints;
    COCO eyes (slots 1, 2) are dropped because those slots are head_bottom/head_top
    here, which COCO does not label (visibility 0 -> no loss on those channels).
    """
    items = []
    for rel_path in c['coco_annotation_files']:
        data = json.loads((ROOT / rel_path).read_text())
        image_dir = c['coco_image_dirs'][rel_path]
        data['annotations'] = [a for a in data['annotations'] if not a.get('iscrowd', 0)]
        for a in data['annotations']:
            kp = list(a['keypoints'])
            kp[3:9] = [0, 0, 0, 0, 0, 0]
            a['keypoints'] = kp
        for r in dataset_items(data, require_labels=True):
            labelled = int((r['keypoints'][:, 2] > 0).sum())
            if labelled < c.get('coco_min_labelled', 6) or r['box'][3] < c.get('coco_min_box_height', 32):
                continue
            rel = f'{image_dir}/' + r['file_name'].rsplit('/', 1)[-1]
            if not (ROOT / 'data' / rel).is_file():
                continue
            items.append({**r, 'file_name': rel})
    return items


PRETRAINING_DOMAINS = {'posetrack18', 'coco'}


def build_population(c, domain_of_basename):
    items = []
    pretraining = set(c['train_domains']) & PRETRAINING_DOMAINS
    if pretraining and set(c['train_domains']) - PRETRAINING_DOMAINS:
        raise ValueError('COCO/PoseTrack18 are pre-training data only and must not be mixed into organiser-data fine-tuning')
    for domain in c['train_domains']:
        if domain == 'posetrack18':
            items.extend(posetrack_items(c))
            continue
        if domain == 'coco':
            items.extend(coco_items(c))
            continue
        rows = dataset_items(load_domain(domain), require_labels=True)
        if domain == 'peta':
            allowed = set(c['peta_train_subdomains'])
            rows = [r for r in rows if domain_of_basename[r['file_name'].rsplit('/', 1)[-1]] in allowed]
        items.extend(rows)
    held = set(c['heldout_subdomains'])
    leaked = [r for r in items if r['file_name'].startswith('peta/') and domain_of_basename[r['file_name'].rsplit('/', 1)[-1]] in held]
    # The all-source arm deliberately fits every camera; its held-out metrics are
    # then only a train diagnostic and must be declared as such.
    fits_heldout = c.get('fit_includes_heldout', False) is True and c['evaluation_role'] == 'train_diagnostic'
    if leaked and not fits_heldout:
        raise ValueError(f'{len(leaked)} held-out PETA items entered the fit')
    if c.get('expected_train_examples') is not None and len(items) != c['expected_train_examples']:
        raise ValueError(f"Training population {len(items)} != expected {c['expected_train_examples']}")
    if c.get('head_zoom'):
        items = head_zoom_items(items, c['head_zoom'])
    return items


HEAD_JOINTS = (0, 1, 2)   # nose, head_bottom, head_top


def head_box(keypoints, box, scale, fallback_fraction):
    """Square box around the head: centre = mean of the labelled head points, side = scale x head length
    (head_top-head_bottom distance; else scale x fallback_fraction x person-box height). None without head labels.
    The same function places the crop at test time from predicted head points."""
    kp = np.asarray(keypoints, dtype=np.float64).reshape(-1, 3)
    vis = [j for j in HEAD_JOINTS if kp[j, 2] > 0]
    if not vis:
        return None
    cx, cy = kp[vis, 0].mean(), kp[vis, 1].mean()
    if kp[1, 2] > 0 and kp[2, 2] > 0:
        length = float(np.hypot(*(kp[2, :2] - kp[1, :2])))
    else:
        length = 0.0
    length = max(length, fallback_fraction * float(box[3]) * 0.5)
    side = scale * length
    return [cx - side / 2, cy - side / 2, side, side]


def head_zoom_items(items, spec):
    """Head-zoom specialist: replace each person box by a square head box (keypoints unchanged; joints outside the crop
    get no target). Persons without labelled head points are dropped."""
    out = []
    for item in items:
        hb = head_box(item['keypoints'], item['box'], float(spec.get('scale', 2.6)), float(spec.get('fallback_fraction', 0.2)))
        if hb is not None:
            out.append({**item, 'box': hb, 'person_box': item['box']})
    return out


def teacher_projection_storage_definition(config, enabled=False):
    """Validate an execution opt-in without changing a saved training recipe."""
    if not enabled:
        return {'enabled': False, 'status': 'original_fp32_storage'}
    if config.get('precision') != 'bf16':
        raise ValueError('--teacher-projection-storage-bf16 requires explicit precision=bf16')
    if not config.get('distill') or not config['distill'].get('teachers'):
        raise ValueError('--teacher-projection-storage-bf16 requires nonempty distillation teachers')
    return {
        'enabled': True, 'status': 'requested_not_built', 'teachers': [],
        'contract': 'Only fully frozen eval RSNPose/ViTPose Conv/ConvTranspose/Linear weight and bias storage to BF16; norms, pos/cls parameters, and all buffers retain FP32/original dtype; CUDA BF16 autocast and teacher execution order unchanged',
        'gpu_parity_evidence': 'artifacts/runs/20261017T0348+0900_teacher_bf16_cuda_plan/batch1_production_teachers_result.json',
        'gpu_parity_evidence_sha256': '28a0e0c788ebc461972e0e130e2ab2bac96cf87e5bb4d18c37066a91426fa312',
        'gpu_parity_scope': 'Three production F teachers, 16 real crops, batch1: raw/normalized/fused maps exact; development teachers separately CPU-exact on the same crops',
        'full_batch_training_peak': 'unmeasured', 'full_training_step_parity': 'unmeasured',
        'teacher_first_adopted_by_this_flag': False, 'source_checkpoints_written': False}


def convert_reviewed_teacher_storage(teacher, allowed_classes):
    if type(teacher) not in allowed_classes:
        raise ValueError('BF16 teacher storage requires a reviewed exact teacher architecture')
    if any(b.is_floating_point() and b.dtype != torch.float32 for b in teacher.buffers()):
        raise ValueError('BF16 teacher storage expects original FP32 floating buffers')
    # Conversion alone must not consume global RNG, including the active CUDA device.
    cpu_rng, python_rng, numpy_rng = torch.get_rng_state(), random.getstate(), np.random.get_state()
    device = next(teacher.parameters()).device
    cuda_rng = torch.cuda.get_rng_state(device) if device.type == 'cuda' else None
    conversion = convert_frozen_affine_parameters_to_bf16(teacher)
    after = np.random.get_state()
    unchanged = (torch.equal(cpu_rng, torch.get_rng_state()) and python_rng == random.getstate()
                 and numpy_rng[0] == after[0] and np.array_equal(numpy_rng[1], after[1])
                 and numpy_rng[2:] == after[2:])
    if cuda_rng is not None:
        unchanged = unchanged and torch.equal(cuda_rng, torch.cuda.get_rng_state(device))
    if not unchanged:
        raise ValueError('Teacher storage conversion changed RNG; no retry')
    return {**conversion, 'conversion_rng_unchanged': True,
            'adoption_status': 'explicit_execution_opt_in_after_recorded_cpu_and_cuda_map_parity'}


def build_teachers(c, *, projection_storage_bf16=False, storage_records=None):
    """Optional fixed heatmap teachers (config 'distill'); each must respect the student's fit boundary."""
    teacher_projection_storage_definition(c, projection_storage_bf16)
    spec = c.get('distill')
    if not spec:
        return None
    held = set(c['heldout_subdomains'])
    teachers = []
    for t in spec['teachers']:
        tc = load_json(ROOT / t['config'])
        if c['evaluation_role'] == 'development':
            # A development student is only scored on unseen cameras if no teacher saw them either.
            if (tc.get('evaluation_role') != 'development' or tc.get('fit_includes_heldout') or set(tc['heldout_subdomains']) != held
                    or held & set(tc.get('peta_train_subdomains', []))):
                raise ValueError(f"teacher {t['config']} crosses the development fit boundary")
        if sha256_file(ROOT / t['weights']) != t['sha256']:
            raise ValueError(f"teacher weights changed: {t['weights']}")
        kind = tc.get('model_type', 'rsn')
        if kind not in ('rsn', 'vit'):
            raise ValueError(f'Unreviewed teacher architecture: {kind}')
        teacher = RSNPose() if kind == 'rsn' else ViTPose(arch=tc['vit_arch'], image_size=tc.get('vit_image_size', 256), drop_path=0.0,
                                                          quality_head=bool(tc.get('quality_head', False)),
                                                          upsample_layers=tc.get('vit_upsample_layers', 2),
                                                          patch_padding=tc.get('vit_patch_padding', 0), with_cls_token=tc.get('vit_with_cls_token', True))
        teacher.load_state_dict(torch.load(ROOT / t['weights'], map_location='cpu', weights_only=False)['model_state'], strict=True)
        teacher = teacher.cuda().eval().requires_grad_(False)
        if projection_storage_bf16:
            conversion = convert_reviewed_teacher_storage(teacher, (RSNPose, ViTPose))
            record = {'weights': t['weights'], 'source_weights_sha256': t['sha256'],
                      'config': t['config'], 'source_config_sha256': sha256_file(ROOT / t['config']),
                      'model_type': kind, 'fusion_weight': float(t['weight']), **conversion}
            if storage_records is not None:
                storage_records.append(record)
            print(json.dumps({'teacher_projection_storage_bf16': t['weights'],
                              'bytes_saved': conversion['bytes_saved'],
                              'source_weights_sha256': t['sha256']}), flush=True)
        teachers.append((teacher, float(t['weight'])))
    return teachers


def teacher_maps(teachers, inputs, size):
    """Single augmented-view fixed-teacher fusion in training space; not deployed TTA fusion."""
    fused, total = None, 0.0
    with torch.no_grad():
        for teacher, weight in teachers:
            maps = teacher(inputs).float()
            if maps.shape[-1] != size:
                maps = torch.nn.functional.interpolate(maps, size=(size, size), mode='bilinear', align_corners=True)
            peak = maps.amax(dim=(-2, -1), keepdim=True)
            maps = maps / torch.where(peak > 1e-8, peak, torch.ones_like(peak))
            fused = weight * maps if fused is None else fused + weight * maps
            total += weight
    return (fused / total).clamp_min(0.0)


def evaluation_cache_spec(config_path, config, epoch, weights_path):
    """Execution-only cache request; no change to the saved training recipe."""
    import evaluate_highcap_highres_inference_v15 as crop_module
    return {'project_root': ROOT, 'weights_path': weights_path, 'config_path': config_path,
        'candidate': config['candidate'], 'epoch': epoch, 'cameras': config['heldout_subdomains'],
        'source_paths': [ROOT / p for p in (
            'scripts/train_rsn_domain_continuation.py', 'scripts/evaluate_resolution_ensemble.py',
            'scripts/evaluate_multimodel_map_fusion.py', 'scripts/evaluate_robust_seed_geometry_ensemble_v7.py',
            'scripts/evaluate_seed_geometry_ensemble_v6.py', 'scripts/pose_square_adapter.py',
            'scripts/evaluate_highcap_highres_inference_v15.py', 'src/upar_pose/evaluation_map_cache.py',
            'src/upar_pose/geometry_tta_v4.py', 'src/upar_pose/flip_tta.py')],
        'data_paths': [ROOT / 'data/annotations/specialization/train_peta.json', ROOT / config['domain_mapping']],
        'execution_context': {'torch': str(torch.__version__), 'exact_crop': bool(crop_module.EXACT_CROP),
            'inference': 'FP32; no autocast', 'cudnn_allow_tf32': torch.backends.cudnn.allow_tf32,
            'matmul_allow_tf32': torch.backends.cuda.matmul.allow_tf32}}


def evaluate_heldout(model, peta, eval_items, domains, output, *, batch_size, workers=0, cache_spec=None):
    generator = torch.Generator().manual_seed(20261004)
    # Crops are deterministic per item, so loader workers only change speed, not predictions.
    loader = DataLoader(SquareInteractionDataset(eval_items, (256, 256)), batch_size=batch_size,
                        num_workers=workers, shuffle=False, pin_memory=True, generator=generator)
    rows, channels = [], torch.tensor(FLIP_CHANNELS, device='cuda')
    side = model_heatmap_side(model)   # RSN/ViT legacy geometry; explicit HRFormer internal_size/4.
    model.eval()
    started = time.monotonic()
    cache, cache_status = None, None
    if cache_spec is not None:
        try:
            if not Path(cache_spec['weights_path']).is_file():
                cache_status = {'status': 'SKIPPED_NO_IMMUTABLE_EPOCH_WEIGHTS',
                                'note': 'No guessed e0 or discarded-epoch model binding; six-view validation continues.'}
            else:
                cache = EvaluationMapCache(output.with_name(output.stem + '_cache_v4'), model=model,
                    expected_ids=[(r['image_id'], r['annotation_id']) for r in eval_items], grid_side=side,
                    **cache_spec)
        except Exception as error:
            cache_status = {'status': 'FAILED_OPTIONAL_CACHE', 'error': str(error), 'stage': 'create'}
    with torch.inference_mode():
        for inputs, geometries, image_ids, annotation_ids in loader:
            if cache is None:
                maps = aligned_model_heatmaps(model, inputs.cuda(), channels, (side, side))
            else:
                maps, view_means = aligned_model_heatmaps(model, inputs.cuda(), channels, (side, side), return_view_means=True)
            geometry = [tuple(map(float, r)) for r in geometries[:, PADDINGS.index(TARGET_PADDING), :].numpy()]
            coords, maxima = decode_heatmaps_refined(maps.cpu().numpy(), geometry, refinement='quadratic')
            rows.extend(prediction(coords[i], maxima[i], int(image_ids[i]), int(annotation_ids[i])) for i in range(len(image_ids)))
            if cache is not None:
                try:
                    cache.append(aligned_v4_from_flip_means(view_means), image_ids, annotation_ids, geometry)
                except Exception as error:
                    cache_status = {'status': 'FAILED_OPTIONAL_CACHE', 'error': str(error), 'stage': 'append'}
                    cache.abort()
                    cache = None
                finally:
                    del view_means
    torch.cuda.synchronize()
    check_ids(rows, {(r['image_id'], r['annotation_id']) for r in eval_items})
    write_json_atomic(output, rows)
    if cache is not None:
        try:
            cache_status = {'status': 'COMPLETE', **cache.finalize(prediction_path=output)}
        except Exception as error:
            cache_status = {'status': 'FAILED_OPTIONAL_CACHE', 'error': str(error), 'stage': 'finalize'}
            cache.abort()
    if cache_spec is not None:
        write_json_atomic(output.with_name(output.stem + '_cache_status.json'), cache_status)
        print(json.dumps({'evaluation_map_cache': cache_status}), flush=True)
    image_ids = {r['image_id'] for r in eval_items}
    gt = subset_ground_truth(peta, image_ids)
    report = evaluate_domain_structure(gt, load_json(output), {i: domains[i] for i in image_ids}, include_oracle=False)
    return {'pooled': report['pooled'], 'subset_mean': report['subset_mean']['zero_fill_mean'],
            'per_subset_AP': {k: v['AP'] for k, v in report['per_subset'].items()},
            'seconds': time.monotonic() - started, 'predictions': relative(output), 'sha256': sha256_file(output),
            **({'map_cache': cache_status} if cache_spec is not None else {})}


def evaluate_quality_scores(model, peta, eval_items, domains, base_predictions_path, output):
    """Re-score the six-view predictions with the averaged quality head (base coords unchanged)."""
    from upar_pose.geometry_tta_v4 import aligned_weighted_mean  # noqa: F401  (same geometry module family)
    generator = torch.Generator().manual_seed(20261004)
    loader = DataLoader(SquareInteractionDataset(eval_items, (256, 256)), batch_size=8, num_workers=0, shuffle=False, generator=generator)
    channels = torch.tensor(FLIP_CHANNELS, device='cuda')
    scored = [i for i in range(17) if i not in (3, 4)]
    qual = {}
    model.eval()
    with torch.inference_mode():
        for inputs, _, image_ids, annotation_ids in loader:
            x = inputs.cuda()
            flat = x.reshape(-1, *x.shape[2:])
            _, q1 = model.forward_with_quality(flat)
            _, q2 = model.forward_with_quality(torch.flip(flat, dims=(-1,)))
            q2 = q2.index_select(1, channels)
            q = torch.sigmoid(0.5 * (q1 + q2)).reshape(x.shape[0], x.shape[1], 17).mean(1)
            for n in range(x.shape[0]):
                qual[int(annotation_ids[n])] = q[n, scored].mean().item()
    base = load_json(base_predictions_path)
    variants = {}
    for name, fn in {'quality': lambda b, qq: qq, 'quality_x_peak': lambda b, qq: (b * qq) ** 0.5}.items():
        rows = [dict(r, score=max(1e-4, float(fn(r['score'], qual[int(r['source_annotation_id'])])))) for r in base]
        image_ids = {r['image_id'] for r in eval_items}
        gt = subset_ground_truth(peta, image_ids)
        rep = evaluate_domain_structure(gt, rows, {i: domains[i] for i in image_ids}, include_oracle=False)
        variants[name] = rep['subset_mean']['zero_fill_mean']['AP']
        write_json_atomic(output.with_name(output.stem + f'_{name}.json'), rows)
    return variants


def run(path, audit=False, resume=False, early_vram_reserve_gib=None, progressive_reserve_minutes=None,
        kd_teacher_first=False, event_path_override=None, event_id_override=None, allocator_cap_gib=None, eval_cache_v4=False,
        teacher_projection_storage_bf16=False):
    c = load_json(path)
    validate_hrformer_recipe(c)
    if c.get('model_type', 'rsn') not in ('rsn', 'vit', 'hrformer'):
        raise ValueError('Unknown student model_type')
    kd_supervision = distillation_supervision_definition(c.get('distill'))
    teacher_storage_execution = teacher_projection_storage_definition(c, teacher_projection_storage_bf16)
    allocator_execution = allocator_cap_execution(c, allocator_cap_gib)
    config_sha256 = sha256_file(path)
    out = ROOT / c['output_dir']
    out = out.with_name(out.name + '_audit') if audit else out
    event_target, completion_id = ((None, None) if audit else
                                  completion_event_target(c, ROOT, event_path_override, event_id_override))
    if kd_teacher_first and not c.get('distill'):
        raise ValueError('--kd-teacher-first requires a distillation recipe')
    # Check the immutable recipe before reserving VRAM or touching run outputs.
    if resume:
        if audit or not (out / 'last_checkpoint.pt').exists() or not (out / 'config.json').exists():
            raise ValueError('--resume needs a non-audit run directory with config.json and last_checkpoint.pt')
        validate_resume_config(c, load_json(out / 'config.json'))
    if torch.cuda.is_available():
        allocator_execution = allocator_cap_execution(c, allocator_cap_gib, torch.cuda.get_device_properties(0).total_memory)
    reserve = early_vram_reserve_gib or c.get('early_vram_reserve_gib')
    if reserve and torch.cuda.is_available():
        # Shared GPU: claim the slot before the slow checkpoint/data/model setup so a concurrent launcher cannot take it.
        # The freed block stays in this process's caching allocator and is reused by the model and optimizer.
        torch.cuda.set_per_process_memory_fraction(allocator_execution['fraction'])
        if progressive_reserve_minutes:
            # Operational: accumulate the claim in 1 GiB blocks as other jobs release memory (single large free blocks
            # are rare on the shared GPU), waiting up to the given minutes; the blocks are then freed into this cache.
            held, got, deadline = [], 0.0, time.time() + 60.0 * progressive_reserve_minutes
            while got < reserve:
                step = min(1.0, reserve - got)
                try:
                    held.append(torch.empty(int(step * 1024**3), dtype=torch.uint8, device='cuda'))
                    got += step
                except torch.OutOfMemoryError:
                    if time.time() > deadline:
                        raise
                    time.sleep(2.0)
            print(json.dumps({'progressive_reserve_gib': got, 'waited_seconds': round(time.time() - deadline + 60.0 * progressive_reserve_minutes, 1)}), flush=True)
            del held
        else:
            torch.empty(int(reserve * 1024**3), dtype=torch.uint8, device='cuda')
    if c.get('init_mode', 'checkpoint') not in ('coco', 'mae') and sha256_file(ROOT / c['init_checkpoint']) != c['init_checkpoint_sha256']:
        raise ValueError('Init checkpoint changed')
    out.mkdir(parents=True, exist_ok=resume)
    execution_path = out / ('execution_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ') + '.json')
    execution_record = {
        'config_sha256': config_sha256, 'trainer_sha256': sha256_file(Path(__file__)),
        'resume': resume, 'kd_teacher_first': kd_teacher_first,
        'allocator_cap_execution': allocator_execution,
        'eval_cache_v4': eval_cache_v4,
        'teacher_projection_storage': teacher_storage_execution,
        'distillation_supervision': kd_supervision,
        'completion_event_path': relative(event_target) if event_target else None, 'completion_event_id': completion_id,
        'teacher_first_rng_contract': 'frozen eval teachers, process RNG checked per batch' if kd_teacher_first else None,
        'teacher_first_gpu_peak_reduction': 'unmeasured', 'teacher_first_cuda_numerical_parity': 'unmeasured'}
    write_json_atomic(execution_path, execution_record)
    if not torch.cuda.is_available():
        raise RuntimeError('Real CUDA required')
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.cuda.set_per_process_memory_fraction(allocator_execution['fraction'])

    domain_map = load_peta_domain_map(ROOT / c['domain_mapping'])
    peta = load_domain('peta')
    domains = image_domains(peta, domain_map)
    held = set(c['heldout_subdomains'])
    eval_items = [r for r in dataset_items(peta, require_labels=False) if domains[r['image_id']] in held]
    items = build_population(c, domain_map)

    seed_everything(c['seed'])
    model_type = c.get('model_type', 'rsn')
    if model_type == 'hrformer':
        model = HRFormerPose(internal_size=c['hrformer_internal_size'],
                             grad_checkpoint=c.get('grad_checkpoint', False))
    else:
        model = RSNPose() if model_type == 'rsn' else ViTPose(arch=c['vit_arch'], image_size=c.get('vit_image_size', 256),
                                                         drop_path=c.get('drop_path', 0.3), quality_head=bool(c.get('quality_head', False)),
                                                         upsample_layers=c.get('vit_upsample_layers', 2),
                                                         patch_padding=c.get('vit_patch_padding', 0), with_cls_token=c.get('vit_with_cls_token', True))
    if model_type == 'vit':
        model.grad_checkpoint = bool(c.get('grad_checkpoint', False))
    if model_type == 'vit' and c.get('vit_image_size', 256) != 256 and c.get('quality_head'):
        raise ValueError('quality_head targets assume 64x64 heatmaps; disable it for internal-resolution ViTs')
    mode = c.get('init_mode', 'checkpoint')
    if model_type == 'vit':
        if mode not in ('mae', 'weights', 'checkpoint'):
            raise ValueError('ViT init_mode must be mae, weights, or a same-architecture checkpoint')
        if sha256_file(ROOT / c['init_checkpoint']) != c['init_checkpoint_sha256']:
            raise ValueError('ViT init checkpoint changed')
        if mode == 'mae':
            write_json_atomic(out / 'initialization.json', model.load_mae(ROOT / c['init_checkpoint']))
        elif mode == 'checkpoint':
            checkpoint = torch.load(ROOT / c['init_checkpoint'], map_location='cpu', weights_only=False)
            # No positional-grid resizing, fresh heads, or missing keys with optimizer continuation.
            model.load_state_dict(checkpoint['model_state'], strict=True)
            source_config_path = (ROOT / c['init_checkpoint']).parent / 'config.json'
            if not source_config_path.is_file():
                raise ValueError('ViT checkpoint continuation requires the source run config.json')
            source_config = load_json(source_config_path)
            validate_vit_continuation_source(c, source_config)
        else:
            # e.g. the COCO+PoseTrack18 keypoint pre-training stage; a fresh quality head may be added here.
            state = torch.load(ROOT / c['init_checkpoint'], map_location='cpu', weights_only=False)['model_state']
            state = resize_pos_embed(state, model.grid)   # no-op unless the internal resolution changes
            missing, unexpected = model.load_state_dict(state, strict=False)
            # A deeper decoder (vit_upsample_layers 3) adds fresh deconv/BN layers after the pre-trained ones.
            fresh_ok = lambda k: k.startswith('quality.') or (c.get('vit_upsample_layers', 2) > 2 and k.startswith('decoder.') and int(k.split('.')[1]) >= 6)
            if unexpected or any(not fresh_ok(k) for k in missing):
                raise ValueError(f'ViT weights init mismatch missing={missing} unexpected={unexpected}')
            write_json_atomic(out / 'initialization.json', {'source': c['init_checkpoint'], 'sha256': c['init_checkpoint_sha256'],
                                                            'fresh_parameters': sorted(missing)})
        model.cuda()
        optimizer = torch.optim.AdamW(layer_decay_param_groups(model, c['learning_rate'], c['weight_decay'], c['layer_decay']),
                                      lr=c['learning_rate'], betas=(0.9, 0.999))
        if mode == 'checkpoint':
            restore_adamw_moments(optimizer, checkpoint['optimizer_state'])
            init_epoch = int(checkpoint['epoch'])
            write_json_atomic(out / 'initialization.json', {
                'source': c['init_checkpoint'], 'sha256': c['init_checkpoint_sha256'], 'init_epoch': init_epoch,
                'source_config': relative(source_config_path), 'source_config_sha256': sha256_file(source_config_path),
                'restored': ['model_state', 'AdamW moments and step'],
                'new_revision': ['config LR schedule starts at epoch 1', 'config data/RNG streams; not an exact resume']})
            del checkpoint
        else:
            init_epoch = 0
    elif model_type == 'hrformer':
        if sha256_file(ROOT / c['init_checkpoint']) != c['init_checkpoint_sha256']:
            raise ValueError('HRFormer init checkpoint changed')
        source_config_path = None
        if mode == 'coco':
            initialization = model.initialize(ROOT / c['init_checkpoint'], reset_head=True)
            init_epoch = 0
        else:
            source_config_path = (ROOT / c['init_checkpoint']).parent / 'config.json'
            if not source_config_path.is_file():
                raise ValueError('HRFormer continuation requires the source run config.json')
            validate_hrformer_continuation_source(c, load_json(source_config_path))
            checkpoint = torch.load(ROOT / c['init_checkpoint'], map_location='cpu', weights_only=False)
            model.load_state_dict(checkpoint['model_state'], strict=True)
            init_epoch = int(checkpoint['epoch']) if mode == 'checkpoint' else 0
            initialization = {'source': c['init_checkpoint'], 'sha256': c['init_checkpoint_sha256'],
                'source_config': relative(source_config_path), 'source_config_sha256': sha256_file(source_config_path),
                'strict_state_load': True, 'init_epoch': init_epoch, 'reinitialized_channels': []}
        model.cuda()
        optimizer = torch.optim.AdamW(hrformer_optimizer_groups(model, c['weight_decay']),
                                      lr=c['learning_rate'], betas=(0.9, 0.999))
        if mode == 'checkpoint':
            restore_adamw_moments(optimizer, checkpoint['optimizer_state'])
            initialization['restored'] = ['model_state', 'AdamW moments and step']
        if mode != 'coco':
            del checkpoint
        initialization['training_loss_contract'] = 'peak-one MSE, ears3/4 masked, divide by masked GT-weight sum; same denominator as legacy weighted MSE on normal ear-zero dataset batches'
        write_json_atomic(out / 'initialization.json', initialization)
    elif mode == 'coco':
        initialization = model.initialize(ROOT / c['init_checkpoint'], expected_sha256=c['init_checkpoint_sha256'])
        write_json_atomic(out / 'initialization.json', initialization)
        model.cuda()
        optimizer = torch.optim.AdamW(model.parameters(), lr=c['learning_rate'], weight_decay=c['weight_decay'])
        init_epoch = 0
    else:
        checkpoint = torch.load(ROOT / c['init_checkpoint'], map_location='cpu', weights_only=False)
        model.load_state_dict(checkpoint['model_state'], strict=True)
        model.cuda()
        optimizer = torch.optim.AdamW(model.parameters(), lr=c['learning_rate'], weight_decay=c['weight_decay'])
        if mode == 'checkpoint':
            optimizer.load_state_dict(checkpoint['optimizer_state'])
        elif mode != 'weights':
            raise ValueError(f'Unknown init_mode {mode}')
        init_epoch = int(checkpoint['epoch']) if str(checkpoint.get('epoch', '')).isdigit() else 0
        del checkpoint
    resumed = None
    if resume:
        # Same config/output dir: continue after the last fully saved epoch (model + optimizer + history).
        checkpoint = torch.load(out / 'last_checkpoint.pt', map_location='cpu', weights_only=False)
        if checkpoint.get('config_sha256', config_sha256) != config_sha256:
            raise ValueError('Resume checkpoint config hash mismatch')
        if checkpoint['init_epoch'] != init_epoch:
            raise ValueError('Resume checkpoint init_epoch mismatch')
        model.load_state_dict(checkpoint['model_state'], strict=True)
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        resumed = {'epoch': int(checkpoint['epoch']), 'history': checkpoint['history']}
        del checkpoint
    for group in optimizer.param_groups:
        group['lr'] = c['learning_rate'] * group.get('lr_scale', 1.0)

    seed_everything(c['data_seed'])
    dataset = DomainFactorDataset(items, c, settings=c['factor_synthesis'], weights=np.ones(5), output=out)
    if getattr(dataset, 'strong', None) is not None:
        # Optional strong augmentation (pose_strong_augmentation.py): resolved defaults + seed on record.
        write_json_atomic(out / 'strong_augmentation_resolved.json', dataset.strong.describe())
    if c.get('camera_quality'):
        dataset = CameraQualityDataset(dataset, c['camera_quality'])
    if c.get('group_sampling_weights'):
        # Each group (market, pa100k, or one PETA camera) receives the declared
        # share of draws regardless of its size; sampling with replacement.
        def group_of(r):
            head = r['file_name'].split('/', 1)[0]
            if head == 'external':
                return 'coco' if r['file_name'].startswith('external/coco/') else 'posetrack18'
            return 'peta/' + domain_map[r['file_name'].rsplit('/', 1)[-1]] if head == 'peta' else head
        groups = [group_of(r) for r in items]
        counts = {g: groups.count(g) for g in set(groups)}
        shares = c['group_sampling_weights']
        missing = set(counts) - set(shares)
        if missing:
            raise KeyError(f'No sampling share for groups {sorted(missing)}')
        item_weights = torch.tensor([shares[g] / counts[g] for g in groups], dtype=torch.double)
        sampler = torch.utils.data.WeightedRandomSampler(item_weights, num_samples=c['samples_per_epoch'], replacement=True,
                                                         generator=torch.Generator().manual_seed(c['data_seed']))
    else:
        sampler = RandomSampler(dataset, replacement=False, num_samples=c['samples_per_epoch'],
                                generator=torch.Generator().manual_seed(c['data_seed']))
    def reseed_worker(worker_id):
        # Distinct augmentation/synthesis streams per worker (workers otherwise copy one RNG state).
        import random as _random
        info = torch.utils.data.get_worker_info()
        # A resumed run offsets the streams so it does not replay epoch 1's augmentation draws.
        seed = (c['data_seed'] * 1000 + worker_id + 1 + 104729 * (resumed['epoch'] if resumed else 0)) % (2**32)
        _random.seed(seed)
        np.random.seed(seed)
        base = info.dataset.base if hasattr(info.dataset, 'base') else info.dataset
        base.factor_rng = np.random.default_rng(seed + 11)
        base.synthetic_rng.base = np.random.default_rng(seed + 13)
        base.synthetic_rng.factor = base.factor_rng
        if getattr(base, 'strong', None) is not None:
            # Strong stream: SeedSequence([config seed, this worker seed]) -> same resume offset as above.
            base.strong.reseed(seed)
        if hasattr(info.dataset, 'rng'):
            info.dataset.rng = np.random.default_rng(seed + 17)
    workers = int(c.get('workers', 0))
    loader = DataLoader(dataset, batch_size=c['batch_size'], sampler=sampler, num_workers=workers, pin_memory=True, drop_last=True,
                        worker_init_fn=reseed_worker if workers else None, persistent_workers=bool(workers))
    population = {}
    for r in items:
        key = r['file_name'].split('/', 1)[0]
        if key == 'external':
            key = 'coco' if r['file_name'].startswith('external/coco/') else 'posetrack18'
        elif key == 'peta':
            key = 'peta/' + domain_map[r['file_name'].rsplit('/', 1)[-1]]
        population[key] = population.get(key, 0) + 1
    if resumed:
        # Advance the sampler generator past the completed epochs so the index stream continues.
        for _ in range(resumed['epoch']):
            for _ in sampler:
                pass
        write_json_atomic(out / f'resume_after_epoch_{resumed["epoch"]:03d}.json', {
            'resumed_after_epoch': resumed['epoch'], 'config_sha256': sha256_file(path), 'checkpoint': relative(out / 'last_checkpoint.pt'),
            'checkpoint_sha256': sha256_file(out / 'last_checkpoint.pt'), 'observed_at': datetime.now(timezone.utc).isoformat(),
            'restored': ['model_state', 'optimizer_state', 'training_history', 'sampler_index_stream'],
            'not_restored': ['worker augmentation RNG (offset instead)', 'global torch/numpy RNG']})
    else:
        write_json_atomic(out / 'config.json', {**c, 'config_sha256': sha256_file(path), 'train_examples': len(items),
            'train_population': population, 'eval_persons': len(eval_items), 'init_epoch': init_epoch, 'audit_only': audit})
    print(json.dumps({'train_examples': len(items), 'population': population, 'eval_persons': len(eval_items)}), flush=True)

    teachers = build_teachers(c, projection_storage_bf16=teacher_projection_storage_bf16,
                              storage_records=teacher_storage_execution.get('teachers'))
    if teacher_projection_storage_bf16:
        teacher_storage_execution.update(status='converted_frozen_teachers',
            total_tensor_storage_bytes_saved=sum(t['bytes_saved'] for t in teacher_storage_execution['teachers']),
            helper_sha256=sha256_file(ROOT / 'src/upar_pose/teacher_storage.py'))
        write_json_atomic(execution_path, execution_record)
    if kd_teacher_first:
        validate_teacher_first_models(teachers, (RSNPose, ViTPose))
    if teachers is not None:
        # Distillation on the scored joints only (the ear slots carry no labels); the GT loss is kept unchanged.
        kd_mask = torch.ones(17, device='cuda')
        kd_mask[[3, 4]] = 0.0
    probe = next(model.parameters()).detach().flatten()[:256].clone()
    history, validations, started = [], [], time.monotonic()
    if resumed:
        history = resumed['history']
        if (out / 'validation_history.json').exists():
            validations = [v for v in load_json(out / 'validation_history.json') if v['epoch'] <= resumed['epoch']]
    if not audit and not resumed and c.get('evaluate_initial', False):
        val = evaluate_heldout(model, peta, eval_items, domains, out / 'epoch_000_heldout_predictions.json', batch_size=c['eval_batch_size'], workers=c.get('eval_workers', 0),
            cache_spec=evaluation_cache_spec(path, c, 0, out / 'epoch_000_weights.pt') if eval_cache_v4 else None)
        val['epoch'] = 0
        if c.get('save_best_weights', True):
            improves_validation(float(val['subset_mean']['AP']), [])
            initial_weights = out / 'epoch_000_weights.pt'
            save_epoch_weights(initial_weights, {'epoch': 0, 'init_epoch': init_epoch, 'model_state': model.state_dict(),
                                              'config_sha256': config_sha256})
            val.update(weights=relative(initial_weights), weights_sha256=sha256_file(initial_weights))
        validations.append(val)
        write_json_atomic(out / 'validation_history.json', validations)
        print(json.dumps({k: val[k] for k in ('epoch', 'subset_mean', 'seconds')}), flush=True)
    for epoch in range((resumed['epoch'] if resumed else 0) + 1, (1 if audit else c['max_epochs']) + 1):
        weights_path = out / f'epoch_{epoch:03d}_weights.pt'
        if weights_path.exists():
            raise FileExistsError(f'Epoch weights already exist: {weights_path}; preserve them and use a new revision')
        lr = scheduled_learning_rate(c, epoch, batches_per_epoch=len(loader))
        if c.get('warmup_epochs') or c.get('lr_schedule'):
            for group in optimizer.param_groups:
                group['lr'] = lr * group.get('lr_scale', 1.0)
        model.train()
        seen, total, tick = 0, 0.0, time.monotonic()
        for bi, batch in enumerate(loader):
            if c.get('warmup_mode') == 'linear_steps' and c.get('warmup_epochs'):
                lr = scheduled_learning_rate(c, epoch, bi, len(loader))
                for group in optimizer.param_groups:
                    group['lr'] = lr * group.get('lr_scale', 1.0)
            inputs, targets, weights, coords, visibility = [x.cuda(non_blocking=True) for x in batch[:5]]
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                guide = None
                if kd_teacher_first:
                    output_side = model_heatmap_side(model)
                    guide = checked_teacher_maps(teachers, inputs, output_side)
                output, stages, aux = model.forward_training(inputs)
                if guide is not None and guide.shape != output.shape:
                    raise ValueError('Teacher-first output geometry does not match the student')
                kd_targets = (gt_supported_training_targets(output, targets, coords, c)
                              if kd_supervision is not None and kd_supervision['mode'] == 'gt_supported' else None)
                if model_type == 'rsn':
                    loss, _ = rsn_loss(model, output, stages, aux, targets, weights, coords, visibility)
                    if teachers is not None:
                        # Final-stage map only; lambda in the units of rsn_loss (its final stage carries 1/5.25 of it).
                        if guide is None:
                            guide = teacher_maps(teachers, inputs, output.shape[-1])
                        kd = distillation_heatmap_loss(output, guide, weights, kd_mask, kd_supervision['mode'], gt_targets=kd_targets)
                        loss = loss + float(c['distill']['lambda']) * kd
                elif model_type == 'hrformer':
                    loss, _ = hrformer_training_loss(model, output, stages, aux, targets, weights, coords, visibility, c)
                    if teachers is not None:
                        if guide is None:
                            guide = teacher_maps(teachers, inputs, output.shape[-1])
                        kd = distillation_heatmap_loss(output, guide, weights, kd_mask, kd_supervision['mode'], gt_targets=kd_targets)
                        loss = loss + float(c['distill']['lambda']) * kd
                else:
                    if output.shape[-1] != targets.shape[-1] or c.get('heatmap_sigma_joint_scale') is not None:
                        # Internal-resolution ViT: same joints/weights, targets redrawn on the finer grid
                        # with the sigma scaled so the Gaussian keeps its width in input pixels;
                        # optional per-joint multipliers (e.g. sharper head targets).
                        size = output.shape[-1]
                        sigma = c.get('heatmap_sigma_out', c['heatmap_sigma'] * (size - 1) / 63.0)
                        if c.get('heatmap_sigma_joint_scale') is not None:
                            sigma = [sigma * float(f) for f in c['heatmap_sigma_joint_scale']]
                        targets = gaussian_targets(coords, size, sigma)
                    # Optional OKS-aware terms (joint_loss_weights / coord_loss, src/upar_pose/oks_loss.py);
                    # with both keys absent this is vit_pose_loss(output, targets, weights) itself.
                    loss, _ = oks_aware_vit_pose_loss(output, targets, weights, coords, c)
                    if teachers is not None:
                        if guide is None:
                            guide = teacher_maps(teachers, inputs, output.shape[-1])
                        kd = distillation_heatmap_loss(output, guide, weights, kd_mask, kd_supervision['mode'], gt_targets=kd_targets)
                        loss = loss + float(c['distill']['lambda']) * kd
                    if aux is not None:
                        target_q, mask = quality_targets(output, coords, weights)
                        bce = torch.nn.functional.binary_cross_entropy_with_logits(aux.float(), target_q, reduction='none')
                        loss = loss + c.get('quality_weight', 0.01) * (bce * mask).sum() / mask.sum().clamp_min(1)
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite RSN loss; no retry')
            loss.backward()
            grads = [torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None]
            if not grads or not torch.stack(grads).all():
                raise FloatingPointError('Nonfinite RSN gradient')
            if c.get('grad_clip'):
                torch.nn.utils.clip_grad_norm_(model.parameters(), c['grad_clip'])
            optimizer.step()
            seen += len(inputs)
            total += float(loss.detach()) * len(inputs)
            if bi % 100 == 0:
                print(json.dumps({'epoch': epoch, 'batch': bi + 1, 'loss': float(loss.detach())}), flush=True)
            if audit and bi == 2:
                break
        torch.cuda.synchronize()
        history.append({'epoch': epoch, 'examples': seen, 'train_loss': total / max(seen, 1), 'seconds': time.monotonic() - tick,
                        'learning_rate': optimizer.param_groups[0]['lr'], 'observed_at': datetime.now(timezone.utc).isoformat()})
        write_json_atomic(out / 'training_history.json', history)
        if audit:
            update = float((next(model.parameters()).detach().flatten()[:256] - probe).abs().max())
            if update <= 0:
                raise ValueError('No real parameter update')
            write_json_atomic(out / 'audit.json', {'audit_only': True, 'batches': 3, 'examples': seen, 'parameter_max_update': update,
                                                    'peak_cuda_reserved_bytes': torch.cuda.max_memory_reserved(), 'finite': True})
            print(json.dumps({'audit': 'ok', 'update': update}), flush=True)
            return
        # Disk economy for large models: per-epoch weights only from a declared epoch (soup window) and the final one.
        keep_weights = epoch >= c.get('save_weights_from_epoch', 1) or epoch == c['max_epochs']
        if keep_weights:
            save_epoch_weights(weights_path, {'epoch': epoch, 'init_epoch': init_epoch, 'model_state': model.state_dict(),
                                           'config_sha256': config_sha256})
        if epoch % c.get('evaluate_every', 1) and epoch != c['max_epochs']:
            save_checkpoint(out / 'last_checkpoint.pt', {'epoch': epoch, 'init_epoch': init_epoch, 'model_state': model.state_dict(),
                                                         'optimizer_state': optimizer.state_dict(), 'history': history,
                                                         'config_sha256': config_sha256})
            continue
        val = evaluate_heldout(model, peta, eval_items, domains, out / f'epoch_{epoch:03d}_heldout_predictions.json', batch_size=c['eval_batch_size'], workers=c.get('eval_workers', 0),
            cache_spec=evaluation_cache_spec(path, c, epoch, weights_path) if eval_cache_v4 else None)
        if getattr(model, 'quality', None) is not None:
            val['quality_score_subset_mean'] = evaluate_quality_scores(model, peta, eval_items, domains, out / f'epoch_{epoch:03d}_heldout_predictions.json',
                                                                        out / f'epoch_{epoch:03d}_heldout_rescored.json')
        # Legacy runs may have lost their highest-scoring epoch. Preserve the best
        # *available* model too, without rewriting the old observed score as recovered.
        is_best = improves_validation(float(val['subset_mean']['AP']), preserved_validations(validations, ROOT))
        if is_best and c.get('save_best_weights', True) and not keep_weights:
            save_epoch_weights(weights_path, {'epoch': epoch, 'init_epoch': init_epoch, 'model_state': model.state_dict(),
                                           'config_sha256': config_sha256})
            keep_weights = True
        val.update(epoch=epoch, weights=relative(weights_path) if keep_weights else None,
                   weights_sha256=sha256_file(weights_path) if keep_weights else None)
        validations.append(val)
        write_json_atomic(out / 'validation_history.json', validations)
        print(json.dumps({k: val[k] for k in ('epoch', 'subset_mean', 'seconds')}), flush=True)
        save_checkpoint(out / 'last_checkpoint.pt', {'epoch': epoch, 'init_epoch': init_epoch, 'model_state': model.state_dict(),
                                                     'optimizer_state': optimizer.state_dict(), 'history': history,
                                                     'config_sha256': config_sha256})
    selection = validation_selection(validations, ROOT)
    # A last checkpoint without matching scored weights is a recovery asset, not
    # the missing historical best. Do not attach the latter's AP to this fallback.
    fallback = {'path': relative(out / 'last_checkpoint.pt'), 'epoch': history[-1]['epoch'] if history else None,
                'purpose': 'resume_only_not_a_recovered_validation_best', 'AP': None}
    write_json_atomic(out / 'result.json', {'status': 'epoch_cap_reached_not_convergence_proof', 'epochs': len(history),
                                           'elapsed_seconds': time.monotonic() - started, 'validations': len(validations),
                                           'evaluation_role': c['evaluation_role'], 'resumed_after_epoch': resumed['epoch'] if resumed else None,
                                           'selection': selection, 'resume_checkpoint': fallback})
    if event_target:
        best = max(validations, key=lambda v: v['subset_mean']['AP']) if validations else {'subset_mean': {'AP': float('nan')}, 'epoch': None}
        write_json_atomic(event_target, {
            'id': completion_id, 'candidate': c['candidate'], 'kind': 'diagnostic', 'candidate_status': 'evaluating',
            'summary': f"Domain continuation ended; held-out subset-mean AP best {best['subset_mean']['AP']:.6f} at epoch {best['epoch']} "
                       f"({c['evaluation_role']}); observed-best weights available={selection['observed_best_has_weights']}. "
                       'See result.selection.best_available_weights for the preserved candidate.',
            'issue': 'Paired C/P comparison and Public-relevance assessment still required; not a Public forecast.',
            'next_action': 'Compare arm validation histories on identical held-out cameras; decide all-source packaging.',
            'scope': {'evaluation_role': c['evaluation_role']}, 'visual': False,
            'evidence': [relative(out / 'result.json'), relative(out / 'validation_history.json')],
            'artifacts': [relative(out / 'result.json'), relative(out / 'last_checkpoint.pt')]})


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--audit', action='store_true')
    p.add_argument('--resume', action='store_true', help='continue the same run from its last_checkpoint.pt')
    p.add_argument('--kd-teacher-first', action='store_true', help='compute frozen deterministic teachers before the student (explicit execution optimization)')
    p.add_argument('--teacher-projection-storage-bf16', action='store_true',
                   help='explicit frozen RSN/ViT Conv/Linear BF16 storage under BF16 autocast; preserve norms/buffers and execution order')
    p.add_argument('--event-path', help='new completion event path for a new scheduler job, without changing the training recipe')
    p.add_argument('--event-id', help='completion event ID override, paired with the scheduler job')
    p.add_argument('--allocator-cap-gib', type=positive_finite_gib, default=None,
                   help='execution-only PyTorch allocator ceiling; default keeps config allocator_cap_gib unchanged')
    p.add_argument('--eval-cache-v4', action='store_true',
                   help='save verified native-grid v4 member maps during existing six-view validation (execution-only; no extra forward)')
    p.add_argument('--early-vram-reserve-gib', type=float, help='claim this much GPU memory at start (operational; overrides the config key)')
    p.add_argument('--progressive-reserve-minutes', type=float, help='accumulate the start-up claim in 1 GiB blocks for up to this many minutes (operational)')
    a = p.parse_args()
    run(a.config, a.audit, a.resume, a.early_vram_reserve_gib, a.progressive_reserve_minutes, a.kd_teacher_first,
        a.event_path, a.event_id, a.allocator_cap_gib, a.eval_cache_v4, a.teacher_projection_storage_bf16)
