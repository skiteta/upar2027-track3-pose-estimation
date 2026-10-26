"""Optional strong fine-tune augmentation for ResearchDataset (config key ``strong_augmentation``).

Absent key (or null/false): this module is not even imported and ResearchDataset is
bit-identical to the pre-patch code (same RNG draws, same samples).

Truncation (test-like partial bodies; independent of half-body): with p=truncation_prob
  the ORIGINAL image is cut BEFORE any cropping -- usually the bottom is removed (keep
  the top U[truncation_keep] fraction of the person box), with p=truncation_side_prob
  instead the left or right side (keep U[truncation_side_keep] of the box width).  The
  person box is clipped to the cut image, joints outside the cut become unlabelled
  (visibility 0 -> heatmap weight 0), and the normal crop pipeline then runs on the new
  box, so the area beyond the cut edge is black exactly as inference pads out-of-image
  regions (Image.crop / EXACT_CROP canvas).  Skipped (draws still consumed) when fewer
  than truncation_min_labelled labelled joints would remain.  Unlike MMPose half-body,
  no image content beyond the cut survives.

Geometry (applied while cropping from the ORIGINAL image, one bilinear resampling):
  * RandomHalfBody (MMPose 1.x semantics): with p=half_body_prob, if >= min_total
    labelled joints, re-centre the square window on the mean of the upper body
    (nose, head_bottom, head_top, shoulders, elbows, wrists; slots 0,1,2,5-10; ears
    3/4 ignored) or lower body (hips, knees, ankles; slots 11-16) joints, side =
    padding * max(extent); upper chosen with p=upper_prioritized_prob when both
    have enough joints.
  * RandomBBoxTransform (MMPose 1.x semantics, truncated-normal on [-1, 1]):
    shift = t*shift_frac*side (p=shift_prob), scale = mid + t*half_width
    (p=scale_prob, from scale_range), rotation = t*rotate_deg (p=rotate_prob).
  The window is a rotated square (centre C, side S, angle theta) composed ON TOP of
  the unchanged base square jitter; base draws are still consumed in the same order.
  Image (PIL corner-continuous coordinates): c_in = C + (S/256) R(theta) (c_out - 128)
  Labels (existing convention, see native.transform_keypoints_highres):
                                              u    = 127.5 + (255/S) R(-theta) (x - C)
  At zero augmentation both reduce exactly to the existing crop/resize mapping
  (image c_in = left + c_out*W/256, labels (x-left)*255/W), so the label/pixel
  convention the evaluation decoder relies on is unchanged.  Downscaled windows
  (S > 256 source px) are pre-resized with PIL's antialiased resize, as the
  existing crop().resize() path does.  Content outside the source image is black,
  like the existing crop padding.

Photometric (after crop, flip, base brightness/contrast and person paste):
  colour (MMPose PhotometricDistortion: brightness, contrast before/after HSV,
  saturation, hue, channel swap; each p=color_jitter.prob) -> one CoarseDropout
  hole (centre uniform over the transformed person box, size U[hole_min_frac,
  hole_max_frac] * 256 per side, synthetic fill; covered labelled joints -> visibility
  1, heatmap weight unchanged, like the person paste) -> Gaussian blur / median blur
  -> Gaussian noise.

RNG: one private numpy Generator per dataset, seeded from the config
(``seed`` or data_seed + 52711).  Every sample consumes exactly DRAWS_PER_SAMPLE
uniforms + one 63-bit integer regardless of which branches fire, so changing one
probability knob does not shift the other knobs' draws of later samples.  Noise
fields and hole textures use a per-sample local Generator from that integer.
"""
from __future__ import annotations

import math
from statistics import NormalDist

import numpy as np
from PIL import Image, ImageFilter

from upar_pose.scratch_cnn import FLIP_PAIRS

SCHEMA = 'strong_augmentation_v1'
SEED_OFFSET = 52711
DRAWS_PER_SAMPLE = 40
UPPER_BODY = (0, 1, 2, 5, 6, 7, 8, 9, 10)
LOWER_BODY = (11, 12, 13, 14, 15, 16)
PERMUTATIONS = ((0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0))

DEFAULTS = {
    # RandomBBoxTransform (MMPose 1.x defaults: shift 0.16 @ p0.3, scale (0.5,1.5) @ p1.0,
    # rotate 80 @ p0.6).  Narrower here: upright pedestrians, axis-aligned test crops.
    'rotate_prob': 0.6, 'rotate_deg': 40.0,
    'scale_prob': 1.0, 'scale_range': [0.65, 1.35],
    'shift_prob': 0.3, 'shift_frac': 0.16,
    # RandomHalfBody (MMPose 1.x defaults).  min_side_frac is a local guard against
    # degenerate (near-coincident) joint sets; MMPose has none.
    'half_body_prob': 0.3, 'half_body_min_total': 9, 'half_body_min_upper': 2,
    'half_body_min_lower': 3, 'half_body_padding': 1.5, 'half_body_upper_prob': 0.7,
    'half_body_min_side_frac': 0.3,
    # Organiser UPAR-Pose recipe (Cormier dissertation §6.2): MMPose photometric
    # p0.5 each, Gaussian blur p0.1, median blur p0.1, CoarseDropout 1 hole <=40%.
    'color_jitter': {'prob': 0.5, 'brightness_delta': 32.0, 'contrast_range': [0.5, 1.5],
                     'saturation_range': [0.5, 1.5], 'hue_delta': 18.0, 'channel_swap_prob': 0.5},
    'hole_prob': 0.5, 'hole_min_frac': 0.1, 'hole_max_frac': 0.4,
    'blur_prob': 0.1, 'blur_sigma': [0.5, 1.5],
    'median_blur_prob': 0.1, 'median_sizes': [3, 5, 7],
    'noise_prob': 0.1, 'noise_sigma': [3.0, 10.0],
    # Test-like truncated persons (Public analysis 20261012T0015_size_domain_gap: partial
    # bodies with the box ~ the whole truncated image; <1 % of our training persons).
    'truncation_prob': 0.15, 'truncation_keep': [0.45, 0.85],
    'truncation_side_prob': 0.2, 'truncation_side_keep': [0.55, 0.85], 'truncation_min_labelled': 4,
}
OPTIONAL = {'seed', 'note'}

_NORMAL = NormalDist()
_LOW = _NORMAL.cdf(-1.0)
_SPAN = _NORMAL.cdf(1.0) - _LOW


def truncnorm(u):
    """Standard normal truncated to [-1, 1] from one uniform (inverse CDF; fixed draw count)."""
    return _NORMAL.inv_cdf(_LOW + float(u) * _SPAN)


def rotation(degrees):
    t = math.radians(degrees)
    c, s = math.cos(t), math.sin(t)
    return np.array([[c, -s], [s, c]])


def resolve_config(block, data_seed):
    if not isinstance(block, dict):
        raise ValueError('strong_augmentation must be an object (or absent/null to disable)')
    unknown = set(block) - set(DEFAULTS) - OPTIONAL
    if unknown:
        raise KeyError(f'Unknown strong_augmentation keys {sorted(unknown)}')
    colour = block.get('color_jitter', {})
    if colour is None:
        colour = {**DEFAULTS['color_jitter'], 'prob': 0.0, 'channel_swap_prob': 0.0}
    unknown = set(colour) - set(DEFAULTS['color_jitter'])
    if unknown:
        raise KeyError(f'Unknown color_jitter keys {sorted(unknown)}')
    c = {**DEFAULTS, **block, 'color_jitter': {**DEFAULTS['color_jitter'], **colour}}
    c['seed'] = int(block.get('seed', int(data_seed) + SEED_OFFSET))
    probs = [k for k in c if k.endswith('_prob')] + ['half_body_upper_prob']
    for k in probs:
        if not 0 <= float(c[k]) <= 1:
            raise ValueError(f'{k} must lie in [0, 1]')
    for k in ('prob', 'channel_swap_prob'):
        if not 0 <= float(c['color_jitter'][k]) <= 1:
            raise ValueError(f'color_jitter.{k} must lie in [0, 1]')
    lo, hi = map(float, c['scale_range'])
    if not 0 < lo <= hi:
        raise ValueError('scale_range must be 0 < low <= high')
    if not 0 <= c['hole_min_frac'] <= c['hole_max_frac'] <= 1:
        raise ValueError('need 0 <= hole_min_frac <= hole_max_frac <= 1')
    if any(int(k) % 2 == 0 or int(k) < 3 for k in c['median_sizes']):
        raise ValueError('median_sizes must be odd and >= 3')
    for k in ('truncation_keep', 'truncation_side_keep'):
        if not 0 < float(c[k][0]) <= float(c[k][1]) <= 1:
            raise ValueError(f'{k} must be 0 < low <= high <= 1')
    if c['rotate_deg'] < 0 or c['shift_frac'] < 0 or c['half_body_padding'] <= 0:
        raise ValueError('rotate_deg/shift_frac must be >= 0 and half_body_padding > 0')
    return c


class StrongAugmentation:
    """Draws, applies and checkpoints the optional strong augmentation of one dataset."""

    def __init__(self, block, data_seed):
        self.config = resolve_config(block, data_seed)
        self.seed = self.config['seed']
        self.rng = np.random.default_rng(self.seed)
        self.reseeded_with = None

    # ---- RNG stream ---------------------------------------------------------------
    def reseed(self, worker_seed):
        """Per-DataLoader-worker stream: SeedSequence([config seed, worker seed])."""
        self.rng = np.random.default_rng([self.seed, int(worker_seed) % 2**32])
        self.reseeded_with = int(worker_seed) % 2**32

    def _check_worker(self):
        if self.reseeded_with is not None:
            return
        import torch
        info = torch.utils.data.get_worker_info()
        if info is not None:
            # Fallback for trainers whose worker_init_fn does not call reseed(): without it
            # every worker would replay one identical stream.  Deterministic given torch's seed.
            self.reseed(info.seed)

    def state_dict(self):
        return {'schema': SCHEMA, 'seed': self.seed, 'reseeded_with': self.reseeded_with,
                'bit_generator': self.rng.bit_generator.state}

    def load_state_dict(self, state):
        if state.get('schema') != SCHEMA or state.get('seed') != self.seed:
            raise ValueError('strong augmentation state from another schema/seed')
        self.reseeded_with = state['reseeded_with']
        if self.reseeded_with is not None:
            self.rng = np.random.default_rng([self.seed, self.reseeded_with])
        self.rng.bit_generator.state = state['bit_generator']

    def describe(self):
        return {'schema': SCHEMA, 'resolved_config': self.config, 'draws_per_sample': DRAWS_PER_SAMPLE + 1,
                'upper_body_slots': UPPER_BODY, 'lower_body_slots': LOWER_BODY,
                'label_convention': 'u = 127.5 + 255/S * R(-theta) (x - C); heatmap = u * 63/255 (unchanged)'}

    # ---- per-sample draws ----------------------------------------------------------------
    def draw(self):
        """All of this sample's strong draws (fixed count), before anything else is decided."""
        self._check_worker()
        return {'u': self.rng.random(DRAWS_PER_SAMPLE), 'local_seed': int(self.rng.integers(0, 2**63 - 1)),
                'truncation': None}

    # ---- truncation (on the original image, before the crop window exists) -------------
    def truncate(self, image, box, keypoints, plan):
        """Return (image, box, keypoints), cut like a test-time truncated person, or unchanged."""
        c, u = self.config, plan['u']
        if not u[32] < c['truncation_prob']:
            return image, box, keypoints
        x, y, w, h = map(float, box[:4])
        if u[34] < c['truncation_side_prob']:
            lo, hi = map(float, c['truncation_side_keep'])
            keep = lo + (hi - lo) * u[35]
            if u[36] < 0.5:
                mode, rect = 'keep_left', (0, 0, int(round(x + keep * w)), image.height)
            else:
                mode, rect = 'keep_right', (int(round(x + (1 - keep) * w)), 0, image.width, image.height)
        else:
            lo, hi = map(float, c['truncation_keep'])
            keep = lo + (hi - lo) * u[33]
            mode, rect = 'keep_top', (0, 0, image.width, int(round(y + keep * h)))
        x0, y0, x1, y1 = max(0, rect[0]), max(0, rect[1]), min(image.width, rect[2]), min(image.height, rect[3])
        new_box = [max(x, x0), max(y, y0), min(x + w, x1), min(y + h, y1)]
        new_box = [new_box[0], new_box[1], new_box[2] - new_box[0], new_box[3] - new_box[1]]
        points = np.asarray(keypoints, dtype=np.float32).copy()
        inside = (points[:, 0] >= x0) & (points[:, 0] < x1) & (points[:, 1] >= y0) & (points[:, 1] < y1)
        remaining = int(((points[:, 2] > 0) & inside).sum())
        if x1 - x0 < 4 or y1 - y0 < 4 or min(new_box[2:]) < 4 or remaining < c['truncation_min_labelled']:
            plan['truncation'] = {'mode': mode, 'skipped': True, 'remaining_labelled': remaining}
            return image, box, keypoints
        points[~inside, 2] = 0          # cut away: unlabelled, as in a truncated test image
        points[:, 0] -= x0
        points[:, 1] -= y0
        new_box[0] -= x0
        new_box[1] -= y0
        plan['truncation'] = {'mode': mode, 'skipped': False, 'keep': keep, 'cut_rect_original': [x0, y0, x1, y1],
                              'box': new_box, 'remaining_labelled': remaining}
        return image.crop((x0, y0, x1, y1)), new_box, points

    # ---- geometry -----------------------------------------------------------------
    def sample(self, plan, keypoints, geometry):
        """Fill the geometry of ``plan``.  ``geometry`` is the base (already jittered) square window."""
        c = self.config
        u = plan['u']
        left, top, width, height = map(float, geometry)
        if abs(width - height) > 1e-6 * max(width, 1.0):
            raise ValueError('Square base window required')
        centre = np.array([left + width / 2, top + height / 2])
        side = width
        half_body = None
        points = np.asarray(keypoints, dtype=np.float64)
        labelled = points[:, 2] > 0
        if labelled.sum() >= c['half_body_min_total'] and u[0] < c['half_body_prob']:
            upper = [i for i in UPPER_BODY if labelled[i]]
            lower = [i for i in LOWER_BODY if labelled[i]]
            ok_upper, ok_lower = len(upper) >= c['half_body_min_upper'], len(lower) >= c['half_body_min_lower']
            chosen = None
            if ok_upper and ok_lower:
                chosen = ('upper', upper) if u[1] < c['half_body_upper_prob'] else ('lower', lower)
            elif ok_upper:
                chosen = ('upper', upper)
            elif ok_lower:
                chosen = ('lower', lower)
            if chosen is not None:
                selected = points[chosen[1], :2]
                extent = float(np.ptp(selected, axis=0).max()) * c['half_body_padding']
                centre = selected.mean(axis=0)
                side = max(extent, c['half_body_min_side_frac'] * width)
                half_body = chosen[0]
        shift = np.zeros(2)
        if u[2] < c['shift_prob']:
            shift = np.array([truncnorm(u[3]), truncnorm(u[4])]) * c['shift_frac'] * side
        scale = 1.0
        if u[5] < c['scale_prob']:
            lo, hi = map(float, c['scale_range'])
            scale = (lo + hi) / 2 + truncnorm(u[6]) * (hi - lo) / 2
        angle = truncnorm(u[8]) * c['rotate_deg'] if u[7] < c['rotate_prob'] else 0.0
        centre = centre + shift
        size = side * scale
        rot = rotation(angle)
        plan.update(centre=centre, side=size, angle=angle, scale=scale, shift=shift, half_body=half_body,
                    # source -> label coordinates (before flip); pixels use the matching corner map
                    label_linear=(255.0 / size) * rot.T, image_linear=(size / 256.0) * rot)
        return plan

    def warp(self, image, plan):
        """Crop the rotated/scaled/shifted window from the ORIGINAL image into 256x256."""
        linear, centre = plan['image_linear'], plan['centre']
        corners = np.array([[0, 0], [256, 0], [0, 256], [256, 256]], dtype=float) - 128.0
        mapped = centre + corners @ linear.T
        factor = plan['side'] / 256.0
        margin = int(math.ceil(2 * max(1.0, factor))) + 2
        x0 = max(0, int(math.floor(mapped[:, 0].min())) - margin)
        y0 = max(0, int(math.floor(mapped[:, 1].min())) - margin)
        x1 = min(image.width, int(math.ceil(mapped[:, 0].max())) + margin)
        y1 = min(image.height, int(math.ceil(mapped[:, 1].max())) + margin)
        if x1 <= x0 or y1 <= y0:
            return Image.new('RGB', (256, 256))
        region = image.crop((x0, y0, x1, y1))
        scale_xy = np.ones(2)
        if factor > 1.0:
            # Same antialiasing as the existing crop().resize() path when downscaling.
            new = (max(1, round((x1 - x0) / factor)), max(1, round((y1 - y0) / factor)))
            scale_xy = np.array([new[0] / (x1 - x0), new[1] / (y1 - y0)])
            region = region.resize(new, Image.Resampling.BILINEAR)
        matrix = np.diag(scale_xy) @ linear
        offset = scale_xy * (centre - [x0, y0] - linear @ [128.0, 128.0])
        data = (matrix[0, 0], matrix[0, 1], offset[0], matrix[1, 0], matrix[1, 1], offset[1])
        return region.transform((256, 256), Image.Transform.AFFINE, data,
                                resample=Image.Resampling.BILINEAR, fillcolor=(0, 0, 0))

    def transform_keypoints(self, keypoints, plan, *, flipped):
        """Same output convention/dtype/flip-swap as native.transform_keypoints_highres."""
        transformed = keypoints.astype(np.float32, copy=True)
        xy = keypoints[:, :2].astype(np.float64) - plan['centre']
        transformed[:, :2] = xy @ plan['label_linear'].T + 127.5
        if flipped:
            transformed[:, 0] = 255.0 - transformed[:, 0]
            for left_index, right_index in FLIP_PAIRS:
                transformed[[left_index, right_index]] = transformed[[right_index, left_index]]
        return transformed

    # ---- photometric ---------------------------------------------------------------
    def person_region(self, box, plan, flipped):
        x, y, w, h = map(float, box[:4])
        corners = np.array([[x, y], [x + w, y], [x, y + h], [x + w, y + h]]) - plan['centre']
        mapped = corners @ plan['label_linear'].T + 127.5
        if flipped:
            mapped[:, 0] = 255.0 - mapped[:, 0]
        low = np.clip(mapped.min(0), 0, 255)
        high = np.clip(mapped.max(0), 0, 255)
        if np.any(high - low < 8):
            return np.zeros(2), np.full(2, 255.0)
        return low, high

    def photometric(self, crop, plan, box, *, flipped):
        """Return (crop, hole rectangle or None).  The caller updates visibility for the hole."""
        c, u = self.config, plan['u']
        local = None
        pixels = None
        cj = c['color_jitter']
        do = [u[9] < cj['prob'], u[11] < cj['prob'], u[14] < cj['prob'], u[16] < cj['prob'],
              u[18] < cj['channel_swap_prob']]
        if any(do):
            pixels = np.asarray(crop, dtype=np.uint8)

            def convert(array, alpha=1.0, beta=0.0):
                return np.clip(array.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)

            lo, hi = cj['contrast_range']
            alpha = lo + (hi - lo) * u[12]
            contrast_first = u[13] < 0.5
            if do[0]:
                pixels = convert(pixels, beta=-cj['brightness_delta'] + 2 * cj['brightness_delta'] * u[10])
            if do[1] and contrast_first:
                pixels = convert(pixels, alpha=alpha)
            if do[2] or do[3]:
                hsv = np.array(Image.fromarray(pixels).convert('HSV'))
                if do[2]:
                    slo, shi = cj['saturation_range']
                    hsv[..., 1] = convert(hsv[..., 1], alpha=slo + (shi - slo) * u[15])
                if do[3]:
                    # hue_delta in OpenCV units (H in [0,180)); PIL H spans [0,256).
                    delta = (-cj['hue_delta'] + 2 * cj['hue_delta'] * u[17]) * 256.0 / 180.0
                    hsv[..., 0] = np.mod(hsv[..., 0].astype(np.int32) + int(round(delta)), 256).astype(np.uint8)
                pixels = np.asarray(Image.fromarray(hsv, 'HSV').convert('RGB'))
            if do[1] and not contrast_first:
                pixels = convert(pixels, alpha=alpha)
            if do[4]:
                pixels = np.ascontiguousarray(pixels[..., PERMUTATIONS[min(5, int(u[19] * 6))]])
        hole = None
        if u[20] < c['hole_prob']:
            local = np.random.default_rng(plan['local_seed'])
            low, high = self.person_region(box, plan, flipped)
            span = c['hole_max_frac'] - c['hole_min_frac']
            hw = max(1, int(round(256 * (c['hole_min_frac'] + span * u[21]))))
            hh = max(1, int(round(256 * (c['hole_min_frac'] + span * u[22]))))
            cx = low[0] + (high[0] - low[0]) * u[23]
            cy = low[1] + (high[1] - low[1]) * u[24]
            x0, y0 = max(0, int(round(cx - hw / 2))), max(0, int(round(cy - hh / 2)))
            x1, y1 = min(256, int(round(cx + hw / 2))), min(256, int(round(cy + hh / 2)))
            if x1 > x0 and y1 > y0:
                if pixels is None:
                    pixels = np.array(crop, dtype=np.uint8)
                elif not pixels.flags.writeable:
                    pixels = pixels.copy()
                if u[25] < 0.5:
                    pixels[y0:y1, x0:x1] = local.integers(0, 256, 3, dtype=np.uint8)
                else:
                    pixels[y0:y1, x0:x1] = local.integers(0, 256, (y1 - y0, x1 - x0, 3), dtype=np.uint8)
                hole = [x0, y0, x1 - x0, y1 - y0]
        if pixels is not None:
            crop = Image.fromarray(pixels)
        if u[26] < c['blur_prob']:
            lo, hi = c['blur_sigma']
            crop = crop.filter(ImageFilter.GaussianBlur(lo + (hi - lo) * u[27]))
        if u[28] < c['median_blur_prob']:
            sizes = c['median_sizes']
            crop = crop.filter(ImageFilter.MedianFilter(int(sizes[min(len(sizes) - 1, int(u[29] * len(sizes)))])))
        if u[30] < c['noise_prob']:
            if local is None:
                local = np.random.default_rng(plan['local_seed'])
            lo, hi = c['noise_sigma']
            noise = local.standard_normal((256, 256, 3), dtype=np.float32) * (lo + (hi - lo) * u[31])
            crop = Image.fromarray(np.clip(np.rint(np.asarray(crop, dtype=np.float32) + noise), 0, 255).astype(np.uint8))
        return crop, hole


class StrongStreams:
    """``synthetic_rng`` wrapper: same random()/integers() API, checkpoint state includes the strong stream.

    Trainers that save/restore ``dataset.synthetic_rng.bit_generator.state`` (e.g.
    continue_pose_research.py; FactorDataset nests this inside SynthesisStreams'
    'base' entry) therefore restore the strong stream exactly.  A legacy state
    (no schema) restores only the base stream.
    """

    def __init__(self, base, strong):
        self.base, self.strong = base, strong

    @property
    def bit_generator(self):
        return self

    @property
    def state(self):
        return {'schema': 'strong_streams_v1', 'base': self.base.bit_generator.state,
                'strong': self.strong.state_dict()}

    @state.setter
    def state(self, state):
        if isinstance(state, dict) and state.get('schema') == 'strong_streams_v1':
            self.base.bit_generator.state = state['base']
            self.strong.load_state_dict(state['strong'])
        else:
            self.base.bit_generator.state = state

    def random(self, *args, **kwargs):
        return self.base.random(*args, **kwargs)

    def integers(self, *args, **kwargs):
        return self.base.integers(*args, **kwargs)
