"""Shared square crops with explicit visibility and source-only person cut-paste.

GT files never change. Cut-paste is rectangular (not a segmentation method).
Only the supplied training items may be donors; PETA is rejected for this cycle.
Optional recipe key ``strong_augmentation`` (pose_strong_augmentation.py) adds a
test-like truncation of the original image, rotation/scale/shift/half-body of the
crop window and photometric ops after the paste, from a separate RNG stream;
absent, every draw and sample is unchanged.
"""
import random
import numpy as np
from PIL import Image, ImageEnhance
import torch
from torch.utils.data import Dataset
from train_scratch_cnn import image_path
from train_scratch_robust_v5 import CENTER_JITTER_FRACTION, SCALE_JITTER_FRACTION
from pose_square_adapter import square_jitter, configured_square_dataset
import train_highcap_robust_native_highres_v16 as native
from upar_pose.scratch_cnn import normalize_image


def update_occluded_visibility(keypoints, rectangle):
    result = keypoints.copy()
    left, top, width, height = rectangle
    covered = ((result[:, 0] >= left) & (result[:, 0] < left + width) &
               (result[:, 1] >= top) & (result[:, 1] < top + height) & (result[:, 2] > 0))
    result[covered, 2] = 1
    return result


class ResearchDataset(Dataset):
    def __init__(self, items, recipe):
        self.items, self.recipe = list(items), recipe
        self.synthetic = recipe.get('person_paste_probability', 0.)
        if not 0 <= self.synthetic <= 1:
            raise ValueError('Invalid paste probability')
        if any(not str(item['file_name']).lower().startswith(('market/', 'pa100k/')) for item in self.items):
            raise ValueError('Only official Market/PA100K source items may enter this fit/donor pool')
        configured_square_dataset({'input_size': [256, 256], 'heatmap_sigma': 2.})
        # Separate stream so adding a synthetic arm does not change standard crop,
        # photometric, shuffle or flip draws, even when additional images are read.
        self.synthetic_rng = np.random.default_rng(recipe['data_seed'] + 73591)
        self.strong = None
        if recipe.get('strong_augmentation') not in (None, False):
            from pose_strong_augmentation import StrongAugmentation, StrongStreams
            if (native.INPUT_WIDTH, native.INPUT_HEIGHT, native.HEATMAP_WIDTH) != (256, 256, 64):
                raise ValueError('strong_augmentation supports the Square256/64 targets only')
            self.strong = StrongAugmentation(recipe['strong_augmentation'], recipe['data_seed'])
            # Checkpointing synthetic_rng.bit_generator.state now includes the strong stream.
            self.synthetic_rng = StrongStreams(self.synthetic_rng, self.strong)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        item = self.items[index]
        with Image.open(image_path(item['file_name'])) as source:
            image = source.convert('RGB')
        box, keypoints = item['box'], item['keypoints']
        if self.strong is not None:
            # All strong draws first (separate stream); optional test-like truncation of the
            # ORIGINAL image redefines the box/labels before the unchanged base jitter below.
            plan = self.strong.draw()
            image, box, keypoints = self.strong.truncate(image, box, keypoints, plan)
        geometry = square_jitter(box,
            center_dx=float(np.random.uniform(-CENTER_JITTER_FRACTION, CENTER_JITTER_FRACTION)),
            center_dy=float(np.random.uniform(-CENTER_JITTER_FRACTION, CENTER_JITTER_FRACTION)),
            scale=float(np.random.uniform(1 - SCALE_JITTER_FRACTION, 1 + SCALE_JITTER_FRACTION)))
        left, top, width, height = geometry
        if self.strong is None:
            crop = image.crop((left, top, left + width, top + height)).resize((256, 256), Image.Resampling.BILINEAR)
        else:
            # Strong window composed on the base jitter (whose draws are still consumed);
            # one warp from the original image, so rotated corners hold real content.
            self.strong.sample(plan, keypoints, geometry)
            crop = self.strong.warp(image, plan)
        flipped = random.random() < .5
        if flipped:
            crop = crop.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        crop = ImageEnhance.Brightness(crop).enhance(random.uniform(.8, 1.2))
        crop = ImageEnhance.Contrast(crop).enhance(random.uniform(.8, 1.2))
        if self.strong is None:
            transformed = native.transform_keypoints_highres(item['keypoints'], geometry, flipped=flipped)
        else:
            transformed = self.strong.transform_keypoints(keypoints, plan, flipped=flipped)
        donor_id, rectangle = -1, [-1, -1, 0, 0]
        if self.synthetic_rng.random() < self.synthetic:
            donor_index = int(self.synthetic_rng.integers(len(self.items) - 1))
            if donor_index >= index:
                donor_index += 1
            donor = self.items[donor_index]
            x, y, w, h = donor['box']
            with Image.open(image_path(donor['file_name'])) as source:
                patch = source.convert('RGB').crop((x, y, x + w, y + h))
            patch_height = int(self.synthetic_rng.integers(96, 225))
            patch_width = min(144, max(24, round(patch_height * w / h)))
            patch = patch.resize((patch_width, patch_height), Image.Resampling.BILINEAR)
            # Truncated rectangular person occluders, not pasted target GT.
            px = int(self.synthetic_rng.integers(-patch_width // 2, 256 - patch_width // 2))
            py = int(self.synthetic_rng.integers(-patch_height // 4, 256 - patch_height // 2))
            crop.paste(patch, (px, py))
            rectangle = [px, py, patch_width, patch_height]
            transformed = update_occluded_visibility(transformed, rectangle)
            donor_id = int(donor['annotation_id'])
        if self.strong is not None:
            crop, hole = self.strong.photometric(crop, plan, box, flipped=flipped)
            if hole is not None:
                # Occluded but still supervised (heatmap weight unchanged), as for the paste.
                transformed = update_occluded_visibility(transformed, hole)
        heatmaps, weights = native.make_heatmaps_highres(transformed)
        visibility = transformed[:, 2].copy()
        # A crop-out is not a visible target for the visibility auxiliary task.
        outside = ((transformed[:, :2] < 0) | (transformed[:, :2] > 255)).any(1)
        visibility[outside & (visibility > 0)] = 1
        return (torch.from_numpy(normalize_image(np.asarray(crop, dtype=np.uint8))),
                torch.from_numpy(heatmaps), torch.from_numpy(weights),
                torch.from_numpy(transformed[:, :2] * (63 / 255)), torch.from_numpy(visibility),
                torch.tensor([int(item['annotation_id']), donor_id, *rectangle]))
