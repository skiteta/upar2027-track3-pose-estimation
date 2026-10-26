"""Square-input bridge reusing the established crop, target and evaluation code."""
import math
import train_highcap_robust_native_highres_v16 as native
import evaluate_highcap_highres_inference_v15 as evaluation
_make_heatmaps = native.make_heatmaps_highres


def square_geometry(box, padding=1.25):
    x,y,w,h = map(float, box[:4])
    if not all(math.isfinite(v) for v in (x,y,w,h,padding)) or min(w,h,padding)<=0:
        raise ValueError('Finite positive box and padding required')
    extent = max(w,h)*padding
    return x+w/2-extent/2, y+h/2-extent/2, extent, extent


def square_jitter(box, *, center_dx, center_dy, scale):
    left,top,width,height = square_geometry(box)
    cx,cy = left+width/2+center_dx*width,top+height/2+center_dy*height
    width,height = width*scale,height*scale
    return cx-width/2,cy-height/2,width,height


def configured_square_dataset(recipe):
    if recipe['input_size'] != [256,256] or recipe['heatmap_sigma'] != 2.0:
        raise ValueError('Only fixed Hourglass square256/sigma2 recipe supported')
    if recipe.get('synthetic_occlusion') or recipe.get('half_body'):
        raise ValueError('Separate base-model comparison, no additional augmentation')
    native.INPUT_WIDTH = native.INPUT_HEIGHT = 256
    native.HEATMAP_WIDTH = native.HEATMAP_HEIGHT = 64
    native.jitter_crop_geometry = square_jitter
    native.make_heatmaps_highres = lambda keypoints: _make_heatmaps(keypoints, sigma=2.0)
    return native.NativeHighResRobustDataset


class SquareInteractionDataset(evaluation.ResolutionInteractionDataset):
    def __init__(self, items, size):
        if tuple(size) != (256,256): raise ValueError('Square256 evaluation only')
        self.items = list(items); self.width = self.height = 256
        # A process-local adapter, no mutation of shared on-disk or active jobs.
        evaluation.scaled_crop_geometry = square_geometry
