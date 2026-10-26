"""Minimal registry/checkpoint compatibility, without importing all of MMPose."""
import logging
from mmcv.utils import Registry

BACKBONES = Registry('upar_litehrnet_backbone')


def channel_shuffle(x, groups):
    batch, channels, height, width = x.shape
    if channels % groups:
        raise ValueError('Channels not divisible by shuffle groups')
    return x.reshape(batch, groups, channels//groups, height, width).transpose(1, 2).contiguous().reshape(batch, channels, height, width)


def get_root_logger():
    return logging.getLogger('upar_litehrnet')


def load_checkpoint(*args, **kwargs):
    raise RuntimeError('Legacy unsafe loader disabled: use hash-verified weights_only=True in adapter')
