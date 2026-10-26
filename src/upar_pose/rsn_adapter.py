"""Pinned COCO-only three-stage RSN, adapted to the existing square pose API.

PredictHeatmap follows OpenMMLab v0.29.0 (Apache-2.0), upstream evidence and
license in the v42 rsn_upstream directory. No network or dynamic imports.
"""
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from mmcv.cnn import ConvModule
from .rsn_upstream import RSN
from .pipeline import sha256_file


class PredictHeatmap(nn.Module):
    def __init__(self, shape=(64, 64)):
        super().__init__()
        self.out_shape = shape
        self.conv_layers = nn.Sequential(
            ConvModule(256, 256, 1, norm_cfg=dict(type='BN'), inplace=False),
            ConvModule(256, 17, 3, padding=1, norm_cfg=dict(type='BN'), act_cfg=None, inplace=False))

    def forward(self, feature):
        return F.interpolate(self.conv_layers(feature), size=self.out_shape,
                             mode='bilinear', align_corners=True)


class RSNPose(nn.Module):
    def __init__(self, output_shape=(64, 64)):
        super().__init__()
        self.backbone = RSN(unit_channels=256, num_stages=3, num_units=4,
                            num_blocks=[3, 4, 6, 3], num_steps=4, norm_cfg=dict(type='BN'))
        self.keypoint_head = nn.Module()
        self.keypoint_head.predict_layers = nn.ModuleList([PredictHeatmap(output_shape) for _ in range(12)])
        self.error_heads = None
        self.register_buffer('coco_mean', torch.tensor([.485, .456, .406]).reshape(1,3,1,1), persistent=False)
        self.register_buffer('coco_std', torch.tensor([.229, .224, .225]).reshape(1,3,1,1), persistent=False)

    def forward_training(self, inputs):
        features = self.backbone((inputs*.25+.5-self.coco_mean)/self.coco_std)
        maps = [head(features[i//4][i%4]) for i, head in enumerate(self.keypoint_head.predict_layers)]
        return maps[-1], maps, {}

    def forward(self, inputs):
        # Auxiliary heads have no influence on the final features; omit eleven
        # unused low-resolution predictions at inference, not during training.
        features = self.backbone((inputs*.25+.5-self.coco_mean)/self.coco_std)
        return self.keypoint_head.predict_layers[-1](features[-1][-1])

    def initialize(self, checkpoint: Path, *, expected_sha256, reset_head=True):
        if sha256_file(checkpoint) != expected_sha256:
            raise ValueError('RSN pretrained artifact identity changed')
        payload = torch.load(checkpoint, map_location='cpu', weights_only=True)
        config = payload['meta']['config']
        required = ('pretrained=None', "encoding='Megvii'", "ann_file='data/coco/annotations/person_keypoints_train2017.json'", 'num_stages=3')
        if any(s not in config for s in required) or payload['meta']['epoch'] != 210:
            raise ValueError('Different upstream fitting boundary or output encoding')
        self.load_state_dict(payload['state_dict'], strict=True)
        # Megvii supervision has peak255. Scaling final BN affine parameters
        # preserves normalized outputs while giving our peak1 loss normal units.
        with torch.no_grad():
            for head in self.keypoint_head.predict_layers:
                last = head.conv_layers[-1]
                last.bn.weight.div_(255.)
                last.bn.bias.div_(255.)
                if reset_head:
                    for index in (1, 2):
                        nn.init.normal_(last.conv.weight[index], std=.001)
                        last.bn.weight[index] = 1.
                        last.bn.bias[index] = 0.
                        last.bn.running_mean[index] = 0.
                        last.bn.running_var[index] = 1.
        return {'initialization':'COCO_train2017', 'checkpoint_sha256':expected_sha256,
            'strict_state_load':True, 'upstream_epoch':210, 'upstream_config':config,
            'output_unit_conversion':'Each final BN weight/bias divided by255; no decoder offset added',
            'reinitialized_channels':[1,2] if reset_head else [], 'loss_ignored_channels':[3,4],
            'upstream_input_width_height':[192,256], 'adapted_input_width_height':[256,256],
            'method_limit':'A new full fine-tuning recipe, not a one-factor capacity comparison or a claimed replication of original RSN training.'}


def rsn_loss(model, output, stages, auxiliary, targets, weights, coordinates, visibility):
    # A source-fine-tuning policy in peak1 units, not the upstream coarse-to-fine
    # Megvii/OHKM target recipe. All12 heads supervise corresponding scales.
    values = [((m.float()-targets.float()).square().mean((2,3))*weights).sum()/weights.sum().clamp_min(1.) for m in stages]
    scales = [1. if i%4 == 3 else .25 for i in range(12)]
    loss = sum(v*s for v,s in zip(values,scales))/sum(scales)
    return loss, {'heatmap':values[-1], 'intermediate':torch.stack(values).mean()}
