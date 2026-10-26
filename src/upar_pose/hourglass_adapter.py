"""COCO-only Hourglass52 with the established UPAR normalization/head contract."""
from pathlib import Path
import torch
from torch import nn
from .hourglass_upstream import HourglassNet
from .pipeline import sha256_file

CHECKPOINT_SHA256 = '4ec713baae921f9b64b7a19bbc2f192427d6e7abcdfd7290d4902b405cf2dc10'


class HourglassPose(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = HourglassNet(num_stacks=1)
        self.keypoint_head = nn.Module()
        self.keypoint_head.multi_final_layers = nn.ModuleList([nn.Conv2d(256, 17, 1)])
        self.register_buffer('coco_mean', torch.tensor([.485,.456,.406]).reshape(1,3,1,1), persistent=False)
        self.register_buffer('coco_std', torch.tensor([.229,.224,.225]).reshape(1,3,1,1), persistent=False)

    def forward(self, inputs):
        normalized = (inputs*.25+.5-self.coco_mean)/self.coco_std
        return self.keypoint_head.multi_final_layers[0](self.backbone(normalized)[0])

    def initialize(self, checkpoint: Path):
        if checkpoint is None or sha256_file(checkpoint) != CHECKPOINT_SHA256:
            raise ValueError('Only the explicitly audited COCO checkpoint is supported')
        payload = torch.load(checkpoint, map_location='cpu', weights_only=True)
        self.load_state_dict(payload['state_dict'], strict=True)
        with torch.no_grad():
            for index in (1,2):
                nn.init.normal_(self.keypoint_head.multi_final_layers[0].weight[index], std=.001)
                self.keypoint_head.multi_final_layers[0].bias[index].zero_()
        return {'initialization': 'COCO_train2017', 'checkpoint_sha256': CHECKPOINT_SHA256,
            'transferred_channels': [0,*range(3,17)], 'reinitialized_channels': [1,2],
            'loss_ignored_channels': [3,4], 'strict_state_load': True,
            'upstream_config': payload['meta']['config'], 'upstream_epoch': payload['meta']['epoch']}
