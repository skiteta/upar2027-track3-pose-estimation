"""Frozen teacher affine storage conversion; caller owns architecture/adoption gates.

The conversion body is promoted unchanged from the CPU/CUDA-audited helper in
artifacts/runs/20261017T0338+0900_teacher_bf16_probe/teacher_storage.py.
"""
import torch
from torch import nn


CONVERTIBLE = (nn.Conv1d, nn.Conv2d, nn.Conv3d,
               nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d, nn.Linear)


def convert_frozen_affine_parameters_to_bf16(model):
    """Convert only Conv/ConvTranspose/Linear weight/bias; never norms/buffers.

    Reject trainable/training, mixed-source-dtype or cross-role shared tensors.
    This changes only this in-memory model; callers own numerical validation.
    It is not permission to quantize arbitrary teachers or overwrite checkpoints.
    """
    if any(module.training for module in model.modules()) or any(p.requires_grad for p in model.parameters()):
        raise ValueError('Only fully frozen eval teachers may be converted')
    if any(p.grad is not None for p in model.parameters()):
        raise ValueError('Frozen teacher must not retain gradients from earlier training')
    if any(p.is_floating_point() and p.dtype != torch.float32 for p in model.parameters()):
        raise ValueError('Expected original FP32 teacher parameters; do not convert twice')
    targets, protected = {}, {id(buffer) for buffer in model.buffers()}
    for module_name, module in model.named_modules():
        for key, parameter in module.named_parameters(recurse=False):
            name = f'{module_name}.{key}' if module_name else key
            if isinstance(module, CONVERTIBLE) and key in ('weight', 'bias'):
                targets.setdefault(id(parameter), {'parameter': parameter, 'names': [], 'owners': []})
                targets[id(parameter)]['names'].append(name)
                targets[id(parameter)]['owners'].append(type(module).__module__ + '.' + type(module).__name__)
            else:
                protected.add(id(parameter))
    if set(targets) & protected:
        raise ValueError('A Conv/Linear parameter is shared with a protected parameter role')
    if not targets:
        raise ValueError('No convertible teacher parameters found')
    records = []
    with torch.no_grad():
        for target in targets.values():
            parameter = target['parameter']
            before = parameter.numel() * parameter.element_size()
            parameter.data = parameter.detach().to(dtype=torch.bfloat16)
            records.append({'names': target['names'], 'owners': target['owners'],
                            'shape': list(parameter.shape), 'from_dtype': 'torch.float32',
                            'to_dtype': str(parameter.dtype), 'bytes_before': before,
                            'bytes_after': parameter.numel() * parameter.element_size()})
    assert all(not p.requires_grad and p.grad is None for p in model.parameters())
    return {'converted': records, 'bytes_saved': sum(r['bytes_before'] - r['bytes_after'] for r in records),
            'norms_and_buffers_converted': False, 'source_checkpoints_written': False,
            'adoption_status': 'experimental_requires_actual_output_parity'}
