"""Eval-only ConvModule fusion on a separate deployment copy, never live fits."""
import torch

def fuse_eval_conv_modules(model):
    if model.training:raise ValueError('Evalmodel required; no trainingmodule edits')
    plan=[]
    for name,module in model.named_modules():
        if not getattr(module,'with_norm',False) or not hasattr(module,'conv'):continue
        if tuple(getattr(module,'order',()))!=('conv','norm','act'):raise ValueError('Unrecognizedoperationorder:'+name)
        norm=getattr(module,module.norm_name)
        if not isinstance(module.conv,torch.nn.Conv2d) or not isinstance(norm,torch.nn.BatchNorm2d):raise ValueError('OnlyConv2d+BatchNorm2d allowed')
        if module.conv.training or norm.training or norm.running_mean is None or norm.running_var is None:raise ValueError('CalibratedevalBN required')
        plan.append((name,module,norm))
    if not plan:raise ValueError('NoeligibleConvModules')
    for name,module,norm in plan:
        module.conv=torch.nn.utils.fuse_conv_bn_eval(module.conv,norm)
        setattr(module,module.norm_name,torch.nn.Identity())
    return [name for name,_,_ in plan]
