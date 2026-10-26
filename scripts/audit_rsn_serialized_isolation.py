#!/usr/bin/env python3
"""Minimal serialized-only validator, deliberately importing no eager model code.

This tests the previously identified process-context hypothesis, not another
export attempt. Original failed checkers and existing bytes stay untouched.
"""
from collections import Counter
from datetime import datetime,timezone
from pathlib import Path
import sys
import time
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from upar_pose.pipeline import sha256_file,write_json_atomic
BASE=ROOT/'artifacts/runs/20261005T025700+0900_mechanism_cycle_v42'


def inspect(module):
    graph=module.inlined_graph.copy();torch._C._jit_pass_inline(graph)
    def nodes(block):
        for node in block.nodes():
            yield node
            for sub in node.blocks():yield from nodes(sub)
    allnodes=list(nodes(graph));convs=[n for n in allnodes if n.kind()=='aten::_convolution']
    return {'convolutions':len(convs),'remaining_callmethod':sum(n.kind()=='prim::CallMethod' for n in allnodes),
        'TF32_flags':dict(Counter(str(list(n.inputs())[-1].toIValue()) for n in convs))}


def main():
    if not torch.__version__.startswith('2.4.1') or torch.cuda.is_available():raise ValueError('CPU official-compatible2.4.1 required')
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    out=BASE/'rsn_serialized_isolation_cpu241';out.mkdir(exist_ok=False)
    source=BASE/'rsn_mutable_export_audit/model.unfrozen.torchscript'
    if sha256_file(source)!='5b11701f4a63354d1e571beb8a071cb83034ffbf3d6a084ead0a85c78ae6a668':raise ValueError('Original serialized model changed')
    fixture_path=BASE/'rsn_cuda_inference_r4/parity_tensors.pt'
    fixture=torch.load(fixture_path,map_location='cpu',weights_only=True)
    start=time.monotonic();saved=torch.jit.load(str(source),map_location='cpu').eval();elapsed=time.monotonic()-start
    before=inspect(saved)
    with torch.inference_mode(),torch.jit.optimized_execution(False):actual=saved(fixture['input'])
    after=inspect(saved);error=float((actual-fixture['CPU_reference']).abs().max())
    report={'at':datetime.now(timezone.utc).isoformat(),'torch':torch.__version__,
        'before':before,'after':after,'CPU_2p4_vs_prior_2p7_actual_tensor_error':error,
        'named_parameters':len(list(saved.named_parameters())),'named_buffers':len(list(saved.named_buffers())),
        'cpu_load_seconds':elapsed,'model_sha256':sha256_file(source),'fixture_sha256':sha256_file(fixture_path),
        'source_sha256':sha256_file(Path(__file__)),
        'adoption':'NOT_ADOPTED_GPU_TRANSFER_AND_RUNTIME_UNMEASURED',
        'interpretation':'Serialized-only process versus prior eager-model-importing audit. A passing result supports a checker/process-context explanation, not a full compiler root cause.',
        'limits':['COCO initialization only, not current UPAR-trained RSN weights.','No retracing, weight change, HPE fitting or CUDA allocation.',
            'Original1e-5 CPU numeric tolerance and627FP32-convolution requirement retained.','CUDA loading speed, same-precision output and deployed wall time remain unmeasured for this format.']}
    write_json_atomic(out/'observed.json',report)
    if before!=after or before['TF32_flags']!={'False':627} or before['remaining_callmethod'] or error>1e-5:
        raise ValueError('Isolated serialized validator failed; observed evidence preserved')
    if not torch.isfinite(actual).all():raise ValueError('Nonfinite actual output')
    report['status']='CPU_SERIALIZED_GRAPH_AND_ABI_PASS'
    write_json_atomic(out/'report.json',report);print(report,flush=True)


if __name__=='__main__':main()
