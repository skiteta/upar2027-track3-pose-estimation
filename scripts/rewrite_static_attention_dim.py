"""Remove capture-blocking traced scalar division; preserve all learned tensors.

Only the current ViT-L archives are accepted: 24 attention layers, width1024,
16 heads. The linear weights require width1024, so the head width is always64.
The edited intermediate archive and a normal torch.jit.save export are retained.
"""
import argparse
import hashlib
import json
import re
import sys
import zipfile
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from upar_pose.pipeline import sha256_file, write_json_atomic


def transform_source(text):
    matches = list(re.finditer(r'(?m)^(    )(_\d+) = int\(torch\.floor_divide\(c, CONSTANTS\.c\d+\)\)$', text))
    if not matches:
        return text, 0
    if len(matches) != 1 or 'class Attention(Module):' not in text:
        raise ValueError('Unexpected scalar-division program')
    match = matches[0]
    if not re.search(r'torch\.reshape\([^\n]+, \[[^\n]+, 3, 16, ' + re.escape(match[2]) + r'\]\)', text):
        raise ValueError('Head dimension is not the expected16-head reshape')
    if 'c = ops.prim.NumToTensor(torch.size(argument_1, 2))' not in text:
        raise ValueError('Unexpected source of channel dimension')
    return text[:match.start()] + match[1] + match[2] + ' = 64' + text[match.end():], 1


def main(source, output, report_path):
    source = Path(source).resolve()
    output = Path(output).absolute()
    intermediate = output.with_name(output.stem + '_source_rewrite.torchscript')
    report_path = Path(report_path)
    if output.exists() or intermediate.exists() or report_path.exists():
        raise FileExistsError('Preserve existing export; no overwrite')
    torch.set_num_threads(1)
    model = torch.jit.load(str(source), map_location='cpu').eval()
    state = model.state_dict()
    for index in range(24):
        prefix = f'blocks.{index}.attn.'
        assert tuple(state[prefix + 'qkv.weight'].shape) == (3072, 1024)
        assert tuple(state[prefix + 'proj.weight'].shape) == (1024, 1024)
    assert len([k for k in state if k.endswith('.attn.qkv.weight')]) == 24
    output.parent.mkdir(parents=True, exist_ok=True)
    changed = []
    copied_hashes = {}
    with zipfile.ZipFile(source) as src, zipfile.ZipFile(intermediate, 'x', allowZip64=True) as dst:
        for entry in src.infolist():
            data = src.read(entry.filename)
            new = data
            if '/code/' in entry.filename and entry.filename.endswith('.py'):
                rewritten, count = transform_source(data.decode())
                if count:
                    changed.append(entry.filename)
                    new = rewritten.encode()
            if new == data:
                copied_hashes[entry.filename] = hashlib.sha256(data).hexdigest()
            dst.writestr(entry, new)
    assert len(changed) == 24
    with zipfile.ZipFile(intermediate) as check:
        assert check.testzip() is None
        assert all(hashlib.sha256(check.read(n)).hexdigest() == h for n, h in copied_hashes.items())
    patched = torch.jit.load(str(intermediate), map_location='cpu').eval()
    new_state = patched.state_dict()
    assert state.keys() == new_state.keys()
    assert all(torch.equal(state[k], new_state[k]) for k in state)
    assert 'aten::floor_divide' not in str(patched.inlined_graph)
    # Standard serializer regenerates program/debug/serialization metadata.
    torch.jit.save(patched, str(output))
    final = torch.jit.load(str(output), map_location='cpu').eval()
    assert 'aten::floor_divide' not in str(final.inlined_graph)
    assert state.keys() == final.state_dict().keys()
    assert all(torch.equal(state[k], final.state_dict()[k]) for k in state)
    with zipfile.ZipFile(output) as check:
        assert check.testzip() is None
    report = {'source': str(source), 'source_sha256': sha256_file(source), 'output': str(output),
              'output_sha256': sha256_file(output), 'intermediate': str(intermediate), 'intermediate_sha256': sha256_file(intermediate),
              'changed_code_members': changed, 'all_other_intermediate_members_byte_identical': True,
              'state_tensors': len(state), 'all_state_tensors_bit_identical': True, 'head_width': 64, 'width': 1024, 'heads': 16,
              'floor_divide_nodes_remaining': 0, 'scope': 'Structural export only. No fitting or numerical optimization. Real eager/graph parity and timing remain required.'}
    write_json_atomic(report_path, report)
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--report', required=True)
    args = parser.parse_args()
    main(args.source, args.output, args.report)
