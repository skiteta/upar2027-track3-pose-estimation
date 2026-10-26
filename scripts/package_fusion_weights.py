#!/usr/bin/env python3
"""Package several saved pose weights (RSN and/or ViTPose) into one fused Codabench ZIP.

Each member is exported exactly as package_rsn_weights.py does (RSN: CPU
Conv/BN fusion + trace; ViT: torch 2.4.1 trace, optional FP16 storage).  The
run.py comes from scripts/templates/fusion_run_template.py, which reuses the
v50 crop/decode/AMP code and adds unit-peak heatmap fusion with model-mean
scores, mirroring scripts/evaluate_multimodel_map_fusion.py.  The extracted ZIP
is run under torch 2.4.1 (CPU) and compared with the development fusion
predictions of the same weights and view configuration.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]
import package_rsn_weights as single  # noqa: E402

TEMPLATE = ROOT / 'scripts/templates/fusion_run_template.py'
VIEW_CONFIGS = {'v6': (1.20, 1.25, 1.30), 'v4': (1.20, 1.30), 'v2': (1.25,)}
ARCHITECTURES = {'vit_b': 'COCOMAE_ViTPose_B16', 'vit_l': 'COCOMAE_ViTPose_L16'}
VIT_SIZES = {'vit_b': 'B16', 'vit_l': 'L16', 'vit_h': 'H16'}
DEFAULT_PRETRAINING = 'COCO 2017 (pose pre-training for RSN, images-only MAE for ViT); PoseTrack18 only where declared by a member config'


def architecture_of(exported: dict) -> str:
    """Parent architecture label; ImageNet-1K MAE -> COCO ViTPose members (patch padding / no cls token) get their own name."""
    arch = exported.get('architecture')
    if arch == 'hrformer_b':
        return 'IN1K_COCOPose_HRFormer_B'
    if arch is None:
        return 'COCO3xRSN50'
    if arch not in VIT_SIZES:
        raise ValueError(f'unknown ViT architecture {arch!r}')
    if exported.get('patch_padding') or exported.get('with_cls_token') is False:
        return f'IN1KMAE_COCOViTPose_{VIT_SIZES[arch]}'
    return ARCHITECTURES.get(arch, f'COCOMAE_ViTPose_{VIT_SIZES[arch]}')


def resolve_member_views(views: str, member_views: list[str] | None, n_members: int) -> list[str]:
    """One view-config name per member: --views for all unless --member-views is given."""
    if member_views is None:
        return [views] * n_members
    if len(member_views) != n_members:
        raise ValueError('--member-views needs one view config per --labels entry')
    if len(set(member_views)) == 1 and member_views[0] != views:
        raise ValueError('uniform --member-views must equal --views (or drop --member-views)')
    return list(member_views)


def fusion_constants(a: argparse.Namespace, files: list[str], shas: list[str], model_weights: list[float], member_views: list[str]) -> str:
    """The #__FUSION_CONSTANTS__ block of run.py: one (paddings, equal scale weights) view set per member."""
    member_paddings = tuple(tuple(VIEW_CONFIGS[views]) for views in member_views)
    return (f'MODEL_TYPE = {a.candidate!r}\n'
            f'MODEL_FILES = {tuple(files)!r}\n'
            f'MODEL_SHA256S = {tuple(shas)!r}\n'
            f'MODEL_WEIGHTS = {tuple(float(w) for w in model_weights)!r}\n'
            f'MEMBER_PADDINGS = {member_paddings!r}\n'
            f'MEMBER_SCALE_WEIGHTS = {tuple(tuple(1.0 / len(paddings) for _ in paddings) for paddings in member_paddings)!r}\n'
            f'GRAPH_MODEL_INDICES = {tuple(a.labels.index(label) for label in a.cuda_graph_labels)!r}\n'
            f'DECODER = {a.decoder!r}\n'
            f'RUNTIME_GUARD = {None if a.runtime_guard is None else (float(a.runtime_guard[0]), int(a.runtime_guard[1]))!r}\n'
            f'EXPECTED_OKS = {None if a.expected_oks is None else (float(a.expected_oks[0]), float(a.expected_oks[1]))!r}\n'
            f'AMBIGUITY_BETA = {float(a.ambiguity_beta)!r}\n'
            f'TORSO_GAMMA = {float(a.torso_gamma)!r}\n'
            f'SQUAT_BOX = {None if a.squat_box is None else (float(a.squat_box[0]), float(a.squat_box[1]))!r}\n'
            f'MEMBER_DROP_GUARD = {None if a.member_drop_guard is None else (float(a.member_drop_guard[0]), int(a.member_drop_guard[1]))!r}\n'
            f'PROBE_MEMBERS = {tuple(a.labels.index(label) for label in a.probe_labels)!r}\n'
            f'PROBE_TIMEOUT = {float(a.probe_timeout)!r}\n'
            f'DIAGNOSTIC_FILES = {tuple(Path(p).name for p in a.diagnostic_probes)!r}\n'
            f'DIAGNOSTIC_TIMEOUT = {float(a.diagnostic_timeout)!r}\n'
            f'FALLBACK_WEIGHTS = {None if a.fallback_weights is None else tuple(float(w) for w in a.fallback_weights)!r}')


def render_run(constants: str) -> str:
    """run.py source: the fusion template with its constants and the embedded helper modules."""
    template = TEMPLATE.read_text()
    if any(template.count(marker) != 1 for marker in ('#__FUSION_CONSTANTS__', '#__CUDA_GRAPH_FORWARD__', '#__DARK_DECODE__', '#__EXPECTED_OKS__')):
        raise ValueError('template marker missing')
    graph_source = (ROOT / 'src/upar_pose/cuda_graph_forward.py').read_text()
    return (template.replace('#__FUSION_CONSTANTS__', constants)
            .replace('#__CUDA_GRAPH_FORWARD__', graph_source)
            .replace('#__DARK_DECODE__', (ROOT / 'src/upar_pose/dark_decode.py').read_text())
            .replace('#__EXPECTED_OKS__', (ROOT / 'src/upar_pose/expected_oks.py').read_text()))


def smoke(archive: Path, fused_predictions: Path, n: int = 12) -> dict:
    import numpy as np
    from train_scratch_cnn import dataset_items, load_domain
    preds = {p['image_id']: p for p in json.loads(fused_predictions.read_text())}
    peta = load_domain('peta')
    images = {int(i['id']): i for i in peta['images']}
    items = [r for r in dataset_items(peta, require_labels=False) if r['image_id'] in preds][:n]
    with tempfile.TemporaryDirectory() as tmp:
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(tmp)
        import torch
        from audit_rsn_serialized_isolation import inspect
        graphs = {}
        for name in sorted(Path(tmp).glob('model_*.torchscript')):
            graphs[name.name] = inspect(torch.jit.load(str(name), map_location='cpu').float().eval())
            if set(graphs[name.name]['TF32_flags']) != {'False'}:
                raise ValueError(f'TF32 convolution in {name.name}: {graphs[name.name]}')
        spec = importlib.util.spec_from_file_location('submission_run', Path(tmp) / 'run.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        module.load_model()
        worst_px, worst_score = 0.0, 0.0
        for item in items:
            image = images[item['image_id']]
            sample = {'image_id': item['image_id'], 'file_name': image['file_name'], 'image_path': str(ROOT / 'data' / image['file_name']),
                      'width': image['width'], 'height': image['height'], 'boxes': [item['box']], 'box_ids': [item['annotation_id']]}
            out = module.predict_image(sample)
            kp = np.asarray(out['keypoints'], dtype=float).reshape(-1, 3)
            ref = np.asarray(preds[item['image_id']]['keypoints'], dtype=float).reshape(-1, 3)
            worst_px = max(worst_px, float(np.abs(kp[:, :2] - ref[:, :2]).max()))
            worst_score = max(worst_score, abs(float(out['scores'][0]) - float(preds[item['image_id']]['score'])))
    return {'graphs_torch241': graphs, 'images': len(items), 'max_abs_pixel_vs_dev_fusion': worst_px,
            'max_abs_score_vs_dev_fusion': worst_score, 'device': 'cpu'}


def main() -> int:
    if len(sys.argv) == 4 and sys.argv[1] == '--smoke-only':
        import torch
        result = smoke(Path(sys.argv[2]), Path(sys.argv[3]))
        result['torch'] = torch.__version__
        print(json.dumps(result))
        return 0
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--weights', type=Path, nargs='+', required=True)
    p.add_argument('--labels', nargs='+', required=True)
    p.add_argument('--model-weights', type=float, nargs='+', help='fixed fusion weights (default equal)')
    p.add_argument('--views', choices=sorted(VIEW_CONFIGS), required=True)
    p.add_argument('--member-views', choices=sorted(VIEW_CONFIGS), nargs='+',
                   help='per-member view configs, one per --labels entry (default: --views for every member); '
                        'crops are shared per padding, each member averages its own views before the fusion')
    p.add_argument('--candidate', required=True)
    p.add_argument('--fit-scope', required=True)
    p.add_argument('--training-configs', type=Path, nargs='+', required=True)
    p.add_argument('--fused-predictions', type=Path, required=True, help='dev fusion predictions of the same weights and views')
    p.add_argument('--selection-note', required=True)
    p.add_argument('--authority', required=True)
    p.add_argument('--store-half', action='store_true', help='FP16 parameter storage for ViT members')
    p.add_argument('--store-half-all', action='store_true', help='FP16 parameter storage for every member (run.py restores FP32 at load)')
    p.add_argument('--decoder', choices=('quadratic', 'dark'), default='quadratic', help='fused-map coordinate refinement')
    p.add_argument('--runtime-guard', type=float, nargs=2, metavar=('SECONDS_PER_IMAGE', 'MIN_IMAGES'),
                   help='switch to centre-padding views once the mean wall time per image exceeds the budget')
    p.add_argument('--cuda-graph-labels', nargs='*', default=[],
                   help='members replayed as CUDA graphs (exactness-checked at load; RSN only has been validated)')
    p.add_argument('--expected-oks', type=float, nargs=2, metavar=('ALPHA', 'TARGET_STD_CELLS'),
                   help='expected-OKS blur of the fused map before decoding (sweep_decoders eoks_a{ALPHA}_t{T})')
    p.add_argument('--ambiguity-beta', type=float, default=0.0, help='score *= 1 - beta * limb far-secondary-peak ratio')
    p.add_argument('--torso-gamma', type=float, default=0.0, help='score *= exp(-gamma * max shoulder/hip far-secondary-peak ratio)')
    p.add_argument('--diagnostic-probes', nargs='*', default=[], type=Path,
                   help='extra TorchScript files shipped only to be loaded and run in a child process at load time and logged; never fused')
    p.add_argument('--diagnostic-timeout', type=float, default=240.0, help='seconds per diagnostic probe')
    p.add_argument('--probe-labels', nargs='*', default=[],
                   help='members first loaded and run in a child process at load time; a failing/timed-out member is left out')
    p.add_argument('--probe-timeout', type=float, default=900.0, help='seconds per member probe')
    p.add_argument('--fallback-weights', type=float, nargs='+', default=None,
                   help='fusion weights (one per member) used instead of --model-weights when any member probe fails; '
                        'members with weight 0 in the active weights are not loaded')
    p.add_argument('--member-drop-guard', type=float, nargs=2, default=None, metavar=('SECONDS_PER_IMAGE', 'WINDOW_IMAGES'),
                   help='after the runtime guard reduced the views: if the mean wall time per image over each window still exceeds '
                        'the budget, drop the slowest member (sampled forward time) so the run finishes inside the time limit')
    p.add_argument('--squat-box', type=float, nargs=2, default=None, metavar=('MAX_H_OVER_W', 'TARGET_H_OVER_W'),
                   help='crop window only: extend boxes with h/w < MAX downward to h = TARGET*w (likely bottom-truncated persons)')
    p.add_argument('--external-pretraining-data', default=DEFAULT_PRETRAINING,
                   help='model_meta.json external_pretraining_data (declare ImageNet-1K MAE / COCO ViTPose members here)')
    p.add_argument('--output-dir', type=Path, required=True)
    a = p.parse_args()
    if set(a.cuda_graph_labels) - set(a.labels):
        raise ValueError('--cuda-graph-labels must name packaged members')
    if len(a.weights) != len(a.labels) or len(a.weights) < 2:
        raise ValueError('need >= 2 weights with one label each')
    model_weights = a.model_weights or [1.0] * len(a.weights)
    if len(model_weights) != len(a.weights):
        raise ValueError('one fusion weight per model')
    if a.fallback_weights is not None and (len(a.fallback_weights) != len(a.weights) or not a.probe_labels):
        raise ValueError('--fallback-weights needs one weight per model and --probe-labels')
    member_views = resolve_member_views(a.views, a.member_views, len(a.labels))
    mixed_views = len(set(member_views)) > 1
    out = a.output_dir
    out.mkdir(parents=True, exist_ok=False)
    staging = out / 'staging'
    staging.mkdir()
    report = {'at': datetime.now(timezone.utc).isoformat(), 'candidate': a.candidate, 'source_sha256': single.sha256(Path(__file__)),
              'template_sha256': single.sha256(TEMPLATE), 'members': []}
    files, shas, parents = [], [], []
    for weights, label, weight in zip(a.weights, a.labels, model_weights):
        member_dir = out / f'export_{label}'
        member_dir.mkdir()
        model_type = single.model_type_of(weights)
        exported = single.export(weights, member_dir, a.store_half_all or (a.store_half and model_type == 'vit'))
        name = f'model_{label}.torchscript'
        shutil.move(str(member_dir / 'model_pose.torchscript'), staging / name)
        (member_dir / 'export_report.json').write_text(json.dumps(exported, ensure_ascii=False, indent=1) + '\n')
        architecture = architecture_of(exported)
        files.append(name)
        shas.append(exported['model_sha256'])
        parents.append({'role': label, 'model_file': name, 'model_sha256': exported['model_sha256'], 'architecture': architecture,
                        'weight': weight, 'source_weights': single.rel(weights), 'epoch': exported.get('epoch'),
                        'manifest_path': single.rel(member_dir / 'export_report.json')})
        report['members'].append({'label': label, 'export': exported})
    (staging / 'run.py').write_text(render_run(fusion_constants(a, files, shas, model_weights, member_views)))
    for name in ('LICENSE_MMPose.txt', 'metadata.yaml'):
        shutil.copy2(single.REFERENCE / name, staging / name)
    for probe in a.diagnostic_probes:   # diagnostic-only files (not fused, not in parent_models)
        if probe.name in files or not probe.name.endswith('.torchscript'):
            raise ValueError(f'bad diagnostic probe {probe}')
        shutil.copy2(probe, staging / probe.name)
        report.setdefault('diagnostic_probes', []).append({'file': probe.name, 'source': single.rel(probe), 'sha256': single.sha256(probe)})
    spec = importlib.util.spec_from_file_location('staged_run', staging / 'run.py')
    staged = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(staged)
    pretraining = a.external_pretraining_data
    if any(p['architecture'] == 'IN1K_COCOPose_HRFormer_B' for p in parents):
        pretraining += '; HRFormer-B: ImageNet-1K classification initialization -> COCO train2017 pose; official MMPose checkpoint ecf0758d7d482b40f71ab909c11c63d94c2c8045de53300ad915796656001152'
    meta = {'schema_version': 1, 'model_type': a.candidate, 'architecture': '+'.join(p['architecture'] for p in parents),
            'input_width': 256, 'input_height': 256, 'heatmap_width': 64, 'heatmap_height': 64, 'num_keypoints': 17,
            'padding_factor': 1.25, 'image_mean': [0.5, 0.5, 0.5], 'image_std': [0.25, 0.25, 0.25],
            'score_excludes_keypoint_indices': [3, 4], 'external_weights': True, 'external_training_data': True,
            'external_pretraining_data': pretraining,
            'external_finetune_data': False, 'hidden_data_used': False, 'inference': staged.expected_inference(),
            'parent_models': parents, 'fit_scope': a.fit_scope, 'submission_fit_scope': a.fit_scope,
            'selection_evidence': {'note': a.selection_note, 'fused_predictions': single.rel(a.fused_predictions), 'views': a.views,
                                   **({'member_views': dict(zip(a.labels, member_views))} if mixed_views else {})},
            'ensemble_training_configs': [json.loads(c.read_text()) for c in a.training_configs], 'submission_authority': a.authority,
            'deployment_status': 'EXACT_STOCK_SUBMISSION_VALIDATION_PENDING'}
    (staging / 'model_meta.json').write_text(json.dumps(meta, ensure_ascii=False, indent=2) + '\n')
    members = ('LICENSE_MMPose.txt', 'metadata.yaml', 'model_meta.json', *files, *(p.name for p in a.diagnostic_probes), 'run.py')
    archive = out / f'{a.candidate}.zip'
    with zipfile.ZipFile(archive, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for name in members:
            info = zipfile.ZipInfo(name, date_time=(2026, 10, 6, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            zf.writestr(info, (staging / name).read_bytes())
    report['archive'] = {'path': single.rel(archive), 'sha256': single.sha256(archive), 'bytes': archive.stat().st_size,
                         'members': {n: {'sha256': single.sha256(staging / n), 'bytes': (staging / n).stat().st_size} for n in members},
                         'views': a.views, 'member_views': dict(zip(a.labels, member_views)), 'model_weights': model_weights}
    validation = subprocess.run([sys.executable, 'scripts/codabench.py', 'validate', str(archive)], cwd=ROOT, capture_output=True, text=True)
    report['archive_validation'] = {'returncode': validation.returncode, 'stdout_tail': validation.stdout[-2000:], 'stderr_tail': validation.stderr[-2000:]}
    if validation.returncode != 0:
        (out / 'package_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=1) + '\n')
        raise SystemExit('archive validation failed')
    env = {**os.environ, 'CUDA_VISIBLE_DEVICES': '', 'PYTHONPATH': 'src:scripts'}
    smoke_run = subprocess.run([str(ROOT / '.venv/bin/python'), '-B', __file__, '--smoke-only', str(archive), str(a.fused_predictions)],
                               cwd=ROOT, capture_output=True, text=True, env=env)
    if smoke_run.returncode != 0:
        report['cpu_smoke'] = {'returncode': smoke_run.returncode, 'stderr_tail': smoke_run.stderr[-3000:]}
        (out / 'package_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=1) + '\n')
        raise SystemExit('torch2.4.1 extracted-ZIP smoke failed')
    report['cpu_smoke'] = json.loads(smoke_run.stdout.strip().splitlines()[-1])
    (out / 'package_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=1) + '\n')
    print(json.dumps({k: report[k] for k in ('archive', 'cpu_smoke')}, ensure_ascii=False)[:2000])
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
