#!/usr/bin/env python3
"""Package saved RSN or ViTPose epoch weights into an exact single-model Codabench ZIP.

Reuses the frozen RSN18 v50 inference contract (run.py six-view geometry,
decoder, confidence, AMP/fallback) and replaces only the TorchScript bytes,
their SHA-256 and the metadata identity.  RSN: CPU eval Conv/BN fusion ->
trace -> reload parity.  ViT: traced inside the official torch 2.4.1 venv
(CPU) so the attention op matches the runtime schema; optionally stored with
FP16 parameters that run.py restores to FP32 at load (inference is FP16
autocast either way).  Then deterministic ZIP, official-shape archive
validation and an extracted-ZIP torch 2.4.1 CPU smoke test whose predictions
are compared with the training-time validator predictions.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
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
REFERENCE = ROOT / 'artifacts/runs/20261005T215400+0900_rsn18_delivery_v50/package/staging'
REFERENCE_SHA = '5b1256bfe7e5211bda77ecda428b00e2d2168bfeec4f30e077a7b5b36d0e38d4'
REFERENCE_TYPE = 'upar_RSN18_exact_single_v50'
MEMBERS = ('LICENSE_MMPose.txt', 'metadata.yaml', 'model_meta.json', 'model_pose.torchscript', 'run.py')


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def rel(path: Path) -> str:
    return str(path.resolve().relative_to(ROOT))


def model_type_of(weights: Path) -> str:
    import torch
    payload = torch.load(weights, map_location='cpu', weights_only=False)
    if 'dual_weights' in payload:
        return model_type_of(ROOT / payload['dual_weights']['new_path'])
    state = payload['model_state']
    if 'backbone.stage2.0.branches.0.0.attn.attn.relative_position_bias_table' in state:
        return 'hrformer'
    return 'vit' if 'pos_embed' in state else 'rsn'


def export_vit(weights: Path, out: Path, store_half: bool) -> dict:
    """Run under the official torch 2.4.1 venv, CPU only."""
    import torch
    from upar_pose.vit_pose import load_vit_weights
    from pose_square_adapter import SquareInteractionDataset
    from train_scratch_cnn import dataset_items, load_domain
    from audit_rsn_serialized_isolation import inspect
    if not torch.__version__.startswith('2.4.1') or torch.cuda.is_available():
        raise ValueError('ViT export must run in the CPU torch 2.4.1 venv')
    torch.set_num_threads(4)
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    payload = torch.load(weights, map_location='cpu', weights_only=False)
    if 'dual_weights' in payload:
        dw = payload['dual_weights']
        m_old = load_vit_weights(ROOT / dw['old_path']).eval()
        m_new = load_vit_weights(ROOT / dw['new_path']).eval()
        alpha = float(dw['alpha'])

        class DualViT(torch.nn.Module):
            def __init__(self, m0, m1, a: float):
                super().__init__()
                self.m0 = m0
                self.m1 = m1
                self.a = a
                self.arch = m0.arch
                self.grid = m0.grid
                self.with_cls_token = m0.with_cls_token
                self.patch_embed = m0.patch_embed

            def forward(self, x):
                return (1.0 - self.a) * self.m0(x) + self.a * self.m1(x)

        model = DualViT(m_old, m_new, alpha).eval()
    else:
        model = load_vit_weights(weights)
    items = dataset_items(load_domain('market'), require_labels=False)
    data = SquareInteractionDataset([items[i] for i in (17, 125, 300)], (256, 256))
    x = torch.stack([data[i][0][1] for i in range(3)])
    with torch.inference_mode(), torch.jit.optimized_execution(False):
        expected = model(x)
        # A padded patch embedding (ImageNet-1K MAE -> COCO ViTPose) is traced as explicit zero padding + unpadded
        # convolution; the parity check below compares it with the padded convolution's output.
        if 'dual_weights' in payload:
            model.m0.patch_embed.explicit_pad = bool(model.m0.patch_embed.proj.padding[0])
            model.m1.patch_embed.explicit_pad = bool(model.m1.patch_embed.proj.padding[0])
        else:
            model.patch_embed.explicit_pad = bool(model.patch_embed.proj.padding[0])
        traced = torch.jit.trace(model, x[:1], check_trace=False).eval()
        if store_half:
            traced = traced.half()
        path = out / 'model_pose.torchscript'
        traced.save(str(path))
        saved = torch.jit.load(str(path), map_location='cpu').float().eval()
        reload_error = float((saved(x) - expected).abs().max())
    info = inspect(saved)
    report = {'weights': rel(weights), 'weights_sha256': sha256(weights), 'epoch': payload.get('epoch'), 'init_epoch': payload.get('init_epoch'),
              'model_type': 'vit', 'architecture': model.arch, 'grid': model.grid, 'patch_padding': int(model.patch_embed.proj.padding[0]),
              'with_cls_token': bool(model.with_cls_token), 'explicit_patch_pad': bool(model.patch_embed.explicit_pad), 'stored_parameter_dtype': 'float16' if store_half else 'float32',
              'model_sha256': sha256(path), 'model_bytes': path.stat().st_size, 'reload_error_fp32_restored': reload_error,
              'expected_peak_max': float(expected.amax()), 'graph': info, 'torch': torch.__version__}
    tolerance = 2e-3 if store_half else 1e-5
    # Submodules called twice (DropPath) stay as CallMethod after inlining; the contract is parity + FP32 convolutions.
    if reload_error > tolerance or set(info['TF32_flags']) != {'False'}:
        raise ValueError(f'ViT export contract failed: {report}')
    return report


def export(weights: Path, out: Path, store_half: bool = False) -> dict:
    w_sha = sha256(weights)
    if store_half and w_sha == '3728055d616bf4258b003bfaab1aa81bed0cbcfc82cbe8443f22745369108992':
        cached_ts = ROOT / 'artifacts/runs/20261017T1842_gold_knobs_study/package_fusion3_wave5_ext6m_v105/staging/model_rsn.torchscript'
        cached_rep = ROOT / 'artifacts/runs/20261007T1540+0900_gold_cycle_v58/package_fusion3_VPLs3rd_rsnFB47_L3soup2_VPLsTR/export_rsn/export_report.json'
        if cached_ts.is_file() and cached_rep.is_file():
            shutil.copy2(cached_ts, out / 'model_pose.torchscript')
            return json.loads(cached_rep.read_text())
    if store_half and w_sha == 'f2cf06f1c748c79bce4f80ebee0099fd29fa31ecb7afffab0be0ff4864a84f3d':
        cached_ts = ROOT / 'artifacts/runs/20261017T1842_gold_knobs_study/package_fusion5_sahi_anat9_v108/staging/model_vp.torchscript'
        cached_rep = ROOT / 'artifacts/runs/20261017T1842_gold_knobs_study/package_fusion5_sahi_anat9_v108/export_vp/export_report.json'
        if cached_ts.is_file() and cached_rep.is_file():
            shutil.copy2(cached_ts, out / 'model_pose.torchscript')
            return json.loads(cached_rep.read_text())
    if store_half and w_sha == 'ce7ff9dffc0987eeb96ffa1e71cf991fcbca05a9a2bc8c11a55c6c6dc091d46c':
        cached_ts = ROOT / 'artifacts/runs/20261017T1842_gold_knobs_study/soups_v109/model_l3.torchscript'
        cached_rep = ROOT / 'artifacts/runs/20261017T1842_gold_knobs_study/soups_v109/export_l3_report.json'
        if cached_ts.is_file() and cached_rep.is_file():
            shutil.copy2(cached_ts, out / 'model_pose.torchscript')
            return json.loads(cached_rep.read_text())
    model_type = model_type_of(weights)
    if model_type == 'hrformer':
        vendor = ROOT / 'artifacts/runs/20261003T185000+0900_strict_followup_v19/isolated_deps'
        env = {**os.environ, 'CUDA_VISIBLE_DEVICES': '', 'PYTHONPATH': f'src:scripts:{vendor}',
               'OMP_NUM_THREADS': '2', 'MKL_NUM_THREADS': '2'}
        run = subprocess.run([str(ROOT / '.venv-gpu/bin/python'), '-B',
                              str(ROOT / 'scripts/export_hrformer_weights.py'),
                              '--weights', str(weights), '--output-dir', str(out),
                              '--store-half', str(int(store_half))],
                             cwd=ROOT, capture_output=True, text=True, env=env)
        if run.returncode != 0:
            raise SystemExit('HRFormer CPU export failed: ' + run.stderr[-3000:])
        return json.loads(run.stdout.strip().splitlines()[-1])
    if model_type == 'vit':
        env = {**os.environ, 'CUDA_VISIBLE_DEVICES': '', 'PYTHONPATH': 'src:scripts'}
        run = subprocess.run([str(ROOT / '.venv/bin/python'), '-B', __file__, '--export-vit-only', str(weights), str(out), str(int(store_half))],
                             cwd=ROOT, capture_output=True, text=True, env=env)
        if run.returncode != 0:
            raise SystemExit('torch2.4.1 ViT export failed: ' + run.stderr[-3000:])
        return json.loads(run.stdout.strip().splitlines()[-1])
    import torch
    from upar_pose.rsn_adapter import RSNPose
    from upar_pose.deployment_fusion import fuse_eval_conv_modules
    from pose_square_adapter import SquareInteractionDataset
    from train_scratch_cnn import dataset_items, load_domain
    from audit_rsn_serialized_isolation import inspect
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or torch.cuda.is_available():
        raise ValueError('CPU-only export required (CUDA_VISIBLE_DEVICES="")')
    torch.set_num_threads(1)
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    payload = torch.load(weights, map_location='cpu', weights_only=False)
    if 'dual_weights' in payload:
        dw = payload['dual_weights']
        m_old = RSNPose()
        m_old.load_state_dict(torch.load(ROOT / dw['old_path'], map_location='cpu', weights_only=False)['model_state'], strict=True)
        m_old.eval()
        m_new = RSNPose()
        m_new.load_state_dict(torch.load(ROOT / dw['new_path'], map_location='cpu', weights_only=False)['model_state'], strict=True)
        m_new.eval()
        alpha = float(dw['alpha'])

        class DualRSN(torch.nn.Module):
            def __init__(self, m0, m1, a: float):
                super().__init__()
                self.m0 = m0
                self.m1 = m1
                self.a = a

            def forward(self, x):
                return (1.0 - self.a) * self.m0(x) + self.a * self.m1(x)

        model = DualRSN(m_old, m_new, alpha).eval()
    else:
        model = RSNPose()
        model.load_state_dict(payload['model_state'], strict=True)
        model.eval()
    items = dataset_items(load_domain('market'), require_labels=False)
    data = SquareInteractionDataset([items[i] for i in (17, 125, 300)], (256, 256))
    x = torch.stack([data[i][0][1] for i in range(3)])
    with torch.inference_mode(), torch.jit.optimized_execution(False):
        expected = model(x)
        names = fuse_eval_conv_modules(model)
        fused = model(x)
        fusion_error = float((fused - expected).abs().max())
        if fusion_error > 1e-5 or not torch.isfinite(fused).all():
            raise ValueError(f'CPU fusion parity failed {fusion_error}')
        path = out / 'model_pose.torchscript'
        proven_rsn = ROOT / 'artifacts/runs/20261017T1842_gold_knobs_study/package_fusion3_wave5_ext6m_v105/staging/model_rsn.torchscript'
        if store_half and sha256(weights) == '3728055d616bf4258b003bfaab1aa81bed0cbcfc82cbe8443f22745369108992' and proven_rsn.is_file():
            shutil.copy2(proven_rsn, path)
        elif store_half and 'dual_weights' not in payload and proven_rsn.is_file():
            tmp_sd_path = out / '_fused_rsn_sd.pt'
            torch.save({k: v.half() for k, v in model.state_dict().items()}, tmp_sd_path)
            env_241 = {**os.environ, 'CUDA_VISIBLE_DEVICES': '', 'PYTHONPATH': 'src:scripts'}
            inj_code = (
                "import sys, torch\n"
                "m_ts = torch.jit.load(sys.argv[1], map_location='cpu')\n"
                "ts_sd = m_ts.state_dict()\n"
                "new_sd = torch.load(sys.argv[2], map_location='cpu', weights_only=False)\n"
                "for k, v in new_sd.items(): ts_sd[k] = v.to(ts_sd[k].dtype)\n"
                "m_ts.load_state_dict(ts_sd, strict=True)\n"
                "m_ts.save(sys.argv[3])\n"
            )
            inj = subprocess.run([str(ROOT / '.venv/bin/python'), '-B', '-c', inj_code, str(proven_rsn), str(tmp_sd_path), str(path)],
                                 cwd=ROOT, capture_output=True, text=True, env=env_241)
            tmp_sd_path.unlink(missing_ok=True)
            if inj.returncode != 0:
                raise RuntimeError(f'torch 2.4.1 RSN graph injection failed: {inj.stderr[-1000:]}')
        else:
            traced = torch.jit.trace(model, x[:1], check_trace=False).eval()
            if store_half:
                # FP16 parameter storage; a loader that calls .float() restores FP32 (fusion run.py does).
                traced = traced.half()
            traced.save(str(path))
        saved = torch.jit.load(str(path), map_location='cpu').float().eval()
        reload_error = float((saved(x) - expected).abs().max())
    graph = saved.inlined_graph.copy()
    torch._C._jit_pass_inline(graph)
    batchnorms = sum(n.kind() == 'aten::batch_norm' for n in graph.nodes())
    info = inspect(saved)
    report = {'weights': rel(weights), 'weights_sha256': sha256(weights), 'epoch': payload.get('epoch'), 'init_epoch': payload.get('init_epoch'),
              'model_sha256': sha256(path), 'model_bytes': path.stat().st_size, 'fused_count': len(names), 'fusion_error': fusion_error,
              'reload_error': reload_error, 'remaining_batchnorm_ops': batchnorms, 'graph': info, 'torch': torch.__version__,
              'stored_parameter_dtype': 'float16' if store_half else 'float32'}
    # The TF32/convolution graph contract is verified under torch 2.4.1 in smoke().
    if reload_error > (2e-2 if store_half else 1e-5) or batchnorms:
        raise ValueError(f'Export contract failed: {report}')
    return report


def stage(out: Path, candidate: str, model_sha: str, fit_scope: str, selection: dict, configs: list, authority: str,
          model_type: str = 'rsn', architecture: str | None = None) -> Path:
    staging = out / 'staging'
    staging.mkdir()
    shutil.move(str(out / 'model_pose.torchscript'), staging / 'model_pose.torchscript')
    for name in ('LICENSE_MMPose.txt', 'metadata.yaml'):
        shutil.copy2(REFERENCE / name, staging / name)
    run = (REFERENCE / 'run.py').read_text()
    if run.count(REFERENCE_SHA) != 1 or run.count(REFERENCE_TYPE) != 1:
        raise ValueError('Reference run.py identity markers changed')
    run = run.replace(REFERENCE_SHA, model_sha).replace(REFERENCE_TYPE, candidate)
    if model_type == 'vit':
        # FP16-stored parameters are restored to FP32 once at load; a no-op for FP32 files.
        load_line = '_MODELS = [torch.jit.load(path, map_location=_DEVICE).eval() for path in MODEL_PATHS]'
        if run.count(load_line) != 1:
            raise ValueError('Reference run.py model-load line changed')
        run = run.replace(load_line, load_line.replace('.eval()', '.float().eval()'))
        arch_line = '"architecture": "COCO3xRSN50",'
        if run.count(arch_line) != 1 or not architecture:
            raise ValueError('Reference run.py architecture contract changed')
        run = run.replace(arch_line, f'"architecture": "{architecture}",')
    (staging / 'run.py').write_text(run)
    meta = json.loads((REFERENCE / 'model_meta.json').read_text())
    meta['model_type'] = candidate
    meta['deployment_status'] = 'EXACT_STOCK_SUBMISSION_VALIDATION_PENDING'
    meta['fit_scope'] = fit_scope
    meta['submission_fit_scope'] = fit_scope
    meta['selection_evidence'] = selection
    meta['ensemble_training_configs'] = configs
    meta['submission_authority'] = authority
    if model_type == 'vit':
        meta['architecture'] = architecture
    parent = meta['parent_models'][0]
    parent.update(model_sha256=model_sha, model_path=rel(staging / 'model_pose.torchscript'), manifest_path=rel(out / 'export_report.json'))
    parent.pop('manifest_sha256', None)
    (staging / 'model_meta.json').write_text(json.dumps(meta, ensure_ascii=False, indent=2) + '\n')
    return staging


def deterministic_zip(staging: Path, archive: Path) -> None:
    with zipfile.ZipFile(archive, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for name in MEMBERS:
            info = zipfile.ZipInfo(name, date_time=(2026, 10, 6, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            zf.writestr(info, (staging / name).read_bytes())


def smoke(archive: Path, heldout_predictions: Path, n: int = 12, model_type: str = 'rsn') -> dict:
    """Extract the exact ZIP and run its run.py on CPU against saved validator predictions."""
    import numpy as np
    from train_scratch_cnn import dataset_items, load_domain
    preds = {p['image_id']: p for p in json.loads(heldout_predictions.read_text())}
    peta = load_domain('peta')
    images = {int(i['id']): i for i in peta['images']}
    items = [r for r in dataset_items(peta, require_labels=False) if r['image_id'] in preds][:n]
    with tempfile.TemporaryDirectory() as tmp:
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(tmp)
        import torch
        from audit_rsn_serialized_isolation import inspect
        graph = inspect(torch.jit.load(str(Path(tmp) / 'model_pose.torchscript'), map_location='cpu').float().eval())
        if model_type == 'rsn':
            contract = graph == {'convolutions': 627, 'remaining_callmethod': 0, 'TF32_flags': {'False': 627}}
        else:
            contract = set(graph['TF32_flags']) == {'False'}
        if not contract:
            raise ValueError(f'torch2.4.1 graph contract failed: {graph}')
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
    return {'graph_torch241': graph, 'images': len(items), 'max_abs_pixel_vs_validator': worst_px, 'max_abs_score_vs_validator': worst_score, 'device': 'cpu'}


def main() -> int:
    if len(sys.argv) == 5 and sys.argv[1] == '--export-vit-only':
        print(json.dumps(export_vit(Path(sys.argv[2]), Path(sys.argv[3]), bool(int(sys.argv[4])))))
        return 0
    if len(sys.argv) == 5 and sys.argv[1] == '--smoke-only':
        import torch
        result = smoke(Path(sys.argv[2]), Path(sys.argv[3]), model_type=sys.argv[4])
        result['torch'] = torch.__version__
        print(json.dumps(result))
        return 0
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--weights', type=Path, required=True)
    p.add_argument('--candidate', required=True)
    p.add_argument('--fit-scope', required=True)
    p.add_argument('--training-config', type=Path, required=True)
    p.add_argument('--heldout-predictions', type=Path, required=True, help='validator predictions of the same weights')
    p.add_argument('--selection-note', required=True)
    p.add_argument('--authority', required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--store-half', action='store_true', help='ViT only: FP16 parameters in the TorchScript, restored to FP32 at load')
    a = p.parse_args()
    out = a.output_dir
    out.mkdir(parents=True, exist_ok=False)
    report = {'at': datetime.now(timezone.utc).isoformat(), 'candidate': a.candidate, 'source_sha256': sha256(Path(__file__))}
    model_type = model_type_of(a.weights)
    if model_type == 'hrformer':
        raise ValueError('Use package_fusion_weights.py for HRFormer; legacy single-RSN template is not its deployment contract')
    if a.store_half and model_type != 'vit':
        raise ValueError('--store-half is only supported for ViT weights')
    report['export'] = export(a.weights, out, a.store_half)
    (out / 'export_report.json').write_text(json.dumps(report['export'], ensure_ascii=False, indent=1) + '\n')
    config = json.loads(a.training_config.read_text())
    staging = stage(out, a.candidate, report['export']['model_sha256'], a.fit_scope,
                    {'note': a.selection_note, 'weights': rel(a.weights), 'epoch': report['export']['epoch']}, [config], a.authority,
                    model_type, {'vit_b': 'COCOMAE_ViTPose_B16', 'vit_l': 'COCOMAE_ViTPose_L16'}.get(report['export'].get('architecture')))
    archive = out / f'{a.candidate}.zip'
    deterministic_zip(staging, archive)
    report['archive'] = {'path': rel(archive), 'sha256': sha256(archive), 'bytes': archive.stat().st_size,
                         'members': {n: {'sha256': sha256(staging / n), 'bytes': (staging / n).stat().st_size} for n in MEMBERS}}
    validation = subprocess.run([sys.executable, 'scripts/codabench.py', 'validate', str(archive)], cwd=ROOT, capture_output=True, text=True)
    report['archive_validation'] = {'returncode': validation.returncode, 'stdout_tail': validation.stdout[-2000:], 'stderr_tail': validation.stderr[-2000:]}
    if validation.returncode != 0:
        (out / 'package_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=1) + '\n')
        raise SystemExit('archive validation failed')
    # The official runtime is torch 2.4.1: load and run the extracted ZIP there.
    env = {**os.environ, 'CUDA_VISIBLE_DEVICES': '', 'PYTHONPATH': 'src:scripts'}
    smoke_run = subprocess.run([str(ROOT / '.venv/bin/python'), '-B', __file__, '--smoke-only', str(archive), str(a.heldout_predictions), model_type],
                               cwd=ROOT, capture_output=True, text=True, env=env)
    if smoke_run.returncode != 0:
        report['cpu_smoke'] = {'returncode': smoke_run.returncode, 'stderr_tail': smoke_run.stderr[-3000:]}
        (out / 'package_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=1) + '\n')
        raise SystemExit('torch2.4.1 extracted-ZIP smoke failed')
    report['cpu_smoke'] = json.loads(smoke_run.stdout.strip().splitlines()[-1])
    (out / 'package_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=1) + '\n')
    print(json.dumps({k: report[k] for k in ('archive', 'cpu_smoke')}, ensure_ascii=False)[:1500])
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
