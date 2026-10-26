#!/usr/bin/env python3
"""Finite COCO-only RSN fine-tuning; shared data, validator and stop policy."""
import argparse
from datetime import datetime, timezone
import gc
import json
from pathlib import Path
import shutil
import sys
import time
import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'scripts')]
from continue_pose_research import validate_model, save_checkpoint, relative
from train_pose_research import export
from train_scratch_hrnet_v4 import seed_everything
from train_scratch_cnn import load_domain, dataset_items
from pose_factor_data import FactorDataset
from upar_pose.rsn_adapter import RSNPose, rsn_loss
from upar_pose.continuation_policy import ValidationPlateau
from upar_pose.training_recovery import training_checkpoint
from upar_pose.pipeline import load_json, sha256_file, write_json_atomic


def run(path, audit=False):
    c = load_json(path)
    for ref in c['frozen_evidence']:
        if sha256_file(ROOT/ref['path']) != ref['sha256']:
            raise ValueError('Frozen RSN dependency changed: '+ref['path'])
    if c['train_domains'] != ['market', 'pa100k'] or c['fit_scope'] != 'holdout_peta':
        raise ValueError('Invalid fit boundary')
    if c['input_size'] != [256,256] or c['workers'] != 0 or c['precision'] != 'bf16':
        raise ValueError('Unsupported tensor/resource recipe')
    out = ROOT/c['output_dir']; out = out.with_name(out.name+'_audit') if audit else out
    out.mkdir(parents=True, exist_ok=False)
    if not torch.cuda.is_available(): raise RuntimeError('Real CUDA required')
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.cuda.set_per_process_memory_fraction(c['allocator_cap_gib']*1024**3/torch.cuda.get_device_properties(0).total_memory)
    seed_everything(c['seed'])
    model = RSNPose()
    initialization = model.initialize(ROOT/c['checkpoint'], expected_sha256=c['checkpoint_sha256'])
    write_json_atomic(out/'initialization.json', initialization)
    model.cuda()
    items = [r for domain in c['train_domains'] for r in dataset_items(load_domain(domain), require_labels=True)]
    if len(items) != 32422: raise ValueError('Changed training population')
    seed_everything(c['data_seed'])
    dataset = FactorDataset(items, c, settings=c['factor_synthesis'], weights=np.ones(5), output=out)
    loader = DataLoader(dataset, batch_size=c['batch_size'], shuffle=True, num_workers=0, pin_memory=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=c['learning_rate'], weight_decay=c['weight_decay'])
    controller = ValidationPlateau(0., 0., **c['early_stopping'])
    scaler = torch.amp.GradScaler('cuda', enabled=False)
    write_json_atomic(out/'config.json', {**c, 'config_sha256':sha256_file(path), 'train_examples':len(items),
        'audit_only':audit, 'parameter_count':sum(p.numel() for p in model.parameters())})
    probe = next(model.parameters()).detach().flatten()[:256].clone()
    history, validations, best_epoch = [], [], 0
    lr, started = c['learning_rate'], time.monotonic()
    reason = 'epoch_cap_reached_not_convergence_proof'
    torch.cuda.reset_peak_memory_stats()
    for epoch in range(1, (1 if audit else c['max_epochs'])+1):
        if shutil.disk_usage(ROOT).free < c['disk_reserve_gib']*1024**3:
            reason = 'disk_limit_last_checkpoint_preserved_not_convergence'; break
        for group in optimizer.param_groups: group['lr'] = lr*min(1., epoch/c['warmup_epochs'])
        model.train(); seen = 0; totals = {}; tick = time.monotonic()
        for bi, batch in enumerate(loader):
            inputs, targets, weights, coords, visibility = [x.cuda(non_blocking=True) for x in batch[:5]]
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                output, stages, aux = model.forward_training(inputs)
                loss, terms = rsn_loss(model, output, stages, aux, targets, weights, coords, visibility)
            if not torch.isfinite(loss): raise FloatingPointError('Nonfinite RSN loss; no retry')
            loss.backward()
            grads = [torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None]
            if not grads or not torch.stack(grads).all(): raise FloatingPointError('Nonfinite RSN gradient')
            optimizer.step(); seen += len(inputs)
            for key, value in {'loss':loss, **terms}.items():
                totals[key] = totals.get(key, 0.)+float(value.detach())*len(inputs)
            if bi%100 == 0: print(json.dumps({'epoch':epoch,'batch':bi+1,'loss':float(loss.detach())}), flush=True)
            if audit and bi == 2: break
        torch.cuda.synchronize()
        history.append({'epoch':epoch, 'examples':seen, 'train_loss':totals['loss']/seen,
            'losses':{k:v/seen for k,v in totals.items()}, 'seconds':time.monotonic()-tick,
            'learning_rate':optimizer.param_groups[0]['lr'], 'observed_at':datetime.now(timezone.utc).isoformat()})
        write_json_atomic(out/'training_history.json', history)
        if audit:
            update = float((next(model.parameters()).detach().flatten()[:256]-probe).abs().max())
            if update <= 0: raise ValueError('No real RSN backbone update')
            write_json_atomic(out/'audit.json', {'audit_only':True, 'batches':3, 'examples':seen,
                'batch_size':c['batch_size'], 'parameter_max_update':update, 'optimizer_updates_before':0,
                'optimizer_updates_after':max(float(v['step']) for v in optimizer.state.values() if 'step' in v),
                'peak_cuda_reserved_bytes':torch.cuda.max_memory_reserved(), 'finite':True})
            return
        del inputs, targets, weights, coords, visibility, output, stages, aux, loss
        action = {'stop':False, 'save_best':False}
        if epoch%c['evaluate_every'] == 0 or epoch == c['max_epochs']:
            val = validate_model(model, out/f'epoch_{epoch:03d}_predictions.json', batch_size=c['eval_batch_size'])
            if epoch >= c['warmup_epochs']:
                action = controller.observe(epoch, val['AP'], lr); lr = action['lr']
            else:
                action['save_best'] = val['AP'] > controller.best_ap
                controller.best_ap = max(controller.best_ap, val['AP'])
                controller.reference_ap = max(controller.reference_ap, val['AP'])
            val.update(epoch=epoch, action=action, policy=controller.state_dict()); validations.append(val)
            write_json_atomic(out/'validation_history.json', validations)
            if action['save_best']:
                best_epoch = epoch
                save_checkpoint(out/'best_weights.pt', {'epoch':epoch, 'AP':val['AP'], 'model_state':model.state_dict()})
                write_json_atomic(out/'best_selection.json', {'epoch':epoch, 'AP':val['AP'],
                    'predictions':val['predictions'], 'weights_sha256':sha256_file(out/'best_weights.pt')})
            print(json.dumps(val), flush=True)
        checkpoint = training_checkpoint(epoch, model, optimizer, controller, scaler)
        checkpoint.update(dataset_synthetic_rng_state=dataset.synthetic_rng.bit_generator.state,
            config_sha256=sha256_file(path), best_epoch=best_epoch, next_learning_rate=lr,
            history=history, validations=validations)
        save_checkpoint(out/'last_checkpoint.pt', checkpoint); del checkpoint
        if action['stop']:
            reason = 'validation_plateau_after_lr_reductions'; break
    del optimizer, loader, dataset; gc.collect()
    if not best_epoch: raise RuntimeError('No complete evaluated checkpoint; outputs retained, not complete')
    selected = torch.load(out/'best_weights.pt', map_location='cpu', weights_only=True)
    model.load_state_dict(selected['model_state'], strict=True); del selected
    error = export(model, out); del model; torch.cuda.empty_cache()
    selection = load_json(out/'best_selection.json')
    result = {'status':reason, 'completed_epochs':len(history), 'best_epoch':best_epoch, 'selected_AP':selection['AP'],
        'selected_model':relative(out/'model.torchscript'), 'selected_model_sha256':sha256_file(out/'model.torchscript'),
        'selected_predictions':selection['predictions'], 'selected_predictions_sha256':sha256_file(ROOT/selection['predictions']),
        'export_max_abs_error':error, 'elapsed_seconds':time.monotonic()-started,
        'limits':['Adaptive PETA development only; no untouched confirmation or Public guarantee.',
            'Fresh three-stage RSN with new loss/batch/schedule: recipe-level comparison, not capacity-only causality.',
            'CPU export parity checked; GPU deployment parity and full runtime still required before submission.']}
    write_json_atomic(out/'result.json', result)
    write_json_atomic(ROOT/c['event_path'], {'id':c['candidate']+'_fit_completed','candidate':c['candidate'],
        'kind':'diagnostic', 'candidate_status':'evaluating', 'summary':'RSN fit ended; selected development AP '+str(selection['AP']),
        'issue':'; '.join(result['limits']), 'next_action':'Pair saved predictions against fixed control, inspect errors/convergence and measure exact deployment parity/runtime.',
        'scope':{'evaluation_role':'development'}, 'visual':False, 'evidence':[relative(out/'result.json'),relative(out/'validation_history.json')],
        'artifacts':[relative(out/'result.json'),relative(out/'best_weights.pt'),relative(out/'last_checkpoint.pt')]})


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--config', type=Path, required=True); p.add_argument('--audit', action='store_true')
    a = p.parse_args(); run(a.config, a.audit)
