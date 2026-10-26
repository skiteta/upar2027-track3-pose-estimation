#!/usr/bin/env python3
"""One config-bound continuation phase with full PETA validation and best restore.

Reuse the frozen model/data/loss/decoder. Do not modify the running v27 source.
The completed parent's AdamW moments and RNG are carried forward, while its
finished OneCycle schedule is explicitly replaced by warmup + validation plateau.
"""
import argparse
from datetime import datetime, timezone
import gc
import json
from pathlib import Path
import random
import shutil
import sys
import time
import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]
from train_pose_research import verify_recipe, export
from train_scratch_hrnet_v4 import seed_everything
from pose_research_data import ResearchDataset
from pose_square_adapter import SquareInteractionDataset
from train_scratch_cnn import load_domain, dataset_items
from evaluate_resolution_ensemble import aligned_model_heatmaps, check_ids
from evaluate_seed_geometry_ensemble_v6 import PADDINGS, TARGET_PADDING, prediction
from compare_pose_predictions import compare
from upar_pose.hourglass_research import HourglassResearchPose, research_loss
from upar_pose.training_recovery import training_checkpoint
from upar_pose.continuation_policy import ValidationPlateau, assess_undertraining
from upar_pose.evaluation import evaluate_coco_keypoints
from upar_pose.flip_tta import FLIP_CHANNELS, decode_heatmaps_refined
from upar_pose.pipeline import load_json, sha256_file, write_json_atomic


def relative(path):
    return str(path.resolve().relative_to(ROOT))


def validate_sources(config):
    for ref in config['frozen_evidence']:
        if sha256_file(ROOT / ref['path']) != ref['sha256']:
            raise ValueError('Changed continuation source: ' + ref['path'])
    recipe_path = ROOT / config['parent_recipe']
    recipe = load_json(recipe_path)
    verify_recipe(recipe)
    parent = ROOT / recipe['arms']['coco']['output_dir']
    manifest = load_json(parent / 'manifest.json')
    fit = manifest['config']
    if manifest.get('audit_only') or fit['fit_scope'] != 'holdout_peta' or fit['train_domains'] != ['market', 'pa100k'] or fit['train_examples'] != 32422:
        raise ValueError('Invalid upstream fit boundary')
    if fit['recipe_config']['sha256'] != sha256_file(recipe_path) or len(manifest['history']) != recipe['epochs']:
        raise ValueError('Parent phase incomplete or recipe changed')
    for i, row in enumerate(manifest['history'], 1):
        if row['epoch'] != i or row['examples'] != 32422 or not np.isfinite(row['train_loss']):
            raise ValueError('Invalid parent curve')
    for name in ('last_checkpoint.pt', 'model.torchscript'):
        if sha256_file(parent / name) != manifest['artifacts'][name]['sha256']:
            raise ValueError('Parent artifact changed')
    evaluation = load_json(ROOT / recipe['evaluation_recipe'])
    binding_path = ROOT / evaluation['prediction_dir'] / 'model_binding.json'
    binding = load_json(binding_path)
    if (binding['manifest_sha256'] != sha256_file(parent / 'manifest.json') or
            binding['model_sha256'] != sha256_file(parent / 'model.torchscript') or
            binding['fit_recipe_sha256'] != sha256_file(recipe_path)):
        raise ValueError('Parent evaluation binding mismatch')
    predictions = ROOT / evaluation['prediction_dir'] / 'predictions.json'
    report = load_json(ROOT / evaluation['comparison']['output_dir'] / 'comparison.json')
    if sha256_file(predictions) != report['prediction_sha256']['other']:
        raise ValueError('Scored parent predictions changed')
    dataset = load_domain('peta')
    rows = load_json(predictions)
    expected = {(r['image_id'], r['annotation_id']) for r in dataset_items(dataset, require_labels=False)}
    check_ids(rows, expected)
    actual = evaluate_coco_keypoints(dataset, rows, max_dets=20)['AP']
    if abs(actual - report['metrics']['candidate']['AP']) > 1e-10:
        raise ValueError('Saved parent AP does not replay')
    return recipe, parent, manifest, evaluation, predictions, actual


def restore_rng(checkpoint, dataset):
    state = checkpoint['rng_state']
    random.setstate(state['python'])
    n = state['numpy']
    np.random.set_state((n[0], np.asarray(n[1], dtype=np.uint32), n[2], n[3], n[4]))
    torch.set_rng_state(state['torch'].cpu())
    torch.cuda.set_rng_state_all([x.cpu() for x in state['cuda']])
    dataset.synthetic_rng.bit_generator.state = checkpoint['dataset_synthetic_rng_state']


def save_checkpoint(path, value):
    tmp = path.with_suffix('.tmp')
    torch.save(value, tmp)
    tmp.replace(path)


def validate_model(model, output, *, batch_size=8):
    """Same six-view geometry, decoder, full population, metric as the parent."""
    dataset = load_domain('peta')
    items = dataset_items(dataset, require_labels=False)
    if len(items) != 10314:
        raise ValueError('PETA count changed')
    # DataLoader creates a base-seed even with workers=0. A private generator
    # prevents validation from advancing the training shuffle/augmentation RNG.
    generator = torch.Generator().manual_seed(20261004)
    loader = DataLoader(SquareInteractionDataset(items, (256, 256)), batch_size=batch_size,
                        num_workers=0, shuffle=False, pin_memory=True, generator=generator)
    rows, channels = [], torch.tensor(FLIP_CHANNELS, device='cuda')
    model.eval()
    started = time.monotonic()
    with torch.inference_mode():
        for inputs, geometries, image_ids, annotation_ids in loader:
            maps = aligned_model_heatmaps(model, inputs.cuda(), channels, (64, 64))
            geometry = [tuple(map(float, r)) for r in geometries[:, PADDINGS.index(TARGET_PADDING), :].numpy()]
            coords, maxima = decode_heatmaps_refined(maps.cpu().numpy(), geometry, refinement='quadratic')
            rows.extend(prediction(coords[i], maxima[i], int(image_ids[i]), int(annotation_ids[i])) for i in range(len(image_ids)))
    torch.cuda.synchronize()
    check_ids(rows, {(r['image_id'], r['annotation_id']) for r in items})
    write_json_atomic(output, rows)
    metric = evaluate_coco_keypoints(dataset, load_json(output), max_dets=20)
    return {'AP': metric['AP'], 'metrics': metric, 'seconds': time.monotonic() - started,
            'predictions': relative(output), 'sha256': sha256_file(output)}


def run(config_path, *, audit=False):
    config = load_json(config_path)
    recipe, parent, manifest, evaluation, parent_predictions, parent_ap = validate_sources(config)
    assessment = assess_undertraining(manifest['history'], has_new_modules=bool(recipe['error_heads'] or recipe['stacks'] > 1))
    out = ROOT / config['output_dir']
    if audit:
        out = out.with_name(out.name + '_audit')
    out.mkdir(parents=True, exist_ok=False)
    write_json_atomic(out / 'config.json', config)
    decision = {'parent_AP_replayed': parent_ap, 'assessment': assessment,
        'parent_manifest': relative(parent / 'manifest.json'), 'parent_manifest_sha256': sha256_file(parent / 'manifest.json'),
        'parent_checkpoint_sha256': sha256_file(parent / 'last_checkpoint.pt'),
        'note': 'One final AP cannot prove undertraining. This is a bounded optimization test, not a new independent holdout.'}
    write_json_atomic(out / 'parent_assessment.json', decision)
    if not assessment['continue_trial'] and not audit:
        write_json_atomic(out / 'result.json', {**decision, 'status': 'no_predeclared_trigger', 'training_started': False})
        return
    if not torch.cuda.is_available():
        raise RuntimeError('No silent CPU training fallback')
    torch.set_num_threads(1)
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    seed_everything(recipe['seed'])
    model = HourglassResearchPose(recipe['stacks'], recipe['error_heads'])
    payload = torch.load(parent / 'last_checkpoint.pt', map_location='cpu', weights_only=True)
    if payload['epoch'] != recipe['epochs']:
        raise ValueError('Checkpoint epoch mismatch')
    model.load_state_dict(payload['model_state'], strict=True)
    model.cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=recipe['weight_decay'])
    optimizer.load_state_dict(payload['optimizer_state'])
    items = [item for domain in recipe['train_domains'] for item in dataset_items(load_domain(domain), require_labels=True)]
    if len(items) != 32422:
        raise ValueError('Training boundary changed')
    dataset = ResearchDataset(items, recipe)
    restore_rng(payload, dataset)
    optimizer_updates_before = max(float(v['step']) for v in optimizer.state.values() if 'step' in v)
    del payload
    loader = DataLoader(dataset, batch_size=recipe['batch_size'], shuffle=True, num_workers=0, pin_memory=True)
    controller = ValidationPlateau(parent_ap, parent_ap, **config['early_stopping'])
    scaler = torch.amp.GradScaler('cuda', enabled=False)
    lr, best_epoch, best_predictions = config['learning_rate'], 0, parent_predictions
    history, validations, stop_reason = [], [], 'epoch_cap_reached_not_convergence_proof'
    start = time.monotonic()
    write_json_atomic(out / 'initialization.json', {**decision,
        'optimizer_moments_restored': True, 'optimizer_updates_before': optimizer_updates_before,
        'optimizer_betas': [list(g['betas']) for g in optimizer.param_groups],
        'training_rng_and_private_synthetic_rng_restored': True,
        'old_OneCycle_replaced_explicitly': True,
        'new_schedule': '3epoch linear warmup to configured LR then validation plateau, independent of maxepochs',
        'audit_only': audit})
    parameter_probe = next(model.parameters()).detach().flatten()[:256].clone()
    for epoch in range(1, (1 if audit else config['max_epochs']) + 1):
        # Guard at a recoverable boundary, allowing room for atomic last save.
        if shutil.disk_usage(ROOT).free < config['disk_reserve_gib'] * 1024**3:
            stop_reason = 'resource_disk_limit_checkpoint_preserved_not_convergence'
            break
        actual_lr = lr * min(1., epoch / config['warmup_epochs'])
        for group in optimizer.param_groups:
            group['lr'] = actual_lr
        model.train()
        totals, seen = {}, 0
        epoch_start = time.monotonic()
        for batch_index, batch in enumerate(loader):
            inputs, targets, weights, coords, visibility = [x.cuda(non_blocking=True) for x in batch[:5]]
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                output, stages, aux = model.forward_training(inputs)
                loss, terms = research_loss(model, output, stages, aux, targets, weights, coords, visibility)
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite loss; preserve last checkpoint; no retry')
            loss.backward()
            if not torch.stack([torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None]).all():
                raise FloatingPointError('Nonfinite gradient; no retry')
            optimizer.step()
            seen += len(inputs)
            for key, value in {'loss': loss, **terms}.items():
                totals[key] = totals.get(key, 0.) + float(value.detach()) * len(inputs)
            if batch_index % 100 == 0:
                print(json.dumps({'epoch': epoch, 'batch': batch_index + 1, 'loss': float(loss.detach())}), flush=True)
            if audit and batch_index == 2:
                break
        torch.cuda.synchronize()
        row = {'epoch': epoch, 'examples': seen, 'train_loss': totals['loss'] / seen,
               'losses': {k: v / seen for k, v in totals.items()}, 'learning_rate': actual_lr,
               'seconds': time.monotonic() - epoch_start, 'observed_at': datetime.now(timezone.utc).isoformat()}
        history.append(row)
        write_json_atomic(out / 'training_history.json', history)
        if audit:
            update = float((next(model.parameters()).detach().flatten()[:256] - parameter_probe).abs().max())
            if update == 0:
                raise ValueError('No measured backbone update')
            write_json_atomic(out / 'audit.json', {'audit_only': True, 'batches': 3, 'examples': seen,
                'parameter_max_update': update, 'optimizer_updates_before': optimizer_updates_before,
                'optimizer_updates_after': max(float(v['step']) for v in optimizer.state.values() if 'step' in v),
                'peak_cuda_reserved_bytes': torch.cuda.max_memory_reserved(), 'finite': True})
            print(load_json(out / 'audit.json'), flush=True)
            return
        del inputs, targets, weights, coords, visibility, output, stages, aux, loss
        action = {'stop': False, 'save_best': False}
        if epoch % config['evaluate_every'] == 0 or epoch == config['max_epochs']:
            val = validate_model(model, out / f'epoch_{epoch:03d}_predictions.json')
            # Warmup completes before any plateau/stop decision.
            if epoch >= config['warmup_epochs']:
                action = controller.observe(epoch, val['AP'], lr)
                lr = action['lr']
            else:
                action['save_best'] = val['AP'] > controller.best_ap
                controller.best_ap = max(controller.best_ap, val['AP'])
                controller.reference_ap = max(controller.reference_ap, val['AP'])
            val.update(epoch=epoch, action=action, policy=controller.state_dict())
            validations.append(val)
            write_json_atomic(out / 'validation_history.json', validations)
            if action['save_best']:
                best_epoch, best_predictions = epoch, ROOT / val['predictions']
                save_checkpoint(out / 'best_weights.pt', {'epoch': epoch, 'model_state': model.state_dict(), 'AP': val['AP']})
                write_json_atomic(out / 'best_selection.json', {'epoch': epoch, 'AP': val['AP'],
                    'predictions': relative(best_predictions), 'weights_sha256': sha256_file(out / 'best_weights.pt')})
            print(json.dumps(val), flush=True)
        checkpoint = training_checkpoint(epoch, model, optimizer, controller, scaler)
        checkpoint.update(dataset_synthetic_rng_state=dataset.synthetic_rng.bit_generator.state,
            continuation_config_sha256=sha256_file(config_path), best_epoch=best_epoch,
            next_learning_rate=lr, history=history, validations=validations)
        save_checkpoint(out / 'last_checkpoint.pt', checkpoint)
        del checkpoint
        if action['stop']:
            stop_reason = 'validation_plateau_after_lr_reductions'
            break
    # Parent is part of the selection set: if the phase never improves AP, use
    # the untouched parent, not a worse final/last checkpoint.
    del optimizer, loader, dataset
    gc.collect()
    if best_epoch:
        selected = torch.load(out / 'best_weights.pt', map_location='cpu', weights_only=True)
        model.load_state_dict(selected['model_state'], strict=True)
        del selected
        real_inputs = SquareInteractionDataset(dataset_items(load_domain('peta'), require_labels=False)[:1], (256, 256))[0][0]
        model.eval()
        with torch.inference_mode():
            expected_real = model(real_inputs.cuda()).float().cpu()
        error = export(model, out)
        selected_model = out / 'model.torchscript'
        saved = torch.jit.load(str(selected_model), map_location='cuda').eval()
        with torch.inference_mode():
            real_error = float((saved(real_inputs.cuda()).float().cpu() - expected_real).abs().max())
        if real_error > 2e-4:
            raise RuntimeError('Selected eager/export real-input parity failed')
        write_json_atomic(out / 'export_real_parity.json', {'max_abs_error': real_error, 'limit': 2e-4,
            'source': 'first PETA sample, all three geometry views', 'model_sha256': sha256_file(selected_model)})
        del saved
    else:
        error, selected_model = None, parent / 'model.torchscript'
    del model
    torch.cuda.empty_cache()
    comparison = {**evaluation['comparison'], 'reference': relative(parent_predictions),
        'other': relative(best_predictions), 'output_dir': relative(out / 'selected_comparison'),
        'reference_label': 'completed parent phase', 'candidate_label': 'validation-selected continuation',
        'title': 'Same model, extra optimization and explicit new schedule',
        'frozen_evidence': config['frozen_evidence'] + [
            {'path': relative(parent_predictions), 'sha256': sha256_file(parent_predictions)},
            {'path': relative(best_predictions), 'sha256': sha256_file(best_predictions)}]}
    report = compare(comparison)
    fit_audit = {'status': 'clean', 'pipeline': 'matched', 'target_mode': 'unseen_group',
        'note': 'Upstream and continuation fit Market+PA100K only. PETA repeatedly selects epochs/LR; development only.',
        'evidence': [relative(out / 'initialization.json'), relative(out / 'config.json')]}
    write_json_atomic(out / 'fit_audit.json', fit_audit)
    result = {'status': stop_reason, 'training_started': bool(history), 'completed_extra_epochs': len(history),
        'best_extra_epoch': best_epoch, 'selected_AP': controller.best_ap, 'parent_AP': parent_ap,
        'selected_model': relative(selected_model), 'selected_model_sha256': sha256_file(selected_model),
        'selected_predictions': relative(best_predictions), 'selected_predictions_sha256': sha256_file(best_predictions),
        'export_max_abs_error': error, 'elapsed_seconds': time.monotonic() - start,
        'limits': ['Adaptive PETA development, not independent confirmation.',
                   'Extra training + changed LR schedule, not pure epoch-count effect.',
                   'Best weights and last optimizer checkpoint can be different epochs.'],
        'config': config}
    write_json_atomic(out / 'result.json', result)
    # Whole submission packaging/runtime checks remain separate.
    cid = config['candidate']
    event = {'id': cid + '_peta', 'candidate': cid, 'kind': 'performance',
        'candidate_status': 'ready' if report['AP_delta'] > 0 else 'parked',
        'summary': f'Validation-controlled continuation: extra{len(history)}epochs, bestextra{best_epoch}, AP{controller.best_ap:.6f}, delta{report["AP_delta"]:+.6f}.',
        'issue': 'Adaptive reused PETA; no independent or hidden-score claim. Exact submission packaging/runtime remains.',
        'next_action': 'Review validation curves, best-vs-parent errors and optimizer budget; verify exact saved-model inference before productionizing.',
        'valid': True, 'guardrails_pass': True, 'is_best': False,
        'metric': {'name': 'COCO_OKS_AP', 'direction': 'max', 'evaluation_id': cid + '_peta',
            'evaluation_ref': relative(out / 'selected_comparison/comparison.json'), 'baseline_id': recipe['candidate'],
            'baseline': parent_ap, 'candidate': controller.best_ap, 'material_delta': .001},
        'scope': {'evaluation_role': 'development', 'evaluation_purpose': 'candidate_comparison',
            'execution': 'FullPETA predictions at fixed2epoch intervals; parent included in best selection',
            'review': 'Same assistant and executable guards, not independent review', **evaluation['group_refs'],
            'group_namespace': 'upar-source-domain-v1', 'fit_boundary_ref': config['commitment'],
            'fit_audit_ref': relative(out / 'fit_audit.json')},
        'critical_check': {'risk': 'Incomplete parent, leakage, or worse last weight called best',
            'finding': 'Parent receipt/recipe/weights/predictions verified, full AP replayed; unchanged fit boundary and parent retained',
            'evidence': relative(out / 'result.json')}, 'visual': True,
        'media': [{'role': 'comparison', 'sample': 'fixed paired OKS quantiles', 'split': 'reused PETA',
            'selection': 'paired quantiles', 'caption': 'Parent vs selected continuation',
            'path': relative(out / 'selected_comparison/comparison.png')}],
        'evidence': [relative(out / 'result.json'), relative(out / 'parent_assessment.json')],
        'artifacts': [relative(out / 'result.json'), relative(out / 'selected_comparison/comparison.json')]}
    write_json_atomic(ROOT / config['event_path'], event)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--audit', action='store_true')
    args = parser.parse_args()
    run(args.config, audit=args.audit)
