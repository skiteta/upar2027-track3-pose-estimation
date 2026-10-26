#!/usr/bin/env python3
"""One fixed recipe: audit or durable train -> export -> saved PETA evaluation.

No adaptive research/selection or external submission is performed by this job.
"""
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
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))
from train_scratch_hrnet_v4 import seed_everything
from train_scratch_cnn import load_domain, dataset_items
from pose_research_data import ResearchDataset
from upar_pose.hourglass_research import HourglassResearchPose, research_loss
from upar_pose.pipeline import load_json, sha256_file, write_json_atomic
from upar_pose.training_recovery import training_checkpoint


def create_model(recipe):
    seed_everything(recipe['seed'])
    model = HourglassResearchPose(recipe['stacks'], recipe['error_heads'])
    initialization = model.initialize(ROOT / recipe['checkpoint'], recipe['seed'])
    if recipe.get('continuation_from'):
        spec = recipe['continuation_from']
        source = ROOT / spec['path']
        if sha256_file(source) != spec['sha256']:
            raise ValueError('Continuation checkpoint changed')
        original = load_json(ROOT / spec['manifest'])
        config = original['config']
        if config['fit_scope'] != 'holdout_peta' or config['train_domains'] != ['market', 'pa100k'] or config['train_examples'] != 32422:
            raise ValueError('Continuation cannot undo upstream leakage')
        payload = torch.load(source, map_location='cpu', weights_only=True)
        if payload['epoch'] != spec['completed_epochs']:
            raise ValueError('Continuation epoch mismatch')
        model.load_state_dict(payload['model_state'], strict=True)
        initialization['continuation'] = {**spec, 'optimizer_restarted': True,
            'interpretation': 'Additional predeclared learning phase, not an uninterrupted60epoch OneCycle fit'}
        del payload
    return model, initialization


def verify_recipe(recipe):
    for spec in recipe['frozen_evidence']:
        if sha256_file(ROOT / spec['path']) != spec['sha256']:
            raise ValueError('Frozen evidence changed: ' + spec['path'])
    if recipe['train_domains'] != ['market', 'pa100k'] or recipe['fit_scope'] != 'holdout_peta':
        raise ValueError('Only this explicit holdout boundary is supported')
    if recipe['input_size'] != [256, 256] or recipe['heatmap_sigma'] != 2. or recipe['workers'] != 0:
        raise ValueError('Fixed square256 geometry and single-process augmentation required')
    if recipe['precision'] != 'bf16' or not recipe['require_finite']:
        raise ValueError('Audited BF16 and finite guards required')


def export(model, directory):
    model = model.cpu().eval()
    with torch.inference_mode():
        traced = torch.jit.freeze(torch.jit.trace(model, torch.zeros(1, 3, 256, 256), check_trace=False))
        traced.save(str(directory / 'model.torchscript'))
        reloaded = torch.jit.load(str(directory / 'model.torchscript')).eval()
        x = torch.randn(3, 3, 256, 256)
        expected, actual = model(x), reloaded(x)
        error = float((expected - actual).abs().max())
        if actual.shape != (3, 17, 64, 64) or not torch.isfinite(actual).all() or error > 1e-5:
            raise RuntimeError('Variable-batch exact export parity failed')
    return error


def train(recipe_path, audit=False):
    recipe = load_json(recipe_path)
    verify_recipe(recipe)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required; never silently use CPU for these fits')
    # Blackwell TF32 convolution rounding is measurable in this deep network.
    # Freeze/export explicit FP32 inference math; training still uses BF16 AMP.
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    directory = ROOT / (recipe['audit_dir'] if audit else recipe['arms']['coco']['output_dir'])
    if shutil.disk_usage(ROOT).free < (5 if audit else 12) * 1024**3:
        raise RuntimeError('Insufficient disk reserve before new work; preserve all existing jobs/artifacts')
    directory.mkdir(parents=True, exist_ok=False)
    items = [item for domain in recipe['train_domains'] for item in dataset_items(load_domain(domain), require_labels=True)]
    if len(items) != 32422:
        raise ValueError('Official training source item count changed')
    model, initialization = create_model(recipe)
    write_json_atomic(directory / 'initialization.json', initialization)
    parameter_count = sum(p.numel() for p in model.parameters())
    model = model.cuda()
    seed_everything(recipe['data_seed'])
    dataset = ResearchDataset(items, recipe)
    loader = DataLoader(dataset, batch_size=recipe['batch_size'], shuffle=True, num_workers=0, pin_memory=True)
    config = {**recipe, 'created_at': datetime.now(timezone.utc).isoformat(),
        'recipe_config': {'path': str(recipe_path), 'sha256': sha256_file(recipe_path)},
        'train_examples': len(items), 'parameter_count': parameter_count,
        'external_weights': True, 'external_pretraining_data': 'COCO train2017',
        'external_finetune_data': False, 'hidden_data_used': False,
        'torch': torch.__version__, 'device_name': torch.cuda.get_device_name(),
        'inference_cudnn_allow_tf32': False, 'inference_matmul_allow_tf32': False,
        'cumulative_epochs_after_completion': recipe['epochs'] + recipe.get('continuation_from', {}).get('completed_epochs', 0)}
    write_json_atomic(directory / 'config.json', config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=recipe['learning_rate'], weight_decay=recipe['weight_decay'])
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=recipe['learning_rate'], epochs=recipe['epochs'],
        steps_per_epoch=len(loader), pct_start=.15, div_factor=10., final_div_factor=100.)
    scaler = torch.amp.GradScaler('cuda', enabled=False)
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    history, initial_batch = [], None
    for epoch in range(1, (1 if audit else recipe['epochs']) + 1):
        model.train()
        epoch_start, seen, pasted, total = time.monotonic(), 0, 0, {}
        for batch_index, batch in enumerate(loader):
            inputs, targets, weights, coordinates, visibility, ids = batch
            if initial_batch is None:
                initial_batch = inputs[:3].clone()
                if audit:
                    torch.save({'inputs': initial_batch}, directory / 'parity_inputs.pt')
                write_json_atomic(directory / 'first_batch_identity.json', {
                    'target_annotation_donor_annotation_paste_rectangle': ids.tolist(),
                    'columns': ['target_annotation', 'donor_annotation_or_minus1', 'x', 'y', 'width', 'height'],
                    'source_pool': recipe['train_domains'], 'count': len(items), 'test_or_peta_donors': False})
                if audit:
                    save_examples(inputs[:6], targets[:6], weights[:6], visibility[:6], ids[:6], directory)
            inputs, targets, weights, coordinates, visibility = [x.cuda(non_blocking=True) for x in (inputs, targets, weights, coordinates, visibility)]
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                output, stages, auxiliary = model.forward_training(inputs)
                loss, terms = research_loss(model, output, stages, auxiliary, targets, weights, coordinates, visibility)
            if not torch.isfinite(loss):
                torch.save({'epoch': epoch, 'batch': batch, 'model_state': model.state_dict()}, directory / 'nonfinite_failure.pt')
                raise FloatingPointError('Nonfinite loss; failure retained; no automatic restart')
            loss.backward()
            if not torch.stack([torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None]).all():
                torch.save({'epoch': epoch, 'batch': batch, 'model_state': model.state_dict()}, directory / 'nonfinite_failure.pt')
                raise FloatingPointError('Nonfinite gradients; failure retained')
            optimizer.step()
            scheduler.step()
            count = len(inputs)
            seen += count
            pasted += int((ids[:, 1] >= 0).sum())
            for name, value in {'loss': loss, **terms}.items():
                total[name] = total.get(name, 0.) + float(value.detach()) * count
            if batch_index == 0 or (batch_index + 1) % 100 == 0:
                print(json.dumps({'epoch': epoch, 'batch': batch_index + 1, 'examples': seen, 'loss': float(loss.detach())}), flush=True)
            if audit and batch_index == 2:
                break
        torch.cuda.synchronize()
        row = {'epoch': epoch, 'examples': seen, 'synthetic_person_pastes': pasted,
               'losses': {key: val / seen for key, val in total.items()},
               'train_loss': total['loss'] / seen, 'learning_rate': optimizer.param_groups[0]['lr'],
               'elapsed_seconds': time.monotonic() - epoch_start,
               'observed_at': datetime.now(timezone.utc).isoformat()}
        history.append(row)
        with (directory / 'training_history.jsonl').open('a') as stream:
            stream.write(json.dumps(row) + '\n')
        if not audit:
            checkpoint = training_checkpoint(epoch, model, optimizer, scheduler, scaler)
            checkpoint['dataset_synthetic_rng_state'] = dataset.synthetic_rng.bit_generator.state
            checkpoint['worker_rng_note'] = 'workers0; process and source-only augmentation generator states captured'
            torch.save(checkpoint, directory / 'last_checkpoint.tmp')
            (directory / 'last_checkpoint.tmp').replace(directory / 'last_checkpoint.pt')
            # A bounded number of milestones preserve learning-amount evidence.
            if epoch in recipe.get('save_epochs', []) and epoch != recipe['epochs']:
                torch.save({'epoch': epoch, 'model_state': model.state_dict()}, directory / f'epoch_{epoch:03d}_weights.pt')
            del checkpoint
        print(json.dumps(row), flush=True)
    training_seconds = time.monotonic() - started
    allocated, reserved = torch.cuda.max_memory_allocated(), torch.cuda.max_memory_reserved()
    model.eval()
    with torch.inference_mode():
        expected_real = model(initial_batch.cuda()).float().cpu()
    del optimizer, scheduler, loss, output, stages, auxiliary, inputs, targets, weights, coordinates, visibility
    gc.collect()
    error = export(model, directory)
    torch.cuda.empty_cache()
    saved = torch.jit.load(str(directory / 'model.torchscript')).eval()
    with torch.inference_mode():
        cpu_real = saved(initial_batch).float()
    cpu_gpu_error = float((cpu_real - expected_real).abs().max())
    write_json_atomic(directory / 'parity_observation.json', {'CPU_export_max_abs_error': error,
        'real_CPU_GPU_max_abs_error': cpu_gpu_error, 'limit': 2e-4,
        'inference_cudnn_allow_tf32': False, 'inference_matmul_allow_tf32': False})
    if cpu_gpu_error > 2e-4:
        raise RuntimeError('Real-input CPU/GPU numerical mismatch')
    # Frozen constants are remapped at load, not by module.cuda().
    saved = torch.jit.load(str(directory / 'model.torchscript'), map_location='cuda').eval()
    with torch.inference_mode():
        jit_gpu = saved(initial_batch.cuda()).float().cpu()
        export_gpu_error = float((jit_gpu - expected_real).abs().max())
        if export_gpu_error > 2e-4:
            raise RuntimeError('GPU exported model mismatch')
        runtime_inputs = initial_batch.repeat(2, 1, 1, 1).cuda()
        for _ in range(2):
            saved(runtime_inputs)
        torch.cuda.synchronize()
        runtime_start = time.monotonic()
        for _ in range(10):
            saved(runtime_inputs)
        torch.cuda.synchronize()
        runtime_seconds = (time.monotonic() - runtime_start) / 10
    write_json_atomic(directory / 'training_curve.json', {'epochs': history, 'note': 'Training objective, not heldout AP; full saved-model evaluation follows'})
    names = ['config.json', 'initialization.json', 'model.torchscript', 'training_curve.json', 'training_history.jsonl']
    if not audit:
        names.append('last_checkpoint.pt')
    manifest = {'candidate': recipe['candidate'], 'config': config, 'history': history, 'audit_only': audit,
                'training_seconds': training_seconds, 'peak_cuda_allocated_bytes': allocated,
                'peak_cuda_reserved_bytes': reserved, 'parameter_count': parameter_count,
                'export_max_abs_error': error, 'real_CPU_GPU_max_abs_error': cpu_gpu_error,
                'real_JIT_GPU_max_abs_error': export_gpu_error, 'gpu_batch6_forward_seconds': runtime_seconds,
                'runtime_scope': '10batch6 forwards on real training pixels; excludes crop/decode/organizer-hardware differences; not full submission feasibility',
                'artifacts': {name: {'path': name, 'sha256': sha256_file(directory / name)} for name in names}}
    write_json_atomic(directory / 'manifest.json', manifest)
    print(json.dumps({'saved_manifest': str(directory / 'manifest.json'), 'peak_reserved_GiB': reserved / 1024**3,
                      'parameters': parameter_count, 'audit_only': audit}), flush=True)
    return manifest


def save_examples(inputs, targets, weights, visibility, ids, directory):
    from PIL import Image, ImageDraw
    canvas = Image.new('RGB', (3 * 256, 2 * 290), 'white')
    for index, pixels in enumerate(inputs):
        array = ((pixels.numpy().transpose(1, 2, 0) * .25 + .5) * 255).clip(0, 255).astype(np.uint8)
        picture = Image.fromarray(array)
        draw = ImageDraw.Draw(picture)
        for joint in range(17):
            if weights[index, joint] > 0:
                y, x = np.unravel_index(int(targets[index, joint].argmax()), (64, 64))
                x, y = x * 255 / 63, y * 255 / 63
                draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill='lime' if visibility[index, joint] == 2 else 'orange')
        ox, oy = index % 3 * 256, index // 3 * 290
        canvas.paste(picture, (ox, oy))
        ImageDraw.Draw(canvas).text((ox + 3, oy + 258), f'train ann={int(ids[index, 0])} donor={int(ids[index, 1])}', fill='black')
    canvas.save(directory / 'training_examples.png')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--audit', action='store_true')
    args = parser.parse_args()
    train(args.config, audit=args.audit)
    if not args.audit:
        # Fresh process releases training activations/optimizer before evaluation.
        import subprocess
        recipe = load_json(args.config)
        subprocess.run([sys.executable, '-B', 'scripts/train_hourglass_transfer.py', 'evaluate',
                        '--config', recipe['evaluation_recipe']], cwd=ROOT, check=True)


if __name__ == '__main__':
    main()
