"""Opt-in, bounded-memory v4 cache from the existing six-view evaluation.

No inference, normalization or decoding is performed here. Maps remain on the
member's native grid; FP16 is the existing offline-dumper storage convention.
Final metadata is published last, only after model/weights/ID/hash validation.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import shutil

import numpy as np
import torch

from .flip_tta import FLIP_CHANNELS
from .geometry_tta_v4 import aligned_weighted_mean

VIEW_PADDINGS = (1.20, 1.30)
INPUT_PADDINGS = (1.20, 1.25, 1.30)
TARGET_PADDING = 1.25


def aligned_v4_from_flip_means(views, paddings=INPUT_PADDINGS):
    """Same ordered arithmetic as evaluate_multimodel_map_fusion --save-maps v4."""
    if tuple(paddings) != INPUT_PADDINGS or views.ndim != 5 or views.shape[1:3] != (3, 17):
        raise ValueError('v4 cache requires [B,3,17,H,W] flip means ordered 1.20,1.25,1.30')
    if views.shape[-2] != views.shape[-1]:
        raise ValueError('v4 cache currently supports square native maps only')
    return aligned_weighted_mean(
        {pad: views[:, paddings.index(pad)] for pad in VIEW_PADDINGS},
        [(pad, 1.0) for pad in VIEW_PADDINGS], target_padding=TARGET_PADDING)


def _sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b''):
            value.update(block)
    return value.hexdigest()


def state_dict_sha256(state):
    """Exact named tensor fingerprint; at most a 4-MiB CPU block per tensor read.

    Also works on mmap checkpoint tensors. Does not alter tensors or RNG. This
    is not the serialized checkpoint hash; both identities are recorded.
    """
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        if not isinstance(tensor, torch.Tensor) or tensor.layout != torch.strided or tensor.is_quantized:
            raise ValueError('cache state binding requires dense unquantized tensor state')
        header = json.dumps([name, str(tensor.dtype), list(tensor.shape)], separators=(',', ':')).encode()
        digest.update(len(header).to_bytes(8, 'little'))
        digest.update(header)
        flat = tensor.detach().contiguous().reshape(-1)
        chunk = max(1, 4 * 1024**2 // tensor.element_size())
        for start in range(0, flat.numel(), chunk):
            raw = flat[start:start + chunk].cpu().view(torch.uint8).numpy()
            digest.update(memoryview(raw))
    return digest.hexdigest()


def _json_bytes(value):
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n').encode()


def _publish_bytes(path, data):
    """Atomic publication without overwriting another process's artifact."""
    temporary = path.with_name('.' + path.name + '.partial')
    with temporary.open('xb') as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    os.link(temporary, path)  # Atomic, but unlike replace refuses an existing target.
    temporary.unlink()


def _ids(image_ids, annotation_ids):
    images, annotations = list(image_ids), list(annotation_ids)
    if len(images) != len(annotations):
        raise ValueError('cache image/annotation ID counts differ')
    result = []
    for image, annotation in zip(images, annotations):
        if isinstance(image, bool) or isinstance(annotation, bool) or int(image) != image or int(annotation) != annotation:
            raise ValueError('cache IDs must be integers')
        result.append((int(image), int(annotation)))
    return result


class EvaluationMapCache:
    """One immutable model/epoch cache; incomplete files never gain reader metadata.

    Require an existing epoch-weights file. This deliberately does not guess the
    identity of an unsaved e0 or an epoch discarded by save_weights_from_epoch.
    The caller owns failure policy and should not stop healthy training merely
    because optional cache IO fails.
    """

    def __init__(self, directory, *, project_root, model, weights_path, config_path,
                 candidate, epoch, expected_ids, cameras, grid_side, dtype='float16',
                 source_paths=(), data_paths=(), execution_context=None):
        self.root, self.directory = Path(project_root).resolve(), Path(directory).resolve()
        self.weights_path, self.config_path = Path(weights_path).resolve(), Path(config_path).resolve()
        for path in (self.directory, self.weights_path, self.config_path):
            if not path.is_relative_to(self.root):
                raise ValueError('cache and binding paths must stay inside the project')
        if self.directory.exists():
            raise FileExistsError('cache directory already exists; do not overwrite or resume partial maps')
        if not isinstance(candidate, str) or not candidate or isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
            raise ValueError('cache requires a candidate ID and nonnegative integer epoch')
        if isinstance(grid_side, bool) or not isinstance(grid_side, int) or grid_side < 2 or dtype not in ('float16', 'float32'):
            raise ValueError('cache requires a native square grid and explicit float16/float32 storage')
        self.expected_ids = _ids(*zip(*expected_ids)) if expected_ids else []
        if (not self.expected_ids or len(set(self.expected_ids)) != len(self.expected_ids)
                or len({a for _, a in self.expected_ids}) != len(self.expected_ids)):
            raise ValueError('cache expected IDs must be nonempty and annotation-unique')
        self.model, self.side, self.dtype, self.offset = model, grid_side, np.dtype(dtype), 0
        self.geometry, self.cameras, self.finished = [], list(cameras), False
        if not source_paths or not execution_context or execution_context.get('exact_crop') is not False:
            raise ValueError('cache requires source provenance and explicit legacy exact_crop=False execution')
        required_bytes = len(self.expected_ids) * 17 * grid_side**2 * self.dtype.itemsize
        if shutil.disk_usage(self.root).free < required_bytes + 1024**3:
            raise OSError('insufficient disk for optional maps plus 1 GiB minimum reserve')
        config = json.loads(self.config_path.read_text())
        self.weights_sha, self.config_sha = _sha(self.weights_path), _sha(self.config_path)
        checkpoint = torch.load(self.weights_path, map_location='cpu', weights_only=True, mmap=True)
        if checkpoint.get('epoch') != epoch or checkpoint.get('config_sha256') != self.config_sha:
            raise ValueError('cache epoch/config does not match immutable saved weights')
        saved_state_sha = state_dict_sha256(checkpoint['model_state'])
        self.model_state_sha = state_dict_sha256(model.state_dict())
        del checkpoint
        if self.model_state_sha != saved_state_sha:
            raise ValueError('cache current model state differs from saved weights')
        if config.get('candidate') != candidate or config.get('input_size') != [256, 256]:
            raise ValueError('cache candidate/input geometry does not match config')
        if set(self.cameras) != set(config.get('heldout_subdomains', [])):
            raise ValueError('cache cameras differ from the declared evaluation population')
        role = config.get('evaluation_role')
        if role not in ('development', 'train_diagnostic'):
            raise ValueError('cache requires explicit development or train_diagnostic role')
        if role == 'development' and (config.get('fit_includes_heldout') or set(self.cameras) & set(config.get('peta_train_subdomains', []))):
            raise ValueError('cache development fit includes evaluation cameras')
        safe_candidate = re.sub(r'[^A-Za-z0-9_.-]', '_', candidate)[:96]
        self.label = f'{safe_candidate}__e{epoch:03d}__{self.weights_sha}'
        self.sources = {}
        snapshots = {}
        for supplied in source_paths:
            path = Path(supplied).resolve()
            if not path.is_relative_to(self.root):
                raise ValueError('cache source path outside project')
            name = str(path.relative_to(self.root))
            data = path.read_bytes()
            self.sources[name] = hashlib.sha256(data).hexdigest()
            snapshots[name] = data
        self.data_refs = []
        for supplied in data_paths:
            path = Path(supplied).resolve()
            if not path.is_relative_to(self.root):
                raise ValueError('cache data path outside project')
            self.data_refs.append({'path': str(path.relative_to(self.root)), 'sha256': _sha(path)})
        self.binding = {'candidate': candidate, 'epoch': epoch,
            'weights': {'path': str(self.weights_path.relative_to(self.root)), 'sha256': self.weights_sha},
            'config': {'path': str(self.config_path.relative_to(self.root)), 'sha256': self.config_sha},
            'model_state_sha256': self.model_state_sha, 'saved_state_bitexact': True,
            'model_class': type(model).__module__ + '.' + type(model).__qualname__,
            'evaluation_role': role, 'fit_includes_heldout': bool(config.get('fit_includes_heldout', False)),
            'peta_train_subdomains': config.get('peta_train_subdomains', []),
            'data_refs': self.data_refs,
            'execution_context': execution_context or {}}
        self.directory.mkdir(parents=True, exist_ok=False)
        for name, data in snapshots.items():
            destination = self.directory / 'source_snapshots' / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            with destination.open('xb') as stream:
                stream.write(data)
        self.partial_path = self.directory / f'maps_{self.label}__v4.npy.partial'
        self.map_path = self.directory / f'maps_{self.label}__v4.npy'
        self.maps = np.lib.format.open_memmap(self.partial_path, mode='w+', dtype=self.dtype,
                                             shape=(len(self.expected_ids), 17, grid_side, grid_side))

    def abort(self):
        """Close only this writer; keep incomplete files for diagnosis, no commit marker."""
        if not self.finished and getattr(self, 'maps', None) is not None:
            if not self.maps._mmap.closed:
                self.maps._mmap.close()
            self.maps = None

    def append(self, maps, image_ids, annotation_ids, geometry):
        if self.finished or self.maps is None:
            raise ValueError('cache is already finalized')
        ids = _ids(image_ids, annotation_ids)
        if not ids or self.expected_ids[self.offset:self.offset + len(ids)] != ids:
            raise ValueError('cache batch IDs/order do not match expected population')
        if tuple(maps.shape) != (len(ids), 17, self.side, self.side):
            raise ValueError('cache maps have an unexpected shape/grid')
        geom = np.asarray(geometry, dtype=np.float64)
        if geom.shape != (len(ids), 4) or not np.isfinite(geom).all() or (geom[:, 2:] <= 0).any():
            raise ValueError('cache geometry must be finite positive-width/height rows')
        array = maps.detach().cpu().numpy() if isinstance(maps, torch.Tensor) else np.asarray(maps)
        if not np.isfinite(array).all():
            raise ValueError('cache maps contain nonfinite values')
        with np.errstate(over='ignore', invalid='ignore'):
            stored = array.astype(self.dtype, copy=False)
        if not np.isfinite(stored).all():
            raise ValueError('cache conversion overflowed the storage dtype')
        self.maps[self.offset:self.offset + len(ids)] = stored
        self.geometry.extend(geom.tolist())
        self.offset += len(ids)

    def finalize(self, *, prediction_path=None):
        if self.finished or self.maps is None or self.offset != len(self.expected_ids):
            raise ValueError('cache cannot publish incomplete or already-finalized data')
        if (_sha(self.weights_path) != self.weights_sha or _sha(self.config_path) != self.config_sha
                or state_dict_sha256(self.model.state_dict()) != self.model_state_sha):
            raise ValueError('cache weights/config/current model state changed during evaluation')
        for name, digest in self.sources.items():
            if _sha(self.root / name) != digest:
                raise ValueError('cache source changed during evaluation: ' + name)
        for reference in self.data_refs:
            if _sha(self.root / reference['path']) != reference['sha256']:
                raise ValueError('cache data changed during evaluation: ' + reference['path'])
        prediction = None
        if prediction_path is not None:
            path = Path(prediction_path).resolve()
            if not path.is_relative_to(self.root):
                raise ValueError('cache prediction path outside project')
            rows = json.loads(path.read_text())
            if _ids([r['image_id'] for r in rows], [r['source_annotation_id'] for r in rows]) != self.expected_ids:
                raise ValueError('cache six-view prediction IDs do not match')
            prediction = {'path': str(path.relative_to(self.root)), 'sha256': _sha(path)}
        self.maps.flush()
        self.maps._mmap.close()
        with self.partial_path.open('rb') as stream:
            os.fsync(stream.fileno())
        maps_sha = _sha(self.partial_path)
        os.link(self.partial_path, self.map_path)
        self.partial_path.unlink()
        metadata = {'image_ids': [i for i, _ in self.expected_ids], 'annotation_ids': [a for _, a in self.expected_ids],
            'geometry': self.geometry, 'labels': [self.label], 'weights': [self.binding['weights']['path']],
            'cameras': self.cameras, 'view': 'v4', 'cache_manifest': 'cache_manifest.json'}
        metadata_bytes = _json_bytes(metadata)
        manifest = {'schema': 1, 'status': 'COMPLETE', 'binding': self.binding,
            'model_label': self.label, 'maps': {'file': self.map_path.name, 'sha256': maps_sha,
                'shape': [len(self.expected_ids), 17, self.side, self.side], 'dtype': self.dtype.name},
            'metadata_sha256': hashlib.sha256(metadata_bytes).hexdigest(),
            'source_hashes': self.sources, 'six_view_predictions': prediction,
            'contract': {'input_size': [256, 256], 'input_paddings': list(INPUT_PADDINGS),
                'view': 'v4', 'paddings': list(VIEW_PADDINGS), 'flip': True, 'flip_channels': [int(i) for i in FLIP_CHANNELS],
                'scale_weights': [1.0, 1.0], 'target_padding': TARGET_PADDING,
                'crop': 'legacy PIL rounded crop then BILINEAR resize; exact_crop=False',
                'alignment': 'bilinear zero-padding align_corners=True', 'normalization': 'none',
                'decoder': 'none', 'native_grid': True,
                'storage_note': 'FP16 matches existing dumper quantization; it is not exact FP32 production output.'}}
        _publish_bytes(self.directory / 'cache_manifest.json', _json_bytes(manifest))
        _publish_bytes(self.directory / 'maps_meta__v4.json', metadata_bytes)  # Commit marker, always last.
        self.finished = True
        return {'directory': str(self.directory.relative_to(self.root)), 'model_label': self.label,
                'manifest_sha256': _sha(self.directory / 'cache_manifest.json'), 'maps_sha256': maps_sha}


def validate_map_cache(directory, *, expected_ids=None, expected_geometry=None):
    """Strict optional reader preflight; compatible metadata is insufficient alone."""
    directory = Path(directory)
    metadata_path = directory / 'maps_meta__v4.json'
    metadata = json.loads(metadata_path.read_text())
    manifest = json.loads((directory / 'cache_manifest.json').read_text())
    if (manifest.get('status') != 'COMPLETE' or metadata.get('view') != 'v4'
            or _sha(metadata_path) != manifest['metadata_sha256']):
        raise ValueError('cache metadata is incomplete, changed or not v4')
    label = manifest['model_label']
    if metadata['labels'] != [label] or metadata['weights'] != [manifest['binding']['weights']['path']]:
        raise ValueError('cache member label/weight binding changed')
    maps_path = directory / manifest['maps']['file']
    if maps_path.name != f'maps_{label}__v4.npy' or maps_path.parent != directory:
        raise ValueError('cache map filename escapes or mismatches the member identity')
    if _sha(maps_path) != manifest['maps']['sha256']:
        raise ValueError('cache maps hash changed')
    maps = np.load(maps_path, mmap_mode='r', allow_pickle=False)
    if list(maps.shape) != manifest['maps']['shape'] or maps.dtype.name != manifest['maps']['dtype']:
        raise ValueError('cache maps shape/dtype changed')
    ids = _ids(metadata['image_ids'], metadata['annotation_ids'])
    if len(ids) != len(maps) or len(set(ids)) != len(ids) or len({a for _, a in ids}) != len(ids):
        raise ValueError('cache identity population invalid')
    if expected_ids is not None and ids != list(expected_ids):
        raise ValueError('cache image/annotation ID order mismatch')
    if expected_geometry is not None and not np.array_equal(metadata['geometry'], expected_geometry):
        raise ValueError('cache geometry mismatch')
    return metadata, manifest
