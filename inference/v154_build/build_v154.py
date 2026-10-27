"""Build v153 = exact v127 archive members + v153 run.py (tier-gated scale TTA replacing pose-SAHI) + meta revision note."""
import hashlib, json, zipfile
from pathlib import Path
ROOT = Path('/home/syouya/ml/lab/2d_human_pose_estimation'); V = ROOT / 'artifacts/runs/20261027T1600_v154'
V127 = ROOT / 'artifacts/runs/20261027T0335_final_submission/package/upar_track3_final_candidate.zip'
ORDER = ['LICENSE_MMPose.txt', 'metadata.yaml', 'model_l3.torchscript', 'model_meta.json', 'model_rsn.torchscript', 'model_vp.torchscript', 'run.py']
stage = V / 'staging'; stage.mkdir(exist_ok=False)
with zipfile.ZipFile(V127) as z:
    for f in ORDER: (stage / f).write_bytes(z.read(f))
assert hashlib.sha256((stage / 'run.py').read_bytes()).hexdigest().startswith('90bd6971')
(stage / 'run.py').write_bytes((V / 'run_prod_v154.py').read_bytes())
meta = json.loads((stage / 'model_meta.json').read_text())
meta['revision_notes'].append({'revision': 'v154_scale_tta_confidence_fusion',
    'change': 'At controller tiers <= 1 every member uses paddings (1.15, 1.2, 1.3, 1.35) with flip (equal weights) instead of (1.2, 1.3); pose-SAHI is not used. The fusion keeps each member per-joint confidence (unit-peak map times its own peak) before the anatomical weighted mean; person score unchanged. Tier >= 2 or after the SAHI time budget: (1.2, 1.3) views. Model files unchanged.',
    'development_evidence': 'hard5 dev 2,314 persons tier 1: AP 0.91727 -> 0.91772, AP75 0.9551 -> 0.9571; PT18 sample AP 0.7928 -> 0.7948, AP_large 0.7998 -> 0.8024.'})
meta['deployment_status'] = 'LOCAL_CANDIDATE_v154_UNSUBMITTED'
(stage / 'model_meta.json').write_text(json.dumps(meta, indent=2))
zp = V / 'upar_track3_final_candidate_v154.zip'
with zipfile.ZipFile(zp, 'w', zipfile.ZIP_DEFLATED) as z:
    for f in ORDER: z.write(stage / f, f)
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
rep = {'archive': str(zp.relative_to(ROOT)), 'archive_sha256': sha(zp), 'archive_bytes': zp.stat().st_size, 'members': {f: sha(stage / f) for f in ORDER},
       'unchanged_from_v127': [f for f in ORDER if f not in ('run.py', 'model_meta.json')]}
with zipfile.ZipFile(zp) as z: rep['crc_ok'] = z.testzip() is None
json.dump(rep, open(V / 'package_report.json', 'w'), indent=2); print(json.dumps({k: rep[k] for k in ('archive', 'archive_sha256', 'archive_bytes', 'crc_ok')}))
