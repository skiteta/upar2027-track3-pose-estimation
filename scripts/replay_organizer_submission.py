"""Offline replay of an extracted organizer submission, no uploads or downloads."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import time
import numpy as np
import torch

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--package',type=Path,required=True)
    p.add_argument('--samples',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    package=a.package.resolve()
    s=importlib.util.spec_from_file_location('submitted_run',package/'run.py')
    m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
    samples=json.loads(a.samples.read_text()); started=time.monotonic()
    rows=[]
    for sample in samples:
        pred=m.predict_image(sample)
        n=len(sample['boxes'])
        kp=np.asarray(pred['keypoints']);scores=np.asarray(pred['scores'])
        if n and kp.shape!=(n,17,3):raise ValueError('Invalid keypoint count/shape')
        if len(kp)!=n or scores.shape!=(n,):raise ValueError('Invalid box order/count contract')
        if not np.isfinite(kp).all() or not np.isfinite(scores).all():raise ValueError('Nonfinite output')
        rows.append({'image_id':sample['image_id'],'box_ids':sample['box_ids'],
                     'keypoints':kp.tolist(),'scores':scores.tolist()})
    report={'torch':torch.__version__,'cuda':torch.version.cuda,
       'run_sha256':hashlib.sha256((package/'run.py').read_bytes()).hexdigest(),
       'seconds':time.monotonic()-started,'predictions':rows,
       'runtime_state':getattr(m,'_RUNTIME',{}),'scope':'Offline validity/replay, not server timing certification'}
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('x') as f:json.dump(report,f,indent=2)
    print(json.dumps({'images':len(rows),'output':str(a.output),'seconds':report['seconds']}))

if __name__=='__main__':main()
