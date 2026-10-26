#!/usr/bin/env python3
"""Cluster saved-prediction differences by exact decoded image, not row ID."""
import argparse
from datetime import datetime,timezone
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'scripts'))
from train_scratch_cnn import load_domain
from upar_pose.evaluation import assigned_prediction_oks_by_image
from upar_pose.pipeline import load_json,sha256_file,write_json_atomic


def cluster_interval(reference,candidate,groups,*,replicates=5000,seed=20261903):
    if set(reference)!=set(candidate):raise ValueError('ID mismatch')
    root={i:i for i in reference}
    for group in groups:
        ids=[int(x['image_id']) for x in group if x['domain']=='peta' and int(x['image_id']) in root]
        for i in ids:root[i]=min(ids)
    grouped={}
    for i in reference:grouped.setdefault(root[i],[]).append(candidate[i]-reference[i])
    sums=np.array([sum(v) for _,v in sorted(grouped.items())]);counts=np.array([len(v) for _,v in sorted(grouped.items())])
    rng=np.random.default_rng(seed);samples=[]
    for start in range(0,replicates,100):
        indices=rng.integers(0,len(sums),size=(min(100,replicates-start),len(sums)))
        samples.extend((sums[indices].sum(axis=1)/counts[indices].sum(axis=1)).tolist())
    low,high=np.quantile(samples,[.025,.975])
    return {'estimate_image_weighted_mean_OKS_delta':float(sums.sum()/counts.sum()),'ci_low':float(low),'ci_high':float(high),
        'image_rows':len(reference),'exact_image_clusters':len(sums),'replicates':replicates,'seed':seed,
        'resampling':'exact-decoded-image clusters, retain row-weighted mean within each replicate',
        'limits':['Assigned OKS interval, not AP interval.','Exact-image clustering does not remove unknown same-person/track/near-duplicate dependence.','Repeated-development selection is not corrected.']}


def main():
    p=argparse.ArgumentParser();p.add_argument('--reference',type=Path,required=True);p.add_argument('--candidate',type=Path,required=True)
    p.add_argument('--groups',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    ds=load_domain('peta')
    r=assigned_prediction_oks_by_image(ds,load_json(args.reference));c=assigned_prediction_oks_by_image(ds,load_json(args.candidate))
    report=cluster_interval(r,c,load_json(args.groups));report['generated_at']=datetime.now(timezone.utc).isoformat()
    report['inputs']={str(x):sha256_file(x) for x in (args.reference,args.candidate,args.groups)}
    write_json_atomic(args.output,report);print(report)


if __name__=='__main__':main()
