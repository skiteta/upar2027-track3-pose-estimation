"""COCO keypoint AP (OKS 0.50:0.95, maxDets 20) of a predictions file against a ground-truth file.

Uses pycocotools with the 17 PoseTrack18 keypoint sigmas of this project (ears are unlabelled in
the UPAR ground truth). The official leaderboard averages AP over four hidden subsets; pass one
ground-truth file per subset and average the printed APs to mirror that.

Usage::

    python inference/evaluate_keypoints.py --gt ground_truth.json --pred predictions.json
"""

import argparse
import contextlib
import io
import json

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

SIGMAS = np.array([0.026, 0.025, 0.025, 0.035, 0.035, 0.079, 0.079, 0.072, 0.072,
                   0.062, 0.062, 0.107, 0.107, 0.087, 0.087, 0.089, 0.089])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gt", required=True)
    parser.add_argument("--pred", required=True)
    args = parser.parse_args()
    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO(args.gt)
        gt.dataset.setdefault("info", {})
        gt.dataset.setdefault("licenses", [])
        rows = [{k: v for k, v in r.items() if k != "box_id"} for r in json.load(open(args.pred))]
        evaluator = COCOeval(gt, gt.loadRes(rows), "keypoints")
        evaluator.params.kpt_oks_sigmas = SIGMAS
        evaluator.params.maxDets = [20]
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
    names = ["AP", "AP50", "AP75", "AP_medium", "AP_large", "AR", "AR50", "AR75", "AR_medium", "AR_large"]
    print(json.dumps({n: round(float(v), 6) for n, v in zip(names, evaluator.stats)}))


if __name__ == "__main__":
    main()
