"""Run an unmodified Codabench submission archive on a Codabench-style input folder, offline.

The input folder has the layout of the official evaluation container::

    input_data/
        images/          image files referenced by boxes.json
        boxes.json       COCO format: images + person boxes (no keypoints)

For every image (in the order of ``boxes.json["images"]``) the script builds the same
``sample`` dict that the official ingestion program passes to ``run.predict_image``
(``image_id``, ``file_name``, absolute ``image_path``, ``width``, ``height``, COCO xywh
``boxes`` and ``box_ids`` in annotation order), calls ``run.load_model()`` once if it
exists, and writes COCO keypoint results (one entry per box) that pycocotools'
``COCO.loadRes`` accepts. The submission code is imported as is; nothing in it is
patched, so its runtime guards behave exactly as written (see README).

Usage::

    python inference/run_codabench_layout.py --submission /path/to/extracted_zip \
        --input /path/to/input_data --output predictions.json
"""

import argparse
import importlib.util
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np


def load_submission(submission: Path):
    spec = importlib.util.spec_from_file_location("run", submission / "run.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["run"] = module
    spec.loader.exec_module(module)
    return module


def build_samples(input_dir: Path):
    data = json.loads((input_dir / "boxes.json").read_text())
    by_image = defaultdict(list)
    for annotation in data["annotations"]:
        by_image[annotation["image_id"]].append(annotation)
    samples = []
    for image in data["images"]:
        annotations = by_image.get(image["id"], [])
        samples.append({
            "image_id": image["id"],
            "file_name": image["file_name"],
            "image_path": str((input_dir / "images" / image["file_name"]).resolve()),
            "width": image["width"],
            "height": image["height"],
            "boxes": [[float(v) for v in a["bbox"]] for a in annotations],
            "box_ids": [a["id"] for a in annotations],
        })
    return samples


def to_rows(sample, prediction):
    if isinstance(prediction, dict):
        keypoints = np.asarray(prediction["keypoints"], dtype=np.float64)
        scores = prediction.get("scores")
    else:
        keypoints, scores = np.asarray(prediction, dtype=np.float64), None
    count = len(sample["boxes"])
    keypoints = keypoints.reshape(count, 17, 3) if count else keypoints.reshape(0, 17, 3)
    scores = np.ones(count) if scores is None else np.asarray(scores, dtype=np.float64).reshape(count)
    if not (np.isfinite(keypoints).all() and np.isfinite(scores).all()):
        raise ValueError(f"non-finite prediction for image {sample['image_id']}")
    return [{"image_id": sample["image_id"], "category_id": 1, "box_id": box_id,
             "keypoints": keypoints[i].reshape(-1).tolist(), "score": float(scores[i])}
            for i, box_id in enumerate(sample["box_ids"])]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--submission", type=Path, required=True, help="extracted submission folder (run.py at its root)")
    parser.add_argument("--input", type=Path, required=True, help="folder with images/ and boxes.json")
    parser.add_argument("--output", type=Path, required=True, help="COCO keypoint results JSON to write")
    parser.add_argument("--limit", type=int, default=None, help="only the first N images (smoke test)")
    args = parser.parse_args()

    submission, input_dir = args.submission.resolve(), args.input.resolve()
    samples = build_samples(input_dir)[: args.limit]
    os.chdir(submission)  # the official container runs from the submission folder
    run = load_submission(submission)
    start = time.perf_counter()
    if hasattr(run, "load_model"):
        run.load_model()
    rows = []
    for index, sample in enumerate(samples, 1):
        rows.extend(to_rows(sample, run.predict_image(sample)))
        if index % 500 == 0:
            print(f"[replay] {index}/{len(samples)} images, {time.perf_counter() - start:.1f}s", flush=True)
    seconds = time.perf_counter() - start
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rows))
    print(json.dumps({"images": len(samples), "predictions": len(rows), "seconds": round(seconds, 1),
                      "output": str(args.output)}))


if __name__ == "__main__":
    main()
