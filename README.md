# UPAR Challenge 2027 — Track 3 (2D Human Pose Estimation): code for Codabench submission 1006463

Participant: Syouya Tobita (Codabench `syouya_tobita`, individual participant).

This repository contains the inference code, training code, configurations and checksums needed to reproduce the
**last submission that appears in the organizers' leaderboard snapshot**:

| Item | Value |
|---|---|
| Codabench submission | **1006463** |
| Submitted archive | `upar_fusion3_sahi_anat9_v109.zip`, SHA-256 `b2210cf1b12e3b1b8632fd090262e44ed6eb0cfb165e635e5b1a1661345f6375` |
| Leaderboard (snapshot CSV) | AP 0.886743, AP50 0.965086 (AP_medium/AP_large listed as 0) |
| Official scoring log | AP 0.886743, AP50 0.965086, AP75 0.925566, AP_medium 0.857837, AP_large 0.679830 |

The method is a three-model top-down heatmap fusion (3×RSN-50 + two ViT-L/16) with per-joint anatomical fusion,
gated pose-guided zoom views and OKS-aware decoding. It is described in the fact sheet that accompanies this
repository; the exact inference code is the `run.py` inside the submitted archive.

## 1. Trained weights

The weights are too large for the repository and are hosted at: **WEIGHTS-URL-TO-BE-FILLED**

| File | Size | SHA-256 | Purpose |
|---|---|---|---|
| `upar_fusion3_sahi_anat9_v109.zip` | 1.26 GB | `b2210cf1…6375` | the exact submitted archive (run.py + 3 FP16 TorchScript members) |
| `soup_L3L4_v109_prod_weights.pt` | 1.24 GB | `ce7ff9df…6c` | trainer-format weights of the ViT-L "L3L4" member |
| `soup_VP_v108_prod_weights.pt` | 1.24 GB | `f2cf06f1…3f` | trainer-format weights of the ViT-L "VP" member |
| `FB_soup_e4to7_weights.pt` | 0.24 GB | `3728055d…08992` | trainer-format weights of the RSN member |

Full checksums: [`weights/SHA256SUMS`](weights/SHA256SUMS). Verify after download with
`cd weights && sha256sum -c SHA256SUMS`.

## 2. Reproduce the Codabench inference (Docker, recommended)

The submission runs in the official evaluation image `pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime`
(Python 3.11, PyTorch 2.4.1, NumPy, Pillow; no network access). The provided `Dockerfile` uses that image and
adds `inference/run_codabench_layout.py`, which reads the Codabench input layout and calls the unmodified
`run.py` exactly like the official ingestion program (`load_model()` once, then `predict_image(sample)` per image).

```bash
# 1) unpack the exact submitted archive (run.py must be at the root of this folder)
mkdir -p submission && unzip weights/upar_fusion3_sahi_anat9_v109.zip -d submission

# 2) input in the Codabench layout:
#    input_data/images/<file_name>   and   input_data/boxes.json  (COCO images + person boxes)

# 3) build and run
docker build -t upar-track3-1006463 .
docker run --rm --gpus all \
    -v "$PWD/submission:/submission" \
    -v "$PWD/input_data:/input_data:ro" \
    -v "$PWD/output:/output" \
    upar-track3-1006463
# -> output/predictions.json : COCO keypoint results, one entry per box (image_id, box_id, keypoints[51], score)

# 4) optional: score against a COCO keypoint ground-truth file (run once per hidden subset and average the APs)
docker run --rm -v "$PWD:/work" --entrypoint python upar-track3-1006463 \
    /opt/upar/inference/evaluate_keypoints.py --gt /work/ground_truth.json --pred /work/output/predictions.json
```

Without Docker, any Python ≥ 3.10 environment with `torch==2.4.1`, `numpy` and `Pillow` works:
`python inference/run_codabench_layout.py --submission submission --input input_data --output predictions.json`
(add `--limit 10` for a quick smoke test). The archive selects CUDA when available and falls back to the CPU.

### Important: the scored run took a degraded path

`run.py` contains wall-clock guards designed for the documented 7,200 s limit. In the official run of 1006463
the CPU post-processing was slow (≈0.60 s/image), so the guards disabled the pose-guided zoom views after 20
images, switched to the centre crop padding at image 8,433, and then dropped the VP and L3 members, so most of
the remaining images were predicted by the RSN member alone (official log in
[`evidence/runtime_1006463.txt`](evidence/runtime_1006463.txt); the log reports completion after 14,224.6 s).
On a faster machine the same unmodified archive keeps more members and views, so its predictions — and
usually its AP — differ from the scored ones. We do not modify the archive to imitate the server timing.

## 3. Training

See [`TRAINING.md`](TRAINING.md): data layout, environment (`docker/train.Dockerfile`), the trainer
`scripts/train_rsn_domain_continuation.py`, the configuration of every stage of the three member lineages
(`artifacts/runs/**`), the weight-soup recipes, and the export to TorchScript.

## 4. Repository layout

| Path | Content |
|---|---|
| `inference/` | offline Codabench-layout runner and COCO keypoint evaluation |
| `Dockerfile`, `docker/train.Dockerfile` | inference image (official base) and training image |
| `src/upar_pose/` | model definitions (ViT-L pose graph, vendored RSN), data, evaluation utilities |
| `scripts/` | trainer, augmentation, soups, TorchScript export and packaging tools |
| `artifacts/runs/**` | the training configurations of all stages, with their original paths and seeds |
| `evidence/` | official scoring/runtime logs, leaderboard snapshot row, package reports, saved competition rules |
| `historical_appendix/` | provenance of external pre-trained weights (URLs, SHA-256), lineage notes, known gaps |
| `requirements/`, `environment.yml` | inference (`local.txt`) and training (`training-local.txt`) dependencies |
| `peta_file_mapping.txt` | PETA file mapping from the official starter kit (used by the configs) |

## 5. Data, pre-training and disclosures

* Fine-tuning used only the organizer-provided Market-1501, PA-100K and PETA annotations (42,736 persons).
  Data are not redistributed; obtain them with the official starter kit.
* Pre-training (permitted by the rules): COCO 2017; PoseTrack18; ImageNet-1K indirectly through the MMPose
  ViTPose-L COCO checkpoint. URLs and SHA-256 of all external checkpoints are listed in
  [`historical_appendix/EXTERNAL_ASSETS_20261012.md`](historical_appendix/EXTERNAL_ASSETS_20261012.md).
* Disclosure: part of our PoseTrack18 pre-training images were PoseTrack 2017 release frames matched to
  PoseTrack18 frame names and annotations; this affects lineages that were used as distillation teachers
  (teachers were run on organizer training crops only). We asked the organizers to confirm this route.
* No test images were used in any way (training, pseudo-labelling, adaptation or selection).

## 6. Known limitations

* The exact scored inference is fully reproducible from the archive. A byte-identical re-run of the complete
  historical multi-stage training is **not** certified: some intermediate checkpoints and the original
  weighted-soup command were not preserved, and the source code evolved during the challenge
  (see [`historical_appendix/GAPS_20261012.md`](historical_appendix/GAPS_20261012.md)).
  The final trainer-format member weights are provided instead.
* The Docker images were written for this release but could not be built on the author's machine (no Docker
  daemon access). The inference runner was tested outside Docker with PyTorch 2.4.1 on the exact archive;
  the archive itself ran in the official container on Codabench.
* No new licence is granted for the code or weights beyond the organizers' verification of this challenge
  entry; third-party notices (e.g. `LICENSE_MMPose.txt` inside the archive) remain unchanged.
