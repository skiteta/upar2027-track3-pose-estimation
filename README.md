# UPAR Challenge 2027 — Track 3 (2D Human Pose Estimation): code for the final Codabench submission

Participant: Syouya Tobita (Codabench `syouya_tobita`, individual participant).

This repository contains the inference code, training code, configurations and checksums needed to reproduce our
**final Codabench submission** (the single submission allowed in the re-opened final window):

| Item | Value |
|---|---|
| Codabench submission | **[final submission ID to be added after submission]** |
| Submitted archive | `upar_track3_final_candidate.zip`, 1,260,249,633 bytes, SHA-256 `83ffd518454828d8a9ca264b4b3f08d8b33563cd1bb713245b398ea90709dca1` |
| Internal model type | `upar_fusion3_sahi_anat9_v118` (`run.py` SHA-256 `90bd6971…2309`), export revision "v127" |
| Official result | **[final score to be added after scoring]** |
| Earlier leaderboard entry | 1006463 (`upar_fusion3_sahi_anat9_v109.zip`, AP 0.886743); see §6 |

The method is a three-model top-down heatmap fusion (3×RSN-50 + two ViT-L/16) with per-joint anatomical fusion,
gated pose-guided zoom views ("pose-SAHI"), adaptive re-cropping of loose single-person boxes and OKS-aware decoding.
Both ViT-L members end with a ground-truth-only loose-box fine-tuning stage ("LB3") run with two seeds whose
weights are averaged. The method is described in the fact sheet that accompanies this repository; the exact
inference code is the `run.py` inside the submitted archive.

## 1. Trained weights

The weights are too large for the repository and are hosted at: **WEIGHTS-URL-TO-BE-FILLED**

| File | Size | SHA-256 | Purpose |
|---|---|---|---|
| `upar_track3_final_candidate.zip` | 1.26 GB | `83ffd518…dca1` | the exact submitted archive (`run.py`, `model_meta.json`, 3 FP16 TorchScript members) |
| `soup_L3_LB3_s12_prod.pt` | 1.24 GB | `ae52299b…ea66` | trainer-format weights of the ViT-L "L3" member (uniform soup of 2 LB3 seeds) |
| `soup_VP_LB3_s12_prod.pt` | 1.24 GB | `08267f03…db67` | trainer-format weights of the ViT-L "VP" member (uniform soup of 2 LB3 seeds) |
| `FB_soup_e4to7_weights.pt` | 0.24 GB | `3728055d…08992` | trainer-format weights of the RSN member (unchanged since 1006463) |

Full checksums: [`weights/SHA256SUMS`](weights/SHA256SUMS). Verify after download with
`cd weights && sha256sum -c SHA256SUMS`.

Archive members (also checked by `run.py` at load time):

| File | SHA-256 | Exported from |
|---|---|---|
| `model_rsn.torchscript` | `9df1ea68…2736` | `FB_soup_e4to7_weights.pt` (same file as in 1006463) |
| `model_l3.torchscript` | `d1570a01…da02` | `soup_L3_LB3_s12_prod.pt` |
| `model_vp.torchscript` | `b64527f2…aaf7` | `soup_VP_LB3_s12_prod.pt` |

## 2. Reproduce the Codabench inference (Docker, recommended)

The submission runs in the official evaluation image `pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime`
(Python 3.11, PyTorch 2.4.1, NumPy, Pillow; no network access). The provided `Dockerfile` uses that image and
adds `inference/run_codabench_layout.py`, which reads the Codabench input layout and calls the unmodified
`run.py` exactly like the official ingestion program (`load_model()` once, then `predict_image(sample)` per image).

```bash
# 1) unpack the exact submitted archive (run.py must be at the root of this folder)
mkdir -p submission && unzip weights/upar_track3_final_candidate.zip -d submission

# 2) input in the Codabench layout:
#    input_data/images/<file_name>   and   input_data/boxes.json  (COCO images + person boxes)

# 3) build and run
docker build -t upar-track3-final .
docker run --rm --gpus all \
    -v "$PWD/submission:/submission" \
    -v "$PWD/input_data:/input_data:ro" \
    -v "$PWD/output:/output" \
    upar-track3-final
# -> output/predictions.json : COCO keypoint results, one entry per box (image_id, box_id, keypoints[51], score)

# 4) optional: score against a COCO keypoint ground-truth file (run once per hidden subset and average the APs)
docker run --rm -v "$PWD:/work" --entrypoint python upar-track3-final \
    /opt/upar/inference/evaluate_keypoints.py --gt /work/ground_truth.json --pred /work/output/predictions.json
```

Without Docker, any Python ≥ 3.10 environment with `torch==2.4.1`, `numpy` and `Pillow` works:
`python inference/run_codabench_layout.py --submission submission --input input_data --output predictions.json`
(add `--limit 10` for a quick smoke test; relative paths are resolved from the calling directory). The archive
selects CUDA when available and falls back to the CPU; on the CPU its first four images always take the zoom path
(a built-in smoke-test switch), so a CPU smoke test exercises that path too. A CPU smoke test of the exact archive
is recorded in [`evidence/smoke_cpu_final_v127.txt`](evidence/smoke_cpu_final_v127.txt).

### Important: the executed path depends on the speed of the machine

`run.py` contains a wall-clock controller designed for the documented limit of 7,200 s for 30,245 images (it never
looks at image content or predictions, only at elapsed time and measured paces):

* every 80 images (after the first 160) the projected total time moves the zoom tier up by one (projection
  ≥ 6,600 s) or down by one (< 6,250 s): tier 0 = RSN and VP zoom views, tier 1 = RSN zoom views, tiers 2–3 = none;
* if the path without zoom views is still projected above 7,000 s (checked after ≥ 480 images), it measures a
  centre-padding-only mode for 80 images and allocates the remaining images between the full and the centre mode
  from the measured paces; if even the centre mode is too slow, it drops the member with the largest sampled
  forward time and measures again;
* the adaptive re-crop of loose single boxes stops after 6,400 s or when the projection exceeds 6,700 s;
* an emergency mode at 7,120 s keeps only the fastest member with centre padding.

All decisions are printed to stdout (`[submission] proportional SAHI controller …`, `measured deadline allocation …`),
so the official ingestion log shows which path ran. On a faster or slower machine the same unmodified archive takes
a different path, so its predictions — and usually its AP — differ from the scored ones. We do not modify the
archive to imitate the server timing.

## 3. Training

See [`TRAINING.md`](TRAINING.md): data layout, environment (`docker/train.Dockerfile`), the trainer
`scripts/train_rsn_domain_continuation.py` (the exact file used for the final LB3 runs, SHA-256 `7689ff95…7b101`),
the configuration of every stage of the three member lineages (`artifacts/runs/**`), the weight-soup recipes,
the TorchScript export and the structural attention export used for the final archive.

## 4. Repository layout

| Path | Content |
|---|---|
| `inference/` | offline Codabench-layout runner and COCO keypoint evaluation |
| `Dockerfile`, `docker/train.Dockerfile` | inference image (official base) and training image |
| `src/upar_pose/` | model definitions (ViT-L pose graph, vendored RSN), data, evaluation utilities |
| `scripts/` | trainer, augmentation, soups, TorchScript export, static-attention export and packaging tools |
| `artifacts/runs/**` | the training configurations of all stages (including the LB3 seed runs), with their original paths and seeds; soup and export logs of the final members |
| `evidence/` | final archive package report, static-attention export and parity reports, CPU smoke test; official scoring/runtime logs of the earlier entry 1006463; saved competition rules |
| `historical_appendix/` | provenance of external pre-trained weights (URLs, SHA-256), lineage notes, known gaps (written 2026-10-12) |
| `requirements/`, `environment.yml` | inference (`local.txt`) and training (`training-local.txt`) dependencies |
| `peta_file_mapping.txt` | PETA file mapping from the official starter kit (used by the configs) |

## 5. Data, pre-training and disclosures

* Fine-tuning used only the organizer-provided Market-1501, PA-100K and PETA annotations (42,736 persons).
  Data are not redistributed; obtain them with the official starter kit.
* Pre-training (permitted by the rules): COCO 2017; PoseTrack18; ImageNet-1K indirectly through the MMPose
  ViTPose-L COCO checkpoint. URLs and SHA-256 of all external checkpoints are listed in
  [`historical_appendix/EXTERNAL_ASSETS_20261012.md`](historical_appendix/EXTERNAL_ASSETS_20261012.md).
* Disclosure: part of our PoseTrack18 pre-training images were PoseTrack 2017 release frames matched to
  PoseTrack18 frame names and annotations; this affects lineages that were used as distillation teachers of the
  parent soups of both ViT members (teachers were run on organizer training crops only; the final LB3 stage uses
  no teachers). We ask the organizers to confirm this route.
* A sample of the PoseTrack18 validation set (1,032 persons) was used as a development diagnostic (evaluation only)
  when choosing the loose-box augmentation strength. It is not independent of the teacher lineages above.
* No test images were used in any way (training, pseudo-labelling, adaptation or selection).

## 6. Earlier leaderboard entry 1006463

The previous commit of this repository (`93ea8fb`) describes submission 1006463 (archive
`upar_fusion3_sahi_anat9_v109.zip`, SHA-256 `b2210cf1…6375`, AP 0.886743, AP50 0.965086, AP75 0.925566,
AP_medium 0.857837, AP_large 0.679830 in the official scoring log; the organizers' snapshot lists AP_medium and
AP_large as 0). Its evidence files are kept in `evidence/` (`*_1006463*`). In its scored run the older runtime
guards of that archive reduced most images to the RSN member (`evidence/runtime_1006463.txt`). The final archive
keeps the same RSN file, replaces both ViT members, and replaces those guards by the controller of §2.

## 7. Known limitations

* The exact scored inference is reproducible from the archive. A byte-identical re-run of the complete
  historical multi-stage training is **not** certified: several intermediate checkpoints of the parent lineages
  and the commands that wrote the weighted parent soups were not preserved, GPU training is not bitwise
  deterministic, and the source code evolved during the challenge (see `TRAINING.md` §6 and
  [`historical_appendix/GAPS_20261012.md`](historical_appendix/GAPS_20261012.md)). The final trainer-format member
  weights are provided instead. The last stage (LB3, two seeds) and its soups were run with the trainer and soup
  script contained in this repository.
* The Docker images were written for this release but could not be built on the author's machine (no Docker
  daemon access). The inference runner was tested outside Docker with PyTorch 2.4.1 on the CPU with the exact
  archive (6 images, `evidence/smoke_cpu_final_v127.txt`). Completion of all 30,245 images within 7,200 s on the
  official hardware was not tested locally.
* Development numbers quoted in the fact sheet come from a reused five-camera PETA development set and are not an
  estimate of the leaderboard score.
* No new licence is granted for the code or weights beyond the organizers' verification of this challenge
  entry; third-party notices (e.g. `LICENSE_MMPose.txt` inside the archive) remain unchanged.
