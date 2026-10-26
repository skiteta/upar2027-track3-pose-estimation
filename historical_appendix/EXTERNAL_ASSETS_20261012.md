# External weights and data — provenance table (2026-10-12)

Rule basis: Codabench Terms §3 ("Pre-training on COCO and PoseTrack18 is permitted. No other external pretraining data or
foundation models are allowed."; snapshot `artifacts/runs/20261005T215400+0900_rsn18_delivery_v50/preflight_001/competition.json`)
and the organiser clarification of 2026-10-10 (starter-kit Issue #1: ImageNet/ImageNet-1K, COCO and PoseTrack18
pre-training permitted, "the restriction applies to the complete pretraining history"; quoted in `.os_state/PROJECT_LOCAL_RULES.md`).

"Licence" = what the repo records. "not recorded" means the repo has no statement; **verify from the official source
before the release** (no network access was used for this draft). sha256 values are copied from the cited download
logs/manifests; "UNKNOWN" = never recorded. `R` = `artifacts/runs/20261007T1540+0900_gold_cycle_v58`.

## 1. External weights

| Asset | Used by | URL | Local path | Bytes | sha256 | Licence (as recorded) | Source of record |
|---|---|---|---|---|---|---|---|
| MMPose 3×RSN-50 COCO 256×192 (0.x, 20201127) | **H: RSN member** (init of `upar_rsn_v42`) | https://download.openmmlab.com/mmpose/top_down/rsn/3xrsn50_coco_256x192-58f57a68_20201127.pth | `artifacts/runs/20261005T025700+0900_mechanism_cycle_v42/rsn_assets/3xrsn50_coco.pth` | 238,847,445 | `58f57a688ca0eaae6f19d902e9f78522cf8e1db7ee74c677ae84ef35e4279561` | Apache-2.0 assumed (MMPose repo licence; "no weight-specific licence document found", survey report) | `…v42/rsn_assets/download_manifest.json`; history: embedded config `pretrained=None`, COCO train2017, 210 ep (`…v42/rsn/initialization.json`) |
| long_seq_mae ViT-L/16, COCO train2017+unlabeled2017, 800 ep × dup5, 448 px | **H: both ViT-L members** (init of `vit_pre_COCOPT_L`) | https://dl.fbaipublicfiles.com/long_seq_mae/pretrained_models/coco%2Bunlabeled_dup5/vitl_dec512d16h8b_800ep_img448_crop0.2-1.0_maskds2.pth | `data/external/coco_mae/vitl_coco_mae_448.pth` | 3,955,856,645 | `270bee3b98ca7c8a3108c2947a0d6f01a640708d57300b1f0b0ba37be1b75216` | CC BY-NC 4.0; repo archived | `artifacts/runs/20261007T002451+0900_pretraining_survey_v55/download_coco_mae.sh/.log`; `…/research_notes/…/gap_stronger_noimagenet.md` §2 (checkpoint `args`: `vit_init_ckpt=''`) |
| long_seq_mae ViT-B/16 (same family) | not in H (older ViT-B arm, Public 965475) | …/coco%2Bunlabeled_dup5/vitb_dec384d12h8b_800ep_img448_crop0.2-1.0_maskds2.pth | `data/external/coco_mae/vitb_coco_mae_448.pth` | 1,208,991,229 | `816522606f04ae65a78614b3f3eb6689b750021fffead2cbb8c8fe3e41c461a6` | CC BY-NC 4.0 | same log |
| MMPose ViTPose-L COCO 256×192 (210e, 20230314) | optional (ViTPose-L members) | https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/topdown_heatmap/coco/td-hm_ViTPose-large_8xb64-210e_coco-256x192-53609f55_20230314.pth | `data/external/in1k_coco_weights/` | 1,234,272,760 | `53609f55473dd4d26aa2e509911fc092a3e4d438e9eb75126be6f905b693c475` | not recorded (MMPose Apache-2.0 assumed) | `R/20261011T2010_in1k_coco_weights_download.log`; chain check `R/vitpose_in1k_coco/20261011T2000_provenance.md` |
| MMPose ViTPose-H COCO 256×192 | optional (ViTPose-H members) | …/td-hm_ViTPose-huge_8xb64-210e_coco-256x192-e32adcd4_20230314.pth | same dir | 2,548,954,167 | `e32adcd41ab0b0ef0b5bf3d167ddae7cdbd45fcf45e7f6a834815ef04d641f2b` | not recorded | same |
| MMPose copy of MAE ViT-L (IN-1K) | optional; provenance check only (init of the ViTPose COCO weights) | https://download.openmmlab.com/mmpose/v1/pretrained_models/mae_pretrain_vit_large_20230913.pth | same dir | 1,213,293,813 | `d34734793fa83833d93a7fe8999ce205f8450f45e0ce6afac9d5efcfa43e005d` | not recorded | same |
| MMPose copy of MAE ViT-H (IN-1K) | optional; provenance check | …/mae_pretrain_vit_huge_20230913.pth | same dir | 2,524,097,013 | `4ac018188e36cbe84254d906f3b89bb943a2c0cb4a296d8cf0e8c5c09f646ff2` | not recorded | same |
| Meta official MAE ViT-L (IN-1K) | optional; provenance check | https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_large.pth | same dir (`official_…`) | 1,213,314,565 | `d2ceff4892889d0c0f5418fa8bcb02e6a25a37fc3bdc0619c3cacd7d3d05c39f` | not recorded (verify facebookresearch/mae licence) | same |
| Meta official MAE ViT-H (IN-1K) | optional; provenance check | https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_huge.pth | same dir (`official_…`) | 2,523,202,725 | `2d0099a83c73e2b006095c90755458e7c9fd87513b720290b86c623881db42ca` | not recorded | same |
| Converted ViTPose inits (this project) | optional | produced by `scripts/convert_mmpose_vitpose.py` (sha256 now `6e264181…`) | `R/vitpose_in1k_coco/20261011T1950_vitpose{L,H}_coco_grid{16,24}.pt` | L grid24 1,235,862,500; H grid24 2,550,944,324 | in the sidecar `.json` (`out_sha256`; e.g. L grid24 `fa2ab8af…`) | derived | `R/vitpose_in1k_coco/` |

Not in any shipped lineage (listed so the fact sheet can say they were not used): LiteHRNet-30 COCO, Hourglass52 COCO,
detectron2 scratch Mask R-CNN (mask-teacher experiments), ViTPose+-S (rejected: trained on AIC/MPII/… —
`artifacts/runs/20261003T022500+0900_vitpose_plus_small_v1/pretraining_audit.json`). Downloaded code (not weights) used for the vendored RSN:
MMPose v0.29.0 `rsn.py` (`a61ca751…`), `3xrsn50_coco_256x192.py` (`49cc1a24…`), `topdown_heatmap_multi_stage_head.py`
(`645a5445…`), LICENSE (`aa2c6f34…`) — `…v42/rsn_upstream/download_manifest.json`, `…v42/rsn_assets/download_manifest.json`.

## 2. External data

| Data | Used by | URL / source | Local path | Size / count | sha256 | Licence (as recorded) | Approval / record |
|---|---|---|---|---|---|---|---|
| UPAR-Pose annotations (organiser) | all fine-tunes | starter kit `download_datasets.py` (organiser JSON bytes unchanged; local patch only hardens download) | `data/annotations/specialization/train_{market,pa100k,peta}.json` | 12,923 / 19,499 / 10,314 persons | `ceba8fbf353973756015977e8204187b633f6dccc0c68476115c3403a00cb3b4`, `b52f4e3d0143c5ae3426d27382bc7170fe025efbba18477ad9a2746f9863865c`, `3b8dbb27b5f7d0e8d064183c767431686e0df06a8a43107605214a6023a68810` (recomputed 2026-10-12; equal to the values recorded in RESULTS.md and v42 `frozen_evidence`) | CC BY-NC-SA 3.0 (Terms §6) | — |
| Market-1501 images | fine-tunes | Google Drive id `0B8-rUzbwVRk0c054eEozWG9COHM` (via `download_datasets.py`) | `data/market/` | 270 MB | UNKNOWN (archive) | "research only … DO NOT distribute" (`data/market/readme.txt`) | — |
| PA-100K images | fine-tunes | Google Drive folder `1d_D0Yh7C262gr0ef9EqkvG_M3fqgAWa2` | `data/pa100k/images/` | 100,000 files | UNKNOWN | non-commercial research (Terms §6) | — |
| PETA images | fine-tunes | https://www.dropbox.com/s/52ylx522hwbdxz6/PETA.zip?dl=1 | `data/peta/images/` | 19,000 files | UNKNOWN | non-commercial research (Terms §6) | — |
| `peta_file_mapping.txt` (PETA original path → renamed file; defines the 10 PETA cameras) | fine-tunes, dev split | starter kit | repo root | 19,000 lines | `a33d1cc7ac826c07d08a35df682690b224088e9138c4a0bfc11a6d522201ccf8` (computed now; no earlier record) | starter kit | — |
| COCO 2017 train person keypoints | ViT keypoint pre-training (RSN only via MMPose) | http://images.cocodataset.org/annotations/annotations_trainval2017.zip, http://images.cocodataset.org/zips/train2017.zip | `data/external/coco/annotations/person_keypoints_train2017.json`, `data/external/coco/train2017/` | 238,884,731 B; 118,287 images; 123,202 persons used (≥ 6 joints, box h ≥ 32) | UNKNOWN (zip deleted after extraction; json > 200 MB never hashed) | not recorded (verify COCO terms: annotations CC BY 4.0, images Flickr terms) | Terms §3; `…v55/download_coco_keypoints.sh/.log` |
| COCO unlabeled2017 images | only inside the long_seq_mae checkpoint (not downloaded by us) | — | — | — | — | — | disclose in the fact sheet |
| PoseTrack18 annotations (MMPose-converted, train + val) | ViT keypoint pre-training | https://download.openmmlab.com/mmpose/datasets/posetrack18_annotations.tar | `data/external/posetrack18/annotations/posetrack18_{train,val}.json` | tar 746,290,176 B | tar `094a0848319519ea89b593c8f1c091457e4bda2af3b02df940129aa13aae0b46` (archived to `/mnt/ARCHIVE/…storage_cleanup_v56…`); val json `c8be3088…` | official terms not found (site offline) | owner approval `artifacts/runs/20261006T003419+0900_resume_v53/posetrack18_user_approval.json` |
| PoseTrack18 frames (official tar parts aa/ab only, ≈ 12 % of videos) | ViT keypoint pre-training (17,348 persons, 113 videos) | figshare mirror https://ndownloader.figshare.com/files/57099065 (`2018.zip`, Hyper.AI upload) | `data/external/posetrack18/images/train/` | 10,007,989,641 B zip; 18,656 jpg | parts aa `e75191d2939ea5f3b5aa24e84ce14cfec0e2415e57adcba3a28e9815e3a702f1`, ab `2195a24ee737a2204f283e92239b21e77fab24aa07e7b9dc120f1cca3289f92c`; zip UNKNOWN | mirror self-asserts CC BY 4.0 (`…/within_rules_strategies.md`) | `…v53/posetrack18_download.sh/.log`; v56 container journal |
| PoseTrack 2017 frames (MPI-INF), mapped to PT18 names | **optional** stage L4x only | https://datasets.d2.mpi-inf.mpg.de/posetrack/ (`posetrack_v0.75_labels.tar`, `posetrack_data_images.tar.batch00…07`) | raw `data/external/posetrack17_mpi/raw/`; frames `…/frames/`; hard links `data/external/posetrack18_from_pt17/` | 66,304,000 + 7 × 5,000,000,000 + 550,535,680 B | labels `d4f810daf4752a3d36ba5c45350a11c1bb2033a973e9e703ea220f74143851d6`; batch00–07 `157e151c…f4ec`, `b3f122eb…2e9f`, `a3f75330…280d`, `f6f9ee6d…6bec`, `b11ebe75…7ab4`, `18ed7345…a77a`, `9b683031…f847`, `a3780c9e…22ea` (full values in `R/20261011T1940_posetrack17_mpi_download.log`) | CC BY-NC 4.0 "shown on posetrack.net (2019 archive)" | owner approval quoted in `R/20261011T1950_posetrack17_mpi_provenance.md` (not in `.os_state/PROJECT_LOCAL_RULES.md`) |
| ImageNet-1K | **optional** only, inside the MAE IN-1K weights | — | — | — | — | — | organiser Issue #1 (2026-10-10) + owner approval 2026-10-11 (`PROJECT_LOCAL_RULES.md`) |

## 3. Software

| Component | Version | Licence | Where |
|---|---|---|---|
| Python | 3.12.3 (`.venv`, `.venv-gpu`) | PSF | `pyvenv.cfg` |
| PyTorch (training, RSN export) | 2.7.0+cu128, CUDA 12.8, cuDNN 9.7.1 | BSD-3 | `.venv-gpu` |
| PyTorch (ViT export, CPU smoke, closest to the container) | 2.4.1+cpu; torchvision 0.19.1+cpu | BSD-3 | `.venv`, `requirements/torch-cpu.txt` |
| numpy / Pillow / pycocotools | 1.26.4 / 10.2.0 / 2.0.10 | BSD / HPND / BSD | both venvs; `requirements/local.txt` |
| mmcv (lite, pure Python), torchvision (GPU side), opencv-python-headless, addict, yapf, PyYAML … | 1.7.2, 0.22.0, 4.10.0.84, 2.4.0, 0.40.2, 6.0.2 | Apache-2.0, BSD, Apache-2.0, MIT, Apache, MIT | `artifacts/runs/20261003T185000+0900_strict_followup_v19/isolated_deps` (pip `--target` style; creation command UNKNOWN) |
| MMPose (vendored, not installed) | v0.29.0 RSN/head code in `src/upar_pose/rsn_upstream.py`, `rsn_adapter.py` (+ LiteHRNet/Hourglass upstream copies) | Apache-2.0 (`src/upar_pose/LITEHRNET_LICENSE` = MMPose LICENSE `aa2c6f34…`) | `src/upar_pose/*_upstream.py` |
| Evaluation container | `pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime` (Python 3.11, torch 2.4.1) | — | Codabench page; `run.py` imports only stdlib, numpy, torch, PIL |
