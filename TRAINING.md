# Training the three members of the final submission

The final archive v154 uses the same trained members as export revision v127 and differs only in inference code
(§5).

All paths are relative to the repository root. Each configuration under `artifacts/runs/**` is the configuration
that was actually used, with its original `output_dir`, `event_path`, seeds and `init_checkpoint`. To retrain a
stage, copy its JSON, point `output_dir`/`event_path` to a new folder and `init_checkpoint` to the parent you
produced (the trainer verifies `init_checkpoint_sha256`; update or remove it when you use a re-trained parent).
A `<run>/config.json` next to a `<run>.json` is the trainer's resolved copy (with `config_sha256` and the training
population counts); the top-level `<run>.json` is the file that was passed to the trainer.

## 1. Environment

* Image: `docker build -f docker/train.Dockerfile -t upar-track3-train .` (PyTorch 2.7.0, CUDA 12.8), or a local
  environment from `requirements/training-local.txt`. No mmcv-full CUDA operators are needed.
* Hardware used: one NVIDIA RTX PRO 6000 (96 GB, shared with other jobs), bf16, gradient checkpointing; a ViT-L
  stage needs ≈15–22 GB.
* Entry point: `PYTHONPATH=src:scripts python scripts/train_rsn_domain_continuation.py --config CONFIG.json`
  (`--allocator-cap-gib N` limits GPU memory; `--audit` runs three batches as a smoke test).
* The four final LB3 runs recorded trainer SHA-256 `7689ff956fb45e25099ccdc24e7484c794caa512fed4eeaa7606ec15d687b101`
  in their execution records; this is the `scripts/train_rsn_domain_continuation.py` of this repository.

## 2. Data layout

```
data/annotations/specialization/train_market.json   # official starter kit annotations
data/annotations/specialization/train_pa100k.json
data/annotations/specialization/train_peta.json
data/<file_name of each image>                       # e.g. data/market/bounding_box_train/....jpg
peta_file_mapping.txt                                # official PETA mapping (camera domains)
data/external/...                                    # pre-training data / checkpoints (see below)
```

Pre-training data and checkpoints (paths expected by the configs, URLs and SHA-256):
[`historical_appendix/EXTERNAL_ASSETS_20261012.md`](historical_appendix/EXTERNAL_ASSETS_20261012.md).

Production configurations train on all 42,736 organizer persons (`fit_includes_heldout: true`; the
`heldout_subdomains` list is then only used for a train-diagnostic evaluation); development configurations (not
needed to reproduce the submission) hold out the PETA cameras CAVIAR4REID, GRID, PRID, SARC3D and i-LID.

## 3. Common recipe

* Square crop `max(w,h)×1.25` around the box, 256×256 input; ViT-L resamples to 384 inside the network
  (96×96 heatmaps), RSN uses 64×64 heatmaps; Gaussian heatmap targets (σ = 2 on the 64 grid).
* AdamW, weight decay 0.1, layer-wise decay 0.8, drop path 0.4 (ViT-L), gradient clipping 1.0, batch 32 (ViT)
  or 16 (RSN), 32k samples per epoch, bf16.
* Augmentation: flip, centre (±4 %) and scale (0.9–1.1) jitter, brightness/contrast (0.8–1.2), person paste
  (p 0.35), factor synthesis, test-like truncation (`strong_augmentation`), camera/dataset-balanced sampling
  (`group_sampling_weights`: Market 0.25, PA-100K 0.25, each of the ten PETA cameras 0.05).
* Knowledge-distillation stages (configs with a `distill` block): ground-truth-supported heatmap distillation
  (λ = 0.5) from the fixed teachers listed in the config, evaluated on organizer training crops.
* **LB3 (final stage of both ViT members):** ground-truth only (no `distill` block), 2 epochs × 32k samples,
  learning rate 5e-6 (L3: `warmup_mode: linear_steps`, i.e. linear warm-up over the first epoch; VP: the legacy
  epoch warm-up, which is a no-op for epoch ≥ 1, so constant), `strong_augmentation` with truncation (p 0.15,
  keep 0.45–0.85; side cut p 0.2, keep 0.55–0.85) plus a **loose-box window scale** (`scale_prob` 0.8,
  `scale_range` [1.05, 2.2]; > 1 enlarges the crop window). Seeds: seed 1 `seed` 20261005 / `data_seed` 20261903,
  seed 2 `seed` 20261105 / `data_seed` 20261913 (both `strong_augmentation.seed` 20261977). The `init_note` text
  of the VP LB3 configs was copied from an older config and is not accurate; `init_mode: weights` (weights only,
  fresh optimizer) is what the trainer used (see `initialization.json` of the runs).

## 4. Member lineages (production arm)

`R1 = artifacts/runs/20261006T003419+0900_resume_v53`, `R2 = artifacts/runs/20261007T1540+0900_gold_cycle_v58`,
`R3 = artifacts/runs/20261017T1842_gold_knobs_study`, `R4 = artifacts/runs/20261022T2030_loosebox_ft`,
`R5 = artifacts/runs/20261023T1225_runtime_candidate_selection`. Soup files record their parents (path, SHA-256,
coefficient) in the `soup_parents` field and the recipe in the `epoch` field. Weighted soups are reproduced with
`scripts/weighted_soup.py`; uniform soups with `scripts/build_vit_weight_soup.py`.

### RSN member (`FB_soup_e4to7_weights.pt`, SHA-256 `3728055d…08992`; unchanged since 1006463)
1. MMPose 3×RSN-50 COCO 256×192 checkpoint (external, see EXTERNAL_ASSETS) → Market-1501 + PA-100K fine-tune,
   24 epochs; the epoch-24 state (`R1/rsn_e24_init/last_checkpoint_e24.pt`) is the next init.
2. `R1/rsn_arm_FB.json`: all organizer data, 8 epochs, lr 6e-5.
3. Uniform average of epochs 4–7 (BatchNorm statistics averaged).

### ViT-L "L3" member (`soup_L3_LB3_s12_prod.pt`, SHA-256 `ae52299b…ea66`)
1. Keypoint pre-training from the COCO-only MAE ViT-L: `R1/vit_pre_COCOPT_L/config.json` →
   `R1/vit_pre_COCOPT_L2/config.json` → `R2/vit_pre_COCOPT_L3/config.json` ("L3", epoch 20, `32aaf8d8…`).
2. Organizer fine-tunes from L3 e20 (12 epochs × 32k, lr 1e-4 → 3e-5 from epoch 9, internal 384):
   `R2/vit_prod_VITLP3_384_TR.json` (truncation augmentation; "L3_TR"), and two seeds without truncation
   `R2/vit_prod_VITLP3_384.json` + `R2/vit_prod_VITLP3_384_s2.json` → uniform soup
   `R2/vit_prod_VITLP3_384_soup2/soup_weights.pt` (`39675e76…`).
3. Distillation: `R3/vit_prod_L3_4mKD_vphs.json` (init L3_TR e12, lr 1e-5, 2 epochs) and
   `R3/vit_prod_L3soup_gen2KD.json` (init soup2, lr 8e-6, 2 epochs) → intermediate soup `soup_L3_v103a`
   = 0.425/0.425·4mKD e1/e2 + 0.075/0.075·gen2KD e1/e2 (`4317f9d7…`) → `R3/vit_prod_L3soup_ext6mKD.json`
   (init `soup_L3_v103a`, lr 5e-6, 2 epochs, six teachers).
4. Parent soup `R3/vit_prod_L3soup_ext6mKD/soup_L3_v104_prod_weights.pt` (`f30f86b7…`)
   = 0.30·4mKD e1 + 0.30·4mKD e2 + 0.05·gen2KD e1 + 0.05·gen2KD e2 + 0.15·ext6mKD e1 + 0.15·ext6mKD e2.
5. LB3, seed 1: `R4/vit_prod_L3soup_LB3.json`; seed 2: `R5/prod_L3_LB3s2.json` (both init from the parent soup).
6. Uniform soup of the two epoch-2 weights (`R5/soup_L3.log` records parents and output hash):
   `python scripts/build_vit_weight_soup.py --weights R4/vit_prod_L3soup_LB3/epoch_002_weights.pt
   R5/prod_L3_LB3s2/epoch_002_weights.pt --output soup_L3_LB3_s12_prod.pt` → `ae52299b…`.

### ViT-L "VP" member (`soup_VP_LB3_s12_prod.pt`, SHA-256 `08267f03…db67`)
1. MMPose ViTPose-L COCO 256×192 checkpoint (ImageNet-1K MAE init) converted into our ViT-L graph with
   `scripts/convert_mmpose_vitpose.py` (`R2/vitpose_in1k_coco/20261015T2035_vitposeL_coco_grid24_L3struct.pt`,
   `405a711e…`).
2. `R2/vit_prod_VPLs_384_TR.json` (12 epochs, truncation augmentation; "VPLs_TR").
3. Distillation: `R3/vit_prod_VPLs_4mKD_vphs.json` (init VPLs_TR `last_checkpoint.pt` incl. AdamW state, lr 1e-5,
   3 epochs) → intermediate soup `soup_VPLs_35base_65kd23` = 0.35·VPLs_TR e12 + 0.65·mean(4mKD e2, e3)
   (`d4b2c431…`) → `R3/vit_prod_VPLs_gen2KD_r1.json` (lr 6e-6, 2 epochs) → intermediate soup `soup_VP_v103a`
   = 0.30/0.275/0.275/0.15 of VPLs_TR e12, 4mKD e2, 4mKD e3, gen2KD e1 (`215b451d…`) →
   `R3/vit_prod_VPLs_ext6mKD.json` (lr 5e-6, 2 epochs, six teachers).
4. Parent soup `R3/vit_prod_VPLs_ext6mKD/soup_VP_v104_prod_weights.pt` (`dfcddb05…`)
   = 0.225·VPLs_TR e12 + 0.20625·4mKD e2 + 0.20625·4mKD e3 + 0.1125·gen2KD e1 + 0.125·ext6mKD e1 + 0.125·ext6mKD e2.
5. LB3, seed 1: `R4/vit_prod_VPLs_LB3.json`; seed 2: `R5/prod_VP_LB3s2.json`.
6. Uniform soup of the two epoch-2 weights (`R5/soup_VP.log`) → `08267f03…`.

The teacher models referenced by the distillation configs (RSN, L3, VP, VPLsPT, VITLP2_448 and ViTPose-H variants,
and soups of them) are themselves trained with the configs in `R2`/`R3`; their pre-training lineage is disclosed in
the fact sheet (some were pre-trained with PoseTrack 2017 frames matched to PoseTrack18). The final LB3 stage uses
no teacher. The two parent soups (`f30f86b7…`, `dfcddb05…`) are not part of the weight upload because of their size.

## 5. Export and packaging

* TorchScript export (FP16, BatchNorm fused, run under PyTorch 2.4.1):
  `python -c "import sys; sys.path[:0]=['scripts','src']; import package_rsn_weights as p; from pathlib import Path; p.export(Path('WEIGHTS.pt'), Path('OUT_DIR'), True)"`
  writes `OUT_DIR/model_pose.torchscript`. For the final ViT members this produced `32408c7e…` (L3) and
  `065dea54…` (VP) (`R5/export_L3.log`, `R5/export_VP.log`).
* Structural attention export (final archive only): `python scripts/rewrite_static_attention_dim.py --source
  model_pose.torchscript --output model_l3.torchscript --report report.json` replaces the traced
  `channels // heads` scalar division in the 24 attention blocks by the constant head width 64 (removes a
  GPU-to-CPU synchronisation); it asserts that all 310 learned/state tensors are bit-identical. This gives the
  archive members `d1570a01…` (L3) and `b64527f2…` (VP) (`evidence/static_attention_export_report_v127.json`;
  development parity: `evidence/static_attention_dev_parity_v127.json`). The RSN member `model_rsn.torchscript`
  (`9df1ea68…`) is the file used since the earlier submissions.
* The submitted `run.py`, `model_meta.json` and `metadata.yaml` are inside the archive; the archive is a plain ZIP of
  these files, `LICENSE_MMPose.txt` and the three TorchScript members (`evidence/package_report_final_v154.json`
  lists member hashes; `evidence/package_report_final_v127.json` those of export revision v127).
* Final archive v154 (`upar_track3_final_candidate_v154.zip`, SHA-256 `82a5f53e…69a4`): no retraining and no new
  export. `inference/v154_build/build_v154.py` copies the v127 archive members unchanged (the three TorchScript files
  above, `LICENSE_MMPose.txt`, `metadata.yaml`), replaces `run.py` by the v127 `run.py` patched with
  `inference/v154_build/patch_v153.py` (tier-gated four-padding scale TTA replacing pose-SAHI) and then
  `inference/v154_build/patch_v154.py` (confidence-preserving fusion), and appends a revision note to
  `model_meta.json`. Usage: `python patch_v153.py run_v127.py run_v153.py` and
  `python patch_v154.py run_v153.py run_v154.py`; the result has SHA-256 `2c37093d…7f2a`, the `run.py` inside the
  final archive (see `evidence/v154_inference_change_and_checks.txt`).

## 6. Not certified

The complete chain above was not re-executed for this release. Several intermediate checkpoints of the parent
lineages were not preserved (e.g. the ViT distillation epoch weights that the parent soups average; the soups
themselves are kept), the commands that wrote the weighted parent soups were not recorded (the coefficients are
recorded in the soup files), one L3 seed run was resumed after a crash (`R2/vit_prod_VITLP3_384_s2`), GPU training
is not bitwise deterministic, and the code evolved during the challenge (details in
`historical_appendix/GAPS_20261012.md`). A byte-identical retraining is therefore not guaranteed. The final member
weights and the exact archive are provided.
