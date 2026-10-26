# Training the three members of submission 1006463

All paths are relative to the repository root. Each configuration under `artifacts/runs/**` is the configuration
that was actually used, with its original `output_dir`, `event_path`, seeds and `init_checkpoint`. To retrain a
stage, copy its JSON, point `output_dir`/`event_path` to a new folder and `init_checkpoint` to the parent you
produced (the trainer verifies `init_checkpoint_sha256`; update or remove it when you use a re-trained parent).

## 1. Environment

* Image: `docker build -f docker/train.Dockerfile -t upar-track3-train .` (PyTorch 2.7.0, CUDA 12.8), or a local
  environment from `requirements/training-local.txt`. No mmcv-full CUDA operators are needed.
* Hardware used: one NVIDIA RTX PRO 6000 (96 GB), bf16, gradient checkpointing; a ViT-L stage needs ≈15–22 GB.
* Entry point: `PYTHONPATH=src:scripts python scripts/train_rsn_domain_continuation.py --config CONFIG.json`
  (`--allocator-cap-gib N` limits GPU memory; `--audit` runs three batches as a smoke test).

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

Production configurations train on all 42,736 organizer persons (`fit_includes_heldout: true`); development
configurations (not needed to reproduce the submission) hold out the PETA cameras CAVIAR4REID, GRID, PRID, SARC3D
and i-LID.

## 3. Common recipe

* Square crop `max(w,h)×1.25` around the box, 256×256 input; ViT-L resamples to 384 inside the network
  (96×96 heatmaps), RSN uses 64×64 heatmaps; Gaussian heatmap targets (σ = 2 on the 64 grid).
* AdamW, weight decay 0.1, layer-wise decay 0.8, drop path 0.4 (ViT-L), gradient clipping 1.0, batch 32 (ViT)
  or 16 (RSN), 32k samples per epoch, bf16.
* Augmentation: flip, centre/scale jitter, brightness/contrast, person paste (p 0.35), factor synthesis,
  test-like truncation (`strong_augmentation`), camera/dataset-balanced sampling (`group_sampling_weights`).
* Knowledge-distillation stages (configs with a `distill` block): ground-truth-supported heatmap distillation
  (λ = 0.5) from the fixed teachers listed in the config, evaluated on organizer training crops.

## 4. Member lineages (production arm)

`R1 = artifacts/runs/20261006T003419+0900_resume_v53`, `R2 = artifacts/runs/20261007T1540+0900_gold_cycle_v58`,
`R3 = artifacts/runs/20261017T1842_gold_knobs_study`. Soup coefficients are recorded inside each soup file
(`epoch` field) and reproduced with `scripts/weighted_soup.py` (uniform soups: `scripts/build_vit_weight_soup.py`).

### RSN member (`FB_soup_e4to7_weights.pt`)
1. MMPose 3×RSN-50 COCO 256×192 checkpoint (external, see EXTERNAL_ASSETS) → Market-1501 + PA-100K fine-tune,
   24 epochs; the epoch-24 state (`R1/rsn_e24_init/last_checkpoint_e24.pt`) is the next init.
2. `R1/rsn_arm_FB.json`: all organizer data, 8 epochs, lr 6e-5.
3. Uniform average of epochs 4–7 (BatchNorm statistics averaged).

### ViT-L "L3L4" member (`soup_L3L4_v109_prod_weights.pt`)
1. Keypoint pre-training from the COCO-only MAE ViT-L: `R1/vit_pre_COCOPT_L/config.json` →
   `R1/vit_pre_COCOPT_L2/config.json` → `R2/vit_pre_COCOPT_L3/config.json` ("L3", epoch 20) and
   `R2/vit_pre_COCOPT_L4r1/config.json` ("L4", epoch 20).
2. Organizer fine-tunes (12 epochs, lr 1e-4, internal 384): `R2/vit_prod_VITLP3_384*.json` (L3 inits:
   `_TR` truncation, `_h07`; the two-seed L3 soup used by `vit_prod_L3soup_gen2KD.json` came from seed configs that
   are not all preserved) and `R2/vit_prod_VITLP4_384*.json` (L4 inits: seed 1, `_s2`, `_TR`, `_h07`).
3. Distillation chain (L3): `R3/vit_prod_L3_4mKD_vphs.json` → `R3/vit_prod_L3soup_gen2KD.json` →
   `R3/vit_prod_L3soup_ext6mKD.json` → `R3/vit_prod_L3soup_okskd8m.json`, `R3/vit_prod_L3soup_hard8_okskd.json`
   (→ intermediate soup `L3_v106_e12`) → `R3/vit_prod_L3soup_sahi_okskd.json` ("w7") →
   `R3/vit_prod_L3soup_gen4_sahi_okskd.json` ("w8"). Each config's `init_checkpoint` names its exact parent.
4. Soups: `soup_L3_v108 = 0.54·L3_v106_e12 + 0.135·w7 + 0.135·w8 + 0.10·h07 + 0.09·okskd8m_e2`;
   `soup_L4_v108 = 0.375·L4 + 0.375·L4_TR + 0.25·L4_h07`;
   **`soup_L3L4_v109 = 0.82·soup_L3_v108 + 0.18·soup_L4_v108`**.

### ViT-L "VP" member (`soup_VP_v108_prod_weights.pt`)
1. MMPose ViTPose-L COCO 256×192 checkpoint (ImageNet-1K MAE init) converted into our ViT-L graph with
   `scripts/convert_mmpose_vitpose.py`.
2. `R2/vit_prod_VPLs_384_TR.json` (12 epochs, truncation augmentation).
3. Distillation chain: `R3/vit_prod_VPLs_4mKD_vphs.json` → `R3/vit_prod_VPLs_gen2KD_r1.json` →
   `R3/vit_prod_VPLs_ext6mKD.json` → `R3/vit_prod_VPLs_hard8_okskd.json` (→ intermediate soup `VP_v106_7030`) →
   `R3/vit_prod_VPLs_sahi_okskd.json` ("w7") → `R3/vit_prod_VPLs_gen4_sahi_okskd.json` ("w8").
4. **`soup_VP_v108 = 0.45·VP_v106_7030 + 0.25·w7 + 0.30·w8`**.

The teacher models referenced by the distillation configs (RSN, L3, VP, VPLsPT, L4, ViTPose-H variants) are
themselves trained with the configs in `R2`/`R3`; their pre-training lineage is disclosed in the fact sheet.

## 5. Export and packaging

* TorchScript export (FP16, BatchNorm fused, run under PyTorch 2.4.1):
  `python -c "import sys; sys.path[:0]=['scripts','src']; import package_rsn_weights as p; from pathlib import Path; p.export(Path('WEIGHTS.pt'), Path('OUT_DIR'), True)"`
  writes `OUT_DIR/model_pose.torchscript`; rename it to `model_rsn|model_l3|model_vp.torchscript`.
* The submitted `run.py`, `model_meta.json` and `metadata.yaml` are inside the archive; the archive is a plain ZIP
  of these files with the three TorchScript members (`evidence/package_report_1006463.json` lists member hashes).

## 6. Not certified

The complete chain above was not re-executed for this release, several intermediate checkpoints were not
preserved, and the code evolved during the challenge, so a byte-identical retraining is not guaranteed (details in
`historical_appendix/GAPS_20261012.md`). The final member weights and the exact archive are provided.
