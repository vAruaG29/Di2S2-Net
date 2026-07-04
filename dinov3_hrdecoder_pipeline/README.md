# `dinov3_hrdecoder_pipeline/` — the model package

A CLI-driven PyTorch Lightning pipeline for **7-class semantic segmentation
of aerial / drone orthoimagery** (Background + Built_Up_Area, Road,
Water_Body, Utility, Bridge, Railway). A **DINOv3 ViT-L/16 (SAT-493M)**
encoder feeds an **HRDecoder** (two-pass low-res + high-res fusion); inputs
are 1024×1024 RGB tiles cut from raw orthomosaics, outputs are per-tile
masks, a stitched full-extent raster, per-class metrics, and a multi-layer
GeoPackage.

This is the code-only package. Data, checkpoints and results live on Google
Drive — see the [repo-root README](../README.md) §0.

---

## The five sub-packages

Each directory has its **own README** with the exact CLI, every argparse
flag, the config keys it reads, and its inputs/outputs.

| Sub-package | What it does | README |
|---|---|---|
| [`configs/`](configs/) | The **single source of truth** for all paths, classes and hyper-parameters (3 YAMLs). No paths are hard-coded. | [configs/README.md](configs/README.md) |
| [`data_prep/`](data_prep/) | Raw rasters → organised dataset → COG → 1024² tiles → rasterised label masks. Run once per data drop. | [data_prep/README.md](data_prep/README.md) |
| [`models/`](models/) | Encoder, HRDecoder (+ 3 alternative decoders), losses, metrics, tile Dataset/DataModule, Lightning module. | [models/README.md](models/README.md) |
| [`training/`](training/) | Three training entrypoints: single-holdout `train`, two-holdout `train_2val`, full-data `train_full` (the submission model). | [training/README.md](training/README.md) |
| [`inference/`](inference/) | Batch infer → memory-safe stitch → evaluate → vectorise to GeoPackage, plus the portal-consumed timing hooks. | [inference/README.md](inference/README.md) |
| [`visualization/`](visualization/) | Stage-wise matplotlib visualisations (data / DINOv3 features / predictions / stitched). | [visualization/README.md](visualization/README.md) |

---

## The dataflow, end to end

![End-to-end dataflow: data prep (convert_to_cog, prepare_labels, tile_raster) → training phase (SegDataModule, DINOv3HRDecoderModule → best checkpoint) → inference phase (batch inference → prediction GeoTIFFs → stitched raster → evaluation report + GeoPackage)](../docs/images/Dataflow_end_to_end.png)

Every stage reads its paths and hyper-parameters from a
[`configs/`](configs/) YAML — nothing is hard-coded, so moving the bundle
only requires rerunning `setup_env.sh` (or one `sed`; see the repo-root
README §6).

---

## Running it

All commands run from the **repo root** with `conda activate svamitva2`
(env created by `../setup_env.sh`). The one-liner:

```bash
# Batch inference over every dataset in tiles/tile_index.csv
python -m dinov3_hrdecoder_pipeline.inference.run_pipeline \
    --checkpoint pretrained/dinov3_hrdecoder_full_best_loss=0.0615.ckpt
```

See each sub-package README for the full stage-by-stage workflow, and the
[repo-root README](../README.md) §5 for the canonical A→E walkthrough.

---

## Feature classes

| ID | Class          | Colour (RGB)  |
|----|----------------|---------------|
| 0  | Background     | —             |
| 1  | Built_Up_Area  | (255, 0, 0)   |
| 2  | Road           | (0, 255, 0)   |
| 3  | Water_Body     | (0, 0, 255)   |
| 4  | Utility        | (255, 255, 0) |
| 5  | Bridge         | (255, 0, 255) |
| 6  | Railway        | (0, 255, 255) |

> **Data-coverage caveat:** in the 4,513-tile labelled corpus, **Railway has
> zero ground truth** and **Bridge appears in only 12 tiles (0.0039 % of
> pixels)**. Both score IoU ≈ 0 by data coverage, not model failure. The
> 7-class head still reserves IDs 5/6, so adding labels later needs no
> architecture change. Present-class (5-class) mIoU is **0.905** in-sample;
> 7-class mIoU is **0.647**. Regenerate the class distribution with
> [`scripts/analyze_dataset_stats.py`](../scripts/analyze_dataset_stats.py); the
> full writeup is in [`docs/documentation.md`](../docs/documentation.md).
