# `configs/` — the single source of truth

Every path, class list, and hyper-parameter used by the pipeline lives in
one of these YAMLs. **Nothing is hard-coded in the Python code** — scripts
take `--config <path>` (defaulting to the file listed below) and read paths
and settings from it.

| File | Default consumer(s) | Purpose |
|---|---|---|
| `data_prep.yaml` | all of `data_prep/*` (except `prepare_masks.py`) | Data-prep paths, COG settings, tiling geometry, shapefile→class map, vectorisation defaults, visualisation colours. |
| `train.yaml` | `training/train.py`, `models/*`, all of `inference/*` (default), `visualization/visualize.py`, `data_prep/prepare_masks.py` | Training config with a **single geographic hold-out** (`test_dataset`) used as validation. |
| `train_2val.yaml` | `training/train_2val.py` | Same as `train.yaml` but **two** hold-out datasets (`test_datasets`) and isolated output dirs (`checkpoints_2val/`, `logs_2val/`, `visualizations_2val/`). |
| `train_full.yaml` | `training/train_full.py`, `inference/run_pipeline.py` (default) | **Full-data** training: all labelled tiles, no validation. This produced the submission model. |

---

## Paths are absolute — and rewritten on move

Every YAML ships with absolute paths baked to the machine the bundle was
prepared on:

```yaml
paths:
  workspace: /path/to/svamitva
```

**If you move or extract the bundle elsewhere, rerun `../../setup_env.sh`** —
its Step 7 auto-detects the old `workspace:` value in `data_prep.yaml` and
`sed`-rewrites every path in all four YAMLs to the new location. Manual
rewrite is in the [repo-root README](../../README.md) §6.

---

## `data_prep.yaml` — key sections

| Section | Keys | Notes |
|---|---|---|
| `paths` | `workspace`, `train_raw_dir`, `test_raw_dir`, `shp_raw_cg`, `shp_raw_pb`, `dataset_dir`, `train_dir`, `test_dir`, `shp_cg`, `shp_pb`, `cog_dir`, `tiles_dir`, `outputs_dir`, `visualizations_dir`, `dinov3_repo` | Raw inputs are read-only; the rest are generated. |
| `crs` | `EPSG:3857` | CRS all shapefiles are merged into before per-tile reprojection. |
| `raster_extensions` | `.tif`, `.tiff`, `.ecw` | Which rasters the walk picks up. |
| `cog` | `compression: DEFLATE`, `blocksize: 512`, `overview_resampling: nearest` | COG conversion settings. |
| `tiling` | `tile_size: 1024`, `overlap: 128`, `nodata_threshold: 0.5`, `output_format: GTiff` | Step = `tile_size − overlap` = 896; tiles with >50 % nodata are dropped. |
| `feature_classes` | shapefile-stem → class-ID map | Covers naming variants (`Road`/`Road_Centre_Line`→2, `Utility`/`Utility_Poly`→4, `Water_Body`/`Water_Body_Line`/`Waterbody_Point`→3, etc.). |
| `feature_normalize` | raw stem → canonical stem | e.g. `Built_Up_Area_typ`→`Built_Up_Area_type`, applied before the class lookup. |
| `vectorization` | `simplify_tolerance: 1.0`, `min_area: 5.0`, `buffer_distance: 0.0` | Defaults for GeoPackage export (CLI flags override). |
| `evaluation` | `iou_threshold: 0.5`, `metrics: [iou, f1, precision, recall]` | |
| `visualization` | `thumbnail_max_width`, `dpi`, `overlay_alpha`, `class_colors` | Per-class hex colours for figures. |

---

## `train*.yaml` — key sections

| Section | Keys | Notes |
|---|---|---|
| `full_training` / `test_dataset` / `test_datasets` | selects the split | `train.yaml`: `test_dataset` = one hold-out village. `train_2val.yaml`: `test_datasets` = two. `train_full.yaml`: `full_training: true`, `test_dataset: ""` → no val. |
| `paths` | `workspace`, `tiles_dir`, `tile_index`, `labels_dir`, `output_dir`, `checkpoint_dir`, `vis_dir`, `log_dir` | `labels_dir` = the `masks/` directory. |
| `label_sources` | list of `{path, type: shapefile}` | Consumed by `prepare_masks.py`. |
| `feature_normalize`, `feature_classes` | as above | Shapefile canonicalisation + class map for `prepare_masks.py`. |
| `classes` | 6 `{name, id, color}` entries | Background (id 0) is added implicitly in code (`num_classes = len(classes)+1`). |
| `model.encoder` | `arch: vitl16`, `embed_dim: 1024`, `patch_size: 16`, `intermediate_layers: [5,11,17,23]`, `freeze_first_n_blocks: 12`, `pretrained: true`, `weights: SAT493M` | The config's `weights: SAT493M` overrides the module default of `LVD1689M`. |
| `model.decoder` | `type: hrdecoder`, `hidden_dim: 256`, `hr_crop_size: [256,256]`, `crop_num: 2`, `scale_ratio: [0.75,1.25]`, `divisible: 8`, `hr_loss_weight: 0.1`, `lr_loss_weight: 0.5`, `fuse_loss_weight: 1.0` | Swap `type` to `upernet`/`segformer`/`skip` to use an alternative decoder (same encoder/loss/data). |
| `training` | `tile_size: 1024`, `batch_size: 4`, `num_workers: 4`, `max_epochs: 50`, `learning_rate: 1e-4`, `encoder_lr_multiplier: 0.1`, `weight_decay: 1e-4`, `scheduler: cosine`, `warmup_epochs: 5`, `accumulate_grad_batches: 2`, `precision: 16-mixed`, `early_stopping_patience` | `batch_size 4 × accum 2 = effective 8`. Patience `10` in `train.yaml`/`train_2val.yaml`, `-1` (disabled) in `train_full.yaml`. `train_full.yaml` also has `save_every_n_epochs: 3`. |
| `loss` | `bce_weight: 0.5`, `dice_weight: 0.3`, `edge_weight: 0.2`, `edge_kernel_size: 3` | The `bce_weight` term is actually Cross-Entropy (`ignore_index=255`). |
| `visualization` | `overlay_alpha`, `dpi: 150`, `save_format`, `num_samples: 10` | |

> The comment `# 512/16=32` next to `training.tile_size` is stale — the
> value is **1024** (→ 64×64 token grid).

---

## `train.yaml` vs `train_full.yaml` (the diff that matters)

| Key | `train.yaml` | `train_full.yaml` |
|---|---|---|
| `full_training` | absent (→ `false`) | `true` |
| `test_dataset` | `NAGUL_450171_MADASE_450172_GHOTPAL_450137_ORTHO` | `""` (no hold-out) |
| `training.early_stopping_patience` | `10` | `-1` (disabled) |
| `training.save_every_n_epochs` | absent | `3` |

Everything else (`max_epochs 50`, batch/accum, LR schedule, `model`,
`classes`, `loss`, `visualization`, `label_sources`) is identical.
`train_2val.yaml` differs from `train.yaml` only by using `test_datasets`
(TIMMOWAL + NAGUL) and the isolated `*_2val` output dirs.
