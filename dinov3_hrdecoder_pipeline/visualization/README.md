# `visualization/` — stage-wise figures

Optional matplotlib (Agg backend) visualisations for sanity-checking each
pipeline stage: the input data, what the DINOv3 encoder "sees", per-tile
predictions vs ground truth, and the final stitched raster. Run from the
**repo root** with `conda activate svamitva2`.

```bash
python -m dinov3_hrdecoder_pipeline.visualization.visualize [--stage STAGE]
```

There is **no `--config` flag** — config is hard-coded to
`configs/train.yaml` (paths, `classes`, `visualization.{dpi,num_samples}`,
`test_dataset`, `model.encoder`).

| `--stage` | Reads | Writes (under `paths.vis_dir/`) |
|---|---|---|
| `data` | `masks/label_index.csv` | `1_data/data_samples.png` (N random labelled tiles: tile / mask / overlay) + `class_distribution.png` (pixel counts over the first 100 labelled tiles). |
| `features` | first 4 labelled tiles + DINOv3 encoder | `2_features/features_sample_{0..3}.png` — input + a **PCA(3)** panel per intermediate layer `[5,11,17,23]`. Loads the encoder (needs the HuggingFace weights). |
| `predictions` | `outputs/predictions/<test_dataset>/prediction_index.csv` + matching `masks/<ds>/*_mask.tif` | `3_predictions/prediction_{i}_{tile}.png` — 5 panels: input, GT, GT overlay, prediction, pred overlay. |
| `stitched` | `outputs/stitched/<test_dataset>_pred.tif` | `4_stitched/stitched_dinov3hrdecoder.png` (PIL-nearest downsampled if > 4096 px). |
| `all` *(default)* | all of the above | Runs `data → features → predictions → stitched`; each wrapped in try/except so one missing input doesn't abort the rest. |

- `dpi` and colours come from the config (`visualization.dpi` = 150,
  per-class colours from `classes`). Random sampling is seeded (`seed=42`) so
  figures are reproducible.
- Each stage warns and returns gracefully if its input (e.g.
  `label_index.csv`, `prediction_index.csv`, the stitched raster) is missing,
  so you can run `--stage predictions` right after inference without having
  run the others.

Use these to produce the qualitative galleries for the deck
([`docs/documentation.md`](../../docs/documentation.md)). Training-curve figures
are a separate script — [`scripts/make_curves.py`](../../scripts/make_curves.py)
→ `docs/figures/`.
