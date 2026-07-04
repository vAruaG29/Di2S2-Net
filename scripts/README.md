# `scripts/` — standalone utilities

Helper scripts that sit alongside the pipeline but aren't part of it. Run
all of them **from the repo root** with `conda activate svamitva2` (they
anchor their paths to the repo root, one level up from here).

| Script | What it does |
|---|---|
| `analyze_dataset_stats.py` | Class-distribution stats over the labelled corpus — per-class pixel / tile / patch counts (the numbers behind the Bridge/Railway data-coverage caveat). |
| `make_curves.py` | Regenerate the training-curve figures in `docs/figures/` from Lightning `csv_logs/metrics.csv`. |
| `clean_unused_data.py` | Prune files under `data/` that aren't referenced by a `dataset/` symlink or a required shapefile label source. |

---

## `analyze_dataset_stats.py`

```bash
python scripts/analyze_dataset_stats.py
python scripts/analyze_dataset_stats.py --config dinov3_hrdecoder_pipeline/configs/train_full.yaml
python scripts/analyze_dataset_stats.py --no-pixels     # tile/label counts only (fast)
python scripts/analyze_dataset_stats.py --csv stats.csv # also dump a CSV
```

Reads `<labels_dir>/label_index.csv` and `tile_index.csv` (paths from the
config; falls back to the repo's `masks/` and `tiles/` if the config holds
another machine's absolute paths). Prints tile/label counts and, unless
`--no-pixels`, the per-class pixel/tile/patch distribution.

## `make_curves.py`

```bash
python scripts/make_curves.py
```

Writes `fig_8_2_curves.png`, `fig_9_1_curves.png`, `fig_full_train_curves.png`
and `fig_val_miou_comparison.png` into `docs/figures/`. **Regeneration record
only** — it reads training-run logs under `logs/` and `logs_2val/`, which are
gitignored and not shipped, so it only reproduces the committed figures on a
machine that still has those run directories.

## `clean_unused_data.py`

```bash
python scripts/clean_unused_data.py                 # dry-run (default): lists what it would delete
python scripts/clean_unused_data.py --apply         # actually delete
python scripts/clean_unused_data.py --apply --keep-sidecars   # spare .pyrx/.aux/.ovr files
```

Dry-run by default. Keeps every `data/` file that a `dataset/` symlink points
at (native resolution + a remote-prefix fallback so it works regardless of the
absolute prefix baked into the symlinks) plus the two shapefile label-source
dirs the training config depends on. Deletes everything else under `data/`.
