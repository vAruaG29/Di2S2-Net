# `data_prep/` — raw rasters → tiles + label masks

Turns a messy drop of orthomosaics and shapefiles into the clean, tiled,
label-aligned corpus the model trains and infers on. **Run once per data
drop, in order.** All scripts read `configs/data_prep.yaml` unless noted;
`prepare_masks.py` is the exception (it reads `configs/train.yaml`).

Run everything from the **repo root** with `conda activate svamitva2`.

```bash
python -m dinov3_hrdecoder_pipeline.data_prep.prepare_dataset
python -m dinov3_hrdecoder_pipeline.data_prep.convert_to_cog
python -m dinov3_hrdecoder_pipeline.data_prep.tile_raster
python -m dinov3_hrdecoder_pipeline.data_prep.prepare_masks          # for training
#   …or, for the CSV-in-place variant used by some tooling:
python -m dinov3_hrdecoder_pipeline.data_prep.prepare_labels --mode multiclass
```

---

## Stage order & data flow

```
data/{train,test}/*.{tif,ecw} + shapefiles
   │  prepare_dataset.py     discover · validate · dedup · organise (symlinks)
   ▼
dataset/{train,test}/{images,labels}/{CG,PB}/ + dataset/manifest.csv
   │  convert_to_cog.py      → Cloud-Optimized GeoTIFF, auto-UTM reproject
   ▼
cog/<split>/<region>/<stem>.tif
   │  tile_raster.py         → 1024² tiles, 128-px overlap, nodata filter
   ▼
tiles/<ds>/<row>_<col>.tif + tiles/tile_index.csv
   │  prepare_masks.py       → rasterise shapefiles onto each tile
   ▼
masks/<ds>/<tile>_mask.tif + masks/label_index.csv
```

---

## `prepare_dataset.py` — discover, validate, dedup, organise

```bash
python -m dinov3_hrdecoder_pipeline.data_prep.prepare_dataset [--copy] [--output DIR] [--config PATH]
```

| Flag | Default | Effect |
|---|---|---|
| `--config` | `configs/data_prep.yaml` | Config path. |
| `--copy` | off (symlink) | Hard-copy rasters instead of symlinking. |
| `--output DIR` | config `paths.dataset_dir` | Override the output dataset dir. |

- **Reads:** rasters under `train_raw_dir`/`test_raw_dir`; shapefiles under `shp_raw_cg` (region CG) / `shp_raw_pb` (region PB).
- **Writes:** `dataset/{train,test}/{images,labels}/{CG,PB}/` (raster symlinks + copied shapefile components) and **`dataset/manifest.csv`** with columns `split, region, filename, status, width, height, bands, crs, size_mb, source` (`status` ∈ `valid` / `corrupted: …` / `duplicate_skipped`).
- **Dedup** (train only): rasters are grouped by a **canonical key** (strip extension, strip trailing `_3857`, normalise case/whitespace/underscores). Within a group the survivor is picked *valid > corrupt*, *.tif > .ecw*, *larger > smaller* — this is what prevents ECW/TIF and `_3857` re-projection copies leaking across the set.
- **Region** (CG / PB / UNKNOWN) is inferred from path markers, village-code regexes, and hard-coded village-name lists (`classify_region`).

## `convert_to_cog.py` — Cloud-Optimized GeoTIFF + auto-UTM

```bash
python -m dinov3_hrdecoder_pipeline.data_prep.convert_to_cog [--input FILE] [--dataset-name NAME] [--config PATH]
```

| Flag | Default | Effect |
|---|---|---|
| `--config` | `configs/data_prep.yaml` | Config path. |
| `--input FILE` | walk train/test | Convert only this one raster (skips the walk). |
| `--dataset-name NAME` | — | Convert only the raster whose basename stem matches. |

- **Reads** `cog.{compression,blocksize,overview_resampling}` and `raster_extensions`.
- **Writes:** real COGs at `cog/<split>/<region>/<stem>.tif`; symlinks at `dataset/<split>/cog/<region>/<stem>.tif`.
- **Auto-UTM:** rasters in `EPSG:4326` are `gdalwarp`-reprojected to the correct UTM zone (`32600+zone` N / `32700+zone` S, computed from the raster centroid) before COG conversion — a projected, metre-accurate CRS is required for correct areas/lengths in the final GeoPackage.
- **ECW:** converted via `gdal_translate` if the GDAL ECW driver exists, else skipped gracefully. Corrupt TIFs are caught by a trial block-read and skipped. Output uses `gdal_translate -of COG` with `BIGTIFF=IF_SAFER`. Idempotent (skips existing COGs); validates all outputs at the end.

## `tile_raster.py` — 1024² tiles with overlap

```bash
python -m dinov3_hrdecoder_pipeline.data_prep.tile_raster [--input FILE] [--dataset-name NAME] [--config PATH]
```

| Flag | Default | Effect |
|---|---|---|
| `--config` | `configs/data_prep.yaml` | Config path. |
| `--input FILE` | walk `cog_dir` | Tile a single raster only. |
| `--dataset-name NAME` | `Path(input).stem` | Override the dataset tag (single-file mode only). |

- **Reads** `tiling.{tile_size:1024, overlap:128, nodata_threshold:0.5}`. Prefers COGs under `cog_dir`; falls back to raw `train_raw_dir`+`test_raw_dir` if no COGs.
- **Writes** `tiles/<ds>/<row:04d>_<col:04d>.tif` (GTiff, deflate, zero-padded to 1024²) and **`tiles/tile_index.csv`** with columns `tile_path, dataset, row, col, col_off, row_off, width, height, nodata_frac, min_x, min_y, max_x, max_y, crs, source_raster`.
- **Geometry:** step = `tile_size − overlap` = 896; per-tile georeferencing preserved via `windows.transform`; edge tiles smaller than `tile_size//2` are skipped, otherwise zero-padded. A tile is dropped if `nodata_frac > 0.5` (all-band nodata, or all-zero if nodata undefined).
- Single-file `--input` mode merges into the existing CSV (other datasets' rows are preserved); a full run overwrites the CSV.

## `tile_append.py` — add one raster without a full re-tile

```bash
python -m dinov3_hrdecoder_pipeline.data_prep.tile_append --input FILE [--force] [--config PATH]
```

| Flag | Default | Effect |
|---|---|---|
| `--input FILE` | **required** | Raster to tile and append. |
| `--config` | `configs/data_prep.yaml` | Config path. |
| `--force` | off | Re-tile even if the dataset is already in the index. |

- Same tiling geometry/nodata logic as `tile_raster.py`, but **appends** rows to `tiles/tile_index.csv`. Skips entirely (exit 0) if the dataset already exists in the index unless `--force`. Existing tile files on disk are not rewritten (only their metadata is appended). *Note:* `--force` appends without dedup, so it can duplicate index rows — used mainly by the portal's per-job tiling.

## `prepare_masks.py` — shapefiles → multiclass masks (training path)

```bash
python -m dinov3_hrdecoder_pipeline.data_prep.prepare_masks       # no flags
```

- **Reads `configs/train.yaml`** (hard-coded): `paths.{labels_dir, tile_index}`, `label_sources[].path`, `feature_normalize`, `feature_classes`.
- **Writes** `masks/<ds>/<tile>_mask.tif` (single-band uint8, `nodata=0`, deflate) for labelled tiles, and **`masks/label_index.csv`** with columns `tile_path, mask_path, dataset, has_label`.
- Shapefiles from all `label_sources` are merged in `EPSG:3857`; per-tile reprojection to the tile CRS is cached by `(class_name, crs)` (reproject once per village, reuse for all its tiles). A GeoPandas **R-tree spatial index** pre-filters candidate features by tile bounds. **Point/MultiPoint geometries (utility poles) are buffered 3 m** so they survive rasterisation. Multiclass overlap resolves last-class-wins (higher ID overwrites lower). This cache + R-tree is what makes label prep near-linear over 4,513 tiles.

## `prepare_labels.py` — masks + in-place `tile_index.csv` update

```bash
python -m dinov3_hrdecoder_pipeline.data_prep.prepare_labels --mode {binary,multiclass} [--class-name NAME] [--dataset NAME] [--config PATH]
```

| Flag | Default | Effect |
|---|---|---|
| `--config` | `configs/data_prep.yaml` | Config path. |
| `--mode` | `binary` | `binary` = one class; `multiclass` = all classes. |
| `--class-name NAME` | min class-ID | Single class to rasterise (binary mode). |
| `--dataset NAME` | all | Only rasterise tiles whose `dataset` column matches. |

- Same rasterisation engine as `prepare_masks.py` (EPSG:3857 merge, `(class,crs)` reproject cache, R-tree pre-filter, 3 m point buffer, last-class-wins). Writes masks to `labels_dir/<ds>/<tile>_mask.tif` (or beside the tile in legacy mode) and **rewrites `tile_index.csv` in place**, appending `has_label` (`yes`/`no`) and `split` (`train`/`test`) columns.
- Use `prepare_masks.py` for the standard training corpus (writes a separate `label_index.csv`); `prepare_labels.py` is the alternative that annotates `tile_index.csv` directly.
