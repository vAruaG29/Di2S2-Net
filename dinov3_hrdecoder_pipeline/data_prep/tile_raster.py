#!/usr/bin/env python3
"""
Step 3: Tile Rasters into Patches
==================================
Tiles COG rasters into 1024x1024 patches with overlap,
preserving georeferencing. Skips nodata-heavy tiles.

Usage:
    python -m dinov3_hrdecoder_pipeline.data_prep.tile_raster
    python -m dinov3_hrdecoder_pipeline.data_prep.tile_raster --config dinov3_hrdecoder_pipeline/configs/data_prep.yaml
    python -m dinov3_hrdecoder_pipeline.data_prep.tile_raster --input /path/to/raster.tif
"""

import os
import sys
import csv
import argparse
from pathlib import Path

import numpy as np
import yaml
import rasterio
from rasterio.windows import Window
from rasterio.transform import from_bounds


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def find_rasters(root_dir: str, extensions: list = None) -> list:
    if extensions is None:
        extensions = [".tif", ".tiff"]
    rasters = []
    for dirpath, _, filenames in os.walk(root_dir):
        for f in filenames:
            ext = os.path.splitext(f)[1].lower()
            if ext in extensions:
                full = os.path.join(dirpath, f)
                if not any(full.endswith(s) for s in (".aux", ".pyrx", ".xml", ".aux.xml", ".ovr")):
                    rasters.append(full)
    return sorted(rasters)


def tile_raster(
    raster_path: str,
    output_dir: str,
    tile_size: int = 1024,
    overlap: int = 128,
    nodata_threshold: float = 0.5,
    dataset_name: str | None = None,
) -> list:
    """
    Tile a raster into patches with overlap.
    Returns list of tile metadata dicts.

    `dataset_name` overrides the auto-derived `Path(raster_path).stem`
    — useful when the on-disk filename is something generic like
    `source.tif` but downstream code expects the original dataset name.
    """
    tiles_meta = []
    if not dataset_name:
        dataset_name = Path(raster_path).stem

    tile_dir = os.path.join(output_dir, dataset_name)
    os.makedirs(tile_dir, exist_ok=True)

    with rasterio.open(raster_path) as src:
        bands = src.count
        width = src.width
        height = src.height
        nodata = src.nodata
        step = tile_size - overlap

        n_cols = max(1, (width - overlap) // step)
        n_rows = max(1, (height - overlap) // step)

        total_tiles = 0
        skipped_tiles = 0

        for row_idx in range(n_rows):
            for col_idx in range(n_cols):
                col_off = col_idx * step
                row_off = row_idx * step

                # Clamp to raster bounds
                win_width = min(tile_size, width - col_off)
                win_height = min(tile_size, height - row_off)

                if win_width < tile_size // 2 or win_height < tile_size // 2:
                    continue

                window = Window(col_off, row_off, win_width, win_height)
                data = src.read(window=window)

                # Check nodata coverage
                if nodata is not None:
                    nodata_mask = np.all(data == nodata, axis=0)
                else:
                    nodata_mask = np.all(data == 0, axis=0)

                nodata_frac = nodata_mask.sum() / nodata_mask.size
                if nodata_frac > nodata_threshold:
                    skipped_tiles += 1
                    continue

                # Pad if necessary (edge tiles)
                if win_width < tile_size or win_height < tile_size:
                    padded = np.zeros((bands, tile_size, tile_size), dtype=data.dtype)
                    padded[:, :win_height, :win_width] = data
                    data = padded

                # Compute transform for this tile
                win_transform = rasterio.windows.transform(window, src.transform)

                # Output path
                tile_name = f"{row_idx:04d}_{col_idx:04d}.tif"
                tile_path = os.path.join(tile_dir, tile_name)

                # Write tile
                profile = src.profile.copy()
                profile.update(
                    driver="GTiff",
                    width=tile_size,
                    height=tile_size,
                    transform=win_transform,
                    compress="deflate",
                )

                with rasterio.open(tile_path, "w", **profile) as dst:
                    dst.write(data)

                # Compute bounds
                bounds = rasterio.transform.array_bounds(
                    tile_size, tile_size, win_transform
                )

                tiles_meta.append(
                    {
                        "tile_path": tile_path,
                        "dataset": dataset_name,
                        "row": row_idx,
                        "col": col_idx,
                        "col_off": col_off,
                        "row_off": row_off,
                        "width": win_width,
                        "height": win_height,
                        "nodata_frac": nodata_frac,
                        "min_x": bounds[0],
                        "min_y": bounds[1],
                        "max_x": bounds[2],
                        "max_y": bounds[3],
                        "crs": str(src.crs),
                        "source_raster": raster_path,
                    }
                )
                total_tiles += 1

        print(
            f"  {dataset_name}: {total_tiles} tiles created, "
            f"{skipped_tiles} skipped (>{nodata_threshold*100:.0f}% nodata)"
        )

    return tiles_meta


def write_tile_index(tiles_meta: list, output_path: str, merge: bool = False):
    """Write tile metadata to CSV index file.

    If `merge=True` and `output_path` already exists, rows from the
    existing CSV whose `dataset` value is NOT in the new write are
    preserved (single-dataset re-tiles don't wipe the rest of the
    index).
    """
    if not tiles_meta and not merge:
        return

    rows = list(tiles_meta)
    if merge and os.path.exists(output_path):
        new_datasets = {r.get("dataset") for r in tiles_meta}
        with open(output_path) as f:
            for r in csv.DictReader(f):
                if r.get("dataset") not in new_datasets:
                    rows.append(r)

    if not rows:
        return

    # Use the field order from one of the new rows (always present in
    # merge mode because we early-return on no rows above).
    template = tiles_meta[0] if tiles_meta else rows[0]
    fieldnames = list(template.keys())
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Tile index written → {output_path} ({len(rows)} entries)")


def main():
    parser = argparse.ArgumentParser(description="Tile rasters into patches")
    parser.add_argument("--config", default=str(__import__("pathlib").Path(__file__).resolve().parent.parent / "configs" / "data_prep.yaml"))
    parser.add_argument("--input", default=None, help="Process a single raster file")
    parser.add_argument(
        "--dataset-name", default=None,
        help="Override the auto-derived dataset name (Path(input).stem). "
             "Tiles will be tagged with this name in tile_index.csv.",
    )
    args = parser.parse_args()

    config_path = args.config
    if not os.path.isabs(config_path):
        config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), config_path)
    cfg = load_config(config_path)

    paths = cfg["paths"]
    tile_cfg = cfg.get("tiling", {})
    tile_size = tile_cfg.get("tile_size", 1024)
    overlap = tile_cfg.get("overlap", 128)
    nodata_threshold = tile_cfg.get("nodata_threshold", 0.5)
    tiles_dir = paths["tiles_dir"]

    print("=" * 70)
    print("  STEP 3: Tile Rasters into Patches")
    print(f"  Tile size: {tile_size}×{tile_size}, Overlap: {overlap}px")
    print("=" * 70)

    all_tiles = []

    if args.input:
        # Process single file
        rasters = [args.input]
    else:
        # Prefer COGs (written by convert_to_cog.py to cog/<split>/<basename>.tif);
        # fall back to raw data if no COGs exist yet.
        cog_dir = paths.get("cog_dir", "")
        rasters = find_rasters(cog_dir) if os.path.isdir(cog_dir) else []
        if rasters:
            print(f"\n  Using COG directory: {cog_dir}")
        else:
            print(f"\n  COG directory not found/empty, using raw data...")
            for data_dir in (paths.get("train_raw_dir"), paths.get("test_raw_dir")):
                if data_dir and os.path.isdir(data_dir):
                    rasters.extend(find_rasters(data_dir, [".tif", ".tiff"]))

    print(f"  Found {len(rasters)} raster files to tile\n")

    for rpath in rasters:
        # --dataset-name overrides the per-file auto-derived name, but
        # only makes sense when we have exactly one input file.
        ds_override = args.dataset_name if (args.dataset_name and len(rasters) == 1) else None
        tiles = tile_raster(
            rpath, tiles_dir, tile_size, overlap, nodata_threshold,
            dataset_name=ds_override,
        )
        all_tiles.extend(tiles)

    # Write combined index. In single-file mode, merge with whatever
    # already exists so we don't blow away tiles for other datasets.
    index_path = os.path.join(tiles_dir, "tile_index.csv")
    write_tile_index(all_tiles, index_path, merge=bool(args.input))

    print(f"\n{'='*70}")
    print(f"  TILING COMPLETE")
    print(f"  Total tiles: {len(all_tiles)}")
    print(f"  Output:      {tiles_dir}")
    print(f"  Index:       {index_path}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
