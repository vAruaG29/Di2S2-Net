#!/usr/bin/env python3
"""
Tile a single new raster and APPEND entries to the existing tile_index.csv.
Existing tiles are never re-processed or overwritten.

Usage:
    conda activate torch
    # run from your <WORKSPACE>
    python scripts/tile_append.py --input <path/to/raster.tif>
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


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def dataset_already_tiled(tile_index_path: str, dataset_name: str) -> bool:
    """Return True if dataset_name already has entries in tile_index.csv."""
    if not os.path.exists(tile_index_path):
        return False
    with open(tile_index_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("dataset") == dataset_name:
                return True
    return False


def tile_raster(
    raster_path: str,
    output_dir: str,
    tile_size: int = 1024,
    overlap: int = 128,
    nodata_threshold: float = 0.5,
) -> list:
    """Tile a raster into patches with overlap. Returns list of tile metadata dicts."""
    tiles_meta = []
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

                win_width = min(tile_size, width - col_off)
                win_height = min(tile_size, height - row_off)

                if win_width < tile_size // 2 or win_height < tile_size // 2:
                    continue

                window = Window(col_off, row_off, win_width, win_height)
                data = src.read(window=window)

                if nodata is not None:
                    nodata_mask = np.all(data == nodata, axis=0)
                else:
                    nodata_mask = np.all(data == 0, axis=0)

                nodata_frac = nodata_mask.sum() / nodata_mask.size
                if nodata_frac > nodata_threshold:
                    skipped_tiles += 1
                    continue

                # Pad edge tiles
                if win_width < tile_size or win_height < tile_size:
                    padded = np.zeros((bands, tile_size, tile_size), dtype=data.dtype)
                    padded[:, :win_height, :win_width] = data
                    data = padded

                win_transform = rasterio.windows.transform(window, src.transform)

                tile_name = f"{row_idx:04d}_{col_idx:04d}.tif"
                tile_path = os.path.join(tile_dir, tile_name)

                # Skip if tile already exists
                if os.path.exists(tile_path):
                    # Still add to metadata
                    bounds = rasterio.transform.array_bounds(tile_size, tile_size, win_transform)
                    tiles_meta.append({
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
                    })
                    total_tiles += 1
                    continue

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

                bounds = rasterio.transform.array_bounds(tile_size, tile_size, win_transform)
                tiles_meta.append({
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
                })
                total_tiles += 1

        print(f"  {dataset_name}: {total_tiles} tiles created/found, "
              f"{skipped_tiles} skipped (>{nodata_threshold*100:.0f}% nodata)")

    return tiles_meta


def append_to_tile_index(new_tiles: list, index_path: str):
    """Append new tile metadata rows to existing tile_index.csv.
    If the file doesn't exist, creates it from scratch."""
    if not new_tiles:
        print("  No new tiles to append.")
        return

    fieldnames = list(new_tiles[0].keys())

    if not os.path.exists(index_path):
        with open(index_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(new_tiles)
        print(f"  Created new tile index → {index_path} ({len(new_tiles)} entries)")
    else:
        with open(index_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writerows(new_tiles)
        print(f"  Appended {len(new_tiles)} entries → {index_path}")


def main():
    parser = argparse.ArgumentParser(description="Tile a single raster and append to tile_index.csv")
    parser.add_argument("--input", required=True, help="Path to raster file to tile")
    parser.add_argument("--config", default=str(__import__("pathlib").Path(__file__).resolve().parent.parent / "configs" / "data_prep.yaml"))
    parser.add_argument("--force", action="store_true",
                        help="Force re-tiling even if dataset already in index")
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
    index_path = os.path.join(tiles_dir, "tile_index.csv")

    dataset_name = Path(args.input).stem

    print("=" * 70)
    print(f"  TILE APPEND: {dataset_name}")
    print(f"  Tile size: {tile_size}×{tile_size}, Overlap: {overlap}px")
    print("=" * 70)

    # Check if already tiled
    if not args.force and dataset_already_tiled(index_path, dataset_name):
        print(f"\n  SKIP: '{dataset_name}' already has entries in tile_index.csv.")
        print(f"  Use --force to re-tile and re-append.")
        sys.exit(0)

    if not os.path.exists(args.input):
        print(f"\n  ERROR: Input file not found: {args.input}")
        sys.exit(1)

    new_tiles = tile_raster(args.input, tiles_dir, tile_size, overlap, nodata_threshold)
    append_to_tile_index(new_tiles, index_path)

    print(f"\n{'='*70}")
    print(f"  TILING COMPLETE")
    print(f"  New tiles: {len(new_tiles)}")
    print(f"  Tile dir:  {os.path.join(tiles_dir, dataset_name)}")
    print(f"  Index:     {index_path}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
