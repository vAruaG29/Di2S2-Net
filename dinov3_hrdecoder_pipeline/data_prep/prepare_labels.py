#!/usr/bin/env python3
"""
Step 4: Prepare Label Masks from Shapefiles
=============================================
Rasterizes shapefile polygons onto each tile to create
binary/multi-class segmentation masks.

Usage:
    python -m dinov3_hrdecoder_pipeline.data_prep.prepare_labels
    python -m dinov3_hrdecoder_pipeline.data_prep.prepare_labels --config dinov3_hrdecoder_pipeline/configs/data_prep.yaml
    python -m dinov3_hrdecoder_pipeline.data_prep.prepare_labels --class-name Built_Up_Area_type  # buildings only
"""

import os
import sys
import csv
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
import yaml
import rasterio
from rasterio.features import rasterize
from rasterio.transform import from_bounds

try:
    import geopandas as gpd
    from shapely.geometry import box, mapping
except ImportError:
    print("ERROR: geopandas not installed. Run: pip install geopandas shapely")
    sys.exit(1)


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def find_shapefiles(shp_dir: str) -> dict:
    shapefiles = {}
    if not os.path.isdir(shp_dir):
        return shapefiles
    for f in os.listdir(shp_dir):
        if f.endswith(".shp"):
            name = os.path.splitext(f)[0]
            shapefiles[name] = os.path.join(shp_dir, f)
    return shapefiles


def load_tile_index(index_path: str) -> list:
    """Load tile metadata from CSV index."""
    tiles = []
    with open(index_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Convert numeric fields
            for key in ["row", "col", "col_off", "row_off", "width", "height"]:
                row[key] = int(row[key])
            for key in ["nodata_frac", "min_x", "min_y", "max_x", "max_y"]:
                row[key] = float(row[key])
            tiles.append(row)
    return tiles


def rasterize_labels_for_tile(
    tile_meta: dict,
    gdf_dict: dict,
    feature_classes: dict,
    mode: str = "binary",
    target_class: str | None = None,
    _gdf_cache: dict | None = None,
) -> np.ndarray | None:
    """
    Rasterize shapefile geometries onto a tile.

    Args:
        tile_meta: dict with tile_path, min_x, min_y, max_x, max_y, crs
        gdf_dict: {class_name: GeoDataFrame}  (any CRS – reprojected here)
        feature_classes: {class_name: class_id}
        mode: 'binary' (single class) or 'multiclass' (all classes)
        target_class: class name for binary mode
        _gdf_cache: mutable dict for caching reprojected GDFs per CRS

    Returns:
        2D numpy array (height x width) with class IDs, or None if no labels
    """
    if _gdf_cache is None:
        _gdf_cache = {}

    tile_path = tile_meta["tile_path"]

    with rasterio.open(tile_path) as src:
        tile_width = src.width
        tile_height = src.height
        tile_transform = src.transform
        tile_crs = src.crs
        tile_bounds = box(
            tile_meta["min_x"], tile_meta["min_y"],
            tile_meta["max_x"], tile_meta["max_y"]
        )

    tile_crs_str = str(tile_crs)

    mask = np.zeros((tile_height, tile_width), dtype=np.uint8)
    has_labels = False

    if mode == "binary":
        classes_to_process = {target_class: 1} if target_class in gdf_dict else {}
    else:
        classes_to_process = feature_classes

    for class_name, class_id in classes_to_process.items():
        if class_name not in gdf_dict:
            continue

        # Reproject shapefile to tile CRS if needed (cached per CRS)
        cache_key = (class_name, tile_crs_str)
        if cache_key not in _gdf_cache:
            gdf_src = gdf_dict[class_name]
            if str(gdf_src.crs) != tile_crs_str:
                _gdf_cache[cache_key] = gdf_src.to_crs(tile_crs)
            else:
                _gdf_cache[cache_key] = gdf_src
        gdf = _gdf_cache[cache_key]

        # Filter to geometries that intersect this tile
        try:
            spatial_idx = gdf.sindex
            possible_idx = list(spatial_idx.intersection(tile_bounds.bounds))
            if not possible_idx:
                continue
            candidates = gdf.iloc[possible_idx]
            clipped = candidates[candidates.geometry.intersects(tile_bounds)]
        except Exception:
            clipped = gdf[gdf.geometry.intersects(tile_bounds)]

        if len(clipped) == 0:
            continue

        # Handle different geometry types
        geom_type = clipped.geometry.geom_type.iloc[0]

        if geom_type in ("Point", "MultiPoint"):
            # Buffer points into small circles (3 meter radius)
            clipped = clipped.copy()
            clipped["geometry"] = clipped.geometry.buffer(3)

        # Prepare shapes for rasterization
        shapes = [
            (mapping(geom), class_id)
            for geom in clipped.geometry
            if geom is not None and not geom.is_empty
        ]

        if not shapes:
            continue

        # Rasterize
        try:
            class_mask = rasterize(
                shapes,
                out_shape=(tile_height, tile_width),
                transform=tile_transform,
                fill=0,
                dtype=np.uint8,
            )
            # For multiclass, higher class_id overwrites lower
            mask = np.where(class_mask > 0, class_mask, mask)
            has_labels = True
        except Exception as e:
            print(f"    WARN: Rasterize failed for {class_name}: {e}")

    return mask if has_labels else None


def save_mask(mask: np.ndarray, tile_meta: dict, suffix: str = "_mask",
              labels_dir: str | None = None):
    """Save a mask as a single-band GeoTIFF.

    Two output locations, decided by `labels_dir`:

      • `labels_dir is None`  → legacy beside-the-tile path
            <tile_dir>/<row>_<col>{suffix}.tif
      • `labels_dir is set`    → <labels_dir>/<dataset>/<tile_stem>_mask.tif
        which is exactly where `run_pipeline.discover_datasets` and
        `evaluate_tile_predictions` look for ground-truth masks. This
        path is REQUIRED for evaluation to find the masks.
    """
    tile_path = tile_meta["tile_path"]
    tile_stem = os.path.splitext(os.path.basename(tile_path))[0]

    if labels_dir:
        ds_name = tile_meta.get("dataset", "")
        out_dir = os.path.join(labels_dir, ds_name) if ds_name else labels_dir
        os.makedirs(out_dir, exist_ok=True)
        # Use the canonical "_mask.tif" suffix that the downstream
        # consumers expect — `discover_datasets` globs for this exactly.
        mask_path = os.path.join(out_dir, f"{tile_stem}_mask.tif")
    else:
        stem = os.path.splitext(tile_path)[0]
        mask_path = stem + suffix + ".tif"

    with rasterio.open(tile_path) as src:
        profile = src.profile.copy()

    profile.update(
        count=1,
        dtype="uint8",
        nodata=0,
        compress="deflate",
    )

    with rasterio.open(mask_path, "w", **profile) as dst:
        dst.write(mask[np.newaxis, :, :])

    return mask_path


def main():
    parser = argparse.ArgumentParser(description="Prepare label masks from shapefiles")
    parser.add_argument("--config", default=str(__import__("pathlib").Path(__file__).resolve().parent.parent / "configs" / "data_prep.yaml"))
    parser.add_argument("--class-name", default=None,
                        help="Single class to rasterize (default: all)")
    parser.add_argument("--mode", choices=["binary", "multiclass"], default="binary",
                        help="binary = single class, multiclass = all classes")
    parser.add_argument("--dataset", default=None,
                        help="Only rasterise tiles whose `dataset` column equals this name.")
    args = parser.parse_args()

    config_path = args.config
    if not os.path.isabs(config_path):
        config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), config_path)
    cfg = load_config(config_path)

    paths = cfg["paths"]
    feature_classes = cfg["feature_classes"]
    name_aliases = cfg.get("feature_normalize", {})
    tiles_dir = paths["tiles_dir"]
    # If `labels_dir` is configured, write masks there so run_pipeline /
    # evaluate_tile_predictions can find them. Otherwise fall back to
    # the legacy beside-the-tile layout.
    labels_dir = paths.get("labels_dir")
    if labels_dir:
        os.makedirs(labels_dir, exist_ok=True)

    # Default binary target = first class in feature_classes (by ID)
    default_target = min(feature_classes, key=feature_classes.get)
    target_class = args.class_name or default_target

    print("=" * 70)
    print("  STEP 4: Prepare Label Masks from Shapefiles")
    print(f"  Mode: {args.mode} | Target class: {target_class}")
    print("=" * 70)

    # Load tile index
    index_path = os.path.join(tiles_dir, "tile_index.csv")
    if not os.path.exists(index_path):
        print(f"  ERROR: Tile index not found: {index_path}")
        print("  Run tile_raster.py first.")
        sys.exit(1)

    tiles = load_tile_index(index_path)
    print(f"  Loaded {len(tiles)} tiles from index")

    if args.dataset:
        before = len(tiles)
        tiles = [t for t in tiles if t.get("dataset") == args.dataset]
        print(f"  Scoped to dataset={args.dataset!r}: {len(tiles)}/{before} tiles")
        if not tiles:
            print(f"  Nothing to do for {args.dataset!r}, exiting.")
            return

    # Shapefile sources: organised CG/PB dirs + raw input dirs (all optional)
    candidate_dirs = [
        paths.get("shp_cg"), paths.get("shp_pb"),
        paths.get("shp_raw_cg"), paths.get("shp_raw_pb"),
    ]
    shp_dirs = [d for d in candidate_dirs if d and os.path.isdir(d)]
    all_gdfs = defaultdict(list)

    for shp_dir in shp_dirs:
        shapefiles = find_shapefiles(shp_dir)
        for class_name, spath in shapefiles.items():
            # Normalise name
            canonical = name_aliases.get(class_name, class_name)
            try:
                gdf = gpd.read_file(spath)
                print(f"  Loaded {canonical} from {os.path.basename(shp_dir)}: "
                      f"{len(gdf)} features (CRS: {gdf.crs})")
                all_gdfs[canonical].append(gdf)
            except Exception as e:
                print(f"  WARN: Could not read {spath}: {e}")

    # Merge GeoDataFrames for each class (keep original CRS – reprojection
    # is done per-tile in rasterize_labels_for_tile via _gdf_cache)
    import pandas as pd
    gdf_dict = {}
    for class_name, gdf_list in all_gdfs.items():
        if gdf_list:
            # Use EPSG:3857 as merge CRS (most shapefiles already use it)
            merge_crs = "EPSG:3857"
            aligned = []
            for gdf in gdf_list:
                if str(gdf.crs) != merge_crs:
                    gdf = gdf.to_crs(merge_crs)
                aligned.append(gdf)
            gdf_dict[class_name] = gpd.GeoDataFrame(
                pd.concat(aligned, ignore_index=True),
                crs=merge_crs,
            )
            print(f"  Merged {class_name}: {len(gdf_dict[class_name])} total features")

    # Shared reprojection cache (avoids re-projecting the same shapefile per tile)
    _gdf_cache = {}

    # Process each tile
    tiles_with_labels = 0
    tiles_without_labels = 0

    labeled_tile_paths = set()

    for i, tile_meta in enumerate(tiles):
        mask = rasterize_labels_for_tile(
            tile_meta, gdf_dict, feature_classes,
            mode=args.mode, target_class=target_class,
            _gdf_cache=_gdf_cache,
        )

        if mask is not None:
            suffix = "_mask" if args.mode == "binary" else "_mask_multi"
            mask_path = save_mask(mask, tile_meta, suffix, labels_dir=labels_dir)
            tiles_with_labels += 1
            labeled_tile_paths.add(tile_meta["tile_path"])
        else:
            tiles_without_labels += 1

        if (i + 1) % 100 == 0:
            print(f"    Processed {i+1}/{len(tiles)} tiles...")

    # ── Update tile_index.csv with has_label and split columns ──────────
    print(f"\n  Updating tile_index.csv with 'has_label' and 'split' columns...")

    # Re-read the full CSV to preserve all existing columns
    updated_rows = []
    with open(index_path, "r") as f:
        reader = csv.DictReader(f)
        original_fieldnames = list(reader.fieldnames)
        for row in reader:
            tile_path = row["tile_path"]

            # Add has_label: check both the new labels_dir location and
            # the legacy beside-the-tile filenames.
            stem = os.path.splitext(tile_path)[0]
            tile_stem = os.path.basename(stem)
            ds_name = row.get("dataset", "")
            in_labels_dir = (
                labels_dir is not None
                and ds_name
                and os.path.exists(os.path.join(labels_dir, ds_name, f"{tile_stem}_mask.tif"))
            )
            has_mask = (
                tile_path in labeled_tile_paths
                or in_labels_dir
                or os.path.exists(stem + "_mask.tif")
                or os.path.exists(stem + "_mask_multi.tif")
            )
            row["has_label"] = "yes" if has_mask else "no"

            # Add split: infer from path if not already present
            if "split" not in original_fieldnames:
                path_lower = tile_path.lower()
                if "test" in path_lower:
                    row["split"] = "test"
                else:
                    row["split"] = "train"

            updated_rows.append(row)

    # Build updated fieldnames
    updated_fieldnames = list(original_fieldnames)
    if "has_label" not in updated_fieldnames:
        updated_fieldnames.append("has_label")
    if "split" not in updated_fieldnames:
        updated_fieldnames.append("split")

    with open(index_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=updated_fieldnames)
        writer.writeheader()
        writer.writerows(updated_rows)

    n_labeled_in_csv = sum(1 for r in updated_rows if r["has_label"] == "yes")
    print(f"  Updated tile_index.csv: {n_labeled_in_csv} tiles marked as labeled")

    print(f"\n{'='*70}")
    print(f"  LABEL PREPARATION COMPLETE")
    print(f"  Tiles with labels:    {tiles_with_labels}")
    print(f"  Tiles without labels: {tiles_without_labels}")
    print(f"  Mode:                 {args.mode}")
    print(f"  Target class:         {target_class if args.mode == 'binary' else 'all'}")
    print(f"  tile_index.csv:       Updated with 'has_label' and 'split' columns")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
