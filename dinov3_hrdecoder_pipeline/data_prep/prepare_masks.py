#!/usr/bin/env python3
"""
Prepare Label Masks for DINOv3+HRDecoder Pipeline
==================================================
Rasterizes shapefile labels onto tiles. Creates multiclass masks
in a separate directory (data/masks/) linked to tile_index.csv.

Usage:
    # run from your <WORKSPACE>
    python -m dinov3_hrdecoder_pipeline.data_prep.prepare_masks
"""

import os
import sys
import csv
from pathlib import Path
from collections import defaultdict

import numpy as np
import yaml
import rasterio
from rasterio.features import rasterize

try:
    import geopandas as gpd
    import pandas as pd
    from shapely.geometry import box, mapping
except ImportError:
    print("ERROR: pip install geopandas shapely")
    sys.exit(1)

PIPE_ROOT = Path(__file__).resolve().parent.parent
CFG_PATH = PIPE_ROOT / "configs" / "train.yaml"


def load_config():
    with open(CFG_PATH) as f:
        return yaml.safe_load(f)


def find_shapefiles(shp_dir: str) -> dict:
    shps = {}
    if not os.path.isdir(shp_dir):
        return shps
    for f in os.listdir(shp_dir):
        if f.endswith(".shp"):
            shps[os.path.splitext(f)[0]] = os.path.join(shp_dir, f)
    return shps


def load_all_shapefiles(cfg: dict) -> dict:
    """Load and merge all label shapefiles into a dict of class_name → GeoDataFrame."""
    name_aliases = cfg.get("feature_normalize", {})

    all_gdfs = defaultdict(list)
    for src in cfg["label_sources"]:
        shp_dir = src["path"]
        if not os.path.isdir(shp_dir):
            continue
        for class_name, spath in find_shapefiles(shp_dir).items():
            canonical = name_aliases.get(class_name, class_name)
            try:
                gdf = gpd.read_file(spath)
                print(f"  Loaded {canonical}: {len(gdf)} features from {os.path.basename(shp_dir)}")
                all_gdfs[canonical].append(gdf)
            except Exception as e:
                print(f"  WARN: Could not read {spath}: {e}")

    merge_crs = "EPSG:3857"
    gdf_dict = {}
    for class_name, gdf_list in all_gdfs.items():
        aligned = []
        for gdf in gdf_list:
            if str(gdf.crs) != merge_crs:
                gdf = gdf.to_crs(merge_crs)
            aligned.append(gdf)
        gdf_dict[class_name] = gpd.GeoDataFrame(
            pd.concat(aligned, ignore_index=True), crs=merge_crs
        )
        print(f"  Merged {class_name}: {len(gdf_dict[class_name])} total features")
    return gdf_dict


def build_class_mapping(cfg: dict) -> dict:
    """shapefile-stem → class-ID mapping. Sourced from config."""
    return dict(cfg["feature_classes"])


def rasterize_tile(tile_meta: dict, gdf_dict: dict, class_map: dict,
                   _cache: dict) -> np.ndarray | None:
    """Rasterize all label classes onto a single tile → multiclass mask."""
    tile_path = tile_meta["tile_path"]
    with rasterio.open(tile_path) as src:
        h, w = src.height, src.width
        transform = src.transform
        tile_crs = src.crs
        tile_bounds = box(
            tile_meta["min_x"], tile_meta["min_y"],
            tile_meta["max_x"], tile_meta["max_y"]
        )

    crs_str = str(tile_crs)
    mask = np.zeros((h, w), dtype=np.uint8)
    has_labels = False

    for class_name, gdf in gdf_dict.items():
        class_id = class_map.get(class_name)
        if class_id is None:
            continue

        cache_key = (class_name, crs_str)
        if cache_key not in _cache:
            if str(gdf.crs) != crs_str:
                _cache[cache_key] = gdf.to_crs(tile_crs)
            else:
                _cache[cache_key] = gdf
        gdf_proj = _cache[cache_key]

        try:
            candidates = gdf_proj.iloc[list(gdf_proj.sindex.intersection(tile_bounds.bounds))]
            clipped = candidates[candidates.geometry.intersects(tile_bounds)]
        except Exception:
            clipped = gdf_proj[gdf_proj.geometry.intersects(tile_bounds)]

        if len(clipped) == 0:
            continue

        geom_type = clipped.geometry.geom_type.iloc[0]
        if geom_type in ("Point", "MultiPoint"):
            clipped = clipped.copy()
            clipped["geometry"] = clipped.geometry.buffer(3)

        shapes = [(mapping(g), class_id) for g in clipped.geometry
                  if g is not None and not g.is_empty]
        if not shapes:
            continue

        try:
            cls_mask = rasterize(shapes, out_shape=(h, w), transform=transform,
                                 fill=0, dtype=np.uint8)
            mask = np.where(cls_mask > 0, cls_mask, mask)
            has_labels = True
        except Exception as e:
            print(f"    WARN: {class_name} rasterize error: {e}")

    return mask if has_labels else None


def main():
    cfg = load_config()
    masks_dir = cfg["paths"]["labels_dir"]
    os.makedirs(masks_dir, exist_ok=True)

    print("=" * 70)
    print("  PREPARE LABEL MASKS (DINOv3+HRDecoder Pipeline)")
    print("=" * 70)

    # Load tile index
    index_path = cfg["paths"]["tile_index"]
    tiles = []
    with open(index_path) as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames)
        for row in reader:
            for k in ["row", "col", "col_off", "row_off", "width", "height"]:
                row[k] = int(row[k])
            for k in ["nodata_frac", "min_x", "min_y", "max_x", "max_y"]:
                row[k] = float(row[k])
            tiles.append(row)
    print(f"  Loaded {len(tiles)} tiles")

    # Load shapefiles
    gdf_dict = load_all_shapefiles(cfg)
    class_map = build_class_mapping(cfg)
    print(f"  Class mapping: {class_map}\n")

    _cache = {}
    labeled = 0
    unlabeled = 0
    label_index = []

    for i, tile in enumerate(tiles):
        mask = rasterize_tile(tile, gdf_dict, class_map, _cache)
        dataset = tile["dataset"]

        # Save mask in data/masks/<dataset>/<row>_<col>_mask.tif
        mask_subdir = os.path.join(masks_dir, dataset)
        os.makedirs(mask_subdir, exist_ok=True)
        tile_name = Path(tile["tile_path"]).stem
        mask_path = os.path.join(mask_subdir, f"{tile_name}_mask.tif")

        if mask is not None:
            with rasterio.open(tile["tile_path"]) as src:
                profile = src.profile.copy()
            profile.update(count=1, dtype="uint8", nodata=0, compress="deflate")
            with rasterio.open(mask_path, "w", **profile) as dst:
                dst.write(mask[np.newaxis])
            labeled += 1
            label_index.append({
                "tile_path": tile["tile_path"],
                "mask_path": mask_path,
                "dataset": dataset,
                "has_label": "yes",
            })
        else:
            unlabeled += 1
            label_index.append({
                "tile_path": tile["tile_path"],
                "mask_path": "",
                "dataset": dataset,
                "has_label": "no",
            })

        if (i + 1) % 200 == 0:
            print(f"    {i+1}/{len(tiles)} tiles processed ({labeled} labeled)...")

    # Save label index
    label_index_path = os.path.join(masks_dir, "label_index.csv")
    with open(label_index_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["tile_path", "mask_path", "dataset", "has_label"])
        writer.writeheader()
        writer.writerows(label_index)

    print(f"\n{'='*70}")
    print(f"  MASK PREPARATION COMPLETE")
    print(f"  Labeled tiles:   {labeled}")
    print(f"  Unlabeled tiles: {unlabeled}")
    print(f"  Masks dir:       {masks_dir}")
    print(f"  Label index:     {label_index_path}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
