#!/usr/bin/env python3
"""
Evaluate Building IoU from Decomposed Prediction Shapefiles
===========================================================
Computes building-only IoU (and precision/recall/F1) between decomposed
prediction shapefiles and training labels (Built_Up_Area_type), per dataset
and as an overall aggregate.

This script is vector-based (area overlap), unlike tile-mask evaluation used
inside model training. It follows the same TP/FP/FN metric style:
  - IoU       = TP / (TP + FP + FN)
  - Precision = TP / (TP + FP)
  - Recall    = TP / (TP + FN)
  - F1        = 2PR / (P + R)

Usage:
  cd <bundle root>
  python -m dinov3_hrdecoder_pipeline.inference.evaluate_iou

  # Custom paths (defaults are pulled from configs/train.yaml otherwise)
  python -m dinov3_hrdecoder_pipeline.inference.evaluate_iou \
      --pred-dir   /path/to/decomposed_train_villages \
      --tile-index <bundle>/tiles/tile_index.csv \
      --gt-shp     <bundle>/dataset/train/labels/CG/Built_Up_Area_type.shp \
      --gt-shp     <bundle>/dataset/train/labels/PB/Built_Up_Area_type.shp
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import sys
from pathlib import Path

try:
    import geopandas as gpd
    import yaml
    from shapely.geometry import box
    from shapely.ops import unary_union
except ImportError as exc:
    print(f"ERROR: Missing dependency: {exc}")
    print("Install required packages: geopandas shapely fiona pyproj pyyaml")
    sys.exit(1)

EPS = 1e-9
PIPE_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CFG = PIPE_ROOT / "configs" / "train.yaml"


def safe_ratio(num: float, den: float) -> float:
    return num / den if den > EPS else 0.0


def parse_args() -> argparse.Namespace:
    # Resolve defaults from train.yaml so they follow the bundle workspace.
    cfg_paths: dict = {}
    try:
        with open(DEFAULT_CFG) as f:
            cfg_paths = yaml.safe_load(f).get("paths", {})
    except Exception:
        pass  # fall back to None defaults; user can pass explicit flags

    workspace = cfg_paths.get("workspace", "")
    default_pred_dir = (
        os.path.join(workspace, "outputs", "decomposed_train_villages")
        if workspace else None
    )
    default_tile_index = cfg_paths.get("tile_index")
    labels_dir = cfg_paths.get("labels_dir") or (
        os.path.join(workspace, "dataset", "train", "labels") if workspace else ""
    )
    default_gt_shps = [
        os.path.join(workspace, "dataset", "train", "labels", "CG",
                     "Built_Up_Area_type.shp"),
        os.path.join(workspace, "dataset", "train", "labels", "PB",
                     "Built_Up_Area_type.shp"),
    ] if workspace else []
    default_out_csv = (
        os.path.join(workspace, "outputs", "evaluation",
                     "building_iou_decomposed.csv")
        if workspace else None
    )

    parser = argparse.ArgumentParser(
        description="Compute building-only IoU from decomposed shapefiles"
    )
    parser.add_argument(
        "--config", default=str(DEFAULT_CFG),
        help="Path to train.yaml (used to resolve default paths)",
    )
    parser.add_argument(
        "--pred-dir", default=default_pred_dir,
        help="Directory containing *_pred_shp_Rectangles_Decomposed.shp",
    )
    parser.add_argument(
        "--pred-pattern",
        default="*_pred_shp_Rectangles_Decomposed.shp",
        help="Glob pattern for prediction shapefiles inside --pred-dir",
    )
    parser.add_argument(
        "--tile-index", default=default_tile_index,
        help="tile_index.csv used to derive dataset AOI extents",
    )
    parser.add_argument(
        "--gt-shp", action="append", default=default_gt_shps,
        help="Ground-truth building shapefile path(s). Repeatable.",
    )
    parser.add_argument(
        "--output-csv", default=default_out_csv,
        help="Output CSV for per-dataset and overall metrics",
    )
    return parser.parse_args()


def build_dataset_aoi_map(tile_index_csv: str) -> dict[str, dict]:
    """Build per-dataset AOI bounding boxes from tile_index.csv."""
    aoi_map: dict[str, dict] = {}

    with open(tile_index_csv, newline="") as f:
        for row in csv.DictReader(f):
            ds = row["dataset"]
            min_x = float(row["min_x"])
            min_y = float(row["min_y"])
            max_x = float(row["max_x"])
            max_y = float(row["max_y"])
            crs = row["crs"]

            if ds not in aoi_map:
                aoi_map[ds] = {
                    "min_x": min_x,
                    "min_y": min_y,
                    "max_x": max_x,
                    "max_y": max_y,
                    "crs": crs,
                }
            else:
                rec = aoi_map[ds]
                rec["min_x"] = min(rec["min_x"], min_x)
                rec["min_y"] = min(rec["min_y"], min_y)
                rec["max_x"] = max(rec["max_x"], max_x)
                rec["max_y"] = max(rec["max_y"], max_y)

    return aoi_map


def normalize_geometries(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Clean invalid/empty geometries and dissolve into a valid geometry set."""
    import warnings
    warnings.filterwarnings('ignore', 'GeoSeries.notna', UserWarning)
    
    if gdf is None or len(gdf) == 0:
        return gpd.GeoDataFrame(geometry=[], crs=getattr(gdf, "crs", None))

    gdf = gdf[~gdf.is_empty & gdf.notna().any(axis=1)].copy()
    if len(gdf) == 0:
        return gpd.GeoDataFrame(geometry=[], crs=gdf.crs)

    # buffer(0) is a practical fix for many self-intersection issues.
    gdf["geometry"] = gdf.geometry.buffer(0)
    gdf = gdf[~gdf.is_empty & gdf.notna().any(axis=1)].copy()
    return gdf


def load_gt_by_crs(gt_paths: list[str], target_crs: str) -> gpd.GeoDataFrame:
    parts = []
    for path in gt_paths:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Ground-truth shapefile not found: {path}")
        gdf = gpd.read_file(path)
        if gdf.crs is None:
            raise ValueError(f"GT shapefile has no CRS: {path}")
        if str(gdf.crs) != str(target_crs):
            gdf = gdf.to_crs(target_crs)
        gdf = normalize_geometries(gdf)
        if len(gdf) > 0:
            parts.append(gdf[["geometry"]])

    if not parts:
        return gpd.GeoDataFrame(geometry=[], crs=target_crs)

    merged = gpd.GeoDataFrame(
        geometry=gpd.GeoSeries(
            [geom for part in parts for geom in part.geometry], crs=target_crs
        ),
        crs=target_crs,
    )
    return normalize_geometries(merged)


def extract_dataset_name(pred_path: str) -> str:
    stem = Path(pred_path).stem
    suffix = "_pred_shp_Rectangles_Decomposed"
    if stem.endswith(suffix):
        return stem[: -len(suffix)]
    return stem


def compute_area_metrics(pred_geom, gt_geom) -> dict[str, float]:
    pred_area = pred_geom.area if pred_geom is not None and not pred_geom.is_empty else 0.0
    gt_area = gt_geom.area if gt_geom is not None and not gt_geom.is_empty else 0.0

    if pred_area <= EPS and gt_area <= EPS:
        # If both are empty in AOI, treat as perfect overlap.
        return {
            "tp": 0.0,
            "fp": 0.0,
            "fn": 0.0,
            "pred_area": 0.0,
            "gt_area": 0.0,
            "intersection_area": 0.0,
            "union_area": 0.0,
            "iou": 1.0,
            "precision": 1.0,
            "recall": 1.0,
            "f1": 1.0,
        }

    if pred_area <= EPS:
        inter = 0.0
        union = gt_area
    elif gt_area <= EPS:
        inter = 0.0
        union = pred_area
    else:
        inter = pred_geom.intersection(gt_geom).area
        union = pred_area + gt_area - inter

    tp = inter
    fp = max(pred_area - inter, 0.0)
    fn = max(gt_area - inter, 0.0)

    precision = safe_ratio(tp, tp + fp)
    recall = safe_ratio(tp, tp + fn)
    f1 = safe_ratio(2.0 * precision * recall, precision + recall)
    iou = safe_ratio(tp, tp + fp + fn)

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "pred_area": pred_area,
        "gt_area": gt_area,
        "intersection_area": inter,
        "union_area": union,
        "iou": iou,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def main() -> None:
    args = parse_args()

    pred_paths = sorted(glob.glob(os.path.join(args.pred_dir, args.pred_pattern)))
    if not pred_paths:
        print("ERROR: No prediction shapefiles found.")
        print(f"  dir: {args.pred_dir}")
        print(f"  pattern: {args.pred_pattern}")
        sys.exit(1)

    aoi_map = build_dataset_aoi_map(args.tile_index)
    gt_cache: dict[str, gpd.GeoDataFrame] = {}

    print("=" * 72)
    print("  BUILDING IoU EVALUATION (DECOMPOSED SHAPEFILES)")
    print(f"  Prediction files: {len(pred_paths)}")
    print(f"  Tile index:       {args.tile_index}")
    print("=" * 72)

    rows = []
    total_tp = 0.0
    total_fp = 0.0
    total_fn = 0.0

    for i, pred_path in enumerate(pred_paths, 1):
        ds_name = extract_dataset_name(pred_path)

        if ds_name not in aoi_map:
            print(f"  WARN [{i}/{len(pred_paths)}] Dataset not found in tile_index: {ds_name}")
            continue

        info = aoi_map[ds_name]
        ds_crs = info["crs"]
        aoi_poly = box(info["min_x"], info["min_y"], info["max_x"], info["max_y"])

        pred_gdf = gpd.read_file(pred_path)
        if pred_gdf.crs is None:
            raise ValueError(f"Prediction shapefile has no CRS: {pred_path}")
        if str(pred_gdf.crs) != str(ds_crs):
            pred_gdf = pred_gdf.to_crs(ds_crs)
        pred_gdf = normalize_geometries(pred_gdf)

        if ds_crs not in gt_cache:
            gt_cache[ds_crs] = load_gt_by_crs(args.gt_shp, ds_crs)
        gt_gdf = gt_cache[ds_crs]

        # Intersect with bounding box first to avoid extremely expensive union over the entire state/region
        pred_sub = pred_gdf.clip(aoi_poly) if len(pred_gdf) > 0 else pred_gdf
        gt_sub = gt_gdf.clip(aoi_poly) if len(gt_gdf) > 0 else gt_gdf

        pred_union = unary_union(pred_sub.geometry.tolist()) if len(pred_sub) > 0 else None
        gt_union = unary_union(gt_sub.geometry.tolist()) if len(gt_sub) > 0 else None

        pred_clip = pred_union
        gt_clip = gt_union

        m = compute_area_metrics(pred_clip, gt_clip)
        total_tp += m["tp"]
        total_fp += m["fp"]
        total_fn += m["fn"]

        rows.append(
            {
                "dataset": ds_name,
                "pred_file": pred_path,
                "pred_area": m["pred_area"],
                "gt_area": m["gt_area"],
                "intersection_area": m["intersection_area"],
                "union_area": m["union_area"],
                "iou": m["iou"],
                "precision": m["precision"],
                "recall": m["recall"],
                "f1": m["f1"],
            }
        )

        print(
            f"  [{i:>2}/{len(pred_paths)}] {ds_name:<70.70s} "
            f"IoU={m['iou']:.4f}  F1={m['f1']:.4f}  "
            f"P={m['precision']:.4f}  R={m['recall']:.4f}"
        )

    overall_iou = safe_ratio(total_tp, total_tp + total_fp + total_fn)
    overall_precision = safe_ratio(total_tp, total_tp + total_fp)
    overall_recall = safe_ratio(total_tp, total_tp + total_fn)
    overall_f1 = safe_ratio(
        2.0 * overall_precision * overall_recall,
        overall_precision + overall_recall,
    )

    rows_sorted = sorted(rows, key=lambda x: x["dataset"])

    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    with open(args.output_csv, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "dataset",
                "pred_file",
                "pred_area",
                "gt_area",
                "intersection_area",
                "union_area",
                "iou",
                "precision",
                "recall",
                "f1",
            ],
        )
        writer.writeheader()
        writer.writerows(rows_sorted)
        writer.writerow(
            {
                "dataset": "OVERALL",
                "pred_file": "-",
                "pred_area": total_tp + total_fp,
                "gt_area": total_tp + total_fn,
                "intersection_area": total_tp,
                "union_area": total_tp + total_fp + total_fn,
                "iou": overall_iou,
                "precision": overall_precision,
                "recall": overall_recall,
                "f1": overall_f1,
            }
        )

    print("\n" + "=" * 72)
    print("  OVERALL BUILDING METRICS")
    print("=" * 72)
    print(f"  Datasets evaluated: {len(rows_sorted)}")
    print(f"  IoU:       {overall_iou:.4f}")
    print(f"  F1:        {overall_f1:.4f}")
    print(f"  Precision: {overall_precision:.4f}")
    print(f"  Recall:    {overall_recall:.4f}")
    print(f"  Saved CSV: {args.output_csv}")
    print("=" * 72)


if __name__ == "__main__":
    main()
