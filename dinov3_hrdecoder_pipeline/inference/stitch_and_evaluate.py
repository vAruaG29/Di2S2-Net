#!/usr/bin/env python3
"""
Stitch Tile Predictions into Full Image + Evaluate
====================================================
Reassembles tile predictions into the full NAGUL image
and computes metrics against ground truth (if available).

Usage:
    cd <bundle root>
    python -m dinov3_hrdecoder_pipeline.inference.stitch_and_evaluate
"""

import os
import sys
import csv
import argparse
from pathlib import Path

import numpy as np
import yaml
import rasterio
from rasterio.transform import from_bounds

PIPE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PIPE_ROOT.parent))

from dinov3_hrdecoder_pipeline.models.metrics import SegmentationMetrics
from dinov3_hrdecoder_pipeline.inference._timing import StepTimer, now_hms

import time as _time_mod


def load_config():
    with open(PIPE_ROOT / "configs" / "train.yaml") as f:
        return yaml.safe_load(f)


def get_source_raster_info(tile_index_path: str, dataset_name: str):
    """Get source raster dimensions and transform from tile index."""
    tiles = []
    source_raster = None

    with open(tile_index_path) as f:
        for row in csv.DictReader(f):
            if row["dataset"] == dataset_name:
                tiles.append(row)
                if source_raster is None:
                    source_raster = row.get("source_raster", "")

    if not tiles:
        return None, None, None, None

    # Compute full extent from tile metadata
    max_col_off = max(int(t["col_off"]) + int(t["width"]) for t in tiles)
    max_row_off = max(int(t["row_off"]) + int(t["height"]) for t in tiles)

    # Get CRS and bounds from source raster if available
    if source_raster and os.path.exists(source_raster):
        with rasterio.open(source_raster) as src:
            return src.width, src.height, src.transform, src.crs

    # ── Fallback 1: recover CRS + transform from a tile GeoTIFF ──
    # Each tile was written with a proper transform by tile_raster.py.
    # We reconstruct the full-image transform from the first tile's
    # transform and its pixel offset within the full image.
    for t in tiles:
        tile_path = t["tile_path"]
        if os.path.exists(tile_path):
            try:
                with rasterio.open(tile_path) as src:
                    tile_transform = src.transform
                    tile_crs = src.crs
                if tile_crs is not None and tile_transform is not None:
                    col_off = int(t["col_off"])
                    row_off = int(t["row_off"])
                    # The tile's origin = full_transform * (col_off, row_off)
                    # So full_transform origin = tile_origin - pixel_size * offset
                    pixel_w = tile_transform.a   # positive
                    pixel_h = tile_transform.e   # negative
                    origin_x = tile_transform.c - col_off * pixel_w
                    origin_y = tile_transform.f - row_off * pixel_h
                    full_transform = rasterio.transform.Affine(
                        pixel_w, 0, origin_x,
                        0, pixel_h, origin_y,
                    )
                    print(f"  INFO: Recovered georeferencing from tile {tile_path}")
                    return max_col_off, max_row_off, full_transform, tile_crs
            except Exception:
                continue

    # ── Fallback 2: recover from tile_index metadata (min_x/min_y etc.) ──
    if "crs" in tiles[0] and "min_x" in tiles[0]:
        try:
            from rasterio.crs import CRS
            tile_crs = CRS.from_user_input(tiles[0]["crs"])
            # Compute full extent from all tiles
            full_min_x = min(float(t["min_x"]) for t in tiles)
            full_min_y = min(float(t["min_y"]) for t in tiles)
            full_max_x = max(float(t["max_x"]) for t in tiles)
            full_max_y = max(float(t["max_y"]) for t in tiles)
            full_transform = from_bounds(
                full_min_x, full_min_y, full_max_x, full_max_y,
                max_col_off, max_row_off,
            )
            print(f"  INFO: Recovered georeferencing from tile_index metadata")
            return max_col_off, max_row_off, full_transform, tile_crs
        except Exception:
            pass

    # Final fallback: no georeferencing available
    print("  WARN: Could not recover georeferencing from tiles or metadata")
    return max_col_off, max_row_off, None, None


def stitch_predictions(pred_dir: str, tile_index_path: str, dataset_name: str,
                       output_path: str, source_type: str = "predictions",
                       timer: "StepTimer | None" = None):
    """
    Stitch tile predictions into a single full-resolution raster.

    Uses rasterio windowed writing to avoid holding the full image in RAM.
    Peak memory = one tile at a time (~few MB) instead of 2× full_h×full_w
    float32 arrays (which can be 10+ GB for large datasets).

    For overlapping pixels, last-write-wins is used since predictions are
    discrete class labels (0-6) and averaging would produce non-integer values.
    """
    if timer is None:
        timer = StepTimer()

    with timer.step("stitch / get_source_raster_info", indent=6):
        full_w, full_h, transform, crs = get_source_raster_info(
            tile_index_path, dataset_name
        )
    if full_w is None:
        print(f"  ERROR: No tiles found for {dataset_name}")
        return None

    print(f"  [{now_hms()}] Full image: {full_w}x{full_h}")
    old_mem_gb = 2 * full_h * full_w * 4 / (1024**3)
    print(f"  (windowed write mode — avoids {old_mem_gb:.1f} GB in-memory allocation)")

    with timer.step("stitch / load tile metadata from tile_index.csv", indent=6):
        tile_meta = {}
        with open(tile_index_path) as f:
            for row in csv.DictReader(f):
                if row["dataset"] == dataset_name:
                    tile_meta[row["tile_path"]] = row

    suffix = "_pred.tif" if source_type == "predictions" else "_refined.tif"
    index_file = "prediction_index.csv" if source_type == "predictions" else "refined_index.csv"
    idx_path = os.path.join(pred_dir, index_file)
    if not os.path.exists(idx_path):
        print(f"  ERROR: Index not found: {idx_path}")
        return None

    with timer.step("stitch / read prediction_index.csv", indent=6):
        with open(idx_path) as f:
            entries = list(csv.DictReader(f))

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    profile = {
        "driver": "GTiff",
        "width": full_w,
        "height": full_h,
        "count": 1,
        "dtype": "uint8",
        "compress": "deflate",
        "tiled": True,
        "blockxsize": 512,
        "blockysize": 512,
    }
    if transform is not None:
        profile["transform"] = transform
    if crs is not None:
        profile["crs"] = crs

    read_t   = timer.cumulative("stitch / per-tile rasterio.read", indent=6)
    resize_t = timer.cumulative("stitch / per-tile resize-to-meta", indent=6)
    write_t  = timer.cumulative("stitch / per-tile windowed write", indent=6)

    placed = 0
    with timer.step("stitch / per-tile placement loop", indent=6):
        with rasterio.open(output_path, "w", **profile) as dst:
            loop_t0 = _time_mod.time()
            for entry in entries:
                tile_path = entry["tile_path"]
                if tile_path not in tile_meta:
                    continue

                meta = tile_meta[tile_path]
                col_off = int(meta["col_off"])
                row_off = int(meta["row_off"])
                w = int(meta["width"])
                h = int(meta["height"])

                pred_file = entry.get("pred_path", "")
                if not os.path.exists(pred_file):
                    continue

                with read_t:
                    with rasterio.open(pred_file) as src:
                        pred = src.read(1)

                if pred.shape != (h, w):
                    with resize_t:
                        from PIL import Image
                        pred_img = Image.fromarray(pred.astype(np.uint8))
                        pred_img = pred_img.resize((w, h), Image.NEAREST)
                        pred = np.array(pred_img)

                r_end = min(row_off + h, full_h)
                c_end = min(col_off + w, full_w)
                ph = r_end - row_off
                pw = c_end - col_off

                window = rasterio.windows.Window(col_off, row_off, pw, ph)
                with write_t:
                    dst.write(pred[:ph, :pw].astype(np.uint8)[np.newaxis], window=window)
                placed += 1

                if placed % 500 == 0:
                    rate = placed / (_time_mod.time() - loop_t0)
                    print(f"    [{now_hms()}] {placed}/{len(entries)} "
                          f"tiles placed  ({rate:.1f} tiles/s)")

    read_t.flush()
    resize_t.flush()
    write_t.flush()

    print(f"  [{now_hms()}] Stitched {placed} tiles → {output_path}")
    return output_path


def evaluate_tile_predictions(idx_path: str, masks_dir: str, cfg: dict,
                               pred_key: str = "pred_path", label: str = "DINOv3",
                               timer: "StepTimer | None" = None):
    """
    Evaluate tile-level predictions against ground truth masks.

    Args:
        idx_path:  path to prediction_index.csv
        masks_dir: directory containing GT masks ({tile_stem}_mask.tif)
        cfg:       pipeline config
        pred_key:  column name for the prediction path in the CSV
        label:     label for printout (e.g. "DINOv3+HRDecoder")

    Returns:
        (metrics dict, summary string) or (None, None) if no evaluation was possible
    """
    if timer is None:
        timer = StepTimer()
    if not os.path.exists(idx_path):
        print(f"  WARN: Index not found: {idx_path}")
        return None, None

    num_classes = len(cfg["classes"]) + 1  # +1 background
    class_names = ["background"] + [c["name"] for c in cfg["classes"]]
    seg_metrics = SegmentationMetrics(num_classes, class_names)

    with timer.step("evaluate / read prediction_index.csv", indent=6):
        with open(idx_path) as f:
            entries = list(csv.DictReader(f))

    read_t   = timer.cumulative("evaluate / per-tile read pred+mask", indent=6)
    update_t = timer.cumulative("evaluate / per-tile metric update",  indent=6)

    evaluated = 0
    import torch
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from PIL import Image

    def _load_pair(entry):
        """Worker: read prediction + GT mask for one tile.

        Returns (pred, gt) or None if either is missing / GT empty.
        Rasterio releases the GIL on read, so threads get real
        parallelism here.
        """
        tile_stem = Path(entry["tile_path"]).stem
        mask_path = os.path.join(masks_dir, f"{tile_stem}_mask.tif")
        if not os.path.exists(mask_path):
            return None
        pred_file = entry.get(pred_key, "")
        if not pred_file or not os.path.exists(pred_file):
            return None
        with rasterio.open(pred_file) as src:
            pred = src.read(1)
        with rasterio.open(mask_path) as src:
            gt = src.read(1)
        if gt.max() == 0:
            return None
        if pred.shape != gt.shape:
            pred_img = Image.fromarray(pred.astype(np.uint8))
            pred_img = pred_img.resize((gt.shape[1], gt.shape[0]), Image.NEAREST)
            pred = np.array(pred_img)
        return pred, gt

    # Walk the tiles in parallel for IO, then update metrics serially
    # (SegmentationMetrics is not thread-safe).
    n_workers = max(4, (os.cpu_count() or 4))
    with timer.step("evaluate / per-tile loop", indent=6):
        loop_t0 = _time_mod.time()
        with ThreadPoolExecutor(max_workers=n_workers) as ex:
            futures = [ex.submit(_load_pair, e) for e in entries]
            for fut in as_completed(futures):
                with read_t:
                    pair = fut.result()
                if pair is None:
                    continue
                pred, gt = pair
                with update_t:
                    pred_t = torch.from_numpy(pred.astype(np.int64)).unsqueeze(0)
                    gt_t = torch.from_numpy(gt.astype(np.int64)).unsqueeze(0)
                    seg_metrics.update(pred_t, gt_t)
                evaluated += 1
                if evaluated and evaluated % 500 == 0:
                    rate = evaluated / (_time_mod.time() - loop_t0)
                    print(f"    [{now_hms()}] evaluated {evaluated} tiles "
                          f"({rate:.1f} tiles/s)")

    read_t.flush()
    update_t.flush()

    if evaluated == 0:
        print(f"  WARN: No tiles with ground truth found for {label} evaluation.")
        return None, None

    with timer.step("evaluate / compute final metrics", indent=6):
        result = seg_metrics.compute()
        summary = f"\n  {label} Evaluation ({evaluated} labeled tiles)\n"
        summary += seg_metrics.summary_str()
    print(summary)
    return result, summary


def save_eval_report(results: dict, path: str, label: str):
    """Save metrics to CSV."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["source", "metric", "value"])
        for key, val in sorted(results.items()):
            writer.writerow([label, key, f"{val:.6f}"])
    print(f"  Saved report → {path}")


def main():
    parser = argparse.ArgumentParser(description="Stitch predictions + evaluate")
    parser.add_argument("--eval-only", action="store_true",
                        help="Skip stitching, only compute metrics")
    args = parser.parse_args()

    cfg = load_config()
    test_dataset = cfg["test_dataset"]
    output_dir = cfg["paths"]["output_dir"]
    tile_index = cfg["paths"]["tile_index"]
    masks_dir = os.path.join(cfg["paths"]["labels_dir"], test_dataset)

    pred_dir = os.path.join(output_dir, "predictions", test_dataset)
    stitched_pred_path = os.path.join(output_dir, "stitched",
                                       f"{test_dataset}_pred.tif")

    timer = StepTimer()

    if not args.eval_only:
        print("=" * 70)
        print(f"  [{now_hms()}] Stitch Tile Predictions → Full Image + Evaluate")
        print(f"  Dataset: {test_dataset}")
        print("=" * 70)

        with timer.step("stitch_predictions (top-level)", indent=4):
            stitch_predictions(
                pred_dir, tile_index, test_dataset, stitched_pred_path,
                source_type="predictions", timer=timer,
            )
    else:
        print("=" * 70)
        print(f"  [{now_hms()}] Evaluate Only (skipping stitching)")
        print(f"  Dataset: {test_dataset}")
        print("=" * 70)

    # ── Evaluate predictions against ground truth ──
    print(f"\n{'='*70}")
    print(f"  [{now_hms()}] EVALUATION (tile-level, labeled tiles only)")
    print(f"{'='*70}")

    eval_dir = os.path.join(output_dir, "evaluation")
    os.makedirs(eval_dir, exist_ok=True)

    pred_idx_path = os.path.join(pred_dir, "prediction_index.csv")
    with timer.step("evaluate_tile_predictions (top-level)", indent=4):
        results, summary = evaluate_tile_predictions(
            pred_idx_path, masks_dir, cfg,
            pred_key="pred_path", label="DINOv3+HRDecoder",
            timer=timer,
        )

    if results:
        save_eval_report(
            results,
            os.path.join(eval_dir, "dinov3_metrics.csv"),
            "DINOv3+HRDecoder",
        )
        report_path = os.path.join(eval_dir, "full_evaluation_report.txt")
        with open(report_path, "w") as f:
            f.write(summary + "\n")
        print(f"  Saved full text report → {report_path}")

    print(f"\n{'='*70}")
    print(f"  [{now_hms()}] COMPLETE")
    print(f"  Prediction: {stitched_pred_path}")
    print(f"  Evaluation: {eval_dir}")
    print(f"{'='*70}\n")

    timer.print_summary(title="Detailed step-wise timing")


if __name__ == "__main__":
    main()

