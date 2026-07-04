#!/usr/bin/env python3
"""
Batch Stitched-to-GeoPackage Conversion
=========================================
Reads dataset names from a text file and converts the corresponding
stitched prediction GeoTIFF files to GeoPackage.

Layer filtering (compared to predictions_to_gpkg.py):
  - Built_Up_Area_type  (Polygon)
  - Road                (Polygon)       ← NO Road_Centre_Line
  - Water_Body          (Polygon)       ← NO Water_Body_Line, NO Waterbody_Point
  - Utility_Poly        (Polygon)  + Utility (Point)  ← both kept, poly shrunk
  - Bridge              (Polygon)
  - Railway             (Line)

Output goes to a new `gpkg/` directory next to the stitched directory.

Usage:
    python batch_stitched_to_gpkg.py \\
        --file-list datasets.txt \\
        --stitched-dir /path/to/stitched \\
        --utility-buffer -1.0

    # datasets.txt contains one dataset name per line:
    # NAGUL_450171_MADASE_450172_GHOTPAL_450137_ORTHO
    # BASANTPUR_434297_ORTHO
"""

import os
import sys
import shutil
import tempfile
import argparse
import time
from pathlib import Path

import numpy as np

# ── Setup path so imports from the pipeline work ────────────────────────────
PIPE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PIPE_ROOT.parent))

# Import existing conversion machinery
import dinov3_hrdecoder_pipeline.inference.predictions_to_gpkg as gpkg_mod
from dinov3_hrdecoder_pipeline.inference.predictions_to_gpkg import (
    load_config,
    get_class_info,
    stitched_to_gpkg,
    write_qgis_layer_style,
    LAYER_COLOURS,
)

try:
    import geopandas as gpd
    import fiona
except ImportError as e:
    print(f"ERROR: Missing dependency: {e}")
    sys.exit(1)


# ── Filtered geometry map ───────────────────────────────────────────────────
# Only the layers we want: no Road line, no Water_Body line/point
GEOMETRY_MAP_FILTERED = {
    1: [  # Built_Up_Area
        ("Built_Up_Area_type", "polygon"),
    ],
    2: [  # Road — polygon only, no centre line
        ("Road", "polygon"),
    ],
    3: [  # Water_Body — polygon only, no line, no point
        ("Water_Body", "polygon"),
    ],
    4: [  # Utility — keep both polygon and point
        ("Utility_Poly", "polygon"),
        ("Utility", "point"),
    ],
    5: [  # Bridge
        ("Bridge", "polygon"),
    ],
    6: [  # Railway
        ("Railway", "line"),
    ],
}


def shrink_utility_polygons(gpkg_path: str, buffer_distance: float):
    """
    Apply a negative buffer (erosion) to Utility_Poly polygons in a GPKG
    to reduce their size.

    Args:
        gpkg_path: Path to the GeoPackage file.
        buffer_distance: Negative buffer distance in CRS units (e.g. -1.0).
                        Must be negative to shrink.
    """
    if buffer_distance >= 0:
        print(f"    WARN: buffer_distance is non-negative ({buffer_distance}), "
              f"skipping utility shrink")
        return

    try:
        layers = fiona.listlayers(gpkg_path)
    except Exception:
        return

    if "Utility_Poly" not in layers:
        return

    print(f"    Shrinking Utility_Poly with buffer={buffer_distance}...", end=" ",
          flush=True)

    gdf = gpd.read_file(gpkg_path, layer="Utility_Poly")
    if len(gdf) == 0:
        print("no features, skipped")
        return

    original_area = gdf.geometry.area.sum()

    # Apply negative buffer to shrink polygons
    gdf["geometry"] = gdf.geometry.buffer(buffer_distance)

    # Remove any geometries that became empty after shrinking
    gdf = gdf[~gdf.geometry.is_empty & gdf.geometry.is_valid]

    # Remove any geometries that collapsed to non-polygon types
    gdf = gdf[gdf.geometry.geom_type.isin(["Polygon", "MultiPolygon"])]

    if len(gdf) == 0:
        print("all polygons collapsed, removing layer")
        return

    # Update area column
    gdf["area"] = gdf.geometry.area
    new_area = gdf.geometry.area.sum()

    # Rewrite the layer using the fast pyogrio writer when available.
    gpkg_mod._write_gpkg_layer(gdf, gpkg_path, "Utility_Poly")
    reduction = (1 - new_area / original_area) * 100 if original_area > 0 else 0
    print(f"{len(gdf)} features, area reduced by {reduction:.1f}%")


def convert_single(stitched_path: str, output_path: str,
                   class_info: list, cfg: dict,
                   utility_buffer: float, workers: int,
                   downsample: int = 1,
                   simplify_tol: float = 0.0,
                   min_area: float = 0.0):
    """
    Convert a single stitched GeoTIFF to a filtered GPKG.

    Writes to a local temp file first, then moves to the final output path.
    This avoids SQLite transaction failures on NAS/network filesystems
    (GPKG is SQLite-based and doesn't handle network FS locking well).
    """
    if not os.path.exists(stitched_path):
        print(f"    ⚠️  File not found: {stitched_path}")
        return False

    print(f"\n    Input:  {stitched_path}")
    print(f"    Output: {output_path}")

    # ── Write to a LOCAL temp file to avoid NAS/SQLite transaction issues ──
    # Use /tmp (local filesystem) for the intermediate GPKG, then copy out.
    tmp_fd, tmp_gpkg = tempfile.mkstemp(suffix=".gpkg", prefix="batch_gpkg_")
    os.close(tmp_fd)
    # Remove the empty file so stitched_to_gpkg can create it fresh
    os.remove(tmp_gpkg)

    print(f"    (writing to local temp: {tmp_gpkg})")

    # Temporarily replace the GEOMETRY_MAP in the module for the
    # sequential / single-process path. For workers > 1 we also pass
    # the map explicitly via the `geometry_map` arg so it propagates
    # across the multiprocessing pickle boundary (parent-process
    # monkey-patches don't carry to spawned workers).
    original_map = gpkg_mod.GEOMETRY_MAP
    gpkg_mod.GEOMETRY_MAP = GEOMETRY_MAP_FILTERED

    # ── FAST PATH: per-tile predictions are usually still on disk in
    #   <output_dir>/predictions/<DS>/*_pred.tif. Polygonising those
    #   1024² tiles in parallel is 5-20× faster than polygonising the
    #   full stitched raster (which can be tens of gigapixels). Auto-
    #   detect and prefer that path; fall back to stitched if missing.
    ds_name = os.path.basename(stitched_path).removesuffix("_pred.tif")
    stitched_dir = os.path.dirname(stitched_path)
    pred_dir = os.path.join(os.path.dirname(stitched_dir),
                            "predictions", ds_name)
    use_tile_path = (
        os.path.isdir(pred_dir)
        and any(p.endswith("_pred.tif") for p in os.listdir(pred_dir))
    )

    try:
        if use_tile_path:
            print(f"\n    Fast path: polygonising per-tile predictions")
            print(f"    Pred dir: {pred_dir}")
            total_features, layers_written = gpkg_mod.tiles_to_gpkg(
                pred_dir, tmp_gpkg, class_info,
                pred_suffix="_pred.tif",
                simplify_tol=simplify_tol,
                min_area=min_area,
                dissolve=True,
                workers=max(workers, 4),  # threads, cheap
                geometry_map=GEOMETRY_MAP_FILTERED,
            )
        else:
            total_features, layers_written = stitched_to_gpkg(
                stitched_path, tmp_gpkg, class_info, cfg,
                workers=workers,
                downsample=downsample,
                simplify_tol=simplify_tol,
                min_area=min_area,
                geometry_map=GEOMETRY_MAP_FILTERED,
            )
    finally:
        # Always restore the original map
        gpkg_mod.GEOMETRY_MAP = original_map

    if layers_written == 0:
        print(f"    ⚠️  No layers written")
        # Clean up temp file
        if os.path.exists(tmp_gpkg):
            os.remove(tmp_gpkg)
        return False

    # Shrink utility polygons if requested (still on local temp file)
    if utility_buffer < 0:
        shrink_utility_polygons(tmp_gpkg, utility_buffer)

    # Write QGIS layer styles (still on local temp file)
    if os.path.exists(tmp_gpkg):
        for layer_name in fiona.listlayers(tmp_gpkg):
            if layer_name == "layer_styles":
                continue
            colour = LAYER_COLOURS.get(layer_name, "#808080")
            if layer_name in ("Utility",):
                gt = "point"
            elif layer_name in ("Railway",):
                gt = "line"
            else:
                gt = "polygon"
            write_qgis_layer_style(tmp_gpkg, layer_name, colour, gt)

    # ── Move temp file to final output path ─────────────────────────────────
    if os.path.exists(output_path):
        os.remove(output_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    shutil.move(tmp_gpkg, output_path)
    print(f"    Moved to final location: {output_path}")

    # Verify
    print(f"\n    ── Verification ──")
    if os.path.exists(output_path):
        layers = fiona.listlayers(output_path)
        data_layers = [l for l in layers if l != "layer_styles"]
        print(f"    Layers: {data_layers}")
        for layer_name in data_layers:
            try:
                gdf = gpd.read_file(output_path, layer=layer_name)
                geom_types = gdf.geometry.geom_type.unique() if len(gdf) > 0 else []
                print(f"      {layer_name}: {len(gdf)} features, "
                      f"types={list(geom_types)}")
            except Exception as e:
                print(f"      {layer_name}: (error: {e})")
        fsize = os.path.getsize(output_path) / 1024 / 1024
        print(f"    File size: {fsize:.2f} MB")

    print(f"    ✅ {total_features:,} features in {layers_written} layers")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Batch convert stitched GeoTIFFs to filtered GeoPackages",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert all datasets listed in datasets.txt:
  python batch_stitched_to_gpkg.py --file-list datasets.txt \\
      --stitched-dir /path/to/final_results/stitched

  # With utility polygon shrinking:
  python batch_stitched_to_gpkg.py --file-list datasets.txt \\
      --stitched-dir /path/to/final_results/stitched \\
      --utility-buffer -1.0
""",
    )
    parser.add_argument("--file-list", required=True,
                        help="Text file with dataset names (one per line)")
    parser.add_argument("--stitched-dir", required=True,
                        help="Directory containing stitched GeoTIFFs "
                             "(e.g. final_results/stitched)")
    parser.add_argument("--output-dir", default=None,
                        help="Output directory for GPKGs (default: "
                             "sibling 'gpkg' dir next to stitched-dir)")
    parser.add_argument("--config", default=None,
                        help="Path to train.yaml")
    parser.add_argument("--utility-buffer", type=float, default=-1.0,
                        help="Negative buffer to shrink Utility_Poly "
                             "(CRS units, default: -1.0). Set to 0 to skip.")
    parser.add_argument("--workers", type=int, default=1,
                        help="Parallel workers for per-class processing")
    parser.add_argument("--downsample", type=int, default=1,
                        help="Read each stitched raster at 1/N resolution "
                             "before polygonising (N=2 → 4× faster, N=4 → "
                             "16× faster). Class IDs preserved via nearest "
                             "resampling. Default: 1 (full res).")
    parser.add_argument("--simplify", type=float, default=0.0,
                        help="Geometry simplification tolerance in CRS "
                             "units (applied INSIDE each worker so the "
                             "pickled return is small). Default: 0.")
    parser.add_argument("--min-area", type=float, default=0.0,
                        help="Drop polygons whose area is below this "
                             "threshold (CRS units²) inside each worker.")

    args = parser.parse_args()

    # ── Read dataset names ──────────────────────────────────────────────────
    if not os.path.exists(args.file_list):
        print(f"ERROR: File list not found: {args.file_list}")
        sys.exit(1)

    dataset_names = []
    with open(args.file_list) as f:
        for line in f:
            name = line.strip()
            if name and not name.startswith("#"):
                dataset_names.append(name)

    if not dataset_names:
        print("ERROR: No dataset names found in the file list")
        sys.exit(1)

    # ── Resolve paths ───────────────────────────────────────────────────────
    stitched_dir = os.path.abspath(args.stitched_dir)
    if not os.path.isdir(stitched_dir):
        print(f"ERROR: Stitched directory not found: {stitched_dir}")
        sys.exit(1)

    # Output directory: default is a sibling 'gpkg' directory
    if args.output_dir:
        output_dir = os.path.abspath(args.output_dir)
    else:
        output_dir = os.path.join(os.path.dirname(stitched_dir), "gpkg")
    os.makedirs(output_dir, exist_ok=True)

    # ── Load config and class info ──────────────────────────────────────────
    cfg = load_config(args.config)
    class_info = get_class_info(cfg)

    # ── Print banner ────────────────────────────────────────────────────────
    print("=" * 70)
    print("  Batch Stitched → GeoPackage (Filtered Layers)")
    print(f"  Datasets:       {len(dataset_names)}")
    print(f"  Stitched dir:   {stitched_dir}")
    print(f"  Output dir:     {output_dir}")
    print(f"  Utility buffer: {args.utility_buffer}")
    print(f"  Workers:        {args.workers}")
    print(f"  Downsample:     ×{args.downsample}")
    print(f"  Simplify tol:   {args.simplify}")
    print(f"  Min area:       {args.min_area}")
    print(f"  Layers: Built_Up_Area_type, Road, Water_Body, "
          f"Utility_Poly, Utility (point), Bridge, Railway")
    print("=" * 70)

    # ── Process each dataset ────────────────────────────────────────────────
    total_start = time.time()
    success_count = 0
    fail_count = 0

    for idx, ds_name in enumerate(dataset_names, 1):
        print(f"\n{'─' * 70}")
        print(f"  [{idx}/{len(dataset_names)}] {ds_name}")
        print(f"{'─' * 70}")

        ds_start = time.time()

        # ── Convert predicted stitched ──────────────────────────────────────
        pred_tif = os.path.join(stitched_dir, f"{ds_name}_pred.tif")
        pred_gpkg = os.path.join(output_dir, f"{ds_name}_pred.gpkg")

        print(f"\n  📌 Converting predicted stitched...")
        if convert_single(pred_tif, pred_gpkg, class_info, cfg,
                         args.utility_buffer, args.workers,
                         downsample=args.downsample,
                         simplify_tol=args.simplify,
                         min_area=args.min_area):
            success_count += 1
        else:
            fail_count += 1

        # ── Convert refined stitched ────────────────────────────────────────
        refined_tif = os.path.join(stitched_dir, f"{ds_name}_refined.tif")
        refined_gpkg = os.path.join(output_dir, f"{ds_name}_refined.gpkg")

        print(f"\n  📌 Converting refined stitched...")
        if convert_single(refined_tif, refined_gpkg, class_info, cfg,
                         args.utility_buffer, args.workers,
                         downsample=args.downsample,
                         simplify_tol=args.simplify,
                         min_area=args.min_area):
            success_count += 1
        else:
            fail_count += 1

        elapsed = time.time() - ds_start
        print(f"\n  ⏱  {ds_name} done in {elapsed:.1f}s")

    # ── Summary ─────────────────────────────────────────────────────────────
    total_elapsed = time.time() - total_start
    print(f"\n{'=' * 70}")
    print(f"  BATCH CONVERSION COMPLETE")
    print(f"  Datasets processed: {len(dataset_names)}")
    print(f"  Conversions:        {success_count} succeeded, {fail_count} failed")
    print(f"  Output directory:   {output_dir}")
    print(f"  Total time:         {total_elapsed:.1f}s")
    print(f"{'=' * 70}\n")

    # Exit non-zero when NOT A SINGLE GeoPackage was written. Callers (the
    # portal's pipeline_runner, CI, shell scripts) otherwise can't tell a
    # run that vectorised nothing apart from a successful one — the portal
    # used to treat that rc=0 as "done" and delete the stitched raster +
    # per-tile predictions in cleanup, leaving an unrecoverable empty
    # result. A partial run (some datasets ok, some failed) still exits 0.
    if success_count == 0:
        print("ERROR: no GeoPackages were produced — every conversion "
              "failed (predictions may be empty or below --min-area).")
        sys.exit(1)


if __name__ == "__main__":
    main()
