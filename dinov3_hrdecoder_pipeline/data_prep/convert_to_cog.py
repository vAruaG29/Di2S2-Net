#!/usr/bin/env python3
"""
Step 2: Convert Rasters to Cloud Optimized GeoTIFF (COG)
=========================================================
Converts all .tif and .ecw rasters in train/ and test/ to COG format
with DEFLATE compression and internal tiling.

Usage:
    python -m dinov3_hrdecoder_pipeline.data_prep.convert_to_cog
    python -m dinov3_hrdecoder_pipeline.data_prep.convert_to_cog --config dinov3_hrdecoder_pipeline/configs/data_prep.yaml
"""

import os
import sys
import argparse
import subprocess
import shutil
from pathlib import Path

import yaml
from osgeo import gdal, osr

# Reuse the CG/PB region heuristic from prepare_dataset so the layout
# stays consistent across data-prep steps.
from dinov3_hrdecoder_pipeline.data_prep.prepare_dataset import classify_region


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def find_rasters(root_dir: str, extensions: list = None) -> list:
    if extensions is None:
        extensions = [".tif", ".tiff", ".ecw"]
    rasters = []
    for dirpath, _, filenames in os.walk(root_dir):
        for f in filenames:
            ext = os.path.splitext(f)[1].lower()
            if ext in extensions:
                full = os.path.join(dirpath, f)
                if not any(full.endswith(s) for s in (".aux", ".pyrx", ".xml", ".aux.xml", ".ovr")):
                    rasters.append(full)
    return sorted(rasters)


def check_gdal():
    """Verify GDAL tools are available."""
    for tool in ["gdal_translate", "gdalinfo"]:
        if shutil.which(tool) is None:
            print(f"ERROR: {tool} not found. Install GDAL: sudo apt install gdal-bin")
            sys.exit(1)
    # Check for rio-cogeo as well
    try:
        import rasterio
        print(f"  rasterio version: {rasterio.__version__}")
    except ImportError:
        print("WARNING: rasterio not installed.")


def get_relative_output_path(input_path: str, root_dir: str, cog_dir: str) -> str:
    """Compute the output COG path preserving folder structure."""
    rel = os.path.relpath(input_path, root_dir)
    # Change extension to .tif for ECW inputs
    stem, ext = os.path.splitext(rel)
    out_rel = stem + ".tif"
    return os.path.join(cog_dir, out_rel)


def get_utm_epsg(lon: float, lat: float) -> int:
    """Return EPSG code for the UTM zone containing (lon, lat)."""
    zone = int((lon + 180) / 6) + 1
    return 32600 + zone if lat >= 0 else 32700 + zone


def check_raster_readable(path: str) -> tuple:
    """Return (ok, reason). Catches corrupt TIFFs and missing drivers."""
    gdal.UseExceptions()
    try:
        ds = gdal.Open(path)
        if ds is None:
            return False, "GDAL returned None"
        # Try reading a small block to catch bad offsets
        band = ds.GetRasterBand(1)
        band.ReadRaster(0, 0, min(256, ds.RasterXSize), min(256, ds.RasterYSize))
        ds = None
        return True, ""
    except Exception as e:
        return False, str(e)
    finally:
        gdal.DontUseExceptions()


def get_crs_epsg(path: str):
    """Return EPSG authority code string or None."""
    ds = gdal.Open(path)
    if ds is None:
        return None
    srs = osr.SpatialReference()
    srs.ImportFromWkt(ds.GetProjection())
    epsg = srs.GetAttrValue("AUTHORITY", 1)
    ds = None
    return epsg


def get_raster_centre(path: str) -> tuple:
    """Return (lon, lat) of raster centre in WGS84."""
    ds = gdal.Open(path)
    gt = ds.GetGeoTransform()
    cx = gt[0] + ds.RasterXSize * gt[1] / 2
    cy = gt[3] + ds.RasterYSize * gt[5] / 2
    srs_src = osr.SpatialReference()
    srs_src.ImportFromWkt(ds.GetProjection())
    epsg = srs_src.GetAttrValue("AUTHORITY", 1)
    ds = None
    # For EPSG:4326 the GeoTransform is already lon/lat — no transform needed
    # (an identity transform with mixed AxisMappingStrategy swaps x/y)
    if epsg == "4326":
        return cx, cy
    # For projected CRS, transform centre to WGS84 lon/lat
    srs_src.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    srs_wgs = osr.SpatialReference()
    srs_wgs.ImportFromEPSG(4326)
    srs_wgs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    ct = osr.CoordinateTransformation(srs_src, srs_wgs)
    lon, lat, _ = ct.TransformPoint(cx, cy)
    return lon, lat


def convert_to_cog(
    input_path: str,
    output_path: str,
    compression: str = "DEFLATE",
    blocksize: int = 512,
    overview_resampling: str = "nearest",
) -> bool:
    """Convert a raster file to Cloud Optimized GeoTIFF.

    Handles:
    - ECW → skipped gracefully if GDAL lacks ECW driver
    - Corrupt TIFFs → skipped
    - EPSG:4326 rasters → auto-reprojected to correct UTM zone first
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if os.path.exists(output_path):
        print(f"  SKIP (exists): {os.path.basename(output_path)}")
        return True

    input_ext = os.path.splitext(input_path)[1].lower()

    # --- Sanity checks ---
    if input_ext == ".ecw":
        # Quick check: try opening to see if ECW driver is available
        ok, reason = check_raster_readable(input_path)
        if not ok:
            print(f"  SKIP (ECW, no driver): {os.path.basename(input_path)}")
            return False

    # For non-ECW, check for corrupt files before spending time
    if input_ext in (".tif", ".tiff"):
        ok, reason = check_raster_readable(input_path)
        if not ok:
            print(f"  SKIP (corrupt TIF): {os.path.basename(input_path)}")
            print(f"    Reason: {reason[:120]}")
            return False

    temp_paths = []  # track temp files to clean up
    source_path = input_path

    # --- ECW → temp GeoTIFF ---
    if input_ext == ".ecw":
        temp_ecw = output_path.replace(".tif", "_from_ecw.tif")
        cmd_ecw = ["gdal_translate", "-of", "GTiff", input_path, temp_ecw]
        print(f"  Converting ECW → GeoTIFF: {os.path.basename(input_path)}")
        result = subprocess.run(cmd_ecw, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"    ERROR (ECW→TIF): {result.stderr[:200]}")
            return False
        temp_paths.append(temp_ecw)
        source_path = temp_ecw

    # --- Reproject EPSG:4326 → UTM ---
    crs = get_crs_epsg(source_path)
    if crs == "4326":
        lon, lat = get_raster_centre(source_path)
        target_epsg = get_utm_epsg(lon, lat)
        print(f"  Reprojecting EPSG:4326 → EPSG:{target_epsg}: {os.path.basename(input_path)}")
        temp_utm = output_path.replace(".tif", "_utm.tif")
        cmd_warp = [
            "gdalwarp",
            "-t_srs", f"EPSG:{target_epsg}",
            "-r", "bilinear",
            "-co", "COMPRESS=DEFLATE",
            "-co", "TILED=YES",
            "-co", "BIGTIFF=IF_SAFER",
            source_path,
            temp_utm,
        ]
        result = subprocess.run(cmd_warp, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"    ERROR (warp): {result.stderr[:200]}")
            for tp in temp_paths:
                if os.path.exists(tp): os.remove(tp)
            return False
        temp_paths.append(temp_utm)
        source_path = temp_utm
        print(f"    ✓ Reprojected to EPSG:{target_epsg}")

    # --- Convert to COG ---
    cmd = [
        "gdal_translate",
        "-of", "COG",
        "-co", f"COMPRESS={compression}",
        "-co", f"BLOCKSIZE={blocksize}",
        "-co", f"OVERVIEW_RESAMPLING={overview_resampling}",
        "-co", "BIGTIFF=IF_SAFER",
        source_path,
        output_path,
    ]

    print(f"  Converting to COG: {os.path.basename(input_path)}...")
    result = subprocess.run(cmd, capture_output=True, text=True)

    # Clean up temp files
    for tp in temp_paths:
        if os.path.exists(tp):
            os.remove(tp)

    # --- legacy (kept for compat) ---
    temp_path = None  # already handled above
    if temp_path and os.path.exists(temp_path):
        os.remove(temp_path)

    if result.returncode != 0:
        print(f"    ERROR: {result.stderr[:300]}")
        return False

    # Validate
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"    ✓ Created COG: {os.path.basename(output_path)} ({size_mb:.1f} MB)")
    return True


def validate_cog(filepath: str) -> bool:
    """Validate a file is a valid COG."""
    try:
        # Use rio cogeo validate if available
        result = subprocess.run(
            ["rio", "cogeo", "validate", filepath],
            capture_output=True, text=True
        )
        if "is a valid cloud optimized GeoTIFF" in result.stdout:
            return True
        elif result.returncode == 0:
            return True
        else:
            # Fallback: check with gdalinfo
            result2 = subprocess.run(
                ["gdalinfo", "-json", filepath],
                capture_output=True, text=True
            )
            return result2.returncode == 0
    except Exception:
        return False


def main():
    parser = argparse.ArgumentParser(description="Convert rasters to COG")
    parser.add_argument("--config", default=str(__import__("pathlib").Path(__file__).resolve().parent.parent / "configs" / "data_prep.yaml"))
    parser.add_argument("--input", default=None,
                        help="Convert only this raster file (path); skips the train/test walk.")
    parser.add_argument("--dataset-name", default=None,
                        help="Convert only the raster whose basename stem matches this name.")
    args = parser.parse_args()

    config_path = args.config
    if not os.path.isabs(config_path):
        config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), config_path)
    cfg = load_config(config_path)

    paths = cfg["paths"]
    cog_cfg = cfg.get("cog", {})
    compression = cog_cfg.get("compression", "DEFLATE")
    blocksize = cog_cfg.get("blocksize", 512)
    overview_resampling = cog_cfg.get("overview_resampling", "nearest")
    extensions = cfg.get("raster_extensions", [".tif", ".tiff", ".ecw"])

    print("=" * 70)
    print("  STEP 2: Convert Rasters to Cloud Optimized GeoTIFF (COG)")
    print("=" * 70)

    check_gdal()

    # Layout (mirrors prepare_dataset.py's CG/PB region convention):
    #   Real COGs:  <cog_dir>/<split>/<region>/<basename>.tif
    #   Symlinks:   <dataset_dir>/<split>/cog/<region>/<basename>.tif → real COG
    # Region is CG | PB | UNKNOWN, determined by classify_region().
    cog_dir = paths["cog_dir"]
    dataset_dir = paths["dataset_dir"]
    splits = {
        "train": paths["train_raw_dir"],
        "test":  paths["test_raw_dir"],
    }

    success = 0
    failed = 0
    skipped = 0
    linked = 0
    name_collisions = []
    per_region_counts = {"CG": 0, "PB": 0, "UNKNOWN": 0}

    # Optional per-dataset scoping: if --input or --dataset-name is given,
    # we keep only the matching raster(s). Saves a *lot* of time when the
    # web portal asks us to (re)process one file.
    abs_input = os.path.abspath(args.input) if args.input else None
    only_stem = args.dataset_name.strip() if args.dataset_name else None
    if abs_input and not os.path.exists(abs_input):
        print(f"  ERROR: --input not found: {abs_input}")
        sys.exit(1)
    if abs_input or only_stem:
        print(f"\n  Scoping COG conversion: "
              f"input={abs_input or '-'}, dataset-name={only_stem or '-'}")

    # ── Fast path: --input was given and points at a file NOT under the
    #   normal train/test dirs (e.g. a portal-workspace symlink). Convert
    #   that one file directly instead of trying to filter the dir walk.
    train_dir = os.path.abspath(splits["train"]) if splits["train"] else None
    test_dir  = os.path.abspath(splits["test"])  if splits["test"]  else None
    def _under(parent: str | None, child: str) -> bool:
        return bool(parent) and child.startswith(parent.rstrip(os.sep) + os.sep)

    if abs_input and not (_under(train_dir, abs_input) or _under(test_dir, abs_input)):
        region = classify_region(abs_input)
        per_region_counts[region] = per_region_counts.get(region, 0) + 1
        split_name = "test"   # uploads / out-of-tree inputs default to "test"
        cog_out_dir = os.path.join(cog_dir, split_name, region)
        link_dir    = os.path.join(dataset_dir, split_name, "cog", region)
        os.makedirs(cog_out_dir, exist_ok=True)
        os.makedirs(link_dir, exist_ok=True)

        # Output filename: prefer --dataset-name so downstream stages see
        # the dataset-aware name; otherwise fall back to the input stem.
        stem = only_stem or os.path.splitext(os.path.basename(abs_input))[0]
        out_path  = os.path.join(cog_out_dir, stem + ".tif")
        link_path = os.path.join(link_dir, stem + ".tif")

        print(f"\n  Direct conversion (out-of-tree input):")
        print(f"    src  → {abs_input}")
        print(f"    cog  → {out_path}")
        print(f"    link → {link_path}")

        if os.path.exists(out_path):
            skipped += 1
            print(f"  SKIP (exists): {region}/{os.path.basename(out_path)}")
        else:
            ok = convert_to_cog(
                abs_input, out_path,
                compression, blocksize, overview_resampling,
            )
            if ok:
                success += 1
            else:
                failed += 1

        if os.path.exists(out_path):
            try:
                if os.path.islink(link_path) or os.path.exists(link_path):
                    os.unlink(link_path)
                os.symlink(os.path.abspath(out_path), link_path)
                linked += 1
            except OSError as e:
                print(f"    WARN: could not symlink {link_path}: {e}")

        # Skip the train/test walk entirely — we handled it directly.
        splits = {}

    def _keep(rpath: str) -> bool:
        if abs_input and os.path.abspath(rpath) != abs_input:
            return False
        if only_stem and os.path.splitext(os.path.basename(rpath))[0] != only_stem:
            return False
        return True

    for split_name, data_dir in splits.items():
        if not os.path.isdir(data_dir):
            print(f"\n  WARNING: {split_name} dir not found: {data_dir}")
            continue

        rasters = [r for r in find_rasters(data_dir, extensions) if _keep(r)]
        print(f"\n  {len(rasters)} raster(s) selected in {split_name}/ ({data_dir})")
        if not rasters:
            continue
        print(f"  → COGs    : {cog_dir}/{split_name}/<region>/")
        print(f"  → symlinks: {dataset_dir}/{split_name}/cog/<region>/")

        seen_basenames = {}
        for rpath in rasters:
            region = classify_region(rpath)
            per_region_counts[region] = per_region_counts.get(region, 0) + 1

            cog_out_dir = os.path.join(cog_dir, split_name, region)
            link_dir = os.path.join(dataset_dir, split_name, "cog", region)
            os.makedirs(link_dir, exist_ok=True)

            stem = os.path.splitext(os.path.basename(rpath))[0]
            out_path = os.path.join(cog_out_dir, stem + ".tif")
            link_path = os.path.join(link_dir, stem + ".tif")

            collision_key = (region, stem)
            if collision_key in seen_basenames and seen_basenames[collision_key] != rpath:
                name_collisions.append((region, stem, seen_basenames[collision_key], rpath))
            seen_basenames[collision_key] = rpath

            if os.path.exists(out_path):
                skipped += 1
                print(f"  SKIP (exists): {region}/{os.path.basename(out_path)}")
            else:
                ok = convert_to_cog(rpath, out_path, compression, blocksize, overview_resampling)
                if ok:
                    success += 1
                else:
                    failed += 1
                    continue  # don't link a file that wasn't produced

            # Create / refresh the symlink in dataset/<split>/cog/<region>/
            try:
                if os.path.islink(link_path) or os.path.exists(link_path):
                    os.unlink(link_path)
                os.symlink(os.path.abspath(out_path), link_path)
                linked += 1
            except OSError as e:
                print(f"    WARN: could not symlink {link_path}: {e}")

    if name_collisions:
        print(f"\n  WARNING: {len(name_collisions)} basename collision(s) within the same region — last write wins:")
        for region, stem, a, b in name_collisions[:10]:
            print(f"    [{region}] {stem}.tif  ←  {a}   AND   {b}")

    print(f"\n{'='*70}")
    print(f"  COG CONVERSION COMPLETE")
    print(f"  Success: {success} | Failed: {failed} | Skipped: {skipped} | Symlinked: {linked}")
    print(f"  Per-region tally: CG={per_region_counts['CG']}, "
          f"PB={per_region_counts['PB']}, UNKNOWN={per_region_counts['UNKNOWN']}")
    print(f"  COG store:   {cog_dir}/{{train,test}}/{{CG,PB}}/")
    print(f"  Symlinks at: {dataset_dir}/{{train,test}}/cog/{{CG,PB}}/")
    print(f"{'='*70}\n")

    # Validate all COGs across split / region sub-dirs
    print("  Validating COGs...")
    cog_files = []
    for split_name in splits:
        for region in ("CG", "PB", "UNKNOWN"):
            d = os.path.join(cog_dir, split_name, region)
            if os.path.isdir(d):
                cog_files.extend(find_rasters(d, [".tif", ".tiff"]))
    valid = 0
    for cf in cog_files:
        if validate_cog(cf):
            valid += 1
        else:
            print(f"    WARN: Not a valid COG: {cf}")
    print(f"  Validated {valid}/{len(cog_files)} COG files.\n")


if __name__ == "__main__":
    main()
