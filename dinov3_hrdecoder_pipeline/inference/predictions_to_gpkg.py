#!/usr/bin/env python3
"""
Convert Predictions to GeoPackage (Multi-Geometry)
=====================================================
Converts multi-class prediction GeoTIFF tiles into a GeoPackage (.gpkg)
with separate layers per class AND geometry type, matching the training
data format exactly.

Output layers match the training shapefile names:
  - Built_Up_Area_type  (Polygon)
  - Road                (Polygon)  + Road_Centre_Line (Line)
  - Water_Body          (Polygon)  + Water_Body_Line  (Line) + Waterbody_Point (Point)
  - Utility_Poly        (Polygon)  + Utility           (Point)
  - Bridge              (Polygon)
  - Railway             (Line)

Two modes:
  1) --pred-dir : Reads tile predictions directly (RECOMMENDED)
  2) --input    : Reads a stitched full-image GeoTIFF

Usage:
    # From tile predictions (recommended):
    python -m dinov3_hrdecoder_pipeline.inference.predictions_to_gpkg \\
        --pred-dir outputs/predictions/<DATASET> \\
        --output predictions.gpkg

    # From stitched raster:
    python -m dinov3_hrdecoder_pipeline.inference.predictions_to_gpkg \\
        --input outputs/stitched/<DATASET>_pred.tif
"""

import os
import sys
import csv
import argparse
import glob
from pathlib import Path

import numpy as np
import yaml
from concurrent.futures import ProcessPoolExecutor, as_completed

try:
    import rasterio
    import rasterio.transform
    from rasterio.features import shapes as rio_shapes
    import geopandas as gpd
    from shapely.geometry import shape, LineString, MultiLineString, Point
    from shapely.ops import unary_union, linemerge
    import fiona
    import pandas as pd
except ImportError as e:
    print(f"ERROR: Missing dependency: {e}")
    print("  pip install rasterio geopandas shapely fiona pandas")
    sys.exit(1)

try:
    from skimage.morphology import skeletonize
except ImportError:
    print("ERROR: scikit-image required for centerline extraction")
    print("  pip install scikit-image")
    sys.exit(1)

# pyogrio is GeoPandas' fast OGR backend. Falls back to fiona silently
# if the wheel isn't present — slow path but still correct.
try:
    import pyogrio  # noqa: F401
    _HAVE_PYOGRIO = True
except Exception:
    _HAVE_PYOGRIO = False


def _write_gpkg_layer(gdf: "gpd.GeoDataFrame", path: str, layer: str) -> None:
    """Write a GeoDataFrame to a GPKG layer using pyogrio when available.

    pyogrio writes ~5-10× faster than fiona for typical layers and is a
    drop-in replacement.
    """
    try:
        if _HAVE_PYOGRIO:
            gdf.to_file(path, layer=layer, driver="GPKG", engine="pyogrio")
            return
    except (TypeError, ValueError, Exception):
        # Old GeoPandas without engine kwarg, or a transient OGR error.
        pass
    gdf.to_file(path, layer=layer, driver="GPKG")

PIPE_ROOT = Path(__file__).resolve().parent.parent
CFG_PATH = PIPE_ROOT / "configs" / "train.yaml"

# ── Layer colours for QGIS styling ──────────────────────────────────────────
LAYER_COLOURS = {
    "Built_Up_Area_type": "#FF0000",
    "Road":               "#00FF00",
    "Road_Centre_Line":   "#228B22",
    "Water_Body":         "#0000FF",
    "Water_Body_Line":    "#4169E1",
    "Waterbody_Point":    "#00BFFF",
    "Utility_Poly":       "#FFD700",
    "Utility":            "#FFA500",
    "Bridge":             "#FF00FF",
    "Railway":            "#00FFFF",
}

# ── Geometry configuration per class ────────────────────────────────────────
# Maps class_id → list of (layer_name, geometry_type)
# geometry_type: "polygon", "line" (skeleton), "point" (centroid of small features)
GEOMETRY_MAP = {
    1: [  # Built_Up_Area
        ("Built_Up_Area_type", "polygon"),
    ],
    2: [  # Road
        ("Road", "polygon"),
        ("Road_Centre_Line", "line"),
    ],
    3: [  # Water_Body
        ("Water_Body", "polygon"),
        ("Water_Body_Line", "line"),
        ("Waterbody_Point", "point"),
    ],
    4: [  # Utility
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


# ═══════════════════════════════════════════════════════════════════════════
#  Helper functions
# ═══════════════════════════════════════════════════════════════════════════

def load_config(config_path: str = None) -> dict:
    path = config_path or str(CFG_PATH)
    with open(path) as f:
        return yaml.safe_load(f)


def get_class_info(cfg: dict) -> list:
    """Return list of (class_id, class_name) sorted by id."""
    return [(c["id"], c["name"]) for c in sorted(cfg["classes"], key=lambda x: x["id"])]


def polygonize_class(raster_data: np.ndarray, transform, crs,
                     class_id: int) -> gpd.GeoDataFrame:
    """Polygonize a single class from a raster."""
    mask = (raster_data == class_id).astype(np.uint8)
    if mask.sum() == 0:
        return gpd.GeoDataFrame(columns=["geometry"], crs=crs)

    geometries = []
    for geom_dict, value in rio_shapes(mask, mask=mask, transform=transform):
        if value == 1:
            geom = shape(geom_dict)
            if not geom.is_empty:
                geometries.append(geom)

    if not geometries:
        return gpd.GeoDataFrame(columns=["geometry"], crs=crs)
    return gpd.GeoDataFrame(geometry=geometries, crs=crs)


def extract_centerlines(raster_data: np.ndarray, transform, crs,
                        class_id: int) -> gpd.GeoDataFrame:
    """
    Skeletonize a class mask and vectorize into LineString geometries.

    Uses scikit-image skeletonize to extract 1-pixel-wide skeletons,
    then traces connected skeleton pixels into LineString geometries.
    """
    mask = (raster_data == class_id).astype(bool)
    if mask.sum() == 0:
        return gpd.GeoDataFrame(columns=["geometry"], crs=crs)

    # Skeletonize the binary mask
    skeleton = skeletonize(mask).astype(np.uint8)

    if skeleton.sum() == 0:
        return gpd.GeoDataFrame(columns=["geometry"], crs=crs)

    # Vectorize skeleton pixels → thin polygons, then extract boundaries as lines
    lines = []
    for geom_dict, value in rio_shapes(skeleton, mask=skeleton, transform=transform):
        if value == 1:
            poly = shape(geom_dict)
            if poly.is_empty:
                continue
            # The skeleton polygons are very thin (1px wide).
            # Extract their boundary or centerline.
            boundary = poly.boundary
            if boundary.is_empty:
                continue
            if boundary.geom_type == "MultiLineString":
                lines.extend(boundary.geoms)
            elif boundary.geom_type == "LineString":
                lines.append(boundary)

    if not lines:
        return gpd.GeoDataFrame(columns=["geometry"], crs=crs)

    # Merge connected line segments
    try:
        merged = linemerge(lines)
        if merged.geom_type == "MultiLineString":
            final_lines = list(merged.geoms)
        elif merged.geom_type == "LineString":
            final_lines = [merged]
        else:
            final_lines = lines
    except Exception:
        final_lines = lines

    # Filter out very short lines (noise)
    final_lines = [l for l in final_lines if not l.is_empty and l.length > 0]

    if not final_lines:
        return gpd.GeoDataFrame(columns=["geometry"], crs=crs)

    return gpd.GeoDataFrame(geometry=final_lines, crs=crs)


def extract_points(polygons_gdf: gpd.GeoDataFrame,
                   area_threshold: float) -> tuple:
    """
    Split polygons into small (→ points) and large (→ keep as polygons).

    Args:
        polygons_gdf: GeoDataFrame of polygons
        area_threshold: area below which polygons become points (CRS units²)

    Returns:
        (points_gdf, remaining_polygons_gdf)
    """
    if len(polygons_gdf) == 0:
        empty = gpd.GeoDataFrame(columns=["geometry"], crs=polygons_gdf.crs)
        return empty, polygons_gdf

    areas = polygons_gdf.geometry.area
    small_mask = areas < area_threshold
    large_mask = ~small_mask

    # Small polygons → centroid points
    small = polygons_gdf[small_mask].copy()
    if len(small) > 0:
        small["geometry"] = small.geometry.centroid
    points_gdf = small

    # Large polygons → keep as polygons
    remaining = polygons_gdf[large_mask].copy()

    return points_gdf, remaining


def _crop_to_class_bbox(raster_data: np.ndarray, transform,
                        class_id: int, pad: int = 2):
    """
    Crop the raster to the tight bounding box of a given class.

    Returns:
        (cropped_data, cropped_transform, (row_min, col_min))
        or (None, None, None) if class not present.
    """
    rows, cols = np.where(raster_data == class_id)
    if len(rows) == 0:
        return None, None, None

    r_min = max(rows.min() - pad, 0)
    r_max = min(rows.max() + pad + 1, raster_data.shape[0])
    c_min = max(cols.min() - pad, 0)
    c_max = min(cols.max() + pad + 1, raster_data.shape[1])

    cropped = raster_data[r_min:r_max, c_min:c_max]

    # Adjust affine transform for the cropped window
    cropped_transform = rasterio.transform.Affine(
        transform.a, transform.b, transform.c + c_min * transform.a,
        transform.d, transform.e, transform.f + r_min * transform.e,
    )

    return cropped, cropped_transform, (r_min, c_min)


def write_qgis_layer_style(gpkg_path: str, layer_name: str, colour_hex: str,
                           geom_type: str = "polygon"):
    """Write QGIS-compatible layer style into the GeoPackage."""
    try:
        import sqlite3
        conn = sqlite3.connect(gpkg_path)
        cur = conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS layer_styles (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                f_table_catalog TEXT DEFAULT '',
                f_table_schema TEXT DEFAULT '',
                f_table_name TEXT NOT NULL,
                f_geometry_column TEXT DEFAULT 'geometry',
                styleName TEXT NOT NULL,
                styleQML TEXT,
                styleSLD TEXT,
                useAsDefault INTEGER DEFAULT 1,
                description TEXT DEFAULT '',
                owner TEXT DEFAULT '',
                ui TEXT,
                update_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        if geom_type == "point":
            qml = f"""<!DOCTYPE qgis PUBLIC 'http://mrcc.com/qgis.dtd' 'SYSTEM'>
<qgis version="3.0">
  <renderer-v2 type="singleSymbol">
    <symbols>
      <symbol name="0" type="marker">
        <layer class="SimpleMarker">
          <prop k="color" v="{colour_hex}"/>
          <prop k="size" v="3"/>
          <prop k="outline_color" v="#000000"/>
          <prop k="outline_width" v="0.4"/>
          <prop k="name" v="circle"/>
        </layer>
      </symbol>
    </symbols>
  </renderer-v2>
</qgis>"""
        elif geom_type == "line":
            qml = f"""<!DOCTYPE qgis PUBLIC 'http://mrcc.com/qgis.dtd' 'SYSTEM'>
<qgis version="3.0">
  <renderer-v2 type="singleSymbol">
    <symbols>
      <symbol name="0" type="line">
        <layer class="SimpleLine">
          <prop k="line_color" v="{colour_hex}"/>
          <prop k="line_width" v="0.5"/>
          <prop k="line_style" v="solid"/>
        </layer>
      </symbol>
    </symbols>
  </renderer-v2>
</qgis>"""
        else:  # polygon
            qml = f"""<!DOCTYPE qgis PUBLIC 'http://mrcc.com/qgis.dtd' 'SYSTEM'>
<qgis version="3.0">
  <renderer-v2 type="singleSymbol">
    <symbols>
      <symbol name="0" type="fill">
        <layer class="SimpleFill">
          <prop k="color" v="{colour_hex}"/>
          <prop k="outline_color" v="#000000"/>
          <prop k="outline_width" v="0.26"/>
          <prop k="style" v="solid"/>
        </layer>
      </symbol>
    </symbols>
  </renderer-v2>
</qgis>"""

        cur.execute(
            "INSERT INTO layer_styles (f_table_name, styleName, styleQML, useAsDefault) "
            "VALUES (?, ?, ?, 1)",
            (layer_name, f"{layer_name}_style", qml),
        )
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"  WARN: Could not write QGIS style for {layer_name}: {e}")


# ═══════════════════════════════════════════════════════════════════════════
#  Core: process one raster (tile or stitched) → per-class multi-geometry
# ═══════════════════════════════════════════════════════════════════════════

def _process_single_class(class_id, class_name, raster_data, transform, crs,
                          point_area_threshold, verbose, crop_bbox,
                          geometry_map=None, simplify_tol=0.0, min_area=0.0):
    """
    Process one class from the raster → per-geometry GeoDataFrames.
    If crop_bbox is True, crops to the class bounding box first.

    `geometry_map` lets the caller pass a filtered map (e.g.
    `GEOMETRY_MAP_FILTERED` from batch_stitched_to_gpkg.py) so that
    parallel workers — which on `spawn` start with a fresh module-level
    `GEOMETRY_MAP` — produce only the desired layers.

    `simplify_tol` and `min_area` are applied INSIDE this worker so the
    pickle round-trip back to the parent carries far fewer vertices.

    Returns:
        list of (layer_name, gdf, geom_type)
    """
    import time
    gmap = geometry_map if geometry_map is not None else GEOMETRY_MAP
    geom_specs = gmap.get(class_id, [(class_name, "polygon")])

    # ── Crop to class bounding box to avoid processing background ──
    if crop_bbox:
        cropped, crop_transform, offsets = _crop_to_class_bbox(
            raster_data, transform, class_id, pad=2)
        if cropped is None:
            return []
        work_data = cropped
        work_transform = crop_transform
        crop_shape = cropped.shape
        n_pixels = int((cropped == class_id).sum())
        if verbose:
            print(f"\n  ── {class_name} (class {class_id}) ──")
            print(f"     {n_pixels:,} pixels detected")
            print(f"     Cropped: {raster_data.shape} → {crop_shape} "
                  f"({100 * crop_shape[0] * crop_shape[1] / (raster_data.shape[0] * raster_data.shape[1]):.1f}% of area)")
    else:
        work_data = raster_data
        work_transform = transform
        n_pixels = int((raster_data == class_id).sum())
        if n_pixels == 0:
            return []
        if verbose:
            print(f"\n  ── {class_name} (class {class_id}) ──")
            print(f"     {n_pixels:,} pixels detected")

    results = []
    base_polys = None

    for layer_name, geom_type in geom_specs:
        if geom_type == "polygon":
            if base_polys is None:
                if verbose:
                    print(f"     Polygonizing...", end=" ", flush=True)
                t0 = time.time()
                base_polys = polygonize_class(work_data, work_transform, crs, class_id)
                if verbose:
                    print(f"{len(base_polys):,} polygons ({time.time()-t0:.1f}s)")
            gdf = base_polys.copy()
            if len(gdf) > 0:
                results.append((layer_name, gdf, "polygon"))

        elif geom_type == "line":
            if verbose:
                print(f"     Skeletonizing for centerlines...", end=" ", flush=True)
            t0 = time.time()
            gdf = extract_centerlines(work_data, work_transform, crs, class_id)
            if verbose:
                print(f"{len(gdf):,} lines ({time.time()-t0:.1f}s)")
            if len(gdf) > 0:
                results.append((layer_name, gdf, "line"))

        elif geom_type == "point":
            if base_polys is None:
                if verbose:
                    print(f"     Polygonizing...", end=" ", flush=True)
                t0 = time.time()
                base_polys = polygonize_class(work_data, work_transform, crs, class_id)
                if verbose:
                    print(f"{len(base_polys):,} polygons ({time.time()-t0:.1f}s)")
            if verbose:
                print(f"     Splitting small features → points (threshold={point_area_threshold})...",
                      end=" ", flush=True)
            points_gdf, remaining_polys = extract_points(
                base_polys, point_area_threshold
            )
            if verbose:
                print(f"{len(points_gdf):,} points, {len(remaining_polys):,} polys remain")
            if len(points_gdf) > 0:
                results.append((layer_name, points_gdf, "point"))
            # Update the polygon layer to keep only large features
            poly_layer = [ln for ln, gt in geom_specs if gt == "polygon"]
            if poly_layer and len(remaining_polys) > 0:
                # Replace or add the polygon layer with filtered polys
                results = [(ln, g, gt) for ln, g, gt in results if ln != poly_layer[0]]
                results.append((poly_layer[0], remaining_polys, "polygon"))

    # ── Apply simplify + min_area filters INSIDE the worker. Massive ────
    # win because the pickle return then carries far fewer vertices.
    if simplify_tol > 0 or min_area > 0:
        filtered: list = []
        for layer_name, gdf, geom_type in results:
            if len(gdf) == 0:
                filtered.append((layer_name, gdf, geom_type))
                continue
            if simplify_tol > 0 and geom_type in ("polygon", "line"):
                gdf = gdf.copy()
                gdf["geometry"] = gdf.geometry.simplify(simplify_tol, preserve_topology=True)
            if min_area > 0 and geom_type == "polygon":
                gdf = gdf[gdf.geometry.area >= min_area]
            filtered.append((layer_name, gdf, geom_type))
        results = filtered

    return results


def process_raster_to_layers(raster_data: np.ndarray, transform, crs,
                             class_info: list,
                             point_area_threshold: float = 100.0,
                             verbose: bool = False,
                             crop_bbox: bool = False,
                             workers: int = 1,
                             geometry_map: dict | None = None,
                             simplify_tol: float = 0.0,
                             min_area: float = 0.0):
    """
    Process a raster into per-class, per-geometry-type GeoDataFrames.

    Args:
        crop_bbox: If True, crop to each class's bounding box before
                   processing (huge speedup for stitched rasters
                   with mostly background).
        workers:   Number of parallel workers (1 = sequential).

    Returns:
        dict of { layer_name: (gdf, geom_type) }
    """
    import time

    # Filter to classes that actually exist in the raster
    active_classes = []
    for class_id, class_name in class_info:
        if class_id == 0:
            continue
        if (raster_data == class_id).any():
            active_classes.append((class_id, class_name))
        elif verbose:
            print(f"  [{class_name}] No pixels found, skipping")

    results = {}

    if workers > 1 and len(active_classes) > 1:
        # Parallel processing per class
        if verbose:
            print(f"\n  Using {workers} parallel workers for {len(active_classes)} classes")
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {}
            for class_id, class_name in active_classes:
                fut = executor.submit(
                    _process_single_class,
                    class_id, class_name,
                    raster_data, transform, crs,
                    point_area_threshold, verbose, crop_bbox,
                    geometry_map, simplify_tol, min_area,
                )
                futures[fut] = (class_id, class_name)

            for fut in as_completed(futures):
                class_id, class_name = futures[fut]
                try:
                    class_results = fut.result()
                    for layer_name, gdf, geom_type in class_results:
                        results[layer_name] = (gdf, geom_type)
                except Exception as e:
                    print(f"  ERROR processing {class_name}: {e}")
    else:
        # Sequential processing
        for class_id, class_name in active_classes:
            class_results = _process_single_class(
                class_id, class_name,
                raster_data, transform, crs,
                point_area_threshold, verbose, crop_bbox,
                geometry_map, simplify_tol, min_area,
            )
            for layer_name, gdf, geom_type in class_results:
                results[layer_name] = (gdf, geom_type)

    return results


# ═══════════════════════════════════════════════════════════════════════════
#  MODE 1: Direct tile predictions → GeoPackage  (RECOMMENDED)
# ═══════════════════════════════════════════════════════════════════════════

def _process_one_tile(tile_path, class_info, point_area_threshold,
                      geometry_map=None):
    """Worker for the threaded tiles_to_gpkg path."""
    with rasterio.open(tile_path) as src:
        data = src.read(1)
        transform = src.transform
        crs = src.crs
    return process_raster_to_layers(
        data, transform, crs, class_info,
        point_area_threshold=point_area_threshold,
        geometry_map=geometry_map,
    )


def tiles_to_gpkg(pred_dir: str, output_path: str, class_info: list,
                  pred_suffix: str = "_pred.tif",
                  simplify_tol: float = 0.0, min_area: float = 0.0,
                  point_area_threshold: float = 100.0,
                  dissolve: bool = True,
                  workers: int = 1,
                  geometry_map: dict | None = None):
    """
    Read tile prediction GeoTIFFs, polygonize per class with multi-geometry,
    merge across tiles, and write to a GeoPackage.

    Much faster than `stitched_to_gpkg` on huge orthomosaics: each tile
    is only 1024² (~1 MB), so per-tile polygonisation is trivially
    cheap; we just parallelise it across cores. The merge step then
    dissolves polygons that cross tile boundaries.
    """
    import time
    tile_files = sorted(glob.glob(os.path.join(pred_dir, f"*{pred_suffix}")))
    if not tile_files:
        print(f"  ERROR: No tile predictions found in {pred_dir} with suffix '{pred_suffix}'")
        return 0, 0

    print(f"  Found {len(tile_files)} tile predictions (workers={workers})")

    # Verify georeferencing on first tile
    with rasterio.open(tile_files[0]) as src:
        ref_crs = src.crs
        print(f"\n  ── Georeferencing (from tile) ──")
        print(f"  CRS:   {ref_crs}")
        print(f"  EPSG:  {ref_crs.to_epsg() if ref_crs else 'NONE'}")
        if ref_crs:
            print(f"  ✅ Tiles are properly georeferenced")
        else:
            print(f"  ⚠️  Tiles lack CRS")

    if os.path.exists(output_path):
        os.remove(output_path)

    # Collect all layers across all tiles
    # { layer_name: [list of GeoDataFrames] }
    all_layers = {}
    layer_geom_types = {}

    t0 = time.time()
    done = 0
    if workers > 1:
        # Rasterio releases the GIL during file IO + numpy ops, so a
        # thread pool gets real parallelism without the pickle overhead
        # a process pool would carry for our per-tile rasters.
        from concurrent.futures import ThreadPoolExecutor, as_completed
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = {
                ex.submit(_process_one_tile, tp, class_info,
                          point_area_threshold, geometry_map): tp
                for tp in tile_files
            }
            for fut in as_completed(futures):
                tile_results = fut.result()
                for layer_name, (gdf, geom_type) in tile_results.items():
                    if len(gdf) > 0:
                        if layer_name not in all_layers:
                            all_layers[layer_name] = []
                            layer_geom_types[layer_name] = geom_type
                        all_layers[layer_name].append(gdf)
                done += 1
                if done % 200 == 0:
                    rate = done / (time.time() - t0)
                    print(f"    {done}/{len(tile_files)} tiles processed "
                          f"({rate:.0f} tiles/s)")
    else:
        for i, tile_path in enumerate(tile_files):
            tile_results = _process_one_tile(
                tile_path, class_info, point_area_threshold, geometry_map
            )
            for layer_name, (gdf, geom_type) in tile_results.items():
                if len(gdf) > 0:
                    if layer_name not in all_layers:
                        all_layers[layer_name] = []
                        layer_geom_types[layer_name] = geom_type
                    all_layers[layer_name].append(gdf)
            if (i + 1) % 200 == 0:
                print(f"    {i+1}/{len(tile_files)} tiles processed...")

    # Merge and write each layer
    total_features = 0
    layers_written = 0

    for layer_name, gdfs in all_layers.items():
        geom_type = layer_geom_types[layer_name]
        print(f"\n  Writing layer: {layer_name} ({geom_type})...")

        merged = gpd.GeoDataFrame(pd.concat(gdfs, ignore_index=True), crs=ref_crs)
        raw_count = len(merged)

        # Dissolve overlapping geometries at tile boundaries
        if dissolve and len(merged) > 1 and geom_type in ("polygon", "line"):
            try:
                dissolved = unary_union(merged.geometry)
                if dissolved.geom_type.startswith("Multi"):
                    geoms = list(dissolved.geoms)
                else:
                    geoms = [dissolved]

                # Flatten any remaining Multi-geometries
                flat = []
                for g in geoms:
                    if g.geom_type.startswith("Multi"):
                        flat.extend(g.geoms)
                    else:
                        flat.append(g)
                geoms = flat

                # Apply filters
                if simplify_tol > 0:
                    geoms = [g.simplify(simplify_tol, preserve_topology=True)
                             for g in geoms]
                if min_area > 0 and geom_type == "polygon":
                    geoms = [g for g in geoms if g.area >= min_area]
                geoms = [g for g in geoms if not g.is_empty]

                merged = gpd.GeoDataFrame(geometry=geoms, crs=ref_crs)
                print(f"    {raw_count:,} raw → {len(merged):,} after dissolve")
            except Exception as e:
                print(f"    WARN: Dissolve failed ({e}), keeping raw")

        if len(merged) == 0:
            continue

        # Add metadata columns
        merged["layer"] = layer_name
        merged["geom_type"] = geom_type
        if geom_type == "polygon":
            merged["area"] = merged.geometry.area
        elif geom_type == "line":
            merged["length"] = merged.geometry.length

        # Write layer (pyogrio when available, ~5-10× faster than fiona)
        _write_gpkg_layer(merged, output_path, layer_name)
        total_features += len(merged)
        layers_written += 1

        colour = LAYER_COLOURS.get(layer_name, "#808080")
        print(f"    → {len(merged):,} features, colour: {colour}")

    return total_features, layers_written


# ═══════════════════════════════════════════════════════════════════════════
#  MODE 2: Stitched raster → GeoPackage
# ═══════════════════════════════════════════════════════════════════════════

def stitched_to_gpkg(input_path: str, output_path: str, class_info: list,
                     cfg: dict, tile_index: str = None, dataset: str = None,
                     simplify_tol: float = 0.0, min_area: float = 0.0,
                     point_area_threshold: float = 100.0,
                     workers: int = 1,
                     downsample: int = 1,
                     geometry_map: dict | None = None):
    """Read a stitched GeoTIFF and convert to multi-geometry GeoPackage."""
    import time
    t_total = time.time()

    with rasterio.open(input_path) as src:
        crs = src.crs
        transform = src.transform
        has_crs = crs is not None
        has_transform = (transform is not None and
                         transform != rasterio.transform.Affine.identity())
        print(f"\n  ── Georeferencing ──")
        print(f"  CRS:  {crs}")
        print(f"  EPSG: {crs.to_epsg() if crs else 'NONE'}")
        if has_crs and has_transform:
            print(f"  ✅ Properly georeferenced")
        else:
            print(f"  ⚠️  Missing georeferencing")

    # ── Step 1: Read raster (optionally downsampled) ──
    # Downsampling N> 1 reads the raster at 1/N resolution using
    # nearest-neighbour resampling. The class IDs are preserved
    # exactly (no interpolation), and the affine transform is scaled
    # accordingly so resulting polygon coordinates remain in the
    # source CRS. A 2× downsample → 4× fewer pixels to polygonise.
    print(f"\n  [1/4] Reading raster from disk"
          f"{' (downsample ×{}'.format(downsample) + ')' if downsample > 1 else ''}...",
          end=" ", flush=True)
    t0 = time.time()
    with rasterio.open(input_path) as src:
        full_w, full_h = src.width, src.height
        if downsample > 1:
            from rasterio.enums import Resampling
            new_h = max(1, full_h // downsample)
            new_w = max(1, full_w // downsample)
            raster_data = src.read(
                1,
                out_shape=(new_h, new_w),
                resampling=Resampling.nearest,
            )
            # Scale the affine to match the new pixel grid.
            transform = src.transform * rasterio.Affine.scale(
                full_w / new_w, full_h / new_h
            )
        else:
            raster_data = src.read(1)
            transform = src.transform
        crs = src.crs
    print(f"done ({time.time()-t0:.1f}s) — shape: {raster_data.shape}, "
          f"{raster_data.nbytes / 1024**2:.0f} MB")

    unique_vals = np.unique(raster_data)
    print(f"      Unique class values: {unique_vals}")
    class_pixel_counts = {v: int((raster_data == v).sum()) for v in unique_vals if v > 0}
    for cid, cnt in class_pixel_counts.items():
        print(f"      Class {cid}: {cnt:,} pixels")

    # Recover georef if missing
    if not (has_crs and has_transform):
        transform, crs = _try_recover_georef(
            cfg, tile_index, dataset, input_path, raster_data.shape)

    if os.path.exists(output_path):
        os.remove(output_path)

    # ── Step 2: Process each class (with bbox cropping for speed) ──
    # The filtered geometry_map (when provided by batch_stitched_to_gpkg)
    # propagates into worker processes here so each only computes the
    # layers we actually keep. simplify_tol + min_area run inside the
    # worker so the pickled return is much smaller.
    print(f"\n  [2/4] Processing classes → geometries (bbox-cropped)...")
    t0 = time.time()
    results = process_raster_to_layers(
        raster_data, transform, crs, class_info,
        point_area_threshold=point_area_threshold,
        verbose=True,
        crop_bbox=True,
        workers=workers,
        geometry_map=geometry_map,
        simplify_tol=simplify_tol,
        min_area=min_area,
    )
    print(f"\n  Processing complete ({time.time()-t0:.1f}s) — "
          f"{len(results)} layers ready")

    # ── Step 3: Write GeoPackage ──
    print(f"\n  [3/4] Writing layers to GeoPackage...")
    total_features = 0
    layers_written = 0
    n_layers = len(results)

    for idx, (layer_name, (gdf, geom_type)) in enumerate(results.items(), 1):
        if len(gdf) == 0:
            continue

        print(f"  [{idx}/{n_layers}] {layer_name} ({geom_type}, {len(gdf):,} features)...",
              end=" ", flush=True)
        t0 = time.time()

        # simplify_tol / min_area were already applied inside the worker
        # (see _process_single_class); only safety net is needed for the
        # rare sequential path that bypassed the args (shouldn't happen).

        # Add metadata
        gdf["layer"] = layer_name
        gdf["geom_type"] = geom_type
        if geom_type == "polygon":
            gdf["area"] = gdf.geometry.area
        elif geom_type == "line":
            gdf["length"] = gdf.geometry.length

        _write_gpkg_layer(gdf, output_path, layer_name)
        total_features += len(gdf)
        layers_written += 1
        print(f"written ({time.time()-t0:.1f}s)")

    # ── Step 4: Done ──
    print(f"\n  [4/4] Finished in {time.time()-t_total:.1f}s total")
    return total_features, layers_written


def _try_recover_georef(cfg, tile_index, dataset, input_path, raster_shape):
    """Try to recover CRS/transform from tiles."""
    tile_index_path = tile_index
    dataset_name = dataset

    if tile_index_path is None:
        candidate = cfg["paths"].get("tile_index", "")
        if candidate and os.path.exists(candidate):
            tile_index_path = candidate

    if dataset_name is None:
        stem = Path(input_path).stem
        for suffix in ["_pred", "_refined"]:
            if stem.endswith(suffix):
                dataset_name = stem[:-len(suffix)]
                break
        else:
            dataset_name = stem

    if not tile_index_path or not os.path.exists(tile_index_path):
        print(f"  Could not find tile_index.csv. Use --tile-index to specify.")
        return None, None

    tiles = []
    with open(tile_index_path) as f:
        for row in csv.DictReader(f):
            if row["dataset"] == dataset_name:
                tiles.append(row)

    for t in tiles:
        tp = t["tile_path"]
        if os.path.exists(tp):
            try:
                with rasterio.open(tp) as src:
                    tt, tc = src.transform, src.crs
                if tc and tt:
                    co, ro = int(t["col_off"]), int(t["row_off"])
                    ox = tt.c - co * tt.a
                    oy = tt.f - ro * tt.e
                    full_t = rasterio.transform.Affine(tt.a, 0, ox, 0, tt.e, oy)
                    print(f"  ✅ Recovered georef from tile: {tp}")
                    return full_t, tc
            except Exception:
                continue
    return None, None


# ═══════════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Convert predictions to multi-geometry GeoPackage (.gpkg)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Direct from tile predictions (recommended):
  python predictions_to_gpkg.py --pred-dir outputs/predictions/<DATASET>

  # From a stitched raster:
  python predictions_to_gpkg.py --input outputs/stitched/<DATASET>_pred.tif
""")

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--pred-dir",
                      help="Directory containing tile prediction GeoTIFFs (RECOMMENDED)")
    mode.add_argument("--input",
                      help="Path to a stitched prediction GeoTIFF")

    parser.add_argument("--output", default=None, help="Output .gpkg path")
    parser.add_argument("--config", default=None, help="Path to train.yaml")
    parser.add_argument("--simplify", type=float, default=0.0,
                        help="Geometry simplification tolerance (CRS units, default: 0)")
    parser.add_argument("--min-area", type=float, default=0.0,
                        help="Min polygon area to keep (CRS units², default: 0)")
    parser.add_argument("--point-area-threshold", type=float, default=100.0,
                        help="Polygon area below which → point (CRS units², default: 100)")
    parser.add_argument("--pred-suffix", default="_pred.tif",
                        help="Tile prediction filename suffix (default: _pred.tif)")
    parser.add_argument("--no-dissolve", action="store_true",
                        help="Skip dissolving at tile boundaries")
    parser.add_argument("--workers", type=int, default=1,
                        help="Parallel workers for per-class processing (default: 1)")
    parser.add_argument("--downsample", type=int, default=1,
                        help="Read stitched raster at 1/N resolution before "
                             "polygonising (default: 1 = full res). Higher "
                             "values trade pixel-level precision for speed "
                             "(N=2 → 4× faster, N=4 → 16× faster).")
    parser.add_argument("--tile-index", default=None,
                        help="tile_index.csv (for georef recovery in --input mode)")
    parser.add_argument("--dataset", default=None,
                        help="Dataset name in tile_index.csv")

    args = parser.parse_args()
    cfg = load_config(args.config)
    class_info = get_class_info(cfg)

    # Determine output path
    if args.output:
        output_path = os.path.abspath(args.output)
    elif args.pred_dir:
        output_path = os.path.abspath(
            os.path.join(args.pred_dir, "..", Path(args.pred_dir).name + ".gpkg"))
    else:
        output_path = os.path.splitext(os.path.abspath(args.input))[0] + ".gpkg"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # ── MODE 1: Tile predictions ──
    if args.pred_dir:
        pred_dir = os.path.abspath(args.pred_dir)
        if not os.path.isdir(pred_dir):
            print(f"ERROR: Directory not found: {pred_dir}")
            sys.exit(1)

        print("=" * 70)
        print("  Tile Predictions → Multi-Geometry GeoPackage")
        print(f"  Pred dir: {pred_dir}")
        print(f"  Output:   {output_path}")
        print(f"  Suffix:   {args.pred_suffix}")
        print(f"  Point threshold: {args.point_area_threshold} CRS units²")
        print("=" * 70)

        total, written = tiles_to_gpkg(
            pred_dir, output_path, class_info,
            pred_suffix=args.pred_suffix,
            simplify_tol=args.simplify,
            min_area=args.min_area,
            point_area_threshold=args.point_area_threshold,
            dissolve=not args.no_dissolve,
        )

    # ── MODE 2: Stitched raster ──
    else:
        input_path = os.path.abspath(args.input)
        if not os.path.exists(input_path):
            print(f"ERROR: File not found: {input_path}")
            sys.exit(1)

        print("=" * 70)
        print("  Stitched Raster → Multi-Geometry GeoPackage")
        print(f"  Input:  {input_path}")
        print(f"  Output: {output_path}")
        print(f"  Point threshold: {args.point_area_threshold} CRS units²")
        print("=" * 70)

        total, written = stitched_to_gpkg(
            input_path, output_path, class_info, cfg,
            tile_index=args.tile_index, dataset=args.dataset,
            simplify_tol=args.simplify, min_area=args.min_area,
            point_area_threshold=args.point_area_threshold,
            workers=args.workers,
            downsample=args.downsample,
        )

    # ── Write QGIS styles per layer ──
    if written > 0:
        print(f"\n  Writing QGIS layer styles...")
        if os.path.exists(output_path):
            for layer_name in fiona.listlayers(output_path):
                if layer_name == "layer_styles":
                    continue
                colour = LAYER_COLOURS.get(layer_name, "#808080")
                # Determine geom type from layer name
                if layer_name in ("Utility", "Waterbody_Point"):
                    gt = "point"
                elif layer_name in ("Road_Centre_Line", "Water_Body_Line", "Railway"):
                    gt = "line"
                else:
                    gt = "polygon"
                write_qgis_layer_style(output_path, layer_name, colour, gt)

    # ── Verify ──
    print(f"\n  ── Output Verification ──")
    if os.path.exists(output_path):
        layers = fiona.listlayers(output_path)
        print(f"  Layers: {[l for l in layers if l != 'layer_styles']}")
        for layer_name in layers:
            if layer_name == "layer_styles":
                continue
            try:
                gdf = gpd.read_file(output_path, layer=layer_name)
                geom_types = gdf.geometry.geom_type.unique() if len(gdf) > 0 else []
                print(f"    {layer_name}: {len(gdf)} features, "
                      f"CRS={gdf.crs}, types={list(geom_types)}")
            except Exception as e:
                print(f"    {layer_name}: (error: {e})")

    # ── Summary ──
    print(f"\n{'=' * 70}")
    print(f"  CONVERSION COMPLETE")
    print(f"  Output:         {output_path}")
    print(f"  Layers written: {written}")
    print(f"  Total features: {total:,}")
    if os.path.exists(output_path):
        print(f"  File size:      {os.path.getsize(output_path) / 1024 / 1024:.2f} MB")
    print(f"{'=' * 70}\n")

    if written > 0:
        print("  ℹ️  Open in QGIS — each class will show as separate layers")
        print("  with correct geometry types (polygon/line/point) and colours.\n")


if __name__ == "__main__":
    main()
