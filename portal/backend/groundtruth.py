"""
Per-dataset ground-truth extraction for the portal.

Background
----------
The pipeline produces per-tile multiclass masks at
`<bundle>/masks/<DATASET>/<tile-stem>_mask.tif` (single-band uint8,
pixel value = class ID, 0=background). These are the canonical
ground truth — exactly the same labels the model is trained against,
after canonicalisation from the raw region-wide shapefiles in
`data/train/{shp-file, PB_training_dataSet_shp_file/shp-file}`.

For the portal we want to show those masks as vector overlays on the
map, one feature class at a time. Strategy:

    1. Discover every `_mask.tif` for the dataset.
    2. For each non-background class ID, run `rasterio.features.shapes`
       on each tile to extract polygons.
    3. Merge polygons across tiles (unary_union — features straddling
       a tile boundary get stitched back together).
    4. Reproject to EPSG:4326 for MapLibre.
    5. Cache the resulting FeatureCollection at
       `<portal_workspace>/_gt_cache/<DATASET>/<class>.geojson`.

The cache is invalidated by source-mask mtime (same pattern as
`gpkg_reader.py`). First fetch of a layer for a fresh dataset is
slow (5-30 s depending on tile count); subsequent loads are instant.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np
import rasterio
import rasterio.features
from rasterio.warp import transform_geom
from shapely.geometry import mapping, shape

from .settings import (
    BUNDLE_ROOT,
    masks_dir,
    portal_gt_cache_dir,
    portal_job_dir,
    feature_classes,
)

WGS84 = "EPSG:4326"


# ── Mask discovery ────────────────────────────────────────────────────

def _mask_dirs_for(dataset_name: str) -> list[Path]:
    """Return every directory that *might* hold per-tile masks for this
    dataset, in priority order (portal-side first, then bundle-side).

    Portal-job runs write masks to `<portal_workspace>/<DS>/masks/<DS>/`.
    Bundle-side `prepare_labels` writes to `<bundle>/masks/<DS>/`.
    """
    candidates = [
        portal_job_dir(dataset_name) / "masks" / dataset_name,
        masks_dir() / dataset_name,
        BUNDLE_ROOT / "masks" / dataset_name,
    ]
    return [p for p in candidates if p.is_dir()]


def _list_mask_tiles(dataset_name: str) -> list[Path]:
    for d in _mask_dirs_for(dataset_name):
        tiles = sorted(d.glob("*_mask.tif"))
        if tiles:
            return tiles
    return []


def has_groundtruth(dataset_name: str) -> bool:
    return bool(_list_mask_tiles(dataset_name))


# ── Class-ID ↔ canonical-layer-name table ─────────────────────────────

def _class_id_to_name() -> dict[int, str]:
    """Invert feature_classes ({name: id}) → {id: name}, skipping
    background. Matches the GPKG layer names the prediction side uses."""
    out: dict[int, str] = {}
    for name, cid in feature_classes().items():
        if int(cid) == 0:
            continue
        # First wins — the YAML happens to be in priority order anyway.
        out.setdefault(int(cid), name)
    return out


def _class_id_for_layer(layer_name: str) -> int | None:
    for cid, name in _class_id_to_name().items():
        if name == layer_name:
            return cid
    # Allow case-insensitive / underscore-loose lookup as a fallback.
    norm = layer_name.lower().replace("_", "")
    for cid, name in _class_id_to_name().items():
        if name.lower().replace("_", "") == norm:
            return cid
    return None


# ── Layer discovery (which classes have any features in this DS) ─────

def list_layers(dataset_name: str) -> list[str]:
    """Names of every class that actually has pixels in this dataset's
    masks (skips classes that are absent from every tile so the UI
    doesn't show empty toggles)."""
    tiles = _list_mask_tiles(dataset_name)
    if not tiles:
        return []

    seen: set[int] = set()
    # Read just the value histogram from each tile — cheap.
    for t in tiles:
        try:
            with rasterio.open(t) as src:
                arr = src.read(1)
                vals = np.unique(arr)
                for v in vals.tolist():
                    if v != 0:
                        seen.add(int(v))
        except Exception:
            continue
        # Quick exit once we've found every known class.
        if len(seen) >= len(_class_id_to_name()):
            break

    id2name = _class_id_to_name()
    out: list[str] = []
    for cid, name in sorted(id2name.items()):
        if cid in seen:
            out.append(name)
    return out


# ── Vectorisation ─────────────────────────────────────────────────────

def _extract_polygons_for_tile(
    tile_path: Path, class_id: int, simplify_px: float = 1.0,
) -> tuple[list[dict], str | None]:
    """Return (list of native-CRS GeoJSON geometries for `class_id`
    pixels in this tile, the tile's CRS as an EPSG string).

    Reprojection to WGS-84 is intentionally NOT done here — see the
    note in `_build_layer_geojson` for why.

    `simplify_px` is the Douglas-Peucker tolerance in *pixels*.  At
    1024×1024 with metre-scale ground sampling, 1 px is a few cm —
    invisible at any reasonable zoom but cuts vertex counts ~5×."""
    with rasterio.open(tile_path) as src:
        arr = src.read(1)
        mask = (arr == class_id).astype(np.uint8)
        if not mask.any():
            return [], None

        # Convert px tolerance to source CRS units using the affine.
        # `src.transform.a` is the pixel width in CRS units.
        tol = float(simplify_px) * abs(src.transform.a)

        out: list[dict] = []
        for geom, val in rasterio.features.shapes(
            mask, mask=mask.astype(bool), transform=src.transform
        ):
            if val != 1:
                continue
            if tol > 0:
                try:
                    g = shape(geom).simplify(tol, preserve_topology=True)
                    if g.is_empty:
                        continue
                    out.append(mapping(g))
                except Exception:
                    out.append(geom)
            else:
                out.append(geom)
        crs_str = str(src.crs) if src.crs else None
        return out, crs_str


def _build_layer_geojson(
    dataset_name: str, layer_name: str,
) -> dict:
    """Vectorise every tile's mask for the requested class and return
    a GeoJSON FeatureCollection in EPSG:4326.

    Perf notes (relative to the previous implementation):

      - **Reprojection is batched.** The old code called
        `transform_geom` once per polygon (often thousands of times);
        now we collect every native-CRS geometry first and reproject
        them in one pass per tile-CRS bucket.  Since all of a single
        dataset's tiles share one CRS, this is effectively a single
        transform per polygon.
      - **No `unary_union`.** Stitching duplicated polygons across the
        128 px tile overlaps used to dominate runtime (O(n²) in the
        worst case for shapely).  We now emit overlapping polygons as
        separate features — visually identical at any reasonable zoom
        and ~10× faster on large datasets.
      - **`simplify(1 px)`** drops the vertex count of every polygon
        by ~5× with no visible loss; this also makes the JSON payload
        smaller for the frontend.
    """
    class_id = _class_id_for_layer(layer_name)
    if class_id is None:
        raise ValueError(f"Unknown ground-truth class {layer_name!r}")
    tiles = _list_mask_tiles(dataset_name)
    if not tiles:
        return {"type": "FeatureCollection", "features": []}

    # Bucket native-CRS geometries by their tile CRS so we can do one
    # batched reprojection per bucket. In practice every tile shares
    # one CRS, so this is a single bucket.
    by_crs: dict[str, list[dict]] = {}
    for t in tiles:
        geoms, crs_str = _extract_polygons_for_tile(t, class_id)
        if not geoms or crs_str is None:
            continue
        by_crs.setdefault(crs_str, []).extend(geoms)

    if not by_crs:
        return {"type": "FeatureCollection", "features": []}

    features: list[dict] = []
    for crs_str, geoms in by_crs.items():
        if crs_str.upper() == WGS84.upper():
            iter_wgs = geoms
        else:
            iter_wgs = []
            for g in geoms:
                try:
                    iter_wgs.append(transform_geom(crs_str, WGS84, g, precision=7))
                except Exception:
                    continue
        for g in iter_wgs:
            features.append({
                "type": "Feature",
                "geometry": g,
                "properties": {"class": layer_name},
            })

    return {"type": "FeatureCollection", "features": features}


# ── Public read-with-cache API ────────────────────────────────────────

def _cache_path(dataset_name: str, layer_name: str) -> Path:
    p = portal_gt_cache_dir() / dataset_name / f"{layer_name}.geojson"
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def _newest_mask_mtime(dataset_name: str) -> float:
    tiles = _list_mask_tiles(dataset_name)
    return max((t.stat().st_mtime for t in tiles), default=0.0)


def read_layer_geojson(dataset_name: str, layer_name: str) -> dict:
    """Return the GT layer as a GeoJSON FeatureCollection in EPSG:4326,
    materialising and caching on first request. Cache is invalidated
    when ANY tile mask is newer than the cached file (rare in practice
    but cheap to check)."""
    cache_p = _cache_path(dataset_name, layer_name)
    src_mtime = _newest_mask_mtime(dataset_name)
    if cache_p.exists() and src_mtime > 0 and cache_p.stat().st_mtime >= src_mtime:
        with open(cache_p) as f:
            return json.load(f)

    fc = _build_layer_geojson(dataset_name, layer_name)
    with open(cache_p, "w") as f:
        json.dump(fc, f)
    return fc
