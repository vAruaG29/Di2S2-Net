"""
GPKG → GeoJSON reader.

Datasets can live in two places:

  - Portal workspace:  <portal_workspace>/<NAME>/outputs/gpkg/<NAME>_pred.gpkg
  - Offline pipeline:  <bundle>/outputs/gpkg/<NAME>_pred.gpkg

We expose two layers of API:

  - `*_at(path, ...)` operate on a raw .gpkg path (used by datasets.py
    once it has already resolved the dataset's origin).
  - `read_layer_geojson(name, layer)` is the resolver used by the
    `/api/datasets/{name}/gpkg/{layer}` route — it picks the portal
    GPKG if present, else falls back to the offline one.

All GeoJSON is reprojected to EPSG:4326 (MapLibre's native CRS) and
cached on disk under `portal_workspace/_gpkg_cache/<NAME>/<LAYER>.geojson`
keyed by source mtime.
"""

from __future__ import annotations

import json
from pathlib import Path

import fiona
import geopandas as gpd

from .settings import (
    gpkg_dir as offline_gpkg_dir,
    gpkg_pred_filename,
    portal_gpkg_cache_dir,
    portal_job_dir,
)


WGS84 = "EPSG:4326"


# ── Path-based primitives (the "_at" variants) ──────────────────────────

# Non-spatial / housekeeping tables that GeoPackage / QGIS keep in the
# .gpkg alongside the real layers. They have no geometry and would just
# clutter the layer panel with zero-feature rows, so we filter them out
# of every listing the portal exposes.
_HIDDEN_LAYERS = {
    "layer_styles",         # QGIS QML styles (written by write_qgis_layer_style)
    "qgis_projects",        # QGIS project blobs
}


def list_layers_at(gpkg: Path) -> list[str]:
    if not gpkg.exists():
        return []
    try:
        layers = list(fiona.listlayers(str(gpkg)))
    except Exception:
        return []
    return [ln for ln in layers if ln not in _HIDDEN_LAYERS]


def count_features_at(gpkg: Path, layer: str) -> int:
    if not gpkg.exists():
        return 0
    try:
        with fiona.open(str(gpkg), layer=layer) as src:
            return len(src)
    except Exception:
        return 0


# ── Dataset-name based resolver ─────────────────────────────────────────

def resolve_gpkg(dataset_name: str) -> Path | None:
    """Find the GPKG for a dataset (`<name>_pred.gpkg`). Portal beats offline.

    For backwards-compatibility we also accept the older `<name>.gpkg`
    naming in case it's already on disk somewhere.
    """
    fname_new = gpkg_pred_filename(dataset_name)             # NAGUL_pred.gpkg
    fname_old = f"{dataset_name}.gpkg"                       # NAGUL.gpkg

    portal_outputs = portal_job_dir(dataset_name) / "outputs" / "gpkg"
    offline_outputs = offline_gpkg_dir()

    for root in (portal_outputs, offline_outputs):
        for fname in (fname_new, fname_old):
            p = root / fname
            if p.exists():
                return p
    return None


def list_layers(dataset_name: str) -> list[str]:
    p = resolve_gpkg(dataset_name)
    return list_layers_at(p) if p else []


def layer_feature_count(dataset_name: str, layer: str) -> int:
    p = resolve_gpkg(dataset_name)
    return count_features_at(p, layer) if p else 0


def read_layer_geojson(dataset_name: str, layer: str) -> dict:
    """Return the layer as a GeoJSON FeatureCollection (EPSG:4326)."""
    gpkg = resolve_gpkg(dataset_name)
    if gpkg is None:
        raise FileNotFoundError(f"No GPKG found for dataset {dataset_name!r}")

    cache_p = portal_gpkg_cache_dir() / dataset_name / f"{layer}.geojson"
    cache_p.parent.mkdir(parents=True, exist_ok=True)
    if cache_p.exists() and cache_p.stat().st_mtime >= gpkg.stat().st_mtime:
        with open(cache_p) as f:
            return json.load(f)

    gdf = gpd.read_file(gpkg, layer=layer)
    if gdf.crs is None:
        gdf.set_crs(WGS84, inplace=True, allow_override=True)
    elif str(gdf.crs).upper() != WGS84.upper():
        gdf = gdf.to_crs(WGS84)

    safe = gdf.copy()
    for col in safe.columns:
        if col == "geometry":
            continue
        if safe[col].dtype.name.startswith("datetime"):
            safe[col] = safe[col].astype(str)

    # `gdf.to_json()` already serialises to a JSON string. Write that
    # string straight to disk (avoids a json.loads/json.dump roundtrip
    # the old code was doing — for big layers this was a real cost).
    # Then parse once to return the dict the FastAPI route needs.
    geojson_str = safe.to_json()
    with open(cache_p, "w") as f:
        f.write(geojson_str)
    return json.loads(geojson_str)
