"""
Dataset discovery + detail.

A "dataset" is anything with a stitched prediction raster on disk. We
look in two places:

  1. Portal workspace (preferred — modern, isolated per-run):
        <portal_workspace>/<NAME>/outputs/stitched/<NAME>_pred.tif
  2. Bundle offline pipeline (historical results, read-only):
        <bundle>/outputs/stitched/<NAME>_pred.tif

Each returned record carries `origin: "portal" | "offline"` so the
frontend can label them and route the correct GPKG/raster paths.
"""

from __future__ import annotations

import csv
import json
import re
import threading
from pathlib import Path
from typing import Any

import rasterio
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from rasterio.warp import transform_bounds

from .. import gpkg_reader, groundtruth, tiler_routes
from ..checkpoints import split_effective
from ..job_paths import list_portal_jobs
from ..settings import (
    PRED_SUFFIX,
    class_colors,
    evaluation_dir,
    find_source_cog,
    gpkg_dir,
    gpkg_pred_filename,
    portal_job_dir,
    stitched_dir,
    stitched_pred_filename,
)


router = APIRouter(prefix="/api/datasets", tags=["datasets"])


def _portal_paths(name: str) -> dict[str, Path]:
    root = portal_job_dir(name)
    out  = root / "outputs"
    return {
        # Pipeline writes both `<name>_pred.tif` and `<name>_pred.gpkg`.
        "stitched":   out / "stitched" / stitched_pred_filename(name),
        "gpkg":       out / "gpkg"     / gpkg_pred_filename(name),
        "metrics":    out / "evaluation" / f"{name}_metrics.csv",
    }


def _offline_paths(name: str) -> dict[str, Path]:
    return {
        "stitched": stitched_dir() / stitched_pred_filename(name),
        "gpkg":     gpkg_dir()     / gpkg_pred_filename(name),
        "metrics":  evaluation_dir() / f"{name}_metrics.csv",
    }


def _resolve(name: str) -> tuple[str, dict[str, Path]] | None:
    """Return (origin, paths-dict) for the dataset, or None if unknown.

    Resolution order — critical for keeping the list endpoint's badge
    and the detail endpoint's layer list consistent:

      1. **Portal predictions** (gpkg / stitched in the per-job
         workspace) win — the user just re-ran inference, this is
         their newest result.
      2. **Offline (LIB) predictions** — for datasets shipped in the
         bundle's `outputs/`, even when a portal-job folder also
         exists (e.g. a stale source-only re-run attempt).
      3. **Source-only portal job** — last resort, for uploads that
         haven't been inferenced yet AND have no LIB counterpart.
         Surfacing these lets the map show basemap + COG immediately
         after upload.

    The old order had (3) ahead of (2), which caused the list endpoint
    to find a 5-layer offline GPKG and tag the row "GPKG · 5", while
    the detail endpoint then resolved via (3) — portal paths with no
    GPKG — and returned `layers: []`. ClassesFloat would (correctly)
    say "no layers" while the list (correctly) said GPKG. That's the
    inconsistency the user reported.
    """
    p = _portal_paths(name)
    if p["gpkg"].exists() or p["stitched"].exists():
        return "portal", p
    o = _offline_paths(name)
    if o["gpkg"].exists() or o["stitched"].exists():
        return "offline", o
    if find_source_cog(name) is not None:
        # Source raster on disk but no predictions anywhere — still a
        # valid portal dataset (basemap + COG will render fine).
        return "portal", p
    return None


def _is_upload_dataset(origin: str, name: str) -> bool:
    """True iff this dataset originated from an upload (chunked or
    multipart). Detected via the `_meta.json` marker dropped by
    `routes/upload.py` (canonical), with a fallback to legacy /
    new folder-name patterns so older uploads still classify
    correctly.

    Offline (LIB) datasets are never uploads — bail early."""
    if origin != "portal":
        return False
    meta_p = portal_job_dir(name) / "_meta.json"
    if meta_p.exists():
        try:
            if (json.loads(meta_p.read_text()) or {}).get("source_kind") == "upload":
                return True
        except Exception:
            pass
    if name.startswith("upload_"):                                # legacy
        return True
    if name.endswith("_uploaded"):
        return True
    if re.search(r"_uploaded_\d+$", name):
        return True
    return False


def _summary(origin: str, name: str, paths: dict[str, Path]) -> dict[str, Any]:
    # `stitched` and `prediction_cog` point at the same file — the
    # `<name>_pred.tif` raster the pipeline produced. The duplicate name
    # is to make it crystal clear in the API which file is the prediction
    # (and therefore the one to overlay on top of the source imagery).
    #
    # `layer_count`: the actual number of vector layers in the GPKG (or
    # 0 if the GPKG is missing / empty). The frontend uses this to
    # avoid showing a misleading "GPKG" badge on rows whose GeoPackage
    # exists but contains no usable layers (we've seen this happen on
    # some legacy datasets where only QGIS style tables made it in).
    pred_path = paths["stitched"]
    gpkg_path = paths["gpkg"]
    layer_count = 0
    gpkg_mtime: float | None = None
    if gpkg_path.exists():
        try:
            layer_count = len(gpkg_reader.list_layers_at(gpkg_path))
        except Exception:
            layer_count = 0
        try:
            gpkg_mtime = gpkg_path.stat().st_mtime
        except OSError:
            gpkg_mtime = None
    # Split a checkpoint-namespaced name (`<base>@@<id>`) back into its
    # parts so the UI can show a clean base name + a "which model" badge.
    # For ordinary datasets `checkpoint` is None and `base_name == name`.
    base_name, checkpoint_id = split_effective(name)
    return {
        "name": name,
        "base_name": base_name,
        "checkpoint": checkpoint_id,
        "origin": origin,
        "stitched":       str(pred_path),
        "prediction_cog": str(pred_path),
        "has_gpkg":       layer_count > 0,                 # ← treat empty GPKGs as no GPKG
        "gpkg_present":   gpkg_path.exists(),              # raw file existence
        "layer_count":    layer_count,
        # File mtime of the resolved GeoPackage — used by the frontend
        # as a cache buster so re-running inference on a dataset with
        # the same name surfaces the NEW features instead of the
        # in-memory GeoJSON the previous run cached.
        "gpkg_mtime":     gpkg_mtime,
        "has_metrics":    paths["metrics"].exists(),
        "prediction_filename": pred_path.name,             # always ends in _pred.tif
        # `is_upload`: the frontend uses this to exclude upload-origin
        # datasets from the Train / Test lists (they only belong in
        # the Upload tab's "recent uploads" pane).
        "is_upload":      _is_upload_dataset(origin, name),
    }


@router.get("")
def list_datasets() -> list[dict[str, Any]]:
    seen: dict[str, dict[str, Any]] = {}

    # Portal-run datasets first — they take precedence over an offline
    # entry of the same name (e.g. a re-run).
    # A portal job counts as "complete" if EITHER the stitched raster
    # OR the GeoPackage is on disk — the stitched raster gets cleaned
    # after a successful run, so checking only that would silently
    # drop every recently-finished upload from the gallery.
    for ds in list_portal_jobs():
        p = _portal_paths(ds)
        if p["gpkg"].exists() or p["stitched"].exists():
            seen[ds] = _summary("portal", ds, p)

    # Offline / historical pipeline results — match either the
    # stitched raster or the GeoPackage by exact name, so a manually-
    # slimmed library (stitched removed, gpkg kept) still surfaces.
    sd = stitched_dir()
    if sd.exists():
        for tif in sorted(sd.glob(f"*{PRED_SUFFIX}")):
            name = tif.name[: -len(PRED_SUFFIX)]
            if not name or name in seen:
                continue
            seen[name] = _summary("offline", name, _offline_paths(name))
    gd = gpkg_dir()
    if gd.exists():
        for gpkg in sorted(gd.glob("*_pred.gpkg")):
            name = gpkg.name[: -len("_pred.gpkg")]
            if not name or name in seen:
                continue
            seen[name] = _summary("offline", name, _offline_paths(name))

    return list(seen.values())


@router.get("/{name}")
def dataset_detail(name: str) -> dict[str, Any]:
    resolved = _resolve(name)
    if resolved is None:
        raise HTTPException(404, f"No stitched raster for {name!r}")
    origin, paths = resolved

    info: dict[str, Any] = _summary(origin, name, paths)

    # ── Source imagery (the actual aerial photo, served via TiTiler).
    # The frontend draws this BELOW the prediction layers so the user
    # sees the satellite content underneath the segmentation overlay.
    src = find_source_cog(name)
    info["source_cog"] = str(src) if src else None
    if src is not None:
        # Use whichever tile path TiTiler actually registered (discovered
        # at startup via `tiler_routes.init_from_app`). PNG is lossless
        # — same pixel fidelity as the source COG. Speed comes from
        # TiTiler reading the COG's internal overview pyramid plus the
        # Cache-Control middleware caching tiles in the browser.
        from urllib.parse import quote
        url_q = quote(str(src), safe="")
        tile_tmpl = tiler_routes.tile_url_template()        # has literal {z}{x}{y}
        prev_path = tiler_routes.preview_path()
        # PNG can only encode 1/2/3/4 bands. Aerial COGs are often 5-band
        # (R, G, B, NIR, alpha) — TiTiler doesn't auto-pick the visible
        # ones, so we have to force band selection to RGB or the tile
        # request 500s with "PNG driver doesn't support N bands".
        # `bidx=1&bidx=2&bidx=3` is harmless on 3-band RGB COGs (just
        # re-selects them in order) and fixes the 4/5/6-band case.
        bidx = "bidx=1&bidx=2&bidx=3"
        info["tiles_url_template"] = (
            f"{tile_tmpl}?url={url_q}&format=png&resampling=bilinear&{bidx}"
        )
        info["tilejson_url"] = info["tiles_url_template"]   # legacy alias
        info["preview_url"]  = (
            f"{prev_path}?url={url_q}&max_size=1024&format=png&{bidx}"
        )
    else:
        info["tiles_url_template"] = None
        info["tilejson_url"] = None
        info["preview_url"] = None

    # Raster bounds in WGS84 (for fitBounds in the map).
    # Prefer the source aerial COG since the stitched prediction
    # raster is deleted by `_cleanup_intermediates` after a successful
    # run. Both rasters share the same extent + CRS by construction,
    # so either is fine — but the COG is always still on disk.
    candidates = [p for p in (find_source_cog(name), paths["stitched"]) if p]
    info["crs"] = None
    info["bounds_wgs84"] = None
    for tif in candidates:
        if not Path(tif).exists():
            continue
        try:
            with rasterio.open(str(tif)) as src:
                b = src.bounds
                crs = src.crs
                wgs_bounds = transform_bounds(
                    crs, "EPSG:4326",
                    b.left, b.bottom, b.right, b.top,
                    densify_pts=21,
                )
            info["crs"] = str(crs)
            info["bounds_wgs84"] = list(wgs_bounds)
            break
        except Exception as exc:
            info["bounds_error"] = str(exc)
            continue

    # Vector layers + feature counts (cheap — Fiona only iterates lengths).
    layers: list[dict[str, Any]] = []
    if info["has_gpkg"]:
        colours = class_colors()
        layer_names = gpkg_reader.list_layers_at(paths["gpkg"])
        for ln in layer_names:
            layers.append({
                "name": ln,
                "feature_count": gpkg_reader.count_features_at(paths["gpkg"], ln),
                "colour": colours.get(ln, "#888888"),
            })

        # Kick off a single background thread that materialises the
        # per-layer GeoJSON cache. The frontend's LayerStack effect
        # will fire its parallel fetches the moment React renders the
        # dataset; without this warm-up the first hit pays the full
        # `gpd.read_file` + reproject + serialise cost (~1-5 s per
        # layer) and the predictions take a noticeable beat to land
        # on the map. ONE thread, sequential, so we don't contend
        # with the in-flight parallel requests on disk/GIL.
        threading.Thread(
            target=_warm_predictions, args=(name, paths["gpkg"], layer_names),
            daemon=True,
        ).start()
    info["layers"] = layers

    # Metrics CSV (optional).
    metrics: list[dict[str, str]] = []
    csv_path = paths["metrics"]
    if csv_path.exists():
        with open(csv_path) as f:
            for row in csv.DictReader(f):
                metrics.append(dict(row))
    info["metrics"] = metrics

    # Ground-truth layers — derived from the per-tile masks. Empty
    # list means no GT for this dataset; the UI hides the toggle.
    # We deliberately DO NOT pre-warm the GT cache: GT vectorisation
    # is heavy and spawning per-layer threads here used to starve the
    # predictions pre-warm above for CPU/GIL. GT is off by default,
    # so the user only pays the cost when they explicitly enable it.
    info["gt_layers"] = groundtruth.list_layers(name)

    return info


def _warm_predictions(name: str, gpkg_path: Path, layer_names: list[str]) -> None:
    """Sequentially materialise the GeoJSON cache for every layer in
    the dataset's prediction GeoPackage. Runs on a single daemon
    thread so it never competes with the frontend's parallel fetches
    on GIL / disk I/O — if the frontend's request lands first, the
    backend just serves the freshly-built cache file."""
    try:
        for ln in layer_names:
            try:
                gpkg_reader.read_layer_geojson(name, ln)
            except Exception:
                # Silent best-effort; the on-demand endpoint will
                # surface real errors if the user actually requests
                # this layer.
                pass
    except Exception:
        pass


@router.get("/{name}/gpkg-download")
def download_gpkg(name: str):
    """Stream the dataset's GeoPackage with a `Content-Disposition:
    attachment` header so the browser pops up a Save-As dialog."""
    resolved = _resolve(name)
    if resolved is None:
        raise HTTPException(404, f"Unknown dataset {name!r}")
    _, paths = resolved
    gpkg = paths["gpkg"]
    if not gpkg.exists():
        raise HTTPException(404, f"No GeoPackage on disk for {name!r}")
    return FileResponse(
        path=str(gpkg),
        media_type="application/geopackage+sqlite3",
        filename=gpkg.name,
    )
