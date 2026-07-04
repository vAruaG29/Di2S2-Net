"""GeoJSON serving for one vector layer of one dataset.

Two parallel endpoints with the same shape — one for prediction
GeoPackage layers and one for ground-truth layers vectorised from
per-tile masks. Both return EPSG:4326 FeatureCollections.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse

from .. import gpkg_reader, groundtruth


router = APIRouter(prefix="/api/datasets", tags=["layers"])


@router.get("/{name}/gpkg/{layer}")
def layer_geojson(name: str, layer: str) -> JSONResponse:
    try:
        fc = gpkg_reader.read_layer_geojson(name, layer)
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))
    except Exception as e:
        raise HTTPException(500, f"Failed to read layer {layer!r}: {e}")
    return JSONResponse(fc)


@router.get("/{name}/gt/{layer}")
def gt_layer_geojson(name: str, layer: str) -> JSONResponse:
    """Vectorise the per-tile masks for `name` into per-class
    polygons. First call is slow (5-30 s); subsequent calls hit the
    on-disk cache under `<portal_workspace>/_gt_cache/<DS>/`."""
    if not groundtruth.has_groundtruth(name):
        raise HTTPException(404, f"No ground-truth masks for {name!r}")
    try:
        fc = groundtruth.read_layer_geojson(name, layer)
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        raise HTTPException(500, f"Failed to vectorise GT layer {layer!r}: {e}")
    return JSONResponse(fc)
