"""
TiTiler route discovery.

The exact URL pattern for COG tiles depends on the installed TiTiler
version — some builds mount `/tiles/cog/tiles/{z}/{x}/{y}`, others
require a `{TileMatrixSetId}` segment (e.g.
`/tiles/cog/tiles/{TileMatrixSetId}/{z}/{x}/{y}`), and some only expose
TileJSON under a TMS-specific path too.

Instead of hard-coding any of those, we walk the FastAPI route table
once at startup, pick the actual paths, and substitute the default TMS
(`WebMercatorQuad`) for any `{TileMatrixSetId}` placeholders so the
template MapLibre receives has only `{z}/{x}/{y}` left.

If the introspection somehow fails, we fall back to the most common
historical paths.
"""

from __future__ import annotations

import re
from typing import Iterable


_DEFAULT_TMS = "WebMercatorQuad"

# Substitute any case variant of the placeholder: {TileMatrixSetId},
# {tileMatrixSetId}, {tilematrixsetid}, etc. Different TiTiler builds
# use different capitalisations.
_TMS_PLACEHOLDER = re.compile(r"\{tile[mM]atrix[sS]et[iI]d\}", re.IGNORECASE)

_tile_template: str | None = None
_preview_path:  str | None = None
_tilejson_path: str | None = None
_all_paths:     list[str] = []


def _has(path: str, *needles: str) -> bool:
    return all(n in path for n in needles)


def _sub_tms(path: str) -> str:
    return _TMS_PLACEHOLDER.sub(_DEFAULT_TMS, path)


def _pick_shortest(paths: Iterable[str]) -> str | None:
    paths = sorted(set(paths), key=lambda p: (len(p), p))
    return paths[0] if paths else None


def init_from_app(app) -> None:
    """Walk `app.routes` once and remember which TiTiler paths exist."""
    global _tile_template, _preview_path, _tilejson_path, _all_paths

    _all_paths = []
    tile_paths: list[str] = []
    preview_paths: list[str] = []
    tilejson_paths: list[str] = []

    for r in app.routes:
        path = getattr(r, "path", "") or ""
        if "/tiles/cog" not in path:
            continue
        _all_paths.append(path)
        if _has(path, "{z}", "{x}", "{y}"):
            tile_paths.append(path)
        if path.endswith("/preview") or path.endswith("/preview.{format}"):
            preview_paths.append(path)
        if path.endswith("/tilejson.json"):
            tilejson_paths.append(path)

    pick = _pick_shortest(tile_paths)
    if pick:
        _tile_template = _sub_tms(pick)
    else:
        _tile_template = "/tiles/cog/tiles/{z}/{x}/{y}"  # historical default

    pick = _pick_shortest(preview_paths)
    _preview_path = _sub_tms(pick) if pick else "/tiles/cog/preview"

    pick = _pick_shortest(tilejson_paths)
    _tilejson_path = _sub_tms(pick) if pick else None


def tile_url_template() -> str:
    """Tile URL with literal {z}/{x}/{y} placeholders, no query string."""
    return _tile_template or "/tiles/cog/tiles/{z}/{x}/{y}"


def preview_path() -> str:
    return _preview_path or "/tiles/cog/preview"


def tilejson_path() -> str | None:
    return _tilejson_path


def all_cog_paths() -> list[str]:
    return list(_all_paths)
