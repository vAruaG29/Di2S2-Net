"""
Portal FastAPI app.

Mounts:
  - /api/*    our own routes (datasets, layers, uploads, inference, jobs)
  - /tiles/cog/*   TiTiler COG endpoints (tilejson, {z}/{x}/{y}, preview, info)

Run locally:
    uvicorn portal.backend.app:app --reload --port 8000
"""

from __future__ import annotations

import os

# ── GDAL/rasterio tuning. Must run before titiler imports rasterio. ─────
# These knobs are the single biggest win for COG-tile latency, especially
# over a slow remote filesystem.
os.environ.setdefault("GDAL_CACHEMAX",                "512")  # MB raster cache
os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
os.environ.setdefault("GDAL_HTTP_MULTIPLEX",          "YES")
os.environ.setdefault("VSI_CACHE",                    "TRUE")
os.environ.setdefault("VSI_CACHE_SIZE",               "536870912")  # 512 MB
os.environ.setdefault("CPL_VSIL_CURL_ALLOWED_EXTENSIONS", ".tif,.tiff,.TIF,.TIFF")
os.environ.setdefault("GDAL_NUM_THREADS",             "ALL_CPUS")

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware

from . import tiler_routes
from .routes import datasets, inference, layers, upload
from .settings import class_colors, feature_classes


# TiTiler is optional at import-time so the app still loads (with a
# friendly error) on machines where titiler isn't installed yet.
try:
    from titiler.core.factory import TilerFactory          # type: ignore
    _HAVE_TITILER = True
except Exception as _e:  # pragma: no cover - env-specific
    _HAVE_TITILER = False
    _TITILER_ERR = str(_e)


app = FastAPI(
    title="DINOv3 + HRDecoder Demo Portal",
    version="0.1.0",
    description="Browse pre-computed segmentation results or run inference live.",
)

# Local dev: vite runs on :5173, FastAPI on :8000. CORS for that origin
# (vite's dev proxy also covers this — belt + braces).
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class _TileCacheControl(BaseHTTPMiddleware):
    """Set aggressive Cache-Control on TiTiler tile responses so the
    browser stops re-requesting the same tiles after every pan/zoom.

    Tiles are immutable per (z,x,y,url) so we can mark them so. The
    tilejson endpoint changes whenever the dataset changes, so we use a
    short cache window there."""

    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        p = request.url.path
        if p.startswith("/tiles/cog/"):
            if p.endswith("tilejson.json"):
                response.headers["Cache-Control"] = "public, max-age=60"
            else:
                response.headers["Cache-Control"] = (
                    "public, max-age=31536000, immutable"
                )
        return response


app.add_middleware(_TileCacheControl)


@app.get("/api/health")
def health() -> dict:
    return {
        "status": "ok",
        "titiler": _HAVE_TITILER,
    }


@app.get("/api/config")
def config() -> dict:
    """Bundle-level config the frontend needs at boot time."""
    return {
        "class_colors": class_colors(),
        "feature_classes": feature_classes(),
    }


app.include_router(datasets.router)
app.include_router(layers.router)
app.include_router(upload.router)
app.include_router(inference.router)


# ── TiTiler mount ──────────────────────────────────────────────────────
if _HAVE_TITILER:
    cog = TilerFactory()
    app.include_router(cog.router, prefix="/tiles/cog", tags=["tiles"])
else:                                                  # pragma: no cover
    @app.get("/tiles/cog/{rest:path}")
    def _titiler_missing(rest: str) -> dict:
        return {
            "error": "titiler.core is not installed",
            "details": _TITILER_ERR,  # noqa: F821
            "hint": "pip install titiler.core",
        }


# ── Discover the real TiTiler URL patterns ────────────────────────────
# Different TiTiler versions register different path shapes. Walk
# `app.routes` once at startup and remember what's actually mounted so
# the dataset detail endpoint can hand MapLibre a URL that 200s.
tiler_routes.init_from_app(app)


@app.get("/api/debug/routes")
def debug_routes() -> dict:
    """List every TiTiler route that's actually registered.

    Useful when tile requests are 404ing — paste the output here so we
    know exactly what to point MapLibre at.
    """
    return {
        "tile_url_template": tiler_routes.tile_url_template(),
        "preview_path":      tiler_routes.preview_path(),
        "tilejson_path":     tiler_routes.tilejson_path(),
        "all_cog_paths":     tiler_routes.all_cog_paths(),
    }
