"""
Portal settings — every path is resolved against the pipeline's own
config files so the portal stays in lockstep with the rest of the
bundle. Re-read on each request (cheap).
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml


# Bundle root = .../final_submission_and_inference  (parent of `portal/`)
BUNDLE_ROOT = Path(__file__).resolve().parent.parent.parent
PIPE_ROOT = BUNDLE_ROOT / "dinov3_hrdecoder_pipeline"

DATA_PREP_CFG = PIPE_ROOT / "configs" / "data_prep.yaml"
TRAIN_CFG     = PIPE_ROOT / "configs" / "train.yaml"


@lru_cache(maxsize=1)
def data_prep_cfg() -> dict:
    with open(DATA_PREP_CFG) as f:
        return yaml.safe_load(f)


@lru_cache(maxsize=1)
def train_cfg() -> dict:
    with open(TRAIN_CFG) as f:
        return yaml.safe_load(f)


def output_dir() -> Path:
    """The directory `run_pipeline.py` writes predictions/stitched/gpkg into."""
    return Path(train_cfg()["paths"]["output_dir"])


def stitched_dir() -> Path:
    return output_dir() / "stitched"


def gpkg_dir() -> Path:
    return output_dir() / "gpkg"


def predictions_dir() -> Path:
    return output_dir() / "predictions"


def evaluation_dir() -> Path:
    return output_dir() / "evaluation"


def gpkg_cache_dir() -> Path:
    p = output_dir() / "gpkg_cache"
    p.mkdir(parents=True, exist_ok=True)
    return p


def test_raw_dir() -> Path:
    return Path(data_prep_cfg()["paths"]["test_raw_dir"])


# ── Portal-managed workspace ────────────────────────────────────────────
# Everything the portal generates (uploads, COGs, tiles, predictions,
# stitched, gpkg) lives here, one sub-dir per dataset. The offline
# pipeline's <bundle>/cog, /tiles, /outputs, /masks are never written to
# by the portal — they stay read-only for "Browse results" of historical
# offline runs.

def portal_workspace_dir() -> Path:
    p = BUNDLE_ROOT / "portal_workspace"
    p.mkdir(parents=True, exist_ok=True)
    return p


def portal_job_dir(dataset_name: str) -> Path:
    """Per-dataset sub-folder under portal_workspace/."""
    return portal_workspace_dir() / dataset_name


# Filename conventions used by the pipeline:
#   stitched raster:  <ds><PRED_SUFFIX>           → e.g. NAGUL_pred.tif
#   vector output:    <ds><PRED_GPKG_SUFFIX>      → e.g. NAGUL_pred.gpkg
# Centralised so every layer of the portal agrees.
PRED_SUFFIX      = "_pred.tif"
PRED_GPKG_SUFFIX = "_pred.gpkg"


def stitched_pred_filename(dataset_name: str) -> str:
    """Canonical filename for a dataset's stitched prediction raster."""
    return f"{dataset_name}{PRED_SUFFIX}"


def gpkg_pred_filename(dataset_name: str) -> str:
    """Canonical filename for a dataset's prediction GeoPackage."""
    return f"{dataset_name}{PRED_GPKG_SUFFIX}"


def portal_job_gpkg(dataset_name: str) -> Path:
    return portal_job_dir(dataset_name) / "outputs" / "gpkg" / gpkg_pred_filename(dataset_name)


def portal_job_stitched(dataset_name: str) -> Path:
    return portal_job_dir(dataset_name) / "outputs" / "stitched" / stitched_pred_filename(dataset_name)


def portal_gpkg_cache_dir() -> Path:
    p = portal_workspace_dir() / "_gpkg_cache"
    p.mkdir(parents=True, exist_ok=True)
    return p


def portal_gt_cache_dir() -> Path:
    """Per-dataset ground-truth GeoJSON cache. Files land at
    `<portal_workspace>/_gt_cache/<DS>/<class>.geojson`."""
    p = portal_workspace_dir() / "_gt_cache"
    p.mkdir(parents=True, exist_ok=True)
    return p


def masks_dir() -> Path:
    """Bundle-side per-tile mask directory, from data_prep.yaml's
    `labels_dir` (falls back to <bundle>/masks for legacy layouts)."""
    paths = data_prep_cfg().get("paths", {})
    return Path(paths.get("labels_dir") or paths.get("masks_dir") or (BUNDLE_ROOT / "masks"))


# ── Source-imagery resolution ───────────────────────────────────────────
# Given just a dataset name, find the COG of the *original* aerial
# imagery (so the map can render the photo underneath the prediction
# layers). We search in three places in order:
#
#   1. Portal job folder              → <portal_workspace>/<DS>/cog/**/<DS>.tif
#   2. Offline bundle COG directory   → <bundle>/cog/**/<DS>.tif
#   3. The portal job's `source.*`    → the raw upload, may not be a COG but
#                                        TiTiler can still serve it (slower).
#   4. data/{train,test} raw rasters  → as a last-ditch fallback.

def _is_prediction(path: Path) -> bool:
    """True if the filename clearly belongs to a prediction artefact."""
    n = path.name.lower()
    return n.endswith("_pred.tif") or n.endswith("_refined.tif")


def find_source_cog(dataset_name: str) -> Path | None:
    """Path to a COG/raster of the **original aerial photo**, or None.

    Crucially this NEVER returns a `<DS>_pred.tif` — prediction rasters
    live under `outputs/stitched/` and are exposed separately. We hard-
    skip anything matching that pattern.
    """
    # 1. portal job COG (convert_to_cog writes <DS>.tif here, never <DS>_pred.tif)
    pj = portal_job_dir(dataset_name) / "cog"
    if pj.exists():
        for p in pj.rglob(f"{dataset_name}.tif"):
            if not _is_prediction(p):
                return p

    # 2. offline COG directory from data_prep.yaml
    cog_root = data_prep_cfg()["paths"].get("cog_dir")
    if cog_root and Path(cog_root).is_dir():
        for p in Path(cog_root).rglob(f"{dataset_name}.tif"):
            if not _is_prediction(p):
                return p

    # 3. portal job source.<ext> (uploaded file)
    pj_src = sorted(portal_job_dir(dataset_name).glob("source.*"))
    for p in pj_src:
        if not _is_prediction(p):
            return p

    # 4. raw data/{train,test} that match by basename stem
    paths = data_prep_cfg()["paths"]
    exts = set(data_prep_cfg().get("raster_extensions", [".tif", ".tiff", ".ecw"]))
    for key in ("train_raw_dir", "test_raw_dir"):
        d = paths.get(key)
        if not d or not Path(d).is_dir():
            continue
        for p in Path(d).rglob("*"):
            if (p.is_file() and p.suffix.lower() in exts
                    and p.stem == dataset_name and not _is_prediction(p)):
                return p

    return None


def class_colors() -> dict[str, str]:
    """layer-name → hex colour. Sourced from data_prep.yaml visualization."""
    return data_prep_cfg().get("visualization", {}).get("class_colors", {})


def feature_classes() -> dict[str, int]:
    return data_prep_cfg().get("feature_classes", {})
