"""
Per-job workspace setup.

Each portal run gets a dedicated folder:

    portal_workspace/
    └── <DATASET_NAME>/
        ├── source.<ext>             ← raw raster (uploaded copy or symlink)
        ├── _configs/
        │   ├── data_prep.yaml       ← derived from the bundle's data_prep.yaml
        │   └── train.yaml           ← derived from the bundle's train.yaml
        ├── _meta.json               ← created_at, source, status (optional)
        ├── cog/                     ← convert_to_cog output
        ├── tiles/
        │   ├── <DATASET_NAME>/...
        │   └── tile_index.csv
        ├── masks/                   ← prepare_labels output (often empty)
        └── outputs/
            ├── predictions/<DATASET_NAME>/
            ├── stitched/<DATASET_NAME>_pred.tif
            ├── evaluation/
            ├── gpkg/<DATASET_NAME>_pred.gpkg
            └── logs/

We never touch the bundle's `<bundle>/cog`, `<bundle>/tiles`,
`<bundle>/masks`, `<bundle>/outputs` — those stay reserved for the
offline pipeline.

For each job we *also* synthesise a pair of YAML configs (a data_prep
and a train config) that point at the per-job directories. The
existing pipeline scripts all accept `--config <path>`, so this is a
drop-in redirect.
"""

from __future__ import annotations

import copy
import json
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

import yaml

from .settings import (
    DATA_PREP_CFG,
    TRAIN_CFG,
    data_prep_cfg,
    portal_job_dir,
    train_cfg,
)


@dataclass
class JobPaths:
    dataset_name: str
    root: Path
    source: Path                   # raw raster file
    data_prep_cfg_path: Path
    train_cfg_path: Path
    cog_dir: Path
    tiles_dir: Path
    tile_index: Path
    masks_dir: Path
    output_dir: Path
    stitched_dir: Path
    gpkg_dir: Path
    log_dir: Path


def _mk(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


#: Sub-directories that hold *generated* intermediates — wiped on a
#  force re-run so the next inference cannot reuse stale artefacts from
#  a previous portal run. Anything else under `<job>/` (notably
#  `source.*` and `_meta.json`) is preserved.
_INTERMEDIATE_SUBDIRS = (
    "cog", "tiles", "masks", "outputs",
    "dataset", "visualizations", "checkpoints", "_configs",
)


def prepare_job_workspace(
    dataset_name: str,
    source_raster: Path,
    overwrite: bool = False,
) -> JobPaths:
    """
    Create / refresh the per-job folder layout AND emit two YAML config
    files (data_prep, train) whose paths point inside this job's folder
    instead of the bundle's offline directories.

      source_raster: path to the raw .tif/.ecw to process. If it isn't
                     already inside <job_root>/source.*, we copy or
                     symlink it in.
      overwrite:     when True, wipe every *generated* intermediate dir
                     (cog, tiles, masks, outputs, dataset,
                     visualizations, checkpoints, _configs) before re-
                     laying the skeleton. `source.*` and `_meta.json`
                     are preserved — critical for uploads, where the
                     raw raster already lives inside the job folder and
                     a full rmtree would delete it.
    """
    job_root  = portal_job_dir(dataset_name)
    if overwrite and job_root.exists():
        for sub in _INTERMEDIATE_SUBDIRS:
            p = job_root / sub
            if p.exists():
                shutil.rmtree(p, ignore_errors=True)
    _mk(job_root)

    # ── Mirror the source raster into the job folder ────────────────────
    source_in = job_root / f"source{source_raster.suffix.lower()}"
    if source_in.resolve() != source_raster.resolve():
        if source_in.exists() or source_in.is_symlink():
            source_in.unlink()
        # Symlink to avoid duplicating GB of raster data on local disk.
        try:
            source_in.symlink_to(source_raster.resolve())
        except OSError:
            shutil.copy2(source_raster, source_in)

    # ── Build the directory skeleton ────────────────────────────────────
    cog_dir       = _mk(job_root / "cog")
    tiles_dir     = _mk(job_root / "tiles")
    masks_dir     = _mk(job_root / "masks")
    output_dir    = _mk(job_root / "outputs")
    stitched_dir  = _mk(output_dir / "stitched")
    gpkg_dir      = _mk(output_dir / "gpkg")
    log_dir       = _mk(output_dir / "logs")
    _mk(output_dir / "predictions")
    _mk(output_dir / "evaluation")
    cfg_dir       = _mk(job_root / "_configs")

    tile_index = tiles_dir / "tile_index.csv"

    # ── Derive a data_prep.yaml that points at this job's folder ────────
    dp = copy.deepcopy(data_prep_cfg())
    paths = dp.setdefault("paths", {})
    # Read inputs from the same place as before (raw shp + raster dirs).
    # ONLY the *output* keys are redirected:
    paths["workspace"] = str(job_root)
    paths["cog_dir"]   = str(cog_dir)
    paths["tiles_dir"] = str(tiles_dir)
    paths["outputs_dir"] = str(output_dir)
    paths["visualizations_dir"] = str(job_root / "visualizations")
    # `labels_dir` lives in train.yaml conceptually, but prepare_labels
    # uses data_prep.yaml — surface it here so masks land at
    #   <job>/masks/<DS>/<tile>_mask.tif
    # which is exactly where run_pipeline.discover_datasets and
    # evaluate_tile_predictions look. Without this, prepare_labels
    # falls back to writing beside the tiles, evaluation sees 0
    # labeled tiles, the dataset is classified `test`, and metrics
    # are silently skipped (this was the NAGUL bug).
    paths["labels_dir"] = str(masks_dir)
    # `dataset_dir` is used by convert_to_cog for the symlink mirror —
    # point it at the job folder too so symlinks land inside.
    paths["dataset_dir"] = str(job_root / "dataset")
    _mk(Path(paths["dataset_dir"]))
    dp_path = cfg_dir / "data_prep.yaml"
    dp_path.write_text(yaml.safe_dump(dp, sort_keys=False))

    # ── Derive a train.yaml that points at this job's folder ────────────
    tr = copy.deepcopy(train_cfg())
    tpaths = tr.setdefault("paths", {})
    tpaths["workspace"]      = str(job_root)
    tpaths["tiles_dir"]      = str(tiles_dir)
    tpaths["tile_index"]     = str(tile_index)
    tpaths["labels_dir"]     = str(masks_dir)
    tpaths["output_dir"]     = str(output_dir)
    tpaths["checkpoint_dir"] = str(_mk(job_root / "checkpoints"))
    tpaths["vis_dir"]        = str(_mk(job_root / "visualizations"))
    tpaths["log_dir"]        = str(log_dir)
    # Keep label_sources pointing at the bundle's raw shapefiles (read-only).
    tr_path = cfg_dir / "train.yaml"
    tr_path.write_text(yaml.safe_dump(tr, sort_keys=False))

    # ── Metadata blob (useful for the UI later) ─────────────────────────
    meta_path = job_root / "_meta.json"
    meta = {
        "dataset_name": dataset_name,
        "source": str(source_raster),
        "created_at": time.time(),
    }
    if meta_path.exists():
        try:
            meta.update(json.loads(meta_path.read_text()))
        except Exception:
            pass
    meta_path.write_text(json.dumps(meta, indent=2))

    return JobPaths(
        dataset_name=dataset_name,
        root=job_root,
        source=source_in,
        data_prep_cfg_path=dp_path,
        train_cfg_path=tr_path,
        cog_dir=cog_dir,
        tiles_dir=tiles_dir,
        tile_index=tile_index,
        masks_dir=masks_dir,
        output_dir=output_dir,
        stitched_dir=stitched_dir,
        gpkg_dir=gpkg_dir,
        log_dir=log_dir,
    )


def existing_job(dataset_name: str) -> JobPaths | None:
    """Return JobPaths for an already-prepared job, or None."""
    job_root = portal_job_dir(dataset_name)
    cfg_dir = job_root / "_configs"
    dp = cfg_dir / "data_prep.yaml"
    tr = cfg_dir / "train.yaml"
    if not (dp.exists() and tr.exists()):
        return None
    sources = list(job_root.glob("source.*"))
    return JobPaths(
        dataset_name=dataset_name,
        root=job_root,
        source=sources[0] if sources else job_root / "source.tif",
        data_prep_cfg_path=dp,
        train_cfg_path=tr,
        cog_dir=job_root / "cog",
        tiles_dir=job_root / "tiles",
        tile_index=job_root / "tiles" / "tile_index.csv",
        masks_dir=job_root / "masks",
        output_dir=job_root / "outputs",
        stitched_dir=job_root / "outputs" / "stitched",
        gpkg_dir=job_root / "outputs" / "gpkg",
        log_dir=job_root / "outputs" / "logs",
    )


def list_portal_jobs() -> list[str]:
    """Names of every prepared portal job (dataset folder under portal_workspace)."""
    from .settings import portal_workspace_dir
    root = portal_workspace_dir()
    if not root.exists():
        return []
    out = []
    for p in sorted(root.iterdir()):
        if p.is_dir() and not p.name.startswith("_"):
            out.append(p.name)
    return out
