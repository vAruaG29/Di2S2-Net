"""
Run the existing CLI pipeline (data_prep + inference) as a subprocess
for a single dataset, parse its stdout, and feed structured events
into a Job (jobs.py) so the SSE endpoint can stream them to the UI.

The CLI emits StepTimer lines like:

    ⏱  [HH:MM:SS] START  some/label
    ⏱  [HH:MM:SS] DONE   some/label  (12.34s)
    ⏱  [HH:MM:SS] cumulative foo: 1.23s over 4 call(s)

We extract those, plus a coarse "phase" marker so the UI can group
sub-steps under the five top-level stages.
"""

from __future__ import annotations

import asyncio
import os
import re
import shutil
import sys
import time
from pathlib import Path

from . import gpkg_reader
from .job_paths import JobPaths, prepare_job_workspace
from .jobs import Job
from .settings import BUNDLE_ROOT, data_prep_cfg, gpkg_pred_filename


_TIMING_LINE = re.compile(
    r"⏱\s+\[(\d{2}:\d{2}:\d{2})\]\s+"
    r"(START|DONE|cumulative)\s+"
    r"(?P<label>.+?)"
    r"(?:\s+\(([0-9.]+)s\))?"
    r"\s*$"
)

PHASES = [
    ("convert_to_cog",          "1/5 Convert to COG"),
    ("tile_raster",             "2/5 Tile rasters"),
    ("prepare_labels",          "3/5 Prepare labels"),
    ("run_pipeline",            "4/5 Inference + stitch + eval"),
    ("batch_stitched_to_gpkg",  "5/5 Vectorise to GeoPackage"),
]


def _now() -> str:
    return time.strftime("%H:%M:%S")


def _push_phase(job: Job, idx: int, status: str, label: str) -> None:
    job.add_event({
        "type": "phase",
        "index": idx,
        "total": len(PHASES),
        "status": status,            # "start" | "done" | "failed" | "skipped"
        "label": label,
        "t_iso": _now(),
    })


def _push_step(job: Job, raw: str, ts: str, action: str,
               label: str, dt: float | None) -> None:
    job.add_event({
        "type": "step",
        "action": action,            # "START" | "DONE" | "cumulative"
        "label": label,
        "t_iso": ts,
        "elapsed_s": dt,
        "raw": raw.rstrip(),
    })


def _push_log(job: Job, line: str) -> None:
    job.add_event({
        "type": "log",
        "line": line.rstrip(),
        "t_iso": _now(),
    })


class _Cancelled(Exception):
    """Raised internally when the user has requested the job stop."""


async def _stream(job: Job, cmd: list[str], phase_idx: int,
                  phase_label: str) -> int:
    """Run one subprocess phase. Stream stdout into the job."""
    _push_phase(job, phase_idx, "start", phase_label)

    # If the user already cancelled before we even started this phase,
    # short-circuit straight to the cancelled state.
    if job.cancel_requested:
        _push_phase(job, phase_idx, "skipped", phase_label)
        raise _Cancelled()

    proc = await asyncio.create_subprocess_exec(
        *cmd,
        cwd=str(BUNDLE_ROOT),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )
    assert proc.stdout is not None
    job.current_proc = proc

    try:
        while True:
            chunk = await proc.stdout.readline()
            if not chunk:
                break
            line = chunk.decode("utf-8", errors="replace")
            m = _TIMING_LINE.search(line)
            if m:
                ts, action, label, dt = m.group(1), m.group(2), m.group("label"), m.group(4)
                _push_step(
                    job, line, ts, action,
                    label.strip(),
                    float(dt) if dt else None,
                )
            else:
                if line.strip():
                    _push_log(job, line)

        rc = await proc.wait()
    finally:
        # Always drop the reference, even if we crashed mid-stream.
        if job.current_proc is proc:
            job.current_proc = None

    # Distinguish a user-requested kill from an organic failure.
    if job.cancel_requested:
        _push_phase(job, phase_idx, "skipped", phase_label)
        raise _Cancelled()

    _push_phase(
        job, phase_idx,
        "done" if rc == 0 else "failed",
        phase_label,
    )
    return rc


def _python() -> str:
    """Use whichever Python interpreter is running the portal."""
    return sys.executable or "python"


def _dir_size_bytes(p: Path) -> int:
    """Total on-disk size of a directory tree, in bytes. Best-effort —
    we silently skip anything we can't stat (broken symlinks, permission
    issues) so the cleanup never crashes the job."""
    if not p.exists():
        return 0
    total = 0
    for f in p.rglob("*"):
        try:
            if f.is_file() and not f.is_symlink():
                total += f.stat().st_size
        except OSError:
            pass
    return total


def _fmt_bytes(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{n} B"
        n /= 1024
    return f"{n:.1f} GB"


def _cleanup_intermediates(job: Job, job_paths: JobPaths) -> None:
    """
    Reclaim disk after a successful run.

    Once the GeoPackage has been written we no longer need:
      - <job>/tiles/                    (raster tiles + tile_index.csv)
      - <job>/masks/                    (per-tile ground-truth masks)
      - <job>/outputs/predictions/      (per-tile prediction TIFs)
      - <job>/outputs/stitched/         (full-extent prediction raster)

    These can each be multi-GB and are not consumed by the portal at
    *display* time — the map shows the aerial COG (kept) plus the GPKG
    (kept). The metrics CSV (kept) and the source raster symlink (kept)
    round out the persisted state.

    The preflight check in routes/inference.py keys off the GPKG, so
    deleting `stitched/` doesn't make the dataset look "unprocessed".
    """
    targets = [
        job_paths.tiles_dir,
        job_paths.masks_dir,
        job_paths.output_dir / "predictions",
        job_paths.stitched_dir,
    ]
    freed = 0
    for p in targets:
        if not p.exists():
            continue
        freed += _dir_size_bytes(p)
        shutil.rmtree(p, ignore_errors=True)
    if freed:
        _push_log(job, f"  🧹 cleaned intermediates: freed {_fmt_bytes(freed)} "
                       f"(tiles, masks, per-tile predictions, stitched raster)")


def _find_raster_for_dataset(dataset_name: str) -> Path | None:
    """
    Locate the raw raster file whose basename stem == dataset_name.
    Looks in both train_raw_dir and test_raw_dir from data_prep.yaml.
    Returns None if not found (caller should fall back to dataset-name filter).
    """
    cfg = data_prep_cfg()
    paths = cfg["paths"]
    exts = cfg.get("raster_extensions", [".tif", ".tiff", ".ecw"])
    for root_key in ("train_raw_dir", "test_raw_dir"):
        root = paths.get(root_key)
        if not root or not Path(root).is_dir():
            continue
        for p in Path(root).rglob("*"):
            if p.is_file() and p.suffix.lower() in exts and p.stem == dataset_name:
                return p
    return None


def _write_one_dataset_file_list(dataset_name: str) -> Path:
    """Tempfile holding one dataset name per line — for batch_stitched_to_gpkg."""
    import tempfile
    fd, path = tempfile.mkstemp(prefix=f"dataset_list_{dataset_name}_", suffix=".txt")
    with os.fdopen(fd, "w") as f:
        f.write(dataset_name + "\n")
    return Path(path)


async def run_for_dataset(
    job: Job,
    raster_path: Path | None = None,
    force: bool = False,
    base_name: str | None = None,
    checkpoint_path: Path | None = None,
) -> None:
    """
    Drive the full pipeline for ONE dataset, writing **every artefact**
    under `<portal_workspace>/<dataset_name>/`. The bundle's offline
    directories (`<bundle>/cog`, `/tiles`, `/masks`, `/outputs`) are
    never written to AND never read from for generated artefacts. Reads
    from outside `portal_workspace/` are limited to:

      - the raw source raster (symlinked in as `<job>/source.<ext>`),
      - ground-truth shapefiles declared in `data_prep.yaml`
        (input data — can't be regenerated),
      - the pretrained model checkpoint under `<bundle>/pretrained/`.

      raster_path:     optional. If omitted, we try to locate a raw
                       raster in the bundle whose stem matches the
                       *base* dataset name (so the user can "re-run" an
                       offline dataset through the portal without
                       overwriting anything).
      force:           when True, the per-job folder's intermediate
                       subdirs (cog/, tiles/, masks/, outputs/, …) are
                       wiped first so the run starts from a clean slate
                       and no cached intermediate from a previous portal
                       run is reused. `source.*` is preserved.
      base_name:       the un-suffixed dataset name. `job.dataset_name`
                       may carry a `@@<checkpoint-id>` suffix (so results
                       from different checkpoints don't collide); the raw
                       raster on disk is named after the base, so raster
                       discovery keys off this. Defaults to the job name.
      checkpoint_path: which `.ckpt` to run. When None, run_pipeline
                       falls back to its own DEFAULT_CHECKPOINT.
    """
    job.status = "running"
    py = _python()
    ds = job.dataset_name                     # effective name (may be suffixed)
    base = base_name or ds                    # raw raster is named after this

    # Find the source raster if not supplied. Match on the BASE name —
    # the effective name's `@@<id>` suffix never appears on disk.
    if raster_path is None:
        raster_path = _find_raster_for_dataset(base)
    if raster_path is None or not Path(raster_path).exists():
        msg = f"No raw raster found for dataset {base!r}. Upload one or pick an existing image."
        job.status = "failed"
        job.error = msg
        job.add_event({"type": "error", "message": msg, "t_iso": _now()})
        return

    # Prepare the per-job workspace + its private YAML configs.
    # On a forced re-run we wipe every intermediate subdir first so no
    # cached COG / tiles / masks / predictions / stitched / gpkg from a
    # previous portal run can be picked up by the subprocesses below.
    job_paths: JobPaths = prepare_job_workspace(ds, Path(raster_path), overwrite=force)
    if force:
        _push_log(job, f"  ↻ force re-run: cleared cog/tiles/masks/outputs in {job_paths.root}")
    _push_log(job, f"  Job workspace: {job_paths.root}")
    _push_log(job, f"  Configs:        {job_paths.data_prep_cfg_path.name}, "
                   f"{job_paths.train_cfg_path.name}")

    dp_cfg = str(job_paths.data_prep_cfg_path)
    tr_cfg = str(job_paths.train_cfg_path)

    # Tempfile of dataset names for batch_stitched_to_gpkg.
    file_list = _write_one_dataset_file_list(ds)

    # ── Preflight: confirm the checkpoint exists before we burn 30
    #   minutes converting + tiling, only to crash at the model-load
    #   step. The chosen checkpoint (or the portal default) ships at
    #   <bundle>/pretrained/.
    ckpt = Path(checkpoint_path) if checkpoint_path else (
        BUNDLE_ROOT / "pretrained" / "dinov3_hrdecoder_full_best_loss=0.0615.ckpt"
    )
    if not ckpt.exists():
        msg = (
            f"Model checkpoint not found:\n"
            f"  {ckpt}\n"
            f"Drop the .ckpt file at that path (or symlink it) and retry.\n"
            f"See pretrained/README.md for the expected file names."
        )
        job.status = "failed"
        job.error = msg
        job.add_event({"type": "error", "message": msg, "t_iso": _now()})
        return
    _push_log(job, f"  Checkpoint:     {ckpt.name}")

    try:
        # ── 1) COG conversion — scoped to this raster, written into
        #      <job>/cog/ via the per-job data_prep.yaml. We pass
        #      --dataset-name so the output file is named after the
        #      portal's dataset, matching what tile_raster and
        #      run_pipeline expect later.
        rc = await _stream(
            job,
            [py, "-m", "dinov3_hrdecoder_pipeline.data_prep.convert_to_cog",
             "--config", dp_cfg,
             "--input",  str(job_paths.source),
             "--dataset-name", ds],
            phase_idx=0, phase_label=PHASES[0][1],
        )
        if rc != 0:
            raise RuntimeError(f"convert_to_cog failed (rc={rc})")

        # ── 2) Tile — feed the REPROJECTED COG from step 1, not the raw
        #      source. `convert_to_cog` reprojects EPSG:4326 inputs to the
        #      correct UTM zone; if we tile the raw source instead, the
        #      tiles inherit EPSG:4326 (degrees) and every downstream
        #      step that depends on metric units silently breaks. Most
        #      visible failure: batch_stitched_to_gpkg's --min-area 5.0
        #      is then "5 square degrees" (~60,000 km²) so every polygon
        #      gets dissolved away and the GPKG ends up with only the
        #      Utility (point) layer. Discover the COG explicitly under
        #      <job>/cog/<split>/<region>/<ds>.tif (convert_to_cog uses
        #      classify_region which we can't easily import here), then
        #      fall back to source.tif if no COG was produced.
        cog_candidates = sorted(job_paths.cog_dir.rglob(f"{ds}.tif"))
        tile_input = str(cog_candidates[0]) if cog_candidates else str(job_paths.source)
        if cog_candidates:
            _push_log(job, f"  tiling from COG: {tile_input}")
        else:
            _push_log(job, f"  no COG found under {job_paths.cog_dir}, "
                           f"falling back to raw source (CRS may be wrong)")
        rc = await _stream(
            job,
            [py, "-m", "dinov3_hrdecoder_pipeline.data_prep.tile_raster",
             "--config", dp_cfg,
             "--input",  tile_input,
             "--dataset-name", ds],
            phase_idx=1, phase_label=PHASES[1][1],
        )
        if rc != 0:
            raise RuntimeError(f"tile_raster failed (rc={rc})")

        # ── 3) Labels — soft-fail if no shapefiles match this dataset.
        rc = await _stream(
            job,
            [py, "-m", "dinov3_hrdecoder_pipeline.data_prep.prepare_labels",
             "--config", dp_cfg,
             "--mode",   "multiclass",
             "--dataset", ds],
            phase_idx=2, phase_label=PHASES[2][1],
        )
        if rc != 0:
            _push_log(job, f"  prepare_labels rc={rc} — continuing without GT.")

        # Count mask files actually produced — if 0 the dataset will be
        # classified as 'test' in run_pipeline and evaluation is skipped.
        # This is the most common reason metrics don't appear for a
        # training dataset like NAGUL.
        masks_for_ds = job_paths.masks_dir / ds
        n_masks = len(list(masks_for_ds.glob("*_mask.tif"))) if masks_for_ds.exists() else 0
        if n_masks > 0:
            _push_log(job, f"  ✓ {n_masks} mask file(s) for {ds!r} — metrics WILL be computed.")
        else:
            _push_log(job, f"  ⚠ 0 mask files for {ds!r} at {masks_for_ds} — metrics will be skipped (treated as test dataset).")

        # ── 4) Inference + stitch + evaluate — uses the per-job train.yaml
        #      so predictions/, stitched/, evaluation/ all land in <job>/outputs.
        #      `--checkpoint` selects the model; without it run_pipeline
        #      falls back to its own DEFAULT_CHECKPOINT.
        run_cmd = [
            py, "-m", "dinov3_hrdecoder_pipeline.inference.run_pipeline",
            "--config", tr_cfg,
            "--datasets", ds,
            "--output-dir", str(job_paths.output_dir),
            "--force",
        ]
        if checkpoint_path:
            run_cmd += ["--checkpoint", str(checkpoint_path)]
        rc = await _stream(
            job, run_cmd,
            phase_idx=3, phase_label=PHASES[3][1],
        )
        if rc != 0:
            raise RuntimeError(f"run_pipeline failed (rc={rc})")

        # ── 5) Vectorise to GPKG — writes <job>/outputs/gpkg/<DS>_pred.gpkg.
        # Speed: parallel per-class workers + 2× downsample + simplify
        # + min-area filter inside each worker. See predictions_to_gpkg
        # for the rationale of each flag.
        n_classes_max = 6        # we have 6 non-background classes
        cpu_workers = max(1, min(n_classes_max, (os.cpu_count() or 4) // 2))
        rc = await _stream(
            job,
            [py, "-m", "dinov3_hrdecoder_pipeline.inference.batch_stitched_to_gpkg",
             "--config", tr_cfg,
             "--file-list",   str(file_list),
             "--stitched-dir", str(job_paths.stitched_dir),
             "--output-dir",   str(job_paths.gpkg_dir),
             "--workers",    str(cpu_workers),
             "--downsample", "2",
             "--simplify",   "0.5",   # 0.5 m simplification — invisible at city zoom
             "--min-area",   "5.0"],  # drop polygons smaller than 5 m²
            phase_idx=4, phase_label=PHASES[4][1],
        )
        if rc != 0:
            raise RuntimeError(
                "Vectorisation produced no GeoPackage — the model's "
                "predictions for this checkpoint may be empty or entirely "
                "below the --min-area threshold. The stitched raster and "
                "per-tile predictions have been kept under "
                f"{job_paths.output_dir} so you can inspect them or "
                "re-vectorise with a smaller --min-area."
            )

        # ── 5b) Guard before reclaiming disk. batch_stitched_to_gpkg can
        #      finish "cleanly" while writing a GeoPackage that has no
        #      usable vector layers (only QGIS style tables, or zero
        #      features after filtering). If we cleaned up in that case
        #      the dataset would look done in the gallery yet show no
        #      predictions AND have no intermediates left to re-vectorise.
        #      Verify ≥1 real layer landed; otherwise fail loudly and KEEP
        #      the stitched raster + predictions for inspection.
        gpkg_out = job_paths.gpkg_dir / gpkg_pred_filename(ds)
        n_layers = len(gpkg_reader.list_layers_at(gpkg_out)) if gpkg_out.exists() else 0
        if n_layers == 0:
            raise RuntimeError(
                f"Vectorisation wrote no usable layers for {ds!r} "
                f"(GeoPackage {'missing' if not gpkg_out.exists() else 'has 0 data layers'}). "
                f"Predictions may be empty for this checkpoint. The stitched "
                f"raster + per-tile predictions are kept under "
                f"{job_paths.output_dir} for inspection / re-vectorising."
            )
        _push_log(job, f"  ✓ GeoPackage has {n_layers} layer(s): {gpkg_out.name}")

        # ── 6) Reclaim disk: the per-tile predictions + the stitched
        #      raster + the tile pyramid + the masks have all served
        #      their purpose now that the GPKG is on disk. Saves several
        #      GB per dataset for the typical aerial run.
        _cleanup_intermediates(job, job_paths)

        job.status = "done"
        job.add_event({
            "type": "complete",
            "dataset_name": ds,
            "job_root": str(job_paths.root),
            "t_iso": _now(),
        })

    except _Cancelled:
        job.status = "cancelled"
        job.add_event({
            "type": "cancelled",
            "dataset_name": ds,
            "job_root": str(job_paths.root),
            "t_iso": _now(),
        })

    except Exception as exc:
        # If we got here because the subprocess was killed by `cancel`,
        # report as cancelled rather than a generic failure.
        if job.cancel_requested:
            job.status = "cancelled"
            job.add_event({
                "type": "cancelled",
                "dataset_name": ds,
                "job_root": str(job_paths.root),
                "t_iso": _now(),
            })
        else:
            job.status = "failed"
            job.error = str(exc)
            job.add_event({
                "type": "error",
                "message": str(exc),
                "t_iso": _now(),
            })
    finally:
        try:
            os.unlink(file_list)
        except OSError:
            pass
