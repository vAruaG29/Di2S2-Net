"""
Trigger a pipeline run for one dataset and stream its events via SSE.

POST /api/inference         { source: "existing"|"upload", dataset_name }
                            → { job_id }
GET  /api/jobs/{id}/events  → SSE stream (event = JSON line)
GET  /api/jobs/{id}         → snapshot of events so far + status
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, HTTPException, Query
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

import shutil

from .. import pipeline_runner
from ..checkpoints import (
    CKPT_SEP,
    effective_name,
    list_checkpoints,
    portal_decoder_type,
    resolve_checkpoint_path,
    split_effective,
)
from ..jobs import REGISTRY, Job
from ..settings import (
    portal_job_dir,
    portal_job_gpkg,
    portal_job_stitched,
)


router = APIRouter(prefix="/api", tags=["inference"])


class InferenceRequest(BaseModel):
    source: str          # "existing" | "upload"
    dataset_name: str    # the BASE dataset name (without any checkpoint suffix)
    force: bool = False  # re-run even if outputs already exist
    # Which checkpoint to run. None / the default checkpoint's id → run
    # under the bare base name; any other id → results land under the
    # namespaced effective name `<base>@@<id>` so they coexist with the
    # default checkpoint's results.
    checkpoint: str | None = None


@router.get("/checkpoints")
def checkpoints() -> dict:
    """Model checkpoints the portal can run (from `<bundle>/pretrained/`).

    Only checkpoints whose decoder architecture matches the pipeline
    (`train.yaml` model.decoder.type) are returned in `checkpoints` — the
    incompatible ones (e.g. a UPerNet checkpoint vs the HRDecoder
    pipeline) would only random-init the seg head and predict garbage, so
    they're kept out of the dropdown. They're still reported under
    `hidden` so the UI can note that something was filtered."""
    allc = list_checkpoints()
    shown = [c for c in allc if c.get("compatible")]
    hidden = [
        {"id": c["id"], "filename": c["filename"], "decoder_type": c["decoder_type"]}
        for c in allc if not c.get("compatible")
    ]
    return {
        "checkpoints": shown,
        "hidden": hidden,
        "expected_decoder": portal_decoder_type(),
        "separator": CKPT_SEP,
    }


def _existing_outputs(name: str) -> dict:
    """Return a snapshot of which portal-job artefacts already exist
    for this dataset. ONLY the per-job workspace under
    `portal_workspace/<name>/` is considered — the bundle's offline
    `outputs/` directory is treated as ephemeral library data that the
    user may have removed, so we never key the dialog logic off it."""
    job_root         = portal_job_dir(name)
    stitched         = portal_job_stitched(name)
    gpkg             = portal_job_gpkg(name)
    metrics_csv      = job_root / "outputs" / "evaluation" / f"{name}_metrics.csv"

    return {
        "dataset_name":  name,
        "job_root":      str(job_root) if job_root.exists() else None,
        "has_stitched":  stitched.exists(),
        "stitched_path": str(stitched) if stitched.exists() else None,
        "has_gpkg":      gpkg.exists(),
        "gpkg_path":     str(gpkg) if gpkg.exists() else None,
        "has_metrics":   metrics_csv.exists(),
        "metrics_path":  str(metrics_csv) if metrics_csv.exists() else None,
    }


@router.get("/inference/preflight/{dataset_name}")
def preflight(
    dataset_name: str,
    checkpoint: str | None = Query(default=None),
) -> dict:
    """
    Check whether a previous portal run already produced outputs for this
    dataset **with this checkpoint**. The frontend hits this before
    kicking off a new run so it can offer "use existing" vs "re-run"
    instead of always re-computing.

    `dataset_name` is the BASE name; the existence check is keyed off the
    *effective* name (`<base>@@<checkpoint-id>` for a non-default
    checkpoint), so each (image, checkpoint) pair is tracked separately.

    The canonical signal is the GeoPackage — the stitched raster and the
    per-tile predictions are deleted after each successful run to save
    disk (see pipeline_runner._cleanup_intermediates), so they will
    typically be missing even on a "complete" portal job.
    """
    name = dataset_name.strip()
    if not name:
        raise HTTPException(400, "dataset_name is required")
    # If a checkpoint-namespaced result (`<base>@@<id>`) is the thing being
    # re-run, recover the true base so the chosen checkpoint re-namespaces
    # cleanly instead of stacking another suffix.
    name, _ = split_effective(name)
    eff = effective_name(name, checkpoint)
    info = _existing_outputs(eff)
    info["outputs_exist"] = info["has_gpkg"]
    return info


@router.post("/inference")
async def start_inference(
    body: InferenceRequest, bg: BackgroundTasks,
) -> dict:
    base = body.dataset_name.strip()
    if not base:
        raise HTTPException(400, "dataset_name is required")
    # Re-running a checkpoint-namespaced result (`<base>@@<id>`) sends the
    # full name back; strip the suffix so we resolve the real raster
    # (named after the base) and re-namespace by the newly-chosen checkpoint
    # rather than appending a second `@@`.
    base, _ = split_effective(base)

    # Effective name = where this (image, checkpoint) pair's outputs live.
    # Bare base for the default checkpoint; `<base>@@<id>` otherwise.
    eff = effective_name(base, body.checkpoint)
    ckpt_path = resolve_checkpoint_path(body.checkpoint)

    # ── Idempotency guard ────────────────────────────────────────────────
    # If a previous portal run already produced the GeoPackage for THIS
    # (image, checkpoint) pair and the caller didn't pass force=true,
    # return the existing outputs and skip the (expensive) re-run. Note
    # we don't require has_stitched because the stitched raster is
    # deleted by _cleanup_intermediates after each successful run; the
    # GPKG is the canonical signal.
    existing = _existing_outputs(eff)
    if not body.force and existing["has_gpkg"]:
        return {
            "status":   "exists",
            "job_id":   None,
            "dataset_name": eff,
            "existing": existing,
        }

    raster_path: Path | None = None
    if body.source == "upload":
        # Locate the raster — uploads land under the BASE upload folder
        # at portal_workspace/<BASE>/source.* (re-running it under a
        # different checkpoint symlinks that same source into the
        # effective job folder).
        job_root = portal_job_dir(base)
        cands = list(job_root.glob("source.*"))
        if not cands:
            raise HTTPException(404, f"Uploaded raster for {base!r} not found")
        raster_path = cands[0]
    elif body.source != "existing":
        raise HTTPException(400, "source must be 'existing' or 'upload'")

    job = REGISTRY.create(dataset_name=eff, source=body.source)
    # Run the pipeline in the event loop (it spawns subprocesses async).
    # `force=True` here propagates all the way into prepare_job_workspace,
    # which wipes the per-job intermediates so the new run cannot reuse
    # anything from a previous portal run or from the bundle's offline
    # outputs/ directories. `base` drives raw-raster discovery (the
    # effective name has the checkpoint suffix and won't match any file
    # on disk); `ckpt_path` is passed to run_pipeline via --checkpoint.
    bg.add_task(
        pipeline_runner.run_for_dataset,
        job, raster_path, body.force, base, ckpt_path,
    )
    return {
        "status":   "queued",
        "job_id":   job.id,
        "dataset_name": eff,
    }


@router.get("/jobs/{job_id}")
def job_snapshot(job_id: str) -> dict:
    job: Job | None = REGISTRY.get(job_id)
    if job is None:
        raise HTTPException(404, "Unknown job")
    return {
        "id": job.id,
        "dataset_name": job.dataset_name,
        "source": job.source,
        "status": job.status,
        "error": job.error,
        "created_at": job.created_at,
        "events": job.events,
    }


@router.post("/jobs/{job_id}/cancel")
def cancel_job(job_id: str) -> dict:
    """
    Ask the running pipeline to stop. Kills the active subprocess and
    flags the job as cancelled. Safe to call on a job that has already
    finished — returns the final state either way.
    """
    job: Job | None = REGISTRY.get(job_id)
    if job is None:
        raise HTTPException(404, "Unknown job")
    if job.status not in ("queued", "running"):
        return {"id": job.id, "status": job.status, "cancelled": False,
                "reason": f"already {job.status}"}
    killed = job.request_cancel()
    return {"id": job.id, "status": job.status, "cancelled": True, "killed_proc": killed}


@router.post("/jobs/{job_id}/cleanup")
def cleanup_job(job_id: str, hard: bool = False) -> dict:
    """
    Delete the per-job workspace folder under `portal_workspace/`.

    By default we only allow cleanup once the job has reached a terminal
    state (cancelled / failed / done). Pass `?hard=true` to also delete a
    workspace that is currently `running` (which also issues a cancel
    first).

    The bundle's offline directories are never touched.
    """
    job: Job | None = REGISTRY.get(job_id)
    if job is None:
        raise HTTPException(404, "Unknown job")

    if job.status == "running":
        if not hard:
            raise HTTPException(
                409,
                "Job is still running. Cancel it first, or pass ?hard=true.",
            )
        job.request_cancel()

    job_root = portal_job_dir(job.dataset_name)
    if not job_root.exists():
        return {"id": job.id, "removed": False, "reason": "workspace not found",
                "path": str(job_root)}

    # Defence-in-depth: refuse to delete anything that isn't inside the
    # portal_workspace tree, in case settings get reconfigured weirdly.
    from ..settings import portal_workspace_dir
    if portal_workspace_dir() not in job_root.parents:
        raise HTTPException(500, f"Refusing to delete {job_root!r}: outside portal_workspace.")

    shutil.rmtree(job_root, ignore_errors=True)
    job.add_event({
        "type": "log",
        "line": f"workspace removed: {job_root}",
        "t_iso": pipeline_runner._now(),
    })
    return {"id": job.id, "removed": True, "path": str(job_root)}


@router.get("/jobs/{job_id}/events")
async def job_events_sse(job_id: str):
    job: Job | None = REGISTRY.get(job_id)
    if job is None:
        raise HTTPException(404, "Unknown job")

    async def gen():
        q = job.subscribe()
        try:
            # Heartbeat so proxies don't close the connection.
            last_beat = asyncio.get_event_loop().time()
            while True:
                try:
                    ev = await asyncio.wait_for(q.get(), timeout=10.0)
                    yield {
                        "event": ev.get("type", "message"),
                        "data": json.dumps(ev),
                    }
                    if ev.get("type") in ("complete", "error", "cancelled"):
                        break
                except asyncio.TimeoutError:
                    yield {"event": "ping", "data": "{}"}
                    last_beat = asyncio.get_event_loop().time()
        finally:
            job.unsubscribe(q)

    return EventSourceResponse(gen())
