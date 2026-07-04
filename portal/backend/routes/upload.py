"""
Raster ingestion.

Three ways to get a raster into the portal — pick the fastest one for
your situation:

  • POST /api/uploads             ← multipart upload (browser → server).
                                    Slow on multi-GB files: every byte
                                    travels over the network.

  • GET  /api/server-files        ← list rasters that already exist on
                                    the server filesystem.

  • POST /api/server-pick         ← register an already-on-disk file as
                                    a portal job (symlink, no copy). Use
                                    this when you have already `scp`'d
                                    or `rsync`'d the file to the server,
                                    or it lives in <bundle>/data/.

All three put the dataset behind the SAME flow downstream — convert →
tile → infer → stitch → gpkg.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import uuid
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from pydantic import BaseModel

from .. import gpkg_reader
from ..settings import (
    BUNDLE_ROOT,
    data_prep_cfg,
    gpkg_pred_filename,
    portal_job_dir,
    portal_job_gpkg,
    portal_job_stitched,
    portal_workspace_dir,
)


router = APIRouter(prefix="/api", tags=["uploads"])

_SAFE_NAME = re.compile(r"[^A-Za-z0-9._-]+")
_ALLOWED_EXT = {".tif", ".tiff", ".ecw"}
_CHUNK = 1024 * 1024  # 1 MiB


def _safe_stem(name: str) -> str:
    """Sanitised filename stem (no whitespace, no path separators).
    Empty input falls back to `"raster"`."""
    return _SAFE_NAME.sub("_", Path(name).stem).strip("_") or "raster"


def _unique_upload_name(stem: str) -> str:
    """Return a fresh portal-job name based on the user's original
    filename stem.  Pattern: `<stem>_uploaded`, `<stem>_uploaded_2`,
    `<stem>_uploaded_3`, … — picks the first that doesn't already
    exist in `portal_workspace/`.

    Keeps the human-readable name from the user's file, only resorting
    to a numeric suffix on actual collision (so a second upload of
    `survey_a.tif` becomes `survey_a_uploaded_2`)."""
    base = f"{stem}_uploaded"
    if not portal_job_dir(base).exists():
        return base
    n = 2
    while portal_job_dir(f"{base}_{n}").exists():
        n += 1
    return f"{base}_{n}"


def _write_upload_meta(job_root: Path, original_filename: str) -> None:
    """Drop a `_meta.json` marker into the job folder so the
    `/api/uploads/list` endpoint can tell this folder apart from
    other portal jobs (Train/Test re-runs etc.) regardless of the
    folder name."""
    meta = {
        "source_kind":       "upload",
        "original_filename": original_filename,
        # `created_at` is also derivable from source.* mtime, but
        # writing it explicitly makes the list endpoint trivial.
    }
    try:
        (job_root / "_meta.json").write_text(json.dumps(meta, indent=2))
    except OSError:
        pass


@router.post("/uploads")
async def upload_raster(file: UploadFile = File(...)) -> dict:
    """Stream-upload a raster via multipart form-data. Slow on huge
    files — prefer `/api/server-pick` if the file is already on the
    server."""
    if not file.filename:
        raise HTTPException(400, "Missing filename")

    ext = Path(file.filename).suffix.lower()
    if ext not in _ALLOWED_EXT:
        raise HTTPException(
            415,
            f"Unsupported extension {ext!r}. Allowed: {sorted(_ALLOWED_EXT)}",
        )

    # Keep the user's original filename stem (sanitised) — the
    # numeric suffix only kicks in if the same name is uploaded
    # twice. No more `upload_<random-id>_<stem>` opaque names.
    dataset_name = _unique_upload_name(_safe_stem(file.filename))

    job_root = portal_job_dir(dataset_name)
    job_root.mkdir(parents=True, exist_ok=True)
    dst = job_root / f"source{ext}"

    written = 0
    with open(dst, "wb") as f:
        while True:
            chunk = await file.read(_CHUNK)
            if not chunk:
                break
            f.write(chunk)
            written += len(chunk)

    _write_upload_meta(job_root, file.filename)

    return {
        "dataset_name": dataset_name,
        "saved_to": str(dst),
        "bytes": written,
    }


# ─────────────────────────────────────────────────────────────────────
# Server-side file picker — no network transfer required.
# ─────────────────────────────────────────────────────────────────────

def _allowed_browse_roots() -> list[Path]:
    """Directories the server-file API is allowed to list / pick from.

    Defaults to the bundle's `data/` (raw inputs) plus the configured
    `train_raw_dir` and `test_raw_dir`. Refusing to look anywhere else
    keeps the endpoint from turning into a generic filesystem browser.
    """
    roots: list[Path] = []
    cfg = data_prep_cfg().get("paths", {})
    for key in ("train_raw_dir", "test_raw_dir"):
        p = cfg.get(key)
        if p:
            roots.append(Path(p))
    roots.append(BUNDLE_ROOT / "data")
    # De-dup + only keep existing dirs
    seen: dict[str, Path] = {}
    for r in roots:
        r = r.resolve()
        if r.is_dir() and str(r) not in seen:
            seen[str(r)] = r
    return list(seen.values())


def _is_safe_path(target: Path) -> bool:
    """True iff `target` resolves to a file *under one of* the allowed roots."""
    try:
        t = target.resolve()
    except (OSError, RuntimeError):
        return False
    if not t.is_file():
        return False
    for root in _allowed_browse_roots():
        try:
            t.relative_to(root)
            return True
        except ValueError:
            continue
    return False


@router.get("/uploads/list")
def list_uploads() -> dict:
    """Every upload-origin job sitting in `portal_workspace/`.

    An entry counts as an upload if EITHER:
      - it has a `_meta.json` with `"source_kind": "upload"` (new
        naming: `<stem>_uploaded[_N]`), or
      - its folder name starts with the legacy `upload_<id>_…`
        prefix (kept so previously-uploaded files still appear after
        the rename).

    Returns one row per upload with display name, source path, byte
    size, completion state, and a `created_at` timestamp for sorting.
    """
    root = portal_workspace_dir()
    if not root.exists():
        return {"uploads": []}

    out: list[dict] = []
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        if child.name.startswith("_"):                  # _gpkg_cache, _gt_cache, …
            continue

        # Detect upload via _meta.json marker (preferred) OR the
        # legacy folder-name prefix (backwards-compat).
        meta_p = child / "_meta.json"
        meta: dict = {}
        if meta_p.exists():
            try:
                meta = json.loads(meta_p.read_text()) or {}
            except Exception:
                meta = {}
        is_upload = (meta.get("source_kind") == "upload"
                     or child.name.startswith("upload_")
                     or child.name.endswith("_uploaded")
                     or re.search(r"_uploaded_\d+$", child.name) is not None)
        if not is_upload:
            continue

        srcs = sorted(child.glob("source.*"))
        if not srcs:
            continue
        src = srcs[0]
        try:
            stat = src.stat()
            size_bytes = stat.st_size
            created_at = stat.st_mtime
        except OSError:
            size_bytes, created_at = 0, 0.0

        # Display name: prefer the original filename recorded in
        # `_meta.json`; otherwise strip the trailing `_uploaded[_N]`
        # (new naming) or `upload_<id>_` prefix (legacy).
        display = meta.get("original_filename")
        if display:
            display = Path(display).stem
        else:
            m_new = re.match(r"^(.+?)_uploaded(?:_\d+)?$", child.name)
            m_old = re.match(r"^upload_[0-9a-fA-F]{6,}_(.+)$", child.name)
            display = (m_new.group(1) if m_new
                       else m_old.group(1) if m_old
                       else child.name)

        gpkg = portal_job_gpkg(child.name)
        stitched = portal_job_stitched(child.name)

        # `has_outputs` must match what the user actually gets when
        # they click the row.  An empty-but-present GPKG (zero
        # readable layers — happens on partial / failed inference
        # runs) used to say "ready" here while ClassesFloat then
        # said "no GeoPackage layers". Now we count the layers and
        # only mark as ready if there's something to show.
        layer_count = 0
        if gpkg.exists():
            try:
                layer_count = len(gpkg_reader.list_layers_at(gpkg))
            except Exception:
                layer_count = 0

        out.append({
            "dataset_name": child.name,
            "display_name": display,
            "source":       str(src),
            "bytes":        size_bytes,
            "has_outputs":  layer_count > 0,
            "has_stitched": stitched.exists(),
            "created_at":   created_at,
        })

    # Newest first — what the user most likely wants to revisit.
    out.sort(key=lambda r: r["created_at"], reverse=True)
    return {"uploads": out}


@router.delete("/uploads/list/{dataset_name}")
def delete_upload(dataset_name: str) -> dict:
    """Remove an upload-origin job folder from `portal_workspace/`.

    Accepts BOTH the legacy `upload_<id>_<stem>` naming and the new
    `<stem>_uploaded[_N]` naming, plus the `_meta.json` marker — same
    detection rules as `list_uploads`.  Refuses anything else (a
    Train/Test re-run, for example) to avoid foot-guns from the
    UploadPane's per-row trash button.
    """
    job_root = portal_job_dir(dataset_name)
    if not job_root.exists():
        raise HTTPException(404, f"No portal job directory for {dataset_name!r}")

    # Detect upload via meta marker OR naming pattern.
    meta_kind: str | None = None
    meta_p = job_root / "_meta.json"
    if meta_p.exists():
        try:
            meta_kind = (json.loads(meta_p.read_text()) or {}).get("source_kind")
        except Exception:
            meta_kind = None
    looks_uploaded = (
        meta_kind == "upload"
        or dataset_name.startswith("upload_")
        or dataset_name.endswith("_uploaded")
        or re.search(r"_uploaded_\d+$", dataset_name) is not None
    )
    if not looks_uploaded:
        raise HTTPException(
            400,
            f"{dataset_name!r} isn't an upload-origin job — refusing to delete.",
        )

    # Defence-in-depth: never touch anything outside portal_workspace.
    if portal_workspace_dir() not in job_root.parents:
        raise HTTPException(500, f"Refusing to delete {job_root!r}: outside portal_workspace.")
    shutil.rmtree(job_root, ignore_errors=True)
    return {"removed": True, "path": str(job_root)}


@router.get("/server-files")
def list_server_files(max_depth: int = 4) -> dict:
    """List rasters already on the server filesystem, grouped by root.

    No bytes transferred — the frontend just shows the result as a
    picker. Use `/api/server-pick` to register a chosen file as a
    portal job.
    """
    out: list[dict] = []
    for root in _allowed_browse_roots():
        items = []
        # Bounded depth so we don't accidentally walk a huge tree.
        for path in _iter_rasters(root, max_depth=max_depth):
            try:
                st = path.stat()
            except OSError:
                continue
            items.append({
                "path":      str(path),
                "name":      path.name,
                "rel":       str(path.relative_to(root)),
                "size_bytes": st.st_size,
                "size_mb":   round(st.st_size / (1024 * 1024), 1),
                "ext":       path.suffix.lower(),
            })
        items.sort(key=lambda x: x["rel"].lower())
        out.append({
            "root":  str(root),
            "label": root.name,
            "count": len(items),
            "files": items,
        })
    return {"roots": out, "allowed_extensions": sorted(_ALLOWED_EXT)}


def _iter_rasters(root: Path, max_depth: int):
    """Generator over raster files under `root` up to `max_depth` deep."""
    root_depth = len(root.parts)
    for dirpath, _dirnames, filenames in os.walk(root):
        depth = len(Path(dirpath).parts) - root_depth
        if depth > max_depth:
            continue
        for fn in filenames:
            ext = Path(fn).suffix.lower()
            if ext in _ALLOWED_EXT:
                yield Path(dirpath) / fn


class ServerPickRequest(BaseModel):
    server_path: str
    # Optional override; defaults to the file's basename stem.
    dataset_name: str | None = None


@router.post("/server-pick")
def pick_server_file(body: ServerPickRequest) -> dict:
    """Register an already-on-disk raster as a portal job.

    Creates `portal_workspace/<dataset_name>/source.<ext>` as a symlink
    to the chosen file. Zero bytes copied. Subsequent flow is identical
    to a normal upload.
    """
    src = Path(body.server_path).expanduser()
    if not src.is_absolute():
        # Resolve relative paths against the bundle root for safety.
        src = (BUNDLE_ROOT / src).resolve()

    if not _is_safe_path(src):
        allowed = ", ".join(str(r) for r in _allowed_browse_roots())
        raise HTTPException(
            403,
            f"Refusing to pick {src!s}. Allowed roots: {allowed}",
        )

    ext = src.suffix.lower()
    if ext not in _ALLOWED_EXT:
        raise HTTPException(
            415,
            f"Unsupported extension {ext!r}. Allowed: {sorted(_ALLOWED_EXT)}",
        )

    # dataset_name: explicit > basename stem (sanitised).
    if body.dataset_name:
        stem = _SAFE_NAME.sub("_", body.dataset_name.strip()).strip("_")
    else:
        stem = _SAFE_NAME.sub("_", src.stem).strip("_")
    if not stem:
        stem = "raster"

    job_root = portal_job_dir(stem)
    job_root.mkdir(parents=True, exist_ok=True)
    link = job_root / f"source{ext}"

    # Remove a stale entry (file OR symlink), then symlink fresh.
    if link.exists() or link.is_symlink():
        link.unlink()
    try:
        link.symlink_to(src.resolve())
        linked = True
    except OSError:
        # Filesystem doesn't allow symlinks — fall back to a hard link.
        # (Still avoids any byte copying when source + dest share an fs.)
        try:
            os.link(src, link)
            linked = False
        except OSError as e:
            raise HTTPException(500, f"Could not link {src} → {link}: {e}")

    return {
        "dataset_name": stem,
        "saved_to":     str(link),
        "linked_to":    str(src.resolve()),
        "symlink":      linked,
        "bytes":        src.stat().st_size,
    }


# ─────────────────────────────────────────────────────────────────────
# Chunked upload — fast multi-GB transfer with parallel chunks + resume.
#
# Multipart POST is bottlenecked by a single TCP stream and can't
# recover from a drop. The chunked API lets the browser:
#   • upload N chunks in parallel (saturates the link),
#   • retry only the failed chunk,
#   • resume after a browser close (chunks already on disk are skipped).
#
# State lives on disk under  portal_workspace/_uploads/<upload_id>/
#   meta.json                ← filename, total_size, chunk_size, …
#   chunks/{0..N-1}          ← one file per chunk
# so it survives uvicorn restarts.
# ─────────────────────────────────────────────────────────────────────

_DEFAULT_CHUNK_BYTES = 16 * 1024 * 1024  # 16 MiB


def _uploads_root() -> Path:
    p = portal_workspace_dir() / "_uploads"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _session_dir(upload_id: str) -> Path:
    # Defensive: upload_id must be a plain hex string. Reject anything else.
    if not re.fullmatch(r"[A-Fa-f0-9]{8,}", upload_id):
        raise HTTPException(400, "Invalid upload_id")
    return _uploads_root() / upload_id


def _read_meta(upload_id: str) -> dict:
    sess = _session_dir(upload_id)
    meta_p = sess / "meta.json"
    if not meta_p.exists():
        raise HTTPException(404, f"Unknown upload_id {upload_id!r}")
    with open(meta_p) as f:
        return json.load(f)


def _write_meta(upload_id: str, meta: dict) -> None:
    sess = _session_dir(upload_id)
    sess.mkdir(parents=True, exist_ok=True)
    (sess / "meta.json").write_text(json.dumps(meta, indent=2))


def _received_chunks(upload_id: str) -> list[int]:
    chunks_dir = _session_dir(upload_id) / "chunks"
    if not chunks_dir.exists():
        return []
    out = []
    for p in chunks_dir.iterdir():
        if p.is_file() and p.name.isdigit():
            out.append(int(p.name))
    out.sort()
    return out


class InitUploadRequest(BaseModel):
    filename: str
    total_size: int
    chunk_size: int = _DEFAULT_CHUNK_BYTES


@router.post("/uploads/init")
def init_upload(body: InitUploadRequest) -> dict:
    """Start a chunked upload session. Returns the upload_id the
    browser then uses for `PUT /api/uploads/<id>/chunk/<index>`."""
    if body.total_size <= 0:
        raise HTTPException(400, "total_size must be positive")
    if body.chunk_size <= 0:
        raise HTTPException(400, "chunk_size must be positive")

    ext = Path(body.filename).suffix.lower()
    if ext not in _ALLOWED_EXT:
        raise HTTPException(
            415,
            f"Unsupported extension {ext!r}. Allowed: {sorted(_ALLOWED_EXT)}",
        )

    upload_id = uuid.uuid4().hex
    sess = _session_dir(upload_id)
    (sess / "chunks").mkdir(parents=True, exist_ok=True)

    total_chunks = (body.total_size + body.chunk_size - 1) // body.chunk_size
    meta = {
        "upload_id":    upload_id,
        "filename":     body.filename,
        "ext":          ext,
        "total_size":   body.total_size,
        "chunk_size":   body.chunk_size,
        "total_chunks": total_chunks,
        "finished":     False,
        "dataset_name": None,
    }
    _write_meta(upload_id, meta)
    return {
        "upload_id":       upload_id,
        "total_chunks":    total_chunks,
        "received_chunks": [],
    }


@router.get("/uploads/{upload_id}/status")
def upload_status(upload_id: str) -> dict:
    """Return which chunks are already on disk — used by the client
    to resume after a tab close / connection drop."""
    meta = _read_meta(upload_id)
    return {
        "upload_id":       upload_id,
        "filename":        meta["filename"],
        "total_size":      meta["total_size"],
        "total_chunks":    meta["total_chunks"],
        "chunk_size":      meta["chunk_size"],
        "received_chunks": _received_chunks(upload_id),
        "finished":        meta.get("finished", False),
        "dataset_name":    meta.get("dataset_name"),
    }


@router.put("/uploads/{upload_id}/chunk/{index}")
async def upload_chunk(upload_id: str, index: int, request: Request) -> dict:
    """Receive one chunk's raw bytes (Content-Type: application/octet-stream).

    Streams the body straight to `chunks/<index>` — never holds the
    whole chunk in memory. Idempotent: re-uploading the same chunk
    just overwrites it.
    """
    meta = _read_meta(upload_id)
    if meta.get("finished"):
        raise HTTPException(409, "Upload already finalised")
    if index < 0 or index >= meta["total_chunks"]:
        raise HTTPException(400, f"Chunk index {index} out of range "
                                 f"[0, {meta['total_chunks']})")

    chunks_dir = _session_dir(upload_id) / "chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)
    out_path = chunks_dir / str(index)
    tmp_path = chunks_dir / f".{index}.partial"

    written = 0
    with open(tmp_path, "wb") as f:
        async for chunk in request.stream():
            if not chunk:
                continue
            f.write(chunk)
            written += len(chunk)
    os.replace(tmp_path, out_path)

    received = _received_chunks(upload_id)
    return {
        "upload_id":       upload_id,
        "index":           index,
        "bytes":           written,
        "received_chunks": received,
        "received_count":  len(received),
        "total_chunks":    meta["total_chunks"],
    }


class FinishUploadRequest(BaseModel):
    # Optional: override the dataset_name. By default it's derived from the
    # original filename's stem the same way `/api/uploads` does.
    dataset_name: str | None = None


@router.post("/uploads/{upload_id}/finish")
def finish_upload(upload_id: str, body: FinishUploadRequest | None = None) -> dict:
    """Concatenate the chunks into the final `source.<ext>` for a new
    portal job. Cleans up the upload session on success."""
    meta = _read_meta(upload_id)
    if meta.get("finished"):
        # Idempotent — return the existing dataset.
        return {
            "dataset_name": meta.get("dataset_name"),
            "saved_to":     meta.get("saved_to"),
            "bytes":        meta["total_size"],
            "already_done": True,
        }

    received = _received_chunks(upload_id)
    total = meta["total_chunks"]
    if len(received) != total or received != list(range(total)):
        missing = sorted(set(range(total)) - set(received))
        raise HTTPException(
            409,
            f"Cannot finish: {len(missing)} chunk(s) still missing "
            f"(e.g. {missing[:10]})",
        )

    # Dataset name: explicit override (sanitised) > original filename
    # stem, with `_uploaded` appended (`_uploaded_N` on collision).
    stem_source = (body.dataset_name if body and body.dataset_name
                   else Path(meta["filename"]).stem)
    dataset_name = _unique_upload_name(_safe_stem(stem_source))

    job_root = portal_job_dir(dataset_name)
    job_root.mkdir(parents=True, exist_ok=True)
    dst = job_root / f"source{meta['ext']}"

    # Concatenate chunks in order. shutil.copyfileobj uses a fast
    # 64 KiB buffer; total cost is one full read + one full write.
    chunks_dir = _session_dir(upload_id) / "chunks"
    written = 0
    with open(dst, "wb") as out_f:
        for i in range(total):
            chunk_p = chunks_dir / str(i)
            with open(chunk_p, "rb") as in_f:
                shutil.copyfileobj(in_f, out_f, length=64 * 1024)
            written += chunk_p.stat().st_size

    if written != meta["total_size"]:
        # Bytes don't match the declared size — keep the session around
        # so the client can re-send missing/corrupt chunks.
        raise HTTPException(
            422,
            f"Reassembled size {written} ≠ declared total_size "
            f"{meta['total_size']}. Re-upload the affected chunks.",
        )

    # Mark finished + clean up the session directory.
    meta["finished"]     = True
    meta["dataset_name"] = dataset_name
    meta["saved_to"]     = str(dst)
    _write_meta(upload_id, meta)

    # Drop a `_meta.json` marker in the job folder so the
    # `/api/uploads/list` endpoint can detect this as an upload
    # regardless of folder name.
    _write_upload_meta(job_root, meta.get("filename") or dataset_name)

    try:
        shutil.rmtree(_session_dir(upload_id))
    except OSError:
        pass

    return {
        "dataset_name": dataset_name,
        "saved_to":     str(dst),
        "bytes":        written,
        "already_done": False,
    }


@router.delete("/uploads/{upload_id}")
def cancel_upload(upload_id: str) -> dict:
    """Drop a partially-uploaded session (frees the disk space)."""
    sess = _session_dir(upload_id)
    if not sess.exists():
        return {"upload_id": upload_id, "removed": False, "reason": "not found"}
    shutil.rmtree(sess, ignore_errors=True)
    return {"upload_id": upload_id, "removed": True}
