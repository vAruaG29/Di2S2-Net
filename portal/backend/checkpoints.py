"""
Checkpoint discovery + dataset-name namespacing for multi-checkpoint
inference.

The portal can run any checkpoint shipped under `<bundle>/pretrained/`.
Because the rest of the portal keys EVERYTHING by dataset name
(`portal_workspace/<NAME>/…` and `outputs/.../<NAME>_pred.*`), running a
second checkpoint on the same image would otherwise overwrite the first
one's results. To let results from different checkpoints coexist in the
gallery, a *non-default* checkpoint's outputs are written under a
suffixed effective dataset name:

    <BASE>@@<checkpoint-id>          e.g.  BADRA_…_ORTHO@@val-best-miou-…

The default checkpoint keeps the bare `<BASE>` name, so existing portal
runs and offline (LIB) datasets are completely unaffected.
"""

from __future__ import annotations

import re
import zipfile
from pathlib import Path

from .settings import BUNDLE_ROOT


# Separator between the base dataset name and the checkpoint id in an
# effective (namespaced) dataset name. Chosen so it can never collide
# with a real dataset name (which only ever contain letters, digits,
# spaces, underscores and hyphens) and survives URL-encoding and use as
# a filesystem path component.
CKPT_SEP = "@@"

PRETRAINED_DIR = BUNDLE_ROOT / "pretrained"

# The checkpoint the portal used before this feature existed. It keeps
# the bare base name so nothing about the existing gallery changes when
# a user picks it.
DEFAULT_CKPT_FILENAME = "dinov3_hrdecoder_full_best_loss=0.0615.ckpt"


def _slug(stem: str) -> str:
    """Short, stable, URL/filesystem-safe id derived from a .ckpt stem.

    Strips the shared `dinov3[_-]hrdecoder[_-]` prefix, lowercases, and
    keeps only `[a-z0-9-]`. Deterministic from the filename, so the same
    checkpoint always maps to the same id across restarts (important —
    the id becomes part of the on-disk dataset folder name)."""
    s = re.sub(r"^dinov3[_-]+hrdecoder[_-]+", "", stem, flags=re.IGNORECASE)
    s = re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")
    return (s or "ckpt")[:40]


# ── Decoder-architecture detection (torch-free) ──────────────────────────
# A torch.save .ckpt is a ZIP whose `*/data.pkl` member holds the
# state-dict structure, including every parameter NAME as a literal
# string. We read just that small member (NOT the multi-GB tensor blobs
# under `*/data/`) and look for signature key names. This lets the portal
# backend classify a checkpoint without importing torch (which it
# deliberately never loads — all heavy work is shelled out to subprocs).

def _classify_keys_blob(blob: bytes) -> str | None:
    def has(s: bytes) -> bool:
        return s in blob
    # HRDecoder: multi-scale fusion + seg_head.  UPerNet: PSP + FPN + cls_seg.
    if has(b"decoder.seg_head") and has(b"decoder.fusion."):
        return "hrdecoder"
    if has(b"decoder.cls_seg") and (has(b"decoder.psp")
                                    or has(b"decoder.fpn_convs")
                                    or has(b"decoder.lateral_convs")):
        return "upernet"
    if has(b"decoder.linear_fuse") or has(b"decoder.linear_c"):
        return "segformer"
    return None


def detect_decoder_type(ckpt_path: Path) -> str | None:
    """Decoder architecture of a checkpoint ('hrdecoder' | 'upernet' |
    'segformer'), or None if it can't be determined (broken symlink,
    unreadable, legacy format). Callers treat None as "don't hide it" —
    the run_pipeline guard is the real safety net."""
    try:
        if zipfile.is_zipfile(ckpt_path):
            with zipfile.ZipFile(ckpt_path) as zf:
                pkls = [n for n in zf.namelist() if n.endswith("data.pkl")]
                if not pkls:
                    return None
                return _classify_keys_blob(zf.read(pkls[0]))
        # Legacy (non-zip) pickle — scan a bounded prefix so we never pull
        # a multi-GB file into memory.
        with open(ckpt_path, "rb") as f:
            return _classify_keys_blob(f.read(8_000_000))
    except Exception:
        return None


def portal_decoder_type() -> str:
    """The decoder the portal's pipeline builds (`train.yaml`
    model.decoder.type). Only checkpoints with this architecture are
    runnable through the portal."""
    try:
        from .settings import train_cfg
        return str((train_cfg().get("model", {}).get("decoder", {}) or {})
                   .get("type", "hrdecoder")).lower()
    except Exception:
        return "hrdecoder"


def list_checkpoints() -> list[dict]:
    """Every `*.ckpt` under `pretrained/`, each tagged with its detected
    `decoder_type` and whether it's `compatible` with the portal pipeline.

    Lists ALL checkpoints (resolution helpers below depend on that); the
    `/api/checkpoints` route is what filters the incompatible ones out of
    the dropdown.

    Not cached — the directory is tiny and re-scanning each request means
    a freshly-dropped checkpoint shows up without a portal restart, in
    keeping with how `settings.py` re-reads config per request.
    """
    expected = portal_decoder_type()
    out: list[dict] = []
    seen: dict[str, int] = {}
    if PRETRAINED_DIR.is_dir():
        for p in sorted(PRETRAINED_DIR.glob("*.ckpt")):
            cid = _slug(p.stem)
            if cid in seen:                       # de-dupe slug collisions
                seen[cid] += 1
                cid = f"{cid}-{seen[cid]}"
            else:
                seen[cid] = 1
            dtype = detect_decoder_type(p)
            out.append({
                "id": cid,
                "label": cid,
                "filename": p.name,
                "path": str(p),
                "is_default": p.name == DEFAULT_CKPT_FILENAME,
                "decoder_type": dtype,
                # Unknown (None) is treated as compatible — don't hide a
                # checkpoint we simply couldn't sniff; the run_pipeline
                # guard still refuses a true mismatch at load time.
                "compatible": dtype is None or dtype == expected,
            })
    # Guarantee exactly one default — prefer a compatible checkpoint if the
    # canonical default file is missing, so the dropdown's default runs.
    if out and not any(c["is_default"] for c in out):
        first_ok = next((c for c in out if c["compatible"]), out[0])
        first_ok["is_default"] = True
    return out


def _is_default_id(checkpoint_id: str | None) -> bool:
    if not checkpoint_id:
        return True
    for c in list_checkpoints():
        if c["id"] == checkpoint_id:
            return bool(c["is_default"])
    # Unknown id → treat as default (safest: don't namespace).
    return True


def effective_name(base: str, checkpoint_id: str | None) -> str:
    """Dataset name to actually run/store under for (base, checkpoint).

    Returns the bare base for the default checkpoint (or an unknown id),
    else `<base>@@<id>`."""
    if _is_default_id(checkpoint_id):
        return base
    return f"{base}{CKPT_SEP}{checkpoint_id}"


def split_effective(name: str) -> tuple[str, str | None]:
    """Inverse of `effective_name`: `(base, checkpoint_id|None)`.

    Real dataset names never contain `@@`, so a present separator
    unambiguously marks a checkpoint-namespaced dataset."""
    if CKPT_SEP in name:
        base, _, cid = name.partition(CKPT_SEP)
        return base, (cid or None)
    return name, None


def resolve_checkpoint_path(checkpoint_id: str | None) -> Path | None:
    """Absolute path to the chosen checkpoint, or the default, or None
    when `pretrained/` is empty."""
    cks = list_checkpoints()
    if not cks:
        return None
    if checkpoint_id:
        m = next((c for c in cks if c["id"] == checkpoint_id), None)
        if m:
            return Path(m["path"])
    default = next((c for c in cks if c["is_default"]), cks[0])
    return Path(default["path"])
