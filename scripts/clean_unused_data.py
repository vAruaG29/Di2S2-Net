#!/usr/bin/env python3
"""
Remove files under data/ that are neither (a) symlinked by dataset/ nor
(b) a required label source. DRY-RUN by default — prints what it would
delete. Pass --apply to actually delete.

Keep-set:
  1. Every file under data/ that a symlink inside dataset/ points at
     (resolved natively, with a remote-path fallback so it works on any
     machine regardless of the absolute prefix baked into the symlinks).
  2. The shapefile label-source dirs the training config depends on:
        data/train/shp-file
        data/train/PB_training_dataSet_shp_file/shp-file

Options:
  --apply           actually delete (default: dry-run)
  --keep-sidecars   do NOT delete .pyrx/.aux/.aux.xml/.ovr sidecars of
                    kept rasters (they're regenerable but accessory to
                    required files)
"""
from __future__ import annotations
import argparse, os, sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # scripts/ → repo root
DATA = os.path.join(REPO, "data")
DATASET = os.path.join(REPO, "dataset")
MARKER = "/final_submission_and_inference/data/"   # for remote-prefix fallback
SIDECAR_EXTS = (".pyrx", ".aux", ".aux.xml", ".ovr")

KEEP_DIRS = [
    os.path.join(DATA, "train", "shp-file"),
    os.path.join(DATA, "train", "PB_training_dataSet_shp_file", "shp-file"),
]


def referenced_under_data() -> set[str]:
    refs: set[str] = set()
    for dp, _, fs in os.walk(DATASET):
        for f in fs:
            p = os.path.join(dp, f)
            if not os.path.islink(p):
                continue
            t = os.readlink(p)
            if not os.path.isabs(t):
                t = os.path.normpath(os.path.join(dp, t))
            # native resolution (works on the box the symlinks were made for)
            rp = os.path.realpath(p)
            for cand in (rp, t):
                if cand.startswith(DATA + os.sep):
                    refs.add(os.path.normpath(cand))
            # remote-prefix fallback: map any .../data/<tail> back into THIS tree
            if MARKER in t:
                tail = t.split(MARKER, 1)[1]
                refs.add(os.path.normpath(os.path.join(DATA, tail)))
    return refs


def under_keep_dir(p: str) -> bool:
    return any(os.path.abspath(p).startswith(os.path.abspath(d) + os.sep) for d in KEEP_DIRS)


def is_sidecar(p: str) -> bool:
    n = p.lower()
    return any(n.endswith(ext) for ext in SIDECAR_EXTS)


def human(n: float) -> str:
    for u in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or u == "TB":
            return f"{n:.1f}{u}"
        n /= 1024


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="actually delete")
    ap.add_argument("--keep-sidecars", action="store_true",
                    help="don't delete .pyrx/.aux/.aux.xml/.ovr sidecars")
    args = ap.parse_args()

    if not os.path.isdir(DATA) or not os.path.isdir(DATASET):
        sys.exit(f"ERROR: expected data/ and dataset/ at the repo root ({REPO})")

    refs = referenced_under_data()
    print(f"Referenced data/ files (via dataset/ symlinks): {len(refs)}")

    cands, freed = [], 0
    for dp, _, fs in os.walk(DATA):
        for f in fs:
            p = os.path.abspath(os.path.join(dp, f))
            if p in refs or under_keep_dir(p):
                continue
            if args.keep_sidecars and is_sidecar(p):
                continue
            try:
                freed += os.path.getsize(p)
            except OSError:
                pass
            cands.append(p)

    print(f"\n{'WOULD DELETE' if not args.apply else 'DELETING'} "
          f"{len(cands)} file(s), {human(freed)}:\n")
    for p in sorted(cands, key=lambda x: -os.path.getsize(x) if os.path.exists(x) else 0):
        try:
            s = human(os.path.getsize(p))
        except OSError:
            s = "?"
        print(f"  {s:>9}   {os.path.relpath(p, DATA)}")
        if args.apply:
            try:
                os.remove(p)
            except OSError as e:
                print(f"            !! failed: {e}")

    if not args.apply:
        print(f"\nDry-run only. Re-run with --apply to delete "
              f"(add --keep-sidecars to spare overview/aux files).")
    else:
        print(f"\nDone. Reclaimed ~{human(freed)}.")


if __name__ == "__main__":
    main()
