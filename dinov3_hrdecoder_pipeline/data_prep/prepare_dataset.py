#!/usr/bin/env python3
"""
Prepare a clean, organized dataset directory from the scattered raw data.
==========================================================================

This script:
  1. Scans all raster files (TIF/ECW) in train/ and test/
  2. Validates each file with rasterio (skips corrupted/unreadable ones)
  3. Detects and removes duplicates (keeps the valid copy)
  4. Organises everything into a clean `dataset/` directory:
       dataset/
         train/
           images/
             CG/   ← Chhattisgarh rasters
             PB/   ← Punjab rasters
           labels/
             CG/   ← CG shapefiles (all component files)
             PB/   ← PB shapefiles (all component files)
         test/
           images/
             CG/
             PB/
  5. Uses **symlinks** to avoid duplicating ~20 GB of rasters
  6. Copies label shapefiles into the correct sub-directory
  7. Writes a manifest CSV with file status and metadata

Usage:
    python -m dinov3_hrdecoder_pipeline.data_prep.prepare_dataset
    python -m dinov3_hrdecoder_pipeline.data_prep.prepare_dataset --copy   # hard-copy instead of symlink
"""

import argparse
import csv
import os
import re
import shutil
import sys
from pathlib import Path

try:
    import rasterio
except ImportError:
    print("ERROR: rasterio not installed. Run: pip install rasterio")
    sys.exit(1)

try:
    import yaml
except ImportError:
    print("ERROR: pyyaml not installed. Run: pip install pyyaml")
    sys.exit(1)


# ─── Constants (not configurable — these are file-format invariants) ──
# Shapefile component extensions: every shapefile is a set of these files,
# and all must be moved together. Not a configuration value.
SHP_COMPONENTS = [".shp", ".shx", ".dbf", ".prj", ".cpg", ".sbn", ".sbx", ".shp.xml"]

# ─── Config loader ────────────────────────────────────────────────────
PIPE_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CFG = PIPE_ROOT / "configs" / "data_prep.yaml"


def load_config(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def classify_region(filepath: str) -> str:
    """Determine whether a raster belongs to CG (Chhattisgarh) or PB (Punjab).

    Heuristic:
      - Path or name contains CG / 45xxxx / Training_dataSet → CG
      - Path or name contains PB / 37xxx / 289xx / NADALA / PINDORI / TIMMOWAL
        / BARNALA / AMRITSAR / FATEHGARH / DIWANA → PB
    """
    fp = filepath.upper()
    name = os.path.basename(filepath).upper()

    # Explicit directory markers
    if "/CG_" in fp or "CG_LIVE" in fp:
        return "CG"
    if "/PB_" in fp or "LIVE_DEMO" in fp:
        return "PB"

    # Village-code based (CG codes are 43xxxx–45xxxx, PB are 28xxx–40xxx)
    cg_codes = re.findall(r"(?:^|_)(4[345]\d{4})(?:_|$)", name)
    pb_codes = re.findall(r"(?:^|_)((?:2[89]|3[0-9]|40)\d{3})(?:_|$)", name)
    if cg_codes and not pb_codes:
        return "CG"
    if pb_codes and not cg_codes:
        return "PB"

    # Punjab village / district names
    pb_names = [
        "NADALA", "PINDORI", "TIMMOWAL", "FATTU", "BAGGA",
        "BARNALA", "AMRITSAR", "FATEHGARH", "DIWANA", "KARTARPUR",
        "BADRA", "BUTTAR", "BUTTER", "ANAITPURA", "TUGALWAL",
    ]
    for pn in pb_names:
        if pn in fp:
            return "PB"

    # CG village names
    cg_names = [
        "SAMLUR", "KUTULNAR", "BINJAM", "JHODIYAWADAM", "BADETUMNAR",
        "BANGAPAL", "CHHOTETUMAR", "MOFALNAR", "KUTRU", "AAKLANKA",
        "MURDANDA", "AWAPALLI", "CHINTAKONTA", "NAGUL", "MADASE",
        "GHOTPAL", "PARAGAON", "BAGAI", "CHANABHATA", "BASANTPUR",
        "GUDBHELI",
    ]
    for cn in cg_names:
        if cn in fp:
            return "CG"

    # Training_dataSet_2 / _3 are CG
    if "TRAINING_DATASET" in fp:
        return "CG"

    return "UNKNOWN"


def validate_raster(path: str) -> dict:
    """Try to open a raster and return metadata, or error info."""
    info = {
        "path": path,
        "valid": False,
        "error": None,
        "width": None,
        "height": None,
        "bands": None,
        "crs": None,
        "size_mb": 0,
    }
    info["size_mb"] = round(os.path.getsize(path) / (1024 * 1024), 1)
    try:
        with rasterio.open(path) as ds:
            info["valid"] = True
            info["width"] = ds.width
            info["height"] = ds.height
            info["bands"] = ds.count
            info["crs"] = str(ds.crs) if ds.crs else None
    except Exception as e:
        info["error"] = str(e)
    return info


def canonical_raster_key(filepath: str, raster_exts: set) -> str:
    """Return a key that identifies the same underlying raster regardless of
    directory or extension, so we can de-duplicate."""
    name = os.path.basename(filepath)
    # Strip extension
    for ext in sorted(raster_exts, key=len, reverse=True):
        if name.lower().endswith(ext):
            name = name[: -len(ext)]
            break
    # Remove common suffixes like _3857
    name = re.sub(r"_3857$", "", name, flags=re.IGNORECASE)
    # Normalise whitespace / underscores
    name = re.sub(r"[\s_]+", "_", name).strip("_").upper()
    return name


def collect_rasters(src_dir: Path, raster_exts: set, workspace: Path) -> list:
    """Walk src_dir and return metadata for every raster file found."""
    results = []
    for root, _dirs, files in os.walk(src_dir):
        for f in files:
            ext = os.path.splitext(f)[1].lower()
            if ext in raster_exts:
                full = os.path.join(root, f)
                info = validate_raster(full)
                info["region"] = classify_region(full)
                info["key"] = canonical_raster_key(full, raster_exts)
                info["relpath"] = os.path.relpath(full, workspace)
                results.append(info)
    return results


def deduplicate(rasters: list[dict]) -> list[dict]:
    """Among rasters sharing the same canonical key, keep only the valid one
    (prefer .tif over .ecw, larger file over smaller)."""
    groups: dict[str, list[dict]] = {}
    for r in rasters:
        groups.setdefault(r["key"], []).append(r)

    kept = []
    skipped = []
    for key, group in groups.items():
        if len(group) == 1:
            kept.append(group[0])
            continue
        # Sort: valid first, then .tif before .ecw, then larger file
        group.sort(
            key=lambda x: (
                not x["valid"],
                not x["path"].lower().endswith(".tif"),
                -x["size_mb"],
            )
        )
        kept.append(group[0])
        for dup in group[1:]:
            dup["_dedup_skipped"] = True
            skipped.append(dup)

    return kept, skipped


def copy_shapefile(src_shp: Path, dst_dir: Path, canonical_name: str = None):
    """Copy a shapefile and all its component files to dst_dir,
    optionally renaming to canonical_name."""
    stem = src_shp.stem
    src_dir = src_shp.parent
    if canonical_name is None:
        canonical_name = stem

    for comp in SHP_COMPONENTS:
        src_comp = src_dir / f"{stem}{comp}"
        if src_comp.exists():
            dst_comp = dst_dir / f"{canonical_name}{comp}"
            shutil.copy2(str(src_comp), str(dst_comp))


def link_or_copy(src: str, dst: str, use_copy: bool = False):
    """Create a symlink (default) or hard copy of src at dst."""
    dst_path = Path(dst)
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    if dst_path.exists() or dst_path.is_symlink():
        dst_path.unlink()
    if use_copy:
        shutil.copy2(src, dst)
    else:
        os.symlink(os.path.abspath(src), dst)


# ─── Main ──────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Prepare clean dataset directory")
    parser.add_argument("--config", default=str(DEFAULT_CFG),
                        help="Path to data_prep.yaml")
    parser.add_argument("--copy", action="store_true",
                        help="Hard-copy rasters instead of symlinking")
    parser.add_argument("--output", default=None,
                        help="Output dataset directory (overrides config paths.dataset_dir)")
    args = parser.parse_args()

    cfg = load_config(Path(args.config))
    paths = cfg["paths"]

    workspace        = Path(paths["workspace"])
    train_src        = Path(paths["train_raw_dir"])
    test_src         = Path(paths["test_raw_dir"])
    shp_raw_cg       = Path(paths["shp_raw_cg"])
    shp_raw_pb       = Path(paths["shp_raw_pb"])
    raster_exts      = set(cfg["raster_extensions"])
    feature_classes  = cfg["feature_classes"]
    feature_normalize = cfg.get("feature_normalize", {})

    dst = Path(args.output) if args.output else Path(paths["dataset_dir"])
    use_copy = args.copy

    print("=" * 70)
    print("  SVAMITVA Dataset Preparation")
    print("=" * 70)
    print(f"  Config:    {args.config}")
    print(f"  Workspace: {workspace}")
    print(f"  Output:    {dst}")

    # ── 1. Collect & validate train rasters ─────────────────────────────
    print("\n[1/6] Scanning training rasters ...")
    train_rasters = collect_rasters(train_src, raster_exts, workspace)
    valid_train = [r for r in train_rasters if r["valid"]]
    failed_train = [r for r in train_rasters if not r["valid"]]

    print(f"  Found {len(train_rasters)} rasters: {len(valid_train)} valid, {len(failed_train)} corrupted/unreadable")
    for f in failed_train:
        print(f"  ✗ CORRUPTED: {f['relpath']}")
        print(f"    Reason: {f['error']}")

    # ── 2. De-duplicate ─────────────────────────────────────────────────
    print("\n[2/6] De-duplicating ...")
    kept_train, skipped_train = deduplicate(valid_train)
    print(f"  Keeping {len(kept_train)} unique rasters, skipping {len(skipped_train)} duplicates")
    for s in skipped_train:
        print(f"  ✗ DUPLICATE: {s['relpath']} (duplicate of kept version)")

    # ── 3. Collect & validate test rasters ──────────────────────────────
    print("\n[3/6] Scanning test rasters ...")
    test_rasters = collect_rasters(test_src, raster_exts, workspace)
    valid_test = [r for r in test_rasters if r["valid"]]
    failed_test = [r for r in test_rasters if not r["valid"]]
    print(f"  Found {len(test_rasters)} rasters: {len(valid_test)} valid, {len(failed_test)} corrupted")
    for f in failed_test:
        print(f"  ✗ CORRUPTED: {f['relpath']}")

    # ── 4. Create directory structure ───────────────────────────────────
    print(f"\n[4/6] Creating dataset directory at: {dst}")
    for split in ["train", "test"]:
        for sub in ["images/CG", "images/PB", "labels/CG", "labels/PB"]:
            (dst / split / sub).mkdir(parents=True, exist_ok=True)

    # ── 5. Link/copy rasters ───────────────────────────────────────────
    print("\n[5/6] Organizing raster files ...")
    action = "Copying" if use_copy else "Symlinking"

    manifest = []

    for r in kept_train:
        region = r["region"]
        fname = os.path.basename(r["path"])
        dest = dst / "train" / "images" / region / fname
        link_or_copy(r["path"], str(dest), use_copy)
        print(f"  {action}: train/images/{region}/{fname}")
        manifest.append({
            "split": "train",
            "region": region,
            "filename": fname,
            "status": "valid",
            "width": r["width"],
            "height": r["height"],
            "bands": r["bands"],
            "crs": r["crs"],
            "size_mb": r["size_mb"],
            "source": r["relpath"],
        })

    for r in valid_test:
        region = r["region"]
        fname = os.path.basename(r["path"])
        dest = dst / "test" / "images" / region / fname
        link_or_copy(r["path"], str(dest), use_copy)
        print(f"  {action}: test/images/{region}/{fname}")
        manifest.append({
            "split": "test",
            "region": region,
            "filename": fname,
            "status": "valid",
            "width": r["width"],
            "height": r["height"],
            "bands": r["bands"],
            "crs": r["crs"],
            "size_mb": r["size_mb"],
            "source": r["relpath"],
        })

    # Add skipped / corrupted to manifest
    for r in failed_train + failed_test:
        manifest.append({
            "split": "train" if "train" in r["relpath"] else "test",
            "region": r["region"],
            "filename": os.path.basename(r["path"]),
            "status": f"corrupted: {r['error'][:80]}",
            "width": r.get("width"),
            "height": r.get("height"),
            "bands": r.get("bands"),
            "crs": r.get("crs"),
            "size_mb": r["size_mb"],
            "source": r["relpath"],
        })
    for s in skipped_train:
        manifest.append({
            "split": "train",
            "region": s["region"],
            "filename": os.path.basename(s["path"]),
            "status": "duplicate_skipped",
            "width": s.get("width"),
            "height": s.get("height"),
            "bands": s.get("bands"),
            "crs": s.get("crs"),
            "size_mb": s["size_mb"],
            "source": s["relpath"],
        })

    # ── 6. Copy label shapefiles ────────────────────────────────────────
    print("\n[6/6] Copying label shapefiles ...")

    for region, src_dir in (("CG", shp_raw_cg), ("PB", shp_raw_pb)):
        if not src_dir.exists():
            print(f"  WARNING: {region} shapefile dir not found: {src_dir}")
            continue
        dst_region = dst / "train" / "labels" / region
        for shp in sorted(src_dir.glob("*.shp")):
            canonical = feature_normalize.get(shp.stem, shp.stem)
            copy_shapefile(shp, dst_region, canonical)
            class_id = feature_classes.get(canonical, "?")
            print(f"  {region}: {canonical} (class {class_id}) — {shp.stem}")

    # ── Write manifest ──────────────────────────────────────────────────
    manifest_path = dst / "manifest.csv"
    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "split", "region", "filename", "status",
            "width", "height", "bands", "crs", "size_mb", "source",
        ])
        writer.writeheader()
        writer.writerows(manifest)
    print(f"\n  Manifest written to: {manifest_path}")

    # ── Summary ─────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  DATASET SUMMARY")
    print("=" * 70)

    train_cg = list((dst / "train" / "images" / "CG").glob("*"))
    train_pb = list((dst / "train" / "images" / "PB").glob("*"))
    test_cg = list((dst / "test" / "images" / "CG").glob("*"))
    test_pb = list((dst / "test" / "images" / "PB").glob("*"))

    print(f"  Train images:  CG={len(train_cg)}, PB={len(train_pb)}, total={len(train_cg)+len(train_pb)}")
    print(f"  Test  images:  CG={len(test_cg)},  PB={len(test_pb)},  total={len(test_cg)+len(test_pb)}")

    label_cg = list((dst / "train" / "labels" / "CG").glob("*.shp"))
    label_pb = list((dst / "train" / "labels" / "PB").glob("*.shp"))
    print(f"  Label classes: CG={len(label_cg)} shapefiles, PB={len(label_pb)} shapefiles")

    print(f"\n  Corrupted files excluded: {len(failed_train) + len(failed_test)}")
    print(f"  Duplicates excluded:     {len(skipped_train)}")

    print(f"\n  Dataset directory: {dst}")
    print("=" * 70)

    # Print the final tree
    print("\n  Directory structure:")
    for root, dirs, files in os.walk(dst):
        level = root.replace(str(dst), "").count(os.sep)
        indent = "  " * (level + 2)
        print(f"{indent}{os.path.basename(root)}/")
        if level < 3:  # Don't print individual shp component files
            sub_indent = "  " * (level + 3)
            for f in sorted(files):
                print(f"{sub_indent}{f}")

    print("\nDone!")


if __name__ == "__main__":
    main()
