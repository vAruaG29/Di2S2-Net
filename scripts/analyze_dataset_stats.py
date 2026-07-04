#!/usr/bin/env python3
"""
Dataset statistics — tile/label counts + per-class PIXEL distribution.
======================================================================
Reproduces, in one pass, the two things the training scripts surface:

  1. Per-dataset TILE / LABELED-tile counts
     (same rule as models/dataset.py::SegDataModule:
      a tile is "labeled" iff label_index.csv has has_label=="yes" AND a mask_path)

  2. Per-class PIXEL distribution per dataset + grand total  (area share)
     (same as SegDataModule::_print_class_distribution:
      counts pixels over num_classes+1 classes = ["background"] + cfg classes)

  3. Per-class TILE presence per dataset + grand total
     (how many 1024x1024 tiles contain >=1 pixel of each class, and % of tiles)

  4. Per-class ViT-PATCH (16x16) presence per dataset + grand total
     (the model is ViT-L/16: each 1024x1024 tile = 64x64 = 4096 patches of 16x16 px.
      This counts how many 16x16 patches contain >=1 pixel of each class, and % of
      all patches — the granularity the encoder actually tokenises at.)

  NOTE on terminology: a "tile" = one 1024x1024 image written by tile_raster.
  A "patch" = a 16x16 px ViT token (patch_size=16). They are NOT the same thing.

Run from the repo root (where configs/ paths resolve), on the server:

    conda activate svamitva2
    python scripts/analyze_dataset_stats.py
    # options:
    python scripts/analyze_dataset_stats.py --config dinov3_hrdecoder_pipeline/configs/train_full.yaml
    python scripts/analyze_dataset_stats.py --no-pixels        # tile/label counts only (fast)
    python scripts/analyze_dataset_stats.py --csv stats.csv    # also dump a CSV

Reads ONLY (never writes into the pipeline):
    <labels_dir>/label_index.csv           -> tile/label counts
    <labels_dir>/<dataset>/*_mask.tif      -> pixel distribution
    <tile_index>                           -> cross-check
"""

import os
import csv
import sys
import glob
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
import yaml

# ── optional fast raster reader; fall back to PIL ────────────────────────────
try:
    import rasterio
    def read_band1(path):
        with rasterio.open(path) as src:
            return src.read(1)
    _READER = "rasterio"
except Exception:
    from PIL import Image
    def read_band1(path):
        return np.array(Image.open(path))
    _READER = "PIL"

REPO_ROOT = Path(__file__).resolve().parent.parent   # scripts/ → repo root
PIPE_ROOT = REPO_ROOT / "dinov3_hrdecoder_pipeline"
DEFAULT_CFG = PIPE_ROOT / "configs" / "train_full.yaml"


def load_cfg(path):
    with open(path) as f:
        return yaml.safe_load(f)


def hr(n=70):
    print("=" * n)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Tile / labeled-tile counts  (SegDataModule logic)
# ─────────────────────────────────────────────────────────────────────────────
def tile_label_counts(labels_dir, tile_index_path):
    label_index = os.path.join(labels_dir, "label_index.csv")
    if not os.path.exists(label_index):
        sys.exit(f"ERROR: {label_index} not found — run prepare_masks first.")

    total = defaultdict(int)
    labeled = defaultdict(int)
    with open(label_index) as f:
        for row in csv.DictReader(f):
            ds = row["dataset"]
            total[ds] += 1
            if row.get("has_label") == "yes" and row.get("mask_path"):
                labeled[ds] += 1

    datasets = sorted(total, key=lambda d: (-labeled[d], d))
    w = max((len(d) for d in datasets), default=8)

    print("\n" + "#" * 70)
    print("# 1. TILE & LABEL COUNTS  (per dataset)")
    print("#" * 70)
    print(f"\n{'Dataset':<{w}}  {'Total':>9}  {'Labeled':>8}  {'Unlabeled':>9}  {'%lbl':>6}  {'Split':>6}")
    print("-" * (w + 46))
    TT = TL = 0
    n_train = n_test = 0
    per_ds = {}
    for d in datasets:
        t, l = total[d], labeled[d]
        u = t - l
        split = "train" if l > 0 else "test"
        n_train += l > 0
        n_test += l == 0
        print(f"{d:<{w}}  {t:>9,}  {l:>8,}  {u:>9,}  {100*l/t:>5.1f}%  {split:>6}")
        TT += t
        TL += l
        per_ds[d] = (t, l, split)
    print("-" * (w + 46))
    print(f"{'TOTAL (' + str(len(datasets)) + ' datasets)':<{w}}  {TT:>9,}  {TL:>8,}  {TT-TL:>9,}  {100*TL/TT:>5.1f}%")
    print(f"\n  Train villages (>=1 labeled tile): {n_train}   |   Test/unlabeled villages: {n_test}")
    print(f"  Total tiles: {TT:,}   Labeled: {TL:,}   Unlabeled: {TT-TL:,}")

    # cross-check against tile_index.csv
    if os.path.exists(tile_index_path):
        with open(tile_index_path) as f:
            ti = sum(1 for _ in f) - 1
        print(f"  Cross-check tile_index.csv = {ti:,} tiles -> "
              f"{'MATCH' if ti == TT else 'MISMATCH (%+d)' % (ti - TT)}")
    return per_ds


# ─────────────────────────────────────────────────────────────────────────────
# 2. Per-class PIXEL distribution  (_print_class_distribution logic)
# ─────────────────────────────────────────────────────────────────────────────
def print_pixel_dist(title, counts, ntiles, names):
    total = counts.sum()
    if total == 0:
        print(f"\n  {title}: (no pixels)")
        return
    print(f"\n  {title} — PIXEL distribution ({ntiles} tiles):")
    print(f"  {'Class':<20s} {'Pixels':>16s} {'%pixels':>9s}")
    print(f"  {'-'*48}")
    for c in range(len(names)):
        print(f"  {names[c]:<20s} {counts[c]:>16,d} {counts[c]/total*100:>8.3f}%")
    print(f"  {'-'*48}")
    print(f"  {'Total':<20s} {total:>16,d}")


def print_tile_dist(title, tiles_with, ntiles, names):
    """tiles_with[c] = # 1024x1024 tiles containing >=1 pixel of class c."""
    if ntiles == 0:
        print(f"\n  {title}: (no tiles)")
        return
    print(f"\n  {title} — TILE presence ({ntiles} tiles of 1024x1024; a tile counts once if it holds the class):")
    print(f"  {'Class':<20s} {'Tiles':>12s} {'%tiles':>9s}")
    print(f"  {'-'*44}")
    for c in range(len(names)):
        print(f"  {names[c]:<20s} {tiles_with[c]:>12,d} {tiles_with[c]/ntiles*100:>8.2f}%")
    print(f"  {'-'*44}")
    print(f"  {'Tiles':<20s} {ntiles:>12,d}")


def print_patch16_dist(title, patches_with, npatches, names, P):
    """patches_with[c] = # PxP ViT patches containing >=1 pixel of class c."""
    if npatches == 0:
        print(f"\n  {title}: (no patches)")
        return
    print(f"\n  {title} — ViT-PATCH ({P}x{P}) presence ({npatches:,} patches; a patch counts once if it holds the class):")
    print(f"  {'Class':<20s} {'Patches':>14s} {'%patches':>10s}")
    print(f"  {'-'*48}")
    for c in range(len(names)):
        print(f"  {names[c]:<20s} {patches_with[c]:>14,d} {patches_with[c]/npatches*100:>9.3f}%")
    print(f"  {'-'*48}")
    print(f"  {'Total patches':<20s} {npatches:>14,d}")


def patch16_presence(m2d, NC, P):
    """For a 2D mask, return per-class count of PxP patches that contain that class.
    Crops to a multiple of P so reshape is exact (edge tiles are already 1024-padded)."""
    H, W = m2d.shape
    Hc, Wc = (H // P) * P, (W // P) * P
    if Hc == 0 or Wc == 0:
        return np.zeros(NC, dtype=np.int64), 0
    mb = m2d[:Hc, :Wc].reshape(Hc // P, P, Wc // P, P)  # (gy, P, gx, P)
    out = np.zeros(NC, dtype=np.int64)
    for c in range(NC):
        present = (mb == c).any(axis=(1, 3))   # (gy, gx) bool
        out[c] = int(present.sum())
    return out, (Hc // P) * (Wc // P)


def scan_and_report(labels_dir, cfg, csv_out=None):
    """Single pass over the mask rasters → PIXEL + TILE + ViT-PATCH(16x16) views."""
    # names exactly as training builds them: ["background"] + cfg classes
    names = ["background"] + [c["name"] for c in cfg["classes"]]
    NC = len(names)  # = num_classes + 1
    P = int(cfg["model"]["encoder"].get("patch_size", 16))  # ViT patch size (16)

    files = sorted(glob.glob(os.path.join(labels_dir, "*", "*_mask.tif")))
    if not files:
        sys.exit(f"ERROR: no *_mask.tif under {labels_dir}")

    pix = defaultdict(lambda: np.zeros(NC, dtype=np.int64))   # pixels per class
    til = defaultdict(lambda: np.zeros(NC, dtype=np.int64))   # 1024x1024 tiles with class
    p16 = defaultdict(lambda: np.zeros(NC, dtype=np.int64))   # PxP patches with class
    ntiles = defaultdict(int)
    npatch = defaultdict(int)                                  # total PxP patches per ds
    ignored = 0  # pixels with class id >= NC (e.g. 255 ignore_index)

    print("\n" + "#" * 70)
    print(f"# SCANNING MASKS  (reader: {_READER}, {len(files):,} masks, ViT patch={P}x{P})")
    print("#" * 70)

    for i, fp in enumerate(files, 1):
        ds = os.path.basename(os.path.dirname(fp))
        try:
            m2d = read_band1(fp)
        except Exception as e:
            print(f"  WARN: could not read {fp}: {e}")
            continue
        bc = np.bincount(m2d.ravel(), minlength=256)
        pix[ds] += bc[:NC]
        ignored += int(bc[NC:].sum())
        til[ds] += (bc[:NC] > 0).astype(np.int64)   # tile presence
        pc, np_tile = patch16_presence(m2d, NC, P)  # 16x16 patch presence
        p16[ds] += pc
        npatch[ds] += np_tile
        ntiles[ds] += 1
        if i % 500 == 0:
            print(f"  ...processed {i:,}/{len(files):,} masks", file=sys.stderr)

    order = sorted(pix, key=lambda d: -ntiles[d])
    g_pix = np.zeros(NC, dtype=np.int64)
    g_til = np.zeros(NC, dtype=np.int64)
    g_p16 = np.zeros(NC, dtype=np.int64)
    g_tiles = 0
    g_patch = 0
    for ds in order:
        g_pix += pix[ds]; g_til += til[ds]; g_p16 += p16[ds]
        g_tiles += ntiles[ds]; g_patch += npatch[ds]

    # ── Section 2: pixel distribution ──
    print("\n" + "#" * 70)
    print("# 2. PER-CLASS PIXEL DISTRIBUTION  (area share)")
    print("#" * 70)
    for ds in order:
        print_pixel_dist(ds, pix[ds], ntiles[ds], names)
    print("\n" + "=" * 70)
    print_pixel_dist("TRAIN — ALL labeled tiles (what full_train prints)",
                     g_pix, g_tiles, names)
    if ignored:
        print(f"\n  (ignored pixels with id >= {NC}, e.g. 255: {ignored:,})")
    print("=" * 70)

    # ── Section 3: tile (1024x1024) presence ──
    print("\n" + "#" * 70)
    print("# 3. PER-CLASS TILE PRESENCE  (how many 1024x1024 tiles contain the class)")
    print("#" * 70)
    for ds in order:
        print_tile_dist(ds, til[ds], ntiles[ds], names)
    print("\n" + "=" * 70)
    print_tile_dist("TRAIN — ALL labeled tiles", g_til, g_tiles, names)
    print("=" * 70)

    # ── Section 4: ViT-patch (PxP) presence ──
    print("\n" + "#" * 70)
    print(f"# 4. PER-CLASS ViT-PATCH ({P}x{P}) PRESENCE  (how many {P}x{P} tokens contain the class)")
    print("#" * 70)
    for ds in order:
        print_patch16_dist(ds, p16[ds], npatch[ds], names, P)
    print("\n" + "=" * 70)
    print_patch16_dist("TRAIN — ALL labeled tiles", g_p16, g_patch, names, P)
    print("=" * 70)

    if csv_out:
        rows = []
        def emit(ds, pixv, tilv, p16v, nt, npt):
            tot = pixv.sum()
            for c in range(NC):
                rows.append([ds, names[c],
                             int(pixv[c]), round(pixv[c]/tot*100, 4) if tot else 0.0,
                             int(tilv[c]), round(tilv[c]/nt*100, 4) if nt else 0.0,
                             int(p16v[c]), round(p16v[c]/npt*100, 4) if npt else 0.0])
        for ds in order:
            emit(ds, pix[ds], til[ds], p16[ds], ntiles[ds], npatch[ds])
        emit("__TOTAL__", g_pix, g_til, g_p16, g_tiles, g_patch)
        with open(csv_out, "w", newline="") as f:
            wri = csv.writer(f)
            wri.writerow(["dataset", "class",
                          "pixels", "percent_pixels",
                          "tiles_with_class", "percent_tiles",
                          f"vit{P}_patches_with_class", "percent_patches"])
            wri.writerows(rows)
        print(f"\n  CSV written -> {csv_out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(DEFAULT_CFG),
                    help="train_full.yaml / train.yaml (for paths + class names)")
    ap.add_argument("--no-pixels", action="store_true",
                    help="skip the (slow) pixel scan; print tile/label counts only")
    ap.add_argument("--csv", default=None, help="optional CSV dump of the pixel table")
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    labels_dir = cfg["paths"]["labels_dir"]
    tile_index = cfg["paths"]["tile_index"]

    # If the config still holds another machine's absolute paths, fall back to
    # paths relative to the repo root (so it "just works" on the server).
    if not os.path.isdir(labels_dir):
        local = REPO_ROOT / "masks"
        if local.is_dir():
            labels_dir = str(local)
    if not os.path.exists(tile_index):
        local_ti = REPO_ROOT / "tiles" / "tile_index.csv"
        if local_ti.exists():
            tile_index = str(local_ti)

    print(f"labels_dir : {labels_dir}")
    print(f"tile_index : {tile_index}")
    print(f"classes    : {['background'] + [c['name'] for c in cfg['classes']]}")

    tile_label_counts(labels_dir, tile_index)
    if not args.no_pixels:
        scan_and_report(labels_dir, cfg, csv_out=args.csv)
    else:
        print("\n(skipped pixel + patch distribution: --no-pixels)")


if __name__ == "__main__":
    main()
