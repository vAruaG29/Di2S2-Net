#!/usr/bin/env python3
"""
Run Full Inference Pipeline on ALL Datasets
============================================
Runs inference → stitch → evaluate for all datasets listed in the
tile index using a trained DINOv3 + HRDecoder checkpoint.

Training datasets (with ground-truth masks) get full per-class metrics.
Test datasets (no masks) get prediction tiles + stitched GeoTIFFs only.

Usage:
    python -m dinov3_hrdecoder_pipeline.inference.run_pipeline \
        --checkpoint checkpoints/<best>.ckpt

    # Run only specific datasets:
    python -m dinov3_hrdecoder_pipeline.inference.run_pipeline \
        --datasets NAGUL_450171_MADASE_450172_GHOTPAL_450137_ORTHO

    # Skip the stitching stage:
    python -m dinov3_hrdecoder_pipeline.inference.run_pipeline --skip-stitch
"""

import os
import sys
import csv
import gc
import argparse
import time
import traceback
from pathlib import Path
from datetime import datetime
from collections import defaultdict

import numpy as np
import torch
import yaml
import rasterio
import pytorch_lightning as pl

PIPE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PIPE_ROOT.parent))

from dinov3_hrdecoder_pipeline.models.lightning_module import DINOv3HRDecoderModule
from dinov3_hrdecoder_pipeline.models.dataset import TileTestDataset
from dinov3_hrdecoder_pipeline.inference.stitch_and_evaluate import (
    stitch_predictions, evaluate_tile_predictions, save_eval_report
)
from dinov3_hrdecoder_pipeline.inference._timing import StepTimer, now_hms

DEFAULT_OUTPUT_DIR = str(PIPE_ROOT.parent / "outputs")
# Shipped submission checkpoints live in <bundle>/pretrained/.
# New training-run checkpoints land in <bundle>/checkpoints/.
DEFAULT_CHECKPOINT = str(
    PIPE_ROOT.parent / "pretrained"
    / "dinov3_hrdecoder_full_best_loss=0.0615.ckpt"
)


# ═════════════════════════════════════════════════════════════════════════════
#  Tee logger — mirrors all stdout/stderr to a log file
# ═════════════════════════════════════════════════════════════════════════════

class TeeLogger:
    def __init__(self, log_path: str, mode: str = "w", stream=None):
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        self._file = open(log_path, mode, encoding="utf-8", buffering=1)
        self._stream = stream or sys.__stdout__

    def write(self, data):
        self._stream.write(data)
        self._file.write(data)

    def flush(self):
        self._stream.flush()
        self._file.flush()

    def fileno(self):
        return self._stream.fileno()

    def isatty(self):
        return False

    def close(self):
        self._file.close()

    def __getattr__(self, name):
        return getattr(self._stream, name)


def setup_logging(output_dir: str) -> str:
    log_dir = os.path.join(output_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(log_dir, f"pipeline_run_{run_ts}.log")
    sys.stdout = TeeLogger(log_path, mode="w", stream=sys.__stdout__)
    sys.stderr = TeeLogger(log_path, mode="a", stream=sys.__stderr__)
    return log_path


# ═════════════════════════════════════════════════════════════════════════════
#  Discover datasets from tile index
# ═════════════════════════════════════════════════════════════════════════════

def discover_datasets(tile_index_path: str, masks_dir: str):
    """
    Group tile_index.csv rows by dataset. Datasets with ground-truth
    masks under `masks_dir/<dataset>/` are classified as 'train'; the
    rest are 'test'.
    """
    dataset_tiles = defaultdict(list)
    with open(tile_index_path) as f:
        for row in csv.DictReader(f):
            dataset_tiles[row["dataset"]].append(row)

    datasets = []
    for ds_name, tiles in sorted(dataset_tiles.items()):
        ds_mask_dir = os.path.join(masks_dir, ds_name)
        num_labeled = 0
        if os.path.isdir(ds_mask_dir):
            num_labeled = len([f for f in os.listdir(ds_mask_dir)
                               if f.endswith("_mask.tif")])
        datasets.append({
            "name": ds_name,
            "type": "train" if num_labeled > 0 else "test",
            "tiles": tiles,
            "num_labeled": num_labeled,
        })
    return datasets


# ═════════════════════════════════════════════════════════════════════════════
#  Stage 1: Run inference
# ═════════════════════════════════════════════════════════════════════════════

def save_prediction_geotiff(pred: np.ndarray, tile_path: str, output_dir: str):
    """Save prediction as georeferenced GeoTIFF."""
    with rasterio.open(tile_path) as src:
        profile = src.profile.copy()
    profile.update(count=1, dtype="uint8", nodata=0, compress="deflate")

    tile_name = Path(tile_path).stem
    pred_path = os.path.join(output_dir, f"{tile_name}_pred.tif")
    os.makedirs(os.path.dirname(pred_path), exist_ok=True)

    if pred.shape != (profile["height"], profile["width"]):
        from PIL import Image
        pred_img = Image.fromarray(pred.astype(np.uint8))
        pred_img = pred_img.resize((profile["width"], profile["height"]),
                                   Image.NEAREST)
        pred = np.array(pred_img)

    with rasterio.open(pred_path, "w", **profile) as dst:
        dst.write(pred[np.newaxis].astype(np.uint8))
    return pred_path


def run_inference(model, dataset_info: dict, cfg: dict, output_dir: str,
                  timer: "StepTimer | None" = None):
    """
    Manual batched inference: avoids accumulating all prediction tensors
    in memory (OOM on large datasets with 10k+ tiles).
    """
    ds_name = dataset_info["name"]
    pred_dir = os.path.join(output_dir, "predictions", ds_name)
    os.makedirs(pred_dir, exist_ok=True)

    if timer is None:
        timer = StepTimer()

    with timer.step("inference / build sample list", indent=6):
        test_samples = [{
            "tile_path": t["tile_path"],
            "dataset": ds_name,
            "row": int(t.get("row", 0)),
            "col": int(t.get("col", 0)),
        } for t in dataset_info["tiles"]]

    if not test_samples:
        print(f"    ⚠️  No tiles found for {ds_name}")
        return pred_dir, []

    with timer.step("inference / build dataloader", indent=6):
        tile_size = cfg["training"]["tile_size"]
        test_ds = TileTestDataset(test_samples, tile_size)
        test_loader = torch.utils.data.DataLoader(
            test_ds,
            batch_size=cfg["training"]["batch_size"],
            shuffle=False,
            num_workers=cfg["training"]["num_workers"],
            pin_memory=False,
        )

    device = next(model.parameters()).device
    pred_index = []
    total_tiles = len(test_ds)
    processed = 0

    forward_t = timer.cumulative("inference / forward pass per batch", indent=6)
    save_t    = timer.cumulative("inference / save_prediction_geotiff per tile", indent=6)

    model.eval()
    # FP16 autocast on the forward pass cuts ViT-L/16 latency by ~50 %
    # on Ampere/Ada GPUs. argmax is computed in FP32 on the host after
    # the cast back, so prediction values are unchanged.
    use_amp = (device.type == "cuda")
    amp_dtype = torch.float16
    with timer.step("inference / batch loop", indent=6):
        with torch.no_grad():
            for batch_idx, batch in enumerate(test_loader):
                images = batch["image"].to(device, non_blocking=True)
                tile_paths = batch["tile_path"]

                with forward_t:
                    if use_amp:
                        with torch.autocast(device_type="cuda", dtype=amp_dtype):
                            logits = model(images)
                        logits = logits.float()
                    else:
                        logits = model(images)
                    preds = logits.argmax(dim=1).cpu().numpy()

                with save_t:
                    for pred, tp in zip(preds, tile_paths):
                        pred_path = save_prediction_geotiff(pred, tp, pred_dir)
                        pred_index.append({"tile_path": tp, "pred_path": pred_path})

                processed += len(tile_paths)
                del images, logits, preds
                if (batch_idx + 1) % 50 == 0:
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    print(f"      [{now_hms()}] {processed}/{total_tiles} tiles predicted...")

    forward_t.flush()
    save_t.flush()

    with timer.step("inference / GPU cleanup + index csv", indent=6):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        idx_path = os.path.join(pred_dir, "prediction_index.csv")
        with open(idx_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["tile_path", "pred_path"])
            writer.writeheader()
            writer.writerows(pred_index)

    return pred_dir, pred_index


# ═════════════════════════════════════════════════════════════════════════════
#  Stage 2: Stitch tiles
# ═════════════════════════════════════════════════════════════════════════════

def run_stitch(tile_dir: str, tile_index_path: str, ds_name: str,
               output_dir: str, timer: "StepTimer | None" = None):
    stitch_dir = os.path.join(output_dir, "stitched")
    os.makedirs(stitch_dir, exist_ok=True)
    stitched_path = os.path.join(stitch_dir, f"{ds_name}_pred.tif")
    return stitch_predictions(
        tile_dir, tile_index_path, ds_name, stitched_path,
        source_type="predictions", timer=timer,
    )


# ═════════════════════════════════════════════════════════════════════════════
#  Stage 3: Evaluate (training datasets only)
# ═════════════════════════════════════════════════════════════════════════════

def run_evaluate(pred_dir: str, masks_dir: str, cfg: dict, ds_name: str,
                 output_dir: str, timer: "StepTimer | None" = None):
    if timer is None:
        timer = StepTimer()
    eval_dir = os.path.join(output_dir, "evaluation")
    os.makedirs(eval_dir, exist_ok=True)
    ds_masks_dir = os.path.join(masks_dir, ds_name)

    pred_idx_path = os.path.join(pred_dir, "prediction_index.csv")
    metrics, summary = evaluate_tile_predictions(
        pred_idx_path, ds_masks_dir, cfg,
        pred_key="pred_path", label=f"DINOv3+HRDecoder ({ds_name})",
        timer=timer,
    )
    if not metrics:
        return None

    save_eval_report(
        metrics,
        os.path.join(eval_dir, f"{ds_name}_metrics.csv"),
        "DINOv3+HRDecoder",
    )

    report_path = os.path.join(eval_dir, f"{ds_name}_report.txt")
    with open(report_path, "w") as f:
        f.write(f"Evaluation Report: {ds_name}\n")
        f.write("=" * 70 + "\n\n")
        if summary:
            f.write(summary + "\n\n")

    return metrics


# ═════════════════════════════════════════════════════════════════════════════
#  Skip check
# ═════════════════════════════════════════════════════════════════════════════

def is_dataset_complete(ds_name: str, ds_type: str, output_dir: str,
                        skip_stitch: bool) -> tuple:
    missing = []
    pred_idx = os.path.join(output_dir, "predictions", ds_name,
                            "prediction_index.csv")
    if not os.path.exists(pred_idx):
        missing.append(f"predictions/{ds_name}/prediction_index.csv")

    if not skip_stitch:
        pred_tif = os.path.join(output_dir, "stitched", f"{ds_name}_pred.tif")
        if not os.path.exists(pred_tif):
            missing.append(f"stitched/{ds_name}_pred.tif")

    if ds_type == "train":
        metrics_csv = os.path.join(output_dir, "evaluation",
                                   f"{ds_name}_metrics.csv")
        if not os.path.exists(metrics_csv):
            missing.append(f"evaluation/{ds_name}_metrics.csv")

    return len(missing) == 0, missing


# ═════════════════════════════════════════════════════════════════════════════
#  Summary report
# ═════════════════════════════════════════════════════════════════════════════

def write_summary_report(all_results: dict, output_dir: str):
    eval_dir = os.path.join(output_dir, "evaluation")
    os.makedirs(eval_dir, exist_ok=True)
    summary_path = os.path.join(eval_dir, "summary_report.csv")
    metric_keys = ["mIoU", "mF1", "mPrecision", "mRecall", "overall_accuracy"]

    with open(summary_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["dataset"] + metric_keys)
        for ds_name, metrics in sorted(all_results.items()):
            if not metrics:
                continue
            row = [ds_name] + [f"{metrics.get(k, 0):.6f}" for k in metric_keys]
            writer.writerow(row)

    print(f"\n  📊 Summary report saved → {summary_path}")
    return summary_path


# ═════════════════════════════════════════════════════════════════════════════
#  Checkpoint ↔ config architecture guard
# ═════════════════════════════════════════════════════════════════════════════

def _checkpoint_decoder_type(ckpt_path: str):
    """Infer a checkpoint's decoder architecture from its state-dict keys.

    Returns 'hrdecoder', 'upernet', 'segformer', or None if undetermined.
    Uses mmap so only the pickle header (key names) is read, not the multi-GB
    tensor payload — cheap even for big checkpoints.
    """
    try:
        import torch
        try:
            ckpt = torch.load(ckpt_path, map_location="cpu", mmap=True,
                              weights_only=False)
        except TypeError:                      # older torch: no mmap/weights_only
            ckpt = torch.load(ckpt_path, map_location="cpu")
        sd = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else {}
        keys = list(sd.keys())
    except Exception as exc:
        print(f"  ⚠ could not inspect checkpoint architecture ({exc}); "
              f"skipping compatibility check")
        return None

    def has(sub: str) -> bool:
        return any(sub in k for k in keys)

    # HRDecoder: multi-scale fusion + seg_head.  UPerNet: PSP + FPN + cls_seg.
    if has("decoder.seg_head") and has("decoder.fusion."):
        return "hrdecoder"
    if has("decoder.cls_seg") and (has("decoder.psp")
                                   or has("decoder.fpn_convs")
                                   or has("decoder.lateral_convs")):
        return "upernet"
    if has("decoder.linear_fuse") or has("decoder.linear_c"):
        return "segformer"
    return None


def _assert_decoder_matches(ckpt_path: str, cfg: dict) -> None:
    """Refuse a checkpoint whose decoder architecture doesn't match the
    config's `model.decoder.type`.

    Without this, `load_from_checkpoint(strict=False)` happily loads a
    mismatched checkpoint by random-initialising every layer that doesn't
    line up (including the segmentation head) — the model then predicts
    garbage that vectorises to nothing. Fail fast with a clear message
    instead of burning a full inference pass on a doomed run.
    """
    cfg_type = (cfg.get("model", {}).get("decoder", {}) or {}).get(
        "type", "hrdecoder").lower()
    ckpt_type = _checkpoint_decoder_type(ckpt_path)
    if ckpt_type and ckpt_type != cfg_type:
        print(f"\n{'='*70}")
        print(f"  ❌ ERROR: checkpoint / config decoder mismatch")
        print(f"     checkpoint : {ckpt_path}")
        print(f"                  weights look like a '{ckpt_type}' decoder")
        print(f"     config     : model.decoder.type = '{cfg_type}'")
        print(f"     This checkpoint was trained with a DIFFERENT decoder and "
              f"is not")
        print(f"     compatible with the '{cfg_type}' pipeline — loading it "
              f"would random-")
        print(f"     initialise the segmentation head and predict garbage.")
        print(f"     Pick a '{cfg_type}' checkpoint instead.")
        print(f"{'='*70}\n")
        sys.exit(2)


# ═════════════════════════════════════════════════════════════════════════════
#  Main
# ═════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Run inference pipeline on all datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config",
                        default=str(PIPE_ROOT / "configs" / "train_full.yaml"),
                        help="Path to config YAML")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT,
                        help="Model checkpoint path")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR,
                        help="Base output directory")
    parser.add_argument("--datasets", nargs="+", default=None, metavar="NAME",
                        help="Dataset name(s) to run (space-separated)")
    parser.add_argument("--datasets-file", default=None, metavar="FILE",
                        help="Text file with one dataset name per line")
    parser.add_argument("--skip-stitch", action="store_true",
                        help="Skip the stitching stage")
    parser.add_argument("--force", action="store_true",
                        help="Re-process datasets even if outputs exist")
    args = parser.parse_args()

    filter_names = set(args.datasets) if args.datasets else set()
    if args.datasets_file:
        with open(args.datasets_file) as _f:
            for _line in _f:
                name = _line.strip()
                if name and not name.startswith("#"):
                    filter_names.add(name)
    if filter_names:
        print(f"  Dataset filter: {len(filter_names)} name(s) specified")

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    log_path = setup_logging(output_dir)

    tile_index_path = cfg["paths"]["tile_index"]
    masks_dir = cfg["paths"]["labels_dir"]

    print("=" * 70)
    print("  DINOv3 + HRDecoder — Inference Pipeline")
    print(f"  Log file:    {log_path}")
    print(f"  Config:      {args.config}")
    print(f"  Checkpoint:  {args.checkpoint}")
    print(f"  Output dir:  {output_dir}")
    print(f"  Skip Stitch: {args.skip_stitch}")
    print("=" * 70)

    all_datasets = discover_datasets(tile_index_path, masks_dir)
    if filter_names:
        all_datasets = [d for d in all_datasets if d["name"] in filter_names]
        found = {d["name"] for d in all_datasets}
        not_found = filter_names - found
        if not_found:
            print(f"  ⚠️  {len(not_found)} name(s) not found in tile index:")
            for n in sorted(not_found):
                print(f"       • {n!r}")

    train_ds = [d for d in all_datasets if d["type"] == "train"]
    test_ds = [d for d in all_datasets if d["type"] == "test"]

    print(f"\n  Found {len(all_datasets)} datasets:")
    print(f"    Training (with labels): {len(train_ds)}")
    for d in train_ds:
        print(f"      - {d['name']} ({len(d['tiles'])} tiles, "
              f"{d['num_labeled']} labeled)")
    print(f"    Test (no labels):       {len(test_ds)}")
    for d in test_ds:
        print(f"      - {d['name']} ({len(d['tiles'])} tiles)")

    # Master timer — used across model load and all dataset stages
    timer = StepTimer()

    print(f"\n{'='*70}")
    print(f"  [{now_hms()}] Loading DINOv3+HRDecoder model...")
    print(f"{'='*70}")
    # Guard against an architecture-mismatched checkpoint BEFORE the
    # (expensive) model build + inference — a UPerNet checkpoint loaded
    # into the HRDecoder config would silently predict garbage.
    _assert_decoder_matches(args.checkpoint, cfg)
    with timer.step("model load_from_checkpoint", indent=4):
        model = DINOv3HRDecoderModule.load_from_checkpoint(
            args.checkpoint, cfg=cfg, strict=False
        )
        model.eval()
    print("  ✅ Model loaded")

    all_eval_results = {}
    total_start = time.time()
    per_dataset_durations = []  # list of (name, stage1, stage2, stage3, total)

    for idx, ds_info in enumerate(all_datasets, 1):
        ds_name = ds_info["name"]
        ds_type = ds_info["type"]
        ds_start = time.time()

        print(f"\n{'='*70}")
        print(f"  [{now_hms()}] [{idx}/{len(all_datasets)}] {ds_name}")
        print(f"  Type: {ds_type.upper()} | Tiles: {len(ds_info['tiles'])} | "
              f"Labeled: {ds_info['num_labeled']}")
        print(f"{'='*70}")

        if not args.force:
            complete, missing = is_dataset_complete(
                ds_name, ds_type, output_dir, skip_stitch=args.skip_stitch
            )
            if complete:
                print(f"  ⏩ SKIPPING — all outputs already exist")
                continue
            print(f"  ⏳ Missing {len(missing)} output(s), will process:")
            for m in missing:
                print(f"       • {m}")

        stage1_dt = stage2_dt = stage3_dt = 0.0
        try:
            # ── Stage 1: Inference ──────────────────────────────────────────
            print(f"\n  📌 Stage 1/3: Running inference...")
            t_s1 = time.time()
            with timer.step(f"[{ds_name}] Stage 1 — inference", indent=4):
                pred_dir, pred_index = run_inference(
                    model, ds_info, cfg, output_dir, timer=timer
                )
            stage1_dt = time.time() - t_s1
            print(f"    ✅ {len(pred_index)} predictions saved to {pred_dir}")

            if not pred_index:
                print(f"    ⚠️  No predictions generated, skipping remaining stages")
                continue

            # ── Stage 2: Stitch ─────────────────────────────────────────────
            if not args.skip_stitch:
                print(f"\n  📌 Stage 2/3: Stitching tiles...")
                t_s2 = time.time()
                with timer.step(f"[{ds_name}] Stage 2 — stitching", indent=4):
                    pred_stitched = run_stitch(
                        pred_dir, tile_index_path, ds_name, output_dir,
                        timer=timer,
                    )
                stage2_dt = time.time() - t_s2
                if pred_stitched:
                    print(f"    ✅ Stitched → {pred_stitched}")
                else:
                    print(f"    ⚠️  Stitching failed")
            else:
                print(f"\n  📌 Stage 2/3: Stitching — SKIPPED")

            # ── Stage 3: Evaluate (training datasets only) ──────────────────
            if ds_type == "train":
                print(f"\n  📌 Stage 3/3: Evaluating against ground truth...")
                t_s3 = time.time()
                with timer.step(f"[{ds_name}] Stage 3 — evaluation", indent=4):
                    metrics = run_evaluate(
                        pred_dir, masks_dir, cfg, ds_name, output_dir, timer=timer
                    )
                stage3_dt = time.time() - t_s3
                all_eval_results[ds_name] = metrics
                if metrics:
                    print(f"    mIoU={metrics['mIoU']:.4f}  "
                          f"mF1={metrics['mF1']:.4f}  "
                          f"mPrec={metrics['mPrecision']:.4f}  "
                          f"mRecall={metrics['mRecall']:.4f}  "
                          f"Accuracy={metrics['overall_accuracy']:.4f}")
            else:
                print(f"\n  📌 Stage 3/3: Evaluation — SKIPPED (no labels)")

            ds_total = time.time() - ds_start
            per_dataset_durations.append(
                (ds_name, stage1_dt, stage2_dt, stage3_dt, ds_total)
            )
            print(f"\n  ✅ [{now_hms()}] {ds_name} completed in {ds_total:.1f}s "
                  f"(infer={stage1_dt:.1f}s, stitch={stage2_dt:.1f}s, eval={stage3_dt:.1f}s)")

        except Exception as e:
            print(f"\n  ❌ ERROR processing {ds_name}: {e}")
            traceback.print_exc()

        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    if all_eval_results:
        write_summary_report(all_eval_results, output_dir)

    total_elapsed = time.time() - total_start
    print(f"\n{'='*70}")
    print(f"  PIPELINE COMPLETE")
    print(f"{'='*70}")
    print(f"  Datasets processed: {len(all_datasets)}")
    print(f"    Training:  {len(train_ds)} (with evaluation)")
    print(f"    Test:      {len(test_ds)} (predictions only)")
    print(f"  Output dir:  {output_dir}")
    print(f"  Total time:  {total_elapsed/60:.1f} minutes")

    if all_eval_results:
        print(f"\n  📊 Training Dataset Metrics:")
        print(f"  {'Dataset':<50s} {'mIoU':>8s} {'mF1':>8s} "
              f"{'mPrec':>8s} {'mRecall':>9s} {'Accuracy':>10s}")
        print(f"  {'─'*95}")
        for ds_name, m in sorted(all_eval_results.items()):
            if not m:
                continue
            print(f"  {ds_name:<50s} {m['mIoU']:>8.4f} "
                  f"{m['mF1']:>8.4f} {m['mPrecision']:>8.4f} "
                  f"{m['mRecall']:>9.4f} {m['overall_accuracy']:>10.4f}")

    # ── Per-dataset stage-duration table ────────────────────────────────────
    if per_dataset_durations:
        print(f"\n  ⏱  Per-dataset stage durations (seconds):")
        print(f"  {'Dataset':<50s} {'Infer':>8s} {'Stitch':>8s} "
              f"{'Eval':>8s} {'Total':>9s}")
        print(f"  {'─'*87}")
        sum1 = sum2 = sum3 = sumT = 0.0
        for name, s1, s2, s3, st in per_dataset_durations:
            print(f"  {name:<50s} {s1:>8.1f} {s2:>8.1f} {s3:>8.1f} {st:>9.1f}")
            sum1 += s1; sum2 += s2; sum3 += s3; sumT += st
        print(f"  {'─'*87}")
        print(f"  {'TOTAL':<50s} {sum1:>8.1f} {sum2:>8.1f} {sum3:>8.1f} {sumT:>9.1f}")

    # ── Detailed sub-step timing table (everything the StepTimer recorded) ──
    timer.print_summary(title="Detailed step-wise timing")

    print(f"\n  Output structure:")
    print(f"    {output_dir}/")
    print(f"    ├── predictions/   (per-tile DINOv3 predictions)")
    print(f"    ├── stitched/      (stitched full-extent GeoTIFFs)")
    print(f"    ├── evaluation/    (metrics CSVs + summary report)")
    print(f"    └── logs/          (pipeline run logs)")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
