#!/usr/bin/env python3
"""
Test DINOv3 + HRDecoder on NAGUL Tiles
========================================
Runs inference on test tiles, saves predictions, computes metrics.

Usage:
    cd <bundle root>
    python -m dinov3_hrdecoder_pipeline.inference.infer --checkpoint <path>
"""

import os
import sys
import argparse
import csv
from pathlib import Path

import numpy as np
import torch
import yaml
import rasterio
import pytorch_lightning as pl

PIPE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PIPE_ROOT.parent))

from dinov3_hrdecoder_pipeline.models.lightning_module import DINOv3HRDecoderModule
from dinov3_hrdecoder_pipeline.models.dataset import SegDataModule


def save_prediction_geotiff(pred: np.ndarray, tile_path: str, output_dir: str):
    """Save prediction as georeferenced GeoTIFF."""
    with rasterio.open(tile_path) as src:
        profile = src.profile.copy()

    profile.update(count=1, dtype="uint8", nodata=0, compress="deflate")

    tile_name = Path(tile_path).stem
    pred_path = os.path.join(output_dir, f"{tile_name}_pred.tif")
    os.makedirs(os.path.dirname(pred_path), exist_ok=True)

    # Resize prediction to original tile size if needed
    if pred.shape != (profile["height"], profile["width"]):
        from PIL import Image
        pred_img = Image.fromarray(pred.astype(np.uint8))
        pred_img = pred_img.resize((profile["width"], profile["height"]),
                                    Image.NEAREST)
        pred = np.array(pred_img)

    with rasterio.open(pred_path, "w", **profile) as dst:
        dst.write(pred[np.newaxis].astype(np.uint8))

    return pred_path


def main():
    parser = argparse.ArgumentParser(description="Test DINOv3+HRDecoder")
    parser.add_argument("--config", default=str(PIPE_ROOT / "configs" / "train.yaml"))
    parser.add_argument("--checkpoint", required=True, help="Model checkpoint path")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    print("=" * 70)
    print("  DINOv3 + HRDecoder Testing on NAGUL Tiles")
    print(f"  Checkpoint: {args.checkpoint}")
    print("=" * 70)

    # Load model
    model = DINOv3HRDecoderModule.load_from_checkpoint(
        args.checkpoint, cfg=cfg, strict=False
    )
    model.eval()

    # DataModule
    dm = SegDataModule(cfg)
    dm.setup(stage="test")

    # Output directory
    pred_dir = os.path.join(cfg["paths"]["output_dir"], "predictions",
                            cfg["test_dataset"])
    os.makedirs(pred_dir, exist_ok=True)

    # Run inference
    trainer = pl.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        precision=cfg["training"]["precision"],
    )

    predictions = trainer.predict(model, dm.test_dataloader())

    # Save predictions
    pred_index = []
    for batch_pred in predictions:
        preds = batch_pred["preds"].cpu().numpy()
        tile_paths = batch_pred["tile_path"]

        for i, (pred, tp) in enumerate(zip(preds, tile_paths)):
            pred_path = save_prediction_geotiff(pred, tp, pred_dir)
            pred_index.append({
                "tile_path": tp,
                "pred_path": pred_path,
            })

    # Save prediction index
    idx_path = os.path.join(pred_dir, "prediction_index.csv")
    with open(idx_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["tile_path", "pred_path"])
        writer.writeheader()
        writer.writerows(pred_index)

    print(f"\n{'='*70}")
    print(f"  TESTING COMPLETE")
    print(f"  Predictions:    {len(pred_index)} tiles")
    print(f"  Output dir:     {pred_dir}")
    print(f"  Pred index:     {idx_path}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
