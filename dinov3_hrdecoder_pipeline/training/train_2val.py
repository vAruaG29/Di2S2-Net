#!/usr/bin/env python3
"""
Train DINOv3 + HRDecoder — 2-image validation hold-out variant
================================================================
Identical to `train.py` except:
  - defaults to `configs/train_2val.yaml` (TIMMOWAL + NAGUL held out for
    validation; the other 8 labeled villages train), and
  - saves THREE "best" checkpoints per run:
        * max  val/mIoU
        * max  val/overall_accuracy
        * min  val/loss
    plus `last.ckpt`.

Everything lands under the config's (fresh) `checkpoint_dir` / `log_dir`
in a timestamped `run_<ts>/` sub-dir, so existing runs are never touched.

Usage:
    cd <bundle root>
    python -m dinov3_hrdecoder_pipeline.training.train_2val
    python -m dinov3_hrdecoder_pipeline.training.train_2val --gpus 2
    python -m dinov3_hrdecoder_pipeline.training.train_2val \
        --config dinov3_hrdecoder_pipeline/configs/train_2val.yaml
"""

import os
import sys
import argparse
from pathlib import Path

import yaml
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    ModelCheckpoint, EarlyStopping, LearningRateMonitor, RichProgressBar
)
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger

# Add project root to path
PIPE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PIPE_ROOT.parent))

from dinov3_hrdecoder_pipeline.models.lightning_module import DINOv3HRDecoderModule
from dinov3_hrdecoder_pipeline.models.dataset import SegDataModule


def main():
    parser = argparse.ArgumentParser(description="Train DINOv3+HRDecoder (2-val hold-out)")
    parser.add_argument("--config", default=str(PIPE_ROOT / "configs" / "train_2val.yaml"))
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    train_cfg = cfg["training"]

    print("=" * 70)
    print("  DINOv3 + HRDecoder Training Pipeline (2-image validation)")
    print(f"  Config:     {args.config}")
    print(f"  Val images: {cfg.get('test_datasets', cfg.get('test_dataset'))}")
    print(f"  Epochs:     {train_cfg['max_epochs']}")
    print(f"  Batch size: {train_cfg['batch_size']}")
    print(f"  Tile size:  {train_cfg['tile_size']}x{train_cfg['tile_size']}")
    print(f"  Precision:  {train_cfg['precision']}")
    print("=" * 70)

    # DataModule
    dm = SegDataModule(cfg)

    # Model
    model = DINOv3HRDecoderModule(cfg)

    # Callbacks
    from datetime import datetime
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    base_ckpt_dir = cfg["paths"]["checkpoint_dir"]
    ckpt_dir = os.path.join(base_ckpt_dir, f"run_{run_id}")
    decoder = cfg["model"]["decoder"]["type"].lower()
    os.makedirs(ckpt_dir, exist_ok=True)

    # 1) Best by mIoU (also keeps `last.ckpt`)
    miou_ckpt = ModelCheckpoint(
        dirpath=ckpt_dir,
        filename=f"dinov3_hrdecoder_{decoder}_best_miou_miou={{val/mIoU:.4f}}_acc={{val/overall_accuracy:.4f}}",
        monitor="val/mIoU",
        mode="max",
        save_top_k=1,
        save_last=True,
        auto_insert_metric_name=False,
    )

    # 2) Best by overall accuracy
    acc_ckpt = ModelCheckpoint(
        dirpath=ckpt_dir,
        filename=f"dinov3_hrdecoder_{decoder}_best_acc_miou={{val/mIoU:.4f}}_acc={{val/overall_accuracy:.4f}}",
        monitor="val/overall_accuracy",
        mode="max",
        save_top_k=1,
        save_last=False,
        auto_insert_metric_name=False,
    )

    # 3) Best by (minimum) validation loss
    loss_ckpt = ModelCheckpoint(
        dirpath=ckpt_dir,
        filename=f"dinov3_hrdecoder_{decoder}_best_loss_valloss={{val/loss:.4f}}_miou={{val/mIoU:.4f}}",
        monitor="val/loss",
        mode="min",
        save_top_k=1,
        save_last=False,
        auto_insert_metric_name=False,
    )

    callbacks = [
        miou_ckpt,
        acc_ckpt,
        loss_ckpt,
        EarlyStopping(
            monitor="val/mIoU",
            mode="max",
            patience=train_cfg["early_stopping_patience"],
            verbose=True,
        ),
        LearningRateMonitor(logging_interval="step"),
    ]

    try:
        callbacks.append(RichProgressBar())
    except Exception:
        pass

    # Loggers
    base_log_dir = cfg["paths"]["log_dir"]
    log_dir = os.path.join(base_log_dir, f"run_{run_id}")
    os.makedirs(log_dir, exist_ok=True)

    loggers = [
        TensorBoardLogger(log_dir, name="tensorboard"),
        CSVLogger(log_dir, name="csv_logs"),
    ]

    # Trainer
    trainer = pl.Trainer(
        max_epochs=train_cfg["max_epochs"],
        accelerator="gpu" if args.gpus > 0 else "cpu",
        devices=args.gpus if args.gpus > 0 else "auto",
        precision=train_cfg["precision"],
        accumulate_grad_batches=train_cfg["accumulate_grad_batches"],
        callbacks=callbacks,
        logger=loggers,
        check_val_every_n_epoch=1,  # validate after every epoch
        gradient_clip_val=1.0,
        deterministic=False,
    )

    # Train
    trainer.fit(model, dm, ckpt_path=args.resume)

    print(f"\n{'='*70}")
    print(f"  TRAINING COMPLETE")
    print(f"  Checkpoints in: {ckpt_dir}")
    for tag, cb in (("Best mIoU", miou_ckpt),
                    ("Best accuracy", acc_ckpt),
                    ("Best (min) val loss", loss_ckpt)):
        try:
            print(f"  {tag:22}: {cb.best_model_score:.4f}  →  {cb.best_model_path}")
        except Exception:
            print(f"  {tag:22}: (unavailable)")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
