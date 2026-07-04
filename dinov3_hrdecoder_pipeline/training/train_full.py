#!/usr/bin/env python3
"""
Train DINOv3 + HRDecoder — FULL TRAINING (All Data, No Validation)
====================================================================
Trains on ALL labeled tiles for final submission. No validation set.
Checkpoints are saved periodically and by best training loss.

Usage:
    cd <bundle root>
    python -m dinov3_hrdecoder_pipeline.training.train_full
    python -m dinov3_hrdecoder_pipeline.training.train_full --resume /path/to/best_checkpoint.ckpt
    python -m dinov3_hrdecoder_pipeline.training.train_full --gpus 2
"""

import os
import sys
import argparse
from pathlib import Path
from datetime import datetime

import yaml
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    ModelCheckpoint, LearningRateMonitor, RichProgressBar
)
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger

# Add project root to path
PIPE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PIPE_ROOT.parent))

from dinov3_hrdecoder_pipeline.models.lightning_module import DINOv3HRDecoderModule
from dinov3_hrdecoder_pipeline.models.dataset import SegDataModule


def main():
    parser = argparse.ArgumentParser(
        description="Train DINOv3+HRDecoder on ALL data (no validation)")
    parser.add_argument("--config",
                        default=str(PIPE_ROOT / "configs" / "train_full.yaml"))
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume from checkpoint (e.g. best model from normal training)")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override max_epochs from config")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    train_cfg = cfg["training"]
    max_epochs = args.epochs or train_cfg["max_epochs"]
    save_every = train_cfg.get("save_every_n_epochs", 5)

    print("=" * 70)
    print("  DINOv3 + HRDecoder — FULL TRAINING (Final Submission)")
    print("  ⚠️  No validation — training on ALL labeled tiles")
    print(f"  Epochs:     {max_epochs}")
    print(f"  Batch size: {train_cfg['batch_size']}")
    print(f"  Tile size:  {train_cfg['tile_size']}x{train_cfg['tile_size']}")
    print(f"  Precision:  {train_cfg['precision']}")
    print(f"  Save every: {save_every} epochs")
    if args.resume:
        print(f"  Resume from: {args.resume}")
    print("=" * 70)

    # DataModule (will use full_training mode from config)
    dm = SegDataModule(cfg)

    # Model
    model = DINOv3HRDecoderModule(cfg)

    # Callbacks
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    base_ckpt_dir = cfg["paths"]["checkpoint_dir"]
    ckpt_dir = os.path.join(base_ckpt_dir, f"full_train_{run_id}")
    decoder = cfg["model"]["decoder"]["type"].lower()
    os.makedirs(ckpt_dir, exist_ok=True)

    # Checkpoint: save every N epochs
    # periodic_ckpt = ModelCheckpoint(
    #     dirpath=ckpt_dir,
    #     filename=f"dinov3_{decoder}_full_epoch{{epoch:03d}}_loss{{train/loss:.4f}}",
    #     every_n_epochs=save_every,
    #     save_top_k=-1,  # Keep all periodic saves
    #     save_last=True,
    #     auto_insert_metric_name=False,
    #     save_on_train_epoch_end=True,  # Save at the end of the epoch with train metrics
    # )

    # Checkpoint: best by training loss
    best_loss_ckpt = ModelCheckpoint(
        dirpath=ckpt_dir,
        filename=f"dinov3_{decoder}_full_best_loss={{train/loss:.4f}}",
        monitor="train/loss",
        mode="min",
        save_top_k=1,
        save_last=True,
        auto_insert_metric_name=False,
        save_on_train_epoch_end=True,  # Save at the end of the epoch with train metrics
    )

    callbacks = [
        # periodic_ckpt,
        best_loss_ckpt,
        LearningRateMonitor(logging_interval="step"),
    ]

    try:
        callbacks.append(RichProgressBar())
    except Exception:
        pass

    # Note: NO EarlyStopping — we train for the full epoch budget

    # Loggers
    base_log_dir = cfg["paths"]["log_dir"]
    log_dir = os.path.join(base_log_dir, f"full_train_{run_id}")
    os.makedirs(log_dir, exist_ok=True)

    loggers = [
        TensorBoardLogger(log_dir, name="tensorboard"),
        CSVLogger(log_dir, name="csv_logs"),
    ]

    # Trainer — no validation
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator="gpu" if args.gpus > 0 else "cpu",
        devices=args.gpus if args.gpus > 0 else "auto",
        precision=train_cfg["precision"],
        accumulate_grad_batches=train_cfg["accumulate_grad_batches"],
        callbacks=callbacks,
        logger=loggers,
        check_val_every_n_epoch=0,  # Disable validation
        limit_val_batches=0,        # No validation batches
        gradient_clip_val=1.0,
        deterministic=False,
        enable_progress_bar=True,
    )

    # Train
    trainer.fit(model, dm, ckpt_path=args.resume)

    print(f"\n{'='*70}")
    print(f"  FULL TRAINING COMPLETE")
    print(f"  Epochs trained: {max_epochs}")
    print(f"  Checkpoints in: {ckpt_dir}")
    try:
        print(f"  Best loss model: {best_loss_ckpt.best_model_path}")
        print(f"  Best train loss: {best_loss_ckpt.best_model_score:.4f}")
    except Exception:
        pass
    print(f"  Last checkpoint: {os.path.join(ckpt_dir, 'last.ckpt')}")
    print(f"{'='*70}")
    print(f"\n  ℹ️  Use the last.ckpt or best_loss checkpoint for inference.")
    print(f"  Run test.py with --checkpoint to generate predictions.\n")


if __name__ == "__main__":
    main()
