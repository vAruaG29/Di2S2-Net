#!/usr/bin/env python3
"""
PyTorch Lightning Module: DINOv3 + HRDecoder Segmentor
=======================================================
"""
import os
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
import yaml
from pathlib import Path

from .encoder import DINOv3Encoder
from .decoder import HRDecoder, build_decoder
from .losses import CombinedLoss
from .metrics import SegmentationMetrics

PIPE_ROOT = Path(__file__).resolve().parent.parent
CFG_PATH = PIPE_ROOT / "configs" / "train.yaml"


def load_config():
    with open(CFG_PATH) as f:
        return yaml.safe_load(f)


class DINOv3HRDecoderModule(pl.LightningModule):
    """
    DINOv3 encoder + HRDecoder for semantic segmentation.

    Training:
        - Multi-scale loss: LR + HR + Fuse (weighted)
        - Edge-aware boundary loss
        - Differential LR for encoder vs decoder
    """

    def __init__(self, cfg: dict = None):
        super().__init__()
        if cfg is None:
            cfg = load_config()
        self.cfg = cfg
        self.learning_rate = cfg["training"]["learning_rate"]
        self.save_hyperparameters(cfg)

        enc_cfg = cfg["model"]["encoder"]
        dec_cfg = cfg["model"]["decoder"]
        loss_cfg = cfg["loss"]
        self.num_classes = len(cfg["classes"]) + 1  # +1 for background (class 0)

        # Encoder
        self.encoder = DINOv3Encoder(
            arch=enc_cfg["arch"],
            intermediate_layers=enc_cfg["intermediate_layers"],
            freeze_first_n_blocks=enc_cfg["freeze_first_n_blocks"],
            embed_dim=enc_cfg["embed_dim"],
            out_dim=dec_cfg["hidden_dim"],
            pretrained=enc_cfg.get("pretrained", True),
            weights=enc_cfg.get("weights", "LVD1689M"),
        )

        patch_size = self.encoder.patch_size
        tile_size = cfg["training"]["tile_size"]

        # Decoder — supports multiple types
        decoder_type = dec_cfg.get("type", "hrdecoder")
        common_kwargs = dict(
            in_dim=dec_cfg["hidden_dim"],
            num_scales=len(enc_cfg["intermediate_layers"]),
            hidden_dim=dec_cfg["hidden_dim"],
            num_classes=self.num_classes,
            patch_size=patch_size,
        )

        if decoder_type == "hrdecoder":
            # hr_crop_size is in PIXEL SPACE (e.g., 256×256 from a 512×512 tile)
            hr_crop = dec_cfg.get("hr_crop_size", dec_cfg.get("hr_scale", [256, 256]))
            self.decoder = build_decoder(
                decoder_type,
                **common_kwargs,
                hr_crop_size=tuple(hr_crop),
                crop_num=dec_cfg["crop_num"],
                scale_ratio=tuple(dec_cfg["scale_ratio"]),
                divisible=dec_cfg["divisible"],
            )
        else:
            # UPerNet, SegFormer, SkipDecoder — pass any extra decoder params
            extra = {k: v for k, v in dec_cfg.items()
                     if k not in ("type", "hidden_dim", "hr_scale", "lr_scale",
                                  "hr_loss_weight", "lr_loss_weight", "fuse_loss_weight",
                                  "crop_num", "scale_ratio", "divisible", "hr_crop_size")}
            self.decoder = build_decoder(decoder_type, **common_kwargs, **extra)

        # Loss
        self.criterion = CombinedLoss(
            num_classes=self.num_classes,
            bce_weight=loss_cfg["bce_weight"],
            dice_weight=loss_cfg["dice_weight"],
            edge_weight=loss_cfg["edge_weight"],
            edge_kernel_size=loss_cfg["edge_kernel_size"],
        )

        # Loss weights for multi-scale (only used by HRDecoder)
        self.hr_loss_weight = dec_cfg.get("hr_loss_weight", 0.1)
        self.lr_loss_weight = dec_cfg.get("lr_loss_weight", 0.5)
        self.fuse_loss_weight = dec_cfg.get("fuse_loss_weight", 1.0)
        self.decoder_type = dec_cfg.get("type", "hrdecoder")

        # Metrics
        class_names = [c["name"] for c in cfg["classes"]]
        self.val_metrics = SegmentationMetrics(
            self.num_classes, ["background"] + class_names
        )
        self.train_metrics = SegmentationMetrics(
            self.num_classes, ["background"] + class_names
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Inference forward → fused logits."""
        features = self.encoder(x)
        target_size = (x.shape[2], x.shape[3])
        logits = self.decoder.forward_test(features, target_size)
        return logits

    def training_step(self, batch, batch_idx):
        images = batch["image"]
        masks = batch["mask"]
        target_size = (images.shape[2], images.shape[3])

        features = self.encoder(images)
        dec_out = self.decoder.forward_train(features, target_size)

        if self.decoder_type == "hrdecoder":
            # Multi-scale loss: LR + HR + Fuse
            lr_losses = self.criterion(dec_out["lr_logits"], masks)
            fuse_losses = self.criterion(dec_out["fuse_logits"], masks)

            # HR loss: crop GT to match HR crops
            hr_logits = dec_out["hr_logits"]
            hr_bboxes = dec_out["hr_bboxes"]
            hr_scale = dec_out["hr_scale"]
            B = images.shape[0]

            hr_masks_list = []
            for bbox in hr_bboxes:
                y1, y2, x1, x2 = bbox
                cropped = masks[:, y1:y2, x1:x2]
                cropped = F.interpolate(
                    cropped.unsqueeze(1).float(), size=hr_scale,
                    mode="nearest"
                ).squeeze(1).long()
                hr_masks_list.append(cropped)
            hr_masks = torch.cat(hr_masks_list, dim=0)
            hr_losses = self.criterion(hr_logits, hr_masks)

            total_loss = (
                self.lr_loss_weight * lr_losses["loss"] +
                self.hr_loss_weight * hr_losses["loss"] +
                self.fuse_loss_weight * fuse_losses["loss"]
            )

            self.log("train/loss", total_loss, on_step=False, on_epoch=True, prog_bar=True)
            self.log("train/ce", fuse_losses["ce_loss"], on_step=False, on_epoch=True)
            self.log("train/dice", fuse_losses["dice_loss"], on_step=False, on_epoch=True)
            self.log("train/edge", fuse_losses["edge_loss"], on_step=False, on_epoch=True)
            self.log("train/lr_loss", lr_losses["loss"], on_step=False, on_epoch=True)
            self.log("train/hr_loss", hr_losses["loss"], on_step=False, on_epoch=True)
            
            # Update train mIoU using fuse_logits
            preds = dec_out["fuse_logits"].argmax(dim=1)
        else:
            # Simple decoders: single fuse_logits output
            losses = self.criterion(dec_out["fuse_logits"], masks)
            total_loss = losses["loss"]

            self.log("train/loss", total_loss, on_step=False, on_epoch=True, prog_bar=True)
            self.log("train/ce", losses["ce_loss"], on_step=False, on_epoch=True)
            self.log("train/dice", losses["dice_loss"], on_step=False, on_epoch=True)
            self.log("train/edge", losses["edge_loss"], on_step=False, on_epoch=True)
            
            # Update train mIoU using fuse_logits
            preds = dec_out["fuse_logits"].argmax(dim=1)

        self.train_metrics.update(preds, masks)
        return total_loss

    def on_train_epoch_end(self):
        metrics = self.train_metrics.compute()
        for key, val in metrics.items():
            self.log(f"train/{key}", val, sync_dist=True)
            
        summary = f"\n{'='*60}\n  Epoch {self.current_epoch} - TRAINING Metrics\n" + self.train_metrics.summary_str()
        print(summary)
        self._save_summary_to_file(summary)
        
        self.train_metrics.reset()

    def validation_step(self, batch, batch_idx):
        images = batch["image"]
        masks = batch["mask"]

        logits = self(images)
        losses = self.criterion(logits, masks)

        self.log("val/loss", losses["loss"], on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val/ce", losses["ce_loss"], on_step=False, on_epoch=True)
        self.log("val/dice", losses["dice_loss"], on_step=False, on_epoch=True)
        self.log("val/edge", losses["edge_loss"], on_step=False, on_epoch=True)

        # Update metrics
        preds = logits.argmax(dim=1)
        self.val_metrics.update(preds, masks)

        return losses["loss"]

    def on_validation_epoch_end(self):
        metrics = self.val_metrics.compute()
        for key, val in metrics.items():
            self.log(f"val/{key}", val, sync_dist=True)
            
        summary = f"\n{'='*60}\n  Epoch {self.current_epoch} - VALIDATION Metrics\n" + self.val_metrics.summary_str()
        print(summary)
        self._save_summary_to_file(summary)
        
        self.val_metrics.reset()

    def _save_summary_to_file(self, summary: str):
        """Append the summary string to a text file in the active log directory."""
        if not self.trainer.is_global_zero:
            return
            
        # Get the directory where TensorBoard/CSV logs go
        if self.trainer.loggers:
            # Typically taking the first logger's directory (e.g., TensorBoardLogger)
            log_dir = self.trainer.loggers[0].log_dir or self.trainer.loggers[0].save_dir
            if getattr(self.trainer.loggers[0], "name", None):
                # The actual run folder we set up in train.py is inside loggers[0].save_dir
                # but if we passed log_dir = "..." in train.py, the logger often puts it in run_id directly
                pass
        else:
            log_dir = self.trainer.default_root_dir
            
        if hasattr(self, "cfg") and "paths" in self.cfg and "log_dir" in self.cfg["paths"]:
            # Fallback to the run_id directory if we can find it
            # Actually, the logger's directory is exactly the run_id log_dir we passed in train.py
            pass

        # Since we initialized the logger with `log_dir=run_id`, the save_dir is that folder
        # Let's cleanly just use the first logger's root dir
        log_dir = self.trainer.loggers[0].save_dir if self.trainer.loggers else self.trainer.default_root_dir
        
        os.makedirs(log_dir, exist_ok=True)
        out_path = os.path.join(log_dir, "epoch_metrics_summary.txt")
        with open(out_path, "a") as f:
            f.write(summary + "\n")

    def predict_step(self, batch, batch_idx):
        images = batch["image"]
        logits = self(images)
        probs = torch.softmax(logits, dim=1)
        preds = probs.argmax(dim=1)
        return {
            "preds": preds,
            "probs": probs,
            "tile_path": batch["tile_path"],
            "dataset": batch["dataset"],
        }

    def configure_optimizers(self):
        train_cfg = self.cfg["training"]
        lr = self.learning_rate  # Use self.learning_rate so LR finder can update it
        enc_mult = train_cfg["encoder_lr_multiplier"]

        # Differential LR
        param_groups = self.encoder.get_param_groups(enc_mult, lr)
        param_groups.append({
            "params": self.decoder.parameters(),
            "lr": lr,
        })

        optimizer = torch.optim.AdamW(
            param_groups,
            weight_decay=train_cfg["weight_decay"],
        )

        # Cosine scheduler with warmup
        total_steps = self.trainer.estimated_stepping_batches
        warmup_steps = int(train_cfg["warmup_epochs"] *
                           total_steps / train_cfg["max_epochs"])

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total_steps - warmup_steps, eta_min=1e-7
        )

        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.01, total_iters=warmup_steps
        )

        combined = torch.optim.lr_scheduler.SequentialLR(
            optimizer, [warmup_scheduler, scheduler], milestones=[warmup_steps]
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": combined,
                "interval": "step",
            },
        }
