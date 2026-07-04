#!/usr/bin/env python3
"""
Dataset & DataModule for DINOv3+HRDecoder Pipeline
===================================================
Reads tile images + masks, handles train/val/test split.
Test = NAGUL tiles, Train/Val = all other labeled tiles.
"""

import os
import csv
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
import rasterio
import yaml
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import random

PIPE_ROOT = Path(__file__).resolve().parent.parent
CFG_PATH = PIPE_ROOT / "configs" / "train.yaml"


def load_config():
    with open(CFG_PATH) as f:
        return yaml.safe_load(f)


class TileSegDataset(Dataset):
    """Loads tile images and corresponding masks."""

    def __init__(self, samples: list, num_classes: int, tile_size: int = 512,
                 transform=None):
        """
        Args:
            samples: list of dicts with 'tile_path' and 'mask_path'
            num_classes: number of segmentation classes
            tile_size: target size for resizing
            transform: albumentations transform
        """
        self.samples = samples
        self.num_classes = num_classes
        self.tile_size = tile_size
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def _read_image(self, path: str) -> np.ndarray:
        """Read raster as HWC uint8 array (first 3 bands → RGB)."""
        with rasterio.open(path) as src:
            bands = min(src.count, 3)
            data = src.read(list(range(1, bands + 1)))  # (C, H, W)
            if bands < 3:
                # Pad to 3 channels
                pad = np.zeros((3 - bands, data.shape[1], data.shape[2]), dtype=data.dtype)
                data = np.concatenate([data, pad], axis=0)
        img = np.transpose(data, (1, 2, 0))  # HWC
        if img.dtype != np.uint8:
            # Normalize to uint8
            if img.max() > 255:
                img = (img / img.max() * 255).astype(np.uint8)
            else:
                img = img.astype(np.uint8)
        return img

    def _read_mask(self, path: str) -> np.ndarray:
        """Read mask as HW uint8 array."""
        with rasterio.open(path) as src:
            mask = src.read(1)  # (H, W)
        return mask.astype(np.int64)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        img = self._read_image(sample["tile_path"])
        mask = self._read_mask(sample["mask_path"])

        # Resize
        img_pil = TF.to_pil_image(img)
        mask_pil = TF.to_pil_image(mask.astype(np.uint8))
        img_pil = TF.resize(img_pil, [self.tile_size, self.tile_size])
        mask_pil = TF.resize(mask_pil, [self.tile_size, self.tile_size],
                             interpolation=T.InterpolationMode.NEAREST)

        if self.transform == "train":
            # Random augmentations
            if random.random() > 0.5:
                img_pil = TF.hflip(img_pil)
                mask_pil = TF.hflip(mask_pil)
            if random.random() > 0.5:
                img_pil = TF.vflip(img_pil)
                mask_pil = TF.vflip(mask_pil)
            k = random.randint(0, 3)
            img_pil = TF.rotate(img_pil, k * 90)
            mask_pil = TF.rotate(mask_pil, k * 90)
            if random.random() > 0.7:
                img_pil = TF.adjust_brightness(img_pil, random.uniform(0.8, 1.2))
                img_pil = TF.adjust_contrast(img_pil, random.uniform(0.8, 1.2))

        # To tensor + normalize
        img = TF.to_tensor(img_pil)  # (3, H, W) float [0,1]
        img = TF.normalize(img, mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
        mask = torch.from_numpy(np.array(mask_pil)).long()

        return {
            "image": img,
            "mask": mask,
            "tile_path": sample["tile_path"],
            "dataset": sample.get("dataset", ""),
        }


class TileTestDataset(Dataset):
    """Test dataset — images only (no masks required)."""

    def __init__(self, samples: list, tile_size: int = 512, transform=None):
        self.samples = samples
        self.tile_size = tile_size
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def _read_image(self, path: str) -> np.ndarray:
        with rasterio.open(path) as src:
            bands = min(src.count, 3)
            data = src.read(list(range(1, bands + 1)))
            if bands < 3:
                pad = np.zeros((3 - bands, data.shape[1], data.shape[2]), dtype=data.dtype)
                data = np.concatenate([data, pad], axis=0)
        img = np.transpose(data, (1, 2, 0))
        if img.dtype != np.uint8:
            if img.max() > 255:
                img = (img / img.max() * 255).astype(np.uint8)
            else:
                img = img.astype(np.uint8)
        return img

    def __getitem__(self, idx):
        sample = self.samples[idx]
        img = self._read_image(sample["tile_path"])

        img_pil = TF.to_pil_image(img)
        img_pil = TF.resize(img_pil, [self.tile_size, self.tile_size])
        img = TF.to_tensor(img_pil)
        img = TF.normalize(img, mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])

        return {
            "image": img,
            "tile_path": sample["tile_path"],
            "dataset": sample.get("dataset", ""),
            "row": sample.get("row", 0),
            "col": sample.get("col", 0),
        }


class SegDataModule(pl.LightningDataModule):
    """
    Lightning DataModule that splits tiles:
    - Train: ALL labeled tiles NOT from test_dataset (no random val split)
    - Val:   Labeled test_dataset tiles (geographic hold-out)
    - Test:  ALL test_dataset tiles (for inference/stitching)

    When full_training=True or test_dataset="", ALL labeled tiles go to train.
    """

    def __init__(self, cfg: dict = None):
        super().__init__()
        if cfg is None:
            cfg = load_config()
        self.cfg = cfg
        self.tile_size = cfg["training"]["tile_size"]
        self.batch_size = cfg["training"]["batch_size"]
        self.num_workers = cfg["training"]["num_workers"]
        # Validation hold-out can be a SINGLE dataset (`test_dataset`,
        # back-compat) or a LIST of datasets (`test_datasets`) held out
        # together. The list wins when present; otherwise we fall back to
        # the single key so existing configs behave exactly as before.
        _val_list = cfg.get("test_datasets")
        if _val_list:
            self.val_dataset_names = [v for v in _val_list if v]
        else:
            _single = cfg.get("test_dataset", "")
            self.val_dataset_names = [_single] if _single else []
        # Human-readable string used only in the split print-outs below.
        self.test_dataset_name = ", ".join(self.val_dataset_names)
        self.num_classes = len(cfg["classes"])
        self.class_names = ["background"] + [c["name"] for c in cfg["classes"]]
        self.full_training = cfg.get("full_training", False) or not self.val_dataset_names

    def setup(self, stage=None):
        masks_dir = self.cfg["paths"]["labels_dir"]
        label_index_path = os.path.join(masks_dir, "label_index.csv")

        # Load tile index for tile metadata
        tile_index_path = self.cfg["paths"]["tile_index"]
        tile_meta = {}
        with open(tile_index_path) as f:
            for row in csv.DictReader(f):
                tile_meta[row["tile_path"]] = row

        train_samples = []
        val_samples = []
        test_samples = []

        with open(label_index_path) as f:
            for row in csv.DictReader(f):
                dataset = row["dataset"]

                if self.full_training:
                    # ── Full training mode: ALL labeled tiles → train ──
                    if row["has_label"] == "yes" and row["mask_path"]:
                        train_samples.append({
                            "tile_path": row["tile_path"],
                            "mask_path": row["mask_path"],
                            "dataset": dataset,
                        })
                else:
                    # ── Normal mode: hold out the val dataset(s) ──
                    is_test = (dataset in self.val_dataset_names)

                    if is_test:
                        meta = tile_meta.get(row["tile_path"], {})
                        test_samples.append({
                            "tile_path": row["tile_path"],
                            "dataset": dataset,
                            "row": int(meta.get("row", 0)),
                            "col": int(meta.get("col", 0)),
                            "col_off": int(meta.get("col_off", 0)),
                            "row_off": int(meta.get("row_off", 0)),
                            "width": int(meta.get("width", 1024)),
                            "height": int(meta.get("height", 1024)),
                        })
                        if row["has_label"] == "yes" and row["mask_path"]:
                            val_samples.append({
                                "tile_path": row["tile_path"],
                                "mask_path": row["mask_path"],
                                "dataset": dataset,
                            })
                    elif row["has_label"] == "yes" and row["mask_path"]:
                        train_samples.append({
                            "tile_path": row["tile_path"],
                            "mask_path": row["mask_path"],
                            "dataset": dataset,
                        })

        self.train_samples = train_samples
        self.val_samples = val_samples
        self.test_samples = test_samples

        if self.full_training:
            print(f"\n  Dataset split (FULL TRAINING — no validation):")
            print(f"    Train: {len(self.train_samples)} labeled tiles (ALL datasets)")
        else:
            print(f"\n  Dataset split:")
            print(f"    Train: {len(self.train_samples)} labeled tiles (all except {self.test_dataset_name})")
            print(f"    Val:   {len(self.val_samples)} labeled {self.test_dataset_name} tiles")
            print(f"    Test:  {len(self.test_samples)} {self.test_dataset_name} tiles (all, for inference)")

        # Print per-class pixel distribution
        self._print_class_distribution(train_samples, "Train")
        if val_samples:
            self._print_class_distribution(val_samples, "Val")

        self.train_ds = TileSegDataset(
            self.train_samples, self.num_classes, self.tile_size, transform="train"
        )
        if self.val_samples:
            self.val_ds = TileSegDataset(
                self.val_samples, self.num_classes, self.tile_size, transform="val"
            )
        else:
            self.val_ds = None
        self.test_ds = TileTestDataset(
            self.test_samples, self.tile_size
        ) if self.test_samples else None

    def _print_class_distribution(self, samples, split_name, max_samples=None):
        """Count pixels per class across mask files and print distribution."""
        num_classes = self.num_classes + 1  # +1 for background (class 0)
        counts = np.zeros(num_classes, dtype=np.int64)

        subset = samples if max_samples is None else samples[:max_samples]
        for s in subset:
            try:
                with rasterio.open(s["mask_path"]) as src:
                    mask = src.read(1)
                for c in range(num_classes):
                    counts[c] += (mask == c).sum()
            except Exception:
                continue

        total = counts.sum()
        if total == 0:
            return

        print(f"\n  {split_name} class distribution ({len(subset)} tiles):")
        print(f"  {'Class':<20s} {'Pixels':>12s} {'%':>8s}")
        print(f"  {'─'*42}")
        for c in range(num_classes):
            name = self.class_names[c] if c < len(self.class_names) else f"class_{c}"
            pct = counts[c] / total * 100
            print(f"  {name:<20s} {counts[c]:>12,d} {pct:>7.2f}%")
        print(f"  {'─'*42}")
        print(f"  {'Total':<20s} {total:>12,d}")

    def train_dataloader(self):
        return DataLoader(
            self.train_ds, batch_size=self.batch_size, shuffle=True,
            num_workers=self.num_workers, pin_memory=True, drop_last=True,
        )

    def val_dataloader(self):
        if self.val_ds is None or len(self.val_samples) == 0:
            return None
        return DataLoader(
            self.val_ds, batch_size=self.batch_size, shuffle=False,
            num_workers=self.num_workers, pin_memory=True,
        )

    def test_dataloader(self):
        if self.test_ds is None or len(self.test_samples) == 0:
            return None
        return DataLoader(
            self.test_ds, batch_size=self.batch_size, shuffle=False,
            num_workers=self.num_workers, pin_memory=True,
        )
