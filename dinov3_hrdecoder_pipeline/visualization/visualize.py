#!/usr/bin/env python3
"""
Visualization at Every Stage
==============================
Generates visualizations for:
1. Input data (tiles, masks, class distribution)
2. DINOv3 features (intermediate layer activations)
3. DINOv3+HRDecoder predictions (per-tile overlays)
4. Full stitched image with predictions

Usage:
    python -m dinov3_hrdecoder_pipeline.visualization.visualize --stage all
    python -m dinov3_hrdecoder_pipeline.visualization.visualize --stage data
    python -m dinov3_hrdecoder_pipeline.visualization.visualize --stage features
    python -m dinov3_hrdecoder_pipeline.visualization.visualize --stage predictions
    python -m dinov3_hrdecoder_pipeline.visualization.visualize --stage stitched
"""

import os
import sys
import csv
import argparse
from pathlib import Path

import numpy as np
import yaml
import rasterio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap

PIPE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PIPE_ROOT.parent))


def load_config():
    with open(PIPE_ROOT / "configs" / "train.yaml") as f:
        return yaml.safe_load(f)


def read_tile_rgb(path: str) -> np.ndarray:
    """Read tile as RGB HWC uint8."""
    with rasterio.open(path) as src:
        bands = min(src.count, 3)
        data = src.read(list(range(1, bands + 1)))
        if bands < 3:
            data = np.concatenate([data, np.zeros((3 - bands, *data.shape[1:]),
                                                   dtype=data.dtype)])
    img = np.transpose(data, (1, 2, 0))
    if img.dtype != np.uint8:
        if img.max() > 255:
            img = (img / img.max() * 255).astype(np.uint8)
        else:
            img = img.astype(np.uint8)
    return img


def read_mask(path: str) -> np.ndarray:
    with rasterio.open(path) as src:
        return src.read(1)


def get_class_colors(cfg: dict):
    """Get color map from config."""
    colors = [[0, 0, 0]]  # background = black
    names = ["background"]
    for cls in cfg["classes"]:
        colors.append(cls["color"])
        names.append(cls["name"])
    return np.array(colors, dtype=np.uint8), names


def mask_to_rgb(mask: np.ndarray, colors: np.ndarray) -> np.ndarray:
    """Convert class mask to RGB image."""
    rgb = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for i, color in enumerate(colors):
        rgb[mask == i] = color
    return rgb


def create_legend(names: list, colors: np.ndarray):
    """Create legend patches for matplotlib."""
    patches = []
    for name, color in zip(names, colors):
        patches.append(mpatches.Patch(
            color=np.array(color) / 255.0, label=name
        ))
    return patches


# ───────────────────── Stage 1: Data Visualization ─────────────────────

def visualize_data(cfg: dict):
    """Visualize input tiles and masks."""
    vis_dir = os.path.join(cfg["paths"]["vis_dir"], "1_data")
    os.makedirs(vis_dir, exist_ok=True)

    masks_dir = cfg["paths"]["labels_dir"]
    label_index = os.path.join(masks_dir, "label_index.csv")

    if not os.path.exists(label_index):
        print("  WARN: label_index.csv not found. Run prepare_masks.py first.")
        return

    colors, names = get_class_colors(cfg)
    legend = create_legend(names, colors)

    # Get labeled samples
    labeled = []
    with open(label_index) as f:
        for row in csv.DictReader(f):
            if row["has_label"] == "yes" and row["mask_path"]:
                labeled.append(row)

    num_samples = min(cfg["visualization"]["num_samples"], len(labeled))
    np.random.seed(42)
    samples = [labeled[i] for i in np.random.choice(len(labeled), num_samples,
                                                     replace=False)]

    fig, axes = plt.subplots(num_samples, 3, figsize=(15, 5 * num_samples))
    if num_samples == 1:
        axes = axes[np.newaxis]

    for i, sample in enumerate(samples):
        img = read_tile_rgb(sample["tile_path"])
        mask = read_mask(sample["mask_path"])
        overlay = mask_to_rgb(mask, colors)

        axes[i, 0].imshow(img)
        axes[i, 0].set_title(f"Tile: {Path(sample['tile_path']).stem}")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(overlay)
        axes[i, 1].set_title("Label Mask")
        axes[i, 1].axis("off")

        # Overlay
        blended = (img.astype(float) * 0.6 + overlay.astype(float) * 0.4).astype(np.uint8)
        axes[i, 2].imshow(blended)
        axes[i, 2].set_title("Overlay")
        axes[i, 2].axis("off")

    fig.legend(handles=legend, loc="lower center", ncol=len(names), fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, "data_samples.png"),
                dpi=cfg["visualization"]["dpi"], bbox_inches="tight")
    plt.close()

    # Class distribution
    class_counts = np.zeros(len(names))
    for sample in labeled[:100]:  # first 100 for speed
        mask = read_mask(sample["mask_path"])
        for c in range(len(names)):
            class_counts[c] += (mask == c).sum()

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(names, class_counts, color=[np.array(c) / 255.0 for c in colors])
    ax.set_title("Class Pixel Distribution (first 100 labeled tiles)")
    ax.set_ylabel("Pixel Count")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, "class_distribution.png"),
                dpi=cfg["visualization"]["dpi"])
    plt.close()

    print(f"  ✓ Data visualizations → {vis_dir}")


# ───────────────── Stage 2: Feature Visualization ──────────────────

def visualize_features(cfg: dict):
    """Visualize DINOv3 intermediate features using PCA."""
    vis_dir = os.path.join(cfg["paths"]["vis_dir"], "2_features")
    os.makedirs(vis_dir, exist_ok=True)

    import torch
    from sklearn.decomposition import PCA
    import torchvision.transforms.functional as TF

    # Load model encoder only
    from dinov3_hrdecoder_pipeline.models.encoder import DINOv3Encoder
    enc_cfg = cfg["model"]["encoder"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    encoder = DINOv3Encoder(
        arch=enc_cfg.get("arch", "vitl16"),
        intermediate_layers=enc_cfg["intermediate_layers"],
        embed_dim=enc_cfg["embed_dim"],
        out_dim=256,
        pretrained=enc_cfg.get("pretrained", True),
        weights=enc_cfg.get("weights", "LVD1689M"),
    ).eval().to(device)

    # Get sample tiles
    masks_dir = cfg["paths"]["labels_dir"]
    label_index = os.path.join(masks_dir, "label_index.csv")
    samples = []
    with open(label_index) as f:
        for row in csv.DictReader(f):
            if row["has_label"] == "yes":
                samples.append(row)
                if len(samples) >= 4:
                    break

    tile_size = cfg["training"]["tile_size"]

    for si, sample in enumerate(samples):
        img = read_tile_rgb(sample["tile_path"])
        img_t = TF.to_pil_image(img)
        img_t = TF.resize(img_t, [tile_size, tile_size])
        img_t = TF.to_tensor(img_t)
        img_t = TF.normalize(img_t, mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
        img_t = img_t.unsqueeze(0).to(device)

        with torch.no_grad():
            features = encoder(img_t)

        n_layers = len(features)
        fig, axes = plt.subplots(1, n_layers + 1, figsize=(5 * (n_layers + 1), 5))

        axes[0].imshow(img)
        axes[0].set_title("Input")
        axes[0].axis("off")

        for li, feat in enumerate(features):
            feat_np = feat[0].cpu().numpy()  # (C, h, w)
            C, h, w = feat_np.shape
            feat_flat = feat_np.reshape(C, -1).T  # (h*w, C)

            pca = PCA(n_components=3)
            pca_feat = pca.fit_transform(feat_flat)
            pca_feat = pca_feat.reshape(h, w, 3)
            pca_feat = (pca_feat - pca_feat.min()) / (pca_feat.max() - pca_feat.min() + 1e-6)

            layer_idx = enc_cfg["intermediate_layers"][li]
            axes[li + 1].imshow(pca_feat)
            axes[li + 1].set_title(f"Layer {layer_idx} (PCA)")
            axes[li + 1].axis("off")

        plt.tight_layout()
        plt.savefig(os.path.join(vis_dir, f"features_sample_{si}.png"),
                    dpi=cfg["visualization"]["dpi"])
        plt.close()

    del encoder
    torch.cuda.empty_cache()
    print(f"  ✓ Feature visualizations → {vis_dir}")


# ───────────────── Stage 3: Prediction Visualization ───────────────

def visualize_predictions(cfg: dict):
    """Visualize per-tile DINOv3+HRDecoder predictions against ground truth."""
    vis_dir = os.path.join(cfg["paths"]["vis_dir"], "3_predictions")
    os.makedirs(vis_dir, exist_ok=True)

    test_dataset = cfg["test_dataset"]
    pred_dir = os.path.join(cfg["paths"]["output_dir"], "predictions", test_dataset)
    idx_path = os.path.join(pred_dir, "prediction_index.csv")
    masks_dir = os.path.join(cfg["paths"]["labels_dir"], test_dataset)

    if not os.path.exists(idx_path):
        print("  WARN: No predictions found. Run test.py first.")
        return

    colors, names = get_class_colors(cfg)
    legend = create_legend(names, colors)

    with open(idx_path) as f:
        entries = list(csv.DictReader(f))

    # Filter to only tiles that have ground truth masks
    labeled_entries = []
    for entry in entries:
        tile_stem = Path(entry["tile_path"]).stem
        mask_path = os.path.join(masks_dir, f"{tile_stem}_mask.tif")
        if os.path.exists(mask_path):
            entry["mask_path"] = mask_path
            labeled_entries.append(entry)

    if not labeled_entries:
        print("  WARN: No predicted tiles with ground truth masks found.")
        return

    num_samples = max(10, cfg["visualization"]["num_samples"])
    num_samples = min(num_samples, len(labeled_entries))
    np.random.seed(42)
    samples = [labeled_entries[i] for i in np.random.choice(
        len(labeled_entries), num_samples, replace=False)]

    # Save each sample as a separate image
    for i, entry in enumerate(samples):
        img = read_tile_rgb(entry["tile_path"])
        gt = read_mask(entry["mask_path"])
        pred = read_mask(entry["pred_path"])
        gt_rgb = mask_to_rgb(gt, colors)
        pred_rgb = mask_to_rgb(pred, colors)

        fig, axes = plt.subplots(1, 5, figsize=(25, 5))
        tile_stem = Path(entry["tile_path"]).stem

        axes[0].imshow(img)
        axes[0].set_title(f"Input: {tile_stem}")
        axes[0].axis("off")

        axes[1].imshow(gt_rgb)
        axes[1].set_title("Ground Truth")
        axes[1].axis("off")

        gt_blended = (img.astype(float) * 0.6 + gt_rgb.astype(float) * 0.4).astype(np.uint8)
        axes[2].imshow(gt_blended)
        axes[2].set_title("GT Overlay")
        axes[2].axis("off")

        axes[3].imshow(pred_rgb)
        axes[3].set_title("Prediction")
        axes[3].axis("off")

        pred_blended = (img.astype(float) * 0.6 + pred_rgb.astype(float) * 0.4).astype(np.uint8)
        axes[4].imshow(pred_blended)
        axes[4].set_title("Pred Overlay")
        axes[4].axis("off")

        fig.legend(handles=legend, loc="lower center", ncol=len(names), fontsize=10)
        plt.tight_layout()
        plt.savefig(os.path.join(vis_dir, f"prediction_{i:02d}_{tile_stem}.png"),
                    dpi=cfg["visualization"]["dpi"], bbox_inches="tight")
        plt.close()

    print(f"  ✓ {num_samples} prediction visualizations → {vis_dir}")


# ───────────────── Stage 4: Stitched Full Image ───────────────────

def visualize_stitched(cfg: dict):
    """Visualize the full stitched prediction image."""
    vis_dir = os.path.join(cfg["paths"]["vis_dir"], "4_stitched")
    os.makedirs(vis_dir, exist_ok=True)

    test_dataset = cfg["test_dataset"]
    colors, names = get_class_colors(cfg)
    legend = create_legend(names, colors)

    stitched_dir = os.path.join(cfg["paths"]["output_dir"], "stitched")
    pred_path = os.path.join(stitched_dir, f"{test_dataset}_pred.tif")

    paths = [("DINOv3+HRDecoder", pred_path)]

    for title, path in paths:
        if not os.path.exists(path):
            print(f"  WARN: {path} not found. Run stitch_and_evaluate.py first.")
            continue

        with rasterio.open(path) as src:
            mask = src.read(1)

        mask_rgb = mask_to_rgb(mask, colors)

        # Downsample for visualization if very large
        max_dim = 4096
        h, w = mask_rgb.shape[:2]
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            new_h, new_w = int(h * scale), int(w * scale)
            from PIL import Image
            mask_rgb = np.array(Image.fromarray(mask_rgb).resize((new_w, new_h),
                                                                   Image.NEAREST))

        fig, ax = plt.subplots(figsize=(20, 20))
        ax.imshow(mask_rgb)
        ax.set_title(f"{test_dataset} - {title}", fontsize=16)
        ax.axis("off")
        ax.legend(handles=legend, loc="lower right", fontsize=12)

        safe_title = title.replace("+", "").replace(" ", "_").lower()
        plt.tight_layout()
        plt.savefig(os.path.join(vis_dir, f"stitched_{safe_title}.png"),
                    dpi=cfg["visualization"]["dpi"], bbox_inches="tight")
        plt.close()

    print(f"  ✓ Stitched visualizations → {vis_dir}")


# ───────────────── Main ──────────────────────────────────────────

STAGES = {
    "data": visualize_data,
    "features": visualize_features,
    "predictions": visualize_predictions,
    "stitched": visualize_stitched,
}


def main():
    parser = argparse.ArgumentParser(description="Pipeline Visualizations")
    parser.add_argument("--stage", default="all",
                        choices=list(STAGES.keys()) + ["all"])
    args = parser.parse_args()

    cfg = load_config()

    print("=" * 70)
    print("  DINOv3 + HRDecoder Pipeline Visualizations")
    print("=" * 70)

    stages = list(STAGES.keys()) if args.stage == "all" else [args.stage]

    for stage in stages:
        print(f"\n  Stage: {stage}")
        print(f"  {'─' * 50}")
        try:
            STAGES[stage](cfg)
        except Exception as e:
            print(f"  ERROR in {stage}: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n{'='*70}")
    print(f"  VISUALIZATION COMPLETE")
    print(f"  Output: {cfg['paths']['vis_dir']}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
