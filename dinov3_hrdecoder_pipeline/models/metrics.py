#!/usr/bin/env python3
"""
Segmentation Metrics: IoU, F1, Precision, Recall per class
============================================================
"""

import torch
import numpy as np


class SegmentationMetrics:
    """Accumulates confusion matrix and computes per-class metrics."""

    def __init__(self, num_classes: int, class_names: list = None, ignore_index: int = 255):
        self.num_classes = num_classes
        self.class_names = class_names or [f"class_{i}" for i in range(num_classes)]
        self.ignore_index = ignore_index
        self.reset()

    def reset(self):
        self.confusion = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)

    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        """
        Args:
            preds: (B, H, W) predicted class indices
            targets: (B, H, W) ground truth class indices
        """
        preds = preds.cpu().numpy().flatten()
        targets = targets.cpu().numpy().flatten()

        # Filter out ignore index
        valid = targets != self.ignore_index
        preds = preds[valid]
        targets = targets[valid]

        # Clamp to valid range
        preds = np.clip(preds, 0, self.num_classes - 1)
        targets = np.clip(targets, 0, self.num_classes - 1)

        for t, p in zip(targets, preds):
            self.confusion[t, p] += 1

    def compute(self) -> dict:
        """Compute per-class and mean metrics."""
        eps = 1e-6
        results = {}

        tp = np.diag(self.confusion)
        fp = self.confusion.sum(axis=0) - tp
        fn = self.confusion.sum(axis=1) - tp

        iou = tp / (tp + fp + fn + eps)
        precision = tp / (tp + fp + eps)
        recall = tp / (tp + fn + eps)
        f1 = 2 * precision * recall / (precision + recall + eps)

        for i in range(self.num_classes):
            name = self.class_names[i] if i < len(self.class_names) else f"class_{i}"
            results[f"iou/{name}"] = iou[i]
            results[f"precision/{name}"] = precision[i]
            results[f"recall/{name}"] = recall[i]
            results[f"f1/{name}"] = f1[i]

        results["mIoU"] = np.nanmean(iou)
        results["mF1"] = np.nanmean(f1)
        results["mPrecision"] = np.nanmean(precision)
        results["mRecall"] = np.nanmean(recall)
        results["overall_accuracy"] = tp.sum() / (self.confusion.sum() + eps)

        return results

    def summary_str(self) -> str:
        """Pretty-print metrics."""
        metrics = self.compute()
        lines = [
            f"{'='*60}",
            f"  Segmentation Metrics",
            f"{'='*60}",
            f"  mIoU:      {metrics['mIoU']:.4f}",
            f"  mF1:       {metrics['mF1']:.4f}",
            f"  mPrec:     {metrics['mPrecision']:.4f}",
            f"  mRecall:   {metrics['mRecall']:.4f}",
            f"  OA:        {metrics['overall_accuracy']:.4f}",
            f"{'─'*60}",
        ]
        for i in range(self.num_classes):
            name = self.class_names[i] if i < len(self.class_names) else f"class_{i}"
            lines.append(
                f"  {name:20s}  IoU={metrics[f'iou/{name}']:.4f}  "
                f"F1={metrics[f'f1/{name}']:.4f}  "
                f"P={metrics[f'precision/{name}']:.4f}  "
                f"R={metrics[f'recall/{name}']:.4f}"
            )
        lines.append(f"{'='*60}")
        return "\n".join(lines)
