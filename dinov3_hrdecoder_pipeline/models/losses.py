#!/usr/bin/env python3
"""
Loss Functions: BCE + Dice + Edge (Sobel boundary)
===================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """Per-class Dice loss."""
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (B, C, H, W) raw logits
            targets: (B, H, W) class indices
        """
        num_classes = logits.shape[1]
        probs = torch.softmax(logits, dim=1)

        # One-hot encode targets
        targets_oh = F.one_hot(targets.clamp(0, num_classes - 1),
                               num_classes).permute(0, 3, 1, 2).float()

        dice_sum = 0.0
        for c in range(num_classes):
            p = probs[:, c]
            g = targets_oh[:, c]
            intersection = (p * g).sum(dim=(1, 2))
            union = p.sum(dim=(1, 2)) + g.sum(dim=(1, 2))
            dice = (2 * intersection + self.smooth) / (union + self.smooth)
            dice_sum += (1 - dice).mean()

        return dice_sum / num_classes


class EdgeLoss(nn.Module):
    """
    Sobel edge-aware boundary loss.
    Encourages prediction edges to align with ground truth boundaries.
    """
    def __init__(self, kernel_size: int = 3):
        super().__init__()
        # Sobel kernels
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                               dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                               dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        self.register_buffer("sobel_x", sobel_x)
        self.register_buffer("sobel_y", sobel_y)

    def _get_edges(self, x: torch.Tensor) -> torch.Tensor:
        """Extract edges from a single-channel tensor (B, 1, H, W)."""
        if x.dim() == 3:
            x = x.unsqueeze(1)
        x = x.float()
        edge_x = F.conv2d(x, self.sobel_x, padding=1)
        edge_y = F.conv2d(x, self.sobel_y, padding=1)
        return torch.sqrt(edge_x ** 2 + edge_y ** 2 + 1e-6)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (B, C, H, W)
            targets: (B, H, W)
        """
        # Prediction edges: use max probability channel
        probs = torch.softmax(logits, dim=1)
        pred_mask = probs.argmax(dim=1).float()  # (B, H, W)

        # Extract edges
        pred_edges = self._get_edges(pred_mask)
        target_edges = self._get_edges(targets.float())

        # Normalize
        pred_edges = pred_edges / (pred_edges.max() + 1e-6)
        target_edges = target_edges / (target_edges.max() + 1e-6)

        # MSE on edge maps (autocast-safe, unlike binary_cross_entropy)
        return F.mse_loss(pred_edges, target_edges)


class CombinedLoss(nn.Module):
    """Combined BCE + Dice + Edge loss."""
    def __init__(self, num_classes: int, bce_weight: float = 0.5,
                 dice_weight: float = 0.3, edge_weight: float = 0.2,
                 edge_kernel_size: int = 3):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.edge_weight = edge_weight

        self.ce_loss = nn.CrossEntropyLoss(ignore_index=255)
        self.dice_loss = DiceLoss()
        self.edge_loss = EdgeLoss(edge_kernel_size)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> dict:
        """
        Returns dict with individual and total losses for logging.
        """
        ce = self.ce_loss(logits, targets)
        dice = self.dice_loss(logits, targets)
        edge = self.edge_loss(logits, targets)

        total = (self.bce_weight * ce +
                 self.dice_weight * dice +
                 self.edge_weight * edge)

        return {
            "loss": total,
            "ce_loss": ce,
            "dice_loss": dice,
            "edge_loss": edge,
        }
