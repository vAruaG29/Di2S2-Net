#!/usr/bin/env python3
"""
Decoders for DINOv3 Segmentation Pipeline
==========================================
Multiple decoder architectures for use with ViT multi-scale features:

1. HRDecoder   - Adapted from CVIU-CSU/HRDecoder (LR/HR two-pass fusion)
2. UPerNet     - FPN + PSP pooling (standard for ViT backbones)
3. SegFormer   - Lightweight MLP-based decoder
4. SkipDecoder - UNet-style progressive upsampling with skip connections
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# Common building blocks
# ─────────────────────────────────────────────────────────────────────────────

class ConvBNReLU(nn.Module):
    """Conv2d + BatchNorm + ReLU block."""
    def __init__(self, in_ch, out_ch, kernel_size=3, padding=1, stride=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class MultiScaleFusion(nn.Module):
    """
    Fuse multi-scale ViT features into a single feature map.
    All ViT features are at the same spatial resolution (H/14 or H/16)
    but from different depths. We concatenate and compress.
    """
    def __init__(self, in_dim: int, num_scales: int, hidden_dim: int):
        super().__init__()
        total_in = in_dim * num_scales
        self.compress = nn.Sequential(
            ConvBNReLU(total_in, hidden_dim, 1, 0),
            ConvBNReLU(hidden_dim, hidden_dim, 3, 1),
        )

    def forward(self, features: list) -> torch.Tensor:
        fused = torch.cat(features, dim=1)
        return self.compress(fused)


class SegHead(nn.Module):
    """Segmentation head: feature map → class logits."""
    def __init__(self, in_dim: int, num_classes: int, upsample_factor: int = 1):
        super().__init__()
        self.upsample_factor = upsample_factor
        self.head = nn.Sequential(
            ConvBNReLU(in_dim, in_dim, 3, 1),
            ConvBNReLU(in_dim, in_dim // 2, 3, 1),
            nn.Conv2d(in_dim // 2, num_classes, 1),
        )

    def forward(self, x: torch.Tensor, target_size: tuple = None) -> torch.Tensor:
        logits = self.head(x)
        if target_size is not None:
            logits = F.interpolate(logits, size=target_size, mode="bilinear",
                                   align_corners=False)
        elif self.upsample_factor > 1:
            logits = F.interpolate(logits, scale_factor=self.upsample_factor,
                                   mode="bilinear", align_corners=False)
        return logits


# ─────────────────────────────────────────────────────────────────────────────
# 1. HRDecoder (fixed)
# ─────────────────────────────────────────────────────────────────────────────

def get_crop_bbox(img_size, hr_crop_size, divisible=8):
    """Random crop bbox in pixel space for HR pass."""
    if img_size == hr_crop_size:
        return (0, img_size[0], 0, img_size[1])
    margin_h = max(img_size[0] - hr_crop_size[0], 0)
    margin_w = max(img_size[1] - hr_crop_size[1], 0)
    offset_h = np.random.randint(0, max((margin_h + 1) // divisible, 1)) * divisible
    offset_w = np.random.randint(0, max((margin_w + 1) // divisible, 1)) * divisible
    return (offset_h, offset_h + hr_crop_size[0], offset_w, offset_w + hr_crop_size[1])


class HRDecoder(nn.Module):
    """
    HRDecoder-style two-pass decoder adapted for ViT features.

    Adapted from CVIU-CSU/HRDecoder. The original operates on very high-res
    images (2048+px) by creating LR features from a resized image. For our
    512px tiles with ViT, we adapt: the LR pass uses the native feature map,
    and HR crops zoom into sub-regions of the feature map for local detail.

    All crop coordinates (hr_crop_size) are in PIXEL SPACE.
    """

    def __init__(
        self,
        in_dim: int = 256,
        num_scales: int = 4,
        hidden_dim: int = 256,
        num_classes: int = 6,
        hr_crop_size: tuple = (256, 256),
        crop_num: int = 2,
        scale_ratio: tuple = (0.75, 1.25),
        divisible: int = 8,
        patch_size: int = 16,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.hr_crop_size = hr_crop_size
        self.crop_num = crop_num
        self.scale_ratio = scale_ratio
        self.divisible = divisible
        self.patch_size = patch_size

        self.fusion = MultiScaleFusion(in_dim, num_scales, hidden_dim)
        self.seg_head = SegHead(hidden_dim, num_classes, upsample_factor=1)

    def _get_random_hr_crop_size(self):
        if isinstance(self.scale_ratio, (tuple, list)) and len(self.scale_ratio) == 2:
            lo, hi = self.scale_ratio
            ratio = np.random.random() * (hi - lo) + lo
            d = max(self.divisible, 1)
            return (
                int(self.hr_crop_size[0] * ratio // d * d),
                int(self.hr_crop_size[1] * ratio // d * d),
            )
        return self.hr_crop_size

    def _resize_crop_feat(self, feat, ori_size, bbox):
        """Crop from feat using pixel-space bbox, resize back to feat spatial size."""
        h_ratio = feat.shape[2] / ori_size[0]
        w_ratio = feat.shape[3] / ori_size[1]
        y1, y2, x1, x2 = bbox
        fy1 = int(y1 * h_ratio)
        fy2 = int(y2 * h_ratio)
        fx1 = int(x1 * w_ratio)
        fx2 = int(x2 * w_ratio)
        fy2 = max(fy2, fy1 + 1)
        fx2 = max(fx2, fx1 + 1)
        cropped = feat[:, :, fy1:fy2, fx1:fx2]
        return F.interpolate(cropped, size=feat.shape[2:], mode="bilinear",
                             align_corners=False)

    def forward_train(self, features: list, target_size: tuple):
        fused = self.fusion(features)

        # LR pass
        lr_logits = self.seg_head(fused)
        lr_logits = F.interpolate(lr_logits, size=target_size, mode="bilinear",
                                  align_corners=False)

        # HR pass: random crops in pixel space
        hr_crop = self._get_random_hr_crop_size()
        hr_bboxes = [get_crop_bbox(target_size, hr_crop, self.divisible)
                     for _ in range(self.crop_num)]

        hr_feats = torch.cat([
            self._resize_crop_feat(fused, target_size, bbox)
            for bbox in hr_bboxes
        ], dim=0)

        hr_logits_all = self.seg_head(hr_feats)
        hr_logits_all = F.interpolate(hr_logits_all, size=hr_crop, mode="bilinear",
                                      align_corners=False)

        # Fusion: paste HR crops back onto LR base
        fuse_logits = lr_logits.clone()
        B = fused.shape[0]
        for i, bbox in enumerate(hr_bboxes):
            y1, y2, x1, x2 = bbox
            hr_crop_logit = hr_logits_all[i * B:(i + 1) * B]
            hr_crop_resized = F.interpolate(
                hr_crop_logit, size=(y2 - y1, x2 - x1), mode="bilinear",
                align_corners=False
            )
            fuse_logits[:, :, y1:y2, x1:x2] = (
                fuse_logits[:, :, y1:y2, x1:x2] + hr_crop_resized
            ) / 2

        return {
            "lr_logits": lr_logits,
            "hr_bboxes": hr_bboxes,
            "hr_logits": hr_logits_all,
            "hr_scale": hr_crop,
            "fuse_logits": fuse_logits,
        }

    def forward_test(self, features: list, target_size: tuple):
        fused = self.fusion(features)

        lr_logits = self.seg_head(fused)
        lr_logits = F.interpolate(lr_logits, size=target_size, mode="bilinear",
                                  align_corners=False)

        h_crop, w_crop = self.hr_crop_size
        H, W = target_size
        count_map = torch.ones_like(lr_logits)
        fuse_logits = lr_logits.clone()

        for y in range(0, max(H - h_crop + 1, 1), h_crop):
            for x in range(0, max(W - w_crop + 1, 1), w_crop):
                y1, x1 = y, x
                y2 = min(y1 + h_crop, H)
                x2 = min(x1 + w_crop, W)
                y1 = max(y2 - h_crop, 0)
                x1 = max(x2 - w_crop, 0)
                bbox = (y1, y2, x1, x2)

                hr_feat = self._resize_crop_feat(fused, target_size, bbox)
                hr_logit = self.seg_head(hr_feat)
                hr_logit = F.interpolate(hr_logit, size=(y2 - y1, x2 - x1),
                                         mode="bilinear", align_corners=False)
                fuse_logits[:, :, y1:y2, x1:x2] += hr_logit
                count_map[:, :, y1:y2, x1:x2] += 1

        fuse_logits = fuse_logits / count_map
        return fuse_logits

    def forward(self, features: list, target_size: tuple, training: bool = False):
        if training:
            return self.forward_train(features, target_size)
        else:
            return self.forward_test(features, target_size)


# ─────────────────────────────────────────────────────────────────────────────
# 2. UPerNet Decoder
# ─────────────────────────────────────────────────────────────────────────────

class PPM(nn.Module):
    """Pyramid Pooling Module (PSPNet)."""
    def __init__(self, in_channels, channels, pool_scales=(1, 2, 3, 6)):
        super().__init__()
        self.stages = nn.ModuleList()
        for scale in pool_scales:
            self.stages.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(scale),
                ConvBNReLU(in_channels, channels, 1, 0),
            ))

    def forward(self, x):
        ppm_outs = []
        for stage in self.stages:
            ppm_out = stage(x)
            ppm_out = F.interpolate(ppm_out, size=x.shape[2:],
                                    mode="bilinear", align_corners=False)
            ppm_outs.append(ppm_out)
        return ppm_outs


class UPerNetDecoder(nn.Module):
    """
    UPerNet decoder adapted for ViT multi-scale features.

    Uses FPN-style lateral connections + PSP pooling on the deepest feature.
    Standard decoder for ViT-based segmentation (used in BEiT, DINOv2 papers).

    Since all ViT features share the same spatial resolution, this decoder
    still benefits from different semantic levels at each layer.
    """

    def __init__(
        self,
        in_dim: int = 256,
        num_scales: int = 4,
        hidden_dim: int = 256,
        num_classes: int = 6,
        pool_scales: tuple = (1, 2, 3, 6),
        patch_size: int = 16,
        **kwargs,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.patch_size = patch_size

        # PSP module on deepest feature
        self.psp = PPM(in_dim, hidden_dim, pool_scales)
        self.psp_bottleneck = ConvBNReLU(
            in_dim + len(pool_scales) * hidden_dim, hidden_dim, 3, 1
        )

        # FPN lateral convs (for all scales except deepest)
        self.lateral_convs = nn.ModuleList([
            ConvBNReLU(in_dim, hidden_dim, 1, 0) for _ in range(num_scales - 1)
        ])
        self.fpn_convs = nn.ModuleList([
            ConvBNReLU(hidden_dim, hidden_dim, 3, 1) for _ in range(num_scales - 1)
        ])

        # Final bottleneck: all FPN levels + PSP concatenated
        self.fpn_bottleneck = ConvBNReLU(num_scales * hidden_dim, hidden_dim, 3, 1)

        # Classification head
        self.cls_seg = nn.Conv2d(hidden_dim, num_classes, 1)

    def forward(self, features: list, target_size: tuple, training: bool = False):
        """
        Args:
            features: list of (B, in_dim, h, w) from encoder (4 scales)
            target_size: (H, W) output resolution
        Returns:
            (B, num_classes, H, W) logits — or dict with 'fuse_logits' if training
        """
        # PSP on deepest feature
        psp_outs = [features[-1]]
        psp_outs.extend(self.psp(features[-1]))
        psp_out = self.psp_bottleneck(torch.cat(psp_outs, dim=1))

        # Build laterals
        laterals = [conv(features[i]) for i, conv in enumerate(self.lateral_convs)]
        laterals.append(psp_out)

        # Top-down pathway (all same spatial res for ViT, but semantics differ)
        for i in range(len(laterals) - 1, 0, -1):
            prev_shape = laterals[i - 1].shape[2:]
            laterals[i - 1] = laterals[i - 1] + F.interpolate(
                laterals[i], size=prev_shape, mode="bilinear", align_corners=False
            )

        # FPN outputs
        fpn_outs = [self.fpn_convs[i](laterals[i]) for i in range(len(self.lateral_convs))]
        fpn_outs.append(laterals[-1])

        # Resize all to largest spatial size and concat
        target_feat_size = fpn_outs[0].shape[2:]
        for i in range(1, len(fpn_outs)):
            fpn_outs[i] = F.interpolate(
                fpn_outs[i], size=target_feat_size, mode="bilinear", align_corners=False
            )

        out = self.fpn_bottleneck(torch.cat(fpn_outs, dim=1))
        logits = self.cls_seg(out)
        logits = F.interpolate(logits, size=target_size, mode="bilinear",
                               align_corners=False)

        if training:
            return {"fuse_logits": logits}
        return logits

    def forward_train(self, features, target_size):
        return self.forward(features, target_size, training=True)

    def forward_test(self, features, target_size):
        return self.forward(features, target_size, training=False)


# ─────────────────────────────────────────────────────────────────────────────
# 3. SegFormer Decoder
# ─────────────────────────────────────────────────────────────────────────────

class LinearMLP(nn.Module):
    """Linear embedding used in SegFormer decoder."""
    def __init__(self, input_dim, embed_dim):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        # x: (B, C, H, W) → (B, H*W, C) → project → (B, H*W, embed_dim)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        x = x.transpose(1, 2).reshape(B, -1, H, W)
        return x


class SegFormerDecoder(nn.Module):
    """
    SegFormer-style MLP decoder adapted for ViT features.

    Lightweight decoder: each scale gets a linear projection to a common
    embedding dim, all are resized to the same spatial resolution, concatenated,
    and fused with a single conv + classification head.

    Very efficient and works well when the backbone provides strong features.
    """

    def __init__(
        self,
        in_dim: int = 256,
        num_scales: int = 4,
        hidden_dim: int = 256,
        num_classes: int = 6,
        embed_dim: int = 256,
        dropout: float = 0.1,
        patch_size: int = 16,
        **kwargs,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.patch_size = patch_size

        # Per-scale MLP projections
        self.linear_projs = nn.ModuleList([
            LinearMLP(in_dim, embed_dim) for _ in range(num_scales)
        ])

        # Fusion conv
        self.linear_fuse = nn.Sequential(
            ConvBNReLU(embed_dim * num_scales, hidden_dim, 1, 0),
        )

        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.cls_seg = nn.Conv2d(hidden_dim, num_classes, 1)

    def forward(self, features: list, target_size: tuple, training: bool = False):
        # Project each scale
        projected = []
        target_feat_size = features[0].shape[2:]
        for i, feat in enumerate(features):
            p = self.linear_projs[i](feat)
            if p.shape[2:] != target_feat_size:
                p = F.interpolate(p, size=target_feat_size, mode="bilinear",
                                  align_corners=False)
            projected.append(p)

        fused = self.linear_fuse(torch.cat(projected, dim=1))
        fused = self.dropout(fused)
        logits = self.cls_seg(fused)
        logits = F.interpolate(logits, size=target_size, mode="bilinear",
                               align_corners=False)

        if training:
            return {"fuse_logits": logits}
        return logits

    def forward_train(self, features, target_size):
        return self.forward(features, target_size, training=True)

    def forward_test(self, features, target_size):
        return self.forward(features, target_size, training=False)


# ─────────────────────────────────────────────────────────────────────────────
# 4. SkipDecoder (UNet-style with progressive upsampling)
# ─────────────────────────────────────────────────────────────────────────────

class UpBlock(nn.Module):
    """Upsample + concat skip + conv refine."""
    def __init__(self, in_ch, skip_ch, out_ch, scale_factor=2):
        super().__init__()
        self.scale_factor = scale_factor
        self.conv = nn.Sequential(
            ConvBNReLU(in_ch + skip_ch, out_ch, 3, 1),
            ConvBNReLU(out_ch, out_ch, 3, 1),
        )

    def forward(self, x, skip=None):
        if self.scale_factor > 1:
            x = F.interpolate(x, scale_factor=self.scale_factor, mode="bilinear",
                              align_corners=False)
        if skip is not None:
            if x.shape[2:] != skip.shape[2:]:
                skip = F.interpolate(skip, size=x.shape[2:], mode="bilinear",
                                     align_corners=False)
            x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class SkipDecoder(nn.Module):
    """
    UNet-style decoder with skip connections from ViT intermediate layers.

    Progressive upsampling: deepest features are refined with shallower features
    via skip connections. Each level upsamples 2× and concatenates with the
    corresponding skip feature. Since all ViT features share the same spatial
    resolution, we create a multi-resolution pyramid by applying strided convs
    to create artificial scale hierarchy before decoding.

    Architecture:
        ViT features (all at H/16):
        [shallow] ──────────────────────────┐
        [mid-shallow] ──────────────────┐   │
        [mid-deep] ──────────────┐      │   │
        [deep] → compress → up2x+skip → up2x+skip → up2x+skip → cls
    """

    def __init__(
        self,
        in_dim: int = 256,
        num_scales: int = 4,
        hidden_dim: int = 256,
        num_classes: int = 6,
        patch_size: int = 16,
        **kwargs,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.patch_size = patch_size
        self.num_scales = num_scales

        # Create scale hierarchy from same-res ViT features using strided convs
        # deepest → 1/4 res, next → 1/2 res, next → 1× res, shallowest → 2× res
        self.scale_adapters = nn.ModuleList()
        for i in range(num_scales):
            depth_idx = num_scales - 1 - i  # reverse: deepest first
            if depth_idx == 0:
                # Shallowest: upsample 2×
                adapter = nn.Sequential(
                    ConvBNReLU(in_dim, hidden_dim, 1, 0),
                    nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                )
            elif depth_idx == 1:
                # Keep at native resolution
                adapter = ConvBNReLU(in_dim, hidden_dim, 1, 0)
            elif depth_idx == 2:
                # Downsample 2×
                adapter = nn.Sequential(
                    ConvBNReLU(in_dim, hidden_dim, 3, 1, stride=2),
                )
            else:
                # Downsample 4×
                adapter = nn.Sequential(
                    ConvBNReLU(in_dim, hidden_dim, 3, 1, stride=2),
                    ConvBNReLU(hidden_dim, hidden_dim, 3, 1, stride=2),
                )

            self.scale_adapters.append(adapter)

        # Decoder blocks: progressively upsample from deepest
        dims = [hidden_dim] * num_scales
        self.up_blocks = nn.ModuleList()
        for i in range(num_scales - 1):
            in_ch = dims[0] if i == 0 else dims[i]
            skip_ch = dims[i + 1]
            out_ch = dims[i + 1]
            self.up_blocks.append(UpBlock(in_ch, skip_ch, out_ch, scale_factor=2))

        # Final upsample to get closer to input resolution
        self.final_up = nn.Sequential(
            ConvBNReLU(hidden_dim, hidden_dim // 2, 3, 1),
            ConvBNReLU(hidden_dim // 2, hidden_dim // 2, 3, 1),
        )
        self.cls_seg = nn.Conv2d(hidden_dim // 2, num_classes, 1)

    def forward(self, features: list, target_size: tuple, training: bool = False):
        # Create multi-resolution pyramid from ViT features
        # features[0]=shallowest, features[-1]=deepest
        # scale_adapters[0] adapts deepest (smallest), scale_adapters[-1] adapts shallowest (largest)
        adapted = []
        for i in range(self.num_scales):
            feat_idx = self.num_scales - 1 - i  # deepest first
            adapted.append(self.scale_adapters[i](features[feat_idx]))

        # Progressive decode: start from deepest (smallest spatial)
        x = adapted[0]
        for i, up_block in enumerate(self.up_blocks):
            x = up_block(x, adapted[i + 1])

        # Final classification
        x = self.final_up(x)
        logits = self.cls_seg(x)
        logits = F.interpolate(logits, size=target_size, mode="bilinear",
                               align_corners=False)

        if training:
            return {"fuse_logits": logits}
        return logits

    def forward_train(self, features, target_size):
        return self.forward(features, target_size, training=True)

    def forward_test(self, features, target_size):
        return self.forward(features, target_size, training=False)


# ─────────────────────────────────────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────────────────────────────────────

DECODER_REGISTRY = {
    "hrdecoder": HRDecoder,
    "upernet": UPerNetDecoder,
    "segformer": SegFormerDecoder,
    "skip": SkipDecoder,
}


def build_decoder(decoder_type: str, **kwargs) -> nn.Module:
    """Build decoder by type name."""
    decoder_type = decoder_type.lower()
    if decoder_type not in DECODER_REGISTRY:
        raise ValueError(
            f"Unknown decoder type '{decoder_type}'. "
            f"Choose from: {list(DECODER_REGISTRY.keys())}"
        )
    return DECODER_REGISTRY[decoder_type](**kwargs)
