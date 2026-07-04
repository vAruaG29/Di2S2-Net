# `models/` — encoder, decoders, loss, metrics, data

The model itself. None of these files are CLI entrypoints — they are
imported by `training/` and `inference/`. They read hyper-parameters from
`configs/train.yaml` (the `model`, `loss`, `training`, `classes` sections).

```
image (B,3,1024,1024)
   │  encoder.py — DINOv3 ViT-L/16, blocks [5,11,17,23] → 4 × (B,256,64,64)
   ▼
decoder.py — HRDecoder: multi-scale fusion → LR pass + HR pass → fuse
   ▼
logits (B,7,1024,1024)
   │  losses.py — 0.5·CE + 0.3·Dice + 0.2·Edge, weighted over Fuse/LR/HR
   ▼  metrics.py — confusion matrix → per-class IoU/F1/P/R + mIoU/OA
```

| File | Role |
|---|---|
| `encoder.py` | `DINOv3Encoder` — ViT-L/16 backbone + per-scale projection heads, partial freeze, robust weight loading. |
| `decoder.py` | `HRDecoder` (default) + `UPerNetDecoder`, `SegFormerDecoder`, `SkipDecoder` behind a `build_decoder()` factory. |
| `losses.py` | `CombinedLoss` = CE + Dice + Sobel-edge; `DiceLoss`, `EdgeLoss`. |
| `metrics.py` | `SegmentationMetrics` — accumulating confusion matrix + per-class metrics. |
| `dataset.py` | `TileSegDataset`, `TileTestDataset`, `SegDataModule` (geographic splits, augmentation, ImageNet norm). |
| `lightning_module.py` | `DINOv3HRDecoderModule` — ties it all together: multi-scale loss, differential LR, cosine+warmup schedule. |

---

## `encoder.py` — `DINOv3Encoder(nn.Module)`

Constructor: `arch="vitl16", intermediate_layers=[5,11,17,23],
freeze_first_n_blocks=12, embed_dim=1024, out_dim=256, pretrained=True,
weights="LVD1689M"` (the config passes `weights="SAT493M"`).

- **Backbone** loaded from the locally cloned DINOv3 repo (`../../models/dinov3/`, injected onto `sys.path`). `patch_size=16` → a 1024² tile becomes a **64×64 token grid**. `arch_map` supports `vits16 / vitb16 / vitl16 / vitl16plus / vith16plus / vit7b16`.
- **Multi-scale extraction:** `get_intermediate_layers([5,11,17,23], norm=True)` yields 4 features at 64×64×1024. Each passes a **projection head** `LayerNorm → Linear(1024→256) → GELU`, reshaped to `(B,256,64,64)`.
- **Partial freeze** (`_freeze_layers`): patch-embed **+ first 12 of 24 blocks** frozen; blocks 12–23 fine-tuned.
- **Robust weight loading** (`_load_dinov3`, 3-strategy fallback): (1) Facebook CDN → (2) **HuggingFace Hub** with a full HF→native key remap (fuses separate `q/k/v` into one `qkv`, builds a zero k-bias, maps `layer_scale/mlp/norm` keys; `rope_embed` treated as computed-not-stored) → (3) local `.pth`/`.safetensors` → random-init fallback with a warning. This is why the gated SAT-493M weights load even where the CDN is blocked.
- `get_param_groups(encoder_lr_mult=0.1, base_lr=1e-4)` returns two groups: projections at `base_lr`, encoder at `base_lr × 0.1` (differential LR).

## `decoder.py` — `HRDecoder` + factory

Building blocks: `ConvBNReLU`, `MultiScaleFusion` (concat 4×256 →
`Conv1×1(1024→256)+BN+ReLU → Conv3×3(256→256)`), `SegHead`
(`Conv3×3(256→256) → Conv3×3(256→128) → Conv1×1(128→7)`, raw logits, no
softmax).

**`HRDecoder`** (`in_dim=256, num_scales=4, hidden_dim=256, num_classes,
hr_crop_size=(256,256), crop_num=2, scale_ratio=(0.75,1.25), divisible=8`):

- **LR (global) pass** — SegHead on the full fused map, bilinear-upsampled to 1024² → scene context.
- **HR (local) pass, train vs test differ deliberately:**
  - *Train* (`forward_train`): one random crop size (base 256² jittered ×0.75–1.25, snapped to a multiple of 8), `crop_num=2` random bboxes; each crop's feature region is resized, SegHead'd, then **pasted back and averaged** with the LR base. Returns a dict `{lr_logits, hr_bboxes, hr_logits, hr_scale, fuse_logits}` for multi-scale supervision.
  - *Test* (`forward_test`): a **deterministic sliding window** of 256² crops tiling the whole image; logits accumulate into a `count_map` (LR weight 1, each covering HR crop +1) and are averaged → seamless, reproducible full-res output. Returns a plain tensor.
- Output: fused logits **1024×1024×7**.

**Alternative decoders** (same encoder / loss / data, swap via config
`model.decoder.type`): `UPerNetDecoder` (PSP + FPN), `SegFormerDecoder`
(all-MLP), `SkipDecoder` (U-Net). `DECODER_REGISTRY = {"hrdecoder",
"upernet", "segformer", "skip"}`; `build_decoder(type, **kwargs)`. This is
what lets the portal sniff a checkpoint's decoder from its weight-key names
and refuse incompatible ones.

## `losses.py` — `CombinedLoss`

`total = 0.5·CE + 0.3·Dice + 0.2·Edge` (weights from `loss:` config), each
returned in a dict `{loss, ce_loss, dice_loss, edge_loss}`.

- **CE** — `nn.CrossEntropyLoss(ignore_index=255)` (the config key is named `bce_weight` for historical reasons but the term is Cross-Entropy).
- **Dice** — softmax + one-hot, per-class overlap, `smooth=1`. Directly optimises overlap → the fix for severe class imbalance (background = 71.7 % of pixels, Utility just 0.57 %).
- **Edge** — fixed 3×3 Sobel-x/-y gradient magnitude of prediction vs GT, each max-normalised, scored with **MSE** (chosen over BCE because MSE is FP16/autocast-safe). A low-weight boundary-consistency regulariser.
- **Multi-scale supervision** (applied in `lightning_module.py`): `total = 1.0·Fuse + 0.5·LR + 0.1·HR`. The HR term crops the GT (nearest-neighbour) to each random HR bbox so the local head is supervised on exactly the region it predicted.

## `metrics.py` — `SegmentationMetrics`

Accumulates an `(num_classes, num_classes)` confusion matrix over batches
(`update(preds, targets)`, ignoring `ignore_index=255`). `compute()` returns
per-class `iou/f1/precision/recall`, plus `mIoU, mF1, mPrecision, mRecall`
(nan-mean) and `overall_accuracy`. `summary_str()` pretty-prints the table.

## `dataset.py` — `SegDataModule` + tile datasets

- **`TileSegDataset`** — reads tile GeoTIFFs (first ≤3 bands, padded to 3, uint8) and masks (band 1) via rasterio; resizes to `tile_size²` (bilinear image / **nearest mask**). Train augmentation: random h/v flip, random 0/90/180/270° rotation, and (p≈0.3) brightness+contrast jitter ×0.8–1.2. **ImageNet normalisation** (`mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]`) — the SAT-493M backbone expects it.
- **`TileTestDataset`** — images only, carries `row/col` for stitching.
- **`SegDataModule`** — reads `label_index.csv` + `tile_index.csv`. **Splits are geographic, not a random %:** *train* = all labelled tiles not in the hold-out; *val* = labelled tiles of `test_dataset`/`test_datasets`; *test* = all tiles of the hold-out. If `full_training: true` (or no hold-out named) → 100 % train, no val. Train loader `shuffle=True, drop_last=True`; empty splits return `None`.

## `lightning_module.py` — `DINOv3HRDecoderModule`

`num_classes = len(classes) + 1` (background added as class 0). Builds the
encoder from `model.encoder`, the decoder via `build_decoder(model.decoder)`,
and `CombinedLoss` from `loss`.

- **`training_step`** — HRDecoder path computes CombinedLoss on `lr_logits` and `fuse_logits`, and on `hr_logits` against the GT cropped+nearest-resized to each HR bbox; `total = 0.5·LR + 0.1·HR + 1.0·Fuse`. Non-HR decoders get a single loss on `fuse_logits`.
- **Optimizer/schedule** (`configure_optimizers`): **AdamW** (`weight_decay=1e-4`), **differential LR** (encoder ×0.1, projections + decoder ×1) via `encoder.get_param_groups()`. **5-epoch linear warmup** (`start_factor=0.01`) chained (`SequentialLR`) into **cosine annealing** (`η_min=1e-7`), stepped per training step.
- **Logged:** `train/loss` (+ `ce/dice/edge/lr_loss/hr_loss`), `val/loss` (+ components), and all per-class + mean metrics under `train/*` and `val/*`. **Checkpoint monitors and early-stopping live in the `training/` scripts, not here.**
