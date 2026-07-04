#!/usr/bin/env python3
"""
DINOv3 Encoder Wrapper
======================
Wraps DINOv3 ViT-L (loaded from local repo) to extract multi-scale
intermediate features for use with HRDecoder.
"""

import sys
import torch
import torch.nn as nn

# Add local DINOv3 repo to path.
# Bundle layout:
#   <BUNDLE>/dinov3_hrdecoder_pipeline/models/encoder.py   (this file)
#   <BUNDLE>/models/dinov3/                                (cloned by setup_env.sh)
DINOV3_REPO = str(
    __import__("pathlib").Path(__file__).resolve().parent.parent.parent
    / "models" / "dinov3"
)
if DINOV3_REPO not in sys.path:
    sys.path.insert(0, DINOV3_REPO)


class DINOv3Encoder(nn.Module):
    """
    DINOv3 ViT encoder that returns multi-scale features from
    intermediate transformer blocks.

    Given a ViT-L with 24 blocks (patch_size=16), extracting from
    layers [5, 11, 17, 23] gives 4 feature scales at (H/16, W/16).

    We project each to a common dimension for the decoder.
    """

    def __init__(
        self,
        arch: str = "vitl16",
        intermediate_layers: list = None,
        freeze_first_n_blocks: int = 12,
        embed_dim: int = 1024,
        out_dim: int = 256,
        pretrained: bool = True,
        weights: str = "LVD1689M",
    ):
        super().__init__()
        if intermediate_layers is None:
            intermediate_layers = [5, 11, 17, 23]

        self.intermediate_layers = intermediate_layers
        self.embed_dim = embed_dim
        self.out_dim = out_dim

        # Load DINOv3 from local repo
        self.backbone = self._load_dinov3(arch, pretrained, weights)
        self.patch_size = self.backbone.patch_size

        # Freeze early layers for stability
        self._freeze_layers(freeze_first_n_blocks)

        # Project each intermediate feature to out_dim
        self.projections = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(embed_dim),
                nn.Linear(embed_dim, out_dim),
                nn.GELU(),
            )
            for _ in intermediate_layers
        ])

    @staticmethod
    def _load_dinov3(arch: str, pretrained: bool, weights: str):
        """
        Load DINOv3 from local repo at models/dinov3/.

        Supports loading weights from:
        1. Facebook CDN (default, via Weights enum)
        2. HuggingFace Hub (if CDN is blocked — gated repo, needs access approval)
        3. Local .pth/.safetensors file path
        """
        from dinov3.hub.backbones import (
            dinov3_vits16, dinov3_vitb16, dinov3_vitl16,
            dinov3_vitl16plus, dinov3_vith16plus, dinov3_vit7b16,
            Weights,
        )

        arch_map = {
            "vits16": dinov3_vits16,
            "vitb16": dinov3_vitb16,
            "vitl16": dinov3_vitl16,
            "vitl16plus": dinov3_vitl16plus,
            "vith16plus": dinov3_vith16plus,
            "vit7b16": dinov3_vit7b16,
        }

        factory = arch_map.get(arch)
        if factory is None:
            raise ValueError(f"Unknown DINOv3 arch: {arch}. Choose from {list(arch_map.keys())}")

        if not pretrained:
            model = factory(pretrained=False)
            print(f"  DINOv3 {arch} loaded (random init)")
            return model

        weight_map = {
            "LVD1689M": Weights.LVD1689M,
            "SAT493M": Weights.SAT493M,
        }

        # Try loading weights via multiple strategies
        # Strategy 1: Direct CDN download (original method)
        w = weight_map.get(weights)
        if w is not None:
            try:
                model = factory(pretrained=True, weights=w)
                print(f"  DINOv3 {arch} loaded (pretrained, weights={weights})")
                return model
            except Exception as e:
                print(f"  CDN download failed ({e}), trying HuggingFace Hub...")

        # Strategy 2: HuggingFace Hub download
        hf_repo_map = {
            ("vitl16", "SAT493M"): "facebook/dinov3-vitl16-pretrain-sat493m",
            ("vitl16", "LVD1689M"): "facebook/dinov3-vitl16-pretrain-lvd1689m",
            ("vitb16", "LVD1689M"): "facebook/dinov3-vitb16-pretrain-lvd1689m",
            ("vits16", "LVD1689M"): "facebook/dinov3-vits16plus-pretrain-lvd1689m",
            ("vith16plus", "LVD1689M"): "facebook/dinov3-vith16plus-pretrain-lvd1689m",
            ("vit7b16", "LVD1689M"): "facebook/dinov3-vit7b16-pretrain-lvd1689m",
            ("vitl16", "SAT493M"): "facebook/dinov3-vitl16-pretrain-sat493m",
            ("vit7b16", "SAT493M"): "facebook/dinov3-vit7b16-pretrain-sat493m",
        }
        hf_repo = hf_repo_map.get((arch, weights))
        if hf_repo:
            try:
                model = factory(pretrained=False)
                state_dict = DINOv3Encoder._load_from_huggingface(hf_repo)
                # strict=False: rope_embed.periods is computed, not in HF weights
                missing, unexpected = model.load_state_dict(state_dict, strict=False)
                if unexpected:
                    print(f"  WARN: Unexpected keys: {unexpected[:5]}")
                if missing:
                    non_trivial = [k for k in missing if "rope_embed" not in k]
                    if non_trivial:
                        raise RuntimeError(f"Missing critical keys: {non_trivial[:5]}")
                print(f"  DINOv3 {arch} loaded from HuggingFace ({hf_repo})")
                return model
            except Exception as e:
                print(f"  HuggingFace download failed ({e})")

        # Strategy 3: Local file path
        if isinstance(weights, str) and (weights.endswith('.pth') or weights.endswith('.safetensors')):
            try:
                import pathlib
                if pathlib.Path(weights).exists():
                    model = factory(pretrained=False)
                    state_dict = DINOv3Encoder._load_local_weights(weights)
                    model.load_state_dict(state_dict, strict=True)
                    print(f"  DINOv3 {arch} loaded from local file ({weights})")
                    return model
            except Exception as e:
                print(f"  Local file loading failed ({e})")

        # Fallback: random init
        print(f"  WARN: All weight loading strategies failed, using random init")
        print(f"  To fix: visit https://huggingface.co/{hf_repo or 'facebook/dinov3-vitl16-pretrain-sat493m'}")
        print(f"  and request access, then run: huggingface-cli login")
        model = factory(pretrained=False)
        return model

    @staticmethod
    def _load_from_huggingface(repo_id: str):
        """Download and load state dict from HuggingFace Hub, converting key names."""
        from huggingface_hub import hf_hub_download

        # Try safetensors first, then pytorch bin
        hf_sd = None
        for filename in ["model.safetensors", "pytorch_model.bin"]:
            try:
                path = hf_hub_download(repo_id=repo_id, filename=filename)
                if filename.endswith(".safetensors"):
                    from safetensors.torch import load_file
                    hf_sd = load_file(path)
                else:
                    hf_sd = torch.load(path, map_location="cpu")
                break
            except Exception:
                continue
        if hf_sd is None:
            raise RuntimeError(f"Could not download weights from {repo_id}")

        # Convert HuggingFace Transformers key names → native DINOv3 format
        return DINOv3Encoder._convert_hf_to_native(hf_sd)

    @staticmethod
    def _convert_hf_to_native(hf_sd: dict) -> dict:
        """
        Convert HuggingFace Transformers state dict keys to native DINOv3 format.

        HF format                          → Native format
        embeddings.cls_token               → cls_token
        embeddings.mask_token              → mask_token (squeeze dim)
        embeddings.patch_embeddings.*      → patch_embed.proj.*
        embeddings.register_tokens         → storage_tokens
        layer.N.attention.{q,k,v}_proj.*   → blocks.N.attn.qkv.* (concatenated)
        layer.N.attention.o_proj.*         → blocks.N.attn.proj.*
        layer.N.layer_scale1.lambda1       → blocks.N.ls1.gamma
        layer.N.layer_scale2.lambda1       → blocks.N.ls2.gamma
        layer.N.mlp.up_proj.*              → blocks.N.mlp.fc1.*
        layer.N.mlp.down_proj.*            → blocks.N.mlp.fc2.*
        layer.N.norm{1,2}.*               → blocks.N.norm{1,2}.*
        norm.*                             → norm.*
        """
        import re

        native_sd = {}

        # Embeddings
        if "embeddings.cls_token" in hf_sd:
            native_sd["cls_token"] = hf_sd["embeddings.cls_token"]
        if "embeddings.mask_token" in hf_sd:
            t = hf_sd["embeddings.mask_token"]
            native_sd["mask_token"] = t.squeeze(1) if t.dim() == 3 else t
        if "embeddings.patch_embeddings.weight" in hf_sd:
            native_sd["patch_embed.proj.weight"] = hf_sd["embeddings.patch_embeddings.weight"]
        if "embeddings.patch_embeddings.bias" in hf_sd:
            native_sd["patch_embed.proj.bias"] = hf_sd["embeddings.patch_embeddings.bias"]
        if "embeddings.register_tokens" in hf_sd:
            native_sd["storage_tokens"] = hf_sd["embeddings.register_tokens"]

        # Norm
        if "norm.weight" in hf_sd:
            native_sd["norm.weight"] = hf_sd["norm.weight"]
        if "norm.bias" in hf_sd:
            native_sd["norm.bias"] = hf_sd["norm.bias"]

        # Find all layer indices
        layer_indices = set()
        for k in hf_sd:
            m = re.match(r"layer\.(\d+)\.", k)
            if m:
                layer_indices.add(int(m.group(1)))

        for idx in sorted(layer_indices):
            prefix_hf = f"layer.{idx}"
            prefix_nat = f"blocks.{idx}"

            # Attention: merge separate q,k,v into fused qkv
            q_w = hf_sd.get(f"{prefix_hf}.attention.q_proj.weight")
            k_w = hf_sd.get(f"{prefix_hf}.attention.k_proj.weight")
            v_w = hf_sd.get(f"{prefix_hf}.attention.v_proj.weight")
            if q_w is not None and k_w is not None and v_w is not None:
                native_sd[f"{prefix_nat}.attn.qkv.weight"] = torch.cat([q_w, k_w, v_w], dim=0)

            q_b = hf_sd.get(f"{prefix_hf}.attention.q_proj.bias")
            v_b = hf_sd.get(f"{prefix_hf}.attention.v_proj.bias")
            if q_b is not None and v_b is not None:
                dim = q_b.shape[0]
                k_b = torch.zeros(dim, dtype=q_b.dtype)
                native_sd[f"{prefix_nat}.attn.qkv.bias"] = torch.cat([q_b, k_b, v_b], dim=0)
                # bias_mask: 1 for q, 0 for k (no bias), 1 for v
                native_sd[f"{prefix_nat}.attn.qkv.bias_mask"] = torch.cat([
                    torch.ones(dim), torch.zeros(dim), torch.ones(dim)
                ])

            # Output projection
            if f"{prefix_hf}.attention.o_proj.weight" in hf_sd:
                native_sd[f"{prefix_nat}.attn.proj.weight"] = hf_sd[f"{prefix_hf}.attention.o_proj.weight"]
            if f"{prefix_hf}.attention.o_proj.bias" in hf_sd:
                native_sd[f"{prefix_nat}.attn.proj.bias"] = hf_sd[f"{prefix_hf}.attention.o_proj.bias"]

            # Layer scales
            if f"{prefix_hf}.layer_scale1.lambda1" in hf_sd:
                native_sd[f"{prefix_nat}.ls1.gamma"] = hf_sd[f"{prefix_hf}.layer_scale1.lambda1"]
            if f"{prefix_hf}.layer_scale2.lambda1" in hf_sd:
                native_sd[f"{prefix_nat}.ls2.gamma"] = hf_sd[f"{prefix_hf}.layer_scale2.lambda1"]

            # MLP
            if f"{prefix_hf}.mlp.up_proj.weight" in hf_sd:
                native_sd[f"{prefix_nat}.mlp.fc1.weight"] = hf_sd[f"{prefix_hf}.mlp.up_proj.weight"]
            if f"{prefix_hf}.mlp.up_proj.bias" in hf_sd:
                native_sd[f"{prefix_nat}.mlp.fc1.bias"] = hf_sd[f"{prefix_hf}.mlp.up_proj.bias"]
            if f"{prefix_hf}.mlp.down_proj.weight" in hf_sd:
                native_sd[f"{prefix_nat}.mlp.fc2.weight"] = hf_sd[f"{prefix_hf}.mlp.down_proj.weight"]
            if f"{prefix_hf}.mlp.down_proj.bias" in hf_sd:
                native_sd[f"{prefix_nat}.mlp.fc2.bias"] = hf_sd[f"{prefix_hf}.mlp.down_proj.bias"]

            # Norms
            for norm_name in ["norm1", "norm2"]:
                for param in ["weight", "bias"]:
                    key_hf = f"{prefix_hf}.{norm_name}.{param}"
                    if key_hf in hf_sd:
                        native_sd[f"{prefix_nat}.{norm_name}.{param}"] = hf_sd[key_hf]

        return native_sd

    @staticmethod
    def _load_local_weights(path: str):
        """Load state dict from a local file."""
        import torch
        if path.endswith(".safetensors"):
            from safetensors.torch import load_file
            return load_file(path)
        else:
            return torch.load(path, map_location="cpu")

    def _freeze_layers(self, n_blocks: int):
        """Freeze patch embed and first n transformer blocks."""
        for param in self.backbone.patch_embed.parameters():
            param.requires_grad = False

        if hasattr(self.backbone, "blocks"):
            for i, block in enumerate(self.backbone.blocks):
                if i < n_blocks:
                    for param in block.parameters():
                        param.requires_grad = False

    def forward(self, x: torch.Tensor) -> list:
        """
        Args:
            x: (B, 3, H, W) input images (H,W must be divisible by patch_size=16)

        Returns:
            list of (B, out_dim, h, w) feature maps from each intermediate layer
        """
        B, _, H, W = x.shape
        h = H // self.patch_size
        w = W // self.patch_size

        # DINOv3's get_intermediate_layers returns spatial features when reshape=True
        features = self.backbone.get_intermediate_layers(
            x, n=self.intermediate_layers, reshape=False, norm=True
        )

        outputs = []
        for i, feat in enumerate(features):
            # feat: (B, num_patches, embed_dim) — CLS/storage already stripped
            # Project and reshape to spatial
            feat = self.projections[i](feat)  # (B, h*w, out_dim)
            feat = feat.permute(0, 2, 1).reshape(B, self.out_dim, h, w)
            outputs.append(feat)

        return outputs

    def get_param_groups(self, encoder_lr_mult: float = 0.1, base_lr: float = 1e-4):
        """Return param groups with different LRs for encoder vs projections."""
        encoder_params = []
        proj_params = []

        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if "projections" in name:
                proj_params.append(param)
            else:
                encoder_params.append(param)

        return [
            {"params": encoder_params, "lr": base_lr * encoder_lr_mult},
            {"params": proj_params, "lr": base_lr},
        ]
