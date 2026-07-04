#!/bin/bash
# ============================================================
# DINOv3 + HRDecoder Segmentation Pipeline — Env Setup
# ============================================================
# The submission bundle itself is the workspace. This script:
#   1. Creates conda env 'svamitva2' (Python 3.10)
#   2. Installs PyTorch (CUDA 12.6 wheels) + GDAL
#   3. Installs Python deps from requirements.txt
#   4. Clones the DINOv3 source repo into <BUNDLE>/models/dinov3/
#   5. Creates the runtime-output directory skeleton
#   6. Substitutes the bundle path into all YAML configs
#   7. Imports the deps to verify the install
# ============================================================
set -e

# Bundle = directory holding this script. The bundle IS the workspace.
BUNDLE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "============================================================"
echo "  DINOv3 + HRDecoder pipeline — environment setup"
echo "============================================================"
echo "  Bundle (workspace): $BUNDLE"
echo ""

# ── 1. Conda environment ────────────────────────────────────
echo "Step 1: Creating conda environment 'svamitva2' (Python 3.10)…"
if conda info --envs | grep -q "^svamitva2 "; then
    echo "  Env 'svamitva2' already exists — reusing."
else
    conda create -n svamitva2 python=3.10 -y
fi
eval "$(conda shell.bash hook)"
conda activate svamitva2

# ── 2. PyTorch with CUDA (adjust the index URL to match your driver) ──
echo ""
echo "Step 2: Installing PyTorch (CUDA 12.6 wheels)…"
pip install torch==2.7.0 torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu126

# ── 3. GDAL (system-level, required by rasterio/fiona) ──────
echo ""
echo "Step 3: Installing GDAL via conda-forge…"
conda install -c conda-forge gdal -y || \
    echo "  (GDAL install skipped — install manually if rasterio/fiona fail.)"

# ── 4. Python deps from requirements.txt ────────────────────
echo ""
echo "Step 4: Installing pinned Python dependencies…"
pip install -r "$BUNDLE/requirements.txt"

# ── 5. Clone DINOv3 source (weights downloaded on first run via HF) ─
echo ""
echo "Step 5: Cloning DINOv3 source repo…"
DINOV3_DIR="$BUNDLE/models/dinov3"
mkdir -p "$BUNDLE/models"
if [ ! -d "$DINOV3_DIR" ]; then
    git clone https://github.com/facebookresearch/dinov3.git "$DINOV3_DIR"
    echo "  ✓ DINOv3 cloned to $DINOV3_DIR"
else
    echo "  DINOv3 already at $DINOV3_DIR"
fi

# ── 6. Runtime output directories ───────────────────────────
# Raw data lives at $BUNDLE/data/{train,test} (already in place).
# Everything else is generated.
echo ""
echo "Step 6: Creating runtime-output directory skeleton…"
mkdir -p \
    "$BUNDLE/dataset"/{train/{images/{CG,PB},labels/{CG,PB},cog/{CG,PB}},test/{images/{CG,PB},cog/{CG,PB}}} \
    "$BUNDLE/cog"/{train,test}/{CG,PB} \
    "$BUNDLE/tiles" \
    "$BUNDLE/masks" \
    "$BUNDLE/checkpoints" \
    "$BUNDLE/outputs"/{predictions,stitched,evaluation,gpkg,logs} \
    "$BUNDLE/visualizations" \
    "$BUNDLE/logs"

# ── 7. Substitute bundle path into YAML configs ─────────────
# Configs ship with absolute paths pointing at the bundle's prior
# location. Detect the current "workspace" value in data_prep.yaml
# and replace it everywhere with $BUNDLE. Safe to re-run.
echo ""
echo "Step 7: Pointing YAML config paths to $BUNDLE…"
SED_INPLACE=(sed -i)
if [[ "$OSTYPE" == "darwin"* ]]; then
    SED_INPLACE=(sed -i '')
fi
OLD_BUNDLE="$(awk '/^[[:space:]]*workspace:/{
    gsub(/^[[:space:]]*workspace:[[:space:]]*"?/, "")
    gsub(/"?[[:space:]]*$/, "")
    print
    exit
}' "$BUNDLE/dinov3_hrdecoder_pipeline/configs/data_prep.yaml")"

if [ -n "$OLD_BUNDLE" ] && [ "$OLD_BUNDLE" != "$BUNDLE" ]; then
    echo "  Rewriting:  $OLD_BUNDLE  →  $BUNDLE"
    for cfg in \
        "$BUNDLE/dinov3_hrdecoder_pipeline/configs/data_prep.yaml" \
        "$BUNDLE/dinov3_hrdecoder_pipeline/configs/train.yaml" \
        "$BUNDLE/dinov3_hrdecoder_pipeline/configs/train_full.yaml" \
        "$BUNDLE/dinov3_hrdecoder_pipeline/configs/train_2val.yaml"; do
        [ -f "$cfg" ] && "${SED_INPLACE[@]}" "s|$OLD_BUNDLE|$BUNDLE|g" "$cfg"
    done
else
    echo "  Configs already point at $BUNDLE — no rewrite needed."
fi

# ── 8. Portal frontend dependencies ─────────────────────────
echo ""
echo "Step 8: Installing portal frontend deps (Node + npm)…"
if command -v npm >/dev/null 2>&1 && [ -d "$BUNDLE/portal/frontend" ]; then
    ( cd "$BUNDLE/portal/frontend" && npm install --no-audit --no-fund ) \
        || echo "  WARN: npm install failed — re-run inside portal/frontend/ later."
else
    echo "  Skipped — install Node.js (https://nodejs.org/) to build the portal frontend."
fi

# ── 9. Verify imports ───────────────────────────────────────
echo ""
echo "Step 9: Verifying installation…"
python - <<'PY' 2>&1 || echo "  WARN: Some packages may need manual installation"
import torch, rasterio, geopandas, numpy, yaml, pytorch_lightning
print(f"  PyTorch:           {torch.__version__}")
print(f"  CUDA available:    {torch.cuda.is_available()}")
print(f"  rasterio:          {rasterio.__version__}")
print(f"  geopandas:         {geopandas.__version__}")
print(f"  pytorch-lightning: {pytorch_lightning.__version__}")
print(f"  numpy:             {numpy.__version__}")
try:
    import fastapi, titiler.core
    print(f"  fastapi:           {fastapi.__version__}")
    print(f"  titiler.core:      {titiler.core.__version__}")
except Exception as exc:
    print(f"  portal deps:       NOT INSTALLED ({exc!r})")
PY

echo ""
echo "============================================================"
echo "  Setup complete."
echo "============================================================"
echo "  Activate env:   conda activate svamitva2"
echo "  HuggingFace:    huggingface-cli login    (for DINOv3 SAT493M weights)"
echo "  Run pipeline:   see README.md → 'End-to-end workflow'"
echo "  Run portal:     bash start_portal.sh   (then http://localhost:5173)"
echo ""
