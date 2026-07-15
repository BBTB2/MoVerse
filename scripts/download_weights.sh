#!/bin/bash
# Download all model weights for MoVerse pipeline.
#
# Usage:
#   ./scripts/download_weights.sh [--checkpoints_dir <dir>]
#
# Default: checkpoints/ relative to project root.
#
# Downloads:
#   1. MoVerse proprietary weights (Orange-3DV-Team/MoVerse)
#      - AutoLevel camera calibration (autolevel.pth)
#      - Pano LoRA (pytorch_lora_weights.safetensors)
#      - Gaussian Predictor (gaussian_predictor_checkpoint.pth)
#      - Self-Forcing Video Gen (self_forcing_video.pt)
#      - TAEHV lightweight VAE (taew2_1.pth)
#   2. Open-source models (from official HF repos)
#      - FLUX.1-Fill-dev (black-forest-labs)
#      - DA3 (depth-anything)
#      - Wan2.1-T2V-1.3B (Wan-AI)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$ROOT_DIR"

# Parse arguments
CHECKPOINTS_DIR="${CHECKPOINTS_DIR:-checkpoints}"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --checkpoints_dir)
            CHECKPOINTS_DIR="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--checkpoints_dir <dir>]"
            exit 1
            ;;
    esac
done

mkdir -p "$CHECKPOINTS_DIR"

echo "============================================================"
echo "  MoVerse Weight Download"
echo "  Target: $CHECKPOINTS_DIR/"
echo "============================================================"

# Check huggingface-cli
if ! command -v huggingface-cli &> /dev/null; then
    echo "[ERROR] huggingface-cli not found. Install with: pip install huggingface_hub"
    exit 1
fi

# ── 1. MoVerse proprietary weights ────────────────────────────────────────
echo ""
echo "[1/3] Downloading MoVerse proprietary weights..."
echo "  Repo: Orange-3DV-Team/MoVerse"

HF_MOVERSE="Orange-3DV-Team/MoVerse"

# AutoLevel camera calibration
huggingface-cli download "$HF_MOVERSE" gimbal360/autolevel.pth --local-dir "$CHECKPOINTS_DIR"

# Pano LoRA
huggingface-cli download "$HF_MOVERSE" gimbal360/pytorch_lora_weights.safetensors --local-dir "$CHECKPOINTS_DIR"

# Gaussian Predictor (includes DINOv3 encoder, no separate DA2 needed)
huggingface-cli download "$HF_MOVERSE" gaussian_predictor_checkpoint.pth --local-dir "$CHECKPOINTS_DIR"

# Self-Forcing Video Generation checkpoint
huggingface-cli download "$HF_MOVERSE" self_forcing_video.pt --local-dir "$CHECKPOINTS_DIR"

# TAEHV lightweight VAE decoder
huggingface-cli download "$HF_MOVERSE" taew2_1.pth --local-dir "$CHECKPOINTS_DIR"

echo "[1/3] MoVerse weights downloaded."

# ── 2. Open-source models ──────────────────────────────────────────────────
echo ""
echo "[2/3] Downloading open-source models..."

# FLUX.1-Fill-dev (inpainting/outpainting base model)
echo "  Downloading FLUX.1-Fill-dev..."
huggingface-cli download black-forest-labs/FLUX.1-Fill-dev --local-dir "$CHECKPOINTS_DIR/FLUX.1-Fill-dev"

# DA3 (Depth Anything 3 — panoramic depth estimation)
echo "  Downloading DA3..."
huggingface-cli download depth-anything/DA3NESTED-GIANT-LARGE-1.1 --local-dir "$CHECKPOINTS_DIR/DA3NESTED-GIANT-LARGE-1.1"

# Wan2.1-T2V-1.3B (text-to-video base model)
echo "  Downloading Wan2.1-T2V-1.3B..."
huggingface-cli download Wan-AI/Wan2.1-T2V-1.3B --local-dir "$CHECKPOINTS_DIR/Wan2.1-T2V-1.3B"

echo "[2/3] Open-source models downloaded."

# ── 3. Verify ──────────────────────────────────────────────────────────────
echo ""
echo "[3/3] Verifying weights..."

verify_file() {
    local path="$1"
    local name="$2"
    if [ -f "$path" ]; then
        local size=$(du -h "$path" | cut -f1)
        echo "  ✓ $name ($size)"
    else
        echo "  ✗ $name — NOT FOUND at $path"
        return 1
    fi
}

ERRORS=0

verify_file "$CHECKPOINTS_DIR/gimbal360/autolevel.pth" "AutoLevel" || ERRORS=$((ERRORS+1))
verify_file "$CHECKPOINTS_DIR/gimbal360/pytorch_lora_weights.safetensors" "Pano LoRA" || ERRORS=$((ERRORS+1))
verify_file "$CHECKPOINTS_DIR/gaussian_predictor_checkpoint.pth" "Gaussian Predictor" || ERRORS=$((ERRORS+1))
verify_file "$CHECKPOINTS_DIR/self_forcing_video.pt" "Self-Forcing Video" || ERRORS=$((ERRORS+1))
verify_file "$CHECKPOINTS_DIR/taew2_1.pth" "TAEHV VAE" || ERRORS=$((ERRORS+1))

# Check directories
for dir in FLUX.1-Fill-dev DA3NESTED-GIANT-LARGE-1.1 Wan2.1-T2V-1.3B; do
    if [ -d "$CHECKPOINTS_DIR/$dir" ]; then
        echo "  ✓ $dir/"
    else
        echo "  ✗ $dir/ — NOT FOUND"
        ERRORS=$((ERRORS+1))
    fi
done

echo ""
if [ $ERRORS -eq 0 ]; then
    echo "============================================================"
    echo "  All weights verified successfully!"
    echo "  Location: $CHECKPOINTS_DIR/"
    echo "============================================================"
else
    echo "============================================================"
    echo "  WARNING: $ERRORS file(s) missing!"
    echo "  Check the download logs above."
    echo "============================================================"
    exit 1
fi
