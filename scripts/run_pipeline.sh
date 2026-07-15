#!/bin/bash
# Headless pipeline: image → panorama → depth → 3DGS → video
#
# Usage:
#   ./scripts/run_pipeline.sh --image <path> --prompt "<text>" [--back_prompt "<text>"] [--traj <traj.txt>] [--output_dir <dir>]
#   ./scripts/run_pipeline.sh --example
#
# Pipeline steps:
#   1. pano_gen/inference.py    — Perspective image → 360° panorama (FluxFill + LoRA)
#   2. gaussian_gen/infer_da3.py — Panorama → depth map (DA3)
#   3. gaussian_gen/infer_pano.py — Panorama + depth → 3DGS PLY (Sharp predictor)
#   4. scripts/run_pipeline.py   — 3DGS + trajectory → video (Self-Forcing pipeline)
#                                 Outputs: gen video, render video, side-by-side comparison

set -e

# ── Default paths (relative to project root) ──────────────────────────────
GIMBAL360_DIR="${GIMBAL360_DIR:-checkpoints/gimbal360}"
FLUX_MODEL_PATH="${FLUX_MODEL_PATH:-checkpoints/FLUX.1-Fill-dev}"
DA3_MODEL_DIR="${DA3_MODEL_DIR:-checkpoints/DA3NESTED-GIANT-LARGE-1.1}"
GAUSSIAN_CKPT="${GAUSSIAN_CKPT:-checkpoints/gaussian_predictor_checkpoint.pth}"
VIDEO_CKPT="${VIDEO_CKPT:-checkpoints/self_forcing_video.pt}"
TAEHV_CKPT="${TAEHV_CKPT:-checkpoints/taew2_1.pth}"
CONFIG_PATH="${CONFIG_PATH:-configs/inference/cond_inference_multi.yaml}"

# ── Parse arguments ────────────────────────────────────────────────────────
EXAMPLE_MODE=false
IMAGE_PATH=""
PROMPT=""
BACK_PROMPT=""
TRAJ_FILE=""
OUTPUT_DIR="output"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --example)
            EXAMPLE_MODE=true
            shift
            ;;
        --image)
            IMAGE_PATH="$2"
            shift 2
            ;;
        --prompt)
            PROMPT="$2"
            shift 2
            ;;
        --back_prompt)
            BACK_PROMPT="$2"
            shift 2
            ;;
        --traj)
            TRAJ_FILE="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# ── Example mode: use alcove defaults ─────────────────────────────────────
if [ "$EXAMPLE_MODE" = true ]; then
    IMAGE_PATH="examples/alcove/input.png"
    PROMPT="a bright breakfast alcove, curved booth seating, small round table, colorful patterned wall tiles, pendant lamp, one window with soft daylight"
    BACK_PROMPT="with additional alcove seating and a small kitchen behind"
    TRAJ_FILE="examples/alcove/trajectory.txt"
    OUTPUT_DIR="examples/alcove"
fi

# ── Validate ──────────────────────────────────────────────────────────────
if [ -z "$IMAGE_PATH" ] || [ -z "$PROMPT" ]; then
    echo "Usage:"
    echo "  $0 --image <path> --prompt \"<text>\" [--back_prompt \"<text>\"] [--traj <traj.txt>] [--output_dir <dir>]"
    echo "  $0 --example"
    exit 1
fi

export PYTHONPATH=.
mkdir -p "$OUTPUT_DIR"

# Compute file paths
BASE_NAME=$(basename "$IMAGE_PATH")
BASE_NAME="${BASE_NAME%.*}"
PANO_FILE="$OUTPUT_DIR/${BASE_NAME}_pano.png"
DEPTH_FILE="$OUTPUT_DIR/depth.npy"
PLY_FILE="$OUTPUT_DIR/${BASE_NAME}_pano.ply"
VIDEO_FILE="$OUTPUT_DIR/video.mp4"

echo "============================================================"
echo "  MoVerse Headless Pipeline"
echo "  Scene:     $BASE_NAME"
echo "  Image:     $IMAGE_PATH"
echo "  Output:    $OUTPUT_DIR"
echo "  Panorama:  $PANO_FILE"
echo "  Depth:     $DEPTH_FILE"
echo "  PLY:       $PLY_FILE"
echo "  Video:     $VIDEO_FILE"
echo "============================================================"

# ── Step 1: Generate panorama (FluxFill + LoRA) ──────────────────────────
echo ""
echo "[Step 1/4] Generating panorama..."
echo "  Input:  $IMAGE_PATH"
echo "  Output: $PANO_FILE"

PANO_ARGS="--image $IMAGE_PATH --prompt \"$PROMPT\""
if [ -n "$BACK_PROMPT" ]; then
    PANO_ARGS="$PANO_ARGS --back_prompt \"$BACK_PROMPT\""
fi
PANO_ARGS="$PANO_ARGS --ckpt_dir $GIMBAL360_DIR --basemodel_name_or_path $FLUX_MODEL_PATH --output_dir $OUTPUT_DIR"

echo "  $ python pano_gen/inference.py $PANO_ARGS"
eval "python pano_gen/inference.py $PANO_ARGS"

# ── Step 2: Estimate depth (DA3) ─────────────────────────────────────────
echo ""
echo "[Step 2/4] Estimating depth (DA3)..."
echo "  Input:  $PANO_FILE"
echo "  Output: $DEPTH_FILE"

python gaussian_gen/infer_da3.py \
    --path "$PANO_FILE" \
    --output_dir "$OUTPUT_DIR" \
    --model_dir "$DA3_MODEL_DIR"

# ── Step 3: Generate 3DGS PLY (Sharp predictor) ──────────────────────────
echo ""
echo "[Step 3/4] Generating 3DGS PLY..."
echo "  Input:  $PANO_FILE + $DEPTH_FILE"
echo "  Output: $PLY_FILE"

python gaussian_gen/infer_pano.py \
    --image "$PANO_FILE" \
    --depth "$DEPTH_FILE" \
    --checkpoint "$GAUSSIAN_CKPT" \
    --output_dir "$OUTPUT_DIR"

# ── Step 4: Render video (Self-Forcing pipeline) ─────────────────────────
echo ""
echo "[Step 4/4] Rendering video..."
echo "  Scene:   $PLY_FILE"
echo "  Traj:    ${TRAJ_FILE:-<none>}"
echo "  Output:  $VIDEO_FILE"

# Derive video prompt from pano prompt
VIDEO_PROMPT="A video of $PROMPT"

python scripts/run_pipeline.py \
    --scene "$PLY_FILE" \
    --traj "$TRAJ_FILE" \
    --config_path "$CONFIG_PATH" \
    --checkpoint_path "$VIDEO_CKPT" \
    --taehv \
    --taehv_checkpoint "$TAEHV_CKPT" \
    --prompt "$VIDEO_PROMPT" \
    --output "$VIDEO_FILE" \
    --num_frames 161

echo ""
echo "============================================================"
echo "  Pipeline complete!"
echo "  Output: $OUTPUT_DIR/"
echo "    - panorama:       $PANO_FILE"
echo "    - depth:          $DEPTH_FILE"
echo "    - 3DGS PLY:       $PLY_FILE"
echo "    - gen video:      $VIDEO_FILE"
echo "    - render video:   ${VIDEO_FILE%.mp4}_render.mp4"
echo "    - side-by-side:   ${VIDEO_FILE%.mp4}_side_by_side.mp4"
echo "============================================================"
