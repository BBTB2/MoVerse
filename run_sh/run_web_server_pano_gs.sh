#!/bin/bash
# ============================================================
# Start MoVerse World Engine (streaming server)
# ============================================================
#
# Usage:
#   bash run_sh/run_web_server_pano_gs.sh
#
# Frontend dev mode (optional, in another terminal):
#   cd world_engine && npm run dev
#   Then open http://localhost:5173/moverse/
#
# Or build frontend and use backend directly:
#   cd world_engine && npm run build
#   Then open http://localhost:8889/moverse/
# ============================================================

export CUDA_HOME=${CUDA_HOME:-$CONDA_PREFIX}
export PYTHONPATH=.

python world_engine/server_pano_gs.py \
    --config_path configs/inference/cond_inference_multi.yaml \
    --scenes_config world_engine/scenes_gs.yaml \
    --translation_step 0.02 \
    --rotation_step_deg 0.05 \
    --render_fov_deg 85 \
    --taehv \
    --show_render \
    --pano_ckpt_dir checkpoints/gimbal360 \
    --flux_model_path checkpoints/FLUX.1-Fill-dev \
    --da3_model_dir checkpoints/DA3NESTED-GIANT-LARGE-1.1 \
    --gaussian_ckpt checkpoints/gaussian_predictor_checkpoint.pth \
    --taehv_checkpoint checkpoints/taew2_1.pth \
    --port 8889
