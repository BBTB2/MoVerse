#!/usr/bin/env python3
"""
world_engine/server_pano_gs.py — Self-Forcing Streaming Web Service (Gaussian Splatting)
=================================================================

Integrates the core inference pipeline into a FastAPI web service,
streaming real-time generated video frames to the browser via WebSocket.

Architecture:
  Browser Vue frontend
    ↕ WebSocket (/ws/frames)
  FastAPI backend (this file)
    ├── Control input: {move_local, look_delta}  ← sent by frontend every ~42ms
    ├── Pipeline: Gaussian render new view → VAE encode → DiT inference → VAE decode → JPEG frame
    └── Frame output: binary JPEG ← real-time stream to browser

Rendering (Gaussian Splatting):
  - Input 3DGS PLY or .pt checkpoint file → load Gaussian parameters
  - Uses gsplat rasterization (GPU CUDA accelerated) to render condition frames at target camera poses
  - Gaussian scene covers training view range, supports free movement within the scene

Scene management:
  - Reads multi-scene config from scenes_gs.yaml (PLY/ckpt file + prompt)
  - Recreates renderer + streaming session on scene switch
  - Caches Gaussian parameters to avoid repeated loading

Example startup:
  CUDA_HOME=$CONDA_PREFIX PYTHONPATH=. python world_engine/server_pano_gs.py \\
      --config_path configs/inference/cond_inference_multi.yaml \\
      --use_ema \\
      --scenes_config world_engine/scenes_gs.yaml \\
      --port 8889
"""

import argparse
import asyncio
import io
import json
import math
import os
import shutil
import sys
import tempfile
import threading
import time
import traceback
import queue
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, field

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from omegaconf import OmegaConf
from PIL import Image

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, Form
from fastapi.responses import JSONResponse, Response, FileResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# ── Path setup ──────────────────────────────────────────────────────────────
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# Upload directory for dynamically uploaded scene files (not persisted)
_UPLOAD_DIR = os.path.join(os.environ.get('TEMP', '/tmp'), 'moverse_uploads')
os.makedirs(_UPLOAD_DIR, exist_ok=True)

from pipeline import CausalInferencePipeline
from utils.camera_controller import CameraController

# ── gsplat import ──────────────────────────────────────────────────────────
try:
    from gsplat.rendering import rasterization
    HAS_GSPLAT = True
except Exception as e:
    HAS_GSPLAT = False
    print(f"[WARNING] gsplat unavailable: {type(e).__name__}: {e}")
    print("[WARNING] Please install gsplat: cd gsplat && pip install -e .")


# =============================================================================
# Gaussian data loading (PLY / .pt checkpoint)
# =============================================================================

def _load_gaussian_ply(ply_path: str, device: str = "cuda") -> dict:
    """
    Load a standard 3DGS PLY file (Inria format).

    PLY attribute conventions (consistent with Inria 3DGS):
      - x, y, z          → Gaussian center coordinates
      - f_dc_0/1/2       → Spherical harmonics DC coefficients (RGB)
      - f_rest_0..N      → Higher-order SH coefficients (Inria layout: every 3 form one SH basis RGB)
      - opacity           → Logit-space opacity (requires sigmoid activation)
      - scale_0/1/2       → Log-space scale (requires exp activation)
      - rot_0/1/2/3       → Quaternion (wxyz)

    Returns:
        dict with keys: means, quats, scales, opacities, colors, sh_degree
    """
    _PLY_DTYPES = {
        "float": np.float32, "float32": np.float32,
        "double": np.float64, "float64": np.float64,
        "int": np.int32, "int32": np.int32,
        "uint": np.uint32, "uint32": np.uint32,
        "uchar": np.uint8, "uint8": np.uint8,
        "short": np.int16, "int16": np.int16,
        "ushort": np.uint16, "uint16": np.uint16,
        "char": np.int8, "int8": np.int8,
    }

    with open(ply_path, "rb") as f:
        # Parse ASCII header
        num_vertices = 0
        properties = []    # [(name, np_dtype)]
        current_element = None

        while True:
            line = f.readline().decode("ascii", errors="ignore").strip()
            if line == "end_header":
                break
            tokens = line.split()
            if not tokens:
                continue
            if tokens[0] == "format":
                assert "binary_little_endian" in line, \
                    f"Only binary_little_endian format supported, got: {line}"
            elif tokens[0] == "element":
                current_element = tokens[1]
                if current_element == "vertex":
                    num_vertices = int(tokens[2])
            elif tokens[0] == "property" and current_element == "vertex":
                if tokens[1] == "list":
                    continue   # Skip list properties
                dtype = _PLY_DTYPES.get(tokens[1], np.float32)
                name = tokens[2]
                properties.append((name, dtype))

        assert num_vertices > 0, f"No vertices found in PLY file: {ply_path}"

        # Build structured numpy dtype and read all at once
        np_dtype = np.dtype([(name, dt) for name, dt in properties])
        raw_bytes = f.read(num_vertices * np_dtype.itemsize)
        data = np.frombuffer(raw_bytes, dtype=np_dtype, count=num_vertices)

    prop_names = {n for n, _ in properties}
    print(f"[gs-load] PLY: {num_vertices:,} Gaussians, {len(properties)} properties")

    # ── Coordinates ──────────────────────────────────────────────────────
    xyz = np.stack([data["x"], data["y"], data["z"]], axis=-1).astype(np.float32)

    # ── SH DC ──────────────────────────────────────────────────
    sh_dc = np.stack(
        [data["f_dc_0"], data["f_dc_1"], data["f_dc_2"]], axis=-1
    ).astype(np.float32)

    # ── SH higher-order ─────────────────────────────────────────
    f_rest_keys = sorted(
        [n for n in prop_names if n.startswith("f_rest_")],
        key=lambda x: int(x.split("_")[-1]),
    )
    if f_rest_keys:
        sh_rest = np.stack(
            [data[k].astype(np.float32) for k in f_rest_keys], axis=-1
        )
    else:
        sh_rest = np.zeros((num_vertices, 0), dtype=np.float32)

    # ── Opacity (logit → sigmoid) ─────────────────────────────
    opacity = data["opacity"].astype(np.float32)

    # ── Scale (log → exp) ──────────────────────────────────────
    scale_keys = sorted(
        [n for n in prop_names if n.startswith("scale_")],
        key=lambda x: int(x.split("_")[-1]),
    )
    scales = np.stack([data[k].astype(np.float32) for k in scale_keys], axis=-1)

    # ── Quaternion ────────────────────────────────────────────
    rot_keys = sorted(
        [n for n in prop_names if n.startswith("rot_")],
        key=lambda x: int(x.split("_")[-1]),
    )
    rots = np.stack([data[k].astype(np.float32) for k in rot_keys], axis=-1)

    # ── Convert to torch tensor + activate ─────────────────────────────
    means = torch.from_numpy(xyz).float().to(device)

    # SH DC: (N, 3) → (N, 1, 3)
    sh0 = torch.from_numpy(sh_dc).float().to(device).unsqueeze(1)

    # SH rest: Inria format (N, K*3) → (N, K, 3)
    if sh_rest.shape[1] > 0:
        assert sh_rest.shape[1] % 3 == 0, \
            f"f_rest count {sh_rest.shape[1]} is not a multiple of 3"
        n_sh_rest = sh_rest.shape[1] // 3
        shN = torch.from_numpy(sh_rest).float().to(device)
        shN = shN.reshape(num_vertices, n_sh_rest, 3)
    else:
        shN = torch.zeros(num_vertices, 0, 3, device=device)

    colors = torch.cat([sh0, shN], dim=1)         # (N, total_K, 3)
    sh_degree = int(math.sqrt(colors.shape[1]) - 1)

    quats = torch.from_numpy(rots).float().to(device)
    quats = F.normalize(quats, p=2, dim=-1)

    scales_t = torch.exp(torch.from_numpy(scales).float().to(device))
    opacities_t = torch.sigmoid(torch.from_numpy(opacity).float().to(device))

    print(f"[gs-load] SH degree={sh_degree}, "
          f"colors shape={list(colors.shape)}")

    return {
        "means": means,
        "quats": quats,
        "scales": scales_t,
        "opacities": opacities_t,
        "colors": colors,
        "sh_degree": sh_degree,
    }


def _load_gaussian_ckpt(ckpt_path: str, device: str = "cuda") -> dict:
    """
    Load a gsplat .pt checkpoint file.

    Checkpoint format (same as gsplat/examples/simple_viewer.py):
      {"splats": {"means", "quats", "scales", "opacities", "sh0", "shN"}}
    or directly a splats dict.

    scales/opacities are stored as pre-activation values (log-space / logit-space).
    """
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    if "splats" in ckpt:
        splats = ckpt["splats"]
    else:
        splats = ckpt

    means = splats["means"].float().to(device)
    quats = F.normalize(splats["quats"].float().to(device), p=2, dim=-1)
    scales = torch.exp(splats["scales"].float().to(device))
    opacities = torch.sigmoid(splats["opacities"].float().to(device))
    sh0 = splats["sh0"].float().to(device)
    shN = splats["shN"].float().to(device)

    colors = torch.cat([sh0, shN], dim=-2)      # (N, total_K, 3)
    sh_degree = int(math.sqrt(colors.shape[-2]) - 1)

    print(f"[gs-load] ckpt: {means.shape[0]:,} Gaussians, "
          f"SH degree={sh_degree}")

    return {
        "means": means,
        "quats": quats,
        "scales": scales,
        "opacities": opacities,
        "colors": colors,
        "sh_degree": sh_degree,
    }


def _load_gaussians(gs_file_path: str, device: str = "cuda") -> dict:
    """Select PLY or .pt loader based on file extension."""
    ext = os.path.splitext(gs_file_path)[1].lower()
    if ext == ".ply":
        return _load_gaussian_ply(gs_file_path, device)
    elif ext in (".pt", ".pth"):
        return _load_gaussian_ckpt(gs_file_path, device)
    else:
        raise ValueError(f"Unsupported Gaussian file format: {ext} (supported: .ply / .pt / .pth)")


# =============================================================================
# Gaussian renderer (wraps gsplat rasterization)
# =============================================================================

class GaussianRenderer:
    """
    Wraps gsplat's rasterization function to provide a simple rendering interface.

    Caches Gaussian parameters internally; each render only needs camera parameters.
    """

    def __init__(
        self,
        means: torch.Tensor,      # (N, 3)
        quats: torch.Tensor,      # (N, 4) normalized
        scales: torch.Tensor,     # (N, 3) linear space
        opacities: torch.Tensor,  # (N,) [0, 1]
        colors: torch.Tensor,     # (N, K, 3) SH coefficients
        sh_degree: int,
        width: int,
        height: int,
        device: str = "cuda",
        near_plane: float = 0.01,
        far_plane: float = 1000.0,
        bg_color: tuple = (0.0, 0.0, 0.0),
    ):
        self.means = means
        self.quats = quats
        self.scales = scales
        self.opacities = opacities
        self.colors = colors
        self.sh_degree = sh_degree
        self.width = width
        self.height = height
        self.device = device
        self.near_plane = near_plane
        self.far_plane = far_plane
        self.num_gaussians = means.shape[0]

        # Background color (1, 3)
        self.backgrounds = torch.tensor(
            [list(bg_color)], dtype=torch.float32, device=device
        )

    def render_batch(
        self,
        viewmats: torch.Tensor,   # (C, 4, 4) w2c matrices on GPU
        Ks: torch.Tensor,         # (C, 3, 3) intrinsics on GPU
    ) -> torch.Tensor:
        """
        Batch render multiple viewpoints.

        Args:
            viewmats: (C, 4, 4) world-to-camera matrices, float32, GPU
            Ks:       (C, 3, 3) camera intrinsics, float32, GPU

        Returns:
            rgb: (C, H, W, 3) float32 tensor [0, 1], GPU
        """
        C = viewmats.shape[0]
        bg = self.backgrounds.expand(C, -1)   # (C, 3)

        render_colors, render_alphas, _meta = rasterization(
            self.means,          # (N, 3)
            self.quats,          # (N, 4)
            self.scales,         # (N, 3)
            self.opacities,      # (N,)
            self.colors,         # (N, K, 3)
            viewmats,            # (C, 4, 4)
            Ks,                  # (C, 3, 3)
            self.width,
            self.height,
            sh_degree=self.sh_degree,
            near_plane=self.near_plane,
            far_plane=self.far_plane,
            backgrounds=bg,
            render_mode="RGB",
            packed=False,
        )
        # render_colors: (C, H, W, 3)
        return render_colors.clamp(0.0, 1.0)

    def render_single(
        self,
        w2c_4x4: torch.Tensor,   # (4, 4)
        K_3x3: torch.Tensor,     # (3, 3)
    ) -> torch.Tensor:
        """Render a single viewpoint, returns (H, W, 3) float32 [0, 1]."""
        return self.render_batch(
            w2c_4x4.unsqueeze(0), K_3x3.unsqueeze(0)
        )[0]


# =============================================================================
# Camera utility functions
# =============================================================================

def _look_at_w2c(
    eye: torch.Tensor,
    target: torch.Tensor,
    world_up: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Build world-to-camera matrix (consistent with CameraController: +X right, +Y up, +Z forward).

    Args:
        eye:    Camera world position (3,)
        target: Target world position (3,)
        world_up: World up direction (3,), default [0, 1, 0]

    Returns:
        w2c: (4, 4) float32
    """
    if world_up is None:
        world_up = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32)

    forward = target - eye
    forward = forward / (forward.norm() + 1e-8)

    # Right-handed system: right = forward x up, ensuring camera Y axis points up
    right = torch.cross(forward, world_up)
    if right.norm() < 1e-6:
        # forward is parallel to world_up, use fallback up
        world_up = torch.tensor([0.0, 0.0, -1.0], dtype=torch.float32)
        right = torch.cross(forward, world_up)
    right = right / (right.norm() + 1e-8)

    up = torch.cross(right, forward)
    up = up / (up.norm() + 1e-8)

    R = torch.stack([right, up, forward], dim=0)   # (3, 3)
    t = -R @ eye                                      # (3,)

    w2c = torch.eye(4, dtype=torch.float32)
    w2c[:3, :3] = R
    w2c[:3, 3] = t
    return w2c


def _compute_initial_w2c(
    gs_data: dict,
    camera_position: Optional[list] = None,
    camera_look_at: Optional[list] = None,
) -> torch.Tensor:
    """
    Compute initial camera pose for a Gaussian scene.

    Prioritizes user-specified camera_position / camera_look_at from scenes_gs.yaml,
    otherwise uses identity matrix (camera at origin, default orientation).
    Intrinsics are computed from the render_fov_deg parameter.
    """
    if camera_position is not None and camera_look_at is not None:
        eye = torch.tensor(camera_position, dtype=torch.float32)
        target = torch.tensor(camera_look_at, dtype=torch.float32)
        return _look_at_w2c(eye, target)

    # When camera parameters are not specified, use identity matrix (camera at origin, identity extrinsics)
    print("[camera] camera_position/camera_look_at not specified, using identity matrix as initial extrinsics")
    return torch.eye(4, dtype=torch.float32)



# =============================================================================
# Control accumulator (thread-safe)
# =============================================================================

class ControlAccumulator:
    """
    Accumulates control commands from the frontend for pipeline steps.

    The frontend sends {move_local, look_delta} at ~42ms intervals, while pipeline steps
    take ~800-1200ms. This class accumulates all command deltas during that time,
    consuming 1/n_frames of the accumulated amount before each frame render to ensure
    smooth camera movement.

    When there is no user input, the generation loop blocks via wait_for_input()
    to avoid idle GPU spinning.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._move = [0.0, 0.0, 0.0]
        self._look = [0.0, 0.0]
        self._recenter = False
        self._input_event = threading.Event()

    def add(self, control_msg: dict):
        move = control_msg.get("move_local", [0, 0, 0])
        look = control_msg.get("look_delta", [0, 0])
        with self._lock:
            if control_msg.get("recenter"):
                self._recenter = True
                self._move = [0.0, 0.0, 0.0]
                self._look = [0.0, 0.0]
                self._input_event.set()
                return
            has_movement = (
                any(m != 0 for m in move[:3]) or
                any(l != 0 for l in look[:2])
            )
            self._move[0] += move[0] if len(move) > 0 else 0
            self._move[1] += move[1] if len(move) > 1 else 0
            self._move[2] += move[2] if len(move) > 2 else 0
            self._look[0] += look[0] if len(look) > 0 else 0
            self._look[1] += look[1] if len(look) > 1 else 0
            if has_movement:
                self._input_event.set()

    def wait_for_input(self, timeout: float = 0.5) -> bool:
        return self._input_event.wait(timeout=timeout)

    def consume_for_frames(self, n_frames: int):
        with self._lock:
            total_move = self._move.copy()
            total_look = self._look.copy()
            recenter = self._recenter
            self._move = [0.0, 0.0, 0.0]
            self._look = [0.0, 0.0]
            self._recenter = False
            self._input_event.clear()
        if recenter:
            return [0, 0, 0], [0, 0], True
        n = max(n_frames, 1)
        per_move = [m / n for m in total_move]
        per_look = [l / n for l in total_look]
        return per_move, per_look, False

    @property
    def has_input(self) -> bool:
        with self._lock:
            return (any(m != 0 for m in self._move) or
                    any(l != 0 for l in self._look) or
                    self._recenter)


def _resolve_image_path(img_path: str) -> str:
    """Resolve a relative path to absolute (relative to project root)."""
    if os.path.isabs(img_path):
        return img_path
    return os.path.join(ROOT_DIR, img_path)


# =============================================================================
# Dynamic scene folder loading (in-memory, not persisted)
# =============================================================================

def _normalize_folder_path(path: str) -> str:
    """
    Normalize a user-provided folder path.

    Converts Windows-style paths (C:\\...) to WSL paths (/mnt/c/...) when the
    server is running on Linux, so that paths entered on the Windows browser
    work correctly when the backend runs inside WSL.
    """
    import re
    path = path.strip().strip('"').strip("'").strip()
    # Detect Windows drive path: C:\... or C:/...
    win_match = re.match(r'^([A-Za-z]):[\\/](.*)$', path)
    if win_match and sys.platform.startswith('linux'):
        drive = win_match.group(1).lower()
        rest = win_match.group(2).replace('\\', '/')
        return f'/mnt/{drive}/{rest}'
    # Normalise backslashes to forward slashes
    return path.replace('\\', '/')


def _scan_scene_folder(folder_path: str) -> dict:
    """
    Scan a local folder for a Gaussian scene file and optional thumbnail/prompt.

    Looks for (in priority order):
      - .ply files (prefer scene.ply / point_cloud.ply)
      - .pt / .pth checkpoint files
      - panorama.png / pano.png / preview.png / thumbnail.png (thumbnail)
      - prompt.txt (text prompt for video generation)

    Returns:
        dict with keys: gs_file, input_image, prompt
    """
    if not os.path.isdir(folder_path):
        raise FileNotFoundError(f"Folder not found: {folder_path}")

    ply_files = []
    pt_files = []
    for f in sorted(os.listdir(folder_path)):
        ext = os.path.splitext(f)[1].lower()
        full_path = os.path.join(folder_path, f)
        if os.path.isfile(full_path):
            if ext == '.ply':
                ply_files.append(full_path)
            elif ext in ('.pt', '.pth'):
                pt_files.append(full_path)

    # Select Gaussian file
    gs_file = None
    if ply_files:
        # Prefer common names
        preferred = ('scene.ply', 'point_cloud.ply', 'output.ply')
        gs_file = next(
            (f for f in ply_files if os.path.basename(f).lower() in preferred),
            ply_files[0],
        )
    elif pt_files:
        gs_file = pt_files[0]
    else:
        raise FileNotFoundError(
            f"No .ply or .pt/.pth file found in folder: {folder_path}"
        )

    # Look for thumbnail image
    input_image = ""
    image_candidates = [
        'panorama.png', 'pano.png', 'preview.png', 'thumbnail.png',
        'panorama.jpg', 'pano.jpg', 'preview.jpg', 'thumbnail.jpg',
    ]
    for name in image_candidates:
        img_path = os.path.join(folder_path, name)
        if os.path.exists(img_path):
            input_image = img_path
            break

    # Look for prompt.txt
    prompt = ""
    prompt_path = os.path.join(folder_path, 'prompt.txt')
    if os.path.exists(prompt_path):
        try:
            with open(prompt_path, 'r', encoding='utf-8') as pf:
                prompt = pf.read().strip()
        except Exception:
            pass

    return {
        'gs_file': gs_file,
        'input_image': input_image,
        'prompt': prompt,
    }


# =============================================================================
# Scene cache
# =============================================================================

@dataclass
class SceneData:
    """Precomputed data for a scene (Gaussian parameters + text condition)."""
    name: str
    display_name: str
    gs_file: str                   # PLY or .pt file path
    prompt: str
    input_image_path: str = ""     # Optional: thumbnail / preview image
    # Gaussian data (cached after first load)
    gs_data: Optional[dict] = None  # {means, quats, scales, opacities, colors, sh_degree}
    # Pre-encoded text condition
    cond_dict: Optional[dict] = None
    initialized: bool = False
    # Optional initial camera parameters (read from YAML)
    camera_position: Optional[list] = None   # [x, y, z]
    camera_look_at: Optional[list] = None    # [x, y, z]


# =============================================================================
# Pipeline step execution
# =============================================================================

@dataclass
class PipelineStepTiming:
    """Per-stage timing (ms) for a single pipeline step, for server logging."""
    step_idx: int        = 0
    frames_out: int      = 1
    camera_ms: float     = 0.0
    render_ms: float     = 0.0
    vae_enc_ms: float    = 0.0
    dit_ms: float        = 0.0
    vae_dec_ms: float    = 0.0
    send_ms: float       = 0.0
    total_ms: float      = 0.0

    def log_line(self) -> str:
        return (
            f"  step={self.step_idx:<4d}  frm={self.frames_out}  "
            f"cam={self.camera_ms:6.1f}  render={self.render_ms:7.1f}  "
            f"vae_enc={self.vae_enc_ms:7.1f}  dit={self.dit_ms:7.1f}  "
            f"vae_dec={self.vae_dec_ms:7.1f}  send={self.send_ms:6.1f}  "
            f"total={self.total_ms:7.1f} ms"
        )


def _now_ms() -> float:
    return time.perf_counter() * 1000.0


def _run_pipeline_step(
    step_idx: int,
    camera: CameraController,
    renderer: Optional[GaussianRenderer],
    pipeline: CausalInferencePipeline,
    device: torch.device,
    dtype: torch.dtype,
    per_frame_move: list,
    per_frame_look: list,
) -> tuple:
    """
    Execute one pipeline step, returns (gen_pixel, render_pixel, timing).

    Rendering: gsplat rasterization (GPU Gaussian rasterization)
      step_idx == 1: render 1 frame, cached_encode(1 frame) → 1 latent → 1 output frame
      step_idx > 1:  render 4 frames, cached_encode(4 frames) → 1 latent → 4 output frames
    """
    frames_per_step = 1 if step_idx == 1 else 4
    tm = PipelineStepTiming(step_idx=step_idx, frames_out=frames_per_step)
    t0_total = _now_ms()

    # ── 1. Camera update ──────────────────────────────────────────────────────
    t0 = _now_ms()
    s_t = camera.translation_step
    s_r = camera.rotation_step_deg
    scaled_move = [m * s_t for m in per_frame_move]
    scaled_look = [l * s_r for l in per_frame_look]

    frame_w2cs = []
    for _ in range(frames_per_step):
        camera.update_from_vectors(scaled_move, scaled_look)
        frame_w2cs.append(camera.w2c.clone())     # Keep on GPU
    tm.camera_ms = _now_ms() - t0

    # ── 2. Gaussian rendering ──────────────────────────────────────────────────────
    cond_latent = None
    render_pixel = None

    if renderer is not None:
        t0 = _now_ms()
        with torch.no_grad():
            viewmats = torch.stack(frame_w2cs)                                   # (T, 4, 4)
            Ks = camera.intrinsics.unsqueeze(0).expand(frames_per_step, -1, -1)  # (T, 3, 3)
            rendered_rgb = renderer.render_batch(viewmats, Ks)                   # (T, H, W, 3)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        tm.render_ms = _now_ms() - t0

        # (T, H, W, 3) → (1, T, 3, H, W), [0,1] → [-1,1]
        render_pixel = (rendered_rgb * 2.0 - 1.0).permute(0, 3, 1, 2).unsqueeze(0).to(device)

        # ── 3. VAE encode ──────────────────────────────────────────────────
        t0 = _now_ms()
        pixel_bct = render_pixel.permute(0, 2, 1, 3, 4).to(device=device, dtype=dtype)
        with torch.no_grad():
            cond_latent = pipeline.vae.encode_to_latent_cached(pixel_bct)
        cond_latent = cond_latent.to(dtype=dtype)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        tm.vae_enc_ms = _now_ms() - t0
    else:
        cond_latent = torch.zeros(1, 1, 16, 60, 104, device=device, dtype=dtype)

    # ── 4. DiT inference + VAE decode ──────────────────────────────────────────
    _, T_lat, C_lat, H_lat, W_lat = cond_latent.shape
    noise = torch.randn(1, T_lat, C_lat, H_lat, W_lat, device=device, dtype=dtype)

    _step_timing = {}
    with torch.no_grad():
        gen_pixel = pipeline.streaming_step(
            noise=noise,
            cond_latent=cond_latent,
            _timing=_step_timing,
        )

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    tm.dit_ms = _step_timing.get("dit_ms", 0.0)
    tm.vae_dec_ms = _step_timing.get("vae_dec_ms", 0.0)
    tm.frames_out = gen_pixel.shape[1]
    tm.total_ms = _now_ms() - t0_total

    return gen_pixel, render_pixel, tm


# =============================================================================
# Explore mode: render-only step (no DiT / VAE)
# =============================================================================

# In explore mode, movement is 10x faster than denoise mode.
# This is hardcoded and NOT affected by frontend speed sliders.
_EXPLORE_SPEED_BOOST = 10.0


def _run_explore_step(
    camera: CameraController,
    renderer: Optional[GaussianRenderer],
    device: torch.device,
    per_frame_move: list,
    per_frame_look: list,
) -> tuple:
    """
    Execute a render-only step (explore mode). No DiT inference, no VAE.

    Movement speed is boosted by _EXPLORE_SPEED_BOOST (10x) and is NOT
    affected by the frontend move/look speed sliders.

    Returns:
        (None, render_pixel, timing): render_pixel is [1, T, 3, H, W] float [-1,1]
    """
    frames_per_step = 4
    tm = PipelineStepTiming(step_idx=0, frames_out=frames_per_step)
    t0_total = _now_ms()

    # ── 1. Camera update ──────────────────────────────────────────────────────
    t0 = _now_ms()
    s_t = camera.translation_step * _EXPLORE_SPEED_BOOST
    s_r = camera.rotation_step_deg * _EXPLORE_SPEED_BOOST
    scaled_move = [m * s_t for m in per_frame_move]
    scaled_look = [l * s_r for l in per_frame_look]

    frame_w2cs = []
    for _ in range(frames_per_step):
        camera.update_from_vectors(scaled_move, scaled_look)
        frame_w2cs.append(camera.w2c.clone())
    tm.camera_ms = _now_ms() - t0

    # ── 2. Gaussian rendering only ─────────────────────────────────────────────
    render_pixel = None
    if renderer is not None:
        t0 = _now_ms()
        with torch.no_grad():
            viewmats = torch.stack(frame_w2cs)
            Ks = camera.intrinsics.unsqueeze(0).expand(frames_per_step, -1, -1)
            rendered_rgb = renderer.render_batch(viewmats, Ks)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        tm.render_ms = _now_ms() - t0

        render_pixel = (rendered_rgb * 2.0 - 1.0).permute(0, 3, 1, 2).unsqueeze(0).to(device)

    tm.total_ms = _now_ms() - t0_total
    return None, render_pixel, tm


def _tensor_frame_to_jpeg(frame_chw: torch.Tensor, quality: int = 80) -> bytes:
    """Convert [3, H, W] float tensor ([0,1]) to JPEG bytes."""
    arr = (frame_chw.permute(1, 2, 0).clamp(0, 1) * 255.0).byte().cpu().numpy()
    arr_bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    _, buf = cv2.imencode(".jpg", arr_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return buf.tobytes()


def _render_tensor_to_jpeg(frame_chw: torch.Tensor, quality: int = 80) -> bytes:
    """Convert render frame [3, H, W] float tensor ([-1,1]) to JPEG bytes."""
    arr = ((frame_chw.permute(1, 2, 0).cpu().float().clamp(-1, 1) * 0.5 + 0.5) * 255.0).byte().numpy()
    arr_bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    _, buf = cv2.imencode(".jpg", arr_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return buf.tobytes()


# =============================================================================
# Background JPEG encoding thread
# =============================================================================

def _encode_and_enqueue_worker(
    encode_queue: queue.Queue,
    session: "Session",
    save_dir: Optional[str],
    show_render: bool,
    scene_name: str,
    stop_event: threading.Event,
):
    """Background JPEG encoding thread: encodes CPU tensor frames to JPEG and enqueues them."""
    while not stop_event.is_set():
        try:
            item = encode_queue.get(timeout=0.5)
        except queue.Empty:
            continue
        if item is None:
            break

        gen_pixel_cpu, render_pixel_cpu, step_idx, frame_counter, tm = item
        t0_send = _now_ms()
        is_explore = gen_pixel_cpu is None  # Explore mode: no DiT output

        # ── Save frames to disk (optional: --save_frames) ──────────────────────
        if save_dir and gen_pixel_cpu is not None:
            for t_idx in range(gen_pixel_cpu.shape[1]):
                gen_np = (gen_pixel_cpu[0, t_idx].permute(1, 2, 0).float().numpy()
                          .clip(0, 1) * 255).astype(np.uint8)
                cv2.imwrite(
                    os.path.join(save_dir, f"gen_{frame_counter + t_idx:05d}.jpg"),
                    cv2.cvtColor(gen_np, cv2.COLOR_RGB2BGR),
                )
        if save_dir and render_pixel_cpu is not None:
            for t_idx in range(render_pixel_cpu.shape[1]):
                rnd_np = ((render_pixel_cpu[0, t_idx].permute(1, 2, 0).float().numpy()
                           .clip(-1, 1) * 0.5 + 0.5) * 255).astype(np.uint8)
                cv2.imwrite(
                    os.path.join(save_dir, f"render_{frame_counter + t_idx:05d}.jpg"),
                    cv2.cvtColor(rnd_np, cv2.COLOR_RGB2BGR),
                )

        # ── Recording: save frames to recording_dir ──────────────────────────
        if session.recording and render_pixel_cpu is not None:
            for t_idx in range(render_pixel_cpu.shape[1]):
                rnd_np = ((render_pixel_cpu[0, t_idx].permute(1, 2, 0).float().numpy()
                           .clip(-1, 1) * 0.5 + 0.5) * 255).astype(np.uint8)
                rnd_path = os.path.join(
                    session.recording_dir,
                    f"render_{len(session.recording_render_frames):05d}.jpg")
                cv2.imwrite(rnd_path, cv2.cvtColor(rnd_np, cv2.COLOR_RGB2BGR))
                session.recording_render_frames.append(rnd_path)
            if gen_pixel_cpu is not None:
                for t_idx in range(gen_pixel_cpu.shape[1]):
                    gen_np = (gen_pixel_cpu[0, t_idx].permute(1, 2, 0).float().numpy()
                              .clip(0, 1) * 255).astype(np.uint8)
                    gen_path = os.path.join(
                        session.recording_dir,
                        f"gen_{len(session.recording_gen_frames):05d}.jpg")
                    cv2.imwrite(gen_path, cv2.cvtColor(gen_np, cv2.COLOR_RGB2BGR))
                    session.recording_gen_frames.append(gen_path)

        # Enqueue render frames first
        if show_render and render_pixel_cpu is not None:
            for t_idx in range(render_pixel_cpu.shape[1]):
                render_jpeg = _render_tensor_to_jpeg(render_pixel_cpu[0, t_idx])
                try:
                    session.render_queue.put_nowait(render_jpeg)
                except queue.Full:
                    try:
                        session.render_queue.get_nowait()
                    except queue.Empty:
                        pass
                    session.render_queue.put_nowait(render_jpeg)

        # Encode generated frames to JPEG
        # Explore mode (gen_pixel is None) or step_idx==1: show render frame
        if is_explore and render_pixel_cpu is not None:
            display_pixel = render_pixel_cpu.clamp(-1, 1) * 0.5 + 0.5
        elif step_idx == 1 and render_pixel_cpu is not None:
            display_pixel = render_pixel_cpu.clamp(-1, 1) * 0.5 + 0.5
        else:
            display_pixel = gen_pixel_cpu

        for t_idx in range(display_pixel.shape[1]):
            jpeg_bytes = _tensor_frame_to_jpeg(display_pixel[0, t_idx])
            try:
                session.frame_queue.put_nowait(jpeg_bytes)
            except queue.Full:
                try:
                    session.frame_queue.get_nowait()
                except queue.Empty:
                    pass
                session.frame_queue.put_nowait(jpeg_bytes)

        tm.send_ms = _now_ms() - t0_send
        tm.total_ms = tm.total_ms + tm.send_ms
        print(f"[encode:{scene_name}] {tm.log_line()}")


# =============================================================================
# Generation loop (background thread)
# =============================================================================

def _generation_loop(
    session: "Session",
    pipeline: CausalInferencePipeline,
    device: torch.device,
    dtype: torch.dtype,
    config,
):
    """
    Background generation loop.

    Mode state machine:
      EXPLORE (default after scene change / reset)
        → user clicks 'Start Denoising' → DENOISE
      DENOISE
        → recenter / scene change → EXPLORE

    In EXPLORE mode: render-only (no DiT/VAE), both panels show render.
    In DENOISE mode: full pipeline (render + VAE + DiT + VAE decode).
    """
    num_frames_for_cache = getattr(config, "num_frames", 21)
    save_dir = getattr(_server_args, 'save_frames', None)
    show_render = getattr(_server_args, 'show_render', False)

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        print(f"[gen-loop:{session.scene_name}] Frame save directory: {save_dir}")
    if show_render:
        print(f"[gen-loop:{session.scene_name}] Render frame push enabled")

    frame_counter = 0

    # ── EXPLORE MODE: skip pipeline init, start rendering immediately ──
    session.mode = "explore"
    session.outgoing_text.put({"type": "mode_change", "mode": "explore"})
    print(f"[gen-loop:{session.scene_name}] EXPLORE mode active (render-only), waiting for control input")

    # Start background JPEG encoding thread
    encode_queue = queue.Queue(maxsize=4)
    encode_thread = threading.Thread(
        target=_encode_and_enqueue_worker,
        args=(encode_queue, session, save_dir, show_render,
              session.scene_name, session.stop_event),
        daemon=True,
    )
    encode_thread.start()

    step_idx = 0
    initial_w2c = session.initial_w2c
    # starting_w2c is initialised to initial_w2c if not set by user
    if session.starting_w2c is None:
        session.starting_w2c = initial_w2c.clone()

    while not session.stop_event.is_set():
        # ── Check for denoise transition request ──────────────────────────
        if session.denoise_requested and session.mode == "explore":
            session.denoise_requested = False
            print(f"[gen-loop:{session.scene_name}] Transitioning to DENOISE mode, initializing pipeline ...")
            with _pipeline_lock, torch.no_grad():
                pipeline.initialize_streaming(
                    batch_size=1,
                    device=device,
                    dtype=dtype,
                    conditional_dict=session.cond_dict,
                    num_frames_for_cache=num_frames_for_cache,
                )
            session.mode = "denoise"
            session.outgoing_text.put({"type": "mode_change", "mode": "denoise"})
            step_idx = 0
            frame_counter = 0
            print(f"[gen-loop:{session.scene_name}] DENOISE mode active")

        # ── Wait for control input ────────────────────────────────────────
        # In EXPLORE mode, skip wait on first iteration (step_idx==0) to render initial frame immediately
        if (session.mode == "denoise" and step_idx > 0) or (session.mode == "explore" and step_idx > 0):
            if not session.control.has_input and not session.denoise_requested:
                got_input = session.control.wait_for_input(timeout=0.5)
                if session.stop_event.is_set():
                    break
                if not got_input:
                    continue

        if session.stop_event.is_set():
            break

        # ── Determine frames per step ─────────────────────────────────────
        if session.mode == "explore":
            frames_per_step = 4
        else:
            frames_per_step = 1 if step_idx == 0 else 4

        per_move, per_look, should_recenter = session.control.consume_for_frames(
            frames_per_step)

        # ── Handle recenter ────────────────────────────────────────────────
        if should_recenter:
            while not encode_queue.empty():
                try:
                    encode_queue.get_nowait()
                except queue.Empty:
                    break

            # Auto-stop recording if active
            if session.recording:
                session.recording = False
                print(f"[gen-loop:{session.scene_name}] Auto-stopping recording due to recenter")
                # Create videos in background thread, send URLs via outgoing_text
                def _bg_create_videos(s=session):
                    try:
                        result = _create_recording_videos(s)
                        s.outgoing_text.put({
                            "type": "recording_complete",
                            "downloads": result,
                        })
                    except Exception as e:
                        print(f"[gen-loop] Background video creation failed: {e}")
                threading.Thread(target=_bg_create_videos, daemon=True).start()

            # Reload Gaussian renderer with current scale factor
            gs_scale_factor = getattr(_server_args, 'gaussian_scale', 1.0)
            scene = _scene_map.get(session.scene_name)
            if scene and scene.gs_data and session.renderer:
                session.renderer.scales = scene.gs_data["scales"] * gs_scale_factor
                print(f"[gen-loop:{session.scene_name}] Applied gaussian scale factor: {gs_scale_factor}")

            if session.mode == "denoise":
                # DENOISE → EXPLORE: reset camera + pipeline, switch to explore
                print(f"[gen-loop:{session.scene_name}] Recenter in DENOISE: resetting to EXPLORE mode")
                session.camera.reset(session.starting_w2c.to(session.camera.device))
                step_idx = 0
                frame_counter = 0
                session.mode = "explore"
                session.outgoing_text.put({"type": "mode_change", "mode": "explore"})
            else:
                # EXPLORE recenter: reset camera to starting_w2c only
                print(f"[gen-loop:{session.scene_name}] Recenter in EXPLORE: resetting camera")
                session.camera.reset(session.starting_w2c.to(session.camera.device))
                step_idx = 0
                frame_counter = 0
            continue

        if session.stop_event.is_set():
            break

        step_idx += 1

        try:
            if session.mode == "explore":
                # ── EXPLORE: render only ───────────────────────────────────
                gen_pixel, render_pixel, tm = _run_explore_step(
                    camera=session.camera,
                    renderer=session.renderer,
                    device=device,
                    per_frame_move=per_move,
                    per_frame_look=per_look,
                )
                gen_pixel_cpu = None
                render_pixel_cpu = render_pixel.cpu() if render_pixel is not None else None
                session.last_step_ms = tm.total_ms

                frames_out = render_pixel.shape[1] if render_pixel is not None else 0
            else:
                # ── DENOISE: full pipeline ────────────────────────────────
                with _pipeline_lock:
                    gen_pixel, render_pixel, tm = _run_pipeline_step(
                        step_idx=step_idx,
                        camera=session.camera,
                        renderer=session.renderer,
                        pipeline=pipeline,
                        device=device,
                        dtype=dtype,
                        per_frame_move=per_move,
                        per_frame_look=per_look,
                    )

                if gen_pixel is None:
                    continue

                gen_pixel_cpu = gen_pixel.cpu()
                render_pixel_cpu = render_pixel.cpu() if render_pixel is not None else None
                session.last_step_ms = tm.total_ms
                frames_out = gen_pixel.shape[1]

            try:
                encode_queue.put_nowait(
                    (gen_pixel_cpu, render_pixel_cpu, step_idx, frame_counter, tm))
            except queue.Full:
                try:
                    encode_queue.get_nowait()
                except queue.Empty:
                    pass
                encode_queue.put_nowait(
                    (gen_pixel_cpu, render_pixel_cpu, step_idx, frame_counter, tm))
            frame_counter += frames_out

        except Exception as e:
            print(f"[gen-loop:{session.scene_name}] Pipeline execution error: {e}")
            traceback.print_exc()
            time.sleep(0.1)

    encode_queue.put(None)
    encode_thread.join(timeout=3.0)
    print(f"[gen-loop:{session.scene_name}] Generation loop stopped")


# =============================================================================
# Session management
# =============================================================================

@dataclass
class Session:
    """Session state for one WebSocket connection."""
    connection_id: str
    scene_name: str
    camera: CameraController
    renderer: Optional[GaussianRenderer]
    initial_w2c: torch.Tensor                 # [4, 4] CPU
    cond_dict: dict
    control: ControlAccumulator = field(default_factory=ControlAccumulator)
    frame_queue: queue.Queue = field(default_factory=lambda: queue.Queue(maxsize=64))
    render_queue: queue.Queue = field(default_factory=lambda: queue.Queue(maxsize=64))
    stop_event: threading.Event = field(default_factory=threading.Event)
    gen_thread: Optional[threading.Thread] = None
    start_time: float = field(default_factory=time.time)
    last_step_ms: float = 1000.0
    # ── Explore / Denoise mode ──
    mode: str = "explore"                          # "explore" or "denoise"
    denoise_requested: bool = False                # Flag to trigger denoise init in gen loop
    starting_w2c: Optional[torch.Tensor] = None    # User-set starting viewpoint (CPU)
    # ── Outgoing text messages (from gen loop to frontend) ──
    outgoing_text: queue.Queue = field(default_factory=queue.Queue)
    # ── Recording ──
    recording: bool = False
    recording_dir: Optional[str] = None
    recording_gen_frames: list = field(default_factory=list)
    recording_render_frames: list = field(default_factory=list)
    recording_id: Optional[str] = None


# =============================================================================
# Global state
# =============================================================================

_pipeline: Optional[CausalInferencePipeline] = None
_device: torch.device = torch.device("cuda:0")
_dtype = torch.bfloat16
_config = None
_scenes: List[SceneData] = []
_scene_map: Dict[str, SceneData] = {}
_active_session: Optional[Session] = None
_session_lock = threading.Lock()
_pipeline_lock = threading.Lock()
# Recording storage: {recording_id: {"render": path, "gen": path, "combined": path, "dir": dir}}
_RECORDINGS: Dict[str, dict] = {}


# =============================================================================
# FastAPI App
# =============================================================================

app = FastAPI(title="Self-Forcing World Engine (Gaussian Splatting)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── HTTP API ────────────────────────────────────────────────────────────────

@app.get("/api/config")
async def api_config():
    show_render = getattr(_server_args, 'show_render', False) if _server_args else False
    gaussian_scale = getattr(_server_args, 'gaussian_scale', 1.0) if _server_args else 1.0
    return {"transport": "tcp", "show_render": show_render, "gaussian_scale": gaussian_scale}


@app.post("/api/gaussian_scale")
async def set_gaussian_scale(body: dict = None):
    """Set gaussian scale factor. Takes effect on next scene (re)connection or recenter (Reset)."""
    import json
    from starlette.requests import Request
    if body is None:
        body = {}
    scale = body.get("scale", 1.0)
    if _server_args:
        _server_args.gaussian_scale = float(scale)
    print(f"[api] gaussian_scale set to {scale}")
    return {"gaussian_scale": float(scale)}


@app.get("/api/scenes")
async def api_scenes():
    scene_list = []
    for s in _scenes:
        scene_list.append({
            "name": s.name,
            "display_name": s.display_name,
            "thumbnail": f"/thumbnails/{s.name}.jpg",
            "description": s.prompt[:60] + "..." if len(s.prompt) > 60 else s.prompt,
        })
    return {"scenes": scene_list}


@app.post("/api/add_scene_folder")
async def add_scene_folder(body: dict = None):
    """
    Add a scene from a local folder path (in-memory only, NOT persisted to YAML).

    Request body: { "folder_path": "C:\\Users\\...\\DancingHall" }

    The backend scans the folder for a .ply/.pt/.pth file, optional thumbnail
    image and prompt.txt, then registers a temporary SceneData in memory.
    """
    if body is None:
        body = {}
    folder_path = body.get("folder_path", "").strip()
    if not folder_path:
        return JSONResponse(
            status_code=400,
            content={"error": "folder_path is required"},
        )

    try:
        # Normalise path (Windows → WSL if needed)
        norm_path = _normalize_folder_path(folder_path)

        # Scan folder for scene files
        scan_result = _scan_scene_folder(norm_path)

        # Derive scene name from folder name
        folder_name = os.path.basename(norm_path.rstrip('/'))
        if not folder_name:
            folder_name = os.path.basename(norm_path)
        # Sanitise for use as URL/API key
        scene_name = folder_name.lower().replace(' ', '_').replace('-', '_')
        # Ensure uniqueness
        base_name = scene_name
        suffix = 2
        while scene_name in _scene_map:
            scene_name = f"{base_name}_{suffix}"
            suffix += 1

        # Create SceneData (in-memory only)
        scene = SceneData(
            name=scene_name,
            display_name=folder_name,
            gs_file=scan_result['gs_file'],
            prompt=scan_result['prompt'],
            input_image_path=scan_result['input_image'],
        )

        # Pre-encode text prompt (handles text_encoder on/off-loading)
        _encode_single_scene_prompt(scene)

        # Register in global scene list and map
        _scenes.append(scene)
        _scene_map[scene_name] = scene

        print(f"[api] Dynamically added scene: '{scene_name}' "
              f"(display: '{folder_name}') "
              f"gs_file={scan_result['gs_file']}")

        return {
            "name": scene.name,
            "display_name": scene.display_name,
            "thumbnail": f"/thumbnails/{scene.name}.jpg",
            "description": scene.prompt[:60] + "..." if len(scene.prompt) > 60 else scene.prompt,
        }

    except FileNotFoundError as e:
        return JSONResponse(status_code=404, content={"error": str(e)})
    except Exception as e:
        print(f"[api] add_scene_folder error: {e}")
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/api/upload_scene")
async def upload_scene(
    file: UploadFile = File(...),
    thumbnail: Optional[UploadFile] = File(None),
    prompt: str = Form(""),
):
    """
    Upload a .ply/.pt/.pth file and create a temporary scene (in-memory, NOT persisted).

    Multipart form fields:
      - file:      Gaussian scene file (.ply / .pt / .pth), required
      - thumbnail: Preview image (optional)
      - prompt:    Text prompt for video generation (optional)
    """
    # Validate file extension
    original_name = file.filename or "uploaded_scene"
    ext = os.path.splitext(original_name)[1].lower()
    if ext not in ('.ply', '.pt', '.pth'):
        return JSONResponse(
            status_code=400,
            content={"error": f"Unsupported file type: {ext}. Supported: .ply, .pt, .pth"},
        )

    # Derive scene name from filename
    base_name = os.path.splitext(original_name)[0]
    scene_name = base_name.lower().replace(' ', '_').replace('-', '_')
    base_scene = scene_name
    suffix = 2
    while scene_name in _scene_map:
        scene_name = f"{base_scene}_{suffix}"
        suffix += 1

    # Save uploaded Gaussian file in 1 MB chunks
    file_path = os.path.join(_UPLOAD_DIR, f"{scene_name}{ext}")
    print(f"[api] Saving uploaded file: {original_name} → {file_path}")
    with open(file_path, "wb") as f:
        while True:
            chunk = await file.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)
    file_size = os.path.getsize(file_path)
    print(f"[api] File saved: {original_name} ({file_size / 1024 / 1024:.1f} MB)")

    # Save thumbnail if provided
    thumbnail_path = ""
    if thumbnail and thumbnail.filename:
        thumb_ext = os.path.splitext(thumbnail.filename)[1].lower()
        if thumb_ext in ('.png', '.jpg', '.jpeg', '.webp'):
            thumbnail_path = os.path.join(_UPLOAD_DIR, f"{scene_name}_thumb{thumb_ext}")
            with open(thumbnail_path, "wb") as tf:
                while True:
                    chunk = await thumbnail.read(1024 * 1024)
                    if not chunk:
                        break
                    tf.write(chunk)

    # Create SceneData (in-memory only)
    scene = SceneData(
        name=scene_name,
        display_name=base_name,
        gs_file=file_path,
        prompt=prompt.strip(),
        input_image_path=thumbnail_path,
    )

    # Pre-encode text prompt (handles text_encoder on/off-loading)
    _encode_single_scene_prompt(scene)

    # Register in global scene list and map
    _scenes.append(scene)
    _scene_map[scene_name] = scene

    print(f"[api] Uploaded scene registered: '{scene_name}' "
          f"(display: '{base_name}', file: {file_path})")

    return {
        "name": scene.name,
        "display_name": scene.display_name,
        "thumbnail": f"/thumbnails/{scene.name}.jpg",
        "description": scene.prompt[:60] + "..." if len(scene.prompt) > 60 else scene.prompt,
    }


# ── Scene generation pipeline (pano → depth → 3DGS) ──────────────────────
_GENERATED_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "generated")
os.makedirs(_GENERATED_DIR, exist_ok=True)


@app.post("/api/generate_pipeline")
async def generate_pipeline(
    image: UploadFile = File(...),
    prompt: str = Form(...),
    back_prompt: str = Form(""),
    scene_name: str = Form(None),
):
    """Generate a complete scene from a perspective image.

    Pipeline:
        1. Panorama generation (FluxFill + AutoLevel)
        2. Depth estimation (DA3)
        3. 3D Gaussian Splatting generation (Sharp)
        4. Scene registration

    Returns an SSE stream with per-step progress updates.
    """
    # Read image data before starting the stream (async → sync boundary)
    image_bytes = await image.read()

    def _sse(data: dict) -> str:
        return f"data: {json.dumps(data)}\n\n"

    def _generate_stream():
        try:
            nonlocal scene_name
            # Derive scene name if not provided
            if not scene_name:
                words = prompt.split()[:3]
                scene_name = "_".join(w.strip(".,!?") for w in words).lower()
            scene_name = scene_name.replace(" ", "_").replace("-", "_").lower()

            # Ensure uniqueness
            base_name = scene_name
            suffix = 2
            while scene_name in _scene_map:
                scene_name = f"{base_name}_{suffix}"
                suffix += 1

            out_dir = os.path.join(_GENERATED_DIR, scene_name)
            os.makedirs(out_dir, exist_ok=True)

            # Save input image
            input_path = os.path.join(out_dir, "input.png")
            with open(input_path, "wb") as f:
                f.write(image_bytes)

            yield _sse({
                "step": "init", "status": "done",
                "scene_name": scene_name,
                "output_dir": out_dir,
                "input_path": input_path,
            })

            # Move streaming pipeline to CPU to free VRAM for FLUX/DA3/Sharp
            _pipeline_on_cpu = False
            _gs_data_on_cpu = False
            if _pipeline is not None:
                print("[pipeline] Moving streaming pipeline to CPU to free VRAM...")
                with _pipeline_lock:
                    _pipeline.generator.to("cpu")
                    _pipeline.vae.to("cpu")
                torch.cuda.empty_cache()
                _pipeline_on_cpu = True
                print(f"[pipeline] Streaming pipeline on CPU. VRAM: {torch.cuda.memory_allocated()/1e9:.1f}GB")

            # Move all scene Gaussian data to CPU to free remaining VRAM
            _gs_moved_scenes = []
            for s in _scenes:
                if s.gs_data is not None and not getattr(s, '_gs_on_cpu', False):
                    for k, v in s.gs_data.items():
                        if isinstance(v, torch.Tensor) and v.is_cuda:
                            s.gs_data[k] = v.cpu()
                    _gs_moved_scenes.append(s)
                    s._gs_on_cpu = True
            if _gs_moved_scenes:
                _gs_data_on_cpu = True
                torch.cuda.empty_cache()
                print(f"[pipeline] Moved {len(_gs_moved_scenes)} scene(s) Gaussian data to CPU. "
                      f"VRAM: {torch.cuda.memory_allocated()/1e9:.1f}GB")

            # Step 1: Panorama generation
            yield _sse({"step": "panorama", "status": "processing",
                        "message": "Generating 360° panorama (FluxFill + AutoLevel)..."})

            from pano_gen.generate import generate as generate_panorama
            from pano_gen.prompt_builder import derive_video_prompt

            pano_ckpt_dir = getattr(_server_args, 'pano_ckpt_dir', 'checkpoints/gimbal360')
            flux_model = getattr(_server_args, 'flux_model_path',
                                  'checkpoints/FLUX.1-Fill-dev')

            pano_image = generate_panorama(
                image=input_path,
                prompt=prompt,
                back_prompt=back_prompt,
                ckpt_dir=pano_ckpt_dir,
                basemodel_name_or_path=flux_model,
            )
            pano_path = os.path.join(out_dir, "panorama.png")
            pano_image.save(pano_path)

            yield _sse({"step": "panorama", "status": "done", "path": pano_path})

            # Free FLUX pipeline GPU memory before DA3 (FLUX ~20GB, DA3 needs ~6GB)
            import pano_gen.generate as _pano_gen_mod
            if _pano_gen_mod._pipeline_cache:
                print("[pipeline] Releasing FLUX pipeline from GPU memory...")
                _pano_gen_mod._pipeline_cache.clear()
                if _pano_gen_mod._erp_converter_cache:
                    for conv in _pano_gen_mod._erp_converter_cache.values():
                        if hasattr(conv, 'model'):
                            del conv.model
                    _pano_gen_mod._erp_converter_cache.clear()
                torch.cuda.empty_cache()
                print(f"[pipeline] GPU memory freed. VRAM: {torch.cuda.memory_allocated()/1e9:.1f}GB")

            # Step 2: Depth estimation (DA3)
            yield _sse({"step": "depth", "status": "processing",
                        "message": "Estimating panoramic depth with DA3..."})

            from gaussian_gen.infer_da3 import infer_panorama_depth

            da3_dir = getattr(_server_args, 'da3_model_dir',
                              'checkpoints/DA3NESTED-GIANT-LARGE-1.1')

            t_da3_start = time.perf_counter()
            print(f"[pipeline] DA3 starting — model_dir={da3_dir}")
            infer_panorama_depth(
                image_path=pano_path,
                output_dir=out_dir,
                model_dir=da3_dir,
            )
            t_da3_end = time.perf_counter()
            print(f"[pipeline] DA3 completed in {t_da3_end - t_da3_start:.1f}s")
            # Free DA3 model from GPU
            import gc
            gc.collect()
            torch.cuda.empty_cache()
            print(f"[pipeline] DA3 GPU memory freed. VRAM: {torch.cuda.memory_allocated()/1e9:.1f}GB")
            depth_path = os.path.join(out_dir, "depth.npy")

            yield _sse({"step": "depth", "status": "done", "path": depth_path})

            # Step 3: Gaussian generation
            yield _sse({"step": "gaussian", "status": "processing",
                        "message": "Generating 3D Gaussians (Sharp)..."})

            from gaussian_gen.infer_pano import (
                build_pano_predictor, load_checkpoint, save_pano_ply,
                load_panorama, load_depth,
            )

            gaussian_ckpt = getattr(_server_args, 'gaussian_ckpt',
                                    'checkpoints/gaussian_predictor_checkpoint.pth')
            gen_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            t_gs_start = time.perf_counter()
            print(f"[pipeline] Gaussian generation starting — ckpt={gaussian_ckpt}")
            gs_model = build_pano_predictor(gen_device, depth_model_path=None, num_layers=1)
            load_checkpoint(gs_model, gaussian_ckpt, gen_device)
            gs_model.eval()
            print(f"[pipeline] Gaussian model loaded in {time.perf_counter() - t_gs_start:.1f}s")

            t_gs_inf_start = time.perf_counter()

            image_np = load_panorama(pano_path, target_height=None, target_width=None)
            img_h, img_w = image_np.shape[:2]
            image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).unsqueeze(0).to(gen_device)

            depth_np = load_depth(Path(depth_path), img_h, img_w, factor=1.0)
            depth_tensor = torch.from_numpy(depth_np).unsqueeze(0).unsqueeze(0).to(gen_device)

            with torch.no_grad():
                predictor_output = gs_model(image_tensor, depth_gt=depth_tensor)
                gaussians = predictor_output.gaussians.to(torch.device("cpu"))
            print(f"[pipeline] Gaussian inference done in {time.perf_counter() - t_gs_inf_start:.1f}s")

            ply_path = os.path.join(out_dir, "scene.ply")
            save_pano_ply(gaussians, (img_h, img_w), ply_path)

            # Free GPU memory
            del gs_model, image_tensor, depth_tensor, predictor_output, gaussians
            torch.cuda.empty_cache()

            yield _sse({"step": "gaussian", "status": "done", "path": ply_path})

            # Step 4: Register scene
            yield _sse({"step": "register", "status": "processing",
                        "message": "Registering scene..."})

            video_prompt = derive_video_prompt(prompt)

            scene = SceneData(
                name=scene_name,
                display_name=scene_name.replace("_", " ").title(),
                gs_file=ply_path,
                prompt=video_prompt,
                input_image_path=pano_path,
            )
            _encode_single_scene_prompt(scene)
            _scenes.append(scene)
            _scene_map[scene_name] = scene

            print(f"[api] Generated scene registered: '{scene_name}' "
                  f"(ply={ply_path}, prompt={video_prompt[:60]}...)")

            # Restore streaming pipeline to GPU for video streaming
            if _pipeline is not None and _pipeline_on_cpu:
                print("[pipeline] Restoring streaming pipeline to GPU...")
                with _pipeline_lock:
                    _pipeline.generator.to(_device)
                    _pipeline.vae.to(_device)
                torch.cuda.empty_cache()
                _pipeline_on_cpu = False
                print(f"[pipeline] Streaming pipeline on GPU. VRAM: {torch.cuda.memory_allocated()/1e9:.1f}GB")

            # Restore scene Gaussian data to GPU
            if _gs_data_on_cpu:
                for s in _gs_moved_scenes:
                    if s.gs_data is not None:
                        for k, v in s.gs_data.items():
                            if isinstance(v, torch.Tensor) and not v.is_cuda:
                                s.gs_data[k] = v.to(_device)
                    s._gs_on_cpu = False
                torch.cuda.empty_cache()
                _gs_data_on_cpu = False
                print(f"[pipeline] Scene Gaussian data restored to GPU. "
                      f"VRAM: {torch.cuda.memory_allocated()/1e9:.1f}GB")

            yield _sse({
                "step": "done", "status": "done",
                "scene_name": scene_name,
                "video_prompt": video_prompt,
                "ply_path": ply_path,
                "panorama_path": pano_path,
                "depth_path": depth_path,
            })

        except Exception as e:
            traceback.print_exc()
            yield _sse({"step": "error", "status": "error", "message": str(e)})
        finally:
            # Safety net: always restore pipeline to GPU if it was moved to CPU
            if _pipeline is not None and _pipeline_on_cpu:
                print("[pipeline] Safety restore: pipeline still on CPU, moving back to GPU...")
                try:
                    with _pipeline_lock:
                        _pipeline.generator.to(_device)
                        _pipeline.vae.to(_device)
                    torch.cuda.empty_cache()
                    print(f"[pipeline] Streaming pipeline restored to GPU. VRAM: {torch.cuda.memory_allocated()/1e9:.1f}GB")
                except Exception as restore_err:
                    print(f"[pipeline] ERROR restoring pipeline to GPU: {restore_err}")
            # Safety net: restore scene Gaussian data to GPU
            if _gs_data_on_cpu:
                print("[pipeline] Safety restore: Gaussian data still on CPU, moving back to GPU...")
                try:
                    for s in _gs_moved_scenes:
                        if s.gs_data is not None:
                            for k, v in s.gs_data.items():
                                if isinstance(v, torch.Tensor) and not v.is_cuda:
                                    s.gs_data[k] = v.to(_device)
                        s._gs_on_cpu = False
                    torch.cuda.empty_cache()
                    print(f"[pipeline] Gaussian data restored to GPU. VRAM: {torch.cuda.memory_allocated()/1e9:.1f}GB")
                except Exception as restore_err:
                    print(f"[pipeline] ERROR restoring Gaussian data to GPU: {restore_err}")

    return StreamingResponse(
        _generate_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/thumbnails/{filename}")
async def get_thumbnail(filename: str):
    scene_name = filename.replace(".jpg", "").replace(".png", "")
    scene = _scene_map.get(scene_name)

    if scene is None:
        return Response(status_code=404, content=b"Scene not found")

    # Prefer input_image (preview thumbnail)
    if scene.input_image_path:
        img_path = _resolve_image_path(scene.input_image_path)
        if os.path.exists(img_path):
            try:
                img = Image.open(img_path).convert("RGB")
                img.thumbnail((256, 256), Image.LANCZOS)
                buf = io.BytesIO()
                img.save(buf, format="JPEG", quality=85)
                return Response(content=buf.getvalue(), media_type="image/jpeg")
            except Exception:
                pass

    # No preview image, return solid color placeholder
    placeholder = np.zeros((128, 128, 3), dtype=np.uint8) + 64
    _, buf = cv2.imencode(".jpg", placeholder)
    return Response(content=buf.tobytes(), media_type="image/jpeg")


# ── Static files ────────────────────────────────────────────────────────────────
_DIST_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dist")
if os.path.isdir(_DIST_DIR):
    from fastapi.staticfiles import StaticFiles
    app.mount("/moverse", StaticFiles(directory=_DIST_DIR, html=True), name="frontend")
    print(f"[init] Frontend static files mounted: {_DIST_DIR}")


# ── WebSocket: TCP frame stream ────────────────────────────────────────────

@app.websocket("/ws/frames")
async def websocket_frames(websocket: WebSocket):
    global _active_session
    connection_id = f"frames-{id(websocket)}"
    await websocket.accept()
    print(f"[ws:{connection_id}] WebSocket connected")

    session: Optional[Session] = None
    frame_sender_task: Optional[asyncio.Task] = None
    message_sender_task: Optional[asyncio.Task] = None

    try:
        while True:
            data = await websocket.receive()

            msg_asgi_type = data.get("type", "")
            if msg_asgi_type == "websocket.disconnect":
                print(f"[ws:{connection_id}] Received disconnect message")
                break

            if "text" not in data:
                continue

            try:
                message = __import__("json").loads(data["text"])
            except Exception:
                continue

            msg_type = message.get("type", "")

            if msg_type == "ping":
                await websocket.send_json({"type": "pong"})

            elif msg_type == "start":
                scene_name = message.get("scene", "")
                print(f"[ws:{connection_id}] Request to start scene: {scene_name}")

                if session is not None:
                    # Auto-stop recording before stopping session
                    if session.recording:
                        session.recording = False
                        try:
                            rec_result = await asyncio.get_event_loop().run_in_executor(
                                None, _create_recording_videos, session)
                            await websocket.send_json({
                                "type": "recording_complete",
                                "downloads": rec_result,
                            })
                        except Exception as e:
                            print(f"[ws] Auto-recording finalization failed: {e}")
                    session.stop_event.set()
                    if session.gen_thread and session.gen_thread.is_alive():
                        session.gen_thread.join(timeout=10.0)
                if frame_sender_task:
                    frame_sender_task.cancel()
                    try:
                        await frame_sender_task
                    except asyncio.CancelledError:
                        pass
                if message_sender_task:
                    message_sender_task.cancel()
                    try:
                        await message_sender_task
                    except asyncio.CancelledError:
                        pass

                try:
                    await websocket.send_json({
                        "type": "queue_status", "status": "active",
                    })
                except Exception:
                    break

                try:
                    session = await asyncio.get_event_loop().run_in_executor(
                        None, _init_scene_session, connection_id, scene_name
                    )
                except Exception as e:
                    print(f"[ws:{connection_id}] Scene initialization failed: {e}")
                    try:
                        await websocket.send_json({
                            "type": "error",
                            "error": "scene_init_failed",
                            "message": str(e),
                        })
                    except Exception:
                        break
                    continue

                with _session_lock:
                    if _active_session is not None:
                        _active_session.stop_event.set()
                    _active_session = session

                # Initial frame: let the generation loop render the first Gaussian frame
                # (previously sent panorama image which was confusing for new scenes)
                # The gen loop skips input wait on step_idx==0 in EXPLORE mode, so first render
                # arrives within ~100ms.

                session.gen_thread = threading.Thread(
                    target=_generation_loop,
                    args=(session, _pipeline, _device, _dtype, _config),
                    daemon=True,
                )
                session.gen_thread.start()

                frame_sender_task = asyncio.create_task(
                    _frame_sender(websocket, session))
                message_sender_task = asyncio.create_task(
                    _message_sender(websocket, session))

                try:
                    await websocket.send_json({"type": "start"})
                except Exception:
                    pass

                print(f"[ws:{connection_id}] Scene {scene_name} generation started")

            elif msg_type == "control":
                if session is not None:
                    session.control.add(message)

            elif msg_type == "recenter":
                if session is not None:
                    session.control.add({"recenter": True})

            elif msg_type == "start_denoising":
                if session is not None and session.mode == "explore":
                    print(f"[ws:{connection_id}] Start denoising requested")
                    session.denoise_requested = True
                    session.control._input_event.set()  # Wake the generation loop

            elif msg_type == "set_starting_point":
                if session is not None and session.mode == "explore":
                    session.starting_w2c = session.camera.w2c.clone().cpu()
                    print(f"[ws:{connection_id}] Starting point set")
                    await websocket.send_json({"type": "starting_point_set"})

            elif msg_type == "start_recording":
                if session is not None and not session.recording:
                    import uuid
                    session.recording = True
                    session.recording_id = str(uuid.uuid4())[:8]
                    session.recording_dir = tempfile.mkdtemp(prefix=f"rec_{session.recording_id}_")
                    session.recording_gen_frames = []
                    session.recording_render_frames = []
                    print(f"[ws:{connection_id}] Recording started (id={session.recording_id})")
                    await websocket.send_json({"type": "recording_started"})

            elif msg_type == "stop_recording":
                if session is not None and session.recording:
                    session.recording = False
                    print(f"[ws:{connection_id}] Recording stopped, creating videos ...")
                    await websocket.send_json({"type": "recording_processing"})
                    result = await asyncio.get_event_loop().run_in_executor(
                        None, _create_recording_videos, session)
                    await websocket.send_json({
                        "type": "recording_complete",
                        "downloads": result,
                    })
                    session.recording_gen_frames = []
                    session.recording_render_frames = []

    except WebSocketDisconnect:
        print(f"[ws:{connection_id}] WebSocket disconnected")
    except Exception as e:
        print(f"[ws:{connection_id}] Error: {type(e).__name__}: {e}")
    finally:
        if session is not None:
            # Auto-finalize recording if active
            if session.recording:
                session.recording = False
                try:
                    _create_recording_videos(session)
                except Exception:
                    pass
            session.stop_event.set()
            if session.gen_thread and session.gen_thread.is_alive():
                session.gen_thread.join(timeout=3.0)
        if frame_sender_task:
            frame_sender_task.cancel()
        if message_sender_task:
            message_sender_task.cancel()
        with _session_lock:
            if _active_session is session:
                _active_session = None
        print(f"[ws:{connection_id}] Cleanup complete")


async def _frame_sender(websocket: WebSocket, session: Session):
    """Async coroutine: reads JPEG frames from queue and sends to WebSocket."""
    loop = asyncio.get_event_loop()
    show_render = getattr(_server_args, 'show_render', False) if _server_args else False
    last_gen_send_time = 0.0

    def _get_one_gen_frame():
        try:
            return session.frame_queue.get(timeout=0.3)
        except queue.Empty:
            return None

    def _get_one_render_frame():
        try:
            return session.render_queue.get_nowait()
        except queue.Empty:
            return None

    while not session.stop_event.is_set():
        try:
            gen_jpeg = await loop.run_in_executor(None, _get_one_gen_frame)
            if gen_jpeg is None:
                continue

            frames_per_step = 4
            target_interval = (session.last_step_ms / frames_per_step) / 1000.0
            target_interval = max(min(target_interval * 0.85, 0.50), 0.04)

            now = time.time()
            wait_needed = target_interval - (now - last_gen_send_time)
            if wait_needed > 0.01:
                await asyncio.sleep(wait_needed)

            if show_render:
                await websocket.send_bytes(b'G' + gen_jpeg)
            else:
                await websocket.send_bytes(gen_jpeg)
            last_gen_send_time = time.time()

            if show_render:
                render_jpeg = await loop.run_in_executor(None, _get_one_render_frame)
                if render_jpeg is not None:
                    await websocket.send_bytes(b'R' + render_jpeg)

        except Exception:
            break


async def _message_sender(websocket: WebSocket, session: Session):
    """Async coroutine: reads text messages from session.outgoing_text and sends to WebSocket.

    Used by the generation loop (running in a thread) to notify the frontend of
    mode changes (explore ↔ denoise) and other events.
    """
    loop = asyncio.get_event_loop()

    def _get_message():
        try:
            return session.outgoing_text.get(timeout=0.3)
        except queue.Empty:
            return None

    while not session.stop_event.is_set():
        try:
            msg = await loop.run_in_executor(None, _get_message)
            if msg is None:
                continue
            await websocket.send_json(msg)
        except Exception:
            break


# ── WebSocket: WebRTC signaling (reserved) ──────────────────────────────────────────

@app.websocket("/ws/session")
async def websocket_session(websocket: WebSocket):
    await websocket.accept()
    await websocket.send_json({
        "type": "error",
        "error": "not_implemented",
        "message": "WebRTC mode not supported. Use TCP mode.",
    })
    await websocket.close()


# =============================================================================
# Recording: video creation + download
# =============================================================================

def _create_recording_videos(session: "Session") -> dict:
    """
    Create render.mp4, gen.mp4 (if available), and combined side-by-side.mp4
    from recorded JPEG frames.

    Returns dict of download URLs:
      {"render": "/api/download/<id>/render", "gen": ..., "combined": ...}
    """
    import imageio

    rec_id = session.recording_id or "unknown"
    rec_dir = session.recording_dir
    if not rec_dir or not os.path.isdir(rec_dir):
        print(f"[recording] No recording directory for {rec_id}")
        return {}

    render_frames = sorted(session.recording_render_frames)
    gen_frames = sorted(session.recording_gen_frames)

    if not render_frames and not gen_frames:
        print(f"[recording] No frames recorded for {rec_id}")
        return {}

    # Target FPS: ~24fps (matching typical streaming rate)
    target_fps = 24

    # ── Create render.mp4 ──────────────────────────────────────────────────
    render_mp4 = os.path.join(rec_dir, "render.mp4")
    if render_frames:
        print(f"[recording] Creating render.mp4 ({len(render_frames)} frames) ...")
        imgs = [imageio.imread(p) for p in render_frames]
        imageio.mimsave(render_mp4, imgs, fps=target_fps, codec="libx264",
                        quality=8, macro_block_size=None,
                        ffmpeg_params=['-movflags', 'faststart', '-pix_fmt', 'yuv420p'])
        print(f"[recording] render.mp4 created")

    # ── Create gen.mp4 ─────────────────────────────────────────────────────
    gen_mp4 = os.path.join(rec_dir, "gen.mp4")
    if gen_frames:
        print(f"[recording] Creating gen.mp4 ({len(gen_frames)} frames) ...")
        imgs = [imageio.imread(p) for p in gen_frames]
        imageio.mimsave(gen_mp4, imgs, fps=target_fps, codec="libx264",
                        quality=8, macro_block_size=None,
                        ffmpeg_params=['-movflags', 'faststart', '-pix_fmt', 'yuv420p'])
        print(f"[recording] gen.mp4 created")

    # ── Create combined side-by-side.mp4 ──────────────────────────────────
    # Use numpy + imageio directly instead of combine_videos.py subprocess,
    # which requires system ffmpeg/ffprobe in PATH.
    combined_mp4 = os.path.join(rec_dir, "combined.mp4")
    if render_frames and gen_frames:
        print(f"[recording] Creating combined.mp4 (numpy+imageio) ...")
        try:
            import numpy as np
            render_imgs = [imageio.imread(p) for p in render_frames]
            gen_imgs = [imageio.imread(p) for p in gen_frames]

            # Match frame counts: pad the shorter one by repeating its last frame
            n = max(len(render_imgs), len(gen_imgs))
            while len(render_imgs) < n:
                render_imgs.append(render_imgs[-1])
            while len(gen_imgs) < n:
                gen_imgs.append(gen_imgs[-1])

            # Resize gen frames to match render frame height (if different)
            target_h = render_imgs[0].shape[0]
            target_w_render = render_imgs[0].shape[1]
            target_w_gen = gen_imgs[0].shape[1]
            # Scale gen frames to same height as render frames
            if gen_imgs[0].shape[0] != target_h:
                scale = target_h / gen_imgs[0].shape[0]
                target_w_gen = int(gen_imgs[0].shape[1] * scale)
                import cv2 as _cv2
                gen_imgs = [_cv2.resize(img, (target_w_gen, target_h)) for img in gen_imgs]

            # Concatenate horizontally: [render | gen]
            combined_imgs = []
            for r_img, g_img in zip(render_imgs, gen_imgs):
                combined = np.zeros((target_h, target_w_render + target_w_gen, 3),
                                    dtype=np.uint8)
                combined[:, :target_w_render] = r_img
                combined[:, target_w_render:] = g_img
                combined_imgs.append(combined)

            imageio.mimsave(combined_mp4, combined_imgs, fps=target_fps,
                            codec="libx264", quality=8, macro_block_size=None,
                            ffmpeg_params=['-movflags', 'faststart', '-pix_fmt', 'yuv420p'])
            print(f"[recording] combined.mp4 created ({len(combined_imgs)} frames)")
        except Exception as e:
            print(f"[recording] combined.mp4 creation error: {e}")
            import traceback
            traceback.print_exc()
            combined_mp4 = None
    else:
        combined_mp4 = None

    # ── Register for download ─────────────────────────────────────────────
    result = {}
    if render_frames and os.path.exists(render_mp4):
        result["render"] = f"/api/download/{rec_id}/render"
    if gen_frames and os.path.exists(gen_mp4):
        result["gen"] = f"/api/download/{rec_id}/gen"
    if combined_mp4 and os.path.exists(combined_mp4):
        result["combined"] = f"/api/download/{rec_id}/combined"

    _RECORDINGS[rec_id] = {
        "dir": rec_dir,
        "render": render_mp4 if os.path.exists(render_mp4) else None,
        "gen": gen_mp4 if os.path.exists(gen_mp4) else None,
        "combined": combined_mp4 if combined_mp4 and os.path.exists(combined_mp4) else None,
    }

    print(f"[recording] Videos created for {rec_id}: {list(result.keys())}")
    return result


@app.get("/api/download/{recording_id}/{video_type}")
async def download_recording(recording_id: str, video_type: str):
    """Download a recorded video file."""
    rec = _RECORDINGS.get(recording_id)
    if rec is None:
        return JSONResponse(status_code=404, content={"error": "Recording not found"})

    file_path = rec.get(video_type)
    if not file_path or not os.path.exists(file_path):
        return JSONResponse(status_code=404, content={"error": f"Video '{video_type}' not found"})

    filename = f"{recording_id}_{video_type}.mp4"
    return FileResponse(
        path=file_path,
        media_type="video/mp4",
        filename=filename,
    )


# =============================================================================
# Scene initialization
# =============================================================================

def _init_scene_session(connection_id: str, scene_name: str) -> Session:
    """
    Initialize a Gaussian scene session.

    Includes:
      1. Load Gaussian data (first load, then cached)
      2. Create GaussianRenderer
      3. Create CameraController (initial pose auto-computed from scene)
      4. Encode text condition
    """
    if torch.cuda.is_available():
        torch.cuda.set_device(_device)

    if not HAS_GSPLAT:
        raise RuntimeError(
            "gsplat not available. Please install gsplat: cd gsplat && pip install -e ."
        )

    scene = _scene_map.get(scene_name)
    if scene is None:
        raise ValueError(f"Unknown scene: {scene_name} (available: {list(_scene_map.keys())})")

    target_h, target_w = 480, 832

    # ── 1. Load Gaussian data (cached) ──────────────────────────────────────────
    if not scene.initialized:
        gs_path = _resolve_image_path(scene.gs_file)
        if not os.path.exists(gs_path):
            raise FileNotFoundError(
                f"Scene Gaussian file not found: {gs_path} "
                f"(original path: {scene.gs_file})"
            )

        print(f"[scene-init:{scene_name}] Loading Gaussian data: {gs_path}", flush=True)
        t0 = time.perf_counter()
        with torch.no_grad():
            scene.gs_data = _load_gaussians(gs_path, str(_device))
        dt = (time.perf_counter() - t0) * 1000
        print(f"[scene-init:{scene_name}] Gaussian load complete ({dt:.0f} ms, "
              f"{scene.gs_data['means'].shape[0]:,} Gaussians)")
        scene.initialized = True
    else:
        print(f"[scene-init:{scene_name}] Using cached Gaussian data")

    # ── 2. GaussianRenderer ──────────────────────────────────────────────
    # Apply Gaussian scale factor (does not modify cached data, only affects current renderer)
    gs_scale_factor = getattr(_server_args, 'gaussian_scale', 1.0)
    scene_scales = scene.gs_data["scales"]
    if gs_scale_factor != 1.0:
        scene_scales = scene_scales * gs_scale_factor
        print(f"[scene-init:{scene_name}] Applying Gaussian scale factor: {gs_scale_factor}")

    print(f"[scene-init:{scene_name}] Creating GaussianRenderer "
          f"({scene.gs_data['means'].shape[0]:,} Gaussians, "
          f"{target_w}x{target_h}) ...", flush=True)
    t0 = time.perf_counter()
    try:
        renderer = GaussianRenderer(
            means=scene.gs_data["means"],
            quats=scene.gs_data["quats"],
            scales=scene_scales,
            opacities=scene.gs_data["opacities"],
            colors=scene.gs_data["colors"],
            sh_degree=scene.gs_data["sh_degree"],
            width=target_w,
            height=target_h,
            device=str(_device),
            near_plane=getattr(_server_args, 'near_plane', 0.01),
            far_plane=getattr(_server_args, 'far_plane', 1000.0),
        )
        print(f"[scene-init:{scene_name}] GaussianRenderer created successfully "
              f"({(time.perf_counter()-t0)*1000:.0f} ms)", flush=True)
    except Exception as e:
        print(f"[scene-init:{scene_name}] !!!! Renderer creation failed: {e}")
        traceback.print_exc()
        raise

    # ── 3. CameraController ──────────────────────────────────────────────
    print(f"[scene-init:{scene_name}] Creating CameraController ...", flush=True)

    fov_h_deg = getattr(_server_args, 'render_fov_deg', 70.0)
    fx = (target_w / 2.0) / math.tan(math.radians(fov_h_deg / 2.0))
    fy = fx
    cx = target_w / 2.0
    cy = target_h / 2.0

    intrinsics = torch.tensor([
        [fx,  0.0, cx],
        [0.0, fy,  cy],
        [0.0, 0.0, 1.0],
    ], dtype=torch.float32)

    initial_w2c = _compute_initial_w2c(
        scene.gs_data,
        camera_position=scene.camera_position,
        camera_look_at=scene.camera_look_at,
    )

    camera = CameraController(
        initial_w2c=initial_w2c,
        intrinsics=intrinsics,
        translation_step=_server_args.translation_step,
        rotation_step_deg=_server_args.rotation_step_deg,
        device=str(_device),
    )

    initial_w2c_cpu = initial_w2c.clone()

    # ── 4. Text condition ──────────────────────────────────────────────────────
    print(f"[scene-init:{scene_name}] Using pre-encoded text condition", flush=True)
    if scene.cond_dict is not None:
        cond_dict = scene.cond_dict
    else:
        print(f"[scene-init:{scene_name}] ⚠ No pre-encoded condition, using zero vector")
        cond_dict = {"prompt_embeds": torch.zeros(1, 512, 4096, dtype=_dtype, device=_device)}

    print(f"[scene-init:{scene_name}] Scene initialization complete "
          f"(FOV={fov_h_deg:.0f}°, fx={fx:.1f})")

    return Session(
        connection_id=connection_id,
        scene_name=scene_name,
        camera=camera,
        renderer=renderer,
        initial_w2c=initial_w2c_cpu,
        cond_dict=cond_dict,
        starting_w2c=initial_w2c_cpu.clone(),
    )


def _make_init_frame(image_path: str) -> Optional[bytes]:
    """Encode input image as JPEG for sending as initial frame to frontend."""
    image_path = _resolve_image_path(image_path)
    if not os.path.exists(image_path):
        return None
    img = cv2.imread(image_path)
    if img is None:
        return None
    img = cv2.resize(img, (832, 480))
    _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return buf.tobytes()


# =============================================================================
# CLI & initialization
# =============================================================================

_server_args = None


def parse_args():
    p = argparse.ArgumentParser(
        description="Self-Forcing World Engine — Gaussian Splatting Web streaming service",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--config_path",     type=str, required=True,
                   help="Inference config file (YAML)")
    p.add_argument("--checkpoint_path", type=str, default=None,
                   help="Model checkpoint path")
    p.add_argument("--scenes_config",   type=str,
                   default=os.path.join(os.path.dirname(__file__), "scenes_gs.yaml"),
                   help="Gaussian scene config file (YAML)")
    p.add_argument("--use_ema",         action="store_true")
    p.add_argument("--fp8",             action="store_true")
    p.add_argument("--taehv",           action="store_true",
                   help="Use TAEHV lightweight VAE decoder")
    p.add_argument("--taehv_checkpoint", type=str, default=None)
    p.add_argument("--translation_step", type=float, default=0.01)
    p.add_argument("--rotation_step_deg", type=float, default=3.0)
    p.add_argument("--prompt", type=str, default="",
                   help="Global text prompt (overrides prompt in scene config)")
    p.add_argument("--save_frames", type=str, default=None,
                   help="Save inference + render frames to specified directory")
    p.add_argument("--show_render", action="store_true",
                   help="Push render condition frames to frontend (left panel)")
    p.add_argument("--render_fov_deg", type=float, default=70.0,
                   help="Render horizontal FOV (degrees, default 70)")
    p.add_argument("--near_plane", type=float, default=0.01,
                   help="Render near plane (default 0.01)")
    p.add_argument("--far_plane", type=float, default=1000.0,
                   help="Render far plane (default 1000.0)")
    p.add_argument("--gaussian_scale", type=float, default=1.0,
                   help="Gaussian scale factor (default 1.0, >1 enlarge, <1 shrink)")
    p.add_argument("--host", type=str, default="0.0.0.0")
    p.add_argument("--port", type=int, default=8889)
    # Scene generation pipeline args
    p.add_argument("--pano_ckpt_dir", type=str, default="checkpoints/gimbal360",
                   help="AutoLevel + LoRA checkpoint directory for pano_gen")
    p.add_argument("--flux_model_path", type=str,
                   default="checkpoints/FLUX.1-Fill-dev",
                   help="FLUX.1-Fill-dev model local path or HF repo ID")
    p.add_argument("--da3_model_dir", type=str,
                   default="checkpoints/DA3NESTED-GIANT-LARGE-1.1",
                   help="DA3 model directory (HuggingFace name or local path)")
    p.add_argument("--gaussian_ckpt", type=str,
                   default="checkpoints/gaussian_predictor_checkpoint.pth",
                   help="Gaussian predictor checkpoint path")
    return p.parse_args()


def _load_pipeline(args, config):
    """Load inference pipeline."""
    global _pipeline, _device, _dtype

    print("[init] Loading CausalInferencePipeline ...")
    t0 = time.perf_counter()
    _pipeline = CausalInferencePipeline(config, device=_device)
    print(f"[init] Pipeline loaded ({(time.perf_counter()-t0)*1000:.0f} ms)")

    ckpt_path = args.checkpoint_path
    if ckpt_path is None and hasattr(config, 'checkpoint_paths'):
        ckpt_path = list(config.checkpoint_paths)[0]
        print(f"[init] Reading checkpoint from config.checkpoint_paths: {ckpt_path}")

    if ckpt_path:
        print(f"[init] Loading checkpoint: {ckpt_path}")
        t0 = time.perf_counter()
        raw_ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

        def _strip_fsdp(sd):
            return {k.replace('._fsdp_wrapped_module', ''): v for k, v in sd.items()}

        if args.use_ema and 'generator_ema' in raw_ckpt:
            ema_sd = _strip_fsdp(raw_ckpt['generator_ema'])
            try:
                _pipeline.generator.load_state_dict(ema_sd)
                print("[init] EMA weights loaded.")
            except RuntimeError:
                remapped = {'model.' + k: v for k, v in ema_sd.items()}
                _pipeline.generator.load_state_dict(remapped)
                print("[init] EMA weights loaded (after remap model. prefix).")
        elif 'generator' in raw_ckpt:
            _pipeline.generator.load_state_dict(_strip_fsdp(raw_ckpt['generator']))
            print("[init] Generator weights loaded.")
        else:
            sd = _strip_fsdp(raw_ckpt)
            if not any(k.startswith('model.') for k in sd):
                sd = {'model.' + k: v for k, v in sd.items()}
            _pipeline.generator.load_state_dict(sd)
            print("[init] Flat format weights loaded.")

        print(f"[init] Checkpoint loaded ({(time.perf_counter()-t0)*1000:.0f} ms)")

    _pipeline = _pipeline.to(dtype=_dtype)
    _pipeline.generator.eval()
    _pipeline.generator.to(device=_device)
    _pipeline.vae.to(device=_device)

    # FP8 quantization
    if args.fp8:
        try:
            from torchao.quantization.quant_api import (
                quantize_, Float8DynamicActivationFloat8WeightConfig)
            quantize_(_pipeline.generator, Float8DynamicActivationFloat8WeightConfig())
            print("[init] FP8 quantization enabled")
        except Exception as e:
            print(f"[init] FP8 quantization failed: {e}")

    # TAEHV
    _TAEHV_DEFAULT_CKPT = "checkpoints/taew2_1.pth"
    use_taehv = args.taehv or bool(getattr(config, 'taehv', False))
    if use_taehv:
        import types
        from demo_utils.taehv import TAEHV

        taehv_ckpt = (
            args.taehv_checkpoint
            or getattr(config, 'taehv_checkpoint', None)
            or _TAEHV_DEFAULT_CKPT
        )
        print(f"[init] Loading TAEHV lightweight VAE decoder: {taehv_ckpt} ...")
        t0 = time.perf_counter()
        _taehv_model = (
            TAEHV(checkpoint_path=taehv_ckpt)
            .to(dtype=torch.float16)
            .eval()
            .requires_grad_(False)
            .to(_device)
        )

        _taehv_latent_cache = []
        _TAEHV_MAX_CACHE = 3

        def _taehv_decode_to_pixel(self_vae, latent, use_cache=False):
            nonlocal _taehv_latent_cache

            if not use_cache:
                pixels = _taehv_model.decode_video(
                    latent.to(dtype=torch.float16), parallel=False)
                return pixels.mul(2).sub(1).float()

            T_new = latent.shape[1]

            if len(_taehv_latent_cache) == 0:
                decode_input = latent
                skip_frames = _taehv_model.frames_to_trim
            else:
                cache_tensor = torch.cat(_taehv_latent_cache, dim=1)
                decode_input = torch.cat([cache_tensor, latent], dim=1)
                skip_frames = cache_tensor.shape[1] * 4

            all_pixels = _taehv_model.decode_video(
                decode_input.to(dtype=torch.float16), parallel=False)
            new_pixels = all_pixels[:, skip_frames:]

            for t in range(T_new):
                _taehv_latent_cache.append(latent[:, t:t+1].detach())
            if len(_taehv_latent_cache) > _TAEHV_MAX_CACHE:
                _taehv_latent_cache = _taehv_latent_cache[-_TAEHV_MAX_CACHE:]

            return new_pixels.mul(2).sub(1).float()

        _pipeline.vae.decode_to_pixel = types.MethodType(
            _taehv_decode_to_pixel, _pipeline.vae)

        _original_clear_cache = _pipeline.vae.model.clear_cache
        def _taehv_aware_clear_cache():
            _original_clear_cache()
            nonlocal _taehv_latent_cache
            _taehv_latent_cache = []
        _pipeline.vae.model.clear_cache = _taehv_aware_clear_cache

        print(f"[init] TAEHV VAE decoder activated (streaming cache mode) "
              f" ({(time.perf_counter()-t0)*1000:.0f} ms)")


def _load_scenes(args):
    """Load Gaussian scene configuration from YAML file."""
    global _scenes, _scene_map

    scenes_cfg_path = args.scenes_config
    if not os.path.exists(scenes_cfg_path):
        print(f"[WARNING] Scene config file not found: {scenes_cfg_path}")
        print("[WARNING] Will create a default scene")
        _scenes = [SceneData(
            name="default",
            display_name="Default Scene",
            gs_file="test/scene.ply",
            prompt=args.prompt or "",
        )]
    else:
        cfg = OmegaConf.load(scenes_cfg_path)
        scenes_list = cfg.get("scenes", [])
        _scenes = []
        for s in scenes_list:
            # Support gs_file / ply_file / ckpt_file field names
            gs_file = (s.get("gs_file", "")
                       or s.get("ply_file", "")
                       or s.get("ckpt_file", ""))
            cam_pos = s.get("camera_position", None)
            cam_look = s.get("camera_look_at", None)
            # OmegaConf ListConfig → Python list
            if cam_pos is not None:
                cam_pos = list(cam_pos)
            if cam_look is not None:
                cam_look = list(cam_look)

            _scenes.append(SceneData(
                name=s.get("name", f"scene_{len(_scenes)}"),
                display_name=s.get("display_name", s.get("name", "")),
                gs_file=gs_file,
                prompt=args.prompt or s.get("prompt", ""),
                input_image_path=s.get("input_image", ""),
                camera_position=cam_pos,
                camera_look_at=cam_look,
            ))
        print(f"[init] Loaded {len(_scenes)} scenes: {[s.name for s in _scenes]}")

    _scene_map = {s.name: s for s in _scenes}

    for s in _scenes:
        gs_path = _resolve_image_path(s.gs_file)
        exists = os.path.exists(gs_path)
        status = "✓" if exists else "✗ not found"
        print(f"[init]   scene '{s.name}': {gs_path} [{status}]")


def _prewarm_scenes():
    """Preload Gaussian data for all scenes at startup."""
    if not HAS_GSPLAT:
        print("[prewarm] gsplat not available, skipping scene prewarm")
        return

    for scene in _scenes:
        if scene.initialized:
            continue

        gs_path = _resolve_image_path(scene.gs_file)
        if not os.path.exists(gs_path):
            print(f"[prewarm] ⚠ scene '{scene.name}' Gaussian file not found: {gs_path}")
            continue

        print(f"[prewarm] Preloading scene '{scene.name}': {gs_path} ...")
        t0 = time.perf_counter()
        try:
            with torch.no_grad():
                scene.gs_data = _load_gaussians(gs_path, str(_device))
            scene.initialized = True
            dt = (time.perf_counter() - t0) * 1000
            print(f"[prewarm] Scene '{scene.name}' Gaussian load complete ({dt:.0f} ms, "
                  f"{scene.gs_data['means'].shape[0]:,} Gaussians)")
        except Exception as e:
            print(f"[prewarm] ⚠ scene '{scene.name}' preload failed: {e}")
            traceback.print_exc()

    ok = sum(1 for s in _scenes if s.initialized)
    print(f"[prewarm] Prewarm complete: {ok}/{len(_scenes)} scenes ready")


def _preencode_prompts():
    """Pre-encode text prompts for all scenes at startup, then offload text_encoder to CPU."""
    print("[preencode] Pre-encoding text conditions for all scenes ...")
    t0 = time.perf_counter()

    _pipeline.text_encoder.to(device=_device)

    for scene in _scenes:
        if scene.cond_dict is not None:
            continue
        if scene.prompt.strip():
            with torch.no_grad():
                cond_dict = _pipeline.text_encoder([scene.prompt])
            scene.cond_dict = {
                k: v.to(dtype=_dtype, device=_device).clone()
                for k, v in cond_dict.items()
            }
            print(f"[preencode]   scene '{scene.name}': text encoding complete")
        else:
            scene.cond_dict = {
                "prompt_embeds": torch.zeros(1, 512, 4096, dtype=_dtype, device=_device)
            }
            print(f"[preencode]   scene '{scene.name}': empty prompt → zero vector")

    _pipeline.text_encoder.to("cpu")
    torch.cuda.empty_cache()

    dt = (time.perf_counter() - t0) * 1000
    print(f"[preencode] Text pre-encoding complete ({dt:.0f} ms), text_encoder offloaded to CPU")


def _encode_single_scene_prompt(scene: SceneData):
    """
    Encode the text prompt for a single scene (used for dynamically added scenes).

    Temporarily moves the text encoder to GPU, encodes the prompt, then moves
    it back to CPU to free VRAM. Thread-safe via _pipeline_lock.
    """
    with _pipeline_lock:
        if scene.prompt.strip():
            print(f"[encode] Encoding prompt for scene '{scene.name}': {scene.prompt[:60]}...")
            _pipeline.text_encoder.to(device=_device)
            with torch.no_grad():
                cond_dict = _pipeline.text_encoder([scene.prompt])
            scene.cond_dict = {
                k: v.to(dtype=_dtype, device=_device).clone()
                for k, v in cond_dict.items()
            }
            _pipeline.text_encoder.to("cpu")
            torch.cuda.empty_cache()
            print(f"[encode] Scene '{scene.name}' prompt encoding complete")
        else:
            scene.cond_dict = {
                "prompt_embeds": torch.zeros(1, 512, 4096, dtype=_dtype, device=_device)
            }
            print(f"[encode] Scene '{scene.name}' empty prompt → zero vector")


def main():
    global _server_args, _config

    args = parse_args()
    _server_args = args

    # Check gsplat
    if not HAS_GSPLAT:
        print("\n[ERROR] gsplat not installed or unavailable.")
        print("[ERROR] Please install gsplat: cd gsplat && pip install -e .")
        sys.exit(1)

    # Load config
    print(f"\n[init] ROOT_DIR = {ROOT_DIR}")
    print(f"[init] CWD     = {os.getcwd()}")
    print(f"[init] Loading config: {args.config_path}")
    config = OmegaConf.load(args.config_path)
    default_cfg = OmegaConf.load(os.path.join(ROOT_DIR, "configs/default_config.yaml"))
    config = OmegaConf.merge(default_cfg, config)
    _config = config

    if hasattr(config, "denoising_step_lists") and not hasattr(config, "denoising_step_list"):
        OmegaConf.update(config, "denoising_step_list",
                         list(config.denoising_step_lists[0]))

    # Load model
    _load_pipeline(args, config)
    _load_scenes(args)

    # Prewarm scenes (load Gaussian data)
    _prewarm_scenes()

    # Pre-encode text conditions → offload text_encoder → free VRAM
    _preencode_prompts()

    # Print startup info
    print(f"\n{'='*60}")
    print(f"  Self-Forcing World Engine (Gaussian Splatting)")
    print(f"  HTTP:  http://{args.host}:{args.port}")
    print(f"  WS:    ws://{args.host}:{args.port}/ws/frames")
    print(f"  scenes: {[s.name for s in _scenes]}")
    print(f"  frontend: {'mounted at ' + _DIST_DIR if os.path.isdir(_DIST_DIR) else 'use vite dev server'}")
    print(f"{'='*60}\n")

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
