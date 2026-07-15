"""Inference script for Panorama -> 3DGS model
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

# Ensure sharp package is findable when running from project root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2
import numpy as np
import torch
from PIL import Image
from plyfile import PlyData, PlyElement

from sharp.models import create_pano_predictor, create_depth_anything_v2
from sharp.models.params import GaussianDecoderParams, InitializerParams, PredictorParams
from sharp.utils.gaussians import Gaussians3D, convert_rgb_to_spherical_harmonics
from sharp.utils.color_space import linearRGB2sRGB


LOGGER = logging.getLogger(__name__)


# ============================================================
# Input loading
# ============================================================
def load_panorama(image_path: Path, target_height: int | None, target_width: int | None) -> np.ndarray:
    """Load equirectangular panorama as float32 [H, W, 3] in [0, 1]."""
    img_pil = Image.open(image_path).convert("RGB")
    if target_height == None or target_width == None:
        target_width, target_height = img_pil.size
        target_height = round(target_height / 16) * 16
        target_width = target_height * 2
    img_pil = img_pil.resize((target_width, target_height), Image.Resampling.LANCZOS)
    return np.asarray(img_pil, dtype=np.float32) / 255.0


def load_depth_npy(depth_path: Path, target_height: int, target_width: int, factor: float = 1.0) -> np.ndarray:
    """Load depth from .npy as float32 [H, W]."""
    depth = np.load(depth_path).astype(np.float32)
    if depth.ndim == 3:
        depth = depth.squeeze(-1) if depth.shape[-1] == 1 else depth.squeeze(0)
    if depth.shape[0] != target_height or depth.shape[1] != target_width:
        depth = cv2.resize(depth, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
    depth = depth * factor
    depth = np.clip(depth, 0.001, None)
    return depth


def load_depth_png(depth_path: Path, target_height: int, target_width: int) -> np.ndarray:
    """Load depth from .png (uint16, millimeters) as float32 [H, W] in meters."""
    depth_uint16 = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
    depth = depth_uint16.astype(np.float32) / 1000.0
    if depth.ndim == 3:
        depth = depth.squeeze(-1) if depth.shape[-1] == 1 else depth.squeeze(0)
    if depth.shape[0] != target_height or depth.shape[1] != target_width:
        depth = cv2.resize(depth, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
    depth = np.clip(depth, 0.001, None)
    return depth


def load_depth(depth_path: Path, target_height: int, target_width: int, factor: float = 1.0) -> np.ndarray:
    """Load depth from .npy or .png, dispatching by extension."""
    suffix = depth_path.suffix.lower()
    if suffix == ".npy":
        return load_depth_npy(depth_path, target_height, target_width, factor=factor)
    elif suffix == ".png":
        return load_depth_png(depth_path, target_height, target_width)
    else:
        raise ValueError(f"Unsupported depth format: {suffix}. Use .npy or .png.")


# ============================================================
# PLY saving
# ============================================================


def save_pano_ply(
    gaussians: Gaussians3D,
    image_shape: tuple[int, int],
    output_path: Path,
) -> None:
    """Save Gaussians3D to a standard 3DGS .ply file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    def _inverse_sigmoid(tensor: torch.Tensor) -> torch.Tensor:
        return torch.log(tensor / (1.0 - tensor))

    xyz = gaussians.mean_vectors.flatten(0, 1)
    scale_logits = torch.log(gaussians.singular_values.clamp(min=1e-8)).flatten(0, 1)
    quaternions = gaussians.quaternions.flatten(0, 1)
    # Model outputs colors in linearRGB space (see composer._color_activation).
    # Convert to sRGB before saving so that external viewers (which assume
    # sRGB / no gamma correction) display correct brightness.
    colors_srgb = linearRGB2sRGB(gaussians.colors.flatten(0, 1))
    spherical_harmonics = convert_rgb_to_spherical_harmonics(colors_srgb)
    opacities_clamped = gaussians.opacities.flatten(0, 1).clamp(min=1e-4, max=1.0 - 1e-4)
    opacity_logits = _inverse_sigmoid(opacities_clamped).unsqueeze(-1)

    attributes = torch.cat(
        (xyz, spherical_harmonics, opacity_logits, scale_logits, quaternions),
        dim=1,
    )

    dtype_full = [
        (attr, "f4")
        for attr in ["x", "y", "z"]
        + [f"f_dc_{i}" for i in range(3)]
        + ["opacity"]
        + [f"scale_{i}" for i in range(3)]
        + [f"rot_{i}" for i in range(4)]
    ]

    num_gaussians = len(xyz)
    elements = np.empty(num_gaussians, dtype=dtype_full)
    elements[:] = list(map(tuple, attributes.detach().cpu().numpy()))
    vertex_element = PlyElement.describe(elements, "vertex")

    image_height, image_width = image_shape

    dtype_image_size = [("image_size", "u4")]
    image_size_array = np.empty(2, dtype=dtype_image_size)
    image_size_array[:] = np.array([image_width, image_height])
    image_size_element = PlyElement.describe(image_size_array, "image_size")

    dummy_focal = float(np.sqrt(image_width**2 + image_height**2)) / 2.0
    dtype_intrinsic = [("intrinsic", "f4")]
    intrinsic_array = np.empty(9, dtype=dtype_intrinsic)
    intrinsic_array[:] = np.array([
        dummy_focal, 0, image_width * 0.5,
        0, dummy_focal, image_height * 0.5,
        0, 0, 1,
    ])
    intrinsic_element = PlyElement.describe(intrinsic_array, "intrinsic")

    dtype_extrinsic = [("extrinsic", "f4")]
    extrinsic_array = np.empty(16, dtype=dtype_extrinsic)
    extrinsic_array[:] = np.eye(4).flatten()
    extrinsic_element = PlyElement.describe(extrinsic_array, "extrinsic")

    dtype_frames = [("frame", "i4")]
    frame_array = np.empty(2, dtype=dtype_frames)
    frame_array[:] = np.array([1, num_gaussians], dtype=np.int32)
    frame_element = PlyElement.describe(frame_array, "frame")

    dtype_disparity = [("disparity", "f4")]
    disparity_array = np.empty(2, dtype=dtype_disparity)
    depth_from_xyz = torch.sqrt(
        gaussians.mean_vectors[0, :, 0] ** 2
        + gaussians.mean_vectors[0, :, 1] ** 2
        + gaussians.mean_vectors[0, :, 2] ** 2
    )
    disparity = 1.0 / depth_from_xyz.clamp(min=1e-4)
    quantiles = (
        torch.quantile(disparity, q=torch.tensor([0.1, 0.9], device=disparity.device))
        .float()
        .cpu()
        .numpy()
    )
    disparity_array[:] = quantiles
    disparity_element = PlyElement.describe(disparity_array, "disparity")

    dtype_color_space = [("color_space", "u1")]
    color_space_array = np.empty(1, dtype=dtype_color_space)
    color_space_array[:] = np.array([0]).flatten()  # 0 = sRGB (colors already converted)
    color_space_element = PlyElement.describe(color_space_array, "color_space")

    dtype_version = [("version", "u1")]
    version_array = np.empty(3, dtype=dtype_version)
    version_array[:] = np.array([1, 5, 0], dtype=np.uint8).flatten()
    version_element = PlyElement.describe(version_array, "version")

    plydata = PlyData([
        vertex_element,
        extrinsic_element,
        intrinsic_element,
        image_size_element,
        frame_element,
        disparity_element,
        color_space_element,
        version_element,
    ])
    plydata.write(str(output_path))
    LOGGER.info("Saved %d Gaussians to %s", num_gaussians, output_path)


# ============================================================
# Model building & checkpoint loading
# ============================================================


def build_pano_predictor(
    device: torch.device,
    depth_model_path: str,
    infer_depth: bool = False,
    num_layers: int = 1,
) -> torch.nn.Module:
    """Build PanoGaussianPredictor with the same architecture as training."""
    params = PredictorParams(
        initializer=InitializerParams(
            num_layers=num_layers,
            stride=2,
            scale_factor=1.0,
            disparity_factor=1.0,
            color_option="all_layers",
            first_layer_depth_option="surface_min",
            rest_layer_depth_option="surface_min",
            normalize_depth=False,
            feature_input_stop_grad=True,
        ),
        gaussian_decoder=GaussianDecoderParams(dim_out=32, stride=2),
    )
    if not infer_depth:
        params.depth_alignment = None

    depth_anything_v2 = create_depth_anything_v2(
        midas_model_type="vitl",
        pretrained_path=depth_model_path,
    )
    predictor = create_pano_predictor(params, depth_anything_v2)
    predictor = predictor.to(device)
    return predictor


def load_checkpoint(
    model: torch.nn.Module, checkpoint_path: str, device: torch.device
) -> None:
    """Load model weights from a training checkpoint."""
    LOGGER.info("Loading checkpoint from %s ...", checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    if missing_keys:
        LOGGER.warning("Missing keys: %s", missing_keys)
    if unexpected_keys:
        LOGGER.warning("Unexpected keys: %s", unexpected_keys)
    LOGGER.info(
        "Checkpoint loaded (step=%s, epoch=%s).",
        checkpoint.get("global_step", "N/A"),
        checkpoint.get("epoch", "N/A"),
    )


# ============================================================
# Main inference
# ============================================================


@torch.no_grad()
def infer(args: argparse.Namespace) -> None:
    """Run inference: predict Gaussians, save PLY, render panorama & depth."""
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    LOGGER.info("Using device: %s", device)

    LOGGER.info("Building PanoGaussianPredictor ...")
    model = build_pano_predictor(device, args.depth_model_path, num_layers=args.num_layers)
    load_checkpoint(model, args.checkpoint, device)
    model.eval()

    LOGGER.info("Loading panorama from %s ...", args.image)
    image_np = load_panorama(args.image, args.pano_height, args.pano_width)
    image_height, image_width = image_np.shape[:2]
    image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).unsqueeze(0).to(device)
    LOGGER.info("Input panorama shape: %s", list(image_tensor.shape))

    LOGGER.info("Loading depth from %s ...", args.depth)
    depth_np = load_depth(args.depth, image_height, image_width, args.depth_factor)
    depth_tensor = torch.from_numpy(depth_np).unsqueeze(0).unsqueeze(0).to(device)
    LOGGER.info("Input depth shape:    %s", list(depth_tensor.shape))   


    LOGGER.info("Running forward pass ...")
    predictor_output = model(
        image_tensor, depth_gt=depth_tensor
    )
    gaussians = predictor_output.gaussians

    # Prepare output directory; file names match input image stem
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(args.image).stem

    # Save PLY
    gaussians_cpu = gaussians.to(torch.device("cpu"))
    ply_path = output_dir / f"{stem}.ply"
    LOGGER.info("Saving Gaussian PLY to %s ...", ply_path)
    save_pano_ply(gaussians_cpu, (image_height, image_width), ply_path)


def main():
    parser = argparse.ArgumentParser(
        description="Infer Panorama -> 3DGS: single GPU, single RGB+depth input, "
                    "save Gaussian PLY + rendered panorama + depth."
    )
    parser.add_argument(
        "--image", "-i", type=Path, required=True,
        help="Path to equirectangular panorama image (jpg/png).",
    )
    parser.add_argument(
        "--depth", "-d", type=Path, required=True,
        help="Path to depth map (.npy or .png).",
    )
    parser.add_argument(
        "--depth_factor", type=float, default=1.0,
        help="Depth factor (default: 1.0).",
    )
    parser.add_argument(
        "--mask", "-m", type=Path, default=None,
        help="Optional sky mask (.png). Black pixels = sky; their depth is set "
             "to 20x the max non-sky depth.",
    )
    parser.add_argument(
        "--checkpoint", "-c", type=str, required=True,
        help="Path to training checkpoint (.pth) containing model_state_dict.",
    )
    parser.add_argument(
        "--output_dir", "-o", type=Path, default=Path("output"),
        help="Output directory (default: output). Files named after input image.",
    )
    parser.add_argument(
        "--depth_model_path", type=str,
        default=None,
        help="Path to DepthAnythingV2 pretrained weights.",
    )
    parser.add_argument("--pano_height", default=None)
    parser.add_argument("--pano_width", default=None)
    parser.add_argument(
        "--face_size", type=int, default=256,
        help="Cubemap face resolution for rendering (default: 256).",
    )
    parser.add_argument(
        "--depth_max", type=float, default=100.0,
        help="Max depth for colorized depth visualization (default: 10.0).",
    )
    parser.add_argument("--device", type=str, default="cuda")

    # Optional multi-view rendering from camera poses
    parser.add_argument(
        "--camera_json", type=Path, default=None,
        help="Path to camera.json with camera poses for novel-view rendering.",
    )
    parser.add_argument(
        "--rotation_npy", type=Path, default=None,
        help="Path to .npy file with camera rotations, shape (N, 3, 3).",
    )
    parser.add_argument(
        "--translation_npy", type=Path, default=None,
        help="Path to .npy file with camera translations, shape (N, 3).",
    )
    parser.add_argument(
        "--save_video", action="store_true", default=False,
        help="Save rendered NVS frames as an MP4 video.",
    )
    parser.add_argument(
        "--video_fps", type=int, default=24,
        help="Frame rate for the NVS video (default: 5).",
    )
    parser.add_argument(
        "--dap_scale_factor", type=float, default=1.0,
    )
    parser.add_argument(
        "--num_layers", type=int, default=1,
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    infer(args)


if __name__ == "__main__":
    main()
