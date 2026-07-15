"""CLI entry point for 360° panorama generation.

Adapted from Gimbal360/inference.py to use prompt_builder for prompt assembly
and support --back_prompt for describing unseen content.

Usage:
    python pano_gen/inference.py \\
        --image input.jpg \\
        --prompt "a bright breakfast alcove, curved booth seating..." \\
        --back_prompt "with additional alcove seating and a small kitchen behind" \\
        --ckpt_dir checkpoints/gimbal360 \\
        --output_dir outputs
"""

import argparse
import os
import sys

# Ensure local imports (fluxfill_pipeline, camera_utils, autolevel) resolve
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

from fluxfill_pipeline import FluxFillPipeline
from camera_utils import Perspective
from autolevel.model import FlowEstimator
from autolevel.geometry import preprocess_image
from prompt_builder import build_pano_prompt


class PerspectiveToERP:
    """Converts a perspective image to equirectangular conditioning image + mask."""

    def __init__(self, pano_height: int, pano_width: int, ckpt_path: str):
        self.pano_height = pano_height
        self.pano_width = pano_width
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._load_model(ckpt_path)
        self.to_tensor = T.ToTensor()

    def _load_model(self, ckpt_path: str) -> FlowEstimator:
        state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        rigid_filter_cfg = state.get("rigid_filter_cfg", {})
        model = FlowEstimator(freeze_backbone=True, rigid_filter_cfg=rigid_filter_cfg)
        model.backbone.load_state_dict(state["backbone"], strict=True)
        model.ll_enc.load_state_dict(state["ll_enc"], strict=True)
        model.decoder.load_state_dict(state["decoder"], strict=True)
        model = model.to(self.device).eval()
        return model

    @torch.no_grad()
    def estimate_camera(self, img_path: str) -> dict:
        img = Image.open(img_path).convert("RGB")
        img_t = self.to_tensor(img)
        crop_proc, _ = preprocess_image(img_t, short_side=320, divisor=32)
        inp = crop_proc.unsqueeze(0).to(self.device)

        out = self.model(inp)
        params = out["params"][0]
        fov_deg, pitch_deg, roll_deg = params[0].item(), params[1].item(), params[2].item()

        return {
            "roll": roll_deg,
            "pitch": pitch_deg,
            "fov": fov_deg,
        }

    def convert(self, img_path: str) -> tuple:
        """
        Returns:
            conditioning_image (PIL.Image, RGB): gray background with projected perspective content
            mask_image (PIL.Image, L):           white = inpaint region, black = known region
            cam_param (dict):                    estimated camera parameters
        """
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            raise FileNotFoundError(f"Cannot read image: {img_path}")

        height_orig, width_orig = img_bgr.shape[:2]
        cam_param = self.estimate_camera(img_path)
        vertical_fov = cam_param["fov"]
        roll = cam_param["roll"]
        pitch = cam_param["pitch"]

        aspect_ratio = width_orig / height_orig
        resized_height = int((vertical_fov / 180) * self.pano_height)
        resized_width = int(resized_height * aspect_ratio)
        img_bgr = cv2.resize(img_bgr, (resized_width, resized_height), interpolation=cv2.INTER_AREA)

        projector = Perspective(img_bgr, vertical_fov, 0, pitch, roll, vfov=True)
        erp_raw, erp_mask = projector.GetEquirec(self.pano_height, self.pano_width)

        erp_mask = cv2.erode(erp_mask.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=5)

        conditioning = np.full((self.pano_height, self.pano_width, 3), 127, dtype=np.uint8)
        conditioning[erp_mask == 1] = erp_raw[erp_mask == 1].astype(np.uint8)

        inpaint_mask = (1 - erp_mask.astype(np.uint8)) * 255

        conditioning_rgb = cv2.cvtColor(conditioning, cv2.COLOR_BGR2RGB)
        return Image.fromarray(conditioning_rgb), Image.fromarray(inpaint_mask[:, :, 0]), cam_param


def parse_args():
    parser = argparse.ArgumentParser(description="Perspective image → 360° panorama")

    parser.add_argument("--basemodel_name_or_path", type=str, default="checkpoints/FLUX.1-Fill-dev")
    parser.add_argument("--ckpt_dir", type=str, required=True,
                        help="Local folder containing 'pytorch_lora_weights.safetensors' and 'autolevel.pth'")
    parser.add_argument("--output_dir", type=str, default="outputs")

    parser.add_argument("--resolution", type=int, default=960, help="Panorama height (width = 2x)")
    parser.add_argument("--guidance_scale", type=float, default=30.0)
    parser.add_argument("--num_inference_steps", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mixed_precision", type=str, default="bf16", choices=["fp16", "bf16", "fp32"])

    parser.add_argument("--image", type=str, required=True, help="Path to perspective image")
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt for panorama generation")
    parser.add_argument("--back_prompt", type=str, default="",
                        help="Optional description of unseen 270° content")

    parser.add_argument("--save_intermediate", action="store_true",
                        help="Save ERP conditioning image and mask")

    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    if not os.path.exists(args.image):
        print(f"Image not found: {args.image}")
        return

    ckpt_dir = Path(args.ckpt_dir)
    autolevel_ckpt_path = ckpt_dir / "autolevel.pth"
    lora_path = ckpt_dir / "pytorch_lora_weights.safetensors"
    if not autolevel_ckpt_path.exists():
        raise FileNotFoundError(f"AutoLevel checkpoint not found: {autolevel_ckpt_path}")
    if not lora_path.exists():
        raise FileNotFoundError(f"LoRA weights not found: {lora_path}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32
    if args.mixed_precision == "fp16":
        dtype = torch.float16
    elif args.mixed_precision == "bf16":
        dtype = torch.bfloat16

    pano_width = args.resolution * 2
    pano_height = args.resolution
    base_name = Path(args.image).stem

    # --- Step 1: Perspective → ERP ---
    print("Initializing AutoLevel camera calibration...")
    erp_converter = PerspectiveToERP(
        pano_height=pano_height,
        pano_width=pano_width,
        ckpt_path=str(autolevel_ckpt_path),
    )
    print("Converting perspective image to ERP...")
    conditioning_image, mask_image, cam_param = erp_converter.convert(args.image)
    conditioning_image = conditioning_image.resize((pano_width, pano_height), Image.BILINEAR)
    mask_image = mask_image.resize((pano_width, pano_height), Image.NEAREST)

    print(f"   Camera: FOV={cam_param['fov']:.1f} pitch={cam_param['pitch']:.1f} roll={cam_param['roll']:.1f}")

    if args.save_intermediate:
        conditioning_image.save(os.path.join(args.output_dir, f"{base_name}_cond.png"))
        mask_image.save(os.path.join(args.output_dir, f"{base_name}_mask.png"))
        print(f"   Saved intermediate files.")

    # --- Step 2: Build prompt via prompt_builder ---
    prompt_text = build_pano_prompt(args.prompt, args.back_prompt)
    print(f"   Full prompt: {prompt_text}")

    # --- Step 3: FluxFill ---
    print(f"Loading FluxFill pipeline...")
    pipeline = FluxFillPipeline.from_pretrained(args.basemodel_name_or_path, torch_dtype=dtype).to(device)

    print(f"Loading LoRA weights from {lora_path}...")
    pipeline.load_lora_weights(str(lora_path))
    pipeline.fuse_lora(lora_scale=1.0)

    generator = torch.Generator(device).manual_seed(args.seed)

    print(f"Generating panorama...")
    with torch.no_grad():
        result = pipeline(
            prompt=prompt_text,
            image=conditioning_image,
            mask_image=mask_image,
            height=pano_height,
            width=pano_width,
            guidance_scale=args.guidance_scale,
            num_inference_steps=args.num_inference_steps,
            generator=generator,
        ).images[0]

    save_path = os.path.join(args.output_dir, f"{base_name}_pano.png")
    result.save(save_path)
    print(f"Done! Saved to {save_path}")


if __name__ == "__main__":
    main()
