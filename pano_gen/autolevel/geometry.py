"""Geometry utilities: perspective → ERP coordinate grids and input preprocessing."""

import numpy as np
import torch
import torch.nn.functional as F


def _build_rotation(pitch_deg: torch.Tensor, roll_deg: torch.Tensor) -> torch.Tensor:
    """Build (B, 3, 3) rotation matrix R = Rz(roll) @ Rx(pitch)."""
    pitch = torch.deg2rad(pitch_deg)
    roll = torch.deg2rad(roll_deg)
    zero = torch.zeros_like(pitch)
    one = torch.ones_like(pitch)

    Rx = torch.stack([
        torch.stack([one,  zero,              zero             ], dim=-1),
        torch.stack([zero, torch.cos(pitch), -torch.sin(pitch)], dim=-1),
        torch.stack([zero, torch.sin(pitch),  torch.cos(pitch)], dim=-1),
    ], dim=-2)

    Rz = torch.stack([
        torch.stack([torch.cos(roll), -torch.sin(roll), zero], dim=-1),
        torch.stack([torch.sin(roll),  torch.cos(roll), zero], dim=-1),
        torch.stack([zero,             zero,             one ], dim=-1),
    ], dim=-2)

    return torch.bmm(Rz, Rx)


def persp_to_erp_grid(
    fov_deg: torch.Tensor,
    pitch_deg: torch.Tensor,
    roll_deg: torch.Tensor,
    H: int,
    W: int,
    device: str = "cpu",
) -> torch.Tensor:
    """Forward mapping: perspective pixel → ERP (u, v) in [-1, 1].

    Returns (B, 2, H, W) where channel 0 is u (horizontal) and channel 1 is v (vertical).

    Args:
        fov_deg: (B,) vertical field of view in degrees.
        pitch_deg: (B,) pitch in degrees (positive = tilt up).
        roll_deg: (B,) roll in degrees.
        H, W: output grid spatial dimensions (= crop H, W).
        device: torch device string.
    """
    fov_deg = fov_deg.to(device)
    pitch_deg = pitch_deg.to(device)
    roll_deg = roll_deg.to(device)
    B = fov_deg.shape[0]
    aspect_ratio = W / float(H)

    y = torch.linspace(-1.0, 1.0, H, device=device, dtype=torch.float32)
    x = torch.linspace(-aspect_ratio, aspect_ratio, W, device=device, dtype=torch.float32)
    grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")

    grid_x = grid_x.unsqueeze(0).unsqueeze(-1).expand(B, -1, -1, -1)
    grid_y = grid_y.unsqueeze(0).unsqueeze(-1).expand(B, -1, -1, -1)

    f = (1.0 / torch.tan(torch.deg2rad(fov_deg / 2.0))).view(B, 1, 1, 1).expand(B, H, W, 1)
    rays = torch.cat([grid_x, grid_y, f], dim=-1)
    rays = rays / torch.norm(rays, dim=-1, keepdim=True)

    R = _build_rotation(pitch_deg, roll_deg)
    rot_rays = torch.bmm(rays.view(B, -1, 3), R.transpose(1, 2)).view(B, H, W, 3)

    x_r, y_r, z_r = rot_rays[..., 0], rot_rays[..., 1], rot_rays[..., 2]
    phi = torch.atan2(x_r, z_r)
    theta = torch.asin(torch.clamp(y_r, -1.0 + 1e-6, 1.0 - 1e-6))

    u = phi / np.pi
    v = theta / (np.pi / 2.0)
    return torch.stack([u, v], dim=1)


def preprocess_image(image: torch.Tensor, short_side: int = 320, divisor: int = 32) -> tuple[torch.Tensor, tuple[float, float]]:
    """Resize short side to `short_side`, then center-crop to multiples of `divisor`.

    Args:
        image: (C, H, W) or (B, C, H, W) float tensor in [0, 1].
        short_side: target short side length.
        divisor: both dimensions are cropped to the nearest multiple.

    Returns:
        processed: same shape type as input but resized/cropped.
        scales: (scale_w, scale_h) — resized / original.
    """
    batched = image.dim() == 4
    if not batched:
        image = image.unsqueeze(0)

    B, C, H, W = image.shape
    if H <= W:
        new_h = short_side
        new_w = round(short_side * W / H)
    else:
        new_w = short_side
        new_h = round(short_side * H / W)

    scale_w = new_w / W
    scale_h = new_h / H

    resized = F.interpolate(image, size=(new_h, new_w), mode="bilinear", align_corners=False, antialias=True)

    crop_h = (new_h // divisor) * divisor
    crop_w = (new_w // divisor) * divisor
    top = (new_h - crop_h) // 2
    left = (new_w - crop_w) // 2
    cropped = resized[:, :, top:top + crop_h, left:left + crop_w]

    if not batched:
        cropped = cropped.squeeze(0)

    return cropped, (scale_w, scale_h)
