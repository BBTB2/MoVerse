"""Contains utility code for gsplat renderer.

For licensing see accompanying LICENSE file.
Copyright (C) 2025 Apple Inc. All Rights Reserved.
"""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import gsplat
import torch
from torch import nn

from sharp.utils import color_space as cs_utils
from sharp.utils import io, vis
from sharp.utils.gaussians import BackgroundColor, Gaussians3D


class RenderingOutputs(NamedTuple):
    """Outputs of 3D Gaussians renderer."""

    color: torch.Tensor
    depth: torch.Tensor
    alpha: torch.Tensor


def write_renderings(rendering: RenderingOutputs, output_folder: Path, filename: str):
    """Write rendered color/depth/alpha to files."""
    batch_size = len(rendering.color)
    if batch_size != 1:
        raise RuntimeError("We only support saving rendering of batch size = 1")

    def _save_image_tensor(tensor: torch.Tensor, suffix: str):
        np_array = tensor.permute(1, 2, 0).numpy()
        io.save_image(np_array, (output_folder / filename).with_suffix(suffix))

    color = (rendering.color[0].cpu() * 255.0).to(dtype=torch.uint8)
    colorized_depth = vis.colorize_depth(rendering.depth[0], val_max=100.0)
    colorized_alpha = vis.colorize_alpha(rendering.alpha[0])

    _save_image_tensor(color, ".color.png")
    _save_image_tensor(colorized_depth, ".depth.png")
    _save_image_tensor(colorized_alpha, ".alpha.png")


class GSplatRenderer(nn.Module):
    """Module to render 3D Gaussians to images using gsplat."""

    color_space: cs_utils.ColorSpace
    background_color: BackgroundColor

    def __init__(
        self,
        color_space: cs_utils.ColorSpace = "sRGB",
        background_color: BackgroundColor = "black",
        low_pass_filter_eps: float = 0.0,
    ) -> None:
        """Initialize gsplat renderer.

        Args:
            color_space: The color space to use for rendering.
            background_color: The background color to use for rendering.
            low_pass_filter_eps: The epsilon value for the low pass filter.
        """
        super().__init__()
        self.color_space = color_space
        self.background_color = background_color
        self.low_pass_filter_eps = low_pass_filter_eps

    def forward(
        self,
        gaussians: Gaussians3D,
        extrinsics: torch.Tensor,
        intrinsics: torch.Tensor,
        image_width: int,
        image_height: int,
    ) -> RenderingOutputs:
        """Predict images from gaussians.

        Args:
            gaussians: The Gaussians to render.
            extrinsics: The extrinsics of the camera to render to in OpenCV format.
            intrinsics: The intriniscs of the camera to render to in OpenCV format.
            image_width: The desired output image width.
            image_height: The desired output image height.
        """
        batch_size = len(gaussians.mean_vectors)
        outputs_list: list[RenderingOutputs] = []

        for ib in range(batch_size):
            colors, alphas, meta = gsplat.rendering.rasterization(
                means=gaussians.mean_vectors[ib],
                quats=gaussians.quaternions[ib],
                scales=gaussians.singular_values[ib],
                opacities=gaussians.opacities[ib],
                colors=gaussians.colors[ib],
                viewmats=extrinsics[ib : ib + 1],
                Ks=intrinsics[ib : ib + 1, :3, :3],
                width=image_width,
                height=image_height,
                render_mode="RGB+D",
                rasterize_mode="classic",
                absgrad=False,
                packed=False,
                eps2d=self.low_pass_filter_eps,
            )

            rendered_color = colors[..., 0:3].permute([0, 3, 1, 2])
            rendered_depth_unnormalized = colors[..., 3:4].permute([0, 3, 1, 2])
            rendered_alpha = alphas.permute([0, 3, 1, 2])

            # Compose with background color.
            rendered_color = self.compose_with_background(
                rendered_color, rendered_alpha, self.background_color
            )

            # Colorspace conversion.
            if self.color_space == "sRGB":
                pass
            elif self.color_space == "linearRGB":
                rendered_color = cs_utils.linearRGB2sRGB(rendered_color)
            else:
                ValueError("Unsupported ColorSpace type.")

            # splats: (B, N, 10)
            cov2d = self._conics_to_covars2d(meta["conics"])
            # Set the cov2d of invisible splats to 1 to avoid nan in condition number calculation..
            splats_visible_mask = meta["depths"] > 1e-2
            cov2d[~splats_visible_mask][..., 0, 0] = 1
            cov2d[~splats_visible_mask][..., 1, 1] = 1
            cov2d[~splats_visible_mask][..., 0, 1] = 0

            # Normalize the depth by alpha.
            rendered_depth = rendered_depth_unnormalized / torch.clip(rendered_alpha, min=1e-8)

            outputs = RenderingOutputs(
                color=rendered_color,
                depth=rendered_depth,
                alpha=rendered_alpha,
            )
            outputs_list.append(outputs)

        return RenderingOutputs(
            color=torch.cat([item.color for item in outputs_list], dim=0).contiguous(),
            depth=torch.cat([item.depth for item in outputs_list], dim=0).contiguous(),
            alpha=torch.cat([item.alpha for item in outputs_list], dim=0).contiguous(),
        )

    @staticmethod
    def compose_with_background(
        rendered_rgb: torch.Tensor,
        rendered_alpha: torch.Tensor,
        background_color: BackgroundColor,
    ) -> torch.Tensor:
        """Compose rendered RGB with background color."""
        if background_color == "black":
            return rendered_rgb
        elif background_color == "white":
            return rendered_rgb + (1.0 - rendered_alpha)
        elif background_color == "random_color":
            return (
                rendered_rgb
                + (1.0 - rendered_alpha)
                * torch.rand(3, dtype=rendered_rgb.dtype, device=rendered_rgb.device)[
                    None, :, None, None
                ]
            )
        elif background_color == "random_pixel":
            return rendered_rgb + (1.0 - rendered_alpha) * torch.rand_like(rendered_rgb)
        else:
            raise ValueError("Unsupported BackgroundColor type.")

    @staticmethod
    def _conics_to_covars2d(conics: torch.Tensor, eps=1e-8) -> torch.Tensor:
        """Convert conics to covariance matrices."""
        a = conics[..., 0]
        b = conics[..., 1]
        c = conics[..., 2]
        # Reconstruct determinant.
        det = 1 / (a * c - b**2 + eps)
        det = det.clamp(min=eps)
        # Reconstruct covars2d.
        covars2d = torch.zeros(*conics.shape[:-1], 2, 2, device=conics.device)
        covars2d[..., 1, 1] = a * det
        covars2d[..., 0, 0] = c * det
        covars2d[..., 0, 1] = -b * det
        covars2d[..., 1, 0] = -b * det
        covars2d = torch.nan_to_num(covars2d, nan=0.0, posinf=0.0, neginf=0.0)
        return covars2d


class MultiCameraRenderingOutputs(NamedTuple):
    """Outputs of multi-camera 3D Gaussians renderer.

    Each field has shape [B, C, ...] where C is the number of cameras.
    """

    color: torch.Tensor   # [B, C, 3, H, W]
    depth: torch.Tensor   # [B, C, 1, H, W]
    alpha: torch.Tensor   # [B, C, 1, H, W]


class GSplatBatchRenderer(nn.Module):
    """Batch renderer for 3D Gaussians using gsplat's native batch dimension support.

    Unlike GSplatRenderer which loops over each batch element, this renderer
    leverages gsplat's batch prefix dimensions (`[..., N, 3]` for means,
    `[..., C, 4, 4]` for viewmats) to rasterize the entire batch in a single
    kernel call.  This avoids Python-level looping and can significantly
    improve GPU utilization.

    Requirement: all batch elements must have the same number of Gaussians N,
    which is naturally satisfied when Gaussians are produced by a fixed-architecture
    predictor (e.g. PanoGaussianPredictor).
    """

    color_space: cs_utils.ColorSpace
    background_color: BackgroundColor

    def __init__(
        self,
        color_space: cs_utils.ColorSpace = "sRGB",
        background_color: BackgroundColor = "black",
        low_pass_filter_eps: float = 0.0,
    ) -> None:
        """Initialize batch gsplat renderer.

        Args:
            color_space: The color space to use for rendering.
            background_color: The background color to use for rendering.
            low_pass_filter_eps: The epsilon value for the low pass filter.
        """
        super().__init__()
        self.color_space = color_space
        self.background_color = background_color
        self.low_pass_filter_eps = low_pass_filter_eps

    def _postprocess(
        self,
        rendered_color: torch.Tensor,
        rendered_depth_unnormalized: torch.Tensor,
        rendered_alpha: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply background composition, colorspace conversion, and depth normalization.

        Args:
            rendered_color: [..., 3, H, W] rendered RGB.
            rendered_depth_unnormalized: [..., 1, H, W] accumulated depth.
            rendered_alpha: [..., 1, H, W] rendered alpha.

        Returns:
            Tuple of (postprocessed_color, normalized_depth).
        """
        rendered_color = GSplatRenderer.compose_with_background(
            rendered_color, rendered_alpha, self.background_color
        )

        if self.color_space == "sRGB":
            pass
        elif self.color_space == "linearRGB":
            rendered_color = cs_utils.linearRGB2sRGB(rendered_color)
        else:
            raise ValueError("Unsupported ColorSpace type.")

        rendered_depth = rendered_depth_unnormalized / torch.clip(rendered_alpha, min=1e-8)
        return rendered_color, rendered_depth

    def forward(
        self,
        gaussians: Gaussians3D,
        extrinsics: torch.Tensor,
        intrinsics: torch.Tensor,
        image_width: int,
        image_height: int,
    ) -> RenderingOutputs:
        """Render images from gaussians for the entire batch in one call.

        gsplat's rasterization accepts batch prefix dimensions:
          - means:    [B, N, 3]
          - viewmats: [B, C, 4, 4]   (here C=1, one camera per batch element)
        This lets us avoid the Python for-loop over batch elements.

        Args:
            gaussians: The Gaussians to render.  Each field has shape [B, N, ...].
            extrinsics: Camera extrinsics in OpenCV format. [B, 4, 4]
            intrinsics: Camera intrinsics in OpenCV format. [B, 3, 3] or [B, 4, 4]
            image_width: The desired output image width.
            image_height: The desired output image height.

        Returns:
            RenderingOutputs with color [B, 3, H, W], depth [B, 1, H, W],
            alpha [B, 1, H, W].
        """
        # gsplat expects viewmats as [B, C, 4, 4] where C is the number of
        # cameras per batch element.  We render one camera per element, so
        # insert a camera dimension: [B, 4, 4] -> [B, 1, 4, 4].
        viewmats_batched = extrinsics.unsqueeze(1)
        intrinsics_3x3 = intrinsics[:, :3, :3] if intrinsics.shape[-1] == 4 else intrinsics
        ks_batched = intrinsics_3x3.unsqueeze(1)

        # Single batched rasterization call: means [B, N, 3], viewmats [B, 1, 4, 4]
        # Output colors: [B, 1, H, W, 4], alphas: [B, 1, H, W, 1]
        colors, alphas, meta = gsplat.rendering.rasterization(
            means=gaussians.mean_vectors,
            quats=gaussians.quaternions,
            scales=gaussians.singular_values,
            opacities=gaussians.opacities,
            colors=gaussians.colors,
            viewmats=viewmats_batched,
            Ks=ks_batched,
            width=image_width,
            height=image_height,
            render_mode="RGB+D",
            rasterize_mode="classic",
            absgrad=False,
            packed=False,
            eps2d=self.low_pass_filter_eps,
        )

        # colors: [B, 1, H, W, 4] -> squeeze camera dim -> [B, H, W, 4]
        # alphas: [B, 1, H, W, 1] -> squeeze camera dim -> [B, H, W, 1]
        colors = colors.squeeze(1)
        alphas = alphas.squeeze(1)

        # Rearrange to [B, C, H, W] format
        rendered_color = colors[..., 0:3].permute(0, 3, 1, 2)
        rendered_depth_unnormalized = colors[..., 3:4].permute(0, 3, 1, 2)
        rendered_alpha = alphas.permute(0, 3, 1, 2)

        rendered_color, rendered_depth = self._postprocess(
            rendered_color, rendered_depth_unnormalized, rendered_alpha
        )

        return RenderingOutputs(
            color=rendered_color,
            depth=rendered_depth,
            alpha=rendered_alpha,
        )

    def render_multi_camera(
        self,
        gaussians: Gaussians3D,
        extrinsics: torch.Tensor,
        intrinsics: torch.Tensor,
        image_width: int,
        image_height: int,
    ) -> MultiCameraRenderingOutputs:
        """Render the same Gaussians to multiple cameras in one rasterization call.

        This leverages gsplat's C (camera) dimension to render all cameras
        simultaneously, avoiding Python-level loops over camera views.
        Ideal for cubemap rendering where the same scene is rendered to 6 faces.

        Args:
            gaussians: The Gaussians to render.  Each field has shape [B, N, ...].
            extrinsics: Camera extrinsics in OpenCV format. [B, C, 4, 4]
                where C is the number of cameras per batch element.
            intrinsics: Camera intrinsics in OpenCV format. [B, C, 3, 3] or [B, C, 4, 4]
            image_width: The desired output image width.
            image_height: The desired output image height.

        Returns:
            MultiCameraRenderingOutputs with:
                color [B, C, 3, H, W],
                depth [B, C, 1, H, W],
                alpha [B, C, 1, H, W].
        """
        # extrinsics already has shape [B, C, 4, 4] — pass directly as viewmats.
        intrinsics_3x3 = (
            intrinsics[..., :3, :3] if intrinsics.shape[-1] == 4 else intrinsics
        )

        # Rasterization: means [B, N, 3], viewmats [B, C, 4, 4]
        # Output colors: [B, C, H, W, 4], alphas: [B, C, H, W, 1]
        colors, alphas, meta = gsplat.rendering.rasterization(
            means=gaussians.mean_vectors,
            quats=gaussians.quaternions,
            scales=gaussians.singular_values,
            opacities=gaussians.opacities,
            colors=gaussians.colors,
            viewmats=extrinsics,
            Ks=intrinsics_3x3,
            width=image_width,
            height=image_height,
            render_mode="RGB+D",
            rasterize_mode="classic",
            absgrad=False,
            packed=False,
            eps2d=self.low_pass_filter_eps,
        )

        # colors: [B, C, H, W, 4], alphas: [B, C, H, W, 1]
        # Rearrange to [B, C, channels, H, W]
        rendered_color = colors[..., 0:3].permute(0, 1, 4, 2, 3)
        rendered_depth_unnormalized = colors[..., 3:4].permute(0, 1, 4, 2, 3)
        rendered_alpha = alphas.permute(0, 1, 4, 2, 3)

        rendered_color, rendered_depth = self._postprocess(
            rendered_color, rendered_depth_unnormalized, rendered_alpha
        )

        return MultiCameraRenderingOutputs(
            color=rendered_color,
            depth=rendered_depth,
            alpha=rendered_alpha,
        )
