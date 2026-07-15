"""Contains modules to initialize Gaussians from RGBD.

For licensing see accompanying LICENSE file.
Copyright (C) 2025 Apple Inc. All Rights Reserved.
"""

from __future__ import annotations

from typing import NamedTuple
import math
import torch.nn.functional as F
from sharp.utils.gaussians import Gaussians3D

import torch
from torch import nn

from .params import ColorInitOption, DepthInitOption, InitializerParams


def create_initializer(params: InitializerParams) -> nn.Module:
    """Create inpainter."""
    return MultiLayerInitializer(
        num_layers=params.num_layers,
        stride=params.stride,
        base_depth=params.base_depth,
        scale_factor=params.scale_factor,
        disparity_factor=params.disparity_factor,
        color_option=params.color_option,
        first_layer_depth_option=params.first_layer_depth_option,
        rest_layer_depth_option=params.rest_layer_depth_option,
        normalize_depth=params.normalize_depth,
        feature_input_stop_grad=params.feature_input_stop_grad,
    )


class GaussianBaseValues(NamedTuple):
    """Base values for gaussian predictor.

    We predict x and y in normalized device coordinates (NDC) where (-1, -1) is the top
    left corner and (1, 1) the bottom right corner. The last component of
    mean_vectors_ndc is inverse depth.
    """

    mean_x_ndc: torch.Tensor
    mean_y_ndc: torch.Tensor
    mean_inverse_z_ndc: torch.Tensor

    scales: torch.Tensor
    quaternions: torch.Tensor
    colors: torch.Tensor
    opacities: torch.Tensor


class InitializerOutput(NamedTuple):
    """Output of initializer."""

    # Gaussian base values.
    gaussian_base_values: GaussianBaseValues

    # Feature input to the Gaussian predictor.
    feature_input: torch.Tensor

    # Global scale to unscale output.
    global_scale: torch.Tensor | None = None


class MultiLayerInitializer(nn.Module):
    """Initialize Gaussians with multilayer representation.

    The returned tensors have the shape

        batch_size x dim x num_layers x height x width

    where dim indicates the dimensionality of the property.
    Some of the dimensions might be set to 1 for efficiency reasons.
    """

    def __init__(
        self,
        num_layers: int,
        stride: int,
        base_depth: float,
        scale_factor: float,
        disparity_factor: float,
        color_option: ColorInitOption = "first_layer",
        first_layer_depth_option: DepthInitOption = "surface_min",
        rest_layer_depth_option: DepthInitOption = "surface_min",
        normalize_depth: bool = True,
        feature_input_stop_grad: bool = True,
    ) -> None:
        """Initialize MultilayerInitializer.

        Args:
            stride: The downsample rate of output feature map.
            base_depth: The depth of the first layer (after the foreground
                layer if use_depth=True).
            scale_factor: Multiply scale of Gaussians by this factor.
            disparity_factor: Factor to convert inverse depth to disparity.
            num_layers: How many layers of Gaussians to predict.
            color_option: Which color option to initialize the multi-layer gaussians.
            first_layer_depth_option: Which depth option to initialize the first layer of gaussians.
            rest_layer_depth_option: Which depth option to initialize the rest layers of gaussians.
            normalize_depth: # Whether to normalize depth to [DepthTransformParam.depth_min,
                DepthTransformParam.depth_max).
            feature_input_stop_grad: Whether to not propagate gradients through feature inputs.
        """
        super().__init__()
        self.num_layers = num_layers
        self.stride = stride
        self.base_depth = base_depth
        self.scale_factor = scale_factor
        self.disparity_factor = disparity_factor
        self.color_option = color_option
        self.first_layer_depth_option = first_layer_depth_option
        self.rest_layer_depth_option = rest_layer_depth_option
        self.normalize_depth = normalize_depth
        self.feature_input_stop_grad = feature_input_stop_grad

    def prepare_feature_input(self, image: torch.Tensor, depth: torch.Tensor) -> torch.Tensor:
        """Prepare the feature input to the Guassian predictor."""
        if self.feature_input_stop_grad:
            image = image.detach()
            depth = depth.detach()

        normalized_disparity = self.disparity_factor / depth
        features_in = torch.cat([image, normalized_disparity], dim=1)
        features_in = 2.0 * features_in - 1.0
        return features_in

    def forward(self, image: torch.Tensor, depth: torch.Tensor) -> InitializerOutput:
        """Construct Gaussian base values and prepare feature input.

        Args:
            image: The image to process.
            depth: The corresponding depth map from the monodepth network.

        Returns:
            The base value for Gaussians.
        """
        image = image.contiguous()
        depth = depth.contiguous()
        device = depth.device
        batch_size, _, image_height, image_width = depth.shape
        base_height, base_width = (
            image_height // self.stride,
            image_width // self.stride,
        )
        # global_scale is the inverse of the depth_factor, which is used to rescale
        # the depth such that it is numerically stable for training.
        global_scale: torch.Tensor | None = None
        if self.normalize_depth:
            depth, depth_factor = _rescale_depth(depth)
            global_scale = 1.0 / depth_factor

        def _create_disparity_layers(num_layers: int = 1) -> torch.Tensor:
            """Create multiple disparity layers."""
            disparity = torch.linspace(1.0 / self.base_depth, 0.0, num_layers + 1, device=device)
            return disparity[None, None, :-1, None, None].repeat(
                batch_size, 1, 1, base_height, base_width
            )

        def _create_surface_layer(
            depth: torch.Tensor,
            depth_pooling_mode: str,
        ) -> torch.Tensor:
            """Create multiple surface layers."""
            disparity = 1.0 / depth
            if depth_pooling_mode == "min":
                disparity = torch.max_pool2d(disparity, self.stride, self.stride)
            elif depth_pooling_mode == "max":
                disparity = -torch.max_pool2d(-disparity, self.stride, self.stride)
            else:
                raise ValueError(f"Invalid depth pooling mode {depth_pooling_mode}.")

            return disparity[:, :, None, :, :]

        # Input disparity dimensions:
        #   (batch_size, num_channels in (1, 2), height, width)

        # Output disparity dimensions:
        #   (batch_size, num_channels=1, num_layers in (1, 2), height, width)
        if self.first_layer_depth_option == "surface_min":
            first_disparity = _create_surface_layer(depth[:, 0:1], "min")
        elif self.first_layer_depth_option == "surface_max":
            first_disparity = _create_surface_layer(depth[:, 0:1], "max")
        elif self.first_layer_depth_option in ("base_depth", "linear_disparity"):
            first_disparity = _create_disparity_layers()
        else:
            raise ValueError(f"Unknown depth init option: {self.first_layer_depth_option}.")

        if self.num_layers == 1:
            disparity = first_disparity
        else:  # Fill in the rest layers.
            following_depth = depth if depth.shape[1] == 1 else depth[:, 1:]
            if self.rest_layer_depth_option == "surface_min":
                following_disparity = _create_surface_layer(following_depth, "min")
            elif self.rest_layer_depth_option == "surface_max":
                following_disparity = _create_surface_layer(following_depth, "max")
            elif self.rest_layer_depth_option == "base_depth":
                following_disparity = torch.cat(
                    [_create_disparity_layers() for i in range(self.num_layers - 1)],
                    dim=2,
                )
            elif self.rest_layer_depth_option == "linear_disparity":
                following_disparity = _create_disparity_layers(self.num_layers - 1)
            else:
                raise ValueError(f"Unknown depth init option: {self.rest_layer_depth_option}.")

            disparity = torch.cat([first_disparity, following_disparity], dim=2)

        # Prepare base values.
        base_x_ndc, base_y_ndc = _create_base_xy(depth, self.stride, self.num_layers)
        disparity_scale_factor = 2 * self.scale_factor * self.stride / float(image_width)
        base_scales = _create_base_scale(disparity, disparity_scale_factor)

        base_quaternions = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device)
        base_quaternions = base_quaternions[None, :, None, None, None]

        # Initializing the opacitiy this way ensures that the initial transmittance
        # is approximately
        #
        #       1 / e ~= (1 - 1 / self.num_layers)**self.num_layers
        #
        # and hence independent of the number of layers.
        #
        base_opacities = torch.tensor([min(1.0 / self.num_layers, 0.5)], device=device)
        base_colors = torch.empty(
            batch_size, 3, self.num_layers, base_height, base_width, device=device
        ).fill_(0.5)
        # Dimensions: (batch_size, num_channels, num_layers, height, width)
        if self.color_option == "none":
            pass
        elif self.color_option == "first_layer":
            base_colors[:, :, 0] = torch.nn.functional.avg_pool2d(image, self.stride, self.stride)
        elif self.color_option == "all_layers":
            temp = torch.nn.functional.avg_pool2d(image, self.stride, self.stride)
            base_colors = temp[:, :, None, :, :].repeat(1, 1, self.num_layers, 1, 1)
        else:
            raise ValueError(f"Unknown color init option: {self.color_option}.")

        features_in = self.prepare_feature_input(image, depth)
        base_gaussians = GaussianBaseValues(
            mean_x_ndc=base_x_ndc,
            mean_y_ndc=base_y_ndc,
            mean_inverse_z_ndc=disparity,
            scales=base_scales,
            quaternions=base_quaternions,
            colors=base_colors,
            opacities=base_opacities,
        )

        return InitializerOutput(
            gaussian_base_values=base_gaussians,
            feature_input=features_in,
            global_scale=global_scale,
        )


class PanoGaussianBaseValues(NamedTuple):
    """Base values for panorama gaussian predictor.

    Unlike pinhole NDC, we store 3D direction vectors on the unit sphere
    and inverse depth (radius).
    All spatial tensors have shape [B, 1, num_layers, H, W].
    """

    # Unit sphere direction vectors, shape: [B, 1, num_layers, H, W]
    dir_x: torch.Tensor
    dir_y: torch.Tensor
    dir_z: torch.Tensor

    # Inverse depth (1/radius), shape: [B, 1, num_layers, H, W]
    inverse_depth: torch.Tensor

    scales: torch.Tensor
    quaternions: torch.Tensor
    colors: torch.Tensor
    opacities: torch.Tensor


def _create_base_directions_equirect(
    depth: torch.Tensor, stride: int, num_layers: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create base direction vectors on the unit sphere for equirectangular input.

    For equirectangular projection:
        pixel (u, v) in [0, W) x [0, H) maps to:
            longitude θ = (u / W - 0.5) * 2π       ∈ [-π, π]
            latitude  φ = (0.5 - v / H) * π         ∈ [-π/2, π/2]
        3D direction:
            dir_x = cos(φ) * sin(θ)
            dir_y = -sin(φ)          (y-down, OpenCV convention)
            dir_z = cos(φ) * cos(θ)  (z-forward)

    Args:
        depth: Depth tensor [B, C, H, W] (used only for shape/device).
        stride: Downsampling stride.
        num_layers: Number of Gaussian layers.

    Returns:
        dir_x, dir_y, dir_z: each [B, 1, num_layers, base_H, base_W]
    """
    device = depth.device
    batch_size, _, image_height, image_width = depth.shape
    base_height = image_height // stride
    base_width = image_width // stride

    # Pixel centers in [0, 1]
    u_coords = torch.linspace(
        0.5 / base_width, 1.0 - 0.5 / base_width, base_width, device=device
    )
    v_coords = torch.linspace(
        0.5 / base_height, 1.0 - 0.5 / base_height, base_height, device=device
    )
    grid_v, grid_u = torch.meshgrid(v_coords, u_coords, indexing="ij")

    # Equirectangular → spherical angles
    theta = (grid_u - 0.5) * 2.0 * math.pi   # longitude [-π, π]
    phi = (0.5 - grid_v) * math.pi            # latitude [-π/2, π/2]

    # Spherical → unit direction vectors (OpenCV: x-right, y-down, z-forward)
    cos_phi = torch.cos(phi)
    dir_x = cos_phi * torch.sin(theta)
    dir_y = -torch.sin(phi)
    dir_z = cos_phi * torch.cos(theta)

    # Expand to [B, 1, num_layers, H, W]
    expand_shape = (batch_size, 1, num_layers, base_height, base_width)
    dir_x = dir_x[None, None, None].expand(expand_shape)
    dir_y = dir_y[None, None, None].expand(expand_shape)
    dir_z = dir_z[None, None, None].expand(expand_shape)

    return dir_x, dir_y, dir_z


def _create_base_scale_equirect(
    disparity: torch.Tensor,
    disparity_scale_factor: float,
    scale_mode: str = "linear",
    exp_growth_rate: float = 1.2,
    transition_depth: float = 100.0,
) -> torch.Tensor:
    """Create base scale for equirectangular, with cos(latitude) correction.

    In equirectangular projection, pixels near the poles cover less solid angle.
    We scale the Gaussian size by cos(φ) to account for this.

    Two depth-to-scale modes are supported:

    - ``"linear"``: scale = depth × factor.  Simple and stable, but far
      regions may show gaps because pixel spacing grows linearly with depth
      while Gaussian coverage grows at the same rate.
    - ``"exp2lin"``: power-law-then-linear (C1-continuous).
      Near range  (depth ≤ D_t): scale = depth^α × factor  (superlinear)
      Far range   (depth >  D_t): linear continuation at the tangent slope.
      This fills gaps more aggressively at mid-range without exploding at
      very large depths (1 000–10 000+).

    Args:
        disparity: Inverse depth [B, 1, num_layers, base_H, base_W].
        disparity_scale_factor: Scale factor for disparity-based sizing.
        scale_mode: ``"linear"`` or ``"exp2lin"``.
        exp_growth_rate: (exp2lin only) α – power exponent. Values > 1
            give superlinear growth (e.g. 1.2).
        transition_depth: (exp2lin only) D_t – depth where the curve
            switches from power-law to linear.

    Returns:
        Base scales [B, 1, num_layers, base_H, base_W].
    """
    device = disparity.device
    _, _, _, base_height, _ = disparity.shape

    # Compute cos(latitude) correction
    v_coords = torch.linspace(
        0.5 / base_height, 1.0 - 0.5 / base_height, base_height, device=device
    )
    phi = (0.5 - v_coords) * math.pi
    cos_phi = torch.cos(phi)  # [base_H]

    inverse_disparity = torch.ones_like(disparity) / disparity

    if scale_mode == "linear":
        base_scales = inverse_disparity * disparity_scale_factor
    elif scale_mode == "exp2lin":
        # Power-law-then-linear (C1-continuous) growth.
        # Near range  (depth <= D_t): scale = depth^α  (superlinear)
        # Far range   (depth >  D_t): linear continuation.
        # exp_growth_rate is used as the power exponent α (e.g. 1.2).
        depth_values = inverse_disparity
        alpha = exp_growth_rate

        # Power piece: f(d) = d^α
        power_piece = depth_values ** alpha

        # Value and slope at transition point for C1 continuity
        #   f(D_t)  = D_t^α
        scale_at_transition = transition_depth ** alpha

        # Linear piece: g(d) = f(D_t) + (d - D_t)
        linear_piece = (
            scale_at_transition
            + (depth_values - transition_depth)
        )

        base_scales = torch.where(
            depth_values <= transition_depth, power_piece, linear_piece
        ) * disparity_scale_factor
    else:
        raise ValueError(
            f"Unknown scale_mode '{scale_mode}'. Expected 'linear' or 'exp2lin'."
        )

    # Apply cos(φ) correction: pixels near poles are smaller
    cos_phi_map = cos_phi[None, None, None, :, None]  # [1, 1, 1, base_H, 1]
    base_scales = base_scales * cos_phi_map.clamp(min=0.1)

    return base_scales


class PanoInitializerOutput(NamedTuple):
    """Output of panorama initializer."""

    gaussian_base_values: PanoGaussianBaseValues
    feature_input: torch.Tensor
    global_scale: torch.Tensor | None = None


class PanoMultiLayerInitializer(nn.Module):
    """Initialize Gaussians from equirectangular panorama RGBD.

    Produces multi-layer base Gaussians whose 3D positions are defined by
    spherical unprojection instead of pinhole NDC.

    Output tensor layout: [B, dim, num_layers, base_H, base_W].
    """

    def __init__(
        self,
        num_layers: int = 2,
        stride: int = 2,
        base_depth: float = 10.0,
        scale_factor: float = 1.0,
        disparity_factor: float = 1.0,
        color_option: ColorInitOption = "all_layers",
        first_layer_depth_option: DepthInitOption = "surface_min",
        rest_layer_depth_option: DepthInitOption = "surface_min",
        normalize_depth: bool = True,
        feature_input_stop_grad: bool = True,
        scale_mode: str = "linear",
        exp_growth_rate: float = 1.2,
        transition_depth: float = 100.0,
    ) -> None:
        """Initialize PanoMultiLayerInitializer.

        Args:
            num_layers: Number of Gaussian layers to produce.
            stride: Spatial downsampling factor.
            base_depth: Depth of background layers (when not using surface init).
            scale_factor: Multiplier for Gaussian scale.
            disparity_factor: Factor to convert inverse depth to disparity.
            color_option: How to initialize Gaussian colors from the input image.
            first_layer_depth_option: Depth init strategy for the first layer.
            rest_layer_depth_option: Depth init strategy for remaining layers.
            normalize_depth: Whether to rescale depth for numerical stability.
            feature_input_stop_grad: Stop gradients through feature inputs.
            scale_mode: Depth-to-scale mapping: ``"linear"`` (original) or
                ``"exp2lin"`` (power-law-then-linear, better gap filling).
            exp_growth_rate: (exp2lin only) Power exponent α (e.g. 1.2).
            transition_depth: (exp2lin only) Depth where curve switches
                from power-law to linear.
        """
        super().__init__()
        self.num_layers = num_layers
        self.stride = stride
        self.base_depth = base_depth
        self.scale_factor = scale_factor
        self.disparity_factor = disparity_factor
        self.color_option = color_option
        self.first_layer_depth_option = first_layer_depth_option
        self.rest_layer_depth_option = rest_layer_depth_option
        self.normalize_depth = normalize_depth
        self.feature_input_stop_grad = feature_input_stop_grad
        self.scale_mode = scale_mode
        self.exp_growth_rate = exp_growth_rate
        self.transition_depth = transition_depth

    def prepare_feature_input(
        self, image: torch.Tensor, depth: torch.Tensor
    ) -> torch.Tensor:
        """Concatenate image and normalized disparity as feature input."""
        if self.feature_input_stop_grad:
            image = image.detach()
            depth = depth.detach()
        # depth+1: to avoid too small depth, which impact the _rescale_depth (introduce too large rescale_factor)
        depth, rescale_factor = _rescale_depth(depth + 1)
        normalized_disparity = self.disparity_factor / depth
        features_in = torch.cat([image, normalized_disparity], dim=1)
        features_in = 2.0 * features_in - 1.0
        return features_in

    def forward(
        self, image: torch.Tensor, depth: torch.Tensor
    ) -> PanoInitializerOutput:
        """Build base Gaussians from an equirectangular panorama and its depth.

        Args:
            image: [B, 3, H, W] equirectangular panorama in [0, 1].
            depth: [B, 1, H, W] metric depth map (positive, in metres).

        Returns:
            PanoInitializerOutput containing PanoGaussianBaseValues, feature
            input, and an optional global_scale for metric recovery.
        """
        image = image.contiguous()
        depth = depth.contiguous()
        device = depth.device
        batch_size, _, image_height, image_width = depth.shape
        base_height = image_height // self.stride
        base_width = image_width // self.stride

        # Optional depth rescaling for numerical stability.
        global_scale: torch.Tensor | None = None
        if self.normalize_depth:
            depth, depth_factor = _rescale_depth(depth)
            global_scale = 1.0 / depth_factor

        # ---- Build disparity layers (reuse original logic) ----
        def _create_disparity_layers(count: int = 1) -> torch.Tensor:
            disparity = torch.linspace(
                1.0 / self.base_depth, 0.0, count + 1, device=device
            )
            return disparity[None, None, :-1, None, None].repeat(
                batch_size, 1, 1, base_height, base_width
            )

        def _create_surface_layer(
            depth_slice: torch.Tensor, mode: str
        ) -> torch.Tensor:
            disparity = 1.0 / (depth_slice + 1e-6)
            if mode == "min":
                disparity = torch.max_pool2d(disparity, self.stride, self.stride)
            elif mode == "max":
                disparity = -torch.max_pool2d(-disparity, self.stride, self.stride)
            else:
                raise ValueError(f"Invalid depth pooling mode {mode}.")
            return disparity[:, :, None, :, :]

        if self.first_layer_depth_option == "surface_min":
            first_disparity = _create_surface_layer(depth[:, 0:1], "min")
        elif self.first_layer_depth_option == "surface_max":
            first_disparity = _create_surface_layer(depth[:, 0:1], "max")
        elif self.first_layer_depth_option in ("base_depth", "linear_disparity"):
            first_disparity = _create_disparity_layers()
        else:
            raise ValueError(
                f"Unknown depth init option: {self.first_layer_depth_option}."
            )

        if self.num_layers == 1:
            disparity = first_disparity
        else:
            following_depth = depth if depth.shape[1] == 1 else depth[:, 1:]
            if self.rest_layer_depth_option == "surface_min":
                following_disparity = _create_surface_layer(following_depth, "min")
            elif self.rest_layer_depth_option == "surface_max":
                following_disparity = _create_surface_layer(following_depth, "max")
            elif self.rest_layer_depth_option == "base_depth":
                following_disparity = torch.cat(
                    [_create_disparity_layers() for _ in range(self.num_layers - 1)],
                    dim=2,
                )
            elif self.rest_layer_depth_option == "linear_disparity":
                following_disparity = _create_disparity_layers(self.num_layers - 1)
            else:
                raise ValueError(
                    f"Unknown depth init option: {self.rest_layer_depth_option}."
                )
            disparity = torch.cat([first_disparity, following_disparity], dim=2)

        # ---- Spherical direction vectors ----
        dir_x, dir_y, dir_z = _create_base_directions_equirect(
            depth, self.stride, self.num_layers
        )

        # ---- Scale with cos(latitude) correction ----
        disparity_scale_factor = (
            2 * self.scale_factor * self.stride / float(image_width)
        )
        base_scales = _create_base_scale_equirect(
            disparity,
            disparity_scale_factor,
            scale_mode=self.scale_mode,
            exp_growth_rate=self.exp_growth_rate,
            transition_depth=self.transition_depth,
        )

        # ---- Quaternions, opacities, colors (same as original) ----
        base_quaternions = torch.tensor(
            [1.0, 0.0, 0.0, 0.0], device=device
        )[None, :, None, None, None]

        base_opacities = torch.tensor(
            [min(1.0 / self.num_layers, 0.5)], device=device
        )

        base_colors = torch.empty(
            batch_size, 3, self.num_layers, base_height, base_width, device=device
        ).fill_(0.5)
        if self.color_option == "none":
            pass
        elif self.color_option == "first_layer":
            base_colors[:, :, 0] = F.avg_pool2d(image, self.stride, self.stride)
        elif self.color_option == "all_layers":
            pooled = F.avg_pool2d(image, self.stride, self.stride)
            base_colors = pooled[:, :, None, :, :].repeat(
                1, 1, self.num_layers, 1, 1
            )
        else:
            raise ValueError(f"Unknown color init option: {self.color_option}.")

        features_in = self.prepare_feature_input(image, depth)

        base_gaussians = PanoGaussianBaseValues(
            dir_x=dir_x,
            dir_y=dir_y,
            dir_z=dir_z,
            inverse_depth=disparity,
            scales=base_scales,
            quaternions=base_quaternions,
            colors=base_colors,
            opacities=base_opacities,
        )

        return PanoInitializerOutput(
            gaussian_base_values=base_gaussians,
            feature_input=features_in,
            global_scale=global_scale,
        )


def create_pano_initializer(params: InitializerParams) -> nn.Module:
    """Create a PanoMultiLayerInitializer from InitializerParams."""
    return PanoMultiLayerInitializer(
        num_layers=params.num_layers,
        stride=params.stride,
        base_depth=params.base_depth,
        scale_factor=params.scale_factor,
        disparity_factor=params.disparity_factor,
        color_option=params.color_option,
        first_layer_depth_option=params.first_layer_depth_option,
        rest_layer_depth_option=params.rest_layer_depth_option,
        normalize_depth=params.normalize_depth,
        feature_input_stop_grad=params.feature_input_stop_grad,
        scale_mode=params.scale_mode,
        exp_growth_rate=params.exp_growth_rate,
        transition_depth=params.transition_depth,
    )


def _create_base_xy(
    depth: torch.Tensor, stride: int, num_layers: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create base x and y coordinates for the gaussians in NDC space."""
    device = depth.device
    batch_size, _, image_height, image_width = depth.shape
    xx = torch.arange(0.5 * stride, image_width, stride, device=device)
    yy = torch.arange(0.5 * stride, image_height, stride, device=device)
    xx = 2 * xx / image_width - 1.0
    yy = 2 * yy / image_height - 1.0

    xx, yy = torch.meshgrid(xx, yy, indexing="xy")
    base_x_ndc = xx[None, None, None].repeat(batch_size, 1, num_layers, 1, 1)
    base_y_ndc = yy[None, None, None].repeat(batch_size, 1, num_layers, 1, 1)

    return base_x_ndc, base_y_ndc


def _create_base_scale(disparity: torch.Tensor, disparity_scale_factor: float) -> torch.Tensor:
    """Create base scale for the gaussians."""
    inverse_disparity = torch.ones_like(disparity) / disparity
    base_scales = inverse_disparity * disparity_scale_factor
    return base_scales


def _rescale_depth(
    depth: torch.Tensor, depth_min: float = 1.0, depth_max: float = 1e2
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rescale a depth image tensor.

    Args:
        depth: The depth tensor to transform.
        depth_min: The min depth to scale depth to.
        depth_max: The max clamp depth after scaling.

    Returns:
        The rescaled depth and rescale factor.
    """
    current_depth_min = depth.flatten(depth.ndim - 3).min(dim=-1).values
    depth_factor = depth_min / (current_depth_min + 1e-6)
    depth = (depth * depth_factor[..., None, None, None]).clamp(max=depth_max)
    return depth, depth_factor


def pano_base_to_gaussians3d(
    base_values: PanoGaussianBaseValues,
    global_scale: torch.Tensor | None = None,
) -> Gaussians3D:
    """Convert PanoGaussianBaseValues to a flat Gaussians3D (no delta).

    This is the "zero-delta" conversion: every delta is assumed to be zero,
    so the final Gaussians are exactly the base values after spherical
    unprojection.

    Args:
        base_values: Output of PanoMultiLayerInitializer.
        global_scale: If depth was normalised, this rescales back to metric.

    Returns:
        Gaussians3D with flattened [B, N, ...] layout.
    """

    # Reference shape from inverse_depth: [B, 1, L, H, W]
    inv_depth = base_values.inverse_depth
    batch_size = inv_depth.shape[0]
    num_layers = inv_depth.shape[2]
    base_height = inv_depth.shape[3]
    base_width = inv_depth.shape[4]
    num_gaussians = num_layers * base_height * base_width

    # Spherical unprojection: position = depth * direction
    depth = 1.0 / (inv_depth + 1e-6)  # [B,1,L,H,W]

    # Broadcast direction vectors to full [B, 1, L, H, W] shape
    dir_x = base_values.dir_x.expand_as(depth)
    dir_y = base_values.dir_y.expand_as(depth)
    dir_z = base_values.dir_z.expand_as(depth)

    mean_x = depth * dir_x
    mean_y = depth * dir_y
    mean_z = depth * dir_z
    mean_vectors = torch.cat([mean_x, mean_y, mean_z], dim=1)  # [B,3,L,H,W]

    # Apply global scale to recover metric coordinates
    scales = base_values.scales  # [B, 1, L, H, W]
    if global_scale is not None:
        mean_vectors = global_scale[:, None, None, None, None] * mean_vectors
        scales = global_scale[:, None, None, None, None] * scales

    # Expand quaternions from [1, 4, 1, 1, 1] to [B, 4, L, H, W]
    quaternions_full = base_values.quaternions.expand(
        batch_size, 4, num_layers, base_height, base_width
    )

    # Colors are already [B, 3, L, H, W] — no expansion needed

    # Expand opacities to [B, 1, L, H, W] then drop channel dim.
    # base_values.opacities may be a scalar tensor [value] (shape [1])
    # or already shaped [B, 1, L, H, W].
    opacities_val = base_values.opacities
    if opacities_val.ndim < 5:
        opacities_full = opacities_val.view(1, 1, 1, 1, 1).expand(
            batch_size, 1, num_layers, base_height, base_width
        )
    else:
        opacities_full = opacities_val.expand(
            batch_size, 1, num_layers, base_height, base_width
        )

    # Flatten: [B, C, L, H, W] → permute to [B, L, H, W, C] → flatten spatial to [B, N, C]
    mean_vectors = mean_vectors.permute(0, 2, 3, 4, 1).flatten(1, 3)  # [B, N, 3]
    singular_values = scales.permute(0, 2, 3, 4, 1).flatten(1, 3)  # [B, N, 1]
    quaternions_flat = quaternions_full.permute(0, 2, 3, 4, 1).flatten(1, 3)  # [B, N, 4]
    colors = base_values.colors.permute(0, 2, 3, 4, 1).flatten(1, 3)  # [B, N, 3]
    opacities_flat = opacities_full[:, 0].flatten(1, 3)  # [B, N]

    # Expand isotropic scale from [B, N, 1] to [B, N, 3]
    if singular_values.shape[-1] == 1:
        singular_values = singular_values.expand(-1, -1, 3)

    return Gaussians3D(
        mean_vectors=mean_vectors,
        singular_values=singular_values,
        quaternions=quaternions_flat,
        colors=colors,
        opacities=opacities_flat,
    )
