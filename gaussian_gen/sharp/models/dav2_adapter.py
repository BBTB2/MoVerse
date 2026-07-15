"""Adapters to integrate DepthAnythingV2 into PanoGaussianPredictor.

Provides:
    - DepthAnythingV2Adapter: wraps DepthAnythingV2 as pano_depth_model,
      returning depth + encoder_features.
    - DPTFeatureHead: a DPTHead-like decoder that outputs ImageFeatures
      (texture + geometry) instead of a 1-channel depth map, so that
      DirectPredictionHead can consume the result.

For licensing see accompanying LICENSE file.
Copyright (C) 2025 Apple Inc. All Rights Reserved.
"""

from __future__ import annotations

from typing import NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from sharp.models.blocks import FeatureFusionBlock2d
from sharp.models.gaussian_decoder import ImageFeatures

from sharp.submodules.DAP.depth_anything_v2_metric.depth_anything_v2.dpt import (
    DPTHead,
    DepthAnythingV2,
    _make_fusion_block,
)
from sharp.submodules.DAP.depth_anything_v2_metric.depth_anything_v2.util.blocks import (
    _make_scratch,
)


# ---------------------------------------------------------------------------
# 1. DepthAnythingV2Adapter  –  pano_depth_model
# ---------------------------------------------------------------------------

class DepthAnythingV2Output(NamedTuple):
    """Structured output of DepthAnythingV2Adapter."""

    depth: torch.Tensor | None
    """Metric depth map [B, 1, H, W]."""

    mask: torch.Tensor | None
    """Predicted mask [B, 1, H, W]."""

    encoder_features: list[torch.Tensor]
    """Intermediate encoder features (4 levels), each [B, C, h, w].
    These are the raw patch-map outputs from DINOv3 *before* the DPT head."""


class DepthAnythingV2Adapter(nn.Module):
    """Wraps DepthAnythingV2 so it can serve as ``pano_depth_model``.

    The original ``DepthAnythingV2.forward`` runs the encoder and both heads
    but does **not** expose the intermediate encoder features.  This adapter
    splits the forward pass so that the encoder features are returned alongside
    the depth prediction, enabling downstream modules (e.g. ``DPTFeatureHead``)
    to reuse them without a redundant encoder pass.
    """

    def __init__(
        self,
        depth_anything_v2: DepthAnythingV2,
        duplicate_depth_layer: bool = False,
    ) -> None:
        """Initialize DepthAnythingV2Adapter.

        Args:
            depth_anything_v2: The wrapped DepthAnythingV2 model.
            duplicate_depth_layer: If True, duplicate the final Conv2d in
                ``depth_head.scratch.output_conv2`` so it outputs 2 depth
                channels instead of 1.  The duplicated layer's weights are
                copied from the original single-channel layer.
        """
        super().__init__()
        self.model = depth_anything_v2
        self.duplicate_depth_layer = duplicate_depth_layer

        if duplicate_depth_layer:
            self._duplicate_final_conv(self.model.depth_head)
            print('Depth head: ', self.model.depth_head.scratch.output_conv2)

    @staticmethod
    def _duplicate_final_conv(depth_head: nn.Module) -> None:
        """Replace the last Conv2d(in, 1) in output_conv2 with Conv2d(in, 2).

        The new layer's two output channels are initialised with identical
        copies of the original single-channel weights and bias.
        """
        output_conv2 = depth_head.scratch.output_conv2
        # Find the last Conv2d in the Sequential
        last_conv_index = None
        for idx in reversed(range(len(output_conv2))):
            if isinstance(output_conv2[idx], nn.Conv2d):
                last_conv_index = idx
                break
        if last_conv_index is None:
            raise RuntimeError(
                "Could not find a Conv2d layer in depth_head.scratch.output_conv2"
            )

        original_conv = output_conv2[last_conv_index]
        in_channels = original_conv.in_channels
        kernel_size = original_conv.kernel_size
        stride = original_conv.stride
        padding = original_conv.padding
        has_bias = original_conv.bias is not None

        # Create new Conv2d with 2 output channels
        new_conv = nn.Conv2d(
            in_channels,
            2,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=has_bias,
        )

        # Copy weights: duplicate the single output channel to both channels
        with torch.no_grad():
            new_conv.weight[0] = original_conv.weight[0].clone()
            new_conv.weight[1] = original_conv.weight[0].clone()
            if has_bias:
                new_conv.bias[0] = original_conv.bias[0].clone()
                new_conv.bias[1] = original_conv.bias[0].clone()

        output_conv2[last_conv_index] = new_conv

    def forward(self, image: torch.Tensor, infer_depth=True) -> DepthAnythingV2Output:
        """Run encoder once, then depth_head and mask_head.

        Args:
            image: Input image [B, 3, H, W].

        Returns:
            DepthAnythingV2Output with depth, mask, and encoder_features.
        """
        patch_size = int(getattr(self.model.pretrained, "patch_size", 16))
        patch_h = image.shape[-2] // patch_size # number of patches in height
        patch_w = image.shape[-1] // patch_size # number of patches in width

        # Encoder: get intermediate features (without class tokens)
        raw_features = self.model.pretrained.get_intermediate_layers(
            image,
            self.model.intermediate_layer_idx[self.model.encoder],
            return_class_token=False,
        )

        # Extract the spatial patch maps for downstream reuse.
        # raw_features is a tuple of n patch_maps [B,C,h,w]. 
        # h = patch_h, w = patch_w
        encoder_features = list(raw_features) 

        # Depth & mask heads (they accept the raw features with cls tokens)
        if infer_depth:
            depth = self.model.depth_head(raw_features, patch_h, patch_w, patch_size) * self.model.max_depth
            mask = self.model.mask_head(raw_features, patch_h, patch_w, patch_size)

            # Ensure [B, C, H, W] shape (C=1 normally, C=2 when duplicate_depth_layer)
            if depth.dim() == 3:
                depth = depth.unsqueeze(1)
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)
        else:
            depth = None
            mask = None

        return DepthAnythingV2Output(
            depth=depth,
            mask=mask,
            encoder_features=encoder_features,
        )


# ---------------------------------------------------------------------------
# 2. DPTFeatureHead  –  feature_model
# ---------------------------------------------------------------------------

class DPTFeatureHead(nn.Module):
    """A DPTHead-like decoder that outputs ``ImageFeatures`` for DirectPredictionHead.

    Architecture mirrors ``DPTHead`` (project → resize → scratch/refinenet
    fusion), but replaces the final depth output layers with two separate
    heads that produce *texture_features* and *geometry_features* at the
    original image resolution.

    The ``forward`` signature matches ``GaussianDensePredictionTransformer``:
        ``forward(input_features, encodings=encoder_features) → ImageFeatures``
    so it can be used as a drop-in ``feature_model`` in PanoGaussianPredictor.
    """

    def __init__(
        self,
        in_channels: int,
        feature_dim: int,
        input_channels: int = 4,
        features: int = 256,
        out_channels: list[int] = [256, 512, 1024, 1024],
        use_bn: bool = False,
        use_clstoken: bool = False,
        patch_size: int = 16,
        stride: int = 2,
    ) -> None:
        """Initialize DPTFeatureHead.

        Args:
            in_channels: Channel dim of each encoder feature map (embed_dim).
            feature_dim: Output channel dim for texture/geometry features.
                Must match ``DirectPredictionHead.feature_dim``.
            input_channels: Number of channels in input_features (e.g. 4 for
                RGB + depth).
            features: Internal feature dim of the DPT fusion blocks.
            out_channels: Per-level projection channels (default [256,512,1024,1024]).
            use_bn: Whether to use BatchNorm in fusion blocks.
            use_clstoken: Whether encoder features include class tokens.
            patch_size: Patch size of the ViT encoder (for upsampling).
            stride: Spatial downsampling stride for the output feature maps.
                The output resolution will be (H / stride, W / stride) relative
                to the original image.
        """
        super().__init__()

        self.use_clstoken = use_clstoken
        self.in_channels = in_channels
        self.feature_dim = feature_dim
        self.patch_size = patch_size
        self.stride = stride

        # ---- Projection & resize (same as DPTHead) ----
        self.projects = nn.ModuleList([
            nn.Conv2d(in_channels, oc, kernel_size=1, stride=1, padding=0)
            for oc in out_channels
        ])

        self.resize_layers = nn.ModuleList([
            nn.ConvTranspose2d(out_channels[0], out_channels[0], kernel_size=4, stride=4, padding=0),
            nn.ConvTranspose2d(out_channels[1], out_channels[1], kernel_size=2, stride=2, padding=0),
            nn.Identity(),
            nn.Conv2d(out_channels[3], out_channels[3], kernel_size=3, stride=2, padding=1),
        ])

        if use_clstoken:
            self.readout_projects = nn.ModuleList([
                nn.Sequential(nn.Linear(2 * in_channels, in_channels), nn.GELU())
                for _ in range(len(self.projects))
            ])

        # ---- Scratch / refinenet fusion (same as DPTHead) ----
        self.scratch = _make_scratch(out_channels, features, groups=1, expand=False)
        self.scratch.stem_transpose = None
        self.scratch.refinenet1 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet2 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet3 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet4 = _make_fusion_block(features, use_bn)

        # ---- Output heads: texture & geometry (replaces depth output_conv) ----
        # output_conv1 halves the channel dim, then we upsample to full resolution,
        # and finally project to feature_dim via separate texture/geometry heads.
        half_features = features // 2
        self.output_conv1 = nn.Conv2d(features, half_features, kernel_size=3, stride=1, padding=1)

        # ---- Input encoder: extract features from input_features (RGB+depth) ----
        # A lightweight conv encoder that maps input_channels → half_features,
        # matching the channel dim of the shared DPT output so they can be fused.
        # The first conv uses kernel_size=stride and stride=stride to downsample
        # the input to match the base Gaussian resolution.
        self.input_encoder = nn.Sequential(
            nn.Conv2d(input_channels, half_features // 2, kernel_size=stride, stride=stride, padding=0),
            nn.ReLU(),
            nn.Conv2d(half_features // 2, half_features, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )

        # ---- Fusion: merge shared DPT features with encoded input features ----
        self.input_fusion = FeatureFusionBlock2d(half_features)

        self.texture_head = nn.Sequential(
            nn.Conv2d(half_features, half_features, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(half_features, feature_dim, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
        )

        self.geometry_head = nn.Sequential(
            nn.Conv2d(half_features, half_features, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(half_features, feature_dim, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
        )

    def _process_encoder_features(
        self,
        encoder_features: list[torch.Tensor],
        patch_h: int,
        patch_w: int,
    ) -> list[torch.Tensor]:
        """Project and resize encoder features (mirrors DPTHead logic)."""
        out = []
        for i, feature_map in enumerate(encoder_features):
            # encoder_features are already spatial [B, C, h, w] from the adapter
            if feature_map.dim() == 3:
                feature_map = feature_map.permute(0, 2, 1).reshape(
                    feature_map.shape[0], feature_map.shape[-1], patch_h, patch_w
                )
            elif feature_map.dim() == 4:
                if feature_map.shape[-2] != patch_h or feature_map.shape[-1] != patch_w:
                    feature_map = F.interpolate(
                        feature_map, size=(patch_h, patch_w), mode="bilinear", align_corners=True
                    )

            feature_map = self.projects[i](feature_map)
            feature_map = self.resize_layers[i](feature_map)
            out.append(feature_map)
        return out

    def forward(
        self,
        input_features: torch.Tensor,
        encodings: list[torch.Tensor],
    ) -> ImageFeatures:
        """Decode encoder features into texture & geometry feature maps.

        Args:
            input_features: Feature input from the initializer (currently unused
                by this head, kept for API compatibility with
                GaussianDensePredictionTransformer).
            encodings: Encoder features from DepthAnythingV2Adapter
                (4 levels, each [B, C, h, w]).

        Returns:
            ImageFeatures with texture_features and geometry_features,
            both [B, feature_dim, H, W].
        """

        # Infer patch grid size from the first encoder feature map
        patch_h, patch_w = encodings[0].shape[-2], encodings[0].shape[-1]

        # Project + resize + refinenet fusion
        projected = self._process_encoder_features(encodings, patch_h, patch_w)
        layer_1, layer_2, layer_3, layer_4 = projected

        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)

        path_4 = self.scratch.refinenet4(layer_4_rn, size=layer_3_rn.shape[2:])
        path_3 = self.scratch.refinenet3(path_4, layer_3_rn, size=layer_2_rn.shape[2:])
        path_2 = self.scratch.refinenet2(path_3, layer_2_rn, size=layer_1_rn.shape[2:])
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn)

        # Shared stem: halve channels
        shared = self.output_conv1(path_1) 

        # Upsample to the strided output resolution (H/stride, W/stride)
        # instead of full image resolution, so that the output matches the
        # spatial dimensions of the base Gaussian values.
        input_h, input_w = input_features.shape[-2], input_features.shape[-1]
        target_h = input_h // self.stride
        target_w = input_w // self.stride
        shared = F.interpolate(
            shared, size=(target_h, target_w), mode="bilinear", align_corners=True
        ) # [B, half_features, H/stride, W/stride]

        # Encode input_features (RGB+depth); input_encoder downsamples by stride
        encoded_input = self.input_encoder(input_features)
        fused = self.input_fusion(shared, encoded_input)

        # Separate texture / geometry heads
        texture_features = self.texture_head(fused)
        geometry_features = self.geometry_head(fused)

        return ImageFeatures(
            texture_features=texture_features,
            geometry_features=geometry_features,
        )
