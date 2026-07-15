"""Contains definition of RGB-only gaussian predictor.

For licensing see accompanying LICENSE file.
Copyright (C) 2025 Apple Inc. All Rights Reserved.
"""

from __future__ import annotations

import logging
from typing import NamedTuple

import torch
from torch import nn

from sharp.models.dav2_adapter import DepthAnythingV2Output
from sharp.models.monodepth import MonodepthWithEncodingAdaptor
from sharp.utils.gaussians import Gaussians3D

from .composer import GaussianComposer

LOGGER = logging.getLogger(__name__)


class PanoPredictorOutput(NamedTuple):
    """Output of PanoGaussianPredictor, bundling Gaussians with delta values."""

    gaussians: Gaussians3D
    delta_values: torch.Tensor
    predicted_depth: torch.Tensor | None = None


class DepthAlignment(nn.Module):
    """Depth alignment in a dedicated nn.Module.

    Wrap scale_map_estimator to perform the conditional logic in a separated torch
    module outside the forward of RGBGaussianPredictor. This module can be then
    excluded during symbolic tracing.
    """

    def __init__(self, scale_map_estimator: nn.Module | None):
        """Initialize DepthAlignmentWrapper.

        Args:
            scale_map_estimator: Module to align monodepth to ground truth depth.
        """
        super().__init__()
        self.scale_map_estimator = scale_map_estimator

    def forward(
        self,
        monodepth: torch.Tensor,
        depth: torch.Tensor,
        depth_decoder_features: torch.Tensor | None = None,
    ):
        """Optionally align monodepth to ground truth with a local scale map.

        Args:
            monodepth: The monodepth model with intermediate features to use.
            depth: Ground truth depth to align predicted depth to.
            depth_decoder_features: The (optional) monodepth decoder features.
        """
        if depth is not None and self.scale_map_estimator is not None:
            depth_alignment_map = self.scale_map_estimator(
                monodepth[:, 0:1], depth, depth_decoder_features
            )
            monodepth = depth_alignment_map * monodepth
        else:
            # Some losses rely on the presence of an alignment map.
            # We ensure that they can be computed by creating a fake alignment map.
            depth_alignment_map = torch.ones_like(monodepth)
        return monodepth, depth_alignment_map


class RGBGaussianPredictor(nn.Module):
    """Predicts 3D Gaussians from images."""

    feature_model: nn.Module

    def __init__(
        self,
        init_model: nn.Module,
        monodepth_model: MonodepthWithEncodingAdaptor,
        feature_model: nn.Module,
        prediction_head: nn.Module,
        gaussian_composer: GaussianComposer,
        scale_map_estimator: nn.Module | None,
    ) -> None:
        """Initialize RGBGaussianPredictor.

        Args:
            init_model: A model mapping image and depth to base values.
            monodepth_model: The monodepth model with intermediate features to use.
            feature_model: The image2image model to predict Gaussians from.
            prediction_head: Head to decode image features.
            gaussian_composer: Module to compose final prediction from deltas and
                base values.
            scale_map_estimator: Module to align monodepth to ground truth depth.

        Note:
        ----
            when monodepth_model is trainable, using local depth alignment can
            result in the monodepth model losing its ability to predict shapes. It is
            hence recommend to deactivate the corresponding flag.
        """
        super().__init__()
        self.init_model = init_model
        self.feature_model = feature_model
        self.monodepth_model = monodepth_model
        self.prediction_head = prediction_head
        self.gaussian_composer = gaussian_composer
        self.depth_alignment = DepthAlignment(scale_map_estimator)

    def forward(
        self,
        image: torch.Tensor,
        disparity_factor: torch.Tensor,
        depth: torch.Tensor | None = None,
    ) -> Gaussians3D:
        """Predict 3D Gaussians.

        Args:
            image: The image to process.
            disparity_factor: Factor to convert depth to disparities.
            depth: Ground truth depth to align predicted depth to.

        Returns:
            The predicted 3D Gaussians.

        Note:
        ----
        During training, it is recommended to feed an additional ground truth depth
        map to the network to align the predicted depth to. During inference, it is
        recommended to use depth_gt=None and use monodepth_disparity output from the
        model instead to compute depth.
        """
        # Estimate depth and align to ground truth (if available).
        monodepth_output = self.monodepth_model(image)
        monodepth_disparity = monodepth_output.disparity

        disparity_factor = disparity_factor[:, None, None, None]
        monodepth = disparity_factor / monodepth_disparity.clamp(min=1e-4, max=1e4)

        # In the model we apply additional alignment to provided ground truth depth
        # as well as additional normalization.
        #
        # The overall graph looks as follows:
        #
        #     monodepth        depth    # Both monodepth and depth are metric here.
        #         |              |
        #         +------+-------+
        #                |
        #        +-------+--------+     # Optionally align monodepth to ground truth
        #        |depth_alignement|     # with a local scale map.
        #        +-------+--------+
        #                |
        #                v
        #       monodepth (aligned)     # Monodepth is now aligned to ground truth.
        #                |
        #          +-----+----+         # Normalize depth and compute base gaussians.
        #          |init_model|         # in these normalized coordinates.
        #          +-----+----+
        #                |
        #                v
        #   +------ init_output         # Init_output consists of features, base
        #   |            |              # gaussians and a global scale.
        #   |     +------+-----+
        #   |     |main network|        # Compute delta values to base gaussians.
        #   |     +------+-----+
        #   |            |
        #   |            V
        #   |        delta_values       # The delta values are computed with normalized depth.
        #   |            |
        #   |    +-------+---------+
        #   +--> |gaussian_composer|    # Add delta to base values and unscale gaussians.
        #        +-------+---------+
        #                |
        #                v
        #            gaussians          # The final Gaussians are metric again.
        #

        # The logic to decide whether to align monodepth to the ground truth is wrapped
        # in a submodule 'DepthAlignement' to facilitate the symbolic tracing of the
        # predictor. This way, the depth alignment submodule containing the conditional
        # logic can be excluded during the tracing and the graph of the predictors is
        # static.
        monodepth, _ = self.depth_alignment(
            monodepth,
            depth,
            monodepth_output.decoder_features,
        )

        init_output = self.init_model(image, monodepth)
        image_features = self.feature_model(
            init_output.feature_input, encodings=monodepth_output.output_features
        )
        delta_values = self.prediction_head(image_features)
        gaussians = self.gaussian_composer(
            delta=delta_values,
            base_values=init_output.gaussian_base_values,
            global_scale=init_output.global_scale,
        )
        return gaussians

    def internal_resolution(self) -> int:
        """Internal resolution."""
        return self.monodepth_model.internal_resolution()

    @property
    def output_resolution(self) -> int:
        """Output resolution of Gaussians."""
        return self.internal_resolution() // 2


class PanoGaussianPredictor(nn.Module):
    """Predicts 3D Gaussians from equirectangular panorama images.

    Pipeline:
        pano_image → pano_depth_model → depth
        depth → depth_alignment (optional, with GT depth)
        (image, depth) → pano_initializer → base Gaussians (spherical)
        (features) → feature_model → delta values
        (base + delta) → pano_composer → 3DGS (metric)
    """

    def __init__(
        self,
        pano_depth_model: nn.Module, # DepthAnythingV2Adapter
        init_model: nn.Module,  # PanoMultiLayerInitializer
        feature_model: nn.Module,  # DPTFeatureHead
        prediction_head: nn.Module,  # DirectPredictionHead
        gaussian_composer: nn.Module,  # PanoGaussianComposer
        scale_map_estimator: nn.Module | None,
    ) -> None:
        """Initialize PanoGaussianPredictor.

        Args:
            pano_depth_model: Your panorama depth estimation model.
                Must accept [B, 3, H, W] and return depth [B, 1, H, W].
            init_model: PanoMultiLayerInitializer.
            feature_model: GaussianDensePredictionTransformer (reuses depth encoder features).
            prediction_head: DirectPredictionHead.
            gaussian_composer: PanoGaussianComposer.
            scale_map_estimator: Optional depth alignment module.
        """
        super().__init__()
        self.pano_depth_model = pano_depth_model
        self.init_model = init_model
        self.feature_model = feature_model
        self.prediction_head = prediction_head
        self.gaussian_composer = gaussian_composer
        self.depth_alignment = DepthAlignment(scale_map_estimator)

    def forward(
        self,
        image: torch.Tensor,
        depth_gt: torch.Tensor | None = None,
        infer_depth: bool = False,
        dap_scale_factor: float = 1.0,
    ) -> PanoPredictorOutput:
        """Predict 3D Gaussians from equirectangular panorama.

        Args:
            image: Equirectangular panorama [B, 3, H, W], values in [0, 1].
            depth_gt: Optional GT depth [B, 1, H, W] for alignment during training.

        Returns:
            PanoPredictorOutput containing the predicted 3D Gaussians and raw
            delta values (for regularization losses).

        Note:
            Unlike RGBGaussianPredictor, no disparity_factor is needed because
            equirectangular projection doesn't have focal length.

        Pipeline:
            image → pano_depth_model → depth
                                         ↓
            (image, depth) → init_model → base_gaussians + features
                                         ↓
            features → feature_model → delta_values
                                         ↓
            (base + delta) → composer → 3DGS
        """
        if not infer_depth and depth_gt is None:
            raise ValueError("depth_gt is required when infer_depth is False.")

        # Step 1: Predict depth from panorama via DepthAnythingV2Adapter
        depth_output: DepthAnythingV2Output = self.pano_depth_model(image, infer_depth)
        encoder_features = depth_output.encoder_features

        # Step 2: Determine depth to use and optionally align to GT
        # Keep raw predicted depth for direct supervision loss.
        raw_predicted_depth = depth_output.depth  # None when infer_depth=False
        if infer_depth:
            raw_predicted_depth = raw_predicted_depth * dap_scale_factor

            depth_gt_align = depth_gt + 1 if depth_gt is not None else None
            depth, _ = self.depth_alignment(raw_predicted_depth + 1, depth_gt_align, encoder_features[0])
        else:
            depth = depth_gt

        # Step 3: Initialize base Gaussians (spherical)
        init_output = self.init_model(image, depth)

        # Step 4: Predict delta values using DPTFeatureHead with shared encoder features
        # init_output.feature_input shape: [B, 3+#_Depth_Channels, H, W], e.g., [B, 4, H, W]
        image_features = self.feature_model(
            init_output.feature_input, encodings=encoder_features
        )

        delta_values = self.prediction_head(image_features)

        # Step 5: Compose final Gaussians (spherical unprojection)
        gaussians = self.gaussian_composer(
            delta=delta_values,
            base_values=init_output.gaussian_base_values,
            global_scale=init_output.global_scale,
        )
        return PanoPredictorOutput(
            gaussians=gaussians,
            delta_values=delta_values,
            predicted_depth=raw_predicted_depth,
        )