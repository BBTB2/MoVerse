"""Contains different Gaussian predictors.

For licensing see accompanying LICENSE file.
Copyright (C) 2025 Apple Inc. All Rights Reserved.
"""

from __future__ import annotations

from sharp.models.monodepth import (
    create_monodepth_adaptor,
    create_monodepth_dpt,
)

from torch import nn

from .alignment import create_alignment
from .composer import GaussianComposer, PanoGaussianComposer
from .dav2_adapter import DepthAnythingV2Adapter, DPTFeatureHead
from .gaussian_decoder import create_gaussian_decoder
from .heads import DirectPredictionHead
from .initializer import create_initializer, create_pano_initializer
from .params import PredictorParams
from .predictor import RGBGaussianPredictor, PanoGaussianPredictor


def create_predictor(params: PredictorParams) -> RGBGaussianPredictor:
    """Create gaussian predictor model specified by name."""
    if params.gaussian_decoder.stride < params.initializer.stride:
        raise ValueError(
            "We donot expected gaussian_decoder has higher resolution than initializer."
        )

    scale_factor = params.gaussian_decoder.stride // params.initializer.stride
    gaussian_composer = GaussianComposer(
        delta_factor=params.delta_factor,
        min_scale=params.min_scale,
        max_scale=params.max_scale,
        color_activation_type=params.color_activation_type,
        opacity_activation_type=params.opacity_activation_type,
        color_space=params.color_space,
        scale_factor=scale_factor,
        base_scale_on_predicted_mean=params.base_scale_on_predicted_mean,
    )
    if params.num_monodepth_layers > 1 and params.initializer.num_layers != 2:
        raise KeyError("We only support num_layers = 2 when num_monodepth_layers > 1.")

    monodepth_model = create_monodepth_dpt(params.monodepth)
    monodepth_adaptor = create_monodepth_adaptor(
        monodepth_model,
        params.monodepth_adaptor,
        params.num_monodepth_layers,
        params.sorting_monodepth,
    )

    if params.num_monodepth_layers == 2:
        monodepth_adaptor.replicate_head(params.num_monodepth_layers)

    gaussian_decoder = create_gaussian_decoder(
        params.gaussian_decoder,
        dims_depth_features=monodepth_adaptor.get_feature_dims(),
    )
    initializer = create_initializer(
        params.initializer,
    )
    prediction_head = DirectPredictionHead(
        feature_dim=gaussian_decoder.dim_out, num_layers=initializer.num_layers
    )
    decoder_dim = monodepth_model.decoder.dims_decoder[-1]
    return RGBGaussianPredictor(
        init_model=initializer,
        feature_model=gaussian_decoder,
        prediction_head=prediction_head,
        monodepth_model=monodepth_adaptor,
        gaussian_composer=gaussian_composer,
        scale_map_estimator=create_alignment(params.depth_alignment, depth_decoder_dim=decoder_dim),
    )


def create_depth_anything_v2(midas_model_type="vitl", pretrained_path: str | None = None):
    from sharp.submodules.DAP.depth_anything_v2_metric.depth_anything_v2.dpt import (
        DepthAnythingV2,
    )
   
    # Pre-defined setting of the model
    model_configs = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }
    
    # Load the pretrained model of depth anything
    import os
    _sharp_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # .../sharp
    dinov3_repo_dir = os.path.join(_sharp_dir, "submodules", "DAP",
                                   "depth_anything_v2_metric", "depth_anything_v2", "dinov3")
    dinov3_arch="dinov3_vitl16"          
    dinov3_weight=""

    depth_anything = DepthAnythingV2(
        **{**model_configs[midas_model_type], 'max_depth': 1.0},
        dinov3_repo_dir=dinov3_repo_dir,
        dinov3_arch=dinov3_arch,
        dinov3_weight=dinov3_weight
    )

    if pretrained_path is not None:
        import torch
        print(f"🔹 Loading model weights from: {pretrained_path}")
        missing_keys, unexpected_keys = depth_anything.load_state_dict(torch.load(pretrained_path, map_location='cpu'))
        print("✅ Model loaded successfully.\n")
        if missing_keys:
            print("Warning: Missing keys in checkpoint:")
            for key in missing_keys:
                print(f"  - {key}")
                
        if unexpected_keys:
            print("Warning: Unexpected keys in checkpoint:")
            for key in unexpected_keys:
                print(f"  - {key}")
        depth_anything.to()
    return depth_anything

def create_pano_predictor(
    params: PredictorParams,
    depth_anything_v2: nn.Module,
) -> PanoGaussianPredictor:
    """Create panorama gaussian predictor with DepthAnythingV2 backbone.

    Args:
        params: Predictor parameters.
        depth_anything_v2: A DepthAnythingV2 instance. Will be wrapped in
            DepthAnythingV2Adapter internally.

    Returns:
        PanoGaussianPredictor ready for training.
    """

    if params.gaussian_decoder.stride < params.initializer.stride:
        raise ValueError(
            "gaussian_decoder should not have higher resolution than initializer."
        )

    scale_factor = params.gaussian_decoder.stride // params.initializer.stride

    gaussian_composer = PanoGaussianComposer(
        delta_factor=params.delta_factor,
        min_scale=params.min_scale,
        max_scale=params.max_scale,
        color_activation_type=params.color_activation_type,
        opacity_activation_type=params.opacity_activation_type,
        color_space=params.color_space,
        scale_factor=scale_factor,
        base_scale_on_predicted_mean=params.base_scale_on_predicted_mean,
    )

    initializer = create_pano_initializer(params.initializer)

    # Wrap DepthAnythingV2 as pano_depth_model
    pano_depth_model = DepthAnythingV2Adapter(depth_anything_v2, duplicate_depth_layer=(int(params.initializer.num_layers)==2))

    # Extract encoder info from DepthAnythingV2 for DPTFeatureHead.
    # DepthAnythingV2 / DPTHead lack type annotations; we use getattr() to
    # access their attributes so the type checker does not complain.
    pretrained = depth_anything_v2.pretrained
    encoder_embed_dim: int = int(getattr(pretrained, "embed_dim"))
    patch_size: int = int(getattr(pretrained, "patch_size"))
    depth_head = getattr(depth_anything_v2, "depth_head")
    use_clstoken: bool = bool(getattr(depth_head, "use_clstoken"))

    # Read out_channels from the existing depth_head's project layers
    depth_head_out_channels: list[int] = [
        int(getattr(proj, "out_channels")) for proj in depth_head.projects
    ]

    # Read the DPT fusion feature dim from scratch.refinenet (same as depth_head)
    refinenet1 = getattr(getattr(depth_head, "scratch"), "refinenet1")
    dpt_features: int = int(getattr(getattr(refinenet1, "out_conv"), "in_channels"))

    # Build DPTFeatureHead as feature_model (mirrors depth_head architecture,
    # but outputs ImageFeatures instead of 1-channel depth).
    # feature_input from PanoMultiLayerInitializer is [B, 3+num_layers, H, W]:
    # 3 RGB channels + num_layers depth channels (one per Gaussian layer).
    num_layers: int = int(params.initializer.num_layers)  # pyright: ignore[reportArgumentType]
    feature_input_channels = 3 + num_layers
    feature_model = DPTFeatureHead(
        in_channels=encoder_embed_dim,
        feature_dim=params.gaussian_decoder.dim_out,
        input_channels=feature_input_channels,
        features=dpt_features,
        out_channels=depth_head_out_channels,
        use_bn=False,
        use_clstoken=use_clstoken,
        patch_size=patch_size,
        stride=params.gaussian_decoder.stride,
    )

    prediction_head = DirectPredictionHead(
        feature_dim=params.gaussian_decoder.dim_out,
        num_layers=num_layers,
    )

    # Depth alignment (optional)
    scale_map_estimator = None
    if params.depth_alignment is not None:
        print("✅ Depth alignment enabled.")
        scale_map_estimator = create_alignment(
            params.depth_alignment,
            depth_decoder_dim=dpt_features,
        )

    return PanoGaussianPredictor(
        pano_depth_model=pano_depth_model,
        init_model=initializer,
        feature_model=feature_model,
        prediction_head=prediction_head,
        gaussian_composer=gaussian_composer,
        scale_map_estimator=scale_map_estimator,
    )


__all__ = [
    "PredictorParams",
    "create_predictor",
]
