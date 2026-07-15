import types
from typing import List, Optional
import torch
from torch import nn
import os

from utils.scheduler import SchedulerInterface, FlowMatchScheduler
from wan.modules.tokenizers import HuggingfaceTokenizer
from wan.modules.model import WanModel, RegisterTokens, GanAttentionBlock
from wan.modules.vae import _video_vae
from wan.modules.t5 import umt5_xxl
from wan.modules.causal_model import CausalWanModel
from wan.modules.causal_mmdit_model import CausalMMDiTModel



class WanTextEncoder(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.text_encoder = umt5_xxl(
            encoder_only=True,
            return_tokenizer=False,
            dtype=torch.float32,
            device=torch.device('cpu')
        ).eval().requires_grad_(False)
        self.text_encoder.load_state_dict(
            torch.load(os.path.join("checkpoints/Wan2.1-T2V-1.3B", "models_t5_umt5-xxl-enc-bf16.pth"),
                       map_location='cpu', weights_only=False)
        )

        self.tokenizer = HuggingfaceTokenizer(
            name=os.path.join("checkpoints/Wan2.1-T2V-1.3B", "google/umt5-xxl"), seq_len=512, clean='whitespace')

    @property
    def device(self):
        # Assume we are always on GPU
        return torch.cuda.current_device()

    def forward(self, text_prompts: List[str]) -> dict:
        ids, mask = self.tokenizer(
            text_prompts, return_mask=True, add_special_tokens=True)
        ids = ids.to(self.device)
        mask = mask.to(self.device)
        seq_lens = mask.gt(0).sum(dim=1).long()
        context = self.text_encoder(ids, mask)

        for u, v in zip(context, seq_lens):
            u[v:] = 0.0  # set padding to 0.0

        return {
            "prompt_embeds": context
        }


class WanVAEWrapper(torch.nn.Module):
    def __init__(self):
        super().__init__()
        mean = [
            -0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508,
            0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921
        ]
        std = [
            2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743,
            3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.9160
        ]
        self.mean = torch.tensor(mean, dtype=torch.float32)
        self.std = torch.tensor(std, dtype=torch.float32)

        vae_path = os.path.join("checkpoints/Wan2.1-T2V-1.3B", "Wan2.1_VAE.pth")
        print(f"Using VAE path: {vae_path}")

        # init model
        self.model = _video_vae(
            pretrained_path=vae_path,
            z_dim=16,
        ).eval().requires_grad_(False)

    def encode_to_latent(self, pixel: torch.Tensor) -> torch.Tensor:
        # pixel: [batch_size, num_channels, num_frames, height, width]
        device, dtype = pixel.device, pixel.dtype
        scale = [self.mean.to(device=device, dtype=dtype),
                 1.0 / self.std.to(device=device, dtype=dtype)]

        output = [
            self.model.encode(u.unsqueeze(0), scale).float().squeeze(0)
            for u in pixel
        ]
        output = torch.stack(output, dim=0)
        # from [batch_size, num_channels, num_frames, height, width]
        # to [batch_size, num_frames, num_channels, height, width]
        output = output.permute(0, 2, 1, 3, 4)
        return output

    def encode_to_latent_cached(self, pixel: torch.Tensor) -> torch.Tensor:
        """Streaming incremental encode: does not clear any VAE caches.

        Unlike encode_to_latent() which calls model.encode() (clears caches),
        this calls model.cached_encode() to preserve temporal encoder context
        across streaming calls.

        Frame requirements:
          - First call: pixel T dimension = 1 (first frame)
          - Subsequent calls: pixel T dimension = 4 (4 frames -> 1 latent)

        Args:
            pixel: [B, C, T, H, W] pixel frames (T=1 first frame or T=4 subsequent)

        Returns:
            latent: [B, 1, z_dim, H_lat, W_lat] always T=1 latent
        """
        device, dtype = pixel.device, pixel.dtype
        scale = [self.mean.to(device=device, dtype=dtype),
                 1.0 / self.std.to(device=device, dtype=dtype)]

        output = self.model.cached_encode(pixel, scale).float()
        # from [B, C, T, H, W] to [B, T, C, H, W]
        output = output.permute(0, 2, 1, 3, 4)
        return output

    def decode_to_pixel(self, latent: torch.Tensor, use_cache: bool = False) -> torch.Tensor:
        # from [batch_size, num_frames, num_channels, height, width]
        # to [batch_size, num_channels, num_frames, height, width]
        zs = latent.permute(0, 2, 1, 3, 4)
        if use_cache:
            assert latent.shape[0] == 1, "Batch size must be 1 when using cache"

        device, dtype = latent.device, latent.dtype
        scale = [self.mean.to(device=device, dtype=dtype),
                 1.0 / self.std.to(device=device, dtype=dtype)]

        if use_cache:
            decode_function = self.model.cached_decode
        else:
            decode_function = self.model.decode

        output = []
        for u in zs:
            output.append(decode_function(u.unsqueeze(0), scale).float().clamp_(-1, 1).squeeze(0))
        output = torch.stack(output, dim=0)
        # from [batch_size, num_channels, num_frames, height, width]
        # to [batch_size, num_frames, num_channels, height, width]
        output = output.permute(0, 2, 1, 3, 4)
        return output


class WanDiffusionWrapper(torch.nn.Module):
    def __init__(
            self,
            model_name="Wan2.1-T2V-1.3B",
            timestep_shift=8.0,
            is_causal=False,
            local_attn_size=-1,
            sink_size=0,
            token_concat_and_share_rope=False,
            recam_concat=False,
            skip_pretrained=False,
            is_mmdit=False,
            double_stream_layers=None,
            cross_attn_norm=True,
            use_flash_attn=False,
            fixed_timestep=-1,
            online_rope_indexing=False,
            memrope=None,
            rope_precision="fp64",
    ):
        """
        Args:
            token_concat_and_share_rope (bool): Usually supplied through config kwarg.
                When True, injects a shared-height RoPE variant into every attention
                block via set_rope_fn() so that height-doubled inputs (noisy | cond
                concatenated along H) receive shared positional embeddings. Must be
                set to True when passing cond_latent to forward() in height-concat mode.
                Has no effect when recam_concat=True.
            recam_concat (bool): When True, uses ReCamMaster-style frame-dimension
                concatenation: cond_latent is concatenated with the noisy input along
                the frame dimension (dim=2 in C-first layout), producing a
                [B, C, 2F, H, W] input to the DiT. Standard RoPE is used (frames
                0..2F-1 each receive their own position), matching rope_shared_frame=False
                in the original ReCamMaster pipeline. The output is sliced back to the
                first F frames (the denoised prediction). No rope manipulation is applied.
                When False (default), height-concat mode is used if cond_latent is given.
            skip_pretrained (bool): When True, initialise the model architecture from
                the saved config only (no weights loaded from disk). Use this when a
                checkpoint will be loaded on top immediately after construction, to
                avoid the cost of loading pretrained weights that would be overwritten.
            is_mmdit (bool): When True, use CausalMMDiTModel as the generator.
                cond_latent is passed as a separate stream (no height-concat).
                Requires is_causal=True.  frame_seq_length stays at 1560 (no doubling).
            double_stream_layers (list[int], optional): Block indices for dual-stream
                processing in CausalMMDiTModel.  Forwarded to CausalMMDiTModel constructor.
                Default: [0, 1, 4, 8, 12, 16, 20, 24, 28, 29].
            cross_attn_norm (bool): Whether to use WanLayerNorm (with learnable weight/bias)
                before cross-attention (norm3 / cond_norm3).  Wan2.1-T2V-1.3B was trained
                with cross_attn_norm=True, so checkpoints initialised from it need this True.
                Default: False (matches CausalMMDiTModel default; set True for Wan 1.3B).
            use_flash_attn (bool): Whether to use FlashAttention in tf/hybrid forcing since flex attention is not available on B300 on PPU.
                Default: False.
            fixed_timestep (float): If >= 0, override the scheduler timestep with this fixed value at every forward pass.
        """
        super().__init__()

        if model_name == "Wan2.1-T2V-1.3B":
            model_path = "checkpoints/Wan2.1-T2V-1.3B/"
            print(f"Using model path: {model_path}")
        elif model_name == "Wan2.1-T2V-14B":
            model_path = "checkpoints/Wan2.1-T2V-14B/"

        self.is_mmdit = is_mmdit and is_causal

        if skip_pretrained:
            # Load architecture config only — no pretrained weights.
            # The caller is responsible for loading a checkpoint afterwards.
            if self.is_mmdit:
                _double = double_stream_layers if double_stream_layers is not None else [0, 1, 4, 8, 12, 16, 20, 24, 28, 29]
                self.model = CausalMMDiTModel(
                    local_attn_size=local_attn_size, sink_size=sink_size,
                    double_stream_layers=_double, cross_attn_norm=cross_attn_norm, use_flash_attn=use_flash_attn)
            elif is_causal:
                cfg = CausalWanModel.load_config(model_path)
                self.model = CausalWanModel.from_config(
                    cfg, local_attn_size=local_attn_size, sink_size=sink_size, use_flash_attn=use_flash_attn)
            else:
                cfg = WanModel.load_config(model_path)
                self.model = WanModel.from_config(cfg)
            print(f"Initialised {model_name} architecture only (skip_pretrained=True).")
        else:
            if self.is_mmdit:
                _double = double_stream_layers if double_stream_layers is not None else [0, 1, 4, 8, 12, 16, 20, 24, 28, 29]
                # Build from pretrained CausalWanModel, then load weights via MMDiT loader
                self.model = CausalMMDiTModel(
                    local_attn_size=local_attn_size, sink_size=sink_size,
                    double_stream_layers=_double, cross_attn_norm=cross_attn_norm, use_flash_attn=use_flash_attn)
                print(f"CausalMMDiTModel initialised (weights must be loaded separately via load_causal_mmdit_weights).")
            elif is_causal:
                self.model = CausalWanModel.from_pretrained(model_path, local_attn_size=local_attn_size, sink_size=sink_size, use_flash_attn=use_flash_attn)
            else:
                self.model = WanModel.from_pretrained(model_path)
        self.model.eval()

        # For non-causal diffusion, all frames share the same timestep
        self.uniform_timestep = not is_causal
        self.fixed_timestep = fixed_timestep

        self.scheduler = FlowMatchScheduler(
            shift=timestep_shift, sigma_min=0.0, extra_one_step=True
        )
        self.scheduler.set_timesteps(1000, training=True)

        # Default seq_len for [B, F=21, C=16, H=60, W=104]; recomputed dynamically
        # when cond_latent is provided (height doubles → seq_len doubles).
        self.seq_len = 32760  # [1, 21, 16, 60, 104]

        self.recam_concat = recam_concat

        # Inject shared-height RoPE variants as instance-level dependencies so
        # that different model instances remain isolated (no module-level patching).
        # recam_concat uses standard RoPE over the doubled frame sequence — no injection needed.
        if token_concat_and_share_rope and not recam_concat:
            from wan.modules.causal_model import causal_rope_apply_shared_height
            from wan.modules.model import rope_apply_shared
            rope_fn = rope_apply_shared
            if is_causal:
                causal_rope_fn = causal_rope_apply_shared_height
                self.model.set_rope_fn(
                    rope_fn=rope_fn,
                    causal_rope_fn=causal_rope_fn,
                )
            else:
                self.model.set_rope_fn(rope_fn=rope_fn)
            print(f"Applied shared-height RoPE variant via set_rope_fn().")
        elif recam_concat:
            print("recam_concat=True: frame-dim concat, standard RoPE (no injection).")

        # MemRoPE Phase 1 — Online RoPE Indexing (inference-only; height-concat causal).
        # Re-indexes RoPE block-relative in the AR/KV-cache path so indices stay within the
        # trained range for >1024-frame rollouts. No effect on training (kv_cache=None).
        self.online_rope_indexing = online_rope_indexing
        if online_rope_indexing:
            if is_causal and not self.is_mmdit and hasattr(self.model, "set_online_rope"):
                self.model.set_online_rope(True)
                print("Enabled MemRoPE Online RoPE Indexing (set_online_rope=True).")
            else:
                raise ValueError(
                    "online_rope_indexing=True requires is_causal=True and non-MMDiT model "
                    f"(got is_causal={is_causal}, is_mmdit={self.is_mmdit}).")

        # MemRoPE Phase 2 — dual-EMA Memory Tokens with per-half (gen/cond) three-tier cache.
        # Requires height-concat conditioning (token_concat_and_share_rope). Supersedes the
        # online_rope_indexing path. See SUMMARY_MEMROPE.md.
        from wan.modules.causal_model import parse_memrope_cfg
        self.memrope_cfg = parse_memrope_cfg(memrope)
        if self.memrope_cfg is not None:
            if not (is_causal and not self.is_mmdit and token_concat_and_share_rope
                    and hasattr(self.model, "set_memrope")):
                raise ValueError(
                    "memrope requires is_causal=True, non-MMDiT, and "
                    "token_concat_and_share_rope=True (height-concat conditioning). "
                    f"Got is_causal={is_causal}, is_mmdit={self.is_mmdit}, "
                    f"token_concat_and_share_rope={token_concat_and_share_rope}.")
            self.model.set_memrope(self.memrope_cfg)
            print(f"Enabled MemRoPE (set_memrope): fused={self.memrope_cfg['fused']} "
                  f"gen={self.memrope_cfg['gen']} cond={self.memrope_cfg['cond']}.")

        # RoPE compute precision for the AR inference rope calls (online + memrope).
        # fp64 (default) = exact; fp32 ≈ 2x faster rope, tiny precision change.
        if is_causal and not self.is_mmdit and hasattr(self.model, "set_rope_precision"):
            self.model.set_rope_precision(rope_precision)
        if str(rope_precision).lower() not in ("fp64", "float64"):
            print(f"RoPE compute precision set to {rope_precision}.")

        self.post_init()
        print(f"WanDiffusionWrapper for {model_name} initialised with is_causal={is_causal}, recam_concat={recam_concat}, token_concat_and_share_rope={token_concat_and_share_rope}, is_mmdit={self.is_mmdit}."
              f" local_attn_size={local_attn_size}, sink_size={sink_size}, cross_attn_norm={cross_attn_norm}, use_flash_attn={use_flash_attn}, fixed_timestep={fixed_timestep}, online_rope_indexing={online_rope_indexing}.")

    def enable_gradient_checkpointing(self) -> None:
        self.model.enable_gradient_checkpointing()

    def adding_cls_branch(self, atten_dim=1536, num_class=4, time_embed_dim=0,
                          gan_layer_indices=None) -> None:
        if gan_layer_indices is None:
            gan_layer_indices = [13, 21, 29]
        self._gan_layer_indices = gan_layer_indices
        num_registers = len(gan_layer_indices)
        mlp_in_dim = atten_dim * num_registers + time_embed_dim
        self._cls_pred_branch = nn.Sequential(
            nn.LayerNorm(mlp_in_dim),
            nn.Linear(mlp_in_dim, atten_dim),
            nn.SiLU(),
            nn.Linear(atten_dim, num_class)
        )
        self._cls_pred_branch.requires_grad_(True)
        self._register_tokens = RegisterTokens(num_registers=num_registers, dim=atten_dim)
        self._register_tokens.requires_grad_(True)

        self._gan_ca_blocks = nn.ModuleList(
            [GanAttentionBlock() for _ in range(num_registers)]
        )
        self._gan_ca_blocks.requires_grad_(True)

    def _convert_flow_pred_to_x0(self, flow_pred: torch.Tensor, xt: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        """
        Convert flow matching's prediction to x0 prediction.
        flow_pred: the prediction with shape [B, C, H, W]
        xt: the input noisy data with shape [B, C, H, W]
        timestep: the timestep with shape [B]

        pred = noise - x0
        x_t = (1-sigma_t) * x0 + sigma_t * noise
        we have x0 = x_t - sigma_t * pred
        see derivations https://chatgpt.com/share/67bf8589-3d04-8008-bc6e-4cf1a24e2d0e
        """
        # use higher precision for calculations
        original_dtype = flow_pred.dtype
        flow_pred, xt, sigmas, timesteps = map(
            lambda x: x.double().to(flow_pred.device), [flow_pred, xt,
                                                        self.scheduler.sigmas,
                                                        self.scheduler.timesteps]
        )

        timestep_id = torch.argmin(
            (timesteps.unsqueeze(0) - timestep.unsqueeze(1)).abs(), dim=1)
        sigma_t = sigmas[timestep_id].reshape(-1, 1, 1, 1)
        x0_pred = xt - sigma_t * flow_pred
        return x0_pred.to(original_dtype)

    @staticmethod
    def _convert_x0_to_flow_pred(scheduler, x0_pred: torch.Tensor, xt: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        """
        Convert x0 prediction to flow matching's prediction.
        x0_pred: the x0 prediction with shape [B, C, H, W]
        xt: the input noisy data with shape [B, C, H, W]
        timestep: the timestep with shape [B]

        pred = (x_t - x_0) / sigma_t
        """
        # use higher precision for calculations
        original_dtype = x0_pred.dtype
        x0_pred, xt, sigmas, timesteps = map(
            lambda x: x.double().to(x0_pred.device), [x0_pred, xt,
                                                      scheduler.sigmas,
                                                      scheduler.timesteps]
        )
        timestep_id = torch.argmin(
            (timesteps.unsqueeze(0) - timestep.unsqueeze(1)).abs(), dim=1)
        sigma_t = sigmas[timestep_id].reshape(-1, 1, 1, 1)
        flow_pred = (xt - x0_pred) / sigma_t
        return flow_pred.to(original_dtype)

    def _compute_seq_len(self, model_input_c_first: torch.Tensor) -> int:
        """
        Compute the transformer sequence length (number of tokens) from a C-first input tensor.

        Args:
            model_input_c_first: Tensor of shape [B, C, F, H, W] as fed to self.model.
        Returns:
            seq_len (int): F_patches * H_patches * W_patches
        """
        _, _, F, H, W = model_input_c_first.shape
        pt, ph, pw = self.model.patch_size
        return (F // pt) * (H // ph) * (W // pw)

    def forward(
        self,
        noisy_image_or_video: torch.Tensor, conditional_dict: dict,
        timestep: torch.Tensor, kv_cache: Optional[List[dict]] = None,
        crossattn_cache: Optional[List[dict]] = None,
        current_start: Optional[int] = None,
        classify_mode: Optional[bool] = False,
        extract_features: Optional[bool] = False,
        concat_time_embeddings: Optional[bool] = False,
        clean_x: Optional[torch.Tensor] = None,
        aug_t: Optional[torch.Tensor] = None,
        cache_start: Optional[int] = None,
        cond_latent: Optional[torch.Tensor] = None,
        cond_mask: Optional[torch.Tensor] = None,
        allow_kv_grad: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            cond_latent (Tensor, optional): Conditioning render/reference latent of shape
                [B, F_cond, C, H, W]. Concatenation mode depends on recam_concat:
                - recam_concat=True (ReCamMaster-style): concatenated along the frame
                  dimension → [B, C, F+F_cond, H, W]. Standard RoPE assigns unique
                  positions to all frames. Output is sliced back to the first F frames.
                - recam_concat=False (height-concat): concatenated along height dim →
                  [B, C, F, H+H_cond, W]. Requires token_concat_and_share_rope=True so
                  both halves share height positional embeddings. Output sliced to top half.
        """
        prompt_embeds = conditional_dict["prompt_embeds"]

        # [B, F] -> [B]
        if self.fixed_timestep >= 0:
            input_timestep = torch.full((timestep.shape[0],), self.fixed_timestep, device=timestep.device, dtype=timestep.dtype)
        else:
            if self.uniform_timestep:
                input_timestep = timestep[:, 0]
            else:
                input_timestep = timestep

        # Permute noisy input from [B, F, C, H, W] to [B, C, F, H, W]
        noisy_c_first = noisy_image_or_video.permute(0, 2, 1, 3, 4)
        F_noisy = noisy_c_first.shape[2]  # original latent frame count
        H_noisy = noisy_c_first.shape[3]  # original latent height

        if cond_latent is not None and self.is_mmdit:
            # MMDiT mode: cond_latent passed as a separate stream — no concat
            cond_x = cond_latent.permute(0, 2, 1, 3, 4).contiguous()  # [B, C, F, H, W]
            model_input = noisy_c_first
            seq_len = self.seq_len
        elif cond_latent is not None:
            cond_c_first = cond_latent.permute(0, 2, 1, 3, 4)  # [B, C, F_cond, H, W]
            if self.recam_concat:
                # ReCamMaster-style: concatenate along frame dim → [B, C, F+F_cond, H, W]
                # Standard RoPE assigns positions 0..F+F_cond-1 (no rope manipulation).
                model_input = torch.cat([noisy_c_first, cond_c_first], dim=2)
            else:
                # Height-concat mode: [B, C, F, H+H_cond, W].  Slice cond along the
                # frame dim to noisy's temporal extent in case caller passed a longer
                # cond_latent (e.g. the full dataset cond_latent threaded through to
                # the post-rollout score forwards when num_training_frames < dataset F).
                # Mirrors the existing slice in the clean_x branch below.
                assert cond_c_first.shape[2] >= F_noisy, (
                    f"cond_latent has fewer frames ({cond_c_first.shape[2]}) than "
                    f"noisy ({F_noisy}) — cannot height-concat."
                )
                cond_c_first = cond_c_first[:, :, :F_noisy]
                model_input = torch.cat([noisy_c_first, cond_c_first], dim=3)
            seq_len = self._compute_seq_len(model_input)
            cond_x = None
        else:
            model_input = noisy_c_first
            seq_len = self.seq_len
            cond_x = None

        logits = None
        features = None
        # X0 prediction
        if kv_cache is not None:
            flow_pred_c_first = self.model(
                model_input,
                t=input_timestep, context=prompt_embeds,
                seq_len=seq_len,
                kv_cache=kv_cache,
                crossattn_cache=crossattn_cache,
                current_start=current_start,
                cache_start=cache_start,
                **({"cond_x": cond_x, "allow_kv_grad": allow_kv_grad} if self.is_mmdit else {}),
            )
        else:
            if clean_x is not None:
                # teacher forcing
                # When cond_latent is supplied AND the model uses height-concat
                # (non-MMDiT, non-recam), the noisy_image_or_video was already
                # height-concatted with cond_latent above. Apply the same height-
                # concat to clean_x so its per-frame token count matches the
                # noisy half (otherwise the TF block_mask in causal_model.py,
                # which is constructed from x.shape[-2:] of the noisy input,
                # would be sized for 2H but clean_x would yield only-H tokens
                # per frame, producing a fatal token-count mismatch).
                clean_x_c_first = clean_x.permute(0, 2, 1, 3, 4)
                if (
                    cond_latent is not None
                    and not self.is_mmdit
                    and not self.recam_concat
                ):
                    # Slice cond_latent along the frame dim to clean_x's temporal
                    # extent in case caller passed a longer cond_latent.
                    F_clean = clean_x_c_first.shape[2]
                    cond_for_clean = cond_latent.permute(0, 2, 1, 3, 4)[:, :, :F_clean]
                    clean_x_c_first = torch.cat(
                        [clean_x_c_first, cond_for_clean], dim=3
                    )
                flow_pred_c_first = self.model(
                    model_input,
                    t=input_timestep, context=prompt_embeds,
                    seq_len=seq_len,
                    clean_x=clean_x_c_first,
                    aug_t=aug_t,
                )
            else:
                if classify_mode:
                    if extract_features:
                        # Combined forward: logits + intermediate features in one pass.
                        # features is a list of [B, seq_len_padded, dim] tensors, one per
                        # gan_layer_index, collected before the GAN CA blocks.
                        flow_pred_c_first, logits, features = self.model(
                            model_input,
                            t=input_timestep, context=prompt_embeds,
                            seq_len=seq_len,
                            classify_mode=True,
                            return_features=True,
                            register_tokens=self._register_tokens,
                            cls_pred_branch=self._cls_pred_branch,
                            gan_ca_blocks=self._gan_ca_blocks,
                            gan_layer_indices=self._gan_layer_indices,
                            concat_time_embeddings=concat_time_embeddings,
                            cond_mask=cond_mask,
                        )
                    else:
                        flow_pred_c_first, logits = self.model(
                            model_input,
                            t=input_timestep, context=prompt_embeds,
                            seq_len=seq_len,
                            classify_mode=True,
                            register_tokens=self._register_tokens,
                            cls_pred_branch=self._cls_pred_branch,
                            gan_ca_blocks=self._gan_ca_blocks,
                            gan_layer_indices=self._gan_layer_indices,
                            concat_time_embeddings=concat_time_embeddings,
                            cond_mask=cond_mask,
                        )
                else:
                    flow_pred_c_first = self.model(
                        model_input,
                        t=input_timestep, context=prompt_embeds,
                        seq_len=seq_len,
                        **({"cond_x": cond_x, "allow_kv_grad": allow_kv_grad} if self.is_mmdit else {}),
                    )

        # When cond_latent was provided, slice output to keep only the noisy half.
        # MMDiT mode outputs x-stream only (no slicing needed).
        if cond_latent is not None and not self.is_mmdit:
            if self.recam_concat:
                flow_pred_c_first = flow_pred_c_first[:, :, :F_noisy, :, :]
            else:
                flow_pred_c_first = flow_pred_c_first[:, :, :, :H_noisy, :]

        # Permute output back to [B, F, C, H, W]
        flow_pred = flow_pred_c_first.permute(0, 2, 1, 3, 4)

        # pred_x0 is derived from the original (non-doubled) noisy input
        pred_x0 = self._convert_flow_pred_to_x0(
            flow_pred=flow_pred.flatten(0, 1),
            xt=noisy_image_or_video.flatten(0, 1),
            timestep=timestep.flatten(0, 1)
        ).unflatten(0, flow_pred.shape[:2])

        if logits is not None:
            if features is not None:
                return flow_pred, pred_x0, logits, features
            return flow_pred, pred_x0, logits

        return flow_pred, pred_x0

    def get_scheduler(self) -> SchedulerInterface:
        """
        Update the current scheduler with the interface's static method
        """
        scheduler = self.scheduler
        scheduler.convert_x0_to_noise = types.MethodType(
            SchedulerInterface.convert_x0_to_noise, scheduler)
        scheduler.convert_noise_to_x0 = types.MethodType(
            SchedulerInterface.convert_noise_to_x0, scheduler)
        scheduler.convert_velocity_to_x0 = types.MethodType(
            SchedulerInterface.convert_velocity_to_x0, scheduler)
        self.scheduler = scheduler
        return scheduler

    def post_init(self):
        """
        A few custom initialization steps that should be called after the object is created.
        Currently, the only one we have is to bind a few methods to scheduler.
        We can gradually add more methods here if needed.
        """
        self.get_scheduler()
