"""
CausalDiffusionInferencePipeline
=================================
Full multi-step diffusion inference pipeline for the causal Wan model with
Classifier-Free Guidance (CFG).  This is the counterpart to
``CausalInferencePipeline`` (few-step) — it uses a proper ODE solver
(UniPC or DPM++) with ~50 denoising steps per temporal block instead of the
short fixed step-list used during training.

Key differences from ``CausalInferencePipeline``:
  * Full multi-step scheduler (FlowUniPCMultistepScheduler / FlowDPMSolverMultistepScheduler)
  * Classifier-Free Guidance — two separate KV caches (pos/neg) are maintained
    and two generator forward passes are made per denoising step.
  * Separate ``current_start_frame`` (= ``start_frame_index``) and
    ``cache_start_frame`` (= 0) for long-video sliding-window support.
  * Height-concat conditioning (``use_cond_latent``) is supported, matching
    the training setup in ``ODERegression._prepare_direct_input``.

Height-concat conditioning
--------------------------
When ``use_cond_latent=True`` in the config:
  * ``frame_seq_length`` is doubled (3120 instead of 1560) because the
    noisy input and the render conditioning are concatenated along the height
    dimension inside ``WanDiffusionWrapper.forward()``.
  * All self-attention blocks get their ``max_attention_size`` patched to
    account for the doubled sequence length.
  * A ``cond_latent`` tensor of shape ``[B, T, C, H, W]`` (same spatial dims
    as the latent) must be passed to ``inference()``.  The appropriate temporal
    slice is forwarded to every generator call.

CFG with height-concat
----------------------
The render conditioning (``cond_latent``) is the same for both the positive
and negative forward passes — only the text conditioning (cond vs. uncond)
differs between them.

When a pre-computed ``conditional_dict`` is supplied (e.g. from scene-id
mode in ``inference.py``), the positive text embeddings are used as-is and
the ``unconditional_dict`` is still computed from ``args.negative_prompt``
so that CFG remains functional.

Workflow per temporal block
---------------------------
1. Initialise / reset KV caches (pos + neg).
2. Cache initial-latent context frames (t=0 forward pass for both pos/neg).
3. For each temporal block:
   a. Run ODE solver loop (``sampling_steps`` iterations):
      - Two forward passes (pos + neg) at each solver step.
      - CFG blend: ``flow = uncond + scale * (cond - uncond)``.
      - Scheduler step to advance latent.
   b. Record denoised output.
   c. Re-run at t=0 (context update) for both pos/neg KV caches.
4. VAE decode output → pixel video.

Example usage (from inference.py)::

    pipeline = CausalDiffusionInferencePipeline(config, device=device)
    video, latents = pipeline.inference(
        noise=noise,                          # [B, T, C, H, W]
        text_prompts=["a scenic outdoor view"],
        cond_latent=cond_latent,             # [B, T, C, H, W] or None
        conditional_dict=cond_dict,          # optional pre-computed embeddings
        return_latents=True,
        low_memory=False,
    )
"""

from tqdm import tqdm
from typing import List, Optional
import torch

from wan.utils.fm_solvers import FlowDPMSolverMultistepScheduler, get_sampling_sigmas, retrieve_timesteps
from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
from utils.wan_wrapper import WanDiffusionWrapper, WanTextEncoder, WanVAEWrapper
from demo_utils.memory import gpu, get_cuda_free_memory_gb, DynamicSwapInstaller, move_model_to_device_with_memory_preservation
from pipeline.causal_base import CausalPipelineBase


class CausalDiffusionInferencePipeline(CausalPipelineBase, torch.nn.Module):
    def __init__(
            self,
            args,
            device,
            generator=None,
            text_encoder=None,
            vae=None
    ):
        torch.nn.Module.__init__(self)
        # Step 1: Initialize all models
        self.generator = WanDiffusionWrapper(
            **getattr(args, "model_kwargs", {}), is_causal=True) if generator is None else generator
        self.text_encoder = WanTextEncoder() if text_encoder is None else text_encoder
        self.vae = WanVAEWrapper() if vae is None else vae

        # Step 2: Initialize scheduler
        self.num_train_timesteps = args.num_train_timestep
        self.sampling_steps = 50
        self.sample_solver = 'unipc'
        self.shift = args.timestep_shift

        self.num_transformer_blocks = 30
        self.num_frame_per_block = getattr(args, "num_frame_per_block", 1)
        self.independent_first_frame = args.independent_first_frame
        self.local_attn_size = self.generator.model.local_attn_size

        # Height-concat conditioning support.
        # When use_cond_latent=True the render conditioning is concatenated with
        # the noisy input along the height dim inside WanDiffusionWrapper, so
        # every frame produces 2× tokens (3120 vs 1560 for 60×52 patch grid).
        self.use_cond_latent = getattr(args, 'use_cond_latent', False)
        self.frame_seq_length = 3120 if self.use_cond_latent else 1560

        # Patch max_attention_size on every self-attention block so the KV cache
        # window is correctly sized for the doubled sequence length.
        if self.use_cond_latent:
            self._patch_attn_max_size(self.generator, self.frame_seq_length)
            print(
                f"Changed self-attention max_attention_size to "
                f"{self.generator.model.blocks[-1].self_attn.max_attention_size} "
                f"to account for frame_seq_length={self.frame_seq_length} "
                f"with use_cond_latent={self.use_cond_latent}"
            )

        self.kv_cache_pos = None
        self.kv_cache_neg = None
        self.crossattn_cache_pos = None
        self.crossattn_cache_neg = None
        self.args = args

        print(f"KV inference with {self.num_frame_per_block} frames per block")

        if self.num_frame_per_block > 1:
            self.generator.model.num_frame_per_block = self.num_frame_per_block

    def inference(
        self,
        noise: torch.Tensor,
        text_prompts: List[str],
        initial_latent: Optional[torch.Tensor] = None,
        return_latents: bool = False,
        start_frame_index: Optional[int] = 0,
        cond_latent: Optional[torch.Tensor] = None,
        conditional_dict: Optional[dict] = None,
        low_memory: bool = False,
    ) -> torch.Tensor:
        """
        Perform full multi-step diffusion inference with CFG and optional
        height-concat conditioning.

        Args:
            noise: Input noise tensor ``[B, T, C, H, W]``.
            text_prompts: List of positive text prompts (length = B).
            initial_latent: Optional context latent ``[B, T_ctx, C, H, W]``
                for I2V or video-extension.  Frames are cached at t=0 before
                the main denoising loop.
            return_latents: If True, also return the raw output latent.
            start_frame_index: Global frame offset for long-video sliding-window
                mode.  Affects ``current_start`` passed to the generator (for
                RoPE positional encoding) but not the index into ``output`` or
                ``cond_latent`` (which always starts at 0).
            cond_latent: Optional render conditioning latent ``[B, T, C, H, W]``.
                Requires ``use_cond_latent=True`` in the config.  Temporal slices
                indexed by ``cache_start_frame`` are forwarded to every generator
                call (same slice for pos and neg CFG passes).
            conditional_dict: Optional pre-computed positive text embeddings
                ``{'prompt_embeds': Tensor [B, 512, 4096]}``.  When provided the
                text encoder is bypassed for the positive pass.  The negative
                (unconditional) dict is still computed from
                ``args.negative_prompt`` for CFG.
            low_memory: If True, apply memory-preserving model swapping after
                text encoding.

        Returns:
            video: Pixel video ``[B, T, C, H, W]`` normalised to ``[0, 1]``.
            (optional) output: Raw denoised latent ``[B, T, C, H, W]``.
        """
        batch_size, num_frames, num_channels, height, width = noise.shape
        num_blocks, num_input_frames, num_output_frames = self._compute_num_blocks(
            num_frames, initial_latent)

        # Text encoding.
        # If conditional_dict is supplied externally (e.g. from scene-id mode),
        # skip the positive encoding but still compute uncond for CFG.
        if conditional_dict is None:
            conditional_dict = self.text_encoder(text_prompts=text_prompts)
        unconditional_dict = self.text_encoder(
            text_prompts=[self.args.negative_prompt] * len(text_prompts)
        )

        if low_memory:
            gpu_memory_preservation = get_cuda_free_memory_gb(gpu) + 5
            move_model_to_device_with_memory_preservation(
                self.text_encoder,
                target_device=gpu,
                preserved_memory_gb=gpu_memory_preservation
            )

        output = torch.zeros(
            [batch_size, num_output_frames, num_channels, height, width],
            device=noise.device,
            dtype=noise.dtype
        )

        # Step 1: Initialize KV cache to all zeros
        if self.kv_cache_pos is None:
            self._initialize_kv_cache(
                batch_size=batch_size,
                dtype=noise.dtype,
                device=noise.device,
                num_output_frames=num_output_frames,
            )
            self._initialize_crossattn_cache(
                batch_size=batch_size,
                dtype=noise.dtype,
                device=noise.device
            )
        else:
            self._reset_crossattn_cache(self.crossattn_cache_pos)
            self._reset_crossattn_cache(self.crossattn_cache_neg)
            self._reset_kv_cache(self.kv_cache_pos, noise.device)
            self._reset_kv_cache(self.kv_cache_neg, noise.device)

        # Step 2: Cache context feature
        # current_start_frame: global frame index used for RoPE positional encoding
        # cache_start_frame:   local index into output / cond_latent tensors (always 0-based)
        current_start_frame = start_frame_index
        cache_start_frame = 0
        if initial_latent is not None:
            timestep = torch.ones([batch_size, 1], device=noise.device, dtype=torch.int64) * 0
            if self.independent_first_frame:
                # Assume num_input_frames is 1 + self.num_frame_per_block * num_input_blocks
                assert (num_input_frames - 1) % self.num_frame_per_block == 0
                num_input_blocks = (num_input_frames - 1) // self.num_frame_per_block
                output[:, :1] = initial_latent[:, :1]
                _cond_block = self._slice_cond(cond_latent, cache_start_frame, 1)
                self.generator(
                    noisy_image_or_video=initial_latent[:, :1],
                    conditional_dict=conditional_dict,
                    timestep=timestep * 0,
                    kv_cache=self.kv_cache_pos,
                    crossattn_cache=self.crossattn_cache_pos,
                    current_start=current_start_frame * self.frame_seq_length,
                    cache_start=cache_start_frame * self.frame_seq_length,
                    cond_latent=_cond_block,
                )
                self.generator(
                    noisy_image_or_video=initial_latent[:, :1],
                    conditional_dict=unconditional_dict,
                    timestep=timestep * 0,
                    kv_cache=self.kv_cache_neg,
                    crossattn_cache=self.crossattn_cache_neg,
                    current_start=current_start_frame * self.frame_seq_length,
                    cache_start=cache_start_frame * self.frame_seq_length,
                    cond_latent=_cond_block,
                )
                current_start_frame += 1
                cache_start_frame += 1
            else:
                # Assume num_input_frames is self.num_frame_per_block * num_input_blocks
                assert num_input_frames % self.num_frame_per_block == 0
                num_input_blocks = num_input_frames // self.num_frame_per_block

            for block_index in range(num_input_blocks):
                current_ref_latents = \
                    initial_latent[:, cache_start_frame:cache_start_frame + self.num_frame_per_block]
                output[:, cache_start_frame:cache_start_frame + self.num_frame_per_block] = current_ref_latents
                _cond_block = self._slice_cond(
                    cond_latent, cache_start_frame, self.num_frame_per_block)
                self.generator(
                    noisy_image_or_video=current_ref_latents,
                    conditional_dict=conditional_dict,
                    timestep=timestep * 0,
                    kv_cache=self.kv_cache_pos,
                    crossattn_cache=self.crossattn_cache_pos,
                    current_start=current_start_frame * self.frame_seq_length,
                    cache_start=cache_start_frame * self.frame_seq_length,
                    cond_latent=_cond_block,
                )
                self.generator(
                    noisy_image_or_video=current_ref_latents,
                    conditional_dict=unconditional_dict,
                    timestep=timestep * 0,
                    kv_cache=self.kv_cache_neg,
                    crossattn_cache=self.crossattn_cache_neg,
                    current_start=current_start_frame * self.frame_seq_length,
                    cache_start=cache_start_frame * self.frame_seq_length,
                    cond_latent=_cond_block,
                )
                current_start_frame += self.num_frame_per_block
                cache_start_frame += self.num_frame_per_block

        # Step 3: Temporal denoising loop
        all_num_frames = self._build_all_num_frames(num_blocks, initial_latent)
        for current_num_frames in all_num_frames:
            noisy_input = noise[
                :, cache_start_frame - num_input_frames:cache_start_frame + current_num_frames - num_input_frames]
            latents = noisy_input

            # Slice the cond_latent for this temporal block (same for pos and neg).
            _cond_block = self._slice_cond(cond_latent, cache_start_frame, current_num_frames)

            # Step 3.1: Spatial denoising loop (full ODE solver)
            sample_scheduler = self._initialize_sample_scheduler(noise)
            for _, t in enumerate(tqdm(sample_scheduler.timesteps)):
                latent_model_input = latents
                timestep = t * torch.ones(
                    [batch_size, current_num_frames], device=noise.device, dtype=torch.float32
                )

                flow_pred_cond, _ = self.generator(
                    noisy_image_or_video=latent_model_input,
                    conditional_dict=conditional_dict,
                    timestep=timestep,
                    kv_cache=self.kv_cache_pos,
                    crossattn_cache=self.crossattn_cache_pos,
                    current_start=current_start_frame * self.frame_seq_length,
                    cache_start=cache_start_frame * self.frame_seq_length,
                    cond_latent=_cond_block,
                )
                flow_pred_uncond, _ = self.generator(
                    noisy_image_or_video=latent_model_input,
                    conditional_dict=unconditional_dict,
                    timestep=timestep,
                    kv_cache=self.kv_cache_neg,
                    crossattn_cache=self.crossattn_cache_neg,
                    current_start=current_start_frame * self.frame_seq_length,
                    cache_start=cache_start_frame * self.frame_seq_length,
                    cond_latent=_cond_block,
                )

                flow_pred = flow_pred_uncond + self.args.guidance_scale * (
                    flow_pred_cond - flow_pred_uncond)

                temp_x0 = sample_scheduler.step(
                    flow_pred,
                    t,
                    latents,
                    return_dict=False)[0]
                latents = temp_x0

            # Step 3.2: record the model's output
            output[:, cache_start_frame:cache_start_frame + current_num_frames] = latents

            # Step 3.3: rerun with timestep zero to update KV cache using clean context
            context_timestep = timestep * 0
            self.generator(
                noisy_image_or_video=latents,
                conditional_dict=conditional_dict,
                timestep=context_timestep,
                kv_cache=self.kv_cache_pos,
                crossattn_cache=self.crossattn_cache_pos,
                current_start=current_start_frame * self.frame_seq_length,
                cache_start=cache_start_frame * self.frame_seq_length,
                cond_latent=_cond_block,
            )
            self.generator(
                noisy_image_or_video=latents,
                conditional_dict=unconditional_dict,
                timestep=context_timestep,
                kv_cache=self.kv_cache_neg,
                crossattn_cache=self.crossattn_cache_neg,
                current_start=current_start_frame * self.frame_seq_length,
                cache_start=cache_start_frame * self.frame_seq_length,
                cond_latent=_cond_block,
            )

            # Step 3.4: update the start and end frame indices
            current_start_frame += current_num_frames
            cache_start_frame += current_num_frames

        # Step 4: Decode the output
        video = self.vae.decode_to_pixel(output, use_cache=False)
        video = (video * 0.5 + 0.5).clamp(0, 1)

        if return_latents:
            return video, output
        else:
            return video

    def _initialize_kv_cache(self, batch_size, dtype, device, num_output_frames=21):
        """Initialize KV caches for both positive (cond) and negative (uncond) CFG passes.

        Args:
            num_output_frames: Total output frames used to size the cache when
                ``local_attn_size == -1``.  With ``use_cond_latent=True``
                ``frame_seq_length`` is already doubled, so the capacity doubles
                automatically.
        """
        kv_cache_size = self._compute_kv_cache_size(self.local_attn_size, num_output_frames)
        self.kv_cache_pos = self._make_kv_cache(batch_size, kv_cache_size, dtype, device)
        self.kv_cache_neg = self._make_kv_cache(batch_size, kv_cache_size, dtype, device)

    def _initialize_crossattn_cache(self, batch_size, dtype, device):
        """Initialize cross-attention caches for both CFG passes."""
        self.crossattn_cache_pos = self._make_crossattn_cache(batch_size, dtype, device)
        self.crossattn_cache_neg = self._make_crossattn_cache(batch_size, dtype, device)

    def _initialize_sample_scheduler(self, noise):
        if self.sample_solver == 'unipc':
            sample_scheduler = FlowUniPCMultistepScheduler(
                num_train_timesteps=self.num_train_timesteps,
                shift=1,
                use_dynamic_shifting=False)
            sample_scheduler.set_timesteps(
                self.sampling_steps, device=noise.device, shift=self.shift)
            self.timesteps = sample_scheduler.timesteps
        elif self.sample_solver == 'dpm++':
            sample_scheduler = FlowDPMSolverMultistepScheduler(
                num_train_timesteps=self.num_train_timesteps,
                shift=1,
                use_dynamic_shifting=False)
            sampling_sigmas = get_sampling_sigmas(self.sampling_steps, self.shift)
            self.timesteps, _ = retrieve_timesteps(
                sample_scheduler,
                device=noise.device,
                sigmas=sampling_sigmas)
        else:
            raise NotImplementedError("Unsupported solver.")
        return sample_scheduler
