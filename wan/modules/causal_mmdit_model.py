"""
CausalMMDiTModel — Causal (autoregressive) MMDiT-style dual-stream video DiT.

Overview
--------
Causal (autoregressive with KV-cache) version of WanMMDiTModel for use as the
generator in the self-forcing pipeline.  The condition latent (cond_x) is kept as a
**separate stream** — no height-concatenation.

Key differences from WanMMDiTModel (Gen3C):
  • Causal FlexAttention block mask during training (not full bidirectional).
  • KV-cache inference path for real-time streaming generation.
  • Dual-stream FlexAttention mask at double-stream blocks: [x_0..x_{N-1}|c_0..c_{N-1}]
    where any token at frame i can attend to all x- and c-tokens at frames 0..i.
  • No teacher-forcing — pure causal rollout during training.
  • Condition stream is always clean (t=0 effectively), only x is noised.

Architecture
------------
Block layout (example with num_layers=30, double_stream_layers=[0,1,4,8,12,16,20,24,28,29]):
    Blocks in double_stream_layers  → CausalMMDiTDoubleBlock
    Remaining blocks                → CausalWanAttentionBlock (x-only, unchanged)

At a CausalMMDiTDoubleBlock:
  • Training: joint FlexAttention on [x_tokens | c_tokens] with dual-stream causal mask.
  • Inference: flash attention per-stream attending to joint [x_kv | c_kv] from cache.
  • At is_last block: c K/V always written to cache; c residual/cross-attn/FFN skipped.

Weight Loading
--------------
Call load_causal_mmdit_weights(model, state_dict) to initialise from a CausalWanModel
checkpoint.  x-stream weights copy directly; c-stream weights are cloned from x-stream
(with 'cond_' prefix).  If state_dict already contains 'cond_' keys, load directly.

Example usage
-------------
    model = CausalMMDiTModel(
        double_stream_layers=[0, 1, 4, 8, 12, 16, 20, 24, 28, 29],
        local_attn_size=-1,
    )
    load_causal_mmdit_weights(model, causal_wan_state_dict)

    # Training (kv_cache=None):
    flow_pred = model(x, t, context, cond_x=cond_x)

    # Inference (KV cache):
    kv_cache = init_dual_kv_cache(...)
    flow_pred = model(x_block, t_block, context, cond_x=cond_x_block,
                      kv_cache=kv_cache, current_start=frame_idx * 1560)
"""

import math
import contextlib
from typing import List, Optional, Set, Tuple

import torch
import torch.nn as nn
from torch.nn.attention.flex_attention import create_block_mask, flex_attention, BlockMask
import torch.distributed as dist

from wan.modules.attention import attention
from wan.modules.model import (
    WanRMSNorm,
    WanLayerNorm,
    WAN_CROSSATTENTION_CLASSES,
    rope_params,
    sinusoidal_embedding_1d,
    rope_apply,
)
from wan.modules.causal_model import (
    causal_rope_apply,
    CausalWanAttentionBlock,
    CausalHead,
)

# Compile flex_attention once (same setting as causal_model.py)
flex_attention = torch.compile(
    flex_attention, dynamic=False, mode="max-autotune-no-cudagraphs")


# ---------------------------------------------------------------------------
# Self-attention projection module (Q/K/V/O + optional RMSNorm)
# ---------------------------------------------------------------------------

class _SelfAttnProj(nn.Module):
    """Q/K/V/O projections + optional RMSNorm for one attention stream.

    Attribute names match CausalWanSelfAttention (q, k, v, o, norm_q, norm_k)
    so that CausalWanModel checkpoint weights load directly into the x-stream
    and can be cloned into the c-stream.
    """

    def __init__(self, dim: int, num_heads: int, eps: float = 1e-6, qk_norm: bool = True):
        super().__init__()
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()


# ---------------------------------------------------------------------------
# CausalMMDiTDoubleBlock
# ---------------------------------------------------------------------------

class CausalMMDiTDoubleBlock(nn.Module):
    """Causal dual-stream MMDiT block.

    Processes x (noisy) and c (condition) streams jointly at self-attention via:
      - Training: FlexAttention with dual-stream causal block mask, layout [x|c].
      - Inference: Flash attention per-stream attending to joint [x_kv | c_kv] cache.

    Sub-layer naming matches CausalWanAttentionBlock for direct x-stream weight loading:
        self_attn, cross_attn, norm1, norm2, norm3, ffn, modulation  (x-stream)
        cond_self_attn, cond_cross_attn, cond_norm1, cond_norm2, cond_norm3,
        cond_ffn, cond_modulation                                      (c-stream)

    At is_last block: c K/V is always written to cache (needed for future frame blocks),
    but c residual update, cross-attn, and FFN are skipped.

    Workflow — training path (kv_cache=None)
    -----------------------------------------
    1. Per-frame modulation (e) of x and c independently.
    2. Project x → x_q/x_k/x_v, c → c_q/c_k/c_v with RMSNorm on q/k.
    3. Apply same RoPE freqs to both x and c q/k (position-coherent).
    4. Concatenate [x|c], pad to 128, FlexAttention with block_mask_dual.
    5. Split output; apply o-proj + gate; x cross-attn + FFN; c cross-attn + FFN.

    Workflow — inference path (kv_cache is not None)
    -------------------------------------------------
    1. Per-frame modulation; project + RoPE with causal offset.
    2. Write new x_k/x_v and c_k/c_v into dual KV cache (with eviction if local attn).
    3. Build joint KV: [x_cache | c_cache]; flash-attn for x_q and c_q.
    4. Residual + cross-attn + FFN updates (c-stream skipped at is_last).
    """

    def __init__(
        self,
        cross_attn_type: str,
        dim: int,
        ffn_dim: int,
        num_heads: int,
        local_attn_size: int = -1,
        sink_size: int = 0,
        qk_norm: bool = True,
        cross_attn_norm: bool = False,
        eps: float = 1e-6,
        is_last: bool = False,
        rope_fn=None,
        causal_rope_fn=None,
        use_flash_attn: bool = False,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.local_attn_size = local_attn_size
        self.sink_size = sink_size
        self.is_last = is_last
        self.use_flash_attn = use_flash_attn

        # ── x-stream (same names as CausalWanAttentionBlock for weight loading) ──
        self.norm1 = WanLayerNorm(dim, eps)
        self.self_attn = _SelfAttnProj(dim, num_heads, eps, qk_norm)
        self.norm3 = WanLayerNorm(dim, eps, elementwise_affine=True) if cross_attn_norm else nn.Identity()
        self.cross_attn = WAN_CROSSATTENTION_CLASSES[cross_attn_type](
            dim, num_heads, (-1, -1), qk_norm, eps)
        self.norm2 = WanLayerNorm(dim, eps)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim), nn.GELU(approximate='tanh'), nn.Linear(ffn_dim, dim))
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim ** 0.5)

        # ── c-stream (cond_ prefix) ──────────────────────────────────────────
        self.cond_norm1 = WanLayerNorm(dim, eps)
        self.cond_self_attn = _SelfAttnProj(dim, num_heads, eps, qk_norm)
        self.cond_modulation = nn.Parameter(torch.randn(1, 6, dim) / dim ** 0.5)

        if not is_last:
            self.cond_norm3 = WanLayerNorm(dim, eps, elementwise_affine=True) if cross_attn_norm else nn.Identity()
            self.cond_cross_attn = WAN_CROSSATTENTION_CLASSES[cross_attn_type](
                dim, num_heads, (-1, -1), qk_norm, eps)
            self.cond_norm2 = WanLayerNorm(dim, eps)
            self.cond_ffn = nn.Sequential(
                nn.Linear(dim, ffn_dim), nn.GELU(approximate='tanh'), nn.Linear(ffn_dim, dim))

        # RoPE function injection (per-instance, no global patching)
        self.rope_fn = rope_fn if rope_fn is not None else rope_apply
        self.causal_rope_fn = causal_rope_fn if causal_rope_fn is not None else causal_rope_apply

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,           # [B, S_x, dim]  S_x = num_frames * frame_seqlen
        c: torch.Tensor,           # [B, S_c, dim]  same S as x
        e_x: torch.Tensor,         # [B, F, 6, dim] per-frame modulation for x
        e_c: torch.Tensor,         # [B, F, 6, dim] per-frame modulation for c
        seq_lens: torch.Tensor,    # [B]
        grid_sizes: torch.Tensor,  # [B, 3]
        freqs: torch.Tensor,       # RoPE freqs
        context: torch.Tensor,     # [B, text_len, dim] projected text
        context_lens,              # None or [B]
        block_mask_dual=None,      # FlexAttention BlockMask (training)
        kv_cache=None,             # dual KV cache dict (inference)
        crossattn_cache=None,      # cross-attn cache dict (inference)
        current_start: int = 0,
        allow_kv_grad: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, S_x, _ = x.shape
        n, d = self.num_heads, self.head_dim
        num_frames = e_x.shape[1]
        frame_seqlen = S_x // num_frames

        # ── Modulation ────────────────────────────────────────────────────────
        # e_x/e_c: [B, F, 6, dim]; modulation: [1, 6, dim] → [1, 1, 6, dim]
        e_x_mod = (self.modulation.unsqueeze(1) + e_x).chunk(6, dim=2)
        e_c_mod = (self.cond_modulation.unsqueeze(1) + e_c).chunk(6, dim=2)
        # Each e_x_mod[i]: [B, F, 1, dim] → broadcasts over frame_seqlen tokens

        def _modulate_x(norm_x, shift, scale):
            """Apply per-frame AdaLN: norm → scale/shift in [B, F, frame_seqlen, dim]."""
            return (norm_x.unflatten(1, (num_frames, frame_seqlen)) * (1 + scale) + shift).flatten(1, 2)

        # ── Compute Q, K, V for both streams ─────────────────────────────────
        x_in = _modulate_x(self.norm1(x), e_x_mod[0], e_x_mod[1])
        c_in = _modulate_x(self.cond_norm1(c), e_c_mod[0], e_c_mod[1])

        x_q = self.self_attn.norm_q(self.self_attn.q(x_in)).view(B, S_x, n, d)
        x_k = self.self_attn.norm_k(self.self_attn.k(x_in)).view(B, S_x, n, d)
        x_v = self.self_attn.v(x_in).view(B, S_x, n, d)

        c_q = self.cond_self_attn.norm_q(self.cond_self_attn.q(c_in)).view(B, S_x, n, d)
        c_k = self.cond_self_attn.norm_k(self.cond_self_attn.k(c_in)).view(B, S_x, n, d)
        c_v = self.cond_self_attn.v(c_in).view(B, S_x, n, d)

        if kv_cache is None:
            # ── Training path ─────────────────────────────────────────────────
            # Apply same RoPE to both streams (position-coherent)
            x_q = self.rope_fn(x_q, grid_sizes, freqs).type_as(x_v)
            x_k = self.rope_fn(x_k, grid_sizes, freqs).type_as(x_v)
            c_q = self.rope_fn(c_q, grid_sizes, freqs).type_as(c_v)
            c_k = self.rope_fn(c_k, grid_sizes, freqs).type_as(c_v)

            if self.use_flash_attn:
                # Per-frame loop with flash attention: frame f attends to
                # joint [x_kv | c_kv] from frames 0..f
                x_attn_parts = []
                c_attn_parts = []
                for f in range(num_frames):
                    f_start = f * frame_seqlen
                    f_end = (f + 1) * frame_seqlen
                    joint_k = torch.cat([x_k[:, :f_end], c_k[:, :f_end]], dim=1)
                    joint_v = torch.cat([x_v[:, :f_end], c_v[:, :f_end]], dim=1)
                    x_attn_parts.append(attention(
                        x_q[:, f_start:f_end], joint_k, joint_v))
                    c_attn_parts.append(attention(
                        c_q[:, f_start:f_end], joint_k, joint_v))
                x_attn = torch.cat(x_attn_parts, dim=1).flatten(2)  # [B, S_x, dim]
                c_attn = torch.cat(c_attn_parts, dim=1).flatten(2)  # [B, S_c, dim]
            else:
                # FlexAttention with dual-stream causal mask
                # Concatenate [x | c] for joint attention
                q = torch.cat([x_q, c_q], dim=1)   # [B, 2*S, n, d]
                k = torch.cat([x_k, c_k], dim=1)
                v = torch.cat([x_v, c_v], dim=1)

                total_len = q.shape[1]
                padded_length = math.ceil(total_len / 128) * 128 - total_len
                pad = lambda t: torch.cat([
                    t, t.new_zeros(B, padded_length, n, d)], dim=1)

                out = flex_attention(
                    query=pad(q).transpose(2, 1),
                    key=pad(k).transpose(2, 1),
                    value=pad(v).transpose(2, 1),
                    block_mask=block_mask_dual,
                )  # [B, n, 2*S+pad, d]
                if padded_length > 0:
                    out = out[:, :, :-padded_length, :]
                out = out.transpose(2, 1)  # [B, 2*S, n, d]

                x_attn = out[:, :S_x, :, :].flatten(2)  # [B, S_x, dim]
                c_attn = out[:, S_x:, :, :].flatten(2)  # [B, S_c, dim]

        else:
            # TODO: KV cache safety: On recomputation, current_end == global_end_index (already updated from the first run), so local_end_index doesn't change and the writes land on the exact same positions with identical values — idempotent. 
            # One thing to watch: if allow_kv_grad=True at the exit step, the KV writes inside checkpointed calls retain gradients, which is the same situation as in causal_model._forward_inference. Since that works fine, this should too.
            # ── Inference path: KV cache + flash attention ────────────────────
            frame_seqlen_kv = math.prod(grid_sizes[0][1:]).item()
            current_start_frame = current_start // frame_seqlen_kv

            x_q = self.causal_rope_fn(x_q, grid_sizes, freqs, start_frame=current_start_frame).type_as(x_v)
            x_k = self.causal_rope_fn(x_k, grid_sizes, freqs, start_frame=current_start_frame).type_as(x_v)
            c_q = self.causal_rope_fn(c_q, grid_sizes, freqs, start_frame=current_start_frame).type_as(c_v)
            c_k = self.causal_rope_fn(c_k, grid_sizes, freqs, start_frame=current_start_frame).type_as(c_v)

            current_end = current_start + x_q.shape[1]
            num_new_tokens = x_q.shape[1]
            sink_tokens = self.sink_size * frame_seqlen_kv
            kv_cache_size = kv_cache["x_k"].shape[1]
            global_end = kv_cache["global_end_index"].item()
            local_end_prev = kv_cache["local_end_index"].item()

            if (self.local_attn_size != -1
                    and current_end > global_end
                    and num_new_tokens + local_end_prev > kv_cache_size):
                # KV eviction: shift buffers left, preserve sink tokens
                num_evicted = num_new_tokens + local_end_prev - kv_cache_size
                num_rolled = local_end_prev - num_evicted - sink_tokens

                def _evict(buf):
                    buf[:, sink_tokens:sink_tokens + num_rolled] = \
                        buf[:, sink_tokens + num_evicted:sink_tokens + num_evicted + num_rolled].clone()

                if allow_kv_grad:
                    _evict(kv_cache["x_k"]); _evict(kv_cache["x_v"])
                    _evict(kv_cache["c_k"]); _evict(kv_cache["c_v"])
                else:
                    with torch.no_grad():
                        _evict(kv_cache["x_k"]); _evict(kv_cache["x_v"])
                        _evict(kv_cache["c_k"]); _evict(kv_cache["c_v"])

                local_end_index = local_end_prev + current_end - global_end - num_evicted
            else:
                local_end_index = local_end_prev + current_end - global_end

            local_start_index = local_end_index - num_new_tokens

            # Write new K/V to both sub-buffers
            ctx = contextlib.nullcontext() if allow_kv_grad else torch.no_grad()
            with ctx:
                kv_cache["x_k"][:, local_start_index:local_end_index] = x_k
                kv_cache["x_v"][:, local_start_index:local_end_index] = x_v
                kv_cache["c_k"][:, local_start_index:local_end_index] = c_k
                kv_cache["c_v"][:, local_start_index:local_end_index] = c_v
                kv_cache["global_end_index"].fill_(current_end)
                kv_cache["local_end_index"].fill_(local_end_index)

            # Build joint KV: [x_cache | c_cache]
            joint_k = torch.cat([
                kv_cache["x_k"][:, :local_end_index],
                kv_cache["c_k"][:, :local_end_index],
            ], dim=1)  # [B, 2*local_end, n, d]
            joint_v = torch.cat([
                kv_cache["x_v"][:, :local_end_index],
                kv_cache["c_v"][:, :local_end_index],
            ], dim=1)

            x_attn = attention(x_q, joint_k, joint_v).flatten(2)  # [B, S_x, dim]
            c_attn = attention(c_q, joint_k, joint_v).flatten(2)  # [B, S_c, dim]

        # ── x-stream: residual + cross-attn + FFN ────────────────────────────
        x_gate = e_x_mod[2]  # [B, F, 1, dim]
        x = x + (self.self_attn.o(x_attn).unflatten(1, (num_frames, frame_seqlen)) * x_gate).flatten(1, 2)
        x = x + self.cross_attn(self.norm3(x), context, context_lens,
                                 crossattn_cache=crossattn_cache)
        x_shift_mlp, x_scale_mlp, x_gate_mlp = e_x_mod[3], e_x_mod[4], e_x_mod[5]
        x_ffn_in = _modulate_x(self.norm2(x), x_shift_mlp, x_scale_mlp)
        x = x + (self.ffn(x_ffn_in).unflatten(1, (num_frames, frame_seqlen)) * x_gate_mlp).flatten(1, 2)

        # ── c-stream: residual + cross-attn + FFN (skipped at is_last) ───────
        if not self.is_last:
            c_gate = e_c_mod[2]
            c = c + (self.cond_self_attn.o(c_attn).unflatten(1, (num_frames, frame_seqlen)) * c_gate).flatten(1, 2)
            c = c + self.cond_cross_attn(self.cond_norm3(c), context, context_lens,
                                          crossattn_cache=crossattn_cache)
            c_shift_mlp, c_scale_mlp, c_gate_mlp = e_c_mod[3], e_c_mod[4], e_c_mod[5]
            c_ffn_in = _modulate_x(self.cond_norm2(c), c_shift_mlp, c_scale_mlp)
            c = c + (self.cond_ffn(c_ffn_in).unflatten(1, (num_frames, frame_seqlen)) * c_gate_mlp).flatten(1, 2)

        return x, c


# ---------------------------------------------------------------------------
# CausalMMDiTModel
# ---------------------------------------------------------------------------

class CausalMMDiTModel(nn.Module):
    """Causal dual-stream MMDiT video diffusion model.

    Architecture overview
    ---------------------
    Blocks at indices in double_stream_layers → CausalMMDiTDoubleBlock.
    All other blocks → CausalWanAttentionBlock (x-only, identical to CausalWanModel).

    Two FlexAttention masks are cached after first forward:
      block_mask      — x-only N*S × N*S causal mask (for regular blocks).
      block_mask_dual — dual-stream 2N*S × 2N*S causal mask (for double blocks).

    Constructor parameters
    ----------------------
    double_stream_layers : list[int]
        Block indices that are CausalMMDiTDoubleBlock.
        Default: [0, 1, 4, 8, 12, 16, 20, 24, 28, 29]
    local_attn_size : int
        Local attention window size in frames (-1 = global).
    sink_size : int
        Number of sink frames preserved during KV eviction.
    rope_fn / causal_rope_fn
        Custom RoPE implementations (injected via set_rope_fn, no global patching).
    """

    def __init__(
        self,
        model_type: str = 't2v',
        patch_size: Tuple[int, int, int] = (1, 2, 2),
        text_len: int = 512,
        in_dim: int = 16,
        dim: int = 1536,
        ffn_dim: int = 8960,
        freq_dim: int = 256,
        text_dim: int = 4096,
        out_dim: int = 16,
        num_heads: int = 12,
        num_layers: int = 30,
        local_attn_size: int = -1,
        sink_size: int = 0,
        qk_norm: bool = True,
        cross_attn_norm: bool = False,
        eps: float = 1e-6,
        double_stream_layers: Optional[List[int]] = None,
        rope_fn=None,
        causal_rope_fn=None,
        use_flash_attn: bool = False,
    ):
        super().__init__()

        if double_stream_layers is None:
            double_stream_layers = [0, 1, 4, 8, 12, 16, 20, 24, 28, 29]

        assert model_type in ('t2v', 'i2v')
        self.model_type = model_type
        self.use_flash_attn = use_flash_attn
        self.patch_size = patch_size
        self.text_len = text_len
        self.in_dim = in_dim
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.freq_dim = freq_dim
        self.text_dim = text_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.local_attn_size = local_attn_size
        self.sink_size = sink_size
        self.eps = eps
        self.double_stream_layers: Set[int] = set(double_stream_layers)

        # ── Patch embeddings (separate per stream) ───────────────────────────
        self.patch_embedding = nn.Conv3d(
            in_dim, dim, kernel_size=patch_size, stride=patch_size)
        self.cond_patch_embedding = nn.Conv3d(
            in_dim, dim, kernel_size=patch_size, stride=patch_size)

        # ── Text & time embeddings ───────────────────────────────────────────
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim), nn.GELU(approximate='tanh'), nn.Linear(dim, dim))
        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))

        # x-stream modulation (same names as CausalWanModel for weight loading)
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 6))
        # c-stream modulation
        self.cond_time_projection = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 6))

        # ── Blocks ────────────────────────────────────────────────────────────
        cross_attn_type = 't2v_cross_attn' if model_type == 't2v' else 'i2v_cross_attn'
        last_double = max(double_stream_layers)
        blocks = []
        for i in range(num_layers):
            if i in self.double_stream_layers:
                blocks.append(CausalMMDiTDoubleBlock(
                    cross_attn_type=cross_attn_type,
                    dim=dim, ffn_dim=ffn_dim, num_heads=num_heads,
                    local_attn_size=local_attn_size, sink_size=sink_size,
                    qk_norm=qk_norm, cross_attn_norm=cross_attn_norm, eps=eps,
                    is_last=(i == last_double),
                    rope_fn=rope_fn, causal_rope_fn=causal_rope_fn,
                    use_flash_attn=use_flash_attn,
                ))
            else:
                blocks.append(CausalWanAttentionBlock(
                    cross_attn_type, dim, ffn_dim, num_heads,
                    local_attn_size, sink_size, qk_norm, cross_attn_norm, eps,
                    rope_fn=rope_fn, causal_rope_fn=causal_rope_fn,
                    use_flash_attn=use_flash_attn,
                ))
        self.blocks = nn.ModuleList(blocks)

        # ── Output head (x-stream only) ───────────────────────────────────────
        self.head = CausalHead(dim, out_dim, patch_size, eps)

        # ── RoPE freqs ────────────────────────────────────────────────────────
        assert (dim % num_heads) == 0 and (dim // num_heads) % 2 == 0
        d = dim // num_heads
        self.freqs = torch.cat([
            rope_params(1024, d - 4 * (d // 6)),
            rope_params(1024, 2 * (d // 6)),
            rope_params(1024, 2 * (d // 6)),
        ], dim=1)

        # ── Cached FlexAttention masks ────────────────────────────────────────
        self.block_mask: Optional[BlockMask] = None
        self.block_mask_dual: Optional[BlockMask] = None

        # ── Runtime flags (match CausalWanModel API) ─────────────────────────
        self.gradient_checkpointing = False
        self.num_frame_per_block = 1
        self.independent_first_frame = False

        self.init_weights()

    # ------------------------------------------------------------------
    # Weight init
    # ------------------------------------------------------------------

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        nn.init.xavier_uniform_(self.patch_embedding.weight.flatten(1))
        nn.init.xavier_uniform_(self.cond_patch_embedding.weight.flatten(1))
        for emb in (self.text_embedding, self.time_embedding):
            for m in emb.modules():
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, std=0.02)
        nn.init.zeros_(self.head.head.weight)

    # ------------------------------------------------------------------
    # Gradient checkpointing
    # ------------------------------------------------------------------

    def enable_gradient_checkpointing(self):
        self.gradient_checkpointing = True

    def _set_gradient_checkpointing(self, module, value=False):
        self.gradient_checkpointing = value

    # ------------------------------------------------------------------
    # RoPE injection
    # ------------------------------------------------------------------

    def set_rope_fn(self, rope_fn=None, causal_rope_fn=None):
        """Replace RoPE functions on all blocks (no global monkey-patching)."""
        for block in self.blocks:
            if isinstance(block, CausalMMDiTDoubleBlock):
                if rope_fn is not None:
                    block.rope_fn = rope_fn
                if causal_rope_fn is not None:
                    block.causal_rope_fn = causal_rope_fn
            else:  # CausalWanAttentionBlock
                if rope_fn is not None:
                    block.self_attn.rope_fn = rope_fn
                if causal_rope_fn is not None:
                    block.self_attn.causal_rope_fn = causal_rope_fn

    # ------------------------------------------------------------------
    # FlexAttention mask creation/caching
    # ------------------------------------------------------------------

    def _get_block_mask(self, num_frames: int, frame_seqlen: int, device) -> BlockMask:
        """Create/cache the x-only causal block mask (same as CausalWanModel)."""
        if self.block_mask is not None:
            return self.block_mask

        from wan.modules.causal_model import CausalWanModel
        self.block_mask = CausalWanModel._prepare_blockwise_causal_attn_mask(
            device=device,
            num_frames=num_frames,
            frame_seqlen=frame_seqlen,
            num_frame_per_block=self.num_frame_per_block,
            local_attn_size=self.local_attn_size,
        )
        return self.block_mask

    def _get_block_mask_dual(self, num_frames: int, frame_seqlen: int, device) -> BlockMask:
        """Create/cache the dual-stream causal block mask.

        Token layout: [x_0..x_{N-1} | c_0..c_{N-1}]
        where N=num_frames, S=frame_seqlen.

        Causal rule: kv_frame <= q_frame (agnostic to x vs c stream).
        This is a 2×2 block-causal structure where each 'super-block' is S×S.
        """
        if self.block_mask_dual is not None:
            return self.block_mask_dual

        N = num_frames
        S = frame_seqlen
        total_length = 2 * N * S
        padded_length = math.ceil(total_length / 128) * 128 - total_length

        def dual_stream_causal_mask(b, h, q_idx, kv_idx):
            # Frame index for query (0..N-1 for x, 0..N-1 for c)
            q_frame = torch.where(q_idx < N * S, q_idx // S, (q_idx - N * S) // S)
            kv_frame = torch.where(kv_idx < N * S, kv_idx // S, (kv_idx - N * S) // S)
            return kv_frame <= q_frame

        self.block_mask_dual = create_block_mask(
            dual_stream_causal_mask,
            B=None, H=None,
            Q_LEN=total_length + padded_length,
            KV_LEN=total_length + padded_length,
            device=device,
            _compile=True,
        )

        if not dist.is_initialized() or dist.get_rank() == 0:
            print(f"[CausalMMDiTModel] Created dual-stream block mask: "
                  f"N={N}, S={S}, total={total_length}")
            print(self.block_mask_dual)

        return self.block_mask_dual

    # ------------------------------------------------------------------
    # Unpatchify
    # ------------------------------------------------------------------

    def unpatchify(self, x_list, grid_sizes):
        c = self.out_dim
        out = []
        for u, v in zip(x_list, grid_sizes.tolist()):
            u = u[:math.prod(v)].view(*v, *self.patch_size, c)
            u = torch.einsum('fhwpqrc->cfphqwr', u)
            u = u.reshape(c, *[i * j for i, j in zip(v, self.patch_size)])
            out.append(u)
        return out

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        x,                                        # [B, C, F, H, W] (C-first) or list
        t,                                        # [B, F] timesteps
        context,                                  # [B, text_len, text_dim]
        seq_len=None,
        grid_sizes=None,
        freqs=None,
        cond_x=None,                              # [B, C, F, H, W] condition latent
        kv_cache=None,                            # list[dict] len=num_layers
        crossattn_cache=None,
        current_start: int = 0,
        cache_start=None,
        allow_kv_grad: bool = False,
        **kwargs,
    ):
        if kv_cache is not None:
            return self._forward_inference(
                x, t, context, seq_len=seq_len,
                cond_x=cond_x, kv_cache=kv_cache, crossattn_cache=crossattn_cache,
                current_start=current_start, allow_kv_grad=allow_kv_grad,
            )
        return self._forward_train(x, t, context, cond_x=cond_x)

    # ------------------------------------------------------------------
    # Training forward
    # ------------------------------------------------------------------

    def _forward_train(self, x, t, context, cond_x=None):
        """Full-sequence FlexAttention training forward.

        Args:
            x:      [B, C, F, H, W] noisy latent (C-first).
            t:      [B, F] timesteps per frame.
            context:[B, text_len, text_dim] raw text embeddings.
            cond_x: [B, C, F, H, W] clean condition latent (C-first), or None.
        Returns:
            flow_pred: list of [C, F', H', W'] tensors (one per batch item), stacked
                       → [B, C, F', H', W'].
        """
        device = self.patch_embedding.weight.device
        if self.freqs.device != device:
            self.freqs = self.freqs.to(device)

        # ── Patchify x ────────────────────────────────────────────────────────
        x_patches = [self.patch_embedding(u.unsqueeze(0)) for u in x]   # list of [1, dim, F, H', W']
        grid_sizes = torch.stack(
            [torch.tensor(u.shape[2:], dtype=torch.long) for u in x_patches])  # [B, 3]
        x_patches = [u.flatten(2).transpose(1, 2) for u in x_patches]  # list of [1, S, dim]
        seq_lens = torch.tensor([u.size(1) for u in x_patches], dtype=torch.long)
        x_tok = torch.cat([
            torch.cat([u, u.new_zeros(1, seq_lens[0] - u.size(1), u.size(2))], dim=1)
            for u in x_patches
        ])  # [B, S, dim]
        S = seq_lens[0].item()
        num_frames = t.shape[1]
        frame_seqlen = S // num_frames

        # ── Patchify cond_x ───────────────────────────────────────────────────
        if cond_x is not None:
            c_patches = [self.cond_patch_embedding(u.unsqueeze(0)) for u in cond_x]
            c_patches = [u.flatten(2).transpose(1, 2) for u in c_patches]
            c_tok = torch.cat([
                torch.cat([u, u.new_zeros(1, seq_lens[0] - u.size(1), u.size(2))], dim=1)
                for u in c_patches
            ])  # [B, S, dim]
        else:
            c_tok = torch.zeros_like(x_tok)

        # ── Time embeddings ───────────────────────────────────────────────────
        B = x_tok.shape[0]
        e = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, t.flatten()).type_as(x_tok))  # [B*F, dim]
        e0_x = self.time_projection(e).unflatten(1, (6, self.dim)).unflatten(0, t.shape)   # [B, F, 6, dim]
        e0_c = self.cond_time_projection(e).unflatten(1, (6, self.dim)).unflatten(0, t.shape)  # [B, F, 6, dim]

        # ── Text context ──────────────────────────────────────────────────────
        context_lens = None
        context = self.text_embedding(torch.stack([
            torch.cat([u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
            for u in context
        ]))  # [B, text_len, dim]

        # ── FlexAttention masks (skipped when use_flash_attn=True) ────────────
        if self.use_flash_attn:
            block_mask = None
            block_mask_dual = None
        else:
            block_mask = self._get_block_mask(num_frames, frame_seqlen, device)
            block_mask_dual = self._get_block_mask_dual(num_frames, frame_seqlen, device)

        # ── Iterate blocks ────────────────────────────────────────────────────
        def _gc_double(blk, x_, c_, ex_, ec_, sl_, gs_, fr_, ctx_, ctxl_, bmd_):
            return blk(x_, c_, ex_, ec_, sl_, gs_, fr_, ctx_, ctxl_,
                       block_mask_dual=bmd_, kv_cache=None)

        def _gc_single(blk, x_, ex_, sl_, gs_, fr_, ctx_, ctxl_, bm_):
            return blk(x_, ex_, sl_, gs_, fr_, ctx_, ctxl_, bm_)

        for i, block in enumerate(self.blocks):
            if i in self.double_stream_layers:
                if torch.is_grad_enabled() and self.gradient_checkpointing:
                    x_tok, c_tok = torch.utils.checkpoint.checkpoint(
                        _gc_double, block, x_tok, c_tok, e0_x, e0_c,
                        seq_lens, grid_sizes, self.freqs, context, context_lens,
                        block_mask_dual, use_reentrant=False,
                    )
                else:
                    x_tok, c_tok = block(
                        x_tok, c_tok, e0_x, e0_c,
                        seq_lens, grid_sizes, self.freqs, context, context_lens,
                        block_mask_dual=block_mask_dual, kv_cache=None,
                    )
            else:
                if torch.is_grad_enabled() and self.gradient_checkpointing:
                    x_tok = torch.utils.checkpoint.checkpoint(
                        _gc_single, block, x_tok, e0_x,
                        seq_lens, grid_sizes, self.freqs, context, context_lens,
                        block_mask, use_reentrant=False,
                    )
                else:
                    x_tok = block(
                        x_tok, e0_x, seq_lens, grid_sizes, self.freqs,
                        context, context_lens, block_mask,
                    )

        # ── Head (x-stream only) ──────────────────────────────────────────────
        # CausalHead.forward(x, e) where e: [B, F, 1, dim]
        e_head = e.unflatten(0, t.shape).unsqueeze(2)  # [B, F, 1, dim]
        x_tok = self.head(x_tok, e_head)
        return torch.stack(self.unpatchify(x_tok, grid_sizes))

    # ------------------------------------------------------------------
    # Inference forward (KV cache)
    # ------------------------------------------------------------------

    def _forward_inference(
        self, x, t, context, seq_len=None,
        cond_x=None, kv_cache=None, crossattn_cache=None,
        current_start: int = 0, allow_kv_grad: bool = False,
    ):
        """Autoregressive generation forward with KV caching.

        Args:
            x:      [B, C, F, H, W] current block's noisy latent (C-first).
            t:      [B, F] timesteps for current block.
            context:[B, text_len, text_dim] raw text embeddings.
            cond_x: [B, C, F, H, W] condition latent for current block, or None.
            kv_cache: list[dict] of length num_layers with dual sub-buffers at
                      double-stream blocks.
            crossattn_cache: list[dict] of length num_layers.
            current_start: token offset for RoPE (= frame_idx * frame_seqlen).
            allow_kv_grad: whether KV writes retain gradients.
        Returns:
            flow_pred: stacked [B, C, F', H', W'].
        """
        device = self.patch_embedding.weight.device
        if self.freqs.device != device:
            self.freqs = self.freqs.to(device)

        # Patchify x per sample (matches CausalWanModel._forward_inference)
        x_patches = [self.patch_embedding(u.unsqueeze(0)) for u in x]
        grid_sizes = torch.stack(
            [torch.tensor(u.shape[2:], dtype=torch.long) for u in x_patches])
        x_patches = [u.flatten(2).transpose(1, 2) for u in x_patches]
        seq_lens = torch.tensor([u.size(1) for u in x_patches], dtype=torch.long)
        x_tok = torch.cat(x_patches)  # [B, S_block, dim]

        # Patchify cond_x
        if cond_x is not None:
            c_patches = [self.cond_patch_embedding(u.unsqueeze(0)) for u in cond_x]
            c_patches = [u.flatten(2).transpose(1, 2) for u in c_patches]
            c_tok = torch.cat(c_patches)  # [B, S_block, dim]
        else:
            c_tok = torch.zeros_like(x_tok)

        num_frames = t.shape[1]
        B = x_tok.shape[0]

        # Time embeddings
        e = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, t.flatten()).type_as(x_tok))
        e0_x = self.time_projection(e).unflatten(1, (6, self.dim)).unflatten(0, t.shape)
        e0_c = self.cond_time_projection(e).unflatten(1, (6, self.dim)).unflatten(0, t.shape)

        # Text context
        context_lens = None
        context = self.text_embedding(torch.stack([
            torch.cat([u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
            for u in context
        ]))

        # Iterate blocks with KV cache
        def _double_forward(blk, x_, c_, ex_, ec_, sl_, gs_, fr_, ctx_, ctxl_, kvc_, cac_, cs_, akv_):
            return blk(x_, c_, ex_, ec_, sl_, gs_, fr_, ctx_, ctxl_,
                       block_mask_dual=None, kv_cache=kvc_, crossattn_cache=cac_,
                       current_start=cs_, allow_kv_grad=akv_)

        def _single_forward(blk, x_, ex_, sl_, gs_, fr_, ctx_, ctxl_, kvc_, cac_, cs_):
            return blk(x_, ex_, sl_, gs_, fr_, ctx_, ctxl_,
                       block_mask=None, kv_cache=kvc_, crossattn_cache=cac_,
                       current_start=cs_)

        use_ckpt = torch.is_grad_enabled() and self.gradient_checkpointing
        for i, block in enumerate(self.blocks):
            ca_cache = crossattn_cache[i] if crossattn_cache is not None else None
            if i in self.double_stream_layers:
                # TODO: investigate this: during training ca_cache can't be passed to avoid CheckpointError: A different number of tensors was saved during the original forward and recomputation.
                if use_ckpt:
                    x_tok, c_tok = torch.utils.checkpoint.checkpoint(
                        _double_forward, block, x_tok, c_tok, e0_x, e0_c,
                        seq_lens, grid_sizes, self.freqs, context, context_lens,
                        kv_cache[i], None, current_start, allow_kv_grad,
                        use_reentrant=False,
                    )
                else:
                    x_tok, c_tok = block(
                        x_tok, c_tok, e0_x, e0_c,
                        seq_lens, grid_sizes, self.freqs, context, context_lens,
                        block_mask_dual=None,
                        kv_cache=kv_cache[i],
                        crossattn_cache=ca_cache,
                        current_start=current_start,
                        allow_kv_grad=allow_kv_grad,
                    )
            else:
                if use_ckpt:
                    x_tok = torch.utils.checkpoint.checkpoint(
                        _single_forward, block, x_tok, e0_x,
                        seq_lens, grid_sizes, self.freqs, context, context_lens,
                        kv_cache[i], None, current_start,
                        use_reentrant=False,
                    )
                else:
                    x_tok = block(
                        x_tok, e0_x, seq_lens, grid_sizes, self.freqs,
                        context, context_lens, block_mask=None,
                        kv_cache=kv_cache[i],
                        crossattn_cache=ca_cache,
                        current_start=current_start,
                    )

        # Head
        e_head = e.unflatten(0, t.shape).unsqueeze(2)
        x_tok = self.head(x_tok, e_head)
        return torch.stack(self.unpatchify(x_tok, grid_sizes))


# ---------------------------------------------------------------------------
# Weight loading
# ---------------------------------------------------------------------------

def load_causal_mmdit_weights(
    model: CausalMMDiTModel,
    state_dict: dict,
    strict: bool = False,
) -> None:
    """Load a CausalWanModel checkpoint into CausalMMDiTModel.

    Mapping logic
    -------------
    If state_dict already contains 'cond_' keys → full MMDiT checkpoint, load directly.

    Otherwise (CausalWanModel checkpoint):
      • Regular blocks (CausalWanAttentionBlock): direct name match.
      • Double-stream blocks x-stream: direct name match.
      • Double-stream blocks c-stream: clone from x-stream (prefix 'cond_').
      • cond_patch_embedding ← clone of patch_embedding.
      • cond_time_projection ← clone of time_projection.

    Non-strict mode (default): print MISSING/UNEXPECTED key warnings, continue.
    Strict mode: raise on any mismatch.
    """
    own_state = model.state_dict()

    # Detect if this is already a full MMDiT checkpoint
    has_cond_keys = any(k for k in state_dict if 'cond_' in k)
    if has_cond_keys:
        # Full MMDiT checkpoint — load directly
        new_state = {k: v for k, v in state_dict.items() if k in own_state}
    else:
        # CausalWanModel checkpoint — copy x-stream, clone c-stream
        new_state = {}

        # Step 1: copy all directly matching weights
        for k, v in state_dict.items():
            if k in own_state:
                new_state[k] = v

        # Step 2: clone c-stream from x-stream for double-stream blocks
        for layer_idx in model.double_stream_layers:
            prefix = f'blocks.{layer_idx}.'
            for k, v in state_dict.items():
                if not k.startswith(prefix):
                    continue
                suffix = k[len(prefix):]
                # x-stream sub-module names → c-stream with 'cond_' prefix
                # e.g. 'self_attn.q.weight' → 'cond_self_attn.q.weight'
                #      'modulation'         → 'cond_modulation'
                #      'norm1.weight'       → 'cond_norm1.weight'
                c_key = prefix + 'cond_' + suffix
                if c_key in own_state:
                    new_state[c_key] = v.clone()

        # Step 3: clone cond_patch_embedding ← patch_embedding
        for suffix in ('weight', 'bias'):
            src = f'patch_embedding.{suffix}'
            dst = f'cond_patch_embedding.{suffix}'
            if src in state_dict and dst in own_state:
                new_state[dst] = state_dict[src].clone()

        # Step 4: clone cond_time_projection ← time_projection
        # time_projection is Sequential(SiLU, Linear) — Linear is at index 1
        for suffix in ('1.weight', '1.bias'):
            src = f'time_projection.{suffix}'
            dst = f'cond_time_projection.{suffix}'
            if src in state_dict and dst in own_state:
                new_state[dst] = state_dict[src].clone()

    missing = [k for k in own_state if k not in new_state]
    unexpected = [k for k in new_state if k not in own_state]

    if strict:
        model.load_state_dict(new_state, strict=True)
    else:
        if missing or unexpected:
            print("WARNING: Non-strict checkpoint load for CausalMMDiTModel")
            for k in missing:
                print(f"  MISSING:    {k}")
            for k in unexpected:
                print(f"  UNEXPECTED: {k}")
        model.load_state_dict(new_state, strict=False)
