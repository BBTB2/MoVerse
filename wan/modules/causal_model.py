from wan.modules.attention import attention
from wan.modules.model import (
    WanRMSNorm,
    rope_apply,
    WanLayerNorm,
    WAN_CROSSATTENTION_CLASSES,
    rope_params,
    MLPProj,
    sinusoidal_embedding_1d
)
from torch.nn.attention.flex_attention import create_block_mask, flex_attention
from diffusers.configuration_utils import ConfigMixin, register_to_config
from torch.nn.attention.flex_attention import BlockMask
from diffusers.models.modeling_utils import ModelMixin
import torch.nn as nn
import torch
import math
import torch.distributed as dist

# wan 1.3B model has a weird channel / head configurations and require max-autotune to work with flexattention
# see https://github.com/pytorch/pytorch/issues/133254
# change to default for other models
flex_attention = torch.compile(
    flex_attention, dynamic=False, mode="max-autotune-no-cudagraphs")


def causal_rope_apply(x, grid_sizes, freqs, start_frame=0, compute_dtype=torch.float64):
    n, c = x.size(2), x.size(3) // 2

    # split freqs
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)

    # loop over samples
    output = []

    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        seq_len = f * h * w

        # precompute multipliers. compute_dtype controls RoPE precision (float64 default,
        # float32 ~2x faster with a tiny precision change — see rope_precision config).
        x_i = torch.view_as_complex(x[i, :seq_len].to(compute_dtype).reshape(
            seq_len, n, -1, 2))
        freqs_i = torch.cat([
            freqs[0][start_frame:start_frame + f].view(f, 1, 1, -1).expand(f, h, w, -1),
            freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
        ],
            dim=-1).reshape(seq_len, 1, -1)

        # apply rotary embedding
        x_i = torch.view_as_real(x_i * freqs_i.to(x_i.dtype)).flatten(2)
        x_i = torch.cat([x_i, x[i, seq_len:]])

        # append to collection
        output.append(x_i)
    return torch.stack(output).type_as(x)


def causal_rope_apply_shared_height(x, grid_sizes, freqs, start_frame=0, compute_dtype=torch.float64):
    """
    Causal RoPE variant for height-doubled (height-concatenated conditioning) inputs.

    When the model input is [noisy | cond] concatenated along the height dimension,
    grid_sizes will contain h_doubled = 2 * h_original. This function assigns
    shared/identical height positions to both halves so that position i and
    position i + h_original receive the same height embedding.

    This is the causal counterpart of rope_apply_shared used in
    generate_conditional_ode_pairs.py for the non-causal model.

    Args:
        x (Tensor): Shape [B, L, num_heads, head_dim]
        grid_sizes (Tensor): Shape [B, 3] — (F_patches, H_doubled_patches, W_patches)
        freqs (Tensor): Precomputed RoPE frequencies [max_len, head_dim // 2]
        start_frame (int): Frame offset for causal generation with KV cache
    """
    n, c = x.size(2), x.size(3) // 2

    # split freqs into temporal, height, width components
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)

    output = []
    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        seq_len = f * h * w
        sh = h // 2  # original height patch count before doubling

        x_i = torch.view_as_complex(
            x[i, :seq_len].to(compute_dtype).reshape(seq_len, n, -1, 2)
        )

        # Build freqs for one half: [f, sh, w, C_freq]
        single_half_freqs = torch.cat([
            freqs[0][start_frame:start_frame + f].view(f, 1, 1, -1).expand(f, sh, w, -1),
            freqs[1][:sh].view(1, sh, 1, -1).expand(f, sh, w, -1),
            freqs[2][:w].view(1, 1, w, -1).expand(f, sh, w, -1),
        ], dim=-1)  # [f, sh, w, C_freq]

        # Duplicate along height: both halves get identical positions 0..sh-1
        shared_freqs = torch.cat([single_half_freqs, single_half_freqs], dim=1)  # [f, 2*sh, w, C_freq]
        freqs_i = shared_freqs.reshape(seq_len, 1, -1)

        x_i = torch.view_as_real(x_i * freqs_i.to(device=x_i.device, dtype=x_i.dtype)).flatten(2)
        x_i = torch.cat([x_i, x[i, seq_len:]])
        output.append(x_i)

    return torch.stack(output).type_as(x)



def parse_memrope_cfg(raw):
    """Normalize a MemRoPE config (dict / DictConfig / bool) into a validated dict.

    Returns ``None`` when disabled, else ``{"gen": {...}, "cond": {...}}`` where each half
    has keys sink, long_mem, short_mem, local, alpha_long, alpha_short. Each tier is in
    *frame* units; long_mem/short_mem are 0 or 1 (full-frame EMA streams). Defaults follow
    the paper-ish setting (sink=3, 1 long + 1 short memory, local=5, αL=0.01, αS=0.1).
    """
    if raw is None:
        return None
    if isinstance(raw, bool):
        if not raw:
            return None
        raw = {}

    def _get(d, key, default):
        try:
            val = d[key] if key in d else default
        except TypeError:
            val = getattr(d, key, default)
        return default if val is None else val

    if not _get(raw, "enabled", True):
        return None

    def _half(h):
        h = h if h is not None else {}
        sink = int(_get(h, "sink", 3))
        long_mem = int(_get(h, "long_mem", 1))
        short_mem = int(_get(h, "short_mem", 1))
        local = int(_get(h, "local", 5))
        assert local >= 1, "memrope: local window must be >= 1 frame"
        assert sink >= 0, "memrope: sink must be >= 0"
        assert long_mem in (0, 1) and short_mem in (0, 1), \
            "memrope: long_mem/short_mem must be 0 or 1 (full-frame EMA streams)"
        return {
            "sink": sink,
            "long_mem": long_mem,
            "short_mem": short_mem,
            "local": local,
            "alpha_long": float(_get(h, "alpha_long", 0.01)),
            "alpha_short": float(_get(h, "alpha_short", 0.1)),
        }

    gen = _half(_get(raw, "gen", {}))
    cond = _half(_get(raw, "cond", {}))
    # When gen and cond configs are identical we use the FUSED fast path: a single
    # [gen|cond] full-frame three-tier cache with one shared-height RoPE call (instead of
    # splitting each frame, roping the two halves separately, and re-interleaving). This is
    # numerically identical to the split path when configs match (per-location EMA on the
    # concatenated halves == per-half EMA with the same alpha) but much cheaper.
    return {"gen": gen, "cond": cond, "fused": gen == cond}


class CausalWanSelfAttention(nn.Module):

    def __init__(self,
                 dim,
                 num_heads,
                 local_attn_size=-1,
                 sink_size=0,
                 qk_norm=True,
                 eps=1e-6,
                 rope_fn=None,
                 causal_rope_fn=None,
                 use_flash_attn=False):
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.local_attn_size = local_attn_size
        self.sink_size = sink_size
        self.qk_norm = qk_norm
        self.eps = eps
        self.use_flash_attn = use_flash_attn
        # MemRoPE Phase 1 — Online RoPE Indexing. When True (set via
        # CausalWanModel.set_online_rope), the AR/KV-cache path stores RAW (un-roped)
        # keys and re-applies RoPE at attention time with a compact, contiguous-from-0
        # block-relative temporal index. This keeps indices within the trained range
        # regardless of total video length (enables >1024-frame rollout). Default False
        # → original behavior (keys stored roped with absolute index).
        self.online_rope_indexing = False
        # MemRoPE Phase 2 — dual-EMA Memory Tokens. None → disabled. When set (via
        # CausalWanModel.set_memrope) to a dict {"gen": {...}, "cond": {...}}, the AR path
        # uses a split gen/cond three-tier cache [sink | µ_L | µ_S | local] with per-half
        # config and full-frame EMA memory. Supersedes online_rope_indexing. Requires
        # height-concat conditioning (frame_seq_length doubled). See SUMMARY_MEMROPE.md.
        self.memrope_cfg = None
        # RoPE compute precision for the AR/KV-cache rope calls (online + memrope paths).
        # float64 = original behavior (exact); float32 ≈ 2x faster rope, tiny precision
        # change. Set via CausalWanModel.set_rope_precision from the rope_precision config.
        self.rope_compute_dtype = torch.float64
        # This will be reset to depend on actual input sequence length, see pipeline/causal_base.py
        self.max_attention_size = 32760 if local_attn_size == -1 else local_attn_size * 1560
        print(f"Initialized CausalWanSelfAttention with max_attention_size={self.max_attention_size} (local_attn_size={local_attn_size}), sink_size={sink_size}")
        # Explicit rope dependencies — avoids monkey-patching module-level globals.
        # rope_fn is used in the training path (kv_cache=None).
        # causal_rope_fn is used in the inference path (kv_cache≠None).
        self.rope_fn = rope_fn if rope_fn is not None else rope_apply
        self.causal_rope_fn = causal_rope_fn if causal_rope_fn is not None else causal_rope_apply

        # layers
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

    def forward(
        self,
        x,
        seq_lens,
        grid_sizes,
        freqs,
        block_mask,
        kv_cache=None,
        current_start=0,
        cache_start=None
    ):
        r"""
        Args:
            x(Tensor): Shape [B, L, num_heads, C / num_heads]
            seq_lens(Tensor): Shape [B]
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
            block_mask (BlockMask)
            kv_cache (dict): {"k": k_cache (shape of [B, kv_cache_size, num_heads, head_dim]), "v": v_cache, "local_end_index": local_end_index, "global_end_index": global_end_index}
        """
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim
        if cache_start is None:
            cache_start = current_start

        # query, key, value function
        def qkv_fn(x):
            q = self.norm_q(self.q(x)).view(b, s, n, d)
            k = self.norm_k(self.k(x)).view(b, s, n, d)
            v = self.v(x).view(b, s, n, d)
            return q, k, v

        q, k, v = qkv_fn(x)

        if kv_cache is None:
            # if it is teacher forcing training?
            is_tf = (s == seq_lens[0].item() * 2)
            if is_tf:
                q_chunk = torch.chunk(q, 2, dim=1)
                k_chunk = torch.chunk(k, 2, dim=1)
                roped_query = []
                roped_key = []
                # rope should be same for clean and noisy parts
                for ii in range(2):
                    rq = self.rope_fn(q_chunk[ii], grid_sizes, freqs).type_as(v)
                    rk = self.rope_fn(k_chunk[ii], grid_sizes, freqs).type_as(v)
                    roped_query.append(rq)
                    roped_key.append(rk)

                roped_query = torch.cat(roped_query, dim=1)
                roped_key = torch.cat(roped_key, dim=1)

                padded_length = math.ceil(q.shape[1] / 128) * 128 - q.shape[1]
                padded_roped_query = torch.cat(
                    [roped_query,
                     torch.zeros([q.shape[0], padded_length, q.shape[2], q.shape[3]],
                                 device=q.device, dtype=v.dtype)],
                    dim=1
                )

                padded_roped_key = torch.cat(
                    [roped_key, torch.zeros([k.shape[0], padded_length, k.shape[2], k.shape[3]],
                                            device=k.device, dtype=v.dtype)],
                    dim=1
                )

                padded_v = torch.cat(
                    [v, torch.zeros([v.shape[0], padded_length, v.shape[2], v.shape[3]],
                                    device=v.device, dtype=v.dtype)],
                    dim=1
                )

                x = flex_attention(
                    query=padded_roped_query.transpose(2, 1),
                    key=padded_roped_key.transpose(2, 1),
                    value=padded_v.transpose(2, 1),
                    block_mask=block_mask
                )[:, :, :-padded_length].transpose(2, 1)

            else:
                roped_query = self.rope_fn(q, grid_sizes, freqs).type_as(v)
                roped_key = self.rope_fn(k, grid_sizes, freqs).type_as(v)

                if self.use_flash_attn:
                    # Per-frame loop with flash attention: frame f attends to frames 0..f
                    num_frames = grid_sizes[0][0].item()
                    frame_seqlen = s // num_frames
                    out_parts = []
                    for f in range(num_frames):
                        f_start = f * frame_seqlen
                        f_end = (f + 1) * frame_seqlen
                        out_parts.append(attention(
                            roped_query[:, f_start:f_end],
                            roped_key[:, :f_end],
                            v[:, :f_end],
                        ))
                    x = torch.cat(out_parts, dim=1)
                else:
                    # pad to multiple of 128 for flash attention efficiency (padding is masked out, so it won't affect results)
                    padded_length = math.ceil(q.shape[1] / 128) * 128 - q.shape[1]
                    padded_roped_query = torch.cat(
                        [roped_query,
                         torch.zeros([q.shape[0], padded_length, q.shape[2], q.shape[3]],
                                     device=q.device, dtype=v.dtype)],
                        dim=1
                    )

                    padded_roped_key = torch.cat(
                        [roped_key, torch.zeros([k.shape[0], padded_length, k.shape[2], k.shape[3]],
                                                device=k.device, dtype=v.dtype)],
                        dim=1
                    )

                    padded_v = torch.cat(
                        [v, torch.zeros([v.shape[0], padded_length, v.shape[2], v.shape[3]],
                                        device=v.device, dtype=v.dtype)],
                        dim=1
                    )

                    x = flex_attention(
                        query=padded_roped_query.transpose(2, 1),
                        key=padded_roped_key.transpose(2, 1),
                        value=padded_v.transpose(2, 1),
                        block_mask=block_mask
                    )[:, :, :-padded_length].transpose(2, 1)
        else:
            if self.memrope_cfg is not None:
                # MemRoPE Phase 2 — split gen/cond three-tier cache with dual-EMA memory.
                # Self-contained path: stores raw keys, manages [sink|µL|µS|local], re-ropes
                # block-relative, attends, returns. Fused fast path when gen==cond (single
                # [gen|cond] buffer, one rope call, cached static rope); split path for
                # asymmetric configs. See SUMMARY_MEMROPE.md.
                if self.memrope_cfg.get("fused", False):
                    x = self._attend_memrope_fused(q, k, v, grid_sizes, freqs, kv_cache, current_start)
                else:
                    x = self._attend_memrope_split(q, k, v, grid_sizes, freqs, kv_cache, current_start)
                x = x.flatten(2)
                x = self.o(x)
                return x
            # AR generation with KV cache. Mask not needed and can theoretically support infinite rollout
            frame_seqlen = math.prod(grid_sizes[0][1:]).item() # 1560 (3120 in height-concat)
            current_start_frame = current_start // frame_seqlen # 0
            num_new_tokens = q.shape[1] # 1560/3120/...
            current_end = current_start + num_new_tokens
            sink_tokens = self.sink_size * frame_seqlen # frame_seqlen * sink_size

            if self.online_rope_indexing:
                # MemRoPE Online RoPE Indexing: store RAW keys; RoPE is (re)applied at
                # attention time below with a compact block-relative index. The query is
                # roped after we know the cache fill level. See [SUMMARY_MEMROPE.md].
                key_to_store = k
            else:
                # Original behavior: rope with the absolute frame index and store roped.
                roped_query = self.causal_rope_fn(
                    q, grid_sizes, freqs, start_frame=current_start_frame,
                    compute_dtype=self.rope_compute_dtype).type_as(v)
                key_to_store = self.causal_rope_fn(
                    k, grid_sizes, freqs, start_frame=current_start_frame,
                    compute_dtype=self.rope_compute_dtype).type_as(v)

            # If we are using local attention and the current KV cache size is larger than the local attention size, we need to truncate the KV cache
            kv_cache_size = kv_cache["k"].shape[1] # kv_cache_size, 32760 for full attn, 1560*local_attn_size for local attn
            # kv_cache["local_end_index"] and kv_cache["global_end_index"] initialized to 0
            if self.local_attn_size != -1 and (current_end > kv_cache["global_end_index"].item()) and (
                    num_new_tokens + kv_cache["local_end_index"].item() > kv_cache_size):
                # KV eviction and rolling when local attention is enabled
                # Calculate the number of new tokens added in this step
                # Shift existing cache content left to discard oldest tokens
                # Clone the source slice to avoid overlapping memory error
                num_evicted_tokens = num_new_tokens + kv_cache["local_end_index"].item() - kv_cache_size # always 1560
                num_rolled_tokens = kv_cache["local_end_index"].item() - num_evicted_tokens - sink_tokens # always 1560
                kv_cache["k"][:, sink_tokens:sink_tokens + num_rolled_tokens] = \
                    kv_cache["k"][:, sink_tokens + num_evicted_tokens:sink_tokens + num_evicted_tokens + num_rolled_tokens].clone()
                kv_cache["v"][:, sink_tokens:sink_tokens + num_rolled_tokens] = \
                    kv_cache["v"][:, sink_tokens + num_evicted_tokens:sink_tokens + num_evicted_tokens + num_rolled_tokens].clone()
                # Insert the new keys/values at the end
                local_end_index = kv_cache["local_end_index"].item() + current_end - \
                    kv_cache["global_end_index"].item() - num_evicted_tokens
                local_start_index = local_end_index - num_new_tokens
                kv_cache["k"][:, local_start_index:local_end_index] = key_to_store
                kv_cache["v"][:, local_start_index:local_end_index] = v
            else:
                # Assign new keys/values directly up to current_end
                local_end_index = kv_cache["local_end_index"].item() + current_end - kv_cache["global_end_index"].item() # 1560, 3120, ...
                local_start_index = local_end_index - num_new_tokens # 0, 1560, ...
                kv_cache["k"][:, local_start_index:local_end_index] = key_to_store # this is current layer's kvcache
                kv_cache["v"][:, local_start_index:local_end_index] = v

            if self.online_rope_indexing:
                # Re-rope the whole cached window with a compact, contiguous-from-0 index.
                # The buffer is physically [sink frames | rolled local window], so sink
                # gets indices 0..S-1 and the local window follows contiguously — never
                # exceeding local_attn_size, so RoPE indices stay in the trained range.
                n_cached_frames = local_end_index // frame_seqlen
                grid_window = grid_sizes.clone()
                grid_window[:, 0] = n_cached_frames
                cached_k = self.causal_rope_fn(
                    kv_cache["k"][:, 0:local_end_index], grid_window, freqs, start_frame=0,
                    compute_dtype=self.rope_compute_dtype).type_as(v)
                # The current block sits at the end of the window: compact start frame =
                # number of frames already present before this block's tokens.
                q_start_frame = (local_end_index - num_new_tokens) // frame_seqlen
                roped_query = self.causal_rope_fn(
                    q, grid_sizes, freqs, start_frame=q_start_frame,
                    compute_dtype=self.rope_compute_dtype).type_as(v)
            else:
                cached_k = kv_cache["k"][:, 0:local_end_index]

            x = attention(
                roped_query,
                cached_k,
                kv_cache["v"][:, 0:local_end_index]
            )
            kv_cache["global_end_index"].fill_(current_end)
            kv_cache["local_end_index"].fill_(local_end_index)

        # output
        x = x.flatten(2)
        x = self.o(x)
        return x

    # ------------------------------------------------------------------
    # MemRoPE Phase 2 — split gen/cond three-tier cache with dual-EMA memory
    # ------------------------------------------------------------------
    def _attend_memrope_split(self, q, k, v, grid_sizes, freqs, kv_cache, current_start):
        """MemRoPE AR attention (SPLIT path): independent gen/cond caches (asymmetric cfg).

        Per height-concat frame the tokens are [gen 1560 | cond 1560]. We split q/k/v into
        the two halves, manage each half's own three-tier cache (raw keys, full-frame EMA
        memory), then re-apply RoPE block-relative (compact, contiguous-from-0) per half and
        attend the (re-interleaved) query against the concatenated [gen_keys | cond_keys].

        Args mirror the AR branch; returns x of shape [B, S, n_heads, head_dim].
        """
        b, s, n, d = q.shape
        f_full, h_full, w = (int(grid_sizes[0][0].item()),
                             int(grid_sizes[0][1].item()),
                             int(grid_sizes[0][2].item()))
        fs_full = h_full * w           # tokens per full (gen+cond) frame, e.g. 3120
        fs = fs_full // 2              # tokens per half-frame, e.g. 1560
        sh = h_full // 2              # single-half height patch count
        num_new_frames = s // fs_full
        cur_frame = current_start // fs_full
        cfg = self.memrope_cfg

        def split_half(t):
            t = t.view(b, num_new_frames, fs_full, n, d)
            gen = t[:, :, :fs].reshape(b, num_new_frames * fs, n, d)
            cond = t[:, :, fs:].reshape(b, num_new_frames * fs, n, d)
            return gen, cond

        q_gen, q_cond = split_half(q)
        k_gen, k_cond = split_half(k)
        v_gen, v_cond = split_half(v)

        is_new = cur_frame > kv_cache["committed_frame"]
        self._memrope_update_half(kv_cache["gen"], cfg["gen"], k_gen, v_gen, fs,
                                  num_new_frames, cur_frame, is_new)
        self._memrope_update_half(kv_cache["cond"], cfg["cond"], k_cond, v_cond, fs,
                                  num_new_frames, cur_frame, is_new)
        if is_new:
            kv_cache["committed_frame"] = cur_frame + num_new_frames - 1

        keys_gen, vals_gen, q_start_gen = self._memrope_assemble(
            kv_cache["gen"], cfg["gen"], fs, sh, w, freqs, grid_sizes, num_new_frames, v.dtype)
        keys_cond, vals_cond, q_start_cond = self._memrope_assemble(
            kv_cache["cond"], cfg["cond"], fs, sh, w, freqs, grid_sizes, num_new_frames, v.dtype)

        # RoPE the query halves with single-height grid + per-half compact temporal index.
        grid_q = grid_sizes.clone()
        grid_q[:, 0] = num_new_frames
        grid_q[:, 1] = sh
        roped_q_gen = causal_rope_apply(q_gen, grid_q, freqs, start_frame=q_start_gen,
                                        compute_dtype=self.rope_compute_dtype).type_as(v)
        roped_q_cond = causal_rope_apply(q_cond, grid_q, freqs, start_frame=q_start_cond,
                                         compute_dtype=self.rope_compute_dtype).type_as(v)
        # Re-interleave back to the original per-frame [gen | cond] token order so the
        # attention output matches the input token layout.
        roped_q = q.new_empty(b, num_new_frames, fs_full, n, d)
        roped_q[:, :, :fs] = roped_q_gen.view(b, num_new_frames, fs, n, d)
        roped_q[:, :, fs:] = roped_q_cond.view(b, num_new_frames, fs, n, d)
        roped_q = roped_q.reshape(b, s, n, d)

        keys = torch.cat([keys_gen, keys_cond], dim=1)
        vals = torch.cat([vals_gen, vals_cond], dim=1)
        return attention(roped_q, keys, vals)

    def _memrope_update_half(self, hc, cfg, k_half, v_half, fs, nf, cur_frame, is_new):
        """Push/overwrite the current block's frames into one half's three-tier cache.

        On a *new* frame: fill sink until full, then push into the local ring (evicting the
        oldest local frame into the dual-EMA memory). On re-denoise / clean-rerun of the
        current frame (same cur_frame): overwrite its existing slot in place.
        """
        S, L = cfg["sink"], cfg["local"]
        has_muL, has_muS = cfg["long_mem"] > 0, cfg["short_mem"] > 0
        aL, aS = cfg["alpha_long"], cfg["alpha_short"]
        for j in range(nf):
            kf = k_half[:, j * fs:(j + 1) * fs]
            vf = v_half[:, j * fs:(j + 1) * fs]
            f_idx = cur_frame + j
            if is_new:
                if hc["sink_count"] < S:
                    slot = hc["sink_count"]
                    hc["sink_k"][:, slot * fs:(slot + 1) * fs] = kf
                    hc["sink_v"][:, slot * fs:(slot + 1) * fs] = vf
                    hc["sink_count"] += 1
                else:
                    if hc["local_count"] == L:
                        # Evict oldest local frame → fold into full-frame dual EMA memory.
                        ek = hc["local_k"][:, 0:fs].clone()
                        ev = hc["local_v"][:, 0:fs].clone()
                        if has_muL:
                            if not hc["muL_init"]:
                                hc["muL_k"].copy_(ek); hc["muL_v"].copy_(ev); hc["muL_init"] = True
                            else:
                                hc["muL_k"].mul_(1 - aL).add_(ek, alpha=aL)
                                hc["muL_v"].mul_(1 - aL).add_(ev, alpha=aL)
                        if has_muS:
                            if not hc["muS_init"]:
                                hc["muS_k"].copy_(ek); hc["muS_v"].copy_(ev); hc["muS_init"] = True
                            else:
                                hc["muS_k"].mul_(1 - aS).add_(ek, alpha=aS)
                                hc["muS_v"].mul_(1 - aS).add_(ev, alpha=aS)
                        # Shift local window left by one frame to discard the evicted slot.
                        hc["local_k"][:, 0:(L - 1) * fs] = hc["local_k"][:, fs:L * fs].clone()
                        hc["local_v"][:, 0:(L - 1) * fs] = hc["local_v"][:, fs:L * fs].clone()
                        hc["local_count"] = L - 1
                    slot = hc["local_count"]
                    hc["local_k"][:, slot * fs:(slot + 1) * fs] = kf
                    hc["local_v"][:, slot * fs:(slot + 1) * fs] = vf
                    hc["local_count"] += 1
            else:
                # Overwrite the current frame's existing slot (re-denoise / clean rerun).
                if f_idx < S:
                    slot = f_idx
                    hc["sink_k"][:, slot * fs:(slot + 1) * fs] = kf
                    hc["sink_v"][:, slot * fs:(slot + 1) * fs] = vf
                else:
                    slot = hc["local_count"] - nf + j
                    hc["local_k"][:, slot * fs:(slot + 1) * fs] = kf
                    hc["local_v"][:, slot * fs:(slot + 1) * fs] = vf

    def _memrope_assemble(self, hc, cfg, fs, sh, w, freqs, grid_sizes, nf, dtype):
        """Concatenate one half's filled tiers, RoPE them compact-from-0, return keys/vals.

        Tier order [sink | µ_L | µ_S | local] mirrors temporal structure: sink anchors the
        earliest frames, memory holds the EMA-compressed history, local is the recent window
        (current frame last). Returns (roped_keys, raw_vals, q_start_frame) where q_start is
        the compact temporal index of the current frame (last `nf` frames of the sequence).
        """
        parts_k, parts_v = [], []
        if hc["sink_count"] > 0:
            parts_k.append(hc["sink_k"][:, 0:hc["sink_count"] * fs])
            parts_v.append(hc["sink_v"][:, 0:hc["sink_count"] * fs])
        if cfg["long_mem"] > 0 and hc["muL_init"]:
            parts_k.append(hc["muL_k"]); parts_v.append(hc["muL_v"])
        if cfg["short_mem"] > 0 and hc["muS_init"]:
            parts_k.append(hc["muS_k"]); parts_v.append(hc["muS_v"])
        if hc["local_count"] > 0:
            parts_k.append(hc["local_k"][:, 0:hc["local_count"] * fs])
            parts_v.append(hc["local_v"][:, 0:hc["local_count"] * fs])
        keys_raw = torch.cat(parts_k, dim=1)
        vals = torch.cat(parts_v, dim=1)
        n_frames = keys_raw.shape[1] // fs
        grid_win = grid_sizes.clone()
        grid_win[:, 0] = n_frames
        grid_win[:, 1] = sh
        keys = causal_rope_apply(keys_raw, grid_win, freqs, start_frame=0,
                                 compute_dtype=self.rope_compute_dtype).to(dtype)
        q_start = n_frames - nf
        return keys, vals, q_start

    # ---- FUSED fast path (symmetric gen==cond) ----
    def _attend_memrope_fused(self, q, k, v, grid_sizes, freqs, kv_cache, current_start):
        """Symmetric MemRoPE: single [gen|cond] full-frame three-tier cache.

        Numerically identical to the split path when gen==cond, but avoids splitting each
        frame, roping the two halves separately, and re-interleaving the query — only one
        shared-height RoPE per call. Two further optimizations (no quality impact):
          (a) the roped *static* prefix (sink|µL|µS|local-minus-current) doesn't change
              across a block's denoising steps, so it is roped once (on the first/`is_new`
              forward) and reused; only the current frame is re-roped each step.
          (b) results are written into preallocated `roped_keys_buf` / `vals_buf` to avoid
              per-step allocation.
        """
        s = q.shape[1]
        h_full, w = int(grid_sizes[0][1].item()), int(grid_sizes[0][2].item())
        ffs = h_full * w               # full-frame token count (gen+cond), e.g. 3120
        num_new = s // ffs
        cur_frame = current_start // ffs
        cfg = self.memrope_cfg["gen"]   # gen == cond in fused mode
        rope_dt = self.rope_compute_dtype

        is_new = cur_frame > kv_cache["committed_frame"]
        self._memrope_update_half(kv_cache, cfg, k, v, ffs, num_new, cur_frame, is_new)
        if is_new:
            kv_cache["committed_frame"] = cur_frame + num_new - 1

        n_mem = (1 if (cfg["long_mem"] > 0 and kv_cache["muL_init"]) else 0) \
              + (1 if (cfg["short_mem"] > 0 and kv_cache["muS_init"]) else 0)
        n_frames = kv_cache["sink_count"] + n_mem + kv_cache["local_count"]
        end = n_frames * ffs
        static_end = (n_frames - num_new) * ffs
        rk = kv_cache["roped_keys_buf"]
        vb = kv_cache["vals_buf"]

        # (a) Rope+cache the static prefix once per block (it is unchanged across the
        # block's denoising steps / clean rerun, which are the non-is_new forwards).
        if is_new and static_end > 0:
            static_k, static_v = self._memrope_gather_static(kv_cache, cfg, ffs, num_new, cur_frame)
            grid_s = grid_sizes.clone(); grid_s[:, 0] = n_frames - num_new
            rk[:, :static_end] = causal_rope_apply_shared_height(
                static_k, grid_s, freqs, start_frame=0, compute_dtype=rope_dt).to(v.dtype)
            vb[:, :static_end] = static_v

        # Current frame(s): the input k/v IS the current frame — rope fresh every step.
        grid_c = grid_sizes.clone(); grid_c[:, 0] = num_new
        rk[:, static_end:end] = causal_rope_apply_shared_height(
            k, grid_c, freqs, start_frame=n_frames - num_new, compute_dtype=rope_dt).to(v.dtype)
        vb[:, static_end:end] = v

        grid_q = grid_sizes.clone(); grid_q[:, 0] = num_new
        roped_q = causal_rope_apply_shared_height(
            q, grid_q, freqs, start_frame=n_frames - num_new, compute_dtype=rope_dt).to(v.dtype)
        return attention(roped_q, rk[:, :end], vb[:, :end])

    def _memrope_gather_static(self, hc, cfg, ffs, num_new, cur_frame):
        """Concatenate the static prefix (everything except the current num_new frames)."""
        S = cfg["sink"]
        pk, pv = [], []
        if cur_frame < S:
            # Still filling sink → static is the earlier sink frames only.
            upto = (hc["sink_count"] - num_new) * ffs
            if upto > 0:
                pk.append(hc["sink_k"][:, :upto]); pv.append(hc["sink_v"][:, :upto])
        else:
            if hc["sink_count"] > 0:
                e = hc["sink_count"] * ffs
                pk.append(hc["sink_k"][:, :e]); pv.append(hc["sink_v"][:, :e])
            if cfg["long_mem"] > 0 and hc["muL_init"]:
                pk.append(hc["muL_k"]); pv.append(hc["muL_v"])
            if cfg["short_mem"] > 0 and hc["muS_init"]:
                pk.append(hc["muS_k"]); pv.append(hc["muS_v"])
            upto = (hc["local_count"] - num_new) * ffs
            if upto > 0:
                pk.append(hc["local_k"][:, :upto]); pv.append(hc["local_v"][:, :upto])
        return torch.cat(pk, dim=1), torch.cat(pv, dim=1)


class CausalWanAttentionBlock(nn.Module):

    def __init__(self,
                 cross_attn_type,
                 dim,
                 ffn_dim,
                 num_heads,
                 local_attn_size=-1,
                 sink_size=0,
                 qk_norm=True,
                 cross_attn_norm=True,
                 eps=1e-6,
                 rope_fn=None,
                 causal_rope_fn=None,
                 use_flash_attn=False):
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.local_attn_size = local_attn_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        # layers
        self.norm1 = WanLayerNorm(dim, eps)
        self.self_attn = CausalWanSelfAttention(
            dim, num_heads, local_attn_size, sink_size, qk_norm, eps,
            rope_fn=rope_fn, causal_rope_fn=causal_rope_fn,
            use_flash_attn=use_flash_attn)
        self.norm3 = WanLayerNorm(
            dim, eps,
            elementwise_affine=True) if cross_attn_norm else nn.Identity()
        self.cross_attn = WAN_CROSSATTENTION_CLASSES[cross_attn_type](dim,
                                                                      num_heads,
                                                                      (-1, -1),
                                                                      qk_norm,
                                                                      eps)
        self.norm2 = WanLayerNorm(dim, eps)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim), nn.GELU(approximate='tanh'),
            nn.Linear(ffn_dim, dim))

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    def forward(
        self,
        x,
        e,
        seq_lens,
        grid_sizes,
        freqs,
        context,
        context_lens,
        block_mask,
        kv_cache=None,
        crossattn_cache=None,
        current_start=0,
        cache_start=None
    ):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
            e(Tensor): Shape [B, F, 6, C]
            seq_lens(Tensor): Shape [B], length of each sequence in batch
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
        """
        num_frames, frame_seqlen = e.shape[1], x.shape[1] // e.shape[1]
        # assert e.dtype == torch.float32
        # with amp.autocast(dtype=torch.float32):
        e = (self.modulation.unsqueeze(1) + e).chunk(6, dim=2)
        # assert e[0].dtype == torch.float32

        # self-attention
        y = self.self_attn(
            (self.norm1(x).unflatten(dim=1, sizes=(num_frames, frame_seqlen)) * (1 + e[1]) + e[0]).flatten(1, 2),
            seq_lens, grid_sizes,
            freqs, block_mask, kv_cache, current_start, cache_start)

        # with amp.autocast(dtype=torch.float32):
        x = x + (y.unflatten(dim=1, sizes=(num_frames, frame_seqlen)) * e[2]).flatten(1, 2)

        # cross-attention & ffn function
        def cross_attn_ffn(x, context, context_lens, e, crossattn_cache=None):
            x = x + self.cross_attn(self.norm3(x), context,
                                    context_lens, crossattn_cache=crossattn_cache)
            y = self.ffn(
                (self.norm2(x).unflatten(dim=1, sizes=(num_frames,
                 frame_seqlen)) * (1 + e[4]) + e[3]).flatten(1, 2)
            )
            # with amp.autocast(dtype=torch.float32):
            x = x + (y.unflatten(dim=1, sizes=(num_frames,
                     frame_seqlen)) * e[5]).flatten(1, 2)
            return x

        x = cross_attn_ffn(x, context, context_lens, e, crossattn_cache)
        return x


class CausalHead(nn.Module):

    def __init__(self, dim, out_dim, patch_size, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.patch_size = patch_size
        self.eps = eps

        # layers
        out_dim = math.prod(patch_size) * out_dim
        self.norm = WanLayerNorm(dim, eps)
        self.head = nn.Linear(dim, out_dim)

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, x, e):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            e(Tensor): Shape [B, F, 1, C]
        """
        # assert e.dtype == torch.float32
        # with amp.autocast(dtype=torch.float32):
        num_frames, frame_seqlen = e.shape[1], x.shape[1] // e.shape[1]
        e = (self.modulation.unsqueeze(1) + e).chunk(2, dim=2)
        x = (self.head(self.norm(x).unflatten(dim=1, sizes=(num_frames, frame_seqlen)) * (1 + e[1]) + e[0]))
        return x


class CausalWanModel(ModelMixin, ConfigMixin):
    r"""
    Wan diffusion backbone supporting both text-to-video and image-to-video.
    """

    ignore_for_config = [
        'patch_size', 'cross_attn_norm', 'qk_norm', 'text_dim',
        'rope_fn', 'causal_rope_fn',
    ]
    _no_split_modules = ['WanAttentionBlock']
    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(self,
                 model_type='t2v',
                 patch_size=(1, 2, 2),
                 text_len=512,
                 in_dim=16,
                 dim=2048,
                 ffn_dim=8192,
                 freq_dim=256,
                 text_dim=4096,
                 out_dim=16,
                 num_heads=16,
                 num_layers=32,
                 local_attn_size=-1,
                 sink_size=0,
                 qk_norm=True,
                 cross_attn_norm=True,
                 eps=1e-6,
                 rope_fn=None,
                 causal_rope_fn=None,
                 use_flash_attn=False):
        r"""
        Initialize the diffusion model backbone.

        Args:
            model_type (`str`, *optional*, defaults to 't2v'):
                Model variant - 't2v' (text-to-video) or 'i2v' (image-to-video)
            patch_size (`tuple`, *optional*, defaults to (1, 2, 2)):
                3D patch dimensions for video embedding (t_patch, h_patch, w_patch)
            text_len (`int`, *optional*, defaults to 512):
                Fixed length for text embeddings
            in_dim (`int`, *optional*, defaults to 16):
                Input video channels (C_in)
            dim (`int`, *optional*, defaults to 2048):
                Hidden dimension of the transformer
            ffn_dim (`int`, *optional*, defaults to 8192):
                Intermediate dimension in feed-forward network
            freq_dim (`int`, *optional*, defaults to 256):
                Dimension for sinusoidal time embeddings
            text_dim (`int`, *optional*, defaults to 4096):
                Input dimension for text embeddings
            out_dim (`int`, *optional*, defaults to 16):
                Output video channels (C_out)
            num_heads (`int`, *optional*, defaults to 16):
                Number of attention heads
            num_layers (`int`, *optional*, defaults to 32):
                Number of transformer blocks
            local_attn_size (`int`, *optional*, defaults to -1):
                Window size for temporal local attention (-1 indicates global attention)
            sink_size (`int`, *optional*, defaults to 0):
                Size of the attention sink, we keep the first `sink_size` frames unchanged when rolling the KV cache
            qk_norm (`bool`, *optional*, defaults to True):
                Enable query/key normalization
            cross_attn_norm (`bool`, *optional*, defaults to False):
                Enable cross-attention normalization
            eps (`float`, *optional*, defaults to 1e-6):
                Epsilon value for normalization layers
        """

        super().__init__()

        assert model_type in ['t2v', 'i2v']
        self.model_type = model_type

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
        self.sink_size = sink_size  # stored on top-level model so _forward_train's
                                    # TF block_mask construction can mirror the
                                    # rollout's KV-eviction window (RAVEN path).
                                    # Inner blocks already get sink_size via
                                    # CausalWanAttentionBlock(..., sink_size, ...).
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        # embeddings
        self.patch_embedding = nn.Conv3d(
            in_dim, dim, kernel_size=patch_size, stride=patch_size)
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim), nn.GELU(approximate='tanh'),
            nn.Linear(dim, dim))

        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.time_projection = nn.Sequential(
            nn.SiLU(), nn.Linear(dim, dim * 6))

        # blocks
        cross_attn_type = 't2v_cross_attn' if model_type == 't2v' else 'i2v_cross_attn'
        self.blocks = nn.ModuleList([
            CausalWanAttentionBlock(cross_attn_type, dim, ffn_dim, num_heads,
                                    local_attn_size, sink_size, qk_norm, cross_attn_norm, eps,
                                    rope_fn=rope_fn, causal_rope_fn=causal_rope_fn, use_flash_attn=use_flash_attn)
            for _ in range(num_layers)
        ])

        print(f"use_flash_attn={use_flash_attn} for CausalWanModel")

        # head
        self.head = CausalHead(dim, out_dim, patch_size, eps)

        # buffers (don't use register_buffer otherwise dtype will be changed in to())
        assert (dim % num_heads) == 0 and (dim // num_heads) % 2 == 0
        d = dim // num_heads
        self.freqs = torch.cat([
            rope_params(1024, d - 4 * (d // 6)),
            rope_params(1024, 2 * (d // 6)),
            rope_params(1024, 2 * (d // 6))
        ],
            dim=1)

        if model_type == 'i2v':
            self.img_emb = MLPProj(1280, dim)

        # initialize weights
        self.init_weights()

        self.gradient_checkpointing = False

        self.block_mask = None

        self.num_frame_per_block = 1
        self.independent_first_frame = False

    def _set_gradient_checkpointing(self, module, value=False):
        self.gradient_checkpointing = value

    @staticmethod
    def _prepare_blockwise_causal_attn_mask(
        device: torch.device | str, num_frames: int = 21,
        frame_seqlen: int = 1560, num_frame_per_block=1, local_attn_size=-1
    ) -> BlockMask:
        """
        we will divide the token sequence into the following format
        [1 latent frame] [1 latent frame] ... [1 latent frame]
        We use flexattention to construct the attention mask
        """
        total_length = num_frames * frame_seqlen # 32,760
        # DONT CALL PRINT SINCE IT CAUSES OOM
        # print(f"total_length = num_frames * frame_seqlen: {total_length}={num_frames} * {frame_seqlen}")

        # we do right padding to get to a multiple of 128
        padded_length = math.ceil(total_length / 128) * 128 - total_length # 8
        # print(f"{padded_length=}")

        # the index of the first token that is not visible to the current token. 
        # For example, if the current token can only attend to the previous 2 frames, then ends[current_token_index] will be the index of the first token in the current frame. 
        # If the current token can attend to all previous tokens, then ends[current_token_index] will be current_token_index + 1.
        ends = torch.zeros(total_length + padded_length,
                           device=device, dtype=torch.long) # 0s of shape [total_length + padded_length] [32768,]

        # Block-wise causal mask will attend to all elements that are before the end of the current chunk
        frame_indices = torch.arange(
            start=0,
            end=total_length,
            step=frame_seqlen * num_frame_per_block,
            device=device
        ) # [0, frame_seqlen, frame_seqlen*2, ..., total_length]

        for tmp in frame_indices:
            ends[tmp:tmp + frame_seqlen * num_frame_per_block] = tmp + \
                frame_seqlen * num_frame_per_block
            # [frame_seqlen * num_frame_per_block (tokens per block), tokens per block, ..., 2* tokens per block, ..., total_length, total_length]
            
        

        def attention_mask(b, h, q_idx, kv_idx):
            if local_attn_size == -1:
                # if local_attn_size is -1, we will use global attention, which means each token can attend to all previous tokens and itself. This is equivalent to a causal mask.
                # the ends logic goes like this: for a query token/row of index i, it can attend to (return true) all key tokens/columns with index < ends[i], and also itself (i).
                return (kv_idx < ends[q_idx]) | (q_idx == kv_idx)
            else:
                # with local_attn_size, a query can only see the most recent local_attn_size frames.
                return ((kv_idx < ends[q_idx]) & (kv_idx >= (ends[q_idx] - local_attn_size * frame_seqlen))) | (q_idx == kv_idx)
            # return ((kv_idx < total_length) & (q_idx < total_length))  | (q_idx == kv_idx) # bidirectional mask

        # _compile=True so create_block_mask evaluates mask_mod block-by-block
        # under torch.compile.  With _compile=False the helper materialises a
        # full [Q_LEN, KV_LEN] dense boolean mask + fp32 sum-buffer to count
        # active blocks, which scales as O(N^2) and OOMs at the doubled-height
        # token counts the height-concat training paths produce (e.g. RAVEN's
        # 131k-token TF input → 128 GB scratch).  The mask is built once per
        # model instance and cached at self.block_mask, so the compile cost
        # is paid exactly once.
        block_mask = create_block_mask(attention_mask, B=None, H=None, Q_LEN=total_length + padded_length,
                                       KV_LEN=total_length + padded_length, _compile=True, device=device)

        import torch.distributed as dist
        if not dist.is_initialized() or dist.get_rank() == 0:
            print(
                f" cache a block wise causal mask with block size of {num_frame_per_block} frames")
            print(block_mask)

        # import imageio
        # import numpy as np
        # from torch.nn.attention.flex_attention import create_mask

        # mask = create_mask(attention_mask, B=None, H=None, Q_LEN=total_length +
        #                    padded_length, KV_LEN=total_length + padded_length, device=device)
        # import cv2
        # mask = cv2.resize(mask[0, 0].cpu().float().numpy(), (1024, 1024))
        # imageio.imwrite("mask_%d.jpg" % (0), np.uint8(255. * mask))

        return block_mask

    @staticmethod
    def _prepare_teacher_forcing_mask(
        device: torch.device | str, num_frames: int = 21,
        frame_seqlen: int = 1560, num_frame_per_block: int = 1,
        local_attn_size: int = -1, sink_size: int = 0,
    ) -> BlockMask:
        """Two-half teacher-forcing block mask for [clean_<1..T> | noisy_<1..T>@u].

        Default behaviour (``local_attn_size == -1``, ``sink_size == 0``)
        reproduces the original global TF mask:
            * clean_t attends to all clean_<=t (block-causal)
            * noisy_t attends to all clean_<t (strict block-causal) + same noisy block
            * self-attention (q == kv) always allowed

        Local-attention mode (``local_attn_size != -1``) mirrors the rollout's
        KV-cache eviction window (see CausalWanSelfAttention.forward).  In
        block units::

            NPB  = num_frame_per_block
            K_b  = local_attn_size // NPB         (cache capacity, blocks)
            S_b  = sink_size // NPB               (sink, blocks)
            W_b  = K_b - S_b - 1                  (rolling clean window, blocks)

        For a query at block t the allowed clean kv block-indices become::

            clean query:  j ∈ [0, min(S_b, t+1)) ∪ [max(S_b, t - W_b), t]
            noisy query:  j ∈ [0, min(S_b, t))   ∪ [max(S_b, t - W_b), t - 1]

        The ``min(S_b, t[+1])`` cap prevents query block t from attending to
        sink blocks that do not yet exist in the rollout's cache at that
        chunk.  ``noisy_t`` additionally attends to its own noisy block
        (already in the original mask).

        Args:
            device:               Mask construction device.
            num_frames:           T (per-half frame count).
            frame_seqlen:         Tokens per frame (1560 base, 3120 with height-concat).
            num_frame_per_block:  Frames per AR block (NPB above).
            local_attn_size:      Sliding-window size in frames; -1 disables.
            sink_size:            Sink size in frames (kept across evictions).

        Returns:
            ``BlockMask`` of length ``2 * num_frames * frame_seqlen`` (+pad).
        """
        # debug
        DEBUG = False
        if DEBUG:
            num_frames = 9
            frame_seqlen = 256

        total_length = num_frames * frame_seqlen * 2

        # we do right padding to get to a multiple of 128
        padded_length = math.ceil(total_length / 128) * 128 - total_length

        clean_ends = num_frames * frame_seqlen
        attention_block_size = frame_seqlen * num_frame_per_block
        num_blocks = num_frames // num_frame_per_block

        # Validate window/sink for local-attention mode and compute block-unit
        # bounds.  ``W_b = num_blocks`` in the global-attention fallback is
        # chosen large enough that ``max(S_b, t - W_b) == S_b == 0`` always,
        # i.e. the recent-window lower bound collapses to 0 (= original).
        if local_attn_size != -1:
            assert sink_size % num_frame_per_block == 0, (
                f"sink_size ({sink_size}) must be a multiple of num_frame_per_block ({num_frame_per_block})"
            )
            assert (local_attn_size - sink_size) % num_frame_per_block == 0, (
                f"(local_attn_size - sink_size) ({local_attn_size - sink_size}) "
                f"must be a multiple of num_frame_per_block ({num_frame_per_block})"
            )
            assert 0 <= sink_size < local_attn_size, (
                f"require 0 <= sink_size ({sink_size}) < local_attn_size ({local_attn_size})"
            )
            S_b = sink_size // num_frame_per_block
            K_b = local_attn_size // num_frame_per_block
            W_b = K_b - S_b - 1
        else:
            S_b = 0
            W_b = num_blocks  # large enough that recent_lo collapses to S_b == 0

        # Per-token index arrays.  Slots not written stay 0; the mask closure
        # treats (start, end) == (0, 0) as an empty range.  ``sink_clean_end``
        # is the per-query upper bound (exclusive) of the sink kv range; the
        # implicit lower bound is 0.  For the noisy-half it is capped at
        # ``min(S_b, t)`` to enforce causality vs. the rollout.
        sink_clean_end       = torch.zeros(total_length + padded_length, device=device, dtype=torch.long)
        context_starts       = torch.zeros(total_length + padded_length, device=device, dtype=torch.long)
        context_ends         = torch.zeros(total_length + padded_length, device=device, dtype=torch.long)
        noise_context_starts = torch.zeros(total_length + padded_length, device=device, dtype=torch.long)
        noise_context_ends   = torch.zeros(total_length + padded_length, device=device, dtype=torch.long)
        noise_noise_starts   = torch.zeros(total_length + padded_length, device=device, dtype=torch.long)
        noise_noise_ends     = torch.zeros(total_length + padded_length, device=device, dtype=torch.long)

        for t in range(num_blocks):
            clean_block_start = t * attention_block_size
            clean_block_end   = clean_block_start + attention_block_size
            noisy_block_start = clean_ends + clean_block_start
            noisy_block_end   = clean_ends + clean_block_end

            recent_lo_blocks    = max(S_b, t - W_b)
            recent_hi_blocks_cl = t + 1   # clean half includes current block
            recent_hi_blocks_no = t       # noisy half excludes current (clean_t not yet in rollout cache)
            sink_end_blocks_cl  = min(S_b, t + 1)
            sink_end_blocks_no  = min(S_b, t)

            # Clean half query at block t
            context_starts[clean_block_start:clean_block_end] = recent_lo_blocks * attention_block_size
            context_ends[clean_block_start:clean_block_end]   = recent_hi_blocks_cl * attention_block_size
            sink_clean_end[clean_block_start:clean_block_end] = sink_end_blocks_cl * attention_block_size

            # Noisy half query at block t — recent clean window + sink + own noisy block
            noise_context_starts[noisy_block_start:noisy_block_end] = recent_lo_blocks * attention_block_size
            noise_context_ends[noisy_block_start:noisy_block_end]   = recent_hi_blocks_no * attention_block_size
            sink_clean_end[noisy_block_start:noisy_block_end]       = sink_end_blocks_no * attention_block_size
            noise_noise_starts[noisy_block_start:noisy_block_end] = noisy_block_start
            noise_noise_ends[noisy_block_start:noisy_block_end]   = noisy_block_end

        def attention_mask(b, h, q_idx, kv_idx):
            # Clean half: sink ∪ recent-clean window
            clean_sink   = kv_idx < sink_clean_end[q_idx]
            clean_recent = (kv_idx >= context_starts[q_idx]) & (kv_idx < context_ends[q_idx])
            clean_mask   = (q_idx < clean_ends) & (clean_sink | clean_recent)

            # Noisy half: sink ∪ recent-clean window ∪ same noisy block
            noisy_clean_recent = (kv_idx >= noise_context_starts[q_idx]) & (kv_idx < noise_context_ends[q_idx])
            noisy_self         = (kv_idx >= noise_noise_starts[q_idx]) & (kv_idx < noise_noise_ends[q_idx])
            noise_mask = (q_idx >= clean_ends) & (clean_sink | noisy_clean_recent | noisy_self)

            eye_mask = q_idx == kv_idx
            return eye_mask | clean_mask | noise_mask

        # _compile=True: see note in _prepare_blockwise_causal_attn_mask.
        # The TF input length is 2 * num_frames * frame_seqlen — height-concat
        # doubles frame_seqlen so RAVEN sits at ~131k tokens, where the
        # dense-mask path costs ~128 GB of scratch.  Compiled mask construction
        # walks blocks directly without materialising the dense Q×KV tensor.
        block_mask = create_block_mask(attention_mask, B=None, H=None, Q_LEN=total_length + padded_length,
                                       KV_LEN=total_length + padded_length, _compile=True, device=device)

        if DEBUG:
            print(block_mask)
            import imageio
            import numpy as np
            from torch.nn.attention.flex_attention import create_mask

            mask = create_mask(attention_mask, B=None, H=None, Q_LEN=total_length +
                               padded_length, KV_LEN=total_length + padded_length, device=device)
            import cv2
            mask = cv2.resize(mask[0, 0].cpu().float().numpy(), (1024, 1024))
            imageio.imwrite("mask_%d.jpg" % (0), np.uint8(255. * mask))

        return block_mask

    @staticmethod
    def _prepare_blockwise_causal_attn_mask_i2v(
        device: torch.device | str, num_frames: int = 21,
        frame_seqlen: int = 1560, num_frame_per_block=4, local_attn_size=-1
    ) -> BlockMask:
        """
        we will divide the token sequence into the following format
        [1 latent frame] [N latent frame] ... [N latent frame]
        The first frame is separated out to support I2V generation
        We use flexattention to construct the attention mask
        """
        total_length = num_frames * frame_seqlen

        # we do right padding to get to a multiple of 128
        padded_length = math.ceil(total_length / 128) * 128 - total_length

        ends = torch.zeros(total_length + padded_length,
                           device=device, dtype=torch.long)

        # special handling for the first frame
        ends[:frame_seqlen] = frame_seqlen

        # Block-wise causal mask will attend to all elements that are before the end of the current chunk
        frame_indices = torch.arange(
            start=frame_seqlen,
            end=total_length,
            step=frame_seqlen * num_frame_per_block,
            device=device
        )

        for idx, tmp in enumerate(frame_indices):
            ends[tmp:tmp + frame_seqlen * num_frame_per_block] = tmp + \
                frame_seqlen * num_frame_per_block

        def attention_mask(b, h, q_idx, kv_idx):
            if local_attn_size == -1:
                return (kv_idx < ends[q_idx]) | (q_idx == kv_idx)
            else:
                return ((kv_idx < ends[q_idx]) & (kv_idx >= (ends[q_idx] - local_attn_size * frame_seqlen))) | \
                    (q_idx == kv_idx)

        # _compile=True: see note in _prepare_blockwise_causal_attn_mask.
        block_mask = create_block_mask(attention_mask, B=None, H=None, Q_LEN=total_length + padded_length,
                                       KV_LEN=total_length + padded_length, _compile=True, device=device)

        if not dist.is_initialized() or dist.get_rank() == 0:
            print(
                f" cache a block wise causal mask with block size of {num_frame_per_block} frames")
            print(block_mask)

        # import imageio
        # import numpy as np
        # from torch.nn.attention.flex_attention import create_mask

        # mask = create_mask(attention_mask, B=None, H=None, Q_LEN=total_length +
        #                    padded_length, KV_LEN=total_length + padded_length, device=device)
        # import cv2
        # mask = cv2.resize(mask[0, 0].cpu().float().numpy(), (1024, 1024))
        # imageio.imwrite("mask_%d.jpg" % (0), np.uint8(255. * mask))

        return block_mask

    def _forward_inference(
        self,
        x,
        t,
        context,
        seq_len,
        clip_fea=None,
        y=None,
        kv_cache: dict = None,
        crossattn_cache: dict = None,
        current_start: int = 0,
        cache_start: int = 0
    ):
        r"""
        Run the diffusion model with kv caching.
        See Algorithm 2 of CausVid paper https://arxiv.org/abs/2412.07772 for details.
        This function will be run for num_frame times.
        Process the latent frames one by one (1560 tokens each)

        Args:
            x (List[Tensor]):
                List of input video tensors, each with shape [C_in, F, H, W]
            t (Tensor):
                Diffusion timesteps tensor of shape [B]
            context (List[Tensor]):
                List of text embeddings each with shape [L, C]
            seq_len (`int`):
                Maximum sequence length for positional encoding
            clip_fea (Tensor, *optional*):
                CLIP image features for image-to-video mode
            y (List[Tensor], *optional*):
                Conditional video inputs for image-to-video mode, same shape as x

        Returns:
            List[Tensor]:
                List of denoised video tensors with original input shapes [C_out, F, H / 8, W / 8]
        """

        if self.model_type == 'i2v':
            assert clip_fea is not None and y is not None
        # params
        device = self.patch_embedding.weight.device
        if self.freqs.device != device:
            self.freqs = self.freqs.to(device)

        if y is not None:
            x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

        # embeddings
        x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
        grid_sizes = torch.stack(
            [torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
        x = [u.flatten(2).transpose(1, 2) for u in x]
        seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
        assert seq_lens.max() <= seq_len
        x = torch.cat(x)
        """
        torch.cat([
            torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))],
                      dim=1) for u in x
        ])
        """

        # When t is 1D [B] (e.g. fixed_timestep or uniform_timestep mode),
        # expand to [B, F] so that time embeddings have the correct per-frame shape.
        if t.ndim == 1:
            F_per_batch = grid_sizes[0, 0].item()
            t = t.unsqueeze(1).expand(-1, F_per_batch)  # [B] -> [B, F]

        # time embeddings
        # with amp.autocast(dtype=torch.float32):
        e = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, t.flatten()).type_as(x))
        e0 = self.time_projection(e).unflatten(
            1, (6, self.dim)).unflatten(dim=0, sizes=t.shape)
        # assert e.dtype == torch.float32 and e0.dtype == torch.float32

        # context
        context_lens = None
        context = self.text_embedding(
            torch.stack([
                torch.cat(
                    [u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
                for u in context
            ]))

        if clip_fea is not None:
            context_clip = self.img_emb(clip_fea)  # bs x 257 x dim
            context = torch.concat([context_clip, context], dim=1)

        # arguments
        kwargs = dict(
            e=e0,
            seq_lens=seq_lens,
            grid_sizes=grid_sizes,
            freqs=self.freqs,
            context=context,
            context_lens=context_lens,
            block_mask=self.block_mask
        )

        def create_custom_forward(module):
            def custom_forward(*inputs, **kwargs):
                return module(*inputs, **kwargs)
            return custom_forward

        for block_index, block in enumerate(self.blocks):
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                kwargs.update(
                    {
                        "kv_cache": kv_cache[block_index],  # {"k": k_cache, "v": v_cache, "local_end_index": local_end_index, "global_end_index": global_end_index}
                        "current_start": current_start,
                        "cache_start": cache_start
                    }
                )
                x = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    x, **kwargs,
                    use_reentrant=False,
                )
            else:
                kwargs.update(
                    {
                        "kv_cache": kv_cache[block_index],
                        "crossattn_cache": crossattn_cache[block_index],
                        "current_start": current_start,
                        "cache_start": cache_start
                    }
                )
                x = block(x, **kwargs)

        # head
        x = self.head(x, e.unflatten(dim=0, sizes=t.shape).unsqueeze(2))
        # unpatchify
        x = self.unpatchify(x, grid_sizes)
        return torch.stack(x)

    def _forward_train(
        self,
        x,
        t,
        context,
        seq_len,
        clean_x=None,
        aug_t=None,
        clip_fea=None,
        y=None,
    ):
        r"""
        Forward pass through the diffusion model

        Args:
            x (List[Tensor]):
                List of input video tensors, each with shape [C_in, F, H, W]
            t (Tensor):
                Diffusion timesteps tensor of shape [B]
            context (List[Tensor]):
                List of text embeddings each with shape [L, C]
            seq_len (`int`):
                Maximum sequence length for positional encoding
            clip_fea (Tensor, *optional*):
                CLIP image features for image-to-video mode
            y (List[Tensor], *optional*):
                Conditional video inputs for image-to-video mode, same shape as x

        Returns:
            List[Tensor]:
                List of denoised video tensors with original input shapes [C_out, F, H / 8, W / 8]
        """
        if self.model_type == 'i2v':
            assert clip_fea is not None and y is not None
        # params
        device = self.patch_embedding.weight.device
        if self.freqs.device != device:
            self.freqs = self.freqs.to(device)

        # Construct blockwise causal attn mask
        if self.block_mask is None:
            if clean_x is not None:
                if self.independent_first_frame:
                    raise NotImplementedError()
                else:
                    self.block_mask = self._prepare_teacher_forcing_mask(
                        device, num_frames=x.shape[2],
                        frame_seqlen=x.shape[-2] * x.shape[-1] // (self.patch_size[1] * self.patch_size[2]),
                        num_frame_per_block=self.num_frame_per_block,
                        local_attn_size=self.local_attn_size,
                        sink_size=self.sink_size,
                    )
            else:
                if self.independent_first_frame:
                    self.block_mask = self._prepare_blockwise_causal_attn_mask_i2v(
                        device, num_frames=x.shape[2],
                        frame_seqlen=x.shape[-2] * x.shape[-1] // (self.patch_size[1] * self.patch_size[2]),
                        num_frame_per_block=self.num_frame_per_block,
                        local_attn_size=self.local_attn_size
                    )
                else:
                    self.block_mask = self._prepare_blockwise_causal_attn_mask(
                        device, num_frames=x.shape[2],
                        frame_seqlen=x.shape[-2] * x.shape[-1] // (self.patch_size[1] * self.patch_size[2]),
                        num_frame_per_block=self.num_frame_per_block,
                        local_attn_size=self.local_attn_size
                    )

        if y is not None: # y is almost always None except i2v
            x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

        # embeddings
        x = [self.patch_embedding(u.unsqueeze(0)) for u in x] # [B, dim, F, H/2, W/2]

        grid_sizes = torch.stack(
            [torch.tensor(u.shape[2:], dtype=torch.long) for u in x]) # [F//f_patch_size, H//h_patch_size, W//w_patch_size]*B, usually [21,60 or 30,104]
        x = [u.flatten(2).transpose(1, 2) for u in x] # [B, F*H*W/4, dim] usually [B, 131,040 or 65,520, dim]

        seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long) # [B], usually [131,040 or 65,520] per sample
        assert seq_lens.max() <= seq_len
        x = torch.cat([
            torch.cat([u, u.new_zeros(1, seq_lens[0] - u.size(1), u.size(2))],
                      dim=1) for u in x
        ]) # pad to the max seq len in the batch, shape [B, seq_len, dim]

        # time embeddings
        # with amp.autocast(dtype=torch.float32):
        e = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, t.flatten()).type_as(x))
        e0 = self.time_projection(e).unflatten(
            1, (6, self.dim)).unflatten(dim=0, sizes=t.shape)
        # assert e.dtype == torch.float32 and e0.dtype == torch.float32

        # context
        context_lens = None
        context = self.text_embedding(
            torch.stack([
                torch.cat(
                    [u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
                for u in context
            ])) # [B, text_len (512), dim]

        if clip_fea is not None:
            context_clip = self.img_emb(clip_fea)  # bs x 257 x dim
            context = torch.concat([context_clip, context], dim=1)

        if clean_x is not None:
            # this is teacher forcing where clean tokens are prepended to noisy tokens and the special
            # block mask is applied
            clean_x = [self.patch_embedding(u.unsqueeze(0)) for u in clean_x]
            clean_x = [u.flatten(2).transpose(1, 2) for u in clean_x]

            seq_lens_clean = torch.tensor([u.size(1) for u in clean_x], dtype=torch.long)
            assert seq_lens_clean.max() <= seq_len
            clean_x = torch.cat([
                torch.cat([u, u.new_zeros(1, seq_lens_clean[0] - u.size(1), u.size(2))], dim=1) for u in clean_x
            ])

            x = torch.cat([clean_x, x], dim=1)
            if aug_t is None:
                aug_t = torch.zeros_like(t)
            e_clean = self.time_embedding(
                sinusoidal_embedding_1d(self.freq_dim, aug_t.flatten()).type_as(x))
            e0_clean = self.time_projection(e_clean).unflatten(
                1, (6, self.dim)).unflatten(dim=0, sizes=t.shape)
            e0 = torch.cat([e0_clean, e0], dim=1)

        # arguments
        kwargs = dict(
            e=e0,
            seq_lens=seq_lens,
            grid_sizes=grid_sizes,
            freqs=self.freqs,
            context=context,
            context_lens=context_lens,
            block_mask=self.block_mask)

        def create_custom_forward(module):
            def custom_forward(*inputs, **kwargs):
                return module(*inputs, **kwargs)
            return custom_forward

        for block in self.blocks:
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                x = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    x, **kwargs,
                    use_reentrant=False,
                )
            else:
                x = block(x, **kwargs)

        if clean_x is not None:
            x = x[:, x.shape[1] // 2:]

        # head
        x = self.head(x, e.unflatten(dim=0, sizes=t.shape).unsqueeze(2))

        # unpatchify
        x = self.unpatchify(x, grid_sizes)
        return torch.stack(x)

    def forward(
        self,
        *args,
        **kwargs
    ):
        if kwargs.get('kv_cache', None) is not None:
            # print("Running inference with kv cache")
            return self._forward_inference(*args, **kwargs)
        else:
            # print("Running training forward without kv cache")
            return self._forward_train(*args, **kwargs)

    def unpatchify(self, x, grid_sizes):
        r"""
        Reconstruct video tensors from patch embeddings.

        Args:
            x (List[Tensor]):
                List of patchified features, each with shape [L, C_out * prod(patch_size)]
            grid_sizes (Tensor):
                Original spatial-temporal grid dimensions before patching,
                    shape [B, 3] (3 dimensions correspond to F_patches, H_patches, W_patches)

        Returns:
            List[Tensor]:
                Reconstructed video tensors with shape [C_out, F, H / 8, W / 8]
        """

        c = self.out_dim
        out = []
        for u, v in zip(x, grid_sizes.tolist()):
            u = u[:math.prod(v)].view(*v, *self.patch_size, c)
            u = torch.einsum('fhwpqrc->cfphqwr', u)
            u = u.reshape(c, *[i * j for i, j in zip(v, self.patch_size)])
            out.append(u)
        return out

    def init_weights(self):
        r"""
        Initialize model parameters using Xavier initialization.
        """

        # basic init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # init embeddings
        nn.init.xavier_uniform_(self.patch_embedding.weight.flatten(1))
        for m in self.text_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=.02)
        for m in self.time_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=.02)

        # init output layer
        nn.init.zeros_(self.head.head.weight)

    def set_rope_fn(self, rope_fn=None, causal_rope_fn=None):
        """Replace the RoPE functions used by every self-attention block.

        Call this after construction (e.g. after from_pretrained) to inject
        custom positional-embedding variants without monkey-patching module-level
        globals.

        Args:
            rope_fn: Callable with the same signature as ``rope_apply``.
                     Used in the training path (kv_cache=None).
            causal_rope_fn: Callable with the same signature as ``causal_rope_apply``.
                            Used in the inference path (kv_cache≠None).
        """
        for block in self.blocks:
            if rope_fn is not None:
                block.self_attn.rope_fn = rope_fn
            if causal_rope_fn is not None:
                block.self_attn.causal_rope_fn = causal_rope_fn

    def set_online_rope(self, enabled: bool):
        """Enable/disable MemRoPE Online RoPE Indexing on every self-attention block.

        When enabled, the AR/KV-cache inference path stores un-roped keys and re-applies
        RoPE at attention time with a compact block-relative temporal index, keeping
        indices within the trained range for arbitrarily long rollouts. No effect on the
        training path (kv_cache=None). See [SUMMARY_MEMROPE.md].
        """
        for block in self.blocks:
            block.self_attn.online_rope_indexing = bool(enabled)

    def set_memrope(self, cfg):
        """Set MemRoPE Phase 2 config on every self-attention block (None disables).

        ``cfg`` is the dict returned by ``parse_memrope_cfg`` ({"gen":..., "cond":...}).
        Stored on the model (``self.memrope_cfg``) so the inference pipeline can size the
        split KV cache, and on each block's self-attention for the AR forward path.
        """
        self.memrope_cfg = cfg
        for block in self.blocks:
            block.self_attn.memrope_cfg = cfg

    def set_rope_precision(self, precision):
        """Set RoPE compute precision for the AR rope calls on every self-attention block.

        ``precision`` is "fp64" (default, exact) or "fp32" (~2x faster rope, tiny precision
        change). Affects the online + memrope inference paths; training path is unchanged.
        """
        dt = {"fp64": torch.float64, "fp32": torch.float32,
              "float64": torch.float64, "float32": torch.float32}[str(precision).lower()]
        for block in self.blocks:
            block.self_attn.rope_compute_dtype = dt

