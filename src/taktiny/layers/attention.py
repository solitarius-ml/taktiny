# Copyright 2026 Shinapri
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Attention modules"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from typing import Optional

from taktiny import nn
from taktiny.layers.posemb import RotaryEmbedding
from taktiny.utils.typing import ShardMode, DType


class Attention(nn.Module):
    def __init__(
        self, 
        hidden_size: int, 
        num_heads: int,
        head_dim: int, 
        num_kv_heads: int = None,
        context_dim: int = None,
        pos_emb: nn.Module = None,
        bias: bool = False, 
        use_qkv_norm: bool = False,
        dtype: DType | str = None,
        window_size: int = None,
        rngs: nn.Rngs = None,
        q_axis_names: Optional[tuple[str | None, ...]] = None,
        k_axis_names: Optional[tuple[str | None, ...]] = None,
        v_axis_names: Optional[tuple[str | None, ...]] = None,
        o_axis_names: Optional[tuple[str | None, ...]] = None,
        q_bias: Optional[bool] = None,
        k_bias: Optional[bool] = None,
        v_bias: Optional[bool] = None,
        o_bias: Optional[bool] = None,
        scaling: Optional[float] = None,
        softcap: Optional[float] = None,
        dropout: float | int = 0.0,
        shard_mode: ShardMode = ShardMode.AUTO,
        quant=None,
        dot_general=None,
    ):
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.num_kv_heads = num_heads if num_kv_heads is None else num_kv_heads
        self.context_dim = hidden_size if context_dim is None else context_dim
        self.use_qkv_norm = use_qkv_norm
        self.window_size = window_size
        self.scaling = scaling
        self.softcap = softcap
        self.dropout = dropout
        
        # For Grouped Query Attention (GQA)
        self.num_kv_groups = self.num_heads // self.num_kv_heads
        
        self.pos_emb = pos_emb
        if bias and any(q_bias, k_bias, v_bias, o_bias):
            bias = False
            q_bias = q_bias == True
            k_bias = k_bias == True
            v_bias = v_bias == True
            o_bias = o_bias == True
        
        # Projections (Leveraging General Linear!)
        self.q_proj = nn.Linear(
            hidden_size, (self.num_heads, self.head_dim), 
            dtype=dtype, bias=q_bias or bias, rngs=rngs, 
            axis_names=q_axis_names, shard_mode=shard_mode, 
            quant=quant, dot_general=dot_general
        )
        self.k_proj = nn.Linear(
            self.context_dim, (self.num_kv_heads, self.head_dim), 
            dtype=dtype, bias=k_bias or bias, rngs=rngs, axis_names=k_axis_names, 
            shard_mode=shard_mode, quant=quant, dot_general=dot_general
        )
        self.v_proj = nn.Linear(
            self.context_dim, (self.num_kv_heads, self.head_dim), 
            dtype=dtype, bias=v_bias or bias, rngs=rngs, axis_names=v_axis_names, 
            shard_mode=shard_mode, quant=quant, dot_general=dot_general
        )
        self.o_proj = nn.Linear(
            (self.num_heads, self.head_dim), hidden_size, 
            dtype=dtype, bias=o_bias or bias, rngs=rngs, axis_names=o_axis_names, 
            shard_mode=shard_mode, quant=quant, dot_general=dot_general
        )

        self.q_norm = self.k_norm = None

        if getattr(self, 'use_qkv_norm', False) or getattr(self, 'use_q_norm', False):
            self.q_norm = nn.RMSNorm(self.head_dim, dtype=dtype, shard_mode=shard_mode)
            
        if getattr(self, 'use_qkv_norm', False) or getattr(self, 'use_k_norm', False):
            self.k_norm = nn.RMSNorm(self.head_dim, dtype=dtype, shard_mode=shard_mode)

    def _scale_query(
        self,
        query: jax.Array,
        position_idx: jax.Array = None,
    ) -> jax.Array:
        return query
        
    @classmethod
    def apply_flash_attention(
        cls,
        query: jax.Array,
        key: jax.Array,
        value: jax.Array,
        segment_ids=None,
        block_kv: int = 128,
        block_q: int = 128,
        mask: jax.Array = None,
        mask_value: float = -1e9,
        cap: float = None,
        scale: float = None,
        **kwargs,
    ) -> jax.Array:
        """Apply FlashAttention block masked kernel on Q, K, V."""
        from taktiny.kernel.attention.flash_attention import flash_attention_block_masked
        import math
        head_dim = query.shape[-1]
        if scale is None:
            scale = 1.0 / math.sqrt(head_dim)

        is_4d = query.ndim == 4
        if is_4d:
            # (Batch, SeqLen, Heads, HeadDim) -> (Batch, Heads, SeqLen, HeadDim)
            B, L, H, D = query.shape
            kv_L = key.shape[1]
            q_in = query.transpose(0, 2, 1, 3) * scale
            k_in = key.transpose(0, 2, 1, 3)
            v_in = value.transpose(0, 2, 1, 3)
            bq = min(block_q, L)
            bk = min(block_kv, kv_L)
        else:
            q_in, k_in, v_in = query * scale, key, value
            bq = min(block_q, query.shape[-2])
            bk = min(block_kv, key.shape[-2])

        out = flash_attention_block_masked(
            q_in, k_in, v_in,
            segment_ids=segment_ids,
            block_kv=bk,
            block_q=bq,
            mask=mask,
            mask_value=mask_value,
            cap=cap,
        )
        if isinstance(out, tuple):
            out = out[0]
        if is_4d:
            out = out.reshape(B, H, L, D).transpose(0, 2, 1, 3)
        return out

    @classmethod
    def apply_ragged_attention(
        cls,
        query: jax.Array,
        key: jax.Array,
        value: jax.Array,
        lengths: jax.Array = None,
        **kwargs,
    ) -> jax.Array:
        """Apply Ragged Attention kernel on sequence batches."""
        from taktiny.kernel.attention.ragged_attention import reference_mha, reference_gqa
        if lengths is None:
            B = query.shape[0]
            seq_len = key.shape[1] if key.ndim == 4 else key.shape[2]
            lengths = jnp.full((B,), seq_len, dtype=jnp.int32)

        num_heads_q = query.shape[-2] if query.ndim == 4 else query.shape[1]
        num_heads_k = key.shape[-2] if key.ndim == 4 else key.shape[1]
        if num_heads_q == num_heads_k:
            out = reference_mha(query, key, value, lengths, **kwargs)
        else:
            out = reference_gqa(query, key, value, lengths, **kwargs)

        if isinstance(out, tuple):
            out = out[0]
        return out

    @classmethod
    def apply_splash_attention(
        cls,
        query: jax.Array,
        key: jax.Array,
        value: jax.Array,
        mask=None,
        segment_ids=None,
        scale: float = None,
        **kwargs,
    ) -> jax.Array:
        """Apply Splash Attention block-sparse reference kernel."""
        from taktiny.kernel.attention.splash_attention import attention_reference
        import math
        head_dim = query.shape[-1]
        if scale is None:
            scale = 1.0 / math.sqrt(head_dim)
        q_scaled = query * scale

        if mask is None:
            q_len = query.shape[1] if query.ndim >= 3 else query.shape[0]
            k_len = key.shape[1] if key.ndim >= 3 else key.shape[0]
            mask = jnp.ones((q_len, k_len), dtype=jnp.bool_)
        if query.ndim == 4:
            # vmap over batch (axis 0) and head (axis 2) dimensions for 4D (B, L, H, D)
            vmapped_attn = jax.vmap(
                jax.vmap(
                    lambda q_2d, k_2d, v_2d: attention_reference(mask, q_2d, k_2d, v_2d, segment_ids, **kwargs),
                    in_axes=(1, 1, 1), out_axes=1
                ),
                in_axes=(0, 0, 0), out_axes=0
            )
            out = vmapped_attn(q_scaled, key, value)
            if isinstance(out, tuple):
                out = out[0]
            return out
        else:
            out = attention_reference(mask, q_scaled, key, value, segment_ids, **kwargs)
            if isinstance(out, tuple):
                out = out[0]
            return out

    @classmethod
    def apply_ring_attention(
        cls,
        query: jax.Array,
        key: jax.Array,
        value: jax.Array,
        axis_name: str = "seq",
        **kwargs,
    ) -> jax.Array:
        """Apply Context Parallel Ring Attention kernel."""
        from taktiny.kernel.attention.tokamax_splash import ring_attention_kernel
        return ring_attention_kernel.ring_attention_custom(query, key, value, axis_name=axis_name, **kwargs)

    @classmethod
    def apply(
        cls,
        query: jax.Array,
        key: jax.Array,
        value: jax.Array,
        kernel: str = "dot_product",
        mask: jax.Array = None,
        bias: jax.Array = None,
        scale: float = None,
        is_causal: bool = False,
        **kwargs,
    ) -> jax.Array:
        """
        Unified Entry Point for Attention Kernel Applications.
        Supported kernel methods: 'dot_product', 'flash', 'ragged', 'splash', 'ring'.
        """
        kernel = kernel.lower()
        if kernel in ("dot_product", "standard", "jax"):
            return jax.nn.dot_product_attention(
                query=query,
                key=key,
                value=value,
                bias=bias,
                mask=mask,
                scale=scale,
                is_causal=is_causal,
            )
        elif kernel in ("flash", "flash_attention"):
            return cls.apply_flash_attention(query, key, value, bias=bias, scale=scale, **kwargs)
        elif kernel in ("ragged", "ragged_attention"):
            return cls.apply_ragged_attention(query, key, value, **kwargs)
        elif kernel in ("splash", "splash_attention"):
            return cls.apply_splash_attention(query, key, value, mask=mask, **kwargs)
        elif kernel in ("ring", "ring_attention"):
            return cls.apply_ring_attention(query, key, value, **kwargs)
        else:
            raise ValueError(
                f"Unknown attention kernel method: '{kernel}'. Choice of ['dot_product', 'flash', 'ragged', 'splash', 'ring']"
            )

    def __call__(
        self, 
        x: jax.Array, 
        context: jax.Array = None,
        attention_mask: jax.Array = None, 
        is_causal: bool = False,
        kv_cache: tuple[jax.Array, jax.Array] = None,
        position_idx: jax.Array = None,
        out_sharding = None,
        kernel: str = "dot_product",
    ) -> jax.Array | tuple[jax.Array, tuple[jax.Array, jax.Array]]:
        
        context_in = context if context is not None else x
        
        # Project directly to [B, L, Heads, HeadDim] thanks to General Linear
        q = self.q_proj(x)
        k = self.k_proj(context_in)
        v = self.v_proj(context_in)

        if self.q_norm is not None:
            q = self.q_norm(q)
        if self.k_norm is not None:
            k = self.k_norm(k)
        
        # Apply Positional Embeddings (if provided)
        if self.pos_emb is not None:
            q, k = self.pos_emb(q, k, position_idx)

        q = self._scale_query(q, position_idx)
        
        if kv_cache is not None:
            k_cache, v_cache = kv_cache
            cache_position = jnp.asarray(position_idx, dtype=jnp.int32)
            if cache_position.ndim == 0:
                k_cache = jax.lax.dynamic_update_slice(
                    k_cache,
                    k,
                    (0, cache_position, 0, 0),
                )
                v_cache = jax.lax.dynamic_update_slice(
                    v_cache,
                    v,
                    (0, cache_position, 0, 0),
                )
            elif cache_position.ndim == 1:
                def update_cache(cache_row, value_row, row_position):
                    return jax.lax.dynamic_update_slice(
                        cache_row,
                        value_row,
                        (row_position, 0, 0),
                    )

                k_cache = jax.vmap(update_cache)(
                    k_cache,
                    k,
                    cache_position,
                )
                v_cache = jax.vmap(update_cache)(
                    v_cache,
                    v,
                    cache_position,
                )
            else:
                raise ValueError(
                    'position_idx must be a scalar or a batch vector'
                )
            
            # Use the full cache for attention
            k = k_cache
            v = v_cache
            
        # Sliding Window / Causal Masking
        if is_causal or self.window_size is not None:
            if self.window_size is not None:
                q_len = q.shape[1]
                k_len = k.shape[1]

                query_start = (
                    jnp.asarray(0, dtype=jnp.int32)
                    if position_idx is None
                    else jnp.asarray(position_idx, dtype=jnp.int32)
                )
                query_offsets = jnp.arange(q_len, dtype=jnp.int32)
                if query_start.ndim == 0:
                    query_positions = query_start + query_offsets
                elif query_start.ndim == 1:
                    query_positions = (
                        query_start[:, None] + query_offsets[None, :]
                    )
                else:
                    raise ValueError(
                        'position_idx must be a scalar or a batch vector'
                    )

                # Cached keys use absolute positions from the start of the
                # sequence. Uncached keys belong to the same local chunk as Q.
                key_start = (
                    jnp.asarray(0, dtype=jnp.int32)
                    if kv_cache is not None
                    else query_start
                )
                key_positions = key_start + jnp.arange(
                    k_len, dtype=jnp.int32
                )

                if query_positions.ndim == 1:
                    causal_mask = (
                        key_positions[None, :]
                        <= query_positions[:, None]
                    )
                    window_mask = key_positions[None, :] >= (
                        query_positions[:, None] - self.window_size + 1
                    )
                else:
                    causal_mask = (
                        key_positions[None, None, :]
                        <= query_positions[:, :, None]
                    )
                    window_mask = key_positions[None, None, :] >= (
                        query_positions[:, :, None]
                        - self.window_size
                        + 1
                    )
                    causal_mask = causal_mask[:, None, :, :]
                    window_mask = window_mask[:, None, :, :]
                sliding_mask = causal_mask & window_mask

                if attention_mask is not None:
                    attention_mask = attention_mask & sliding_mask
                else:
                    attention_mask = sliding_mask

                # The absolute-position mask handles causality itself.
                is_causal = False
            
        attention_bias = None
        if self.softcap is not None:
            scale = (
                self.scaling
                if self.scaling is not None
                else self.head_dim ** -0.5
            )
            batch_size, query_length, _, _ = q.shape
            key_length = k.shape[1]
            grouped_q = q.reshape(
                batch_size,
                query_length,
                self.num_kv_heads,
                self.num_kv_groups,
                self.head_dim,
            )
            scores = jnp.einsum(
                'btkgh,bskh->bkgts',
                grouped_q,
                k,
            ) * scale
            capped_scores = self.softcap * jnp.tanh(scores / self.softcap)
            attention_bias = (capped_scores - scores).reshape(
                batch_size,
                self.num_heads,
                query_length,
                key_length,
            )

        # Apply attention using requested kernel via Attention.apply entry point!
        out = self.apply(
            query=q,
            key=k,
            value=v,
            kernel=kernel,
            bias=attention_bias,
            mask=attention_mask,
            scale=self.scaling,
            is_causal=is_causal,
        )
        
        # Output projection from (Batch, SeqLen, Heads, HeadDim) directly to (Batch, SeqLen, HiddenSize)
        out = self.o_proj(out, out_sharding=out_sharding)
        
        if kv_cache is not None:
            return out, (k_cache, v_cache)
        
        return out, None

class JointAttention(nn.Module):
    """
    Generic Joint/Double-Stream Attention for Multimodal architectures (e.g. MM-DiT).
    Takes two separate streams, projects them to Q, K, V independently, 
    concatenates them for a joint self-attention operation, and splits the output back.
    """
    def __init__(
        self, 
        hidden_size1: int, 
        hidden_size2: int, 
        num_heads: int,
        head_dim: int,
        use_qkv_norm: bool = False,
        pos_emb: nn.Module = None,
        seed: nn.Rngs = None
    ):
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.pos_emb = pos_emb
        
        # Stream 1 Projections
        self.q_proj_1 = nn.Linear(hidden_size1, num_heads * head_dim, bias=False, seed=seed)
        self.k_proj_1 = nn.Linear(hidden_size1, num_heads * head_dim, bias=False, seed=seed)
        self.v_proj_1 = nn.Linear(hidden_size1, num_heads * head_dim, bias=False, seed=seed)
        self.o_proj_1 = nn.Linear(num_heads * head_dim, hidden_size1, bias=False, seed=seed)
        
        # Stream 2 Projections
        self.q_proj_2 = nn.Linear(hidden_size2, num_heads * head_dim, bias=False, seed=seed)
        self.k_proj_2 = nn.Linear(hidden_size2, num_heads * head_dim, bias=False, seed=seed)
        self.v_proj_2 = nn.Linear(hidden_size2, num_heads * head_dim, bias=False, seed=seed)
        self.o_proj_2 = nn.Linear(num_heads * head_dim, hidden_size2, bias=False, seed=seed)
        
        if use_qkv_norm:
            self.q_norm_1 = nn.RMSNorm(head_dim)
            self.k_norm_1 = nn.RMSNorm(head_dim)
            self.q_norm_2 = nn.RMSNorm(head_dim)
            self.k_norm_2 = nn.RMSNorm(head_dim)
        else:
            self.q_norm_1 = self.k_norm_1 = None
            self.q_norm_2 = self.k_norm_2 = None
            
    def __call__(
        self, 
        x1: jax.Array, 
        x2: jax.Array,
        # We can pass modulation chunks dynamically (like from AdaLN)
        mod1: tuple[jax.Array, ...] = None,
        mod2: tuple[jax.Array, ...] = None,
        position_idx: jax.Array = None
    ) -> tuple[jax.Array, jax.Array]:
        B, L1, _ = x1.shape
        _, L2, _ = x2.shape
        
        # 1. Project Stream 1
        q1 = self.q_proj_1(x1).reshape(B, L1, self.num_heads, self.head_dim)
        k1 = self.k_proj_1(x1).reshape(B, L1, self.num_heads, self.head_dim)
        v1 = self.v_proj_1(x1).reshape(B, L1, self.num_heads, self.head_dim)
        
        # 2. Project Stream 2
        q2 = self.q_proj_2(x2).reshape(B, L2, self.num_heads, self.head_dim)
        k2 = self.k_proj_2(x2).reshape(B, L2, self.num_heads, self.head_dim)
        v2 = self.v_proj_2(x2).reshape(B, L2, self.num_heads, self.head_dim)
        
        # 3. Apply QK Norms if specified
        if self.q_norm_1 is not None:
            q1, k1 = self.q_norm_1(q1), self.k_norm_1(k1)
            q2, k2 = self.q_norm_2(q2), self.k_norm_2(k2)
            
        # 4. Apply specific modulations if provided by caller (e.g. DiT scale/shift)
        # mod is expected to be (shift_q, scale_q, shift_k, scale_k, shift_v, scale_v) or None
        if mod1 is not None:
            shift_q1, scale_q1, shift_k1, scale_k1, shift_v1, scale_v1 = mod1
            shift_q1, scale_q1 = shift_q1.reshape(B, 1, self.num_heads, self.head_dim), scale_q1.reshape(B, 1, self.num_heads, self.head_dim)
            shift_k1, scale_k1 = shift_k1.reshape(B, 1, self.num_heads, self.head_dim), scale_k1.reshape(B, 1, self.num_heads, self.head_dim)
            shift_v1, scale_v1 = shift_v1.reshape(B, 1, self.num_heads, self.head_dim), scale_v1.reshape(B, 1, self.num_heads, self.head_dim)
            
            q1 = q1 * (1 + scale_q1) + shift_q1
            k1 = k1 * (1 + scale_k1) + shift_k1
            v1 = v1 * (1 + scale_v1) + shift_v1
            
        if mod2 is not None:
            shift_q2, scale_q2, shift_k2, scale_k2, shift_v2, scale_v2 = mod2
            shift_q2, scale_q2 = shift_q2.reshape(B, 1, self.num_heads, self.head_dim), scale_q2.reshape(B, 1, self.num_heads, self.head_dim)
            shift_k2, scale_k2 = shift_k2.reshape(B, 1, self.num_heads, self.head_dim), scale_k2.reshape(B, 1, self.num_heads, self.head_dim)
            shift_v2, scale_v2 = shift_v2.reshape(B, 1, self.num_heads, self.head_dim), scale_v2.reshape(B, 1, self.num_heads, self.head_dim)
            
            q2 = q2 * (1 + scale_q2) + shift_q2
            k2 = k2 * (1 + scale_k2) + shift_k2
            v2 = v2 * (1 + scale_v2) + shift_v2
            
        # 5. Concatenate streams for joint attention!
        q = jnp.concatenate([q1, q2], axis=1)
        k = jnp.concatenate([k1, k2], axis=1)
        v = jnp.concatenate([v1, v2], axis=1)
        
        # Apply Positional Embeddings (e.g. RoPE)
        if self.pos_emb is not None:
            q, k = self.pos_emb(q, k, position_idx)
            
        # 6. Apply JAX native Attention
        out = jax.nn.dot_product_attention(q, k, v)
        
        # 7. Split streams back apart
        out1, out2 = jnp.split(out, [L1], axis=1)
        
        # 8. Final Output Projections
        out1 = self.o_proj_1(out1.reshape(B, L1, -1))
        out2 = self.o_proj_2(out2.reshape(B, L2, -1))
        
        return out1, out2
