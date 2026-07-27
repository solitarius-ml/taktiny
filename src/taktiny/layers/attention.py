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
        
    def __call__(
        self, 
        x: jax.Array, 
        context: jax.Array = None,
        attention_mask: jax.Array = None, 
        is_causal: bool = False,
        kv_cache: tuple[jax.Array, jax.Array] = None,
        position_idx: jax.Array = None,
        out_sharding = None
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
        
        if kv_cache is not None:
            k_cache, v_cache = kv_cache
            # Update cache at position_idx using dynamic_update_slice
            k_cache = jax.lax.dynamic_update_slice(k_cache, k, (0, position_idx, 0, 0))
            v_cache = jax.lax.dynamic_update_slice(v_cache, v, (0, position_idx, 0, 0))
            
            # Use the full cache for attention
            k = k_cache
            v = v_cache
            
        # Sliding Window / Causal Masking
        if is_causal or self.window_size is not None:
            if self.window_size is not None:
                q_len = q.shape[1]
                k_len = k.shape[1]
                
                # Standard causal mask
                causal_mask = jnp.tril(jnp.ones((q_len, k_len), dtype=jnp.bool_))
                
                # Window mask
                window_mask = jnp.triu(jnp.ones((q_len, k_len), dtype=jnp.bool_), k=-self.window_size + 1)
                
                sliding_mask = causal_mask & window_mask
                
                if attention_mask is not None:
                    attention_mask = attention_mask & sliding_mask
                else:
                    attention_mask = sliding_mask
                
                # Turn off dot_product_attention's internal causal since we handle it manually
                is_causal = False
            
        # JAX's dot_product_attention natively handles GQA (num_kv_heads < num_heads)!
        # It expects shape [Batch, SeqLen, Heads, HeadDim] for query, key, and value.
        out = jax.nn.dot_product_attention(
            query=q, 
            key=k, 
            value=v, 
            mask=attention_mask,
            is_causal=is_causal
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