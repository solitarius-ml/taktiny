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
"""Position embedding modules"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from taktiny import nn


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1, x2 = jnp.split(x, 2, axis=-1)
    return jnp.concatenate((-x2, x1), axis=-1)

class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_position_embeddings: int = 4096, base: float = 10000.0, rope_scaling: dict = None):
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.rope_scaling = rope_scaling

    def __call__(self, q: jax.Array, k: jax.Array, position_idx: jax.Array = None) -> tuple[jax.Array, jax.Array]:
        # q and k are expected to have shape [batch, seq_len, num_heads, head_dim]
        seq_len = q.shape[1]
        
        inv_freq = 1.0 / (self.base ** (jnp.arange(0, self.dim, 2, dtype=jnp.float32) / self.dim))
        
        if self.rope_scaling is not None and self.rope_scaling.get("rope_type") == "llama3":
            import math
            factor = self.rope_scaling.get("factor", 8.0)
            low_freq_factor = self.rope_scaling.get("low_freq_factor", 1.0)
            high_freq_factor = self.rope_scaling.get("high_freq_factor", 4.0)
            old_context_len = self.rope_scaling.get("original_max_position_embeddings", 8192)
            
            low_freq_wavelen = old_context_len / low_freq_factor
            high_freq_wavelen = old_context_len / high_freq_factor
            
            wavelen = 2 * math.pi / inv_freq
            
            inv_freq_llama = jnp.where(wavelen > low_freq_wavelen, inv_freq / factor, inv_freq)
            smooth_factor = (old_context_len / wavelen - low_freq_factor) / (high_freq_factor - low_freq_factor)
            smoothed_inv_freq = (1 - smooth_factor) * inv_freq_llama / factor + smooth_factor * inv_freq_llama
            
            is_medium_freq = ~(wavelen < high_freq_wavelen) & ~(wavelen > low_freq_wavelen)
            inv_freq = jnp.where(is_medium_freq, smoothed_inv_freq, inv_freq_llama)
        
        positions = jnp.arange(seq_len, dtype=jnp.float32)
        if position_idx is not None:
            position_idx = jnp.asarray(position_idx, dtype=jnp.float32)
            if position_idx.ndim == 0:
                positions = positions + position_idx
            elif position_idx.ndim == 1:
                positions = position_idx[:, None] + positions[None, :]
            else:
                raise ValueError(
                    'position_idx must be a scalar or a batch vector'
                )

        if positions.ndim == 1:
            freqs = jnp.einsum('s,d->sd', positions, inv_freq)
        else:
            freqs = jnp.einsum('bs,d->bsd', positions, inv_freq)
        emb = jnp.concatenate((freqs, freqs), axis=-1)

        if emb.ndim == 2:
            cos = jnp.cos(emb)[None, :, None, :].astype(q.dtype)
            sin = jnp.sin(emb)[None, :, None, :].astype(q.dtype)
        else:
            cos = jnp.cos(emb)[:, :, None, :].astype(q.dtype)
            sin = jnp.sin(emb)[:, :, None, :].astype(q.dtype)
        
        q_embed = (q * cos) + (rotate_half(q) * sin)
        k_embed = (k * cos) + (rotate_half(k) * sin)
        
        return q_embed, k_embed
