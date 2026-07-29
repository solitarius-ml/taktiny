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

from __future__ import annotations

import jax
import jax.numpy as jnp

from taktiny import nn
from taktiny.cosettes._common import TransformerDecoderLayer
from taktiny.layers import Attention, GateMLP, RotaryEmbedding
from taktiny.layers.posemb import rotate_half
from taktiny.utils.typing import ShardMode


class QwenRotaryEmbedding(RotaryEmbedding):
    """Qwen 1 rotary embedding with dynamic-NTK scaling."""

    def __init__(
        self,
        dim,
        max_position_embeddings,
        base,
        *,
        use_dynamic_ntk=True,
    ):
        super().__init__(dim, max_position_embeddings, base)
        self.use_dynamic_ntk = use_dynamic_ntk

    def __call__(self, q, k, position_idx=None):
        seq_len = q.shape[1]
        position_start = (
            jnp.asarray(0, dtype=jnp.int32)
            if position_idx is None
            else jnp.asarray(position_idx, dtype=jnp.int32)
        )
        total_length = position_start + seq_len

        ntk_alpha = jnp.asarray(1.0, dtype=jnp.float32)
        if self.use_dynamic_ntk:
            context_ratio = jnp.maximum(
                total_length / self.max_position_embeddings,
                1.0,
            )
            context_value = jnp.log2(context_ratio) + 1.0
            ntk_alpha = jnp.maximum(
                jnp.power(2.0, jnp.ceil(context_value)) - 1.0,
                1.0,
            )

        base = self.base * jnp.power(
            ntk_alpha,
            self.dim / (self.dim - 2),
        )
        exponent = (
            jnp.arange(0, self.dim, 2, dtype=jnp.float32) / self.dim
        )
        offsets = jnp.arange(seq_len, dtype=jnp.int32)
        if position_start.ndim == 0:
            inv_freq = 1.0 / jnp.power(base, exponent)
            positions = (position_start + offsets).astype(jnp.float32)
            freqs = jnp.einsum('s,d->sd', positions, inv_freq)
        elif position_start.ndim == 1:
            inv_freq = 1.0 / jnp.power(base[:, None], exponent[None, :])
            positions = (
                position_start[:, None] + offsets[None, :]
            ).astype(jnp.float32)
            freqs = jnp.einsum('bs,bd->bsd', positions, inv_freq)
        else:
            raise ValueError(
                'position_idx must be a scalar or a batch vector'
            )
        emb = jnp.concatenate((freqs, freqs), axis=-1)
        if emb.ndim == 2:
            cos = jnp.cos(emb)[None, :, None, :]
            sin = jnp.sin(emb)[None, :, None, :]
        else:
            cos = jnp.cos(emb)[:, :, None, :]
            sin = jnp.sin(emb)[:, :, None, :]

        def apply_rotary(x):
            dtype = x.dtype
            x = x.astype(jnp.float32)
            return ((x * cos) + (rotate_half(x) * sin)).astype(dtype)

        return apply_rotary(q), apply_rotary(k)


class QwenAttention(Attention):
    """Qwen 1 attention with biased QKV and log-n query scaling."""

    def __init__(
        self,
        *args,
        seq_length,
        use_logn_attn=True,
        output_bias=False,
        **kwargs,
    ):
        self.seq_length = seq_length
        self.use_logn_attn = use_logn_attn
        kwargs.update(
            bias=False,
            q_bias=True,
            k_bias=True,
            v_bias=True,
            o_bias=output_bias,
        )
        super().__init__(*args, **kwargs)

    def _scale_query(self, query, position_idx=None):
        if not self.use_logn_attn:
            return query

        position_start = (
            jnp.asarray(0, dtype=jnp.int32)
            if position_idx is None
            else jnp.asarray(position_idx, dtype=jnp.int32)
        )
        offsets = jnp.arange(query.shape[1], dtype=jnp.int32) + 1
        if position_start.ndim == 0:
            positions = position_start + offsets
        elif position_start.ndim == 1:
            positions = position_start[:, None] + offsets[None, :]
        else:
            raise ValueError(
                'position_idx must be a scalar or a batch vector'
            )
        scale = jnp.where(
            positions > self.seq_length,
            jnp.log(positions.astype(jnp.float32))
            / jnp.log(jnp.asarray(self.seq_length, dtype=jnp.float32)),
            1.0,
        )
        if scale.ndim == 1:
            scale = scale[None, :, None, None]
        else:
            scale = scale[:, :, None, None]
        return query * scale.astype(query.dtype)


class QwenMLP(GateMLP):
    """Qwen 1 gated MLP using its checkpoint-facing module names."""

    def __init__(
        self,
        hidden_size,
        intermediate_size,
        activation=jax.nn.silu,
        bias=False,
        dtype=None,
        rngs=None,
        gate_axis_names=None,
        up_axis_names=None,
        down_axis_names=None,
        shard_mode=ShardMode.AUTO,
        quant=None,
        dot_general=None,
    ):
        self.activation = (
            activation
            if callable(activation)
            else getattr(jax.nn, activation)
        )
        feedforward_size = intermediate_size // 2
        self.w1 = nn.Linear(
            hidden_size,
            feedforward_size,
            bias=bias,
            dtype=dtype,
            rngs=rngs,
            axis_names=up_axis_names,
            shard_mode=shard_mode,
            quant=quant,
            dot_general=dot_general,
        )
        self.w2 = nn.Linear(
            hidden_size,
            feedforward_size,
            bias=bias,
            dtype=dtype,
            rngs=rngs,
            axis_names=gate_axis_names,
            shard_mode=shard_mode,
            quant=quant,
            dot_general=dot_general,
        )
        self.c_proj = nn.Linear(
            feedforward_size,
            hidden_size,
            bias=bias,
            dtype=dtype,
            rngs=rngs,
            axis_names=down_axis_names,
            shard_mode=shard_mode,
            quant=quant,
            dot_general=dot_general,
        )

    def __call__(self, x, out_sharding=None):
        return self.c_proj(
            self.w1(x) * self.activation(self.w2(x)),
            out_sharding=out_sharding,
        )


class QwenDecoderLayer(TransformerDecoderLayer):
    def __init__(self, config, rngs: nn.Rngs, layer_idx=None):
        shard_mode = getattr(config, 'shard_mode', ShardMode.AUTO)
        quant = getattr(config, 'quant', None)
        dot_general = getattr(config, 'dot_general', None)
        dtype = (
            getattr(config, 'torch_dtype', None)
            or getattr(config, 'dtype', None)
            or 'bfloat16'
        )

        attention = QwenAttention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            head_dim=config.head_dim,
            num_kv_heads=config.num_key_value_heads,
            pos_emb=QwenRotaryEmbedding(
                config.head_dim,
                config.max_position_embeddings,
                config.rope_theta,
                use_dynamic_ntk=bool(config.use_dynamic_ntk),
            ),
            dtype=dtype,
            rngs=rngs,
            q_axis_names=('embed', 'heads', 'head_dim'),
            k_axis_names=('embed', 'kv_heads', 'head_dim'),
            v_axis_names=('embed', 'kv_heads', 'head_dim'),
            o_axis_names=('heads', 'head_dim', 'embed'),
            seq_length=config.seq_length,
            use_logn_attn=bool(config.use_logn_attn),
            output_bias=not bool(config.no_bias),
            shard_mode=shard_mode,
            quant=quant,
            dot_general=dot_general,
        )

        super().__init__(
            config,
            rngs=rngs,
            layer_idx=layer_idx,
            ln_1=nn.RMSNorm,
            attn=attention,
            ln_2=nn.RMSNorm,
            mlp=QwenMLP,
        )


class Qwen2Attention(Attention):
    """Qwen2 attention with bias on Q/K/V projections only."""

    def __init__(self, *args, **kwargs):
        kwargs.update(
            bias=False,
            q_bias=True,
            k_bias=True,
            v_bias=True,
            o_bias=False,
        )
        super().__init__(*args, **kwargs)


class Qwen2DecoderLayer(TransformerDecoderLayer):
    def __init__(self, config, rngs: nn.Rngs, layer_idx=None):
        super().__init__(
            config,
            rngs=rngs,
            layer_idx=layer_idx,
            input_layernorm=nn.RMSNorm,
            self_attn=Qwen2Attention,
            post_attention_layernorm=nn.RMSNorm,
            mlp=GateMLP,
        )


__all__ = [
    'QwenRotaryEmbedding',
    'QwenAttention',
    'QwenMLP',
    'QwenDecoderLayer',
    'Qwen2Attention',
    'Qwen2DecoderLayer',
]
