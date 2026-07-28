# Copyright 2026 Shinapri
# Copyright 2024 Google Inc. HuggingFace Inc. team. All rights reserved.
#
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import jax, jax.numpy as jnp
from jax.nn.initializers import normal

from taktiny import nn
from taktiny.cosettes._common import TransformerDecoderLayer
from taktiny.layers import Attention, GateMLP, RotaryEmbedding
from taktiny.utils.typing import ShardMode


class GemmaTextScaledWordEmbedding(nn.Embedding):
    def __init__(
        self, num_embeddings: int, 
        embedding_dim: int, *, 
        rngs: nn.Rngs = None, 
        dtype=jnp.float32,
        initializer = normal(0.02),
        quant=None,
    ):
        super().__init__(
            num_embeddings,
            embedding_dim,
            rngs=rngs,
            dtype=dtype,
            initializer=initializer,
            quant=quant,
        )
        self.embedding_scale = embedding_dim ** 0.5

    def __call__(self, indices: jax.Array):
        return super().__call__(indices) * self.embedding_scale
    

class GemmaRMSNorm(nn.RMSNorm):
    def __call__(self, x, out_sharding=None):
        dtype = x.dtype
        var = jnp.mean(jnp.square(x), axis=-1, keepdims=True)
        x_norm = x * jax.lax.rsqrt(var + self.eps)
        
        if self.with_scale:
            x_norm = x_norm * (1.0 + self.weight)
            
        if self.shard_mode == ShardMode.EXPLICIT and out_sharding is not None:
            x_norm = jax.lax.with_sharding_constraint(x_norm, out_sharding)
            
        return x_norm.astype(dtype)


class GemmaDecoderLayer(TransformerDecoderLayer):
    def __init__(self, config, rngs: nn.Rngs, layer_idx=None):
        super().__init__(
            config,
            rngs=rngs,
            layer_idx=layer_idx,
            input_layernorm=GemmaRMSNorm,
            self_attn=Attention,
            post_attention_layernorm=GemmaRMSNorm,
            mlp=GateMLP,
        )


class Gemma2Attention(Attention):
    """Gemma2 attention configured by the shared decoder layer."""


class Gemma2DecoderLayer(TransformerDecoderLayer):
    def __init__(self, config, rngs: nn.Rngs, layer_idx=None):
        super().__init__(
            config,
            rngs=rngs,
            layer_idx=layer_idx,
            input_layernorm=GemmaRMSNorm,
            self_attn=Gemma2Attention,
            post_attention_layernorm=GemmaRMSNorm,
            pre_feedforward_layernorm=GemmaRMSNorm,
            mlp=GateMLP,
            post_feedforward_layernorm=GemmaRMSNorm,
        )


class Gemma3TextScaledWordEmbedding(GemmaTextScaledWordEmbedding):
    """Gemma 3 embedding with its scale rounded to the embedding dtype."""

    def __call__(self, indices: jax.Array):
        x = nn.Embedding.__call__(self, indices)
        scale = jnp.asarray(self.embedding_scale, dtype=x.dtype)
        return x * scale


class Gemma3RMSNorm(nn.RMSNorm):
    def __init__(
        self,
        hidden_size,
        eps=1e-6,
        dtype=jnp.float32,
        with_scale=True,
        axis_names=None,
        shard_mode=ShardMode.AUTO,
        initializer=jnp.zeros,
    ):
        super().__init__(
            hidden_size,
            eps=eps,
            dtype=dtype,
            with_scale=with_scale,
            axis_names=axis_names,
            shard_mode=shard_mode,
            initializer=initializer,
        )

    def __call__(self, x, out_sharding=None):
        dtype = x.dtype
        x = x.astype(jnp.float32)
        variance = jnp.mean(jnp.square(x), axis=-1, keepdims=True)
        x = x * jax.lax.rsqrt(variance + self.eps)

        if self.with_scale:
            x = x * (1.0 + self.weight.value.astype(jnp.float32))

        if (
            self.shard_mode == ShardMode.EXPLICIT
            and out_sharding is not None
        ):
            x = jax.lax.with_sharding_constraint(x, out_sharding)

        return x.astype(dtype)


class Gemma3Attention(Attention):
    """Gemma 3 attention with zero-centered Q/K normalization."""

    def __init__(
        self,
        *args,
        norm_eps=1e-6,
        norm_dtype=jnp.float32,
        shard_mode=ShardMode.AUTO,
        **kwargs,
    ):
        super().__init__(*args, shard_mode=shard_mode, **kwargs)
        self.q_norm = Gemma3RMSNorm(
            self.head_dim,
            eps=norm_eps,
            dtype=norm_dtype,
            axis_names=('head_dim',),
            shard_mode=shard_mode,
        )
        self.k_norm = Gemma3RMSNorm(
            self.head_dim,
            eps=norm_eps,
            dtype=norm_dtype,
            axis_names=('head_dim',),
            shard_mode=shard_mode,
        )


class Gemma3DecoderLayer(TransformerDecoderLayer):
    def __init__(self, config, rngs: nn.Rngs, layer_idx=None):
        if layer_idx is None:
            raise ValueError('Gemma3DecoderLayer requires layer_idx')

        shard_mode = getattr(config, 'shard_mode', ShardMode.AUTO)
        quant = getattr(config, 'quant', None)
        dot_general = getattr(config, 'dot_general', None)
        dtype = (
            getattr(config, 'torch_dtype', None)
            or getattr(config, 'dtype', None)
            or 'bfloat16'
        )
        layer_type = config.layer_types[layer_idx]
        rope_parameters = getattr(config, 'rope_parameters', None) or {}
        layer_rope_parameters = rope_parameters.get(layer_type) or {}

        if layer_type == 'sliding_attention':
            rope_theta = (
                layer_rope_parameters.get('rope_theta')
                or getattr(config, 'rope_local_base_freq', None)
                or 10_000.0
            )
            sliding_window = config.sliding_window
        else:
            rope_theta = (
                layer_rope_parameters.get('rope_theta')
                or getattr(config, 'rope_theta', None)
                or 1_000_000.0
            )
            sliding_window = None

        attention = Gemma3Attention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            head_dim=config.head_dim,
            num_kv_heads=config.num_key_value_heads,
            pos_emb=RotaryEmbedding(
                config.head_dim,
                config.max_position_embeddings,
                rope_theta,
            ),
            bias=False,
            q_bias=bool(config.attention_bias),
            k_bias=bool(config.attention_bias),
            v_bias=bool(config.attention_bias),
            o_bias=bool(config.attention_bias),
            dtype=dtype,
            rngs=rngs,
            q_axis_names=('embed', 'heads', 'head_dim'),
            k_axis_names=('embed', 'kv_heads', 'head_dim'),
            v_axis_names=('embed', 'kv_heads', 'head_dim'),
            o_axis_names=('heads', 'head_dim', 'embed'),
            window_size=sliding_window,
            scaling=config.query_pre_attn_scalar ** -0.5,
            softcap=getattr(config, 'attn_logit_softcapping', None),
            dropout=getattr(config, 'attention_dropout', None) or 0.0,
            norm_eps=config.rms_norm_eps,
            shard_mode=shard_mode,
            quant=quant,
            dot_general=dot_general,
        )

        super().__init__(
            config,
            rngs=rngs,
            layer_idx=layer_idx,
            input_layernorm=Gemma3RMSNorm,
            self_attn=attention,
            post_attention_layernorm=Gemma3RMSNorm,
            pre_feedforward_layernorm=Gemma3RMSNorm,
            mlp=GateMLP,
            post_feedforward_layernorm=Gemma3RMSNorm,
        )


__all__ = [
    'GemmaTextScaledWordEmbedding',
    'GemmaRMSNorm',
    'GemmaDecoderLayer',
    'Gemma2Attention',
    'Gemma2DecoderLayer',
    'Gemma3TextScaledWordEmbedding',
    'Gemma3RMSNorm',
    'Gemma3Attention',
    'Gemma3DecoderLayer',
]
