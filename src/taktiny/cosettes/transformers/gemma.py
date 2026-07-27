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
from taktiny.layers import Attention, GateMLP
from taktiny.utils.typing import ShardMode


class GemmaTextScaledWordEmbedding(nn.Embedding):
    def __init__(
        self, num_embeddings: int, 
        embedding_dim: int, *, 
        rngs: nn.Rngs = None, 
        dtype=jnp.float32,
        initializer = normal(0.02)
    ):
        super().__init__(
            num_embeddings,
            embedding_dim,
            rngs=rngs,
            dtype=dtype,
            initializer=initializer,
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


__all__ = [
    'GemmaTextScaledWordEmbedding',
    'GemmaRMSNorm',
    'GemmaDecoderLayer',
    'Gemma2Attention',
    'Gemma2DecoderLayer',
]
