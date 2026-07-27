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
from taktiny.utils.typing import ShardMode
from taktiny.layers import RotaryEmbedding, GateMLP, Attention


class LlamaDecoder(nn.Module):
    def __init__(self, config, rngs: nn.Rngs):
        shard_mode = getattr(config, 'shard_mode', ShardMode.AUTO)
        quant = getattr(config, 'quant', None)
        dot_general = getattr(config, 'dot_general', None)

        assert (hidden_size := config.hidden_size) is not None
        assert (dtype := config.torch_dtype) is not None
        assert (num_heads := config.num_attention_heads) is not None
        assert (num_kv_heads := config.num_key_value_heads) is not None
        assert (max_position_embeddings := config.max_position_embeddings) is not None
        assert (rope_theta := config.rope_theta) is not None
        assert (attention_bias := config.attention_bias) is not None
        assert (hidden_act := config.hidden_act) is not None 
        assert (intermediate_size := config.intermediate_size) is not None 

        if (head_dim := config.head_dim) is None:
            head_dim = hidden_size // num_heads
        
        if (mlp_bias := config.mlp_bias) is None:
            mlp_bias = False

        if (eps := config.rms_norm_eps) is None:
            eps = 1e-6
            
        rope_scaling = config.rope_scaling
        
        self.norm1 = nn.RMSNorm(
            hidden_size, 
            eps=eps, 
            dtype=jnp.float32, 
            shard_mode=shard_mode, 
            axis_names=('embed',)
        )
        
        self.attn = Attention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            head_dim=head_dim,
            num_kv_heads=num_kv_heads,
            pos_emb=RotaryEmbedding(
                head_dim, 
                max_position_embeddings,
                rope_theta,
                rope_scaling
            ),
            bias=attention_bias,
            dtype=dtype,
            rngs=rngs,
            # Fully Granular Logical Axes
            q_axis_names=('embed', 'heads', 'head_dim'),
            k_axis_names=('embed', 'kv_heads', 'head_dim'),
            v_axis_names=('embed', 'kv_heads', 'head_dim'),
            o_axis_names=('heads', 'head_dim', 'embed'),
            shard_mode=shard_mode,
            quant=quant,
            dot_general=dot_general
        )
        self.norm2 = nn.RMSNorm(
            hidden_size, 
            eps=eps, 
            dtype=jnp.float32, 
            shard_mode=shard_mode, 
            axis_names=('embed',)
        )
        
        activation_fn = getattr(jax.nn, hidden_act)
        self.mlp = GateMLP(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            activation=activation_fn,
            bias=mlp_bias,
            dtype=dtype,
            rngs=rngs,
            # Fully Granular Logical Axes
            gate_axis_names=('embed', 'mlp'),
            up_axis_names=('embed', 'mlp'),
            down_axis_names=('mlp', 'embed'),
            shard_mode=shard_mode,
            quant=quant,
            dot_general=dot_general
        )

    def __call__(
        self, 
        x: jax.Array, 
        attention_mask: jax.Array = None, 
        kv_cache: tuple[jax.Array, jax.Array] = None,
        position_idx: jax.Array = None,
        is_causal: bool = False,
        out_sharding = None
    ):
        attn_out, new_cache = self.attn(
            self.norm1(x), 
            attention_mask=attention_mask, 
            is_causal=is_causal, 
            kv_cache=kv_cache, 
            position_idx=position_idx,
            out_sharding=out_sharding
        )
        
        x = x + attn_out
        x = x + self.mlp(
            self.norm2(x), 
            out_sharding=out_sharding
        )
        return x, new_cache

LlamaTransformerBlock = LlamaDecoder


class LlamaDecoderLayer(TransformerDecoderLayer):
    def __init__(self, config, rngs: nn.Rngs):
        super().__init__(
            config,
            rngs=rngs,
            input_layernorm=nn.RMSNorm,
            self_attn=Attention,
            post_attention_layernorm=nn.RMSNorm,
            mlp=GateMLP,
        )


__all__ = ['LlamaDecoder', 'LlamaDecoderLayer']
