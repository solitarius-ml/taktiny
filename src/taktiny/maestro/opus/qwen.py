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
"""Qwen architectures"""

from __future__ import annotations

import numpy as np

from taktiny.maestro._livret import repertoire
from taktiny.maestro._config import ModelConfig
from taktiny.cosettes._common import (
    TransformerCausalLM,
    TransformerConditionalGeneration,
)
from taktiny.cosettes.transformers.qwen import (
    QwenDecoderLayer,
    Qwen2DecoderLayer,
)
from taktiny import nn


class Qwen(TransformerCausalLM):
    def __init__(
        self,
        config,
        rngs: nn.Rngs = None,
        mesh=None,
        sharding_rules=None,
    ):
        if rngs is None:
            rngs = nn.Rngs(42)

        config.num_key_value_heads = (
            getattr(config, 'num_key_value_heads', None)
            or config.num_attention_heads
        )
        config.head_dim = (
            getattr(config, 'head_dim', None)
            or getattr(config, 'kv_channels', None)
            or config.hidden_size // config.num_attention_heads
        )
        config.rope_theta = (
            getattr(config, 'rope_theta', None)
            or getattr(config, 'rotary_emb_base', None)
            or 10_000.0
        )
        config.rms_norm_eps = (
            getattr(config, 'rms_norm_eps', None)
            or getattr(config, 'layer_norm_epsilon', None)
            or 1e-6
        )
        config.hidden_act = (
            getattr(config, 'hidden_act', None)
            or 'silu'
        )
        config.attention_bias = False
        config.mlp_bias = not bool(getattr(config, 'no_bias', True))
        config.seq_length = (
            getattr(config, 'seq_length', None)
            or config.max_position_embeddings
        )

        super().__init__(
            config,
            rngs=rngs,
            decoder=QwenDecoderLayer,
            norm=nn.RMSNorm,
            mesh=mesh,
            sharding_rules=sharding_rules,
        )

    @classmethod
    def from_pretrained(
        cls,
        path_or_repo,
        mesh=None,
        sharding_rules=None,
        local=False,
        **kwargs,
    ):
        if 'config' in kwargs:
            config = kwargs.pop('config')
        else:
            config = ModelConfig.load_config(path_or_repo, local=local)

        def split_qkv(value):
            return np.split(value, 3, axis=0)

        module_map = [
            ('transformer.wte.weight', 'model.embed_tokens.embedding'),
            ('transformer.h.', 'model.layers.'),
            ('transformer.ln_f.', 'model.norm.'),
            (
                '.attn.c_attn.weight',
                [
                    '.attn.q_proj.weight',
                    '.attn.k_proj.weight',
                    '.attn.v_proj.weight',
                ],
                split_qkv,
            ),
            (
                '.attn.c_attn.bias',
                [
                    '.attn.q_proj.bias',
                    '.attn.k_proj.bias',
                    '.attn.v_proj.bias',
                ],
                split_qkv,
            ),
            ('.attn.c_proj.', '.attn.o_proj.'),
        ]

        return cls._load_from_pretrained(
            path_or_repo,
            config,
            module_map,
            local=local,
            mesh=mesh,
            sharding_rules=sharding_rules,
            **kwargs,
        )


class Qwen2(TransformerCausalLM):
    def __init__(
        self,
        config,
        rngs: nn.Rngs = None,
        mesh=None,
        sharding_rules=None,
    ):
        if rngs is None:
            rngs = nn.Rngs(42)

        super().__init__(
            config,
            rngs=rngs,
            decoder=Qwen2DecoderLayer,
            norm=nn.RMSNorm,
            mesh=mesh,
            sharding_rules=sharding_rules,
        )


class Qwen3(TransformerCausalLM):
    def __init__(self, config, rngs: nn.Rngs = None, mesh=None, sharding_rules=None, **kwargs):
        if rngs is None:
            rngs = nn.Rngs(42)
        super().__init__(
            config,
            rngs=rngs,
            decoder=Qwen2DecoderLayer,
            norm=nn.RMSNorm,
            mesh=mesh,
            sharding_rules=sharding_rules,
            **kwargs,
        )

class Qwen3MoE(TransformerCausalLM):
    def __init__(self, config, rngs: nn.Rngs = None, mesh=None, sharding_rules=None, **kwargs):
        if rngs is None:
            rngs = nn.Rngs(42)
        super().__init__(
            config,
            rngs=rngs,
            decoder=Qwen2DecoderLayer,
            norm=nn.RMSNorm,
            mesh=mesh,
            sharding_rules=sharding_rules,
            **kwargs,
        )


class Qwen3Next(TransformerCausalLM):
    def __init__(self, config, rngs: nn.Rngs = None, mesh=None, sharding_rules=None, **kwargs):
        if rngs is None:
            rngs = nn.Rngs(42)
        super().__init__(
            config,
            rngs=rngs,
            decoder=Qwen2DecoderLayer,
            norm=nn.RMSNorm,
            mesh=mesh,
            sharding_rules=sharding_rules,
            **kwargs,
        )


class Qwen3_5MoE(TransformerConditionalGeneration):
    def __init__(self, config, rngs: nn.Rngs = None, mesh=None, sharding_rules=None, **kwargs):
        if rngs is None:
            rngs = nn.Rngs(42)
        super().__init__(
            config,
            rngs=rngs,
            decoder=Qwen2DecoderLayer,
            norm=nn.RMSNorm,
            mesh=mesh,
            sharding_rules=sharding_rules,
            **kwargs,
        )


repertoire.register('QwenForCausalLM', Qwen)
repertoire.register('QWenLMHeadModel', Qwen)
repertoire.register('Qwen2ForCausalLM', Qwen2)
repertoire.register('Qwen3ForCausalLM', Qwen3)
repertoire.register('Qwen3MoeForCausalLM', Qwen3MoE)
repertoire.register('Qwen3NextForCausalLM', Qwen3Next)
repertoire.register('Qwen3_5MoeForConditionalGeneration', Qwen3_5MoE)

__all__ = [
    'Qwen',
    'Qwen2',
    'Qwen3',
    'Qwen3MoE',
    'Qwen3Next',
    'Qwen3_5MoE'
]
