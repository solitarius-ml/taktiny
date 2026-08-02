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
"""Gemma architectures"""

from __future__ import annotations
from typing import Any


import jax.numpy as jnp

from taktiny.maestro._livret import repertoire
from taktiny.cosettes._common import (
    TransformerCausalLM,
    TransformerConditionalGeneration,
    DiffusionLM,
    TransformerConditionalGeneration,
)
from taktiny.cosettes.transformers.gemma import (
    GemmaTextScaledWordEmbedding,
    GemmaRMSNorm,
    GemmaDecoderLayer,
    Gemma2DecoderLayer,
    Gemma3TextScaledWordEmbedding,
    Gemma3RMSNorm,
    Gemma3DecoderLayer,
)
from taktiny import nn


class Gemma(TransformerCausalLM):
    def __init__(
        self,
        config: Any,
        rngs: nn.Rngs | None = None,
        mesh: Any=None,
        sharding_rules: Any=None,
        **kwargs: Any
    ) -> None:
        if rngs is None:
            rngs = nn.Rngs(42)

        config.tie_word_embeddings = True
        super().__init__(
            config,
            rngs=rngs,
            embedding=GemmaTextScaledWordEmbedding,
            decoder=GemmaDecoderLayer,
            norm=GemmaRMSNorm,
            mesh=mesh,
            sharding_rules=sharding_rules,
            **kwargs
        )

    @classmethod
    def from_pretrained(cls, path_or_repo: Any, mesh: Any=None, sharding_rules: Any=None, local: bool=False, **kwargs: Any) -> Any:
        from taktiny.maestro._config import ModelConfig
        if 'config' in kwargs:
            config = kwargs.pop('config')
        else:
            config = ModelConfig.load_config(path_or_repo, local=local)
        config.tie_word_embeddings = True

        return super().from_pretrained(
            path_or_repo,
            mesh=mesh,
            sharding_rules=sharding_rules,
            local=local,
            config=config,
            **kwargs,
        )


class Gemma2(TransformerCausalLM):
    def __init__(
        self,
        config: Any,
        rngs: nn.Rngs | None = None,
        mesh: Any=None,
        sharding_rules: Any=None,
        **kwargs: Any
    ) -> None:
        if rngs is None:
            rngs = nn.Rngs(42)

        config.tie_word_embeddings = True
        if getattr(config, 'layer_types', None) is None:
            config.layer_types = [
                (
                    'sliding_attention'
                    if (layer_idx + 1) % 2
                    else 'full_attention'
                )
                for layer_idx in range(config.num_hidden_layers)
            ]
        super().__init__(
            config,
            rngs=rngs,
            embedding=GemmaTextScaledWordEmbedding,
            decoder=Gemma2DecoderLayer,
            norm=GemmaRMSNorm,
            mesh=mesh,
            sharding_rules=sharding_rules,
            **kwargs
        )
        self.final_logit_softcapping = getattr(
            config,
            'final_logit_softcapping',
            None,
        )

    def __call__(
        self,
        x: Any,
        attention_mask: Any=None,
        position_ids: Any=None,
        ctx: Any=None,
        logits_to_keep: int=0,
    ) -> tuple[Any, ...]:
        logits, ctx = super().__call__(
            x,
            attention_mask=attention_mask,
            position_ids=position_ids,
            ctx=ctx,
            logits_to_keep=logits_to_keep,
        )
        if self.final_logit_softcapping is not None:
            cap = self.final_logit_softcapping
            logits = cap * jnp.tanh(logits / cap)
        return logits, ctx


class Gemma3(TransformerCausalLM):
    def __init__(
        self,
        config: Any,
        rngs: nn.Rngs | None = None,
        mesh: Any=None,
        sharding_rules: Any=None,
        **kwargs: Any
    ) -> None:
        if rngs is None:
            rngs = nn.Rngs(42)

        if bool(getattr(config, 'use_bidirectional_attention', False)):
            raise NotImplementedError(
                'Gemma3 bidirectional attention is not supported'
            )

        config.tie_word_embeddings = True
        config.head_dim = (
            getattr(config, 'head_dim', None)
            or config.hidden_size // config.num_attention_heads
        )
        config.num_key_value_heads = (
            getattr(config, 'num_key_value_heads', None)
            or config.num_attention_heads
        )
        config.rope_theta = (
            getattr(config, 'rope_theta', None)
            or 1_000_000.0
        )
        config.rope_local_base_freq = (
            getattr(config, 'rope_local_base_freq', None)
            or 10_000.0
        )
        config.query_pre_attn_scalar = (
            getattr(config, 'query_pre_attn_scalar', None)
            or 256
        )
        config.attention_bias = bool(
            getattr(config, 'attention_bias', False)
        )
        config.rms_norm_eps = (
            getattr(config, 'rms_norm_eps', None)
            or 1e-6
        )

        if getattr(config, 'layer_types', None) is None:
            pattern = (
                getattr(config, 'sliding_window_pattern', None)
                or 6
            )
            config.layer_types = [
                (
                    'sliding_attention'
                    if (layer_idx + 1) % pattern
                    else 'full_attention'
                )
                for layer_idx in range(config.num_hidden_layers)
            ]

        super().__init__(
            config,
            rngs=rngs,
            embedding=Gemma3TextScaledWordEmbedding,
            decoder=Gemma3DecoderLayer,
            norm=Gemma3RMSNorm,
            mesh=mesh,
            sharding_rules=sharding_rules,
            **kwargs
        )
        self.final_logit_softcapping = getattr(
            config,
            'final_logit_softcapping',
            None,
        )

    def __call__(
        self,
        x: Any,
        attention_mask: Any=None,
        position_ids: Any=None,
        ctx: Any=None,
        logits_to_keep: int=0,
    ) -> tuple[Any, ...]:
        logits, ctx = super().__call__(
            x,
            attention_mask=attention_mask,
            position_ids=position_ids,
            ctx=ctx,
            logits_to_keep=logits_to_keep,
        )
        if self.final_logit_softcapping is not None:
            cap = self.final_logit_softcapping
            logits = cap * jnp.tanh(logits / cap)
        return logits, ctx

    @classmethod
    def from_pretrained(
        cls,
        path_or_repo: Any,
        mesh: Any=None,
        sharding_rules: Any=None,
        local: bool=False,
        **kwargs: Any,
    ) -> Any:
        from taktiny.maestro._config import ModelConfig

        if 'config' in kwargs:
            config = kwargs.pop('config')
        else:
            config = ModelConfig.load_config(path_or_repo, local=local)
        config.tie_word_embeddings = True

        return super().from_pretrained(
            path_or_repo,
            mesh=mesh,
            sharding_rules=sharding_rules,
            local=local,
            config=config,
            **kwargs,
        )


class Gemma3ConditionalGeneration(TransformerConditionalGeneration):
    def __init__(self, config: Any, rngs: nn.Rngs | None = None, mesh: Any=None, sharding_rules: Any=None, **kwargs: Any) -> None:
        if rngs is None:
            rngs = nn.Rngs(42)
        super().__init__(
            config,
            rngs=rngs,
            decoder=Gemma3DecoderLayer,
            norm=nn.RMSNorm,
            mesh=mesh,
            sharding_rules=sharding_rules,
            **kwargs,
        )


class Gemma4(TransformerConditionalGeneration):
    def __init__(self, config: Any, rngs: nn.Rngs | None = None, mesh: Any=None, sharding_rules: Any=None, **kwargs: Any) -> None:
        if rngs is None:
            rngs = nn.Rngs(42)
        super().__init__(
            config,
            rngs=rngs,
            decoder=Gemma3DecoderLayer,
            norm=nn.RMSNorm,
            mesh=mesh,
            sharding_rules=sharding_rules,
            **kwargs,
        )


class Gemma4Unified(TransformerConditionalGeneration):
    def __init__(self, config: Any, rngs: nn.Rngs | None = None, mesh: Any=None, sharding_rules: Any=None, **kwargs: Any) -> None:
        if rngs is None:
            rngs = nn.Rngs(42)
        super().__init__(
            config,
            rngs=rngs,
            decoder=Gemma3DecoderLayer,
            norm=nn.RMSNorm,
            mesh=mesh,
            sharding_rules=sharding_rules,
            **kwargs,
        )


class DiffusionGemma(DiffusionLM):
    def __init__(self) -> None:
        raise NotImplementedError(f'There is a plan to implement {self.__class__.__name__}.')


repertoire.register('GemmaForCausalLM', Gemma)
repertoire.register('Gemma2ForCausalLM', Gemma2)
repertoire.register('Gemma3ForCausalLM', Gemma3)
repertoire.register(
    'Gemma3ForConditionalGeneration',
    Gemma3ConditionalGeneration,
)
repertoire.register('Gemma4ForConditionalGeneration', Gemma4)
repertoire.register('Gemma4UnifiedForConditionalGeneration', Gemma4Unified)
repertoire.register('DiffusionGemmaForBlockDiffusion', DiffusionGemma)

__all__ = [
    'Gemma',
    'Gemma2',
    'Gemma3',
    'Gemma3ConditionalGeneration',
    'Gemma4',
    'Gemma4Unified',
    'DiffusionGemma'
]
