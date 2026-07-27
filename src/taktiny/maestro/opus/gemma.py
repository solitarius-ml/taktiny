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

import jax.numpy as jnp

from taktiny.maestro._livret import repertoire
from taktiny.cosettes._common import (
    TransformerCausalLM,
    DiffusionLM,
    TransformerMM,
)
from taktiny.cosettes.transformers.gemma import (
    GemmaTextScaledWordEmbedding,
    GemmaRMSNorm,
    GemmaDecoderLayer,
    Gemma2DecoderLayer,
)
from taktiny import nn


class Gemma(TransformerCausalLM):
    def __init__(
        self, 
        config, 
        rngs: nn.Rngs = None, 
        mesh=None, 
        sharding_rules=None
    ):
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
        )

    @classmethod
    def from_pretrained(cls, path_or_repo, mesh=None, sharding_rules=None, local=False, **kwargs):
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
        config,
        rngs: nn.Rngs = None,
        mesh=None,
        sharding_rules=None,
    ):
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
        )
        self.final_logit_softcapping = getattr(
            config,
            'final_logit_softcapping',
            None,
        )

    def __call__(self, x, attention_mask=None, ctx=None):
        logits, ctx = super().__call__(
            x,
            attention_mask=attention_mask,
            ctx=ctx,
        )
        if self.final_logit_softcapping is not None:
            cap = self.final_logit_softcapping
            logits = cap * jnp.tanh(logits / cap)
        return logits, ctx


class Gemma3(TransformerMM):
    def __init__(self):
        raise NotImplementedError(f'There is a plan to implement {self.__class__.__name__}.')


class Gemma4(TransformerMM):
    def __init__(self):
        raise NotImplementedError(f'There is a plan to implement {self.__class__.__name__}.')


class Gemma4Unified(TransformerMM):
    def __init__(self):
        raise NotImplementedError(f'There is a plan to implement {self.__class__.__name__}.')


class DiffusionGemma(DiffusionLM):
    def __init__(self):
        raise NotImplementedError(f'There is a plan to implement {self.__class__.__name__}.')


repertoire.register('GemmaForCausalLM', Gemma)
repertoire.register('Gemma2ForCausalLM', Gemma2)
repertoire.register('Gemma3ForCausalLM', Gemma3)
repertoire.register('Gemma3ForConditionalGeneration', Gemma3)
repertoire.register('Gemma4ForConditionalGeneration', Gemma4)
repertoire.register('Gemma4UnifiedForConditionalGeneration', Gemma4Unified)
repertoire.register('DiffusionGemmaForBlockDiffusion', DiffusionGemma)

__all__ = [
    'Gemma', 
    'Gemma2', 
    'Gemma3', 
    'Gemma4', 
    'Gemma4Unified', 
    'DiffusionGemma'
]
