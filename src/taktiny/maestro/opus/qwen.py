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

from taktiny.maestro._livret import repertoire
from taktiny.cosettes._common import (
    TransformerCausalLM,
    TransformerMM,
)
from taktiny.cosettes.transformers.qwen import Qwen2DecoderLayer
from taktiny import nn


class Qwen(TransformerCausalLM):
    def __init__(self, *args, **kwargs):
        raise NotImplementedError(f'There is a plan to implement {self.__class__.__name__}.')

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
    def __init__(self, *args, **kwargs):
        raise NotImplementedError(f'There is a plan to implement {self.__class__.__name__}.')
    

class Qwen3MoE(TransformerCausalLM):
    def __init__(self, *args, **kwargs):
        raise NotImplementedError(f'There is a plan to implement {self.__class__.__name__}.')


class Qwen3Next(TransformerCausalLM):
    def __init__(self, *args, **kwargs):
        raise NotImplementedError(f'There is a plan to implement {self.__class__.__name__}.')


class Qwen3HMoE(TransformerMM):
    def __init__(self, *args, **kwargs):
        raise NotImplementedError(f'There is a plan to implement {self.__class__.__name__}.')


repertoire.register('QwenForCausalLM', Qwen)
repertoire.register('Qwen2ForCausalLM', Qwen2)
repertoire.register('Qwen3ForCausalLM', Qwen3)
repertoire.register('Qwen3MoeForCausalLM', Qwen3MoE)
repertoire.register('Qwen3NextForCausalLM', Qwen3Next)
repertoire.register('Qwen3_5MoeForConditionalGeneration', Qwen3HMoE)

__all__ = [
    'Qwen',
    'Qwen2',
    'Qwen3',
    'Qwen3MoE',
    'Qwen3Next',
    'Qwen3HMoE'
]
