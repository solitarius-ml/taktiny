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
"""Deepseek architectures"""

from __future__ import annotations
from typing import Any


from taktiny.maestro._livret import repertoire
from taktiny.cosettes._common import TransformerCausalLM


from taktiny.cosettes.transformers.llama import LlamaDecoderLayer
from taktiny import nn


class Deepseek(TransformerCausalLM):
    def __init__(self, config: Any, rngs: nn.Rngs | None = None, mesh: Any=None, sharding_rules: Any=None, **kwargs: Any) -> None:
        if rngs is None:
            rngs = nn.Rngs(42)
        super().__init__(
            config,
            rngs=rngs,
            decoder=LlamaDecoderLayer,
            norm=nn.RMSNorm,
            mesh=mesh,
            sharding_rules=sharding_rules,
            **kwargs,
        )


class DeepseekV2(TransformerCausalLM):
    def __init__(self, config: Any, rngs: nn.Rngs | None = None, mesh: Any=None, sharding_rules: Any=None, **kwargs: Any) -> None:
        if rngs is None:
            rngs = nn.Rngs(42)
        super().__init__(
            config,
            rngs=rngs,
            decoder=LlamaDecoderLayer,
            norm=nn.RMSNorm,
            mesh=mesh,
            sharding_rules=sharding_rules,
            **kwargs,
        )


class DeepseekV3(TransformerCausalLM):
    def __init__(self, config: Any, rngs: nn.Rngs | None = None, mesh: Any=None, sharding_rules: Any=None, **kwargs: Any) -> None:
        if rngs is None:
            rngs = nn.Rngs(42)
        super().__init__(
            config,
            rngs=rngs,
            decoder=LlamaDecoderLayer,
            norm=nn.RMSNorm,
            mesh=mesh,
            sharding_rules=sharding_rules,
            **kwargs,
        )


class DeepseekV3_2(TransformerCausalLM):
    def __init__(self, config: Any, rngs: nn.Rngs | None = None, mesh: Any=None, sharding_rules: Any=None, **kwargs: Any) -> None:
        if rngs is None:
            rngs = nn.Rngs(42)
        super().__init__(
            config,
            rngs=rngs,
            decoder=LlamaDecoderLayer,
            norm=nn.RMSNorm,
            mesh=mesh,
            sharding_rules=sharding_rules,
            **kwargs,
        )


class DeepseekV4(TransformerCausalLM):
    def __init__(self, config: Any, rngs: nn.Rngs | None = None, mesh: Any=None, sharding_rules: Any=None, **kwargs: Any) -> None:
        if rngs is None:
            rngs = nn.Rngs(42)
        super().__init__(
            config,
            rngs=rngs,
            decoder=LlamaDecoderLayer,
            norm=nn.RMSNorm,
            mesh=mesh,
            sharding_rules=sharding_rules,
            **kwargs,
        )


repertoire.register('DeepseekForCausalLM', Deepseek)
repertoire.register('DeepseekV2ForCausalLM', DeepseekV2)
repertoire.register('DeepseekV3ForCausalLM', DeepseekV3)
repertoire.register('DeepseekV32ForCausalLM', DeepseekV3_2)
repertoire.register('DeepseekV4ForCausalLM', DeepseekV4)

__all__ = [
    'Deepseek',
    'DeepseekV2',
    'DeepseekV3',
    'DeepseekV3_2',
    'DeepseekV4',
]
