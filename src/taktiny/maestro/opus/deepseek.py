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

import jax.numpy as jnp

from taktiny.maestro._livret import repertoire
from taktiny.cosettes._common import TransformerLM
from taktiny import nn


class Deepseek(TransformerLM):
    def __init__(self):
        raise NotImplementedError(f'There is a plan to implement {self.__class__.__name__}.')


class DeepseekV2(TransformerLM):
    def __init__(self):
        raise NotImplementedError(f'There is a plan to implement {self.__class__.__name__}.')

class DeepseekV3(TransformerLM):
    def __init__(self):
        raise NotImplementedError(f'There is a plan to implement {self.__class__.__name__}.')
    

class DeepseekV3_2(TransformerLM):
    def __init__(self):
        raise NotImplementedError(f'There is a plan to implement {self.__class__.__name__}.')


class DeepseekV4(TransformerLM):
    def __init__(self):
        raise NotImplementedError(f'There is a plan to implement {self.__class__.__name__}.')


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