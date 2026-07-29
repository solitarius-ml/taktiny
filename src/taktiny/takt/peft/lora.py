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
"""LoRA model transformation."""

from __future__ import annotations

import re

from taktiny.nn.lora import LoRALinear
from taktiny.nn.module import Module, iter_children
from taktiny.nn.rng import Rngs
from taktiny.takt._prelude import Takt
from taktiny.takt.peft._config import LoraConfig


def _replace_child(parent, name, child):
    if name.isdigit() and hasattr(parent, 'layers'):
        position = int(name)
        if isinstance(parent.layers, tuple):
            updated = list(parent.layers)
            updated[position] = child
            parent.layers = tuple(updated)
        else:
            parent.layers[position] = child
        return

    if '.' in name:
        attribute, index = name.rsplit('.', 1)
        sequence = getattr(parent, attribute)
        position = int(index)
        if isinstance(sequence, tuple):
            updated = list(sequence)
            updated[position] = child
            setattr(parent, attribute, tuple(updated))
        else:
            sequence[position] = child
        return

    setattr(parent, name, child)


@Takt.register_peft(LoraConfig)
def _apply_lora(model: Module, config: LoraConfig):
    if not isinstance(model, Module):
        raise TypeError('LoRA currently requires a Taktiny nn.Module model')

    rngs = config.rngs or Rngs(0)
    matched = []
    adapters = []

    def transform(module, prefix=''):
        for name, child in list(iter_children(module)):
            full_name = f'{prefix}.{name}' if prefix else name
            is_target = any(
                re.search(pattern, full_name)
                for pattern in config.target_modules
            )

            if is_target:
                if isinstance(child, LoRALinear):
                    raise ValueError(
                        f'LoRA is already applied to {full_name}'
                    )
                if not (
                    hasattr(child, 'in_features')
                    and hasattr(child, 'out_features')
                ):
                    raise TypeError(
                        f'PEFT target {full_name} is not a linear module'
                    )

                replacement = LoRALinear(
                    base_layer=child,
                    rank=config.rank,
                    alpha=config.alpha,
                    rngs=rngs,
                )
                _replace_child(module, name, replacement)
                matched.append(full_name)
                adapters.extend(
                    (replacement.lora_A, replacement.lora_B)
                )
            elif isinstance(child, Module):
                transform(child, full_name)

    transform(model)
    if not matched:
        patterns = ', '.join(config.target_modules)
        raise ValueError(
            f'No modules matched the PEFT target patterns: {patterns}'
        )

    for parameter in model.flat_parameter_dict().values():
        parameter.trainable = False
    for parameter in adapters:
        parameter.trainable = True

    return model


__all__ = []
