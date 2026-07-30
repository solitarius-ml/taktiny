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

import jax.numpy as jnp
import qwix

from taktiny.nn.lora import LoRALinear
from taktiny.nn.module import Module, iter_children
from taktiny.nn.rng import Rngs
from taktiny.takt._prelude import Takt, _replace_child
from taktiny.takt.peft._config import LoraConfig
from taktiny.utils.quantization import (
    quantize_linear_weight,
    resolve_quantization_rule,
)


@Takt.register_peft(LoraConfig)
def _apply_lora(model: Module, config: LoraConfig):
    if not isinstance(model, Module):
        raise TypeError('LoRA currently requires a Taktiny nn.Module model')

    rngs = config.rngs or Rngs(0)
    matched = []
    adapters = []
    trainable_state = {
        name: parameter.trainable
        for name, parameter in model.flat_parameter_dict().items()
    }

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

    peft_config = {
        'peft_type': 'LORA',
        'target_modules': list(config.target_modules),
        'rank': int(config.rank),
        'alpha': float(config.alpha),
    }
    base_model = getattr(model, 'base_model_name_or_path', None)
    if base_model is not None:
        peft_config['base_model_name_or_path'] = str(base_model)
    model.peft_config = peft_config
    model._peft_trainable_state = trainable_state

    return model


@Takt.register_peft_merger(LoRALinear)
def _merge_lora(module, *, dtype, quant, module_path):
    base_layer = module.base_layer
    weight = getattr(base_layer, 'weight', None)
    if weight is None:
        raise TypeError(
            f'LoRA base module {module_path} has no mergeable weight'
        )

    base_value = weight.value
    if isinstance(base_value, qwix.QArray):
        base_value = qwix.dequantize(base_value)

    if dtype is None:
        target_dtype = base_value.dtype
    else:
        target_dtype = jnp.dtype(dtype)
    if not jnp.issubdtype(target_dtype, jnp.floating):
        raise TypeError(
            'Merged LoRA dtype must be floating-point; use quant= for '
            'quantized output'
        )

    delta = jnp.matmul(
        module.lora_A.value.astype(jnp.float32),
        module.lora_B.value.astype(jnp.float32),
    ).reshape(base_value.shape)
    merged = (
        base_value.astype(jnp.float32)
        + delta * module.scaling
    ).astype(target_dtype)

    weight.value = merged
    weight.quantization = None
    if quant is not None:
        rule = resolve_quantization_rule(
            quant,
            module_path,
        )
        if rule is not None:
            weight.value = quantize_linear_weight(
                merged,
                weight,
                rule,
            )
            weight.quantization = rule

    return base_layer


__all__ = []
