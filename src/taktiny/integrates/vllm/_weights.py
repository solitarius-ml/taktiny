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
"""Lazy weight views used by the vLLM synchronization adapters."""

from __future__ import annotations

from collections.abc import Callable, Iterator
import math

import jax.numpy as jnp
import qwix

from taktiny.nn.lora import LoRALinear
from taktiny.nn.module import Module, iter_children


WeightMapper = Callable[[str, object], tuple[str, object] | None]


def _dequantize(value):
    if isinstance(value, qwix.QArray):
        return qwix.dequantize(value)
    return value


def _compact_parameter(name, parameters):
    parameter = parameters.get(name)
    if parameter is not None:
        return parameter

    parts = name.split('.')
    for index, part in enumerate(parts):
        if not part.isdigit():
            continue
        candidate = list(parts)
        candidate[index] = 'stacked'
        parameter = parameters.get('.'.join(candidate))
        if parameter is not None:
            return parameter
    return None


def _lora_scalings(model):
    scalings = {}

    def collect(module, prefix=''):
        for name, child in iter_children(module):
            child_prefix = f'{prefix}.{name}' if prefix else name
            if isinstance(child, LoRALinear):
                scalings[child_prefix] = child.scaling
            elif isinstance(child, Module):
                collect(child, child_prefix)

    collect(model)
    return scalings


def _expanded_lora_scaling(name, scalings):
    prefix = name.removesuffix('.base_layer.weight')
    scaling = scalings.get(prefix)
    if scaling is not None:
        return scaling

    parts = prefix.split('.')
    for index, part in enumerate(parts):
        if not part.isdigit():
            continue
        candidate = list(parts)
        candidate[index] = 'stacked'
        scaling = scalings.get('.'.join(candidate))
        if scaling is not None:
            return scaling
    raise ValueError(f'LoRA scaling metadata is missing for {prefix!r}')


def _expanded_state_and_parameters(model):
    flat_state_dict = getattr(model, 'flat_state_dict', None)
    flat_parameter_dict = getattr(model, 'flat_parameter_dict', None)
    if not callable(flat_state_dict) or not callable(flat_parameter_dict):
        raise TypeError(
            'vLLM synchronization requires a model with flat_state_dict '
            'and flat_parameter_dict'
        )

    state = flat_state_dict()
    expand = getattr(model, '_expand_stacked_state_dict', None)
    if callable(expand):
        state = expand(state)
    return state, flat_parameter_dict()


def iter_internal_weight_specs(model):
    """Yield logical names, shapes, and parameter metadata without merging."""
    state, parameters = _expanded_state_and_parameters(model)
    for name, value in state.items():
        if name.endswith(('.lora_A', '.lora_B')):
            continue
        logical_name = name.replace('.base_layer.', '.')
        parameter = _compact_parameter(name, parameters)
        if parameter is None:
            raise ValueError(f'Parameter metadata is missing for {name!r}')
        yield logical_name, value.shape, parameter


def iter_internal_weights(model) -> Iterator[tuple[str, object, object]]:
    """Yield logical TakTiny weights with stacks expanded and LoRA merged."""
    state, parameters = _expanded_state_and_parameters(model)
    scalings = _lora_scalings(model) if isinstance(model, Module) else {}

    for name, value in state.items():
        if name.endswith(('.lora_A', '.lora_B')):
            continue

        parameter_name = name
        logical_name = name
        if '.base_layer.' in name:
            logical_name = name.replace('.base_layer.', '.')

        if name.endswith('.base_layer.weight'):
            prefix = name.removesuffix('.base_layer.weight')
            lora_a_name = f'{prefix}.lora_A'
            lora_b_name = f'{prefix}.lora_B'
            if lora_a_name not in state or lora_b_name not in state:
                raise ValueError(
                    f'LoRA weights are incomplete for {prefix!r}'
                )
            base = _dequantize(value)
            lora_a = state[lora_a_name]
            lora_b = state[lora_b_name]
            delta = jnp.matmul(lora_a, lora_b).reshape(base.shape)
            value = base + (
                delta * _expanded_lora_scaling(name, scalings)
            ).astype(
                base.dtype,
            )
        else:
            value = _dequantize(value)

        parameter = _compact_parameter(parameter_name, parameters)
        if parameter is None:
            raise ValueError(
                f'Parameter metadata is missing for {parameter_name!r}'
            )
        yield logical_name, value, parameter


def _checkpoint_weight(name, value, parameter):
    if name.endswith('.embed_tokens.embedding'):
        name = name.removesuffix('.embedding') + '.weight'

    input_axis_count = getattr(parameter, 'input_axis_count', None)
    if name.endswith('.weight') and input_axis_count is not None:
        input_size = math.prod(value.shape[:input_axis_count])
        output_size = math.prod(value.shape[input_axis_count:])
        value = value.reshape(input_size, output_size).T
    elif name.endswith('.bias') and value.ndim > 1:
        value = value.reshape(-1)

    return name, value


def iter_checkpoint_weights(
    model,
    mapper: WeightMapper | None = None,
) -> Iterator[tuple[str, object]]:
    """Yield Hugging Face checkpoint-format names and tensor layouts."""
    for name, value, parameter in iter_internal_weights(model):
        name, value = _checkpoint_weight(name, value, parameter)
        if mapper is not None:
            mapped = mapper(name, value)
            if mapped is None:
                continue
            if (
                not isinstance(mapped, tuple)
                or len(mapped) != 2
                or not isinstance(mapped[0], str)
            ):
                raise TypeError(
                    'weight_mapper must return (name, value) or None'
                )
            name, value = mapped
        yield name, value


__all__ = [
    'WeightMapper',
    'iter_checkpoint_weights',
    'iter_internal_weight_specs',
    'iter_internal_weights',
]
