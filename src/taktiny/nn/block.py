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
"""Utilities modules for stack/group other modules"""
from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from typing import Any

import jax
import jax.numpy as jnp

from taktiny import transforms as tt
from taktiny.nn.module import Module
from taktiny.utils.typing import PyTree


def _stack_modules(modules: Iterable[Module]) -> tuple[Module, int]:
    modules = tuple(modules)
    if not modules:
        raise ValueError('modules must contain at least one Module')

    for index, module in enumerate(modules):
        if not isinstance(module, Module):
            raise TypeError(
                f'modules[{index}] must be a Module, got '
                f'{type(module).__name__}'
            )

    structure = jax.tree_util.tree_structure(modules[0])
    for index, module in enumerate(modules[1:], start=1):
        if jax.tree_util.tree_structure(module) != structure:
            raise ValueError(
                'all modules must have the same PyTree structure; '
                f'modules[0] and modules[{index}] differ'
            )

    stacked = jax.tree_util.tree_map(
        lambda *values: jnp.stack(values),
        *modules,
    )

    for parameter in stacked.flat_parameter_dict().values():
        if hasattr(parameter, 'axis_names') and parameter.axis_names is not None:
            parameter.axis_names = (None,) + tuple(parameter.axis_names)
        if hasattr(parameter, 'quantization_batch_axis_count'):
            parameter.quantization_batch_axis_count += 1

    return stacked, len(modules)


class List(Module):
    def __init__(self, *modules: Module) -> None:
        self.layers = list(modules)

    def __getitem__(self, idx: int) -> Module:
        return self.layers[idx]

    def __len__(self) -> int:
        return len(self.layers)

    def __iter__(self) -> Iterator[Module]:
        return iter(self.layers)

    def extra_repr(self) -> str:
        return f"{len(self.layers)}"


class Dict(Module): ...


class Sequential(Module):
    def __init__(self, *modules: Module) -> None:
        self.layers = tuple(modules)

    def __call__(self, x: PyTree, *args: Any, **kwargs: Any) -> PyTree:
        for layer in self.layers:
            x = layer(x, *args, **kwargs)
        return x

    def extra_repr(self) -> str:
        return f"{len(self.layers)}"

class SeqStack(Module):
    def __init__(self, modules: Iterable[Module]) -> None:
        self.stacked, self.num_stack = _stack_modules(modules)

    def __call__(
        self,
        f: Callable[..., Any],
        carry: PyTree,
        *args: Any,
        **kwargs: Any,
    ) -> tuple[PyTree, PyTree]:
        @tt.scan()
        def apply_fn(carry: Any, layer: Any, *broadcast_args: Any) -> Any:
            return f(layer, carry, *broadcast_args, **kwargs)

        return apply_fn(carry, self.stacked, *args)

    def extra_repr(self) -> str:
        return f"{self.num_stack}"


class Stack(Module):
    def __init__(self, modules: Iterable[Module]) -> None:
        self.stacked, self.num_stack = _stack_modules(modules)

    def __call__(
        self,
        *args: Any,
        in_axes: int | tuple[int | None, ...] = 0,
        out_axes: int | tuple[int | None, ...] = 0,
        **kwargs: Any,
    ) -> PyTree:
        if isinstance(in_axes, tuple):
            if len(in_axes) != len(args):
                raise ValueError(
                    'tuple in_axes must have one entry per positional argument'
                )
            vmap_in_axes = (0,) + in_axes
        else:
            vmap_in_axes = (0,) + (in_axes,) * len(args)

        @tt.vmap(in_axes=vmap_in_axes, out_axes=out_axes)
        def apply_fn(layer: Any, *positional_args: Any) -> Any:
            return layer(*positional_args, **kwargs)

        return apply_fn(self.stacked, *args)

    def extra_repr(self) -> str:
        return f"{self.num_stack}"


__all__ = ['List', 'Sequential', 'SeqStack', 'Stack']
