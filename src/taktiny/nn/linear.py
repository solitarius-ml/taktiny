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
"""Linear modules."""

from __future__ import annotations
from typing import Any


import jax
import jax.numpy as jnp
import qwix
from jax.nn.initializers import lecun_uniform

from taktiny.nn.module import Module, Parameter
from taktiny.nn.rng import Rngs
from taktiny.utils.typing import DType, ShardMode


default_linear_initializer = lecun_uniform()


class Linear(Module):
    """General linear projection with optional Qwix-quantized weights."""

    def __init__(
        self,
        in_features: int | tuple[int, ...],
        out_features: int | tuple[int, ...],
        *,
        bias: bool = True,
        dtype: DType | str = jnp.float32,
        rngs: Rngs | None = None,
        seed: Rngs | None = None,
        initializer: Any=default_linear_initializer,
        quant: Any=None,
        dot_general: Any=None,
        axis_names: tuple[str | None, ...] | None = None,
        shard_mode: ShardMode = ShardMode.AUTO,
    ) -> None:
        if isinstance(in_features, int):
            in_features = (in_features,)
        else:
            in_features = tuple(in_features)

        if isinstance(out_features, int):
            out_features = (out_features,)
        else:
            out_features = tuple(out_features)

        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = bias
        self.dot_general = dot_general
        self.shard_mode = shard_mode

        if rngs is None and seed is None:
            raise ValueError('A rngs must be provided to initialize Linear layer')
        if rngs is None:
            import warnings

            warnings.warn('seed is deprecated. use `rngs` instead')
            rngs = seed

        weight_shape = in_features + out_features
        self.weight = Parameter(
            initializer(rngs(), weight_shape, dtype)
        )
        self.weight.quantization = quant
        self.weight.input_axis_count = len(in_features)
        self.weight.quantization_batch_axis_count = 0

        if axis_names is not None:
            if len(axis_names) != len(weight_shape):
                raise ValueError(
                    f'axis_names length {len(axis_names)} must match '
                    f'weight dimensions {len(weight_shape)}'
                )
            self.weight.axis_names = axis_names

        if bias:
            self.bias = Parameter(jnp.zeros(out_features, dtype=dtype))
            if axis_names is not None:
                self.bias.axis_names = axis_names[-len(out_features):]

    def __call__(
        self,
        x: jax.Array,
        out_sharding: jax.sharding.Sharding | None = None,
    ) -> jax.Array:
        in_dims = len(self.in_features)
        x_contracting_dims = tuple(range(x.ndim - in_dims, x.ndim))
        weight_contracting_dims = tuple(range(in_dims))
        dimension_numbers = (
            (x_contracting_dims, weight_contracting_dims),
            ((), ()),
        )
        weight = self.weight.value

        if isinstance(weight, qwix.QArray):
            out = qwix.dot_general(x, weight, dimension_numbers)
        elif self.dot_general is not None:
            out = self.dot_general(x, weight, dimension_numbers)
        else:
            out = jax.lax.dot_general(x, weight, dimension_numbers)

        if self.shard_mode == ShardMode.EXPLICIT and out_sharding is not None:
            out = jax.lax.with_sharding_constraint(out, out_sharding)

        if self.use_bias:
            out += self.bias.value

        return out

    def extra_repr(self) -> str:
        in_str = 'x'.join(map(str, self.in_features))
        out_str = 'x'.join(map(str, self.out_features))
        quantized = (
            isinstance(self.weight.value, qwix.QArray)
            or self.weight.quantization is not None
        )
        quant_str = ' (Qwix PTQ)' if quantized else ''
        custom_dot = ' (custom dot_general)' if self.dot_general is not None else ''
        return f'{in_str} -> {out_str}{quant_str}{custom_dot}'


__all__ = ['Linear']
