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
"""Base module for normalization"""
from __future__ import annotations

from typing import Any


import jax
import jax.numpy as jnp

from taktiny.nn.module import Module, Parameter
from taktiny.utils.typing import DType, ShardMode


class LayerNorm(Module):
    def __init__(self, hidden_size: int, eps: float = 1e-5, axis_names: tuple[str | None, ...] | None = None) -> None:
        self.hidden_size = hidden_size
        self.eps = eps
        self.weight = Parameter(jnp.ones((hidden_size,), dtype=jnp.float32))
        self.bias = Parameter(jnp.zeros((hidden_size,), dtype=jnp.float32))

        if axis_names is not None:
            self.weight.axis_names = axis_names
            self.bias.axis_names = axis_names

    def __call__(self, x: jax.Array) -> jax.Array:
        mean = jnp.mean(x, axis=-1, keepdims=True)
        var = jnp.var(x, axis=-1, keepdims=True)
        x_norm = (x - mean) * jax.lax.rsqrt(var + self.eps)
        return x_norm * self.weight.value + self.bias.value

    def extra_repr(self) -> str:
        return f"{self.hidden_size}, eps={self.eps}"

class RMSNorm(Module):
    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-5,
        dtype: DType | str = jnp.float32,
        with_scale: bool = True,
        axis_names: tuple[str | None, ...] | None = None,
        shard_mode: ShardMode = ShardMode.AUTO,
        initializer: Any = jnp.ones
    ) -> None:
        self.hidden_size = hidden_size
        self.eps = eps
        self.with_scale = with_scale
        self.shard_mode = shard_mode

        if with_scale:
            self.weight = Parameter(initializer((hidden_size,), dtype=dtype))
            if axis_names is not None:
                self.weight.axis_names = axis_names

    def __call__(
        self,
        x: jax.Array,
        out_sharding: jax.sharding.Sharding | None = None,
    ) -> jax.Array:
        dtype = x.dtype
        var = jnp.mean(jnp.square(x), axis=-1, keepdims=True)
        x_norm = x * jax.lax.rsqrt(var + self.eps)

        if self.with_scale:
            x_norm = x_norm * self.weight

        if self.shard_mode == ShardMode.EXPLICIT and out_sharding is not None:
            x_norm = jax.lax.with_sharding_constraint(x_norm, out_sharding)

        return x_norm.astype(dtype)

    def extra_repr(self) -> str:
        return f"{self.hidden_size}, eps={self.eps}"

class GroupNorm(Module):
    def __init__(self, num_groups: int, num_channels: int, eps: float = 1e-5) -> None:
        self.num_groups = num_groups
        self.num_channels = num_channels
        self.eps = eps
        self.weight = Parameter(jnp.ones((num_channels,), dtype=jnp.float32))
        self.bias = Parameter(jnp.zeros((num_channels,), dtype=jnp.float32))

        if num_channels % num_groups != 0:
            raise ValueError(f"num_channels ({num_channels}) must be divisible by num_groups ({num_groups})")

    def __call__(self, x: jax.Array) -> jax.Array:
        # x is (B, H, W, C) or (H, W, C)
        is_unbatched = x.ndim == 3
        if is_unbatched:
            x = jnp.expand_dims(x, 0)

        B, H, W, C = x.shape
        G = self.num_groups
        D = C // G

        # Reshape to (B, H, W, G, D)
        x_reshaped = x.reshape((B, H, W, G, D))

        # Calculate mean and variance over H, W, and D
        # We want to normalize over the spatial dimensions and the channel group
        mean = jnp.mean(x_reshaped, axis=(1, 2, 4), keepdims=True)
        var = jnp.var(x_reshaped, axis=(1, 2, 4), keepdims=True)

        x_norm = (x_reshaped - mean) * jax.lax.rsqrt(var + self.eps)
        x_norm = x_norm.reshape((B, H, W, C))

        out = x_norm * self.weight.value + self.bias.value

        if is_unbatched:
            out = jnp.squeeze(out, 0)

        return out

    def extra_repr(self) -> str:
        return f"{self.num_groups}, {self.num_channels}, eps={self.eps}"
