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
"""LEGACY: Normalization layer modules"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from taktiny import nn


class AdaLayerNorm(nn.Module):
    """
    Generic Adaptive Layer Normalization.
    Computes a parameter-free normalization on `x` and projects a condition vector `vec`
    into a desired dimension. The caller is responsible for splitting the modulation
    output into scale/shift/gate chunks as needed for their specific architecture.
    """
    def __init__(
        self,
        embedding_dim: int,
        out_dim: int,
        norm_type: str = "layer_norm",
        eps: float = 1e-6,
        seed: nn.Rngs | None = None
    ) -> None:
        self.eps = eps
        self.norm_type = norm_type

        # Linear projection from the condition vector (e.g. time/text)
        self.linear = nn.Linear(embedding_dim, out_dim, seed=seed)

    def __call__(self, x: jax.Array, vec: jax.Array) -> tuple[jax.Array, jax.Array]:
        # 1. Parameter-free normalization
        if self.norm_type == "layer_norm":
            mean = jnp.mean(x, axis=-1, keepdims=True)
            var = jnp.var(x, axis=-1, keepdims=True)
            normed_x = (x - mean) * jax.lax.rsqrt(var + self.eps)
        elif self.norm_type == "rms_norm":
            var = jnp.mean(jnp.square(x), axis=-1, keepdims=True)
            normed_x = x * jax.lax.rsqrt(var + self.eps)
        else:
            raise ValueError(f"Unsupported norm_type: {self.norm_type}")

        # 2. Compute modulation (typically SiLU is applied before projection in DiTs)
        modulation = self.linear(jax.nn.silu(vec))

        return normed_x, modulation

class AdaLayerNormChunks(AdaLayerNorm):
    """
    Adaptive Layer Normalization that generates multiple chunks (e.g. 6 chunks for shift/scale/gate of Q/K/V).
    """
    def __init__(
        self,
        embedding_dim: int,
        out_dim: int,
        num_chunks: int,
        norm_type: str = "layer_norm",
        eps: float = 1e-6,
        seed: nn.Rngs | None = None
    ) -> None:
        self.num_chunks = num_chunks
        super().__init__(
            embedding_dim=embedding_dim,
            out_dim=out_dim * num_chunks,
            norm_type=norm_type,
            eps=eps,
            seed=seed
        )

    def __call__(self, x: jax.Array, vec: jax.Array) -> tuple[jax.Array, tuple[jax.Array, ...]]:
        normed_x, modulation = super().__call__(x, vec)

        # Split modulation into chunks along the last dimension
        chunks = tuple(jnp.split(modulation, self.num_chunks, axis=-1))

        # Return the parameter-free normalized x and the chunks tuple
        return normed_x, chunks



class SpatialNorm(nn.Module):
    """
    Spatially conditioned normalization as defined in https://huggingface.co/papers/2209.09002.

    Args:
        f_channels (`int`):
            The number of channels for input to group normalization layer, and output of the spatial norm layer.
        zq_channels (`int`):
            The number of channels for the quantized vector as described in the paper.
    """

    def __init__(
        self,
        f_channels: int,
        zq_channels: int,
        rngs: nn.Rngs
    ) -> None:
        super().__init__()
        # self.norm_layer = nn.GroupNorm(num_channels=f_channels, num_groups=32, eps=1e-6, affine=True)
        self.norm_layer = nn.GroupNorm(num_channels=f_channels, num_groups=32, eps=1e-6)
        self.conv_y = nn.Conv2d(zq_channels, f_channels, kernel_size=1, stride=1, padding=0, rngs=rngs)
        self.conv_b = nn.Conv2d(zq_channels, f_channels, kernel_size=1, stride=1, padding=0, rngs=rngs)

    def forward(self, f: jax.Array, zq: jax.Array) -> jax.Array:
        zq = jax.image.resize(zq, shape=f.shape, method='nearest')
        norm_f = self.norm_layer(f)
        new_f = norm_f * self.conv_y(zq) + self.conv_b(zq)
        return new_f