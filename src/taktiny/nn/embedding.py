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
"""Embedding modules"""
from __future__ import annotations

from typing import Any


import jax
import math
import jax.numpy as jnp
import qwix
from jax.nn.initializers import normal

from taktiny.nn.module import Module, Parameter
from taktiny.nn.rng import Rngs
from taktiny.utils.typing import DType

class Embedding(Module):
    def __init__(
        self, num_embeddings: int,
        embedding_dim: int, *,
        rngs: Rngs | None = None,
        seed: Rngs | None = None,
        dtype: DType | str = jnp.float32,
        initializer: Any = normal(0.02),
        quant: Any=None,
    ) -> None:
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        if rngs is None and seed is None:
            raise ValueError("A rngs must be provided to initialize Embedding layer")

        if rngs is None and seed is not None:
            import warnings
            warnings.warn('seed is deprecated. use `rngs` instead')
            rngs = seed

        key = rngs()
        self.embedding = Parameter(
            initializer(key, (num_embeddings, embedding_dim), dtype)
        )
        self.embedding.quantization = quant
        self.embedding.quantization_kind = 'embedding'

    def __call__(self, indices: jax.Array) -> jax.Array:
        table = self.embedding.value
        if isinstance(table, qwix.QArray):
            return qwix.dequantize(table[indices])
        return table[indices]

    @classmethod
    def apply_gather_reduce(
        cls,
        operand: jax.Array,
        indices: jax.Array,
        weights: jax.Array | None = None,
        reduce_group_size: int = 1,
        **kwargs: Any,
    ) -> jax.Array:
        """Apply Stream Gather Reduce kernel for sparse embeddings / reductions."""
        from taktiny.kernels.gather_reduce_sc import sc_gather_reduce
        if jax.default_backend() != "tpu":
            gathered = operand[indices]
            if weights is not None:
                gathered = gathered * weights[..., None]
            return gathered
        return sc_gather_reduce(operand, indices, topk_weights=weights, reduce_group_size=reduce_group_size, **kwargs)

    @classmethod
    def apply_ragged_gather(
        cls,
        operand: jax.Array,
        offsets: jax.Array,
        lengths: jax.Array,
        **kwargs: Any,
    ) -> jax.Array:
        """Apply Ragged Gather kernel."""
        from taktiny.kernels.ragged.ragged_gather import ragged_gather
        return ragged_gather(operand, offsets, lengths, **kwargs)

    @classmethod
    def apply(
        cls,
        operand: jax.Array,
        indices: jax.Array,
        kernel: str = "gather_reduce",
        **kwargs: Any,
    ) -> jax.Array:
        """Unified entry point for Embedding sparse gather kernels."""
        if kernel in ("gather_reduce", "sc_gather_reduce"):
            return cls.apply_gather_reduce(operand, indices, **kwargs)
        elif kernel in ("ragged", "ragged_gather"):
            return cls.apply_ragged_gather(operand, indices, **kwargs)
        else:
            return operand[indices]

    def extra_repr(self) -> str:
        return f"{self.num_embeddings} → {self.embedding_dim}"

class SinusoidalPositionalEmbedding(Module):
    def __init__(self, embedding_dim: int) -> None:
        self.embedding_dim = embedding_dim

    def __call__(self, timesteps: jax.Array) -> jax.Array:
        is_scalar = timesteps.ndim == 0
        if is_scalar:
            timesteps = jnp.expand_dims(timesteps, 0)

        half_dim = self.embedding_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = jnp.exp(jnp.arange(half_dim, dtype=jnp.float32) * -emb)
        # timesteps shape is (B,) or (1,) if it was scalar
        emb = timesteps[:, None] * emb[None, :]
        emb = jnp.concatenate([jnp.sin(emb), jnp.cos(emb)], axis=-1)

        # If embedding_dim is odd, pad by zero
        if self.embedding_dim % 2 == 1:
            emb = jnp.pad(emb, ((0, 0), (0, 1)))

        return jnp.squeeze(emb, 0) if is_scalar else emb
