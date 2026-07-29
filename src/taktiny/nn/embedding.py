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
        rngs: Rngs = None, 
        seed: Rngs = None, 
        dtype: DType | str = jnp.float32,
        initializer = normal(0.02),
        quant=None,
    ):
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

    def extra_repr(self):
        return f"{self.num_embeddings} → {self.embedding_dim}"

class SinusoidalPositionalEmbedding(Module):
    def __init__(self, embedding_dim: int):
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
