# Getting Started with TakTiny

This guide helps you set up TakTiny and run your first models and custom kernels in 5 minutes.

---

## 1. Installation

TakTiny is built on top of JAX and can be installed via `uv` or `pip`:

```bash
# Using uv (recommended)
uv add taktiny

# Or using pip
pip install taktiny
```

Ensure JAX is installed with appropriate hardware support (CPU, GPU, or TPU).

---

## 2. Quickstart Examples

### Example A: Zero-Allocation Model Inspection (`Maestro.eval_shape`)

Inspect the exact shape, layer hierarchy, and parameters of large models (such as `google/gemma-4-12B-it`) with zero memory allocation:

```python
import jax
from taktiny.maestro import Maestro

# Construct abstract model without loading or allocating physical GPU/TPU memory
abstract_model = Maestro.eval_shape("google/gemma-4-12B-it")

print("Model Class:", type(abstract_model).__name__)
print("Language Model:", type(abstract_model.language_model).__name__)
print("Embedding Shape:", abstract_model.language_model.model.embed_tokens.embedding.value.shape)
```

---

### Example B: Building a Custom Transformer Module

Use TakTiny's OOP `nn.Module` semantics and `nn.Rngs` state management:

```python
import jax
import jax.numpy as jnp
from taktiny import nn, layers

class CustomTransformerBlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, head_dim: int, rngs: nn.Rngs):
        self.attn_norm = nn.RMSNorm(hidden_size)
        self.attn = layers.Attention(hidden_size, num_heads, head_dim, rngs=rngs)
        self.mlp_norm = nn.RMSNorm(hidden_size)
        self.mlp = layers.GateMLP(hidden_size, intermediate_size=hidden_size * 4, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        # Residual attention block
        h = x + self.attn(self.attn_norm(x))[0]
        # Residual MLP block
        out = h + self.mlp(self.mlp_norm(h))
        return out

rngs = nn.Rngs(42)
block = CustomTransformerBlock(hidden_size=256, num_heads=8, head_dim=32, rngs=rngs)

x = jnp.ones((2, 16, 256))
output = block(x)
print("Output shape:", output.shape)  # (2, 16, 256)
```

---

### Example C: Invoking Custom Kernel Entry Points

Apply high-performance kernels directly via classmethods on layer classes:

```python
import jax
import jax.numpy as jnp
from taktiny.layers import Attention, MoeFFN

# 1. FlashAttention
q = jnp.ones((2, 8, 4, 16))
k = jnp.ones((2, 8, 4, 16))
v = jnp.ones((2, 8, 4, 16))
flash_out = Attention.apply_flash_attention(q, k, v)
print("FlashAttention output shape:", flash_out.shape)

# 2. Megablox MoE Grouped Matrix Multiply (GMM)
tokens = jnp.ones((16, 32))
indices = jnp.array([0, 1, 0, 1, 2, 2, 1, 0, 0, 1, 2, 0, 1, 2, 0, 1])
sorted_tokens, group_sizes = MoeFFN.apply_route(tokens, indices, num_groups=3)

weights = jnp.ones((3, 32, 64))
gmm_out = MoeFFN.apply_gmm(sorted_tokens, weights, group_sizes)
print("MoE GMM output shape:", gmm_out.shape)  # (16, 64)
```

---

## Next Steps

- Explore [Core Concepts](core_concepts.md) to understand state and sharding.
- Browse [Maestro Architectures](maestro_architectures.md) for pre-trained model loading.
- Check [Custom Kernels](custom_kernels.md) for Pallas attention and MoE ops.
