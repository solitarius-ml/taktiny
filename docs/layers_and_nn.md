# Layers & Neural Networks (`taktiny.nn` & `taktiny.layers`)

TakTiny provides neural network building blocks, container modules, and specialized transformer layers.

---

## 1. Primitives (`taktiny.nn`)

### `nn.Linear`
General linear projection supporting multi-dimensional output shapes, quantization (`quant`), and axis sharding.

```python
from taktiny import nn

# Linear mapping from hidden_size to (num_heads, head_dim)
linear = nn.Linear(768, (12, 64), bias=False, rngs=rngs)
```

### `nn.Embedding`
Sparse lookup table with optional scaling (`GemmaTextScaledWordEmbedding`).

```python
from taktiny import nn

embed = nn.Embedding(vocab_size=32000, embedding_dim=768, rngs=rngs)
```

### `nn.RMSNorm` & `nn.LayerNorm`
Normalizations with configurable epsilon and axis sharding.

```python
from taktiny import nn

norm = nn.RMSNorm(hidden_size=768, eps=1e-6)
```

---

## 2. Container Modules (`taktiny.nn`)

- **`nn.List`**: Python list of heterogeneous or homogeneous submodules. Recommended for hybrid/alternating transformer layer stacks (e.g. Gemma 3/4 sliding vs. full attention layers).
- **`nn.SeqStack`**: Executes homogeneous stacked modules using JAX's `jax.lax.scan` for memory efficiency.
- **`nn.Sequential`**: Sequential pipeline executor for chaining modules.
- **`nn.Block`**: Abstract block container.

---

## 3. Transformer Layers (`taktiny.layers`)

### `layers.Attention`
Multi-Head (MHA), Multi-Query (MQA), and Grouped-Query (GQA) Attention module with custom kernel entry points:

```python
from taktiny import nn, layers

attn = layers.Attention(
    hidden_size=768,
    num_heads=12,
    head_dim=64,
    num_kv_heads=4,  # GQA
    rngs=rngs,
)

# Standard call with kernel selection ("dot_product", "flash", "splash", "ragged", "ring")
output, _ = attn(x, kernel="flash")
```

### `layers.MoeFFN` & `layers.GateMLP`
Mixture-of-Experts (MoE) FFN module with Megablox Grouped Matrix Multiply (GMM) and token sorting:

```python
from taktiny.layers import MoeFFN

# Route tokens to experts
sorted_tokens, group_sizes = MoeFFN.apply_route(tokens, expert_indices, num_groups=8)

# Execute Grouped Matrix Multiply (GMM) across experts
gmm_out = MoeFFN.apply_gmm(sorted_tokens, expert_weights, group_sizes)

# Restore original token order
unrouted = MoeFFN.apply_unroute(gmm_out, expert_indices)
```

### `layers.RotaryEmbedding` (RoPE)
Rotary Position Embedding computation supporting variable frequency scaling:

```python
from taktiny.layers import RotaryEmbedding

rope = RotaryEmbedding(head_dim=64, max_position_embeddings=8192, rope_theta=10000.0)
```
