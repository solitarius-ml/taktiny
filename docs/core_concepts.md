# Core Concepts in TakTiny

TakTiny bridges object-oriented Python design with JAX's functional transformation model. This page explains key architectural concepts.

---

## 1. Object-Oriented JAX Modules (`nn.Module`)

In TakTiny, neural network components inherit from `nn.Module`. Unlike standard Flax or PyTorch modules, TakTiny modules manage state and parameters cleanly while remaining 100% compatible with JAX transformations (`jax.jit`, `jax.vmap`, `jax.grad`, `jax.eval_shape`).

### Parameter Initialization
Parameters are declared using TakTiny's initializers and stored on module instances:

```python
from taktiny import nn
import jax.numpy as jnp

class MyLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, rngs: nn.Rngs):
        self.in_features = in_features
        self.out_features = out_features
        # Parameter registered as a PyTree leaf
        self.weight = nn.Parameter(
            jax.random.normal(rngs(), (in_features, out_features)) * 0.02
        )

    def __call__(self, x):
        return jnp.dot(x, self.weight.value)
```

---

## 2. PRNG Key Management (`nn.Rngs`)

Random number generator (PRNG) state management is handled through `nn.Rngs`. Passing an `nn.Rngs` instance enables reproducible parameter initializations and dropout generation:

```python
from taktiny import nn

# Initialize Rngs with an integer seed or JAX key
rngs = nn.Rngs(42)

# Calling rngs() splits the internal key and returns a new subkey
key1 = rngs()
key2 = rngs()
```

---

## 3. Sharding & Parallelism Modes (`ShardMode`)

TakTiny supports explicit and automatic device mesh sharding via `ShardMode`:

- **`ShardMode.AUTO`**: Lets JAX auto-shard tensors across available devices.
- **`ShardMode.EXPLICIT`**: Uses logical-to-mesh axis mapping rules (e.g. Tensor Parallelism `'tp'`, FSDP `'fsdp'`).

### Explicit Device Mesh Sharding
```python
import jax
from jax.sharding import Mesh
from jax.experimental import mesh_utils
from taktiny import nn, ShardMode

# Define 2D Device Mesh (2 FSDP x 4 TP)
devices = mesh_utils.create_device_mesh((2, 4))
mesh = Mesh(devices, ('fsdp', 'tp'))

# Specify model with explicit sharding rules
model = MyModel(config, rngs=rngs, mesh=mesh, shard_mode=ShardMode.EXPLICIT)
```

---

## 4. Context-Wide Rematerialization (`enable_remat`)

Gradient rematerialization (activation checkpointing) reduces VRAM usage during backward passes by recomputing intermediate activations instead of storing them:

```python
model = TransformerCausalLM(config, rngs=rngs, decoder=LlamaDecoderLayer)

# Enable rematerialization across all decoder layers
model.enable_remat()
```
