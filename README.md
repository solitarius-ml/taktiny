# Taktiny

Taktiny is an experimental neural-network library built directly on JAX. It
provides object-oriented modules that remain valid JAX PyTrees, a small set of
transformer building blocks, and `Maestro` for loading compatible Hugging Face
checkpoints into native Taktiny models.

The project is under active development. APIs, checkpoint mappings, and model
coverage may change between revisions.

## Highlights

- Stateful `nn.Module` and `nn.Parameter` objects registered as JAX PyTrees
- Native support for `jax.jit`, `jax.value_and_grad`, and Optax
- Safetensors checkpoint loading and saving
- Abstract model construction through `jax.eval_shape`
- Reusable transformer decoder, model, and causal-LM components
- KV-cached autoregressive generation
- Logical parameter axes and optional JAX mesh sharding
- Experimental quantized linear layers and LoRA utilities

## Requirements

- Python 3.11 or newer
- JAX 0.10.2 or newer


## Quick Start

The following example loads a Qwen2 checkpoint and generates text:

```python
from taktiny import Maestro
from transformers import AutoTokenizer

repo = "Qwen/Qwen2.5-0.5B"

tokenizer = AutoTokenizer.from_pretrained(repo)
model = Maestro.from_pretrained(repo)

input_ids = tokenizer.encode(
    "Once upon a time",
    return_tensors="np",
)

output_ids = model.generate(
    input_ids,
    max_new_tokens=50,
    temperature=0.7,
    top_p=0.9,
)

print(tokenizer.decode(output_ids[0], skip_special_tokens=True))
```

Model loading and generation materialize checkpoint parameters and KV caches.
Choose a checkpoint and dtype that fit the available device and host memory.

## Implemented Architectures

| Hugging Face architecture | Taktiny class | Status |
| --- | --- | --- |
| `LlamaForCausalLM` | `Llama` | Implemented |
| `Qwen2ForCausalLM` | `Qwen2` | Implemented |
| `GemmaForCausalLM` | `Gemma` | Implemented |

Other architecture names may appear in the internal repertoire as development
placeholders. Registration alone does not mean that checkpoint loading or
inference is implemented.

You can inspect the architecture registry with:

```python
from taktiny import Maestro

print(Maestro.available())
```

## Inspecting Shapes

`Maestro.eval_shape` constructs an abstract model from repository
configuration without allocating parameter buffers or downloading checkpoint
weights:

```python
from taktiny import Maestro

abstract_model = Maestro.eval_shape("Qwen/Qwen2.5-0.5B")
print(abstract_model)
```

This is useful for inspecting parameter counts, shapes, and dtypes before
loading a checkpoint. It does not estimate temporary compilation memory or KV
cache usage.

## Building Modules

Taktiny modules keep parameters directly on the object while participating in
JAX transformations:

```python
import jax

from taktiny import nn


class MLP(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: int,
        *,
        rngs: nn.Rngs,
    ):
        self.input = nn.Linear(
            in_features,
            hidden_features,
            rngs=rngs,
        )
        self.output = nn.Linear(
            hidden_features,
            out_features,
            rngs=rngs,
        )

    def __call__(self, x):
        return self.output(jax.nn.silu(self.input(x)))


model = MLP(64, 128, 10, rngs=nn.Rngs(42))
jitted_model = jax.jit(model)
```

Parameters can be inspected or restored through flat and nested state
dictionaries:

```python
flat_state = model.flat_state_dict()
state = model.state_dict()

model.load_state_dict(state)
```

Models derived from `PretrainedModel` can also write Safetensors checkpoints:

```python
model.save_pretrained("./checkpoint")
```

## Defining A Transformer Family

`TransformerDecoderLayer` creates modules in the order supplied by the family
implementation. Normalization modules transform the active hidden state;
attention and MLP modules are applied as residual branches.

```python
from taktiny import nn
from taktiny.cosettes._common import (
    TransformerCausalLM,
    TransformerDecoderLayer,
)
from taktiny.layers import Attention, GateMLP


class ExampleDecoderLayer(TransformerDecoderLayer):
    def __init__(self, config, rngs: nn.Rngs):
        super().__init__(
            config,
            rngs=rngs,
            input_layernorm=nn.RMSNorm,
            self_attn=Attention,
            post_attention_layernorm=nn.RMSNorm,
            mlp=GateMLP,
        )


class ExampleForCausalLM(TransformerCausalLM):
    def __init__(
        self,
        config,
        rngs: nn.Rngs = None,
        mesh=None,
        sharding_rules=None,
    ):
        if rngs is None:
            rngs = nn.Rngs(42)

        super().__init__(
            config,
            rngs=rngs,
            decoder=ExampleDecoderLayer,
            norm=nn.RMSNorm,
            mesh=mesh,
            sharding_rules=sharding_rules,
        )
```

Checkpoint-facing attribute names such as `self_attn`, `input_layernorm`, and
`mlp` should match the source checkpoint wherever possible. This minimizes
weight-mapping rules.

## Training

The experimental `Trainer` accepts native Taktiny models and uses Optax for
updates:

```python
import optax

from taktiny import DatasetConfig, Trainer, TrainingConfig

trainer = Trainer(
    model=model,
    loss_fn=loss_fn,
    training_config=TrainingConfig(
        epochs=1,
        max_steps=1_000,
        optimizer=optax.adamw(3e-4),
        log_interval=10,
        jit_compile=True,
    ),
    dataset_config=DatasetConfig(dataloader=train_batches),
)

trainer.train()
```

The trainer currently uses heuristic parameter freezing for large and
quantized parameters. Review the trainable parameter set before using it for a
real training run.

## Project Layout

```text
src/taktiny/
├── nn/                 Object-oriented JAX modules and parameters
├── layers/             Attention, feed-forward, positional, and vision layers
├── cosettes/           Reusable model implementations
├── maestro/            Architecture registry and checkpoint orchestration
├── trainer/            Experimental training utilities
└── utils/              Sharding, quantization, typing, and weight mapping
```

## License

Taktiny is distributed under the Apache License 2.0. See
[`LICENSE.md`](LICENSE.md).
