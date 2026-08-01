# Taktiny Documentation

Taktiny is an experimental neural-network library built directly on JAX. It
provides object-oriented modules that are JAX PyTrees, transformer building
blocks, Hugging Face Safetensors loading, Qwix weight-only quantization, PEFT,
and an Optax-based training loop.

The project is under active development. APIs and model coverage can change
between revisions.

## Current Scope

- `nn.Module` and `nn.Parameter` objects compatible with JAX transformations
- Causal language models with KV-cached generation and streaming
- Hugging Face configuration and checkpoint loading through `Maestro`
- Full-model and LoRA-adapter Safetensors serialization
- Qwix INT8, INT4, NF4, and selective PTQ while loading
- Logical parameter axes, JAX mesh placement, and decoder rematerialization
- Grain-compatible data operations and resumable trainer checkpoints
- Low-level attention, MoE, ragged, and SparseCore kernel entry points

The implemented causal model families are Llama, original Qwen, Qwen2,
Qwen3, Gemma, Gemma2, and text-only Gemma3. Other architecture names in the
internal registry are development placeholders and are not supported merely
because they are registered.

Multimodal generation, native Taktiny execution through vLLM TPU, and concrete
RL algorithms remain experimental or incomplete. Their current boundaries are
documented explicitly rather than presented as production features.

## Documentation

- [Getting Started](getting_started.md): install, load, inspect, and generate
- [Core Concepts](core_concepts.md): modules, parameters, RNGs, stacks, sharding,
  and rematerialization
- [Models and Checkpoints](models_and_checkpoints.md): Maestro, implemented
  families, quantized loading, serialization, and Hub upload
- [Generation](generation.md): native batched generation, sampling, streaming,
  and forward contexts
- [Data Pipelines](data.md): Grain composition, batched tokenization, packing,
  workers, and resume boundaries
- [Training](training.md): losses, Optax, gradient controls, evaluation,
  callbacks, sharding, checkpointing, and resume
- [PEFT](peft.md): LoRA, QLoRA-style loading, adapter checkpoints, and merging
- [Layers](layers.md): primitives, transformer layers, containers, and module
  transformations
- [Kernels](kernels.md): attention, MoE, embedding entry points, and hardware
  constraints
- [Experimental APIs](experimental.md): accurate multimodal, vLLM, RL, and
  architecture-placeholder boundaries
- [API Reference](api_reference.md): compact signatures and symbols by module

## Version Note

These pages describe the latest `experiment` implementation, not the older
source snapshot carried by the documentation worktree itself.
