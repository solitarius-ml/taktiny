# TakTiny Documentation

Welcome to the documentation for **TakTiny**, a high-performance Deep Learning framework built on JAX. TakTiny features object-oriented modeling semantics, high-performance custom kernels, unified multimodal conditional generation, and zero-allocation model shape evaluation.

---

## Key Highlights

- **Object-Oriented JAX Modeling**: Clean, modular `nn.Module` classes with automatic state and PRNG key handling via `nn.Rngs`.
- **Zero-Allocation Shape Inspection**: Abstractly inspect massive models (such as Gemma 4 12B or Llama 4) with `Maestro.eval_shape` without downloading or allocating physical GPU/TPU memory.
- **High-Performance Custom Kernels**: Custom Pallas, Mosaic, and Triton kernels for FlashAttention, SplashAttention, RingAttention, Megablox Grouped Matrix Multiply (GMM), activation routing/sorting, and SparseCore gather-reduce.
- **Unified Multimodal Conditional Generation**: The `TransformerConditionalGeneration` base class unifies text, vision, and audio encoder integration, multimodal embedding fusion, and KV-cache autoregressive generation.
- **Flexible Sharding & Rematerialization**: Native support for tensor-parallel and FSDP mesh sharding via `ShardMode` and context-wide gradient rematerialization with `enable_remat()`.
- **PEFT & Training Lifecycle**: Built-in Parameter-Efficient Fine-Tuning with LoRA (`LoraConfig`) and end-to-end training via `Takt`.

---

## Documentation Navigation

- [Getting Started](getting_started.md): Installation, quickstart guide, and 5-minute practical examples.
- [Core Concepts](core_concepts.md): Design philosophy, state management with `nn.Rngs`, sharding, and rematerialization.
- [Maestro Architectures](maestro_architectures.md): Architectural registry (`repertoire`), `eval_shape`, `from_pretrained`, and supported model families.
- [Layers & Neural Networks](layers_and_nn.md): Neural network primitives (`Linear`, `Embedding`, `RMSNorm`), containers (`SeqStack`, `List`), and layers (`Attention`, `MoeFFN`, `GateMLP`).
- [Custom Kernels](custom_kernels.md): Custom JAX/Pallas kernels and classmethod entry points (`Attention.apply`, `MoeFFN.apply_gmm`).
- [Multimodal Generation](multimodal_generation.md): `TransformerConditionalGeneration`, vision/audio feature encoding, embedding fusion, and generation.
- [PEFT & Training](peft_and_training.md): Fine-tuning with LoRA (`LoraConfig`) and full-lifecycle training with `Takt`.
- [API Reference](api_reference.md): Complete index of symbols, modules, and classes.
