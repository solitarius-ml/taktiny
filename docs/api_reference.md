# Complete API Reference

This document provides a comprehensive sitemap of symbols, subpackages, modules, and classes available in TakTiny.

---

## 1. Top-Level Imports (`taktiny`)

| Symbol | Module / Path | Description |
| :--- | :--- | :--- |
| `Maestro` | `taktiny.maestro._prelude` | Architectural registry, `eval_shape`, and `from_pretrained` entry point |
| `ModelConfig` | `taktiny.maestro._config` | Configuration class for HF `config.json` loading and resolution |
| `Takt` | `taktiny.takt` | Full-lifecycle training framework |
| `LoraConfig` | `taktiny.peft` | Low-Rank Adaptation (LoRA) configuration |
| `apply_peft` | `taktiny.peft` | Applies PEFT adapters to modules |
| `tt` | `taktiny.transforms` | Data transformations and preprocessing subpackage |

---

## 2. Neural Network Primitives (`taktiny.nn`)

| Symbol | Description |
| :--- | :--- |
| `nn.Module` | Base class for object-oriented JAX modules |
| `nn.Parameter` | Parameter leaf container for module weights |
| `nn.Rngs` | PRNG key manager for parameter initialization and stochastic ops |
| `nn.Linear` | Multi-dimensional linear projection layer |
| `nn.Embedding` | Sparse lookup table |
| `nn.RMSNorm` | Root Mean Square Normalization |
| `nn.LayerNorm` | Layer Normalization |
| `nn.List` | Container list of submodules |
| `nn.SeqStack` | Homogeneous module stack executed via `jax.lax.scan` |
| `nn.Sequential` | Chained sequential module pipeline |

---

## 3. High-Level Layers (`taktiny.layers`)

| Layer Class | Custom Kernel Entry Points | Description |
| :--- | :--- | :--- |
| `Attention` | `apply_flash_attention`, `apply_splash_attention`, `apply_ragged_attention`, `apply_ring_attention` | MHA, MQA, and GQA Attention module |
| `MoeFFN` | `apply_gmm`, `apply_route`, `apply_unroute` | Mixture-of-Experts FFN layer |
| `GateMLP` | `apply` | SwiGLU / Gated MLP block |
| `FusedGateMLP` | `apply` | Fused Gate-Up MLP projection block |
| `RotaryEmbedding` | N/A | Rotary Position Embeddings (RoPE) |

---

## 4. Custom Kernels (`taktiny.kernel`)

| Subpackage | Kernel Name | Target Hardware |
| :--- | :--- | :--- |
| `taktiny.kernel.attention` | `flash_attention_block_masked` | GPU / TPU |
| `taktiny.kernel.attention` | `splash_attention` | TPU / Mosaic |
| `taktiny.kernel.attention` | `ragged_mqa` / `ragged_gqa` | GPU / TPU |
| `taktiny.kernel.attention` | `ring_attention` | Distributed Mesh |
| `taktiny.kernel.megablox` | `gmm` / `gmm_v2` | TPU / Mosaic |
| `taktiny.kernel.megablox` | `route` / `unroute` | CPU / GPU / TPU |
| `taktiny.kernel.ragged` | `sc_gather_reduce` | TPU SparseCore |

---

## 5. Model Base Classes (`taktiny.cosettes._common`)

| Class | Description |
| :--- | :--- |
| `PretrainedModel` | Base pretrained model handling config, weights, and remat |
| `TransformerModel` | Decoder layer stack wrapper with `inputs_embeds` support |
| `TransformerCausalLM` | Causal language model wrapper with autoregressive generation |
| `TransformerConditionalGeneration` | Unified multimodal base class (Text, Vision, Audio) |
| `DiffusionLM` / `DiffusionIM` | Base classes for Diffusion language & image models |

---

## 6. Supported Maestro Architectures (`taktiny.maestro.opus`)

| HF Architecture Name | TakTiny Class | Subpackage |
| :--- | :--- | :--- |
| `Gemma4UnifiedForConditionalGeneration` | `Gemma4Unified` | `taktiny.maestro.opus.gemma` |
| `Gemma4ForConditionalGeneration` | `Gemma4` | `taktiny.maestro.opus.gemma` |
| `Gemma3ForConditionalGeneration` | `Gemma3ConditionalGeneration` | `taktiny.maestro.opus.gemma` |
| `Llama4ForConditionalGeneration` | `Llama4` | `taktiny.maestro.opus.llama` |
| `Qwen3_5MoeForConditionalGeneration` | `Qwen3_5MoE` | `taktiny.maestro.opus.qwen` |
| `DeepseekV4ForCausalLM` | `DeepseekV4` | `taktiny.maestro.opus.deepseek` |
| `GptOssForCausalLM` | `GPTOSS` | `taktiny.maestro.opus.gpt` |
