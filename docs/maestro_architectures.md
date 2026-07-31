# Maestro Architectures

The `Maestro` module provides the architecture registry (`repertoire`), abstract shape evaluation (`eval_shape`), and weight loading (`from_pretrained`).

---

## 1. Zero-Allocation Shape Inspection (`Maestro.eval_shape`)

`Maestro.eval_shape` constructs an abstract model under `jax.eval_shape`. It downloads or reads configuration metadata (`config.json`), resolves the model class registered in `repertoire`, and returns an abstract model tree of `jax.ShapeDtypeStruct` shapes **without allocating physical GPU/TPU memory** or downloading checkpoint weights.

```python
from taktiny.maestro import Maestro

# Evaluate Gemma 4 12B abstractly
abstract_model = Maestro.eval_shape("google/gemma-4-12B-it")

print("Model Class:", type(abstract_model).__name__)
print("Language Model:", type(abstract_model.language_model).__name__)
print("Embedding Shape:", abstract_model.language_model.model.embed_tokens.embedding.value.shape)
```

---

## 2. Loading Pretrained Weights (`Maestro.from_pretrained`)

`Maestro.from_pretrained` downloads safetensors weights from Hugging Face Hub or reads from a local directory, maps checkpoint tensors to TakTiny module paths, and instantiates the initialized model:

```python
from taktiny.maestro import Maestro

# Load pretrained model
model = Maestro.from_pretrained("google/gemma-2-9b-it")
```

---

## 3. Supported Model Architectures

TakTiny registers architectures via `repertoire.register(hf_architecture_name, model_class)` in `taktiny.maestro.opus`:

### Multimodal Conditional Generation (`TransformerConditionalGeneration`)
- **Gemma 3 & 4**: `Gemma3ForConditionalGeneration`, `Gemma4ForConditionalGeneration`, `Gemma4UnifiedForConditionalGeneration`
- **Llama 4**: `Llama4ForConditionalGeneration`
- **Qwen 3.5 MoE**: `Qwen3_5MoeForConditionalGeneration`

### Causal Language Models (`TransformerCausalLM`)
- **Gemma Series**: `GemmaForCausalLM`, `Gemma2ForCausalLM`, `Gemma3ForCausalLM`
- **Llama Series**: `LlamaForCausalLM`
- **DeepSeek Series**: `DeepseekForCausalLM`, `DeepseekV2ForCausalLM`, `DeepseekV3ForCausalLM`, `DeepseekV32ForCausalLM`, `DeepseekV4ForCausalLM`
- **Qwen Series**: `QwenForCausalLM`, `Qwen2ForCausalLM`, `Qwen3ForCausalLM`, `Qwen3MoeForCausalLM`, `Qwen3NextForCausalLM`
- **GPT Series**: `GptOssForCausalLM`

### Diffusion Models (`DiffusionIM` / `DiffusionLM`)
- **Flux / DiT**: `DiffusionGemmaForBlockDiffusion`, `FluxForDiffusion`

---

## 4. Custom Architecture Registration

Register new custom architectures in `repertoire` to enable `Maestro.eval_shape` and `from_pretrained`:

```python
from taktiny.maestro._livret import repertoire
from taktiny.cosettes._common import TransformerCausalLM

class MyCustomLM(TransformerCausalLM):
    def __init__(self, config, rngs, **kwargs):
        super().__init__(config, rngs=rngs, decoder=MyCustomDecoderLayer, **kwargs)

# Register Hugging Face architecture name
repertoire.register('MyCustomLMForCausalLM', MyCustomLM)
```
