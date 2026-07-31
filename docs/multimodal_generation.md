# Multimodal Conditional Generation (`TransformerConditionalGeneration`)

`TransformerConditionalGeneration` is TakTiny's unified base class for multimodal language models (such as Gemma 3/4, Llama 4, and Qwen 3.5 VL).

---

## 1. Class Architecture

```text
TransformerConditionalGeneration
├── language_model: TransformerCausalLM (Text Backbone)
├── vision_tower: Vision Encoder Module (e.g., SigLIP, ViT)
├── multi_modal_projector: Vision Projection Layer
├── audio_tower: Audio Encoder Module (e.g., Whisper, Conformer)
└── audio_projector: Audio Projection Layer
```

---

## 2. Feature Encoding & Embedding Fusion

Visual and audio inputs are encoded via `encode_vision` and `encode_audio`, and merged into text embedding sequences at specified placeholder token positions (`image_token_id`, `audio_token_id`):

```python
import jax
import jax.numpy as jnp
from taktiny.cosettes._common import TransformerConditionalGeneration, ModelConfig
from taktiny import nn

# Instantiate multimodal model
config = ModelConfig(
    vocab_size=1000,
    hidden_size=64,
    num_hidden_layers=2,
    num_attention_heads=4,
    num_key_value_heads=4,
    intermediate_size=128,
    max_position_embeddings=2048,
    layer_types=["full", "full"],
    query_pre_attn_scalar=16.0,
    head_dim=16,
    rms_norm_eps=1e-6,
)

model = TransformerConditionalGeneration(
    config=config,
    rngs=nn.Rngs(42),
    image_token_id=256,
)

# Text prompt input IDs containing placeholder token ID (256)
input_ids = jnp.array([[1, 2, 256, 256, 3, 4]])

# Forward pass with pixel values
logits, _ = model(input_ids=input_ids, pixel_values=pixel_values)
print("Logits shape:", logits.shape)  # (1, 6, 1000)
```

---

## 3. Autoregressive Generation (`generate`)

Run text and multimodal prompt prefilling and autoregressive decoding using the unified `.generate()` method:

```python
generated_tokens = model.generate(
    input_ids=input_ids,
    pixel_values=pixel_values,
    max_new_tokens=20,
    temperature=0.7,
    top_k=50,
)
print("Generated token IDs shape:", generated_tokens.shape)
```
