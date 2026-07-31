import jax.numpy as jnp

from taktiny import nn
from taktiny.layers import RotaryEmbedding
from taktiny.maestro._config import ModelConfig
from taktiny.maestro.opus.llama import Llama


def test_nested_model_config_supports_mapping_get():
    config = ModelConfig(
        rope_scaling={
            'rope_type': 'llama3',
            'factor': 8.0,
            'original_max_position_embeddings': 8192,
        }
    )

    assert config.rope_scaling.get('rope_type') == 'llama3'
    assert config.rope_scaling.get('factor', 1.0) == 8.0
    assert config.rope_scaling.get('low_freq_factor', 1.0) == 1.0


def test_rotary_embedding_accepts_nested_model_config():
    config = ModelConfig(
        rope_scaling={
            'rope_type': 'llama3',
            'factor': 8.0,
            'low_freq_factor': 1.0,
            'high_freq_factor': 4.0,
            'original_max_position_embeddings': 8192,
        }
    )
    rotary = RotaryEmbedding(8, rope_scaling=config.rope_scaling)
    q = jnp.ones((1, 4, 2, 8), dtype=jnp.float32)
    k = jnp.ones((1, 4, 1, 8), dtype=jnp.float32)

    rotated_q, rotated_k = rotary(q, k)

    assert rotated_q.shape == q.shape
    assert rotated_k.shape == k.shape
    assert jnp.all(jnp.isfinite(rotated_q))
    assert jnp.all(jnp.isfinite(rotated_k))


def test_scanned_llama_accepts_nested_rope_scaling_config():
    config = ModelConfig(
        num_hidden_layers=2,
        vocab_size=16,
        hidden_size=8,
        intermediate_size=16,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
        max_position_embeddings=32,
        rope_theta=500000.0,
        rope_scaling={
            'rope_type': 'llama3',
            'factor': 8.0,
            'low_freq_factor': 1.0,
            'high_freq_factor': 4.0,
            'original_max_position_embeddings': 8192,
        },
        rms_norm_eps=1e-5,
        dtype='float32',
    )
    model = Llama(config, rngs=nn.Rngs(0), use_list=False)

    logits, context = model(jnp.asarray([[1, 2, 3]], dtype=jnp.int32))

    assert logits.shape == (1, 3, config.vocab_size)
    assert context is None
    assert jnp.all(jnp.isfinite(logits))
