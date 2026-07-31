import jax
import jax.numpy as jnp
import pytest

from taktiny import nn
from taktiny.cosettes._common import TransformerCausalLM, TransformerModel
from taktiny.maestro._config import ModelConfig


class RematTestLayer(nn.Module):
    def __init__(self, config, rngs, layer_idx=None):
        self.layer_idx = layer_idx
        self.proj = nn.Linear(
            config.hidden_size,
            config.hidden_size,
            bias=False,
            dtype='float32',
            rngs=rngs,
        )

    def __call__(self, x, **kwargs):
        return jax.nn.gelu(self.proj(x)), None


@pytest.mark.parametrize('use_list', [True, False])
def test_transformer_remat_preserves_forward_and_backward(use_list):
    config = ModelConfig(
        num_hidden_layers=2,
        vocab_size=8,
        hidden_size=4,
        dtype='float32',
    )
    model = TransformerModel(
        config,
        rngs=nn.Rngs(0),
        module=RematTestLayer,
        embedding=nn.Embedding,
        use_list=use_list,
    )
    input_ids = jnp.asarray([[1, 2, 3]])

    expected, _ = model(input_ids)
    model.enable_remat()
    actual, _ = model(input_ids)

    assert jnp.allclose(actual, expected)

    def loss(candidate):
        output, _ = candidate(input_ids)
        return jnp.sum(output)

    gradients = jax.grad(loss)(model)
    assert all(
        jnp.all(jnp.isfinite(leaf))
        for leaf in jax.tree.leaves(gradients)
    )
    differentiated = str(jax.make_jaxpr(jax.grad(loss))(model))
    assert 'remat' in differentiated


def test_causal_lm_enable_remat_forwards_to_transformer_model():
    causal_lm = object.__new__(TransformerCausalLM)
    causal_lm.model = type(
        'RematTarget',
        (),
        {
            'enable_remat': lambda self: setattr(self, 'enabled', True),
            'enabled': False,
        },
    )()

    causal_lm.enable_remat()

    assert causal_lm.model.enabled is True
