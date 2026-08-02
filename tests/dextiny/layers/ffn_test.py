import dextiny as dx
import jax
import jax.numpy as jnp
import pytest

from taktiny import nn
from taktiny.layers import FusedGateMLP, GateMLP, MLP


def linear(x, weight, bias=None, input_dims=1):
    x_axes = tuple(range(x.ndim - input_dims, x.ndim))
    weight_axes = tuple(range(input_dims))
    output = jax.lax.dot_general(
        x,
        weight,
        ((x_axes, weight_axes), ((), ())),
    )
    return output if bias is None else output + bias


def gated_hidden(x, gate_weight, up_weight, *, activation):
    return activation(linear(x, gate_weight)) * linear(x, up_weight)


def fused_gated_hidden(x, weight, *, activation):
    hidden, gate = jnp.split(linear(x, weight), 2, axis=-1)
    return hidden * activation(gate)


def activated_hidden(x, weight, bias, *, activation):
    return activation(linear(x, weight, bias))


def output_without_bias(x, weight):
    return linear(x, weight)


def output_with_bias(x, weight, bias):
    return linear(x, weight, bias)


def assert_matches(compiled, target):
    assert compiled.verify(target), compiled.report(target).render()


@pytest.mark.parametrize(
    ("activation", "activation_name"),
    ((jax.nn.silu, "silu"), (jax.nn.gelu, "gelu")),
)
def test_gate_mlp_matches_reference_trace(activation, activation_name):
    hidden = dx.AbstractArray("B S D", dtype="float32")
    reference = (
        hidden
        >> dx.AbstractModule(
            gated_hidden,
            dx.AbstractArray("D I", dtype="float32", name="gate_weight"),
            dx.AbstractArray("D I", dtype="float32", name="up_weight"),
            name=f"{activation_name}_gate",
            kwargs={"activation": activation},
        )
        >> dx.AbstractModule(
            output_without_bias,
            dx.AbstractArray("I D", dtype="float32", name="down_weight"),
            name="down_projection",
        )
    )
    compiled = reference.compile(B=2, S=8, D=16, I=32)
    module = GateMLP(
        hidden_size=16,
        intermediate_size=32,
        activation=activation,
        bias=False,
        dtype="float32",
        rngs=nn.Rngs(0),
    )

    assert_matches(compiled, module)


def test_fused_gate_mlp_matches_reference_trace():
    hidden = dx.AbstractArray("B S D", dtype="float32")
    reference = (
        hidden
        >> dx.AbstractModule(
            fused_gated_hidden,
            dx.AbstractArray("D 2*I", dtype="float32", name="in_weight"),
            name="fused_silu_gate",
            kwargs={"activation": jax.nn.silu},
        )
        >> dx.AbstractModule(
            output_without_bias,
            dx.AbstractArray("I D", dtype="float32", name="out_weight"),
            name="output_projection",
        )
    )
    compiled = reference.compile(B=2, S=8, D=16, I=32)
    module = FusedGateMLP(
        hidden_size=16,
        intermediate_size=32,
        activation=jax.nn.silu,
        bias=False,
        dtype="float32",
        seed=nn.Rngs(0),
    )

    assert_matches(compiled, module)


def test_mlp_matches_reference_trace():
    hidden = dx.AbstractArray("B S D", dtype="float32")
    reference = (
        hidden
        >> dx.AbstractModule(
            activated_hidden,
            dx.AbstractArray("D I", dtype="float32", name="fc1_weight"),
            dx.AbstractArray("I", dtype="float32", name="fc1_bias"),
            name="gelu_projection",
            kwargs={"activation": jax.nn.gelu},
        )
        >> dx.AbstractModule(
            output_with_bias,
            dx.AbstractArray("I D", dtype="float32", name="fc2_weight"),
            dx.AbstractArray("D", dtype="float32", name="fc2_bias"),
            name="output_projection",
        )
    )
    compiled = reference.compile(B=2, S=8, D=16, I=32)
    module = MLP(
        hidden_size=16,
        intermediate_size=32,
        activation=jax.nn.gelu,
        bias=True,
        dtype="float32",
        rngs=nn.Rngs(0),
    )

    assert_matches(compiled, module)
