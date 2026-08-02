import dextiny as dx
import jax
import jax.numpy as jnp
import pytest

from taktiny import nn
from taktiny.layers import ResnetBlock2D


def linear(x, weight, bias):
    output = jax.lax.dot_general(
        x,
        weight,
        (((x.ndim - 1,), (0,)), ((), ())),
    )
    return output + bias


def conv2d(x, weight, bias, *, padding):
    output = jax.lax.conv_general_dilated(
        lhs=x,
        rhs=weight,
        window_strides=(1, 1),
        padding=padding,
        dimension_numbers=("NHWC", "HWIO", "NHWC"),
        feature_group_count=1,
    )
    return output + bias


def group_norm(x, weight, bias, *, num_groups, eps):
    batch, height, width, channels = x.shape
    grouped = x.reshape(
        batch,
        height,
        width,
        num_groups,
        channels // num_groups,
    )
    mean = jnp.mean(grouped, axis=(1, 2, 4), keepdims=True)
    variance = jnp.var(grouped, axis=(1, 2, 4), keepdims=True)
    normalized = (grouped - mean) * jax.lax.rsqrt(variance + eps)
    normalized = normalized.reshape(batch, height, width, channels)
    return normalized * weight + bias


def resnet_block(
    x,
    norm1_weight,
    norm1_bias,
    conv1_weight,
    conv1_bias,
    norm2_weight,
    norm2_bias,
    conv2_weight,
    conv2_bias,
    *,
    num_groups,
    eps,
):
    residual = x
    hidden = group_norm(
        x,
        norm1_weight,
        norm1_bias,
        num_groups=num_groups,
        eps=eps,
    )
    hidden = conv2d(
        jax.nn.silu(hidden),
        conv1_weight,
        conv1_bias,
        padding="SAME",
    )
    hidden = group_norm(
        hidden,
        norm2_weight,
        norm2_bias,
        num_groups=num_groups,
        eps=eps,
    )
    hidden = conv2d(
        jax.nn.silu(hidden),
        conv2_weight,
        conv2_bias,
        padding="SAME",
    )
    return hidden + residual


def conditioned_resnet_block(
    x,
    time_embedding,
    residual_weight,
    residual_bias,
    norm1_weight,
    norm1_bias,
    conv1_weight,
    conv1_bias,
    time_weight,
    time_bias,
    norm2_weight,
    norm2_bias,
    conv2_weight,
    conv2_bias,
    *,
    num_groups,
    eps,
):
    residual = conv2d(
        x,
        residual_weight,
        residual_bias,
        padding="SAME",
    )
    hidden = group_norm(
        x,
        norm1_weight,
        norm1_bias,
        num_groups=num_groups,
        eps=eps,
    )
    hidden = conv2d(
        jax.nn.silu(hidden),
        conv1_weight,
        conv1_bias,
        padding="SAME",
    )
    time = linear(jax.nn.silu(time_embedding), time_weight, time_bias)
    hidden = hidden + time[:, None, None, :]
    hidden = group_norm(
        hidden,
        norm2_weight,
        norm2_bias,
        num_groups=num_groups,
        eps=eps,
    )
    hidden = conv2d(
        jax.nn.silu(hidden),
        conv2_weight,
        conv2_bias,
        padding="SAME",
    )
    return hidden + residual


def test_resnet_block_matches_reference_trace():
    features = dx.AbstractArray("B H W C", dtype="float32")
    reference = features >> dx.AbstractModule(
        resnet_block,
        dx.AbstractArray("C", dtype="float32", name="norm1_weight"),
        dx.AbstractArray("C", dtype="float32", name="norm1_bias"),
        dx.AbstractArray("3 3 C C", dtype="float32", name="conv1_weight"),
        dx.AbstractArray("C", dtype="float32", name="conv1_bias"),
        dx.AbstractArray("C", dtype="float32", name="norm2_weight"),
        dx.AbstractArray("C", dtype="float32", name="norm2_bias"),
        dx.AbstractArray("3 3 C C", dtype="float32", name="conv2_weight"),
        dx.AbstractArray("C", dtype="float32", name="conv2_bias"),
        name="resnet_block",
        kwargs={"num_groups": 2, "eps": 1e-5},
    )
    compiled = reference.compile(B=2, H=4, W=4, C=4)
    module = ResnetBlock2D(
        in_channels=4,
        groups=2,
        eps=1e-5,
        seed=nn.Rngs(0),
    )

    assert compiled.verify(module), compiled.report(module).render()


@pytest.mark.filterwarnings("ignore:seed is deprecated")
def test_conditioned_projected_resnet_block_matches_reference_trace():
    features = dx.AbstractArray("B H W C", dtype="float32")
    reference = features >> dx.AbstractModule(
        conditioned_resnet_block,
        dx.AbstractArray("B T", dtype="float32", name="time_embedding"),
        dx.AbstractArray("1 1 C O", dtype="float32", name="residual_weight"),
        dx.AbstractArray("O", dtype="float32", name="residual_bias"),
        dx.AbstractArray("C", dtype="float32", name="norm1_weight"),
        dx.AbstractArray("C", dtype="float32", name="norm1_bias"),
        dx.AbstractArray("3 3 C O", dtype="float32", name="conv1_weight"),
        dx.AbstractArray("O", dtype="float32", name="conv1_bias"),
        dx.AbstractArray("T O", dtype="float32", name="time_weight"),
        dx.AbstractArray("O", dtype="float32", name="time_bias"),
        dx.AbstractArray("O", dtype="float32", name="norm2_weight"),
        dx.AbstractArray("O", dtype="float32", name="norm2_bias"),
        dx.AbstractArray("3 3 O O", dtype="float32", name="conv2_weight"),
        dx.AbstractArray("O", dtype="float32", name="conv2_bias"),
        name="conditioned_resnet_block",
        kwargs={"num_groups": 2, "eps": 1e-5},
    )
    compiled = reference.compile(B=2, H=4, W=4, C=4, O=8, T=6)
    module = ResnetBlock2D(
        in_channels=4,
        out_channels=8,
        time_emb_dim=6,
        groups=2,
        eps=1e-5,
        seed=nn.Rngs(0),
    )
    time_embedding = jnp.ones((2, 6), dtype=jnp.float32)

    assert compiled.verify(
        module,
        time_embedding,
    ), compiled.report(module, time_embedding).render()
