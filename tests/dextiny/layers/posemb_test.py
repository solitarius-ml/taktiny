from dataclasses import dataclass
import math

import dextiny as dx
import jax.numpy as jnp
import pytest

from taktiny.layers import RotaryEmbedding, rotate_half


LLAMA3_SCALING = {
    "rope_type": "llama3",
    "factor": 8.0,
    "low_freq_factor": 1.0,
    "high_freq_factor": 4.0,
    "original_max_position_embeddings": 8192,
}


@dataclass(frozen=True)
class RotaryCase:
    name: str
    position_shape: str | None = None
    rope_scaling: dict | None = None


CASES = (
    RotaryCase("implicit"),
    RotaryCase("scalar", position_shape="scalar"),
    RotaryCase("batched", position_shape="B"),
    RotaryCase("per_token", position_shape="B S"),
    RotaryCase("llama3_scaled", position_shape="B S", rope_scaling=LLAMA3_SCALING),
)


def rotate_half_reference(x):
    first, second = jnp.split(x, 2, axis=-1)
    return jnp.concatenate((-second, first), axis=-1)


def rotary_reference(
    query,
    key,
    position_idx=None,
    *,
    dim,
    base,
    rope_scaling,
):
    sequence_length = query.shape[1]
    inv_freq = 1.0 / (
        base
        ** (
            jnp.arange(0, dim, 2, dtype=jnp.float32)
            / dim
        )
    )

    if rope_scaling is not None:
        factor = rope_scaling["factor"]
        low_factor = rope_scaling["low_freq_factor"]
        high_factor = rope_scaling["high_freq_factor"]
        old_context = rope_scaling["original_max_position_embeddings"]
        low_wavelength = old_context / low_factor
        high_wavelength = old_context / high_factor
        wavelength = 2 * math.pi / inv_freq
        scaled = jnp.where(
            wavelength > low_wavelength,
            inv_freq / factor,
            inv_freq,
        )
        smooth_factor = (
            old_context / wavelength - low_factor
        ) / (high_factor - low_factor)
        smoothed = (
            (1 - smooth_factor) * scaled / factor
            + smooth_factor * scaled
        )
        medium = ~(
            wavelength < high_wavelength
        ) & ~(wavelength > low_wavelength)
        inv_freq = jnp.where(medium, smoothed, scaled)

    positions = jnp.arange(sequence_length, dtype=jnp.float32)
    if position_idx is not None:
        position_idx = jnp.asarray(position_idx, dtype=jnp.float32)
        if position_idx.ndim == 0:
            positions = positions + position_idx
        elif position_idx.ndim == 1:
            positions = position_idx[:, None] + positions[None, :]
        else:
            positions = position_idx

    if positions.ndim == 1:
        frequencies = jnp.einsum("s,d->sd", positions, inv_freq)
    else:
        frequencies = jnp.einsum("bs,d->bsd", positions, inv_freq)
    embedding = jnp.concatenate((frequencies, frequencies), axis=-1)

    if embedding.ndim == 2:
        cosine = jnp.cos(embedding)[None, :, None, :].astype(query.dtype)
        sine = jnp.sin(embedding)[None, :, None, :].astype(query.dtype)
    else:
        cosine = jnp.cos(embedding)[:, :, None, :].astype(query.dtype)
        sine = jnp.sin(embedding)[:, :, None, :].astype(query.dtype)

    return (
        query * cosine + rotate_half_reference(query) * sine,
        key * cosine + rotate_half_reference(key) * sine,
    )


def position_values(case, batch_size, sequence_length):
    if case.position_shape is None:
        return None
    if case.position_shape == "scalar":
        return jnp.asarray(3, dtype=jnp.int32)
    if case.position_shape == "B":
        return jnp.arange(batch_size, dtype=jnp.int32) * sequence_length
    return jnp.broadcast_to(
        jnp.arange(sequence_length, dtype=jnp.int32),
        (batch_size, sequence_length),
    )


def test_rotate_half_matches_reference_trace():
    compiled = (
        dx.AbstractArray("B S H K", dtype="float32")
        >> dx.AbstractModule(rotate_half_reference, name="rotate_half")
    ).compile(B=2, S=8, H=4, K=4)

    assert compiled.verify(rotate_half), compiled.report(rotate_half).render()


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.name)
def test_rotary_embedding_matches_reference_trace(case):
    batch_size = 2
    sequence_length = 8
    num_heads = 4
    head_dim = 4
    query = dx.AbstractArray("B S H K", dtype="float32")
    key = dx.AbstractArray("B S G K", dtype="float32", name="key")
    position_idx = position_values(case, batch_size, sequence_length)
    operands = [key]
    if position_idx is not None:
        operands.append(position_idx)
    reference = query >> dx.AbstractModule(
        rotary_reference,
        *operands,
        name=f"rotary_{case.name}",
        kwargs={
            "dim": head_dim,
            "base": 10000.0,
            "rope_scaling": case.rope_scaling,
        },
    )
    compiled = reference.compile(
        B=batch_size,
        S=sequence_length,
        H=num_heads,
        G=2,
        K=head_dim,
    )
    module = RotaryEmbedding(
        dim=head_dim,
        base=10000.0,
        rope_scaling=case.rope_scaling,
    )
    actual_key = jnp.ones(
        (batch_size, sequence_length, 2, head_dim),
        dtype=jnp.float32,
    )
    actual_args = [actual_key]
    if position_idx is not None:
        actual_args.append(position_idx)

    assert compiled.verify(
        module,
        *actual_args,
    ), compiled.report(module, *actual_args).render()
