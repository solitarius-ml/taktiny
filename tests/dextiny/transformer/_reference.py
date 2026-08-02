import jax
import jax.numpy as jnp


def linear(x, weight, bias=None, input_dims=1):
    x_axes = tuple(range(x.ndim - input_dims, x.ndim))
    weight_axes = tuple(range(input_dims))
    output = jax.lax.dot_general(
        x,
        weight,
        ((x_axes, weight_axes), ((), ())),
    )
    return output if bias is None else output + bias


def rms_norm(x, weight, *, eps):
    dtype = x.dtype
    variance = jnp.mean(jnp.square(x), axis=-1, keepdims=True)
    normalized = x * jax.lax.rsqrt(variance + eps)
    return (normalized * weight).astype(dtype)


def gemma_norm(x, weight, *, eps):
    dtype = x.dtype
    variance = jnp.mean(jnp.square(x), axis=-1, keepdims=True)
    normalized = x * jax.lax.rsqrt(variance + eps)
    return (normalized * (1.0 + weight)).astype(dtype)


def gemma3_norm(x, weight, *, eps):
    dtype = x.dtype
    x = x.astype(jnp.float32)
    variance = jnp.mean(jnp.square(x), axis=-1, keepdims=True)
    x = x * jax.lax.rsqrt(variance + eps)
    x = x * (1.0 + weight.astype(jnp.float32))
    return x.astype(dtype)


def begin_norm(hidden_states, weight, *, eps, norm):
    return hidden_states, norm(hidden_states, weight, eps=eps)


def residual_then_norm(state, weight, *, eps, norm):
    residual, branch = state
    hidden_states = residual + branch
    return hidden_states, norm(hidden_states, weight, eps=eps)


def norm_then_residual(state, weight, *, eps, norm):
    residual, branch = state
    return residual + norm(branch, weight, eps=eps)


def rotate_half(x):
    first, second = jnp.split(x, 2, axis=-1)
    return jnp.concatenate((-second, first), axis=-1)


def rotary(query, key, *, head_dim, rope_theta, **_):
    sequence_length = query.shape[1]
    inv_freq = 1.0 / (
        rope_theta
        ** (
            jnp.arange(0, head_dim, 2, dtype=jnp.float32)
            / head_dim
        )
    )
    positions = jnp.arange(sequence_length, dtype=jnp.float32)
    frequencies = jnp.einsum("s,d->sd", positions, inv_freq)
    embedding = jnp.concatenate((frequencies, frequencies), axis=-1)
    cosine = jnp.cos(embedding)[None, :, None, :].astype(query.dtype)
    sine = jnp.sin(embedding)[None, :, None, :].astype(query.dtype)
    return (
        query * cosine + rotate_half(query) * sine,
        key * cosine + rotate_half(key) * sine,
    )


def qwen_rotary(
    query,
    key,
    *,
    head_dim,
    rope_theta,
    max_position_embeddings,
    use_dynamic_ntk,
    **_,
):
    sequence_length = query.shape[1]
    position_start = jnp.asarray(0, dtype=jnp.int32)
    total_length = position_start + sequence_length

    ntk_alpha = jnp.asarray(1.0, dtype=jnp.float32)
    if use_dynamic_ntk:
        context_ratio = jnp.maximum(
            total_length / max_position_embeddings,
            1.0,
        )
        context_value = jnp.log2(context_ratio) + 1.0
        ntk_alpha = jnp.maximum(
            jnp.power(2.0, jnp.ceil(context_value)) - 1.0,
            1.0,
        )

    base = rope_theta * jnp.power(
        ntk_alpha,
        head_dim / (head_dim - 2),
    )
    exponent = (
        jnp.arange(0, head_dim, 2, dtype=jnp.float32) / head_dim
    )
    offsets = jnp.arange(sequence_length, dtype=jnp.int32)
    inv_freq = 1.0 / jnp.power(base, exponent)
    positions = (position_start + offsets).astype(jnp.float32)
    frequencies = jnp.einsum("s,d->sd", positions, inv_freq)
    embedding = jnp.concatenate((frequencies, frequencies), axis=-1)
    cosine = jnp.cos(embedding)[None, :, None, :]
    sine = jnp.sin(embedding)[None, :, None, :]

    def apply(x):
        dtype = x.dtype
        x = x.astype(jnp.float32)
        return (x * cosine + rotate_half(x) * sine).astype(dtype)

    return apply(query), apply(key)


def _sliding_mask(query, key, window_size):
    query_length = query.shape[1]
    key_length = key.shape[1]
    query_start = jnp.asarray(0, dtype=jnp.int32)
    query_positions = query_start + jnp.arange(
        query_length,
        dtype=jnp.int32,
    )
    key_positions = query_start + jnp.arange(
        key_length,
        dtype=jnp.int32,
    )
    causal_mask = key_positions[None, :] <= query_positions[:, None]
    window_mask = key_positions[None, :] >= (
        query_positions[:, None] - window_size + 1
    )
    return causal_mask & window_mask


def _softcap_bias(
    query,
    key,
    *,
    num_heads,
    num_kv_heads,
    head_dim,
    scaling,
    softcap,
):
    scale = scaling if scaling is not None else head_dim ** -0.5
    batch_size, query_length = query.shape[:2]
    key_length = key.shape[1]
    grouped_query = query.reshape(
        batch_size,
        query_length,
        num_kv_heads,
        num_heads // num_kv_heads,
        head_dim,
    )
    scores = jnp.einsum(
        "btkgh,bskh->bkgts",
        grouped_query,
        key,
    ) * scale
    capped_scores = softcap * jnp.tanh(scores / softcap)
    return (capped_scores - scores).reshape(
        batch_size,
        num_heads,
        query_length,
        key_length,
    )


def _attention(
    state,
    q_weight,
    k_weight,
    v_weight,
    o_weight,
    *,
    q_bias=None,
    k_bias=None,
    v_bias=None,
    o_bias=None,
    q_norm_weight=None,
    k_norm_weight=None,
    qk_norm=None,
    norm_eps=1e-6,
    rotary_fn=rotary,
    head_dim,
    num_heads,
    num_kv_heads,
    rope_theta,
    max_position_embeddings,
    use_dynamic_ntk=False,
    use_logn_attn=False,
    sequence_length=None,
    window_size=None,
    scaling=None,
    softcap=None,
):
    residual, hidden_states = state
    query = linear(hidden_states, q_weight, q_bias)
    key = linear(hidden_states, k_weight, k_bias)
    value = linear(hidden_states, v_weight, v_bias)

    if qk_norm is not None:
        query = qk_norm(query, q_norm_weight, eps=norm_eps)
        key = qk_norm(key, k_norm_weight, eps=norm_eps)

    query, key = rotary_fn(
        query,
        key,
        head_dim=head_dim,
        rope_theta=rope_theta,
        max_position_embeddings=max_position_embeddings,
        use_dynamic_ntk=use_dynamic_ntk,
    )

    if use_logn_attn:
        position_start = jnp.asarray(0, dtype=jnp.int32)
        offsets = jnp.arange(query.shape[1], dtype=jnp.int32) + 1
        positions = position_start + offsets
        query_scale = jnp.where(
            positions > sequence_length,
            jnp.log(positions.astype(jnp.float32))
            / jnp.log(jnp.asarray(sequence_length, dtype=jnp.float32)),
            1.0,
        )[None, :, None, None]
        query = query * query_scale.astype(query.dtype)

    attention_mask = None
    is_causal = True
    if window_size is not None:
        attention_mask = _sliding_mask(query, key, window_size)
        is_causal = False

    attention_bias = None
    if softcap is not None:
        attention_bias = _softcap_bias(
            query,
            key,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            scaling=scaling,
            softcap=softcap,
        )

    attended = jax.nn.dot_product_attention(
        query=query,
        key=key,
        value=value,
        bias=attention_bias,
        mask=attention_mask,
        scale=scaling,
        is_causal=is_causal,
    )
    return residual, linear(attended, o_weight, o_bias, input_dims=2)


def attention_no_bias(
    state,
    q_weight,
    k_weight,
    v_weight,
    o_weight,
    **kwargs,
):
    return _attention(
        state,
        q_weight,
        k_weight,
        v_weight,
        o_weight,
        **kwargs,
    )


def attention_qkv_bias(
    state,
    q_weight,
    q_bias,
    k_weight,
    k_bias,
    v_weight,
    v_bias,
    o_weight,
    **kwargs,
):
    return _attention(
        state,
        q_weight,
        k_weight,
        v_weight,
        o_weight,
        q_bias=q_bias,
        k_bias=k_bias,
        v_bias=v_bias,
        **kwargs,
    )


def attention_qkv_and_output_bias(
    state,
    q_weight,
    q_bias,
    k_weight,
    k_bias,
    v_weight,
    v_bias,
    o_weight,
    o_bias,
    **kwargs,
):
    return _attention(
        state,
        q_weight,
        k_weight,
        v_weight,
        o_weight,
        q_bias=q_bias,
        k_bias=k_bias,
        v_bias=v_bias,
        o_bias=o_bias,
        **kwargs,
    )


def attention_with_qk_norm(
    state,
    q_weight,
    k_weight,
    v_weight,
    o_weight,
    q_norm_weight,
    k_norm_weight,
    *,
    qk_norm,
    **kwargs,
):
    return _attention(
        state,
        q_weight,
        k_weight,
        v_weight,
        o_weight,
        q_norm_weight=q_norm_weight,
        k_norm_weight=k_norm_weight,
        qk_norm=qk_norm,
        **kwargs,
    )


def gate_mlp_residual(state, gate_weight, up_weight, down_weight):
    residual, hidden_states = state
    gate = jax.nn.silu(linear(hidden_states, gate_weight))
    up = linear(hidden_states, up_weight)
    return residual + linear(gate * up, down_weight)


def gate_mlp_branch(state, gate_weight, up_weight, down_weight):
    residual, hidden_states = state
    gate = jax.nn.silu(linear(hidden_states, gate_weight))
    up = linear(hidden_states, up_weight)
    return residual, linear(gate * up, down_weight)


def qwen_mlp_residual(state, w1_weight, w2_weight, c_proj_weight):
    residual, hidden_states = state
    up = linear(hidden_states, w1_weight)
    gate = jax.nn.silu(linear(hidden_states, w2_weight))
    return residual + linear(up * gate, c_proj_weight)


def render_and_assert(compiled, target):
    report = compiled.report(target)
    print(f"\n{report.render()}")
    assert report.valid, report.render()
