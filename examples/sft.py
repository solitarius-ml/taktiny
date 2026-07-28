"""Shared supervised fine-tuning helpers for the example notebooks."""

from __future__ import annotations

from collections.abc import Sequence

import jax.numpy as jnp
import numpy as np
import optax


def encode_open_thoughts_row(tokenizer, row, *, max_length):
    """Tokenize one OpenThoughts conversation with assistant-only labels."""
    turns = row['conversations']
    user_turn = next(
        turn for turn in turns if turn['from'] in ('user', 'human')
    )
    assistant_turn = next(
        turn for turn in turns if turn['from'] in ('assistant', 'gpt')
    )

    prompt_messages = []
    if row.get('system'):
        prompt_messages.append(
            {'role': 'system', 'content': row['system']}
        )
    prompt_messages.append(
        {'role': 'user', 'content': user_turn['value']}
    )
    full_messages = prompt_messages + [
        {'role': 'assistant', 'content': assistant_turn['value']}
    ]

    prompt_ids = tokenizer.apply_chat_template(
        prompt_messages,
        tokenize=True,
        add_generation_prompt=True,
    )
    full_ids = tokenizer.apply_chat_template(
        full_messages,
        tokenize=True,
        add_generation_prompt=False,
    )
    if full_ids[:len(prompt_ids)] != prompt_ids:
        raise ValueError(
            'Chat-template prompt is not a prefix of the full sample'
        )
    if len(prompt_ids) >= max_length:
        return None

    full_ids = full_ids[:max_length]
    labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids):]
    if not any(label != -100 for label in labels):
        return None
    attention_mask = [1] * len(full_ids)

    padding = max_length - len(full_ids)
    full_ids += [tokenizer.pad_token_id] * padding
    labels += [-100] * padding
    attention_mask += [0] * padding
    return full_ids, attention_mask, labels


def load_open_thoughts_samples(
    tokenizer,
    *,
    max_samples,
    max_length,
    seed=42,
    candidate_multiplier=3,
):
    """Stream a deterministic subset of verified OpenThoughts math rows."""
    from datasets import load_dataset

    dataset = load_dataset(
        'open-r1/OpenThoughts-114k-math',
        split='train',
        streaming=True,
    )
    dataset = dataset.filter(lambda row: bool(row['correct']))
    dataset = dataset.shuffle(seed=seed, buffer_size=10_000)

    samples = []
    candidate_count = max_samples * candidate_multiplier
    for row in dataset.take(candidate_count):
        sample = encode_open_thoughts_row(
            tokenizer,
            row,
            max_length=max_length,
        )
        if sample is not None:
            samples.append(sample)
        if len(samples) == max_samples:
            break

    if len(samples) < max_samples:
        raise RuntimeError(
            f'Only {len(samples)} usable samples remained after truncation'
        )
    return samples


def collate_sft_samples(samples):
    """Stack encoded samples into one NumPy batch."""
    return (
        np.asarray([sample[0] for sample in samples], dtype=np.int32),
        np.asarray([sample[1] for sample in samples], dtype=np.bool_),
        np.asarray([sample[2] for sample in samples], dtype=np.int32),
    )


def make_sft_batches(samples: Sequence, *, batch_size):
    """Create fixed-size host batches, dropping an incomplete final batch."""
    usable_count = len(samples) - len(samples) % batch_size
    if usable_count == 0:
        raise ValueError('Not enough samples to create one complete batch')
    samples = samples[:usable_count]
    return [
        collate_sft_samples(samples[start:start + batch_size])
        for start in range(0, usable_count, batch_size)
    ]


def assistant_only_sft_loss(model, batch):
    """Calculate next-token cross entropy over assistant labels only."""
    input_ids, attention_mask, labels = batch
    logits, _ = model(
        input_ids,
        attention_mask=attention_mask,
    )

    target_ids = labels[:, 1:]
    loss_mask = target_ids != -100
    safe_target_ids = jnp.where(loss_mask, target_ids, 0)
    token_losses = optax.softmax_cross_entropy_with_integer_labels(
        logits[:, :-1, :],
        safe_target_ids,
    )
    loss_mask = loss_mask.astype(token_losses.dtype)
    return jnp.sum(token_losses * loss_mask) / jnp.maximum(
        jnp.sum(loss_mask),
        1,
    )


__all__ = [
    'assistant_only_sft_loss',
    'collate_sft_samples',
    'encode_open_thoughts_row',
    'load_open_thoughts_samples',
    'make_sft_batches',
]
