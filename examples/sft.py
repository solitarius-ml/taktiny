"""Shared supervised fine-tuning helpers for the example notebooks."""

from __future__ import annotations

from collections.abc import Sequence

import jax.numpy as jnp
import numpy as np
import optax


def _tokenize_with_assistant_labels(
    tokenizer,
    prompt_messages,
    full_messages,
    *,
    max_length,
):
    prompt_text = tokenizer.apply_chat_template(
        prompt_messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    full_text = tokenizer.apply_chat_template(
        full_messages,
        tokenize=False,
        add_generation_prompt=False,
    )

    if full_text.startswith(prompt_text):
        try:
            encoded = tokenizer(
                full_text,
                add_special_tokens=False,
                return_offsets_mapping=True,
                truncation=True,
                max_length=max_length,
            )
            full_ids = list(encoded['input_ids'])
            labels = [
                token_id if end > len(prompt_text) else -100
                for token_id, (_, end) in zip(
                    full_ids,
                    encoded['offset_mapping'],
                    strict=True,
                )
            ]
            return full_ids, labels
        except (KeyError, NotImplementedError, TypeError, ValueError):
            pass

    prompt_ids = tokenizer.apply_chat_template(
        prompt_messages,
        tokenize=True,
        add_generation_prompt=True,
    )
    full_ids = tokenizer.apply_chat_template(
        full_messages,
        tokenize=True,
        add_generation_prompt=False,
    )[:max_length]

    prompt_length = 0
    for prompt_id, full_id in zip(prompt_ids, full_ids):
        if prompt_id != full_id:
            break
        prompt_length += 1
    labels = [-100] * prompt_length + full_ids[prompt_length:]
    return full_ids, labels


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

    full_ids, labels = _tokenize_with_assistant_labels(
        tokenizer,
        prompt_messages,
        full_messages,
        max_length=max_length,
    )
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
