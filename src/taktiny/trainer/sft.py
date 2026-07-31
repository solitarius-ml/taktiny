# Copyright 2026 Shinapri
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Supervised fine-tuning utilities for causal language models."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
import math
import pickle

import jax
import jax.numpy as jnp
import numpy as np
import optax

from taktiny.cosettes._common import TransformerContext
from taktiny.trainer.config import (
    SFTDatasetConfig,
    SFTTrainingConfig,
)
from taktiny.trainer.trainer import (
    Trainer,
    _GrainEpochLoader,
    _load_dataset_splits,
)


def _as_token_list(value, name):
    array = np.asarray(value)
    if array.ndim == 2 and array.shape[0] == 1:
        array = array[0]
    if array.ndim != 1:
        raise ValueError(f'{name} must be a one-dimensional token sequence')
    return [int(item) for item in array.tolist()]


def _truncate(values, max_length, *, side='right'):
    if max_length is None or len(values) <= max_length:
        return values
    if side == 'left':
        return values[-max_length:]
    return values[:max_length]


def _tokenizer_ids(tokenizer, text, *, max_length, add_special_tokens):
    kwargs = {
        'add_special_tokens': add_special_tokens,
        'truncation': max_length is not None,
    }
    if max_length is not None:
        kwargs['max_length'] = max_length
    encoded = tokenizer(text, **kwargs)
    return _as_token_list(encoded['input_ids'], 'input_ids')


def _common_prefix_length(left, right):
    length = 0
    for left_id, right_id in zip(left, right):
        if left_id != right_id:
            break
        length += 1
    return length


def _chat_template_ids(
    tokenizer,
    messages,
    *,
    max_length,
    add_generation_prompt=False,
):
    ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=add_generation_prompt,
    )
    if isinstance(ids, Mapping):
        ids = ids['input_ids']
    ids = _as_token_list(ids, 'input_ids')
    side = getattr(tokenizer, 'truncation_side', 'right')
    return _truncate(ids, max_length, side=side)


def _assistant_chat_labels(tokenizer, messages, ids, *, max_length):
    chat_template = getattr(tokenizer, 'chat_template', None)
    supports_assistant_mask = not (
        isinstance(chat_template, str)
        and '{% generation' not in chat_template
    )
    try:
        if not supports_assistant_mask:
            raise NotImplementedError
        kwargs = {
            'tokenize': True,
            'add_generation_prompt': False,
            'return_dict': True,
            'return_assistant_tokens_mask': True,
            'truncation': max_length is not None,
        }
        if max_length is not None:
            kwargs['max_length'] = max_length
        encoded = tokenizer.apply_chat_template(messages, **kwargs)
        assistant_mask = encoded.get('assistant_masks')
        if assistant_mask is None:
            assistant_mask = encoded.get('assistant_mask')
        masked_ids = _as_token_list(encoded['input_ids'], 'input_ids')
        if assistant_mask is not None:
            assistant_mask = _as_token_list(
                assistant_mask,
                'assistant_mask',
            )
            if any(assistant_mask):
                return masked_ids, assistant_mask
    except (KeyError, NotImplementedError, TypeError, ValueError):
        pass

    assistant_indices = [
        index
        for index, message in enumerate(messages)
        if message.get('role') == 'assistant'
    ]
    if not assistant_indices:
        return ids, [0] * len(ids)

    assistant_mask = [0] * len(ids)
    for assistant_index in assistant_indices:
        prompt_ids = _chat_template_ids(
            tokenizer,
            messages[:assistant_index],
            max_length=max_length,
            add_generation_prompt=True,
        )
        response_ids = _chat_template_ids(
            tokenizer,
            messages[:assistant_index + 1],
            max_length=max_length,
        )
        start = _common_prefix_length(prompt_ids, ids)
        stop = _common_prefix_length(response_ids, ids)
        for index in range(start, stop):
            assistant_mask[index] = 1
    return ids, assistant_mask


def _append_eos(ids, labels, tokenizer, config, ignore_index):
    eos_id = getattr(tokenizer, 'eos_token_id', None)
    if not config.append_eos or eos_id is None or not ids:
        return ids, labels
    if ids[-1] == eos_id:
        return ids, labels

    train_eos = labels[-1] != ignore_index
    if config.max_length is not None and len(ids) >= config.max_length:
        ids[-1] = int(eos_id)
        labels[-1] = int(eos_id) if train_eos else ignore_index
    else:
        ids.append(int(eos_id))
        labels.append(int(eos_id) if train_eos else ignore_index)
    return ids, labels


def _pretokenized_record(row, config, trainer_config):
    ids = _as_token_list(row['input_ids'], 'input_ids')
    labels_value = row.get('labels')
    labels = (
        list(ids)
        if labels_value is None
        else _as_token_list(labels_value, 'labels')
    )
    if len(labels) != len(ids):
        raise ValueError('labels and input_ids must have the same length')

    mask_value = row.get('attention_mask')
    valid = None
    if mask_value is not None:
        mask = _as_token_list(mask_value, 'attention_mask')
        if len(mask) != len(ids):
            raise ValueError(
                'attention_mask and input_ids must have the same length'
            )
        valid = [bool(value) for value in mask]

    if labels_value is None:
        loss_mask = None
        if trainer_config.assistant_only_loss:
            loss_mask = row.get('assistant_masks')
            if loss_mask is None:
                loss_mask = row.get('assistant_mask')
        elif trainer_config.completion_only_loss:
            loss_mask = row.get('completion_mask')
        if loss_mask is not None:
            loss_mask = _as_token_list(loss_mask, 'loss mask')
            if len(loss_mask) != len(ids):
                raise ValueError(
                    'The loss mask and input_ids must have the same length'
                )
            labels = [
                token_id if keep else trainer_config.ignore_index
                for token_id, keep in zip(labels, loss_mask)
            ]

    if valid is not None:
        ids = [value for value, keep in zip(ids, valid) if keep]
        labels = [value for value, keep in zip(labels, valid) if keep]

    side = getattr(config.tokenizer, 'truncation_side', 'right')
    ids = _truncate(ids, config.max_length, side=side)
    labels = _truncate(labels, config.max_length, side=side)
    return ids, labels


def _prompt_completion_record(row, config, trainer_config):
    tokenizer = config.tokenizer
    prompt = row[config.prompt_field]
    completion = row[config.completion_field]

    if isinstance(prompt, Sequence) and not isinstance(prompt, str):
        prompt_messages = list(prompt)
        if isinstance(completion, Mapping):
            completion_messages = [completion]
        elif isinstance(completion, Sequence) and not isinstance(
            completion,
            str,
        ):
            completion_messages = list(completion)
        else:
            completion_messages = [
                {'role': 'assistant', 'content': str(completion)}
            ]
        ids = _chat_template_ids(
            tokenizer,
            prompt_messages + completion_messages,
            max_length=config.max_length,
        )
        prompt_ids = _chat_template_ids(
            tokenizer,
            prompt_messages,
            max_length=config.max_length,
            add_generation_prompt=True,
        )
    else:
        if not isinstance(prompt, str) or not isinstance(completion, str):
            raise TypeError(
                'prompt and completion must both be strings or '
                'conversational message sequences'
            )
        prompt_ids = _tokenizer_ids(
            tokenizer,
            prompt,
            max_length=config.max_length,
            add_special_tokens=True,
        )
        remaining = (
            None
            if config.max_length is None
            else max(config.max_length - len(prompt_ids), 0)
        )
        completion_ids = (
            []
            if remaining == 0
            else _tokenizer_ids(
                tokenizer,
                completion,
                max_length=remaining,
                add_special_tokens=False,
            )
        )
        ids = prompt_ids + completion_ids

    prompt_length = _common_prefix_length(prompt_ids, ids)
    completion_only = trainer_config.completion_only_loss
    if completion_only is None:
        completion_only = True
    labels = list(ids)
    if completion_only:
        labels[:prompt_length] = [
            trainer_config.ignore_index
        ] * prompt_length
    return ids, labels


def _conversational_record(row, config, trainer_config):
    if trainer_config.completion_only_loss:
        raise ValueError(
            'completion_only_loss=True requires prompt-completion records'
        )
    messages = row[config.messages_field]
    ids = _chat_template_ids(
        config.tokenizer,
        messages,
        max_length=config.max_length,
    )
    if not trainer_config.assistant_only_loss:
        return ids, list(ids)

    ids, assistant_mask = _assistant_chat_labels(
        config.tokenizer,
        messages,
        ids,
        max_length=config.max_length,
    )
    labels = [
        token_id if keep else trainer_config.ignore_index
        for token_id, keep in zip(ids, assistant_mask)
    ]
    return ids, labels


def _text_record(row, config, trainer_config):
    if trainer_config.completion_only_loss:
        raise ValueError(
            'completion_only_loss=True requires prompt-completion records'
        )
    if trainer_config.assistant_only_loss:
        raise ValueError(
            'assistant_only_loss=True requires conversational records'
        )
    text = row[config.dataset_text_field]
    if not isinstance(text, str):
        raise TypeError(
            f'{config.dataset_text_field!r} must contain strings'
        )
    ids = _tokenizer_ids(
        config.tokenizer,
        text,
        max_length=config.max_length,
        add_special_tokens=True,
    )
    return ids, list(ids)


def _encode_sft_record(row, config, trainer_config):
    if config.formatting_fn is not None:
        row = config.formatting_fn(row)
    if isinstance(row, str):
        row = {config.dataset_text_field: row}
    if not isinstance(row, Mapping):
        raise TypeError(
            'Each SFT record must be a mapping or formatting_fn must '
            'return a mapping or string'
        )

    if 'input_ids' in row:
        ids, labels = _pretokenized_record(
            row,
            config,
            trainer_config,
        )
    elif (
        config.prompt_field in row
        and config.completion_field in row
    ):
        ids, labels = _prompt_completion_record(
            row,
            config,
            trainer_config,
        )
    elif config.messages_field in row:
        ids, labels = _conversational_record(
            row,
            config,
            trainer_config,
        )
    elif config.dataset_text_field in row:
        ids, labels = _text_record(
            row,
            config,
            trainer_config,
        )
    else:
        raise ValueError(
            'SFT records must contain input_ids, text, messages, or '
            'prompt and completion fields'
        )

    ids, labels = _append_eos(
        ids,
        labels,
        config.tokenizer,
        config,
        trainer_config.ignore_index,
    )
    if len(ids) < 2:
        return None
    if not any(
        label != trainer_config.ignore_index
        for label in labels[1:]
    ):
        return None
    return {
        'input_ids': ids,
        'labels': labels,
    }


def _round_up(value, multiple):
    return math.ceil(value / multiple) * multiple


def _collate_sft_records(records, config, trainer_config):
    if config.collate_fn is not None:
        return config.collate_fn(records)
    if not records:
        raise ValueError('Cannot collate an empty SFT batch')

    lengths = [len(record['input_ids']) for record in records]
    if config.padding == 'max_length' or config.packing:
        target_length = config.max_length
    else:
        target_length = max(lengths)
        if config.pad_to_multiple_of is not None:
            target_length = _round_up(
                target_length,
                config.pad_to_multiple_of,
            )
        if config.max_length is not None:
            target_length = min(target_length, config.max_length)

    tokenizer = config.tokenizer
    pad_id = getattr(tokenizer, 'pad_token_id', None)
    if pad_id is None:
        pad_id = getattr(tokenizer, 'eos_token_id', None)
    if pad_id is None:
        raise ValueError(
            'The tokenizer must define pad_token_id or eos_token_id'
        )
    padding_side = getattr(tokenizer, 'padding_side', 'right')

    input_ids = []
    labels = []
    segment_ids = []
    has_segments = any('segment_ids' in record for record in records)
    for record in records:
        record_ids = record['input_ids'][:target_length]
        record_labels = record['labels'][:target_length]
        record_segments = record.get(
            'segment_ids',
            [0] * len(record_ids),
        )[:target_length]
        padding = target_length - len(record_ids)
        if padding_side == 'left':
            input_ids.append([pad_id] * padding + record_ids)
            labels.append(
                [trainer_config.ignore_index] * padding + record_labels
            )
            segment_ids.append([-1] * padding + record_segments)
        else:
            input_ids.append(record_ids + [pad_id] * padding)
            labels.append(
                record_labels
                + [trainer_config.ignore_index] * padding
            )
            segment_ids.append(record_segments + [-1] * padding)

    input_ids = np.asarray(input_ids, dtype=np.int32)
    labels = np.asarray(labels, dtype=np.int32)
    segment_ids = np.asarray(segment_ids, dtype=np.int32)
    valid = segment_ids != -1
    if has_segments:
        attention_mask = (
            segment_ids[:, :, None] == segment_ids[:, None, :]
        )
        attention_mask &= valid[:, :, None] & valid[:, None, :]
        attention_mask = attention_mask[:, None, :, :]
    else:
        attention_mask = valid[:, None, None, :]

    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask.astype(np.bool_),
        'labels': labels,
    }


class _SFTBatchIterator:
    def __init__(self, iterator, config, trainer_config):
        self.iterator = iterator
        self.config = config
        self.trainer_config = trainer_config
        self.pending_fragment = None
        self.exhausted = False

    def __iter__(self):
        return self

    def _next_record(self):
        while True:
            row = next(self.iterator)
            record = _encode_sft_record(
                row,
                self.config,
                self.trainer_config,
            )
            if record is not None:
                return record

    @staticmethod
    def _slice_record(record, start, stop):
        return {
            'input_ids': record['input_ids'][start:stop],
            'labels': record['labels'][start:stop],
        }

    def _next_pack(self):
        max_length = self.config.max_length
        packed_ids = []
        packed_labels = []
        packed_segments = []
        segment = 0

        while len(packed_ids) < max_length:
            if self.pending_fragment is not None:
                record = self.pending_fragment
                self.pending_fragment = None
            else:
                try:
                    record = self._next_record()
                except StopIteration:
                    self.exhausted = True
                    break

            take = min(
                max_length - len(packed_ids),
                len(record['input_ids']),
            )
            fragment = self._slice_record(record, 0, take)
            fragment_labels = list(fragment['labels'])
            fragment_labels[0] = self.trainer_config.ignore_index
            packed_ids.extend(fragment['input_ids'])
            packed_labels.extend(fragment_labels)
            packed_segments.extend([segment] * take)
            segment += 1

            if take < len(record['input_ids']):
                self.pending_fragment = self._slice_record(
                    record,
                    take,
                    len(record['input_ids']),
                )

        if not packed_ids:
            raise StopIteration
        return {
            'input_ids': packed_ids,
            'labels': packed_labels,
            'segment_ids': packed_segments,
        }

    def __next__(self):
        records = []
        while len(records) < self.config.batch_size:
            try:
                record = (
                    self._next_pack()
                    if self.config.packing
                    else self._next_record()
                )
            except StopIteration:
                break
            records.append(record)

        if not records:
            raise StopIteration
        if (
            self.config.drop_remainder
            and len(records) < self.config.batch_size
        ):
            raise StopIteration
        return _collate_sft_records(
            records,
            self.config,
            self.trainer_config,
        )


class _StatefulSFTBatchIterator(_SFTBatchIterator):
    def get_state(self):
        return pickle.dumps({
            'iterator': self.iterator.get_state(),
            'pending_fragment': self.pending_fragment,
            'exhausted': self.exhausted,
        })

    def set_state(self, state):
        state = pickle.loads(state)
        self.iterator.set_state(state['iterator'])
        self.pending_fragment = state['pending_fragment']
        self.exhausted = state['exhausted']


class _SFTDataLoader:
    def __init__(
        self,
        source,
        config,
        trainer_config,
        *,
        streaming,
        shuffle,
    ):
        self.source = source
        self.config = config
        self.trainer_config = trainer_config
        self.streaming = streaming
        self.shuffle = shuffle
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        source_size = len(self.source)
        process_count = jax.process_count()
        process_index = jax.process_index()
        local_size = max(
            0,
            (
                source_size
                + process_count
                - 1
                - process_index
            ) // process_count,
        )
        if self.config.drop_remainder:
            return local_size // self.config.batch_size
        return math.ceil(local_size / self.config.batch_size)

    def _source_iterator(self):
        if not self.streaming:
            loader = _GrainEpochLoader(
                self.source,
                shuffle=self.shuffle,
                seed=self.config.seed,
            )
            loader.set_epoch(self.epoch)
            return iter(loader)

        source = self.source
        shuffle = getattr(source, 'shuffle', None)
        if self.shuffle and callable(shuffle):
            try:
                source = shuffle(
                    seed=self.config.seed + self.epoch,
                    buffer_size=self.config.shuffle_buffer_size,
                )
            except TypeError:
                source = shuffle(seed=self.config.seed + self.epoch)
        return iter(source)

    def __iter__(self):
        iterator = self._source_iterator()
        iterator_type = (
            _StatefulSFTBatchIterator
            if (
                callable(getattr(iterator, 'get_state', None))
                and callable(getattr(iterator, 'set_state', None))
            )
            else _SFTBatchIterator
        )
        return iterator_type(
            iterator,
            self.config,
            self.trainer_config,
        )


def _sft_loss(model, batch, *, ignore_index):
    if isinstance(batch, Mapping):
        input_ids = batch['input_ids']
        attention_mask = batch.get('attention_mask')
        labels = batch['labels']
    else:
        if len(batch) != 3:
            raise ValueError(
                'SFT batches must contain input_ids, attention_mask, labels'
            )
        input_ids, attention_mask, labels = batch

    context = TransformerContext(
        key_cache=None,
        value_cache=None,
        position_idx=None,
        is_causal=True,
    )
    outputs = model(
        input_ids,
        attention_mask=attention_mask,
        ctx=context,
    )
    logits = outputs[0] if isinstance(outputs, tuple) else outputs

    target_ids = labels[:, 1:]
    loss_mask = target_ids != ignore_index
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


class SFTTrainer(Trainer):
    """Supervised causal-language-model trainer built on :class:`Trainer`.

    The default loss shifts labels by one token, enables causal attention, and
    averages cross entropy only over labels different from ``ignore_index``.
    A custom ``loss_fn`` can replace it without changing dataset preparation.

    Args:
        model: Callable causal language model trained by the base ``Trainer``.
        training_config: Optimizer, checkpoint, callback, and SFT loss options.
        dataset_config: Dataset loading, tokenization, batching, and packing
            options.
        loss_fn: Optional replacement for the default causal language-model
            loss.
        callbacks: Optional Trainer callback or iterable of callbacks.
        compute_metrics: Optional validation metric function.
    """

    def __init__(
        self,
        model,
        training_config: SFTTrainerConfig,
        dataset_config: SFTDatasetConfig,
        *,
        loss_fn=None,
        callbacks=None,
        compute_metrics=None,
    ):
        if not isinstance(training_config, SFTTrainerConfig):
            raise TypeError(
                'training_config must be an SFTTrainerConfig'
            )
        if not isinstance(dataset_config, SFTDatasetConfig):
            raise TypeError(
                'dataset_config must be an SFTDatasetConfig'
            )

        if dataset_config.dataloader is None:
            train_source, validation_source = _load_dataset_splits(
                dataset_config
            )
        else:
            train_source = dataset_config.dataloader
            validation_source = dataset_config.validation_dataloader

        if dataset_config.skip_prepare_dataset:
            train_dataloader = train_source
            validation_dataloader = validation_source
        else:
            train_dataloader = _SFTDataLoader(
                train_source,
                dataset_config,
                training_config,
                streaming=dataset_config.streaming,
                shuffle=dataset_config.shuffle,
            )
            validation_dataloader = (
                None
                if validation_source is None
                else _SFTDataLoader(
                    validation_source,
                    dataset_config,
                    training_config,
                    streaming=dataset_config.streaming,
                    shuffle=False,
                )
            )

        prepared_dataset_config = replace(
            dataset_config,
            dataloader=train_dataloader,
            validation_dataloader=validation_dataloader,
        )
        if loss_fn is None:
            def loss_fn(candidate, batch):
                return _sft_loss(
                    candidate,
                    batch,
                    ignore_index=training_config.ignore_index,
                )

        super().__init__(
            model,
            loss_fn,
            training_config,
            prepared_dataset_config,
            callbacks=callbacks,
            compute_metrics=compute_metrics,
        )


__all__ = ['SFTTrainer']
