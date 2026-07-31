import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import taktiny
from taktiny import nn
from taktiny.trainer import (
    SFTDatasetConfig,
    SFTTrainer,
    SFTTrainerConfig,
)
from taktiny.trainer.sft import (
    _SFTDataLoader,
    _collate_sft_records,
    _encode_sft_record,
    _sft_loss,
)


class FakeTokenizer:
    pad_token_id = 0
    bos_token_id = 1
    eos_token_id = 2
    padding_side = 'right'
    truncation_side = 'right'

    @staticmethod
    def _content_ids(text):
        return [3 + ord(character) % 23 for character in text]

    def __call__(
        self,
        text,
        *,
        add_special_tokens=True,
        truncation=False,
        max_length=None,
        **kwargs,
    ):
        ids = self._content_ids(text)
        if add_special_tokens:
            ids = [self.bos_token_id] + ids
        if truncation and max_length is not None:
            ids = ids[:max_length]
        return {'input_ids': ids}

    def apply_chat_template(
        self,
        messages,
        *,
        tokenize,
        add_generation_prompt=False,
        return_dict=False,
        return_assistant_tokens_mask=False,
        truncation=False,
        max_length=None,
        **kwargs,
    ):
        assert tokenize
        ids = [self.bos_token_id]
        assistant_mask = [0]
        role_ids = {'system': 26, 'user': 27, 'assistant': 28}
        for message in messages:
            message_ids = [
                role_ids[message['role']],
                *self._content_ids(message['content']),
                self.eos_token_id,
            ]
            ids.extend(message_ids)
            assistant_mask.extend([
                int(message['role'] == 'assistant')
            ] * len(message_ids))
        if add_generation_prompt:
            ids.append(role_ids['assistant'])
            assistant_mask.append(0)
        if truncation and max_length is not None:
            ids = ids[:max_length]
            assistant_mask = assistant_mask[:max_length]
        if return_dict:
            result = {'input_ids': ids}
            if return_assistant_tokens_mask:
                result['assistant_masks'] = assistant_mask
            return result
        return ids


class NoAssistantMaskTokenizer(FakeTokenizer):
    chat_template = 'template without generation blocks'


class TinyCausalLM(nn.Module):
    def __init__(self, vocab_size=8):
        self.vocab_size = vocab_size
        self.scale = nn.Parameter(jnp.asarray(0.0))

    def __call__(self, input_ids, attention_mask=None, ctx=None):
        targets = (input_ids + 1) % self.vocab_size
        logits = (
            jax.nn.one_hot(targets, self.vocab_size)
            * self.scale.value
        )
        return logits, ctx


def make_dataset_config(records, **kwargs):
    return SFTDatasetConfig(
        dataloader=records,
        tokenizer=FakeTokenizer(),
        shuffle=False,
        drop_remainder=False,
        **kwargs,
    )


def test_sft_configs_are_exported():
    assert taktiny.SFTTrainer is SFTTrainer
    assert taktiny.SFTTrainerConfig is SFTTrainerConfig
    assert taktiny.SFTDatasetConfig is SFTDatasetConfig


@pytest.mark.parametrize(
    'factory',
    [
        lambda: SFTTrainerConfig(completion_only_loss='yes'),
        lambda: SFTTrainerConfig(assistant_only_loss='yes'),
        lambda: SFTTrainerConfig(ignore_index=True),
        lambda: SFTDatasetConfig(dataloader=[]),
        lambda: SFTDatasetConfig(
            dataloader=[],
            tokenizer=FakeTokenizer(),
            batch_size=0,
        ),
        lambda: SFTDatasetConfig(
            dataloader=[],
            tokenizer=FakeTokenizer(),
            max_length=1,
        ),
        lambda: SFTDatasetConfig(
            dataloader=[],
            tokenizer=FakeTokenizer(),
            padding='all',
        ),
        lambda: SFTDatasetConfig(
            dataloader=[],
            tokenizer=FakeTokenizer(),
            padding='max_length',
            max_length=None,
        ),
        lambda: SFTDatasetConfig(
            dataloader=[],
            tokenizer=FakeTokenizer(),
            max_length=10,
            padding='max_length',
            pad_to_multiple_of=4,
        ),
        lambda: SFTDatasetConfig(
            dataloader=[],
            tokenizer=FakeTokenizer(),
            packing=True,
            max_length=None,
        ),
    ],
)
def test_sft_config_validation(factory):
    with pytest.raises((TypeError, ValueError)):
        factory()


def test_text_record_uses_full_language_model_labels():
    dataset_config = make_dataset_config([], append_eos=True)
    trainer_config = SFTTrainerConfig()

    record = _encode_sft_record(
        {'text': 'abc'},
        dataset_config,
        trainer_config,
    )

    assert record['input_ids'] == record['labels']
    assert record['input_ids'][-1] == FakeTokenizer.eos_token_id


def test_prompt_completion_masks_prompt_by_default():
    dataset_config = make_dataset_config([], append_eos=False)
    trainer_config = SFTTrainerConfig()

    record = _encode_sft_record(
        {'prompt': 'ab', 'completion': 'cd'},
        dataset_config,
        trainer_config,
    )

    prompt_length = 3
    assert record['labels'][:prompt_length] == [-100] * prompt_length
    assert record['labels'][prompt_length:] == record['input_ids'][
        prompt_length:
    ]


def test_prompt_completion_can_train_full_sequence():
    dataset_config = make_dataset_config([], append_eos=False)
    trainer_config = SFTTrainerConfig(completion_only_loss=False)

    record = _encode_sft_record(
        {'prompt': 'ab', 'completion': 'cd'},
        dataset_config,
        trainer_config,
    )

    assert record['labels'] == record['input_ids']


def test_assistant_only_conversation_uses_template_mask():
    dataset_config = make_dataset_config([], append_eos=False)
    trainer_config = SFTTrainerConfig(assistant_only_loss=True)
    messages = [
        {'role': 'user', 'content': 'question'},
        {'role': 'assistant', 'content': 'answer'},
    ]

    record = _encode_sft_record(
        {'messages': messages},
        dataset_config,
        trainer_config,
    )

    assistant_marker = 28
    assistant_start = record['input_ids'].index(assistant_marker)
    assert record['labels'][:assistant_start] == [-100] * assistant_start
    assert record['labels'][assistant_start:] == record['input_ids'][
        assistant_start:
    ]


def test_assistant_fallback_masks_every_assistant_turn():
    tokenizer = NoAssistantMaskTokenizer()
    dataset_config = SFTDatasetConfig(
        dataloader=[],
        tokenizer=tokenizer,
        max_length=64,
        shuffle=False,
        append_eos=False,
    )
    trainer_config = SFTTrainerConfig(assistant_only_loss=True)
    messages = [
        {'role': 'user', 'content': 'one'},
        {'role': 'assistant', 'content': 'first'},
        {'role': 'user', 'content': 'two'},
        {'role': 'assistant', 'content': 'second'},
    ]

    record = _encode_sft_record(
        {'messages': messages},
        dataset_config,
        trainer_config,
    )

    assistant_markers = [
        index
        for index, token_id in enumerate(record['input_ids'])
        if token_id == 28
    ]
    assert len(assistant_markers) == 2
    for marker in assistant_markers:
        response_index = marker + 1
        assert (
            record['labels'][response_index]
            == record['input_ids'][response_index]
        )


def test_pretokenized_labels_take_precedence_over_loss_modes():
    dataset_config = make_dataset_config([], append_eos=False)
    trainer_config = SFTTrainerConfig(
        assistant_only_loss=True,
        completion_only_loss=True,
    )

    record = _encode_sft_record(
        {
            'input_ids': [1, 4, 5],
            'labels': [-100, 4, 5],
            'assistant_mask': [0, 0, 0],
            'completion_mask': [0, 0, 0],
        },
        dataset_config,
        trainer_config,
    )

    assert record['labels'] == [-100, 4, 5]


def test_dynamic_padding_and_left_padding_alignment():
    tokenizer = FakeTokenizer()
    tokenizer.padding_side = 'left'
    config = SFTDatasetConfig(
        dataloader=[],
        tokenizer=tokenizer,
        batch_size=2,
        max_length=16,
        padding='longest',
        pad_to_multiple_of=4,
        shuffle=False,
    )
    trainer_config = SFTTrainerConfig()
    records = [
        {'input_ids': [1, 2], 'labels': [1, 2]},
        {'input_ids': [1, 3, 4, 2], 'labels': [1, 3, 4, 2]},
    ]

    batch = _collate_sft_records(records, config, trainer_config)

    assert batch['input_ids'].shape == (2, 4)
    np.testing.assert_array_equal(batch['input_ids'][0], [0, 0, 1, 2])
    np.testing.assert_array_equal(
        batch['labels'][0],
        [-100, -100, 1, 2],
    )
    assert batch['attention_mask'].shape == (2, 1, 1, 4)
    np.testing.assert_array_equal(
        batch['attention_mask'][0, 0, 0],
        [False, False, True, True],
    )


def test_packing_builds_block_diagonal_attention_and_boundary_labels():
    config = make_dataset_config(
        [
            {'input_ids': [1, 4, 2]},
            {'input_ids': [1, 5, 2]},
        ],
        batch_size=1,
        max_length=8,
        packing=True,
        padding='max_length',
        append_eos=False,
    )
    trainer_config = SFTTrainerConfig()
    loader = _SFTDataLoader(
        config.dataloader,
        config,
        trainer_config,
        streaming=False,
        shuffle=False,
    )

    batch = next(iter(loader))
    mask = batch['attention_mask'][0, 0]

    assert batch['input_ids'].shape == (1, 8)
    assert mask[1, 2]
    assert not mask[1, 4]
    assert not mask[4, 1]
    assert batch['labels'][0, 0] == -100
    assert batch['labels'][0, 3] == -100


def test_sft_grain_iterator_state_restores_next_batch():
    config = make_dataset_config(
        [{'text': str(index)} for index in range(6)],
        batch_size=2,
        max_length=8,
        append_eos=True,
    )
    loader = _SFTDataLoader(
        config.dataloader,
        config,
        SFTTrainerConfig(),
        streaming=False,
        shuffle=False,
    )

    first_iterator = iter(loader)
    next(first_iterator)
    state = first_iterator.get_state()
    expected = next(first_iterator)

    resumed_iterator = iter(loader)
    resumed_iterator.set_state(state)
    actual = next(resumed_iterator)
    np.testing.assert_array_equal(
        actual['input_ids'],
        expected['input_ids'],
    )
    np.testing.assert_array_equal(
        actual['labels'],
        expected['labels'],
    )


def test_packed_iterator_state_restores_pending_fragment():
    config = make_dataset_config(
        [
            {'input_ids': [1, 4, 2]},
            {'input_ids': [1, 5, 2]},
            {'input_ids': [1, 6, 2]},
        ],
        batch_size=1,
        max_length=4,
        packing=True,
        padding='max_length',
        append_eos=False,
    )
    loader = _SFTDataLoader(
        config.dataloader,
        config,
        SFTTrainerConfig(),
        streaming=False,
        shuffle=False,
    )

    first_iterator = iter(loader)
    next(first_iterator)
    state = first_iterator.get_state()
    expected = next(first_iterator)

    resumed_iterator = iter(loader)
    resumed_iterator.set_state(state)
    actual = next(resumed_iterator)
    np.testing.assert_array_equal(
        actual['input_ids'],
        expected['input_ids'],
    )
    np.testing.assert_array_equal(
        actual['attention_mask'],
        expected['attention_mask'],
    )


def test_streaming_sft_loader_does_not_require_length():
    class StreamingSource:
        def __iter__(self):
            yield {'text': 'one'}
            yield {'text': 'two'}

    trainer = SFTTrainer(
        TinyCausalLM(),
        training_config=SFTTrainerConfig(
            max_steps=1,
            learning_rate=0.1,
        ),
        dataset_config=SFTDatasetConfig(
            dataloader=StreamingSource(),
            tokenizer=FakeTokenizer(),
            streaming=True,
            shuffle=False,
            batch_size=1,
            max_length=8,
            drop_remainder=False,
            prefetch_size=0,
        ),
    )

    trainer.train()
    assert trainer.global_step == 1


def test_default_sft_loss_is_shifted_and_causal():
    model = TinyCausalLM(vocab_size=8)
    batch = {
        'input_ids': jnp.asarray([[1, 2, 3]], dtype=jnp.int32),
        'attention_mask': jnp.ones((1, 1, 1, 3), dtype=jnp.bool_),
        'labels': jnp.asarray([[1, 2, 3]], dtype=jnp.int32),
    }

    loss = _sft_loss(model, batch, ignore_index=-100)

    assert float(loss) == pytest.approx(math.log(8), rel=1e-6)


def test_sft_trainer_inherits_training_loop_and_updates_model():
    model = TinyCausalLM(vocab_size=8)
    batch = {
        'input_ids': np.asarray([[1, 2, 3]], dtype=np.int32),
        'attention_mask': np.ones((1, 1, 1, 3), dtype=np.bool_),
        'labels': np.asarray([[1, 2, 3]], dtype=np.int32),
    }
    trainer = SFTTrainer(
        model,
        training_config=SFTTrainerConfig(
            max_steps=1,
            learning_rate=0.1,
            log_interval=1,
        ),
        dataset_config=SFTDatasetConfig(
            dataloader=[batch],
            skip_prepare_dataset=True,
            shuffle=False,
            prefetch_size=0,
        ),
    )

    trainer.train()

    assert trainer.global_step == 1
    assert float(model.scale.value) > 0
