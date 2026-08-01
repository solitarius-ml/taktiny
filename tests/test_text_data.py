import grain.python as grain
import numpy as np
import pytest
from datasets import Dataset

from taktiny.data_utils import (
    ApplyTemplate,
    CausalLMBatch,
    DatasetUtils,
    Map,
    PackSequences,
)


def test_apply_template_reuses_template_without_mutating_it():
    template = [
        {'role': 'user', 'content': '{question}'},
        {'role': 'assistant', 'content': '{answer}'},
    ]
    operation = ApplyTemplate(
        template,
        lambda messages: ' | '.join(
            message['content'] for message in messages
        ),
    )

    first = operation.map({'question': 'One?', 'answer': 'First'})
    second = operation.map({'question': 'Two?', 'answer': 'Second'})

    assert first['template'] == 'One? | First'
    assert second['template'] == 'Two? | Second'
    assert template == [
        {'role': 'user', 'content': '{question}'},
        {'role': 'assistant', 'content': '{answer}'},
    ]


def test_apply_template_preserves_source_record_and_nested_values():
    source = {'name': 'Ada', 'metadata': 3}
    operation = ApplyTemplate(
        {'messages': ['Hello {name}', {'literal': 7}]},
        column='formatted',
    )

    result = operation.map(source)

    assert result == {
        'name': 'Ada',
        'metadata': 3,
        'formatted': {
            'messages': ['Hello Ada', {'literal': 7}],
        },
    }
    assert source == {'name': 'Ada', 'metadata': 3}


def tokenize(row):
    values = np.asarray(row['tokens'], dtype=np.int32)
    return {
        'input_ids': values,
        'labels': values.copy(),
    }


def test_packing_pipeline_produces_isolated_causal_lm_batch():
    source = Dataset.from_dict({
        'tokens': [[10, 11, 12], [20, 21], [30, 31, 32, 33]],
    })
    dataloader = DatasetUtils.from_datasets(
        source,
        num_epochs=1,
        operations=[
            Map(tokenize),
            PackSequences(5),
            grain.Batch(2, drop_remainder=False),
            CausalLMBatch(),
        ],
    )

    batch = next(iter(dataloader))

    np.testing.assert_array_equal(batch['input_ids'], [
        [10, 11, 12, 20, 21],
        [30, 31, 32, 33, 0],
    ])
    np.testing.assert_array_equal(batch['position_ids'], [
        [0, 1, 2, 0, 1],
        [0, 1, 2, 3, 0],
    ])
    np.testing.assert_array_equal(batch['labels'], [
        [-100, 11, 12, -100, 21],
        [-100, 31, 32, 33, -100],
    ])
    assert 'attention_mask' not in batch
    assert 'segment_ids' not in batch
    assert batch['position_ids'].shape == (2, 5)


def test_pack_sequences_splits_long_records_without_losing_tokens():
    source = Dataset.from_dict({'tokens': [[1, 2, 3, 4, 5, 6]]})
    dataloader = DatasetUtils.from_datasets(
        source,
        num_epochs=1,
        operations=[Map(tokenize), PackSequences(4)],
    )

    records = list(dataloader)

    np.testing.assert_array_equal(records[0]['input_ids'], [1, 2, 3, 4])
    np.testing.assert_array_equal(records[1]['input_ids'], [5, 6, 0, 0])
    np.testing.assert_array_equal(records[1]['position_ids'], [0, 1, 0, 0])


def test_pack_sequences_can_truncate_long_records():
    source = Dataset.from_dict({'tokens': [[1, 2, 3, 4, 5, 6]]})
    dataloader = DatasetUtils.from_datasets(
        source,
        num_epochs=1,
        operations=[
            Map(tokenize),
            PackSequences(4, overflow='truncate'),
        ],
    )

    records = list(dataloader)

    assert len(records) == 1
    np.testing.assert_array_equal(records[0]['input_ids'], [1, 2, 3, 4])


def test_pack_sequences_does_not_split_regular_records_between_packs():
    source = Dataset.from_dict({'tokens': [[1, 2, 3], [4, 5, 6]]})
    dataloader = DatasetUtils.from_datasets(
        source,
        num_epochs=1,
        operations=[Map(tokenize), PackSequences(5)],
    )

    records = list(dataloader)

    np.testing.assert_array_equal(records[0]['input_ids'], [1, 2, 3, 0, 0])
    np.testing.assert_array_equal(records[1]['input_ids'], [4, 5, 6, 0, 0])


def test_pack_sequences_validates_overflow_mode():
    with pytest.raises(ValueError, match='overflow'):
        PackSequences(4, overflow='continue')


def test_pack_sequences_can_drop_partial_final_record():
    source = Dataset.from_dict({'tokens': [[1, 2, 3]]})
    dataloader = DatasetUtils.from_datasets(
        source,
        num_epochs=1,
        operations=[
            Map(tokenize),
            PackSequences(4, drop_remainder=True),
        ],
    )

    assert list(dataloader) == []


def test_pack_sequences_rejects_batched_tokenizer_output():
    packer = PackSequences(4)
    record = grain.Record(
        metadata=grain.RecordMetadata(index=0),
        data={
            'input_ids': np.asarray([[1, 2]]),
            'labels': np.asarray([[1, 2]]),
        },
    )

    with pytest.raises(ValueError, match='one-dimensional before packing'):
        list(packer(iter([record])))


def test_causal_lm_batch_must_run_after_batching():
    operation = CausalLMBatch()

    with pytest.raises(ValueError, match='must run after batching'):
        operation.map({
            'labels': np.asarray([1, 2]),
            'position_ids': np.asarray([0, 1]),
        })
