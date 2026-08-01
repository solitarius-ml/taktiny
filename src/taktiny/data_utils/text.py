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
"""Text-specific Grain operations."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any

import grain.python as grain
import numpy as np


class PackSequences:
    """Greedily concatenate token sequences into fixed-length records.

    This is a Grain iterator operation and must run before ``grain.Batch``.
    Every output record includes reset ``position_ids`` so a later
    model-specific operation can preserve boundaries between independent
    examples without materializing a dense attention mask.

    Args:
        sequence_length: Number of tokens in every packed record.
        sequence_keys: Record fields packed along their only dimension. All
            fields must have the same length.
        padding_values: Optional padding value for each sequence field.
        position_key: Output field containing positions within each source
            example.
        overflow: Handle source records longer than ``sequence_length`` by
            splitting them into chunks or truncating them.
        drop_remainder: Drop the final partially filled packed record instead
            of padding it.
    """

    def __init__(
        self,
        sequence_length: int,
        *,
        sequence_keys: Sequence[str] = ('input_ids', 'labels'),
        padding_values: Mapping[str, Any] | None = None,
        position_key: str = 'position_ids',
        overflow: str = 'split',
        drop_remainder: bool = False,
    ):
        if (
            isinstance(sequence_length, bool)
            or not isinstance(sequence_length, int)
            or sequence_length < 1
        ):
            raise ValueError('sequence_length must be a positive integer')
        if (
            isinstance(sequence_keys, (str, bytes))
            or not sequence_keys
            or any(
                not isinstance(key, str) or not key
                for key in sequence_keys
            )
        ):
            raise TypeError(
                'sequence_keys must be a non-empty sequence of strings'
            )
        if len(set(sequence_keys)) != len(sequence_keys):
            raise ValueError('sequence_keys must not contain duplicates')
        if not isinstance(position_key, str) or not position_key:
            raise TypeError('position_key must be a non-empty string')
        if position_key in sequence_keys:
            raise ValueError('packing output field names must be unique')
        if overflow not in {'split', 'truncate'}:
            raise ValueError('overflow must be "split" or "truncate"')
        if not isinstance(drop_remainder, bool):
            raise TypeError('drop_remainder must be a boolean')
        if padding_values is not None and not isinstance(
            padding_values,
            Mapping,
        ):
            raise TypeError('padding_values must be a mapping or None')

        unknown_padding_keys = set(padding_values or ()) - set(sequence_keys)
        if unknown_padding_keys:
            names = ', '.join(sorted(unknown_padding_keys))
            raise ValueError(f'padding_values contains unknown fields: {names}')

        defaults = {
            key: (-100 if key == 'labels' else 0)
            for key in sequence_keys
        }
        defaults.update(padding_values or {})

        self.sequence_length = sequence_length
        self.sequence_keys = tuple(sequence_keys)
        self.padding_values = defaults
        self.position_key = position_key
        self.overflow = overflow
        self.drop_remainder = drop_remainder

    def _normalize_record(self, value):
        if not isinstance(value, Mapping):
            raise TypeError('PackSequences expects mapping records')

        arrays = {}
        length = None
        for key in self.sequence_keys:
            if key not in value:
                raise KeyError(f'packed record is missing {key!r}')
            array = np.asarray(value[key])
            if array.ndim != 1:
                raise ValueError(
                    f'{key!r} must be one-dimensional before packing; '
                    f'got shape {array.shape}'
                )
            if length is None:
                length = len(array)
            elif len(array) != length:
                raise ValueError(
                    'all packed sequence fields must have equal lengths'
                )
            arrays[key] = array

        return arrays, length or 0

    def _finish_pack(
        self,
        values,
        position_ids,
        *,
        drop_incomplete,
    ):
        valid_length = len(position_ids)
        if valid_length == 0:
            return None
        if drop_incomplete and valid_length < self.sequence_length:
            return None

        padding = self.sequence_length - valid_length
        packed = {}
        for key in self.sequence_keys:
            array = np.concatenate(values[key], axis=0)
            if padding:
                array = np.pad(
                    array,
                    (0, padding),
                    constant_values=self.padding_values[key],
                )
            packed[key] = array

        packed[self.position_key] = np.pad(
            np.asarray(position_ids, dtype=np.int32),
            (0, padding),
        )
        return packed

    def __call__(self, input_iterator: Iterable[grain.Record]):
        values = {key: [] for key in self.sequence_keys}
        position_ids = []
        metadata = None
        pack_metadata = None

        def reset():
            nonlocal values, position_ids, pack_metadata
            values = {key: [] for key in self.sequence_keys}
            position_ids = []
            pack_metadata = None

        for record in input_iterator:
            arrays, record_length = self._normalize_record(record.data)
            metadata = record.metadata
            if self.overflow == 'truncate':
                fragments = [(0, min(record_length, self.sequence_length))]
            else:
                fragments = [
                    (offset, min(offset + self.sequence_length, record_length))
                    for offset in range(0, record_length, self.sequence_length)
                ]

            for offset, stop in fragments:
                fragment_length = stop - offset
                if fragment_length == 0:
                    continue

                # Keep ordinary source records intact. Only an explicitly
                # split overlength record may produce multiple fragments.
                if (
                    position_ids
                    and len(position_ids) + fragment_length
                    > self.sequence_length
                ):
                    packed = self._finish_pack(
                        values,
                        position_ids,
                        drop_incomplete=False,
                    )
                    yield grain.Record(
                        metadata=pack_metadata.remove_record_key(),
                        data=packed,
                    )
                    reset()

                for key in self.sequence_keys:
                    values[key].append(arrays[key][offset:stop])
                position_ids.extend(range(fragment_length))
                pack_metadata = metadata

                if len(position_ids) == self.sequence_length:
                    packed = self._finish_pack(
                        values,
                        position_ids,
                        drop_incomplete=False,
                    )
                    yield grain.Record(
                        metadata=metadata.remove_record_key(),
                        data=packed,
                    )
                    reset()

        packed = self._finish_pack(
            values,
            position_ids,
            drop_incomplete=self.drop_remainder,
        )
        if packed is not None and metadata is not None:
            yield grain.Record(
                metadata=metadata.remove_record_key(),
                data=packed,
            )


class CausalLMBatch(grain.MapTransform):
    """Convert a batch of packed sequences into causal-LM training inputs.

    Labels at each example boundary and at padding positions are set to
    ``ignore_index`` so next-token loss never crosses those boundaries. The
    reset ``position_ids`` field is retained for the model; no quadratic
    attention mask is materialized by the dataloader.
    """

    def __init__(
        self,
        *,
        labels_key: str = 'labels',
        position_key: str = 'position_ids',
        ignore_index: int = -100,
    ):
        for name, key in (
            ('labels_key', labels_key),
            ('position_key', position_key),
        ):
            if not isinstance(key, str) or not key:
                raise TypeError(f'{name} must be a non-empty string')
        if isinstance(ignore_index, bool) or not isinstance(ignore_index, int):
            raise TypeError('ignore_index must be an integer')

        super().__init__()
        self.labels_key = labels_key
        self.position_key = position_key
        self.ignore_index = ignore_index

    def map(self, batch):
        if not isinstance(batch, Mapping):
            raise TypeError('CausalLMBatch expects a mapping batch')
        for key in (self.labels_key, self.position_key):
            if key not in batch:
                raise KeyError(f'causal LM batch is missing {key!r}')

        position_ids = np.asarray(batch[self.position_key])
        labels = np.asarray(batch[self.labels_key])
        if position_ids.ndim != 2:
            raise ValueError(
                'CausalLMBatch must run after batching; position_ids must '
                f'have shape [batch, sequence], got {position_ids.shape}'
            )
        if labels.shape != position_ids.shape:
            raise ValueError('labels and position_ids must have equal shapes')

        labels = np.where(
            position_ids == 0,
            self.ignore_index,
            labels,
        )

        result = dict(batch)
        result[self.labels_key] = labels
        return result


class ApplyTemplate(grain.MapTransform):
    """Format a reusable template with values from each dataset record.

    Templates may contain nested strings, mappings, lists, and tuples. The
    original template and source record are never mutated.
    """

    def __init__(
        self,
        template,
        format_fn: Callable[[Any], Any] | None = None,
        column: str = 'template',
    ):
        if format_fn is not None and not callable(format_fn):
            raise TypeError('format_fn must be callable or None')
        
        if not isinstance(column, str) or not column:
            raise TypeError('column must be a non-empty string')

        super().__init__()
        self.template = template
        self.column = column
        self.format_fn = format_fn

    @classmethod
    def _format(cls, template, element):
        if isinstance(template, str):
            return template.format_map(element)
        
        if isinstance(template, Mapping):
            return {
                key: cls._format(value, element)
                for key, value in template.items()
            }
        
        if isinstance(template, list):
            return [cls._format(value, element) for value in template]
        
        if isinstance(template, tuple):
            return tuple(cls._format(value, element) for value in template)
        
        return template

    def map(self, element):
        if not isinstance(element, Mapping):
            raise TypeError('ApplyTemplate expects mapping records')

        value = self._format(self.template, element)
        if self.format_fn is not None:
            value = self.format_fn(value)

        result = dict(element)
        result[self.column] = value
        return result


__all__ = ['PackSequences', 'CausalLMBatch', 'ApplyTemplate']
