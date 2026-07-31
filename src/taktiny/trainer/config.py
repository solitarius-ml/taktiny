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
"""Trainer config"""

from __future__ import annotations

from dataclasses import dataclass, field
from os import PathLike
from typing import Any, Optional, Iterable, Callable


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int = 1
    max_steps: Optional[int] = None
    learning_rate: float = 1e-3
    schedule: Optional[Callable] = None
    optimizer: Any = None # Optax optimizer, defaults to adamw if None
    weight_decay: float = 0.0
    log_interval: int = 10
    seed: int = 42
    jit_compile: bool = True
    donate_batch: bool = False
    output_dir: str | PathLike | None = None
    save_steps: Optional[int] = None
    save_total_limit: Optional[int] = None
    save_at_end: bool = False
    save_optimizer_state: bool = True
    save_async: bool = False
    max_shard_size: int | str = '5GB'
    eval_strategy: str = 'no'
    eval_steps: Optional[int] = None
    metric_for_best_model: str = 'eval_loss'
    greater_is_better: Optional[bool] = None
    load_best_model_at_end: bool = False
    gradient_accumulation_steps: int = 1
    max_grad_norm: Optional[float] = None
    skip_non_finite: bool = True
    loss_scale: float | str | None = None
    initial_loss_scale: float = 32768.0
    loss_scale_growth_interval: int = 2000

    def __post_init__(self):
        if self.epochs < 1:
            raise ValueError('epochs must be a positive integer')
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise TypeError('seed must be an integer')
        if self.max_steps is not None and self.max_steps < 1:
            raise ValueError('max_steps must be a positive integer or None')
        if self.log_interval < 1:
            raise ValueError('log_interval must be a positive integer')
        if self.schedule is not None and not callable(self.schedule):
            raise TypeError('schedule must be callable or None')
        if (
            self.save_steps is not None
            and (
                isinstance(self.save_steps, bool)
                or self.save_steps < 1
            )
        ):
            raise ValueError('save_steps must be a positive integer or None')
        if (
            self.save_total_limit is not None
            and (
                isinstance(self.save_total_limit, bool)
                or self.save_total_limit < 1
            )
        ):
            raise ValueError(
                'save_total_limit must be a positive integer or None'
            )
        if not isinstance(self.save_at_end, bool):
            raise TypeError('save_at_end must be a boolean')
        if not isinstance(self.save_optimizer_state, bool):
            raise TypeError('save_optimizer_state must be a boolean')
        if not isinstance(self.save_async, bool):
            raise TypeError('save_async must be a boolean')
        if self.eval_strategy not in {'no', 'steps', 'epoch'}:
            raise ValueError(
                'eval_strategy must be "no", "steps", or "epoch"'
            )
        if (
            self.eval_steps is not None
            and (
                isinstance(self.eval_steps, bool)
                or not isinstance(self.eval_steps, int)
                or self.eval_steps < 1
            )
        ):
            raise ValueError('eval_steps must be a positive integer or None')
        if self.eval_strategy == 'steps' and self.eval_steps is None:
            raise ValueError(
                'eval_steps is required when eval_strategy="steps"'
            )
        if not (
            isinstance(self.metric_for_best_model, str)
            and self.metric_for_best_model
        ):
            raise TypeError(
                'metric_for_best_model must be a non-empty string'
            )
        if (
            self.greater_is_better is not None
            and not isinstance(self.greater_is_better, bool)
        ):
            raise TypeError('greater_is_better must be a boolean or None')
        if not isinstance(self.load_best_model_at_end, bool):
            raise TypeError('load_best_model_at_end must be a boolean')
        if (
            self.load_best_model_at_end
            and self.eval_strategy == 'no'
        ):
            raise ValueError(
                'load_best_model_at_end requires evaluation'
            )
        if (
            isinstance(self.gradient_accumulation_steps, bool)
            or not isinstance(self.gradient_accumulation_steps, int)
            or self.gradient_accumulation_steps < 1
        ):
            raise ValueError(
                'gradient_accumulation_steps must be a positive integer'
            )
        if (
            self.max_grad_norm is not None
            and (
                isinstance(self.max_grad_norm, bool)
                or not isinstance(self.max_grad_norm, (int, float))
                or self.max_grad_norm <= 0
            )
        ):
            raise ValueError('max_grad_norm must be positive or None')
        if not isinstance(self.skip_non_finite, bool):
            raise TypeError('skip_non_finite must be a boolean')
        if not (
            self.loss_scale is None
            or self.loss_scale == 'dynamic'
            or (
                isinstance(self.loss_scale, (int, float))
                and not isinstance(self.loss_scale, bool)
                and self.loss_scale > 0
            )
        ):
            raise ValueError(
                'loss_scale must be None, "dynamic", or a positive number'
            )
        if (
            isinstance(self.initial_loss_scale, bool)
            or not isinstance(self.initial_loss_scale, (int, float))
            or self.initial_loss_scale <= 0
        ):
            raise ValueError('initial_loss_scale must be positive')
        if (
            isinstance(self.loss_scale_growth_interval, bool)
            or not isinstance(self.loss_scale_growth_interval, int)
            or self.loss_scale_growth_interval < 1
        ):
            raise ValueError(
                'loss_scale_growth_interval must be a positive integer'
            )
        if (
            self.output_dir is None
            and (
                self.save_steps is not None
                or self.save_at_end
                or self.load_best_model_at_end
            )
        ):
            raise ValueError(
                'output_dir is required when checkpoint saving is enabled'
            )

@dataclass(frozen=True)
class DatasetConfig:
    """Configure an existing dataloader or an automatic HF dataset source."""

    # A generic iterable that yields batches (e.g. Grain, PyTorch, or custom).
    # When supplied, all repo-loading options below are ignored.
    dataloader: Optional[Iterable[Any]] = None
    validation_dataloader: Optional[Iterable[Any]] = None
    batch_size: int = 1
    repo_id: Optional[str] = None
    process_fn: Optional[Callable] = None
    streaming: bool = False
    hf_token: Optional[str] = field(default=None, repr=False)
    # Sharding applied to every batch leaf, or a matching sharding PyTree.
    batch_sharding: Any = None
    shuffle: bool = True
    seed: int = 42
    prefetch_size: int = 2

    def __post_init__(self):
        if self.dataloader is None:
            if not self.repo_id:
                raise ValueError(
                    'repo_id is required when dataloader is not provided'
                )
            if self.process_fn is not None and not callable(self.process_fn):
                raise TypeError('process_fn must be callable or None')
            if not isinstance(self.streaming, bool):
                raise TypeError('streaming must be a boolean')
            if (
                self.hf_token is not None
                and not isinstance(self.hf_token, str)
            ):
                raise TypeError('hf_token must be a string or None')
            if not isinstance(self.shuffle, bool):
                raise TypeError('shuffle must be a boolean')
            if isinstance(self.seed, bool) or not isinstance(self.seed, int):
                raise TypeError('seed must be an integer')
        if (
            isinstance(self.prefetch_size, bool)
            or not isinstance(self.prefetch_size, int)
            or self.prefetch_size < 0
        ):
            raise ValueError('prefetch_size must be a non-negative integer')


@dataclass(frozen=True)
class SFTTrainingConfig(TrainingConfig):
    """Training and loss settings for supervised causal language modeling.

    ``completion_only_loss=None`` selects completion-only loss automatically
    for prompt-completion records and full-sequence loss for text records.
    ``assistant_only_loss`` uses assistant masks supplied by conversational
    chat templates. Explicit labels in pretokenized records always take
    precedence over both options.
    """

    completion_only_loss: Optional[bool] = None
    assistant_only_loss: bool = False
    ignore_index: int = -100

    def __post_init__(self):
        super().__post_init__()
        if (
            self.completion_only_loss is not None
            and not isinstance(self.completion_only_loss, bool)
        ):
            raise TypeError(
                'completion_only_loss must be a boolean or None'
            )
        if not isinstance(self.assistant_only_loss, bool):
            raise TypeError('assistant_only_loss must be a boolean')
        if (
            isinstance(self.ignore_index, bool)
            or not isinstance(self.ignore_index, int)
        ):
            raise TypeError('ignore_index must be an integer')


@dataclass(frozen=True)
class SFTDatasetConfig(DatasetConfig):
    """Dataset preparation settings for supervised fine-tuning.

    The source may contain plain text, conversational messages,
    prompt-completion pairs, or pretokenized records. ``process_fn`` transforms
    an automatically loaded dataset once; ``formatting_fn`` transforms each
    record before format detection. Dynamic padding uses the longest sequence
    in each batch. Packing fills fixed ``max_length`` sequences and creates
    block-diagonal attention masks between original examples.
    """

    tokenizer: Any = field(default=None, repr=False)
    max_length: Optional[int] = 1024
    padding: str = 'longest'
    pad_to_multiple_of: Optional[int] = None
    packing: bool = False
    drop_remainder: bool = True
    append_eos: bool = True
    dataset_text_field: str = 'text'
    messages_field: str = 'messages'
    prompt_field: str = 'prompt'
    completion_field: str = 'completion'
    formatting_fn: Optional[Callable] = None
    collate_fn: Optional[Callable] = None
    skip_prepare_dataset: bool = False
    shuffle_buffer_size: int = 10_000

    def __post_init__(self):
        super().__post_init__()
        if not isinstance(self.streaming, bool):
            raise TypeError('streaming must be a boolean')
        if not isinstance(self.shuffle, bool):
            raise TypeError('shuffle must be a boolean')
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise TypeError('seed must be an integer')
        if self.tokenizer is None and not self.skip_prepare_dataset:
            raise ValueError(
                'tokenizer is required unless skip_prepare_dataset=True'
            )
        if (
            isinstance(self.batch_size, bool)
            or not isinstance(self.batch_size, int)
            or self.batch_size < 1
        ):
            raise ValueError('batch_size must be a positive integer')
        if (
            self.max_length is not None
            and (
                isinstance(self.max_length, bool)
                or not isinstance(self.max_length, int)
                or self.max_length < 2
            )
        ):
            raise ValueError(
                'max_length must be at least 2 or None'
            )
        if self.padding not in {'longest', 'max_length'}:
            raise ValueError(
                'padding must be "longest" or "max_length"'
            )
        if self.padding == 'max_length' and self.max_length is None:
            raise ValueError(
                'max_length is required when padding="max_length"'
            )
        if (
            self.pad_to_multiple_of is not None
            and (
                isinstance(self.pad_to_multiple_of, bool)
                or not isinstance(self.pad_to_multiple_of, int)
                or self.pad_to_multiple_of < 1
            )
        ):
            raise ValueError(
                'pad_to_multiple_of must be a positive integer or None'
            )
        if (
            self.max_length is not None
            and self.pad_to_multiple_of is not None
            and self.max_length % self.pad_to_multiple_of != 0
        ):
            raise ValueError(
                'max_length must be divisible by pad_to_multiple_of '
                'when both are specified'
            )
        if not isinstance(self.packing, bool):
            raise TypeError('packing must be a boolean')
        if self.packing and self.max_length is None:
            raise ValueError('max_length is required when packing=True')
        if not isinstance(self.drop_remainder, bool):
            raise TypeError('drop_remainder must be a boolean')
        if not isinstance(self.append_eos, bool):
            raise TypeError('append_eos must be a boolean')
        for name in (
            'dataset_text_field',
            'messages_field',
            'prompt_field',
            'completion_field',
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise TypeError(f'{name} must be a non-empty string')
        if self.formatting_fn is not None and not callable(
            self.formatting_fn
        ):
            raise TypeError('formatting_fn must be callable or None')
        if self.collate_fn is not None and not callable(self.collate_fn):
            raise TypeError('collate_fn must be callable or None')
        if not isinstance(self.skip_prepare_dataset, bool):
            raise TypeError('skip_prepare_dataset must be a boolean')
        if (
            isinstance(self.shuffle_buffer_size, bool)
            or not isinstance(self.shuffle_buffer_size, int)
            or self.shuffle_buffer_size < 1
        ):
            raise ValueError(
                'shuffle_buffer_size must be a positive integer'
            )


__all__ = [
    'DatasetConfig',
    'SFTDatasetConfig',
    'SFTTrainingConfig',
    'TrainingConfig',
]
