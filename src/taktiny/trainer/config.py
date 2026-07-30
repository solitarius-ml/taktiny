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

from dataclasses import dataclass
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
    jit_compile: bool = False
    donate_batch: bool = False
    remat: bool = False
    output_dir: str | PathLike | None = None
    save_steps: Optional[int] = None
    save_total_limit: Optional[int] = None
    save_at_end: bool = False
    save_optimizer_state: bool = True
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
        if self.max_steps is not None and self.max_steps < 1:
            raise ValueError('max_steps must be a positive integer or None')
        if self.log_interval < 1:
            raise ValueError('log_interval must be a positive integer')
        if self.schedule is not None and not callable(self.schedule):
            raise TypeError('schedule must be callable or None')
        if not isinstance(self.remat, bool):
            raise TypeError('remat must be a boolean')
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
    # A generic iterable that yields batches (e.g. tf.data, PyTorch DataLoader, or custom generator)
    dataloader: Iterable[Any]
    validation_dataloader: Optional[Iterable[Any]] = None
    repo_id: Optional[str] = None
    process_fn: Optional[Callable] = None        # process dataset downloaded from given `repo_id`; return Tuple[train, validation] otherwise train
    loader_type: Optional[str | Callable] = None # apply loader after `process_fn` e.g. 'grain', or Callable return custom loader 
    batch_sharding: Any = None   # Sharding applied to every batch leaf, or a matching sharding PyTree
    shuffle: bool = True
    seed: int = 42              # shuffle seed if `shuffle == True`
    prefetch_size: int = 2

    def __post_init__(self):
        if self.prefetch_size < 0:
            raise ValueError('prefetch_size must be a non-negative integer')


__all__ = ['TrainingConfig', 'DatasetConfig']
