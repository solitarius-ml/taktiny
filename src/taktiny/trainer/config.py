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
from typing import Any, Optional, Iterable, Callable


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int = 1
    max_steps: Optional[int] = None
    learning_rate: float = 1e-3
    optimizer: Any = None # Optax optimizer, defaults to adamw if None
    weight_decay: float = 0.0
    log_interval: int = 10
    seed: int = 42
    jit_compile: bool = False
    donate_batch: bool = False
    remat: bool = False

    def __post_init__(self):
        if self.epochs < 1:
            raise ValueError('epochs must be a positive integer')
        if self.max_steps is not None and self.max_steps < 1:
            raise ValueError('max_steps must be a positive integer or None')
        if self.log_interval < 1:
            raise ValueError('log_interval must be a positive integer')
        if not isinstance(self.remat, bool):
            raise TypeError('remat must be a boolean')

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
