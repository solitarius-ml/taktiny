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
"""Platform-specific vLLM weight synchronization."""

from __future__ import annotations
from typing import Any


import importlib
from typing import Protocol, runtime_checkable

import jax
import numpy as np

from ._weights import (
    WeightMapper,
    iter_checkpoint_weights,
)


@runtime_checkable
class WeightSyncAdapter(Protocol):
    """Transfers a TakTiny model into a started local vLLM engine."""

    def configure(self, llm_options: dict) -> None:
        """Add platform-specific options before constructing ``vllm.LLM``."""

    def sync(self, engine: Any, model: Any, *, policy_version: int, **kwargs: Any) -> None:
        """Publish the current model weights to vLLM."""

    def close(self) -> None:
        """Release adapter-owned transfer resources."""


class _LocalWeightSyncClient:
    def __init__(self, llm: Any, policy_version: Any) -> None:
        self.llm = llm
        self.policy_version = str(policy_version)

    def init_weight_transfer_engine(self, init_info: Any) -> None:
        self.llm.init_weight_transfer_engine({
            'init_info': init_info,
        })

    def start_weight_update(self) -> None:
        self.llm.start_weight_update()

    def update_weights(self, update_info: Any) -> None:
        self.llm.update_weights({
            'update_info': update_info,
        })

    def finish_weight_update(self, weight_version: Any=None) -> None:
        self.llm.finish_weight_update(
            weight_version or self.policy_version,
        )


def _torch_from_jax(value: Any, torch: Any, device: str) -> Any:
    value = jax.block_until_ready(value)
    try:
        devices = value.devices()
    except AttributeError:
        devices = set()

    if len(devices) == 1:
        source_device = next(iter(devices))
        if getattr(source_device, 'platform', None) == 'gpu':
            try:
                tensor = torch.utils.dlpack.from_dlpack(value)
            except (BufferError, RuntimeError, TypeError):
                pass
            else:
                if device is None or str(tensor.device) == str(device):
                    return tensor
                return tensor.to(device)

    host = np.asarray(jax.device_get(value))
    if str(host.dtype) == 'bfloat16':
        tensor = torch.from_numpy(host.view(np.uint16)).view(
            torch.bfloat16
        )
    else:
        tensor = torch.from_numpy(host)
    return tensor.to(device or 'cuda')


class _JaxWeightSource:
    def __init__(self, model: Any, *, mapper: Any, torch: Any, param_meta: Any, device: str) -> None:
        self.model = model
        self.mapper = mapper
        self.torch = torch
        self.param_meta = param_meta
        self.device = device

    def __iter__(self) -> Any:
        for name, value in iter_checkpoint_weights(
            self.model,
            self.mapper,
        ):
            yield name, _torch_from_jax(
                value,
                self.torch,
                self.device,
            )

    def metadata(self) -> Any:
        metadata = []
        for name, tensor in self:
            metadata.append(
                self.param_meta(
                    name,
                    tensor.dtype,
                    tuple(tensor.shape),
                )
            )
        return metadata


class GPUWeightSync:
    """Synchronize checkpoint-format weights with vLLM CUDA IPC."""

    def __init__(
        self,
        *,
        mapper: WeightMapper | None = None,
        packed: bool = True,
        buffer_size: int | None = None,
        device: str | None=None,
    ) -> None:
        if not isinstance(packed, bool):
            raise TypeError('packed must be a boolean')
        if buffer_size is not None and buffer_size <= 0:
            raise ValueError('buffer_size must be positive')
        self.mapper = mapper
        self.packed = packed
        self.buffer_size = buffer_size
        self.device = device
        self._trainer_engine = None
        self._source = None
        self._llm = None
        self._tensor_parallel_size = 1

    def configure(self, llm_options: Any) -> None:
        self._tensor_parallel_size = int(
            llm_options.get('tensor_parallel_size', 1)
        )
        config = llm_options.get('weight_transfer_config')
        if config is None:
            try:
                config_module = importlib.import_module('vllm.config')
                config_cls = getattr(
                    config_module,
                    'WeightTransferConfig',
                )
            except (ImportError, AttributeError) as error:
                raise ImportError(
                    'The installed vLLM does not provide weight-transfer '
                    'configuration support'
                ) from error
            llm_options['weight_transfer_config'] = config_cls(
                backend='ipc',
            )
        elif getattr(config, 'backend', None) != 'ipc':
            raise ValueError(
                'GPUWeightSync requires weight_transfer_config backend '
                "'ipc'"
            )

    def _initialize(self, engine: Any, model: Any, policy_version: Any) -> None:
        try:
            torch = importlib.import_module('torch')
            base = importlib.import_module(
                'vllm.distributed.weight_transfer.base'
            )
            factory = importlib.import_module(
                'vllm.distributed.weight_transfer.factory'
            )
            ipc = importlib.import_module(
                'vllm.distributed.weight_transfer.ipc_engine'
            )
        except ImportError as error:
            raise ImportError(
                'This vLLM installation does not provide CUDA IPC weight '
                'transfer'
            ) from error

        source = _JaxWeightSource(
            model,
            mapper=self.mapper,
            torch=torch,
            param_meta=base.ParamMeta,
            device=self.device,
        )
        init_kwargs = {
            'rank': 0,
            'packed': self.packed,
        }
        if self.buffer_size is not None:
            init_kwargs['packed_buffer_size_bytes'] = self.buffer_size
        init_info = ipc.IPCTrainerInitInfo(**init_kwargs)
        client = _LocalWeightSyncClient(engine.llm, policy_version)
        trainer_engine = factory.WeightTransferTrainerFactory.trainer_init(
            init_info=init_info,
            client=client,
            source=source,
        )
        self._trainer_engine = trainer_engine
        self._source = source
        self._llm = engine.llm

    def sync(self, engine: Any, model: Any, *, policy_version: Any, **kwargs: Any) -> None:
        if kwargs:
            names = ', '.join(sorted(kwargs))
            raise TypeError(f'Unexpected GPU sync options: {names}')
        if jax.process_count() != 1:
            raise NotImplementedError(
                'The local CUDA IPC adapter supports one JAX process. '
                'Use a custom WeightSyncAdapter for multi-host training.'
            )
        if self._tensor_parallel_size != 1:
            raise NotImplementedError(
                'The local CUDA IPC adapter currently supports vLLM '
                'tensor_parallel_size=1. Use a custom WeightSyncAdapter '
                'for multi-GPU inference.'
            )
        if self._trainer_engine is None:
            self._initialize(engine, model, policy_version)
        elif self._llm is not engine.llm:
            raise RuntimeError(
                'GPU weight synchronizer is bound to a different vLLM '
                'instance'
            )
        else:
            self._trainer_engine.client.policy_version = str(
                policy_version
            )
        self._trainer_engine.send_weights()

    def close(self) -> None:
        if self._trainer_engine is not None:
            shutdown = getattr(self._trainer_engine, 'shutdown', None)
            if callable(shutdown):
                shutdown()
        self._trainer_engine = None
        self._source = None
        self._llm = None


def default_weight_sync(platform: str, **kwargs: Any) -> Any:
    if platform == 'gpu':
        return GPUWeightSync(**kwargs)
    if platform == 'tpu':
        raise NotImplementedError(
            'TakTiny does not provide a vLLM TPU weight synchronizer. '
            'vLLM TPU models use a different model and KV-cache contract.'
        )
    raise NotImplementedError(
        f'No vLLM weight synchronizer for platform {platform!r}'
    )


__all__ = [
    'GPUWeightSync',
    'WeightSyncAdapter',
    'default_weight_sync',
]
