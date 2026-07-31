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

import importlib
import math
from typing import Protocol, runtime_checkable

import jax
import numpy as np
import qwix

from ._weights import (
    WeightMapper,
    iter_checkpoint_weights,
    iter_internal_weight_specs,
    iter_internal_weights,
)


@runtime_checkable
class WeightSyncAdapter(Protocol):
    """Transfers a TakTiny model into a started local vLLM engine."""

    def configure(self, llm_options: dict) -> None:
        """Add platform-specific options before constructing ``vllm.LLM``."""

    def sync(self, engine, model, *, policy_version: int, **kwargs) -> None:
        """Publish the current model weights to vLLM."""

    def close(self) -> None:
        """Release adapter-owned transfer resources."""


class _LocalWeightSyncClient:
    def __init__(self, llm, policy_version):
        self.llm = llm
        self.policy_version = str(policy_version)

    def init_weight_transfer_engine(self, init_info):
        self.llm.init_weight_transfer_engine({
            'init_info': init_info,
        })

    def start_weight_update(self):
        self.llm.start_weight_update()

    def update_weights(self, update_info):
        self.llm.update_weights({
            'update_info': update_info,
        })

    def finish_weight_update(self, weight_version=None):
        self.llm.finish_weight_update(
            weight_version or self.policy_version,
        )


def _torch_from_jax(value, torch, device):
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
    def __init__(self, model, *, mapper, torch, param_meta, device):
        self.model = model
        self.mapper = mapper
        self.torch = torch
        self.param_meta = param_meta
        self.device = device

    def __iter__(self):
        for name, value in iter_checkpoint_weights(
            self.model,
            self.mapper,
        ):
            yield name, _torch_from_jax(
                value,
                self.torch,
                self.device,
            )

    def metadata(self):
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
        device=None,
    ):
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

    def configure(self, llm_options):
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

    def _initialize(self, engine, model, policy_version):
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

    def sync(self, engine, model, *, policy_version, **kwargs):
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

    def close(self):
        if self._trainer_engine is not None:
            shutdown = getattr(self._trainer_engine, 'shutdown', None)
            if callable(shutdown):
                shutdown()
        self._trainer_engine = None
        self._source = None
        self._llm = None


def _flat_nnx_state(state):
    flat_state = getattr(state, 'flat_state', None)
    if not callable(flat_state):
        raise TypeError(
            'The TPU vLLM model runner does not expose an NNX state'
        )
    return {
        '.'.join(str(part) for part in path): variable
        for path, variable in flat_state()
    }


def _target_candidates(name):
    candidates = [name]
    if '.embed_tokens.embedding' in name:
        candidates.append(
            name.replace(
                '.embed_tokens.embedding',
                '.embed.embedding',
            )
        )

    if name.endswith('.weight'):
        stem = name.removesuffix('.weight')
        if (
            'layernorm' in stem
            or stem.endswith('.norm')
            or '.norm.' in stem
        ):
            candidates.append(f'{stem}.scale')
        else:
            candidates.append(f'{stem}.kernel')

    if name == 'lm_head.weight':
        candidates.extend(('model.lm_head', 'lm_head'))

    return tuple(dict.fromkeys(candidates))


def _get_variable_value(variable):
    getter = getattr(variable, 'get_value', None)
    return getter() if callable(getter) else getattr(variable, 'value')


def _set_variable_value(variable, value):
    setter = getattr(variable, 'set_value', None)
    if callable(setter):
        setter(value)
    else:
        variable.value = value


class TPUWeightSync:
    """Synchronize weights directly into an in-process vLLM JAX model."""

    def configure(self, llm_options):
        if 'weight_transfer_config' in llm_options:
            raise ValueError(
                'TPUWeightSync uses direct JAX state transfer and does not '
                'accept weight_transfer_config'
            )

    @staticmethod
    def _model_runner(llm):
        try:
            return llm.llm_engine.model_executor.driver_worker.model_runner
        except AttributeError as error:
            raise RuntimeError(
                'TPU weight synchronization requires an in-process vLLM '
                'model runner'
            ) from error

    def sync(
        self,
        engine,
        model,
        *,
        policy_version,
        strict=True,
        **kwargs,
    ):
        if kwargs:
            names = ', '.join(sorted(kwargs))
            raise TypeError(f'Unexpected TPU sync options: {names}')
        if not isinstance(strict, bool):
            raise TypeError('strict must be a boolean')

        llm = engine.llm
        runner = self._model_runner(llm)
        target = _flat_nnx_state(getattr(runner, 'state', None))
        missing = []
        targets = {}

        for name, shape, _ in iter_internal_weight_specs(model):
            variable = next(
                (
                    target[candidate]
                    for candidate in _target_candidates(name)
                    if candidate in target
                ),
                None,
            )
            if variable is None:
                missing.append(name)
                continue

            target_value = _get_variable_value(variable)
            if isinstance(target_value, qwix.QArray):
                raise NotImplementedError(
                    'Direct synchronization into a Qwix-quantized vLLM '
                    'target is not supported'
                )
            if shape != target_value.shape:
                if math.prod(shape) != target_value.size:
                    raise ValueError(
                        f'vLLM target shape mismatch for {name!r}: '
                        f'{shape} != {target_value.shape}'
                    )
            targets[name] = (variable, target_value)

        if strict and missing:
            preview = ', '.join(missing[:8])
            if len(missing) > 8:
                preview += f', ... ({len(missing)} total)'
            raise ValueError(
                'vLLM target has no matching parameters for: '
                f'{preview}'
            )

        reset_prefix_cache = getattr(llm, 'reset_prefix_cache', None)
        if callable(reset_prefix_cache):
            reset_prefix_cache()

        collective_rpc = getattr(llm, 'collective_rpc', None)
        if not callable(collective_rpc):
            collective_rpc = getattr(
                getattr(llm, 'llm_engine', None),
                'collective_rpc',
                None,
            )
        if not callable(collective_rpc):
            raise RuntimeError(
                'TPU weight synchronization requires vLLM collective_rpc'
            )
        collective_rpc('delete_kv_cache')

        try:
            for name, value, _ in iter_internal_weights(model):
                target_entry = targets.get(name)
                if target_entry is None:
                    continue
                variable, target_value = target_entry
                if value.shape != target_value.shape:
                    value = value.reshape(target_value.shape)
                value = value.astype(target_value.dtype)
                sharding = getattr(target_value, 'sharding', None)
                if sharding is not None:
                    value = jax.device_put(value, sharding)
                _set_variable_value(variable, value)
            jax.effects_barrier()

            if hasattr(runner, 'state_leaves'):
                runner.state_leaves = tuple(
                    jax.tree_util.tree_leaves(runner.state)
                )
            update_version = getattr(llm, 'update_weight_version', None)
            if callable(update_version):
                update_version(str(policy_version))
        finally:
            collective_rpc('reinitialize_kv_cache')

    def close(self):
        pass


def default_weight_sync(platform, **kwargs):
    if platform == 'gpu':
        return GPUWeightSync(**kwargs)
    if platform == 'tpu':
        if kwargs:
            names = ', '.join(sorted(kwargs))
            raise TypeError(f'Unexpected TPU synchronizer options: {names}')
        return TPUWeightSync()
    raise NotImplementedError(
        f'No vLLM weight synchronizer for platform {platform!r}'
    )


__all__ = [
    'GPUWeightSync',
    'TPUWeightSync',
    'WeightSyncAdapter',
    'default_weight_sync',
]
