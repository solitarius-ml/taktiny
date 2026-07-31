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
"""Local offline vLLM engine for TakTiny token generation."""

from __future__ import annotations

import importlib
import os

import jax
import jax.numpy as jnp
import numpy as np


def _config_value(config, name):
    if config is None:
        return None
    if isinstance(config, dict):
        return config.get(name)
    return getattr(config, name, None)


def _model_source(model, explicit_source=None):
    if explicit_source is not None:
        return os.fspath(explicit_source)

    source = getattr(model, 'base_model_name_or_path', None)
    if source is not None:
        return os.fspath(source)

    config = getattr(model, 'config', None)
    for name in ('_name_or_path', 'name_or_path'):
        source = _config_value(config, name)
        if source:
            return os.fspath(source)

    raise ValueError(
        'Unable to determine the vLLM model source. Load the model with '
        'Maestro.from_pretrained or pass model_path to VLLM.'
    )


def _token_ids(value):
    if value is None:
        return None
    if isinstance(value, (str, bytes)):
        raise TypeError('token IDs must be integers')
    values = np.asarray(jax.device_get(value))
    if values.ndim == 0:
        return [int(values)]
    if values.ndim != 1:
        raise ValueError('token IDs must be a scalar or one-dimensional')
    return [int(token_id) for token_id in values]


class LocalVLLMEngine:
    """Offline vLLM engine selected from the active JAX platform."""

    def __init__(
        self,
        model,
        *,
        platform='auto',
        model_path=None,
        use_tqdm=False,
        sync_adapter=None,
        weight_mapper=None,
        sync_packed=True,
        sync_buffer_size=None,
        sync_device=None,
        **llm_options,
    ):
        if platform == 'auto':
            platform = jax.default_backend()
        if platform not in ('gpu', 'tpu'):
            raise NotImplementedError(
                'Local vLLM inference requires a GPU or TPU JAX backend, '
                f'got {platform!r}'
            )
        if not isinstance(use_tqdm, bool) and not callable(use_tqdm):
            raise TypeError('use_tqdm must be a boolean or callable')
        if not isinstance(sync_packed, bool):
            raise TypeError('sync_packed must be a boolean')
        if sync_buffer_size is not None and sync_buffer_size <= 0:
            raise ValueError('sync_buffer_size must be positive')

        self.model = model
        self.platform = platform
        self.model_path = _model_source(model, model_path)
        self.use_tqdm = use_tqdm
        self.llm_options = dict(llm_options)
        self.sync_adapter = sync_adapter
        self.weight_mapper = weight_mapper
        self.sync_packed = sync_packed
        self.sync_buffer_size = sync_buffer_size
        self.sync_device = sync_device
        self._llm = None
        self._sampling_params_cls = None
        self._weight_sync = None

    @property
    def llm(self):
        """Return the underlying ``vllm.LLM`` instance after startup."""
        if self._llm is None:
            raise RuntimeError('vLLM engine has not been started')
        return self._llm

    def _import_vllm(self):
        if self.platform == 'tpu':
            os.environ.setdefault('TPU_BACKEND_TYPE', 'jax')
            os.environ.setdefault('MODEL_IMPL_TYPE', 'flax_nnx')
            os.environ['VLLM_ENABLE_V1_MULTIPROCESSING'] = '0'
        try:
            module = importlib.import_module('vllm')
        except ModuleNotFoundError as error:
            if error.name != 'vllm':
                raise
            package = 'vllm-tpu' if self.platform == 'tpu' else 'vllm'
            raise ImportError(
                f'{package} is required for vLLM inference on '
                f'{self.platform.upper()}'
            ) from error

        llm_cls = getattr(module, 'LLM', None)
        sampling_params_cls = getattr(module, 'SamplingParams', None)
        if llm_cls is None or sampling_params_cls is None:
            raise ImportError(
                'The installed vLLM package does not expose LLM and '
                'SamplingParams'
            )
        return llm_cls, sampling_params_cls

    def start(self):
        if self._llm is not None:
            return
        llm_cls, self._sampling_params_cls = self._import_vllm()
        options = dict(self.llm_options)
        if self.sync_adapter is None:
            from ._sync import default_weight_sync

            sync_options = {}
            if self.platform == 'gpu':
                sync_options = {
                    'mapper': self.weight_mapper,
                    'packed': self.sync_packed,
                    'buffer_size': self.sync_buffer_size,
                    'device': self.sync_device,
                }
            elif self.weight_mapper is not None:
                raise ValueError(
                    'weight_mapper is only used by GPU checkpoint-format '
                    'synchronization'
                )
            self._weight_sync = default_weight_sync(
                self.platform,
                **sync_options,
            )
        else:
            self._weight_sync = self.sync_adapter

        configure = getattr(self._weight_sync, 'configure', None)
        if callable(configure):
            configure(options)
        options.setdefault('skip_tokenizer_init', True)
        options.setdefault('generation_config', 'vllm')
        self._llm = llm_cls(
            model=self.model_path,
            **options,
        )

    def _prepare_inputs(self, input_ids, attention_mask):
        input_ids = np.asarray(jax.device_get(input_ids))
        if input_ids.ndim != 2:
            raise ValueError(
                'input_ids must have shape [batch, sequence]'
            )
        if not np.issubdtype(input_ids.dtype, np.integer):
            raise TypeError('input_ids must contain integer token IDs')

        if attention_mask is None:
            attention_mask = np.ones_like(input_ids, dtype=np.bool_)
        else:
            attention_mask = np.asarray(
                jax.device_get(attention_mask),
                dtype=np.bool_,
            )
            if attention_mask.shape != input_ids.shape:
                raise ValueError(
                    'attention_mask must have the same shape as input_ids'
                )

        prompt_lengths = attention_mask.sum(axis=-1)
        if np.any(prompt_lengths == 0):
            raise ValueError('each prompt must contain at least one token')

        prompts = [
            {
                'prompt_token_ids': [
                    int(token_id)
                    for token_id in row[mask]
                ],
            }
            for row, mask in zip(input_ids, attention_mask, strict=True)
        ]
        return input_ids, prompts

    def _sampling_params(
        self,
        *,
        max_new_tokens,
        temperature,
        top_k,
        top_p,
        seed,
        repetition_penalty,
        eos_token_id,
    ):
        if not isinstance(top_k, int) or top_k < 0:
            raise ValueError('top_k must be a non-negative integer')
        if not 0 < top_p <= 1:
            raise ValueError('top_p must be in the interval (0, 1]')
        if repetition_penalty <= 0:
            raise ValueError('repetition_penalty must be positive')
        if not isinstance(seed, int):
            raise TypeError('seed must be an integer')

        config = getattr(self.model, 'config', None)
        if eos_token_id is None:
            eos_token_id = _config_value(config, 'eos_token_id')

        return self._sampling_params_cls(
            temperature=max(float(temperature), 0.0),
            top_k=top_k,
            top_p=float(top_p),
            seed=seed,
            repetition_penalty=float(repetition_penalty),
            stop_token_ids=_token_ids(eos_token_id),
            max_tokens=max_new_tokens,
            detokenize=False,
            skip_special_tokens=False,
        )

    def _normalize_outputs(
        self,
        input_ids,
        outputs,
        *,
        pad_token_id,
        eos_token_id,
        max_new_tokens,
    ):
        batch_size = input_ids.shape[0]
        if len(outputs) != batch_size:
            raise RuntimeError(
                'vLLM returned an unexpected number of outputs: '
                f'expected {batch_size}, got {len(outputs)}'
            )

        generated = []
        for output in outputs:
            completions = getattr(output, 'outputs', None)
            if not completions:
                raise RuntimeError(
                    'vLLM returned a request without a completion'
                )
            token_ids = np.asarray(
                getattr(completions[0], 'token_ids', ()),
                dtype=input_ids.dtype,
            )
            if token_ids.ndim != 1:
                raise RuntimeError(
                    'vLLM completion token_ids must be one-dimensional'
                )
            if token_ids.size > max_new_tokens:
                raise RuntimeError(
                    'vLLM returned more tokens than max_new_tokens'
                )
            generated.append(token_ids)

        generated_length = max(
            (token_ids.size for token_ids in generated),
            default=0,
        )
        if generated_length == 0:
            return jnp.asarray(input_ids)

        config = getattr(self.model, 'config', None)
        if pad_token_id is None:
            pad_token_id = _config_value(config, 'pad_token_id')
        if pad_token_id is None:
            eos_values = _token_ids(
                eos_token_id
                if eos_token_id is not None
                else _config_value(config, 'eos_token_id')
            )
            pad_token_id = eos_values[0] if eos_values else 0
        pad_token_id = int(pad_token_id)

        completion_ids = np.full(
            (batch_size, generated_length),
            pad_token_id,
            dtype=input_ids.dtype,
        )
        for row, token_ids in enumerate(generated):
            completion_ids[row, :token_ids.size] = token_ids

        return jnp.asarray(
            np.concatenate([input_ids, completion_ids], axis=1)
        )

    def generate(
        self,
        input_ids,
        max_new_tokens,
        temperature=1.0,
        top_k=50,
        top_p=1.0,
        seed=42,
        attention_mask=None,
        repetition_penalty=1.0,
        eos_token_id=None,
        pad_token_id=None,
        streamer=None,
    ):
        if not isinstance(max_new_tokens, int) or max_new_tokens < 0:
            raise ValueError(
                'max_new_tokens must be a non-negative integer'
            )
        if streamer is not None:
            raise NotImplementedError(
                'streamer requires the asynchronous vLLM engine'
            )

        input_ids, prompts = self._prepare_inputs(
            input_ids,
            attention_mask,
        )
        if max_new_tokens == 0:
            return jnp.asarray(input_ids)

        sampling_params = self._sampling_params(
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            seed=seed,
            repetition_penalty=repetition_penalty,
            eos_token_id=eos_token_id,
        )
        outputs = self.llm.generate(
            prompts,
            sampling_params,
            use_tqdm=self.use_tqdm,
        )
        return self._normalize_outputs(
            input_ids,
            outputs,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            max_new_tokens=max_new_tokens,
        )

    def sync(self, model, *, policy_version, **kwargs):
        if self._llm is None or self._weight_sync is None:
            raise RuntimeError('vLLM engine has not been started')
        sync = getattr(self._weight_sync, 'sync', None)
        if not callable(sync):
            raise TypeError(
                'sync_adapter must provide a callable sync method'
            )
        sync(
            self,
            model,
            policy_version=policy_version,
            **kwargs,
        )

    def close(self):
        if self._llm is None:
            return

        adapter_close = getattr(self._weight_sync, 'close', None)
        if callable(adapter_close):
            adapter_close()

        shutdown_targets = (
            self._llm,
            getattr(self._llm, 'llm_engine', None),
            getattr(
                getattr(self._llm, 'llm_engine', None),
                'engine_core',
                None,
            ),
        )
        for target in shutdown_targets:
            shutdown = getattr(target, 'shutdown', None)
            if callable(shutdown):
                shutdown()
                break
        self._llm = None
        self._sampling_params_cls = None
        self._weight_sync = None
