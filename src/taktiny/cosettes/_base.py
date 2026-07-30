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

from __future__ import annotations

import os
import json
import jax
from huggingface_hub import hf_hub_download
from safetensors.flax import save_file
from taktiny.nn import Module, Rngs


class PretrainedModel(Module):
    """Base class for models that load and save pretrained checkpoints.

    Models are serialized as Safetensors together with a weight index. Loading
    first constructs an abstract parameter tree with ``jax.eval_shape``, then
    maps checkpoint names to module paths, applies any requested quantization,
    and places arrays using parameter sharding metadata.

    Subclasses are expected to accept ``config`` and ``rngs`` in their
    constructor. They may provide module-mapping rules to translate external
    checkpoint names and may expose default logical sharding rules.
    """

    def save_pretrained(self, path):
        """
        Saves the model's weights to a local directory in safetensors format.
        Always saves an index json for unified loading.
        """
        os.makedirs(path, exist_ok=True)
        weights_path = os.path.join(path, "model.safetensors")
        index_path = os.path.join(path, "model.safetensors.index.json")
        
        # Extract flat state dict (mapping string paths to JAX arrays)
        state_dict = self.flat_state_dict()
        save_file(state_dict, weights_path)
        
        # Always create an index file for consistency
        index_data = {
            "metadata": {"total_size": os.path.getsize(weights_path)},
            "weight_map": {k: "model.safetensors" for k in state_dict.keys()}
        }
        with open(index_path, "w") as f:
            json.dump(index_data, f, indent=2)
            

    @classmethod
    def from_pretrained(
        cls, 
        path_or_repo, 
        config, 
        module_map=None, 
        local=False, 
        dtype=None, 
        quant=None,
        subfolder=None, 
        mesh=None, 
        sharding_rules=None, 
        **kwargs
    ):
        """
        Loads safetensors weights into a newly instantiated model.
        Supports both single-file (model.safetensors) and sharded models.
        """
        uniform_quant = None
        if dtype is not None:
            dtype_name = dtype.lower() if isinstance(dtype, str) else None
            quantized_dtypes = {'fp8', 'int8', 'int4', 'nf4'}
            if dtype_name in quantized_dtypes:
                compute_dtype = (
                    getattr(config, 'torch_dtype', None)
                    or getattr(config, 'dtype', None)
                )
                if (
                    compute_dtype is None
                    or (
                        isinstance(compute_dtype, str)
                        and compute_dtype.lower() in quantized_dtypes
                    )
                ):
                    compute_dtype = 'bfloat16'

                uniform_quant = dtype_name
                setattr(config, 'dtype', compute_dtype)
                setattr(config, 'torch_dtype', compute_dtype)
            else:
                setattr(config, 'dtype', dtype)
                setattr(config, 'torch_dtype', dtype)
        if quant is not None and uniform_quant is not None:
            from ..utils.quantization import merge_quantization

            setattr(
                config,
                'quant',
                merge_quantization(quant, uniform_quant),
            )
        elif quant is not None:
            setattr(config, 'quant', quant)
        elif uniform_quant is not None:
            setattr(config, 'quant', uniform_quant)

        path_or_repo_str = str(path_or_repo)
        module_map = module_map or []
        if isinstance(module_map, dict):
            module_map = list(module_map.items())

        # 1. Determine if model is sharded or single file
        is_sharded = False
        if local:
            index_path = os.path.join(path_or_repo_str, subfolder if subfolder else "", "model.safetensors.index.json")
            if os.path.exists(index_path):
                is_sharded = True
        else:
            from huggingface_hub import repo_info
            try:
                info = repo_info(repo_id=path_or_repo_str)
                files = [f.rfilename for f in info.siblings]
                target_index = f"{subfolder}/model.safetensors.index.json" if subfolder else "model.safetensors.index.json"
                if target_index in files:
                    is_sharded = True
                    index_path = hf_hub_download(repo_id=path_or_repo_str, subfolder=subfolder, filename="model.safetensors.index.json")
            except Exception as e:
                print(f"Failed to fetch repo info: {e}")
                is_sharded = False

        # 2. Build files_to_load mapping: file_name -> list of keys (or None for all)
        files_to_load = {}
        if is_sharded:
            with open(index_path, "r") as f:
                index_data = json.load(f)
            weight_map = index_data.get("weight_map", {})
            for k_str, file_name in weight_map.items():
                if file_name not in files_to_load:
                    files_to_load[file_name] = []
                files_to_load[file_name].append(k_str)
        else:
            files_to_load["model.safetensors"] = None

        # 3. Resolve every checkpoint file before materializing any parameters.
        resolved_files = {}
        for file_name in files_to_load:
            if local:
                resolved_files[file_name] = os.path.join(
                    path_or_repo_str,
                    subfolder if subfolder else "",
                    file_name,
                )
            else:
                resolved_files[file_name] = hf_hub_download(
                    repo_id=path_or_repo_str,
                    subfolder=subfolder,
                    filename=file_name,
                )

        # 4. Instantiate model skeleton using eval_shape (no memory allocation)
        rngs = kwargs.pop('rngs', Rngs(0))
        state = jax.eval_shape(
            lambda: cls(
                config,
                rngs=rngs,
                mesh=mesh,
                sharding_rules=sharding_rules,
                **kwargs,
            )
        )
        current_state_dict = state.flat_parameter_dict()
        new_state = {}
        not_found_some = False

        # 5. Load weights
        import re
        import numpy as np
        from safetensors import safe_open
        from ..utils.quantization import (
            quantize_embedding_weight,
            quantize_linear_weight,
            resolve_quantization_rule,
        )
        from ..utils.weights import map_state_dict

        cpu_device = jax.devices('cpu')[0]

        def parameter_sharding(
            parameter,
            axis_names=None,
            *,
            use_explicit=True,
        ):
            sharding = (
                getattr(parameter, 'sharding', None)
                if use_explicit
                else None
            )
            if (
                sharding is None
                and axis_names is not None
                and mesh is not None
            ):
                from ..utils.sharding import create_sharding

                sharding = create_sharding(
                    mesh,
                    axis_names,
                    rules=sharding_rules,
                )
            return sharding

        def place_qarray(value, parameter):
            axis_names = getattr(parameter, 'axis_names', None)
            qvalue_sharding = parameter_sharding(parameter, axis_names)

            scale_axis_names = None
            if axis_names is not None:
                scale_axis_names = tuple(
                    axis_name if size != 1 else None
                    for axis_name, size in zip(
                        axis_names,
                        value.scale.shape,
                    )
                )
            scale_sharding = parameter_sharding(
                parameter,
                scale_axis_names,
                use_explicit=False,
            )

            zero_point = value.zero_point
            if zero_point is not None:
                zero_axis_names = None
                if axis_names is not None:
                    zero_axis_names = tuple(
                        axis_name if size != 1 else None
                        for axis_name, size in zip(
                            axis_names,
                            zero_point.shape,
                        )
                    )
                zero_point = jax.device_put(
                    zero_point,
                    parameter_sharding(
                        parameter,
                        zero_axis_names,
                        use_explicit=False,
                    ),
                )

            return value.replace(
                qvalue=jax.device_put(value.qvalue, qvalue_sharding),
                scale=jax.device_put(value.scale, scale_sharding),
                zero_point=zero_point,
            )

        def materialize_parameter(key, value, parameter):
            quantization = getattr(parameter, 'quantization', None)
            quantization_kind = getattr(
                parameter,
                'quantization_kind',
                'linear',
            )
            rule = resolve_quantization_rule(
                quantization,
                key.rsplit('.', 1)[0],
                op_name=quantization_kind,
            )
            if rule is not None:
                parameter.trainable = False
                with jax.default_device(cpu_device):
                    if quantization_kind == 'embedding':
                        quantized = quantize_embedding_weight(
                            value,
                            parameter,
                            rule,
                        )
                    else:
                        quantized = quantize_linear_weight(
                            value,
                            parameter,
                            rule,
                        )
                return place_qarray(quantized, parameter)

            target_dtype = np.dtype(parameter.dtype)
            if value.dtype != target_dtype:
                value = value.astype(target_dtype, copy=False)
            return jax.device_put(
                value,
                parameter_sharding(
                    parameter,
                    getattr(parameter, 'axis_names', None),
                ),
            )

        stacked_states = {}
        grouped_mapping = any(
            len(rule) == 3
            and isinstance(rule[0], (list, tuple))
            and len(rule[0]) > 1
            for rule in module_map
        )

        for file_name, keys_in_file in files_to_load.items():
            shard_path = resolved_files[file_name]
            with safe_open(shard_path, framework="np", device="cpu") as f:
                keys_to_process = keys_in_file if keys_in_file is not None else f.keys()

                if grouped_mapping:
                    shard = {key: f.get_tensor(key) for key in keys_to_process}
                    mapped_items = map_state_dict(shard, module_map).items()
                else:
                    mapped_items = (
                        item
                        for key in keys_to_process
                        for item in map_state_dict(
                            {key: f.get_tensor(key)},
                            module_map,
                        ).items()
                    )

                for k_mapped, value in mapped_items:
                    if k_mapped in current_state_dict:
                        target_var = current_state_dict[k_mapped]
                        
                        if value.ndim == 2:
                            if k_mapped.endswith(".weight") or ".lora_" in k_mapped:
                                value = value.T
                        if value.shape != target_var.shape:
                            value = value.reshape(target_var.shape)
                        new_state[k_mapped] = materialize_parameter(
                            k_mapped,
                            value,
                            target_var,
                        )

                    else:
                        # Check if it belongs to a SeqStack
                        match = re.search(r'\.(\d+)\.', k_mapped)
                        if match:
                            idx = int(match.group(1))
                            k_stacked = k_mapped[:match.start()] + '.stacked.' + k_mapped[match.end():]
                            
                            if k_stacked in current_state_dict:
                                target_var = current_state_dict[k_stacked]
                                
                                layer_shape = target_var.shape[1:]
                                if value.ndim == 2:
                                    if k_mapped.endswith(".weight") or ".lora_" in k_mapped:
                                        value = value.T
                                if value.shape != layer_shape:
                                    value = value.reshape(layer_shape)
                                    
                                if k_stacked not in stacked_states:
                                    stacked_states[k_stacked] = np.zeros(target_var.shape, dtype=value.dtype)
                                stacked_states[k_stacked][idx] = value
                                continue
                                
                        not_found_some = True
                        print(f"Warning: mapped key {k_mapped} found in checkpoint but not in model.")

        # Move accumulated SeqStack weights to JAX
        for k_stacked, stacked_array in stacked_states.items():
            target_var = current_state_dict[k_stacked]
            new_state[k_stacked] = materialize_parameter(
                k_stacked,
                stacked_array,
                target_var,
            )

        if not_found_some:
            print("\nSome modules from the checkpoint were not found in this model.")
            print("You can try to map module names using module_map.")
            print("e.g. module_map = {'target_module': 'name_to_change'}")

        missing_parameters = sorted(set(current_state_dict) - set(new_state))
        if missing_parameters:
            preview = ', '.join(missing_parameters[:8])
            if len(missing_parameters) > 8:
                preview += f', ... ({len(missing_parameters)} total)'
            raise ValueError(
                'Checkpoint did not provide values for model parameters: '
                f'{preview}'
            )

        # 6. Inject actual arrays into the PyTree skeleton
        state.load_flat_state_dict(new_state)
        return state
    
__all__ = ['PretrainedModel']