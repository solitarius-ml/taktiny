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
"""
Public API for retrieving registered architectures from the Hugging Face Hub
if implemented in this library.
"""

from __future__ import annotations

from taktiny.maestro import opus
from taktiny.maestro._config import ModelConfig
from taktiny.maestro._livret import repertoire
from taktiny.nn import Rngs

import jax
from jax.sharding import Mesh
from jax.experimental import mesh_utils
from huggingface_hub import hf_hub_download
import json
from pprint import pprint


class Maestro:
    @classmethod
    def list(cls):
        return repertoire.available_classes()
    
    @classmethod
    def available(cls):
        return repertoire.available()
    
    @classmethod
    def supported(cls, model_class: str):
        return True if model_class in repertoire.available() else False
    
    @classmethod
    def from_pretrained(
        cls, 
        repo_or_path, 
        mesh=None, 
        sharding_rules=None, 
        local=False,
        **kwargs
    ):
        try:
            config_path = hf_hub_download(repo_or_path, 'config.json')
            with open(config_path, 'r') as config_file:
                config_dict = json.load(config_file)

        except Exception as e:
            print(f'{e}')
            exit(0)

        keys = config_dict.get('architectures', [])
        
        assert len(keys) == 1, \
            'Unsupported architectures.'

        key = keys[0]
        if key not in repertoire.available():
            raise NotImplementedError("Unsupported architectures.")
            
        model_cls = repertoire.get(key)
        
        # Parse Mesh if provided as a dict (e.g. {'data': 4, 'model': 2})
        if isinstance(mesh, dict):
            axis_names = tuple(mesh.keys())
            shape = tuple(mesh.values())
            devices = mesh_utils.create_device_mesh(shape)
            mesh = Mesh(devices, axis_names)
            
        if sharding_rules is None and hasattr(model_cls, 'default_sharding_rules'):
            sharding_rules = model_cls.default_sharding_rules
            
        return model_cls.from_pretrained(
            repo_or_path, 
            mesh=mesh, 
            sharding_rules=sharding_rules, 
            local=local,
            **kwargs
        )

    @classmethod
    def eval_shape(
        cls,
        repo_or_path,
        mesh=None,
        sharding_rules=None,
        local=False,
        **kwargs,
    ):
        config = kwargs.pop('config', None)
        if config is None:
            config = ModelConfig.load_config(repo_or_path, local=local)
        if config is None:
            raise ValueError(f'Unable to load config from {repo_or_path}')

        architectures = getattr(config, 'architectures', None) or []
        if len(architectures) != 1:
            raise ValueError(
                'Expected config.architectures to contain exactly one architecture'
            )

        architecture = architectures[0]
        if not repertoire.get(architecture):
            raise NotImplementedError(
                f'Unsupported architecture: {architecture}'
            )
        model_cls = repertoire.get(architecture)

        if isinstance(mesh, dict):
            axis_names = tuple(mesh)
            shape = tuple(mesh.values())
            devices = mesh_utils.create_device_mesh(shape)
            mesh = Mesh(devices, axis_names)

        if sharding_rules is None and hasattr(model_cls, 'default_sharding_rules'):
            sharding_rules = model_cls.default_sharding_rules

        rngs = kwargs.pop('rngs', None)
        if rngs is None:
            rngs = Rngs(0)

        return jax.eval_shape(
            lambda: model_cls(
                config,
                rngs=rngs,
                mesh=mesh,
                sharding_rules=sharding_rules,
                **kwargs,
            )
        )


__all__ = ['Maestro']
