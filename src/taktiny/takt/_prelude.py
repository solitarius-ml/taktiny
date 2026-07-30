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
"""Public model-transformation API."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from taktiny.nn.module import Module, iter_children


def _replace_child(parent, name, child):
    if name.isdigit() and hasattr(parent, 'layers'):
        position = int(name)
        if isinstance(parent.layers, tuple):
            updated = list(parent.layers)
            updated[position] = child
            parent.layers = tuple(updated)
        else:
            parent.layers[position] = child
        return

    if '.' in name:
        attribute, index = name.rsplit('.', 1)
        sequence = getattr(parent, attribute)
        position = int(index)
        if isinstance(sequence, tuple):
            updated = list(sequence)
            updated[position] = child
            setattr(parent, attribute, tuple(updated))
        else:
            sequence[position] = child
        return

    setattr(parent, name, child)


class Takt:
    """Apply registered transformations to existing model instances.

    ``Takt`` complements ``Maestro``: Maestro constructs and loads a model,
    while Takt transforms a model that already exists. PEFT implementations
    are selected by configuration type so new methods can be added without
    changing the public ``apply_peft`` signature.
    """

    _peft_methods: dict[type, Callable[[Any, Any], Any]] = {}
    _peft_mergers: dict[type, Callable[..., Any]] = {}

    @classmethod
    def register_peft(cls, config_type: type):
        """Register an implementation for a PEFT configuration type.

        Args:
            config_type: Configuration class used to select the implementation.

        Returns:
            A decorator that registers a ``(model, config)`` callable.

        Raises:
            ValueError: If the configuration type already has a different
                registered implementation.
        """

        def decorator(implementation):
            registered = cls._peft_methods.get(config_type)
            if registered is not None and registered is not implementation:
                raise ValueError(
                    f'{config_type.__name__} already has a registered '
                    'PEFT implementation'
                )
            cls._peft_methods[config_type] = implementation
            return implementation

        return decorator

    @classmethod
    def register_peft_merger(cls, module_type: type):
        """Register the merge implementation for a PEFT wrapper module."""

        def decorator(implementation):
            registered = cls._peft_mergers.get(module_type)
            if registered is not None and registered is not implementation:
                raise ValueError(
                    f'{module_type.__name__} already has a registered '
                    'PEFT merger'
                )
            cls._peft_mergers[module_type] = implementation
            return implementation

        return decorator

    @classmethod
    def apply_peft(cls, model, config):
        """Apply a PEFT configuration to a model in place.

        The registered implementation may replace modules inside ``model``.
        The same model instance is returned for convenient assignment.

        Args:
            model: Existing model instance to transform.
            config: Registered PEFT configuration instance.

        Returns:
            The transformed model.

        Raises:
            NotImplementedError: If no implementation is registered for the
                supplied configuration type.
        """
        implementation = cls._peft_methods.get(type(config))
        if implementation is None:
            implementation = next(
                (
                    candidate
                    for config_type, candidate in cls._peft_methods.items()
                    if isinstance(config, config_type)
                ),
                None,
            )
        if implementation is None:
            raise NotImplementedError(
                'Unsupported PEFT configuration: '
                f'{type(config).__name__}'
            )
        return implementation(model, config)

    @classmethod
    def merge_peft(cls, model, *, dtype=None, quant=None):
        """Merge PEFT adapter weights into their base modules in place.

        Adapter calculations are performed in float32. ``dtype`` controls the
        merged dense-weight dtype, while ``quant`` optionally requantizes the
        merged weights using Taktiny's Qwix quantization rules.

        Args:
            model: Existing Taktiny model containing PEFT wrapper modules.
            dtype: Optional floating-point dtype for merged dense weights.
                Dense base weights retain their dtype when omitted. Quantized
                weights use their dequantized dtype.
            quant: Optional Qwix quantization rule, provider, or qtype string
                applied after merging.

        Returns:
            The same model instance with adapters merged and removed.

        Raises:
            TypeError: If ``model`` is not a Taktiny module.
            ValueError: If no registered mergeable PEFT modules are found.
        """
        if not isinstance(model, Module):
            raise TypeError(
                'PEFT merging currently requires a Taktiny nn.Module model'
            )

        merged = []

        def merger_for(module):
            implementation = cls._peft_mergers.get(type(module))
            if implementation is not None:
                return implementation
            return next(
                (
                    candidate
                    for module_type, candidate in cls._peft_mergers.items()
                    if isinstance(module, module_type)
                ),
                None,
            )

        def transform(module, prefix=''):
            for name, child in list(iter_children(module)):
                full_name = f'{prefix}.{name}' if prefix else name
                implementation = merger_for(child)
                if implementation is not None:
                    replacement = implementation(
                        child,
                        dtype=dtype,
                        quant=quant,
                        module_path=full_name,
                    )
                    _replace_child(module, name, replacement)
                    merged.append(full_name)
                elif isinstance(child, Module):
                    transform(child, full_name)

        transform(model)
        if not merged:
            raise ValueError('No mergeable PEFT modules were found in model')

        trainable_state = getattr(
            model,
            '_peft_trainable_state',
            None,
        )
        if trainable_state is not None:
            for name, parameter in model.flat_parameter_dict().items():
                if name in trainable_state:
                    parameter.trainable = trainable_state[name]
            delattr(model, '_peft_trainable_state')

        if hasattr(model, 'peft_config'):
            delattr(model, 'peft_config')
        return model


__all__ = ['Takt']
