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


class Takt:
    """Apply registered transformations to existing model instances.

    ``Takt`` complements ``Maestro``: Maestro constructs and loads a model,
    while Takt transforms a model that already exists. PEFT implementations
    are selected by configuration type so new methods can be added without
    changing the public ``apply_peft`` signature.
    """

    _peft_methods: dict[type, Callable[[Any, Any], Any]] = {}

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


__all__ = ['Takt']
