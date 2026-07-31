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
"""Common policy-runtime lifecycle for reinforcement-learning trainers."""

from __future__ import annotations

from dataclasses import dataclass, field
import pickle
from typing import Any, Mapping, Protocol, runtime_checkable

from taktiny.trainer.trainer import Trainer


@runtime_checkable
class PolicyRuntime(Protocol):
    """Inference runtime synchronized from a trainable policy model."""

    model: Any

    @property
    def policy_version(self) -> int:
        """Return the version currently served by the runtime."""

    def generate(self, *args, **kwargs):
        """Generate samples with the currently synchronized policy."""

    def sync(
        self,
        *,
        policy_version: int | None = None,
        **kwargs,
    ) -> int:
        """Publish model weights and return the served policy version."""


@dataclass(frozen=True)
class Rollout:
    """A generated response tied to the exact policy that produced it.

    Token fields intentionally accept arrays or Python sequences so rollout
    storage remains independent of a particular accelerator or data loader.
    Algorithm-specific values such as rewards and advantages belong in
    ``metadata`` or in a more specialized rollout type.
    """

    prompt_ids: Any
    output_ids: Any
    policy_version: int
    logprobs: Any = None
    finish_reason: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if (
            isinstance(self.policy_version, bool)
            or not isinstance(self.policy_version, int)
            or self.policy_version < 0
        ):
            raise ValueError(
                'policy_version must be a non-negative integer'
            )
        if (
            self.finish_reason is not None
            and not isinstance(self.finish_reason, str)
        ):
            raise TypeError('finish_reason must be a string or None')
        if not isinstance(self.metadata, Mapping):
            raise TypeError('metadata must be a mapping')


def _runtime_methods(runtime):
    return (
        getattr(runtime, 'model', None),
        getattr(runtime, 'generate', None),
        getattr(runtime, 'sync', None),
    )


def _is_policy_runtime(runtime):
    model, generate, sync = _runtime_methods(runtime)
    return (
        model is not None
        and callable(generate)
        and callable(sync)
    )


def _validate_policy_version(value, *, name='policy_version'):
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < 0
    ):
        raise ValueError(f'{name} must be a non-negative integer')
    return value


class _RolloutDataLoader:
    """Lazily turn prompt batches into optimizer-ready RL batches."""

    def __init__(self, trainer, source):
        self.trainer = trainer
        self.source = source

    def __iter__(self):
        return _RolloutIterator(self.trainer, iter(self.source))

    def __len__(self):
        try:
            return len(self.source)
        except (AttributeError, TypeError) as error:
            raise TypeError('rollout source has no length') from error

    def set_epoch(self, epoch):
        return self.trainer._set_dataloader_epoch(self.source, epoch)


class _RolloutIterator:
    """Stateful iterator that prevents rollouts from being prefetched."""

    def __init__(self, trainer, source):
        self.trainer = trainer
        self.source = source
        self.position = 0

    def __iter__(self):
        return self

    def __next__(self):
        prompt_batch = next(self.source)
        self.position += 1
        return self.trainer._prepare_rollout_batch(prompt_batch)

    def get_state(self):
        get_state = getattr(self.source, 'get_state', None)
        source_state = get_state() if callable(get_state) else None
        return pickle.dumps({
            'position': self.position,
            'source_state': source_state,
        })

    def set_state(self, state):
        try:
            state = pickle.loads(state)
        except Exception as error:
            raise ValueError('Rollout dataloader state is invalid') from error
        if not isinstance(state, dict):
            raise ValueError('Rollout dataloader state is invalid')
        position = state.get('position')
        if (
            isinstance(position, bool)
            or not isinstance(position, int)
            or position < 0
        ):
            raise ValueError(
                'Rollout dataloader state has an invalid position'
            )

        source_state = state.get('source_state')
        set_state = getattr(self.source, 'set_state', None)
        if source_state is not None and callable(set_state):
            set_state(source_state)
        else:
            for _ in range(position):
                next(self.source)
        self.position = position

    def close(self):
        close = getattr(self.source, 'close', None)
        if callable(close):
            close()


class RLBaseTrainer(Trainer):
    """Base trainer for offline and online reinforcement learning.

    ``model`` may be either a trainable model or a :class:`PolicyRuntime`.
    Runtime wrappers are unwrapped before the normal :class:`Trainer`
    initialization, keeping parameter traversal, optimization, sharding, and
    checkpointing entirely on the original TakTiny model.

    For online RL, :meth:`train` lazily transforms each prompt batch into
    rollouts, rewards, and an optimizer-ready training batch. Successful
    optimizer updates mark the policy dirty. The runtime is synchronized before
    the next rollout batch and once after the final update, making rollout
    batches the policy synchronization boundary.

    Args:
        model: Trainable model or runtime wrapping the trainable model.
        loss_fn: Loss function accepted by :class:`Trainer`.
        training_config: Generic training configuration.
        dataset_config: Generic dataset configuration.
        runtime: Optional runtime supplied separately from ``model``. Its
            ``model`` attribute must reference the same model instance.
        callbacks: Optional Trainer callbacks.
        compute_metrics: Optional evaluation metric function.
    """

    def __init__(
        self,
        model,
        loss_fn,
        training_config,
        dataset_config,
        *,
        runtime: PolicyRuntime | None = None,
        callbacks=None,
        compute_metrics=None,
    ):
        inferred_runtime = model if _is_policy_runtime(model) else None
        if runtime is not None and not _is_policy_runtime(runtime):
            raise TypeError(
                'runtime must expose model, generate(), and sync()'
            )
        if (
            runtime is not None
            and inferred_runtime is not None
            and runtime is not inferred_runtime
        ):
            raise ValueError(
                'model and runtime refer to different policy runtimes'
            )

        if runtime is None:
            runtime = inferred_runtime
        if runtime is not None:
            policy_model = runtime.model
            if inferred_runtime is None and policy_model is not model:
                raise ValueError(
                    'runtime.model must be the same object as model'
                )
        else:
            policy_model = model

        self.runtime = runtime
        runtime_version = (
            getattr(runtime, 'policy_version', 0)
            if runtime is not None
            else 0
        )
        self.policy_version = _validate_policy_version(
            runtime_version,
            name='runtime.policy_version',
        )
        # A runtime may have loaded the original checkpoint rather than the
        # current in-memory model (for example, after applying an adapter).
        self._policy_dirty = runtime is not None

        super().__init__(
            policy_model,
            loss_fn,
            training_config,
            dataset_config,
            callbacks=callbacks,
            compute_metrics=compute_metrics,
        )
        if self.runtime is not None:
            self._prompt_dataloader = self._train_dataloader
            self._train_dataloader = _RolloutDataLoader(
                self,
                self._prompt_dataloader,
            )
        else:
            self._prompt_dataloader = None

    @property
    def has_runtime(self):
        """Whether online generation is backed by a policy runtime."""
        return self.runtime is not None

    @property
    def policy_dirty(self):
        """Whether model updates have not yet been published to the runtime."""
        return self._policy_dirty

    def _require_runtime(self):
        if self.runtime is None:
            raise RuntimeError(
                'A policy runtime is required for online rollout generation'
            )
        return self.runtime

    def _after_optimizer_step(self, params, logs):
        del logs
        self._inject_params(params)
        self.mark_policy_updated()

    def mark_policy_updated(self):
        """Mark the in-memory policy as newer than the inference runtime."""
        if self.runtime is not None:
            self._policy_dirty = True

    def sync_policy(self, *, force=False, **kwargs):
        """Synchronize a dirty policy and return its served version.

        ``force=True`` publishes a new version even when no optimizer update
        has occurred. The dirty flag and policy version change only after the
        runtime reports a successful synchronization.
        """
        runtime = self._require_runtime()
        if not isinstance(force, bool):
            raise TypeError('force must be a boolean')
        if not force and not self._policy_dirty:
            return self.policy_version

        next_version = self.policy_version + 1
        synchronized_version = runtime.sync(
            policy_version=next_version,
            **kwargs,
        )
        if synchronized_version is None:
            synchronized_version = next_version
        synchronized_version = _validate_policy_version(
            synchronized_version,
            name='synchronized policy version',
        )
        if synchronized_version != next_version:
            raise RuntimeError(
                'Policy runtime synchronized an unexpected version: '
                f'expected {next_version}, got {synchronized_version}'
            )

        self.policy_version = synchronized_version
        self._policy_dirty = False
        return synchronized_version

    def generate_rollouts(
        self,
        prompt_batch,
    ):
        """Generate backend output for one prompt batch.

        Concrete algorithms may override this hook to select prompt fields,
        sampling arguments, or multiple completions. Synchronization is handled
        by the surrounding :meth:`train` pipeline.
        """
        runtime = self._require_runtime()
        return runtime.generate(prompt_batch)

    def build_rollouts(self, prompt_batch, generated):
        """Convert runtime output into one or more :class:`Rollout` objects."""
        raise NotImplementedError(
            'RLBaseTrainer subclasses must implement build_rollouts'
        )

    def create_rollout(
        self,
        *,
        prompt_ids,
        output_ids,
        logprobs=None,
        finish_reason=None,
        metadata=None,
    ):
        """Create a rollout stamped with the currently served policy."""
        self._require_runtime()
        if self._policy_dirty:
            raise RuntimeError(
                'Cannot create a rollout while the policy runtime is stale'
            )
        return Rollout(
            prompt_ids=prompt_ids,
            output_ids=output_ids,
            logprobs=logprobs,
            finish_reason=finish_reason,
            policy_version=self.policy_version,
            metadata={} if metadata is None else metadata,
        )

    def validate_rollouts(self, rollouts):
        """Reject rollout records produced by another policy version."""
        if isinstance(rollouts, Rollout):
            rollouts = (rollouts,)
        try:
            rollouts = tuple(rollouts)
        except TypeError as error:
            raise TypeError(
                'rollouts must be a Rollout or iterable of Rollout objects'
            ) from error

        for rollout in rollouts:
            if not isinstance(rollout, Rollout):
                raise TypeError(
                    'rollouts must contain only Rollout objects'
                )
            if rollout.policy_version != self.policy_version:
                raise ValueError(
                    'Stale rollout policy version: '
                    f'expected {self.policy_version}, '
                    f'got {rollout.policy_version}'
                )
        return rollouts

    def compute_rewards(self, rollouts):
        """Compute algorithm-specific rewards for rollout records."""
        raise NotImplementedError(
            'RLBaseTrainer subclasses must implement compute_rewards'
        )

    def prepare_training_batch(self, rollouts, rewards):
        """Build the batch consumed by ``loss_fn`` from rewarded rollouts."""
        raise NotImplementedError(
            'RLBaseTrainer subclasses must implement prepare_training_batch'
        )

    def _prepare_rollout_batch(self, prompt_batch):
        self.sync_policy()
        generated = self.generate_rollouts(prompt_batch)
        rollouts = self.build_rollouts(prompt_batch, generated)
        rollouts = self.validate_rollouts(rollouts)
        rewards = self.compute_rewards(rollouts)
        return self.prepare_training_batch(rollouts, rewards)

    def _before_train_end(self):
        if self.runtime is not None and self._policy_dirty:
            self.sync_policy()

    def _trainer_state(self, *, step, epoch, step_in_epoch):
        state = super()._trainer_state(
            step=step,
            epoch=epoch,
            step_in_epoch=step_in_epoch,
        )
        state['policy_version'] = self.policy_version
        return state

    def _load_resume_state(self, checkpoint_path):
        state = super()._load_resume_state(checkpoint_path)
        self.policy_version = _validate_policy_version(
            state.get('policy_version', 0),
        )
        # A freshly constructed runtime must receive the restored weights,
        # even when the old process had synchronized before checkpointing.
        self._policy_dirty = self.runtime is not None
        return state


__all__ = [
    'PolicyRuntime',
    'RLBaseTrainer',
    'Rollout',
]
