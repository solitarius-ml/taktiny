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

from collections import deque
from itertools import islice
import json
import math
import os
import re
import shutil

import jax
import qwix
from taktiny.trainer.config import TrainingConfig, DatasetConfig
from taktiny.nn.module import Module, Parameter

import jax.numpy as jnp
import optax

def _is_trainable_value(value):
    if isinstance(value, qwix.QArray):
        return False
    if not hasattr(value, 'dtype'):
        return False
    if not jnp.issubdtype(value.dtype, jnp.inexact):
        return False
    return value.dtype != getattr(jnp, 'float8_e4m3fn', None)


def _parameter_labels(params):
    """Build Optax labels from explicit Taktiny parameter metadata."""
    if not isinstance(params, Module):
        return jax.tree.map(
            lambda value: (
                'trainable'
                if _is_trainable_value(value)
                else 'frozen'
            ),
            params,
        )

    def label_parameter(parameter):
        trainable = (
            getattr(parameter, 'trainable', True)
            and _is_trainable_value(parameter.value)
        )
        label = 'trainable' if trainable else 'frozen'
        return jax.tree.map(lambda _: label, parameter)

    return jax.tree.map(
        label_parameter,
        params,
        is_leaf=lambda value: isinstance(value, Parameter),
    )


def _partition_params(params, labels):
    trainable = jax.tree.map(
        lambda value, label: (
            value if label == 'trainable' else None
        ),
        params,
        labels,
    )
    frozen = jax.tree.map(
        lambda value, label: (
            None if label == 'trainable' else value
        ),
        params,
        labels,
    )
    return trainable, frozen


def _combine_params(trainable, frozen):
    return jax.tree.map(
        lambda trainable_value, frozen_value: (
            frozen_value
            if trainable_value is None
            else trainable_value
        ),
        trainable,
        frozen,
        is_leaf=lambda value: value is None,
    )


def _tree_shardings(tree):
    return jax.tree.map(
        lambda value: (
            value.sharding if isinstance(value, jax.Array) else None
        ),
        tree,
    )


def _parameter_mesh(params):
    for value in jax.tree.leaves(params):
        sharding = getattr(value, 'sharding', None)
        if isinstance(sharding, jax.sharding.NamedSharding):
            return sharding.mesh
    return None


def _sharding_mesh(sharding):
    for value in jax.tree.leaves(sharding):
        if isinstance(value, jax.sharding.NamedSharding):
            return value.mesh
    return None


def _uses_multiple_devices(tree):
    for value in jax.tree.leaves(tree):
        sharding = getattr(value, 'sharding', None)
        if sharding is not None and len(sharding.device_set) > 1:
            return True
    return False


def _validate_parameter_placement(params, batch_mesh):
    parameter_mesh = _parameter_mesh(params)
    if (
        parameter_mesh is not None
        and batch_mesh is not None
        and parameter_mesh != batch_mesh
    ):
        raise ValueError(
            'Model parameters and batches must use the same device mesh.'
        )
    if batch_mesh is None or batch_mesh.size <= 1:
        return
    if parameter_mesh is not None or _uses_multiple_devices(params):
        return
    raise ValueError(
        'Multi-device batch sharding requires pre-sharded model parameters. '
        'Load the model with the same mesh and parameter sharding rules; '
        'Trainer will not replicate an unsharded model automatically.'
    )


def _place_trainable_params(tree, mesh):
    if mesh is None or mesh.size <= 1:
        return tree

    replicated = jax.sharding.NamedSharding(
        mesh,
        jax.sharding.PartitionSpec(),
    )

    def place(value):
        if not isinstance(value, jax.Array):
            return value
        if isinstance(value.sharding, jax.sharding.NamedSharding):
            return value
        return jax.device_put(value, replicated)

    return jax.tree.map(place, tree)


def _place_optimizer_state(tree, mesh):
    if mesh is None or mesh.size <= 1:
        return tree

    replicated = jax.sharding.NamedSharding(
        mesh,
        jax.sharding.PartitionSpec(),
    )

    def place(value):
        if not isinstance(value, jax.Array):
            return value
        if isinstance(value.sharding, jax.sharding.NamedSharding):
            return value
        if value.ndim == 0:
            return jax.device_put(value, replicated)
        raise ValueError(
            'A non-scalar optimizer state did not inherit parameter sharding. '
            'Provide an optimizer whose parameter-shaped state preserves '
            'input shardings.'
        )

    return jax.tree.map(place, tree)


def _prefetch(iterable, place, size):
    """Place a bounded number of batches ahead of consumption."""
    iterator = iter(iterable)
    if size == 0:
        for value in iterator:
            yield place(value)
        return

    queue = deque()
    for _ in range(size):
        try:
            queue.append(place(next(iterator)))
        except StopIteration:
            break

    while queue:
        yield queue.popleft()
        try:
            queue.append(place(next(iterator)))
        except StopIteration:
            pass


def _format_iteration_time(seconds):
    if seconds < 1:
        return f'{seconds * 1000:.1f} ms/it'
    if seconds < 60:
        return f'{seconds:.1f} s/it'
    return f'{seconds / 60:.1f} min/it'


class Trainer:
    def __init__(self, model, loss_fn, training_config: TrainingConfig, dataset_config: DatasetConfig):
        self.model = model
        self.loss_fn = loss_fn
        self.training_config = training_config
        self.dataset_config = dataset_config
        self.model_type = self._diagnose_model_type(model)
        self._mesh = None
        self.global_step = 0
        self.saved_checkpoints = []
        self.log_history = []
        self.best_metric = None
        self.best_model_checkpoint = None
        self._best_step = None
        self._compiled_eval_step = None
        self.loss_scale = self._initial_loss_scale()
        self.loss_scale_good_steps = 0
        self.skipped_updates = 0
        self.micro_step = 0
        self.last_grad_norm = None
        self.last_update_skipped = False

    def _initial_loss_scale(self):
        loss_scale = self.training_config.loss_scale
        if loss_scale == 'dynamic':
            return float(self.training_config.initial_loss_scale)
        if loss_scale is None:
            return 1.0
        return float(loss_scale)
        
    def _diagnose_model_type(self, model) -> str:
        # Detect Taktiny models
        if isinstance(model, Module):
            return "taktiny"
            
        # Detect Flax NNX models
        if hasattr(model, "__module__") and "flax.nnx" in model.__module__:
            return "nnx"
            
        # Detect classic Flax Linen models
        if hasattr(model, "__module__") and "flax.linen" in model.__module__:
            return "flax_linen"
            
        # Detect Equinox models
        if hasattr(model, "__module__") and "equinox" in model.__module__:
            return "equinox"
            
        return "unknown"
        
    def extract_params(self):
        """Extract params based on the diagnosed model type."""
        if self.model_type == "taktiny":
            # Taktiny models are fully registered PyTrees
            return self.model
        elif self.model_type == "nnx":
            from flax import nnx
            _, params = nnx.split(self.model)
            return params
        elif self.model_type == "flax_linen":
            # Assume self.model is a dict of params for Flax Linen in this simplified design
            # (In reality, Flax Trainer would need model.init or params passed in)
            return self.model
        elif self.model_type == "equinox":
            import equinox as eqx
            return eqx.filter(self.model, eqx.is_array)
        else:
            raise ValueError("Unsupported model type")
            
    def _setup_optimizer(self, params):
        """Configure an optimizer for the trainable parameter partition."""
        base_opt = self.training_config.optimizer
        if base_opt is None:
            learning_rate = (
                self.training_config.schedule
                if self.training_config.schedule is not None
                else self.training_config.learning_rate
            )
            base_opt = optax.adamw(
                learning_rate,
                weight_decay=self.training_config.weight_decay,
            )
        return base_opt

    def _learning_rate_at_step(self, step):
        """Return the rate used by a completed optimizer update."""
        schedule = self.training_config.schedule
        if schedule is None:
            if self.training_config.optimizer is not None:
                return None
            return float(self.training_config.learning_rate)
        value = schedule(max(0, step - 1))
        return float(jax.device_get(value))

    def _place_batch(self, batch):
        sharding = self.dataset_config.batch_sharding
        if sharding is None:
            if self._mesh is None:
                return jax.tree.map(jax.device_put, batch)
            sharding = jax.sharding.NamedSharding(
                self._mesh,
                jax.sharding.PartitionSpec(),
            )
        if isinstance(sharding, jax.sharding.Sharding):
            return jax.tree.map(
                lambda value: jax.device_put(value, sharding),
                batch,
            )
        return jax.tree.map(
            lambda value, value_sharding: jax.device_put(
                value,
                value_sharding,
            ),
            batch,
            sharding,
        )

    def _evaluate_params(self, params):
        dataloader = self.dataset_config.validation_dataloader
        if dataloader is None:
            raise ValueError(
                'validation_dataloader is required for evaluation'
            )

        losses = []
        batches = _prefetch(
            dataloader,
            self._place_batch,
            self.dataset_config.prefetch_size,
        )
        for batch in batches:
            if (
                self._compiled_eval_step is None
                and self.training_config.jit_compile
            ):
                self._compiled_eval_step = jax.jit(
                    lambda candidate, value: self.loss_fn(
                        candidate,
                        value,
                    ),
                    in_shardings=(
                        _tree_shardings(params),
                        _tree_shardings(batch),
                    ),
                    out_shardings=None,
                )
            eval_step = self._compiled_eval_step or self.loss_fn
            value = eval_step(params, batch)
            if isinstance(value, jax.Array):
                value = value.item()
            losses.append(float(value))
        batches.close()

        if not losses:
            raise ValueError(
                'validation_dataloader produced no evaluation batches'
            )
        return {
            'eval_loss': sum(losses) / len(losses),
        }

    def evaluate(self):
        """Evaluate the current model using ``validation_dataloader``."""
        params = self.extract_params()
        parameter_mesh = _parameter_mesh(params)
        batch_mesh = _sharding_mesh(self.dataset_config.batch_sharding)
        _validate_parameter_placement(params, batch_mesh)
        self._mesh = parameter_mesh or batch_mesh
        return self._evaluate_params(params)

    def _record_evaluation(self, params, *, step, epoch):
        metrics = self._evaluate_params(params)
        record = {
            'step': step,
            'epoch': epoch,
            **metrics,
        }
        self.log_history.append(record)

        metric_name = self.training_config.metric_for_best_model
        if not metric_name.startswith('eval_'):
            metric_name = f'eval_{metric_name}'
        if metric_name not in metrics:
            raise ValueError(
                f'Evaluation did not produce metric {metric_name!r}'
            )
        metric = float(metrics[metric_name])
        greater_is_better = self.training_config.greater_is_better
        if greater_is_better is None:
            greater_is_better = not metric_name.endswith('loss')
        is_best = (
            self.best_metric is None
            or (
                metric > self.best_metric
                if greater_is_better
                else metric < self.best_metric
            )
        )
        if is_best:
            self.best_metric = metric
            self._best_step = step
            self.best_model_checkpoint = None
        return metrics, is_best

    def _checkpoint_directory(self, step):
        return os.path.join(
            os.fspath(self.training_config.output_dir),
            f'checkpoint-{step}',
        )

    def _checkpoint_paths(self):
        output_dir = self.training_config.output_dir
        if output_dir is None or not os.path.isdir(output_dir):
            return []

        checkpoint_pattern = re.compile(r'checkpoint-(\d+)')
        checkpoints = []
        for entry in os.scandir(output_dir):
            match = checkpoint_pattern.fullmatch(entry.name)
            if entry.is_dir() and match is not None:
                checkpoints.append((int(match.group(1)), entry.path))
        checkpoints.sort()
        return checkpoints

    def _rotate_checkpoints(self):
        limit = self.training_config.save_total_limit
        if limit is None:
            return

        checkpoints = self._checkpoint_paths()
        available_paths = {
            checkpoint_path
            for _, checkpoint_path in checkpoints
        }
        retained = set()
        if self.best_model_checkpoint in available_paths:
            retained.add(self.best_model_checkpoint)
        remaining = max(0, limit - len(retained))
        for _, checkpoint_path in reversed(checkpoints):
            if checkpoint_path in retained:
                continue
            if remaining == 0:
                break
            retained.add(checkpoint_path)
            remaining -= 1

        for _, checkpoint_path in checkpoints:
            if checkpoint_path in retained:
                continue
            shutil.rmtree(checkpoint_path)
        self.saved_checkpoints = [
            path
            for path in self.saved_checkpoints
            if path in retained
        ]

    def _resolve_resume_checkpoint(self, checkpoint):
        if checkpoint != 'latest':
            checkpoint_path = os.fspath(checkpoint)
            if not os.path.isdir(checkpoint_path):
                raise FileNotFoundError(
                    f'Resume checkpoint was not found: {checkpoint_path}'
                )
            return checkpoint_path

        if self.training_config.output_dir is None:
            raise ValueError(
                'output_dir is required when resuming from "latest"'
            )
        checkpoints = self._checkpoint_paths()
        if not checkpoints:
            raise FileNotFoundError(
                'No checkpoint-* directories were found in output_dir'
            )
        return checkpoints[-1][1]

    def _load_resume_state(self, checkpoint_path):
        trainer_state_path = os.path.join(
            checkpoint_path,
            'trainer_state.json',
        )
        if not os.path.isfile(trainer_state_path):
            raise FileNotFoundError(
                f'Trainer state was not found: {trainer_state_path}'
            )
        with open(trainer_state_path) as trainer_state_file:
            state = json.load(trainer_state_file)

        for key in ('global_step', 'epoch', 'step_in_epoch'):
            value = state.get(key)
            if not isinstance(value, int) or value < 0:
                raise ValueError(
                    f'trainer_state.json has invalid {key}: {value!r}'
                )
        history = state.get('log_history', [])
        if not isinstance(history, list):
            raise ValueError(
                'trainer_state.json log_history must be a list'
            )
        accumulation_steps = state.get('gradient_accumulation_steps', 1)
        if accumulation_steps != (
            self.training_config.gradient_accumulation_steps
        ):
            raise ValueError(
                'Cannot resume with a different '
                'gradient_accumulation_steps value'
            )
        for key in ('loss_scale_good_steps', 'skipped_updates', 'micro_step'):
            value = state.get(key, 0)
            if not isinstance(value, int) or value < 0:
                raise ValueError(
                    f'trainer_state.json has invalid {key}: {value!r}'
                )
        loss_scale = state.get('loss_scale', self._initial_loss_scale())
        if not isinstance(loss_scale, (int, float)) or loss_scale <= 0:
            raise ValueError(
                'trainer_state.json has invalid loss_scale: '
                f'{loss_scale!r}'
            )
        return state

    def _load_checkpoint_model(self, checkpoint_path):
        adapter_config = os.path.join(
            checkpoint_path,
            'adapter_config.json',
        )
        has_adapter = (
            os.path.isfile(adapter_config)
            and (
                os.path.isfile(os.path.join(
                    checkpoint_path,
                    'adapter_model.safetensors',
                ))
                or os.path.isfile(os.path.join(
                    checkpoint_path,
                    'adapter_model.safetensors.index.json',
                ))
            )
        )
        has_model = (
            os.path.isfile(os.path.join(
                checkpoint_path,
                'model.safetensors',
            ))
            or os.path.isfile(os.path.join(
                checkpoint_path,
                'model.safetensors.index.json',
            ))
        )
        if has_adapter and has_model:
            raise ValueError(
                'Resume checkpoint contains both full-model and adapter '
                'weights'
            )
        if has_adapter:
            from taktiny.takt import Takt

            Takt.load_peft(
                self.model,
                checkpoint_path,
                local=True,
            )
            return
        if not has_model:
            raise FileNotFoundError(
                'Resume checkpoint contains neither model nor adapter '
                'Safetensors'
            )

        load_pretrained = getattr(self.model, 'load_pretrained', None)
        if not callable(load_pretrained):
            raise TypeError(
                f'{type(self.model).__name__} cannot load full model '
                'checkpoints in place'
            )
        load_pretrained(checkpoint_path)

    def _write_trainer_state(
        self,
        checkpoint_path,
        *,
        step,
        epoch,
        step_in_epoch,
    ):
        trainer_state_path = os.path.join(
            checkpoint_path,
            'trainer_state.json',
        )
        with open(trainer_state_path, 'w') as trainer_state_file:
            json.dump(
                {
                    'global_step': step,
                    'epoch': epoch,
                    'step_in_epoch': step_in_epoch,
                    'log_history': self.log_history,
                    'best_metric': self.best_metric,
                    'best_model_checkpoint': self.best_model_checkpoint,
                    'gradient_accumulation_steps': (
                        self.training_config.gradient_accumulation_steps
                    ),
                    'loss_scale': self.loss_scale,
                    'loss_scale_good_steps': self.loss_scale_good_steps,
                    'skipped_updates': self.skipped_updates,
                    'micro_step': self.micro_step,
                },
                trainer_state_file,
                indent=2,
            )

    def _save_checkpoint(
        self,
        step,
        trainable_params,
        frozen_params,
        opt_state,
        *,
        epoch,
        step_in_epoch,
    ):
        save_pretrained = getattr(self.model, 'save_pretrained', None)
        if not callable(save_pretrained):
            raise TypeError(
                f'{type(self.model).__name__} does not support '
                'save_pretrained checkpoints'
            )

        self._inject_params(
            _combine_params(trainable_params, frozen_params)
        )
        checkpoint_path = self._checkpoint_directory(step)
        save_pretrained(
            checkpoint_path,
            max_shard_size=self.training_config.max_shard_size,
        )

        if self.training_config.save_optimizer_state:
            import orbax.checkpoint as ocp

            optimizer_path = os.path.join(
                checkpoint_path,
                'optimizer_state',
            )
            checkpointer = ocp.StandardCheckpointer()
            try:
                checkpointer.save(
                    optimizer_path,
                    opt_state,
                    force=True,
                )
                checkpointer.wait_until_finished()
            finally:
                checkpointer.close()

        self._write_trainer_state(
            checkpoint_path,
            step=step,
            epoch=epoch,
            step_in_epoch=step_in_epoch,
        )

        self.saved_checkpoints.append(checkpoint_path)
        self._rotate_checkpoints()
        return checkpoint_path

    def _ensure_checkpoint(
        self,
        step,
        trainable_params,
        frozen_params,
        opt_state,
        *,
        epoch,
        step_in_epoch,
    ):
        checkpoint_path = self._checkpoint_directory(step)
        if (
            checkpoint_path in self.saved_checkpoints
            and os.path.isdir(checkpoint_path)
        ):
            self._write_trainer_state(
                checkpoint_path,
                step=step,
                epoch=epoch,
                step_in_epoch=step_in_epoch,
            )
            return checkpoint_path
        return self._save_checkpoint(
            step,
            trainable_params,
            frozen_params,
            opt_state,
            epoch=epoch,
            step_in_epoch=step_in_epoch,
        )
            
    def train(self, resume_from_checkpoint=None):
        """Train the configured model, optionally resuming a checkpoint.

        Args:
            resume_from_checkpoint: A ``checkpoint-<step>`` directory or
                ``"latest"`` to select the highest numbered checkpoint in
                ``output_dir``. Resuming restores model or adapter weights,
                optimizer state, trainer history, and the saved epoch and batch
                position. The dataloader must reproduce the same per-epoch
                ordering so consumed batches can be skipped deterministically.
        """
        from rich.console import Console
        from rich.progress import (
            BarColumn,
            Progress,
            TextColumn,
            TimeElapsedColumn,
            TimeRemainingColumn,
        )

        console = Console()
        console.print(
            f'[bold green]Starting training for a '
            f'[cyan]{self.model_type.upper()}[/cyan] model[/bold green]'
        )
        console.print(
            f'Epochs: [bold]{self.training_config.epochs}[/bold] | '
            f'Max Steps: [bold]{self.training_config.max_steps}[/bold]'
        )

        resume_state = None
        resume_checkpoint = None
        if resume_from_checkpoint is not None:
            resume_checkpoint = self._resolve_resume_checkpoint(
                resume_from_checkpoint
            )
            resume_state = self._load_resume_state(resume_checkpoint)
            self._load_checkpoint_model(resume_checkpoint)
            self.global_step = resume_state['global_step']
            self.log_history = list(
                resume_state.get('log_history', [])
            )
            self.best_metric = resume_state.get('best_metric')
            self.best_model_checkpoint = resume_state.get(
                'best_model_checkpoint'
            )
            self.loss_scale = float(
                resume_state.get('loss_scale', self._initial_loss_scale())
            )
            self.loss_scale_good_steps = resume_state.get(
                'loss_scale_good_steps',
                0,
            )
            self.skipped_updates = resume_state.get('skipped_updates', 0)
            self.micro_step = resume_state.get('micro_step', 0)
            if self.best_model_checkpoint is not None:
                match = re.search(
                    r'checkpoint-(\d+)$',
                    self.best_model_checkpoint,
                )
                if match is not None:
                    self._best_step = int(match.group(1))
            self.saved_checkpoints = [
                path
                for _, path in self._checkpoint_paths()
            ]
            console.print(
                f'[dim]Resuming from {resume_checkpoint} at step '
                f'{self.global_step}[/dim]'
            )

        saving_enabled = (
            self.training_config.save_steps is not None
            or self.training_config.save_at_end
            or self.training_config.load_best_model_at_end
        )
        if (
            self.training_config.eval_strategy != 'no'
            and self.dataset_config.validation_dataloader is None
        ):
            raise ValueError(
                'validation_dataloader is required when evaluation is enabled'
            )
        if saving_enabled and not callable(
            getattr(self.model, 'save_pretrained', None)
        ):
            raise TypeError(
                f'{type(self.model).__name__} does not support '
                'save_pretrained checkpoints'
            )
        if saving_enabled:
            os.makedirs(self.training_config.output_dir, exist_ok=True)
        
        # 1. Initialize Optimizer
        if self.training_config.remat:
            enable_remat = getattr(self.model, 'enable_remat', None)
            if not callable(enable_remat):
                raise TypeError(
                    f'{type(self.model).__name__} does not support remat'
                )
            enable_remat()

        params = self.extract_params()
        parameter_mesh = _parameter_mesh(params)
        batch_mesh = _sharding_mesh(self.dataset_config.batch_sharding)
        _validate_parameter_placement(params, batch_mesh)
        self._mesh = parameter_mesh or batch_mesh
        labels = _parameter_labels(params)
        trainable_params, frozen_params = _partition_params(params, labels)
        del labels, params

        trainable_params = _place_trainable_params(
            trainable_params,
            self._mesh,
        )
        if self.model_type == 'taktiny':
            self._inject_params(
                _combine_params(trainable_params, frozen_params)
            )

        optimizer = self._setup_optimizer(trainable_params)
        opt_state = optimizer.init(trainable_params)
        opt_state = _place_optimizer_state(
            opt_state,
            self._mesh,
        )
        if resume_checkpoint is not None:
            import orbax.checkpoint as ocp

            optimizer_path = os.path.join(
                resume_checkpoint,
                'optimizer_state',
            )
            if not os.path.isdir(optimizer_path):
                raise FileNotFoundError(
                    f'Optimizer state was not found: {optimizer_path}'
                )
            checkpointer = ocp.StandardCheckpointer()
            try:
                opt_state = checkpointer.restore(
                    optimizer_path,
                    target=opt_state,
                )
            finally:
                checkpointer.close()

        # 2. Define independently compilable gradient and optimizer phases.
        def calculate_loss(candidate_trainable, current_frozen, batch):
            current_params = _combine_params(
                candidate_trainable,
                current_frozen,
            )
            return self.loss_fn(current_params, batch)

        use_loss_scaling = self.training_config.loss_scale is not None

        def scaled_loss(
            candidate_trainable,
            current_frozen,
            batch,
            current_loss_scale,
        ):
            loss = calculate_loss(
                candidate_trainable,
                current_frozen,
                batch,
            )
            return loss * current_loss_scale, loss

        loss_and_grad = jax.value_and_grad(scaled_loss, has_aux=True)

        def gradient_step(
            current_trainable,
            current_frozen,
            batch,
            current_loss_scale,
        ):
            (_, loss), grads = loss_and_grad(
                current_trainable,
                current_frozen,
                batch,
                current_loss_scale,
            )
            if use_loss_scaling:
                grads = jax.tree.map(
                    lambda grad: (
                        grad.astype(jnp.float32)
                        / current_loss_scale.astype(jnp.float32)
                    ),
                    grads,
                )
            return loss, grads

        def optimizer_step(current_trainable, current_opt_state, grads):
            updates, new_opt_state = optimizer.update(
                grads,
                current_opt_state,
                current_trainable,
            )
            new_trainable = optax.apply_updates(
                current_trainable,
                updates,
            )
            return new_trainable, new_opt_state

        compiled_gradient_step = None
        compiled_optimizer_step = None

        # 3. Training Loop
        import time

        step = self.global_step
        should_stop = (
            self.training_config.max_steps is not None
            and step >= self.training_config.max_steps
        )
        start_time = time.time()
        steps_since_log = 0
        steps_run_this_call = 0
        microbatches_run_this_call = 0
        loss = next(
            (
                record['loss']
                for record in reversed(self.log_history)
                if 'loss' in record
            ),
            None,
        )
        grad_norm = None
        update_skipped = False
        start_epoch = resume_state['epoch'] if resume_state else 0
        resume_step_in_epoch = (
            resume_state['step_in_epoch']
            if resume_state
            else 0
        )
        epoch = start_epoch
        step_in_epoch = resume_step_in_epoch
        accumulation_steps = (
            self.training_config.gradient_accumulation_steps
        )
        accumulated_grads = None
        accumulated_loss = None
        accumulated_microbatches = 0

        # Try to guess total optimizer updates if dataloader has __len__.
        total_steps = None
        if hasattr(self.dataset_config.dataloader, '__len__'):
            updates_per_epoch = math.ceil(
                len(self.dataset_config.dataloader) / accumulation_steps
            )
            total_steps = updates_per_epoch * self.training_config.epochs
        if self.training_config.max_steps is not None:
            total_steps = self.training_config.max_steps

        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            TextColumn("• [bold magenta]Loss: {task.fields[loss]:.4f}[/bold magenta]"),
            console=console
        ) as progress:

            task_id = progress.add_task(
                "[cyan]Training...",
                total=total_steps,
                completed=step,
                loss=float(loss) if loss is not None else 0.0,
            )

            def finish_accumulation(current_epoch, current_step_in_epoch):
                nonlocal accumulated_grads
                nonlocal accumulated_loss
                nonlocal accumulated_microbatches
                nonlocal compiled_optimizer_step
                nonlocal grad_norm
                nonlocal loss
                nonlocal opt_state
                nonlocal should_stop
                nonlocal start_time
                nonlocal step
                nonlocal steps_run_this_call
                nonlocal steps_since_log
                nonlocal trainable_params
                nonlocal update_skipped

                divisor = jnp.asarray(
                    accumulated_microbatches,
                    dtype=jnp.float32,
                )
                averaged_grads = jax.tree.map(
                    lambda value: value / divisor,
                    accumulated_grads,
                )
                averaged_loss = accumulated_loss / divisor
                norm_grads = jax.tree.map(
                    lambda value: value.astype(jnp.float32),
                    averaged_grads,
                )
                current_grad_norm = optax.tree.norm(norm_grads)
                finite = (
                    jnp.isfinite(averaged_loss)
                    & jnp.isfinite(current_grad_norm)
                )

                if self.training_config.max_grad_norm is not None:
                    clip_scale = jnp.minimum(
                        jnp.asarray(1.0, dtype=jnp.float32),
                        (
                            self.training_config.max_grad_norm
                            / (current_grad_norm + 1e-6)
                        ),
                    )
                    averaged_grads = jax.tree.map(
                        lambda value: value * clip_scale.astype(value.dtype),
                        averaged_grads,
                    )

                loss_value, grad_norm_value, finite_value = jax.device_get(
                    (averaged_loss, current_grad_norm, finite)
                )
                loss_value = float(loss_value)
                grad_norm_value = float(grad_norm_value)
                finite_value = bool(finite_value)
                update_skipped = (
                    not finite_value
                    and self.training_config.skip_non_finite
                )

                if not update_skipped:
                    if (
                        compiled_optimizer_step is None
                        and self.training_config.jit_compile
                    ):
                        compiled_optimizer_step = jax.jit(
                            optimizer_step,
                            in_shardings=(
                                _tree_shardings(trainable_params),
                                _tree_shardings(opt_state),
                                _tree_shardings(averaged_grads),
                            ),
                            out_shardings=(
                                _tree_shardings(trainable_params),
                                _tree_shardings(opt_state),
                            ),
                            donate_argnums=(0, 1, 2),
                        )
                    update_fn = (
                        compiled_optimizer_step or optimizer_step
                    )
                    trainable_params, opt_state = update_fn(
                        trainable_params,
                        opt_state,
                        averaged_grads,
                    )
                else:
                    self.skipped_updates += 1

                if self.training_config.loss_scale == 'dynamic':
                    if finite_value:
                        self.loss_scale_good_steps += 1
                        if (
                            self.loss_scale_good_steps
                            >= self.training_config.loss_scale_growth_interval
                        ):
                            self.loss_scale *= 2.0
                            self.loss_scale_good_steps = 0
                    else:
                        self.loss_scale = max(1.0, self.loss_scale / 2.0)
                        self.loss_scale_good_steps = 0

                step += 1
                self.global_step = step
                self.last_grad_norm = (
                    grad_norm_value if math.isfinite(grad_norm_value) else None
                )
                self.last_update_skipped = update_skipped
                loss = loss_value if math.isfinite(loss_value) else None
                grad_norm = self.last_grad_norm
                steps_run_this_call += 1
                steps_since_log += 1
                progress.update(
                    task_id,
                    advance=1,
                    loss=loss_value,
                )

                accumulated_grads = None
                accumulated_loss = None
                accumulated_microbatches = 0

                if step % self.training_config.log_interval == 0:
                    elapsed = time.time() - start_time
                    seconds_per_step = elapsed / max(1, steps_since_log)
                    iteration_time = _format_iteration_time(
                        seconds_per_step
                    )
                    learning_rate = self._learning_rate_at_step(step)
                    self.log_history.append({
                        'step': step,
                        'epoch': current_epoch,
                        'loss': loss,
                        'seconds_per_step': seconds_per_step,
                        'learning_rate': learning_rate,
                        'grad_norm': grad_norm,
                        'loss_scale': self.loss_scale,
                        'skipped_update': update_skipped,
                    })
                    loss_text = (
                        f'{loss:<7.4f}'
                        if loss is not None
                        else 'non-finite'
                    )
                    learning_rate_text = (
                        f' ┃ LR: [bold green]'
                        f'{learning_rate:.3e}[/bold green]'
                        if learning_rate is not None
                        else ''
                    )
                    progress.console.print(
                        f"[bold cyan]Epoch {current_epoch:<3}[/bold cyan] ┃ "
                        f"[bold yellow]Step {step:<6}[/bold yellow] ┃ "
                        f"Loss: [bold magenta]{loss_text}"
                        f"[/bold magenta]{learning_rate_text} ┃ "
                        f"[dim]{iteration_time:>11}[/dim]"
                    )
                    start_time = time.time()
                    steps_since_log = 0

                should_evaluate = (
                    self.training_config.eval_strategy == 'steps'
                    and step % self.training_config.eval_steps == 0
                )
                if should_evaluate:
                    metrics, is_best = self._record_evaluation(
                        _combine_params(
                            trainable_params,
                            frozen_params,
                        ),
                        step=step,
                        epoch=current_epoch,
                    )
                    progress.console.print(
                        f"[bold blue]Evaluation[/bold blue] ┃ "
                        f"[bold yellow]Step {step:<6}[/bold yellow] ┃ "
                        f"Loss: [bold magenta]"
                        f"{metrics['eval_loss']:.4f}[/bold magenta]"
                    )
                    if (
                        is_best
                        and self.training_config.load_best_model_at_end
                    ):
                        self.best_model_checkpoint = (
                            self._checkpoint_directory(step)
                        )
                        self._ensure_checkpoint(
                            step,
                            trainable_params,
                            frozen_params,
                            opt_state,
                            epoch=current_epoch,
                            step_in_epoch=current_step_in_epoch,
                        )

                if (
                    self.training_config.save_steps is not None
                    and step % self.training_config.save_steps == 0
                ):
                    if self._best_step == step:
                        self.best_model_checkpoint = (
                            self._checkpoint_directory(step)
                        )
                    checkpoint_path = self._ensure_checkpoint(
                        step,
                        trainable_params,
                        frozen_params,
                        opt_state,
                        epoch=current_epoch,
                        step_in_epoch=current_step_in_epoch,
                    )
                    progress.console.print(
                        f'[dim]Saved checkpoint to '
                        f'{checkpoint_path}[/dim]'
                    )

                if (
                    self.training_config.max_steps is not None
                    and step >= self.training_config.max_steps
                ):
                    should_stop = True

            for epoch in range(start_epoch, self.training_config.epochs):
                if should_stop:
                    break

                epoch_updates_run = 0
                skip_batches = (
                    resume_step_in_epoch
                    if epoch == start_epoch
                    else 0
                )
                epoch_batches = self.dataset_config.dataloader
                if skip_batches:
                    epoch_batches = islice(
                        epoch_batches,
                        skip_batches,
                        None,
                    )
                batches = _prefetch(
                    epoch_batches,
                    self._place_batch,
                    self.dataset_config.prefetch_size,
                )
                for step_in_epoch, batch in enumerate(
                    batches,
                    start=skip_batches + 1,
                ):
                    if (
                        compiled_gradient_step is None
                        and self.training_config.jit_compile
                    ):
                        donate_argnums = ()
                        if self.training_config.donate_batch:
                            donate_argnums = (2,)
                        compiled_gradient_step = jax.jit(
                            gradient_step,
                            in_shardings=(
                                _tree_shardings(trainable_params),
                                _tree_shardings(frozen_params),
                                _tree_shardings(batch),
                                None,
                            ),
                            out_shardings=(
                                None,
                                _tree_shardings(trainable_params),
                            ),
                            donate_argnums=donate_argnums,
                        )
                    current_gradient_step = (
                        compiled_gradient_step or gradient_step
                    )
                    microbatch_loss, microbatch_grads = (
                        current_gradient_step(
                            trainable_params,
                            frozen_params,
                            batch,
                            jnp.asarray(
                                self.loss_scale,
                                dtype=jnp.float32,
                            ),
                        )
                    )
                    if accumulated_grads is None:
                        accumulated_grads = microbatch_grads
                        accumulated_loss = microbatch_loss.astype(jnp.float32)
                    else:
                        accumulated_grads = jax.tree.map(
                            lambda total, value: total + value,
                            accumulated_grads,
                            microbatch_grads,
                        )
                        accumulated_loss = (
                            accumulated_loss
                            + microbatch_loss.astype(jnp.float32)
                        )
                    accumulated_microbatches += 1
                    self.micro_step += 1
                    microbatches_run_this_call += 1

                    if accumulated_microbatches == accumulation_steps:
                        finish_accumulation(epoch, step_in_epoch)
                        epoch_updates_run += 1
                    if should_stop:
                        break
                batches.close()

                if accumulated_microbatches and not should_stop:
                    finish_accumulation(epoch, step_in_epoch)
                    epoch_updates_run += 1

                if (
                    self.training_config.eval_strategy == 'epoch'
                    and epoch_updates_run > 0
                ):
                    metrics, is_best = self._record_evaluation(
                        _combine_params(
                            trainable_params,
                            frozen_params,
                        ),
                        step=step,
                        epoch=epoch,
                    )
                    progress.console.print(
                        f"[bold blue]Evaluation[/bold blue] ┃ "
                        f"[bold yellow]Epoch {epoch:<3}[/bold yellow] ┃ "
                        f"Loss: [bold magenta]"
                        f"{metrics['eval_loss']:.4f}[/bold magenta]"
                    )
                    if (
                        is_best
                        and self.training_config.load_best_model_at_end
                    ):
                        self.best_model_checkpoint = (
                            self._checkpoint_directory(step)
                        )
                        self._ensure_checkpoint(
                            step,
                            trainable_params,
                            frozen_params,
                            opt_state,
                            epoch=epoch,
                            step_in_epoch=step_in_epoch,
                        )
                resume_step_in_epoch = 0

            if microbatches_run_this_call == 0 and step == 0:
                raise ValueError('dataloader produced no training batches')

            has_current_training_log = any(
                record.get('step') == step and 'loss' in record
                for record in reversed(self.log_history)
            )
            if steps_run_this_call > 0 and not has_current_training_log:
                seconds_per_step = (
                    (time.time() - start_time)
                    / max(1, steps_since_log)
                )
                learning_rate = self._learning_rate_at_step(step)
                self.log_history.append({
                    'step': step,
                    'epoch': epoch,
                    'loss': loss,
                    'seconds_per_step': seconds_per_step,
                    'learning_rate': learning_rate,
                    'grad_norm': grad_norm,
                    'loss_scale': self.loss_scale,
                    'skipped_update': update_skipped,
                })
                loss_text = (
                    f'{loss:<7.4f}' if loss is not None else 'non-finite'
                )
                learning_rate_text = (
                    f' ┃ LR: [bold green]'
                    f'{learning_rate:.3e}[/bold green]'
                    if learning_rate is not None
                    else ''
                )
                progress.console.print(
                    f"[bold cyan]Epoch {epoch:<3}[/bold cyan] ┃ "
                    f"[bold yellow]Step {step:<6}[/bold yellow] ┃ "
                    f"Loss: [bold magenta]{loss_text}"
                    f"[/bold magenta]{learning_rate_text} ┃ "
                    f"[dim]{_format_iteration_time(seconds_per_step):>11}"
                    f"[/dim]"
                )
                if saving_enabled:
                    final_checkpoint_path = self._checkpoint_directory(step)
                    if final_checkpoint_path in self.saved_checkpoints:
                        self._write_trainer_state(
                            final_checkpoint_path,
                            step=step,
                            epoch=epoch,
                            step_in_epoch=step_in_epoch,
                        )

            progress.update(
                task_id,
                completed=step,
                loss=float(loss) if loss is not None else float('nan'),
            )

        # 4. Inject back into the object if needed
        params = _combine_params(trainable_params, frozen_params)
        self._inject_params(params)
        if (
            self.training_config.save_at_end
            and steps_run_this_call > 0
            and (
                self.training_config.save_steps is None
                or step % self.training_config.save_steps != 0
            )
        ):
            if self._best_step == step:
                self.best_model_checkpoint = self._checkpoint_directory(step)
            checkpoint_path = self._ensure_checkpoint(
                step,
                trainable_params,
                frozen_params,
                opt_state,
                epoch=epoch,
                step_in_epoch=step_in_epoch,
            )
            console.print(
                f'[dim]Saved final checkpoint to {checkpoint_path}[/dim]'
            )
        if self.training_config.load_best_model_at_end:
            if self.best_model_checkpoint is None:
                raise ValueError(
                    'No best checkpoint was produced during evaluation'
                )
            self._load_checkpoint_model(self.best_model_checkpoint)
            console.print(
                f'[dim]Loaded best checkpoint from '
                f'{self.best_model_checkpoint}[/dim]'
            )
        console.print("[bold green]✨ Training complete![/bold green]")
        
    def _inject_params(self, params):
        if self.model_type == "taktiny":
            # The returned PyTree is a new taktiny Module. We can update self.model in-place.
            self.model.load_state_dict(params.state_dict())
        elif self.model_type == "nnx":
            from flax import nnx
            # params is the state dict, we merge it back into the graph
            nnx.update(self.model, params)
        elif self.model_type == "flax_linen":
            self.params = params
        elif self.model_type == "equinox":
            self.model = params


__all__ = ['Trainer']
