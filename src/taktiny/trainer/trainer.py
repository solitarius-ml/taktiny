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
import json
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
            base_opt = optax.adamw(
                self.training_config.learning_rate,
                weight_decay=self.training_config.weight_decay,
            )
        return base_opt

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

    def _checkpoint_directory(self, step):
        return os.path.join(
            os.fspath(self.training_config.output_dir),
            f'checkpoint-{step}',
        )

    def _rotate_checkpoints(self):
        limit = self.training_config.save_total_limit
        if limit is None:
            return

        output_dir = os.fspath(self.training_config.output_dir)
        checkpoint_pattern = re.compile(r'checkpoint-(\d+)')
        checkpoints = []
        for entry in os.scandir(output_dir):
            match = checkpoint_pattern.fullmatch(entry.name)
            if entry.is_dir() and match is not None:
                checkpoints.append((int(match.group(1)), entry.path))
        checkpoints.sort()

        for _, checkpoint_path in checkpoints[:-limit]:
            shutil.rmtree(checkpoint_path)
        retained = {
            checkpoint_path
            for _, checkpoint_path in checkpoints[-limit:]
        }
        self.saved_checkpoints = [
            path
            for path in self.saved_checkpoints
            if path in retained
        ]

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
            
    def train(self):
        from rich.console import Console
        from rich.progress import Progress, TextColumn, BarColumn, TimeElapsedColumn, TimeRemainingColumn
        
        console = Console()
        console.print(f"[bold green]Starting training for a [cyan]{self.model_type.upper()}[/cyan] model[/bold green]")
        console.print(f"Epochs: [bold]{self.training_config.epochs}[/bold] | Max Steps: [bold]{self.training_config.max_steps}[/bold]")

        saving_enabled = (
            self.training_config.save_steps is not None
            or self.training_config.save_at_end
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
        
        # 2. Define train_step
        def calculate_loss(candidate_trainable, current_frozen, batch):
            current_params = _combine_params(
                candidate_trainable,
                current_frozen,
            )
            return self.loss_fn(current_params, batch)

        loss_and_grad = jax.value_and_grad(calculate_loss)

        def train_step(current_trainable, current_frozen, opt_state, batch):
            loss, grads = loss_and_grad(
                current_trainable,
                current_frozen,
                batch,
            )
            updates, new_opt_state = optimizer.update(
                grads,
                opt_state,
                current_trainable,
            )
            new_trainable = optax.apply_updates(
                current_trainable,
                updates,
            )
            return new_trainable, new_opt_state, loss

        compiled_train_step = None

        # 3. Training Loop
        import time
        step = 0
        should_stop = False
        start_time = time.time()
        steps_since_log = 0
        loss = None
        
        # Try to guess total steps if dataloader has __len__
        total_steps = None
        if hasattr(self.dataset_config.dataloader, "__len__"):
            total_steps = len(self.dataset_config.dataloader) * self.training_config.epochs
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
            
            task_id = progress.add_task("[cyan]Training...", total=total_steps, loss=0.0)
            
            for epoch in range(self.training_config.epochs):
                if should_stop:
                    break
                    
                batches = _prefetch(
                    self.dataset_config.dataloader,
                    self._place_batch,
                    self.dataset_config.prefetch_size,
                )
                for step_in_epoch, batch in enumerate(batches, start=1):
                    if (
                        compiled_train_step is None
                        and self.training_config.jit_compile
                    ):
                        donate_argnums = (0, 2)
                        if self.training_config.donate_batch:
                            donate_argnums += (3,)
                        compiled_train_step = jax.jit(
                            train_step,
                            in_shardings=(
                                _tree_shardings(trainable_params),
                                _tree_shardings(frozen_params),
                                _tree_shardings(opt_state),
                                _tree_shardings(batch),
                            ),
                            out_shardings=(
                                _tree_shardings(trainable_params),
                                _tree_shardings(opt_state),
                                None,
                            ),
                            donate_argnums=donate_argnums,
                        )
                    step_fn = compiled_train_step or train_step
                    trainable_params, opt_state, loss = step_fn(
                        trainable_params,
                        frozen_params,
                        opt_state,
                        batch,
                    )
                    step += 1
                    self.global_step = step
                    steps_since_log += 1
                    
                    if step % self.training_config.log_interval == 0:
                        loss = loss.item() if isinstance(loss, jax.Array) else loss
                        progress.update(task_id, advance=self.training_config.log_interval, loss=float(loss))
                        
                        # Calculate timing
                        elapsed = time.time() - start_time
                        seconds_per_step = (
                            elapsed
                            / max(1, steps_since_log)
                        )
                        iteration_time = _format_iteration_time(
                            seconds_per_step
                        )
                        self.log_history.append({
                            'step': step,
                            'epoch': epoch,
                            'loss': float(loss),
                            'seconds_per_step': seconds_per_step,
                        })
                        
                        # Persistent log above the progress bar
                        log_msg = (
                            f"[bold cyan]Epoch {epoch:<3}[/bold cyan] ┃ "
                            f"[bold yellow]Step {step:<6}[/bold yellow] ┃ "
                            f"Loss: [bold magenta]{loss:<7.4f}[/bold magenta] ┃ "
                            f"[dim]{iteration_time:>11}[/dim]"
                        )
                        progress.console.print(log_msg)
                        start_time = time.time()
                        steps_since_log = 0

                    if (
                        self.training_config.save_steps is not None
                        and step % self.training_config.save_steps == 0
                    ):
                        checkpoint_path = self._save_checkpoint(
                            step,
                            trainable_params,
                            frozen_params,
                            opt_state,
                            epoch=epoch,
                            step_in_epoch=step_in_epoch,
                        )
                        progress.console.print(
                            f'[dim]Saved checkpoint to '
                            f'{checkpoint_path}[/dim]'
                        )
                        
                    if self.training_config.max_steps is not None and step >= self.training_config.max_steps:
                        should_stop = True
                        break
                batches.close()
                        
            if loss is None:
                raise ValueError('dataloader produced no training batches')

            if (
                not self.log_history
                or self.log_history[-1]['step'] != step
            ):
                loss = loss.item() if isinstance(loss, jax.Array) else loss
                seconds_per_step = (
                    (time.time() - start_time)
                    / max(1, steps_since_log)
                )
                self.log_history.append({
                    'step': step,
                    'epoch': epoch,
                    'loss': float(loss),
                    'seconds_per_step': seconds_per_step,
                })
                progress.console.print(
                    f"[bold cyan]Epoch {epoch:<3}[/bold cyan] ┃ "
                    f"[bold yellow]Step {step:<6}[/bold yellow] ┃ "
                    f"Loss: [bold magenta]{float(loss):<7.4f}"
                    f"[/bold magenta] ┃ "
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
                loss=float(loss),
            )
                
        # 4. Inject back into the object if needed
        params = _combine_params(trainable_params, frozen_params)
        self._inject_params(params)
        if (
            self.training_config.save_at_end
            and (
                self.training_config.save_steps is None
                or step % self.training_config.save_steps != 0
            )
        ):
            checkpoint_path = self._save_checkpoint(
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
