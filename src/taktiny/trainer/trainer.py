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


def _stop_gradient_frozen(params, labels):
    return jax.tree.map(
        lambda value, label: (
            value
            if label == 'trainable'
            else jax.lax.stop_gradient(value)
        ),
        params,
        labels,
    )


def _apply_update(parameter, update):
    if getattr(update, 'dtype', None) == jax.dtypes.float0:
        return parameter
    return parameter + update


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


def _place_unsharded(tree, mesh):
    if mesh is None:
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

class Trainer:
    def __init__(self, model, loss_fn, training_config: TrainingConfig, dataset_config: DatasetConfig):
        self.model = model
        self.loss_fn = loss_fn
        self.training_config = training_config
        self.dataset_config = dataset_config
        self.model_type = self._diagnose_model_type(model)
        self._mesh = None
        
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
            
    def _setup_optimizer(self, params, labels):
        """Configure an optimizer using explicit trainability labels."""
        base_opt = self.training_config.optimizer
        if base_opt is None:
            base_opt = optax.adamw(
                self.training_config.learning_rate,
                weight_decay=self.training_config.weight_decay,
            )

        return optax.multi_transform(
            {'trainable': base_opt, 'frozen': optax.set_to_zero()},
            lambda _: labels,
        )

    def _place_batch(self, batch):
        sharding = self.dataset_config.batch_sharding
        if sharding is None:
            if self._mesh is None:
                return batch
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
            
    def train(self):
        from rich.console import Console
        from rich.progress import Progress, TextColumn, BarColumn, TimeElapsedColumn, TimeRemainingColumn
        
        console = Console()
        console.print(f"[bold green]Starting training for a [cyan]{self.model_type.upper()}[/cyan] model[/bold green]")
        console.print(f"Epochs: [bold]{self.training_config.epochs}[/bold] | Max Steps: [bold]{self.training_config.max_steps}[/bold]")
        
        # 1. Initialize Optimizer
        params = self.extract_params()
        self._mesh = (
            _parameter_mesh(params)
            or _sharding_mesh(self.dataset_config.batch_sharding)
        )
        params = _place_unsharded(params, self._mesh)
        labels = _parameter_labels(params)
        optimizer = self._setup_optimizer(params, labels)
        opt_state = optimizer.init(params)
        opt_state = _place_unsharded(
            opt_state,
            self._mesh,
        )
        
        # 2. Define train_step
        def train_step(params, opt_state, batch):
            def calculate_loss(current_params):
                differentiable_params = _stop_gradient_frozen(
                    current_params,
                    labels,
                )
                return self.loss_fn(differentiable_params, batch)

            loss, grads = jax.value_and_grad(
                calculate_loss,
                allow_int=True,
            )(params)
            updates, new_opt_state = optimizer.update(grads, opt_state, params)
            new_params = jax.tree.map(_apply_update, params, updates)
            return new_params, new_opt_state, loss

        compiled_train_step = None

        # 3. Training Loop
        import time
        step = 0
        should_stop = False
        start_time = time.time()
        
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
                    
                for batch in self.dataset_config.dataloader:
                    batch = self._place_batch(batch)
                    if (
                        compiled_train_step is None
                        and self.training_config.jit_compile
                    ):
                        compiled_train_step = jax.jit(
                            train_step,
                            in_shardings=(
                                _tree_shardings(params),
                                _tree_shardings(opt_state),
                                _tree_shardings(batch),
                            ),
                            out_shardings=(
                                _tree_shardings(params),
                                _tree_shardings(opt_state),
                                None,
                            ),
                        )
                    step_fn = compiled_train_step or train_step
                    params, opt_state, loss = step_fn(
                        params,
                        opt_state,
                        batch,
                    )
                    step += 1
                    
                    if step % self.training_config.log_interval == 0:
                        loss = loss.item() if isinstance(loss, jax.Array) else loss
                        progress.update(task_id, advance=self.training_config.log_interval, loss=float(loss))
                        
                        # Calculate timing
                        elapsed = time.time() - start_time
                        ms_per_step = (elapsed / max(1, self.training_config.log_interval)) * 1000
                        
                        # Persistent log above the progress bar
                        log_msg = (
                            f"[bold cyan]Epoch {epoch:<3}[/bold cyan] ┃ "
                            f"[bold yellow]Step {step:<6}[/bold yellow] ┃ "
                            f"Loss: [bold magenta]{loss:<7.4f}[/bold magenta] ┃ "
                            f"[dim]{ms_per_step:>6.1f} ms/it[/dim]"
                        )
                        progress.console.print(log_msg)
                        start_time = time.time()
                        
                    if self.training_config.max_steps is not None and step >= self.training_config.max_steps:
                        should_stop = True
                        break
                        
            # Force finish the progress bar
            progress.update(task_id, completed=total_steps if total_steps else step, loss=float(loss))
                
        # 4. Inject back into the object if needed
        self._inject_params(params)
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
