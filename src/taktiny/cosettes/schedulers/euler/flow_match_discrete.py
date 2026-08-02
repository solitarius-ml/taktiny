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
from typing import Any


import math
from typing import Literal, Optional, Union

import jax
import jax.numpy as jnp
import numpy as np

from taktiny import nn


class FlowMatchEulerDiscreteScheduler(nn.Module):
    """
    Euler scheduler for flow-matching
    """

    def __init__(
        self,
        num_train_timesteps: int = 1000,
        shift: float = 1.0,
        use_dynamic_shifting: bool = False,
        base_shift: float = 0.5,
        max_shift: float = 1.15,
        base_image_seq_len: int = 256,
        max_image_seq_len: int = 4096,
        invert_sigmas: bool = False,
        shift_terminal: float | None = None,
        use_karras_sigmas: bool = False,
        use_exponential_sigmas: bool = False,
        use_beta_sigmas: bool = False,
        time_shift_type: Literal["exponential", "linear"] = "exponential",
        stochastic_sampling: bool = False,
    ) -> None:
        self.num_train_timesteps = num_train_timesteps
        self.shift = shift
        self.use_dynamic_shifting = use_dynamic_shifting
        self.base_shift = base_shift
        self.max_shift = max_shift
        self.base_image_seq_len = base_image_seq_len
        self.max_image_seq_len = max_image_seq_len
        self.invert_sigmas = invert_sigmas
        self.shift_terminal = shift_terminal
        self.use_karras_sigmas = use_karras_sigmas
        self.use_exponential_sigmas = use_exponential_sigmas
        self.use_beta_sigmas = use_beta_sigmas
        self.time_shift_type = time_shift_type
        self.stochastic_sampling = stochastic_sampling

        if time_shift_type not in {"exponential", "linear"}:
            raise ValueError("`time_shift_type` must either be 'exponential' or 'linear'.")

        timesteps = jnp.linspace(1, num_train_timesteps, num_train_timesteps, dtype=jnp.float32)[::-1]
        sigmas = timesteps / num_train_timesteps

        if not use_dynamic_shifting:
            sigmas = shift * sigmas / (1 + (shift - 1) * sigmas)

        self.timesteps = sigmas * num_train_timesteps
        self.sigmas = sigmas
        self.sigma_min = float(self.sigmas[-1])
        self.sigma_max = float(self.sigmas[0])

        self.step_index = None

    def _sigma_to_t(self, sigma: Any) -> float:
        return sigma * self.num_train_timesteps

    def time_shift(self, mu: float, sigma: float, t: jnp.ndarray) -> jnp.ndarray:
        if self.time_shift_type == "exponential":
            return math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** sigma)
        elif self.time_shift_type == "linear":
            return mu / (mu + (1 / t - 1) ** sigma)

    def stretch_shift_to_terminal(self, t: jnp.ndarray) -> jnp.ndarray:
        one_minus_z = 1 - t
        scale_factor = one_minus_z[-1] / (1 - self.shift_terminal)
        stretched_t = 1 - (one_minus_z / scale_factor)
        return stretched_t

    def set_timesteps(
        self,
        num_inference_steps: int,
        mu: float | None = None,
        sigmas: list[float] | None = None,
        timesteps: list[float] | None = None,
    ) -> None:
        if self.use_dynamic_shifting and mu is None:
            raise ValueError("`mu` must be passed when `use_dynamic_shifting` is set to be `True`")

        self.num_inference_steps = num_inference_steps
        is_timesteps_provided = timesteps is not None

        if is_timesteps_provided:
            timesteps = jnp.array(timesteps, dtype=jnp.float32)

        if sigmas is None:
            if timesteps is None:
                timesteps = jnp.linspace(
                    self._sigma_to_t(self.sigma_max),
                    self._sigma_to_t(self.sigma_min),
                    num_inference_steps,
                )
            sigmas = timesteps / self.num_train_timesteps
        else:
            sigmas = jnp.array(sigmas, dtype=jnp.float32)
            num_inference_steps = len(sigmas)

        if self.use_dynamic_shifting:
            sigmas = self.time_shift(mu, 1.0, sigmas)
        else:
            sigmas = self.shift * sigmas / (1 + (self.shift - 1) * sigmas)

        if self.shift_terminal:
            sigmas = self.stretch_shift_to_terminal(sigmas)

        if self.use_karras_sigmas:
            sigmas = self._convert_to_karras(sigmas, num_inference_steps)
        elif self.use_exponential_sigmas:
            sigmas = self._convert_to_exponential(sigmas, num_inference_steps)
        elif self.use_beta_sigmas:
            sigmas = self._convert_to_beta(sigmas, num_inference_steps)

        if not is_timesteps_provided:
            timesteps = sigmas * self.num_train_timesteps

        if self.invert_sigmas:
            sigmas = 1.0 - sigmas
            timesteps = sigmas * self.num_train_timesteps
            sigmas = jnp.concatenate([sigmas, jnp.ones((1,))])
        else:
            sigmas = jnp.concatenate([sigmas, jnp.zeros((1,))])

        self.timesteps = timesteps
        self.sigmas = sigmas
        self.step_index = None

    def _convert_to_karras(self, in_sigmas: jnp.ndarray, num_inference_steps: int) -> jnp.ndarray:
        sigma_min = float(in_sigmas[-1])
        sigma_max = float(in_sigmas[0])
        rho = 7.0
        ramp = jnp.linspace(0, 1, num_inference_steps)
        min_inv_rho = sigma_min ** (1 / rho)
        max_inv_rho = sigma_max ** (1 / rho)
        sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
        return sigmas

    def _convert_to_exponential(self, in_sigmas: jnp.ndarray, num_inference_steps: int) -> jnp.ndarray:
        sigma_min = float(in_sigmas[-1])
        sigma_max = float(in_sigmas[0])
        sigmas = jnp.exp(jnp.linspace(math.log(sigma_max), math.log(sigma_min), num_inference_steps))
        return sigmas

    def _convert_to_beta(
        self, in_sigmas: jnp.ndarray, num_inference_steps: int, alpha: float = 0.6, beta: float = 0.6
    ) -> jnp.ndarray:
        import scipy.stats

        sigma_min = float(in_sigmas[-1])
        sigma_max = float(in_sigmas[0])

        sigmas = np.array(
            [
                sigma_min + (ppf * (sigma_max - sigma_min))
                for ppf in [
                    scipy.stats.beta.ppf(timestep, alpha, beta)
                    for timestep in 1 - np.linspace(0, 1, num_inference_steps)
                ]
            ]
        )
        return jnp.array(sigmas, dtype=jnp.float32)

    def step(
        self,
        model_output: jnp.ndarray,
        timestep: Union[float, jnp.ndarray],
        sample: jnp.ndarray,
        generator: Optional[jax.Array] = None,
        return_dict: bool = False,
    ) -> tuple:

        if self.step_index is None:
            # Initialize step index based on closest timestep
            diffs = jnp.abs(self.timesteps - timestep)
            self.step_index = int(jnp.argmin(diffs))

        sigma = self.sigmas[self.step_index]
        sigma_next = self.sigmas[self.step_index + 1]
        dt = sigma_next - sigma

        if self.stochastic_sampling:
            x0 = sample - sigma * model_output
            # Generate random noise for stochastic sampling
            if generator is None:
                generator = jax.random.PRNGKey(0) # Fallback key if not provided
            noise = jax.random.normal(generator, shape=sample.shape, dtype=sample.dtype)
            prev_sample = (1.0 - sigma_next) * x0 + sigma_next * noise
        else:
            prev_sample = sample + dt * model_output

        self.step_index += 1

        if not return_dict:
            return (prev_sample,)

        return prev_sample

    def __len__(self) -> int:
        return self.num_train_timesteps