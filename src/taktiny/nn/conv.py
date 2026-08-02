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
"""Convolution modules"""
from __future__ import annotations


import jax
import jax.numpy as jnp
import math

from taktiny import nn

class Conv2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int],
        stride: int | tuple[int, int] = 1,
        padding: str | tuple[int, int] | tuple[tuple[int, int], tuple[int, int]] = "SAME",
        groups: int = 1,
        use_bias: bool = True,
        *, rngs: nn.Rngs
    ) -> None:
        self.in_channels = in_channels
        self.out_channels = out_channels

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        self.kernel_size = kernel_size

        if isinstance(stride, int):
            stride = (stride, stride)
        self.stride = stride

        if isinstance(padding, int):
            padding = ((padding, padding), (padding, padding))
        elif (
            isinstance(padding, tuple)
            and len(padding) == 2
            and all(isinstance(value, int) for value in padding)
        ):
            padding = tuple((value, value) for value in padding)
        self.padding = padding
        self.groups = groups
        self.use_bias = use_bias

        if in_channels % groups != 0:
            raise ValueError(f"in_channels ({in_channels}) must be divisible by groups ({groups})")
        if out_channels % groups != 0:
            raise ValueError(f"out_channels ({out_channels}) must be divisible by groups ({groups})")

        in_channels_per_group = in_channels // groups

        # LeCun uniform initialization
        k = math.sqrt(1.0 / (in_channels_per_group * kernel_size[0] * kernel_size[1]))

        w_key = rngs()
        b_key = rngs()
        # Shape: (H, W, I, O)
        w_shape = (*kernel_size, in_channels_per_group, out_channels)
        w_init = jax.random.uniform(w_key, w_shape, minval=-k, maxval=k)

        if use_bias:
            b_init = jax.random.uniform(b_key, (out_channels,), minval=-k, maxval=k)
        else:
            b_init = None

        self.weight = nn.Parameter(w_init)
        if use_bias:
            self.bias = nn.Parameter(b_init)
        else:
            self.bias = None

    def __call__(self, x: jax.Array) -> jax.Array:
        # x shape can be (N, H, W, C) or unbatched (H, W, C)
        is_unbatched = x.ndim == 3
        if is_unbatched:
            x = jnp.expand_dims(x, 0)

        out = jax.lax.conv_general_dilated(
            lhs=x,
            rhs=self.weight.value,
            window_strides=self.stride,
            padding=self.padding,
            dimension_numbers=("NHWC", "HWIO", "NHWC"),
            feature_group_count=self.groups
        )

        if is_unbatched:
            out = jnp.squeeze(out, 0)

        if self.use_bias:
            out = out + self.bias.value

        return out

class Upsample2d(nn.Module):
    def __init__(self, scale_factor: int = 2, method: str = "nearest") -> None:
        self.scale_factor = scale_factor
        self.method = method

    def __call__(self, x: jax.Array) -> jax.Array:
        is_unbatched = x.ndim == 3
        if is_unbatched:
            H, W, C = x.shape
            new_shape = (H * self.scale_factor, W * self.scale_factor, C)
        else:
            B, H, W, C = x.shape
            new_shape = (B, H * self.scale_factor, W * self.scale_factor, C)

        return jax.image.resize(x, shape=new_shape, method=self.method)
