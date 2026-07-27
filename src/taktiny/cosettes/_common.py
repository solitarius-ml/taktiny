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
"""Common base modules for transformer architectures"""

from __future__ import annotations

from taktiny import nn
from taktiny.cosettes._base import PretrainedModel
from taktiny.maestro._config import ModelConfig
from taktiny.utils.typing import ShardMode
from taktiny.utils.sharding import create_sharding
from taktiny.layers import RotaryEmbedding, GateMLP, Attention

from dataclasses import dataclass
import jax
import jax.numpy as jnp
from dataclasses import replace
from typing import *
from functools import partial


@partial(
    jax.tree_util.register_dataclass,
    data_fields=['key_cache', 'value_cache', 'position_idx'],
    meta_fields=['is_causal'],
)
@dataclass(frozen=True)
class TransformerContext:
    key_cache: jax.Array | None
    value_cache: jax.Array | None
    position_idx: jax.Array | None
    is_causal: bool


class TransformerDecoderLayer(nn.Module):
    """An ordered transformer decoder block assembled from module types.

    Modules are created and executed in the same order as the keyword arguments
    passed to the constructor. Normalization modules transform the current
    hidden state, while attention and feed-forward modules form residual
    branches. Consecutive normalization modules allow architectures such as
    Gemma 2 to place norms on both sides of a residual branch.

    Attention modules receive the attention mask, causal flag, position index,
    and optional KV cache. The returned cache has the same per-layer
    ``(key_cache, value_cache)`` structure as the input cache.

    Args:
        config: Model configuration containing the hidden size, attention
            dimensions, positional embedding settings, and MLP settings.
        rngs: Random number generator used to initialize parameterized modules.
        layer_idx: Index of this layer in the model. This selects per-layer
            attention modes such as Gemma2's alternating sliding/full pattern.
        **modules: Ordered mapping from checkpoint-facing module names to
            ``nn.Module`` subclasses or initialized module instances. Supported
            types are normalization, ``Attention``, and ``GateMLP`` modules.

    Returns:
        A tuple containing the transformed hidden states and the updated KV
        cache, or ``None`` when no cache was supplied.
    """

    def __init__(self, config, *, rngs, layer_idx=None, **modules):
        shard_mode = getattr(config, 'shard_mode', ShardMode.AUTO)
        quant = getattr(config, 'quant', None)
        dot_general = getattr(config, 'dot_general', None)

        hidden_size = getattr(config, 'hidden_size', None)
        num_heads = getattr(config, 'num_attention_heads', None)
        num_kv_heads = getattr(config, 'num_key_value_heads', None)
        max_position_embeddings = getattr(config, 'max_position_embeddings', None)
        rope_parameters = getattr(config, 'rope_parameters', None) or {}
        rope_theta = (
            getattr(config, 'rope_theta', None)
            or rope_parameters.get('rope_theta')
        )
        intermediate_size = getattr(config, 'intermediate_size', None)

        required = {
            'hidden_size': hidden_size,
            'num_attention_heads': num_heads,
            'num_key_value_heads': num_kv_heads,
            'max_position_embeddings': max_position_embeddings,
            'rope_theta': rope_theta,
            'intermediate_size': intermediate_size,
        }
        missing = [name for name, value in required.items() if value is None]
        if missing:
            raise ValueError(
                f"Missing required transformer config values: {', '.join(missing)}"
            )

        dtype = (
            getattr(config, 'torch_dtype', None)
            or getattr(config, 'dtype', None)
            or 'bfloat16'
        )
        head_dim = getattr(config, 'head_dim', None) or hidden_size // num_heads
        mlp_bias = getattr(config, 'mlp_bias', None) or False
        attention_bias = getattr(config, 'attention_bias', None) or False
        eps = getattr(config, 'rms_norm_eps', None) or 1e-6
        rope_scaling = getattr(config, 'rope_scaling', None)
        sliding_window = getattr(config, 'sliding_window', None)
        if getattr(config, 'use_sliding_window', None) is False:
            sliding_window = None
        layer_types = getattr(config, 'layer_types', None)
        if layer_idx is not None and layer_types is not None:
            layer_type = layer_types[layer_idx]
            if layer_type in ('full_attention', 'full'):
                sliding_window = None
        hidden_act = (
            getattr(config, 'hidden_act', None)
            or getattr(config, 'hidden_activation', None)
            or getattr(config, 'act', None)
            or 'silu'
        )
        if hidden_act in ('gelu_pytorch_tanh', 'gelu_new', 'gelu_fast'):
            hidden_act = partial(jax.nn.gelu, approximate=True)
        attention_scaling = None
        query_pre_attn_scalar = getattr(config, 'query_pre_attn_scalar', None)
        if query_pre_attn_scalar is not None:
            attention_scaling = query_pre_attn_scalar ** -0.5
        attention_softcap = getattr(config, 'attn_logit_softcapping', None)
        attention_dropout = getattr(config, 'attention_dropout', None) or 0.0

        if hidden_size % num_heads != 0 and getattr(config, 'head_dim', None) is None:
            raise ValueError(
                'hidden_size must be divisible by num_attention_heads when '
                'head_dim is not configured'
            )
        if num_heads % num_kv_heads != 0:
            raise ValueError(
                'num_attention_heads must be divisible by num_key_value_heads'
            )

        self.hidden_size = hidden_size
        self.dtype = dtype
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.max_position_embeddings = max_position_embeddings
        self.rope_theta = rope_theta
        self.attention_bias = attention_bias
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.head_dim = head_dim
        self.mlp_bias = mlp_bias
        self.eps = eps
        self.rope_scaling = rope_scaling
        self.sliding_window = sliding_window
        self.layer_idx = layer_idx

        if not modules:
            raise ValueError('TransformerDecoderLayer requires at least one module')

        module_order = []
        module_kinds = []
        for name, module_type in modules.items():
            module, kind = self._create_module(
                name=name,
                module_type=module_type,
                hidden_size=hidden_size,
                intermediate_size=intermediate_size,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                max_position_embeddings=max_position_embeddings,
                rope_theta=rope_theta,
                rope_scaling=rope_scaling,
                sliding_window=sliding_window,
                hidden_act=hidden_act,
                attention_bias=attention_bias,
                attention_scaling=attention_scaling,
                attention_softcap=attention_softcap,
                attention_dropout=attention_dropout,
                mlp_bias=mlp_bias,
                eps=eps,
                dtype=dtype,
                shard_mode=shard_mode,
                quant=quant,
                dot_general=dot_general,
                rngs=rngs,
            )
            setattr(self, name, module)
            module_order.append(name)
            module_kinds.append(kind)

        self.module_order = tuple(module_order)
        self.module_kinds = tuple(module_kinds)

    @staticmethod
    def _create_module(
        *,
        name,
        module_type,
        hidden_size,
        intermediate_size,
        num_heads,
        num_kv_heads,
        head_dim,
        max_position_embeddings,
        rope_theta,
        rope_scaling,
        sliding_window,
        hidden_act,
        attention_bias,
        attention_scaling,
        attention_softcap,
        attention_dropout,
        mlp_bias,
        eps,
        dtype,
        shard_mode,
        quant,
        dot_general,
        rngs,
    ):
        if isinstance(module_type, nn.Module):
            module = module_type
            module_type = type(module)
        elif not isinstance(module_type, type) or not issubclass(module_type, nn.Module):
            raise TypeError(f'{name} must be an nn.Module subclass or instance')
        elif issubclass(module_type, nn.RMSNorm):
            module = module_type(
                hidden_size,
                eps=eps,
                dtype=jnp.float32,
                shard_mode=shard_mode,
                axis_names=('embed',),
            )
        elif issubclass(module_type, nn.LayerNorm):
            module = module_type(
                hidden_size,
                eps=eps,
                axis_names=('embed',),
            )
        elif issubclass(module_type, Attention):
            module = module_type(
                hidden_size=hidden_size,
                num_heads=num_heads,
                head_dim=head_dim,
                num_kv_heads=num_kv_heads,
                pos_emb=RotaryEmbedding(
                    head_dim,
                    max_position_embeddings,
                    rope_theta,
                    rope_scaling,
                ),
                bias=False,
                q_bias=attention_bias,
                k_bias=attention_bias,
                v_bias=attention_bias,
                o_bias=attention_bias,
                dtype=dtype,
                rngs=rngs,
                q_axis_names=('embed', 'heads', 'head_dim'),
                k_axis_names=('embed', 'kv_heads', 'head_dim'),
                v_axis_names=('embed', 'kv_heads', 'head_dim'),
                o_axis_names=('heads', 'head_dim', 'embed'),
                window_size=sliding_window,
                scaling=attention_scaling,
                softcap=attention_softcap,
                dropout=attention_dropout,
                shard_mode=shard_mode,
                quant=quant,
                dot_general=dot_general,
            )
        elif issubclass(module_type, GateMLP):
            module = module_type(
                hidden_size=hidden_size,
                intermediate_size=intermediate_size,
                activation=hidden_act,
                bias=mlp_bias,
                dtype=dtype,
                rngs=rngs,
                gate_axis_names=('embed', 'mlp'),
                up_axis_names=('embed', 'mlp'),
                down_axis_names=('mlp', 'embed'),
                shard_mode=shard_mode,
                quant=quant,
                dot_general=dot_general,
            )
        else:
            raise TypeError(
                f'Unsupported decoder module {name}: {module_type.__name__}'
            )

        if issubclass(module_type, (nn.RMSNorm, nn.LayerNorm)):
            kind = 'norm'
        elif issubclass(module_type, Attention):
            kind = 'attention'
        elif issubclass(module_type, GateMLP):
            kind = 'residual'
        else:
            raise TypeError(
                f'Unsupported decoder module {name}: {module_type.__name__}'
            )

        return module, kind

    @staticmethod
    def _apply_norm(module, x, out_sharding):
        if isinstance(module, nn.RMSNorm):
            return module(x, out_sharding=out_sharding)
        return module(x)

    def __call__(
        self,
        x: jax.Array,
        attention_mask: jax.Array = None,
        kv_cache: tuple[jax.Array, jax.Array] = None,
        position_idx: jax.Array = None,
        is_causal: bool = False,
        out_sharding=None,
    ):
        residual = x
        pending = None
        new_cache = None

        for index, (name, kind) in enumerate(
            zip(self.module_order, self.module_kinds)
        ):
            module = getattr(self, name)

            if kind == 'norm':
                if pending is None:
                    x = self._apply_norm(module, x, out_sharding)
                    continue

                remaining_kinds = self.module_kinds[index:]
                next_residual = next(
                    (
                        offset
                        for offset, next_kind in enumerate(remaining_kinds)
                        if next_kind != 'norm'
                    ),
                    None,
                )

                if next_residual == 1:
                    x = residual + pending
                    residual = x
                    pending = None
                    x = self._apply_norm(module, x, out_sharding)
                else:
                    pending = self._apply_norm(module, pending, out_sharding)
                    if next_residual is not None:
                        x = residual + pending
                        residual = x
                        pending = None
                continue

            if pending is not None:
                x = residual + pending
                residual = x
                pending = None

            if kind == 'attention':
                pending, new_cache = module(
                    x,
                    attention_mask=attention_mask,
                    is_causal=is_causal,
                    kv_cache=kv_cache,
                    position_idx=position_idx,
                    out_sharding=out_sharding,
                )
            else:
                pending = module(x, out_sharding=out_sharding)

        if pending is not None:
            x = residual + pending

        return x, new_cache


class TransformerModel(nn.Module):
    """Token embedding followed by a list of transformer decoder layers.

    The supplied decoder type is instantiated ``config.num_hidden_layers``
    times and stored in an ``nn.List``. Each layer owns independent parameters
    initialized from the shared RNG stream. During a forward pass, a stacked KV
    cache is sliced by layer and rebuilt with the updated per-layer cache values.

    Args:
        config: Model configuration containing ``num_hidden_layers``,
            ``vocab_size``, ``hidden_size``, and the decoder settings.
        rngs: Random number generator used for embeddings and decoder layers.
        module: Decoder-layer ``nn.Module`` subclass to repeat. It receives its
            zero-based ``layer_idx`` when instantiated.
        embedding: Embedding ``nn.Module`` subclass or initialized instance.
        norm: Optional final normalization module type or instance.

    Returns:
        A tuple containing the final hidden states and an updated stacked
        ``(key_cache, value_cache)``, or ``None`` when caching is disabled.
    """

    def __init__(self, config, *, rngs, module, embedding, norm=None):
        num_hidden_layers = getattr(config, 'num_hidden_layers', None)
        vocab_size = getattr(config, 'vocab_size', None)
        hidden_size = getattr(config, 'hidden_size', None)

        if num_hidden_layers is None:
            raise ValueError(
                'Missing required transformer config value: num_hidden_layers'
            )
        if vocab_size is None:
            raise ValueError('Missing required transformer config value: vocab_size')
        if hidden_size is None:
            raise ValueError('Missing required transformer config value: hidden_size')
        dtype = (
            getattr(config, 'torch_dtype', None)
            or getattr(config, 'dtype', None)
            or 'bfloat16'
        )
        if not isinstance(num_hidden_layers, int) or num_hidden_layers < 1:
            raise ValueError('num_hidden_layers must be a positive integer')
        if not isinstance(module, type) or not issubclass(module, nn.Module):
            raise TypeError('module must be an nn.Module subclass')
        if isinstance(embedding, nn.Module):
            embed_tokens = embedding
        elif isinstance(embedding, type) and issubclass(embedding, nn.Module):
            embed_tokens = embedding(
                vocab_size,
                hidden_size,
                rngs=rngs,
                dtype=dtype,
            )
        else:
            raise TypeError('embedding must be an nn.Module subclass or instance')

        self.config = config
        self.num_hidden_layers = num_hidden_layers
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.embed_tokens = embed_tokens
        if hasattr(self.embed_tokens, 'embedding'):
            self.embed_tokens.embedding.axis_names = ('vocab', 'embed')

        self.layers = nn.List(
            *(
                module(config, rngs=rngs, layer_idx=layer_idx)
                for layer_idx in range(num_hidden_layers)
            )
        )

        self.norm = None
        if isinstance(norm, nn.Module):
            self.norm = norm
        elif isinstance(norm, type) and issubclass(norm, nn.RMSNorm):
            self.norm = norm(
                hidden_size,
                eps=getattr(config, 'rms_norm_eps', None) or 1e-6,
                dtype=jnp.float32,
                shard_mode=getattr(config, 'shard_mode', ShardMode.AUTO),
                axis_names=('embed',),
            )
        elif isinstance(norm, type) and issubclass(norm, nn.LayerNorm):
            self.norm = norm(
                hidden_size,
                eps=getattr(config, 'layer_norm_eps', None) or 1e-5,
                axis_names=('embed',),
            )
        elif norm is not None:
            raise TypeError(
                'norm must be a normalization nn.Module subclass or instance'
            )

    def __call__(
        self,
        x: jax.Array,
        attention_mask: jax.Array = None,
        kv_cache: tuple[jax.Array, jax.Array] = None,
        position_idx: jax.Array = None,
        is_causal: bool = False,
        out_sharding=None,
    ):
        x = self.embed_tokens(x)

        if kv_cache is not None:
            if len(kv_cache) != 2:
                raise ValueError('kv_cache must contain key and value caches')

            key_cache, value_cache = kv_cache
            if key_cache.shape[0] != self.num_hidden_layers:
                raise ValueError(
                    'key cache must have one entry for each transformer layer'
                )
            if value_cache.shape[0] != self.num_hidden_layers:
                raise ValueError(
                    'value cache must have one entry for each transformer layer'
                )

        new_key_cache = []
        new_value_cache = []
        for layer_idx, layer in enumerate(self.layers):
            layer_cache = None
            if kv_cache is not None:
                layer_cache = (
                    key_cache[layer_idx],
                    value_cache[layer_idx],
                )

            x, new_cache = layer(
                x,
                attention_mask=attention_mask,
                kv_cache=layer_cache,
                position_idx=position_idx,
                is_causal=is_causal,
                out_sharding=out_sharding,
            )

            if new_cache is not None:
                new_key_cache.append(new_cache[0])
                new_value_cache.append(new_cache[1])

        new_cache = None
        if new_key_cache:
            new_cache = (
                jnp.stack(new_key_cache),
                jnp.stack(new_value_cache),
            )

        if self.norm is not None:
            if isinstance(self.norm, nn.RMSNorm):
                x = self.norm(x, out_sharding=out_sharding)
            else:
                x = self.norm(x)

        return x, new_cache


class TransformerCausalLM(PretrainedModel):
    """Causal language model composed from an embedding, decoder, and LM head.

    ``TransformerModel`` owns the token embedding and repeated decoder layers.
    This wrapper projects its final hidden states to vocabulary logits. When
    word embeddings are tied and no explicit LM head is supplied, logits are
    computed directly with the embedding matrix so the parameter is registered
    only once. Otherwise, ``nn.Linear`` is used as the default LM head.

    Tying is read from ``config.tie_word_embeddings`` and also accepts the
    legacy ``tied_word_embeddings`` and ``tied_word_embedding`` spellings.

    Args:
        config: Model configuration containing vocabulary, hidden, decoder, and
            optional weight-tying settings.
        rngs: Random number generator used to initialize the model.
        embedding: Optional embedding module type or instance. Defaults to
            ``nn.Embedding``.
        decoder: Required decoder-layer ``nn.Module`` subclass repeated by
            ``TransformerModel``.
        norm: Optional final normalization module type or instance.
        lm_head: Optional output-head module type or instance. When omitted, a
            tied embedding projection or ``nn.Linear`` is selected from config.
        mesh: Optional JAX device mesh used for explicit sharding.
        sharding_rules: Optional logical-to-mesh axis mapping rules.

    Returns:
        A tuple containing vocabulary logits and the updated
        ``TransformerContext``, or ``None`` when no context was supplied.
    """

    default_sharding_rules = [
        ('vocab', 'tp'),
        ('embed', None),
        ('heads', 'tp'),
        ('kv_heads', 'tp'),
        ('head_dim', None),
        ('mlp', 'tp'),
        ('batch', 'fsdp'),
        ('sequence', None),
    ]

    def __init__(
        self, config: ModelConfig,
        *, rngs: nn.Rngs,
        embedding = None,
        decoder = None,
        norm = None,
        lm_head = None,
        mesh: jax.sharding.Mesh = None,
        sharding_rules: Optional[List[Tuple]] = None
    ):
        if decoder is None:
            raise ValueError('decoder is required')
        
        if embedding is None:
            embedding = nn.Embedding

        if (vocab_size := getattr(config, 'vocab_size', None)) is None:
            raise ValueError('Missing required transformer config value: vocab_size')
        
        if (hidden_size := getattr(config, 'hidden_size', None)) is None:
            raise ValueError('Missing required transformer config value: hidden_size')

        self.config = config
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.shard_mode = getattr(config, 'shard_mode', ShardMode.AUTO)
        self.quant = getattr(config, 'quant', None)
        self.dot_general = getattr(config, 'dot_general', None)
        self.dtype = (
            getattr(config, 'torch_dtype', None)
            or getattr(config, 'dtype', None)
            or 'bfloat16'
        )

        self.model = TransformerModel(
            config,
            rngs=rngs,
            module=decoder,
            embedding=embedding,
            norm=norm,
        )

        tied = getattr(config, 'tie_word_embeddings', None) # maybe
        if tied is None:
            tied = getattr(config, 'tied_word_embeddings', None)
        if tied is None:
            tied = getattr(config, 'tied_word_embedding', False)

        self.tied_word_embeddings = bool(tied and lm_head is None)
        if self.tied_word_embeddings:
            if not hasattr(self.model.embed_tokens, 'embedding'):
                raise TypeError(
                    'A tied embedding must expose its weight as `embedding`'
                )
        else:
            if lm_head is None:
                lm_head = nn.Linear

            if isinstance(lm_head, nn.Module):
                self.lm_head = lm_head
            elif isinstance(lm_head, type) and issubclass(lm_head, nn.Module):
                self.lm_head = lm_head(
                    hidden_size,
                    vocab_size,
                    bias=False,
                    dtype=self.dtype,
                    rngs=rngs,
                    axis_names=('embed', 'vocab'),
                    shard_mode=self.shard_mode,
                    quant=self.quant,
                    dot_general=self.dot_general,
                )
            else:
                raise TypeError('lm_head must be an nn.Module subclass or instance')

        if sharding_rules is None:
            sharding_rules = self.default_sharding_rules

        self.model_out_sharding = None
        self.logits_out_sharding = None
        if mesh is not None and self.shard_mode == ShardMode.EXPLICIT:
            self.model_out_sharding = create_sharding(
                mesh,
                ('batch', 'sequence', 'embed'),
                rules=sharding_rules,
            )
            self.logits_out_sharding = create_sharding(
                mesh,
                ('batch', 'sequence', 'vocab'),
                rules=sharding_rules,
            )

    def __getattr__(self, name):
        if (
            name == 'lm_head'
            and self.__dict__.get('tied_word_embeddings', False)
        ):
            return self.model.embed_tokens
        raise AttributeError(
            f"'{self.__class__.__name__}' object has no attribute '{name}'"
        )

    def __call__(
        self,
        x: jax.Array,
        attention_mask: jax.Array = None,
        ctx: TransformerContext = None,
    ):
        if ctx is not None and not isinstance(ctx, TransformerContext):
            raise TypeError('ctx must be a TransformerContext or None')

        kv_cache = None
        position_idx = None
        is_causal = False
        if ctx is not None:
            position_idx = ctx.position_idx
            is_causal = ctx.is_causal
            if (ctx.key_cache is None) != (ctx.value_cache is None):
                raise ValueError(
                    'TransformerContext must contain both key and value caches'
                )
            if ctx.key_cache is not None:
                kv_cache = (ctx.key_cache, ctx.value_cache)

        x, new_cache = self.model(
            x,
            attention_mask=attention_mask,
            kv_cache=kv_cache,
            position_idx=position_idx,
            is_causal=is_causal,
            out_sharding=self.model_out_sharding,
        )

        if self.tied_word_embeddings:
            weight = self.model.embed_tokens.embedding.value
            logits = jnp.einsum('...d,vd->...v', x, weight)
            if self.logits_out_sharding is not None:
                logits = jax.lax.with_sharding_constraint(
                    logits,
                    self.logits_out_sharding,
                )
        else:
            logits = self.lm_head(
                x,
                out_sharding=self.logits_out_sharding,
            )

        if ctx is not None and new_cache is not None:
            ctx = replace(
                ctx,
                key_cache=new_cache[0],
                value_cache=new_cache[1],
            )

        return logits, ctx

    @classmethod
    def _load_from_pretrained(cls, path_or_repo, config, module_map, **kwargs):
        module_map = module_map or []
        if isinstance(module_map, dict):
            module_map = list(module_map.items())
            
        tied = getattr(config, 'tie_word_embeddings', False)
        
        new_module_map = []
        for rule in module_map:
            if len(rule) == 2:
                source, target = rule
                if tied and target == "embed_tokens.embedding":
                    new_module_map.append((source, ["embed_tokens.embedding", "lm_head.weight"], lambda x: [x, x]))
                    continue

            new_module_map.append(rule)
            
        return super().from_pretrained(path_or_repo, config=config, module_map=new_module_map, **kwargs)

    @classmethod
    def from_pretrained(cls, path_or_repo, mesh=None, sharding_rules=None, local=False, **kwargs):
        # Load config
        if 'config' in kwargs:
            config = kwargs.pop('config')
        else:
            config = ModelConfig.load_config(path_or_repo, local=local)
        
        # We define how HuggingFace weights map to our components using our new Tuple format
        module_map = [
            ("model.embed_tokens.weight", "model.embed_tokens.embedding"),
        ]

        # Call the base class safetensors loader
        # (Note: PretrainedModel.from_pretrained will need to be updated to pass mesh and sharding_rules down!)
        return cls._load_from_pretrained(
            path_or_repo, 
            config, 
            module_map, 
            local=local, 
            mesh=mesh,
            sharding_rules=sharding_rules,
            **kwargs
        )
    
    def _sample(
        self, 
        logits: jax.Array, 
        temperature: float, 
        top_k: int, 
        top_p: float, 
        key: jax.Array
    ) -> jax.Array:
        logits = logits / jnp.maximum(temperature, 1e-5)
        
        if top_k > 0:
            top_k_logits, _ = jax.lax.top_k(logits, top_k)
            min_top_k = top_k_logits[:, -1:]
            logits = jnp.where(logits >= min_top_k, logits, -jnp.inf)
            
        if top_p < 1.0:
            sorted_indices = jnp.argsort(logits, axis=-1)[:, ::-1]
            sorted_logits = jnp.take_along_axis(logits, sorted_indices, axis=-1)
            cumulative_probs = jnp.cumsum(jax.nn.softmax(sorted_logits, axis=-1), axis=-1)
            
            # Remove tokens with cumulative probability above the threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            # Shift the mask to the right to keep the first token that crosses the threshold
            sorted_indices_to_remove = jnp.roll(sorted_indices_to_remove, 1, axis=-1)
            sorted_indices_to_remove = sorted_indices_to_remove.at[:, 0].set(False)
            
            # Map back to original order
            indices_to_remove = jnp.empty_like(sorted_indices_to_remove)
            indices_to_remove = indices_to_remove.at[
                jnp.arange(logits.shape[0])[:, None], sorted_indices
            ].set(sorted_indices_to_remove)
            
            logits = jnp.where(indices_to_remove, -jnp.inf, logits)
            
        return jax.random.categorical(key, logits)[:, None]

    @partial(jax.jit, static_argnames=['max_seq_len', 'top_k', 'top_p'])
    def _decode_step(
        self, carry, 
        max_seq_len: int = None, 
        temperature: float = 1.0, 
        top_k: int = 50, 
        top_p: float = 1.0
    ):
        token, k_cache, v_cache, pos, rng = carry
        
        decode_ctx = TransformerContext(
            key_cache=k_cache,
            value_cache=v_cache,
            position_idx=pos,
            is_causal=False
        )
        
        # Mask to attend to all past tokens up to pos
        mask = jnp.arange(max_seq_len) <= pos
        mask = mask.reshape(1, 1, 1, max_seq_len)
        
        step_logits, decode_ctx = self(token, attention_mask=mask, ctx=decode_ctx)
        
        rng, subkey = jax.random.split(rng)
        next_t = self._sample(step_logits[:, -1, :], temperature, top_k, top_p, subkey)
        
        return (
            next_t,
            decode_ctx.key_cache,
            decode_ctx.value_cache,
            pos + 1,
            rng,
        ), next_t

    def generate(
        self, 
        input_ids: jax.Array, 
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 1.0,
        key: jax.Array = None
    ) -> jax.Array:
        if key is None:
            key = jax.random.key(42)
            
        batch_size, seq_len = input_ids.shape
        max_seq_len = seq_len + max_new_tokens

        assert (num_layers := self.config.num_hidden_layers) is not None, \
            'Cannot specified `num_hidden_layers` for key value cache generation.'

        assert (num_attention_heads := self.config.num_attention_heads) is not None, \
            'Cannot specified `num_attention_heads` for key value cache generation.'
            
        assert (num_kv_heads := self.config.num_key_value_heads) is not None, \
            'Cannot specified `num_key_value_heads` for key value cache generation.'
            
        assert (hidden_size := self.config.hidden_size) is not None, \
            'Cannot specified `head_dim` for key value cache generation'
            
        head_dim = (
            getattr(self.config, 'head_dim', None)
            or hidden_size // num_attention_heads
        )
        
        # Initialize KV Cache with the model's actual dtype (e.g. bfloat16)
        leaves = jax.tree_util.tree_leaves(self)
        arrays = [leaf for leaf in leaves if getattr(leaf, 'dtype', None) is not None]
        model_dtype = arrays[0].dtype if arrays else jnp.float32
        
        k_cache = jnp.zeros((num_layers, batch_size, max_seq_len, num_kv_heads, head_dim), dtype=model_dtype)
        v_cache = jnp.zeros((num_layers, batch_size, max_seq_len, num_kv_heads, head_dim), dtype=model_dtype)
        
        # Prefill phase
        position_idx = jnp.array(0, dtype=jnp.int32)
        ctx = TransformerContext(
            key_cache=k_cache,
            value_cache=v_cache,
            position_idx=position_idx,
            is_causal=True # JAX native dot_product_attention handles causal masking if True
        )
        
        logits, ctx = self(input_ids, attention_mask=None, ctx=ctx)
        next_token_logits = logits[:, -1, :]
        
        key, subkey = jax.random.split(key)
        next_token = self._sample(next_token_logits, temperature, top_k, top_p, subkey)
        
        # 3. Decoding phase
        def scan_decode_step(carry, _):
            return self._decode_step(
                carry, 
                max_seq_len=max_seq_len, 
                temperature=temperature, 
                top_k=top_k, 
                top_p=top_p
            )
            
        initial_pos = jnp.array(seq_len, dtype=jnp.int32)
        initial_carry = (
            next_token,
            ctx.key_cache,
            ctx.value_cache,
            initial_pos,
            key,
        )
        _, new_tokens = jax.lax.scan(scan_decode_step, initial_carry, None, length=max_new_tokens - 1)
        
        # new_tokens is shape [max_new_tokens - 1, batch_size, 1] -> swap to [batch_size, max_new_tokens - 1]
        new_tokens = new_tokens.swapaxes(0, 1).reshape(batch_size, -1)
        
        return jnp.concatenate([input_ids, next_token, new_tokens], axis=1)


class TransformerMM(PretrainedModel):
    def __init__(self):
        raise NotImplementedError(f'There is a plan to implement {self.__class__.__name__}.')


class DiffusionIM(PretrainedModel):
    def __init__(
        self,
        **kwargs
    ):
        raise NotImplementedError(f'There is a plan to implement {self.__class__.__name__}.')


class DiffusionLM(PretrainedModel):
    def __init__(self):
        raise NotImplementedError(f'There is a plan to implement {self.__class__.__name__}.')


__all__ = [
    'TransformerContext',
    'TransformerDecoderLayer',
    'TransformerModel',
    'TransformerCausalLM',
    'TransformerMM',
    'DiffusionLM',
    'DiffusionIM',
]
