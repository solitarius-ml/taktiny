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
"""Base module class"""

import jax
import operator
import qwix
from jax.tree_util import register_pytree_node_class

def _format_scaled(value, scale, suffix):
    number = f"{value / scale:.2f}".rstrip('0').rstrip('.')
    return f"{number}{suffix}"

def format_params(size):
    for scale, suffix in (
        (1_000_000_000_000, 'T'),
        (1_000_000_000, 'B'),
        (1_000_000, 'M'),
        (1_000, 'K'),
    ):
        if size >= scale:
            return _format_scaled(size, scale, suffix)
    return f"{size:,}"

def format_bytes(size):
    for scale, suffix in (
        (1024**4, 'TB'),
        (1024**3, 'GB'),
        (1024**2, 'MB'),
        (1024, 'KB'),
    ):
        if size >= scale:
            number = f"{size / scale:.2f}".rstrip('0').rstrip('.')
            return f"{number} {suffix}"
    return f"{int(size)} B"

def format_dtype(dtype):
    name = dtype.name
    if name == 'float32': return 'f32'
    if name == 'float16': return 'f16'
    if name == 'bfloat16': return 'bf16'
    if name == 'int32': return 'i32'
    if name == 'int64': return 'i64'
    return name

def iter_children(obj):
    if not hasattr(obj, '__dict__'): return
    for k, v in obj.__dict__.items():
        if isinstance(v, (Module, Parameter)):
            yield k, v
        elif isinstance(v, (list, tuple)) and all(isinstance(x, (Module, Parameter)) for x in v):
            for i, x in enumerate(v):
                name = str(i) if k == 'layers' else f"{k}.{i}"
                yield name, x

def build_tree_repr(name, obj, prefix="", is_last=True, is_root=False):
    lines = []
    total_params = 0
    total_bytes = 0
    
    current_prefix = "" if is_root else prefix + ("└── " if is_last else "├── ")
    child_prefix = "" if is_root else prefix + ("    " if is_last else "│   ")
    
    if isinstance(obj, Parameter):
        if isinstance(obj.value, qwix.QArray):
            p = obj.value.qvalue.size
            b = sum(
                getattr(leaf, 'nbytes', 0)
                for leaf in jax.tree_util.tree_leaves(obj.value)
            )
            dt = f'qwix[{obj.value.qtype}]'
        else:
            p = obj.value.size
            b = getattr(obj.value, 'nbytes', 0)
            dt = format_dtype(obj.value.dtype)
        sh = ", ".join(map(str, obj.value.shape))
        lines.append(f"{current_prefix}{name}: {dt}[{sh}]")
        return lines, p, b
        
    elif isinstance(obj, Module):
        children_items = list(iter_children(obj))
                        
        child_lines = []
        for i, (c_name, c_obj) in enumerate(children_items):
            c_is_last = (i == len(children_items) - 1)
            c_lines, c_p, c_b = build_tree_repr(c_name, c_obj, child_prefix, c_is_last)
            child_lines.extend(c_lines)
            total_params += c_p
            total_bytes += c_b
            
        extra = obj.extra_repr()
        title = f"{obj.__class__.__name__}({extra})" if extra else obj.__class__.__name__
        
        if is_root:
            node_str = f"{title} ({format_params(total_params)} parameters, {format_bytes(total_bytes)})"
        else:
            node_str = (
                f"{current_prefix}{name}: {title} "
                f"({format_params(total_params)} parameters, "
                f"{format_bytes(total_bytes)})"
            )
            
        lines.insert(0, node_str)
        lines.extend(child_lines)
        return lines, total_params, total_bytes
        
    return [], 0, 0

def _is_dynamic(v):
    if isinstance(v, (Module, Parameter, jax.Array)):
        return True
    if hasattr(jax.numpy, 'ndarray') and isinstance(v, jax.numpy.ndarray):
        return True
    if isinstance(v, (list, tuple)) and len(v) > 0 and all(_is_dynamic(x) for x in v):
        return True
    if isinstance(v, dict) and len(v) > 0 and all(_is_dynamic(x) for x in v.values()):
        return True
    return False

class Module:
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        register_pytree_node_class(cls)

    def extra_repr(self): return ""
    def __repr__(self):
        lines, _, _ = build_tree_repr("", self, is_root=True)
        return "\n".join(lines)

    def tree_flatten(self):
        dynamic_names = []
        dynamic_vals = []
        static_data = {}
        
        for k, v in self.__dict__.items():
            if _is_dynamic(v):
                dynamic_names.append(k)
                dynamic_vals.append(v)
            else:
                static_data[k] = v
                
        return tuple(dynamic_vals), (tuple(dynamic_names), static_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        obj = object.__new__(cls)
        dynamic_names, static_data = aux_data
        
        obj.__dict__.update(static_data)
        for k, v in zip(dynamic_names, children):
            obj.__dict__[k] = v
            
        return obj

    def flat_state_dict(self, prefix=''):
        state = {}
        for name, child in iter_children(self):
            if isinstance(child, Parameter):
                state[prefix + name] = child.value
            elif isinstance(child, Module):
                state.update(child.flat_state_dict(prefix + name + '.'))
        return state
        
    def flat_parameter_dict(self, prefix=''):
        state = {}
        for name, child in iter_children(self):
            if isinstance(child, Parameter):
                state[prefix + name] = child
            elif isinstance(child, Module):
                state.update(child.flat_parameter_dict(prefix + name + '.'))
        return state

    def state_dict(self):
        state = {}
        for name, child in iter_children(self):
            if isinstance(child, Parameter):
                state[name] = child.value
            elif isinstance(child, Module):
                state[name] = child.state_dict()
        return state

    def load_flat_state_dict(self, state, prefix=''):
        for name, child in iter_children(self):
            if isinstance(child, Parameter):
                full_name = prefix + name
                if full_name in state:
                    child.value = state[full_name]
            elif isinstance(child, Module):
                child.load_flat_state_dict(state, prefix + name + '.')

    def load_state_dict(self, state):
        for name, child in iter_children(self):
            if isinstance(child, Parameter):
                if name in state:
                    child.value = state[name]
            elif isinstance(child, Module):
                if name in state:
                    child.load_state_dict(state[name])

class Parameter(Module):
    def __init__(self, array: jax.Array, *, trainable: bool = True):
        self.value = array
        self.trainable = trainable

    def __repr__(self):
        return (
            "Parameter("
            f"shape={getattr(self.value, 'shape', 'None')}, "
            f"dtype={getattr(self.value, 'dtype', 'None')}, "
            f"trainable={self.trainable}"
            ")"
        )

    def __jax_array__(self):
        if isinstance(self.value, qwix.QArray):
            return qwix.dequantize(self.value)
        return self.value

    def __getattr__(self, name):
        return getattr(self.value, name)
        
    def tree_flatten(self):
        aux = {k: v for k, v in self.__dict__.items() if k != 'value'}
        return (self.value,), aux
        
    @classmethod
    def tree_unflatten(cls, aux_data, children):
        obj = object.__new__(cls)
        obj.value = children[0]
        if aux_data:
            for k, v in aux_data.items():
                setattr(obj, k, v)
        return obj

def _make_magic_methods():
    for op in ['add', 'sub', 'mul', 'truediv', 'floordiv', 'mod', 'pow', 'matmul', 
               'eq', 'ne', 'lt', 'le', 'gt', 'ge']:
        magic = f'__{op}__'
        rmagic = f'__r{op}__'
        setattr(Parameter, magic, lambda self, other, o=op: getattr(operator, o)(self.value, other))
        setattr(Parameter, rmagic, lambda self, other, o=op: getattr(operator, o)(other, self.value))
    for op in ['neg', 'pos', 'abs', 'invert']:
        magic = f'__{op}__'
        setattr(Parameter, magic, lambda self, o=op: getattr(operator, o)(self.value))
    
    setattr(Parameter, '__getitem__', lambda self, key: operator.getitem(self.value, key))

_make_magic_methods()
