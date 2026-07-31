# Custom Kernels (`taktiny.kernel`)

TakTiny provides high-performance custom kernels implemented in Pallas, Mosaic, Triton, and JAX custom VJPs for TPU, GPU, and CPU execution.

---

## 1. Attention Kernels (`taktiny.kernel.attention`)

- **FlashAttention** (`flash_attention_block_masked`): Block-sparse flash attention computing online softmax without instantiating full attention matrices.
- **SplashAttention** (`splash_attention` / `attention_reference`): Reference and hardware-accelerated block-sparse attention.
- **Ragged Attention** (`ragged_mqa`, `ragged_mha`, `ragged_gqa`): Attention kernels operating over packed, unpadded sequences.
- **RingAttention** (`ring_attention_kernel`): Sequence-parallel distributed ring attention across device meshes.

### Kernel Classmethod Entry Points (`Attention`)

Call custom attention kernels directly on the `Attention` class:

```python
from taktiny.layers import Attention

# 1. FlashAttention
out_flash = Attention.apply_flash_attention(query, key, value)

# 2. SplashAttention
out_splash = Attention.apply_splash_attention(query, key, value, mask=mask)

# 3. Ragged Attention
out_ragged = Attention.apply_ragged_attention(query, key, value, lengths=lengths)

# 4. Ring Attention
out_ring = Attention.apply_ring_attention(query, key, value, axis_name="seq")
```

---

## 2. Mixture-of-Experts (MoE) Kernels (`taktiny.kernel.megablox`)

- **Grouped Matrix Multiply (GMM)** (`gmm`, `gmm_v2`): Pallas Mosaic TPU v2 and JIT-compatible matrix multiplication for variable-sized expert token groups.
- **Token Activation Routing** (`route`, `unroute`): Custom VJP ops for sorting tokens by expert selection and un-sorting back to original sequence order.

### Kernel Classmethod Entry Points (`MoeFFN` / `GateMLP`)

```python
from taktiny.layers import MoeFFN

# Token routing and sorting
sorted_tokens, group_sizes = MoeFFN.apply_route(tokens, indices, num_groups=8)

# Grouped Matrix Multiply across experts
gmm_out = MoeFFN.apply_gmm(sorted_tokens, expert_weights, group_sizes)

# Un-routing back to token sequence order
unrouted = MoeFFN.apply_unroute(gmm_out, indices)
```

---

## 3. SparseCore & Ragged Gather/Reduce Kernels (`taktiny.kernel.ragged`)

- **SparseCore Gather-Reduce** (`sc_gather_reduce`): TPU SparseCore gather-reduce kernel for sparse embedding lookups and reductions.
- **Ragged Gather** (`ragged_gather`): Gather operations over variable-length sequence buffers.

### Kernel Classmethod Entry Point (`Embedding`)

```python
from taktiny.nn import Embedding

gathered = Embedding.apply(embed_table, indices, kernel="gather_reduce")
```
