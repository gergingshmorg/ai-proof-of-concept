# STATE_MAPPING.md — HOPE Architecture: Stateful vs Static Weight Mapping

> How `obekt/HOPE-nested-learning` separates **stateful test-time fast-memory tensors**
> (updated inside the forward pass via the Delta Rule / inner-loop gradient descent)
> from **static `nn.Parameter` weights** (updated only by the outer optimizer).

---

## 1. Architecture Overview

The HOPE model (`train_hope.py::HOPE`) has four component groups:

| Component | Class | Role |
|---|---|---|
| **Embedding** | `nn.Embedding(vocab_size, d_model)` | Token → vector lookup |
| **Fast Memory** | `SelfModifyingLayer(d_model)` | Inner-loop delta-rule memory — the only recurrent/stateful component |
| **CMS Layers** | `ModuleList[ContinuumMemoryBlock]` × `n_layers` | Position-wise FFN blocks (no cross-position mixing) |
| **Head** | `nn.Linear(d_model, vocab_size)` | Final logit projection (tied embeddings optional) |

---

## 2. The Two Weight Categories

### 2A. Static `nn.Parameter` Weights (Outer-Loop / Optimizer-Updated)

These are standard PyTorch parameters. They are **never modified inside `forward()`**.
The outer optimizer (AdamW) updates them at frequencies controlled by `cms_tiers`.

| State Dict Key Pattern | Shape | Component | CMS Tier |
|---|---|---|---|
| `embedding.weight` | `[vocab_size, d_model]` | Embedding | Tier 0 (fastest) |
| `fast_memory.proj_q.{weight,bias}` | `[d_model, d_model]` / `[d_model]` | Fast Memory Q projection | Tier 0 |
| `fast_memory.proj_k.{weight,bias}` | `[d_model, d_model]` / `[d_model]` | Fast Memory K projection | Tier 0 |
| `fast_memory.proj_v.{weight,bias}` | `[d_model, d_model]` / `[d_model]` | Fast Memory V projection | Tier 0 |
| `fast_memory.proj_out.{weight,bias}` | `[d_model, d_model]` / `[d_model]` | Fast Memory output projection | Tier 0 |
| `fast_memory.gate_alpha.{weight,bias}` | `[1, d_model]` / `[1]` | Per-token forget gate (α) | Tier 0 |
| `fast_memory.gate_beta.{weight,bias}` | `[1, d_model]` / `[1]` | Per-token inner learning rate (β) | Tier 0 |
| `norm_fast.{weight,bias}` | `[d_model]` | LayerNorm after fast memory residual | Tier 0 |
| `cms_layers.{i}.net.0.{weight,bias}` | `[d_model*4, d_model]` / `[d_model*4]` | CMS block `i` FFN up-projection | Tier by layer index |
| `cms_layers.{i}.net.2.{weight,bias}` | `[d_model, d_model*4]` / `[d_model]` | CMS block `i` FFN down-projection | Tier by layer index |
| `cms_layers.{i}.norm.{weight,bias}` | `[d_model]` | CMS block `i` LayerNorm | Tier by layer index |
| `head.{weight,bias}` | `[vocab_size, d_model]` / `[vocab_size]` | Output head | Tier 0 |

#### CMS Tier Assignment (default `cms_tiers = [[8,1], [5,4], [3,16]]` with `n_layers=16`):

| Tier | CMS Layer Indices | Update Period | Description |
|---|---|---|---|
| **Tier 0 (Fast)** | 0–7 + embedding, fast_memory, norm_fast, head | Every 1 step | Tracks data at highest frequency |
| **Tier 1 (Medium)** | 8–12 | Every 4 steps | Gradients averaged over 4 steps |
| **Tier 2 (Slow)** | 13–15 | Every 16 steps | Consolidates slowly, most stable |

### 2B. Stateful Test-Time Fast Memory (Forward-Pass / Inner-Loop Updated)

The **memory matrix `M`** is the sole stateful tensor. It is:

- **NOT an `nn.Parameter`** — it is never in `model.state_dict()` or touched by the optimizer
- **Created dynamically** in `SelfModifyingLayer.forward()` as `torch.zeros(B, D, D)` when no prior state exists
- **Updated every token** via the Delta Rule inner-loop gradient descent step
- **Passed between calls** as the `state` argument for autoregressive generation
- **Always `torch.float32`** regardless of model weight dtype (fp32 accumulation over full context)

#### The Delta Rule Recurrence (per token `t`):

```
M_t = α_t · M_{t-1} + β_t · k_t^T · (v_t − k_t · M_{t-1})
                                       ╰── prediction error ──╯
```

Where:
- `M` : `[B, d_model, d_model]` — the associative memory matrix (the **stateful tensor**)
- `α_t` : `[B, 1, 1]` — sigmoid forget gate (from `gate_alpha` **nn.Parameter** + bias +4.0)
- `β_t` : `[B, 1, 1]` — sigmoid inner learning rate (from `gate_beta` **nn.Parameter** − bias 2.0)
- `q_t, k_t, v_t` : `[B, 1, d_model]` — from `proj_q/k/v` **nn.Parameter** projections
- `k` is L2-normalized before use (`F.normalize(k, dim=-1)`)
- Read-out: `o_t = q_t · M_{t-1}` (read **before** the write at step `t`)

#### Implementation Paths:

| Sequence Length | Path | Location |
|---|---|---|
| `T > 1` (training/prefill) | Chunk-parallel exact scan (`_chunked_delta_scan`) | `train_hope.py:98–158` |
| `T == 1` (generation step) | Per-token loop (`_forward_loop`) | `train_hope.py:189–216` |
| `T == 0` | Early return (zeros output, state passed through) | `train_hope.py:242–243` |

---

## 3. Data Flow Diagram

```
Input token IDs [B, T]
        │
        ▼
   nn.Embedding (STATIC nn.Parameter)
        │
        ▼ h [B, T, D]
   ┌────────────────────────────────────┐
   │  SelfModifyingLayer                │
   │  ┌──────────────────────────────┐  │
   │  │ proj_q/k/v (STATIC Params)  │  │
   │  │ gate_alpha/beta (STATIC)     │  │
   │  │          │                   │  │
   │  │          ▼                   │  │
   │  │  Delta Rule Recurrence       │  │
   │  │  M_t = α·M + β·k^T·error   │  │
   │  │  ╰── STATEFUL [B,D,D] ──╯   │  │
   │  │          │                   │  │
   │  │  proj_out (STATIC Param)     │  │
   │  └──────────────────────────────┘  │
   └────────────────────────────────────┘
        │ fast_out [B, T, D]
        ▼
   h = LayerNorm(h + fast_out)   (norm_fast: STATIC)
        │
        ▼
   CMS Layers 0..N-1 (all STATIC nn.Parameters)
   Each: FFN(GELU) + Dropout + LayerNorm residual
        │
        ▼
   nn.Linear head (STATIC nn.Parameter)
        │
        ▼
   logits [B, T, vocab_size]
```

---

## 4. Blueprint V2.7 Mapping: 95% Frozen Core / 5% Plastic Split

For our project, we re-partition HOPE's weights into two groups:

| Group | What It Contains | Training Behavior |
|---|---|---|
| **95% Frozen Core** | CMS layers (FFN blocks), embedding, norm_fast, head | Hard-frozen during `/study`; slow-learning `3e-4` during pre-training |
| **5% Plastic Memory** | `fast_memory.proj_q/k/v/out`, `gate_alpha/beta` + their associated biases | Always trainable; `1e-4` learning rate |

### Stateful Tensor in Cartridge `.pt` Files

The memory matrix `M` is **also saved in cartridges** because after a `/study` session
the fast memory has been written to by the delta rule during the consolidation mini-training.
The cartridge format saves:
1. The 5% plastic `nn.Parameter` deltas (weight diffs from base)
2. The accumulated memory matrix `M` state from the study session
3. BM25 index + cleaned text chunks

---

## 5. Key Invariants

1. **`model.state_dict()` contains ONLY static `nn.Parameter` weights** — never the memory matrix `M`
2. **`M` is created fresh as zeros** at the start of each new sequence / conversation
3. **`M` accumulates in `float32`** even when model weights are in `bfloat16`
4. **The chunked scan and per-token loop produce identical results** (verified to `~1e-13` in float64)
5. **CMS blocks are purely position-wise** (FFN + LayerNorm) — only `SelfModifyingLayer` integrates across time
6. **Padding masks** prevent memory corruption: masked tokens set `α=1, β=0` (memory passes through unchanged)
