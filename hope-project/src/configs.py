"""
configs.py — Fully parametric HOPE configuration with three hardware-tuned presets.

95% Frozen Core / 5% Plastic split is encoded here:
  - `core_params`: embedding, CMS layers, norm_fast, head (slow LR 3e-4)
  - `plastic_params`: fast_memory projections + gates (fast LR 1e-4)

All sizing is derived from (d_model, num_layers, vocab_size, seq_len, fast_ratio).
"""

from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import List, Tuple, Optional


@dataclass
class HOPEConfig:
    """Fully parametric HOPE configuration."""

    # --- Architecture ---
    d_model: int = 512
    num_layers: int = 16
    vocab_size: int = 50257
    seq_len: int = 512
    ffn_expansion: int = 4
    dropout: float = 0.1

    # --- Fast memory ---
    fast_chunk_size: int = 64
    fast_force_loop: bool = False

    # --- 95/5 split ---
    fast_ratio: float = 0.05  # target fraction of params in plastic group

    # --- Training ---
    micro_batch_size: int = 8
    gradient_accumulation_steps: int = 8
    core_lr: float = 3e-4       # 95% frozen core learning rate
    plastic_lr: float = 1e-4    # 5% plastic fast-weight learning rate
    weight_decay: float = 0.1
    betas: Tuple[float, float] = (0.9, 0.95)
    grad_clip: float = 1.0
    warmup_steps: int = 500
    max_steps: int = 10000
    checkpoint_every: int = 500

    # --- Hardware ---
    dtype: str = "bfloat16"  # torch dtype string

    # --- Residual clamp ---
    residual_clamp: float = 1.0  # Hard Residual Clamp bound for tanh gate

    # --- Derived (computed post-init) ---
    _total_params: Optional[int] = field(default=None, repr=False)
    _core_params: Optional[int] = field(default=None, repr=False)
    _plastic_params: Optional[int] = field(default=None, repr=False)

    def __post_init__(self):
        self._compute_param_counts()

    def _compute_param_counts(self):
        D = self.d_model
        V = self.vocab_size
        L = self.num_layers
        E = self.ffn_expansion

        # Embedding: V * D (tied with head weight, so count once for params, head bias = V)
        emb_params = V * D

        # Fast memory (SelfModifyingLayer) — the PLASTIC group:
        #   proj_q: D*D + D,  proj_k: D*D + D,  proj_v: D*D + D,  proj_out: D*D + D
        #   gate_alpha: D + 1,  gate_beta: D + 1
        fast_proj_params = 4 * (D * D + D)  # q, k, v, out projections
        fast_gate_params = 2 * (D + 1)       # alpha, beta gates
        fast_total = fast_proj_params + fast_gate_params

        # norm_fast: 2 * D (weight + bias)
        norm_fast_params = 2 * D

        # CMS layers: each has FFN(D → D*E → D) + LayerNorm(D)
        #   FFN up: D * (D*E) + D*E
        #   FFN down: (D*E) * D + D
        #   LN: 2 * D
        cms_per_layer = D * (D * E) + (D * E) + (D * E) * D + D + 2 * D
        cms_total = L * cms_per_layer

        # Head: D * V + V (weight + bias) — if tied, weight is shared with embedding
        head_params = V * D + V  # we keep head separate (untied) for sizing clarity

        # Totals
        self._plastic_params = fast_total
        self._core_params = emb_params + norm_fast_params + cms_total + head_params
        self._total_params = self._core_params + self._plastic_params

    @property
    def total_params(self) -> int:
        return self._total_params

    @property
    def core_params(self) -> int:
        return self._core_params

    @property
    def plastic_params(self) -> int:
        return self._plastic_params

    @property
    def actual_fast_ratio(self) -> float:
        return self._plastic_params / self._total_params

    @property
    def effective_batch_size(self) -> int:
        return self.micro_batch_size * self.gradient_accumulation_steps

    def summary(self) -> str:
        lines = [
            f"=== {self.preset_name} Configuration ===",
            f"  d_model:     {self.d_model}",
            f"  num_layers:  {self.num_layers}",
            f"  vocab_size:  {self.vocab_size:,}",
            f"  seq_len:     {self.seq_len}",
            f"  ffn_expand:  {self.ffn_expansion}x",
            f"  dtype:       {self.dtype}",
            f"  ---",
            f"  Total params:   {self._total_params:>12,}  ({self._total_params/1e6:.1f}M)",
            f"  Core (95%):     {self._core_params:>12,}  ({self._core_params/1e6:.1f}M)",
            f"  Plastic (5%):   {self._plastic_params:>12,}  ({self._plastic_params/1e6:.1f}M)",
            f"  Actual ratio:   {self.actual_fast_ratio:.2%}",
            f"  ---",
            f"  Effective batch: {self.effective_batch_size} ({self.micro_batch_size} × {self.gradient_accumulation_steps})",
            f"  Core LR:        {self.core_lr}",
            f"  Plastic LR:     {self.plastic_lr}",
            f"  Residual clamp: {self.residual_clamp}",
        ]
        return "\n".join(lines)

    @property
    def preset_name(self) -> str:
        total_m = self._total_params / 1e6
        if total_m < 20:
            return "dev_15m"
        elif total_m < 300:
            return "local_200m"
        else:
            return "prod_500m"


# ============================================================================
# Hardware-Tuned Presets
# ============================================================================

def dev_15m() -> HOPEConfig:
    """~15M params, vocab_size=8192, seq_len=512.
    10-minute sanity check for end-to-end training verification.
    Compact BPE tokenizer keeps embedding table at ~2M params.
    """
    return HOPEConfig(
        d_model=256,
        num_layers=8,
        vocab_size=8192,
        seq_len=512,
        ffn_expansion=4,
        dropout=0.1,
        micro_batch_size=8,
        gradient_accumulation_steps=4,
        core_lr=3e-4,
        plastic_lr=1e-4,
        max_steps=2000,
        warmup_steps=200,
        checkpoint_every=500,
        residual_clamp=1.0,
    )


def local_200m() -> HOPEConfig:
    """~200M total params (~190M frozen / ~10M plastic), GPT-2 tokenizer.
    Overnight single-GPU run on RTX 3060 12GB.
    """
    return HOPEConfig(
        d_model=768,
        num_layers=16,
        vocab_size=50257,
        seq_len=512,
        ffn_expansion=4,
        dropout=0.1,
        micro_batch_size=8,
        gradient_accumulation_steps=8,
        core_lr=3e-4,
        plastic_lr=1e-4,
        max_steps=20000,
        warmup_steps=1000,
        checkpoint_every=500,
        residual_clamp=1.0,
    )


def prod_500m() -> HOPEConfig:
    """~500M total params (~475M frozen / ~25M plastic), GPT-2 tokenizer.
    Full-scale architecture for 10B–20B token runs.
    """
    return HOPEConfig(
        d_model=1024,
        num_layers=24,
        vocab_size=50257,
        seq_len=1024,
        ffn_expansion=4,
        dropout=0.1,
        micro_batch_size=4,
        gradient_accumulation_steps=16,
        core_lr=3e-4,
        plastic_lr=1e-4,
        max_steps=50000,
        warmup_steps=2000,
        checkpoint_every=500,
        residual_clamp=1.0,
    )


PRESETS = {
    "dev_15m": dev_15m,
    "local_200m": local_200m,
    "prod_500m": prod_500m,
}


def get_config(name: str = "dev_15m") -> HOPEConfig:
    """Get a preset config by name, or raise ValueError."""
    if name not in PRESETS:
        raise ValueError(f"Unknown preset '{name}'. Available: {list(PRESETS.keys())}")
    return PRESETS[name]()


if __name__ == "__main__":
    for name, factory in PRESETS.items():
        cfg = factory()
        print(cfg.summary())
        print()
