"""
model.py — HOPE architecture with 95% Frozen Core / 5% Plastic split,
Hard Residual Clamp, and @torch.compile fallback for sequential paths.

Based on obekt/HOPE-nested-learning (SelfModifyingLayer + ContinuumMemoryBlock),
re-engineered for Blueprint V2.7.
"""

from __future__ import annotations
import math
import sys
from typing import Optional, Tuple, Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as grad_checkpoint

from configs import HOPEConfig


# ============================================================================
# Triangular solve probe (cached per device)
# ============================================================================

_TRI_SOLVE_OK: dict = {}

def _tri_solve_ok(device: torch.device, dtype: torch.dtype = torch.float32) -> bool:
    key = (str(device), dtype)
    if key not in _TRI_SOLVE_OK:
        try:
            a = torch.eye(2, device=device, dtype=dtype)
            b = torch.ones(2, 1, device=device, dtype=dtype)
            torch.linalg.solve_triangular(a, b, upper=False, unitriangular=True)
            _TRI_SOLVE_OK[key] = True
        except Exception:
            _TRI_SOLVE_OK[key] = False
    return _TRI_SOLVE_OK[key]


# ============================================================================
# Chunk-Parallel Delta Scan (exact, from obekt/HOPE-nested-learning)
# ============================================================================

def _chunked_delta_scan(
    q: torch.Tensor,       # [B, T, D]
    k: torch.Tensor,       # [B, T, D]
    v: torch.Tensor,       # [B, T, D]
    log_alpha: torch.Tensor,  # [B, T, 1]
    beta: torch.Tensor,       # [B, T, 1]
    memory: torch.Tensor,     # [B, D, D]
    mask: Optional[torch.Tensor],  # [B, T] float or None
    chunk_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Exact chunk-parallel evaluation of the gated delta-rule recurrence.

    M_t = alpha_t * M_{t-1} + beta_t * k_t^T (v_t - k_t M_{t-1})
    o_t = q_t M_{t-1}   (read-out BEFORE the write at t)

    Mathematically identical to the per-token loop.
    """
    batch_size, seq_len, _ = q.shape
    al = log_alpha.squeeze(-1)  # [B, T]
    be = beta.squeeze(-1)       # [B, T]
    if mask is not None:
        al = al * mask
        be = be * mask

    outs = []
    for start in range(0, seq_len, chunk_size):
        end = min(start + chunk_size, seq_len)
        qc, kc, vc = q[:, start:end], k[:, start:end], v[:, start:end]
        alc, bec = al[:, start:end], be[:, start:end]
        length = end - start

        g = torch.cumsum(alc, dim=1)
        gs = g - alc
        decay = torch.exp(gs.unsqueeze(2) - g.unsqueeze(1))
        kkT = torch.bmm(kc, kc.transpose(1, 2))
        A = torch.tril(decay * kkT * bec.unsqueeze(1), diagonal=-1)
        eye = torch.eye(length, device=A.device, dtype=A.dtype)

        rhs = vc - torch.bmm(kc * torch.exp(gs).unsqueeze(-1), memory)
        E = torch.linalg.solve_triangular(eye + A, rhs, upper=False, unitriangular=True)

        qkT = torch.bmm(qc, kc.transpose(1, 2))
        o = (torch.bmm(qc * torch.exp(gs).unsqueeze(-1), memory)
             + torch.bmm(torch.tril(decay * qkT, diagonal=-1), bec.unsqueeze(-1) * E))
        outs.append(o)

        gL = g[:, -1].unsqueeze(1)
        wK = kc * torch.exp(gL - g).unsqueeze(-1) * bec.unsqueeze(-1)
        memory = torch.exp(gL).unsqueeze(-1) * memory + torch.bmm(wK.transpose(1, 2), E)

    return torch.cat(outs, dim=1), memory


# ============================================================================
# SelfModifyingLayer — Fast-weight memory with Delta Rule inner loop
# ============================================================================

class SelfModifyingLayer(nn.Module):
    """Fast-weight memory trained by an inner-loop delta rule.

    The memory matrix M is the STATEFUL tensor:
    - NOT an nn.Parameter (never in state_dict)
    - Created as zeros in forward() when no state is passed
    - Updated every token via delta rule
    - Always float32 regardless of model dtype

    All projections (proj_q/k/v/out, gate_alpha/beta) are the PLASTIC nn.Parameters.
    """

    def __init__(self, dim: int, chunk_size: int = 64, force_loop: bool = False):
        super().__init__()
        self.dim = dim
        self.chunk_size = chunk_size
        self.force_loop = force_loop

        self.proj_q = nn.Linear(dim, dim)
        self.proj_k = nn.Linear(dim, dim)
        self.proj_v = nn.Linear(dim, dim)
        self.proj_out = nn.Linear(dim, dim)

        self.gate_alpha = nn.Linear(dim, 1)
        self.gate_beta = nn.Linear(dim, 1)

    def _forward_loop(self, q, k, v, alpha, beta, memory, mask):
        """Per-token recurrence for T==1 generation steps."""
        batch_size, seq_len, _ = q.shape
        outputs = []
        for t in range(seq_len):
            q_t = q[:, t, :].unsqueeze(1)
            k_t = k[:, t, :].unsqueeze(1)
            v_t = v[:, t, :].unsqueeze(1)
            a_t = alpha[:, t, :].unsqueeze(1)
            b_t = beta[:, t, :].unsqueeze(1)

            read_out = torch.bmm(q_t, memory)
            pred = torch.bmm(k_t, memory)
            error = v_t - pred
            update = b_t * torch.bmm(k_t.transpose(1, 2), error)
            new_memory = a_t * memory + update

            if mask is not None:
                m_t = mask[:, t].view(batch_size, 1, 1)
                memory = (1 - m_t) * memory + m_t * new_memory
            else:
                memory = new_memory

            outputs.append(read_out)

        return torch.cat(outputs, dim=1), memory

    def forward(
        self, x: torch.Tensor, state: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        q, k, v = self.proj_q(x), self.proj_k(x), self.proj_v(x)
        k = F.normalize(k, dim=-1)
        batch_size, seq_len, _ = x.shape

        z_alpha = self.gate_alpha(x) + 4.0
        beta = torch.sigmoid(self.gate_beta(x) - 2.0)

        # Delta-rule scan always in fp32 for numerical stability
        q, k, v = q.float(), k.float(), v.float()
        z_alpha, beta = z_alpha.float(), beta.float()
        if mask is not None:
            mask = mask.float()

        memory = (state if state is not None
                  else torch.zeros(batch_size, self.dim, self.dim,
                                   device=x.device, dtype=torch.float32))

        if seq_len == 0:
            return torch.zeros_like(x), memory

        use_chunked = (
            seq_len > 1
            and self.chunk_size
            and not self.force_loop
            and _tri_solve_ok(x.device, torch.float32)
        )

        if use_chunked:
            out, memory = _chunked_delta_scan(
                q, k, v, F.logsigmoid(z_alpha), beta, memory, mask, self.chunk_size)
        else:
            alpha = torch.sigmoid(z_alpha)
            out, memory = self._forward_loop(q, k, v, alpha, beta, memory, mask)

        return self.proj_out(out.to(self.proj_out.weight.dtype)), memory


# ============================================================================
# ContinuumMemoryBlock — Position-wise FFN + LayerNorm residual
# ============================================================================

class ContinuumMemoryBlock(nn.Module):
    """Single CMS (Continuum Memory System) block.
    Pure position-wise: FFN + residual + LayerNorm. No cross-position mixing.
    """

    def __init__(self, dim: int, expansion: int = 4, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * expansion),
            nn.GELU(),
            nn.Linear(dim * expansion, dim),
            nn.Dropout(dropout),
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x + self.net(x))


# ============================================================================
# Hard Residual Clamp
# ============================================================================

class HardResidualClamp(nn.Module):
    """Wraps fast-weight output: output = x + clamp * tanh(fast_out / clamp).

    Ensures plastic layers steer context without overpowering the frozen core.
    The clamp bound limits the L-infinity norm of the fast contribution.
    """

    def __init__(self, clamp_bound: float = 1.0):
        super().__init__()
        self.clamp_bound = clamp_bound

    def forward(self, residual: torch.Tensor, fast_out: torch.Tensor) -> torch.Tensor:
        clamped = self.clamp_bound * torch.tanh(fast_out / max(self.clamp_bound, 1e-6))
        return residual + clamped


# ============================================================================
# HOPE Model
# ============================================================================

class HOPE(nn.Module):
    """HOPE architecture with 95% Frozen Core / 5% Plastic split.

    Components:
    - Embedding (core)
    - SelfModifyingLayer (plastic) + HardResidualClamp
    - ContinuumMemoryBlocks (core)
    - Output head (core, optionally tied with embedding)

    The forward pass supports:
    - Full sequence (training/prefill)
    - Token-by-token with state passing (generation)
    - last_only mode for efficient prefill
    - Gradient checkpointing for memory savings
    """

    def __init__(self, config: HOPEConfig, tie_embeddings: bool = True):
        super().__init__()
        self.config = config
        self.d_model = config.d_model
        self.vocab_size = config.vocab_size

        # Embedding
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)

        # Fast memory (plastic)
        self.fast_memory = SelfModifyingLayer(
            config.d_model,
            chunk_size=config.fast_chunk_size,
            force_loop=config.fast_force_loop,
        )
        self.norm_fast = nn.LayerNorm(config.d_model)

        # Hard Residual Clamp
        self.residual_clamp = HardResidualClamp(config.residual_clamp)

        # CMS layers (core)
        self.cms_layers = nn.ModuleList([
            ContinuumMemoryBlock(config.d_model, config.ffn_expansion, config.dropout)
            for _ in range(config.num_layers)
        ])

        # Output head
        self.head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Tie embeddings
        if tie_embeddings:
            self.head.weight = self.embedding.weight

        self._init_weights()

        # Gradient checkpointing flag
        self._gradient_checkpointing = False

        # Try torch.compile for sequential step path
        self._compiled_step = None

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def enable_gradient_checkpointing(self):
        """Enable gradient checkpointing for CMS layers during training."""
        self._gradient_checkpointing = True

    def disable_gradient_checkpointing(self):
        self._gradient_checkpointing = False

    def core_parameters(self) -> List[nn.Parameter]:
        """Return the 95% frozen core parameters (slow LR group)."""
        core_params = []
        core_params.extend(self.embedding.parameters())
        core_params.extend(self.norm_fast.parameters())
        for layer in self.cms_layers:
            core_params.extend(layer.parameters())
        # head.weight is tied to embedding.weight, so skip if tied
        if self.head.weight is not self.embedding.weight:
            core_params.extend(self.head.parameters())
        elif self.head.bias is not None:
            core_params.append(self.head.bias)
        return core_params

    def plastic_parameters(self) -> List[nn.Parameter]:
        """Return the 5% plastic fast-weight parameters (fast LR group)."""
        return list(self.fast_memory.parameters())

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        last_only: bool = False,
        pad_token_id: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: token IDs [B, T]
            state: fast-memory state [B, D, D] or None
            mask: padding mask [B, T] float (1=real, 0=pad), or None (auto from pad_token_id)
            last_only: only compute logits for last position (efficient prefill)
            pad_token_id: used for auto-mask generation

        Returns:
            logits [B, T', vocab_size], new_state [B, D, D]
        """
        if mask is None:
            mask = (x != pad_token_id).float()

        h = self.embedding(x)

        # Fast memory (plastic) with hard residual clamp
        fast_out, new_state = self.fast_memory(h, state=state, mask=mask)
        h = self.norm_fast(self.residual_clamp(h, fast_out))

        if last_only:
            h = h[:, -1:, :]

        # CMS layers (core) with optional gradient checkpointing
        for layer in self.cms_layers:
            if self._gradient_checkpointing and self.training:
                h = grad_checkpoint(layer, h, use_reentrant=False)
            else:
                h = layer(h)

        return self.head(h), new_state

    def try_compile_step(self, device: torch.device) -> bool:
        """Attempt to torch.compile the T=1 generation path.

        Returns True if compilation succeeded, False otherwise.
        Falls back gracefully if Triton/inductor is unavailable (Windows).
        """
        try:
            self._compiled_step = torch.compile(self, dynamic=False)
            # Warmup with a dummy token
            dummy = torch.zeros(1, 1, dtype=torch.long, device=device)
            with torch.no_grad():
                logits, state = self._compiled_step(dummy, state=None, last_only=True)
                for _ in range(2):
                    logits, state = self._compiled_step(dummy, state=state, last_only=True)
            return True
        except Exception as e:
            self._compiled_step = None
            print(f"[INFO] torch.compile unavailable ({type(e).__name__}: {e}); using eager mode")
            return False

    def step(
        self, x: torch.Tensor, state: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Single-token generation step, using compiled path if available."""
        model = self._compiled_step if self._compiled_step is not None else self
        return model(x, state=state, last_only=True, **kwargs)


# ============================================================================
# Factory
# ============================================================================

def build_model(config: HOPEConfig, device: str = "cuda") -> HOPE:
    """Build a HOPE model from config and move to device."""
    model = HOPE(config)
    model = model.to(device)
    return model


def count_parameters(model: HOPE) -> Dict[str, int]:
    """Count parameters by group."""
    core = sum(p.numel() for p in model.core_parameters())
    plastic = sum(p.numel() for p in model.plastic_parameters())
    total = sum(p.numel() for p in model.parameters())
    return {
        "total": total,
        "core": core,
        "plastic": plastic,
        "ratio": plastic / max(total, 1),
    }


if __name__ == "__main__":
    from configs import dev_15m, local_200m, prod_500m

    for name, factory in [("dev_15m", dev_15m), ("local_200m", local_200m), ("prod_500m", prod_500m)]:
        config = factory()
        print(config.summary())
        model = HOPE(config)
        counts = count_parameters(model)
        print(f"  Actual total:   {counts['total']:,}")
        print(f"  Actual core:    {counts['core']:,}")
        print(f"  Actual plastic: {counts['plastic']:,}")
        print(f"  Actual ratio:   {counts['ratio']:.2%}")
        print()
