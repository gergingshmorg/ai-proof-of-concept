"""
cartridge.py — Smart .pt Cartridge Format & Parallel Stacking Engine (Blueprint V2.7).

Features:
  - Validation header: {"base_arch": "<preset_name>", "version": "2.7", "topic": "..."}
  - Stores BOTH:
      1. 5% plastic nn.Parameter deltas (weight diffs from base) in fp16 + zlib
      2. Stateful test-time associative memory matrix M ([1, D, D]) in fp16 + zlib
  - Built-in BM25 index + raw cleaned text chunks for hybrid retrieval
  - Parallel output stacking: x + w_A * fast_A(x) + w_B * fast_B(x) (no matrix averaging)
"""

from __future__ import annotations
import io
import os
import re
import zlib
import copy
import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from model import HOPE, SelfModifyingLayer


# ============================================================================
# Compression Helpers (fp16 + zlib)
# ============================================================================

def _compress_tensor_dict(tensors: Dict[str, torch.Tensor]) -> bytes:
    """Cast tensors to fp16 (with safe clamping) and compress with zlib."""
    fp16_dict = {}
    for k, v in tensors.items():
        t = v.detach().cpu().float().clamp(-65000.0, 65000.0).half()
        fp16_dict[k] = t
    buf = io.BytesIO()
    torch.save(fp16_dict, buf)
    return zlib.compress(buf.getvalue(), level=6)


def _decompress_tensor_dict(blob: bytes, device: str = "cpu") -> Dict[str, torch.Tensor]:
    """Decompress zlib blob back into a dict of tensors."""
    raw = zlib.decompress(blob)
    buf = io.BytesIO(raw)
    fp16_dict = torch.load(buf, map_location=device, weights_only=True)
    return fp16_dict


# ============================================================================
# Lightweight Self-Contained BM25 Index
# ============================================================================

class SimpleBM25:
    """Fast, pure-Python BM25 retriever serialized cleanly inside .pt cartridges."""

    def __init__(self, chunks: List[str], k1: float = 1.5, b: float = 0.75):
        self.chunks = chunks
        self.k1 = k1
        self.b = b
        self.doc_freqs: Dict[str, int] = {}
        self.doc_lens: List[int] = []
        self.term_freqs: List[Dict[str, int]] = []

        for chunk in chunks:
            tokens = self._tokenize(chunk)
            self.doc_lens.append(len(tokens))
            tf: Dict[str, int] = {}
            for tok in tokens:
                tf[tok] = tf.get(tok, 0) + 1
            self.term_freqs.append(tf)
            for tok in tf:
                self.doc_freqs[tok] = self.doc_freqs.get(tok, 0) + 1

        self.avgdl = sum(self.doc_lens) / max(len(self.doc_lens), 1)
        self.n_docs = len(chunks)

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return re.findall(r"[a-z0-9_]+", text.lower())

    def search(self, query: str, top_k: int = 2) -> List[Tuple[float, str]]:
        if self.n_docs == 0:
            return []
        q_tokens = self._tokenize(query)
        scores = [0.0] * self.n_docs

        for tok in q_tokens:
            df = self.doc_freqs.get(tok, 0)
            if df == 0:
                continue
            idf = math.log((self.n_docs - df + 0.5) / (df + 0.5) + 1.0)
            for idx, tf_map in enumerate(self.term_freqs):
                tf = tf_map.get(tok, 0)
                if tf == 0:
                    continue
                dl = self.doc_lens[idx]
                denom = tf + self.k1 * (1.0 - self.b + self.b * (dl / max(self.avgdl, 1e-6)))
                scores[idx] += idf * (tf * (self.k1 + 1.0)) / denom

        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        return [(score, self.chunks[idx]) for idx, score in ranked[:top_k] if score > 0.0]

    def to_dict(self) -> dict:
        return {
            "chunks": self.chunks,
            "k1": self.k1,
            "b": self.b,
            "doc_freqs": self.doc_freqs,
            "doc_lens": self.doc_lens,
            "term_freqs": self.term_freqs,
            "avgdl": self.avgdl,
            "n_docs": self.n_docs,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SimpleBM25":
        obj = cls.__new__(cls)
        obj.chunks = data["chunks"]
        obj.k1 = data["k1"]
        obj.b = data["b"]
        obj.doc_freqs = data["doc_freqs"]
        obj.doc_lens = data["doc_lens"]
        obj.term_freqs = data["term_freqs"]
        obj.avgdl = data["avgdl"]
        obj.n_docs = data["n_docs"]
        return obj


# ============================================================================
# Cartridge Save & Load
# ============================================================================

def save_cartridge(
    path: str,
    base_model: HOPE,
    studied_model: HOPE,
    memory_state: Optional[torch.Tensor],
    chunks: List[str],
    topic: str,
) -> str:
    """Save a .pt cartridge containing plastic parameter deltas, memory state M,
    BM25 index, cleaned chunks, and Blueprint V2.7 validation header.
    """
    base_sd = base_model.fast_memory.state_dict()
    studied_sd = studied_model.fast_memory.state_dict()

    deltas: Dict[str, torch.Tensor] = {}
    for key in studied_sd:
        deltas[key] = studied_sd[key].detach().cpu().float() - base_sd[key].detach().cpu().float()

    if memory_state is None:
        d = studied_model.d_model
        memory_state = torch.zeros(1, d, d, dtype=torch.float32)
    else:
        memory_state = memory_state.detach().cpu().float()
        if memory_state.dim() == 3 and memory_state.shape[0] > 1:
            memory_state = memory_state.mean(dim=0, keepdim=True)

    bm25 = SimpleBM25(chunks)

    payload = {
        "header": {
            "base_arch": studied_model.config.preset_name,
            "version": "2.7",
            "topic": topic,
            "d_model": studied_model.config.d_model,
            "num_layers": studied_model.config.num_layers,
            "vocab_size": studied_model.config.vocab_size,
        },
        "plastic_deltas_zlib": _compress_tensor_dict(deltas),
        "memory_state_zlib": _compress_tensor_dict({"M": memory_state}),
        "bm25": bm25.to_dict(),
        "chunks": chunks,
    }

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(payload, path)
    return path


def load_cartridge(path: str, expected_arch: Optional[str] = None, device: str = "cpu") -> dict:
    """Load and validate a .pt cartridge."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Cartridge not found: {path}")

    payload = torch.load(path, map_location="cpu", weights_only=False)
    header = payload.get("header", {})

    if header.get("version") != "2.7":
        raise ValueError(f"Invalid cartridge version: {header.get('version')} (expected '2.7')")
    if expected_arch and header.get("base_arch") != expected_arch:
        raise ValueError(
            f"Architecture mismatch: cartridge is '{header.get('base_arch')}', "
            f"active model is '{expected_arch}'"
        )

    deltas_fp16 = _decompress_tensor_dict(payload["plastic_deltas_zlib"], device=device)
    mem_fp16 = _decompress_tensor_dict(payload["memory_state_zlib"], device=device)

    memory_state = mem_fp16["M"].float().to(device)
    bm25 = SimpleBM25.from_dict(payload["bm25"])

    return {
        "header": header,
        "deltas": deltas_fp16,
        "memory_state": memory_state,
        "bm25": bm25,
        "chunks": payload.get("chunks", []),
    }


# ============================================================================
# Parallel Cartridge Stacking Engine (No Matrix Averaging)
# ============================================================================

class StackedCartridgeRuntime:
    """Runs one or more loaded cartridges in parallel alongside the frozen core:
        output = x + sum_i (w_i * fast_i(x))

    Each cartridge maintains its own instantiated SelfModifyingLayer and independent
    mutable memory state M_i during conversation turns.
    """

    def __init__(self, base_model: HOPE, device: str = "cuda"):
        self.base_model = base_model
        self.device = device
        self.slots: List[dict] = []

    def clear(self):
        self.slots.clear()

    def reset_conversation_states(self):
        """Restore each slot's associative memory M to its cartridge-initial state."""
        for slot in self.slots:
            slot["runtime_state"] = slot["init_state"].clone().to(self.device)

    def load_stack(self, specs: List[Tuple[float, str]]):
        """Load weighted cartridges: e.g. [(0.7, 'python.pt'), (0.3, 'project.pt')]."""
        self.clear()
        base_sd = self.base_model.fast_memory.state_dict()
        model_dtype = next(self.base_model.parameters()).dtype

        for weight, path in specs:
            cart = load_cartridge(
                path,
                expected_arch=self.base_model.config.preset_name,
                device=self.device,
            )
            layer = SelfModifyingLayer(
                self.base_model.d_model,
                chunk_size=self.base_model.config.fast_chunk_size,
                force_loop=self.base_model.config.fast_force_loop,
            ).to(self.device).to(model_dtype)
            layer.eval()

            new_sd = {}
            for k, base_val in base_sd.items():
                delta = cart["deltas"][k].to(device=self.device, dtype=base_val.dtype)
                new_sd[k] = base_val + delta
            layer.load_state_dict(new_sd)

            init_state = cart["memory_state"].float().to(self.device)
            self.slots.append({
                "weight": float(weight),
                "path": path,
                "topic": cart["header"].get("topic", os.path.basename(path)),
                "layer": layer,
                "init_state": init_state,
                "runtime_state": init_state.clone(),
                "bm25": cart["bm25"],
            })

    def retrieve_context(self, query: str, top_k: int = 2) -> List[str]:
        """Query all loaded cartridges' BM25 indexes and return top-k chunks."""
        if not self.slots:
            return []
        scored_chunks: List[Tuple[float, str]] = []
        for slot in self.slots:
            w = slot["weight"]
            for score, chunk in slot["bm25"].search(query, top_k=top_k):
                scored_chunks.append((score * w, chunk))

        scored_chunks.sort(key=lambda x: x[0], reverse=True)
        seen = set()
        results = []
        for _, chunk in scored_chunks:
            if chunk not in seen:
                seen.add(chunk)
                results.append(chunk)
            if len(results) >= top_k:
                break
        return results

    @torch.no_grad()
    def forward_with_stack(
        self,
        x: torch.Tensor,
        base_state: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        last_only: bool = False,
        pad_token_id: int = 0,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Run HOPE forward pass with parallel cartridge blending."""
        if not self.slots:
            return self.base_model(
                x, state=base_state, mask=mask, last_only=last_only, pad_token_id=pad_token_id
            )

        if mask is None:
            mask = (x != pad_token_id).float()

        h = self.base_model.embedding(x)
        B = x.shape[0]

        fast_combined = torch.zeros_like(h)
        for slot in self.slots:
            st = slot["runtime_state"]
            if st.shape[0] != B:
                st = st.expand(B, -1, -1).contiguous()
            out_i, new_st = slot["layer"](h, state=st, mask=mask)
            slot["runtime_state"] = new_st[:1]
            fast_combined = fast_combined + slot["weight"] * out_i

        h = self.base_model.norm_fast(self.base_model.residual_clamp(h, fast_combined))

        if last_only:
            h = h[:, -1:, :]

        for layer in self.base_model.cms_layers:
            h = layer(h)

        return self.base_model.head(h), base_state


# ============================================================================
# Automated Round-Trip & Parallel Stacking Test (dev_15m)
# ============================================================================

def run_self_test():
    """Verify cartridge save/load round-trip and parallel output stacking on dev_15m."""
    import tempfile
    from configs import dev_15m

    print("=" * 65)
    print("  CARTRIDGE ROUND-TRIP & PARALLEL STACKING TEST (dev_15m)")
    print("=" * 65)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = dev_15m()
    base_model = HOPE(cfg).to(device).eval()

    model_a = copy.deepcopy(base_model)
    model_b = copy.deepcopy(base_model)
    with torch.no_grad():
        for p in model_a.plastic_parameters():
            p.add_(torch.randn_like(p) * 0.05)
        for p in model_b.plastic_parameters():
            p.add_(torch.randn_like(p) * 0.05)

    dummy_ids = torch.randint(1, cfg.vocab_size, (1, 32), device=device)
    with torch.no_grad():
        _, mem_a = model_a(dummy_ids)
        _, mem_b = model_b(dummy_ids)

    chunks_a = [
        "Python decorators wrap functions using the @decorator syntax.",
        "List comprehensions in Python provide concise list creation.",
    ]
    chunks_b = [
        "HOPE uses a 95 percent frozen core and 5 percent plastic fast weights.",
        "Cartridges store fp16 zlib-compressed deltas and BM25 indexes.",
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        path_a = os.path.join(tmpdir, "python.pt")
        path_b = os.path.join(tmpdir, "project.pt")

        save_cartridge(path_a, base_model, model_a, mem_a, chunks_a, topic="python")
        save_cartridge(path_b, base_model, model_b, mem_b, chunks_b, topic="project")

        size_a = os.path.getsize(path_a) / 1024
        size_b = os.path.getsize(path_b) / 1024
        print(f"  Saved python.pt  ({size_a:.1f} KB)")
        print(f"  Saved project.pt ({size_b:.1f} KB)")

        loaded_a = load_cartridge(path_a, expected_arch="dev_15m", device=device)
        assert loaded_a["header"]["version"] == "2.7"
        assert loaded_a["memory_state"].shape == (1, cfg.d_model, cfg.d_model)
        assert loaded_a["memory_state"].dtype == torch.float32
        print("  ✓ Single-cartridge state & fp32 memory M restored cleanly")

        runtime = StackedCartridgeRuntime(base_model, device=device)
        runtime.load_stack([(0.7, path_a), (0.3, path_b)])

        hits = runtime.retrieve_context("How do Python decorators and HOPE cartridges work?", top_k=2)
        assert len(hits) == 2
        print(f"  ✓ BM25 hybrid retrieval returned {len(hits)} chunks:")
        for idx, h in enumerate(hits, 1):
            print(f"      [{idx}] {h}")

        test_x = torch.randint(1, cfg.vocab_size, (2, 16), device=device)
        logits, _ = runtime.forward_with_stack(test_x)
        assert logits.shape == (2, 16, cfg.vocab_size), f"Unexpected shape: {logits.shape}"
        assert torch.isfinite(logits).all(), "Non-finite logits detected!"
        print(f"  ✓ Parallel stacking forward pass verified: logits shape {list(logits.shape)}, all finite")

    print("\nAll cartridge round-trip and parallel stacking tests PASSED. ✓")


if __name__ == "__main__":
    run_self_test()
