# Bashful (`>_<`) — A Modular, Continual-Learning AI Console

> **Proof of Concept:** An experimental, vibecoded local AI console built around the **HOPE (Nested Learning)** architecture—featuring real-time fast/slow memory allocation, plug-and-play subject cartridges, and an expressive retro pixel-art mascot.

---

## What is Bashful?

**Bashful** addresses the "anterograde amnesia" of standard LLMs (where weights freeze after training) by implementing a PyTorch-based **HOPE Continuum Memory System** for live, modular continual learning on consumer hardware (Apple Silicon/MPS, CUDA, or CPU). It features an interactive console fronted by an expressive pixel-art mascot (`Bashful`) that visually reflects real-time system states like thinking, active updates, or errors.

---

## Core Features

* **Real-Time Fast & Slow Memory:** Splits memory into **Fast Weights** (`SelfModifyingLayer`), which update per token via a gated delta rule for live context, and **Slow Weights** (tiered FFN blocks) for durable knowledge.
* **Linked Memory Budget Sliders (5% Cap):** Balances **Fast** and **Slow** memory shares within a strict 5% combined active parameter budget, keeping 95% of the base model frozen for stability.
* **Plug-and-Play Subject Cartridges:** Trains lightweight, domain-specific portable adapters (currently PyTorch `.pt`/`.pth` during prototyping, migrating to `.safetensors` for final releases) without overwriting the base checkpoint.
* **Community Library & Deterministic Benchmarks:** Validates trained cartridges via seed-locked custom benchmark suites (`probe_model.py` / `eval_suite.py`).
* **Fast Consumer Inference:** Achieves a 5.2× training speedup via chunk-parallel delta scans and O(N) state-passing inference (~380 tok/s on CPU, ~500 tok/s on Apple MPS).

---

## Project Status & Honest Caveats

Validated across 86M-parameter HOPE training runs:
* **What works:** Fluent English output, structured Q&A/Reasoning formats, O(N) state-passing chat interfaces (`chat.py` and Gradio `app.py`), and measurable in-context fast-memory adaptation.
* **Known limits:** Deeper slow tiers (blocks 8–15) under-train relative to the output head, and stacking 16 position-wise CMS blocks attenuates linear readout of the fast-memory context bag ($R^2 = 0.066$). Future iterations will target direct pathways from `fast_out` to the output head.

---

## Repository Map

| File / Directory | Purpose |
| :--- | :--- |
| `train_hope.py` | Core HOPE architecture (`SelfModifyingLayer`, chunked delta scan, CMS tiers) & training loop |
| `chat.py` / `app.py` / `generate.py` | Interactive terminal console, Gradio web UI, and CLI generator |
| `run_pipeline.py` | End-to-end Phase 1 (Wikipedia) + Phase 2 (Q&A) training runner |
| `extend_training.py` / `extend_phase2.py` | Checkpoint resume & LR re-warm drivers for extended runs |
| `probe_model.py` / `probe_stats.jsonl` | Deterministic factual cloze & generation probe harness |
| `surgery/` | Closed-form output-layer weight surgery & frozen two-split evaluation suite |
| `EXPERIMENT.md` / `EXPERIMENT_LOG.md` | Full writeup and decision log (D1–D7) from the 48k-step training marathon |

---

## License & Attribution

* **Code License:** Released under the [MIT License](LICENSE).
* **Research Credit:** Inspired by *“Nested Learning: The Illusion of Deep Learning”* (Ali Behrouz, Meisam Razaviyayn, Peilin Zhong, Vahab Mirrokni — Google Research).
* **Disclaimer:** An independent community proof-of-concept not affiliated with or endorsed by Google.
