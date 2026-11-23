
Bashful (>_<) — A Modular, Continual-Learning AI Console

Proof of Concept: An experimental, vibecoded local AI console built around the HOPE (Nested Learning) architecture—featuring real-time fast/slow memory allocation, plug-and-play .pt subject cartridges, and a retro pixel-art mascot that reacts to what the model is doing under the hood.

What is Bashful?

Standard LLMs suffer from "Anterograde Amnesia"—once training finishes, their weights freeze. Bashful wraps a from-scratch PyTorch implementation of the HOPE Continuum Memory System in an interactive console designed to test live, modular continual learning on consumer hardware (Mac Apple Silicon / MPS, CUDA, or CPU).

Meet the Mascot

The console is fronted by Bashful, a retro pixel-art cartridge with a bash prompt face (>_ turned into >_<) that shifts expressions for real-time visual feedback on system state:

Expression	State	What Bashful is Doing
o_O	Thinking	Ingesting prompts and learning new information
>_<	Active / Bashful	Deep in thought, running fast-memory state updates
^_^	Happy	Locked in at peak performance
X_X	Error	Hit an exception, tensor mismatch, or OOM state
U_U / -_-	Idle / Sleep	Waiting on input or paused between training runs


Core Features

• Real-Time Fast & Slow Memory: Decouples memory into two active systems. Fast Weights (SelfModifyingLayer) update per token via a gated delta rule (M_t = α_t·M_{t-1} + β_t·k_tᵀ(v_t − k_t·M_{t-1})) to track live context, while Slow Weights (tiered FFN blocks) consolidate durable knowledge over longer intervals.
• Linked Memory Budget Sliders (5% Cap): Two linked controls—Fast Memory % and Slow Memory %—share a strict 5% combined active parameter budget (keeping 95% of the base model frozen for speed and stability).
	• Long coding session? Slide toward Fast Memory (e.g., 3% Fast / 2% Slow) to track live variables without losing context.
	• Need durable domain retention? Dial up Slow Memory.
• Plug-and-Play Subject Cartridges (.pt): Queue up a domain (Python, Math, Dialogue) to train a lightweight, portable .pt cartridge (≤ 10% of base parameters) for targeted specialization without overwriting the base checkpoint.
• Community Library & Deterministic Benchmarks: Share trained cartridges and validate them against deterministic, seed-locked custom benchmarks (probe_model.py / eval_suite.py) to earn reproducible stats.
• Fast Consumer Inference: Built on top of a chunk-parallel delta-rule scan (5.2× training speedup) and O(N) state-passing inference (~380 tok/s on CPU, ~500 tok/s on Apple MPS).

Project Status & Honest Caveats

This repo is an experimental proof of concept built on top of our documented 86M-parameter HOPE marathon (v0.1-marathon):
• What works: Fluent English generation, structured Q&A/Reasoning formatting, O(N) state-passing chat (chat.py & Gradio app.py), and measurable in-context fast-memory adaptation.
• Known architectural limits: At 86M parameters / ~786M tokens, deep slow tiers (blocks 8–15) under-train compared to the output head, and stacking 16 position-wise CMS blocks attenuates linear readout of the fast-memory context bag (R² = 0.066). Upcoming cartridge and wiring experiments focus on giving fast_out a more direct path to the output head.

Quickstart


1. Install

bash
git clone https://github.com
cd HOPE-nested-learning
python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
Use code with caution.

2. Grab a Pretrained Checkpoint (or Train From Scratch)

bash
# Download the 86M marathon checkpoints (~986 MB each)
gh release download v0.1-marathon --repo obekt/HOPE-nested-learning --pattern "*.pth"

# Or run the automated 2-phase pipeline (Wikipedia foundation + Q&A fine-tune)
python run_pipeline.py
Use code with caution.

3. Launch Bashful

• Interactive CLI Console:bash
python chat.py --model hope_final_best.pth
Use code with caution.
(Supports live commands: /temp 0.7, /tokens 250, /reasoning on|off, /format raw|qa)
• Web UI (Gradio):bash
python app.py
Use code with caution.
• Single-Shot Generation:bash
python generate.py --prompt "Why is the sky blue?" --temperature 0.3
Use code with caution.

Repository Map


File / Directory	Purpose
train_hope.py	Core HOPE architecture (SelfModifyingLayer, chunked delta scan, CMS tiers) & training loop
chat.py / app.py / generate.py	Interactive terminal console, Gradio web UI, and CLI generator
run_pipeline.py	End-to-end Phase 1 (Wikipedia) + Phase 2 (Q&A) training runner
extend_training.py / extend_phase2.py	Checkpoint resume & LR re-warm drivers for extended runs
probe_model.py / probe_stats.jsonl	Deterministic factual cloze & generation probe harness
surgery/	Closed-form output-layer weight surgery & frozen two-split evaluation suite
EXPERIMENT.md / EXPERIMENT_LOG.md	Full writeup and decision log (D1–D7) from the 48k-step training marathon


Credits & Research Basis

• Architecture:## License & Attribution

* **Code License:** Released under the [MIT License](LICENSE). Feel free to fork, modify, and experiment with it.
* **Research Credit:** Built as an unofficial, from-scratch PyTorch implementation inspired by **"Nested Learning: The Illusion of Deep Learning"** (Ali Behrouz, Meisam Razaviyayn, Peilin Zhong, Vahab Mirrokni — Google Research).
* **Disclaimer:** This is an independent community proof-of-concept and is not affiliated with, endorsed by, or maintained by Google.
 Unofficial PyTorch implementation inspired by "Nested Learning: The Illusion of Deep Learning" (Behrouz et al., Google Research).
• Mascot & Console Concept: Vibecoded as an open sandbox for modular continual learning and portable .pt adapters.
