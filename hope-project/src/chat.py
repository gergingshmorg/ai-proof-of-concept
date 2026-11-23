"""
chat.py — Interactive HOPE Chat, Deterministic /study Consolidation & Cartridge Loader.

Commands:
  /study <topic>          Fetch web articles deterministically via Python search + trafilatura,
                          hard-freeze the 95% core, consolidate into the 5% plastic weights
                          for 30-60s with cosine LR decay, and save cartridges/<topic>.pt
  /study ./path/to/dir    Ingest local .txt / .md / .py files deterministically and consolidate
  /load <spec>            Load single or weighted parallel cartridges:
                            /load cartridges/python.pt
                            /load 0.7cartridges/python.pt + 0.3cartridges/project.pt
  /unload                 Unload active cartridges and return to base model
  /reset                  Reset conversation memory state
  /quit                   Exit
"""

from __future__ import annotations
import os
import re
import sys
import copy
import time
import math
import glob
import random
import argparse
from typing import List, Tuple, Optional

import torch
import torch.nn.functional as F

from configs import get_config, HOPEConfig
from model import HOPE, count_parameters
from build_dataset import TokenizerWrapper
from cartridge import save_cartridge, StackedCartridgeRuntime


# ============================================================================
# Deterministic Fetching & Text Chunking (Never uses the RNN)
# ============================================================================

def _chunk_text(text: str, chunk_chars: int = 450, overlap: int = 80) -> List[str]:
    """Split cleaned text into overlapping chunks suitable for seq_len + BM25."""
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if not text:
        return []
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_chars, len(text))
        piece = text[start:end].strip()
        if len(piece) >= 40:
            chunks.append(piece)
        if end == len(text):
            break
        start += max(1, chunk_chars - overlap)
    return chunks


def fetch_local_folder(folder_path: str) -> List[str]:
    """Deterministically read local text, markdown, and code files."""
    exts = ("*.txt", "*.md", "*.py", "*.json", "*.rst", "*.csv")
    files = []
    for ext in exts:
        files.extend(glob.glob(os.path.join(folder_path, "**", ext), recursive=True))

    chunks: List[str] = []
    for fp in sorted(set(files)):
        try:
            with open(fp, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
            header = f"[Source: {os.path.basename(fp)}]\n"
            for c in _chunk_text(content):
                chunks.append(header + c)
        except Exception as e:
            print(f"  [WARN] Skipped {fp}: {e}")
    return chunks


def fetch_web_topic(topic: str, max_results: int = 6) -> List[str]:
    """Deterministically search the web and strip HTML boilerplate via trafilatura."""
    try:
        from duckduckgo_search import DDGS
    except ImportError:
        try:
            from ddgs import DDGS
        except ImportError:
            print("ERROR: Install duckduckgo-search (`pip install duckduckgo-search trafilatura`)")
            return []

    try:
        import trafilatura
    except ImportError:
        print("ERROR: Install trafilatura (`pip install trafilatura`)")
        return []

    print(f"  Searching web deterministically for: '{topic}'...")
    urls = []
    snippets = []
    try:
        with DDGS() as ddgs:
            for res in ddgs.text(topic, max_results=max_results):
                if res.get("href"):
                    urls.append(res["href"])
                if res.get("body"):
                    snippets.append(f"{res.get('title', '')}: {res['body']}")
    except Exception as e:
        print(f"  [WARN] Search warning: {e}")

    chunks: List[str] = []
    for url in urls:
        try:
            downloaded = trafilatura.fetch_url(url)
            if downloaded:
                extracted = trafilatura.extract(downloaded, include_comments=False, include_tables=True)
                if extracted:
                    for c in _chunk_text(extracted):
                        chunks.append(c)
                    print(f"    ✓ Extracted {len(extracted):,} chars from {url[:60]}")
        except Exception:
            continue

    if len(chunks) < 4 and snippets:
        for s in snippets:
            chunks.extend(_chunk_text(s))

    return chunks


# ============================================================================
# Multi-Epoch Plastic Consolidation (95% Core Hard-Frozen)
# ============================================================================

def consolidate_study_session(
    base_model: HOPE,
    tokenizer: TokenizerWrapper,
    chunks: List[str],
    topic: str,
    cartridge_dir: str = "cartridges",
    duration_sec: float = 35.0,
    lr: float = 2e-4,
    device: str = "cuda",
) -> str:
    """Hard-freeze the 95% core and run a multi-epoch shuffled consolidation loop
    with cosine LR decay over ONLY the 5% plastic weights, plus final memory pass.
    """
    studied_model = copy.deepcopy(base_model).to(device)
    studied_model.train()

    for p in studied_model.core_parameters():
        p.requires_grad = False
    for p in studied_model.plastic_parameters():
        p.requires_grad = True

    seq_len = studied_model.config.seq_len
    sequences: List[torch.Tensor] = []
    for c in chunks:
        ids = tokenizer.encode(c) + [tokenizer.eos_id]
        if len(ids) < 8:
            continue
        ids = ids[: seq_len + 1]
        if len(ids) < seq_len + 1:
            ids = ids + [tokenizer.pad_id] * (seq_len + 1 - len(ids))
        sequences.append(torch.tensor(ids, dtype=torch.long))

    if not sequences:
        raise ValueError("No valid tokenized sequences produced from study source.")

    optimizer = torch.optim.AdamW(
        studied_model.plastic_parameters(),
        lr=lr,
        betas=(0.9, 0.95),
        weight_decay=0.01,
    )

    print(f"  Consolidating {len(chunks)} chunks ({len(sequences)} sequences) for ~{duration_sec:.0f}s...")
    print("  95% Core: HARD-FROZEN | 5% Plastic Weights: ACTIVE")

    start_time = time.time()
    epoch = 0
    step = 0
    batch_size = min(4, len(sequences))

    while (time.time() - start_time) < duration_sec:
        epoch += 1
        random.shuffle(sequences)

        for i in range(0, len(sequences), batch_size):
            elapsed = time.time() - start_time
            if elapsed >= duration_sec:
                break

            progress = min(1.0, elapsed / max(duration_sec, 1e-3))
            cur_lr = lr * (0.05 + 0.95 * 0.5 * (1.0 + math.cos(math.pi * progress)))
            for pg in optimizer.param_groups:
                pg["lr"] = cur_lr

            batch = torch.stack(sequences[i : i + batch_size]).to(device)
            inputs = batch[:, :-1]
            targets = batch[:, 1:]

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type="cuda" if device == "cuda" else "cpu", dtype=torch.bfloat16):
                logits, _ = studied_model(inputs, pad_token_id=tokenizer.pad_id)
                loss = F.cross_entropy(
                    logits.reshape(-1, studied_model.vocab_size),
                    targets.reshape(-1),
                    ignore_index=tokenizer.pad_id,
                )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(studied_model.plastic_parameters(), 1.0)
            optimizer.step()
            step += 1

        if epoch % 5 == 0 or (time.time() - start_time) >= duration_sec:
            print(f"    Epoch {epoch:>3} | Step {step:>4} | Loss: {loss.item():.4f} | LR: {cur_lr:.2e}")

    studied_model.eval()
    memory_state: Optional[torch.Tensor] = None
    with torch.no_grad():
        for seq in sequences:
            inp = seq[:-1].unsqueeze(0).to(device)
            _, memory_state = studied_model(inp, state=memory_state, pad_token_id=tokenizer.pad_id)

    for p in studied_model.parameters():
        p.requires_grad = True

    safe_name = re.sub(r"[^a-zA-Z0-9_-]+", "_", topic.strip()).strip("_").lower() or "study"
    out_path = os.path.join(cartridge_dir, f"{safe_name}.pt")
    save_cartridge(
        path=out_path,
        base_model=base_model,
        studied_model=studied_model,
        memory_state=memory_state,
        chunks=chunks,
        topic=topic,
    )
    print(f"  ✓ Saved cartridge to: {out_path} ({os.path.getsize(out_path)/1024:.1f} KB)")
    return out_path


# ============================================================================
# Cartridge /load Parser
# ============================================================================

def parse_load_command(arg_str: str) -> List[Tuple[float, str]]:
    """Parse `/load 0.7python.pt + 0.3project.pt` or `/load python.pt`."""
    parts = [p.strip() for p in arg_str.split("+") if p.strip()]
    specs: List[Tuple[float, str]] = []
    for part in parts:
        m = re.match(r"^([0-9]*\.?[0-9]+)\s*(.+)$", part)
        if m:
            weight = float(m.group(1))
            path = m.group(2).strip()
        else:
            weight = 1.0
            path = part
        specs.append((weight, path))
    return specs


# ============================================================================
# Autoregressive Generation
# ============================================================================

@torch.no_grad()
def generate_reply(
    runtime: StackedCartridgeRuntime,
    tokenizer: TokenizerWrapper,
    prompt: str,
    base_state: Optional[torch.Tensor],
    max_new_tokens: int = 120,
    temperature: float = 0.7,
    top_k: int = 40,
    device: str = "cuda",
) -> Tuple[str, Optional[torch.Tensor]]:
    ids = tokenizer.encode(prompt)
    if not ids:
        ids = [tokenizer.eos_id]

    x = torch.tensor([ids], dtype=torch.long, device=device)
    amp_type = "cuda" if device == "cuda" else "cpu"

    with torch.amp.autocast(device_type=amp_type, dtype=torch.bfloat16):
        logits, base_state = runtime.forward_with_stack(
            x, base_state=base_state, last_only=True, pad_token_id=tokenizer.pad_id
        )

    generated: List[int] = []
    for _ in range(max_new_tokens):
        next_logits = logits[:, -1, :].float()
        for prev_tid in set(generated[-48:]):
            next_logits[:, prev_tid] = torch.where(next_logits[:, prev_tid] > 0, next_logits[:, prev_tid] / 1.25, next_logits[:, prev_tid] * 1.25)
        next_logits = next_logits / 0.55
        if top_k > 0:
            v, _ = torch.topk(next_logits, min(top_k, next_logits.size(-1)))
            next_logits[next_logits < v[:, [-1]]] = -float("inf")
        probs = F.softmax(next_logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        tok_id = next_id.item()
        if tok_id == tokenizer.eos_id:
            break
        generated.append(tok_id)
        tail = tokenizer.decode(generated[-6:])
        if len(generated) > 4 and any(stop in tail for stop in ["\nUser:", "\nAssistant:", "User:", "Assistant:"]):
            break
        if len(generated) > 12 and ("\n\n" in tokenizer.decode(generated[-4:]) or "User:" in tokenizer.decode(generated[-4:])):
            break

        with torch.amp.autocast(device_type=amp_type, dtype=torch.bfloat16):
            logits, _ = runtime.forward_with_stack(
                next_id, base_state=base_state, last_only=True, pad_token_id=tokenizer.pad_id
            )

    return tokenizer.decode(generated), base_state


# ============================================================================
# Interactive CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="HOPE Interactive Chat & /study Runtime")
    parser.add_argument("--preset", default="dev_15m", choices=["dev_15m", "local_200m", "prod_500m"])
    parser.add_argument("--checkpoint", default=None, help="Path to trained .pt checkpoint")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--cartridge-dir", default="cartridges")
    parser.add_argument("--study-seconds", type=float, default=35.0)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = get_config(args.preset)

    if args.preset == "dev_15m":
        tok_path = os.path.join(args.data_dir, "tokenizer_8k.json")
        if not os.path.exists(tok_path):
            print(f"ERROR: {tok_path} not found. Run `python src/build_dataset.py --preset dev_15m` first.")
            sys.exit(1)
        tokenizer = TokenizerWrapper.for_dev_15m([], args.data_dir)
    else:
        tokenizer = TokenizerWrapper.for_gpt2()

    model = HOPE(config).to(device)
    ckpt_path = args.checkpoint or os.path.join("checkpoints", f"{args.preset}_latest_checkpoint.pt")
    if os.path.exists(ckpt_path):
        print(f"Loading model checkpoint: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
    else:
        print(f"[WARN] Checkpoint not found at {ckpt_path}; starting with initialized weights.")

    model.eval()
    runtime = StackedCartridgeRuntime(model, device=device)
    base_state: Optional[torch.Tensor] = None

    print("\n" + "=" * 65)
    print(f"  HOPE Local AI ({args.preset}) — Device: {device}")
    print("  Commands: /study <topic|./folder> | /load <cart.pt> | /unload | /reset | /quit")
    print("=" * 65 + "\n")

    while True:
        try:
            user_input = input("You> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not user_input:
            continue

        if user_input.lower() in ("/quit", "/exit"):
            break

        if user_input.lower() == "/reset":
            base_state = None
            runtime.reset_conversation_states()
            print("[System] Conversation memory state reset.\n")
            continue

        if user_input.lower() == "/unload":
            runtime.clear()
            base_state = None
            print("[System] Unloaded all cartridges.\n")
            continue

        if user_input.startswith("/study "):
            target = user_input[len("/study "):].strip()
            if os.path.isdir(target):
                print(f"[Study] Reading local folder: {target}")
                chunks = fetch_local_folder(target)
                topic_name = os.path.basename(os.path.abspath(target))
            else:
                chunks = fetch_web_topic(target)
                topic_name = target

            if not chunks:
                print("[Study] No text extracted. Try another topic or folder.\n")
                continue

            cart_path = consolidate_study_session(
                base_model=model,
                tokenizer=tokenizer,
                chunks=chunks,
                topic=topic_name,
                cartridge_dir=args.cartridge_dir,
                duration_sec=args.study_seconds,
                device=device,
            )
            runtime.load_stack([(1.0, cart_path)])
            base_state = None
            print(f"[System] Auto-loaded cartridge: {cart_path}\n")
            continue

        if user_input.startswith("/load "):
            spec_str = user_input[len("/load "):].strip()
            try:
                specs = parse_load_command(spec_str)
                runtime.load_stack(specs)
                base_state = None
                loaded_desc = ", ".join(f"{w}x {p}" for w, p in specs)
                print(f"[System] Loaded parallel cartridges: {loaded_desc}\n")
            except Exception as e:
                print(f"[Error] Failed to load cartridge: {e}\n")
            continue

        context_chunks = runtime.retrieve_context(user_input, top_k=2)
        if context_chunks:
            ctx_block = "\n".join(f"[Reference {i+1}]: {c}" for i, c in enumerate(context_chunks))
            prompt = f"{ctx_block}\n\nUser: {user_input}\nAssistant:"
            print(f"[BM25 Context: {len(context_chunks)} chunk(s) injected]")
        else:
            prompt = f"User: {user_input}\nAssistant:"

        reply, base_state = generate_reply(
            runtime=runtime,
            tokenizer=tokenizer,
            prompt=prompt,
            base_state=base_state,
            device=device,
        )
        reply = reply.split("User:")[0].split("Assistant:")[0].strip()
        print(f"HOPE> {reply}\n")


if __name__ == "__main__":
    main()
