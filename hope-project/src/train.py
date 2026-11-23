"""
train.py — HOPE training loop for Blueprint V2.7.

Features:
  - torch.bfloat16 (native Ampere, no GradScaler)
  - Gradient checkpointing across CMS blocks
  - Differential AdamW (3e-4 slow core / 1e-4 fast projections) with cosine schedule
  - Windows-safe lazy np.memmap loading (num_workers=0, pin_memory=True)
  - Automatic power-off recovery: rolling latest_checkpoint.pt every 500 steps + on Ctrl+C
"""

from __future__ import annotations
import os
import sys
import time
import math
import json
import signal
import argparse
from pathlib import Path
from datetime import timedelta

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from configs import HOPEConfig, get_config
from model import HOPE, count_parameters


# ============================================================================
# Memmap Dataset (Windows-safe: lazy open in __getitem__)
# ============================================================================

class MemmapDataset(Dataset):
    """Lazily-loaded np.memmap dataset, Windows-safe.

    Opens the memmap file lazily in __getitem__ to avoid Windows multiprocessing
    pickle errors and RAM duplication. With num_workers=0, this is opened once
    per process.
    """

    def __init__(self, bin_path: str, meta_path: str = None):
        self.bin_path = bin_path
        if meta_path is None:
            meta_path = bin_path.replace('.bin', '_meta.json')

        with open(meta_path, 'r') as f:
            self.meta = json.load(f)

        self.n_chunks = self.meta['n_chunks']
        self.chunk_len = self.meta['chunk_len']
        self.dtype = np.uint16 if self.meta['dtype'] == 'uint16' else np.uint32
        self.pad_id = self.meta.get('pad_id', 0)
        self._mm = None

    def _ensure_open(self):
        if self._mm is None:
            self._mm = np.memmap(
                self.bin_path, dtype=self.dtype, mode='r',
                shape=(self.n_chunks, self.chunk_len),
            )

    def __len__(self):
        return self.n_chunks

    def __getitem__(self, idx):
        self._ensure_open()
        chunk = np.array(self._mm[idx], dtype=np.int64)  # copy to regular array
        return torch.from_numpy(chunk)


# ============================================================================
# Learning rate schedule
# ============================================================================

def cosine_lr_lambda(step: int, warmup_steps: int, max_steps: int) -> float:
    """Linear warmup + cosine decay to 1% of peak."""
    if step < warmup_steps:
        return step / max(warmup_steps, 1)
    progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    return 0.01 + 0.5 * (1.0 - 0.01) * (1.0 + math.cos(math.pi * progress))


# ============================================================================
# Checkpoint management
# ============================================================================

def save_checkpoint(
    model: HOPE, optimizer_core, optimizer_plastic,
    scheduler_core, scheduler_plastic,
    step: int, best_val_loss: float,
    config: HOPEConfig, path: str,
):
    """Save a full checkpoint for resumption."""
    checkpoint = {
        'model_state': model.state_dict(),
        'optimizer_core_state': optimizer_core.state_dict(),
        'optimizer_plastic_state': optimizer_plastic.state_dict(),
        'scheduler_core_state': scheduler_core.state_dict(),
        'scheduler_plastic_state': scheduler_plastic.state_dict(),
        'step': step,
        'best_val_loss': best_val_loss,
        'config_preset': config.preset_name,
        'd_model': config.d_model,
        'num_layers': config.num_layers,
        'vocab_size': config.vocab_size,
        'seq_len': config.seq_len,
    }
    # Atomic write: write to temp then rename
    tmp_path = path + '.tmp'
    torch.save(checkpoint, tmp_path)
    if os.path.exists(path):
        os.replace(tmp_path, path)
    else:
        os.rename(tmp_path, path)
    return path


def load_checkpoint(path: str, model: HOPE, optimizer_core, optimizer_plastic,
                    scheduler_core, scheduler_plastic, device: str):
    """Load checkpoint and restore all state."""
    if not os.path.exists(path):
        return 0, float('inf')

    print(f"Loading checkpoint from {path}...")
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state'])

    if 'optimizer_core_state' in checkpoint:
        optimizer_core.load_state_dict(checkpoint['optimizer_core_state'])
    if 'optimizer_plastic_state' in checkpoint:
        optimizer_plastic.load_state_dict(checkpoint['optimizer_plastic_state'])
    if 'scheduler_core_state' in checkpoint:
        scheduler_core.load_state_dict(checkpoint['scheduler_core_state'])
    if 'scheduler_plastic_state' in checkpoint:
        scheduler_plastic.load_state_dict(checkpoint['scheduler_plastic_state'])

    step = checkpoint.get('step', 0)
    best_val_loss = checkpoint.get('best_val_loss', float('inf'))
    print(f"  Resumed from step {step}, best val loss: {best_val_loss:.4f}")
    return step, best_val_loss


# ============================================================================
# Validation
# ============================================================================

@torch.no_grad()
def run_validation(
    model: HOPE, val_loader: DataLoader, device: str,
    config: HOPEConfig, max_batches: int = 50,
) -> float:
    model.eval()
    total_loss = 0.0
    n_batches = 0

    for batch in val_loader:
        batch = batch.to(device)
        inputs = batch[:, :-1]
        targets = batch[:, 1:]

        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            logits, _ = model(inputs, pad_token_id=0)
            loss = F.cross_entropy(
                logits.reshape(-1, config.vocab_size),
                targets.reshape(-1),
                ignore_index=0,  # pad
            )

        total_loss += loss.item()
        n_batches += 1
        if n_batches >= max_batches:
            break

    model.train()
    return total_loss / max(n_batches, 1)


# ============================================================================
# Training loop
# ============================================================================

def train(
    preset: str = "dev_15m",
    data_dir: str = "data",
    output_dir: str = "checkpoints",
    resume: bool = True,
):
    # Setup
    config = get_config(preset)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  HOPE Training — {preset}")
    print(f"{'='*60}")
    print(config.summary())
    print(f"  Device: {device}")

    if device == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  BF16: {torch.cuda.is_bf16_supported()}")

    # Build model
    model = HOPE(config).to(device)
    model.enable_gradient_checkpointing()
    counts = count_parameters(model)
    print(f"\n  Model params: {counts['total']:,} total "
          f"({counts['core']:,} core + {counts['plastic']:,} plastic, "
          f"ratio={counts['ratio']:.2%})")

    # Differential optimizers
    optimizer_core = torch.optim.AdamW(
        model.core_parameters(),
        lr=config.core_lr,
        betas=config.betas,
        weight_decay=config.weight_decay,
    )
    optimizer_plastic = torch.optim.AdamW(
        model.plastic_parameters(),
        lr=config.plastic_lr,
        betas=config.betas,
        weight_decay=config.weight_decay,
    )

    # Cosine schedules
    scheduler_core = torch.optim.lr_scheduler.LambdaLR(
        optimizer_core,
        lambda step: cosine_lr_lambda(step, config.warmup_steps, config.max_steps),
    )
    scheduler_plastic = torch.optim.lr_scheduler.LambdaLR(
        optimizer_plastic,
        lambda step: cosine_lr_lambda(step, config.warmup_steps, config.max_steps),
    )

    # Resume from checkpoint
    ckpt_path = os.path.join(output_dir, f"{preset}_latest_checkpoint.pt")
    start_step = 0
    best_val_loss = float('inf')

    if resume and os.path.exists(ckpt_path):
        start_step, best_val_loss = load_checkpoint(
            ckpt_path, model, optimizer_core, optimizer_plastic,
            scheduler_core, scheduler_plastic, device,
        )

    # Data loaders (Windows-safe: num_workers=0, pin_memory=True)
    train_bin = os.path.join(data_dir, f"{preset}_train.bin")
    val_bin = os.path.join(data_dir, f"{preset}_val.bin")

    if not os.path.exists(train_bin):
        print(f"ERROR: Training data not found at {train_bin}")
        print(f"Run: python build_dataset.py --preset {preset} --data-dir {data_dir}")
        sys.exit(1)

    train_dataset = MemmapDataset(train_bin)
    val_dataset = MemmapDataset(val_bin) if os.path.exists(val_bin) else None

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.micro_batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.micro_batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
        )

    print(f"\n  Train chunks: {len(train_dataset):,}")
    if val_dataset:
        print(f"  Val chunks:   {len(val_dataset):,}")

    # Training state
    model.train()
    step = start_step
    running_loss = 0.0
    steps_in_epoch = 0
    interrupted = False

    # Ctrl+C handler
    def handle_interrupt(signum, frame):
        nonlocal interrupted
        interrupted = True
        print("\n\n[Ctrl+C] Saving checkpoint and exiting...")

    signal.signal(signal.SIGINT, handle_interrupt)

    print(f"\n  Starting training from step {step}...")
    print(f"  Target: {config.max_steps} steps")
    print(f"  Checkpoint every {config.checkpoint_every} steps")
    print(f"  Press Ctrl+C to save and exit.\n")

    start_time = time.time()
    last_log_time = time.time()
    last_val_time = time.time()
    epoch = 0

    try:
        while step < config.max_steps and not interrupted:
            epoch += 1
            train_iter = iter(train_loader)
            while step < config.max_steps and not interrupted:
                optimizer_core.zero_grad(set_to_none=True)
                optimizer_plastic.zero_grad(set_to_none=True)
                accum_loss = 0.0
                for micro_step in range(config.gradient_accumulation_steps):
                    try:
                        batch = next(train_iter)
                    except StopIteration:
                        train_iter = iter(train_loader)
                        batch = next(train_iter)
                    batch_device = batch.to(device)
                    inputs = batch_device[:, :-1]
                    targets = batch_device[:, 1:]

                    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                        logits, _ = model(inputs, pad_token_id=0)
                        loss = F.cross_entropy(
                            logits.reshape(-1, config.vocab_size),
                            targets.reshape(-1),
                            ignore_index=0,
                        )
                        loss = loss / config.gradient_accumulation_steps

                    loss.backward()
                    accum_loss += loss.item()

                # Clip and step
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
                optimizer_core.step()
                optimizer_plastic.step()
                scheduler_core.step()
                scheduler_plastic.step()

                step += 1
                running_loss += accum_loss

                # Logging
                if time.time() - last_log_time > 2.0 or step % 50 == 0:
                    elapsed = time.time() - start_time
                    avg_loss = running_loss / max(step - start_step, 1)
                    tokens_per_step = config.micro_batch_size * config.seq_len * config.gradient_accumulation_steps
                    tok_per_sec = tokens_per_step * (step - start_step) / max(elapsed, 0.001)
                    eta_sec = (config.max_steps - step) * elapsed / max(step - start_step, 1)
                    core_lr = scheduler_core.get_last_lr()[0]
                    plastic_lr = scheduler_plastic.get_last_lr()[0]

                    print(f"  Step {step:>6}/{config.max_steps} | "
                          f"Loss: {accum_loss:.4f} (avg {avg_loss:.4f}) | "
                          f"LR: {core_lr:.2e}/{plastic_lr:.2e} | "
                          f"{tok_per_sec:,.0f} tok/s | "
                          f"ETA: {timedelta(seconds=int(eta_sec))}")
                    last_log_time = time.time()

                # Validation
                if val_loader and (time.time() - last_val_time > 60 or step % 500 == 0):
                    val_loss = run_validation(model, val_loader, device, config)
                    print(f"  >>> Validation loss: {val_loss:.4f} (best: {best_val_loss:.4f})")
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_path = os.path.join(output_dir, f"{preset}_best.pt")
                        save_checkpoint(model, optimizer_core, optimizer_plastic,
                                        scheduler_core, scheduler_plastic,
                                        step, best_val_loss, config, best_path)
                        print(f"  >>> New best model saved to {best_path}")
                    last_val_time = time.time()

                # Periodic checkpoint
                if step % config.checkpoint_every == 0:
                    save_checkpoint(model, optimizer_core, optimizer_plastic,
                                    scheduler_core, scheduler_plastic,
                                    step, best_val_loss, config, ckpt_path)
                    print(f"  >>> Checkpoint saved at step {step}")

    except KeyboardInterrupt:
        interrupted = True
        print("\n\n[KeyboardInterrupt] Saving checkpoint...")

    # Final save
    save_checkpoint(model, optimizer_core, optimizer_plastic,
                    scheduler_core, scheduler_plastic,
                    step, best_val_loss, config, ckpt_path)
    print(f"\n{'='*60}")
    print(f"  Training {'interrupted' if interrupted else 'complete'} at step {step}")
    print(f"  Checkpoint saved to: {ckpt_path}")
    print(f"  Best val loss: {best_val_loss:.4f}")
    elapsed = time.time() - start_time
    print(f"  Total time: {timedelta(seconds=int(elapsed))}")
    print(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train HOPE model")
    parser.add_argument("--preset", default="dev_15m", choices=["dev_15m", "local_200m", "prod_500m"])
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default="checkpoints")
    parser.add_argument("--no-resume", action="store_true", help="Don't resume from checkpoint")
    args = parser.parse_args()

    train(
        preset=args.preset,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        resume=not args.no_resume,
    )
