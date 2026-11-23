"""
benchmark.py — Shape, VRAM, and single-step throughput benchmark.
Tests dev_15m and local_200m on RTX 3060 12GB.
"""

import sys
import time
import gc

import torch
import torch.nn.functional as F

from configs import dev_15m, local_200m
from model import HOPE, count_parameters


def format_bytes(b: int) -> str:
    if b >= 1024**3:
        return f"{b / 1024**3:.2f} GB"
    return f"{b / 1024**2:.1f} MB"


def benchmark_config(name: str, config, device: str = "cuda", n_warmup: int = 3, n_iters: int = 10):
    print(f"\n{'='*70}")
    print(f"  BENCHMARK: {name}")
    print(f"{'='*70}")
    print(config.summary())

    # Clear GPU
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    dtype = torch.bfloat16

    # Build model
    model = HOPE(config).to(device).to(dtype)
    model.eval()
    counts = count_parameters(model)
    print(f"\n--- Actual Parameter Counts ---")
    print(f"  Total:   {counts['total']:>12,}  ({counts['total']/1e6:.1f}M)")
    print(f"  Core:    {counts['core']:>12,}  ({counts['core']/1e6:.1f}M)")
    print(f"  Plastic: {counts['plastic']:>12,}  ({counts['plastic']/1e6:.1f}M)")
    print(f"  Ratio:   {counts['ratio']:.2%}")

    B = config.micro_batch_size  # 8
    T = config.seq_len           # 512

    # Shape test
    print(f"\n--- Shape Test (B={B}, T={T}) ---")
    x = torch.randint(0, config.vocab_size, (B, T), device=device)

    with torch.no_grad(), torch.amp.autocast('cuda', dtype=dtype):
        logits, state = model(x, pad_token_id=0)

    print(f"  Input:   {list(x.shape)}")
    print(f"  Logits:  {list(logits.shape)}  (expected [{B}, {T}, {config.vocab_size}])")
    print(f"  State:   {list(state.shape)}  (expected [{B}, {config.d_model}, {config.d_model}])")
    assert logits.shape == (B, T, config.vocab_size), f"Logit shape mismatch!"
    assert state.shape == (B, config.d_model, config.d_model), f"State shape mismatch!"
    print(f"  ✓ Shapes correct")

    # VRAM measurement
    torch.cuda.reset_peak_memory_stats()
    gc.collect()
    torch.cuda.empty_cache()

    # Measure forward pass VRAM
    with torch.no_grad(), torch.amp.autocast('cuda', dtype=dtype):
        for _ in range(2):
            logits, state = model(x, pad_token_id=0)

    vram_after_fwd = torch.cuda.max_memory_allocated()
    print(f"\n--- VRAM (forward only, {dtype}) ---")
    print(f"  Peak allocated: {format_bytes(vram_after_fwd)}")

    # Measure forward+backward VRAM (training)
    torch.cuda.reset_peak_memory_stats()
    gc.collect()
    torch.cuda.empty_cache()

    model.train()
    model.enable_gradient_checkpointing()

    with torch.amp.autocast('cuda', dtype=dtype):
        logits, state = model(x, pad_token_id=0)
        targets = torch.randint(0, config.vocab_size, (B, T), device=device)
        loss = F.cross_entropy(logits.view(-1, config.vocab_size), targets.view(-1))
        loss.backward()

    vram_train = torch.cuda.max_memory_allocated()
    print(f"  Peak train (grad ckpt): {format_bytes(vram_train)}")

    # Throughput
    model.eval()
    model.disable_gradient_checkpointing()
    model.zero_grad(set_to_none=True)
    gc.collect()
    torch.cuda.empty_cache()

    print(f"\n--- Throughput (B={B}, T={T}, {n_iters} iters) ---")

    # Warmup
    with torch.no_grad(), torch.amp.autocast('cuda', dtype=dtype):
        for _ in range(n_warmup):
            logits, state = model(x, pad_token_id=0)
    torch.cuda.synchronize()

    # Timed
    start = time.perf_counter()
    with torch.no_grad(), torch.amp.autocast('cuda', dtype=dtype):
        for _ in range(n_iters):
            logits, state = model(x, pad_token_id=0)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    tokens_per_iter = B * T
    total_tokens = tokens_per_iter * n_iters
    tok_per_sec = total_tokens / elapsed
    sec_per_step = elapsed / n_iters

    print(f"  Total time:    {elapsed:.3f}s for {n_iters} iters")
    print(f"  Per step:      {sec_per_step*1000:.1f}ms")
    print(f"  Throughput:    {tok_per_sec:,.0f} tok/s (inference)")

    # Training throughput estimate (with backward + grad ckpt)
    model.train()
    model.enable_gradient_checkpointing()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # Warmup training
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast('cuda', dtype=dtype):
            logits, state = model(x, pad_token_id=0)
            targets = torch.randint(0, config.vocab_size, (B, T), device=device)
            loss = F.cross_entropy(logits.view(-1, config.vocab_size), targets.view(-1))
        loss.backward()
        optimizer.step()
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(n_iters):
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast('cuda', dtype=dtype):
            logits, state = model(x, pad_token_id=0)
            targets = torch.randint(0, config.vocab_size, (B, T), device=device)
            loss = F.cross_entropy(logits.view(-1, config.vocab_size), targets.view(-1))
        loss.backward()
        optimizer.step()
    torch.cuda.synchronize()
    elapsed_train = time.perf_counter() - start

    train_tok_per_sec = total_tokens / elapsed_train
    train_sec_per_step = elapsed_train / n_iters

    # With gradient accumulation
    effective_tok_per_sec = train_tok_per_sec  # each micro-step processes B*T tokens
    # Hours per 100M tokens
    hours_per_100m = (100_000_000 / effective_tok_per_sec) / 3600

    print(f"\n--- Training Throughput ---")
    print(f"  Per micro-step:  {train_sec_per_step*1000:.1f}ms")
    print(f"  Throughput:      {train_tok_per_sec:,.0f} tok/s (single micro-batch, fwd+bwd+optim)")
    print(f"  Est. hours/100M: {hours_per_100m:.1f}h")

    # Peak VRAM during training
    vram_final = torch.cuda.max_memory_allocated()
    print(f"  Peak VRAM:       {format_bytes(vram_final)}")

    # Cleanup
    del model, optimizer, x, logits, state
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "name": name,
        "total_params": counts["total"],
        "core_params": counts["core"],
        "plastic_params": counts["plastic"],
        "plastic_ratio": counts["ratio"],
        "peak_vram_inference": vram_after_fwd,
        "peak_vram_training": vram_final,
        "inference_tok_per_sec": tok_per_sec,
        "train_tok_per_sec": train_tok_per_sec,
        "hours_per_100m": hours_per_100m,
    }


def main():
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available. This benchmark requires a GPU.")
        sys.exit(1)

    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem = torch.cuda.get_device_properties(0).total_memory
    print(f"GPU: {gpu_name}")
    print(f"Total VRAM: {format_bytes(gpu_mem)}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA: {torch.version.cuda}")
    print(f"BF16 support: {torch.cuda.is_bf16_supported()}")

    results = []

    # Benchmark dev_15m
    cfg_15m = dev_15m()
    r1 = benchmark_config("dev_15m", cfg_15m)
    results.append(r1)

    # Benchmark local_200m
    cfg_200m = local_200m()
    r2 = benchmark_config("local_200m", cfg_200m)
    results.append(r2)

    # Summary table
    print(f"\n{'='*70}")
    print(f"  SUMMARY")
    print(f"{'='*70}")
    print(f"{'Preset':<15} {'Params':>10} {'VRAM Inf':>10} {'VRAM Train':>12} {'Tok/s Inf':>10} {'Tok/s Train':>12} {'h/100M':>8}")
    print(f"{'-'*15} {'-'*10} {'-'*10} {'-'*12} {'-'*10} {'-'*12} {'-'*8}")
    for r in results:
        print(f"{r['name']:<15} "
              f"{r['total_params']/1e6:>8.1f}M "
              f"{format_bytes(r['peak_vram_inference']):>10} "
              f"{format_bytes(r['peak_vram_training']):>12} "
              f"{r['inference_tok_per_sec']:>10,.0f} "
              f"{r['train_tok_per_sec']:>12,.0f} "
              f"{r['hours_per_100m']:>7.1f}h")

    print(f"\nBenchmark complete. All shapes verified. ✓")


if __name__ == "__main__":
    main()
