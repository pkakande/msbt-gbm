#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
benchmark_model.py

Measure forward and backward throughput of your GPT model on CPU
using synthetic data.

Usage:
  python scripts/benchmark_model.py \
    --vocab-size 1000 \
    --seq-len 128 \
    --emb-dim 256 \
    --n-heads 4 \
    --n-layers 4 \
    --batch-size 32 \
    --iters 20
"""

import time
import argparse
import sys
from pathlib import Path

import torch

# ────────────────────────────────────────────────────────────
# Ensure project root is on path
# ────────────────────────────────────────────────────────────
ROOT = Path(__file__).parents[1].resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.model import GPT, GPTConfig, __version__  # noqa:E402

def benchmark(model, batch, iters=10, device="cpu"):
    """
    Run `iters` forward+backward passes and return avg times (s).
    """
    model = model.to(device)
    batch = batch.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss()

    # Warm-up
    for _ in range(3):
        logits = model(batch)
        loss = criterion(logits.view(-1, logits.size(-1)),
                         batch.view(-1))
        optimizer.zero_grad()
        loss.backward()

    # Forward benchmark
    torch.cuda.empty_cache() if device.startswith("cuda") else None
    t0 = time.time()
    for _ in range(iters):
        _ = model(batch)
    t1 = time.time()

    # Backward benchmark
    optimizer.zero_grad()
    t2 = time.time()
    for _ in range(iters):
        logits = model(batch)
        loss = criterion(logits.view(-1, logits.size(-1)),
                         batch.view(-1))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    t3 = time.time()

    fwd_time = (t1 - t0) / iters
    bwd_time = (t3 - t2) / iters
    return fwd_time, bwd_time

def main():
    parser = argparse.ArgumentParser(description="Benchmark GPT Model Throughput")
    parser.add_argument("--version", action="store_true", help="Show version and exit")
    parser.add_argument("--vocab-size", type=int, required=True, help="Vocabulary size")
    parser.add_argument("--seq-len",    type=int, default=128, help="Sequence length")
    parser.add_argument("--emb-dim",    type=int, default=256, help="Embedding dimension")
    parser.add_argument("--n-heads",    type=int, default=4, help="Attention heads")
    parser.add_argument("--n-layers",   type=int, default=4, help="Transformer layers")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--iters",      type=int, default=20, help="Number of iterations")
    parser.add_argument("--device",     type=str, default="cpu", help="Device: cpu or cuda")
    args = parser.parse_args()

    if args.version:
        print(__version__)
        sys.exit(0)

    config = GPTConfig(
        vocab_size=args.vocab_size,
        max_seq_len=args.seq_len,
        emb_dim=args.emb_dim,
        n_heads=args.n_heads,
        n_layers=args.n_layers
    )
    model = GPT(config)

    # synthetic batch of token IDs
    batch = torch.randint(
        low=0,
        high=args.vocab_size,
        size=(args.batch_size, args.seq_len),
        dtype=torch.long
    )

    print(f"Benchmarking v{__version__} on {args.device.upper()}")
    print(f"Config: vocab={args.vocab_size}, seq_len={args.seq_len}, "
          f"emb_dim={args.emb_dim}, heads={args.n_heads}, layers={args.n_layers}")
    print(f"Batch: size={args.batch_size}, iterations={args.iters}")

    fwd_time, bwd_time = benchmark(model, batch, args.iters, args.device)
    tokens_per_sec_fwd = args.batch_size * args.seq_len / fwd_time
    tokens_per_sec_bwd = args.batch_size * args.seq_len / bwd_time

    print(f"\nResults (avg over {args.iters} iters):")
    print(f"  Forward pass: {fwd_time:.4f}s → {tokens_per_sec_fwd:,.0f} tokens/s")
    print(f"  Backward pass (incl. opt.step): {bwd_time:.4f}s → "
          f"{tokens_per_sec_bwd:,.0f} tokens/s")

if __name__ == "__main__":
    main()