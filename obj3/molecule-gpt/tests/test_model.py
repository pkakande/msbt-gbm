# tests/test_model.py

import os
import sys

# ────────────────────────────────────────────────────────────
# Ensure project root is on PYTHONPATH so `import src.model` works
# ────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch
from src.model import GPT, GPTConfig

def test_gpt_forward_pass():
    """
    Basic forward-pass test for the GPT model:
    - Creates a small GPTConfig
    - Instantiates GPT
    - Runs a dummy batch through forward()
    - Checks output shape and non-constant logits
    """
    # 1) Build a tiny config
    vocab_size = 50
    max_seq_len = 20
    config = GPTConfig(
        vocab_size=vocab_size,
        max_seq_len=max_seq_len,
        emb_dim=64,
        n_heads=4,
        n_layers=2,
        dropout=0.0
    )

    # 2) Instantiate model
    model = GPT(config)

    # 3) Create dummy input (batch_size=3, seq_len=10)
    batch_size, seq_len = 3, 10
    dummy_input = torch.randint(0, vocab_size, (batch_size, seq_len), dtype=torch.long)

    # 4) Forward pass
    logits = model(dummy_input)

    # 5) Assertions
    assert logits.shape == (batch_size, seq_len, vocab_size), (
        f"Expected shape {(batch_size, seq_len, vocab_size)}, got {logits.shape}"
    )
    # Make sure not all logits are identical
    assert torch.any(logits != logits[0,0,0]), "Logits appear constant, check model initialization"