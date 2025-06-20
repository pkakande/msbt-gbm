#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
train.py

CPU-optimized training loop for MolGPT.

Features:
  - Reads hyperparameters and paths from CLI.
  - Loads processed SMILES dataset & token→ID vocab.
  - Pads/truncates to fixed seq_len, builds input/target pairs.
  - Builds GPT, optimizer, loss; trains with checkpointing + resume.
  - Exposes --version flag.
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# ────────────────────────────────────────────────────────────
# Ensure src/ is on PYTHONPATH
# ────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parents[1].resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.model import GPT, GPTConfig, __version__  # noqa: E402


class SmilesDataset(Dataset):
    """
    Dataset for tokenized SMILES.

    Each row in the CSV must have a column "tokens_union" containing a
    JSON list of tokens (e.g. ["C","C","(","O",")",...]). We map each token
    → an integer ID, pad/truncate to seq_len, and create input/target pairs.

    Args:
      csv_path: path to processed CSV
      vocab_json: path to token→ID JSON (expects a dict {token: id})
      seq_len: maximum sequence length; longer are truncated, shorter are padded
    """
    def __init__(self, csv_path: str, vocab_json: str, seq_len: int):
        self.seq_len = seq_len

        # load token→ID mapping
        with open(vocab_json, "r") as f:
            self.token2id = json.load(f)
        # optional: define a PAD token ID; if not present, use 0
        self.pad_id = self.token2id.get("<pad>", 0)

        # load the CSV and parse the JSON tokens_union column
        import pandas as pd
        df = pd.read_csv(
            csv_path,
            usecols=["tokens_union"],
            converters={"tokens_union": lambda s: json.loads(s.replace("'", '"'))}
        )
        self.token_lists = df.tokens_union.tolist()
        # pre-build ID sequences
        self.seqs = [self._tokens_to_ids(toks) for toks in self.token_lists]

    def _tokens_to_ids(self, tokens):
        """
        Map list-of-str tokens → list-of-int IDs, then pad/truncate to seq_len.
        """
        ids = [self.token2id.get(tok, self.pad_id) for tok in tokens]
        # truncate
        if len(ids) > self.seq_len:
            ids = ids[: self.seq_len]
        # pad
        pad_len = self.seq_len - len(ids)
        if pad_len > 0:
            ids = ids + [self.pad_id] * pad_len
        return ids

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        """
        Returns:
          input_ids: LongTensor [seq_len], the token ID sequence.
          target_ids: LongTensor [seq_len], same as input but shifted left by 1,
                      with pad_id appended at end.
        """
        ids = self.seqs[idx]
        input_ids = torch.LongTensor(ids)
        # shift for next-token target
        target_ids = ids[1:] + [self.pad_id]
        target_ids = torch.LongTensor(target_ids)
        return input_ids, target_ids


def save_checkpoint(state, checkpoint_dir: Path, epoch: int):
    ckpt_path = checkpoint_dir / f"checkpoint_epoch{epoch}.pt"
    torch.save(state, ckpt_path)
    print(f"[train] Saved checkpoint: {ckpt_path}")


def load_checkpoint(checkpoint_path, model, optimizer):
    print(f"[train] Loading checkpoint: {checkpoint_path}")
    state = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state["model_state"])
    optimizer.load_state_dict(state["optim_state"])
    cfg_dict = state["config"]
    return cfg_dict, state.get("epoch", 0)


def main():
    parser = argparse.ArgumentParser(description="Train MolGPT model")
    parser.add_argument("--version", action="store_true", help="Show version and exit")
    parser.add_argument("--data-csv", required=True, help="Path to processed SMILES CSV")
    parser.add_argument("--vocab-json", required=True, help="Path to token→ID JSON")
    parser.add_argument("--seq-len", type=int, default=128, help="Max sequence length")
    parser.add_argument("--vocab-size", type=int, required=True, help="Size of vocab")
    parser.add_argument("--emb-dim", type=int, default=256, help="Embedding dimension")
    parser.add_argument("--n-heads", type=int, default=4, help="Number of attention heads")
    parser.add_argument("--n-layers", type=int, default=4, help="Number of transformer layers")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--checkpoint-dir", default="outputs/checkpoints", help="Where to save checkpoints")
    parser.add_argument("--resume", help="Path to checkpoint to resume from")
    args = parser.parse_args()

    if args.version:
        print(__version__)
        sys.exit(0)

    # prepare checkpoint directory
    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # build model config & instance
    config = GPTConfig(
        vocab_size=args.vocab_size,
        max_seq_len=args.seq_len,
        emb_dim=args.emb_dim,
        n_heads=args.n_heads,
        n_layers=args.n_layers
    )
    model = GPT(config)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    start_epoch = 1
    if args.resume:
        cfg_dict, start_epoch = load_checkpoint(args.resume, model, optimizer)
        # TODO: optionally override args from cfg_dict

    # build dataset + dataloader
    dataset = SmilesDataset(args.data_csv, args.vocab_json, args.seq_len)
    dl = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # training loop
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        total_loss = 0.0

        for input_ids, target_ids in dl:
            logits = model(input_ids)  # (bsz, seq_len, vocab_size)
            loss = criterion(
                logits.view(-1, args.vocab_size),
                target_ids.view(-1)
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(dl)
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{now}] Epoch {epoch}/{args.epochs} — loss: {avg_loss:.4f}")

        # save checkpoint
        state = {
            "model_state": model.state_dict(),
            "optim_state": optimizer.state_dict(),
            "config": config.get_config(),
            "version": __version__,
            "epoch": epoch,
        }
        save_checkpoint(state, ckpt_dir, epoch)


if __name__ == "__main__":
    main()