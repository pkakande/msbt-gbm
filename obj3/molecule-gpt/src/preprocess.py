#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
preprocess.py

A distributed SMILES-preprocessing pipeline using Ray.

Features:
  - Reads raw SMILES from CSV (with header) or TSV (one per line).
  - Filters and canonicalizes molecules via RDKit in parallel.
  - Tokenizes canonical SMILES three ways: char, regex, SELFIES.
  - Builds and dumps vocabularies: char, regex, selfies, union.
  - Logs cluster topology and resource utilization.
  - Exposes --version, --input, --smiles-col, --output, --vocab flags.

Usage:
  python src/preprocess.py --input data/raw/my.csv \
                           --smiles-col smiles \
                           --output data/processed/mols.csv \
                           --vocab data/processed/vocab
"""

import sys
import argparse
from pathlib import Path

# ────────────────────────────────────────────────────────────
# Version: single source-of-truth
# ────────────────────────────────────────────────────────────
__version__ = Path(__file__).parents[1].joinpath("VERSION").read_text().strip()

import os
import re
import json
import logging
from typing import Optional, Tuple, List

import pandas as pd
import ray
from rdkit import Chem
import selfies

# ────────────────────────────────────────────────────────────
# Logger configuration
# ────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("preprocess")


# ────────────────────────────────────────────────────────────
# Tokenizer implementations
# ────────────────────────────────────────────────────────────
def char_tokenizer(smiles: str) -> List[str]:
    """Split a SMILES string into single-character tokens."""
    return list(smiles)


_SMILES_TOKEN_RE = re.compile(
    r'Cl|Br|'
    r'[\[\]\(\)=#%@\+\-\\\/\.:\*]|'
    r'\d+|'
    r'[BCOHNOPSFIabcnopsHh]|.'
)

def regex_tokenizer(smiles: str) -> List[str]:
    """Tokenize SMILES using a custom regex pattern."""
    return _SMILES_TOKEN_RE.findall(smiles)


def selfies_tokenizer(smiles: str) -> List[str]:
    """
    Encode SMILES to SELFIES, then split into tokens.
    Returns [] on errors.
    """
    try:
        sf = selfies.encoder(smiles)
        return list(selfies.split_selfies(sf))
    except Exception:
        return []


# ────────────────────────────────────────────────────────────
# Ray‐remote function for canonicalization
# ────────────────────────────────────────────────────────────
@ray.remote
def canonicalize(raw: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Convert a raw SMILES to its RDKit canonical form.

    Args:
        raw: Original SMILES string.

    Returns:
        (raw, canonical) if valid, else (None, None).
    """
    mol = Chem.MolFromSmiles(raw)
    if mol is None:
        return None, None
    can = Chem.MolToSmiles(mol, canonical=True)
    return raw, can


# ────────────────────────────────────────────────────────────
# Main function
# ────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Distributed SMILES preprocessing + vocab export"
    )
    parser.add_argument(
        "--version", action="store_true",
        help="Show version and exit"
    )
    parser.add_argument(
        "--input", "-i", required=True,
        help="Path to CSV (header) or TSV (one SMILES per line) input"
    )
    parser.add_argument(
        "--smiles-col", default="smiles",
        help="Column name for SMILES in CSV (default: smiles)"
    )
    parser.add_argument(
        "--output", "-o", required=True,
        help="Output CSV path for raw, canonical, tokens_*"
    )
    parser.add_argument(
        "--vocab", "-v",
        help="Base path (no .json) to write vocab JSONs (_char, _regex, _selfies, _union)"
    )
    args = parser.parse_args()

    # Version handling
    if args.version:
        print(__version__)
        sys.exit(0)

    # Initialize Ray
    head_addr = os.getenv("RAY_HEAD_ADDRESS")
    if head_addr:
        logger.info("Connecting to Ray at %s", head_addr)
        ray.init(address=head_addr)
    else:
        logger.info("Starting local Ray cluster")
        ray.init(_system_config={"num_heartbeats_timeout": 60})

    # Log cluster topology
    nodes = ray.nodes()
    alive = sum(1 for n in nodes if n["Alive"])
    logger.info("Detected %d Ray nodes (%d alive, %d dead)",
                len(nodes), alive, len(nodes) - alive)
    for n in nodes:
        addr = n.get("NodeManagerAddress", "unknown")
        status = "alive" if n["Alive"] else "dead"
        cpus = n.get("Resources", {}).get("CPU", "?")
        logger.info("  • %s [%s], CPUs=%s", addr, status, cpus)

    # Log resource summary
    total_res = ray.cluster_resources()
    avail_res = ray.available_resources()
    logger.info("Total cluster resources: %s", total_res)
    logger.info("Available cluster resources: %s", avail_res)

    # Load SMILES
    logger.info("Loading SMILES from %s", args.input)
    if args.input.lower().endswith(".csv"):
        df = pd.read_csv(args.input, usecols=[args.smiles_col], dtype=str)
        df = df.rename(columns={args.smiles_col: "smiles"})
    else:
        df = pd.read_csv(
            args.input, sep="\t", names=["smiles"], comment="#", dtype=str
        )
    total = len(df)
    logger.info("%d total entries", total)

    # Canonicalize in parallel
    logger.info("Canonicalizing in parallel")
    futures = [canonicalize.remote(s) for s in df.smiles]
    results = ray.get(futures)
    valid = [(r, c) for r, c in results if c]
    valid_df = pd.DataFrame(valid, columns=["raw", "canonical"])
    logger.info("%d/%d valid molecules", len(valid_df), total)

    # Tokenize
    logger.info("Applying tokenizers (char, regex, selfies)")
    valid_df["tokens_char"]    = valid_df.canonical.map(char_tokenizer)
    valid_df["tokens_regex"]   = valid_df.canonical.map(regex_tokenizer)
    valid_df["tokens_selfies"] = valid_df.canonical.map(selfies_tokenizer)

    # Write processed CSV
    logger.info("Writing processed CSV to %s", args.output)
    valid_df.to_csv(args.output, index=False)

    # Build and dump vocabularies
    if args.vocab:
        logger.info("Building vocabularies with base: %s", args.vocab)
        tok_sets = {"char": set(), "regex": set(), "selfies": set()}
        for row in valid_df.itertuples():
            tok_sets["char"].update(row.tokens_char)
            tok_sets["regex"].update(row.tokens_regex)
            tok_sets["selfies"].update(row.tokens_selfies)

        base = args.vocab.rstrip(".json")
        union_set = set()
        for name, toks in tok_sets.items():
            path = f"{base}_{name}.json"
            with open(path, "w") as f:
                json.dump(sorted(toks), f, indent=2)
            logger.info("Saved %d tokens to %s", len(toks), path)
            union_set.update(toks)

        union_path = f"{base}_union.json"
        with open(union_path, "w") as f:
            json.dump(sorted(union_set), f, indent=2)
        logger.info("Saved %d union tokens to %s", len(union_set), union_path)


if __name__ == "__main__":
    main()