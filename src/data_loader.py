"""
# ============================================================
# ID            : MOD-DATA-001
# Requirement   : Load a Hugging Face dataset and return clean
#                 (texts, labels) or (features, labels) tuples.
# Purpose       : Centralise all I/O so the rest of the pipeline
#                 consumes a uniform interface regardless of the
#                 dataset chosen.
# Rationale     : Isolation of data-loading simplifies dataset
#                 swapping - only this module needs to change.
# Inputs        : dataset_name, split, text_col, label_col,
#                 max_samples (all configurable from main.py).
# Outputs       : Tuple[List[str], List[int]]
# Side Effects  : HF cache written to ~/.cache/huggingface/
# Failure Modes : ValueError on bad column name; network error
#                 on first download.
# Verification  : tests/test_data_loader.py
# References    : https://huggingface.co/docs/datasets
# ============================================================
"""

from __future__ import annotations

import re
from typing import List, Optional, Tuple


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _clean_text(text: str) -> str:
    """
    # ID: HELP-CLEAN-001
    # Purpose   : Strip surrounding whitespace; collapse internal runs.
    # Inputs    : text - raw string
    # Outputs   : normalised string
    """
    return re.sub(r"\s+", " ", text.strip())


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_hf_dataset(
    dataset_name: str = "ag_news",
    split: str = "train",
    text_column: str = "text",
    label_column: Optional[str] = "label",
    max_samples: int = 5000,
) -> Tuple[List[str], List[int]]:
    """
    # ID: FUNC-LOAD-001
    # Requirement   : Return (texts, labels) for the chosen HF dataset.
    # Inputs        :
    #   dataset_name - HF dataset slug (e.g. 'ag_news', 'imdb').
    #   split        - 'train', 'test', or 'validation'.
    #   text_column  - Column name containing raw text.
    #   label_column - Column name for integer labels (None = unsupervised).
    #   max_samples  - Hard cap; keeps RAM usage predictable.
    # Outputs       :
    #   texts  - list[str] of cleaned text, len <= max_samples.
    #   labels - list[int] of integer labels (empty if label_column=None).
    # Preconditions : Network available on first call.
    # Error Handling: Raises ValueError with available column names.
    """
    # --- Guard: max_samples ---
    if max_samples < 2:
        raise ValueError(f"max_samples must be >= 2, got {max_samples}.")

    # --- Load (deferred import keeps module importable without 'datasets') ---
    from datasets import load_dataset

    print(f"[data_loader] Loading '{dataset_name}' split='{split}' ...")
    dataset = load_dataset(dataset_name, split=split)

    # --- Validate columns ---
    available = dataset.column_names
    if text_column not in available:
        raise ValueError(
            f"text_column='{text_column}' not found. Available: {available}"
        )
    if label_column is not None and label_column not in available:
        raise ValueError(
            f"label_column='{label_column}' not found. Available: {available}"
        )

    # --- Subsample ---
    n = min(max_samples, len(dataset))
    dataset = dataset.select(range(n))
    print(f"[data_loader] Using {n} samples.")

    # --- Extract ---
    texts: List[str] = [_clean_text(str(row[text_column])) for row in dataset]
    labels: List[int] = (
        [int(row[label_column]) for row in dataset]
        if label_column is not None
        else []
    )

    return texts, labels
