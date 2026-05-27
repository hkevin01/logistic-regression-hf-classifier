"""
# ============================================================
# ID            : MOD-VIZ-001
# Requirement   : Generate and save two plots:
#                 1. Confusion matrix heatmap.
#                 2. Top-N TF-IDF feature importance bar chart
#                    (skipped automatically in SBERT mode).
# Purpose       : Provide visual diagnostics for classification
#                 quality and feature relevance.
# Rationale     : A confusion matrix reveals which classes are
#                 confused; feature importance helps audit TF-IDF
#                 decision boundaries.
# Inputs        :
#   y_true       - true labels (N,).
#   y_pred       - predicted labels (N,).
#   label_names  - class name strings.
#   output_dir   - folder path for PNG files.
#   vectorizer   - fitted TfidfVectorizer (None for SBERT mode).
#   model        - fitted LogisticRegression (for feature importance).
#   top_n        - how many top features to plot per class.
# Outputs       : PNG files in output_dir; returns list of paths.
# Side Effects  : File I/O to output_dir.
# Failure Modes : OSError if output_dir not writable.
# Verification  : tests/test_visualize.py
# References    : https://matplotlib.org/
# ============================================================
"""

from __future__ import annotations

import os
from typing import Any, List, Optional

import matplotlib
import numpy as np


# ---------------------------------------------------------------------------
# Confusion matrix
# ---------------------------------------------------------------------------

def plot_confusion_matrix(
    y_true: Any,
    y_pred: Any,
    label_names: Optional[List[str]] = None,
    output_path: Optional[str] = None,
    title: str = "Confusion Matrix",
) -> str:
    """
    # ID: FUNC-CM-001
    # Requirement   : Save a normalised confusion-matrix heatmap as PNG.
    # Inputs        :
    #   y_true      - ground-truth labels.
    #   y_pred      - predicted labels.
    #   label_names - class display names (None = use integers).
    #   output_path - PNG save path (None = interactive display).
    #   title       - figure title.
    # Outputs       : path string of saved PNG.
    # Postconditions: File exists at output_path when not None.
    """
    from sklearn.metrics import confusion_matrix  # deferred
    import seaborn as sns  # deferred
    import matplotlib.pyplot as plt  # deferred

    if output_path is not None:
        matplotlib.use("Agg")

    cm = confusion_matrix(y_true, y_pred)
    # Normalise rows to percentages
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    n_classes = cm.shape[0]
    fig_size = max(6, n_classes * 1.2)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size - 1))

    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=label_names or range(n_classes),
        yticklabels=label_names or range(n_classes),
        ax=ax,
        linewidths=0.5,
    )
    ax.set_title(title, fontsize=13)
    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("True", fontsize=11)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150)
        print(f"[visualize] Confusion matrix saved to '{output_path}'.")
    else:
        plt.show()

    plt.close(fig)
    return output_path or ""


# ---------------------------------------------------------------------------
# TF-IDF feature importance
# ---------------------------------------------------------------------------

def plot_feature_importance(
    model: Any,
    vectorizer: Any,
    label_names: Optional[List[str]] = None,
    top_n: int = 15,
    output_path: Optional[str] = None,
    title: str = "Top TF-IDF Features per Class",
) -> str:
    """
    # ID: FUNC-FEAT-001
    # Requirement   : Plot the top-N TF-IDF features per class using
    #                 LogisticRegression coefficients.
    # Purpose       : Audit which tokens drive each classification decision.
    # Inputs        :
    #   model       - fitted LogisticRegression with .coef_ attribute.
    #   vectorizer  - fitted TfidfVectorizer with .get_feature_names_out().
    #   label_names - class display names.
    #   top_n       - features to show per class.
    #   output_path - PNG save path (None = interactive display).
    # Outputs       : path string of saved PNG.
    # Preconditions : vectorizer is not None (TF-IDF mode).
    # Error Handling: Returns empty string silently if vectorizer is None.
    """
    if vectorizer is None:
        print("[visualize] Skipping feature importance (SBERT mode - no vectoriser).")
        return ""

    import matplotlib.pyplot as plt  # deferred

    if output_path is not None:
        matplotlib.use("Agg")

    feature_names = np.array(vectorizer.get_feature_names_out())
    coefs = model.coef_  # shape (n_classes, n_features)
    n_classes = coefs.shape[0]
    names = label_names or [f"Class {i}" for i in range(n_classes)]

    # One subplot per class
    fig, axes = plt.subplots(
        1, n_classes, figsize=(top_n * 0.8, 4), sharey=False
    )
    if n_classes == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        top_idx = np.argsort(coefs[i])[-top_n:][::-1]
        top_weights = coefs[i][top_idx]
        top_features = feature_names[top_idx]
        colors = ["#2196F3" if w >= 0 else "#F44336" for w in top_weights]

        ax.barh(range(top_n), top_weights[::-1], color=colors[::-1])
        ax.set_yticks(range(top_n))
        ax.set_yticklabels(top_features[::-1], fontsize=8)
        ax.set_title(names[i], fontsize=10)
        ax.axvline(0, color="black", linewidth=0.6)

    fig.suptitle(title, fontsize=12, y=1.01)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"[visualize] Feature importance saved to '{output_path}'.")
    else:
        plt.show()

    plt.close(fig)
    return output_path or ""
