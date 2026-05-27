"""
# ============================================================
# ID            : MOD-EVAL-001
# Requirement   : Compute and display classification metrics -
#                 accuracy, precision, recall, F1 - for a set
#                 of ground-truth and predicted labels.
# Purpose       : Provide a reusable evaluation step that prints
#                 a formatted report and returns a metrics dict.
# Rationale     : Wrapping sklearn metrics in one function keeps
#                 main.py concise and makes metric selection easy.
# Inputs        :
#   y_true       - array-like of true integer labels.
#   y_pred       - array-like of predicted integer labels.
#   label_names  - optional list[str] for display.
# Outputs       :
#   dict with keys: accuracy, precision, recall, f1, report.
# Side Effects  : Prints formatted report to stdout.
# Failure Modes : ValueError if label arrays differ in length.
# Verification  : tests/test_evaluate.py
# References    : https://scikit-learn.org/stable/modules/model_evaluation.html
# ============================================================
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def evaluate(
    y_true: Any,
    y_pred: Any,
    label_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    # ID: FUNC-EVAL-001
    # Requirement   : Return a metrics dictionary and print a report.
    # Purpose       : Single evaluation entry-point for main.py.
    # Inputs        :
    #   y_true      - ground-truth integer labels (N,).
    #   y_pred      - predicted integer labels (N,).
    #   label_names - optional human-readable class names.
    # Outputs       :
    #   dict with keys: accuracy, precision, recall, f1, report (str).
    # Preconditions : len(y_true) == len(y_pred) >= 1.
    # Error Handling: Propagates sklearn ValueError on shape mismatch.
    """
    # --- Core metrics (macro average is class-balanced) ---
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    report: str = classification_report(
        y_true,
        y_pred,
        target_names=label_names,
        zero_division=0,
    )

    # --- Print formatted summary ---
    print("\n" + "=" * 60)
    print("EVALUATION REPORT")
    print("=" * 60)
    print(f"  Accuracy  : {acc:.4f}")
    print(f"  Precision : {prec:.4f}  (macro)")
    print(f"  Recall    : {rec:.4f}  (macro)")
    print(f"  F1 Score  : {f1:.4f}  (macro)")
    print("\nPer-class breakdown:")
    print(report)
    print("=" * 60 + "\n")

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "report": report,
    }
