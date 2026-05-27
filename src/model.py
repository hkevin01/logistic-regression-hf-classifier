"""
# ============================================================
# ID            : MOD-MODEL-001
# Requirement   : Train a Logistic Regression classifier on a
#                 pre-built feature matrix and return the fitted
#                 model together with its predictions.
# Purpose       : Isolate all sklearn model code so hyperparameters
#                 can be tuned and models swapped in one place.
# Rationale     : LogisticRegression with saga solver scales to
#                 large TF-IDF matrices and supports multi-class
#                 classification natively via softmax.
# Inputs        :
#   X_train - np.ndarray (N_train, F) float32.
#   y_train - list[int] or np.ndarray (N_train,).
#   C       - float  : inverse regularisation strength.
#   max_iter- int    : solver iteration budget.
#   solver  - str    : sklearn solver name.
# Outputs       :
#   model   - fitted LogisticRegression instance.
# Side Effects  : None (pure computation).
# Failure Modes : ConvergenceWarning if max_iter too low.
# Error Handling: Raises ValueError on shape mismatch.
# Verification  : tests/test_model.py
# References    : https://scikit-learn.org/stable/modules/
#                 linear_model.html#logistic-regression
# ============================================================
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def train_logistic_regression(
    X_train: np.ndarray,
    y_train: Any,
    C: float = 1.0,
    max_iter: int = 1000,
    solver: str = "saga",
    random_state: int = 42,
) -> LogisticRegression:
    """
    # ID: FUNC-TRAIN-001
    # Requirement   : Fit LogisticRegression and return the trained model.
    # Purpose       : Core training step called by main.py.
    # Inputs        :
    #   X_train     - float32 feature matrix (N_train, F).
    #   y_train     - integer label array (N_train,).
    #   C           - regularisation inverse; smaller = stronger L2 penalty.
    #   max_iter    - maximum solver iterations; increase if not converging.
    #   solver      - 'saga' (large sparse), 'lbfgs' (dense small data).
    #   random_state- RNG seed for reproducibility.
    # Outputs       : fitted LogisticRegression.
    # Preconditions : X_train.shape[0] == len(y_train).
    # Error Handling: Raises ValueError on shape mismatch (sklearn).
    """
    # --- Shape validation ---
    if len(X_train) != len(y_train):
        raise ValueError(
            f"X_train has {len(X_train)} rows but y_train has {len(y_train)} "
            "labels."
        )

    n_classes = len(set(y_train))
    print(
        f"[model] Training LogisticRegression: "
        f"n_samples={len(X_train)}, n_features={X_train.shape[1]}, "
        f"n_classes={n_classes}, C={C}, solver={solver} ..."
    )

    clf = LogisticRegression(
        C=C,
        max_iter=max_iter,
        solver=solver,
        # multi_class removed in sklearn 1.7 - multinomial is the default
        random_state=random_state,
        # n_jobs removed from effect in sklearn 1.8
    )
    clf.fit(X_train, y_train)
    print("[model] Training complete.")

    return clf


def predict(model: LogisticRegression, X: np.ndarray) -> np.ndarray:
    """
    # ID: FUNC-PREDICT-001
    # Purpose   : Convenience wrapper around model.predict().
    # Inputs    : model - fitted LR; X - (N, F) feature matrix.
    # Outputs   : np.ndarray (N,) integer predictions.
    """
    return model.predict(X)
