"""
# ============================================================
# ID            : MAIN-001
# Requirement   : Provide a single runnable script that executes
#                 the full Logistic Regression classification
#                 pipeline end-to-end.
# Purpose       : Orchestrate data loading, preprocessing,
#                 train/test split, training, evaluation, and
#                 visualisation using the src/ modules.
# Rationale     : All business logic lives in src/; main.py only
#                 wires the steps together and holds configuration.
# Inputs        : None - all options in PIPELINE_CONFIG below.
# Outputs       :
#   - Console evaluation report (accuracy, precision, recall, F1).
#   - outputs/confusion_matrix.png
#   - outputs/feature_importance.png  (TF-IDF mode only)
# Preconditions : pip install -r requirements.txt completed.
# Side Effects  : HF/model caches written; PNG files saved.
# Verification  : Run `python main.py` and inspect outputs/.
# ============================================================
"""

from __future__ import annotations

import os
from typing import Optional

from sklearn.model_selection import train_test_split

from src.data_loader import load_hf_dataset
from src.preprocess import build_features
from src.model import train_logistic_regression, predict
from src.evaluate import evaluate
from src.visualize import plot_confusion_matrix, plot_feature_importance


# ============================================================
# Pipeline Configuration
# Edit any value here to swap dataset, mode, or hyperparameters.
# ============================================================

PIPELINE_CONFIG = {
    # ---- Dataset --------------------------------------------------------
    # Swap to any HF dataset that has a text + integer-label column.
    # Examples at the bottom of this file.
    "dataset_name":  "ag_news",
    "split":         "train",
    "text_column":   "text",
    "label_column":  "label",
    "max_samples":   5000,        # reduce for a quick smoke-test

    # Human-readable class names (None = use integers).
    # ag_news classes: 0=World 1=Sports 2=Business 3=Sci/Tech
    "label_names":   ["World", "Sports", "Business", "Sci/Tech"],

    # ---- Feature engineering --------------------------------------------
    # mode: 'tfidf'  -> fast TF-IDF baseline (recommended for first run)
    # mode: 'sbert'  -> richer embeddings (needs sentence-transformers)
    "embedding_mode": "tfidf",
    "tfidf_max_features": 20_000,
    "sbert_model": "sentence-transformers/all-MiniLM-L6-v2",

    # ---- Train / test split ---------------------------------------------
    "test_size":    0.20,
    "random_state": 42,

    # ---- Logistic Regression hyperparameters ----------------------------
    "C":           1.0,
    "max_iter":    1000,
    "solver":      "saga",

    # ---- Output ---------------------------------------------------------
    "output_dir":  "outputs",
}


# ============================================================
# Alternative dataset snippets (replace the dataset block above)
# ============================================================
# imdb (2-class sentiment):
#   "dataset_name": "imdb", "text_column": "text",
#   "label_column": "label", "max_samples": 5000,
#   "label_names": ["negative", "positive"],
#
# emotion (6-class):
#   "dataset_name": "dair-ai/emotion", "text_column": "text",
#   "label_column": "label", "max_samples": 5000,
#   "label_names": ["sadness","joy","love","anger","fear","surprise"],
#
# 20 newsgroups via HF:
#   "dataset_name": "SetFit/20_newsgroups", "text_column": "text",
#   "label_column": "label", "max_samples": 5000,
#   "label_names": None,
# ============================================================


def main() -> None:
    """
    # ID: FUNC-MAIN-001
    # Requirement   : Run the full pipeline with PIPELINE_CONFIG settings.
    # Preconditions : requirements.txt packages installed.
    # Postconditions: Metrics printed; PNGs saved in outputs/.
    """
    cfg = PIPELINE_CONFIG
    os.makedirs(cfg["output_dir"], exist_ok=True)

    # ------------------------------------------------------------------
    # Step 1 - Load dataset
    # ------------------------------------------------------------------
    texts, labels = load_hf_dataset(
        dataset_name=cfg["dataset_name"],
        split=cfg["split"],
        text_column=cfg["text_column"],
        label_column=cfg["label_column"],
        max_samples=cfg["max_samples"],
    )

    # ------------------------------------------------------------------
    # Step 2 - Train / test split
    # Split BEFORE feature engineering to prevent data leakage.
    # ------------------------------------------------------------------
    texts_train, texts_test, y_train, y_test = train_test_split(
        texts,
        labels,
        test_size=cfg["test_size"],
        random_state=cfg["random_state"],
        stratify=labels,  # preserve class distribution in both splits
    )
    print(
        f"[main] Train: {len(texts_train)} samples | "
        f"Test: {len(texts_test)} samples"
    )

    # ------------------------------------------------------------------
    # Step 3 - Feature engineering
    # Fit on training data, transform both splits.
    # ------------------------------------------------------------------
    X_train, vectorizer = build_features(
        texts=texts_train,
        mode=cfg["embedding_mode"],
        max_features=cfg["tfidf_max_features"],
        model_name=cfg["sbert_model"],
        vectorizer=None,  # fit on train
    )

    X_test, _ = build_features(
        texts=texts_test,
        mode=cfg["embedding_mode"],
        max_features=cfg["tfidf_max_features"],
        model_name=cfg["sbert_model"],
        vectorizer=vectorizer,  # transform-only on test
    )

    # ------------------------------------------------------------------
    # Step 4 - Train Logistic Regression
    # ------------------------------------------------------------------
    model = train_logistic_regression(
        X_train=X_train,
        y_train=y_train,
        C=cfg["C"],
        max_iter=cfg["max_iter"],
        solver=cfg["solver"],
    )

    # ------------------------------------------------------------------
    # Step 5 - Predict on test set
    # ------------------------------------------------------------------
    y_pred = predict(model, X_test)

    # ------------------------------------------------------------------
    # Step 6 - Evaluate
    # ------------------------------------------------------------------
    metrics = evaluate(
        y_true=y_test,
        y_pred=y_pred,
        label_names=cfg["label_names"],
    )

    # ------------------------------------------------------------------
    # Step 7 - Confusion matrix plot
    # ------------------------------------------------------------------
    cm_path = os.path.join(cfg["output_dir"], "confusion_matrix.png")
    plot_confusion_matrix(
        y_true=y_test,
        y_pred=y_pred,
        label_names=cfg["label_names"],
        output_path=cm_path,
        title=f"Confusion Matrix - {cfg['dataset_name']} "
              f"(F1={metrics['f1']:.3f})",
    )

    # ------------------------------------------------------------------
    # Step 8 - Feature importance (TF-IDF mode only)
    # ------------------------------------------------------------------
    fi_path = os.path.join(cfg["output_dir"], "feature_importance.png")
    plot_feature_importance(
        model=model,
        vectorizer=vectorizer,
        label_names=cfg["label_names"],
        top_n=15,
        output_path=fi_path,
        title=f"Top TF-IDF Features - {cfg['dataset_name']}",
    )

    print(f"[main] Done. Outputs saved to '{cfg['output_dir']}/'.")


if __name__ == "__main__":
    main()
