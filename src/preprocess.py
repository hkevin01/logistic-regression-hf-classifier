"""
# ============================================================
# ID            : MOD-PREPROCESS-001
# Requirement   : Transform a list of raw text strings into a
#                 numerical feature matrix ready for sklearn.
# Purpose       : Decouple feature engineering from model code
#                 so vectorisers can be swapped without touching
#                 training or evaluation logic.
# Rationale     : Two modes are offered -
#                 'tfidf'  : fast, no GPU, good baseline.
#                 'sbert'  : richer semantic embeddings, slower.
# Inputs        :
#   texts        - list[str]   : raw text samples.
#   mode         - str         : 'tfidf' | 'sbert'.
#   max_features - int         : TF-IDF vocabulary cap.
#   model_name   - str         : SentenceTransformer model id.
#   vectorizer   - fitted obj  : pass a pre-fitted vectoriser
#                                for transform-only (test set).
# Outputs       :
#   X            - np.ndarray (N, F) float32 feature matrix.
#   vectorizer   - fitted object (TfidfVectorizer or None for
#                  SBERT where the model encodes directly).
# Side Effects  : Model weights cached on first SBERT call.
# Failure Modes : ValueError on unknown mode; ImportError if
#                 sentence-transformers not installed for SBERT.
# Verification  : tests/test_preprocess.py
# References    : https://scikit-learn.org/stable/modules/
#                 feature_extraction.html#text-feature-extraction
# ============================================================
"""

from __future__ import annotations

from typing import Any, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_features(
    texts: list,
    mode: str = "tfidf",
    max_features: int = 20_000,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    vectorizer: Optional[Any] = None,
) -> Tuple[np.ndarray, Any]:
    """
    # ID: FUNC-FEATURES-001
    # Requirement   : Produce a float32 feature matrix from text.
    # Purpose       : Single entry-point for the feature-engineering
    #                 step used by both training and inference paths.
    # Inputs        :
    #   texts       - list[str] of N text samples.
    #   mode        - 'tfidf' or 'sbert'.
    #   max_features- TF-IDF only: vocabulary size.
    #   model_name  - SBERT only: HF model identifier.
    #   vectorizer  - If supplied, call .transform() not .fit_transform()
    #                 (used on the test split to avoid data leakage).
    # Outputs       :
    #   X           - np.ndarray (N, F) float32.
    #   vectorizer  - fitted TfidfVectorizer (or None for SBERT).
    # Preconditions : len(texts) >= 1.
    # Error Handling: Raises ValueError on unknown mode.
    """
    # --- Input validation ---
    if not texts:
        raise ValueError("build_features: texts list is empty.")

    mode = mode.lower()

    if mode == "tfidf":
        return _tfidf_features(texts, max_features, vectorizer)
    elif mode == "sbert":
        return _sbert_features(texts, model_name)
    else:
        raise ValueError(
            f"Unknown mode='{mode}'. Choose 'tfidf' or 'sbert'."
        )


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _tfidf_features(
    texts: list,
    max_features: int,
    vectorizer: Optional[Any],
) -> Tuple[np.ndarray, Any]:
    """
    # ID: HELP-TFIDF-001
    # Purpose   : Fit (or reuse) a TF-IDF vectoriser and return a
    #             sparse-to-dense float32 matrix.
    # Inputs    : texts, max_features, optional pre-fitted vectorizer.
    # Outputs   : (X float32, fitted TfidfVectorizer).
    # Rationale : Fitting on training data only prevents data leakage.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer  # deferred

    if vectorizer is None:
        # Fit on training data
        print(
            f"[preprocess] Fitting TF-IDF vectoriser "
            f"(max_features={max_features}) ..."
        )
        vectorizer = TfidfVectorizer(
            max_features=max_features,
            sublinear_tf=True,       # log(1+tf) - compresses high-frequency terms
            strip_accents="unicode",
            analyzer="word",
            ngram_range=(1, 2),      # unigrams + bigrams
            min_df=2,                # ignore rare terms
        )
        X = vectorizer.fit_transform(texts)
    else:
        print("[preprocess] Transforming with pre-fitted TF-IDF vectoriser ...")
        X = vectorizer.transform(texts)

    # Convert sparse -> dense float32
    X_dense = X.toarray().astype(np.float32)
    print(f"[preprocess] TF-IDF feature matrix shape: {X_dense.shape}")
    return X_dense, vectorizer


def _sbert_features(
    texts: list,
    model_name: str,
) -> Tuple[np.ndarray, None]:
    """
    # ID: HELP-SBERT-001
    # Purpose   : Encode texts with a SentenceTransformer model.
    # Inputs    : texts, model_name.
    # Outputs   : (X float32 L2-normalised, None) - no vectoriser object.
    # Rationale : SBERT embeddings are already contextual; no fitting
    #             step needed - same model used for train and test.
    """
    from sentence_transformers import SentenceTransformer  # deferred

    print(f"[preprocess] Loading SBERT model '{model_name}' ...")
    model = SentenceTransformer(model_name)
    print(f"[preprocess] Encoding {len(texts)} texts ...")
    X = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype(np.float32)
    print(f"[preprocess] SBERT feature matrix shape: {X.shape}")
    return X, None
