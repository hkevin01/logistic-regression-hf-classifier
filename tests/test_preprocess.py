"""
# ID: TEST-PRE-001
# Purpose: Unit tests for src/preprocess.build_features() in TF-IDF mode.
#          No network access required.
"""
import numpy as np
import pytest
from src.preprocess import build_features


SAMPLE_TEXTS = [
    "the quick brown fox jumps over the lazy dog",
    "python is a great programming language",
    "machine learning with sklearn is fun",
    "natural language processing and text mining",
    "deep learning models are powerful tools",
]


class TestBuildFeaturesTFIDF:
    def test_shape(self):
        X, vec = build_features(SAMPLE_TEXTS, mode="tfidf", max_features=100)
        assert X.shape[0] == len(SAMPLE_TEXTS)
        assert X.shape[1] <= 100

    def test_dtype_float32(self):
        X, _ = build_features(SAMPLE_TEXTS, mode="tfidf", max_features=100)
        assert X.dtype == np.float32

    def test_reuse_vectorizer(self):
        X_train, vec = build_features(SAMPLE_TEXTS[:3], mode="tfidf", max_features=100)
        X_test, vec2 = build_features(SAMPLE_TEXTS[3:], mode="tfidf",
                                       max_features=100, vectorizer=vec)
        # Same number of features (vocabulary fixed at fit time)
        assert X_train.shape[1] == X_test.shape[1]
        assert vec2 is vec  # same object returned

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="empty"):
            build_features([], mode="tfidf")

    def test_unknown_mode_raises(self):
        with pytest.raises(ValueError, match="Unknown mode"):
            build_features(SAMPLE_TEXTS, mode="invalid_mode")
