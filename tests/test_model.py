"""
# ID: TEST-MODEL-001
# Purpose: Unit tests for src/model.train_logistic_regression() and predict().
#          Uses synthetic in-memory data - no network required.
"""
import numpy as np
import pytest
from src.model import train_logistic_regression, predict


def _make_data(n=60, n_features=20, n_classes=3, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, n_features)).astype(np.float32)
    y = (np.arange(n) % n_classes).tolist()
    return X, y


class TestTrainLogisticRegression:
    def test_returns_model(self):
        X, y = _make_data()
        model = train_logistic_regression(X, y)
        assert hasattr(model, "predict")

    def test_predict_shape(self):
        X, y = _make_data()
        model = train_logistic_regression(X, y)
        preds = predict(model, X)
        assert preds.shape == (len(y),)

    def test_shape_mismatch_raises(self):
        X, y = _make_data()
        with pytest.raises(ValueError):
            train_logistic_regression(X, y[:-1])  # one label fewer
