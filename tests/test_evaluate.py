"""
# ID: TEST-EVAL-001
# Purpose: Unit tests for src/evaluate.evaluate().
#          Uses synthetic perfect and random predictions.
"""
import pytest
from src.evaluate import evaluate


class TestEvaluate:
    def test_perfect_predictions(self):
        y = [0, 1, 2, 0, 1, 2]
        metrics = evaluate(y, y)
        assert metrics["accuracy"] == pytest.approx(1.0)
        assert metrics["f1"] == pytest.approx(1.0)

    def test_keys_present(self):
        y_true = [0, 1, 0, 1]
        y_pred = [0, 0, 0, 1]
        metrics = evaluate(y_true, y_pred)
        for key in ("accuracy", "precision", "recall", "f1", "report"):
            assert key in metrics

    def test_accuracy_range(self):
        y_true = [0, 1, 2] * 10
        y_pred = [0, 1, 0] * 10
        metrics = evaluate(y_true, y_pred)
        assert 0.0 <= metrics["accuracy"] <= 1.0
