"""Tests for evaluation provenance / audit trail surfaces.

Covers ``_check_success_criteria`` - the helper that decides whether
evaluation metrics meet caller-supplied gates and emits the
``success_criteria_met`` / ``success_criteria_results`` audit fields the
downstream model-registry, mlflow_logger, and monitoring code consume.

(``chosen_threshold_source`` provenance assertions live alongside the
threshold-tuning tests in ``test_threshold_selection.py`` because they
are inseparable from the threshold-tuning behaviour they describe.)

Split from ``test_evaluator.py`` in 1A-M-6. Test names preserved
verbatim (CI history follows ``Class::method``).
"""

from src.agents.ml_foundation.model_trainer.nodes.evaluator import (
    _check_success_criteria,
)


class TestCheckSuccessCriteria:
    """Test success criteria checking."""

    def test_all_criteria_met(self):
        """Should return True when all criteria met."""
        test_metrics = {"accuracy": 0.85, "roc_auc": 0.90}
        success_criteria = {"accuracy": 0.80, "auc": 0.85}

        result = _check_success_criteria(test_metrics, success_criteria, "binary_classification")

        assert result["success_criteria_met"] is True

    def test_criteria_not_met(self):
        """Should return False when criteria not met."""
        test_metrics = {"accuracy": 0.75, "roc_auc": 0.80}
        success_criteria = {"accuracy": 0.90}

        result = _check_success_criteria(test_metrics, success_criteria, "binary_classification")

        assert result["success_criteria_met"] is False
        assert result["success_criteria_results"]["accuracy"] is False

    def test_lower_is_better_metrics(self):
        """Should correctly handle metrics where lower is better."""
        test_metrics = {"rmse": 0.1, "mae": 0.05}
        success_criteria = {"rmse": 0.2, "mae": 0.1}

        result = _check_success_criteria(test_metrics, success_criteria, "regression")

        assert result["success_criteria_met"] is True

    def test_empty_criteria_returns_true(self):
        """Should return True when no criteria specified."""
        result = _check_success_criteria({}, {}, "binary_classification")

        assert result["success_criteria_met"] is True

    def test_handles_missing_metrics(self):
        """Should handle missing metrics gracefully."""
        test_metrics = {"accuracy": 0.85}
        success_criteria = {"accuracy": 0.80, "nonexistent_metric": 0.5}

        result = _check_success_criteria(test_metrics, success_criteria, "binary_classification")

        assert result["success_criteria_met"] is False
        assert result["success_criteria_results"]["nonexistent_metric"] is False
