"""CATE availability, heterogeneity evidence, and segment uncertainty are distinct."""

from __future__ import annotations

import numpy as np

from src.causal_engine.heterogeneity import analyze_heterogeneity


class _CoefficientInference:
    def __init__(self, pvalues):
        self._pvalues = np.asarray(pvalues, dtype=float)

    def pvalue(self):
        return self._pvalues


class _ATEInference:
    stderr_mean = 0.08

    def __init__(self, center):
        self._center = center

    def conf_int_mean(self):
        return self._center - 0.16, self._center + 0.16


class _LinearCateModel:
    def __init__(self, pvalues, cate):
        self._pvalues = pvalues
        self._cate = np.asarray(cate, dtype=float)

    def coef__inference(self):
        return _CoefficientInference(self._pvalues)

    def ate_inference(self, X):
        # X's first column indexes the rows in this deterministic test model.
        indices = np.asarray(X[:, 0], dtype=int)
        return _ATEInference(float(np.mean(self._cate[indices])))


def test_cate_availability_does_not_imply_detected_heterogeneity():
    cate = np.linspace(0.3, 0.7, 20)
    X = np.column_stack([np.arange(20), np.ones(20)])
    result = analyze_heterogeneity(model=_LinearCateModel([0.30, 0.40], cate), X=X, cate=cate)
    assert result.cate_available is True
    assert result.detected is False
    assert result.p_value == 0.60
    assert result.segments == []


def test_detected_heterogeneity_emits_interval_bearing_exploratory_strata():
    cate = np.linspace(0.1, 0.9, 20)
    X = np.column_stack([np.arange(20), np.ones(20)])
    result = analyze_heterogeneity(model=_LinearCateModel([0.001, 0.50], cate), X=X, cate=cate)
    assert result.detected is True
    assert result.p_value == 0.002
    assert {segment["segment"] for segment in result.segments} == {"High CATE", "Low CATE"}
    for segment in result.segments:
        assert segment["cate_ci_lower"] < segment["cate"] < segment["cate_ci_upper"]
        assert segment["standard_error"] > 0
        assert segment["validation_status"] == "exploratory_model_score_stratum"


def test_cate_without_inference_is_not_called_detected():
    cate = np.asarray([0.1, 0.9, 0.2, 0.8])
    result = analyze_heterogeneity(
        model=object(), X=np.arange(4, dtype=float).reshape(-1, 1), cate=cate
    )
    assert result.cate_available is True
    assert result.detected is False
    assert result.method == "inference_unavailable"
    assert result.segments == []
