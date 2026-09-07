"""model_performance.build_grounding is metric-generic (2026-09-07).

The insight used to read the ACCURACY trend while the page card defaulted to
auc_roc, so the two could disagree about the same model. The grounding now
names the trended metric and carries the tracker's one-sentence basis for the
label so the LLM does not narrate a single-fold wobble as decay.
"""

from __future__ import annotations

from src.insights import model_performance as mp


def test_grounding_names_the_trended_metric_and_carries_the_reason():
    g = mp.build_grounding(
        model_version="hcp_adoption_remibrutinib_goldstd_lr_v1",
        current_value=0.7094,
        baseline_value=0.8152,
        trend="stable",
        confusion=None,
        auc=0.79,
        alerts=[],
        metric_name="auc_roc",
        trend_reason="-13.0% vs the 8-fold baseline is within sampling noise at n=134 (z=-2.1; ±2.5 needed)",
    )
    assert g["metric_name"] == "auc_roc"
    assert g["trend_summary"].startswith("AUC-ROC 0.709 vs baseline 0.815")
    assert "within sampling noise" in g["trend_summary"]
    labels = [c["label"] for c in g["grounding"]]
    assert labels[:3] == ["Current AUC-ROC", "Baseline AUC-ROC", "Trend"]
    # The holdout ROC AUC is a DIFFERENT number from the walk-forward trend
    # value; its chip says so.
    assert "Holdout AUC" in labels
    assert g["auc_summary"] == "holdout ROC AUC 0.790"
    assert "accuracy_summary" not in g


def test_grounding_default_metric_matches_the_page_default_and_fallback_reads_it():
    g = mp.build_grounding("m", 0.9, 0.88, "improving", None, None, None)
    assert g["metric_name"] == "auc_roc"
    assert g["trend_summary"] == "AUC-ROC 0.900 vs baseline 0.880 (Δ+0.020), trend improving"
    fb = mp._fallback(g)
    assert fb["is_fallback"] is True
    assert g["trend_summary"] in fb["insight"]
    assert fb["key_takeaways"][0] == g["trend_summary"]


def test_grounding_for_a_rate_metric_uses_its_label():
    g = mp.build_grounding("m", 0.61, 0.66, "stable", None, None, None, metric_name="recall")
    assert g["trend_summary"].startswith("Recall 0.610 vs baseline 0.660")
    assert g["grounding"][0]["label"] == "Current Recall"
