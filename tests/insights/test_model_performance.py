from src.insights.model_performance import build_grounding, generate_insight


def test_build_grounding_derives_prf_and_chips():
    g = build_grounding(
        model_version="csu_adherence_v3",
        current_accuracy=0.86,
        baseline_accuracy=0.81,
        trend="improving",
        confusion={"tn": 80, "fp": 10, "fn": 12, "tp": 98},
        auc=0.88,
        alerts=[{"metric_name": "precision", "severity": "warning"}],
    )
    assert any(c["label"] == "Accuracy" and c["value"].startswith("0.86") for c in g["grounding"])
    assert "precision" in g["confusion_summary"].lower()


def test_generate_insight_fallback_grounded():
    g = build_grounding(
        "m1", 0.86, 0.81, "improving", {"tn": 80, "fp": 10, "fn": 12, "tp": 98}, 0.88, []
    )
    out = generate_insight(g)
    assert out["is_fallback"] is True
    assert "0.86" in out["insight"] and "0.88" in out["insight"]
