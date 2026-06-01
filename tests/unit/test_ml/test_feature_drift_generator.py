"""#577 WS1-MP-009: the data_generator mirror for ml_drift_history must be COHERENT — the
generated rows are self-reproducible and grounded in the generated feature_distributions, so a
from-scratch regenerate yields coherent (not byte-identical) PSI, exactly like migration 053.

Construction is cheap (E2IDataGenerator.__init__ only initializes empty lists), so we inject a
known feature_distributions and call the method directly — no full generate_all().
"""

from src.ml.data_generator import _DRIFT_MAP, E2IDataGenerator, _compute_feature_psi


def _gen_with_dist(dist):
    g = E2IDataGenerator()
    g.preprocessing_metadata = {"feature_distributions": dist}
    g._generate_ml_drift_history()
    return g.ml_drift_history


def test_generated_rows_are_self_reproducible_and_coherent():
    """Each generated row's test_statistic recomputes from its OWN stored baseline/current
    mean/std (<5e-7); baseline EQUALS the input feature_distributions; the drift is real (PSI>0)."""
    dist = {
        "age": {"mean": 0.6178, "std": 0.2537},
        "risk_score": {"mean": 0.6739, "std": 0.1466},
        "comorbidity_count": {"mean": 0.4261, "std": 0.1788},
    }
    rows = _gen_with_dist(dist)
    assert {r["feature_name"] for r in rows} == set(dist)
    for r in rows:
        f = r["feature_name"]
        # baseline mirrors the source distribution exactly (the YAML "source").
        assert r["baseline_mean"] == round(dist[f]["mean"], 4)
        assert r["baseline_std"] == round(dist[f]["std"], 4)
        # the stored PSI is reproducible from the stored (rounded) stats.
        recomputed = _compute_feature_psi(
            r["baseline_mean"], r["baseline_std"], r["current_mean"], r["current_std"]
        )
        assert abs(recomputed - r["test_statistic"]) < 5e-7
        assert r["test_statistic"] > 0.0  # genuinely drifted (not no-drift)
        assert r["test_type"] == "psi" and r["drift_type"] == "data"
        assert r["model_id"] is None  # honest NULL — ml_model_registry has 0 rows
        assert r["detected_by"] == "kpi_577_seed"


def test_generated_severity_is_coherent_and_below_medium():
    """drift_detected = (PSI > 0.10); severity 'low' iff detected else 'none' — STRICTLY below
    'medium' (so the omitted 017 alert trigger could never fire)."""
    rows = _gen_with_dist({"age": {"mean": 0.6178, "std": 0.2537}})
    for r in rows:
        assert r["drift_detected"] == (r["test_statistic"] > 0.10)
        assert r["severity"] == ("low" if r["drift_detected"] else "none")
        assert r["severity"] in ("none", "low")


def test_generator_drift_is_deterministic():
    """The per-feature drift (mean/std/PSI/severity) is deterministic (no RNG) — two runs give
    identical drift rows. The random uuid 'id' and the wall-clock window timestamps are
    inherently per-run and are excluded; the DRIFT statistic is what must be reproducible."""
    dist = {"prior_rx_count": {"mean": 0.3795, "std": 0.2731}}
    a = _gen_with_dist(dist)
    b = _gen_with_dist(dist)
    volatile = {"id", "baseline_start", "baseline_end", "current_start", "current_end"}
    strip = lambda rows: [{k: v for k, v in r.items() if k not in volatile} for r in rows]  # noqa: E731
    assert strip(a) == strip(b)


def test_generator_uses_the_shared_drift_map():
    """The mapped features all carry their _DRIFT_MAP params in raw_results — proving the
    generator and migration 053 share the SAME deterministic per-feature drift contract."""
    dist = {f: {"mean": 0.5, "std": 0.2} for f in _DRIFT_MAP}
    rows = _gen_with_dist(dist)
    for r in rows:
        shift, mult = _DRIFT_MAP[r["feature_name"]]
        assert r["raw_results"]["mean_shift_std"] == shift
        assert r["raw_results"]["std_mult"] == mult
        assert r["current_mean"] == round(r["baseline_mean"] + shift * r["baseline_std"], 4)
        assert r["current_std"] == round(r["baseline_std"] * mult, 4)
