"""``src.data.per_hcp_cohort_collapse``: the per-(hcp, brand) exposure the adoption DGP's
channel term is computed from (lane T1) -- and, in lane T2, the twin loader's collapse.

Pinned: the SUM / MEAN / FIRST rules per column, the within-brand median exposure bit
(``value > median``, the estimator's own contrast), the shift formula
``sum_k beta_k * (tbin_k - 0.5)``, non-joined HCPs get shift 0, and the module imports without
the twin or sklearn (it is what the light writers import).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data import per_hcp_cohort_collapse as collapse
from src.data.per_hcp_cohort_columns import (
    ADOPTION_CHANNEL_LOGIT_BETA,
    INTERVENTION_TREATMENT_MAP,
)

_REPO = Path(__file__).resolve().parents[3]
_CHANNELS = sorted(set(INTERVENTION_TREATMENT_MAP.values()))


def _rollups() -> pd.DataFrame:
    """Three rows: HCP a has two metric_dates for brand X, HCP b one row; region constant per pair."""
    return pd.DataFrame(
        {
            "hcp_id": ["a", "a", "b"],
            "brand": ["X", "X", "X"],
            "metric_date": pd.to_datetime(["2026-05-01", "2026-09-21", "2026-06-15"]),
            "region": ["northeast", "northeast", "west"],
            "specialty": ["oncology", "oncology", "hematology"],
            "market_share": [0.2, 0.4, 0.5],
            "triggers_total_count": [3, 5, 7],
            "call_frequency": [1.0, 2.0, 10.0],
            "email_campaign_count": [4.0, 6.0, 1.0],
            "speaker_program_count": [0.0, 1.0, 2.0],
            "sample_volume": [10.0, 20.0, 5.0],
            "engagement_score": [2.0, 4.0, 9.0],
            "peer_influence_score": [1.0, 3.0, 8.0],
            "rep_training_score": [5.0, 7.0, 1.0],
            "patient_support_enrollment": [0.1, 0.3, 0.9],
        }
    )


def test_module_imports_without_the_twin_or_sklearn():
    code = (
        "import sys; import src.data.per_hcp_cohort_collapse; "
        "print(sorted(m for m in sys.modules if m in ('src.digital_twin','sklearn','dowhy','shap','econml')))"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True, cwd=_REPO
    ).stdout.strip()
    assert out == "[]"


def test_collapse_rules_cover_every_channel_and_confounder_once():
    assert set(collapse.COLLAPSE_SUM) == {
        "call_frequency",
        "email_campaign_count",
        "speaker_program_count",
        "sample_volume",
        "triggers_total_count",
    }
    assert set(collapse.COLLAPSE_MEAN) == {
        "engagement_score",
        "peer_influence_score",
        "rep_training_score",
        "patient_support_enrollment",
        "market_share",
    }
    assert set(collapse.COLLAPSE_FIRST) == {"region", "specialty"}
    assert set(_CHANNELS) <= set(collapse.COLLAPSE_SUM) | set(collapse.COLLAPSE_MEAN)
    assert set(collapse.COLLAPSE_RULES) == (
        set(collapse.COLLAPSE_SUM) | set(collapse.COLLAPSE_MEAN) | set(collapse.COLLAPSE_FIRST)
    )
    assert collapse.CHANNEL_COLUMNS == tuple(_CHANNELS)


def test_collapse_per_hcp_brand_sums_counts_means_scores_takes_first_labels():
    out = collapse.collapse_per_hcp_brand(_rollups()).set_index(["hcp_id", "brand"])
    assert len(out) == 2
    a = out.loc[("a", "X")]
    assert a["call_frequency"] == 3.0  # SUM
    assert a["email_campaign_count"] == 10.0
    assert a["speaker_program_count"] == 1.0
    assert a["sample_volume"] == 30.0
    assert a["triggers_total_count"] == 8
    assert a["engagement_score"] == 3.0  # MEAN
    assert a["peer_influence_score"] == 2.0
    assert a["rep_training_score"] == 6.0
    assert a["patient_support_enrollment"] == pytest.approx(0.2)
    assert a["market_share"] == pytest.approx(0.3)
    assert a["region"] == "northeast"  # FIRST
    assert a["specialty"] == "oncology"
    assert pd.Timestamp(a["max_metric_date"]) == pd.Timestamp("2026-09-21")
    assert a["n_metric_rows"] == 2
    b = out.loc[("b", "X")]
    assert b["call_frequency"] == 10.0 and b["engagement_score"] == 9.0 and b["n_metric_rows"] == 1


def test_collapse_without_specialty_or_metric_date_columns_still_works():
    df = _rollups().drop(columns=["specialty", "metric_date"])
    out = collapse.collapse_per_hcp_brand(df)
    assert "specialty" not in out.columns and "max_metric_date" not in out.columns
    assert len(out) == 2


def test_channel_tbin_is_above_the_within_brand_median():
    coll = pd.DataFrame(
        {
            "hcp_id": ["a", "b", "c", "d", "e"],
            "brand": ["X", "X", "X", "Y", "Y"],
            **{c: [1.0, 2.0, 3.0, 10.0, 20.0] for c in _CHANNELS},
        }
    )
    coll.loc[coll["hcp_id"] == "b", "call_frequency"] = 3.0  # tie with c on the median
    tb = collapse.channel_tbin(coll)
    assert list(tb.columns[:2]) == ["hcp_id", "brand"]
    assert list(tb.columns[2:]) == [f"tbin_{c}" for c in _CHANNELS]
    x = tb.set_index("hcp_id")
    # Brand X medians: engagement 2.0 -> only c above; call_frequency median 3.0 -> nobody above
    assert x.loc[["a", "b", "c"], "tbin_engagement_score"].tolist() == [0.0, 0.0, 1.0]
    assert x.loc[["a", "b", "c"], "tbin_call_frequency"].tolist() == [0.0, 0.0, 0.0]
    # Brand Y is its own population: median 15 -> e above, d not.
    assert x.loc[["d", "e"], "tbin_engagement_score"].tolist() == [0.0, 1.0]


def test_adoption_channel_shift_is_sum_of_centred_bits_times_beta():
    channels = _CHANNELS
    tb = pd.DataFrame({"hcp_id": ["a", "b"], "brand": ["X", "X"]})
    for c in channels:
        tb[f"tbin_{c}"] = [1.0, 0.0]
    beta = collapse.beta_by_column()
    shift = collapse.adoption_channel_shift(tb)
    assert shift.tolist() == pytest.approx([0.5 * sum(beta.values()), -0.5 * sum(beta.values())])
    # Only the null channel exposed: shift is exactly zero either way.
    tb2 = tb.copy()
    for c in channels:
        tb2[f"tbin_{c}"] = 0.0
    tb2["tbin_rep_training_score"] = [1.0, 0.0]
    assert collapse.adoption_channel_shift(tb2).tolist() == pytest.approx(
        [-0.5 * sum(v for k, v in beta.items() if k != "rep_training_score")] * 2
    )


def test_beta_by_column_maps_interventions_to_planted_columns():
    beta = collapse.beta_by_column()
    assert set(beta) == set(_CHANNELS)
    for intervention, col in INTERVENTION_TREATMENT_MAP.items():
        assert beta[col] == ADOPTION_CHANNEL_LOGIT_BETA[intervention]


def test_align_shift_gives_non_joined_hcps_zero():
    shift = pd.Series([0.4, -0.2], index=pd.Index(["b", "d"], name="hcp_id"))
    aligned = collapse.align_channel_shift(["a", "b", "c", "d"], shift)
    np.testing.assert_array_equal(aligned, np.array([0.0, 0.4, 0.0, -0.2]))


def test_dgp_true_and_stratified_rd_helpers():
    rng = np.random.default_rng(3)
    n = 4000
    tb = pd.DataFrame({"hcp_id": [str(i) for i in range(n)], "brand": "X"})
    for c in _CHANNELS:
        tb[f"tbin_{c}"] = (rng.random(n) < 0.5).astype(float)
    shift = collapse.adoption_channel_shift(tb).to_numpy()
    logit = -0.9 + rng.normal(0, 1.2, n) + shift
    true_rd = collapse.dgp_true_channel_rd(logit, tb)
    assert set(true_rd) == set(_CHANNELS)
    assert true_rd["rep_training_score"] == 0.0
    assert true_rd["engagement_score"] > true_rd["sample_volume"] > 0
    adopted = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    strat = collapse.stratified_channel_rd(adopted, tb)
    assert set(strat) == set(_CHANNELS)
    assert abs(strat["engagement_score"] - true_rd["engagement_score"]) < 0.05
