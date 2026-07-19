"""The adherence latent carries TWO arms (treatment_arm + copay_support). Each
arm's counterfactual RD must be computed against an effective baseline that
INCLUDES the other arm's contribution — otherwise each arm's ground truth
absorbs the other's effect and the recovery gate validates the wrong number."""

import numpy as np
import pytest

from src.ml.synthetic.dgp.adherence_outcomes import generate_adherence_outcomes


def _inputs(n=4000, seed=21):
    rng = np.random.default_rng(seed)
    severity = np.clip(rng.normal(5.0, 2.0, n), 0, 10)
    return {
        "treatment_arm": (rng.random(n) < 0.4).astype(int),
        "disease_severity": severity,
        "academic_hcp": (rng.random(n) < 0.3).astype(int),
        "segment": np.where(
            severity > 7, "high_severity", np.where(severity > 4, "medium_severity", "low_severity")
        ),
        "cate_map": {"high_severity": 0.70, "medium_severity": 0.30, "low_severity": 0.10},
    }


@pytest.mark.unit
def test_copay_absent_is_identical_to_phase0():
    """Passing no second arm must reproduce the Phase 0 result EXACTLY, so the
    shipped behaviour is unchanged when the arm is not wired."""
    a = generate_adherence_outcomes(rng=np.random.default_rng(5), **_inputs())
    b = generate_adherence_outcomes(
        rng=np.random.default_rng(5), copay_support=None, copay_cate=None, **_inputs()
    )
    np.testing.assert_array_equal(a["adherent_180d"], b["adherent_180d"])
    np.testing.assert_array_equal(a["low_gap_180d"], b["low_gap_180d"])
    assert a["adherent_rd_by_segment"] == b["adherent_rd_by_segment"]


@pytest.mark.unit
def test_copay_ground_truth_is_returned_and_ordered():
    ins = _inputs()
    n = len(ins["disease_severity"])
    copay = (np.random.default_rng(9).random(n) < 0.35).astype(int)
    out = generate_adherence_outcomes(
        rng=np.random.default_rng(5),
        copay_support=copay,
        copay_cate={"high_severity": 0.44, "medium_severity": 0.18, "low_severity": 0.06},
        **ins,
    )
    rd = out["copay_low_gap_rd_by_segment"]
    assert rd["high_severity"] > rd["medium_severity"] > rd["low_severity"]
    assert 0.02 < float(np.mean(list(rd.values()))) < 0.25


@pytest.mark.unit
def test_proxy_consistency_survives_the_second_arm():
    """The stored continuous proxies must still never contradict the stored
    binaries (HIGH-3 zero-contradiction), with copay in the latent."""
    ins = _inputs()
    n = len(ins["disease_severity"])
    copay = (np.random.default_rng(9).random(n) < 0.35).astype(int)
    out = generate_adherence_outcomes(
        rng=np.random.default_rng(5),
        copay_support=copay,
        copay_cate={"high_severity": 0.44, "medium_severity": 0.18, "low_severity": 0.06},
        **ins,
    )
    assert np.all((out["adherence_rate"] >= 0.80) == (out["adherent_180d"] == 1))
    assert np.all((out["gap_days"] <= 30) == (out["low_gap_180d"] == 1))


# --- Phase 2: psp_enrolled is a THIRD arm on the same adherence latent ---------------
_PSP_CATE = {"high_severity": 0.38, "medium_severity": 0.13, "low_severity": 0.05}
_COPAY_CATE = {"high_severity": 0.44, "medium_severity": 0.18, "low_severity": 0.06}


@pytest.mark.unit
def test_psp_absent_leaves_the_copay_path_unchanged():
    """Passing no psp arm must reproduce the copay-only result EXACTLY, so wiring
    the psp default branch does not perturb the shipped Phase 1 behaviour."""
    ins = _inputs()
    n = len(ins["disease_severity"])
    copay = (np.random.default_rng(9).random(n) < 0.35).astype(int)
    a = generate_adherence_outcomes(
        rng=np.random.default_rng(5), copay_support=copay, copay_cate=_COPAY_CATE, **ins
    )
    b = generate_adherence_outcomes(
        rng=np.random.default_rng(5),
        copay_support=copay,
        copay_cate=_COPAY_CATE,
        psp_enrolled=None,
        psp_cate=None,
        **ins,
    )
    np.testing.assert_array_equal(a["adherent_180d"], b["adherent_180d"])
    assert a["copay_adherent_rd_by_segment"] == b["copay_adherent_rd_by_segment"]


@pytest.mark.unit
def test_psp_ground_truth_is_returned_and_ordered():
    ins = _inputs()
    n = len(ins["disease_severity"])
    psp = (np.random.default_rng(11).random(n) < 0.38).astype(int)
    out = generate_adherence_outcomes(
        rng=np.random.default_rng(5), psp_enrolled=psp, psp_cate=_PSP_CATE, **ins
    )
    rd = out["psp_adherent_rd_by_segment"]
    assert rd["high_severity"] > rd["medium_severity"] > rd["low_severity"], rd
    assert 0.02 < float(np.mean(list(rd.values()))) < 0.25, rd


@pytest.mark.unit
def test_three_arms_each_carry_their_own_ordered_ground_truth():
    """With treatment_arm + copay + psp all in the latent, EACH commercial arm's
    per-segment RD is its OWN ordered effect (the effective-baseline folds the other
    two arms in), not a blend — so the recovery gate validates the right number."""
    ins = _inputs()
    n = len(ins["disease_severity"])
    copay = (np.random.default_rng(9).random(n) < 0.35).astype(int)
    psp = (np.random.default_rng(11).random(n) < 0.38).astype(int)
    out = generate_adherence_outcomes(
        rng=np.random.default_rng(5),
        copay_support=copay,
        copay_cate=_COPAY_CATE,
        psp_enrolled=psp,
        psp_cate=_PSP_CATE,
        **ins,
    )
    for key in ("copay_adherent_rd_by_segment", "psp_adherent_rd_by_segment"):
        rd = out[key]
        assert rd["high_severity"] > rd["medium_severity"] > rd["low_severity"], (key, rd)
        assert all(v > 0 for v in rd.values()), (key, rd)
    # Proxy consistency still holds with three arms in the latent.
    assert np.all((out["adherence_rate"] >= 0.80) == (out["adherent_180d"] == 1))
