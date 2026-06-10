"""P2 — fidelity + causal generation-time gates (Shard 10).

These are GENERATION-TIME gates run against the in-memory patient frame BEFORE
the 6 parquets are written: a KS distributional check against header-derived
``csu_data.xlsx`` marginals, a correlation-recovery check (the realized latent
correlations match the DGP design), and a causal-role gate that recovers the
embedded TRUE_ATE of the engagement -> initiation effect.
"""

import numpy as np
import pytest

from src.ml.synthetic.claims.config import ClaimsDGPConfig
from src.ml.synthetic.claims.patient_state import generate_patients
from src.ml.synthetic.claims.validation import (
    CLAIMS_TRUE_ATE,
    build_engagement_initiation_frame,
    causal_role_gate,
    correlation_recovery_gate,
    ks_fidelity_gate,
)


def _pats(n=2000, seed=42):
    cfg = ClaimsDGPConfig(n_patients=n, seed=seed)
    return generate_patients(np.random.default_rng(seed), cfg), cfg


def test_ks_fidelity_gate_passes_on_in_distribution_panel():
    pats, _ = _pats()
    result = ks_fidelity_gate(pats)
    # The gate is HEADER-DERIVED (csu_rwd._load_from_excel raises
    # NotImplementedError), so it compares against marginals reconstructed from
    # csu_data.xlsx headers, not the full distribution — documented in result.
    assert result["header_derived"] is True
    assert result["passed"] is True, result
    # Age marginal must be within the plausible adult-CSU band.
    assert 18 <= result["marginals"]["age_mean"] <= 80


def test_correlation_recovery_gate_matches_dgp_design():
    pats, _ = _pats()
    result = correlation_recovery_gate(pats)
    # severity should POSITIVELY correlate with response_propensity (init logit
    # is +severity) and NEGATIVELY with adherence_propensity (logit is -severity).
    assert result["corr_severity_response"] > 0.10
    assert result["corr_severity_adherence"] < -0.10
    # tx_burden is an INDEPENDENT axis -> near-zero correlation with severity.
    assert abs(result["corr_severity_tx"]) < 0.15
    assert result["passed"] is True


def test_causal_role_gate_recovers_true_ate():
    pats, _ = _pats()
    # Engagement (treatment) is a function of tx_burden; the embedded TRUE_ATE on
    # initiation is recovered within tolerance via the reused CausalValidator.
    result = causal_role_gate(pats)
    assert result["true_ate"] == pytest.approx(CLAIMS_TRUE_ATE, abs=1e-9)
    assert result["estimated_ate"] is not None
    assert result["ate_error"] <= result["tolerance"], result
    assert result["passed"] is True


def test_causal_role_gate_fails_on_scrambled_treatment():
    pats, _ = _pats()
    frame = build_engagement_initiation_frame(pats)
    # Destroy the engagement->initiation link by permuting the treatment column.
    rng = np.random.default_rng(0)
    frame = frame.copy()
    frame["engagement_score"] = rng.permutation(frame["engagement_score"].to_numpy())
    result = causal_role_gate(pats, frame=frame)
    # A scrambled treatment recovers ~0 ATE, far outside the TRUE_ATE tolerance.
    assert result["estimated_ate"] is not None
    assert abs(result["estimated_ate"]) < CLAIMS_TRUE_ATE / 2
    assert result["passed"] is False
