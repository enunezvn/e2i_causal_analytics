"""#604: the synthetic manifest must declare the legacy ``ml_patients()`` fixture
predictors as pre-index declared-safe contracts.

The Layer-3 FDR confident-set firing driver (#538) over-drops the legitimately
outcome-correlated columns the legacy ``default``/``adverse``/``clean`` fixtures
emit (``days_on_therapy``, ``hcp_visits``, ``prior_treatments`` — sample_data.py
ml_patients risk model, lines 649-658). The #604 fix re-enables FDR for those
fixtures and protects the legit columns via the Layer-1 declared-safe carve-out,
which requires each column to resolve to a ``knowable_at <= index`` contract under
``data_source="synthetic"``. These tests pin that registration.
"""

from __future__ import annotations

import pytest

from src.data.manifests.synthetic_feature_manifest import synthetic_contract_for

# The exact column names ml_patients() emits for the over-dropping legacy
# fixtures (verified at src/repositories/sample_data.py:689-691). All three are
# pre-index patient features that carry DESIGNED signal — legit, not leakage.
_LEGACY_FIXTURE_PREDICTORS = ("days_on_therapy", "hcp_visits", "prior_treatments")


@pytest.mark.parametrize("name", _LEGACY_FIXTURE_PREDICTORS)
def test_legacy_fixture_predictor_declared_pre_index(name: str) -> None:
    """Each over-dropped legacy fixture predictor resolves to a declared-safe
    (knowable_at <= index) synthetic contract, so the FDR carve-out can protect
    it instead of auto-dropping it."""
    contract = synthetic_contract_for(name)
    assert contract is not None, (
        f"{name!r} must be registered in the synthetic manifest so "
        "lookup_feature_contract returns a contract and layer_1_declared_safe=True"
    )
    assert contract.knowable_at.is_pre_or_at_index() is True, (
        f"{name!r} must be declared knowable_at<=index (pre-index legit predictor)"
    )


def test_borderline_genuine_feature_still_registered() -> None:
    """No regression: the pre-existing v5 Gate C2 contract must remain."""
    assert synthetic_contract_for("borderline_genuine_feature") is not None


def test_unregistered_feature_returns_none() -> None:
    """A name not in the manifest (e.g. a genuine leak / outcome proxy) must NOT
    resolve to a contract — it stays subject to FDR auto-drop."""
    assert synthetic_contract_for("journey_status") is None
    assert synthetic_contract_for("definitely_not_a_feature") is None
