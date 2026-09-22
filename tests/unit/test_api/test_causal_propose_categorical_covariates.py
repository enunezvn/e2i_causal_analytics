"""``GET /causal/propose-questions`` must screen with the loader's EXPANDED
columns (codex r1 on Lane C, 2026-09-22).

The estimation loader one-hot encodes every categorical covariate it is asked
for and returns the expanded names (``geographic_region`` becomes
``geographic_region=south`` ...). ``propose_causal_questions`` discarded that
list and residualised on the RAW covariate names, so the FWL screen indexed a
column the frame no longer had -> ``KeyError`` -> HTTP 500. Reachable on main
for the DEFAULT dataset (``patient_journeys`` offers ``geographic_region``,
which ``_CAUSAL_CATEGORICAL_COLUMNS`` one-hots) and, with Lane C, for
``csu_escalation_causal`` (seven categoricals). The discovery leaderboard's
pre-rank already used the expanded list (discovery.py ``_prerank_signal``);
this pins the propose route to the same contract.

Fake-client seam as in test_causal_csu_escalation_registry.py.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from src.api.routes.causal import catalog
from src.api.routes.causal import datasets as datasets_mod
from src.api.routes.causal.datasets import _CAUSAL_DATASET_SPECS
from src.ml.synthetic.generators.csu_escalation_causal import generate_csu_escalation_cohort

pytestmark = pytest.mark.unit

_CLIENT_FACTORY = "src.memory.services.factories.get_async_supabase_client"


class _FakeQuery:
    def __init__(self, rows):
        self._rows = rows

    def select(self, *_a, **_k):
        return self

    def eq(self, col, value, *_a, **_k):
        if col == "is_synthetic":
            self._rows = [r for r in self._rows if bool(r.get("is_synthetic")) is bool(value)]
        else:
            self._rows = [r for r in self._rows if r.get(col) == value]
        return self

    def limit(self, n, *_a, **_k):
        self._rows = self._rows[:n]
        return self

    async def execute(self):
        return type("R", (), {"data": self._rows})()


class _FakeClient:
    def __init__(self, rows):
        self._rows = rows

    def table(self, *_a, **_k):
        return _FakeQuery(list(self._rows))


def _patient_journey_rows(n: int = 240):
    """Synthetic-gold shaped rows: every universal covariate present, the
    categorical region populated, two treatment columns (``treatment_initiated``
    is treatment AND outcome in the spec), random so every FWL residual has
    variance (a deterministic column makes the screen return None)."""
    import numpy as np

    rng = np.random.default_rng(1991)
    regions = ("midwest", "south", "northeast", "west")
    rows = []
    for i in range(n):
        rows.append(
            {
                "treatment_arm": float(rng.integers(0, 2)),
                "persistent_180d": float(rng.integers(0, 2)),
                "discontinued_180d": float(rng.integers(0, 2)),
                "treatment_initiated": float(rng.integers(0, 2)),
                "adherent_180d": float(rng.integers(0, 2)),
                "low_gap_180d": float(rng.integers(0, 2)),
                "disease_severity": float(rng.random()),
                "engagement_score": float(rng.random()),
                "age_at_diagnosis": float(rng.integers(30, 70)),
                "academic_hcp": float(rng.integers(0, 2)),
                "geographic_region": regions[i % 4],
                "is_synthetic": True,
            }
        )
    return rows


def _loadable_pairs(dataset: str, rows) -> set:
    """The route's own enumeration (every spec treatment x outcome, t != o)
    restricted to the pairs whose two columns the rows carry -- the loader
    omits the others (HTTPException -> None), never fabricates them."""
    spec = _CAUSAL_DATASET_SPECS[dataset]
    present = set(rows[0])
    return {
        (t, o)
        for t in spec["treatment"]
        for o in spec["outcome"]
        if t != o and t in present and o in present
    }


async def _propose(monkeypatch, dataset: str, rows):
    monkeypatch.setenv("E2I_INCLUDE_SYNTHETIC", "true")
    monkeypatch.setattr(_CLIENT_FACTORY, AsyncMock(return_value=_FakeClient(rows)))
    return await catalog.propose_causal_questions(dataset=dataset, user={"role": "analyst"})


@pytest.mark.asyncio
async def test_default_dataset_with_its_categorical_covariate_is_screened(monkeypatch):
    rows = _patient_journey_rows()
    resp = await _propose(monkeypatch, "patient_journeys", rows)
    assert resp.dataset == "patient_journeys"
    # Every loadable (treatment, outcome) pair is screened -- none is silently
    # omitted by a KeyError on the raw categorical name (the loader one-hots
    # geographic_region and drops the all-NULL insurance_access_score).
    expected = _loadable_pairs("patient_journeys", rows)
    assert {(c.treatment, c.outcome) for c in resp.candidates} == expected
    assert len(expected) == 9  # treatment_arm x 5 + treatment_initiated x 4
    assert all(c.n_rows == 240 for c in resp.candidates)
    assert all(0.0 <= c.association_strength <= 1.0 for c in resp.candidates)


@pytest.mark.asyncio
async def test_csu_escalation_dataset_with_seven_categoricals_is_screened(monkeypatch):
    # The synthetic backing is read ONLY under the planted-truth seam (the
    # deployment-wide E2I_INCLUDE_SYNTHETIC the helper sets does not unlock it).
    monkeypatch.setattr(datasets_mod, "PLANTED_TRUTH_RUN", True)
    frame, _ = generate_csu_escalation_cohort(n=300, seed=5)
    rows = frame.astype(object).where(frame.notna(), None).to_dict(orient="records")
    resp = await _propose(monkeypatch, "csu_escalation_causal", rows)
    assert resp.dataset == "csu_escalation_causal"
    # 1 treatment x 4 outcomes, every pair screened on the one-hot frame.
    expected = _loadable_pairs("csu_escalation_causal", rows)
    assert {(c.treatment, c.outcome) for c in resp.candidates} == expected
    assert len(expected) == 4
    assert {c.treatment for c in resp.candidates} == {"treatment_remibrutinib"}
    assert all(c.n_rows == 300 for c in resp.candidates)
