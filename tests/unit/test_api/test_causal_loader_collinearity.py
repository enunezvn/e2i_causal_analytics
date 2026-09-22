"""Lane A (codex r2 HIGH): the agent loader prunes EXACTLY collinear covariates so
the estimators never see a rank-deficient design.

Measured 2026-09-22 on the real Optum biologic-persistence frame (n=15,209,
77 resolved covariates; ``docs/demos/results/2026-09-22_optum_biologic_persistence_cert/
collinearity_probe.json``): the design has rank 61 -- 16 columns are exact
linear combinations of earlier ones (Elixhauser flags duplicating Charlson flags,
a risk band implied by its score, payer dummies implied by a coarser payer axis).
econml's final stage then warns "Co-variance matrix is underdetermined. Inference
will be invalid!" and the wrapper served that CI anyway. Dropping the 16 columns
leaves the ATE at 0.03353 (was 0.03353) and the SE at 0.00858 (was 0.00855) with
no warning: the redundant columns carry no information, so removing them changes
no estimand. The prune runs ONCE at load time, on the full frame, in registry
order (earlier columns win), so the estimation node and the refutation rebuild
(which takes ``common_causes`` from the same resolved list) see the SAME design.

Below ``n <= k + 1`` a rank test says nothing about the COLUMNS (any design is
rank-deficient there), so the prune is skipped and the estimator wrappers' own
invalid-inference refusal is the guard.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import numpy as np
import pytest

from src.api.routes.causal.datasets import _CAUSAL_DATASET_SPECS
from src.api.routes.causal.loaders import _load_agent_estimation_frame

DATASET = "optum_biologic_persistence"
TREATMENT = "treatment_dupixent"
OUTCOME = "persistent_at_180d_g28"
_CLIENT_FACTORY = "src.memory.services.factories.get_async_supabase_client"


class _FakeQuery:
    def __init__(self, rows):
        self._rows = rows

    def select(self, *_a, **_k):
        return self

    def eq(self, *_a, **_k):
        return self

    def limit(self, *_a, **_k):
        return self

    async def execute(self):
        return type("R", (), {"data": self._rows})()


class _FakeClient:
    def __init__(self, rows):
        self._rows = rows

    def table(self, *_a, **_k):
        return _FakeQuery(self._rows)


def _rows(n: int = 60, duplicate: bool = True):
    rng = np.random.default_rng(5)
    rows = []
    for i in range(n):
        cci_hiv = int(rng.random() < 0.3)
        elx_chf = int(rng.random() < 0.4)
        rows.append(
            {
                TREATMENT: int(i % 2),
                OUTCOME: int(rng.random() < 0.6),
                "age_at_index": int(rng.integers(20, 80)),
                "cci_hiv": cci_hiv,
                "elx_chf": elx_chf,
                # exact duplicate of an EARLIER registry column (or an independent flag)
                "elx_aids_hiv": cci_hiv if duplicate else int(rng.random() < 0.3),
            }
        )
    return rows


@pytest.fixture
def covariates():
    order = _CAUSAL_DATASET_SPECS[DATASET]["covariate"]
    cols = ["age_at_index", "cci_hiv", "elx_chf", "elx_aids_hiv"]
    assert all(c in order for c in cols)
    # registry order decides which of an exactly collinear pair survives
    assert order.index("cci_hiv") < order.index("elx_aids_hiv")
    return sorted(cols, key=order.index)


@pytest.mark.asyncio
async def test_loader_drops_an_exactly_collinear_later_column(monkeypatch, covariates, caplog):
    monkeypatch.setattr(_CLIENT_FACTORY, AsyncMock(return_value=_FakeClient(_rows())))
    with caplog.at_level("WARNING", logger="src.api.routes.causal.loaders"):
        frame, cols = await _load_agent_estimation_frame(
            dataset=DATASET,
            treatment_var=TREATMENT,
            outcome_var=OUTCOME,
            covariates=covariates,
            limit=100,
        )
    assert "elx_aids_hiv" not in cols
    assert cols == [TREATMENT, OUTCOME, "age_at_index", "cci_hiv", "elx_chf"]
    design = np.column_stack([np.ones(len(frame)), frame[cols[2:]].to_numpy(dtype=float)])
    assert np.linalg.matrix_rank(design) == design.shape[1]
    assert any("elx_aids_hiv" in r.message and "collinear" in r.message for r in caplog.records)


@pytest.mark.asyncio
async def test_loader_keeps_a_full_rank_design_unchanged(monkeypatch, covariates):
    monkeypatch.setattr(
        _CLIENT_FACTORY, AsyncMock(return_value=_FakeClient(_rows(duplicate=False)))
    )
    frame, cols = await _load_agent_estimation_frame(
        dataset=DATASET,
        treatment_var=TREATMENT,
        outcome_var=OUTCOME,
        covariates=covariates,
        limit=100,
    )
    assert cols == [TREATMENT, OUTCOME, "age_at_index", "cci_hiv", "elx_chf", "elx_aids_hiv"]
    assert "elx_aids_hiv" in frame.columns


@pytest.mark.asyncio
async def test_loader_does_not_prune_when_rows_cannot_rank_the_columns(monkeypatch, covariates):
    # n=3 rows, 4 covariates: every design is rank-deficient here, which says nothing
    # about the columns -- the wrappers' invalid-inference refusal owns this case.
    monkeypatch.setattr(_CLIENT_FACTORY, AsyncMock(return_value=_FakeClient(_rows(n=3))))
    _frame, cols = await _load_agent_estimation_frame(
        dataset=DATASET,
        treatment_var=TREATMENT,
        outcome_var=OUTCOME,
        covariates=covariates,
        limit=100,
    )
    assert cols == [TREATMENT, OUTCOME, "age_at_index", "cci_hiv", "elx_chf", "elx_aids_hiv"]
