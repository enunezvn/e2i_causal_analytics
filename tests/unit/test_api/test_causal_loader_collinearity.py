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
import pandas as pd
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


# --- The criterion itself (codex r3 HIGH): translation- and unit-invariant, exact only ---------


def _prune(df, cols):
    from src.api.routes.causal.loaders import _prune_exactly_collinear

    return _prune_exactly_collinear(df, cols)


def test_prune_keeps_a_large_offset_column():
    # x = 1e10 + arange: a genuinely varying column whose raw norm dwarfs its
    # variation -- a residual-vs-raw-norm test dropped it (codex r3 HIGH).
    n = 100
    rng = np.random.default_rng(1)
    df = pd.DataFrame({"a": 1e10 + np.arange(n, dtype=float), "b": rng.normal(size=n)})
    assert _prune(df, ["a", "b"]) == (["a", "b"], [])


def test_prune_decision_is_invariant_to_column_units():
    n = 120
    rng = np.random.default_rng(2)
    base = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n)})
    base["dup"] = base["a"]
    kept, dropped = _prune(base, ["a", "b", "dup"])
    scaled = base.copy()
    scaled["a"] = scaled["a"] * 1e6
    scaled["dup"] = scaled["dup"] * 1e-9
    assert _prune(scaled, ["a", "b", "dup"]) == (kept, dropped) == (["a", "b"], ["dup"])


def test_prune_leaves_a_near_collinear_column_to_the_estimator_guard():
    n = 200
    rng = np.random.default_rng(3)
    df = pd.DataFrame({"a": rng.normal(size=n)})
    df["almost"] = df["a"] + 1e-4 * rng.normal(size=n)  # informative, not exact
    assert _prune(df, ["a", "almost"]) == (["a", "almost"], [])


def test_prune_runs_at_n_equal_to_k_plus_one_and_skips_below():
    rng = np.random.default_rng(4)
    df = pd.DataFrame({"a": rng.normal(size=5), "b": rng.normal(size=5), "c": rng.normal(size=5)})
    df["dup"] = df["a"]
    # k=4 columns offered: n = k+1 = 5 rows can rank intercept + 4 columns
    assert _prune(df.iloc[:5], ["a", "b", "c", "dup"]) == (["a", "b", "c"], ["dup"])
    # n=4 < k+1: skipped, nothing dropped
    assert _prune(df.iloc[:4], ["a", "b", "c", "dup"]) == (["a", "b", "c", "dup"], [])


def test_prune_drops_a_constant_column_as_collinear_with_the_intercept():
    df = pd.DataFrame({"a": np.arange(10, dtype=float), "k": np.full(10, 7.0)})
    assert _prune(df, ["a", "k"]) == (["a"], ["k"])


# codex r4 HIGH: the constant test and the fixed 1e-8 tolerance were not translation-
# or scale-invariant. Constant = exact represented equality; the offset is removed
# by subtracting a reference value BEFORE centering; the column is scaled by a power
# of two (exact); the tolerance is the same machine-precision convention econml's
# ``lstsq(rcond=None)`` uses to declare the final stage underdetermined.


def test_prune_keeps_a_varying_column_with_a_1e14_offset():
    df = pd.DataFrame({"x": 1e14 + np.arange(100, dtype=float)})
    assert _prune(df, ["x"]) == (["x"], [])


def test_prune_keeps_an_independent_component_below_1e8_relative():
    # 5e-9 relative noise is far above machine precision: econml's own rank check
    # would NOT call this design underdetermined, so the prune must not drop it.
    rng = np.random.default_rng(5)
    a = rng.normal(size=100)
    df = pd.DataFrame({"a": a, "almost": a + 5e-9 * rng.normal(size=100)})
    assert _prune(df, ["a", "almost"]) == (["a", "almost"], [])


def test_prune_is_safe_under_extreme_unit_rescaling():
    rng = np.random.default_rng(6)
    df = pd.DataFrame({"a": rng.normal(size=100) * 1e-200, "b": rng.normal(size=100) * 1e-200})
    assert _prune(df, ["a", "b"]) == (["a", "b"], [])


def test_prune_drops_only_the_exact_duplicate_when_both_carry_a_large_offset():
    a = 1e14 + np.arange(100, dtype=float)
    df = pd.DataFrame({"a": a, "b": 2.0 * a + 3.0})
    assert _prune(df, ["a", "b"]) == (["a"], ["b"])


def test_prune_is_finite_safe_at_float_max():
    # codex r5 MED: [-1e308, 1e308] overflowed the reference subtraction to inf and
    # floor(log2(inf)) raised. Scale by a power of two BEFORE subtracting.
    df = pd.DataFrame({"a": [-1e308, 1e308, 5e307, 0.0, 1.0], "b": [1.0, 2.0, 4.0, 8.0, 16.0]})
    assert _prune(df, ["a", "b"]) == (["a", "b"], [])
    df["half_a"] = df["a"] * 0.5  # exact duplicate at the edge of the range
    assert _prune(df, ["a", "b", "half_a"]) == (["a", "b"], ["half_a"])


@pytest.mark.asyncio
async def test_loader_evaluates_numeric_columns_before_one_hot_dummies(monkeypatch):
    """The resolved order is numeric registry columns first, then the one-hot
    dummies (in their categoricals' registry order): a dummy that exactly equals
    an earlier NUMERIC column is the one dropped, whatever the two columns'
    registry positions. Documented, not reordered -- reordering the design would
    change every dataset's feature-index-dependent forest fits."""
    rows = _rows()
    # payer_category has two levels; make cci_hiv an exact copy of the 'medicare' dummy
    for i, r in enumerate(rows):
        r["payer_category"] = "medicare" if i % 3 == 0 else "commercial"
        r["cci_hiv"] = 1 if i % 3 == 0 else 0
        r["elx_aids_hiv"] = int(i % 5 == 0)
    monkeypatch.setattr(_CLIENT_FACTORY, AsyncMock(return_value=_FakeClient(rows)))
    _frame, cols = await _load_agent_estimation_frame(
        dataset=DATASET,
        treatment_var=TREATMENT,
        outcome_var=OUTCOME,
        covariates=["age_at_index", "payer_category", "cci_hiv", "elx_chf", "elx_aids_hiv"],
        limit=100,
    )
    assert "cci_hiv" in cols
    assert not any(c.startswith("payer_category=") for c in cols)
