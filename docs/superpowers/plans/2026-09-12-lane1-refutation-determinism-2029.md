# Lane 1 — Refutation determinism (#2029) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the DoWhy placebo and random-common-cause refits deterministic per estimate id, persist every test's seed in its row, and stop a single out-of-range `delta_percent` from dropping a whole suite's persistence.

**Architecture:** One 31-bit seed already derives from the estimate id (`_resample_seed_for`); the same seed is threaded as `random_state=` into the two DoWhy refuters and written into every perturbation test's `details`. The column widens to NUMERIC(12,4) by migration 139 and one clamp moves to the repository's row builder so every test type is bounded at the write boundary.

**Tech Stack:** Python 3.12, DoWhy 0.14 (`random_state` kwarg), pandas/numpy, Postgres via `docker exec supabase-db psql`, pytest (`-n 0`, `heavy_ml` marker for real-DoWhy fits).

Spec: `docs/superpowers/specs/2026-09-12-debts-3-4-wave-design.md` §4. Lane protocol: worktree `.worktrees/lane-2029`, branch `claude/2029-refutation-determinism`, every pytest `-n 0`, ruff `--no-cache`, codex read-only round per task, ONE push at the end.

Worktree setup (once):
```bash
cd /home/enunez/Projects/e2i_causal_analytics
git fetch origin && git worktree add -b claude/2029-refutation-determinism .worktrees/lane-2029 origin/main
cd .worktrees/lane-2029 && git branch --show-current
```
Every command below runs from `.worktrees/lane-2029` with `.venv/bin/python` / `.venv/bin/pytest` of the main checkout (`/home/enunez/Projects/e2i_causal_analytics/.venv/bin/pytest`). Assert `python -c "import src; print(src.__file__)"` prints the worktree path before each task.

---

## File map

| file | change |
|---|---|
| `src/causal_engine/refutation_runner.py` | thread `random_state` into `_run_placebo_test` / `_run_random_common_cause_test`; write `random_state` / `resample_seed` into details; retire `_DELTA_PERCENT_COLUMN_MAX` clamp in `_run_negative_control_test` |
| `src/agents/causal_impact/nodes/refutation.py` | calibration probe passes `random_state` |
| `src/repositories/causal_validation.py` | `DELTA_PERCENT_COLUMN_MAX = 99999999.9999`, `_clamp_delta_percent`, applied in `_test_to_row` and `save_single_test` |
| `database/migrations/139_causal_validations_delta_percent_numeric_12_4.sql` | new |
| `tests/unit/test_causal_engine/test_refutation_seed_identity_2029.py` | new (heavy_ml) |
| `tests/unit/test_causal_engine/test_refutation_runner_negative_control_2007.py` | clamp test moves out |
| `tests/unit/test_repositories/test_causal_validation_delta_clamp_2029.py` | new |
| `tests/unit/test_database/test_migration_139_delta_percent.py` | new |

---

### Task 1: Seed the two DoWhy refuters from the estimate id

**Files:**
- Modify: `src/causal_engine/refutation_runner.py` (`run_all_tests` dispatch ~1565–1608; `_run_placebo_test` ~1971–2000; `_run_random_common_cause_test` ~2068–2140)
- Test: `tests/unit/test_causal_engine/test_refutation_seed_identity_2029.py`

- [ ] **Step 1: Write the failing identity test (real DoWhy, tiny frame)**

```python
"""#2029: placebo_treatment and random_common_cause are seeded from the estimate id.

Two runs with the same estimate id give byte-identical refits; two estimate ids
differ (the positive control against a seed that ignores its input).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytestmark = [pytest.mark.heavy_ml, pytest.mark.xdist_group(name="dowhy_seed_2029")]


def _fitted():
    from dowhy import CausalModel

    rng = np.random.default_rng(2029)
    n = 400
    w = rng.normal(size=n)
    t = (rng.normal(size=n) + 0.8 * w > 0).astype(int)
    y = 0.3 * t + 0.5 * w + rng.normal(size=n)
    frame = pd.DataFrame({"t": t, "y": y, "w": w})
    model = CausalModel(data=frame, treatment="t", outcome="y", common_causes=["w"])
    estimand = model.identify_effect(proceed_when_unidentifiable=True)
    estimate = model.estimate_effect(estimand, method_name="backdoor.linear_regression")
    return model, estimand, estimate, float(estimate.value)


def _runner():
    from src.causal_engine.refutation_runner import RefutationRunner

    return RefutationRunner(
        config={
            "placebo_treatment": {"num_simulations": 2},
            "random_common_cause": {"num_simulations": 2},
        }
    )


def _one(test: str, estimate_id: str):
    model, estimand, estimate, ate = _fitted()
    runner = _runner()
    seed = runner._seed_for(estimate_id)
    if test == "placebo_treatment":
        r = runner._run_placebo_test(ate, model, estimand, estimate, True, random_state=seed)
    else:
        r = runner._run_random_common_cause_test(
            ate, (ate - 0.1, ate + 0.1), model, estimand, estimate, True, random_state=seed
        )
    return (r.refuted_effect, r.p_value, r.status.value, r.details.get("random_state"))


@pytest.mark.parametrize("test", ["placebo_treatment", "random_common_cause"])
def test_same_estimate_id_is_byte_identical(test):
    a = _one(test, "estimate-A")
    b = _one(test, "estimate-A")
    assert a == b
    assert a[3] is not None  # the seed is recorded


@pytest.mark.parametrize("test", ["placebo_treatment", "random_common_cause"])
def test_different_estimate_ids_differ(test):
    a = _one(test, "estimate-A")
    b = _one(test, "estimate-B")
    assert a[3] != b[3]
    assert a[0] != b[0]  # at 2 sims the refit means differ for different seeds


def test_no_estimate_id_means_no_seed():
    runner = _runner()
    assert runner._seed_for(None) is None
    assert runner._seed_for("") is None
```

- [ ] **Step 2: Run it to verify it fails**

Run: `.venv/bin/pytest tests/unit/test_causal_engine/test_refutation_seed_identity_2029.py -n 0 -q`
Expected: FAIL — `AttributeError: 'RefutationRunner' object has no attribute '_seed_for'` (and `TypeError: unexpected keyword argument 'random_state'` once `_seed_for` exists).

- [ ] **Step 3: Add `_seed_for` and the `random_state` parameters**

In `src/causal_engine/refutation_runner.py`, inside `class RefutationRunner` right after `__init__`:

```python
    @staticmethod
    def _seed_for(estimate_id: Optional[str]) -> Optional[int]:
        """The ONE seed a run uses for every random refit (#2029).

        Same derivation as the in-house resample loops (``_resample_seed_for``):
        a stable 31-bit integer from the estimate id, ``None`` when there is no
        id -- an unseeded run stays unseeded and says so in its details.
        """
        return _resample_seed_for(estimate_id)
```

Change `_run_placebo_test`'s signature and call:

```python
    def _run_placebo_test(
        self,
        original_effect: float,
        causal_model: Optional[Any],
        identified_estimand: Optional[Any],
        estimate: Optional[Any],
        use_dowhy: bool,
        *,
        random_state: Optional[int] = None,
    ) -> RefutationResult:
```
and in the `refute_estimate(...)` call add, after `num_simulations=...`:
```python
                    # #2029: DoWhy 0.14 converts an int ``random_state`` to one
                    # RandomState shared across the simulations, so the whole
                    # permutation sequence is reproducible from the estimate id.
                    # (``random_seed`` would only seed numpy's GLOBAL rng.)
                    random_state=random_state,
```
and in its `details` dict add `"random_state": random_state,`.

Change `_run_random_common_cause_test`: add `random_state: Optional[int] = None,` to the keyword-only block after `refit_n`; add to `_rcc_kwargs`:
```python
                if random_state is not None:
                    _rcc_kwargs["random_state"] = random_state
```
(unconditional is also fine — DoWhy accepts `None`; keep it conditional so a `None` never appears in kwargs logs). After `details.update(config_details)` add `details["random_state"] = random_state`.

In `run_all_tests`, right after `resample_seed = _resample_seed_for(estimate_id)`:
```python
        # #2029: the same seed feeds the two DoWhy refits.
        random_state = self._seed_for(estimate_id)
```
and add `random_state=random_state,` to the two `_run_test_with_tracing(...)` calls for `placebo_treatment` and `random_common_cause` (the wrapper forwards `**kwargs` to the test function).

- [ ] **Step 4: Run the test to verify it passes**

Run: `.venv/bin/pytest tests/unit/test_causal_engine/test_refutation_seed_identity_2029.py -n 0 -q`
Expected: 5 passed.

- [ ] **Step 5: Run the neighbours that pin the runner**

Run: `.venv/bin/pytest tests/unit/test_causal_engine/test_refutation_runner.py tests/unit/test_causal_engine/test_refutation_runner_real_evidence.py tests/unit/test_causal_engine/test_refutation_runner_rcc_scale_free_2005.py tests/unit/test_causal_engine/test_refutation_bands_enumeration.py -n 0 -q`
Expected: all pass (the bands enumeration is untouched by construction).

- [ ] **Step 6: Commit**

```bash
git add src/causal_engine/refutation_runner.py tests/unit/test_causal_engine/test_refutation_seed_identity_2029.py
git commit -m "fix(refutation): seed placebo and random_common_cause from the estimate id (#2029)"
```

---

### Task 2: Persist the resample seed for subset and bootstrap

**Files:**
- Modify: `src/causal_engine/refutation_runner.py` (`_run_data_subset_test` details ~2318; `_run_bootstrap_test` details ~2465)
- Test: `tests/unit/test_causal_engine/test_refutation_seed_identity_2029.py`

- [ ] **Step 1: Write the failing test**

Append to the test file:

```python
def test_subset_and_bootstrap_details_carry_resample_seed():
    from src.causal_engine.refutation_runner import RefutationRunner, RefutationTestType

    model, estimand, estimate, ate = _fitted()
    runner = RefutationRunner(
        config={"data_subset": {"num_subsets": 2}, "bootstrap": {"num_bootstraps": 2}}
    )
    ci = (ate - 0.1, ate + 0.1)
    sub = runner._run_data_subset_test(ate, ci, model, estimand, estimate, True, resample_seed=17)
    boot = runner._run_bootstrap_test(ate, ci, model, estimand, estimate, True, resample_seed=17)
    assert sub.test_name == RefutationTestType.DATA_SUBSET
    assert sub.details["resample_seed"] == 17
    assert boot.details["resample_seed"] == 17
```

- [ ] **Step 2: Run it to verify it fails**

Run: `.venv/bin/pytest tests/unit/test_causal_engine/test_refutation_seed_identity_2029.py::test_subset_and_bootstrap_details_carry_resample_seed -n 0 -q`
Expected: FAIL — `KeyError: 'resample_seed'`.

- [ ] **Step 3: Record the seed**

In `_run_data_subset_test`'s returned `details` dict add `"resample_seed": resample_seed,` next to `"stopped_for_budget"`; do the same in `_run_bootstrap_test`. Check the signatures: both already take `resample_seed: Optional[int] = None` (keyword-only after `deadline`); if the test's positional call fails on `deadline`, pass `deadline=None, resample_seed=17` explicitly in the test instead.

- [ ] **Step 4: Run the test to verify it passes**

Run: same command. Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/causal_engine/refutation_runner.py tests/unit/test_causal_engine/test_refutation_seed_identity_2029.py
git commit -m "feat(refutation): persist the resample seed in subset and bootstrap details (#2029)"
```

---

### Task 3: The node's calibration probe uses the seed too

**Files:**
- Modify: `src/agents/causal_impact/nodes/refutation.py` (~2047–2055 the `num_simulations=1` probe)
- Test: `tests/unit/test_agents/test_causal_impact/test_refutation_seed_probe_2029.py`

- [ ] **Step 1: Write the failing test**

```python
"""#2029: the per-refit calibration probe passes the run's seed to DoWhy."""
from __future__ import annotations

import inspect

from src.agents.causal_impact.nodes import refutation as node


def test_probe_source_passes_random_state():
    src = inspect.getsource(node)
    probe = src[src.index("per-refit calibration") - 2000 : src.index("per-refit calibration") + 400]
    assert 'method_name="placebo_treatment_refuter"' in probe
    assert "random_state=" in probe, "the calibration probe must be seeded like the scored refits"
```

- [ ] **Step 2: Run it to verify it fails**

Run: `.venv/bin/pytest tests/unit/test_agents/test_causal_impact/test_refutation_seed_probe_2029.py -n 0 -q`
Expected: FAIL on the `random_state=` assertion.

- [ ] **Step 3: Pass the seed**

In `nodes/refutation.py`, immediately before the `if deadline is not None and time.monotonic() + per_refit_hint <= deadline:` block that contains the probe, add:
```python
            probe_seed = self.runner._seed_for(query_id)
```
and inside the `run_bounded_with_budget(causal_model.refute_estimate, ...)` call add `random_state=probe_seed,` after `num_simulations=1,`. (`query_id` is the same variable later passed as `estimate_id=query_id`.)

- [ ] **Step 4: Run the test and the node suite**

Run: `.venv/bin/pytest tests/unit/test_agents/test_causal_impact/test_refutation_seed_probe_2029.py tests/unit/test_agents/test_causal_impact/test_refutation.py -n 0 -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/agents/causal_impact/nodes/refutation.py tests/unit/test_agents/test_causal_impact/test_refutation_seed_probe_2029.py
git commit -m "fix(refutation-node): seed the per-refit calibration probe (#2029)"
```

---

### Task 4: Migration 139 widens `delta_percent`

**Files:**
- Create: `database/migrations/139_causal_validations_delta_percent_numeric_12_4.sql`
- Test: `tests/unit/test_database/test_migration_139_delta_percent.py`

- [ ] **Step 1: Write the failing content-lock test**

```python
"""Migration 139 content lock: causal_validations.delta_percent DECIMAL(8,4) -> NUMERIC(12,4)."""
from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
MIGRATION = REPO_ROOT / "database" / "migrations" / "139_causal_validations_delta_percent_numeric_12_4.sql"


def _statements() -> str:
    return "\n".join(
        ln for ln in MIGRATION.read_text(encoding="utf-8").splitlines() if not ln.strip().startswith("--")
    )


def test_migration_exists_and_is_numbered_139():
    assert MIGRATION.exists()
    assert len(list((REPO_ROOT / "database").rglob("139_*"))) == 1


def test_widens_the_column_idempotently():
    sql = _statements()
    assert re.search(r"ALTER\s+TABLE\s+public\.causal_validations", sql, re.I)
    assert re.search(r"ALTER\s+COLUMN\s+delta_percent\s+TYPE\s+NUMERIC\(12,\s*4\)", sql, re.I)
    # idempotent: guarded on the current type so a re-run is a no-op
    assert "numeric_precision" in sql and "12" in sql


def test_no_transaction_control_of_its_own():
    sql = _statements().upper()
    assert "BEGIN;" not in sql and "COMMIT;" not in sql
```

- [ ] **Step 2: Run it to verify it fails**

Run: `.venv/bin/pytest tests/unit/test_database/test_migration_139_delta_percent.py -n 0 -q`
Expected: FAIL — file does not exist.

- [ ] **Step 3: Write the migration**

```sql
-- ============================================================================
-- Migration 139: causal_validations.delta_percent DECIMAL(8,4) -> NUMERIC(12,4)
-- (#2029, lane 1 of the #1991 debts 3/4 wave)
-- ============================================================================
-- WHAT: widen public.causal_validations.delta_percent from DECIMAL(8,4)
--   (max 9999.9999, database/ml/010_causal_validation_tables.sql:84) to
--   NUMERIC(12,4) (max 99999999.9999).
-- WHY: every refuter's delta_percent is |delta| / |original| * 100 and the
--   four perturbation tests compute it UNCLAMPED with a 1e-10 floor
--   (refutation_runner.py _run_placebo_test / _run_random_common_cause_test /
--   _run_data_subset_test / _run_bootstrap_test). A near-zero claim overflows
--   the column, and CausalValidationRepository.save_suite inserts the whole
--   suite in ONE call, so one overflowing row loses every row's persistence
--   (logged, returns []). Lane G clamped only the negative-control test.
--   The repository now clamps EVERY row at the write boundary to this column's
--   bound (src/repositories/causal_validation.py DELTA_PERCENT_COLUMN_MAX) and
--   keeps the exact value in details_json.delta_percent_exact.
-- SAFETY: widening a numeric type is a metadata-only change on Postgres 15
--   (no rewrite for precision increase with same scale); idempotent -- guarded
--   on information_schema so a re-run is a no-op. No BEGIN/COMMIT of its own:
--   scripts/run_migrations.sh wraps the file in --single-transaction.
-- ============================================================================

DO $$
BEGIN
    IF EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_schema = 'public' AND table_name = 'causal_validations'
          AND column_name = 'delta_percent'
          AND (numeric_precision IS DISTINCT FROM 12 OR numeric_scale IS DISTINCT FROM 4)
    ) THEN
        ALTER TABLE public.causal_validations ALTER COLUMN delta_percent TYPE NUMERIC(12, 4);
    END IF;
END $$;

COMMENT ON COLUMN public.causal_validations.delta_percent IS
    'Percentage change from the original effect, |delta|/|original|*100, clamped by the '
    'repository to 99999999.9999 (migration 139, #2029); the exact value rides in '
    'details_json.delta_percent_exact when clamping occurred.';
```

- [ ] **Step 4: Run the test to verify it passes**

Run: same command. Expected: 3 passed.

- [ ] **Step 5: Rehearse on the live database in BEGIN…ROLLBACK with a positive control**

Run:
```bash
{ echo "BEGIN;"; cat database/migrations/139_causal_validations_delta_percent_numeric_12_4.sql; \
  echo "SELECT numeric_precision, numeric_scale FROM information_schema.columns WHERE table_name='causal_validations' AND column_name='delta_percent'; ROLLBACK;"; } \
  | docker exec -i supabase-db psql -U postgres -d postgres -v ON_ERROR_STOP=1
```
Expected output: `12|4` before ROLLBACK; then re-check reads `8|4` (unchanged live). Record the two readings in `docs/demos/results/2026-09-12_lane2029/rehearsal_139.txt` (untracked evidence dir).

- [ ] **Step 6: Commit**

```bash
git add database/migrations/139_causal_validations_delta_percent_numeric_12_4.sql tests/unit/test_database/test_migration_139_delta_percent.py
git commit -m "feat(db): migration 139 widens causal_validations.delta_percent to NUMERIC(12,4) (#2029)"
```

---

### Task 5: One clamp at the write boundary

**Files:**
- Modify: `src/repositories/causal_validation.py` (`_test_to_row` ~465–507, `save_single_test` row ~187–210)
- Modify: `src/causal_engine/refutation_runner.py` (`_DELTA_PERCENT_COLUMN_MAX` ~716–720; `_run_negative_control_test` ~2711–2713)
- Modify: `tests/unit/test_causal_engine/test_refutation_runner_negative_control_2007.py` (~653–671)
- Test: `tests/unit/test_repositories/test_causal_validation_delta_clamp_2029.py`

- [ ] **Step 1: Write the failing repository test**

```python
"""#2029: every persisted row's delta_percent is clamped to the column bound at the write
boundary, with the exact value kept in details_json.delta_percent_exact."""
from __future__ import annotations

from src.causal_engine.refutation_runner import (
    GateDecision,
    RefutationResult,
    RefutationStatus,
    RefutationSuite,
    RefutationTestType,
)
from src.repositories.causal_validation import DELTA_PERCENT_COLUMN_MAX, CausalValidationRepository


def _suite(delta: float) -> RefutationSuite:
    test = RefutationResult(
        test_name=RefutationTestType.PLACEBO_TREATMENT,
        status=RefutationStatus.PASSED,
        original_effect=1e-8,
        refuted_effect=0.01,
        p_value=0.4,
        delta_percent=delta,
        details={"message": "m"},
    )
    return RefutationSuite(
        passed=True, confidence_score=0.9, tests=[test], gate_decision=GateDecision.PROCEED,
        treatment_variable="t", outcome_variable="y", brand="B",
    )


def test_bound_is_the_numeric_12_4_maximum():
    assert DELTA_PERCENT_COLUMN_MAX == 99999999.9999


def test_row_is_clamped_and_exact_value_kept():
    repo = CausalValidationRepository(client=None)
    suite = _suite(1e8)  # 0.01 / 1e-8 * 100
    row = repo._test_to_row(test=suite.tests[0], suite=suite, estimate_id="e", estimate_source="causal_paths")
    assert row["delta_percent"] == DELTA_PERCENT_COLUMN_MAX
    assert row["details_json"]["delta_percent_exact"] == 1e8


def test_row_below_bound_is_untouched():
    repo = CausalValidationRepository(client=None)
    suite = _suite(42.5)
    row = repo._test_to_row(test=suite.tests[0], suite=suite, estimate_id="e", estimate_source="causal_paths")
    assert row["delta_percent"] == 42.5
    assert "delta_percent_exact" not in row["details_json"]
```

- [ ] **Step 2: Run it to verify it fails**

Run: `.venv/bin/pytest tests/unit/test_repositories/test_causal_validation_delta_clamp_2029.py -n 0 -q`
Expected: FAIL — `ImportError: cannot import name 'DELTA_PERCENT_COLUMN_MAX'`.

- [ ] **Step 3: Implement the clamp**

In `src/repositories/causal_validation.py`, module level (after `logger`):
```python
# causal_validations.delta_percent is NUMERIC(12,4) since migration 139 (#2029):
# the largest value the column stores. Every writer clamps here, once, and keeps
# the exact value in details_json.delta_percent_exact -- one overflowing row used
# to fail the whole bulk save_suite insert.
DELTA_PERCENT_COLUMN_MAX = 99999999.9999


def _clamp_delta_percent(value: Any, details: Dict[str, Any]) -> Any:
    """Return the column-safe delta_percent; record the exact value when clamped."""
    if value is None:
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return value
    if v > DELTA_PERCENT_COLUMN_MAX:
        details["delta_percent_exact"] = v
        return DELTA_PERCENT_COLUMN_MAX
    return v
```
In `_test_to_row`, before the `return {...}`:
```python
        details = dict(test.details)
        delta_percent = _clamp_delta_percent(test.delta_percent, details)
```
and use `"delta_percent": delta_percent,` and `"details_json": to_plain_json(details),`. Apply the same two lines in `save_single_test`'s row construction.

In `refutation_runner.py`: delete `_DELTA_PERCENT_COLUMN_MAX` and its comment; in `_run_negative_control_test` replace `delta_percent = min(ratio, _DELTA_PERCENT_COLUMN_MAX)` with `delta_percent = ratio` and shorten the comment to: `# The column bound is applied ONCE at the write boundary (CausalValidationRepository, #2029); the exact ratio also rides in details["control_to_claimed_ratio"].`

Move the old test: in `test_refutation_runner_negative_control_2007.py` replace `test_delta_percent_is_clamped_to_the_column_range_and_the_exact_ratio_is_kept` with:
```python
def test_delta_percent_is_the_exact_ratio_the_repository_clamps_later():
    """#2029 moved the column clamp to CausalValidationRepository._test_to_row; the
    runner now reports the exact ratio (a 50000 % ratio stays 50000)."""
    result = RefutationRunner()._run_negative_control_test(
        0.0001, (NC_OUTCOME, 0.05, (0.04, 0.06), 1500)
    )
    assert result.status == F
    assert result.delta_percent == 50000.0
    assert result.details["control_to_claimed_ratio"] == 50000.0
```

- [ ] **Step 4: Run the tests**

Run: `.venv/bin/pytest tests/unit/test_repositories/test_causal_validation_delta_clamp_2029.py tests/unit/test_repositories/test_causal_validation.py tests/unit/test_causal_engine/test_refutation_runner_negative_control_2007.py -n 0 -q`
Expected: all pass. Also `grep -rn "_DELTA_PERCENT_COLUMN_MAX" src tests` → no hits.

- [ ] **Step 5: Commit**

```bash
git add src/repositories/causal_validation.py src/causal_engine/refutation_runner.py tests/unit/test_repositories/test_causal_validation_delta_clamp_2029.py tests/unit/test_causal_engine/test_refutation_runner_negative_control_2007.py
git commit -m "fix(persistence): clamp delta_percent once at the write boundary, keep the exact value (#2029)"
```

---

### Task 6: Lane close — lint, targeted suites, codex round, push, PR

- [ ] **Step 1: Lint and format (no cache)**

Run: `.venv/bin/ruff check --no-cache src/causal_engine/refutation_runner.py src/repositories/causal_validation.py src/agents/causal_impact/nodes/refutation.py tests/unit/test_causal_engine/test_refutation_seed_identity_2029.py tests/unit/test_repositories/test_causal_validation_delta_clamp_2029.py tests/unit/test_database/test_migration_139_delta_percent.py tests/unit/test_agents/test_causal_impact/test_refutation_seed_probe_2029.py && .venv/bin/ruff format --check --no-cache <same files>`
Expected: clean. Scoped mypy: `.venv/bin/mypy --config-file pyproject.toml src/repositories/causal_validation.py` → no new errors.

- [ ] **Step 2: Targeted suites**

Run: `.venv/bin/pytest tests/unit/test_causal_engine tests/unit/test_repositories/test_causal_validation.py tests/unit/test_agents/test_causal_impact tests/unit/test_database/test_migration_139_delta_percent.py -n 0 -q -p no:cacheprovider`
Expected: all pass. Record counts in the PR body.

- [ ] **Step 3: Codex read-only round**

Write `.claude/codex/lane2029-brief.md` with the diff summary, the spec §4 text, and the mandatory pushback paragraph from CLAUDE.md ("If a recommendation solves a labeling problem…"). Run: `codex exec -C /home/enunez/Projects/e2i_causal_analytics/.worktrees/lane-2029 --sandbox read-only "$(cat .claude/codex/lane2029-brief.md)" < /dev/null > docs/demos/results/2026-09-12_lane2029/codex_r1.txt`. Iterate until `VERDICT: ACCEPT`; fold every HIGH.

- [ ] **Step 4: Push once, open the PR**

```bash
git branch --show-current   # claude/2029-refutation-determinism
git push -u origin claude/2029-refutation-determinism
gh pr create --title "fix(refutation): seed the DoWhy refuters, persist seeds, clamp delta_percent at the write boundary (#2029)" --body-file .claude/codex/lane2029-pr-body.md
```
PR body: what/why/measured (identity test, near-zero suite persists), migration 139 rehearsal readings, post-deploy cert plan (spec §4 "Cert"), and the attribution footer from the session.

- [ ] **Step 5: CI, owner go, merge, deploy watch, cert**

CI via `gh api "repos/enunezvn/e2i_causal_analytics/actions/runs?head_sha=<FULL40>"`. On the owner's go: `gh api -X PUT repos/enunezvn/e2i_causal_analytics/pulls/<N>/merge -f merge_method=merge`. Deploy gate: 0 non-terminal on `workflows/deploy.yml/runs`, `docker inspect e2i_api --format '{{.Config.Image}}'` == main HEAD. Cert (spec §4): ledger has `migrations/139_…`; `\d public.causal_validations` shows `numeric(12,4)`; two consecutive `run_discovery.py` runs on the deployed image (`docs/demos/results/2026-09-10_sensitivity_calibration/run_discovery.py <label> <dir>`), compare `placebo_treatment` and `random_common_cause` `refuted_effect`/`p_value`/status pair-for-pair (identical), and:
```sql
with d as (select test_type::text tt, case when jsonb_typeof(details_json)='string' then (details_json #>> '{}')::jsonb else details_json end dj from public.causal_validations where created_at > '<post_started_at>')
select tt, count(*), count(*) filter (where dj ? 'random_state' or dj ? 'resample_seed') from d group by 1 order by 1;
```
Expected: placebo/rcc/data_subset/bootstrap rows all carry a seed key; sensitivity/negative-control none. Negative control: the same query over the pre-deploy window reads 0 seeded rows. Write `docs/demos/results/2026-09-12_lane2029/cert.md`, comment #2029, close it.
