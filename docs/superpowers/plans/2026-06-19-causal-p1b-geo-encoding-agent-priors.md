# P1b — Geographic-Region One-Hot Encoding + SSOT Agent-Prior Seeding Implementation Plan

> For agentic workers: REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax. Execute in an isolated git worktree; TDD red-first; commit per task; drive to a green fixed point and use codex:codex-rescue when stuck.

**Goal:** Complete the patient-grain rigor by (A) one-hot encoding the categorical `geographic_region` so the modeled RETENTION adjustment set `{disease_severity, academic_hcp, geographic_region}` is fully usable end-to-end, and (B) seeding the causal_impact agent's guided-discovery priors from each question's *modeled* confounders (`confounder->treatment`, `confounder->outcome` required edges) instead of the generic `KNOWN_CAUSAL_RELATIONSHIPS`/`COMMON_CONFOUNDERS` constants that have ~0 overlap with the real covariates.

**Architecture:** Add a per-dataset categorical-covariate allowlist mirroring `_CAUSAL_NUMERIC_COLUMNS`; `_load_agent_estimation_frame` admits `geographic_region` through the same column security gate, leaves it as a string through the numeric-coercion loop, then one-hot expands it (post-coercion) into stable `geographic_region=<level>` 0/1 float dummies, returning the EXPANDED covariate names so the dummies flow through the FWL pre-rank, the executors, and the agent's `confounders`. Separately, thread the resolved/modeled confounders into a new `modeled_confounders` state channel; `graph_builder._run_discovery` turns them into `CausalPriorKnowledge.required_edges` (each `conf->treatment` and `conf->outcome`) for guided PC, where `pc_wrapper` already feeds `prior_knowledge` into causal-learn's `BackgroundKnowledge`.

**Tech Stack:** Python 3.12, FastAPI, pandas/numpy, networkx, causal-learn (via the discovery runner), pytest. No new deps.

**Scope (this plan):** patient grain backend + the causal_impact graph_builder discovery prior. **Pre-req:** P1 (`2026-06-19-causal-p1-patient-grain-backend.md`) MERGED — this plan consumes its `_discover_candidate_questions` / `_CandidateQuestion` / `DiscoveredEffect.adjustment_set` and reseeded `causal_paths`. **Plan against POST-#1030 code** for `graph_builder.py` (the #1030 worktree `_construct_dag` already draws `conf->treatment`/`conf->outcome` edges and has DELETED `COMMON_CONFOUNDERS`; the guided-discovery prior is the remaining generic-constant gap). **Out of scope → other plans:** P0 (FE page), P2 (HCP), P3 (trigger), enrichment.

---

## File Structure

| File | Create/Modify | Responsibility |
|---|---|---|
| `src/api/routes/causal.py` | Modify | Add `_CAUSAL_CATEGORICAL_COLUMNS`; add `geographic_region` to the `patient_journeys` covariate allowlist; add `_one_hot_categoricals()` helper; make `_load_agent_estimation_frame` admit + one-hot expand categorical covariates and return expanded names; thread resolved covariate names into `_run_agent_analysis_task` |
| `src/agents/causal_impact/state.py` | Modify | Add `modeled_confounders: NotRequired[List[str]]` channel to `CausalImpactState` |
| `src/agents/causal_impact/nodes/graph_builder.py` | Modify (post-#1030) | In `_run_discovery`, seed guided `CausalPriorKnowledge.required_edges` from `state['modeled_confounders']` (`conf->treatment`, `conf->outcome`), replacing the generic-constant role for the prior |
| `tests/unit/test_api/test_causal_geo_encoding.py` | Create (Test) | One-hot helper + loader expansion + allowlist gate coverage |
| `tests/unit/test_agents/test_causal_impact/test_graph_builder_priors.py` | Create (Test) | `_run_discovery` seeds confounder->treatment/outcome required edges from `modeled_confounders` |
| `tests/unit/test_api/test_causal_agent_analyze.py` | Modify (Test) | Assert `_run_agent_analysis_task` receives the EXPANDED covariate names (geo dummies, not the raw categorical) |

---

### Task 1: Add the categorical-covariate allowlist + one-hot helper

The loader today coerces every covariate to float and drops `geographic_region` because it is a string. We add a second per-dataset allowlist (`_CAUSAL_CATEGORICAL_COLUMNS`) parallel to `_CAUSAL_NUMERIC_COLUMNS`, register `geographic_region` in both the `patient_journeys` covariate spec and the categorical set, and add a pure `_one_hot_categoricals()` helper. One-hot is correct: `geographic_region` is an UNORDERED 4-level categorical (live values `midwest/south/northeast/west`, 100% populated), so a numeric cast would be meaningless; dummies are 0/1 floats the DoWhy/EconML executors and the FWL screen consume directly. `drop_first=True` drops one level as the reference category to avoid the dummy-variable trap (perfect collinearity with the intercept).

**Files:**
- Modify: `src/api/routes/causal.py:804-842` (specs + numeric set; add categorical set)
- Modify: `src/api/routes/causal.py` (add `_one_hot_categoricals` near `_load_agent_estimation_frame` ~1410)
- Test: `tests/unit/test_api/test_causal_geo_encoding.py` (create)

- [ ] **Step 1: Write the failing test for the one-hot helper + allowlist registration**

```python
# tests/unit/test_api/test_causal_geo_encoding.py
"""Coverage for geographic_region one-hot encoding in the causal loader.

geographic_region is an unordered 4-level categorical (midwest/south/northeast/
west). It is part of the modeled RETENTION adjustment set but was dropped by the
numeric-only allowlist intersection. P1b admits it through the same column
security gate and one-hot expands it into 0/1 float dummies the executors and the
FWL screen consume directly.
"""

import pandas as pd

from src.api.routes.causal import (
    _CAUSAL_CATEGORICAL_COLUMNS,
    _CAUSAL_DATASET_SPECS,
    _CAUSAL_NUMERIC_COLUMNS,
    _one_hot_categoricals,
)


def test_geographic_region_registered_as_categorical_covariate():
    """geographic_region is an allowed covariate AND flagged categorical (NOT
    numeric) so the loader expands rather than float-coerces it."""
    spec = _CAUSAL_DATASET_SPECS["patient_journeys"]
    assert "geographic_region" in spec["covariate"]
    assert "geographic_region" in _CAUSAL_CATEGORICAL_COLUMNS["patient_journeys"]
    # It must NOT be in the numeric set — that would float-coerce it to None.
    assert "geographic_region" not in _CAUSAL_NUMERIC_COLUMNS["patient_journeys"]


def test_one_hot_expands_into_stable_drop_first_float_dummies():
    df = pd.DataFrame(
        {
            "treatment_arm": [1.0, 0.0, 1.0, 0.0],
            "persistent_180d": [1.0, 0.0, 1.0, 1.0],
            "disease_severity": [2.0, 1.0, 3.0, 2.0],
            "geographic_region": ["south", "west", "midwest", "northeast"],
        }
    )
    out, dummy_names = _one_hot_categoricals(df, ["geographic_region"])
    # Reference level (first sorted = "midwest") dropped; 3 dummies remain.
    assert dummy_names == [
        "geographic_region=northeast",
        "geographic_region=south",
        "geographic_region=west",
    ]
    # Original categorical column is gone; dummies are float 0/1.
    assert "geographic_region" not in out.columns
    for name in dummy_names:
        assert name in out.columns
        assert out[name].dtype == float
        assert set(out[name].unique()) <= {0.0, 1.0}
    # midwest row encodes as all-zero across the dummies (the reference level).
    midwest_row = out.iloc[2]
    assert midwest_row["geographic_region=northeast"] == 0.0
    assert midwest_row["geographic_region=south"] == 0.0
    assert midwest_row["geographic_region=west"] == 0.0
    # Non-categorical columns are untouched.
    assert list(out["treatment_arm"]) == [1.0, 0.0, 1.0, 0.0]


def test_one_hot_noop_when_no_categoricals_present():
    df = pd.DataFrame({"treatment_arm": [1.0, 0.0], "disease_severity": [2.0, 1.0]})
    out, dummy_names = _one_hot_categoricals(df, ["geographic_region"])
    assert dummy_names == []
    assert list(out.columns) == ["treatment_arm", "disease_severity"]
```

- [ ] **Step 2: Run it to confirm it fails**

Run: `pytest tests/unit/test_api/test_causal_geo_encoding.py -v`
Expected: FAIL — `ImportError: cannot import name '_CAUSAL_CATEGORICAL_COLUMNS'` / `_one_hot_categoricals`.

- [ ] **Step 3: Register `geographic_region` in the specs + add the categorical allowlist**

In `src/api/routes/causal.py`, add `geographic_region` to the `patient_journeys` covariate list (insert after `"academic_hcp",` at line 812):

```python
        "covariate": [
            "disease_severity",
            "engagement_score",
            "age_at_diagnosis",
            "academic_hcp",
            "geographic_region",
            "egfr",
            "proteinuria_g_day",
            "ldh_ratio",
            "urticaria_severity_uas7",
            "ecog_performance_status",
        ],
```

Then add the categorical allowlist immediately after the `_CAUSAL_NUMERIC_COLUMNS` block (after line 842):

```python
# Categorical covariates that are ONE-HOT ENCODED before the frame reaches the
# executors (DoWhy/EconML require numeric inputs). These are DELIBERATELY absent
# from _CAUSAL_NUMERIC_COLUMNS so the loader does NOT float-coerce them to None;
# _one_hot_categoricals expands each into stable ``<col>=<level>`` 0/1 float
# dummies (drop_first reference level). geographic_region is the modeled
# RETENTION confounder (treatment_arm -> persistent_180d / discontinued_180d):
# an unordered 4-level region (midwest/south/northeast/west, 100% populated).
_CAUSAL_CATEGORICAL_COLUMNS: Dict[str, set] = {
    "patient_journeys": {"geographic_region"},
}
```

- [ ] **Step 4: Add the `_one_hot_categoricals` helper**

In `src/api/routes/causal.py`, add this pure helper directly above `_load_agent_estimation_frame` (before line 1410):

```python
def _one_hot_categoricals(
    df: "pd.DataFrame",  # type: ignore[name-defined] # noqa: F821
    categorical_cols: List[str],
) -> tuple["pd.DataFrame", List[str]]:  # type: ignore[name-defined] # noqa: F821
    """One-hot encode the categorical columns present in ``df`` into stable
    ``<col>=<level>`` 0/1 float dummies, dropping the original column.

    Unordered categoricals (e.g. geographic_region) cannot be float-coerced into
    a meaningful covariate; the DoWhy/EconML executors and the FWL screen require
    numeric inputs, so each is expanded into indicator columns. ``drop_first``
    drops the first sorted level as the reference category to avoid the
    dummy-variable trap (perfect collinearity with the intercept). Level order is
    sorted so the dummy names are deterministic across runs. Columns not present
    in ``df`` are silently skipped (a brand subset may lack a level — still
    deterministic because the names are derived from the observed levels).

    Returns ``(expanded_df, dummy_names)`` where ``dummy_names`` is the ordered
    list of every generated dummy column (the names the caller substitutes for
    the original categorical in the covariate/adjustment set).
    """
    import pandas as pd

    present = [c for c in categorical_cols if c in df.columns]
    if not present:
        return df, []
    out = df.copy()
    dummy_names: List[str] = []
    for col in present:
        levels = sorted(str(v) for v in out[col].dropna().unique())
        # drop_first: the first sorted level is the reference category.
        for level in levels[1:]:
            name = f"{col}={level}"
            out[name] = (out[col].astype(str) == level).astype(float)
            dummy_names.append(name)
        out = out.drop(columns=[col])
    return out, dummy_names
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/unit/test_api/test_causal_geo_encoding.py -v`
Expected: PASS (all three).

- [ ] **Step 6: Commit**

```bash
git add src/api/routes/causal.py tests/unit/test_api/test_causal_geo_encoding.py
git commit -m "feat(causal): register geographic_region as a one-hot categorical covariate (+helper)"
```

---

### Task 2: Make `_load_agent_estimation_frame` admit + expand categorical covariates

The loader's column security gate (allowlist), provenance filter, treatment/outcome usability drop, and numeric coercion are PRESERVED. We only (1) let an allowed categorical covariate skip float-coercion (kept as a string in the record), and (2) one-hot expand AFTER record assembly, returning the EXPANDED `select_cols` so callers learn the dummy names. `geographic_region` is never a treatment/outcome (it is not in those spec lists), so the usability gate is unaffected.

**Files:**
- Modify: `src/api/routes/causal.py:1410-1501` (`_load_agent_estimation_frame`)
- Test: `tests/unit/test_api/test_causal_geo_encoding.py` (add)

- [ ] **Step 1: Write the failing loader test (mock the Supabase client)**

```python
# tests/unit/test_api/test_causal_geo_encoding.py  (append)
import pytest
from unittest.mock import AsyncMock, patch

from src.api.routes import causal as causal_routes

# _load_agent_estimation_frame does a FUNCTION-LOCAL
# ``from src.memory.services.factories import get_async_supabase_client`` (the
# name is re-bound from the source module at call time), so the patch target is
# the SOURCE module, not ``causal_routes``.
_CLIENT_FACTORY = "src.memory.services.factories.get_async_supabase_client"


class _FakeQuery:
    """Minimal async-execute PostgREST stub: select/eq/limit chain -> rows."""

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


@pytest.mark.asyncio
async def test_loader_expands_geographic_region_into_dummies():
    rows = [
        {"treatment_arm": 1, "persistent_180d": 1, "disease_severity": 2,
         "academic_hcp": 1, "geographic_region": "south"},
        {"treatment_arm": 0, "persistent_180d": 0, "disease_severity": 1,
         "academic_hcp": 0, "geographic_region": "west"},
        {"treatment_arm": 1, "persistent_180d": 1, "disease_severity": 3,
         "academic_hcp": 1, "geographic_region": "midwest"},
    ]
    with patch(_CLIENT_FACTORY, AsyncMock(return_value=_FakeClient(rows))):
        df, select_cols = await causal_routes._load_agent_estimation_frame(
            dataset="patient_journeys",
            treatment_var="treatment_arm",
            outcome_var="persistent_180d",
            covariates=["disease_severity", "academic_hcp", "geographic_region"],
            limit=1500,
        )
    # The raw categorical is gone; dummies (drop_first ref=midwest) are present.
    assert "geographic_region" not in df.columns
    assert "geographic_region=south" in df.columns
    assert "geographic_region=west" in df.columns
    assert "geographic_region=midwest" not in df.columns  # reference level
    # Returned select_cols carries the EXPANDED names (not the raw categorical),
    # numeric columns coerced to float, treatment/outcome retained.
    assert "geographic_region" not in select_cols
    assert "geographic_region=south" in select_cols
    assert "geographic_region=west" in select_cols
    assert df["disease_severity"].dtype == float
    assert df["geographic_region=south"].dtype == float


@pytest.mark.asyncio
async def test_loader_rejects_unallowed_column_still_400():
    """The security gate is preserved: an off-allowlist column still 400s. The
    allowlist check runs BEFORE the client import, so this raises without ever
    touching the store; the patch is defensive."""
    with patch(_CLIENT_FACTORY, AsyncMock(return_value=_FakeClient([]))):
        with pytest.raises(causal_routes.HTTPException) as ei:
            await causal_routes._load_agent_estimation_frame(
                dataset="patient_journeys",
                treatment_var="treatment_arm",
                outcome_var="persistent_180d",
                covariates=["totally_made_up_col"],
                limit=10,
            )
    assert ei.value.status_code == 400
```

- [ ] **Step 2: Run them to confirm they fail**

Run: `pytest tests/unit/test_api/test_causal_geo_encoding.py -k loader -v`
Expected: FAIL — `test_loader_expands_geographic_region_into_dummies` fails because the current loader float-coerces `geographic_region` to None (it is a covariate, not treatment/outcome, so the row survives but the column is `None`), so no dummy columns exist and `select_cols` still contains the raw name. (`test_loader_rejects_unallowed_column_still_400` already passes — it pins the preserved gate.)

- [ ] **Step 3: Implement categorical-aware loading + expansion**

In `src/api/routes/causal.py`, edit `_load_agent_estimation_frame`. After the `numeric_cols = _CAUSAL_NUMERIC_COLUMNS.get(dataset, set())` line (1471), add the categorical set; in the per-column loop, skip coercion for categorical columns; after the `records` loop, one-hot expand and rewrite `select_cols`.

Replace the block from line 1471 (`numeric_cols = ...`) through the final `return` (line 1501) with:

```python
    numeric_cols = _CAUSAL_NUMERIC_COLUMNS.get(dataset, set())
    categorical_cols = _CAUSAL_CATEGORICAL_COLUMNS.get(dataset, set())
    records: List[Dict[str, Any]] = []
    for row in rows:
        record: Dict[str, Any] = {}
        usable = True
        for col in select_cols:
            value = row.get(col)
            if col in numeric_cols and value is not None:
                try:
                    value = float(value)
                except (TypeError, ValueError):
                    value = None
            # Categorical covariates (e.g. geographic_region) are kept as their
            # raw string; _one_hot_categoricals expands them below. They are
            # never treatment/outcome, so they do not gate row usability.
            if col in (treatment_var, outcome_var) and value is None:
                usable = False
                break
            record[col] = value
        if usable:
            records.append(record)

    if not records:
        raise HTTPException(
            status_code=503,
            detail=(
                "No usable estimation rows for the requested variables "
                f"({treatment_var} -> {outcome_var}) in dataset '{dataset}'."
            ),
        )

    import pandas as pd

    frame = pd.DataFrame(records)
    # One-hot expand any categorical covariates into numeric 0/1 dummies the
    # executors + FWL screen consume; substitute the dummy names for the raw
    # categorical in the returned column list (the caller's covariate/adjustment
    # set). The numeric-coercion security gate above is unchanged.
    requested_categoricals = [c for c in select_cols if c in categorical_cols]
    frame, dummy_names = _one_hot_categoricals(frame, requested_categoricals)
    expanded_cols = [c for c in select_cols if c not in categorical_cols] + dummy_names

    return frame, expanded_cols
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/test_api/test_causal_geo_encoding.py -v`
Expected: PASS (all loader + helper tests).

- [ ] **Step 5: Commit**

```bash
git add src/api/routes/causal.py tests/unit/test_api/test_causal_geo_encoding.py
git commit -m "feat(causal): one-hot expand geographic_region in the agent estimation loader"
```

---

### Task 3: Thread the EXPANDED covariate names into the agent run

`run_causal_agent_analysis` discards the loader's returned `_select_cols` and passes the RAW `covariates` (still containing `geographic_region`) to `_run_agent_analysis_task`, which sets `initial_state["confounders"] = covariates`. After Task 2 the frame has `geographic_region=<level>` dummies but no `geographic_region` column, so the agent would receive a confounder name that is not a column. We derive the resolved covariate names from the loader's expanded `select_cols` (minus treatment/outcome) and pass THOSE.

**Files:**
- Modify: `src/api/routes/causal.py:1551-1589` (`run_causal_agent_analysis`)
- Test: `tests/unit/test_api/test_causal_agent_analyze.py` (add)

- [ ] **Step 1: Write the failing test asserting expanded covariates reach the task**

```python
# tests/unit/test_api/test_causal_agent_analyze.py  (append)
import pandas as pd
import pytest
from unittest.mock import AsyncMock, patch

from src.api.routes import causal as causal_routes
from src.api.schemas.causal import AgentCausalAnalysisRequest


@pytest.mark.asyncio
async def test_agent_analyze_passes_expanded_geo_dummies_as_covariates():
    """run_causal_agent_analysis must hand _run_agent_analysis_task the EXPANDED
    covariate names (geo dummies), not the raw categorical 'geographic_region'."""
    frame = pd.DataFrame(
        {
            "treatment_arm": [1.0, 0.0],
            "persistent_180d": [1.0, 0.0],
            "disease_severity": [2.0, 1.0],
            "academic_hcp": [1.0, 0.0],
            "geographic_region=south": [1.0, 0.0],
            "geographic_region=west": [0.0, 1.0],
        }
    )
    expanded_cols = [
        "treatment_arm",
        "persistent_180d",
        "disease_severity",
        "academic_hcp",
        "geographic_region=south",
        "geographic_region=west",
    ]

    captured: dict = {}

    async def _fake_task(analysis_id, request, df, covariates, data_source):
        captured["covariates"] = covariates

    req = AgentCausalAnalysisRequest(
        treatment_var="treatment_arm",
        outcome_var="persistent_180d",
        dataset="patient_journeys",
        covariates=["disease_severity", "academic_hcp", "geographic_region"],
        limit=1500,
    )

    class _BG:
        def add_task(self, fn, *args):
            # Execute synchronously so we can capture the covariates argument.
            import asyncio

            asyncio.get_event_loop().create_task(fn(*args))

    with patch.object(
        causal_routes,
        "_load_agent_estimation_frame",
        AsyncMock(return_value=(frame, expanded_cols)),
    ), patch.object(
        causal_routes, "_run_agent_analysis_task", _fake_task
    ), patch.object(
        causal_routes._agent_analysis_store, "set", AsyncMock()
    ):
        await causal_routes.run_causal_agent_analysis(
            req, _BG(), user={"sub": "t"}
        )
        # Let the scheduled task run.
        import asyncio

        await asyncio.sleep(0)

    assert "geographic_region" not in captured["covariates"]
    assert "geographic_region=south" in captured["covariates"]
    assert "geographic_region=west" in captured["covariates"]
    # Treatment/outcome are never covariates.
    assert "treatment_arm" not in captured["covariates"]
    assert "persistent_180d" not in captured["covariates"]
```

- [ ] **Step 2: Run it to confirm it fails**

Run: `pytest tests/unit/test_api/test_causal_agent_analyze.py -k expanded_geo -v`
Expected: FAIL — `captured["covariates"]` contains `"geographic_region"` (the raw request covariate), not the dummies.

- [ ] **Step 3: Derive + pass the resolved covariates**

In `src/api/routes/causal.py`, edit `run_causal_agent_analysis`. After the `_load_agent_estimation_frame` call (line 1561-1568) that already binds `df, _select_cols`, derive the resolved covariates from `_select_cols` and pass them to the background task. Replace the `_select_cols` discard + the `background_tasks.add_task(...)` line.

Change line 1561 `df, _select_cols = await _load_agent_estimation_frame(` to `df, select_cols = await _load_agent_estimation_frame(` and immediately after the call closes (after line 1568 `)`), add:

```python
    # The loader EXPANDS categorical covariates (e.g. geographic_region) into
    # one-hot dummies; the agent's confounders must be the resolved frame columns
    # (the dummy names), not the raw categorical. Derive them from the loader's
    # returned column list, excluding treatment/outcome.
    resolved_covariates = [
        c for c in select_cols if c not in (request.treatment_var, request.outcome_var)
    ]
```

Then change the background-task dispatch (line 1587-1589) to pass `resolved_covariates`:

```python
    background_tasks.add_task(
        _run_agent_analysis_task, analysis_id, request, df, resolved_covariates, data_source
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/test_api/test_causal_agent_analyze.py -v`
Expected: PASS (new test + all existing agent-analyze tests, which exercise pure mappers unaffected by this wiring).

- [ ] **Step 5: Commit**

```bash
git add src/api/routes/causal.py tests/unit/test_api/test_causal_agent_analyze.py
git commit -m "fix(causal): pass loader-expanded covariate names (geo dummies) to the agent run"
```

---

### Task 4: Add the `modeled_confounders` state channel

The agent's `state["confounders"]` is the loaded covariate set (used as the manual-DAG adjustment set). For guided discovery priors we want the SAME modeled confounders threaded explicitly so the graph_builder can seed required edges without re-deriving them. LangGraph only persists declared TypedDict channels, so an undeclared key returned/read between nodes is dropped — the channel MUST be declared (this is the documented `estimation_data` lesson in `state.py:249`).

**Files:**
- Modify: `src/agents/causal_impact/state.py:202-208` (add channel near `brand`)
- Test: covered structurally by Task 5 (a state with `modeled_confounders` flows into `_run_discovery`); no standalone test for a bare TypedDict field.

- [ ] **Step 1: Add the channel**

In `src/agents/causal_impact/state.py`, inside `class CausalImpactState`, add the field immediately after the `brand: NotRequired[str]` line (line 202):

```python
    brand: NotRequired[str]  # Brand context
    # Modeled confounders for the question (the SSOT ``confounders_controlled`` /
    # the loaded adjustment set). Threaded SEPARATELY from ``confounders`` so the
    # graph_builder's GUIDED discovery can seed CausalPriorKnowledge.required_edges
    # (confounder->treatment, confounder->outcome) from the question's MODELED
    # backdoor set instead of the generic KNOWN_CAUSAL_RELATIONSHIPS constants
    # (~0 overlap with real covariates). Declared so LangGraph persists it across
    # nodes (undeclared channels are dropped — see ``estimation_data`` note below).
    modeled_confounders: NotRequired[List[str]]
```

- [ ] **Step 2: Set it in `_run_agent_analysis_task`'s initial_state**

In `src/api/routes/causal.py`, in `_run_agent_analysis_task`, add `modeled_confounders` to `initial_state` (after the `"confounders": covariates,` line at 1649):

```python
        "treatment_var": request.treatment_var,
        "outcome_var": request.outcome_var,
        "confounders": covariates,
        # The modeled adjustment set (== the resolved/expanded covariates here)
        # threaded so guided discovery seeds confounder->treatment/outcome priors.
        "modeled_confounders": covariates,
        "data_source": data_source,
```

- [ ] **Step 3: Run a smoke import to confirm no syntax/type break**

Run: `python -c "from src.agents.causal_impact.state import CausalImpactState; from src.api.routes import causal; print('ok')"`
Expected: prints `ok` (no import/syntax error).

- [ ] **Step 4: Commit**

```bash
git add src/agents/causal_impact/state.py src/api/routes/causal.py
git commit -m "feat(causal): add modeled_confounders state channel + populate it in the agent task"
```

---

### Task 5: Seed guided-discovery priors from `modeled_confounders` (post-#1030 graph_builder)

`graph_builder._run_discovery` builds a guided `CausalPriorKnowledge` with `tiers` and a single `required_edges=[(treatment, outcome)]`. The generic `KNOWN_CAUSAL_RELATIONSHIPS` constant (and, pre-#1030, `COMMON_CONFOUNDERS`) names pharma-commercial variables (`hcp_engagement_level`, `geographic_region`->`hcp_engagement_level`, …) with ~0 overlap with the real covariates (`disease_severity`, `academic_hcp`, `geographic_region=<level>` dummies, `egfr`, …), so they never fire for real questions. We seed the modeled confounders as REQUIRED edges (`conf->treatment` AND `conf->outcome`) so guided PC anchors them as confounders by construction while the data still selects the rest. `build_background_knowledge` already name-guards against columns not in the frame, and `pc_wrapper.discover` feeds `prior_knowledge` straight into causal-learn — verified, not assumed.

**Plan against the POST-#1030 file** at `/home/enunez/Projects/wt_causal_discovery_revamp/src/agents/causal_impact/nodes/graph_builder.py` (its `_run_discovery` is identical to the current one at lines 498-605; the #1030 diff is in `_construct_dag` + the deletion of `COMMON_CONFOUNDERS`, which this task does not touch).

**Files:**
- Modify: `src/agents/causal_impact/nodes/graph_builder.py` (`_run_discovery`, the guided-prior branch ~544-562 post-#1030)
- Test: `tests/unit/test_agents/test_causal_impact/test_graph_builder_priors.py` (create)

- [ ] **Step 1: Write the failing test (capture the DiscoveryConfig passed to the runner)**

```python
# tests/unit/test_agents/test_causal_impact/test_graph_builder_priors.py
"""Guided discovery seeds CausalPriorKnowledge.required_edges from the question's
MODELED confounders (confounder->treatment, confounder->outcome), replacing the
generic KNOWN_CAUSAL_RELATIONSHIPS constants that never match real covariates."""

import pandas as pd
import pytest

from src.agents.causal_impact.nodes.graph_builder import GraphBuilderNode
from src.causal_engine.discovery.base import DiscoveryResult
from src.causal_engine.discovery.gate import GateEvaluation, GateDecision


@pytest.mark.asyncio
async def test_run_discovery_seeds_confounder_edges_from_modeled_confounders(monkeypatch):
    node = GraphBuilderNode()

    df = pd.DataFrame(
        {
            "treatment_arm": [1.0, 0.0, 1.0, 0.0],
            "persistent_180d": [1.0, 0.0, 1.0, 1.0],
            "disease_severity": [2.0, 1.0, 3.0, 2.0],
            "academic_hcp": [1.0, 0.0, 1.0, 0.0],
            "geographic_region=south": [1.0, 0.0, 0.0, 1.0],
        }
    )

    captured: dict = {}

    async def _fake_discover_dag(*, data, config, session_id):
        captured["config"] = config
        return DiscoveryResult(success=True, config=config, ensemble_dag=None, edges=[])

    monkeypatch.setattr(node.discovery_runner, "discover_dag", _fake_discover_dag)
    monkeypatch.setattr(
        node.discovery_gate,
        "evaluate",
        lambda result, expected: GateEvaluation(
            decision=GateDecision.REVIEW, confidence=0.5, reasons=[]
        ),
    )

    state = {
        "data_cache": {"estimation_data": df},
        "discovery_guided": True,
        "modeled_confounders": ["disease_severity", "academic_hcp", "geographic_region=south"],
    }
    await node._run_discovery(state, "treatment_arm", "persistent_180d")

    prior = captured["config"].prior_knowledge
    assert prior is not None
    edges = set(prior.required_edges or [])
    # The estimand edge is still required.
    assert ("treatment_arm", "persistent_180d") in edges
    # Each modeled confounder -> treatment AND -> outcome is required.
    for conf in ("disease_severity", "academic_hcp", "geographic_region=south"):
        assert (conf, "treatment_arm") in edges
        assert (conf, "persistent_180d") in edges


@pytest.mark.asyncio
async def test_run_discovery_skips_modeled_confounders_absent_from_frame(monkeypatch):
    """A modeled confounder not present as a column is not forced as an edge
    (build_background_knowledge would ignore it; keep required_edges clean)."""
    node = GraphBuilderNode()
    df = pd.DataFrame(
        {
            "treatment_arm": [1.0, 0.0, 1.0],
            "persistent_180d": [1.0, 0.0, 1.0],
            "disease_severity": [2.0, 1.0, 3.0],
        }
    )
    captured: dict = {}

    async def _fake_discover_dag(*, data, config, session_id):
        captured["config"] = config
        return DiscoveryResult(success=True, config=config, ensemble_dag=None, edges=[])

    monkeypatch.setattr(node.discovery_runner, "discover_dag", _fake_discover_dag)
    monkeypatch.setattr(
        node.discovery_gate,
        "evaluate",
        lambda result, expected: GateEvaluation(
            decision=GateDecision.REVIEW, confidence=0.5, reasons=[]
        ),
    )
    state = {
        "data_cache": {"estimation_data": df},
        "discovery_guided": True,
        "modeled_confounders": ["disease_severity", "geographic_region=west"],
    }
    await node._run_discovery(state, "treatment_arm", "persistent_180d")
    edges = set(captured["config"].prior_knowledge.required_edges or [])
    assert ("disease_severity", "treatment_arm") in edges
    assert ("disease_severity", "persistent_180d") in edges
    # The absent confounder is NOT seeded as a required edge.
    assert ("geographic_region=west", "treatment_arm") not in edges
    assert ("geographic_region=west", "persistent_180d") not in edges
```

- [ ] **Step 2: Run it to confirm it fails**

Run: `pytest tests/unit/test_agents/test_causal_impact/test_graph_builder_priors.py -v`
Expected: FAIL — only `(treatment_arm, persistent_180d)` is in `required_edges`; the confounder edges are absent.

- [ ] **Step 3: Seed the confounder edges in `_run_discovery`**

In `src/agents/causal_impact/nodes/graph_builder.py`, in `_run_discovery`, replace the guided-prior construction block. The current block (post-#1030, ~lines 544-562) is:

```python
        guided = bool(state.get("discovery_guided", False))
        prior_knowledge: Optional[CausalPriorKnowledge] = None
        if guided and treatment in data.columns and outcome in data.columns:
            from src.repositories.provenance import PROVENANCE_DROP_COLS

            covariate_cols = [
                c
                for c in data.columns
                if c not in (treatment, outcome) and c not in PROVENANCE_DROP_COLS
            ]
            tiers = (
                [covariate_cols, [treatment], [outcome]]
                if covariate_cols
                else [[treatment], [outcome]]
            )
            prior_knowledge = CausalPriorKnowledge(
                tiers=tiers,
                required_edges=[(treatment, outcome)],
            )
            # Only PC consumes BackgroundKnowledge; restrict to it so the ensemble
            # is not polluted by unconstrained orientations from other algorithms.
            algorithms = [DiscoveryAlgorithmType.PC]
```

Replace it with (adds confounder->treatment / confounder->outcome required edges from `modeled_confounders`):

```python
        guided = bool(state.get("discovery_guided", False))
        prior_knowledge: Optional[CausalPriorKnowledge] = None
        if guided and treatment in data.columns and outcome in data.columns:
            from src.repositories.provenance import PROVENANCE_DROP_COLS

            covariate_cols = [
                c
                for c in data.columns
                if c not in (treatment, outcome) and c not in PROVENANCE_DROP_COLS
            ]
            tiers = (
                [covariate_cols, [treatment], [outcome]]
                if covariate_cols
                else [[treatment], [outcome]]
            )
            # Seed the question's MODELED confounders as REQUIRED edges so guided
            # PC anchors them as confounders (confounder->treatment AND
            # confounder->outcome) by construction, while the data still selects
            # any remaining structure. This replaces the generic
            # KNOWN_CAUSAL_RELATIONSHIPS / COMMON_CONFOUNDERS constants (~0 overlap
            # with real covariates) as the source of the discovery prior. Restrict
            # to confounders actually present as columns — build_background_knowledge
            # name-matches against the frame and would silently drop the rest, but
            # keeping required_edges clean keeps the prior honest. The estimand
            # edge (treatment->outcome) is always required.
            modeled = [
                c
                for c in (state.get("modeled_confounders") or [])
                if c in data.columns and c not in (treatment, outcome)
            ]
            required_edges: List[Tuple[str, str]] = [(treatment, outcome)]
            for conf in modeled:
                required_edges.append((conf, treatment))
                required_edges.append((conf, outcome))
            prior_knowledge = CausalPriorKnowledge(
                tiers=tiers,
                required_edges=required_edges,
            )
            # Only PC consumes BackgroundKnowledge; restrict to it so the ensemble
            # is not polluted by unconstrained orientations from other algorithms.
            algorithms = [DiscoveryAlgorithmType.PC]
```

(`List` and `Tuple` are already imported at the top of the file — line 15.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/test_agents/test_causal_impact/test_graph_builder_priors.py tests/unit/test_agents/test_causal_impact/test_graph_builder.py -v`
Expected: PASS (new prior tests + the existing graph_builder tests, which do not set `discovery_guided` so they are unaffected).

- [ ] **Step 5: Commit**

```bash
git add src/agents/causal_impact/nodes/graph_builder.py tests/unit/test_agents/test_causal_impact/test_graph_builder_priors.py
git commit -m "feat(causal): seed guided-discovery priors from the question's modeled confounders"
```

---

## Verification

- [ ] **Targeted suite:** `pytest tests/unit/test_api/test_causal_geo_encoding.py tests/unit/test_api/test_causal_agent_analyze.py tests/unit/test_agents/test_causal_impact/test_graph_builder_priors.py tests/unit/test_agents/test_causal_impact/test_graph_builder.py -v` — all green.
- [ ] **Regression (P1 contract intact):** `pytest tests/unit/test_api/test_causal_discover_effects.py tests/unit/test_repositories/test_causal_path.py -v` — still green (P1b adds `geographic_region` to the covariate allowlist, so `_discover_candidate_questions`'s `allowed_cov = set(spec["covariate"]) & numeric` is unchanged — `geographic_region` is NOT in `_CAUSAL_NUMERIC_COLUMNS`, so the SSOT enumeration still drops it from the *numeric* intersection there; the encoding is applied in the LOADER, the path P1's FWL pre-rank and the agent run both go through).
- [ ] **Lint:** `ruff check src/ && ruff format --check src/` — clean (a red Lint gate cascade-skips the backend test jobs in CI).
- [ ] **Types (scoped — do NOT run whole-tree mypy on the droplet):** `mypy src/api/routes/causal.py src/agents/causal_impact/nodes/graph_builder.py src/agents/causal_impact/state.py` — clean. CI's `Type Check (MyPy)` gate is the arbiter.
- [ ] **Faithful live run** (real backend, after P1's gated reseed; the encoding/prior do not require their own prod write): submit `POST /causal/discover-effects?brand=Kisqali`, poll to completion, then drill into the RETENTION row (`treatment_arm -> persistent_180d`) via `GET /causal/agent-analyze/{id}` and confirm:
  - the DAG `nodes` include `geographic_region=<level>` dummy node(s) AND `disease_severity` / `academic_hcp` (the full modeled retention set is now adjusted, not just the numeric subset),
  - the DAG `edges` include `disease_severity -> treatment_arm` / `disease_severity -> persistent_180d` and the geo-dummy confounder edges (guided priors fired),
  - a non-empty `adjustment_sets` containing the geo dummy,
  - a stable ATE (the retention adjustment set GREW from `{disease_severity, academic_hcp}` to additionally include the geo dummies, so a small ATE shift is EXPECTED and correct — the prior estimate was under-adjusted; it should not swing wildly nor go NaN).
- [ ] **Adversarial multi-lens review** before PR (this surface has repeatedly shipped CI-passing presentation/honesty bugs).

## Self-Review

- **Spec coverage:** §6 "Categorical-encode `geographic_region` so retention can adjust for it" → Tasks 1-3 (allowlist + helper + loader expansion + wiring). §5.2 step 4 "Seed agent priors — thread `confounders_controlled` into `CausalPriorKnowledge` required edges (confounder->treatment, confounder->outcome), bypassing the generic constants" → Tasks 4-5. P1's deferral note ("`geographic_region` one-hot encoding … + agent-prior seeding … depends on post-#1030 base") is exactly this plan.
- **Cheapest-disproof of the load-bearing assumptions (done during planning, not theorized):**
  1. *"One-hot dummies survive the executors + FWL screen."* `_adjusted_partial_corr` does `df[covariates].to_numpy(dtype=float)` (causal.py:977); 0/1 float dummies pass. Confirmed by reading the helper.
  2. *"required_edges actually reach the structure-learning algorithm."* `pc_wrapper.discover` (pc_wrapper.py:104-110) reads `config.prior_knowledge` and passes `build_background_knowledge(prior, node_names)` into causal-learn's PC with `node_names = list(data.columns)`; `build_background_knowledge` (background_knowledge.py:55-57) `add_required_by_node` for each in-frame edge. Confirmed by reading both — the prior is consumed, not dropped.
  3. *"geographic_region is unordered + populated."* Live probe: `midwest 6381 / south 6242 / northeast 6208 / west 6169`, 0 nulls — one-hot (not ordinal) is correct.
- **Placeholder scan:** none — every code/test step shows complete content grounded in real signatures (`_load_agent_estimation_frame`, `_run_discovery`, `CausalPriorKnowledge(tiers, required_edges)`, `_one_hot_categoricals`).
- **Type consistency:** `_one_hot_categoricals(df, List[str]) -> (DataFrame, List[str])`; loader returns the same `tuple[DataFrame, List[str]]` shape it already declares (line 1418); `modeled_confounders: NotRequired[List[str]]` matches `state.get("modeled_confounders") or []` consumption; `required_edges: List[Tuple[str, str]]` matches `CausalPriorKnowledge.required_edges: Optional[List[Tuple[str, str]]]` (base.py:111). `List`/`Tuple` already imported in graph_builder (line 15).
- **Security gate preserved:** the column allowlist (400 on off-allowlist), provenance filter, treatment/outcome usability drop, and numeric coercion are all retained in Task 2 (a dedicated test pins the 400). `geographic_region` enters the allowlist as a covariate ONLY and is never offered as a treatment/outcome.
- **No regression to P1:** P1's `_discover_candidate_questions` intersects the modeled set with the NUMERIC allowlist (`& numeric`), so adding `geographic_region` to `covariate` (but NOT to `_CAUSAL_NUMERIC_COLUMNS`) leaves that enumeration unchanged; the geo covariate is realized only in the loader, which both the FWL pre-rank and the agent run traverse.
