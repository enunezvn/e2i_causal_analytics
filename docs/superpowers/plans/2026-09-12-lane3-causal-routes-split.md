# Lane 3 — `routes/causal.py` split by concern + module-size ratchet Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the 6,601-line `src/api/routes/causal.py` into the package `src/api/routes/causal/` with one module per concern and no behaviour change (OpenAPI byte-identical), and add a CI ratchet that fails any `src/` file over 1,500 lines unless pinned at its current size.

**Architecture:** Code moves, it is not edited: each cluster's routes and private helpers are cut from `causal.py` by line range into its owning module; shared layers (`_common`, `datasets`, `loaders`) sit below the route modules so imports run one way; `__init__.py` composes one `router` so `main.py` is untouched. Tests that patch symbols move to patching the owner. The ratchet is a meta test with a measured allowlist that can only shrink.

**Tech Stack:** FastAPI `APIRouter.include_router`, Python packages, pytest `-n 0`, `scripts/export_openapi.py`, `openapi-typescript` (verify-types gate), `git mv` is NOT used (a file becomes a package; history is preserved by `git log --follow` on the new modules via similarity detection).

Spec: `docs/superpowers/specs/2026-09-12-debts-3-4-wave-design.md` §6. Worktree `.worktrees/lane-split`, branch `claude/1991-causal-routes-split`, ONE push at the end. Runs AFTER lanes 1 and 2 are merged.

```bash
cd /home/enunez/Projects/e2i_causal_analytics && git fetch origin
git worktree add -b claude/1991-causal-routes-split .worktrees/lane-split origin/main && cd .worktrees/lane-split
```

Line numbers below are from the 2026-09-12 map of `causal.py` at main `2b43ee85e`; re-measure with `grep -n` before cutting — lanes 1/2 may have shifted them by a few lines.

---

## File map (target)

| module | takes from `causal.py` (line ranges at `2b43ee85e`) |
|---|---|
| `causal/_common.py` | 161–237 constants (`CAUSAL_COMPLETED_EVENT_TYPE`, `_AGENT_HARD_TIMEOUT_S`, `_REFUTATION_COMPUTE_BUDGET_S`, `_GENERIC_500_DETAIL`, `_ROBUSTNESS_*`, `_NON_DAG_STRUCTURAL_WARNING`, `_CYCLE_IRRELEVANT_WARNING`), 271 `_CAUSAL_JOB_TTL_SECONDS`, 3756 `_opt_float`, 5905 `_parse_occurred_at`, 5923 `_as_float`, 6035 `_as_optional_float`, 5074 `_dowhy_interval`, 6391 `_te_pvalue_from_z`, 4247–4270 `_NO_REAL_DATA_BACKEND_DETAIL` / `_NO_RESOLVABLE_DATA_DETAIL` / `_DATA_REQUIRED_LIBRARIES` |
| `causal/datasets.py` | 137–144 `column_labels` re-exports (`_COLUMN_DEFINITIONS`, `_COLUMN_LABELS`, `_column_label`), 826–1393 the dataset registry block (`_CAUSAL_DATASET_SPECS` … `_CAUSAL_CATEGORICAL_COLUMNS`, all `_derive_*`), 1394 `_list_dataset_brands`, 1047 `_is_randomized_treatment`, 1119–1160 negative-control map + `_negative_control_outcome` |
| `causal/loaders.py` | 6228–6290 `_TE_PAGE_SIZE`, `_TE_MAX_PAGES`, `_te_paged_select` (moved DOWN from treatment-effects), 2645–3331 (`_coerce_estimation_row`, `_load_hcp_profile_centrality`, `_te_paged_select_all_brands`, `_require_covariate_role`, `_load_hcp_adoption_join_frame`, `_one_hot_categoricals`, `_resolve_requested_baselines`, `_NBA_BASELINE_CATEGORICALS`, `_NBA_JOIN_MAX_PAGES`, `_load_trigger_question_rows`, `_load_patient_baseline_rows`, `_load_nba_triggers_join_frame`, `_load_agent_estimation_frame`), 1767 `_get_causal_path_repo` |
| `causal/catalog.py` | 1430–1715 (`/brands`, `/variables`, `/propose-questions` + `_adjusted_partial_corr`), 2445–2514 `/clinical-context`, 2516–2638 `/estimation-data`, 1755–1757 clinical-context service (made lazy) |
| `causal/discovery.py` | 1717–1754 store + markers, 1760 `_CandidateQuestion`, 1607–1638 `_prerank_signal` / `_prerank_questions`, 1778–2288 helpers + `_run_discover_effects_task`, 2290–2443 the four routes |
| `causal/agent.py` | 273–276 `_agent_analysis_store`, 3333–4114 routes, task, MLflow, `_refutation_tests_from_state`, `_estimator_comparison_from_estimation`, `_agent_state_to_response` |
| `causal/pipelines.py` | 255–256 `_pipeline_cache`, `_validation_cache` (254 `_analysis_cache` goes to hierarchical); 4116–4245 sequential route + task; 4278–5253 wiring block; 5255–5458 parallel route + `_run_library_analysis`; 5460–5482 status; 5484–5578 validate |
| `causal/hierarchical.py` | 254 `_analysis_cache`, 278–692, plus 4309 `_resolve_hierarchical_dataframe` |
| `causal/activity.py` | 171–172 activity cache, 5580–5753 estimator registry + `/estimators`, 5755–5941 `/health` + activity readers, 5943–6000 `/history`, 6002–6219 value chains, 6221–6601 treatment effects (minus what moved to `_common`/`loaders`) |
| `causal/__init__.py` | the aggregator |

Owner map for the 23 patched symbols (tests patch the OWNER after the split):
`_list_dataset_brands` → datasets · `_load_agent_estimation_frame`, `_load_patient_baseline_rows`, `_load_trigger_question_rows`, `_load_hcp_profile_centrality`, `_te_paged_select`, `_get_causal_path_repo` → loaders · `_run_agent_analysis_task`, `_agent_analysis_store` → agent · `_discover_effects_store`, `_prerank_questions`, `_prerank_signal`, `_attach_clinical_context`, `_discover_candidate_questions`, `_run_discover_effects_task`, `_DISCOVERY_HEARTBEAT_TTL_SECONDS`, `_DISCOVERY_HEARTBEAT_INTERVAL_SECONDS` → discovery · `_run_real_sequential_pipeline`, `_run_real_parallel_pipeline` → pipelines · `_execute_hierarchical_analysis` → hierarchical · `get_recent_memories`, `count_memories_by_type`, `get_async_supabase_client`, `apply_provenance_filter` → imported by name at module level in `activity.py` (health/history) AND `loaders.py` (supabase client) — patch whichever module the test's route reads through.

---

### Task 1: The ratchet test, red against `causal.py`

**Files:**
- Create: `tests/unit/test_tests_meta/test_module_size_ratchet.py`

- [ ] **Step 1: Measure the allowlist**

Run: `find src -name "*.py" | xargs wc -l | awk '$1>1500 && $2!="total"{print $2, $1}' | sort > /tmp/over1500.txt; cat /tmp/over1500.txt`
Expected (2026-09-12): 32 lines, `src/api/routes/causal.py 6601` among them.

- [ ] **Step 2: Write the test with every measured file EXCEPT `causal.py` pinned**

```python
"""Module-size ratchet (#1991 debt 4): no file under src/ may exceed LIMIT lines unless it is
pinned here at its current size, and a pinned file may only shrink.

Why a ratchet and not a hard cap: 31 files already exceed the limit (measured 2026-09-12).
A hard cap would either fail forever or exempt them forever. Pins can only move DOWN:
the test fails if a pinned file grows past its pin AND if a pin is above the file's
actual size (so the number on record is always the real one). Delete a pin when the
file drops under LIMIT.
"""
from __future__ import annotations

from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
SRC = REPO / "src"
LIMIT = 1500

# path (relative to repo) -> pinned line count. Measured, not guessed. Only shrinks.
ALLOWLIST: dict[str, int] = {
    # paste /tmp/over1500.txt here, one entry per line, EXCLUDING src/api/routes/causal.py, e.g.
    "src/data/causal_role_classifier.py": 6525,
    "src/api/routes/copilotkit.py": 6149,
    "src/agents/tool_composer/tool_registrations.py": 4511,
    # ... (31 entries total)
}


def _lines(p: Path) -> int:
    with p.open("rb") as fh:
        return sum(1 for _ in fh)


def _all_py() -> list[Path]:
    return sorted(p for p in SRC.rglob("*.py") if "__pycache__" not in p.parts)


def test_no_unpinned_file_exceeds_limit():
    offenders = []
    for p in _all_py():
        n = _lines(p)
        rel = p.relative_to(REPO).as_posix()
        if n > LIMIT and rel not in ALLOWLIST:
            offenders.append(f"{rel}: {n} > {LIMIT}")
    assert not offenders, "split these by concern or pin them (pins only shrink):\n" + "\n".join(offenders)


@pytest.mark.parametrize("rel,pin", sorted(ALLOWLIST.items()))
def test_pinned_file_has_not_grown_and_pin_is_current(rel, pin):
    p = REPO / rel
    assert p.exists(), f"{rel} is pinned but missing — delete its pin"
    n = _lines(p)
    assert n <= pin, f"{rel} grew to {n} lines (pin {pin}); shrink it, do not raise the pin"
    assert n == pin, f"{rel} is {n} lines but pinned at {pin}; lower the pin to {n}"
    assert n > LIMIT, f"{rel} is under {LIMIT}; delete its pin"
```
Fill `ALLOWLIST` from the measurement (31 entries).

- [ ] **Step 3: Run it to verify it fails on `causal.py`**

Run: `.venv/bin/pytest tests/unit/test_tests_meta/test_module_size_ratchet.py -n 0 -q`
Expected: 1 failed (`test_no_unpinned_file_exceeds_limit` names `src/api/routes/causal.py: 6601 > 1500`), 31 passed. Confirm the directory is CI-collected: `grep -n "tests/unit/test_tests_meta" .github/workflows/backend-tests.yml` → listed.

- [ ] **Step 4: Commit the red test**

```bash
git add tests/unit/test_tests_meta/test_module_size_ratchet.py
git commit -m "test(meta): module-size ratchet at 1,500 lines with a measured allowlist — red on routes/causal.py (#1991 debt 4)"
```

---

### Task 2: The OpenAPI contract snapshot

**Files:**
- Create: `tests/unit/test_api/fixtures/causal_openapi_paths.json`, `tests/unit/test_api/test_causal_openapi_unchanged_1991.py`

- [ ] **Step 1: Capture the baseline from main BEFORE any move**

```bash
.venv/bin/python -m scripts.export_openapi --output /tmp/openapi_main.json
.venv/bin/python - <<'EOF'
import json
spec = json.load(open("/tmp/openapi_main.json"))
paths = {k: v for k, v in spec["paths"].items() if k.startswith("/api/causal")}
json.dump(paths, open("tests/unit/test_api/fixtures/causal_openapi_paths.json", "w"), indent=2, sort_keys=True)
print(len(paths), "paths")
EOF
```
Expected: `23 paths`.

- [ ] **Step 2: Write the test**

```python
"""The routes split is a MOVE, not an edit: every /api/causal path, method, operationId,
parameters, request body and responses are byte-identical to the pre-split baseline."""
from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("E2I_TESTING_MODE", "true")

FIXTURE = Path(__file__).parent / "fixtures" / "causal_openapi_paths.json"


def test_causal_openapi_paths_unchanged():
    from src.api.main import app

    spec = app.openapi()
    now = {k: v for k, v in spec["paths"].items() if k.startswith("/api/causal")}
    baseline = json.loads(FIXTURE.read_text())
    assert sorted(now) == sorted(baseline)
    for path in baseline:
        assert json.dumps(now[path], sort_keys=True) == json.dumps(baseline[path], sort_keys=True), path
```

- [ ] **Step 3: Run it (green on main), commit**

Run: `.venv/bin/pytest tests/unit/test_api/test_causal_openapi_unchanged_1991.py -n 0 -q` → 1 passed.
```bash
git add tests/unit/test_api/fixtures/causal_openapi_paths.json tests/unit/test_api/test_causal_openapi_unchanged_1991.py
git commit -m "test(api): pin the /api/causal OpenAPI contract before the split (#1991 debt 4)"
```

---

### Task 3: Create the package skeleton and the three shared layers

**Files:**
- Create: `src/api/routes/causal/__init__.py`, `_common.py`, `datasets.py`, `loaders.py`
- Modify: `src/api/routes/causal.py` → becomes `src/api/routes/causal/_legacy.py` temporarily (so imports keep resolving while modules are carved out)

- [ ] **Step 1: Move the file into the package as `_legacy.py` and make the aggregator re-export it**

```bash
mkdir -p src/api/routes/causal && git mv src/api/routes/causal.py src/api/routes/causal/_legacy.py
cat > src/api/routes/causal/__init__.py <<'EOF'
"""Causal inference routes (/api/causal/*), split by concern (#1991 debt 4).

Transitional: while modules are carved out of ``_legacy``, this package re-exports
``_legacy``'s public surface so nothing breaks between commits. The final commit of
the lane deletes ``_legacy`` and this file composes the sub-routers.
"""
from ._legacy import *  # noqa: F401,F403
from ._legacy import router  # noqa: F401
EOF
```
`import *` skips underscore names, so add the explicit re-export list the tests/`segments.py` need until the final step: append to `__init__.py`:
```python
from ._legacy import (  # noqa: F401  transitional private re-exports
    _CAUSAL_DATASET_SPECS, _brand_scoped_covariates, _get_causal_path_repo, _COLUMN_DEFINITIONS,
    _column_label, _list_dataset_brands, _CAUSAL_NUMERIC_COLUMNS, _CAUSAL_NUMERIC_DERIVATIONS,
    _coerce_estimation_row, _agent_state_to_response, _COLUMN_LABELS, _refutation_tests_from_state,
    _CAUSAL_NEGATIVE_CONTROL_OUTCOMES, _derive_is_prior_c5, list_causal_variables,
)
```
Run: `.venv/bin/pytest tests/unit/test_api/test_routes/test_causal.py tests/unit/test_api/test_causal_openapi_unchanged_1991.py -n 0 -q` → pass (nothing moved yet).
Commit: `git commit -am "refactor(causal): routes/causal.py becomes package causal/ with a transitional _legacy module"`.

- [ ] **Step 2: Carve `_common.py`**

Create `src/api/routes/causal/_common.py` with a module docstring `"""Shared constants and small numeric/date helpers for the causal routes package."""`, the needed imports (`logging`, `math`, `datetime`, `Optional`, `Any`, `Tuple`; `user_safe_503_detail` from wherever `_legacy` imports it), and CUT (not copy) from `_legacy.py` the symbols listed in the file map row for `_common`. In `_legacy.py` add `from ._common import (…every moved name…)  # noqa: F401`. Keep each moved function byte-identical.
Run: `.venv/bin/pytest tests/unit/test_api/test_routes/test_causal.py tests/api/test_causal_value_chains.py tests/unit/test_api/test_causal_dowhy_se_interval_2014.py -n 0 -q` → pass.
Commit: `git commit -am "refactor(causal): extract _common (constants, numeric/date helpers)"`.

- [ ] **Step 3: Carve `datasets.py`**

Same procedure for the `datasets` row (registry block 826–1393, the `column_labels` re-exports, `_list_dataset_brands`, `_is_randomized_treatment`, negative-control map). `datasets.py` imports only `_common` and non-package modules. `_legacy.py` imports the moved names from `.datasets`.
Run: `.venv/bin/pytest tests/unit/test_api/test_brand_scoped_covariates.py tests/unit/test_api/test_causal_brands.py tests/unit/test_api/test_causal_covariate_roles.py tests/unit/test_kpi/test_causal_datasets_copay.py tests/unit/test_api/test_routes/test_segments.py -n 0 -q` → pass (they still import via the package `__init__`).
Commit: `git commit -am "refactor(causal): extract datasets (registry, brand scoping, derivations)"`.

- [ ] **Step 4: Carve `loaders.py`**

Move the `loaders` row INCLUDING `_TE_PAGE_SIZE`, `_TE_MAX_PAGES`, `_te_paged_select` from the treatment-effects block (this removes the backward dependency), `_get_causal_path_repo`, and import `get_async_supabase_client` BY NAME at module level (`from src.memory.services.factories import get_async_supabase_client` — copy the exact import line from `_legacy.py:155`) so tests can `monkeypatch.setattr(loaders, "get_async_supabase_client", …)`.
Run: `.venv/bin/pytest tests/unit/test_api/test_causal_loaders.py tests/unit/test_api/test_causal_hcp_adoption.py tests/unit/test_api/test_causal_nba_backdoor.py tests/unit/test_api/test_causal_nba_baselines.py tests/unit/test_api/test_causal_triggers_dataset.py tests/unit/test_api/test_causal_geo_encoding.py -n 0 -q` → some FAIL on attribute patches (`monkeypatch.setattr(causal_routes, "_load_agent_estimation_frame", …)` now patches the package re-export, not the owner). Fix those tests in this step: replace the patched target with `from src.api.routes.causal import loaders` … `monkeypatch.setattr(loaders, "…", …)`. Re-run → pass.
Commit: `git commit -am "refactor(causal): extract loaders (frames, paging, supabase seam); tests patch the owner"`.

---

### Task 4: Carve the route modules and compose the router

**Files:** `catalog.py`, `discovery.py`, `agent.py`, `pipelines.py`, `hierarchical.py`, `activity.py`, `__init__.py`; delete `_legacy.py`

Each route module starts with:
```python
from fastapi import APIRouter
router = APIRouter()
```
and its `@router.get/post(...)` decorators are moved verbatim (paths stay relative, e.g. `"/brands"`; the `/causal` prefix, tags and shared `responses` live on the aggregator). Explicit `operation_id=` values move with their routes.

- [ ] **Step 1: `hierarchical.py`** (278–692 + `_analysis_cache` + `_resolve_hierarchical_dataframe`). Run `tests/api/test_hierarchical_defab.py tests/api/test_causal_endpoints.py -n 0 -q`; fix the string patch `"src.api.routes.causal._execute_hierarchical_analysis"` → `"src.api.routes.causal.hierarchical._execute_hierarchical_analysis"`. Commit.
- [ ] **Step 2: `catalog.py`** (brands, variables, propose-questions, clinical context, estimation data). Make the clinical-context service lazy:
```python
_clinical_context_service: Optional["ClinicalContextService"] = None

def _get_clinical_context_service() -> "ClinicalContextService":
    """Built on first use, not at import (#1991 debt 4): the constructor builds four HTTP clients."""
    global _clinical_context_service
    if _clinical_context_service is None:
        from src.services.clinical_context import ClinicalContextService
        _clinical_context_service = ClinicalContextService()
    return _clinical_context_service
```
and replace the two uses (`catalog` `/clinical-context`, `discovery._attach_clinical_context`) with the accessor. Tests that patch `_clinical_context_service.get_context` on the instance switch to `monkeypatch.setattr(catalog, "_get_clinical_context_service", lambda: fake)`. Run `tests/unit/test_api/test_causal_clinical_context.py tests/unit/test_api/test_causal_leaderboard_clinical_context.py tests/unit/test_api/test_causal_propose.py tests/unit/test_api/test_causal_brands.py tests/unit/test_causal_engine/test_pipeline/test_first_class_estimation_data.py -n 0 -q`. Commit.
- [ ] **Step 3: `agent.py`** (3333–4114 + `_agent_analysis_store`). Keep the refutation-config override at its line. Run `tests/unit/test_api/test_causal_agent_analyze*.py tests/unit/test_agents/test_tool_composer/test_cross_surface_consistency_2014.py -n 0 -q`; update `_agent_state_to_response` / `_refutation_tests_from_state` imports in those 4 test files to `from src.api.routes.causal.agent import …`. Commit.
- [ ] **Step 4: `discovery.py`** (store, markers, heartbeat, prerank, candidate discovery, ranking, task, 4 routes). It imports `_run_agent_analysis_task` and `_agent_analysis_store` from `.agent`. Run `tests/unit/test_api/test_causal_discover_effects*.py -n 0 -q`; move the attribute patches (`_discover_effects_store`, `_prerank_*`, `_attach_clinical_context`, `_discover_candidate_questions`, `_run_discover_effects_task`, heartbeat constants) to `discovery`; `_run_agent_analysis_task` patches → `agent` (the discovery task must call it THROUGH the `agent` module namespace: `from . import agent as _agent` and `_agent._run_agent_analysis_task(...)`, so a patch on `agent` takes effect). Commit.
- [ ] **Step 5: `pipelines.py`** (sequential/parallel/status/validate + the wiring block). `_SurfaceCSequentialPipeline` and `_dowhy_interval`'s callers in treatment-effects import from `.pipelines` / `._common`. Run `tests/api/test_causal_pipeline_*.py tests/api/test_pipeline_graph_quality_surface.py tests/unit/test_api/test_routes/test_causal_heavy_compute_bound.py -n 0 -q`; patches for `_run_real_sequential_pipeline` / `_run_real_parallel_pipeline` → `pipelines`. Commit.
- [ ] **Step 6: `activity.py`** (estimators, health, history, value chains, treatment effects). Import `get_recent_memories`, `count_memories_by_type`, `apply_provenance_filter` by name at module level (copy the exact lines from `_legacy.py:152–156`). Run `tests/api/test_causal_value_chains.py tests/unit/test_api/test_routes/test_causal.py tests/unit/test_security/test_sentinel_external_unreachable.py -n 0 -q`; patches → `activity`. Commit.
- [ ] **Step 7: Compose the router, delete `_legacy.py`**

`src/api/routes/causal/__init__.py` becomes:
```python
"""Causal inference routes (/api/causal/*), one module per concern (#1991 debt 4).

Import direction is one-way: _common <- datasets <- loaders <- route modules; discovery
imports agent (the job fans out over the agent task) and activity imports pipelines (the
treatment-effects estimator). Nothing imports upward.
"""
from fastapi import APIRouter

from src.api.schemas.common import ErrorResponse, ValidationErrorResponse

from . import activity, agent, catalog, discovery, hierarchical, pipelines

router = APIRouter(
    prefix="/causal",
    tags=["Causal Inference"],
    responses={
        401: {"model": ErrorResponse, "description": "Unauthorized"},
        422: {"model": ValidationErrorResponse, "description": "Validation Error"},
        500: {"model": ErrorResponse, "description": "Internal Server Error"},
    },
)
for _sub in (hierarchical, catalog, discovery, agent, pipelines, activity):
    router.include_router(_sub.router)

__all__ = ["router"]
```
(copy the exact `responses` dict from the old `router = APIRouter(...)` at `_legacy.py:239–247` — descriptions must match byte for byte). Route ORDER matters for path matching: keep sub-router inclusion in the same relative order the routes had in the file (`/pipeline/{pipeline_id}` after `/pipeline/sequential` etc. is within `pipelines`, preserved by keeping decorator order inside each module). Then `git rm src/api/routes/causal/_legacy.py`.
Run: `.venv/bin/pytest tests/unit/test_api/test_causal_openapi_unchanged_1991.py tests/unit/test_tests_meta/test_module_size_ratchet.py -n 0 -q` → both pass (ratchet green: no module over 1,500 — if `pipelines.py` lands over 1,500 because the wiring block is ~980 lines plus routes, move `_sequential_output_to_response`/`_parallel_output_to_response`/`_build_stage_result_from_output` into `pipelines_mapping.py` and import them).
Commit: `git commit -am "refactor(causal): compose the package router; delete the transitional module"`.

---

### Task 5: Consumers outside the package

**Files:** `src/api/routes/segments.py` (8 import sites), `scripts/calibration/reband_sensitivity_readings.py` (2 sites), `tests/unit/test_utils/test_query_log_redaction_guard.py` (allowlist), every test that imports a private symbol from `src.api.routes.causal`

- [ ] **Step 1: Point `segments.py` and the script at the owners**

Replace each `from src.api.routes.causal import (...)` with owner imports: `_CAUSAL_DATASET_SPECS`, `_brand_scoped_covariates`, `_COLUMN_DEFINITIONS`, `_column_label`, `_list_dataset_brands`, `_CAUSAL_NUMERIC_COLUMNS`, `_CAUSAL_NUMERIC_DERIVATIONS`, `_derive_is_accepted` → `from src.api.routes.causal.datasets import …`; `_get_causal_path_repo`, `_coerce_estimation_row`, `_load_agent_estimation_frame`, `_load_patient_baseline_rows`, `_load_trigger_question_rows` → `from src.api.routes.causal.loaders import …`.

- [ ] **Step 2: Redaction guard**

In `tests/unit/test_utils/test_query_log_redaction_guard.py` replace `"src/api/routes/causal.py",` with the package's modules that carry query text (measure: `grep -ln "query" src/api/routes/causal/*.py`) — list each file explicitly, one per line.

- [ ] **Step 3: Sweep the remaining test imports**

Run: `grep -rn "from src.api.routes.causal import" tests | grep -v "import router"` and rewrite each to the owner per the file map (`_agent_state_to_response` → `.agent`, `_CAUSAL_DATASET_SPECS` → `.datasets`, etc.). Keep `from src.api.routes.causal import router` as is.

- [ ] **Step 4: Run the 42 files**

Run: `.venv/bin/pytest $(grep -rl "src.api.routes.causal" tests | tr '\n' ' ') tests/unit/test_utils/test_query_log_redaction_guard.py -n 0 -q -p no:cacheprovider`
Expected: all pass. Commit: `git commit -am "refactor(causal): consumers import from the owning modules (#1991 debt 4)"`.

---

### Task 6: Import side effect and the lazy service

**Files:** `tests/unit/test_api/test_causal_import_is_pure_1991.py`

- [ ] **Step 1: Write the test**

```python
"""Importing the causal routes package builds no external HTTP clients (#1991 debt 4)."""
import importlib
import sys
from unittest.mock import patch


def test_import_does_not_construct_clinical_context_service():
    for m in [k for k in sys.modules if k.startswith("src.api.routes.causal")]:
        del sys.modules[m]
    with patch("src.services.clinical_context.service.ClinicalContextService.__init__", side_effect=AssertionError("built at import")):
        importlib.import_module("src.api.routes.causal")


def test_accessor_builds_once():
    from src.api.routes.causal import catalog

    catalog._clinical_context_service = None
    a = catalog._get_clinical_context_service()
    b = catalog._get_clinical_context_service()
    assert a is b
```
Run → pass (Task 4 step 2 made it lazy). Commit.

---

### Task 7: Lane close

- [ ] **Step 1:** ruff `--no-cache` check + format on `src/api/routes/causal/`, `src/api/routes/segments.py`, `scripts/calibration/reband_sensitivity_readings.py`, touched tests; scoped mypy on `src/api/routes/causal/*.py`.
- [ ] **Step 2:** `.venv/bin/pytest tests/unit/test_api tests/api tests/unit/test_tests_meta tests/unit/test_kpi tests/unit/test_security tests/unit/test_utils -n 0 -q -p no:cacheprovider` → pass.
- [ ] **Step 3:** Contract: `.venv/bin/python -m scripts.export_openapi --output /tmp/openapi_split.json && diff <(python -c "import json;print(json.dumps(json.load(open('/tmp/openapi_main.json')),sort_keys=True,indent=1))") <(python -c "import json;print(json.dumps(json.load(open('/tmp/openapi_split.json')),sort_keys=True,indent=1))")` → empty; `cd frontend && npx openapi-typescript ../openapi_split.json -o /tmp/api.ts && diff -q /tmp/api.ts src/types/generated/api.ts && cd ..` → identical.
- [ ] **Step 4:** codex read-only round with the mandatory pushback paragraph; iterate to `VERDICT: ACCEPT`.
- [ ] **Step 5:** ONE push; PR `refactor(causal): split routes/causal.py by concern; module-size ratchet (#1991 debt 4)`, body lists the module map, the OpenAPI diff (empty), the ratchet allowlist size (31), and the patch-site moves.
- [ ] **Step 6:** Cert after deploy (spec §6): OpenAPI from the pre-lane and post-lane images diffed (fetch `/openapi.json` from the API in the scratch container, or `docker exec e2i_api python -m scripts.export_openapi` — read-only) → identical; `api.ts` byte-identical; one authenticated smoke call per module (`/api/causal/brands`, `/discover-effects/questions`, an `/agent-analyze/{id}` read, `/pipeline/{id}` read, `/health`, `/history`, `/value-chains`, `/treatment-effects`) returning the same status codes and top-level keys as before the flip. Record in `docs/demos/results/2026-09-12_lane_split/cert.md`; comment #1991.
