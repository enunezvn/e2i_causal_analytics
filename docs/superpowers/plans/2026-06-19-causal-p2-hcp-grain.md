# P2 — HCP-Grain SSOT-Derived Leaderboard (JOIN Loader + causal_paths HCP Edges) Implementation Plan

> For agentic workers: REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (- [ ]) syntax. Execute in an isolated git worktree; TDD red-first; commit per task; drive to a green fixed point and use codex:codex-rescue when stuck.

**Goal:** Add the HCP unit-of-analysis to the causal leaderboard — a new `hcp_adoption` dataset (`hcp_brand_adoption ⋈ hcp_profiles` on `hcp_id`) with two questions per brand (`peer_influence_score → adopted` with an EMPTY backdoor, and `treatment_arm → adopted` adjusting `{centrality_z}`), enumerated from the same `causal_paths` SSOT as the patient grain.

**Architecture:** Add 6 HCP edges (2 questions × 3 brands) to the `causal_paths` generator so P1's `CausalPathRepository.get_distinct_questions()` enumerates them identically to patient edges; add `hcp_adoption` to `_CAUSAL_DATASET_SPECS` + `_CAUSAL_NUMERIC_COLUMNS`; add a JOIN-aware branch to `_load_agent_estimation_frame` (and `_list_dataset_brands`) that reuses the existing, proven two-reads-plus-pandas-merge HCP loader pattern (`_te_paged_select` + merge from the treatment-effects surface) and preserves the column-allowlist + numeric-coercion security gate; and scope P1's `_discover_candidate_questions` by grain so each dataset only enumerates its own questions.

**Tech Stack:** Python 3.12, FastAPI, pandas/numpy, Supabase (PostgREST async client), pytest. No new deps.

**Scope (this plan):** HCP grain only; backend only. **Out of scope → other plans:** P0 (unified FE page), P1 (patient grain — MERGES FIRST, provides the SSOT-derivation machinery this plan reuses), P3 (trigger grain), enrichment. The agent run path, refutation, DAG construction, and #1030's estimator-comparison panel are dataset-agnostic and need no change — once the JOIN loader hands the agent a frame, P1's `_run_discover_effects_task` + `_effect_from_agent_response` + the agent graph handle the HCP grain unchanged.

**Sequencing (HARD dependency):** This plan **builds on P1 merged** and **#1030 merged**. P1 introduces `_CandidateQuestion`, `_get_causal_path_repo`, `_discover_candidate_questions(dataset, brand)`, `_prerank_questions`, `CausalPathRepository.get_distinct_questions()`, and adds `brand` + `adjustment_set` to `DiscoveredEffect`; it also decouples the generator's patient brand×outcome loop and rewrites `tests/unit/test_api/test_causal_discover_effects.py`. #1030 adds `summary` to `DiscoveredEffect` and `estimator_comparison` to `AgentCausalAnalysisResponse`, and adds a `summary=` kwarg inside `_effect_from_agent_response` — none of which this plan touches. Plan against the post-P1, post-#1030 state. (Read the #1030 branch worktree at `/home/enunez/Projects/wt_causal_discovery_revamp` for the merged schema/route shape.)

---

## File Structure

| File | Create/Modify | Responsibility |
|---|---|---|
| `src/ml/synthetic/generators/causal_paths_generator.py` | Modify | Emit 6 HCP edges (`peer_influence_score→adopted` ∅-backdoor; `treatment_arm→adopted` adj `{centrality_z}`) × 3 brands, additive after the patient loop, so the SSOT enumerates HCP questions the same way as patient. |
| `tests/unit/test_synthetic/test_causal_paths_generator.py` | Modify | Add HCP-edge coverage; scope the existing universal `start_node == "treatment_arm"` assert to patient rows (HCP rows start at `peer_influence_score`). |
| `src/api/routes/causal.py` | Modify | Add `hcp_adoption` to `_CAUSAL_DATASET_SPECS` + `_CAUSAL_NUMERIC_COLUMNS`; add `_load_hcp_adoption_join_frame()` and branch `_load_agent_estimation_frame` for it (preserve the allowlist/coercion gate, derive `centrality_z`); make `_list_dataset_brands` JOIN-aware; scope `_discover_candidate_questions` by grain. |
| `tests/unit/test_api/test_causal_hcp_adoption.py` | Create | Unit coverage for the HCP spec, the JOIN frame builder (mocked client), the allowlist gate on the JOIN path, and the grain-scoped enumeration. |

---

### Task 1: Emit HCP edges into the `causal_paths` SSOT generator

The leaderboard derives its questions from `causal_paths` (P1). Today the generator emits ONLY patient edges (`treatment_arm → {treatment_initiated, persistent_180d, discontinued_180d}`). Add the two HCP questions per brand so they enumerate identically. The HCP edges are ADDITIVE (a fixed 6-row block, independent of `n_records`) so the SSOT always carries all of them regardless of the patient `n_records` knob.

**Reconciliation with P1:** P1's Task 1 rewrites the patient loop body (decoupling brand×outcome). This task appends a NEW block AFTER that loop, just before `return pd.DataFrame(rows)`. It does not touch the patient loop, so it composes with P1 cleanly. The HCP block reuses the same row-dict shape (every NOT-NULL column populated). `mediators_identified` stays non-empty for HCP rows too (the existing test asserts `len(m) >= 1` for EVERY row) — the HCP edges carry a single mediator that is NOT the treatment or an outcome, so `_clean_causal_chains` does not drop a node and `causal_chain.nodes == [start, end]` stays a clean 2-hop path.

**Files:**
- Modify: `src/ml/synthetic/generators/causal_paths_generator.py` (add `_HCP_QUESTIONS` constant near line 38; append the HCP block before `return pd.DataFrame(rows)` at line 93)
- Test: `tests/unit/test_synthetic/test_causal_paths_generator.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/test_synthetic/test_causal_paths_generator.py  (append)
def test_hcp_adoption_edges_emitted_per_brand():
    """The SSOT must carry BOTH HCP questions for EVERY brand so the HCP-grain
    leaderboard enumerates them the same way as patient edges:
      peer_influence_score -> adopted (EMPTY backdoor, exogenous root)
      treatment_arm        -> adopted (adjust {centrality_z})."""
    df = CausalPathsGenerator(GeneratorConfig(seed=5, n_records=12)).generate()
    hcp = df[df["end_node"] == "adopted"]
    cells = set(zip(hcp["start_node"], hcp["brand"]))
    brands = {"Remibrutinib", "Kisqali", "Fabhalta"}
    assert cells == {(s, b) for s in ("peer_influence_score", "treatment_arm") for b in brands}


def test_hcp_adoption_confounder_sets_are_modeled():
    """peer_influence_score is exogenous (EMPTY backdoor); treatment_arm adjusts
    for centrality_z. These are the SSOT adjustment sets the loader will honor."""
    df = CausalPathsGenerator(GeneratorConfig(seed=5, n_records=12)).generate()
    exo = df[(df["start_node"] == "peer_influence_score") & (df["end_node"] == "adopted")].iloc[0]
    assert list(exo["confounders_controlled"]) == []
    rep = df[(df["start_node"] == "treatment_arm") & (df["end_node"] == "adopted")].iloc[0]
    assert list(rep["confounders_controlled"]) == ["centrality_z"]


def test_hcp_adoption_chain_is_clean_two_hop():
    """HCP chains terminate at adopted with a non-empty mediator list (existing
    invariant) and causal_chain.nodes starts at the treatment, ends at adopted."""
    df = CausalPathsGenerator(GeneratorConfig(seed=5, n_records=12)).generate()
    for _, row in df[df["end_node"] == "adopted"].iterrows():
        nodes = row["causal_chain"]["nodes"]
        assert nodes[0] == row["start_node"]
        assert nodes[-1] == "adopted"
        assert len(row["mediators_identified"]) >= 1
        assert "adopted" not in row["mediators_identified"]
        assert row["start_node"] not in row["mediators_identified"]
```

Then fix the existing universal start-node assert in `test_causal_paths_cover_all_three_gold_standard_cohort_outcomes` so it excludes the new HCP rows. Replace this block:

```python
    assert (df["start_node"] == "treatment_arm").all()
    for _, row in df.iterrows():
        nodes = row["causal_chain"]["nodes"]
        assert nodes[0] == "treatment_arm"
        assert nodes[-1] == row["end_node"]
```

with:

```python
    # Patient chains all start at treatment_arm; HCP chains (end_node 'adopted')
    # start at peer_influence_score / treatment_arm and are asserted separately.
    patient = df[df["end_node"] != "adopted"]
    assert (patient["start_node"] == "treatment_arm").all()
    for _, row in patient.iterrows():
        nodes = row["causal_chain"]["nodes"]
        assert nodes[0] == "treatment_arm"
        assert nodes[-1] == row["end_node"]
```

- [ ] **Step 2: Run the new + existing tests to confirm the new ones fail**

Run: `.venv/bin/python -m pytest tests/unit/test_synthetic/test_causal_paths_generator.py -v`
Expected: the 3 new `test_hcp_adoption_*` tests FAIL (no `adopted` end_node rows yet); the existing cohort-outcome test PASSES with the scoped assert (HCP rows not yet present, so the scope is a no-op until Step 3).

- [ ] **Step 3: Emit the HCP edges**

In `src/ml/synthetic/generators/causal_paths_generator.py`, add this constant immediately after the `_COHORT_CONFOUNDERS` dict (after line 38):

```python
# HCP-grain adoption edges (Shard 06.3 cohort: hcp_brand_adoption JOIN
# hcp_profiles). TWO questions per brand, ADDITIVE to the patient cohort edges so
# the leaderboard enumerates the HCP grain from the same causal_paths SSOT:
#   peer_influence_score -> adopted : EXOGENOUS centrality, EMPTY backdoor.
#   treatment_arm        -> adopted : rep engagement, confounded by centrality_z
#                                     (= log1p(influence_network_size)).
# centrality_z is the modeled backdoor for the rep-engagement arm; the loader
# derives it from hcp_profiles. A single non-treatment/non-outcome mediator keeps
# every chain a clean 2-hop path AND non-empty (the generator's mediator
# invariant). HCP edges are brand-replicated for all three gold-standard brands.
_HCP_QUESTIONS = (
    ("peer_influence_score", "adopted", [], "centrality_diffusion"),
    ("treatment_arm", "adopted", ["centrality_z"], "rep_engagement_path"),
)
```

Then, in `generate()`, insert this block immediately BEFORE `return pd.DataFrame(rows)` (currently line 93):

```python
        # HCP-grain adoption edges — ADDITIVE, fixed 6-row block (2 questions x 3
        # brands), independent of n_records, so the SSOT always carries every HCP
        # question for the hcp_adoption-dataset leaderboard.
        for brand in _BRANDS:
            for start_node, end_node, confounders, mediator in _HCP_QUESTIONS:
                effect = round(float(self._rng.uniform(0.10, 0.55)), 4)
                direct = round(effect * float(self._rng.uniform(0.4, 0.8)), 4)
                indirect = round(effect - direct, 4)
                disc = (now - timedelta(days=int(self._rng.integers(0, 25)))).date()
                rows.append(
                    {
                        "path_id": f"scp_{uuid.uuid4().hex[:13]}",
                        "discovery_date": disc.isoformat(),
                        "causal_chain": {"nodes": [start_node, mediator, end_node]},
                        "start_node": start_node,
                        "end_node": end_node,
                        "intermediate_nodes": [mediator],
                        "path_length": 2,
                        "causal_effect_size": effect,
                        "confidence_level": round(float(self._rng.uniform(0.80, 0.95)), 3),
                        "method_used": "backdoor.linear_regression",
                        "confounders_controlled": list(confounders),
                        "mediators_identified": [mediator],
                        "time_lag_days": int(self._rng.integers(7, 60)),
                        "validation_status": "validated",
                        "business_impact_estimate": round(
                            effect * float(self._rng.uniform(1e5, 5e5)), 2
                        ),
                        "data_split": "unassigned",
                        "direct_effect": direct,
                        "indirect_effect": indirect,
                        "brand": brand,
                        "region": str(self._rng.choice(_REGIONS)),
                        "confirmation_count": int(self._rng.integers(1, 5)),
                        "created_at": now.isoformat(),
                        "is_synthetic": True,
                    }
                )
        return pd.DataFrame(rows)
```

(Delete the original bare `return pd.DataFrame(rows)` line — it is now the last line of the block above.)

- [ ] **Step 4: Run the suite to verify it passes**

Run: `.venv/bin/python -m pytest tests/unit/test_synthetic/test_causal_paths_generator.py -v`
Expected: PASS (the 3 new HCP tests + the existing patient tests, including the scoped cohort-outcome test).

- [ ] **Step 5: Commit**

```bash
git add src/ml/synthetic/generators/causal_paths_generator.py tests/unit/test_synthetic/test_causal_paths_generator.py
git commit -m "feat(synthetic): emit HCP adoption edges into the causal_paths SSOT (centrality + rep-engagement)"
```

---

### Task 2: Reseed prod `causal_paths` with the HCP edges (GATED prod write)

**This task writes to prod.** Execute ONLY after explicit user authorization. It re-runs the same scoped `causal_paths` loader P1 Task 2 uses (the generator now also emits HCP edges) and re-syncs FalkorDB. The reseed replaces `is_synthetic=true` rows only (real rows untouched — the generator emits `is_synthetic=True` exclusively). **Coordinate with P1 Task 2:** if P1's reseed already ran in the SAME session against this generator, this is the same load and need not run twice — verify the HCP edges are present (Step 4); if they are, this task is already satisfied.

**Files:** none changed (ops task using existing scripts).

- [ ] **Step 1: Snapshot current synthetic rows (rollback safety)**

```bash
docker exec -i supabase-db psql -U postgres -d postgres -c \
  "\copy (SELECT * FROM causal_paths WHERE is_synthetic IS TRUE) TO STDOUT WITH CSV HEADER" \
  > /tmp/causal_paths_synthetic_backup_p2_$(date +%s).csv
wc -l /tmp/causal_paths_synthetic_backup_p2_*.csv
```

- [ ] **Step 2: Confirm the scoped causal_paths loader invocation**

Run: `.venv/bin/python scripts/load_synthetic_data.py --help`
Expected: identify the flag(s) that scope the load to `causal_paths` (the script instantiates `CausalPathsGenerator` at `scripts/load_synthetic_data.py:340`; `batch_loader` upserts `is_synthetic=true` rows). Note the exact invocation; do NOT run a full multi-table reload. The patient `n_records` knob does not affect the HCP block (it is a fixed 6-row additive block), so any `n_records ≥ 90` that gives ≥1 row per patient cell also yields all 6 HCP edges.

- [ ] **Step 3: Regenerate + upsert, then sync FalkorDB**

Run the scoped loader (causal_paths only). Then:

```bash
set -a; source .env; set +a
.venv/bin/python scripts/sync_causal_paths_to_falkordb.py --execute
```

(The FalkorDB sync is grain-agnostic — it MERGEs `(:Variable)-[:CAUSES]->(:Variable)` for EVERY validated chain, so the HCP edges `peer_influence_score→adopted` and `treatment_arm→adopted` appear in the KG automatically.)

- [ ] **Step 4: Verify the HCP edges exist (live)**

```bash
docker exec -i supabase-db psql -U postgres -d postgres -c \
  "SELECT start_node, end_node, count(DISTINCT brand) brands, count(*) n
   FROM causal_paths WHERE is_synthetic IS TRUE AND end_node = 'adopted'
   GROUP BY 1,2 ORDER BY 1;"
```
Expected: 2 rows — `peer_influence_score -> adopted` (brands=3) and `treatment_arm -> adopted` (brands=3), each `confounders_controlled` matching the modeled set (`[]` and `["centrality_z"]`).

- [ ] **Step 5: Commit** — nothing to commit (data-only). Record the verification output in the PR description.

---

### Task 3: Register the `hcp_adoption` dataset spec + numeric columns

Add `hcp_adoption` to the two security/spec constants. Its `treatment` allowlist carries both HCP treatments (`peer_influence_score`, `treatment_arm`); its `outcome` is `adopted`; its `covariate` carries the derived `centrality_z`. The numeric set lists every loadable column so the coercion gate applies. The treatment/outcome/covariate names MUST match the SSOT `start_node`/`end_node`/`confounders_controlled` emitted in Task 1, and the columns the JOIN loader (Task 4) produces.

**Files:**
- Modify: `src/api/routes/causal.py` (`_CAUSAL_DATASET_SPECS` at line 804; `_CAUSAL_NUMERIC_COLUMNS` at line 826)
- Test: `tests/unit/test_api/test_causal_hcp_adoption.py` (create)

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_api/test_causal_hcp_adoption.py
"""Unit coverage for the HCP-grain causal dataset (hcp_adoption).

hcp_adoption is a JOIN dataset: hcp_brand_adoption (treatment_arm, adopted, brand)
JOIN hcp_profiles (peer_influence_score, influence_network_size -> centrality_z)
on hcp_id. These tests are CI-safe (no DB, no agent run); the live JOIN is covered
by a faithful check.
"""

import pytest


@pytest.mark.unit
def test_hcp_adoption_spec_registered():
    from src.api.routes.causal import _CAUSAL_DATASET_SPECS, _CAUSAL_NUMERIC_COLUMNS

    assert "hcp_adoption" in _CAUSAL_DATASET_SPECS
    spec = _CAUSAL_DATASET_SPECS["hcp_adoption"]
    assert set(spec["treatment"]) == {"peer_influence_score", "treatment_arm"}
    assert spec["outcome"] == ["adopted"]
    assert spec["covariate"] == ["centrality_z"]
    # Every loadable column is numeric-coerced (the gate covers treatment+outcome+cov).
    numeric = _CAUSAL_NUMERIC_COLUMNS["hcp_adoption"]
    assert {"peer_influence_score", "treatment_arm", "adopted", "centrality_z"} <= numeric
```

- [ ] **Step 2: Run it to confirm it fails**

Run: `.venv/bin/python -m pytest tests/unit/test_api/test_causal_hcp_adoption.py::test_hcp_adoption_spec_registered -v`
Expected: FAIL — `KeyError: 'hcp_adoption'`.

- [ ] **Step 3: Register the spec**

In `src/api/routes/causal.py`, add the `hcp_adoption` entry to `_CAUSAL_DATASET_SPECS` (after the `patient_journeys` block, before the closing brace at line 820):

```python
    # HCP grain: hcp_brand_adoption (treatment_arm, adopted, brand) JOIN
    # hcp_profiles (peer_influence_score, influence_network_size) on hcp_id. The
    # JOIN loader derives centrality_z = zscore(log1p(influence_network_size)) as
    # the modeled backdoor for the rep-engagement arm. peer_influence_score is the
    # EXOGENOUS-centrality treatment (empty backdoor); treatment_arm is the rep
    # engagement arm (adjust centrality_z). adopted is the binary outcome.
    "hcp_adoption": {
        "treatment": ["peer_influence_score", "treatment_arm"],
        "outcome": ["adopted"],
        "covariate": ["centrality_z"],
    },
```

Then add the `hcp_adoption` entry to `_CAUSAL_NUMERIC_COLUMNS` (after the `patient_journeys` block, before the closing brace at line 842):

```python
    "hcp_adoption": {
        "peer_influence_score",
        "treatment_arm",
        "adopted",
        "centrality_z",
    },
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `.venv/bin/python -m pytest tests/unit/test_api/test_causal_hcp_adoption.py::test_hcp_adoption_spec_registered -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/api/routes/causal.py tests/unit/test_api/test_causal_hcp_adoption.py
git commit -m "feat(causal): register hcp_adoption dataset spec + numeric columns (HCP grain)"
```

---

### Task 4: JOIN-aware loader for `hcp_adoption` (preserve the allowlist/coercion gate)

`_load_agent_estimation_frame` (line 1410) and `_list_dataset_brands` (line 845) both issue a single-table `client.table(dataset).select(...)`. `hcp_adoption` is NOT a real table — it is a JOIN. Add a JOIN branch that reuses the EXISTING, proven two-reads-plus-pandas-merge HCP pattern from the treatment-effects surface (`_te_paged_select` at line 4000 + the merge in `_resolve_treatment_effect_frame` at line 4030), derives `centrality_z`, and then applies the SAME column-allowlist + numeric-coercion + drop-missing-treatment/outcome gate. This keeps the security gate intact on the JOIN path (spec §11 risk: "HCP JOIN loader bypasses the column security gate" → mitigation: apply the same allowlist + numeric coercion).

**Why two-reads-plus-merge, not a single embedded select:** the existing HCP loader deliberately reads the two tables separately and merges in pandas because `hcp_profiles` is NOT brand-partitioned (its centrality covariates are brand-agnostic) and PostgREST embed paging across the FK is fiddly. A live probe confirmed both the PostgREST embed AND the two-reads-merge work; this plan REUSES the proven two-reads-merge helper rather than introducing a second JOIN mechanism.

**centrality_z derivation:** `centrality_z = zscore(log1p(influence_network_size))`, matching the generator's `centrality_z = (log1p(network_size) - mean) / std` (`hcp_adoption_artifact.py:106-108`), so the loaded confounder equals the value the DGP confounds `treatment_arm` on. `math.log1p` is already imported (`causal.py:29`).

**Files:**
- Modify: `src/api/routes/causal.py` (new `_load_hcp_adoption_join_frame()` after `_load_agent_estimation_frame`; branch inside `_load_agent_estimation_frame`; JOIN branch inside `_list_dataset_brands`)
- Test: `tests/unit/test_api/test_causal_hcp_adoption.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/test_api/test_causal_hcp_adoption.py  (append)
import math
from unittest.mock import AsyncMock, patch

import pandas as pd
from fastapi import HTTPException

from src.api.routes import causal as causal_routes


def _fake_join_rows():
    # adoption rows (brand-filtered read) and profile rows (un-filtered read).
    adoption = [
        {"hcp_id": "h1", "treatment_arm": 1, "adopted": 1},
        {"hcp_id": "h2", "treatment_arm": 0, "adopted": 0},
        {"hcp_id": "h3", "treatment_arm": 1, "adopted": 1},
    ]
    profiles = [
        {"hcp_id": "h1", "peer_influence_score": 3.0, "influence_network_size": 25},
        {"hcp_id": "h2", "peer_influence_score": 1.0, "influence_network_size": 2},
        {"hcp_id": "h3", "peer_influence_score": 2.5, "influence_network_size": 14},
    ]
    return adoption, profiles


@pytest.mark.asyncio
async def test_hcp_join_frame_builds_treatment_outcome_and_centrality_z():
    adoption, profiles = _fake_join_rows()

    async def fake_paged(client, table, columns, brand):
        return adoption  # the brand-filtered adoption read

    async def fake_profiles(client):
        return profiles

    with (
        patch.object(causal_routes, "get_async_supabase_client", AsyncMock(return_value=object())),
        patch.object(causal_routes, "_te_paged_select", side_effect=fake_paged),
        patch.object(causal_routes, "_load_hcp_profile_centrality", side_effect=fake_profiles),
    ):
        df, select_cols = await causal_routes._load_agent_estimation_frame(
            dataset="hcp_adoption",
            treatment_var="treatment_arm",
            outcome_var="adopted",
            covariates=["centrality_z"],
            limit=1500,
            brand="Kisqali",
        )
    assert set(select_cols) == {"treatment_arm", "adopted", "centrality_z"}
    assert list(df.columns) == ["treatment_arm", "adopted", "centrality_z"]
    assert len(df) == 3
    # centrality_z = zscore(log1p(influence_network_size)) — h1 (size 25) is the highest.
    raw = [math.log1p(25), math.log1p(2), math.log1p(14)]
    mean = sum(raw) / 3
    std = (sum((x - mean) ** 2 for x in raw) / 3) ** 0.5
    assert df.loc[0, "centrality_z"] == pytest.approx((raw[0] - mean) / std, rel=1e-6)


@pytest.mark.asyncio
async def test_hcp_join_empty_backdoor_question_loads_just_treatment_outcome():
    adoption, profiles = _fake_join_rows()

    async def fake_paged(client, table, columns, brand):
        return adoption

    async def fake_profiles(client):
        return profiles

    with (
        patch.object(causal_routes, "get_async_supabase_client", AsyncMock(return_value=object())),
        patch.object(causal_routes, "_te_paged_select", side_effect=fake_paged),
        patch.object(causal_routes, "_load_hcp_profile_centrality", side_effect=fake_profiles),
    ):
        df, select_cols = await causal_routes._load_agent_estimation_frame(
            dataset="hcp_adoption",
            treatment_var="peer_influence_score",
            outcome_var="adopted",
            covariates=[],  # EXOGENOUS root: empty backdoor
            limit=1500,
            brand="Kisqali",
        )
    assert list(df.columns) == ["peer_influence_score", "adopted"]
    assert len(df) == 3


@pytest.mark.asyncio
async def test_hcp_join_rejects_disallowed_column():
    """The allowlist gate still applies on the JOIN path — an off-allowlist column 400s."""
    with patch.object(causal_routes, "get_async_supabase_client", AsyncMock(return_value=object())):
        with pytest.raises(HTTPException) as ei:
            await causal_routes._load_agent_estimation_frame(
                dataset="hcp_adoption",
                treatment_var="treatment_arm",
                outcome_var="adopted",
                covariates=["specialty"],  # not in the hcp_adoption allowlist
                limit=1500,
                brand="Kisqali",
            )
    assert ei.value.status_code == 400
```

- [ ] **Step 2: Run them to confirm they fail**

Run: `.venv/bin/python -m pytest tests/unit/test_api/test_causal_hcp_adoption.py -k "hcp_join" -v`
Expected: FAIL — `_load_hcp_adoption_join_frame` / `_load_hcp_profile_centrality` not defined, and `_load_agent_estimation_frame` does not branch on `hcp_adoption` (it raises 400 because `select_cols` columns are not in a single `hcp_adoption` table read).

- [ ] **Step 3: Implement the JOIN frame builder + a centrality reader, and add the import alias**

First, expose `get_async_supabase_client` at module scope so the loader (and the tests) reference it directly. At the top of `src/api/routes/causal.py`, the function is imported lazily inside helpers; add a module-level import next to the other `from src.memory.services.factories import ...` usages — locate the first lazy import line `from src.memory.services.factories import get_async_supabase_client` (inside `_list_dataset_brands`/loaders) and ALSO add, near the top of the file with the other top-level imports (after the existing `from src.repositories.provenance import ...` line), this line:

```python
from src.memory.services.factories import get_async_supabase_client
```

(The existing lazy `from src.memory.services.factories import get_async_supabase_client` statements inside the helpers can remain — they shadow harmlessly — but the module-level import is what the new helper and the tests patch via `causal_routes.get_async_supabase_client`.)

Then add these two helpers immediately BEFORE `_load_agent_estimation_frame` (before line 1410):

```python
async def _load_hcp_profile_centrality(client: Any) -> List[Dict[str, Any]]:
    """Paged read of hcp_profiles centrality covariates (hcp_id,
    peer_influence_score, influence_network_size). hcp_profiles is NOT
    brand-partitioned, so this reads across the whole (synthetic) table — mirrors
    the proven treatment-effects HCP loader. Provenance-aware (synthetic-gold)."""
    rows: List[Dict[str, Any]] = []
    for page in range(_TE_MAX_PAGES):
        offset = page * _TE_PAGE_SIZE
        query = client.table("hcp_profiles").select(
            "hcp_id,peer_influence_score,influence_network_size"
        )
        query = apply_provenance_filter(query)
        query = query.range(offset, offset + _TE_PAGE_SIZE - 1)
        result = await query.execute()
        batch: List[Dict[str, Any]] = result.data or []
        rows.extend(batch)
        if len(batch) < _TE_PAGE_SIZE:
            break
    return rows


async def _load_hcp_adoption_join_frame(
    *,
    treatment_var: str,
    outcome_var: str,
    covariates: List[str],
    limit: int,
    brand: Optional[str],
) -> tuple["pd.DataFrame", List[str]]:  # type: ignore[name-defined] # noqa: F821
    """Build the hcp_adoption estimation frame: hcp_brand_adoption (treatment_arm,
    adopted, hcp_id) JOIN hcp_profiles (peer_influence_score,
    influence_network_size) on hcp_id, deriving centrality_z =
    zscore(log1p(influence_network_size)).

    Reuses the proven two-reads-plus-pandas-merge HCP pattern (``_te_paged_select``
    + a separate hcp_profiles read) rather than a single-table select, because
    hcp_profiles is not brand-partitioned. Applies the SAME column-allowlist +
    numeric-coercion + drop-missing-treatment/outcome security gate as
    :func:`_load_agent_estimation_frame`. Fail-closed: 400 disallowed column, 503
    no store / no usable rows. Never fabricates rows.
    """
    import pandas as pd

    spec = _CAUSAL_DATASET_SPECS["hcp_adoption"]
    allowed = set(spec["treatment"]) | set(spec["outcome"]) | set(spec["covariate"])
    requested = [treatment_var, outcome_var, *covariates]
    not_allowed = [c for c in requested if c not in allowed]
    if not_allowed:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Column(s) {not_allowed} are not permitted for dataset "
                f"'hcp_adoption'. Allowed: {sorted(allowed)}"
            ),
        )
    select_cols = list(dict.fromkeys(requested))

    client = await get_async_supabase_client()
    if client is None:
        raise HTTPException(status_code=503, detail="Causal data store unavailable")

    # brand scopes the adoption read (brand lives on hcp_brand_adoption); the
    # centrality covariates are brand-agnostic on hcp_profiles.
    adoption_rows = await _te_paged_select(
        client, "hcp_brand_adoption", "hcp_id,treatment_arm,adopted", brand or ""
    ) if brand else await _te_paged_select_all_brands(client)
    if not adoption_rows:
        raise HTTPException(
            status_code=503,
            detail=(
                "No usable estimation rows for the requested variables "
                f"({treatment_var} -> {outcome_var}) in dataset 'hcp_adoption'."
            ),
        )
    profile_rows = await _load_hcp_profile_centrality(client)
    if not profile_rows:
        raise HTTPException(status_code=503, detail="hcp_profiles centrality unavailable")

    adoption_df = pd.DataFrame(adoption_rows)
    profile_df = pd.DataFrame(profile_rows).drop_duplicates(subset="hcp_id")
    merged = adoption_df.merge(profile_df, on="hcp_id", how="inner")
    if merged.empty:
        raise HTTPException(status_code=503, detail="hcp_adoption JOIN produced no rows")

    # Derive centrality_z = zscore(log1p(influence_network_size)) — matches the DGP
    # (hcp_adoption_artifact.py) the rep-engagement arm is confounded on.
    ins = pd.to_numeric(merged["influence_network_size"], errors="coerce")
    centrality = ins.map(lambda v: math.log1p(v) if pd.notna(v) else None).astype(float)
    std = centrality.std(ddof=0)
    merged["centrality_z"] = (
        (centrality - centrality.mean()) / std if std and std > 0 else 0.0
    )
    merged["peer_influence_score"] = pd.to_numeric(
        merged.get("peer_influence_score"), errors="coerce"
    )

    # Same numeric-coercion + drop-missing-treatment/outcome gate as the patient loader.
    numeric_cols = _CAUSAL_NUMERIC_COLUMNS.get("hcp_adoption", set())
    records: List[Dict[str, Any]] = []
    for _, row in merged.iterrows():
        record: Dict[str, Any] = {}
        usable = True
        for col in select_cols:
            value = row.get(col)
            if col in numeric_cols and value is not None:
                try:
                    value = float(value)
                except (TypeError, ValueError):
                    value = None
            if col in (treatment_var, outcome_var) and (value is None or pd.isna(value)):
                usable = False
                break
            record[col] = None if (value is not None and pd.isna(value)) else value
        if usable:
            records.append(record)
        if len(records) >= limit:
            break

    if not records:
        raise HTTPException(
            status_code=503,
            detail=(
                "No usable estimation rows for the requested variables "
                f"({treatment_var} -> {outcome_var}) in dataset 'hcp_adoption'."
            ),
        )
    return pd.DataFrame(records), select_cols


async def _te_paged_select_all_brands(client: Any) -> List[Dict[str, Any]]:
    """Brand-agnostic paged read of hcp_brand_adoption (all brands) for the
    hcp_adoption frame when no brand filter is set."""
    rows: List[Dict[str, Any]] = []
    for page in range(_TE_MAX_PAGES):
        offset = page * _TE_PAGE_SIZE
        query = client.table("hcp_brand_adoption").select("hcp_id,treatment_arm,adopted")
        query = apply_provenance_filter(query)
        query = query.range(offset, offset + _TE_PAGE_SIZE - 1)
        result = await query.execute()
        batch: List[Dict[str, Any]] = result.data or []
        rows.extend(batch)
        if len(batch) < _TE_PAGE_SIZE:
            break
    return rows
```

- [ ] **Step 4: Branch `_load_agent_estimation_frame` to the JOIN builder**

In `_load_agent_estimation_frame`, immediately after the docstring and BEFORE `spec = _CAUSAL_DATASET_SPECS.get(dataset)` (line 1428), add the JOIN dispatch:

```python
    # hcp_adoption is a JOIN dataset (hcp_brand_adoption JOIN hcp_profiles), not a
    # single table — route it to the JOIN-aware loader (same allowlist/coercion gate).
    if dataset == "hcp_adoption":
        return await _load_hcp_adoption_join_frame(
            treatment_var=treatment_var,
            outcome_var=outcome_var,
            covariates=covariates,
            limit=limit,
            brand=brand,
        )
```

- [ ] **Step 5: Make `_list_dataset_brands` JOIN-aware**

`_list_dataset_brands` (line 845) reads `client.table(dataset).select("brand")`. For `hcp_adoption` the brand-bearing table is `hcp_brand_adoption`. Add a mapping so the brand enumeration reads the right physical table. Replace the body line `query = client.table(dataset).select("brand")` with:

```python
        # hcp_adoption is a JOIN dataset; its brand column lives on hcp_brand_adoption.
        brand_table = "hcp_brand_adoption" if dataset == "hcp_adoption" else dataset
        query = client.table(brand_table).select("brand")
```

- [ ] **Step 6: Run the HCP tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/unit/test_api/test_causal_hcp_adoption.py -v`
Expected: PASS (spec + 3 JOIN-frame tests).

- [ ] **Step 7: Commit**

```bash
git add src/api/routes/causal.py tests/unit/test_api/test_causal_hcp_adoption.py
git commit -m "feat(causal): JOIN-aware hcp_adoption loader (centrality_z) preserving the allowlist gate"
```

---

### Task 5: Scope `_discover_candidate_questions` by grain so each dataset enumerates only its own questions

P1's `_discover_candidate_questions(dataset, brand)` reads ALL `causal_paths` distinct questions via `get_distinct_questions(brand=...)` (the SSOT has no `grain` column) and intersects each row's confounders with the dataset's numeric allowlist. With HCP edges now in the SSOT, the patient dataset would pull HCP questions (and vice versa). Scope by grain: keep a candidate only if its `treatment` is in the dataset's `spec["treatment"]` AND its `outcome` is in `spec["outcome"]`. This is correct for BOTH grains (patient questions' treatment/outcome are only in the patient spec; HCP questions' only in the HCP spec) and is the spec's "one derivation path feeds every grain" (§5.3).

**Reconciliation with P1:** this EXTENDS P1's `_discover_candidate_questions` with one filter. P1's helper already loops over SSOT rows and builds `_CandidateQuestion`s; this adds a `spec`-membership guard inside that loop. Apply against the post-P1 body.

**Files:**
- Modify: `src/api/routes/causal.py` (`_discover_candidate_questions`)
- Test: `tests/unit/test_api/test_causal_hcp_adoption.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/test_api/test_causal_hcp_adoption.py  (append)
@pytest.mark.asyncio
async def test_hcp_dataset_enumerates_only_hcp_questions():
    """The HCP dataset must NOT surface patient questions (treatment_arm ->
    persistent_180d) even though they share the causal_paths SSOT."""
    ssot = [
        {"treatment": "peer_influence_score", "outcome": "adopted", "brand": "Kisqali", "confounders": []},
        {"treatment": "treatment_arm", "outcome": "adopted", "brand": "Kisqali", "confounders": ["centrality_z"]},
        {"treatment": "treatment_arm", "outcome": "persistent_180d", "brand": "Kisqali",
         "confounders": ["disease_severity", "academic_hcp", "geographic_region"]},
    ]
    with patch.object(causal_routes, "_get_causal_path_repo") as mk:
        mk.return_value.get_distinct_questions = AsyncMock(return_value=ssot)
        qs = await causal_routes._discover_candidate_questions("hcp_adoption", brand="Kisqali")
    outcomes = {q.outcome for q in qs}
    assert outcomes == {"adopted"}
    treatments = {q.treatment for q in qs}
    assert treatments == {"peer_influence_score", "treatment_arm"}
    # The exogenous-root question keeps an EMPTY adjustment set.
    exo = next(q for q in qs if q.treatment == "peer_influence_score")
    assert exo.adjustment_set == []
    # The rep-engagement question keeps centrality_z (numeric, in the HCP allowlist).
    rep = next(q for q in qs if q.treatment == "treatment_arm")
    assert rep.adjustment_set == ["centrality_z"]


@pytest.mark.asyncio
async def test_patient_dataset_still_excludes_hcp_questions():
    """Symmetry: the patient dataset must NOT surface the adopted-outcome HCP rows."""
    ssot = [
        {"treatment": "treatment_arm", "outcome": "treatment_initiated", "brand": "Fabhalta",
         "confounders": ["disease_severity", "age_at_diagnosis"]},
        {"treatment": "peer_influence_score", "outcome": "adopted", "brand": "Fabhalta", "confounders": []},
    ]
    with patch.object(causal_routes, "_get_causal_path_repo") as mk:
        mk.return_value.get_distinct_questions = AsyncMock(return_value=ssot)
        qs = await causal_routes._discover_candidate_questions("patient_journeys", brand=None)
    assert {q.outcome for q in qs} == {"treatment_initiated"}
    assert all(q.treatment != "peer_influence_score" for q in qs)
```

- [ ] **Step 2: Run them to confirm they fail**

Run: `.venv/bin/python -m pytest tests/unit/test_api/test_causal_hcp_adoption.py -k "enumerates_only_hcp or excludes_hcp" -v`
Expected: FAIL — without the grain filter, the HCP dataset includes the `persistent_180d` patient question (and the patient dataset includes the `adopted` HCP question).

- [ ] **Step 3: Add the grain-scope filter**

In `_discover_candidate_questions` (post-P1), the loop builds `_CandidateQuestion`s from SSOT rows. Inside that loop, after `t, o = r["treatment"], r["outcome"]` and the existing `if t == o or o in _COMPLEMENT_OUTCOMES_SKIP: continue` guard, add a grain-membership guard so only this dataset's questions survive:

```python
        # Grain scope: the causal_paths SSOT is shared across grains (no grain
        # column), so keep only questions whose treatment AND outcome belong to
        # THIS dataset's allowlist — the HCP grain (adopted) and the patient grain
        # (initiation/retention) thus never cross-contaminate the leaderboard.
        if t not in spec["treatment"] or o not in spec["outcome"]:
            continue
```

(`spec` is already bound in P1's helper as `spec = _CAUSAL_DATASET_SPECS[dataset]`. The existing confounder-intersection line — `adj = [c for c in r.get("confounders", []) if c in allowed_cov and c not in (t, o)]` — already keeps `centrality_z` for the HCP rep arm because it is in the HCP numeric allowlist, and yields `[]` for the exogenous-root row.)

- [ ] **Step 4: Run the HCP suite to verify it passes**

Run: `.venv/bin/python -m pytest tests/unit/test_api/test_causal_hcp_adoption.py -v`
Expected: PASS (all HCP unit tests).

- [ ] **Step 5: Commit**

```bash
git add src/api/routes/causal.py tests/unit/test_api/test_causal_hcp_adoption.py
git commit -m "feat(causal): scope leaderboard enumeration by grain (HCP vs patient) over the shared SSOT"
```

---

## Verification

- [ ] Targeted suite green:
  `.venv/bin/python -m pytest tests/unit/test_synthetic/test_causal_paths_generator.py tests/unit/test_api/test_causal_hcp_adoption.py tests/unit/test_api/test_causal_discover_effects.py -v`
  (the discover-effects suite is P1's; the grain-scope change must not regress it.)
- [ ] Lint clean (the Lint gate cascade-skips backend tests if red):
  `ruff check src/ && ruff format --check src/`
- [ ] Types clean (scoped — do NOT run whole-tree mypy on the droplet; CI is the arbiter):
  `mypy src/api/routes/causal.py src/ml/synthetic/generators/causal_paths_generator.py`
- [ ] **Faithful live run** (after the gated reseed, on the real backend with `E2I_INCLUDE_SYNTHETIC=true`):
  1. `GET /causal/brands?dataset=hcp_adoption` → returns `["Fabhalta","Kisqali","Remibrutinib"]` (proves the JOIN-aware brand enumeration reads `hcp_brand_adoption`).
  2. `POST /causal/discover-effects?dataset=hcp_adoption&brand=Kisqali` → poll `GET /causal/discover-effects/{job_id}` to `completed`. Expect exactly 2 effects (`peer_influence_score→adopted`, `treatment_arm→adopted`), each carrying `brand="Kisqali"` and `adjustment_set` (`[]` and `["centrality_z"]`), and NO patient outcomes (`persistent_180d`/`treatment_initiated`).
  3. Drill into the `peer_influence_score→adopted` row via `GET /causal/agent-analyze/{analysis_id}` → DAG nodes `{peer_influence_score, adopted}` (2-node, empty backdoor), a non-fabricated ATE, `n_rows ≈ 5000`. The live probe confirmed strong signal (high-centrality adopt 0.603 vs low 0.210 for Kisqali), so this question should estimate a positive, significant effect.
  4. `POST /causal/agent-analyze` body `{"treatment_var":"treatment_arm","outcome_var":"adopted","dataset":"hcp_adoption","brand":"Kisqali"}` → poll to completion; `discovered_confounders` includes `centrality_z`; ATE stable on re-run.
- [ ] Adversarial multi-lens review before PR (has repeatedly caught CI-passing honesty/presentation bugs on this surface).

## Self-Review

- **Spec coverage (§5.3, §6 HCP bullet, §11):**
  - "Make `causal_paths` the universal SSOT for all grains (… incl. ∅-backdoor rows)" → Task 1 emits both HCP edges incl. the empty-backdoor exogenous root; Task 5 makes the single derivation path grain-correct.
  - "HCP: add the `hcp_adoption` spec + JOIN loader; add HCP edges to `causal_paths`" → Tasks 1–4.
  - "+6 questions, incl. exogenous-treatment ∅-backdoor" → 2 questions × 3 brands = 6; the `peer_influence_score→adopted` row carries `confounders_controlled=[]` → `adjustment_set=[]`.
  - Risk "HCP JOIN loader bypasses the column security gate" → Task 4's `_load_hcp_adoption_join_frame` re-applies the SAME allowlist + numeric coercion + drop-missing-treatment/outcome gate; `test_hcp_join_rejects_disallowed_column` proves it.
- **Placeholder scan:** none — every code/test step contains complete code; Task 2 is an ops task with exact commands + a `--help` confirmation. No "TBD"/"similar to"/"handle edge cases".
- **Type consistency:**
  - `_load_hcp_adoption_join_frame(...)` returns `tuple[pd.DataFrame, List[str]]`, identical to `_load_agent_estimation_frame`'s return, so the `_run_discover_effects_task` / `run_causal_agent_analysis` call sites consume it unchanged.
  - `_te_paged_select(client, table, columns, brand)` is reused verbatim (existing signature at `causal.py:4000`); `_te_paged_select_all_brands(client)` and `_load_hcp_profile_centrality(client)` mirror its paged-`.range()` shape and provenance handling.
  - The new spec value matches `Dict[str, List[str]]` (`treatment`/`outcome`/`covariate` lists) and the numeric set is a `set` of column names, consistent with `_CAUSAL_DATASET_SPECS` / `_CAUSAL_NUMERIC_COLUMNS` typing.
  - `centrality_z` is a single consistent string token across the SSOT (`confounders_controlled`), the spec `covariate`, the numeric set, and the loader's derived column.
- **Reasoning checks (REASON-BEFORE-RULES / cheapest-disproof, run during planning):**
  - The PostgREST embedded JOIN and the two-reads-merge were BOTH live-probed against prod; this plan reuses the EXISTING proven two-reads-merge helper (`_te_paged_select`) rather than introducing a second JOIN mechanism — consistent with the already-shipped treatment-effects HCP loader.
  - `hcp_profiles` has NO `brand` column (verified) → the brand filter applies to `hcp_brand_adoption` only; `centrality_z` covariates are brand-agnostic.
  - There is a PRE-EXISTING `hcp_adoption` *cohort* string in a DIFFERENT endpoint family (treatment-effects / value-chains, `_TE_COHORTS`); this plan adds `hcp_adoption` as a `_CAUSAL_DATASET_SPECS` *dataset* for the discover-effects/agent-analyze path — no collision (distinct constants, distinct endpoints), and it deliberately REUSES that family's `_te_paged_select` loader helper.

## Cross-plan assumptions (reconcile before executing)

1. **P1 merges first** and provides, in `causal.py`: `_CandidateQuestion`, `_get_causal_path_repo`, `_discover_candidate_questions(dataset, brand)` (with `spec = _CAUSAL_DATASET_SPECS[dataset]` and the confounder-intersection line), `_prerank_questions`, and a `_run_discover_effects_task` that iterates `_CandidateQuestion`s and calls `_load_agent_estimation_frame(dataset=…, covariates=q.adjustment_set, brand=q.brand)`. Task 5 extends P1's `_discover_candidate_questions`; Task 4's loader is invoked by P1's task unchanged.
2. **P1 adds `brand` + `adjustment_set` to `DiscoveredEffect`** and **#1030 adds `summary`** (both already merged); this plan touches none of those fields nor `_effect_from_agent_response`. The HCP effects flow through P1's `_effect_from_agent_response` exactly like patient effects.
3. **P1's generator decoupling (Task 1) is in place** before this plan's Task 1 appends the HCP block; the HCP block is additive (it appends to `rows` before `return pd.DataFrame(rows)`), so it composes regardless of P1's exact patient-loop body.
4. **P1's reseed (P1 Task 2) and this plan's reseed (Task 2) are the SAME scoped `causal_paths` load** against the now-HCP-aware generator — run once per session; this plan's Task 2 Step 4 verifies the HCP edges are present.
5. **`GeneratorConfig` is imported from `src.ml.synthetic.generators.base`** (NOT `src.ml.synthetic.config` — that module has no `GeneratorConfig`). This plan's generator test uses the correct path; if P1's Task 1 test used `src.ml.synthetic.config`, fix it to `src.ml.synthetic.generators.base` when reconciling.
6. **Live substrate is present and verified:** `hcp_brand_adoption` (15000 rows, 3 brands, `adopted`~0.40, `treatment_arm` 100% populated 0/1) and `hcp_profiles` (5000 rows, `peer_influence_score`/`influence_network_size` 100% populated, FK `hcp_brand_adoption_hcp_id_fkey`→`hcp_profiles`). The `treatment_arm` column on `hcp_brand_adoption` came from `scripts/backfill_hcp_treatment_arm.py`; if a fresh DB lacks it, that backfill must run before the live verification step.
