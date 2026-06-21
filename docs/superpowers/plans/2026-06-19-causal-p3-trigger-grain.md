# P3 — Trigger Grain (the NBA RCT) Implementation Plan

> For agentic workers: REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (- [ ]) syntax. Execute in an isolated git worktree; TDD red-first; commit per task; drive to a green fixed point and use codex:codex-rescue when stuck.

**Goal:** Add the trigger grain — the only true RCT in the gold standard — by emitting trigger edges into the `causal_paths` SSOT, registering an `nba_triggers` dataset (spec + numeric set + brand_column) over the `triggers` table, and writing a loader that preserves the column-allowlist security gate while coercing the trigger booleans/text (`control_group_flag`, `action_taken`, `conversion_flag`, `acceptance_status=accepted`) to numeric 0/1 with NULL→0 (never row-drop) for designed-NULL outcomes.

**Architecture:** Reuse Plan 1's SSOT-derivation machinery verbatim (`CausalPathRepository.get_distinct_questions()`, `_CandidateQuestion`, `_discover_candidate_questions()`, `_prerank_questions()`, `DiscoveredEffect.brand`/`adjustment_set`) — P3 adds nothing to the enumeration path; it adds the trigger rows the SSOT enumerates and a trigger-aware data loader. The `nba_triggers` loader is a thin variant of `_load_agent_estimation_frame` that (a) resolves the brand filter against `brand_id` (triggers have no `brand` column), (b) derives allowlisted boolean/text columns to 0/1 before float-coercion, and (c) treats NULL as 0 for the designed-NULL trigger outcomes instead of dropping the row. `conversion_flag` is a DB STORED-GENERATED column (`outcome_value > 0`) — no generator change needed for it; the generator already emits `outcome_value`.

**Tech Stack:** Python 3.12, FastAPI, pandas/numpy, Supabase (PostgREST), pytest. No new deps.

**Scope (this plan):** trigger grain only; backend only (generator + batch_loader + causal.py specs/loader/enumeration-glue). **Out of scope → other plans:** P0 (unified FE page + grain facet rendering), P1 (patient SSOT machinery — a hard prerequisite, see Cross-plan dependencies), P2 (HCP grain), enrichment.

**Pre-req note (HARD):** P3 **depends on P1 being merged.** P1 introduces `_CandidateQuestion`, `_discover_candidate_questions()`, `_prerank_questions()`, `CausalPathRepository.get_distinct_questions()`, and the `DiscoveredEffect.brand`/`adjustment_set` fields that Task 6 wires the trigger questions through. P3 is **#1030-independent** for the files it MODIFIES — verified that #1030's diff touches only `causal.py` (agent-analyze / estimator-comparison area, +66 lines), `schemas/causal.py` (+55), `graph_builder.py`, `interpretation.py`, and the FE; it does NOT touch `causal_paths_generator.py`, `trigger_generator.py`, `batch_loader.py`, the dataset specs (`_CAUSAL_DATASET_SPECS` ~804), the loaders (`get_causal_estimation_data` ~1309, `_load_agent_estimation_frame` ~1410), or `_list_dataset_brands` (~845). Plan against the merged (post-#1030, post-P1) `main`.

---

## File Structure

| File | Change | Responsibility |
|---|---|---|
| `src/ml/synthetic/generators/causal_paths_generator.py` | Modify | Emit two trigger-grain edges into the SSOT (the RCT `control_group_flag→action_taken`, ∅-backdoor; and `acceptance_status→conversion_flag`, priority as effect modifier) alongside the existing patient edges. |
| `tests/unit/test_synthetic/test_causal_paths_generator.py` | Modify | Add trigger-edge coverage; scope the existing `start_node=="treatment_arm"` invariant to patient-grain rows. |
| `src/api/routes/causal.py` | Modify | Add `nba_triggers` to `_CAUSAL_DATASET_SPECS` + `_CAUSAL_NUMERIC_COLUMNS`; add per-dataset `_CAUSAL_BRAND_COLUMN` + `_CAUSAL_NUMERIC_DERIVATIONS` + `_CAUSAL_FILL_ZERO_OUTCOMES`; teach `_list_dataset_brands`, `get_causal_estimation_data`, `_load_agent_estimation_frame` to resolve the brand column + apply derivations + NULL→0 fill (security gate preserved). |
| `tests/unit/test_api/test_causal_triggers_dataset.py` | Create | Spec/numeric/brand-column registration + derivation/NULL-fill unit coverage for the trigger loader, with the allowlist gate still enforced. |
| `tests/unit/test_api/test_causal_discover_effects.py` | Modify | Assert the SSOT enumeration yields the two trigger questions with ∅ adjustment sets when the dataset is `nba_triggers` (reuses P1's `_discover_candidate_questions`). |

No new migrations, no new scripts. `conversion_flag` is DB STORED-GENERATED (`outcome_value > 0`); the `triggers` TABLE_COLUMNS allowlist in `batch_loader.py` already registers `action_taken` + `control_group_flag` and correctly omits the generated `conversion_flag` — no `batch_loader.py` edit is required.

---

### Task 1: LIVE-VERIFY trigger-grain variance (gate before building)

Per the spec (§4 trigger row, §8 honesty boundary, §11 risk "Trigger variance not yet verified → P3 live-verify gate before building") and the cheapest-disproof directive, the single assumption P3 depends on is: **the trigger columns carry real variance (a non-degenerate RCT signal and a non-degenerate effect-modifier signal).** Disprove it cheaply FIRST with a read-only psql probe against the faithful self-hosted Supabase before writing any code.

**Files:** none (read-only ops gate; record output in the PR description).

- [ ] **Step 1: Probe row count + the four rates against the live faithful DB**

Run (read-only; the faithful target is the local self-hosted `supabase-db`):

```bash
docker exec -i supabase-db psql -U postgres -d postgres -c \
  "SELECT count(*) AS n_rows FROM triggers;"
docker exec -i supabase-db psql -U postgres -d postgres -c \
  "SELECT control_group_flag, count(*) FROM triggers GROUP BY 1 ORDER BY 1;"
docker exec -i supabase-db psql -U postgres -d postgres -c \
  "SELECT (action_taken IS NOT NULL) AS action_present, count(*) FROM triggers GROUP BY 1 ORDER BY 1;"
docker exec -i supabase-db psql -U postgres -d postgres -c \
  "SELECT acceptance_status, count(*) FROM triggers GROUP BY 1 ORDER BY 2 DESC;"
docker exec -i supabase-db psql -U postgres -d postgres -c \
  "SELECT conversion_flag, count(*) FROM triggers GROUP BY 1 ORDER BY 1;"
```

Expected (verified live 2026-06-19 — the build proceeds ONLY if these hold; if any signal is degenerate, STOP and escalate):
- `n_rows` = 37378 (abundant).
- `control_group_flag`: f≈26889 (71.9% treatment), t≈10489 (28.1% control) — both arms non-empty.
- `action_taken` present: t≈13305 (35.6%), f≈24073 — outcome has variance; NULL = no action.
- `acceptance_status`: accepted≈15920 (42.6%), pending≈10365, rejected≈6330, expired≈4763 — 4 levels.
- `conversion_flag`: t≈6373, NULL≈31005, **no `f`** (NULL ⇒ not-converted; the column is STORED-GENERATED `outcome_value > 0`).

- [ ] **Step 2: Confirm BOTH effects are non-degenerate (the actual signal, not just marginal variance)**

```bash
docker exec -i supabase-db psql -U postgres -d postgres -c \
  "SELECT control_group_flag,
          count(*) FILTER (WHERE action_taken IS NOT NULL) AS acted,
          count(*) AS total,
          round(100.0*count(*) FILTER (WHERE action_taken IS NOT NULL)/count(*),1) AS pct
   FROM triggers GROUP BY 1 ORDER BY 1;"
docker exec -i supabase-db psql -U postgres -d postgres -c \
  "SELECT acceptance_status,
          count(*) FILTER (WHERE conversion_flag IS TRUE) AS converted,
          count(*) AS total,
          round(100.0*count(*) FILTER (WHERE conversion_flag IS TRUE)/count(*),1) AS pct
   FROM triggers GROUP BY 1 ORDER BY 4 DESC;"
```

Expected (verified live):
- RCT incrementality: treatment (f) acted 37.6% vs control (t) acted 30.4% = **+7.2pp**, randomized ⇒ ∅ backdoor.
- Effect modifier: accepted converts 40.0% vs rejected/pending/expired 0.0% — a clean designed effect (`conversion_flag = outcome_value > 0`, and `outcome_value` is only set for accepted triggers).

- [ ] **Step 3: Confirm the brand column divergence (triggers have NO `brand`, only `brand_id`)**

```bash
docker exec -i supabase-db psql -U postgres -d postgres -c \
  "SELECT column_name FROM information_schema.columns
   WHERE table_name='triggers' AND column_name IN ('brand','brand_id') ORDER BY 1;"
docker exec -i supabase-db psql -U postgres -d postgres -c \
  "SELECT brand_id, count(*) FROM triggers GROUP BY 1 ORDER BY 2 DESC;"
docker exec -i supabase-db psql -U postgres -d postgres -c \
  "SELECT column_name, is_generated, generation_expression FROM information_schema.columns
   WHERE table_name='triggers' AND column_name='conversion_flag';"
```

Expected (verified live): only `brand_id` exists (NOT `brand`); brand_id ∈ {Remibrutinib≈12599, Kisqali≈12453, Fabhalta≈12326}; `conversion_flag` `is_generated = ALWAYS`, expression `(outcome_value > (0)::numeric)`. This is WHY Task 4 adds `_CAUSAL_BRAND_COLUMN` (`brand_id` for triggers) and WHY no `conversion_flag` write is needed.

- [ ] **Step 4: Commit** — nothing to commit (read-only gate). Paste the probe output into the PR description as the live-verify evidence. If every expectation held, proceed to Task 2.

---

### Task 2: Emit the two trigger-grain edges into the `causal_paths` SSOT

The leaderboard derives its questions from `causal_paths` (P1's `get_distinct_questions`). To surface the trigger grain, the generator must emit trigger edges with their modeled (∅) backdoor sets. Append them AFTER the patient loop so this composes with P1's patient-cell change regardless of P1's exact loop shape (P1 decoupled brand×outcome into a `cells` list; P3 only adds rows).

**Files:**
- Modify: `src/ml/synthetic/generators/causal_paths_generator.py` (module constants near line 20-38; `generate()` return near line 92-93)
- Test: `tests/unit/test_synthetic/test_causal_paths_generator.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/test_synthetic/test_causal_paths_generator.py  (append)
def test_trigger_grain_edges_emitted_with_empty_and_modeled_backdoor():
    """The trigger grain seeds two SSOT edges: the randomized RCT
    (control_group_flag -> action_taken, EMPTY backdoor) and the effect-modifier
    question (acceptance_status -> conversion_flag, EMPTY backdoor; priority is an
    effect modifier, not a confounder). Both per brand."""
    df = CausalPathsGenerator(GeneratorConfig(seed=11, n_records=30)).generate()
    edges = set(zip(df["start_node"], df["end_node"]))
    assert ("control_group_flag", "action_taken") in edges
    assert ("acceptance_status", "conversion_flag") in edges

    rct = df[(df.start_node == "control_group_flag") & (df.end_node == "action_taken")]
    assert len(rct) >= 1
    # The RCT is randomized -> its modeled backdoor set is empty.
    for _, row in rct.iterrows():
        assert list(row["confounders_controlled"]) == []
        assert row["grain"] == "trigger"
        # causal_chain stays a direct edge (no mediators injected for the RCT).
        assert row["causal_chain"]["nodes"] == ["control_group_flag", "action_taken"]
    # Trigger edges exist for every brand (so a brand-scoped leaderboard sees them).
    rct_brands = set(rct["brand"])
    assert {"Remibrutinib", "Kisqali", "Fabhalta"} <= rct_brands

    mod = df[(df.start_node == "acceptance_status") & (df.end_node == "conversion_flag")]
    assert len(mod) >= 1
    for _, row in mod.iterrows():
        assert list(row["confounders_controlled"]) == []
        assert row["grain"] == "trigger"


def test_patient_edges_retain_treatment_arm_start_and_grain():
    """Patient-grain rows are unchanged: they still start at treatment_arm and are
    tagged grain='patient' (the trigger edges must not perturb the patient cells)."""
    df = CausalPathsGenerator(GeneratorConfig(seed=11, n_records=30)).generate()
    patient = df[df["grain"] == "patient"]
    assert (patient["start_node"] == "treatment_arm").all()
    assert set(patient["end_node"]) <= {
        "treatment_initiated", "persistent_180d", "discontinued_180d"
    }
```

- [ ] **Step 2: Run them to confirm they fail**

Run: `pytest tests/unit/test_synthetic/test_causal_paths_generator.py::test_trigger_grain_edges_emitted_with_empty_and_modeled_backdoor tests/unit/test_synthetic/test_causal_paths_generator.py::test_patient_edges_retain_treatment_arm_start_and_grain -v`
Expected: FAIL — `KeyError: 'grain'` / the trigger edges are absent (`AssertionError` on the `edges` membership).

- [ ] **Step 3: Add the trigger-edge constants + tag patient rows with a grain**

In `src/ml/synthetic/generators/causal_paths_generator.py`, after the `_COHORT_CONFOUNDERS` block (currently ending at line 38), add:

```python
# Trigger-grain edges (the NBA RCT). The triggers table carries the only TRUE
# randomized experiment in the gold standard: control_group_flag is a randomized
# holdout, so control_group_flag -> action_taken has an EMPTY backdoor set (no
# confounder to adjust for — randomization breaks every back-door path). The
# second edge, acceptance_status -> conversion_flag, is the designed effect
# (conversion_flag is the DB STORED-GENERATED column outcome_value>0, set only for
# accepted triggers) with priority as an EFFECT MODIFIER (not a confounder), so its
# modeled backdoor set is also empty. Both are direct edges (no mediator injected).
_TRIGGER_EDGES = (
    # (start_node, end_node, confounders_controlled)
    ("control_group_flag", "action_taken", []),
    ("acceptance_status", "conversion_flag", []),
)
```

- [ ] **Step 4: Tag the patient rows and append the trigger rows in `generate()`**

In `generate()`, set `grain` on every patient row dict (add the key inside the existing `rows.append({...})`, alongside `"is_synthetic": True`):

```python
                    "is_synthetic": True,
                    "grain": "patient",
                }
            )
```

Then, immediately before `return pd.DataFrame(rows)` (currently line 93), append the trigger edges (one row per brand × edge, with empty confounders + a direct two-node chain):

```python
        # Trigger grain: emit each RCT/effect-modifier edge for every brand so a
        # brand-scoped leaderboard surfaces the trigger questions too. Empty
        # confounders_controlled (randomized / effect-modifier — no backdoor set);
        # direct two-node causal_chain; no mediators (mediators_identified=[] so the
        # FalkorDB sync builds a clean direct (:Variable start)-[:CAUSES]->(:Variable end)).
        for brand in _BRANDS:
            for start_node, end_node, confounders in _TRIGGER_EDGES:
                effect = round(float(self._rng.uniform(0.05, 0.25)), 4)
                disc = (now - timedelta(days=int(self._rng.integers(0, 25)))).date()
                rows.append(
                    {
                        "path_id": f"scp_{uuid.uuid4().hex[:13]}",
                        "discovery_date": disc.isoformat(),
                        "causal_chain": {"nodes": [start_node, end_node]},
                        "start_node": start_node,
                        "end_node": end_node,
                        "intermediate_nodes": [],
                        "path_length": 1,
                        "causal_effect_size": effect,
                        "confidence_level": round(float(self._rng.uniform(0.80, 0.95)), 3),
                        "method_used": "backdoor.linear_regression",
                        "confounders_controlled": list(confounders),
                        "mediators_identified": [],
                        "time_lag_days": int(self._rng.integers(7, 60)),
                        "validation_status": "validated",
                        "business_impact_estimate": round(
                            effect * float(self._rng.uniform(1e5, 5e5)), 2
                        ),
                        "data_split": "unassigned",
                        "direct_effect": effect,
                        "indirect_effect": 0.0,
                        "brand": brand,
                        "region": str(self._rng.choice(_REGIONS)),
                        "confirmation_count": int(self._rng.integers(1, 5)),
                        "created_at": now.isoformat(),
                        "is_synthetic": True,
                        "grain": "trigger",
                    }
                )
        return pd.DataFrame(rows)
```

- [ ] **Step 5: Fix the existing test's now-too-strong invariant**

The existing `test_causal_paths_cover_all_three_gold_standard_cohort_outcomes` asserts `(df["start_node"] == "treatment_arm").all()` over ALL rows — trigger edges break that. Scope it to patient rows. In `tests/unit/test_synthetic/test_causal_paths_generator.py`, replace the two assertion lines inside that test:

```python
    df = CausalPathsGenerator(GeneratorConfig(seed=3, n_records=30)).generate()
    end_nodes = set(df["end_node"])
    assert {"treatment_initiated", "persistent_180d", "discontinued_180d"} <= end_nodes
    # Every PATIENT chain starts at the treatment arm and terminates at its
    # end_node (trigger-grain edges start at control_group_flag / acceptance_status
    # and are asserted separately).
    patient = df[df["grain"] == "patient"]
    assert (patient["start_node"] == "treatment_arm").all()
    for _, row in patient.iterrows():
        nodes = row["causal_chain"]["nodes"]
        assert nodes[0] == "treatment_arm"
        assert nodes[-1] == row["end_node"]
        # No repeated nodes (would be dropped by _clean_causal_chains).
        assert len(nodes) == len(set(nodes))
```

(Only the post-`generate()` body changes: swap the bare `df` iteration for the `patient`-filtered one. The `test_causal_paths_nonnull_effect_and_mediators_and_tagged` test still passes — trigger rows have non-NULL `causal_effect_size` and `mediators_identified == []` would FAIL its `len(m) >= 1`; so also scope that one.) Replace the mediator assertion line in `test_causal_paths_nonnull_effect_and_mediators_and_tagged`:

```python
    # CM-005: patient chains carry >=1 mediator (trigger RCT edges are direct, no mediator).
    patient = df[df["grain"] == "patient"]
    assert patient["mediators_identified"].apply(lambda m: len(m) >= 1).all()
```

(Replace the original `assert df["mediators_identified"].apply(lambda m: len(m) >= 1).all()` line with the two lines above; every other assertion in that test holds for all rows.)

- [ ] **Step 6: Run the full generator suite**

Run: `pytest tests/unit/test_synthetic/test_causal_paths_generator.py -v`
Expected: PASS (the two new tests + the three updated existing tests).

- [ ] **Step 7: Commit**

```bash
git add src/ml/synthetic/generators/causal_paths_generator.py tests/unit/test_synthetic/test_causal_paths_generator.py
git commit -m "feat(synthetic): emit trigger-grain causal_paths edges (NBA RCT + effect-modifier, empty backdoor)"
```

---

### Task 3: Register the `nba_triggers` dataset spec + numeric set + brand column + derivations

Add the trigger dataset to the spec maps. The trigger questions need: bool/text → 0/1 derivations (`control_group_flag`, `conversion_flag` are bool; `action_taken` is NULL-or-text; `acceptance_status` treatment = `== 'accepted'`), the brand filter resolved against `brand_id` (no `brand` column), and designed-NULL outcomes (`action_taken`, `conversion_flag`) filled to 0 instead of row-dropped. The allowlist still gates which columns may be read.

**Files:**
- Modify: `src/api/routes/causal.py` — after `_CAUSAL_NUMERIC_COLUMNS` (ends line 842) add the trigger spec entry, numeric entry, and three new per-dataset maps.
- Test: `tests/unit/test_api/test_causal_triggers_dataset.py` (create)

- [ ] **Step 1: Write the failing tests (registration only)**

```python
# tests/unit/test_api/test_causal_triggers_dataset.py
import pytest

from src.api.routes.causal import (
    _CAUSAL_BRAND_COLUMN,
    _CAUSAL_DATASET_SPECS,
    _CAUSAL_FILL_ZERO_OUTCOMES,
    _CAUSAL_NUMERIC_COLUMNS,
    _CAUSAL_NUMERIC_DERIVATIONS,
)


@pytest.mark.unit
def test_nba_triggers_spec_registered_with_rct_and_modifier_questions():
    spec = _CAUSAL_DATASET_SPECS["nba_triggers"]
    # RCT: control_group_flag -> action_taken; modifier: acceptance_status -> conversion_flag.
    assert "control_group_flag" in spec["treatment"]
    assert "acceptance_status" in spec["treatment"]
    assert "action_taken" in spec["outcome"]
    assert "conversion_flag" in spec["outcome"]


@pytest.mark.unit
def test_nba_triggers_numeric_and_derivation_and_fill_registered():
    numeric = _CAUSAL_NUMERIC_COLUMNS["nba_triggers"]
    # All four question columns coerce to numeric 0/1.
    assert {"control_group_flag", "action_taken", "conversion_flag", "acceptance_status"} <= numeric
    deriv = _CAUSAL_NUMERIC_DERIVATIONS["nba_triggers"]
    # acceptance_status derives to the "is accepted" indicator; action_taken to presence.
    assert deriv["acceptance_status"]("accepted") == 1.0
    assert deriv["acceptance_status"]("rejected") == 0.0
    assert deriv["acceptance_status"](None) == 0.0
    assert deriv["action_taken"]("called_patient") == 1.0
    assert deriv["action_taken"](None) == 0.0
    # Designed-NULL outcomes fill to 0 instead of dropping the row.
    assert {"action_taken", "conversion_flag"} <= _CAUSAL_FILL_ZERO_OUTCOMES["nba_triggers"]


@pytest.mark.unit
def test_nba_triggers_brand_column_is_brand_id():
    # triggers has NO `brand` column — the filter resolves against brand_id.
    assert _CAUSAL_BRAND_COLUMN.get("nba_triggers") == "brand_id"
    # patient_journeys keeps the default `brand` column.
    assert _CAUSAL_BRAND_COLUMN.get("patient_journeys", "brand") == "brand"
```

- [ ] **Step 2: Run them to confirm they fail**

Run: `pytest tests/unit/test_api/test_causal_triggers_dataset.py -v`
Expected: FAIL — `ImportError`/`KeyError`: `_CAUSAL_BRAND_COLUMN`, `_CAUSAL_NUMERIC_DERIVATIONS`, `_CAUSAL_FILL_ZERO_OUTCOMES`, and `nba_triggers` are not defined.

- [ ] **Step 3: Register the trigger dataset + the three new maps**

In `src/api/routes/causal.py`, add to `_CAUSAL_DATASET_SPECS` (inside the dict, after the `patient_journeys` block that ends at line 819):

```python
    "nba_triggers": {
        # The triggers table — the ONLY true RCT in the gold standard.
        # Treatments: the randomized holdout flag (control_group_flag) and the
        # "trigger accepted" indicator (acceptance_status). Outcomes: action_taken
        # (an action was taken) and conversion_flag (the DB STORED-GENERATED
        # outcome_value>0). No covariates: the RCT is randomized (empty backdoor)
        # and acceptance->conversion is a designed effect with priority as an
        # effect MODIFIER (surfaced, not adjusted), so no confounder is offered.
        "treatment": ["control_group_flag", "acceptance_status"],
        "outcome": ["action_taken", "conversion_flag"],
        "covariate": [],
    },
```

Add to `_CAUSAL_NUMERIC_COLUMNS` (after the `patient_journeys` set that ends at line 841):

```python
    "nba_triggers": {
        # Every trigger question column coerces to numeric 0/1 (booleans via
        # float(bool); acceptance_status/action_taken via the derivations below).
        "control_group_flag",
        "action_taken",
        "conversion_flag",
        "acceptance_status",
    },
```

Then, immediately after the `_CAUSAL_NUMERIC_COLUMNS` dict (after line 842) add the three new per-dataset maps:

```python
# Per-dataset brand-filter column. The triggers table has NO `brand` column — it
# carries `brand_id` (text, NOT NULL). Datasets absent here default to "brand"
# (patient_journeys). Used by _list_dataset_brands + the loaders' brand filter.
_CAUSAL_BRAND_COLUMN: Dict[str, str] = {
    "nba_triggers": "brand_id",
}

# Per-dataset value derivations applied BEFORE float-coercion, for columns whose
# raw value is not directly float()-able into the modeled 0/1 (categorical /
# text / bool). Only allowlisted columns are reachable (the allowlist gate runs
# first), so a derivation can never read an arbitrary column. Each maps a raw cell
# (possibly None) to a float.
def _derive_is_accepted(value: Any) -> float:
    """acceptance_status -> 1.0 when 'accepted' (case-insensitive), else 0.0."""
    return 1.0 if str(value).strip().lower() == "accepted" else 0.0


def _derive_presence(value: Any) -> float:
    """A nullable text/flag column -> 1.0 when a non-empty value is present."""
    if value is None:
        return 0.0
    text = str(value).strip()
    return 0.0 if text == "" or text.lower() in {"none", "false", "0"} else 1.0


_CAUSAL_NUMERIC_DERIVATIONS: Dict[str, Dict[str, Callable[[Any], float]]] = {
    "nba_triggers": {
        "acceptance_status": _derive_is_accepted,
        "action_taken": _derive_presence,
    },
}

# Per-dataset outcome columns whose NULL is a DESIGNED zero (not missing data), so
# a NULL value fills to 0.0 rather than dropping the row. On triggers, action_taken
# is NULL when no action was taken (= 0) and conversion_flag is NULL when not
# converted (the STORED-GENERATED outcome_value>0 is NULL-not-false); dropping those
# rows would discard the RCT control arm / the non-converters and bias the estimate.
_CAUSAL_FILL_ZERO_OUTCOMES: Dict[str, set] = {
    "nba_triggers": {"action_taken", "conversion_flag"},
}
```

Ensure `Callable` and `Any` are imported at the top of `causal.py`. Check the existing `from typing import ...` line (it already imports `Any`, `Dict`, `List`, `Optional`); add `Callable` to it if absent:

```python
from typing import Any, Callable, Dict, List, Optional  # (extend the existing import; keep the other names already present)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/test_api/test_causal_triggers_dataset.py -v`
Expected: PASS (the three registration tests).

- [ ] **Step 5: Commit**

```bash
git add src/api/routes/causal.py tests/unit/test_api/test_causal_triggers_dataset.py
git commit -m "feat(causal): register nba_triggers dataset spec + numeric/derivation/fill-zero/brand-column maps"
```

---

### Task 4: Apply derivations + NULL→0 fill + `brand_id` filter in the loaders (security gate preserved)

Teach `_list_dataset_brands`, `get_causal_estimation_data`, and `_load_agent_estimation_frame` to consult the new maps. The column-allowlist + numeric-coercion gate stays exactly as-is (the allowlist check runs unchanged, first); the derivations only transform the value of an already-allowlisted column, and the fill-zero set only changes the drop decision for designed-NULL outcomes.

**Files:**
- Modify: `src/api/routes/causal.py` — `_list_dataset_brands` (845-871), `get_causal_estimation_data` coercion loop (1367-1395), `_load_agent_estimation_frame` brand fetch + coercion loop (1461-1497)
- Test: `tests/unit/test_api/test_causal_triggers_dataset.py` (add a coercion-helper test)

- [ ] **Step 1: Write the failing test (the row-coercion helper)**

Factor the per-row coercion into a pure, testable helper so the trigger semantics are unit-covered without a DB. Add to `tests/unit/test_api/test_causal_triggers_dataset.py`:

```python
from src.api.routes.causal import _coerce_estimation_row


@pytest.mark.unit
def test_coerce_row_derives_bool_text_and_fills_designed_null_zero():
    # RCT row: control arm (control_group_flag True), no action taken (NULL).
    rec = _coerce_estimation_row(
        {"control_group_flag": True, "action_taken": None},
        select_cols=["control_group_flag", "action_taken"],
        dataset="nba_triggers",
        treatment_var="control_group_flag",
        outcome_var="action_taken",
    )
    assert rec == {"control_group_flag": 1.0, "action_taken": 0.0}  # NULL outcome -> 0, NOT dropped


@pytest.mark.unit
def test_coerce_row_modifier_question_accepted_and_converted():
    rec = _coerce_estimation_row(
        {"acceptance_status": "accepted", "conversion_flag": True},
        select_cols=["acceptance_status", "conversion_flag"],
        dataset="nba_triggers",
        treatment_var="acceptance_status",
        outcome_var="conversion_flag",
    )
    assert rec == {"acceptance_status": 1.0, "conversion_flag": 1.0}
    rec2 = _coerce_estimation_row(
        {"acceptance_status": "rejected", "conversion_flag": None},
        select_cols=["acceptance_status", "conversion_flag"],
        dataset="nba_triggers",
        treatment_var="acceptance_status",
        outcome_var="conversion_flag",
    )
    assert rec2 == {"acceptance_status": 0.0, "conversion_flag": 0.0}


@pytest.mark.unit
def test_coerce_row_patient_outcome_null_still_drops():
    # patient_journeys is NOT in _CAUSAL_FILL_ZERO_OUTCOMES: a NULL outcome still
    # drops the row (returns None) -> the existing gate is unchanged.
    rec = _coerce_estimation_row(
        {"treatment_arm": 1, "persistent_180d": None},
        select_cols=["treatment_arm", "persistent_180d"],
        dataset="patient_journeys",
        treatment_var="treatment_arm",
        outcome_var="persistent_180d",
    )
    assert rec is None
```

- [ ] **Step 2: Run them to confirm they fail**

Run: `pytest tests/unit/test_api/test_causal_triggers_dataset.py -k coerce -v`
Expected: FAIL — `ImportError: _coerce_estimation_row`.

- [ ] **Step 3: Add the `_coerce_estimation_row` helper**

In `src/api/routes/causal.py`, add immediately above `get_causal_estimation_data` (before line 1303's decorator — i.e. after the `_adjusted_partial_corr` helper / propose-questions block, in the loader region):

```python
def _coerce_estimation_row(
    row: Dict[str, Any],
    *,
    select_cols: List[str],
    dataset: str,
    treatment_var: str,
    outcome_var: str,
) -> Optional[Dict[str, Any]]:
    """Coerce one raw DB row into a numeric estimation record, or None to drop it.

    Mirrors the per-row logic shared by the estimation loaders, extended for the
    trigger grain:
      * derivations (``_CAUSAL_NUMERIC_DERIVATIONS``) map an allowlisted
        categorical/text/bool column to 0/1 BEFORE float-coercion;
      * numeric columns (``_CAUSAL_NUMERIC_COLUMNS``) float()-coerce; an
        uncoercible value becomes None;
      * a missing treatment/outcome value drops the row (returns None) UNLESS the
        outcome is a designed-NULL-zero column (``_CAUSAL_FILL_ZERO_OUTCOMES``),
        in which case NULL fills to 0.0 and the row is kept.
    Only allowlisted columns reach this function (the allowlist gate runs first in
    the caller), so a derivation can never read an arbitrary column.
    """
    derivations = _CAUSAL_NUMERIC_DERIVATIONS.get(dataset, {})
    numeric_cols = _CAUSAL_NUMERIC_COLUMNS.get(dataset, set())
    fill_zero = _CAUSAL_FILL_ZERO_OUTCOMES.get(dataset, set())
    record: Dict[str, Any] = {}
    for col in select_cols:
        value = row.get(col)
        if col in derivations:
            # Derivation owns the full mapping (incl. None -> 0.0); no further coercion.
            value = derivations[col](value)
        elif col in numeric_cols and value is not None:
            try:
                value = float(value)
            except (TypeError, ValueError):
                value = None
        if col in (treatment_var, outcome_var) and value is None:
            if col in fill_zero:
                value = 0.0  # designed NULL == 0 (e.g. unconverted / no action)
            else:
                return None  # missing treatment/outcome -> unusable row
        record[col] = value
    return record
```

- [ ] **Step 4: Route both loaders through the helper**

In `get_causal_estimation_data`, replace the coercion loop **starting at the `numeric_cols = _CAUSAL_NUMERIC_COLUMNS.get(dataset, set())` line (1367) through the `if usable: records.append(record)` (1386)** — the whole span, so the now-unused `numeric_cols` local is removed (else ruff F841 fails the Lint gate) — with:

```python
    records: List[Dict[str, Any]] = []
    for row in rows:
        record = _coerce_estimation_row(
            row,
            select_cols=select_cols,
            dataset=dataset,
            treatment_var=treatment_var,
            outcome_var=outcome_var,
        )
        if record is not None:
            records.append(record)
```

In `_load_agent_estimation_frame`, replace the coercion loop **starting at the `numeric_cols = _CAUSAL_NUMERIC_COLUMNS.get(dataset, set())` line (1471) through the `if usable: records.append(record)` (1488)** — the whole span (same shape; drops the unused `numeric_cols` local) — with the identical block:

```python
    records: List[Dict[str, Any]] = []
    for row in rows:
        record = _coerce_estimation_row(
            row,
            select_cols=select_cols,
            dataset=dataset,
            treatment_var=treatment_var,
            outcome_var=outcome_var,
        )
        if record is not None:
            records.append(record)
```

- [ ] **Step 5: Resolve the brand column against `_CAUSAL_BRAND_COLUMN` in both the brand-list + the agent loader**

In `_list_dataset_brands` (845-871), replace the `query = client.table(dataset).select("brand")` line and the `seen` comprehension to use the dataset's brand column:

```python
    brand_col = _CAUSAL_BRAND_COLUMN.get(dataset, "brand")
    try:
        query = client.table(dataset).select(brand_col)
        query = apply_provenance_filter(query)
        result = await query.limit(20000).execute()
    except Exception as e:  # noqa: BLE001 — missing column / store hiccup => no brands
        logger.warning(f"causal brands: could not enumerate brands for '{dataset}': {e}")
        return []
    seen = {
        str(row[brand_col])
        for row in (result.data or [])
        if isinstance(row, dict) and row.get(brand_col)
    }
    return sorted(seen)
```

In `_load_agent_estimation_frame`, replace the brand fetch/filter (lines 1461-1467) to resolve the brand column:

```python
    fetch_cols = list(select_cols)
    brand_col = _CAUSAL_BRAND_COLUMN.get(dataset, "brand")
    if brand:
        fetch_cols = list(dict.fromkeys([*select_cols, brand_col]))
    query = client.table(dataset).select(",".join(fetch_cols))
    query = apply_provenance_filter(query)
    if brand:
        query = query.eq(brand_col, brand)
    result = await query.limit(limit).execute()
    rows = result.data or []
```

- [ ] **Step 6: Run the whole trigger-dataset suite (helper + registration)**

Run: `pytest tests/unit/test_api/test_causal_triggers_dataset.py -v`
Expected: PASS (registration + coercion-helper tests).

- [ ] **Step 7: Run the patient-path regression tests (the refactor must not change patient behavior)**

Run: `pytest tests/unit/test_api/test_causal_agent_analyze.py tests/unit/test_api/test_causal_discover_effects.py -v`
Expected: PASS — the patient loaders behave identically (no derivations / no fill-zero / brand column = "brand" for `patient_journeys`).

- [ ] **Step 8: Commit**

```bash
git add src/api/routes/causal.py tests/unit/test_api/test_causal_triggers_dataset.py
git commit -m "feat(causal): trigger loader derivations + NULL->0 designed outcomes + brand_id filter (gate preserved)"
```

---

### Task 5: SSOT enumeration yields the trigger questions (reuses P1's `_discover_candidate_questions`)

P1's `_discover_candidate_questions(dataset, brand)` reads the SSOT via `get_distinct_questions` and intersects each row's `confounders_controlled` with the dataset's numeric allowlist. For `nba_triggers` the trigger edges (Task 2) carry empty `confounders_controlled`, and the spec's covariate list is empty, so each trigger question yields an empty `adjustment_set`. Add a test that locks this contract (no production code change — this validates the reuse).

**Files:**
- Test: `tests/unit/test_api/test_causal_discover_effects.py` (add)

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_api/test_causal_discover_effects.py  (append)
import pytest
from unittest.mock import AsyncMock, patch
from src.api.routes import causal as causal_routes


@pytest.mark.asyncio
async def test_trigger_questions_from_ssot_have_empty_adjustment_set():
    """The nba_triggers grain enumerates the RCT + effect-modifier questions from
    the SSOT, each with an EMPTY modeled adjustment set (randomized / effect
    modifier). Reuses P1's _discover_candidate_questions verbatim."""
    fake = [
        {"treatment": "control_group_flag", "outcome": "action_taken",
         "brand": "Kisqali", "confounders": []},
        {"treatment": "acceptance_status", "outcome": "conversion_flag",
         "brand": "Kisqali", "confounders": []},
    ]
    with patch.object(causal_routes, "_get_causal_path_repo") as mk:
        mk.return_value.get_distinct_questions = AsyncMock(return_value=fake)
        qs = await causal_routes._discover_candidate_questions("nba_triggers", brand="Kisqali")

    by_outcome = {q.outcome: q for q in qs}
    assert by_outcome["action_taken"].treatment == "control_group_flag"
    assert by_outcome["action_taken"].adjustment_set == []
    assert by_outcome["action_taken"].brand == "Kisqali"
    assert by_outcome["conversion_flag"].treatment == "acceptance_status"
    assert by_outcome["conversion_flag"].adjustment_set == []
```

- [ ] **Step 2: Run it to confirm it fails (or passes only if P1 is present)**

Run: `pytest tests/unit/test_api/test_causal_discover_effects.py -k trigger_questions_from_ssot -v`
Expected on a clean P1 base: PASS immediately (P1's `_discover_candidate_questions` handles `nba_triggers` because Task 3 registered its spec + numeric set). If it ERRORS with `AttributeError: _discover_candidate_questions` / `_get_causal_path_repo`, **P1 has not been merged** — STOP and merge P1 first (this is the hard prerequisite). The test is the guard that P3's reuse contract holds; it must not be made to pass by reimplementing P1's helper.

- [ ] **Step 3: (only if the test surfaced a real gap) confirm `_COMPLEMENT_OUTCOMES_SKIP` does not eat a trigger outcome**

`_COMPLEMENT_OUTCOMES_SKIP = {"discontinued_180d"}` (causal.py:1071) only names a patient outcome, so it cannot drop `action_taken`/`conversion_flag`. No change needed — assert this by inspection (the test in Step 1 already proves both trigger outcomes survive). If a future complement is added for triggers, it would be registered there.

- [ ] **Step 4: Commit**

```bash
git add tests/unit/test_api/test_causal_discover_effects.py
git commit -m "test(causal): SSOT enumeration yields trigger questions with empty adjustment sets (P1 reuse)"
```

---

### Task 6: Reseed prod `causal_paths` with trigger edges + sync KG (GATED prod write)

**This task writes to prod.** Execute ONLY after explicit user authorization (spec §6, §8 "the reseed is a prod data write — gated on explicit user OK", §13). It regenerates the synthetic `causal_paths` rows (now including the trigger edges from Task 2) and re-syncs FalkorDB. The generator only emits `is_synthetic=True`, so real rows are untouched.

**Note on overlap with P1's Task 2:** P1 also reseeds `causal_paths` (for the 9 patient cells). If P1's reseed has already run, this reseed REPLACES the synthetic rows again with the SAME patient cells PLUS the trigger edges — re-running is idempotent for the patient cells (upsert on `path_id` mints fresh ids each run, so it inserts new synthetic rows; the snapshot+verify below is the safety net). Coordinate timing with P1: run P3's reseed AFTER P1's so the final synthetic set carries both grains. If only one reseed is run, run THIS one (it is a superset).

**Files:** none changed (ops task using existing scripts).

- [ ] **Step 1: Snapshot current synthetic rows (rollback safety)**

```bash
docker exec -i supabase-db psql -U postgres -d postgres -c \
  "\copy (SELECT * FROM causal_paths WHERE is_synthetic IS TRUE) TO STDOUT WITH CSV HEADER" \
  > /tmp/causal_paths_synthetic_backup_$(date +%s).csv
wc -l /tmp/causal_paths_synthetic_backup_*.csv
```

- [ ] **Step 2: Confirm the loader's causal_paths-only invocation**

Run: `python scripts/load_synthetic_data.py --help`
Expected: identify the flag that scopes the load to `causal_paths` (the script instantiates `CausalPathsGenerator` at `scripts/load_synthetic_data.py:340` and registers `"causal_paths"` in `batch_loader.py` TABLE_COLUMNS at line 452). Note the exact invocation; do NOT run a full multi-table reload.

- [ ] **Step 3: Regenerate + upsert (patient cells + trigger edges)**

Run the scoped loader (causal_paths only) with enough records for ≥1 row per patient cell (the trigger edges are emitted unconditionally — 2 edges × 3 brands = 6 rows per `generate()` regardless of `n_records`). Then sync to FalkorDB:

```bash
python scripts/sync_causal_paths_to_falkordb.py
```

- [ ] **Step 4: Verify the trigger edges exist (live)**

```bash
docker exec -i supabase-db psql -U postgres -d postgres -c \
  "SELECT start_node, end_node, brand, count(*)
   FROM causal_paths
   WHERE is_synthetic IS TRUE
     AND start_node IN ('control_group_flag','acceptance_status')
   GROUP BY 1,2,3 ORDER BY 1,2,3;"
```

Expected: 6 rows — `control_group_flag -> action_taken` and `acceptance_status -> conversion_flag`, each × {Remibrutinib, Kisqali, Fabhalta}, every `confounders_controlled` empty (`{}`). The patient 9-cell set from P1 is also present (re-verify with P1's query if P3's reseed superseded P1's).

- [ ] **Step 5: Commit** — nothing to commit (data-only). Record the verification output in the PR description.

---

## Verification (whole plan)

- [ ] `pytest tests/unit/test_synthetic/test_causal_paths_generator.py tests/unit/test_api/test_causal_triggers_dataset.py tests/unit/test_api/test_causal_discover_effects.py tests/unit/test_api/test_causal_agent_analyze.py -v` — all green (new + the patient-path regressions).
- [ ] `ruff check src/ && ruff format --check src/` — clean (the Lint gate cascade-skips backend tests if red; `ruff check` ≠ `ruff format --check` — run both).
- [ ] `mypy src/api/routes/causal.py src/ml/synthetic/generators/causal_paths_generator.py` — clean (scoped; do NOT run whole-tree `mypy` on the droplet — CI's MyPy gate is the arbiter).
- [ ] **Faithful live run** (after the gated reseed): submit `POST /causal/discover-effects?dataset=nba_triggers&brand=Kisqali`, poll `GET /causal/discover-effects/{job_id}` to completion. Expect 2 SSOT-derived trigger rows, each carrying `brand="Kisqali"` + `adjustment_set=[]`: the RCT (`control_group_flag→action_taken`) should recover a small positive ATE (~+0.07, matching the live +7.2pp), and `acceptance_status→conversion_flag` a large positive ATE (the designed ~+0.40 separation) — both with a non-fabricated estimate (status completed/needs_review, not failed). Also `POST /causal/agent-analyze` with `{"dataset":"nba_triggers","treatment_var":"control_group_flag","outcome_var":"action_taken","covariates":[],"brand":"Kisqali"}` → a connected DAG + stable ATE. Confirm via `docker logs e2i_api` that the request loaded rows (no 503 "no usable estimation rows") — the NULL→0 fill must keep the control arm / non-converters.
- [ ] Adversarial multi-lens review before PR (it has repeatedly caught CI-passing honesty/presentation bugs in this codebase).
- [ ] Per-commit check-runs pinned to head SHA; CI is the gate-arbiter.

## Self-Review (done)

- **Spec coverage:** §4 trigger-grain row + §11 "Trigger variance not yet verified → P3 live-verify gate" → Task 1 (live-verify FIRST, with the cheapest-disproof of both signals). §6 "Trigger: live-verify variance first; add `nba_triggers` spec + loader; add trigger edges to causal_paths" → Tasks 2 (edges), 3 (spec), 4 (loader). §5.3 "each grain is a dataset with an allowlist + a loader" → Tasks 3/4. §5.2 "questions derive from the SSOT, never hand-curation" + "∅-backdoor rows" → Tasks 2/5 (empty `confounders_controlled` flows to empty `adjustment_set`). §8 "Trigger variance is code-inferred until P3's live verify" + "reseed is a gated prod write" → Tasks 1/6. The brief's explicit note "booleans control_group_flag/action_taken/conversion_flag must coerce to numeric 0/1" → Task 4's `_coerce_estimation_row` (bool via `float(bool)`, text/categorical via derivations) + Task 3's `_CAUSAL_NUMERIC_DERIVATIONS`/`_CAUSAL_FILL_ZERO_OUTCOMES`. The security-gate preservation requirement → Task 4 keeps the allowlist check unchanged (runs first), derivations only transform allowlisted values, fill-zero only changes the drop decision for designed-NULL outcomes (proven by `test_coerce_row_patient_outcome_null_still_drops`).
- **Placeholder scan:** none — every code/test step shows complete content. Task 1 and Task 6 are ops tasks with exact commands + a `--help` confirmation (not placeholders). No "TBD/TODO/handle edge cases/similar to Task N".
- **Type consistency:** `_CAUSAL_BRAND_COLUMN: Dict[str, str]`, `_CAUSAL_NUMERIC_DERIVATIONS: Dict[str, Dict[str, Callable[[Any], float]]]`, `_CAUSAL_FILL_ZERO_OUTCOMES: Dict[str, set]` (mirrors `_CAUSAL_NUMERIC_COLUMNS: Dict[str, set]`). `_coerce_estimation_row(...) -> Optional[Dict[str, Any]]` returns a record or None-to-drop; both loaders consume it identically. `Callable`/`Any` added to the `typing` import. The generator's trigger rows carry the exact `causal_paths` columns the patient rows do (verified against `batch_loader.py` TABLE_COLUMNS line 452-475) plus the new `grain` key — `grain` is NOT in TABLE_COLUMNS, so the loader strips it before insert (no schema error); it exists only for in-DataFrame test/derivation filtering. `_CandidateQuestion(treatment, outcome, brand, adjustment_set)` (P1) is reused unchanged in Task 5.
- **Cross-grain non-interference:** the trigger edges are APPENDED after the patient loop (robust to P1's patient-loop reshape); the patient regression tests in Verification confirm the loader refactor leaves `patient_journeys` byte-identical (no derivations, no fill-zero, brand column `"brand"`).
- **`conversion_flag` correctness:** it is a DB STORED-GENERATED column (`outcome_value > 0`, verified live) — NOT in the generator and correctly NOT in TABLE_COLUMNS; the modifier question reads it as an outcome (NULL→0 via fill-zero). No generator/loader write for it. Verified the generator already emits `outcome_value` (trigger_generator.py:262), so a fresh load auto-populates `conversion_flag`.

## Cross-plan dependencies (reconcile before execution)
- **P1 MUST be merged first (hard).** Task 5 reuses P1's `_discover_candidate_questions`, `_get_causal_path_repo`, `_prerank_questions`, `CausalPathRepository.get_distinct_questions`, and `DiscoveredEffect.brand`/`adjustment_set`. P3 adds NO enumeration code — it only adds the SSOT rows + the trigger loader. If P1 is absent, Task 5's test errors with `AttributeError` (the documented STOP signal).
- **#1030 merges before P1/P3** (spec §12). P3 is #1030-independent for the files it modifies (verified: #1030 does not touch the generator, batch_loader, the dataset specs, the loaders, or `_list_dataset_brands`).
- **Generator-edit coordination with P1/P2** (brief: "extend; sequence after P1/P2 … coordinates with P1/P2 on the generator edit"). All three phases edit `causal_paths_generator.py`. P3's edit is an APPEND of trigger rows + a `grain` tag, designed to compose with P1's patient-cell decoupling and P2's HCP edges. If executed out of order, re-base onto the merged generator and keep the trigger-append block intact.
- **The `causal_paths` reseed (Task 6) overlaps P1's reseed (P1 Task 2).** Run P3's reseed AFTER P1's; P3's is a superset (patient cells + trigger edges). Both are GATED prod writes requiring explicit user authorization; snapshot before, verify after.
