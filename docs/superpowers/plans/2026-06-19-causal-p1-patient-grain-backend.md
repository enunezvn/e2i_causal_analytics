# P1 — Patient-Grain SSOT-Derived Leaderboard (Backend) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. Execute in an isolated git worktree; TDD red-first; commit per task; drive to a green fixed point (ralph-loop) and use codex:codex-rescue when stuck.

**Goal:** Make the causal leaderboard enumerate its patient-grain questions from the gold-standard `causal_paths` SSOT (never the hand-curated cross-product), attach each question's *modeled* adjustment set, scope each to its brand, fix the brand×outcome lockstep data bug, and pre-rank candidates with the cheap FWL screen.

**Architecture:** Replace only the *enumeration* hat of `_CAUSAL_DATASET_SPECS` (keep its `/variables` dropdown source + the column-allowlist/400 security gate + numeric coercion). Questions come from `CausalPathRepository` rows, each carrying `start_node`/`end_node`/`brand`/`confounders_controlled`. The generator's brand-outcome diagonal is decoupled so all 9 patient cells seed; prod `causal_paths` is reseeded (gated). A data-ranked FWL pre-rank orders candidates before the serial agent runs.

**Tech Stack:** Python 3.12, FastAPI, pandas/numpy, Supabase (PostgREST), pytest. No new deps.

**Scope (this plan):** patient grain only; backend only. **Out of scope → follow-on plans:** P1b (`geographic_region` one-hot encoding to complete the retention adjustment set + agent-prior seeding into `graph_builder.CausalPriorKnowledge` — depends on post-#1030 base); P0 (unified FE page); P2 (HCP); P3 (trigger); enrichment.

**Pre-req note:** P1 is **#1030-independent** (it touches `causal.py` enumeration, the generator, the repo, and `schemas/causal.py` `DiscoveredEffect` — none of which #1030 modifies). It can proceed before or after the #1030 merge.

---

### Task 1: Decouple the generator's brand×outcome lockstep

**Bug:** `causal_paths_generator.py` keys both `brand` and `outcome` on `i % 3` (lines 50, 55), so they are perfectly correlated → only the diagonal `{(Remi,init),(Kis,persist),(Fab,disc)}` is ever emitted. Combined with the older init-only seed, prod holds only 5 of 9 cells.

**Files:**
- Modify: `src/ml/synthetic/generators/causal_paths_generator.py:49-55`
- Test: `tests/unit/test_synthetic/test_causal_paths_generator.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_synthetic/test_causal_paths_generator.py  (APPEND to the existing file —
# it already imports GeneratorConfig from src.ml.synthetic.generators.base; do NOT overwrite it)
from src.ml.synthetic.generators.base import GeneratorConfig
from src.ml.synthetic.generators.causal_paths_generator import CausalPathsGenerator


def test_all_brand_outcome_cells_emitted():
    """Every (brand x outcome) cell must appear — not just the i%3 diagonal."""
    df = CausalPathsGenerator(GeneratorConfig(seed=5, n_records=27)).generate()
    cells = set(zip(df["brand"], df["end_node"]))
    brands = {"Remibrutinib", "Kisqali", "Fabhalta"}
    outcomes = {"treatment_initiated", "persistent_180d", "discontinued_180d"}
    assert cells == {(b, o) for b in brands for o in outcomes}


def test_confounders_match_modeled_set_per_outcome():
    df = CausalPathsGenerator(GeneratorConfig(seed=5, n_records=27)).generate()
    row = df[(df.brand == "Kisqali") & (df.end_node == "persistent_180d")].iloc[0]
    assert set(row["confounders_controlled"]) == {
        "disease_severity", "academic_hcp", "geographic_region",
    }
    row2 = df[df.end_node == "treatment_initiated"].iloc[0]
    assert set(row2["confounders_controlled"]) == {"disease_severity", "age_at_diagnosis"}
```

- [ ] **Step 2: Run it to confirm it fails**

Run: `pytest tests/unit/test_synthetic/test_causal_paths_generator.py::test_all_brand_outcome_cells_emitted -v`
Expected: FAIL — `cells` is missing 4 combos (diagonal only).

- [ ] **Step 3: Decouple brand from outcome**

Replace the loop head in `generate()` (lines 49-55):

```python
        now = datetime.now(timezone.utc)
        rows = []
        # Full brand x outcome matrix — decoupled so EVERY cohort is represented
        # for EVERY brand (the old `_BRANDS[i%3]` + `_COHORT_OUTCOMES[i%3]` keyed
        # both on i%3 -> a diagonal that emitted only 3 of 9 cells).
        cells = [(b, o) for b in _BRANDS for o in _COHORT_OUTCOMES]
        for i in range(self.config.n_records):
            brand, outcome = cells[i % len(cells)]
            effect = round(float(self._rng.uniform(0.10, 0.55)), 4)  # recoverable band
```

(Delete the old `brand = _BRANDS[i % 3]` and `outcome = _COHORT_OUTCOMES[i % len(_COHORT_OUTCOMES)]` lines; everything below `effect = ...` is unchanged.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/test_synthetic/test_causal_paths_generator.py -v`
Expected: PASS (new tests + the existing ones).

- [ ] **Step 5: Commit**

```bash
git add src/ml/synthetic/generators/causal_paths_generator.py tests/unit/test_synthetic/test_causal_paths_generator.py
git commit -m "fix(synthetic): decouple causal_paths brand x outcome so all 9 cells seed"
```

---

### Task 2: Reseed prod `causal_paths` + sync KG (GATED prod write)

**This task writes to prod.** Execute ONLY after explicit user authorization. The reseed replaces `is_synthetic=true` rows (real rows are untouched — the generator only emits `is_synthetic=True`) and de-hairballs the KG (helps open #1031).

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
Expected: identify the flag that scopes the load to `causal_paths` (the script instantiates `CausalPathsGenerator` at `scripts/load_synthetic_data.py:340`). Note the exact invocation; do NOT run a full multi-table reload.

- [ ] **Step 3: Regenerate + upsert the full 9-cell matrix**

Run the scoped loader (causal_paths only) with enough records for ≥1 row per cell (e.g. `n_records` ≥ 90 → ~10/cell). The `batch_loader` upserts `is_synthetic=true` rows to `causal_paths`.
Then sync to FalkorDB:

```bash
python scripts/sync_causal_paths_to_falkordb.py
```

- [ ] **Step 4: Verify all 9 cells exist (live)**

```bash
docker exec -i supabase-db psql -U postgres -d postgres -c \
  "SELECT start_node, end_node, brand, count(*) FROM causal_paths
   WHERE is_synthetic IS TRUE GROUP BY 1,2,3 ORDER BY 2,3;"
```
Expected: 9 rows — `treatment_arm -> {treatment_initiated, persistent_180d, discontinued_180d}` × `{Remibrutinib, Kisqali, Fabhalta}`, each `confounders_controlled` populated.

- [ ] **Step 5: Commit** — nothing to commit (data-only). Record the verification output in the PR description.

---

### Task 3: `CausalPathRepository.get_distinct_questions()`

**Files:**
- Modify: `src/repositories/causal_path.py`
- Test: `tests/unit/test_repositories/test_causal_path.py` (create if absent)

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_repositories/test_causal_path.py
import pytest
from unittest.mock import AsyncMock
from src.repositories.causal_path import CausalPathRepository


@pytest.mark.asyncio
async def test_get_distinct_questions_dedups_and_carries_confounders():
    repo = CausalPathRepository(supabase_client=AsyncMock())
    repo.get_many = AsyncMock(return_value=[
        {"start_node": "treatment_arm", "end_node": "persistent_180d", "brand": "Kisqali",
         "confounders_controlled": ["disease_severity", "academic_hcp", "geographic_region"]},
        {"start_node": "treatment_arm", "end_node": "persistent_180d", "brand": "Kisqali",
         "confounders_controlled": ["disease_severity", "academic_hcp", "geographic_region"]},
        {"start_node": "treatment_arm", "end_node": "treatment_initiated", "brand": "Fabhalta",
         "confounders_controlled": ["disease_severity", "age_at_diagnosis"]},
    ])
    qs = await repo.get_distinct_questions(include_synthetic=True)
    assert len(qs) == 2  # the duplicate Kisqali/persistent collapses
    kis = next(q for q in qs if q["brand"] == "Kisqali")
    assert kis["treatment"] == "treatment_arm"
    assert kis["outcome"] == "persistent_180d"
    assert kis["confounders"] == ["disease_severity", "academic_hcp", "geographic_region"]
```

- [ ] **Step 2: Run it to confirm it fails**

Run: `pytest tests/unit/test_repositories/test_causal_path.py -v`
Expected: FAIL — `AttributeError: get_distinct_questions`.

- [ ] **Step 3: Implement the method** (append inside `CausalPathRepository`)

```python
    async def get_distinct_questions(
        self,
        *,
        brand: Optional[str] = None,
        limit: int = 2000,
        include_synthetic: bool = True,
    ) -> List[dict]:
        """Distinct (treatment, outcome, brand) causal questions from the SSOT,
        each carrying its modeled backdoor set (``confounders_controlled``).

        This is the source of truth for the discovery leaderboard's questions
        (replaces the hand-curated cross-product). ``include_synthetic`` defaults
        True because the gold-standard substrate is synthetic.
        """
        filters = {"brand": brand} if brand else None
        rows = await self.get_many(
            filters=filters, limit=limit, include_synthetic=include_synthetic
        )
        seen: dict = {}
        for r in rows:
            key = (r.get("start_node"), r.get("end_node"), r.get("brand"))
            if None in (key[0], key[1]) or key in seen:
                continue
            seen[key] = {
                "treatment": r["start_node"],
                "outcome": r["end_node"],
                "brand": r.get("brand"),
                "confounders": list(r.get("confounders_controlled") or []),
            }
        return list(seen.values())
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/test_repositories/test_causal_path.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/repositories/causal_path.py tests/unit/test_repositories/test_causal_path.py
git commit -m "feat(repo): CausalPathRepository.get_distinct_questions (SSOT questions + backdoor sets)"
```

---

### Task 4: Derive leaderboard questions from the SSOT (with per-question adjustment set + brand)

Replace the `_discover_candidate_pairs(spec)` cross-product with an SSOT-derived enumeration. Each question carries its treatment, outcome, **brand**, and its **modeled adjustment set** (intersected with the dataset's numeric allowlist so the loader's security gate still passes — `geographic_region` is dropped here and restored in P1b's encoding task).

**Files:**
- Modify: `src/api/schemas/causal.py` (add `brand`, `adjustment_set` to `DiscoveredEffect`)
- Modify: `src/api/routes/causal.py` (new `_CandidateQuestion` + `_discover_candidate_questions`; thread through `discover_causal_effects` and `_run_discover_effects_task`)
- Test: `tests/unit/test_api/test_causal_discover_effects.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/test_api/test_causal_discover_effects.py  (add)
import pytest
from unittest.mock import AsyncMock, patch
from src.api.routes import causal as causal_routes


@pytest.mark.asyncio
async def test_candidate_questions_come_from_ssot_with_modeled_adjustment_set():
    fake_questions = [
        {"treatment": "treatment_arm", "outcome": "persistent_180d", "brand": "Kisqali",
         "confounders": ["disease_severity", "academic_hcp", "geographic_region"]},
        {"treatment": "treatment_arm", "outcome": "treatment_initiated", "brand": "Fabhalta",
         "confounders": ["disease_severity", "age_at_diagnosis"]},
    ]
    with patch.object(causal_routes, "_get_causal_path_repo") as mk:
        mk.return_value.get_distinct_questions = AsyncMock(return_value=fake_questions)
        qs = await causal_routes._discover_candidate_questions("patient_journeys", brand=None)

    by_outcome = {q.outcome: q for q in qs}
    # retention: geographic_region dropped (non-numeric), numeric confounders kept
    assert by_outcome["persistent_180d"].adjustment_set == ["disease_severity", "academic_hcp"]
    assert by_outcome["persistent_180d"].brand == "Kisqali"
    # initiation: all-numeric set preserved
    assert by_outcome["treatment_initiated"].adjustment_set == ["disease_severity", "age_at_diagnosis"]


@pytest.mark.asyncio
async def test_brand_filter_subsets_questions():
    fake = [
        {"treatment": "treatment_arm", "outcome": "persistent_180d", "brand": "Kisqali",
         "confounders": ["disease_severity"]},
        {"treatment": "treatment_arm", "outcome": "discontinued_180d", "brand": "Fabhalta",
         "confounders": ["disease_severity"]},
    ]
    with patch.object(causal_routes, "_get_causal_path_repo") as mk:
        mk.return_value.get_distinct_questions = AsyncMock(return_value=fake)
        qs = await causal_routes._discover_candidate_questions("patient_journeys", brand="Kisqali")
    assert [q.brand for q in qs] == ["Kisqali"]
```

- [ ] **Step 2: Run them to confirm they fail**

Run: `pytest tests/unit/test_api/test_causal_discover_effects.py -k candidate_questions -v`
Expected: FAIL — `_discover_candidate_questions` / `_get_causal_path_repo` not defined.

- [ ] **Step 3: Add the schema fields**

In `src/api/schemas/causal.py`, on `DiscoveredEffect`, add (next to the existing fields):

```python
    brand: Optional[str] = Field(None, description="Brand this question is scoped to (SSOT-derived)")
    adjustment_set: List[str] = Field(default_factory=list, description="Modeled backdoor set used for this estimate")
```

- [ ] **Step 4: Implement the SSOT enumeration in `causal.py`**

Add near the other discover-effects helpers (after `_COMPLEMENT_OUTCOMES_SKIP`, replacing `_discover_candidate_pairs`'s role):

```python
from typing import NamedTuple  # (add to the existing typing import line)


class _CandidateQuestion(NamedTuple):
    treatment: str
    outcome: str
    brand: Optional[str]
    adjustment_set: List[str]


def _get_causal_path_repo() -> "CausalPathRepository":
    from src.repositories.causal_path import CausalPathRepository
    from src.memory.services.factories import get_supabase_client
    return CausalPathRepository(supabase_client=get_supabase_client())


async def _discover_candidate_questions(
    dataset: str, brand: Optional[str]
) -> List[_CandidateQuestion]:
    """SSOT-derived leaderboard questions (replaces the hand-curated cross-product).

    Reads distinct (treatment, outcome, brand) from ``causal_paths`` and attaches
    each row's modeled ``confounders_controlled`` intersected with the dataset's
    numeric allowlist (so the loader's column/coercion gate still passes;
    geographic_region is non-numeric and is added back in P1b's encoding task).
    Self-pairs and complement outcomes are dropped, mirroring the prior dedup.
    """
    spec = _CAUSAL_DATASET_SPECS[dataset]
    numeric = _CAUSAL_NUMERIC_COLUMNS.get(dataset, set())
    allowed_cov = set(spec["covariate"]) & numeric
    repo = _get_causal_path_repo()
    rows = await repo.get_distinct_questions(brand=brand, include_synthetic=True)
    out: List[_CandidateQuestion] = []
    for r in rows:
        t, o = r["treatment"], r["outcome"]
        if t == o or o in _COMPLEMENT_OUTCOMES_SKIP:
            continue
        # Grain-scope guard (shared with P2/P3): causal_paths has NO `grain` column
        # and P2/P3 add HCP/trigger edges to the same table, so restrict each
        # dataset's leaderboard to questions whose treatment AND outcome belong to
        # THIS dataset's spec. No-op for patient (all rows pass); load-bearing once
        # other grains share the SSOT. (Reconciliation D2 — owned here in P1.)
        if t not in spec["treatment"] or o not in spec["outcome"]:
            continue
        adj = [c for c in r.get("confounders", []) if c in allowed_cov and c not in (t, o)]
        out.append(_CandidateQuestion(treatment=t, outcome=o, brand=r.get("brand"), adjustment_set=adj))
    return out
```

- [ ] **Step 5: Thread questions through `discover_causal_effects` + `_run_discover_effects_task`**

In `discover_causal_effects`, replace `pairs = _discover_candidate_pairs(spec)` with:

```python
    questions = await _discover_candidate_questions(dataset, brand)
```

and update the initial response + background task to use `questions` (carry brand + adjustment_set):

```python
    initial = DiscoverEffectsResponse(
        job_id=job_id, status="pending", dataset=dataset, brand=brand,
        total=len(questions), completed=0,
        effects=[DiscoveredEffect(treatment=q.treatment, outcome=q.outcome,
                                  brand=q.brand, adjustment_set=q.adjustment_set,
                                  status="pending") for q in questions],
    )
    await _discover_effects_store.set(job_id, initial)
    background_tasks.add_task(_run_discover_effects_task, job_id, dataset, questions, data_source, brand)
```

In `_run_discover_effects_task`, change the signature `pairs: List[tuple]` → `questions: List[_CandidateQuestion]`, key `effects` by `(q.treatment, q.outcome, q.brand)`, and replace the per-pair covariate line:

```python
        for q in questions:
            t, o = q.treatment, q.outcome
            ...
            cov = q.adjustment_set  # modeled backdoor set, NOT the blanket pool
            df, _select = await _load_agent_estimation_frame(
                dataset=dataset, treatment_var=t, outcome_var=o,
                covariates=cov, limit=1500, brand=q.brand or brand,
            )
            ...
            req = AgentCausalAnalysisRequest(
                treatment_var=t, outcome_var=o, dataset=dataset, limit=1500,
                auto_discover=True, brand=q.brand or brand,
            )
            ...
            effects[(t, o, q.brand)] = _effect_from_agent_response(t, o, resp, aid)
```

Update `_effect_from_agent_response` to also set `brand=` and `adjustment_set=` on the returned `DiscoveredEffect` (pass `q` in). Delete `_discover_candidate_pairs` (now unused).

- [ ] **Step 6: Run the full discover-effects suite**

Run: `pytest tests/unit/test_api/test_causal_discover_effects.py -v`
Expected: PASS (new + existing, after updating any existing test that constructed `pairs`).

- [ ] **Step 7: Commit**

```bash
git add src/api/routes/causal.py src/api/schemas/causal.py tests/unit/test_api/test_causal_discover_effects.py
git commit -m "feat(causal): derive leaderboard questions + modeled adjustment sets from causal_paths SSOT"
```

---

### Task 5: Data-ranked FWL pre-rank before the serial agent runs

The serial agent loop is expensive (minutes/effect). Order the candidates by the cheap FWL adjusted-partial-correlation (`_adjusted_partial_corr`, already implemented) so high-signal questions validate first and the progressively-filled leaderboard surfaces winners early. This folds the vestigial `/propose-questions` logic into the leaderboard path.

**Files:**
- Modify: `src/api/routes/causal.py` (`_run_discover_effects_task` — sort `questions` by pre-rank)
- Test: `tests/unit/test_api/test_causal_discover_effects.py`

- [ ] **Step 1: Write the failing test**

```python
@pytest.mark.asyncio
async def test_questions_are_fwl_preranked(monkeypatch):
    qs = [
        causal_routes._CandidateQuestion("treatment_arm", "treatment_initiated", "Kisqali", ["disease_severity"]),
        causal_routes._CandidateQuestion("treatment_arm", "persistent_180d", "Kisqali", ["disease_severity"]),
    ]
    # weak signal for initiation, strong for persistence
    strengths = {"treatment_initiated": 0.05, "persistent_180d": 0.60}

    async def fake_prerank(dataset, q):
        return strengths[q.outcome]

    monkeypatch.setattr(causal_routes, "_prerank_signal", fake_prerank)
    ordered = await causal_routes._prerank_questions("patient_journeys", qs)
    assert [q.outcome for q in ordered] == ["persistent_180d", "treatment_initiated"]
```

- [ ] **Step 2: Run it to confirm it fails**

Run: `pytest tests/unit/test_api/test_causal_discover_effects.py -k preranked -v`
Expected: FAIL — `_prerank_questions` / `_prerank_signal` not defined.

- [ ] **Step 3: Implement the pre-rank**

```python
async def _prerank_signal(dataset: str, q: "_CandidateQuestion") -> float:
    """Cheap FWL screen for one question; 0.0 when undefined / unloadable."""
    try:
        df, _ = await _load_agent_estimation_frame(
            dataset=dataset, treatment_var=q.treatment, outcome_var=q.outcome,
            covariates=q.adjustment_set, limit=1500, brand=q.brand,
        )
    except HTTPException:
        return 0.0
    pc = _adjusted_partial_corr(df, q.treatment, q.outcome, q.adjustment_set)
    return abs(pc) if pc is not None else 0.0


async def _prerank_questions(
    dataset: str, questions: List["_CandidateQuestion"]
) -> List["_CandidateQuestion"]:
    """Order candidates by descending data-driven association so strong effects
    validate first (the leaderboard fills progressively)."""
    scored = await asyncio.gather(*[_prerank_signal(dataset, q) for q in questions])
    return [q for _, q in sorted(zip(scored, questions), key=lambda p: p[0], reverse=True)]
```

In `_run_discover_effects_task`, before the loop: `questions = await _prerank_questions(dataset, questions)`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/test_api/test_causal_discover_effects.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/api/routes/causal.py tests/unit/test_api/test_causal_discover_effects.py
git commit -m "feat(causal): FWL data-ranked pre-rank of leaderboard questions (folds in propose-questions)"
```

---

## Verification (whole plan)

- [ ] `pytest tests/unit/test_synthetic/test_causal_paths_generator.py tests/unit/test_repositories/test_causal_path.py tests/unit/test_api/test_causal_discover_effects.py -v` — all green.
- [ ] `ruff check src/ && ruff format --check src/` — clean (the Lint gate cascade-skips backend tests if red).
- [ ] `mypy src/api/routes/causal.py src/repositories/causal_path.py src/ml/synthetic/generators/causal_paths_generator.py` — clean (scoped; do NOT run whole-tree mypy on the droplet).
- [ ] **Faithful live run** (after the gated reseed): submit `POST /causal/discover-effects?brand=Kisqali`, poll to completion — expect SSOT-derived questions, each row carrying its `brand` + `adjustment_set`, ATE stable vs. the pre-change run (the patient-grain numeric adjustment set ⊆ the old blanket set, so estimates should not regress).
- [ ] Adversarial multi-lens review before PR.

## Self-Review (done)

- **Spec coverage:** §5.2 derivation (T3/T4/T5), §6 lockstep+reseed (T1/T2), "keep security hats" (T4 honors the allowlist), "no hand-curation" (T4 deletes `_discover_candidate_pairs`). `geographic_region` encoding + agent-prior seeding explicitly deferred to P1b (stated in scope).
- **Placeholder scan:** none — every code/test step has concrete content; Task 2 is an ops task with exact commands + a `--help` confirmation (not a placeholder).
- **Type consistency:** `_CandidateQuestion(treatment, outcome, brand, adjustment_set)` used consistently across T4/T5; `DiscoveredEffect.brand`/`adjustment_set` added in T4 and consumed in T4's response construction; repo `get_distinct_questions` returns dicts with `treatment/outcome/brand/confounders`, consumed by `_discover_candidate_questions`.

## Follow-on plans (separate worktrees/PRs)
- **P1b** — `geographic_region` one-hot encoding (completes the retention backdoor set) + agent-prior seeding (`confounders_controlled` → `CausalPriorKnowledge.required_edges` in `graph_builder`, post-#1030).
- **P0** — unified agent-led FE page (leaderboard landing, grain/brand facets, drill-down reuse, "pose your own question", retire `/causal-discovery`).
- **P2 / P3 / Enrichment** — per the spec.
