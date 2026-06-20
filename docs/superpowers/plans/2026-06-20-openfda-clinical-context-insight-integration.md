# OpenFDA Clinical Context + Insight Integration — Implementation Plan

> **For agentic workers:** Execute task-by-task via superpowers:subagent-driven-development. Each task is TDD (red → green → commit). Full design + grounded data + exact seams: `docs/superpowers/specs/2026-06-20-openfda-clinical-context-insight-integration-design.md`. Steps use `- [ ]`.

**Goal:** Ground every causal insight in therapy label (OpenFDA indications/limitations/boxed-warning, real-first) + market landscape (curated competitors), and weave both into insight generation (leaderboard rows, causal_impact narrative, gap-analysis density).

**Architecture:** Extend the existing `clinical_context` service (2 new fragments) + 3 fail-open consumers. All additive.

**Tech stack:** Python 3.12, FastAPI, LangGraph, httpx (sync + `asyncio.to_thread` at async edges), React/TS, OpenFDA `drug/label.json` (`OPENFDA_API_KEY`).

**Worktree:** `/home/enunez/Projects/wt_openfda` (branch `feat/causal-openfda-label-context`).

**Local gates (faithful; stacked PRs get no CI until retargeted to main):**
- Backend scoped: `env PYTHONPATH=. .venv/bin/python -m mypy --config-file pyproject.toml <file>`; conftest-free pytest `-n0`.
- FE: `cd frontend && npx tsc -b --noEmit`; `vitest run --no-file-parallelism --pool=forks`.
- Full lint before PR: `ruff check src/ tests/` + `ruff format --check src/ tests/`.

---

### Task 1: OpenFDA client

**Files:**
- Modify: `src/services/clinical_context/clients.py` (add `_OpenFDAClient`)
- Test: `tests/unit/test_services/test_clinical_context/test_openfda_client.py` (create)

Contract (search by INN via `generic_name`; co-pack disambiguation; fail-closed to None):
```python
class _OpenFDAClient:
    BASE = "https://api.fda.gov/drug/label.json"
    def __init__(self, api_key: str | None = None, timeout: float = 30.0) -> None: ...
    def fetch_label(self, drug_name: str) -> dict | None: ...           # generic_name first; pick single-ingredient record (generic == [drug_name]); brand_name fallback; None on 404/empty/error
    @staticmethod
    def approved_indications(label: dict) -> list[str]: ...             # parse indications_and_usage[0]; bullets before "Limitations of Use"
    @staticmethod
    def limitations_of_use(label: dict) -> str | None: ...             # substring from "Limitations of Use" marker; None if absent
    @staticmethod
    def boxed_warning(label: dict) -> str | None: ...                  # label.get("boxed_warning",[None])[0]
```

- [ ] **Step 1 — Red:** Write tests with mocked httpx responses (real probe payloads as fixtures): ribociclib → standalone `KISQALI` (single-ingredient picked over co-pack `letrozole and ribociclib`); remibrutinib → `RHAPSIDO` via generic; `limitations_of_use` present for RHAPSIDO, None for ribociclib/iptacopan; `boxed_warning` present for iptacopan; 404/empty → `fetch_label` None. Run → fail.
- [ ] **Step 2 — Green:** Implement `_OpenFDAClient` per contract. Read `OPENFDA_API_KEY` from env (never log it). Single attempt, 30s timeout, all exceptions → None.
- [ ] **Step 3 — Verify:** pytest the new file `-n0`; scoped mypy on `clients.py`.
- [ ] **Step 4 — Commit:** `feat(clinical-context): OpenFDA label client (indications/LoU/boxed-warning)`

---

### Task 2: Brand map extension (grounded data)

**Files:**
- Modify: `src/services/clinical_context/brand_map.py`
- Test: `tests/unit/test_services/test_clinical_context/test_brand_map.py` (extend)

Add to `BrandClinicalProfile` (frozen): `indications_fallback: list[str]`, `limitations_fallback: str | None`, `boxed_warning_fallback: str | None`, `competitor_map: dict[str, list[str]]` (disease→competitor brand(generic) strings; keys lower-cased). Populate `_STATIC_ENRICHMENT` for the 3 brands with the **grounded** values from spec §0 (inline evidence comments, e.g. `# ATC L01EF CDK4/6i (probe-confirmed)`):
- Kisqali (`ribociclib`, "breast cancer"): competitors `["Ibrance (palbociclib)", "Verzenio (abemaciclib)"]`; indications_fallback HR+/HER2- adv + early adjuvant.
- Fabhalta (`iptacopan`): competitor_map `{"paroxysmal nocturnal hemoglobinuria": ["Soliris (eculizumab)","Ultomiris (ravulizumab)","Empaveli (pegcetacoplan)","Voydeya (danicopan)"], "iga nephropathy": ["Tarpeyo (budesonide)","Filspari (sparsentan)"]}`; boxed_warning_fallback serious-infections.
- Remibrutinib (`remibrutinib`, "chronic spontaneous urticaria"): competitors `["Xolair (omalizumab)","Dupixent (dupilumab)"]`; limitations_fallback "Not indicated for other forms of urticaria".

- [ ] **Step 1 — Red:** Test each brand resolves a non-empty `competitor_map` for its disease; Remibrutinib has `limitations_fallback`; Fabhalta has `boxed_warning_fallback`. Run → fail.
- [ ] **Step 2 — Green:** Add fields + populate. Keep dataclass frozen; default empty for the new fields so existing construction is unaffected.
- [ ] **Step 3 — Verify + Commit:** pytest + scoped mypy. `feat(clinical-context): brand-map indications + curated competitor map (grounded)`

---

### Task 3: Indications + Competitor providers

**Files:**
- Modify: `src/services/clinical_context/providers.py`
- Test: `tests/unit/test_services/test_clinical_context/test_providers.py` (extend)

`OpenFDAIndicationsProvider` (real-first → static fallback) returns `IndicationsFragment(approved_indications, limitations_of_use, boxed_warning, source)` `source ∈ {"openfda","static_fallback"}`. `CuratedCompetitorProvider` returns `CompetitorFragment(competitors, count, source="curated")` from `profile.competitor_map[profile.disease.lower()]` (empty → count 0; **no OpenFDA/ATC call**).

- [ ] **Step 1 — Red:** Indications: live client → source "openfda"; raising client → "static_fallback" with brand_map values. Competitors: always source "curated"; unknown disease → `[]`, count 0. Run → fail.
- [ ] **Step 2 — Green:** Implement both per the `ClinicalContextProvider.enrich(profile)` pattern (idempotent `except Exception → fallback`).
- [ ] **Step 3 — Verify + Commit:** pytest + scoped mypy. `feat(clinical-context): OpenFDA-indications + curated-competitor providers`

---

### Task 4: Service fan-out + result

**Files:**
- Modify: `src/services/clinical_context/service.py`
- Test: `tests/unit/test_services/test_clinical_context/test_service.py` (extend)

`_fan_out` returns 5 fragments `(moa, eps, cite, indications, competitors)`; cache key stays `(brand, disease)`. `fully_live` adds `indications.source == "openfda"` (curated competitors count as **live**, not degradation). `get_context` adds `approved_indications {indications, limitations_of_use, boxed_warning, source}` + `competitor_landscape {competitors, count, source}`.

- [ ] **Step 1 — Red:** 5-fragment fan-out; cache reuse across two outcomes of one brand (1 live fan-out); `fully_live` true when indications=openfda & competitors=curated; degraded TTL self-heal still holds. Run → fail.
- [ ] **Step 2 — Green:** Wire the 2 new providers into `__init__` + `_fan_out` + `get_context`. Update `_FragmentTuple` to 5-tuple.
- [ ] **Step 3 — Verify + Commit:** pytest + scoped mypy. `feat(clinical-context): 5-fragment fan-out incl OpenFDA + competitors`

---

### Task 5: API schemas

**Files:**
- Modify: `src/api/schemas/causal.py`
- Test: `tests/unit/test_api/test_causal_schemas.py` (extend or create)

Add `ApprovedIndications` + `CompetitorLandscape` models; add optional fields to `ClinicalContext`; add `clinical_context: Optional[ClinicalContext] = None` to `DiscoveredEffect`.

- [ ] **Step 1 — Red:** Construct `ClinicalContext` with the new sub-objects; `DiscoveredEffect` default `clinical_context is None`. Run → fail.
- [ ] **Step 2 — Green:** Add models/fields (all optional/defaulted — additive, no OpenAPI break; avoid `List[Tuple]` → use `List[List[str]]`/`List[str]`).
- [ ] **Step 3 — Verify + Commit:** pytest + scoped mypy. `feat(causal): clinical_context schema (indications + competitors) on DiscoveredEffect`

---

### Task 6: Leaderboard attachment (fail-open)

**Files:**
- Modify: `src/api/routes/causal.py` (`_effect_from_agent_response`, ~1343–1372)
- Test: `tests/unit/test_api/test_causal_discover_effects.py` (extend)

Populate `clinical_context` via `await asyncio.to_thread(_clinical_context_service.get_context, brand, outcome)` in try/except (KeyError/unknown brand → None). `_FRAGMENT_CACHE` (brand,disease) → 1 live fan-out per brand across its candidate pairs.

- [ ] **Step 1 — Red:** Happy path → row has `clinical_context` with competitor count; raising service → row still returned, `clinical_context is None` (fail-open). Run → fail.
- [ ] **Step 2 — Green:** Implement; never let context failure fail the row.
- [ ] **Step 3 — Verify + Commit:** pytest + scoped mypy. `feat(causal): attach clinical context to leaderboard rows (fail-open)`

---

### Task 7: Interpretation narrative weave (fail-open)

**Files:**
- Modify: `src/agents/causal_impact/nodes/interpretation.py` (after robustness block, ~line 255)
- Test: `tests/unit/test_agents/test_causal_impact/test_interpretation_clinical_context.py` (create)

After the robustness narrative, if `state.get("brand")` and `state.get("outcome_var")`, fetch context (sync service; wrap in `asyncio.to_thread` only if the node path is async — confirm at impl) and append one sentence: on-/off-label framing (outcome vs approved indication / limitation of use) + competitor count. **Wrapped in try/except → on any failure narrative is emitted unchanged.**

- [ ] **Step 1 — Red:** With brand+outcome + stub context → narrative contains the context sentence; with a raising context fetch → narrative identical to the no-context baseline (fail-open), node still returns. Run → fail.
- [ ] **Step 2 — Green:** Implement fail-open weave; do NOT alter ate/gate/recommendations.
- [ ] **Step 3 — Verify + Commit:** pytest `-n0` + scoped mypy. `feat(causal-impact): fail-open clinical-context sentence in interpretation`

---

### Task 8: Gap-analysis competitor density (surface-only)

**Files:**
- Modify: `src/agents/gap_analyzer/state.py` (`ROIEstimate`), `src/agents/gap_analyzer/nodes/roi_calculator.py`
- Test: `tests/unit/test_agents/test_gap_analyzer/test_roi_competitor_density.py` (create)

Add informational `competitor_products_count`, `competitor_density_label`, `competitor_drug_names` to `ROIEstimate`. Populate from the curated map for `state["brand"]` + gap outcome (reuse `_clinical_context_service`); fail-open to count 0 / "unknown". **Do NOT modify `risk_adjusted_roi` or the ranking.**

- [ ] **Step 1 — Red:** Bet carries competitor count/label; `risk_adjusted_roi` value is **unchanged** vs no-density baseline (assert equality); raising service → count 0, bet unaffected. Run → fail.
- [ ] **Step 2 — Green:** Implement surface-only; assert no ROI math change.
- [ ] **Step 3 — Verify + Commit:** pytest `-n0` + scoped mypy. `feat(gap-analyzer): surface competitor density on bets (no ROI change)`

---

### Task 9: Frontend

**Files:**
- Modify: `frontend/src/types/causal.ts`, `frontend/src/components/causal/ClinicalContextPanel.tsx`, `frontend/src/pages/CausalAnalysis.tsx`, gap-analysis bet component
- Test: co-located vitest specs

Types for `ApprovedIndications`/`CompetitorLandscape`/extended `ClinicalContext` + `DiscoveredEffect.clinical_context`. Panel: "Regulatory / Label" subsection (indications, limitations, boxed-warning w/ warning chip) + "Market landscape" (competitor chips + count), each with a source chip (`openfda`/`curated`/`static_fallback`). Leaderboard row: compact competitor-count + on/off-label badge. Gap bet: competitor-density badge.

- [ ] **Step 1 — Red:** Component tests (renderWithProviders/QueryClient): panel renders source chips + boxed-warning; leaderboard badge from `clinical_context`. Run → fail.
- [ ] **Step 2 — Green:** Implement.
- [ ] **Step 3 — Verify + Commit:** `tsc -b --noEmit` + `vitest run --no-file-parallelism --pool=forks`. `feat(causal-fe): label + competitor context in panel, leaderboard, gap bets`

---

### Task 10: Faithful live check + final review

- [ ] **Step 1:** Conftest-free script instantiating `_OpenFDAClient` + service against real OpenFDA for the 3 brands (no app import → no droplet OOM); confirm real indications/LoU/boxed-warning + curated competitors + honesty labels. Confirm Kisqali resolves the standalone label (not co-pack).
- [ ] **Step 2:** Full `ruff check src/ tests/` + `ruff format --check src/ tests/`; scoped mypy on all changed files (ceiling stays ≤61).
- [ ] **Step 3:** Final code review (superpowers:code-reviewer) against the spec; adversarial honesty pass (no fabricated label/competitor; every consumer fail-open; sources disclosed).
- [ ] **Step 4:** Open PR to main; verify CI green before merge.

---

## Notes
- Order is dependency-correct (1→4 build the provider; 5→8 consume it; 9 FE; 10 verify). Tasks 6/7/8 are independent of each other (parallelizable).
- Every consumer fail-open; every fragment source-tagged; never fabricate.
- DEFERRED (not in this plan): OpenFDA-automated competitors/UMLS; predictive/time-series/feature-importance/model-performance integration; FAERS.
