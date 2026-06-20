# OpenFDA Clinical Context + Insight Integration — Design Spec

**Date:** 2026-06-20
**Branch/worktree:** `feat/causal-openfda-label-context` (`/home/enunez/Projects/wt_openfda`, off main `033764d1`)
**Status:** DRAFT — awaiting user review (no implementation yet)

**Goal:** Extend the existing `clinical_context` enrichment so every causal insight is grounded in (a) the **therapy label** (FDA-approved indications, limitations of use, boxed warning) and (b) the **market landscape** (curated, evidence-grounded competitors), and weave that context **into insight generation** on the causal-analysis page and gap-analysis — not merely display it.

**Architecture (one line):** The `clinical_context` service becomes a reusable brand+outcome context provider (2 new fragments: OpenFDA label + curated competitors); the API attaches it to leaderboard rows and the deep analysis response; the causal_impact interpretation node weaves a fail-open context sentence into the narrative; gap-analysis surfaces competitor density as a transparent ROI-risk factor.

**Tech stack:** Python 3.12, FastAPI, LangGraph, httpx (sync, wrapped in `asyncio.to_thread` at async boundaries), React/TypeScript. OpenFDA `drug/label.json` API (`OPENFDA_API_KEY` in `.env`).

---

## 0. What the cheapest-disproof probes established (evidence base)

Live probes against `api.fda.gov` + RxClass for our actual brands (not the sample's Keytruda/Lipitor):

| Brand | OpenFDA label | Limitations of Use | Boxed warning | Curated competitors (grounded) |
|---|---|---|---|---|
| **Kisqali** (ribociclib) | ✅ `KISQALI` via `generic_name:ribociclib` (⚠️ `brand_name:Kisqali` returns the **co-pack** `KISQALI FEMARA CO-PACK` first) | none | none | Ibrance (palbociclib), Verzenio (abemaciclib) — ATC L01EF CDK4/6i confirmed |
| **Fabhalta** (iptacopan) | ✅ `FABHALTA` (PNH + IgAN) | none | **✅ real** (serious infections) | Soliris (eculizumab), Ultomiris (ravulizumab), Empaveli (pegcetacoplan), Voydeya (danicopan) — ATC L04AJ complement inhibitors confirmed |
| **Remibrutinib** (RHAPSIDO) | ✅ via `generic_name:remibrutinib` (⚠️ `brand_name:Remibrutinib` 404s — brand is **RHAPSIDO**) | **✅ real** ("not indicated for other forms of urticaria") | Xolair (omalizumab), Dupixent (dupilumab) — both confirmed CSU-indicated via OpenFDA |

**Disproof that shaped the design:** the sample's automated ATC competitor chain is clinically **misleading** for 2/3 brands (Remibrutinib → broad "selective immunosuppressants" bucket of transplant drugs; Kisqali co-pack → no RxCUI/class). Hence **curated competitors**. Indications/LoU/boxed-warning are real and usable for all three → **OpenFDA real-first**.

**Two brand-mapping gotchas (must encode):**
- Search OpenFDA by **`generic_name` (INN)**, not the platform brand label. `drug_name`: Kisqali→`ribociclib`, Fabhalta→`iptacopan`, Remibrutinib→`remibrutinib`.
- Kisqali: among `generic_name:ribociclib` results, **prefer the record whose `openfda.generic_name == ["ribociclib"]` exactly** (the standalone KISQALI), not the co-pack (`"letrozole and ribociclib"`).

---

## 1. Decisions locked with the user

1. **Indications/Limitations/Boxed-warning** via OpenFDA, real-first with honest static fallback. (Task #10)
2. **Competitors** via a **curated, sourced map** — `source="curated"` (not OpenFDA/ATC). (Task #11)
3. **Insight-integration scope: causal page + gap-analysis** (the reusable provider lets predictive/others adopt later). (User's "all insights take into account label + market landscape")
4. Every fragment carries a `source`; the FE discloses provenance; **never fabricate** a label or a competitor. Synthetic-estimate / real-context honesty label preserved.

---

## 2. Components

### 2.1 OpenFDA client — `src/services/clinical_context/clients.py` (extend)
Add `_OpenFDAClient` alongside the ChEMBL/CT.gov/PubMed clients. Sync httpx (consistent with the existing clients; async boundary handled by the service's `asyncio.to_thread` wrap, already in the route).

Methods (all keyed on `drug_name` = INN):
- `fetch_label(drug_name) -> dict | None` — GET `drug/label.json?search=openfda.generic_name:"<drug_name>"&limit=5` (+ `api_key`); pick the result whose `openfda.generic_name` contains `drug_name` as a **single ingredient** (co-pack disambiguation); fall back to `brand_name` search; return `None` on 404/empty/error.
- `approved_indications(label) -> list[str]` — parse `indications_and_usage[0]`; split the indication bullets (best-effort: the leading "INDICATIONS AND USAGE" block, cut at "Limitations of Use").
- `limitations_of_use(label) -> str | None` — extract the substring from the `"Limitations of Use"` marker in `indications_and_usage[0]` (probe-confirmed: embedded, not a separate field). `None` if absent.
- `boxed_warning(label) -> str | None` — `label.get("boxed_warning", [None])[0]`.

Rate/robustness: 30s timeout, single attempt, fail-closed to `None` (provider converts to static fallback). No secret echo.

### 2.2 Two new providers — `src/services/clinical_context/providers.py` (extend)
Follow the existing `ClinicalContextProvider.enrich(profile) -> Fragment` pattern (idempotent degradation: `except Exception → static fallback`).

- **`OpenFDAIndicationsProvider`** → `IndicationsFragment(approved_indications, limitations_of_use, boxed_warning, source)`; `source ∈ {"openfda","static_fallback"}`. Real-first via `_OpenFDAClient`; fallback to `profile.indications_fallback` / `profile.limitations_fallback` / `profile.boxed_warning_fallback`.
- **`CuratedCompetitorProvider`** → `CompetitorFragment(competitors, count, source)`; **`source="curated"`** always (single source of truth). Reads `profile.competitor_map[profile.disease]`. Empty → `count=0` (honest, no fabrication). *No OpenFDA/ATC call* (per the disproof).

### 2.3 Brand map — `src/services/clinical_context/brand_map.py` (extend)
Add to `BrandClinicalProfile` (frozen dataclass): `indications_fallback: list[str]`, `limitations_fallback: str | None`, `boxed_warning_fallback: str | None`, `competitor_map: dict[str, list[str]]`. Populate `_STATIC_ENRICHMENT` for the 3 brands with the **grounded** values from §0 (curated competitors carry an inline comment citing the probe evidence — e.g., `# ATC L01EF CDK4/6 inhibitors`). Disease keys lower-cased for matching.

### 2.4 Service fan-out + result — `src/services/clinical_context/service.py` (extend)
- `_fan_out` now returns **5** fragments `(moa, eps, cite, indications, competitors)`. Cache key stays `(brand, disease)` — outcome-independent, so indications+competitors (brand/disease-scoped) fit the existing `_FRAGMENT_CACHE` self-heal exactly. `fully_live` now also requires `indications.source == "openfda"` (competitors are curated, so curated does **not** count against `fully_live` — it's the intended source, not a degradation). **Decision:** treat `competitors.source == "curated"` as a *live* state for honesty-label purposes (it is the chosen SSOT, not a fallback).
- `get_context(brand, outcome)` returns two new keys: `approved_indications {indications, limitations_of_use, boxed_warning, source}` and `competitor_landscape {competitors, count, source}`.

### 2.5 API schema — `src/api/schemas/causal.py` (extend)
Add `ApprovedIndications` + `CompetitorLandscape` Pydantic models; add optional fields to `ClinicalContext`. Add `clinical_context: Optional[ClinicalContext] = None` to `DiscoveredEffect` (None until estimated / unknown brand → honest omission).

### 2.6 Leaderboard attachment — `src/api/routes/causal.py`
In `_effect_from_agent_response()` (~1343–1372), after building the row, populate `clinical_context` via `await asyncio.to_thread(_clinical_context_service.get_context, brand, outcome)` wrapped in try/except (KeyError/unknown brand → leave `None`). The `_FRAGMENT_CACHE` keyed on (brand, disease) means the many candidate pairs for one brand trigger **one** live fan-out. **Fail-open:** a context failure never fails the effect row.

### 2.7 causal_impact interpretation — `src/agents/causal_impact/nodes/interpretation.py`
Weave a **fail-open** context sentence into the narrative after the robustness block (~line 255), before recommendations. Reads `state["brand"]`, `state["outcome_var"]`. Fetch via the sync service (the node path: confirm async-ness at implementation; if sync, call directly; if async, `asyncio.to_thread`). **Hard rule:** wrapped in try/except → on any failure the narrative is emitted unchanged (the causal estimate is never blocked or altered by clinical-context I/O). Sentence form: on-/off-label framing (does `outcome_var` map to an approved indication / collide with a limitation of use?) + competitor count.

> **Decision (approved 2026-06-20): ship both.** The structured attachment (§2.6) is the reliable backbone (always present for the FE); the narrative-weaving makes the generated text itself on-/off-label + competitor aware. Weaving is strictly fail-open — any context failure → narrative emitted unchanged, estimate never blocked.

### 2.8 gap-analysis competitor density — `src/agents/gap_analyzer/nodes/roi_calculator.py` + `state.py`
Add to `ROIEstimate` (all **informational** — no change to the ROI value or the ranking): `competitor_products_count: int`, `competitor_density_label: str` (`"open"` / `"moderate"` / `"crowded (>5)"`), `competitor_drug_names: list[str]`. Source from the curated competitor map for `state["brand"]` + the gap's outcome (reuse `_clinical_context_service`); fail-open to count 0 / `"unknown"`.

> **Decision (approved 2026-06-20): surface-only.** Competitor density is attached to each strategic bet so the human sees market saturation alongside the bet — it does **not** alter `risk_adjusted_roi` or the ranking. The mapper's silent `×0.7/×0.85` multipliers are unvalidated magic numbers and are rejected. A validated, transparent ROI adjustment is deferred until coefficients can be justified with data.

### 2.9 Frontend
- `ClinicalContextPanel.tsx`: new "Regulatory / Label" subsection (approved indications, limitations of use, boxed warning with a warning chip) + "Market landscape" subsection (competitor chips + count), each with a source chip (`openfda` / `curated` / `static_fallback`).
- Leaderboard row (`CausalAnalysis.tsx`): compact context affordance (competitor count + on/off-label badge) from `DiscoveredEffect.clinical_context`; full panel on drill-down.
- gap-analysis bets: competitor-density badge + raw-vs-adjusted ROI disclosure.
- Types in `frontend/src/types/causal.ts`; hook already exists (`useClinicalContext`).

---

## 3. Data flow

```
brand + outcome
  └─ ClinicalContextService.get_context(brand, outcome)
       └─ resolve_brand_profile(brand)  # brand→INN, disease, fallbacks, competitor_map
       └─ _fan_out(profile)  # cache key (brand, disease); 1 live fan-out per brand
            ├─ MoA (ChEMBL)            source: chembl | static_fallback
            ├─ Endpoints (CT.gov)      source: clinicaltrials.gov | static_fallback
            ├─ RWE (PubMed)            source: pubmed | pubmed_seed | unavailable
            ├─ Indications (OpenFDA)   source: openfda | static_fallback     [NEW]
            └─ Competitors (curated)   source: curated                       [NEW]
  └─ consumed by:
       ├─ leaderboard rows (DiscoveredEffect.clinical_context)         [presentation]
       ├─ causal_impact interpretation narrative (fail-open weave)     [insight]
       └─ gap-analysis ROI (competitor density, transparent)           [insight]
```

## 4. Honesty & error handling
- Real-first; every fragment carries `source`; FE discloses it. Never fabricate a label or competitor.
- All consumers (leaderboard, interpretation, gap ROI) are **fail-open**: a clinical-context failure degrades to honest omission and never breaks an estimate, a row, a narrative, or a bet.
- Boxed warning surfaced prominently (real safety signal — Fabhalta).
- Curated competitor lists are documented (inline evidence citations) and treated as SSOT, not as a degraded fallback.

## 5. Testing strategy (TDD, red-first)
- `_OpenFDAClient`: co-pack disambiguation (ribociclib→standalone KISQALI), brand→generic mapping (Remibrutinib→RHAPSIDO via generic), LoU extraction (RHAPSIDO present / Fabhalta absent), boxed-warning (Fabhalta present), 404/empty → None. (Mock httpx; one optional live-marked test.)
- Providers: real path + static-fallback path; curated competitor source is always `"curated"`; empty disease → count 0.
- Service: 5-fragment fan-out; cache reuse across two outcomes of one brand; `fully_live` accounting with curated competitors counted as live.
- Leaderboard/interpretation/gap: fail-open (inject a raising provider → row/narrative/bet unaffected); context present on happy path.
- FE: panel renders each source chip; leaderboard badge; vitest `--no-file-parallelism --pool=forks`.
- Faithful live check: conftest-free script hitting real OpenFDA for the 3 brands (no app import → no droplet OOM).

## 6. Out of scope / deferred
- OpenFDA-automated competitors / UMLS / RxClass (disproved as misleading for our brands).
- Wiring into predictive/time-series/feature-importance/model-performance (provider is reusable; deferred per scope decision).
- OpenFDA adverse-event (FAERS) signals.

## 7. Resolved decisions (user: "go ahead", 2026-06-20)
1. **gap-analysis: surface-only** — competitor density shown per bet; no ROI/ranking math change (no unvalidated multiplier). (§2.8)
2. **interpretation: both** — structured attachment + fail-open narrative weave. (§2.7)
3. **Co-pack: standalone** — use the standalone ribociclib KISQALI label; ignore `KISQALI FEMARA CO-PACK`.
