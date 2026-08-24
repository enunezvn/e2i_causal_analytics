# Clinical-Context Narrative Distillation — Design

**Date**: 2026-08-24
**Status**: Approved (design review with user; approach/result/fragments decisions confirmed)
**Page**: `/causal-analysis` — the per-effect drill-down (`CausalAnalysisDetail`)

## Problem

The ClinicalContextPanel on the causal-analysis drill-down renders the clinical /
competitive intelligence gathered from the external biomedical sources (ChEMBL,
ClinicalTrials.gov, PubMed, openFDA, Open Targets, Europe PMC, plus the curated
brand map) as ~8 disconnected sections: analysis framing, mechanism, trial
endpoints, label considerations, evidence block, RWE, approved use, market
landscape. Each fragment is individually honest and source-labeled, but the
panel has no through-line — the distillation is mediocre regardless of which
causal question is being interrogated. The panel also never sees the causal
result, so the clinical story and the estimate are two separate reads.

**Goal**: one narrative, distilled from the full multi-source context, that
reads the specific causal analysis (treatment → outcome, effect direction,
robustness gate) through the brand's clinical and competitive reality.

## Decisions (user-confirmed)

1. **Approach**: LLM distillation via the existing `src/insights/` strategic-
   insight machinery (DSPy signature → Redis cache → honest fallback). A
   deterministic composer was rejected (conditional-template prose is what
   produced the current feel); folding into the causal agent's interpretation
   node was rejected (crosses the additive-presentation-only boundary, loses
   per-analysis cacheability).
2. **Result included**: the narrative incorporates the ATE, CI, and refutation
   gate verdict — this is what makes it "support the narrative of the causal
   question". Digits are allowed: this surface reports effect figures, per the
   causal-discovery insight precedent (NOT the digit-free executive-brief/HTE
   guard regime).
3. **Fragments collapse**: the narrative is the primary read; the existing
   fragment sections move into a collapsed "Sources & provenance" section with
   every source chip preserved. The synthetic/real honesty label stays
   always-visible outside the collapse.

## Target quality (worked example — the reported remibrutinib case; the ATE
figure below is illustrative, the real one comes from the run)

> Remibrutinib (Rhapsido), an oral BTK inhibitor, is approved for chronic
> spontaneous urticaria in adults who remain symptomatic despite H1
> antihistamines — a later-line, antihistamine-refractory population. In that
> setting it competes at initiation against two injectable biologics, Xolair
> (omalizumab) and Dupixent (dupilumab), so which therapy a patient starts is
> itself confounding structure for this estimate. This analysis asks what being
> on remibrutinib does to prescriber adoption; the estimate (+0.14, gate:
> proceed) survived all robustness checks. The brand's pivotal trials measured
> urticaria activity (UAS7, ISS7, HSS7 at week 12), not adoption — the outcome
> is unmapped to any registered endpoint — and no real-world persistence
> literature names the brand yet, expected for a recent approval. The estimate
> therefore stands on the synthetic cohort alone, read against a real clinical
> position: the first oral option in a refractory population currently served
> by injectables.

Every clause traces to a fragment; absences (no RWE, unmapped outcome) are
woven in honestly rather than rendered as empty sections.

## Architecture

```
CausalAnalysisDetail (has result: ATE/CI/gate + brand/grain/treatment/outcome)
  ├─ useClinicalContext ──────────── GET /causal/clinical-context (unchanged)
  ├─ useClinicalNarrative (NEW) ──── POST /insights/clinical-narrative (NEW)
  │       fires once result + clinical context are both loaded (auto, shimmer)
  │                                    │
  │                                    ├─ server-side: ClinicalContextService
  │                                    │   .get_context(brand, outcome,
  │                                    │    treatment, include_causal_evidence
  │                                    │    =True)  (per-worker caches warm)
  │                                    ├─ src/insights/clinical_narrative.py
  │                                    │   build_grounding → run_signature
  │                                    │   → guard → fallback on any failure
  │                                    └─ Redis cache (1h), key includes a
  │                                        hash of the fragment content
  └─ ClinicalContextPanel
        ├─ narrative lead (LLM-synthesized chip + provenance line)
        ├─ "Sources & provenance" collapsible (today's fragment sections,
        │    default-collapsed with narrative, expanded on fallback/absence)
        └─ honesty label (always visible, outside the collapse)
```

The `src/services/clinical_context/` package stays the untouched fact layer.
Nothing here touches the causal math, adjustment sets, or estimation frames.

## Component 1 — `src/insights/clinical_narrative.py` (new)

Mirrors `src/insights/causal_discovery.py` exactly: DSPy signature guarded by
`try: import dspy`, a `build_grounding(...)`, a `_fallback(g)`, a
`generate_insight(g)` calling `run_signature` from `src/insights/common.py`.

**Signature inputs** (all composed strings, built in `build_grounding` from the
`ClinicalContextService.get_context` payload dict + caller-supplied result):

- `analysis`: the service's `analysis_framing` sentence + treatment kind
  (`drug_therapy` / `commercial` / `clinical_covariate`) + grain.
- `result`: ATE with sign, CI, gate verdict phrase (proceed → "survived all
  robustness checks", review → "needs review (mixed robustness)", block →
  "failed robustness checks"), plus the synthetic-cohort honesty statement.
- `clinical_position`: mechanism of action, the approved indication verbatim,
  limitations of use, and the curated `format_clinical_positioning(brand)`
  target-population/line-of-therapy text.
- `competitive_position`: the `analysis_grounding.competitive_context`
  initiation-choice framing + the curated rival list.
- `trial_endpoints`: the registered endpoint measures (capped) + whether OUR
  outcome is mapped to one of them ("mapped to X" / "not mapped to any").
- `evidence`: label considerations bearing on the outcome (or their honest
  absence, distinguishing "label read, nothing bears" from "label unreadable"),
  the Open Targets indication edge, RWE citation title(s) or the honest
  "no real-world evidence yet" state.

**Signature instructions** (the load-bearing part):

- Write ONE flowing narrative (2–4 short paragraphs, no headings, no bullet
  lists) that reads the causal result through the clinical and competitive
  context. STRICTLY grounded: use ONLY the facts provided; NEVER invent trial
  results, citations, numbers, competitors, or label claims.
- A commercial lever (kind=commercial) is not a therapy: the mechanism,
  endpoints and label describe the therapy, never the lever — do not read them
  as evidence about the lever (mirrors the panel's `TREATMENT_KIND_NOTE`).
- Weave absences in honestly (no RWE yet, outcome unmapped to registered
  endpoints, evidence unavailable) instead of omitting them.
- The estimate comes from a synthetic cohort; the clinical context is real —
  keep that boundary explicit, never oversell the estimate as clinical
  evidence.

**Output**: `narrative: str` only. No key-takeaway bullets — the deliverable is
one narrative.

**Post-generation guard** (proportionate, fail-closed to fallback):
- reject empty output;
- reject output containing PMID / NCT-id / DOI / URL patterns not present in
  the grounding strings (regex scan) — the cheapest fabrication tell for this
  content type.

**`_fallback(g)`**: deterministic factual summary of the grounding strings with
`is_fallback: True` (same shape as the causal-discovery fallback). The frontend
treats fallback/absent narrative identically: fragments render expanded as the
primary read (today's behavior).

## Component 2 — `POST /insights/clinical-narrative` (new, `insights_strategic.py`)

**Request model** `ClinicalNarrativeRequest`: `brand`, `grain`, `treatment`,
`outcome`, `ate: float | None`, `ate_ci_lower/upper: float | None`,
`gate_decision: str | None`. Caller supplies scope + result — the same trust
model as the existing causal-discovery insight (which accepts caller effects);
the clinical FACTS are fetched server-side so a bogus scope can only produce an
honest 404/absence, never a grounded-looking narrative from arbitrary data.
Auth: `require_analyst` (matches sibling insight endpoints).

**Server-side fetch**: `ClinicalContextService.get_context(brand, outcome,
treatment=treatment, include_causal_evidence=True)` off the event loop via
`asyncio.to_thread`, bounded by an overall timeout (the
`fetch_clinical_payload` precedent, but with treatment + evidence since this IS
the analyst-opened drill-down). Unknown brand → 404. Fetch failure → fallback
narrative from whatever the request carried (result-only), `is_fallback: True`.

**Caching**: `cache_key("clinical-narrative", brand, {...})` → Redis via
`cache_get`/`cache_set` (1h TTL). The key includes treatment, outcome, grain,
ATE rounded to 4dp, gate, AND a stable hash of the composed grounding strings —
so a narrative written from a degraded-source payload is never served for the
live payload or vice versa. Underlying fragment fetches are already cached
per-worker in the service.

**Response**: the existing `StrategicInsightResponse` (`insight`,
`key_takeaways` empty, `grounding` chips: brand, treatment→outcome, gate,
sources-live count, `is_fallback`, `provenance`). Provenance string: "LLM
synthesis of the labeled clinical-context sources; facts drawn only from them."

## Component 3 — Frontend

- **`useClinicalNarrative`** (new, `frontend/src/hooks/api/use-causal.ts` or
  sibling): TanStack mutation/query POSTing the new endpoint; fired by
  `CausalAnalysisDetail` once `clinicalContext.data` AND the result are
  available (auto-fire with the stale-scope guards the page already uses for
  the strategic insight; loading shimmer; no button).
- **`ClinicalContextPanel`** gains an optional `narrative` prop (+ loading /
  fallback flags):
  - narrative present → renders as the lead: paragraphs + a small
    "LLM-synthesized · sources below" chip + provenance line; the existing
    fragment sections wrap in a collapsible "Sources & provenance"
    (default-collapsed);
  - narrative absent/fallback/loading-failed → fragments render expanded
    exactly as today (no regression path);
  - the synthetic/real honesty label stays always-visible at the panel bottom,
    OUTSIDE the collapse.
- **Wire types**: new request/response flow through the generated OpenAPI
  types + `api-schemas.ts` Zod schemas (WireSchemas strip unknown keys — new
  fields must be declared) + a parse test; `verify-types.yml` gates drift.
- **MSW**: handler for the new endpoint in the test setup (a missing handler is
  a run-killer on this box).

## Error handling summary

| Failure | Behavior |
|---|---|
| LLM unavailable / empty / guard-rejected | `is_fallback: true`; frontend shows fragments expanded (today's rendering) |
| Clinical-context fetch fails server-side | fallback narrative from result-only grounding, `is_fallback: true` |
| Unknown brand | 404 (matches the clinical-context endpoint) |
| Narrative endpoint unreachable from frontend | panel renders fragments expanded; no error banner needed (additive feature) |

## Testing

**Backend** (`tests/unit/insights/test_clinical_narrative.py` + route test):
- `build_grounding` permutations: drug_therapy vs commercial lever vs clinical
  covariate; mapped vs unmapped outcome; RWE present/absent; label
  considerations present / "nothing bears" / "label unreadable"; degraded
  sources reflected in the grounding hash.
- Guard: fabricated-PMID/NCT/DOI output rejected → fallback; grounding-present
  identifiers pass.
- Fallback shape and `is_fallback` flag.
- Endpoint: stubbed `run_signature`; cache hit/miss; unknown brand 404; fetch
  failure → result-only fallback. Assert the DERIVED grounding strings, not
  just the boolean verdict (wave-27 lesson: print what was computed).
- Tests must fail on the broken state first (pin values, never `is not None`).

**Frontend**: panel with narrative (collapsed sources, chip, honesty label
visible), without narrative (today's snapshot), fallback flag; hook fire
conditions (only after both inputs ready; stale-scope guard); Zod parse test.

## Scope & non-goals

- Only the causal-analysis drill-down panel. Leaderboard, gap-analysis page,
  executive brief, HTE and the causal agent's own interpretation are unchanged.
- `src/services/clinical_context/` is not modified (fact layer stays as-is).
- No change to causal estimation, adjustment sets, or the DAG.
- No streaming; the narrative is a single cached response.

## Implementation step 1 — cheapest disproof (gate before wiring)

The single assumption: a standard-tier LLM given these fragments + result
writes a materially better single narrative without fabricating. Before any
endpoint/frontend work: scratch-run the draft signature against the REAL
remibrutinib payload (the reported case) on this box with the real DSPy config
(`src/optimization/dspy_lm.py` default), inspect the actual output with the
user. Iterate on the signature there; only then build around it. If the output
cannot be made non-fabricating and better-than-fragments, stop and revisit the
deterministic-composer option.
