# Label-Gater: Indicated-Population Flag + De-Prioritization — Design

**Date:** 2026-06-20 · **Branch:** `feat/causal-label-gater` (stacked on `feat/causal-openfda-label-context`, PR #1055) · **Status:** proposed

## Problem & intent

This is a drug-**adoption** intelligence platform (README: "Causal Analytics for Pharmaceutical Drug Adoption"), not clinical decision support. The FDA label is a **strategic guardrail**: a commercial recommendation should not chase causal uplift into a population the drug's label does not support. Example (Remibrutinib/RHAPSIDO, CSU "in adults who remain symptomatic despite H1-antihistamine treatment"): a low-severity *treatment-naive* segment may show the highest CATE ("better responders"), but it is **off-label for promotion** — the gater must flag it and sink it in the ranking, reframing the bet toward the on-label antihistamine-refractory population (contested by Xolair/Dupixent).

User constraint (verbatim): *"nothing should be hardcoded, the labels should depend on the brands information as per the apis we used to extract this context and match it to the existing data."*

## Cheapest-disproof results (data, not theory)

1. **Real DB:** the entire synthetic population is pre-filtered to the indicated population — `prior_antihistamine_therapy` is constant `true` (8420/8420 Remibrutinib), `hr_status`/`her2_status` constant, every clinical column varies only *within* the indicated range. **Zero off-label patients exist today** → a gater would never fire without a data fix.
2. **Data-load path:** neither agent calls `cohort_constructor`; both load the **full brand population** from `patient_journeys`/`business_metrics` (no cohort pre-filter). `deployment_includes_synthetic()` is **true** (`.env E2I_INCLUDE_SYNTHETIC=true`), so `is_synthetic=true` rows ARE loaded. → off-label rows added as `is_synthetic=true` will reach CATE/gap segmentation.
3. **Criteria source:** deterministic extraction from real label text is **brittle** (misses UAS7/ECOG not stated numerically; mis-binds/duplicates Fabhalta's dual-indication thresholds). → do NOT extract criteria from free text for a high-stakes flag.

## Architecture

### 1. Criteria SSOT + label validation (honors "derive from API, no hardcode")
- **SSOT = `cohort_constructor.configs.get_brand_config(brand)`** — the existing, clinically-reviewed, column-bound `CohortConfig` (`Criterion(field, operator, value, type, clinical_rationale)`), already "Based on FDA label." The gater invents **no** criteria of its own (not a parallel dict).
- **New `LabelCriteriaProvider.derive(brand)`** (in `src/services/clinical_context/`, stacked on #1055):
  1. Resolve the brand's `CohortConfig` (SSOT criteria, bound to real columns).
  2. Fetch the **live OpenFDA label** via the existing `ClinicalContextProvider` pipeline (#1055).
  3. **Validate + provenance-tag** each criterion against the live label text: set `label_corroborated: bool` + `label_evidence: str` (matching snippet) where the disease / concept token appears; flag `drift` where the live label contradicts; overall `source ∈ {openfda_validated, cohort_config_only, unavailable}`.
  4. Fail-open: live label unreachable → `cohort_config_only`, honestly tagged (mirrors #1055).
- Returns `IndicatedPopulation(brand, criteria: List[GateCriterion], source, label_evidence_by_field)`.

### 2. Segment gate evaluation (deterministic, testable)
- **`label_gate.py`**: `evaluate_segment(segment_descriptor, indicated_population) -> GateVerdict`.
  - `segment_descriptor` = the column/value(s) defining a segment (e.g. `{"prior_antihistamine_therapy": False}` or a band `{"urticaria_severity_uas7": "<16"}`).
  - Verdict: `on_label | off_label | indeterminate` (+ `failed_criteria: List[str]`, human reason from `clinical_rationale`). A segment is `off_label` iff it provably violates an inclusion criterion (or hits an exclusion); `indeterminate` when the segment's columns don't intersect any criterion (never silently "off").
- No network in the evaluator; pure function over the criteria + descriptor.

### 3. Integration seams (flag + de-prioritize; surface-not-silent)
- **heterogeneous_optimizer** (`nodes/policy_learner.py`, before the rank/sort): add `brand` to state (currently dropped — `agent.py _build_initial_state`); for each segment, evaluate the gate; set `off_label`/`off_label_reason` on `PolicyRecommendation`; **de-prioritize** by sinking off-label segments below on-label (sort key `(not off_label, expected_incremental_outcome)`) — never delete, always surfaced.
- **gap_analyzer** (`nodes/prioritizer.py`, after causal/instrument adjustments, before sort): set `off_label_flag`/`off_label_reason` on `ROIEstimate`; de-prioritize identically. Surface-only on ROI value (no magic multiplier; ranking demotion only).
- Both fail-open: a gate error never breaks a recommendation/bet (mirrors #1055 + the existing competitor-density precedent).

### 4. Data fix (makes the gater fire on real data — additive, recoverable)
- Extend the synthetic DGP (`patient_generator.py`) to emit a realistic **on/off-label MIX** with the heterogeneity planted (off-label band has *higher* raw treatment response — the "better responder but off-label" story), driven by a config knob `off_label_fraction` (default 0 → existing behavior unchanged unless opted in).
- **Additive reseed**: ADD off-label rows (`is_synthetic=true`, a recoverable provenance marker) for the 3 brands; **leave existing on-label rows untouched** (gold-standard models trained on them stay valid). Recoverable by the marker. Document the seed + marker.
- Off-label columns set to violate the brand's inclusion criterion (e.g. CSU `prior_antihistamine_therapy=False` and/or `urticaria_severity_uas7<16`; Kisqali `hr_status='negative'`/early stage; Fabhalta `ldh_ratio<1.5`).

### 5. Brand-aware segmentation (so there ARE label-relevant segments)
- The agents only segment on caller-supplied `segment_vars`/`segments` (region/specialty today). Add **brand-aware default label-relevant segment columns** (resolved from the brand's `CohortConfig` fields that have data variance) so CATE/gap naturally produce on- vs off-label bands. Opt-in/defaulted; never forces label columns when the caller supplies their own.

### 6. Frontend (don't make me fish for it)
- Surface the off-label flag wherever recommendations/bets render: a clear "Off-label — <reason>" badge on the de-prioritized segment/bet, with the label-evidence snippet + source chip (openfda_validated/cohort_config) consistent with #1055's `ClinicalContextPanel`. Heterogeneous-optimizer segment table + gap-analysis bets.

## Test plan (TDD red-first, real results, no mocking)
- **Unit:** `LabelCriteriaProvider.derive` against **captured real-label fixtures** (real OpenFDA payloads, not mocks) — criteria match cohort_config; label_corroborated set; fail-open path. `evaluate_segment` truth table (on/off/indeterminate) per brand.
- **Integration (real DB, `E2I_DB_INTEGRATION=1`):** after the additive reseed, run each agent for a brand and assert an off-label band is present, flagged, and ranked below the on-label bands; assert the gate never empties or reorders on-label results incorrectly.
- **Faithful live:** `derive(brand)` against live OpenFDA for all 3 brands (source=openfda_validated).
- **FE:** vitest for the badge + honest empty/indeterminate state.

## Non-goals / honesty
- The gater does **not** alter ROI magnitude by a fabricated factor — de-prioritization is rank demotion only.
- It does **not** gate clinical care — it is a commercial-strategy guardrail.
- `indeterminate` segments are surfaced as such, never silently treated as off-label.

## Blast radius & mitigation
- Additive reseed (existing rows untouched, recoverable marker) minimizes impact on gold-standard models/other sessions. CI batched to the end; deploy gated on user authorization.

## Codex review — adopted revisions (2026-06-20, supersedes the above where in conflict)

A skeptical codex pass returned REVISE-BEFORE-BUILD with 4 HIGH findings; all adopted:

1. **Criteria source is label-PRIMARY (HIGH#1).** The criteria are not "cohort_config SSOT validated by label." Instead: the reviewed `CohortConfig` supplies the *candidate* (column+operator+value, the reviewed reconciliation cache), and the **live OpenFDA label is the gating authority** — a deterministic evidence-matcher confirms each candidate against the live label text (disease name, "adult", status tokens "HR-positive"/"HER2-negative", prior-therapy phrases "despite/inadequately controlled by … antihistamine", stated thresholds). Each criterion is tagged `label_evidenced` | `config_unconfirmed`. The gate may **hard** off-label ONLY on a `label_evidenced` inclusion violation; a `config_unconfirmed` violation surfaces as **review/indeterminate**, never a silent hardcoded flag. Label unreachable → `unavailable`, fail-open. (This deterministically honors "derive from API" — the label decides which criteria are active — without the brittle value-extraction the prototype disproved.)
2. **Indication-scoped, not brand-only (HIGH#2).** `IndicatedPopulation` and the agents carry `indication`; resolve it from the population's diagnosis distribution (BRAND_DIAGNOSIS) when unambiguous, else `indeterminate`. Fabhalta must not silently default to PNH.
3. **Brand-aware segmentation must be wired or it's inert (HIGH#3).** Add `brand`+`indication` to the segments request/state/output (state field done). A resolver injects the brand's label-relevant `CohortConfig` columns into `segment_vars` ONLY when present+variant in the loaded frame (opt-in/defaulted on the direct API), so on/off-label bands form.
4. **Carry the flag end-to-end (HIGH#4).** Extend API Pydantic response models + `_convert_policies`/`_convert_opportunities` + persisted analysis models + FE TS types + render, in the SAME change, with converter-level and UI tests.
5. **Scenario-isolated data fix (MED#5).** Off-label rows carry a `scenario_id`/marker; the standard loaders default-EXCLUDE the gater scenario (existing causal estimates / gold-std / eval untouched); the label-gater segmentation mode opts IN. One opt-in flag drives both label-segmentation columns and scenario inclusion. Recoverable by the marker.
6. **Structured `SegmentDescriptor` (MED#6).** `(field, operator, value|range, source)` not a stringified `segment_value`; gate verdict ∈ `{on_label, off_label, indeterminate, mixed}` (mixed = a banded segment straddling a threshold).
7. **Explicit ranking contract (LOW#7).** Partition by verdict first, preserve the existing metric within each partition; update `quick_wins`/`strategic_bets` + their tests in the same PR; document the rank-contract change.
