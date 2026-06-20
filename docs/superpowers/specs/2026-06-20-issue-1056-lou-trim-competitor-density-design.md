# Issue #1056 — OpenFDA clinical-context follow-ups (design)

**Date:** 2026-06-20
**Issue:** #1056 (deferred from PR #1055 `feat/causal-openfda-label-context`)
**Branch:** `fix/issue-1056-lou-competitor-density` (worktree off `origin/main` @ `e71d50bf`)

## Cheapest-disproof findings (verified against real code, not the issue text)

1. The issue references code that exists **only on `origin/main`** (PR #1055 / #1060), not on the
   stale `fix/home-dashboard-ux` branch the repo was checked out on. The worktree is branched off
   `origin/main` so the referenced symbols are present.
2. **Part 1 file/symbol confirmed:** `_OpenFDAClient.limitations_of_use` at
   `src/services/clinical_context/clients.py:395`.
3. **Part 2 premise is INCOMPLETE.** The issue says the FE doesn't show density "because the gaps FE
   types don't carry the fields" — implying FE-only work. But the API Pydantic response model
   `ROIEstimate` in `src/api/routes/gaps.py:186` does **not** declare the competitor fields, and
   `_convert_opportunities` (`:829`) does not pass them. Pydantic drops the extra keys, so the data
   **never reaches the FE**. FE-only changes would render a permanently-empty badge. The real fix is
   the full chain: **API schema + serializer → FE types → FE render.** (Real backend field names:
   `competitor_products_count`, `competitor_density_label`, `competitor_drug_names` — the issue's
   `_density_label`/`_drug_names` are shorthand.)

## Part 1 — Trim the Limitations-of-Use extraction

**Bug:** `limitations_of_use` returns `text[m.start():]` — the first "Limitations of Use" marker
through the **end** of `indications_and_usage[0]`. OpenFDA concatenates the Highlights summary and the
full-text section, so for RHAPSIDO the first marker sweeps a repeated indication block and a second
LoU copy.

**Real fixture** `tests/fixtures/openfda_labels/remibrutinib.json` `indications_and_usage[0]`:
> `… Limitations of Use: RHAPSIDO is not indicated for other forms of urticaria. RHAPSIDO ® is a
> kinase inhibitor indicated for the treatment of … ( 1 ) Limitations of Use: Not indicated for other
> forms of urticaria. ( 1 )`

**Target output:** `"Limitations of Use: RHAPSIDO is not indicated for other forms of urticaria."`

**Trim rule (pure index-based, preserves original formatting):** from the first marker, keep the run
of *limitation* sentences and stop at the earliest boundary:
- a **positive indication restart** — a sentence containing `indicated` but **not** `not indicated`
  (the duplicated indication; `is not indicated` is a limitation and is *kept*);
- a **duplicated** `Limitations of Use` marker;
- a full-text **subsection header** `\d+\.\d+ [A-Z]`.
Then strip a trailing Highlights reference tag `( N )` / `( N.N )`. Return `None` if empty.

**Invariants (unchanged):** fail-open contract; brands with **no** LoU marker (iptacopan, ribociclib)
still return `None`; the existing substring/bleed assertions still pass.

**Regression test:** load the real `remibrutinib.json` fixture and assert **exact** trimmed equality
(the existing `test_limitations_of_use_extracts_text` only asserts substring containment, so it does
not catch the over-grab).

## Part 2 — Surface competitor density end-to-end

Backend already computes density per bet (`roi_calculator._competitor_density`, surface-only, never
alters ROI or ranking). Make it reach and render in the FE:

1. **API schema** (`src/api/routes/gaps.py`): add `competitor_products_count: Optional[int]`,
   `competitor_density_label: Optional[str]`, `competitor_drug_names: Optional[List[str]]` to the
   Pydantic `ROIEstimate`, and map them in `_convert_opportunities` via `roi_data.get(...)` — mirroring
   the existing `off_label*` fields.
2. **FE types** (`frontend/src/types/gaps.ts`): add the three optional snake_case fields to
   `ROIEstimate` (the FE consumes raw snake_case — no camel mapping layer).
3. **FE component** `frontend/src/components/insights/CompetitorDensityBadge.tsx` (+ test): mirror
   `LabelGateBadge` and the causal `ClinicalContextPanel` "N rivals" treatment — `Building2` icon,
   `Market landscape (N rival/s)`, saturation label (`limited/moderate/crowded`), competitor name
   chips. **Honest empty state:** render nothing when count is `0`/absent or label is `unknown`.
4. **Wire** into the `GapAnalysis.tsx` opportunity card next to `LabelGateBadge`.

**Invariants:** surface-only — no reordering/re-ranking on FE or BE; honest empty/unknown state.

## Testing (TDD red-first, real data — no mocking of the unit under test)
- Backend: `test_openfda_client.py` regression (real fixture, exact); API route test asserting the
  three fields serialize through `_convert_opportunities`.
- FE: `CompetitorDensityBadge.test.tsx` (renders rivals + count; empty on 0/unknown);
  `GapAnalysis` assertion that a bet with density renders the badge.

## Non-goals
No change to ROI math, ranking, the fail-open contract, or no-LoU brands. No network calls added.
