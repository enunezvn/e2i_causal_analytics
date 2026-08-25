# Clinical-Context Narrative Distillation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** One LLM-distilled narrative on the causal-analysis drill-down that reads the specific causal result (ATE, CI, robustness gate) through the brand's clinical and competitive context, with today's fragment sections collapsing into an expandable "Sources & provenance" block.

**Architecture:** A new `src/insights/clinical_narrative.py` module (mirrors `src/insights/causal_discovery.py`: DSPy signature → `build_grounding` → guard → honest fallback), a new `POST /insights/clinical-narrative` endpoint in `src/api/routes/insights_strategic.py` that fetches the clinical facts SERVER-side from `ClinicalContextService` and caches per grounding-content in Redis, and frontend wiring: a `useClinicalNarrativeInsight` mutation auto-fired by `CausalAnalysisDetail`, rendered by `ClinicalContextPanel` as the lead with fragments collapsed. The spec is `docs/superpowers/specs/2026-08-24-clinical-narrative-distillation-design.md`.

**Tech Stack:** Python 3.12 / FastAPI / DSPy / Redis (backend); React 18 + TypeScript + TanStack Query + Radix Collapsible + Vitest (frontend).

---

## Environment ground rules (READ FIRST)

- **This box is PROD == DEV** (the droplet). Never run whole-tree `mypy src/` or whole-tree `pytest` here — CI is the arbiter. Every check below is SCOPED to the changed files.
- Python is **`./.venv/bin/python`** / **`./.venv/bin/pytest`** from the repo root (`/home/enunez/Projects/e2i_causal_analytics`). Bare `python3` lacks redis/supabase.
- Branch: all work goes on **`feat/clinical-narrative-distillation`** (already exists; the spec is committed there as `aefc50c1b`). Run `git branch --show-current` before EVERY commit and confirm it prints `feat/clinical-narrative-distillation`.
- Never squash-merge. Commits end with `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`.
- **Task 1 is a HARD GATE.** Stop after it and show the output to the user. Do not start Task 2 until the user approves the prototype narrative quality.
- `$SCRATCH` below means the session scratchpad directory (never commit files placed there). If you have no scratchpad path, use any directory OUTSIDE the repo.

---

## File structure

| File | Action | Responsibility |
|---|---|---|
| `$SCRATCH/clinical_narrative_prototype.py` | Create (NOT committed) | Task-1 disproof: real payload + draft signature → printed narrative |
| `src/insights/clinical_narrative.py` | Create | Signature, `build_grounding`, `build_result_only_grounding`, `fallback`, fabrication guard, `generate_insight` |
| `tests/insights/test_clinical_narrative.py` | Create | Unit tests for the module (fallback path is forced by `tests/insights/conftest.py`) |
| `src/api/routes/insights_strategic.py` | Modify | `ClinicalNarrativeRequest` + `POST /insights/clinical-narrative` |
| `tests/api/test_insights_strategic_routes.py` | Modify | Route tests (stubbed `ClinicalContextService.get_context`) |
| `frontend/src/types/generated/api.ts` | Regenerate | OpenAPI contract baseline (CI's verify-types gate diffs it byte-for-byte) |
| `frontend/src/types/insights.ts` | Modify | `ClinicalNarrativeRequest` interface |
| `frontend/src/api/insights.ts` | Modify | `getClinicalNarrativeInsight` |
| `frontend/src/api/insights.test.ts` | Modify | POST-shape + timeout test |
| `frontend/src/hooks/api/use-insights.ts` | Modify | `useClinicalNarrativeInsight` |
| `frontend/src/hooks/api/index.ts` | Modify | Re-export the new hook |
| `frontend/src/components/causal/ClinicalContextPanel.tsx` | Modify | `narrative`/`narrativeLoading` props; collapse fragments under "Sources & provenance" |
| `frontend/src/components/causal/ClinicalContextPanel.test.tsx` | Modify | Narrative / fallback / no-narrative rendering tests |
| `frontend/src/components/causal/CausalAnalysisDetail.tsx` | Modify | Auto-fire the narrative once context + result are ready (stale-scope guarded) |
| `frontend/src/components/causal/CausalAnalysisDetail.test.tsx` | Modify | Extend the `@/hooks/api` mock factory (MISSING MOCK = RUN-KILLER) |

**Not touched:** `src/services/clinical_context/` (fact layer), causal estimation/DAG code, `frontend/src/lib/api-schemas.ts` (verified: no insight endpoint is Zod-parsed — insights flow through plain typed axios, and the `ClinicalContext` GET wire shape is unchanged, so no Zod schema work exists for this feature).

---

### Task 1: Cheapest-disproof prototype (GATE — STOP after this task)

The single assumption this feature depends on: *a standard-tier LLM given these fragments + the result writes a materially better single narrative without fabricating.* Falsify it BEFORE building anything, against the REAL remibrutinib payload, with the real DSPy config on this box.

**Files:**
- Create: `$SCRATCH/clinical_narrative_prototype.py` (do NOT commit)

- [ ] **Step 1: Write the prototype script**

Save the following as `$SCRATCH/clinical_narrative_prototype.py`. It fetches the real payload through `ClinicalContextService` (live ChEMBL/CT.gov/PubMed/openFDA/Open Targets calls — this box has network), composes the draft grounding strings, runs the draft signature through `run_signature` (the real production LM path), and prints grounding + narrative + a fabrication scan.

```python
"""Cheapest-disproof prototype for the clinical-narrative distillation (spec
docs/superpowers/specs/2026-08-24-clinical-narrative-distillation-design.md).

Fetches the REAL clinical-context payload, composes the draft grounding
strings, runs the draft DSPy signature through the real default LM, and prints
grounding + narrative + a fabrication scan. NOT part of the product — never
commit this file. The composers here are the DRAFT of what Task 2 formalizes
under TDD; if the signature is iterated here, carry the final wording into
Task 2.

Run from the repo root:
  set -a; source .env; set +a
  ./.venv/bin/python $SCRATCH/clinical_narrative_prototype.py \
      --brand Remibrutinib --grain hcp \
      --treatment treatment_arm --outcome adopted \
      --ate 0.14 --ci-lower 0.05 --ci-upper 0.23 --gate proceed
"""

import argparse
import re
import sys

sys.path.insert(0, "/home/enunez/Projects/e2i_causal_analytics")

import dspy  # noqa: E402

from src.insights.common import run_signature  # noqa: E402
from src.insights.clinical_context import format_clinical_positioning  # noqa: E402
from src.services.clinical_context.service import ClinicalContextService  # noqa: E402


class ClinicalNarrativeSignature(dspy.Signature):
    """Write ONE flowing narrative (2-4 short paragraphs, no headings, no
    bullet lists) that reads a single causal analysis through the brand's
    clinical and competitive reality, for a pharma brand analyst.

    STRICTLY grounded: use ONLY the facts provided in the inputs. NEVER invent
    trial results, citations, PMIDs, NCT ids, numbers, competitors, or label
    claims. The ONLY numbers allowed are ones present in the inputs (the
    effect estimate, its confidence interval, and figures quoted verbatim
    inside the provided facts).

    When the analysis input says the treatment is a commercial lever
    (access/promotion), the mechanism, endpoints and label describe the
    THERAPY, never the lever — do not read them as evidence about the lever.

    Weave absences in honestly (no real-world evidence yet, outcome not
    mapped to any registered endpoint, evidence unavailable) instead of
    omitting them — an absence woven into the story is part of the story.

    The estimate comes from a SYNTHETIC patient cohort; the clinical and
    competitive context is REAL. Keep that boundary explicit and never
    present the estimate as clinical evidence."""

    analysis: str = dspy.InputField(desc="What this causal analysis asks: framing, treatment kind, grain")
    result: str = dspy.InputField(desc="The estimate: signed ATE, CI, robustness gate verdict, synthetic-cohort boundary")
    clinical_position: str = dspy.InputField(desc="Mechanism of action, approved indication verbatim, limitations of use, labeled target population / line of therapy")
    competitive_position: str = dspy.InputField(desc="Competitive framing for this analysis + the curated rival list")
    trial_endpoints: str = dspy.InputField(desc="Registered pivotal trial endpoint measures + whether OUR outcome maps to one")
    evidence: str = dspy.InputField(desc="Label considerations bearing on this outcome (or their honest absence), public-KG indication edge, real-world evidence or its honest absence")

    narrative: str = dspy.OutputField(desc="2-4 flowing paragraphs; every clause traceable to an input fact; absences woven in honestly")


GATE_PHRASES = {
    "proceed": "survived all robustness checks",
    "review": "needs review (mixed robustness)",
    "block": "failed robustness checks",
}
MAX_ENDPOINTS = 5
IDENTIFIER_PATTERNS = (
    re.compile(r"\bNCT\d{7,8}\b", re.IGNORECASE),
    re.compile(r"\bPMID[:\s]*\d{6,9}\b", re.IGNORECASE),
    re.compile(r"\b10\.\d{4,9}/[^\s\)\]]+"),
    re.compile(r"https?://\S+", re.IGNORECASE),
)


def compose(payload, grain, ate, lo, hi, gate):
    brand = str(payload.get("brand") or "")
    treatment = str(payload.get("our_treatment") or "")
    outcome = str(payload.get("our_outcome") or "")
    tc = payload.get("treatment_context") or {}

    parts = []
    if payload.get("analysis_framing"):
        parts.append(str(payload["analysis_framing"]))
    kind = tc.get("kind")
    label = tc.get("label") or treatment
    if kind == "commercial":
        parts.append(
            f"The treatment '{label}' is a commercial (access/promotion) lever, not a "
            "therapy: the clinical sources below describe the therapy, never this lever."
        )
    elif kind == "clinical_covariate":
        parts.append(f"The treatment '{label}' is a patient-state variable used as an observational treatment.")
    elif kind == "drug_therapy":
        parts.append(f"The treatment '{label}' is a therapy contrast — the clinical sources describe it directly.")
    parts.append(f"Analysis grain: {grain}.")
    analysis = " ".join(parts)

    if ate is None:
        est = f"No effect estimate was provided for {treatment} -> {outcome}."
    else:
        ci = f" [95% CI {lo:+.4f}, {hi:+.4f}]" if lo is not None and hi is not None else ""
        est = f"Estimated effect of {treatment} on {outcome}: ATE {ate:+.4f}{ci}."
    phrase = GATE_PHRASES.get((gate or "").lower())
    est += f" Robustness gate: {gate} — the estimate {phrase}." if phrase else " Robustness gate: not reported."
    result = est + (
        " The estimate comes from a synthetic patient cohort (gold-standard demo data); "
        "the clinical and competitive context is real."
    )

    ind = payload.get("approved_indications") or {}
    lines = [f"{payload.get('drug_name')} — {payload.get('disease')}."]
    mech = (payload.get("mechanism") or {}).get("mechanism_of_action")
    if mech:
        lines.append(f"Mechanism of action: {mech}.")
    if ind.get("indications"):
        lines.append("Approved indication (label, verbatim): " + " | ".join(ind["indications"]))
    if ind.get("limitations_of_use"):
        lines.append(f"Limitations of use: {ind['limitations_of_use']}")
    positioning = format_clinical_positioning(brand)
    if positioning:
        lines.append(positioning)
    clinical_position = " ".join(lines)

    ag = payload.get("analysis_grounding") or {}
    comp_lines = []
    if ag.get("competitive_context"):
        comp_lines.append(str(ag["competitive_context"]))
    rivals = list((payload.get("competitor_landscape") or {}).get("competitors") or [])
    if rivals:
        comp_lines.append("Curated rivals: " + "; ".join(rivals) + ".")
    competitive_position = " ".join(comp_lines) or "No competitive context is established for this analysis."

    eps = payload.get("pivotal_endpoints") or {}
    measures = [str(e.get("measure")) for e in (eps.get("endpoints") or []) if e.get("measure")]
    ep_lines = []
    if measures:
        extra = f" (+{len(measures) - MAX_ENDPOINTS} more)" if len(measures) > MAX_ENDPOINTS else ""
        ep_lines.append("Registered pivotal trial endpoint measures: " + "; ".join(measures[:MAX_ENDPOINTS]) + extra + ".")
    else:
        ep_lines.append("No registered trial endpoints are available for this brand.")
    mapped = payload.get("mapped_endpoint")
    if mapped:
        ep_lines.append(f"Our outcome '{outcome}' maps to the real endpoint: {mapped}.")
    else:
        ep_lines.append(f"Our outcome '{outcome}' is not mapped to any registered endpoint.")
    trial_endpoints = " ".join(ep_lines)

    ev_lines = []
    considerations = list(ag.get("label_considerations") or [])
    if considerations:
        for c in considerations:
            ev_lines.append(f"Label consideration ({c.get('section')}): {c.get('title')} — {c.get('detail')}")
    elif ind.get("source") == "openfda":
        ev_lines.append("The FDA label was read and carries nothing bearing on this outcome.")
    else:
        ev_lines.append("The FDA label could not be read for this analysis (curated fallback in use).")
    ce = payload.get("causal_evidence") or {}
    edge = ce.get("indication_edge")
    if edge:
        verb = "an approved therapy for" if edge.get("predicate") == "treats" else "in development for"
        ev_lines.append(
            f"Open Targets records {edge.get('drug_name')} as {verb} {edge.get('disease_name')} "
            f"(max clinical stage: {edge.get('max_clinical_stage')})."
        )
    if ce.get("note"):
        ev_lines.append(str(ce["note"]))
    rwe_titles = []
    for key in ("seminal_real_world_evidence", "real_world_evidence"):
        r = payload.get(key)
        if r and r.get("title"):
            pm = f" (PMID {r['pmid']})" if r.get("pmid") else ""
            rwe_titles.append(f"{r['title']}{pm}")
    if rwe_titles:
        ev_lines.append("Real-world evidence: " + " | ".join(rwe_titles))
    else:
        ev_lines.append(
            "No real-world evidence names this brand yet — expected for a recent approval; "
            "real-world evidence typically lags approval by years."
        )
    evidence = " ".join(ev_lines)

    return {
        "analysis": analysis,
        "result": result,
        "clinical_position": clinical_position,
        "competitive_position": competitive_position,
        "trial_endpoints": trial_endpoints,
        "evidence": evidence,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--brand", default="Remibrutinib")
    ap.add_argument("--grain", default="hcp")
    ap.add_argument("--treatment", default="treatment_arm")
    ap.add_argument("--outcome", default="adopted")
    ap.add_argument("--ate", type=float, default=0.14, help="Use the REAL drilled effect's ATE when available; the default is illustrative.")
    ap.add_argument("--ci-lower", type=float, default=0.05)
    ap.add_argument("--ci-upper", type=float, default=0.23)
    ap.add_argument("--gate", default="proceed")
    args = ap.parse_args()

    print("== Fetching REAL clinical-context payload (live fan-out; cold cache can take ~30-60s) ==")
    payload = ClinicalContextService().get_context(
        args.brand, args.outcome, treatment=args.treatment, include_causal_evidence=True
    )
    g = compose(payload, args.grain, args.ate, args.ci_lower, args.ci_upper, args.gate)

    print("\n== GROUNDING (what the LM sees; ATE/CI are the CLI args — illustrative unless you passed the real ones) ==")
    for k, v in g.items():
        print(f"\n[{k}]\n{v}")

    print("\n== Running the draft signature through the real default LM ==")
    pred = run_signature(ClinicalNarrativeSignature, **g)
    if pred is None:
        print("run_signature returned None — DSPy/LM not configured. Check .env was sourced.")
        sys.exit(1)
    narrative = str(getattr(pred, "narrative", "")).strip()
    print("\n== NARRATIVE ==\n")
    print(narrative)

    grounding_text = " ".join(g.values())
    fabricated = [
        m for pat in IDENTIFIER_PATTERNS for m in pat.findall(narrative) if m not in grounding_text
    ]
    print("\n== FABRICATION SCAN ==")
    print("FABRICATED identifiers:", fabricated or "none")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it against the real remibrutinib payload**

From the repo root:

```bash
set -a; source .env; set +a
./.venv/bin/python $SCRATCH/clinical_narrative_prototype.py \
    --brand Remibrutinib --grain hcp --treatment treatment_arm --outcome adopted \
    --ate 0.14 --ci-lower 0.05 --ci-upper 0.23 --gate proceed
```

Expected: the six grounding sections print with real remibrutinib facts (BTK inhibitor mechanism, UAS7/ISS7/HSS7 endpoints, RHAPSIDO indication, Xolair/Dupixent rivals, Open Targets edge, "No real-world evidence names this brand yet"), then a 2–4 paragraph narrative, then `FABRICATED identifiers: none`. If the real drilled effect's ATE/CI/gate are known from the live page, pass those instead of the illustrative defaults.

- [ ] **Step 3: Judge the output against the gate**

Verdict criteria (all three must hold):
1. No fabricated identifiers, competitors, trial results, or label claims (spot-check every named fact against the printed grounding).
2. Absences woven in (RWE absence, unmapped outcome), not omitted.
3. Materially better single read than the current 8 fragments (compare with the target-quality example in the spec).

Run it 2–3 times (`run_signature` uses the LM cache; add `lm_cache=False` to the `run_signature` call in the script for fresh samples) to see variance. If it fails, iterate ONLY the signature docstring/descs in the scratch script and re-run.

- [ ] **Step 4: STOP — show the user**

Paste the grounding + narrative + scan output to the user and get an explicit go/no-go. **Do not proceed to Task 2 without it.** If the output cannot be made non-fabricating and better-than-fragments after iteration, the spec says stop and revisit the deterministic-composer option — surface that to the user instead of continuing. If the signature wording changed during iteration, carry the final wording into Task 2's module verbatim.

---

### Task 2: `src/insights/clinical_narrative.py` — grounding composers (TDD)

**Files:**
- Create: `src/insights/clinical_narrative.py`
- Test: `tests/insights/test_clinical_narrative.py`

`tests/insights/conftest.py` already forces the no-LLM fallback path for every test in that directory (it patches `ensure_dspy_configured` to `False`) — the tests below are hermetic and offline by construction.

- [ ] **Step 1: Write the failing tests for `build_grounding` / `build_result_only_grounding`**

Create `tests/insights/test_clinical_narrative.py`:

```python
"""Unit tests for the clinical-narrative insight module (spec 2026-08-24).

The fallback path is forced by tests/insights/conftest.py (no live LLM). These
tests pin the DERIVED grounding strings — never bare booleans — so a silent
composition change fails loudly (wave-27: assert the derivation, not the
decision)."""

from types import SimpleNamespace

from src.insights import clinical_narrative


def _payload(**overrides):
    """A real-shaped ClinicalContextService.get_context payload (remibrutinib)."""
    base = {
        "brand": "Remibrutinib",
        "drug_name": "remibrutinib",
        "disease": "Chronic spontaneous urticaria",
        "our_outcome": "adopted",
        "our_treatment": "treatment_arm",
        "mapped_endpoint": None,
        "treatment_context": {
            "column": "treatment_arm",
            "label": "on remibrutinib therapy",
            "framing": "being on remibrutinib",
            "kind": "drug_therapy",
            "source": "curated",
        },
        "analysis_framing": "This analysis asks what being on remibrutinib does to prescriber adoption.",
        "analysis_grounding": {
            "label_considerations": [],
            "competitive_context": "At initiation the choice is between remibrutinib and two injectable biologics.",
            "note": None,
            "outcome_theme": None,
        },
        "mechanism": {
            "mechanism_of_action": "Bruton tyrosine kinase (BTK) inhibitor",
            "source": "chembl",
        },
        "pivotal_endpoints": {
            "endpoints": [
                {"measure": "Change from baseline in UAS7 at Week 12", "time_frame": "Week 12", "nct_id": "NCT05030311"},
                {"measure": "Change from baseline in ISS7 at Week 12", "time_frame": "Week 12", "nct_id": "NCT05030311"},
                {"measure": "Change from baseline in HSS7 at Week 12", "time_frame": "Week 12", "nct_id": "NCT05030311"},
            ],
            "source": "clinicaltrials.gov",
        },
        "real_world_evidence": None,
        "seminal_real_world_evidence": None,
        "approved_indications": {
            "indications": [
                "RHAPSIDO is indicated for the treatment of chronic spontaneous urticaria "
                "in adults who remain symptomatic despite H1 antihistamine treatment."
            ],
            "limitations_of_use": None,
            "boxed_warning": None,
            "source": "openfda",
        },
        "competitor_landscape": {
            "competitors": ["Xolair (omalizumab)", "Dupixent (dupilumab)"],
            "count": 2,
            "source": "curated",
        },
        "causal_evidence": {
            "status": "found",
            "indication_edge": {
                "predicate": "treats",
                "drug_id": "CHEMBL4650485",
                "drug_name": "remibrutinib",
                "disease_id": "EFO_0005854",
                "disease_name": "chronic spontaneous urticaria",
                "max_clinical_stage": "PHASE_3",
                "source": "open_targets",
            },
            "sources_unavailable": [],
            "citations": [],
            "note": None,
        },
        "honesty_label": "Effect estimate = a SYNTHETIC patient cohort ...",
    }
    base.update(overrides)
    return base


def _grounding(**overrides):
    kwargs = dict(grain="hcp", ate=0.14, ate_ci_lower=0.05, ate_ci_upper=0.23, gate_decision="proceed")
    kwargs.update({k: overrides.pop(k) for k in list(overrides) if k in kwargs})
    return clinical_narrative.build_grounding(_payload(**overrides), **kwargs)


class TestBuildGrounding:
    def test_result_string_pins_signed_ate_ci_and_gate_phrase(self):
        g = _grounding()
        assert "ATE +0.1400 [95% CI +0.0500, +0.2300]" in g["result"]
        assert "Robustness gate: proceed — the estimate survived all robustness checks." in g["result"]
        assert "synthetic patient cohort" in g["result"]

    def test_gate_phrases_review_block_and_missing(self):
        assert "needs review (mixed robustness)" in _grounding(gate_decision="review")["result"]
        assert "failed robustness checks" in _grounding(gate_decision="block")["result"]
        assert "Robustness gate: not reported." in _grounding(gate_decision=None)["result"]

    def test_missing_ate_is_reported_not_invented(self):
        g = _grounding(ate=None, ate_ci_lower=None, ate_ci_upper=None)
        assert "No effect estimate was provided for treatment_arm -> adopted." in g["result"]

    def test_analysis_carries_framing_kind_and_grain(self):
        g = _grounding()
        assert "prescriber adoption" in g["analysis"]
        assert "therapy contrast" in g["analysis"]
        assert "Analysis grain: hcp." in g["analysis"]

    def test_clinical_covariate_gets_the_observational_sentence(self):
        payload = _payload(
            treatment_context={
                "column": "disease_severity",
                "label": "high disease severity",
                "framing": "severe disease",
                "kind": "clinical_covariate",
                "source": "curated",
            }
        )
        g = clinical_narrative.build_grounding(
            payload, grain="patient", ate=0.05, ate_ci_lower=None, ate_ci_upper=None, gate_decision=None
        )
        assert (
            "The treatment 'high disease severity' is a patient-state variable "
            "used as an observational treatment." in g["analysis"]
        )

    def test_commercial_lever_gets_the_boundary_sentence(self):
        payload = _payload(
            treatment_context={
                "column": "copay_support",
                "label": "copay support active",
                "framing": "copay support",
                "kind": "commercial",
                "source": "curated",
            }
        )
        g = clinical_narrative.build_grounding(
            payload, grain="patient", ate=0.02, ate_ci_lower=None, ate_ci_upper=None, gate_decision="review"
        )
        assert "commercial (access/promotion) lever" in g["analysis"]
        assert "never this lever" in g["analysis"]

    def test_unmapped_outcome_is_stated(self):
        g = _grounding()
        assert "Our outcome 'adopted' is not mapped to any registered endpoint." in g["trial_endpoints"]
        assert "Change from baseline in UAS7 at Week 12" in g["trial_endpoints"]

    def test_mapped_outcome_names_the_endpoint(self):
        g = _grounding(mapped_endpoint="Treatment persistence / duration of therapy")
        assert (
            "Our outcome 'adopted' maps to the real endpoint: "
            "Treatment persistence / duration of therapy." in g["trial_endpoints"]
        )

    def test_endpoint_list_is_capped_with_honest_overflow(self):
        eps = [
            {"measure": f"Endpoint {i}", "time_frame": None, "nct_id": None} for i in range(7)
        ]
        g = _grounding(pivotal_endpoints={"endpoints": eps, "source": "clinicaltrials.gov"})
        assert "Endpoint 4" in g["trial_endpoints"]
        assert "Endpoint 5" not in g["trial_endpoints"]
        assert "(+2 more)" in g["trial_endpoints"]

    def test_rwe_absence_is_woven_not_blank(self):
        assert "No real-world evidence names this brand yet" in _grounding()["evidence"]

    def test_rwe_presence_carries_title_and_pmid(self):
        g = _grounding(
            real_world_evidence={
                "pmid": "35642282",
                "title": "CDK4/6 inhibitor treatment use in women with advanced breast cancer.",
                "journal": "J Oncol Pharm Pract",
                "pubdate": "2023 Jul",
                "doi": None,
                "url": "https://pubmed.ncbi.nlm.nih.gov/35642282/",
                "source": "pubmed",
                "search_term": None,
            }
        )
        assert "CDK4/6 inhibitor treatment use" in g["evidence"]
        assert "(PMID 35642282)" in g["evidence"]
        assert "No real-world evidence" not in g["evidence"]

    def test_label_read_vs_unreadable_are_different_claims(self):
        read = _grounding()  # openfda source, no considerations
        assert "The FDA label was read and carries nothing bearing on this outcome." in read["evidence"]
        unreadable = _grounding(
            approved_indications={
                "indications": ["curated indication text"],
                "limitations_of_use": None,
                "boxed_warning": None,
                "source": "static_fallback",
            }
        )
        assert "The FDA label could not be read for this analysis" in unreadable["evidence"]

    def test_label_considerations_render_verbatim(self):
        g = _grounding(
            analysis_grounding={
                "label_considerations": [
                    {
                        "title": "Antihistamine-refractory population",
                        "detail": "Indicated only after H1 antihistamines.",
                        "section": "indications",
                        "references": "1",
                        "source": "openfda",
                    }
                ],
                "competitive_context": None,
                "note": None,
                "outcome_theme": None,
            }
        )
        assert (
            "Label consideration (indications): Antihistamine-refractory population — "
            "Indicated only after H1 antihistamines." in g["evidence"]
        )

    def test_open_targets_edge_is_composed(self):
        g = _grounding()
        assert (
            "Open Targets records remibrutinib as an approved therapy for "
            "chronic spontaneous urticaria (max clinical stage: PHASE_3)." in g["evidence"]
        )

    def test_clinical_position_carries_moa_indication_and_positioning(self):
        g = _grounding()
        assert "Bruton tyrosine kinase (BTK) inhibitor" in g["clinical_position"]
        assert "RHAPSIDO is indicated" in g["clinical_position"]
        # Curated positioning from src.insights.clinical_context._CLINICAL_POSITIONING:
        assert "antihistamine-refractory, later-line population" in g["clinical_position"]

    def test_competitive_position_carries_framing_and_rivals(self):
        g = _grounding()
        assert "two injectable biologics" in g["competitive_position"]
        assert "Curated rivals: Xolair (omalizumab); Dupixent (dupilumab)." in g["competitive_position"]

    def test_grounding_chips(self):
        g = _grounding()
        chips = {c["label"]: c["value"] for c in g["grounding"]}
        assert chips["Brand"] == "Remibrutinib"
        assert chips["Analysis"] == "treatment_arm -> adopted"
        assert chips["Gate"] == "proceed"
        # chembl + clinicaltrials.gov + openfda live; RWE is None -> 3/4
        assert chips["Live sources"] == "3/4"


class TestResultOnlyGrounding:
    def test_marks_context_unavailable_and_still_pins_the_result(self):
        g = clinical_narrative.build_result_only_grounding(
            brand="Remibrutinib",
            grain="hcp",
            treatment="treatment_arm",
            outcome="adopted",
            ate=0.14,
            ate_ci_lower=0.05,
            ate_ci_upper=0.23,
            gate_decision="proceed",
        )
        assert g["context_unavailable"] is True
        assert "ATE +0.1400 [95% CI +0.0500, +0.2300]" in g["result"]
        assert "Causal analysis of treatment_arm -> adopted for Remibrutinib" in g["analysis"]
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
cd /home/enunez/Projects/e2i_causal_analytics
./.venv/bin/pytest tests/insights/test_clinical_narrative.py -x -q
```

Expected: `ModuleNotFoundError`/`AttributeError` — `src.insights.clinical_narrative` does not exist.

- [ ] **Step 3: Implement the module (composers + signature)**

Create `src/insights/clinical_narrative.py`. If Task 1 iterated the signature wording, use THAT wording; the structure below stays.

```python
"""Clinical-context narrative distillation for ONE causal analysis.

One flowing narrative that reads the specific causal result (signed ATE, CI,
robustness gate) through the brand's clinical and competitive context — the
drill-down panel's primary read (spec 2026-08-24). Mirrors causal_discovery.py:
DSPy signature guarded by import, build_grounding, honest fallback, and a
post-generation fabrication guard (identifiers not present in the grounding
reject the sample). Facts come from ClinicalContextService.get_context, fetched
SERVER-side by the route; this module never calls the network itself.

Unlike the digit-free executive-brief/HTE surfaces, this surface REPORTS effect
figures (the causal-discovery insight precedent) — digits are allowed.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Optional

from src.insights.clinical_context import format_clinical_positioning
from src.insights.common import run_signature

logger = logging.getLogger(__name__)

try:
    import dspy

    class ClinicalNarrativeSignature(dspy.Signature):
        """Write ONE flowing narrative (2-4 short paragraphs, no headings, no
        bullet lists) that reads a single causal analysis through the brand's
        clinical and competitive reality, for a pharma brand analyst.

        STRICTLY grounded: use ONLY the facts provided in the inputs. NEVER
        invent trial results, citations, PMIDs, NCT ids, numbers, competitors,
        or label claims. The ONLY numbers allowed are ones present in the
        inputs (the effect estimate, its confidence interval, and figures
        quoted verbatim inside the provided facts).

        When the analysis input says the treatment is a commercial lever
        (access/promotion), the mechanism, endpoints and label describe the
        THERAPY, never the lever — do not read them as evidence about the
        lever.

        Weave absences in honestly (no real-world evidence yet, outcome not
        mapped to any registered endpoint, evidence unavailable) instead of
        omitting them — an absence woven into the story is part of the story.

        The estimate comes from a SYNTHETIC patient cohort; the clinical and
        competitive context is REAL. Keep that boundary explicit and never
        present the estimate as clinical evidence."""

        analysis: str = dspy.InputField(
            desc="What this causal analysis asks: framing, treatment kind, grain"
        )
        result: str = dspy.InputField(
            desc="The estimate: signed ATE, CI, robustness gate verdict, synthetic-cohort boundary"
        )
        clinical_position: str = dspy.InputField(
            desc=(
                "Mechanism of action, approved indication verbatim, limitations of "
                "use, labeled target population / line of therapy"
            )
        )
        competitive_position: str = dspy.InputField(
            desc="Competitive framing for this analysis + the curated rival list"
        )
        trial_endpoints: str = dspy.InputField(
            desc="Registered pivotal trial endpoint measures + whether OUR outcome maps to one"
        )
        evidence: str = dspy.InputField(
            desc=(
                "Label considerations bearing on this outcome (or their honest absence), "
                "public-KG indication edge, real-world evidence or its honest absence"
            )
        )

        narrative: str = dspy.OutputField(
            desc="2-4 flowing paragraphs; every clause traceable to an input fact; absences woven in honestly"
        )

    DSPY_AVAILABLE = True
except ImportError:
    DSPY_AVAILABLE = False
    ClinicalNarrativeSignature = None  # type: ignore[assignment,misc]


_GATE_PHRASES = {
    "proceed": "survived all robustness checks",
    "review": "needs review (mixed robustness)",
    "block": "failed robustness checks",
}
# Mirrors the panel's MAX_ENDPOINTS_SHOWN: the endpoint list grounds outcome
# definitions, it is not a data table.
_MAX_ENDPOINTS = 5
_LIVE_SOURCES = {"chembl", "clinicaltrials.gov", "pubmed", "openfda"}
_SYNTHETIC_NOTE = (
    "The estimate comes from a synthetic patient cohort (gold-standard demo data); "
    "the clinical and competitive context is real."
)


def _result_sentence(
    treatment: str,
    outcome: str,
    ate: Optional[float],
    lo: Optional[float],
    hi: Optional[float],
    gate: Optional[str],
) -> str:
    if ate is None:
        est = f"No effect estimate was provided for {treatment} -> {outcome}."
    else:
        ci = f" [95% CI {lo:+.4f}, {hi:+.4f}]" if lo is not None and hi is not None else ""
        est = f"Estimated effect of {treatment} on {outcome}: ATE {ate:+.4f}{ci}."
    phrase = _GATE_PHRASES.get((gate or "").lower())
    if phrase:
        est += f" Robustness gate: {gate} — the estimate {phrase}."
    else:
        est += " Robustness gate: not reported."
    return est + " " + _SYNTHETIC_NOTE


def build_grounding(
    payload: dict[str, Any],
    *,
    grain: str,
    ate: Optional[float],
    ate_ci_lower: Optional[float],
    ate_ci_upper: Optional[float],
    gate_decision: Optional[str],
) -> dict[str, Any]:
    """Compose the six grounding strings from a ClinicalContextService payload
    + the caller-supplied result. Every string is honest about absences — the
    LM is instructed to weave them in, never to fill them."""
    brand = str(payload.get("brand") or "")
    treatment = str(payload.get("our_treatment") or "")
    outcome = str(payload.get("our_outcome") or "")
    tc = payload.get("treatment_context") or {}

    # -- analysis --------------------------------------------------------
    parts: list[str] = []
    if payload.get("analysis_framing"):
        parts.append(str(payload["analysis_framing"]))
    kind = tc.get("kind")
    label = tc.get("label") or treatment
    if kind == "commercial":
        parts.append(
            f"The treatment '{label}' is a commercial (access/promotion) lever, not a "
            "therapy: the clinical sources below describe the therapy, never this lever."
        )
    elif kind == "clinical_covariate":
        parts.append(
            f"The treatment '{label}' is a patient-state variable used as an observational treatment."
        )
    elif kind == "drug_therapy":
        parts.append(
            f"The treatment '{label}' is a therapy contrast — the clinical sources describe it directly."
        )
    parts.append(f"Analysis grain: {grain}.")
    analysis = " ".join(parts)

    # -- result ----------------------------------------------------------
    result = _result_sentence(treatment, outcome, ate, ate_ci_lower, ate_ci_upper, gate_decision)

    # -- clinical_position ----------------------------------------------
    ind = payload.get("approved_indications") or {}
    lines = [f"{payload.get('drug_name')} — {payload.get('disease')}."]
    mech = (payload.get("mechanism") or {}).get("mechanism_of_action")
    if mech:
        lines.append(f"Mechanism of action: {mech}.")
    if ind.get("indications"):
        lines.append("Approved indication (label, verbatim): " + " | ".join(ind["indications"]))
    if ind.get("limitations_of_use"):
        lines.append(f"Limitations of use: {ind['limitations_of_use']}")
    positioning = format_clinical_positioning(brand)
    if positioning:
        lines.append(positioning)
    clinical_position = " ".join(lines)

    # -- competitive_position -------------------------------------------
    ag = payload.get("analysis_grounding") or {}
    comp_lines: list[str] = []
    if ag.get("competitive_context"):
        comp_lines.append(str(ag["competitive_context"]))
    rivals = list((payload.get("competitor_landscape") or {}).get("competitors") or [])
    if rivals:
        comp_lines.append("Curated rivals: " + "; ".join(rivals) + ".")
    competitive_position = (
        " ".join(comp_lines) or "No competitive context is established for this analysis."
    )

    # -- trial_endpoints -------------------------------------------------
    eps = payload.get("pivotal_endpoints") or {}
    measures = [str(e.get("measure")) for e in (eps.get("endpoints") or []) if e.get("measure")]
    ep_lines: list[str] = []
    if measures:
        extra = f" (+{len(measures) - _MAX_ENDPOINTS} more)" if len(measures) > _MAX_ENDPOINTS else ""
        ep_lines.append(
            "Registered pivotal trial endpoint measures: "
            + "; ".join(measures[:_MAX_ENDPOINTS])
            + extra
            + "."
        )
    else:
        ep_lines.append("No registered trial endpoints are available for this brand.")
    mapped = payload.get("mapped_endpoint")
    if mapped:
        ep_lines.append(f"Our outcome '{outcome}' maps to the real endpoint: {mapped}.")
    else:
        ep_lines.append(f"Our outcome '{outcome}' is not mapped to any registered endpoint.")
    trial_endpoints = " ".join(ep_lines)

    # -- evidence --------------------------------------------------------
    ev_lines: list[str] = []
    considerations = list(ag.get("label_considerations") or [])
    if considerations:
        for c in considerations:
            ev_lines.append(
                f"Label consideration ({c.get('section')}): {c.get('title')} — {c.get('detail')}"
            )
    elif ind.get("source") == "openfda":
        # Provenance discrimination (#1767): an empty list under openfda means
        # "read, nothing bears"; under the fallback it means "could not read".
        ev_lines.append("The FDA label was read and carries nothing bearing on this outcome.")
    else:
        ev_lines.append("The FDA label could not be read for this analysis (curated fallback in use).")
    ce = payload.get("causal_evidence") or {}
    edge = ce.get("indication_edge")
    if edge:
        verb = "an approved therapy for" if edge.get("predicate") == "treats" else "in development for"
        ev_lines.append(
            f"Open Targets records {edge.get('drug_name')} as {verb} {edge.get('disease_name')} "
            f"(max clinical stage: {edge.get('max_clinical_stage')})."
        )
    if ce.get("note"):
        ev_lines.append(str(ce["note"]))
    rwe_titles: list[str] = []
    for key in ("seminal_real_world_evidence", "real_world_evidence"):
        r = payload.get(key)
        if r and r.get("title"):
            pm = f" (PMID {r['pmid']})" if r.get("pmid") else ""
            rwe_titles.append(f"{r['title']}{pm}")
    if rwe_titles:
        ev_lines.append("Real-world evidence: " + " | ".join(rwe_titles))
    else:
        ev_lines.append(
            "No real-world evidence names this brand yet — expected for a recent approval; "
            "real-world evidence typically lags approval by years."
        )
    evidence = " ".join(ev_lines)

    # -- chips -----------------------------------------------------------
    live = sum(
        1
        for s in (
            (payload.get("mechanism") or {}).get("source"),
            eps.get("source"),
            ind.get("source"),
            (payload.get("real_world_evidence") or {}).get("source"),
        )
        if s in _LIVE_SOURCES
    )
    chips = [
        {"label": "Brand", "value": brand},
        {"label": "Analysis", "value": f"{treatment} -> {outcome}"},
        {"label": "Gate", "value": str(gate_decision or "n/a")},
        {"label": "Live sources", "value": f"{live}/4"},
    ]

    return {
        "analysis": analysis,
        "result": result,
        "clinical_position": clinical_position,
        "competitive_position": competitive_position,
        "trial_endpoints": trial_endpoints,
        "evidence": evidence,
        "grounding": chips,
        "context_unavailable": False,
    }


def build_result_only_grounding(
    *,
    brand: str,
    grain: str,
    treatment: str,
    outcome: str,
    ate: Optional[float],
    ate_ci_lower: Optional[float],
    ate_ci_upper: Optional[float],
    gate_decision: Optional[str],
) -> dict[str, Any]:
    """Grounding for the fetch-failed path: the result is all we can honestly
    say. The route renders it through fallback() — never through the LM."""
    unavailable = "The clinical-context sources could not be fetched for this analysis."
    return {
        "analysis": f"Causal analysis of {treatment} -> {outcome} for {brand} at the {grain} grain.",
        "result": _result_sentence(treatment, outcome, ate, ate_ci_lower, ate_ci_upper, gate_decision),
        "clinical_position": unavailable,
        "competitive_position": unavailable,
        "trial_endpoints": unavailable,
        "evidence": unavailable,
        "grounding": [
            {"label": "Brand", "value": brand},
            {"label": "Analysis", "value": f"{treatment} -> {outcome}"},
            {"label": "Gate", "value": str(gate_decision or "n/a")},
            {"label": "Clinical context", "value": "unavailable"},
        ],
        "context_unavailable": True,
    }
```

(The guard, `fallback`, and `generate_insight` land in Task 3 — this step only needs the composers the Step-1 tests exercise.)

- [ ] **Step 4: Run the tests to verify they pass**

```bash
./.venv/bin/pytest tests/insights/test_clinical_narrative.py -q
```

Expected: all `TestBuildGrounding` + `TestResultOnlyGrounding` tests PASS.

- [ ] **Step 5: Commit**

```bash
git branch --show-current   # MUST print feat/clinical-narrative-distillation
git add src/insights/clinical_narrative.py tests/insights/test_clinical_narrative.py
git commit -m "feat(insights): clinical-narrative grounding composers (spec 2026-08-24)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 3: Fabrication guard + fallback + `generate_insight` (TDD)

**Files:**
- Modify: `src/insights/clinical_narrative.py` (append)
- Test: `tests/insights/test_clinical_narrative.py` (append)

- [ ] **Step 1: Write the failing tests**

Append to `tests/insights/test_clinical_narrative.py`:

```python
class TestFallback:
    def test_fallback_composes_the_grounding_strings(self):
        g = _grounding()
        out = clinical_narrative.fallback(g)
        assert out["is_fallback"] is True
        assert out["key_takeaways"] == []
        assert g["result"] in out["insight"]
        assert g["clinical_position"] in out["insight"]
        assert "(Factual summary — LLM narrative unavailable.)" in out["insight"]

    def test_result_only_fallback_says_sources_unfetchable(self):
        g = clinical_narrative.build_result_only_grounding(
            brand="Remibrutinib",
            grain="hcp",
            treatment="treatment_arm",
            outcome="adopted",
            ate=None,
            ate_ci_lower=None,
            ate_ci_upper=None,
            gate_decision=None,
        )
        out = clinical_narrative.fallback(g)
        assert out["is_fallback"] is True
        assert "could not be fetched" in out["insight"]
        # The unavailable filler strings must not be repeated as if they were facts.
        assert out["insight"].count("could not be fetched") == 1


class TestGenerateInsight:
    def test_no_lm_returns_fallback(self):
        # conftest forces ensure_dspy_configured -> False, so run_signature -> None.
        out = clinical_narrative.generate_insight(_grounding())
        assert out["is_fallback"] is True

    def test_good_narrative_passes(self, monkeypatch):
        monkeypatch.setattr(
            clinical_narrative,
            "run_signature",
            lambda *a, **k: SimpleNamespace(
                narrative="Remibrutinib, a BTK inhibitor, showed ATE +0.1400 on adoption."
            ),
        )
        out = clinical_narrative.generate_insight(_grounding())
        assert out["is_fallback"] is False
        assert out["insight"].startswith("Remibrutinib, a BTK inhibitor")
        assert out["key_takeaways"] == []

    def test_empty_narrative_falls_back(self, monkeypatch):
        monkeypatch.setattr(
            clinical_narrative, "run_signature", lambda *a, **k: SimpleNamespace(narrative="   ")
        )
        assert clinical_narrative.generate_insight(_grounding())["is_fallback"] is True

    def test_fabricated_pmid_is_rejected(self, monkeypatch):
        monkeypatch.setattr(
            clinical_narrative,
            "run_signature",
            lambda *a, **k: SimpleNamespace(
                narrative="A registry study (PMID 99999999) proved adoption doubles."
            ),
        )
        assert clinical_narrative.generate_insight(_grounding())["is_fallback"] is True

    def test_fabricated_nct_and_url_are_rejected(self, monkeypatch):
        for bad in (
            "See trial NCT99999999 for confirmation.",
            "Details at https://example.com/made-up.",
        ):
            monkeypatch.setattr(
                clinical_narrative, "run_signature", lambda *a, _b=bad, **k: SimpleNamespace(narrative=_b)
            )
            assert clinical_narrative.generate_insight(_grounding())["is_fallback"] is True

    def test_grounded_pmid_passes_the_guard(self, monkeypatch):
        g = _grounding(
            real_world_evidence={
                "pmid": "35642282",
                "title": "CDK4/6 inhibitor treatment use in women with advanced breast cancer.",
                "journal": None,
                "pubdate": None,
                "doi": None,
                "url": "https://pubmed.ncbi.nlm.nih.gov/35642282/",
                "source": "pubmed",
                "search_term": None,
            }
        )
        monkeypatch.setattr(
            clinical_narrative,
            "run_signature",
            lambda *a, **k: SimpleNamespace(
                narrative="Real-world use is documented (PMID 35642282) alongside the estimate."
            ),
        )
        out = clinical_narrative.generate_insight(g)
        assert out["is_fallback"] is False
```

- [ ] **Step 2: Run the tests to verify the new ones fail**

```bash
./.venv/bin/pytest tests/insights/test_clinical_narrative.py -q
```

Expected: Task-2 tests still pass; every `TestFallback`/`TestGenerateInsight` test FAILS with `AttributeError: ... has no attribute 'fallback'` / `'generate_insight'`.

- [ ] **Step 3: Implement guard + fallback + generate**

Append to `src/insights/clinical_narrative.py`:

```python
# The cheapest fabrication tell for this content type: a citation-shaped
# identifier (PMID / NCT id / DOI / URL) the grounding never contained.
# Plain numbers are NOT scanned — the ATE/CI digits are legitimate here.
_IDENTIFIER_PATTERNS = (
    re.compile(r"\bNCT\d{7,8}\b", re.IGNORECASE),
    re.compile(r"\bPMID[:\s]*\d{6,9}\b", re.IGNORECASE),
    re.compile(r"\b10\.\d{4,9}/[^\s\)\]]+"),
    re.compile(r"https?://\S+", re.IGNORECASE),
)

_GROUNDING_STRING_KEYS = (
    "analysis",
    "result",
    "clinical_position",
    "competitive_position",
    "trial_endpoints",
    "evidence",
)


def _fabricated_identifiers(narrative: str, g: dict[str, Any]) -> list[str]:
    grounding_text = " ".join(str(g.get(k, "")) for k in _GROUNDING_STRING_KEYS)
    return [
        m
        for pat in _IDENTIFIER_PATTERNS
        for m in pat.findall(narrative)
        if m not in grounding_text
    ]


def fallback(g: dict[str, Any]) -> dict[str, Any]:
    """Deterministic factual summary of the grounding strings. Public because
    the route calls it directly on the fetch-failed (result-only) path."""
    parts = [g["analysis"], g["result"]]
    if g.get("context_unavailable"):
        parts.append(
            "The clinical-context sources could not be fetched for this analysis, so no "
            "clinical or competitive read can be offered right now."
        )
    else:
        parts.extend(
            [g["clinical_position"], g["competitive_position"], g["trial_endpoints"], g["evidence"]]
        )
    parts.append("(Factual summary — LLM narrative unavailable.)")
    return {
        "insight": "\n\n".join(parts),
        "key_takeaways": [],
        "grounding": g["grounding"],
        "is_fallback": True,
    }


def generate_insight(g: dict[str, Any]) -> dict[str, Any]:
    pred = run_signature(
        ClinicalNarrativeSignature,
        analysis=g["analysis"],
        result=g["result"],
        clinical_position=g["clinical_position"],
        competitive_position=g["competitive_position"],
        trial_endpoints=g["trial_endpoints"],
        evidence=g["evidence"],
    )
    if pred is None:
        return fallback(g)
    narrative = str(getattr(pred, "narrative", "")).strip()
    if not narrative:
        return fallback(g)
    fabricated = _fabricated_identifiers(narrative, g)
    if fabricated:
        logger.warning("clinical narrative rejected — fabricated identifiers: %s", fabricated)
        return fallback(g)
    return {
        "insight": narrative,
        "key_takeaways": [],
        "grounding": g["grounding"],
        "is_fallback": False,
    }
```

Also fix the result-only fallback double-mention: in `build_result_only_grounding`, the filler strings would repeat inside `fallback` — but `fallback` already branches on `context_unavailable` and skips them, so the single-mention assertion holds. Verify this while implementing; if the test says otherwise, the branch is wrong, not the test.

- [ ] **Step 4: Run the tests to verify they pass**

```bash
./.venv/bin/pytest tests/insights/test_clinical_narrative.py -q
```

Expected: ALL tests PASS.

- [ ] **Step 5: Scoped lint + type check, then commit**

```bash
./.venv/bin/ruff check src/insights/clinical_narrative.py tests/insights/test_clinical_narrative.py
./.venv/bin/mypy --config-file pyproject.toml src/insights/clinical_narrative.py
git branch --show-current   # MUST print feat/clinical-narrative-distillation
git add src/insights/clinical_narrative.py tests/insights/test_clinical_narrative.py
git commit -m "feat(insights): clinical-narrative guard, fallback and generate_insight

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 4: `POST /insights/clinical-narrative` endpoint (TDD)

**Files:**
- Modify: `src/api/routes/insights_strategic.py`
- Test: `tests/api/test_insights_strategic_routes.py` (append)

The route file's autouse fixture already forces the LM fallback and overrides `require_analyst` — the new tests inherit both.

- [ ] **Step 1: Write the failing route tests**

Append to `tests/api/test_insights_strategic_routes.py`:

```python
def _clinical_payload():
    """Real-shaped ClinicalContextService payload (remibrutinib, trimmed)."""
    return {
        "brand": "Remibrutinib",
        "drug_name": "remibrutinib",
        "disease": "Chronic spontaneous urticaria",
        "our_outcome": "adopted",
        "our_treatment": "treatment_arm",
        "mapped_endpoint": None,
        "treatment_context": {
            "column": "treatment_arm",
            "label": "on remibrutinib therapy",
            "framing": "being on remibrutinib",
            "kind": "drug_therapy",
            "source": "curated",
        },
        "analysis_framing": "This analysis asks what being on remibrutinib does to prescriber adoption.",
        "analysis_grounding": None,
        "mechanism": {"mechanism_of_action": "Bruton tyrosine kinase (BTK) inhibitor", "source": "chembl"},
        "pivotal_endpoints": {
            "endpoints": [
                {"measure": "Change from baseline in UAS7 at Week 12", "time_frame": "Week 12", "nct_id": "NCT05030311"}
            ],
            "source": "clinicaltrials.gov",
        },
        "real_world_evidence": None,
        "seminal_real_world_evidence": None,
        "approved_indications": {
            "indications": ["RHAPSIDO is indicated for chronic spontaneous urticaria in adults."],
            "limitations_of_use": None,
            "boxed_warning": None,
            "source": "openfda",
        },
        "competitor_landscape": {
            "competitors": ["Xolair (omalizumab)", "Dupixent (dupilumab)"],
            "count": 2,
            "source": "curated",
        },
        "causal_evidence": None,
        "honesty_label": "Effect estimate = a SYNTHETIC patient cohort ...",
    }


_NARRATIVE_BODY = {
    "brand": "Remibrutinib",
    "grain": "hcp",
    "treatment": "treatment_arm",
    "outcome": "adopted",
    "ate": 0.14,
    "ate_ci_lower": 0.05,
    "ate_ci_upper": 0.23,
    "gate_decision": "proceed",
}


def test_clinical_narrative_fallback_grounds_in_server_fetched_facts(test_client, monkeypatch):
    # Facts are fetched SERVER-side: stub the service, not the request.
    monkeypatch.setattr(
        "src.services.clinical_context.service.ClinicalContextService.get_context",
        lambda self, brand, outcome, treatment=None, include_causal_evidence=False: _clinical_payload(),
    )
    r = test_client.post("/api/insights/clinical-narrative", json=_NARRATIVE_BODY)
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["is_fallback"] is True  # conftest forces the no-LM path
    # Pin the DERIVED grounding, not booleans: the fallback composes the strings.
    assert "ATE +0.1400 [95% CI +0.0500, +0.2300]" in data["insight"]
    assert "survived all robustness checks" in data["insight"]
    assert "Bruton tyrosine kinase (BTK) inhibitor" in data["insight"]
    assert "Xolair (omalizumab)" in data["insight"]
    chips = {c["label"]: c["value"] for c in data["grounding"]}
    assert chips["Analysis"] == "treatment_arm -> adopted"
    assert data["key_takeaways"] == []
    assert data["provenance"].startswith("LLM synthesis of the labeled clinical-context sources")


def test_clinical_narrative_unknown_brand_404(test_client):
    r = test_client.post(
        "/api/insights/clinical-narrative", json={**_NARRATIVE_BODY, "brand": "NotABrand"}
    )
    assert r.status_code == 404
    assert "NotABrand" in r.json()["detail"]


def test_clinical_narrative_fetch_failure_degrades_to_result_only(test_client, monkeypatch):
    def _boom(self, brand, outcome, treatment=None, include_causal_evidence=False):
        raise RuntimeError("upstream down")

    monkeypatch.setattr(
        "src.services.clinical_context.service.ClinicalContextService.get_context", _boom
    )
    r = test_client.post("/api/insights/clinical-narrative", json=_NARRATIVE_BODY)
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["is_fallback"] is True
    assert "could not be fetched" in data["insight"]
    # The result the caller supplied is still honestly summarized.
    assert "ATE +0.1400 [95% CI +0.0500, +0.2300]" in data["insight"]


def test_clinical_narrative_fallback_is_cached_briefly(test_client, monkeypatch):
    # Pin the degraded-TTL discipline (exec-brief/HTE precedent): a fallback is
    # transient — cache 300s, never the full hour.
    monkeypatch.setattr(
        "src.services.clinical_context.service.ClinicalContextService.get_context",
        lambda self, brand, outcome, treatment=None, include_causal_evidence=False: _clinical_payload(),
    )

    async def _no_cached(key):
        return None

    seen: dict = {}

    async def _capture(key, value, ttl_seconds=3600):
        seen["ttl"] = ttl_seconds
        seen["is_fallback"] = value.get("is_fallback")

    monkeypatch.setattr("src.api.routes.insights_strategic.cache_get", _no_cached)
    monkeypatch.setattr("src.api.routes.insights_strategic.cache_set", _capture)
    r = test_client.post("/api/insights/clinical-narrative", json=_NARRATIVE_BODY)
    assert r.status_code == 200, r.text
    assert seen == {"ttl": 300, "is_fallback": True}
```

- [ ] **Step 2: Run to verify they fail**

```bash
./.venv/bin/pytest tests/api/test_insights_strategic_routes.py -q -k clinical_narrative
```

Expected: FAIL with 404 on `/api/insights/clinical-narrative` (route does not exist).

- [ ] **Step 3: Implement the endpoint**

In `src/api/routes/insights_strategic.py`:

(a) Change the fastapi import at the top of the file:

```python
from fastapi import APIRouter, Depends, HTTPException
```

(b) Add the request model at the end of the `# ---- Request models` section (after `TreatmentEffectInsightRequest`):

```python
class ClinicalNarrativeRequest(BaseModel):
    """Caller supplies the SCOPE + the RESULT (the same trust model as
    CausalInsightRequest, which accepts caller effects); the clinical FACTS are
    fetched SERVER-side from ClinicalContextService, so a bogus scope can only
    produce an honest 404/absence — never a grounded-looking narrative from
    arbitrary caller data."""

    brand: str
    grain: str
    treatment: str
    outcome: str
    ate: float | None = None
    ate_ci_lower: float | None = None
    ate_ci_upper: float | None = None
    gate_decision: str | None = None
```

(c) Add the endpoint at the end of the file:

```python
# Bound the server-side clinical fan-out: with the causal-evidence block the
# cold-cache path is several live API calls (tens of seconds worst case). On
# timeout the narrative degrades honestly to the result-only fallback.
_CLINICAL_NARRATIVE_FETCH_TIMEOUT_S = 30.0


@router.post("/clinical-narrative", response_model=StrategicInsightResponse)
async def clinical_narrative_insight(
    req: ClinicalNarrativeRequest, user: dict[str, Any] = Depends(require_analyst)
) -> StrategicInsightResponse:
    """ONE flowing narrative reading THIS causal analysis (treatment -> outcome,
    signed ATE/CI, robustness gate) through the brand's clinical and competitive
    context — server-fetched from the labeled clinical-context sources (spec
    2026-08-24). Fragment provenance stays on the panel; this is the through-line."""
    from src.insights import clinical_narrative
    from src.services.clinical_context.brand_map import resolve_brand_profile
    from src.services.clinical_context.service import ClinicalContextService

    provenance = (
        "LLM synthesis of the labeled clinical-context sources; facts drawn only from them."
    )
    # Unknown brand -> 404 BEFORE any fan-out (matches GET /causal/clinical-context).
    try:
        resolve_brand_profile(req.brand)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Unknown brand '{req.brand}'") from None

    def _fetch() -> dict[str, Any]:
        # A fresh instance is fine: every cache in the service module is
        # module-level, so this shares the per-worker warm caches with the
        # panel's own GET /causal/clinical-context route.
        return ClinicalContextService().get_context(
            req.brand, req.outcome, treatment=req.treatment, include_causal_evidence=True
        )

    try:
        payload_ctx = await asyncio.wait_for(
            asyncio.to_thread(_fetch), timeout=_CLINICAL_NARRATIVE_FETCH_TIMEOUT_S
        )
        g = clinical_narrative.build_grounding(
            payload_ctx,
            grain=req.grain,
            ate=req.ate,
            ate_ci_lower=req.ate_ci_lower,
            ate_ci_upper=req.ate_ci_upper,
            gate_decision=req.gate_decision,
        )
    except Exception as e:  # noqa: BLE001 — degrade honestly, never 500
        logger.warning("clinical-narrative context fetch failed for %s: %s", req.brand, e)
        g = clinical_narrative.build_result_only_grounding(
            brand=req.brand,
            grain=req.grain,
            treatment=req.treatment,
            outcome=req.outcome,
            ate=req.ate,
            ate_ci_lower=req.ate_ci_lower,
            ate_ci_upper=req.ate_ci_upper,
            gate_decision=req.gate_decision,
        )
        return _finalize(clinical_narrative.fallback(g), provenance=provenance)

    # Key on the composed grounding strings: they encode treatment, outcome,
    # grain, the ATE/CI/gate AND the fragment content — so a narrative written
    # from a degraded-source payload is never served for the live payload, and
    # any fact change produces a fresh generation (sibling-route discipline).
    key = cache_key(
        "clinical-narrative",
        f"{req.brand}:{req.grain}",
        {
            "a": g["analysis"],
            "r": g["result"],
            "cp": g["clinical_position"],
            "co": g["competitive_position"],
            "te": g["trial_endpoints"],
            "ev": g["evidence"],
        },
    )
    cached = await cache_get(key)
    if cached is not None:
        payload = cached
    else:
        payload = await asyncio.to_thread(clinical_narrative.generate_insight, g)
        # A fallback marks a transient state (LM outage / guard-rejected
        # sample): cache briefly so the panel self-heals instead of pinning
        # the factual summary for the hour (exec-brief/HTE precedent).
        await cache_set(key, payload, ttl_seconds=300 if payload.get("is_fallback") else 3600)
    return _finalize(payload, provenance=provenance)
```

- [ ] **Step 4: Run to verify they pass**

```bash
./.venv/bin/pytest tests/api/test_insights_strategic_routes.py -q -k clinical_narrative
./.venv/bin/pytest tests/api/test_insights_strategic_routes.py -q   # no regression in siblings
```

Expected: all PASS.

- [ ] **Step 5: Scoped lint + type check, then commit**

```bash
./.venv/bin/ruff check src/api/routes/insights_strategic.py tests/api/test_insights_strategic_routes.py
./.venv/bin/mypy --config-file pyproject.toml src/api/routes/insights_strategic.py
git branch --show-current   # MUST print feat/clinical-narrative-distillation
git add src/api/routes/insights_strategic.py tests/api/test_insights_strategic_routes.py
git commit -m "feat(api): POST /insights/clinical-narrative — server-fetched facts, grounded narrative

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 5: Regenerate the OpenAPI type baseline

CI's `verify-types.yml` regenerates `frontend/src/types/generated/api.ts` from the backend schema and diffs it byte-for-byte against the committed baseline — a new endpoint with an uncommitted regeneration is a guaranteed CI red.

**Files:**
- Regenerate: `frontend/src/types/generated/api.ts`

- [ ] **Step 1: Regenerate**

```bash
cd /home/enunez/Projects/e2i_causal_analytics
make generate-types
```

Expected: `openapi.json` is written at the repo root (it is disposable — check whether it is gitignored with `git status --short openapi.json`; if untracked-and-ignored, leave it) and `frontend/src/types/generated/api.ts` gains the `/insights/clinical-narrative` path + `ClinicalNarrativeRequest` schema.

- [ ] **Step 2: Verify the diff is only the new endpoint**

```bash
git diff --stat frontend/src/types/generated/api.ts
git diff frontend/src/types/generated/api.ts | grep -c "clinical" 
```

Expected: the diff mentions `clinical-narrative` / `ClinicalNarrativeRequest`; no unrelated removals. (If unrelated drift appears, main's baseline was stale — note it in the commit message rather than reverting the drift, since CI regenerates from the same source.)

- [ ] **Step 3: Compile-check the generated file and commit**

```bash
cd frontend && npx tsc --noEmit --strict --skipLibCheck src/types/generated/api.ts && cd ..
git branch --show-current   # MUST print feat/clinical-narrative-distillation
git add frontend/src/types/generated/api.ts
git commit -m "chore(types): regenerate OpenAPI baseline for /insights/clinical-narrative

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 6: Frontend request type + API function + hook (TDD)

**Files:**
- Modify: `frontend/src/types/insights.ts`
- Modify: `frontend/src/api/insights.ts`
- Modify: `frontend/src/hooks/api/use-insights.ts`
- Modify: `frontend/src/hooks/api/index.ts`
- Test: `frontend/src/api/insights.test.ts`

- [ ] **Step 1: Write the failing API test**

Append inside the `describe('insights api', ...)` block in `frontend/src/api/insights.test.ts` (and add `getClinicalNarrativeInsight` to the import list from `'./insights'`):

```typescript
  it('POSTs scope+result to /insights/clinical-narrative with the extended timeout', async () => {
    const resp = {
      insight: 'x', key_takeaways: [], grounding: [], is_fallback: false,
      generated_at: 't', provenance: 'p',
    };
    const spy = vi.spyOn(apiClient, 'post').mockResolvedValue(resp as never);
    const out = await getClinicalNarrativeInsight({
      brand: 'Remibrutinib', grain: 'hcp', treatment: 'treatment_arm', outcome: 'adopted',
      ate: 0.14, ate_ci_lower: 0.05, ate_ci_upper: 0.23, gate_decision: 'proceed',
    });
    expect(spy).toHaveBeenCalledWith(
      '/insights/clinical-narrative',
      {
        brand: 'Remibrutinib', grain: 'hcp', treatment: 'treatment_arm', outcome: 'adopted',
        ate: 0.14, ate_ci_lower: 0.05, ate_ci_upper: 0.23, gate_decision: 'proceed',
      },
      // Cold scope = clinical fan-out server-side + LM; Redis caches per grounding.
      { timeout: 95000 }
    );
    expect(out.is_fallback).toBe(false);
  });
```

- [ ] **Step 2: Run to verify it fails**

```bash
cd /home/enunez/Projects/e2i_causal_analytics/frontend
npx vitest run src/api/insights.test.ts
```

Expected: FAIL — `getClinicalNarrativeInsight` is not exported.

- [ ] **Step 3: Implement type, API function, hook, and re-export**

In `frontend/src/types/insights.ts`, after the `TreatmentEffectInsightRequest` interface (keep neighbors' doc style):

```typescript
/** POST /insights/clinical-narrative — the caller supplies only the SCOPE +
 *  RESULT (same trust model as CausalInsightRequest); the clinical facts are
 *  fetched server-side from the labeled clinical-context sources. */
export interface ClinicalNarrativeRequest {
  brand: string;
  grain: string;
  treatment: string;
  outcome: string;
  ate?: number | null;
  ate_ci_lower?: number | null;
  ate_ci_upper?: number | null;
  gate_decision?: string | null;
}
```

In `frontend/src/api/insights.ts`: add `ClinicalNarrativeRequest` to the type import list, then append:

```typescript
export const getClinicalNarrativeInsight = (r: ClinicalNarrativeRequest) =>
  post<StrategicInsightResponse, ClinicalNarrativeRequest>(
    `${BASE}/clinical-narrative`,
    r,
    // Server fetches the clinical-context fan-out (cold: tens of seconds) then
    // runs the LM; Redis caches per (scope + grounding content). nginx allows 120s.
    { timeout: 95000 }
  );
```

In `frontend/src/hooks/api/use-insights.ts`: add `getClinicalNarrativeInsight` to the `@/api/insights` import, `ClinicalNarrativeRequest` to the `@/types/insights` import, then append:

```typescript
export const useClinicalNarrativeInsight = () =>
  useMutation<StrategicInsightResponse, ApiError, ClinicalNarrativeRequest>({
    mutationFn: getClinicalNarrativeInsight,
  });
```

In `frontend/src/hooks/api/index.ts`: add `useClinicalNarrativeInsight,` to the export list from `'./use-insights'` (the STRATEGIC INSIGHTS block).

- [ ] **Step 4: Run to verify it passes**

```bash
npx vitest run src/api/insights.test.ts
npm run typecheck
```

Expected: test PASSES; typecheck clean.

- [ ] **Step 5: Commit**

```bash
cd /home/enunez/Projects/e2i_causal_analytics
git branch --show-current   # MUST print feat/clinical-narrative-distillation
git add frontend/src/types/insights.ts frontend/src/api/insights.ts \
        frontend/src/api/insights.test.ts frontend/src/hooks/api/use-insights.ts \
        frontend/src/hooks/api/index.ts
git commit -m "feat(frontend): clinical-narrative insight request type, API fn, hook

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 7: `ClinicalContextPanel` — narrative lead + collapsed sources (TDD)

**Files:**
- Modify: `frontend/src/components/causal/ClinicalContextPanel.tsx`
- Test: `frontend/src/components/causal/ClinicalContextPanel.test.tsx`

- [ ] **Step 1: Write the failing tests**

Append to the `describe('ClinicalContextPanel', ...)` block in `ClinicalContextPanel.test.tsx` (the `FULL` fixture already exists at the top of the file). Add to the file's imports: `import userEvent from '@testing-library/user-event';` (v14 is installed).

```typescript
  const NARRATIVE = {
    insight:
      'Ribociclib, a CDK4/6 inhibitor, is approved for HR+/HER2- advanced breast cancer.\n\n' +
      'The estimate (+0.14, gate: proceed) survived all robustness checks.',
    key_takeaways: [],
    grounding: [],
    is_fallback: false,
    generated_at: '2026-08-24T00:00:00Z',
    provenance: 'LLM synthesis of the labeled clinical-context sources; facts drawn only from them.',
  };

  it('leads with the narrative and collapses fragments under Sources & provenance', () => {
    render(<ClinicalContextPanel context={FULL} narrative={NARRATIVE} />);
    // Narrative paragraphs render as the primary read, labeled as synthesized.
    expect(screen.getByText(/survived all robustness checks/)).toBeInTheDocument();
    expect(screen.getByText(/LLM-synthesized · sources below/)).toBeInTheDocument();
    expect(screen.getByText(/facts drawn only from them/)).toBeInTheDocument();
    // Fragments are collapsed by default: the MoA fragment is not in the DOM.
    expect(screen.getByRole('button', { name: /Sources & provenance/i })).toBeInTheDocument();
    expect(screen.queryByText('CDK4/6 inhibitor')).not.toBeInTheDocument();
    // The honesty label stays visible OUTSIDE the collapse.
    expect(screen.getByText(/SYNTHETIC/)).toBeInTheDocument();
  });

  it('expands the fragments when the sources trigger is clicked', async () => {
    render(<ClinicalContextPanel context={FULL} narrative={NARRATIVE} />);
    await userEvent.click(screen.getByRole('button', { name: /Sources & provenance/i }));
    expect(screen.getByText('CDK4/6 inhibitor')).toBeInTheDocument();
  });

  it('renders fragments expanded (no chip, no collapse) on a fallback narrative', () => {
    render(
      <ClinicalContextPanel context={FULL} narrative={{ ...NARRATIVE, is_fallback: true }} />
    );
    expect(screen.queryByText(/LLM-synthesized/)).not.toBeInTheDocument();
    expect(screen.queryByRole('button', { name: /Sources & provenance/i })).not.toBeInTheDocument();
    expect(screen.getByText('CDK4/6 inhibitor')).toBeInTheDocument();
  });

  it('renders fragments expanded exactly as before with no narrative at all', () => {
    render(<ClinicalContextPanel context={FULL} />);
    expect(screen.queryByText(/LLM-synthesized/)).not.toBeInTheDocument();
    expect(screen.getByText('CDK4/6 inhibitor')).toBeInTheDocument();
  });

  it('shows a loading shimmer while the narrative is pending, fragments still visible', () => {
    const { container } = render(
      <ClinicalContextPanel context={FULL} narrativeLoading />
    );
    expect(container.querySelector('.animate-pulse')).not.toBeNull();
    expect(screen.getByText('CDK4/6 inhibitor')).toBeInTheDocument();
  });
```

(If `@testing-library/user-event` is not installed, replace the expand test's click with `fireEvent.click` from `@testing-library/react` — check `frontend/package.json` first; use whatever sibling tests use.)

- [ ] **Step 2: Run to verify they fail**

```bash
cd /home/enunez/Projects/e2i_causal_analytics/frontend
npx vitest run src/components/causal/ClinicalContextPanel.test.tsx
```

Expected: new tests FAIL (unknown props / missing elements); the three pre-existing tests still pass.

- [ ] **Step 3: Implement the panel changes**

In `ClinicalContextPanel.tsx`:

(a) Add imports:

```typescript
import { ChevronRight, Sparkles } from 'lucide-react';  // merge into the existing lucide import
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from '@/components/ui/collapsible';
import type { StrategicInsightResponse } from '@/types/insights';
```

(b) Change the component signature:

```typescript
export function ClinicalContextPanel({
  context,
  narrative,
  narrativeLoading = false,
}: {
  context: ClinicalContext;
  /** LLM-synthesized single narrative for THIS analysis. A fallback response or
   *  an absent narrative renders the fragments expanded exactly as before —
   *  the no-regression path. */
  narrative?: StrategicInsightResponse | null;
  narrativeLoading?: boolean;
}) {
```

(c) Inside the component body (after the existing destructuring and `endpointsFromCtgov`), add:

```typescript
  // A fallback narrative is a factual summary of the same fragments — showing
  // it ABOVE the fragments would duplicate them, so fallback renders the
  // fragments expanded, exactly like no narrative at all.
  const hasNarrative = Boolean(narrative && !narrative.is_fallback && narrative.insight.trim());
```

(d) Wrap the existing fragment sections. Everything from the `{analysis_framing && (` block down to and including the `{competitor_landscape && ...}` block (i.e. every section EXCEPT the outer wrapper `<div>`, the header block, and the final honesty-label `<p>`) moves VERBATIM into a local variable declared just before the `return`:

```typescript
  const fragments = (
    <>
      {/* ... the existing sections, moved verbatim, in the same order:
          analysis framing, drug+MoA, mapped outcome, pivotal endpoints,
          analysis grounding, causal evidence, seminal RWE, live RWE,
          approved indications, competitor landscape ... */}
    </>
  );
```

(e) The returned JSX becomes:

```tsx
  return (
    <div className="space-y-4 rounded-md border p-4">
      {/* ... the existing header block, unchanged ... */}

      {/* Loading shimmer for the narrative; fragments stay visible so a failed
          fetch never leaves a hole (additive feature, no regression path). */}
      {narrativeLoading && !hasNarrative && (
        <div className="space-y-2">
          <div className="h-4 w-3/4 animate-pulse rounded bg-muted" />
          <div className="h-4 w-full animate-pulse rounded bg-muted" />
          <div className="h-4 w-5/6 animate-pulse rounded bg-muted" />
        </div>
      )}

      {hasNarrative && narrative && (
        <div className="space-y-2">
          <div className="flex items-center gap-2">
            <Sparkles className="h-3.5 w-3.5 text-muted-foreground" />
            <Badge variant="outline" className="text-xs">
              LLM-synthesized · sources below
            </Badge>
          </div>
          {narrative.insight.split(/\n{2,}/).map((para) => (
            <p key={para.slice(0, 40)} className="text-sm whitespace-pre-line">
              {para}
            </p>
          ))}
          <p className="text-xs text-muted-foreground">{narrative.provenance}</p>
        </div>
      )}

      {hasNarrative ? (
        <Collapsible>
          <CollapsibleTrigger className="group flex items-center gap-1 text-sm text-muted-foreground hover:text-foreground">
            <ChevronRight className="h-4 w-4 transition-transform group-data-[state=open]:rotate-90" />
            Sources &amp; provenance
          </CollapsibleTrigger>
          <CollapsibleContent className="mt-3 space-y-4">{fragments}</CollapsibleContent>
        </Collapsible>
      ) : (
        fragments
      )}

      {/* The synthetic/real honesty boundary — always shown, OUTSIDE the collapse */}
      <p className="border-t pt-3 text-xs text-muted-foreground">{context.honesty_label}</p>
    </div>
  );
```

Note: the fragment sections were previously direct children of a `space-y-4` div; when rendered expanded (`fragments` outside the Collapsible) they lose inter-section spacing since a fragment (`<>`) is not a flex/space container — the parent `space-y-4` still applies to the fragment's children because `space-y-*` targets all direct DOM children, and React fragments don't create DOM nodes. So spacing is preserved with NO extra wrapper. Inside `CollapsibleContent`, the `space-y-4` class on the content div restores the same rhythm.

- [ ] **Step 4: Run to verify all panel tests pass**

```bash
npx vitest run src/components/causal/ClinicalContextPanel.test.tsx
npm run typecheck
```

Expected: ALL tests PASS (the 3 pre-existing ones prove the no-narrative path is byte-compatible).

- [ ] **Step 5: Commit**

```bash
cd /home/enunez/Projects/e2i_causal_analytics
git branch --show-current   # MUST print feat/clinical-narrative-distillation
git add frontend/src/components/causal/ClinicalContextPanel.tsx \
        frontend/src/components/causal/ClinicalContextPanel.test.tsx
git commit -m "feat(frontend): ClinicalContextPanel narrative lead + collapsible sources

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 8: `CausalAnalysisDetail` — auto-fire the narrative (TDD)

**Files:**
- Modify: `frontend/src/components/causal/CausalAnalysisDetail.tsx`
- Test: `frontend/src/components/causal/CausalAnalysisDetail.test.tsx`

⚠️ **Run-killer warning:** `CausalAnalysisDetail.test.tsx` mocks the ENTIRE `@/hooks/api` module with an explicit factory (`vi.mock('@/hooks/api', () => ({ useClinicalContext: ... }))`). The component will now consume a SECOND hook from that module — if the factory does not provide it, every test in the file dies on `undefined is not a function`. Update the factory FIRST.

- [ ] **Step 1: Write the failing tests**

In `CausalAnalysisDetail.test.tsx`, replace the existing `vi.mock('@/hooks/api', ...)` block with:

```typescript
const mockNarrativeMutate = vi.fn();
const mockNarrativeReset = vi.fn();
vi.mock('@/hooks/api', () => ({
  useClinicalContext: vi.fn(() => ({ data: undefined })),
  useClinicalNarrativeInsight: vi.fn(() => ({
    data: undefined,
    isPending: false,
    mutate: mockNarrativeMutate,
    reset: mockNarrativeReset,
  })),
}));
```

Then add (import `useClinicalContext` is already imported at the top of the file; also add `beforeEach` to the vitest import if absent, and `beforeEach(() => { mockNarrativeMutate.mockClear(); mockNarrativeReset.mockClear(); })` inside the describe):

```typescript
  it('does NOT fire the narrative before the clinical context has loaded', () => {
    vi.mocked(useClinicalContext).mockReturnValue({ data: undefined } as never);
    renderWithProviders(<CausalAnalysisDetail result={RESULT} brand="Remibrutinib" />);
    expect(mockNarrativeMutate).not.toHaveBeenCalled();
  });

  it('fires the narrative exactly once when context + result are both ready', () => {
    vi.mocked(useClinicalContext).mockReturnValue({ data: CLINICAL } as never);
    const { rerender } = renderWithProviders(
      <CausalAnalysisDetail result={RESULT} brand="Remibrutinib" />
    );
    expect(mockNarrativeMutate).toHaveBeenCalledTimes(1);
    expect(mockNarrativeMutate).toHaveBeenCalledWith({
      brand: 'Remibrutinib',
      grain: 'patient',
      treatment: 'treatment_arm',
      outcome: 'persistent_180d',
      ate: 0.0875,
      ate_ci_lower: 0.0867,
      ate_ci_upper: 0.0884,
      gate_decision: RESULT.refutation.gate_decision ?? null,
    });
    // A re-render with the same result must not re-fire (keyed auto-fire).
    rerender(<CausalAnalysisDetail result={RESULT} brand="Remibrutinib" />);
    expect(mockNarrativeMutate).toHaveBeenCalledTimes(1);
  });

  it('does NOT fire without a brand (the narrative is brand-scoped)', () => {
    vi.mocked(useClinicalContext).mockReturnValue({ data: CLINICAL } as never);
    renderWithProviders(<CausalAnalysisDetail result={RESULT} />);
    expect(mockNarrativeMutate).not.toHaveBeenCalled();
  });
```

`CLINICAL` is a minimal `ClinicalContext` — add near `RESULT` (import the type: `import type { AgentCausalAnalysisResponse, ClinicalContext } from '@/types/causal';`):

```typescript
const CLINICAL: ClinicalContext = {
  brand: 'Remibrutinib',
  drug_name: 'remibrutinib',
  disease: 'Chronic spontaneous urticaria',
  our_outcome: 'persistent_180d',
  mapped_endpoint: null,
  mechanism: { mechanism_of_action: 'BTK inhibitor', source: 'chembl' },
  pivotal_endpoints: { endpoints: [], source: 'clinicaltrials.gov' },
  real_world_evidence: null,
  approved_indications: {
    indications: [], limitations_of_use: null, boxed_warning: null, source: 'openfda',
  },
  competitor_landscape: { competitors: [], count: 0, source: 'curated' },
  honesty_label: 'Effect estimate = a SYNTHETIC patient cohort.',
};
```

(Match the real `ClinicalContext` type — if the type requires fields this fixture omits (e.g. `treatment_context`, `analysis_framing`, `causal_evidence`, `analysis_grounding`, `seminal_real_world_evidence`, `our_treatment`), add them as `null`/`undefined`-valid values; `npm run typecheck` is the arbiter.)

- [ ] **Step 2: Run to verify the new tests fail**

```bash
cd /home/enunez/Projects/e2i_causal_analytics/frontend
npx vitest run src/components/causal/CausalAnalysisDetail.test.tsx
```

Expected: new tests FAIL (`mockNarrativeMutate` never called — the component doesn't use the hook yet). Pre-existing tests must still PASS (the widened mock factory keeps them alive).

- [ ] **Step 3: Implement the wiring**

In `CausalAnalysisDetail.tsx`:

(a) Extend imports (line 31 area):

```typescript
import { useClinicalContext, useClinicalNarrativeInsight } from '@/hooks/api';
```

Also ensure `useEffect`, `useRef`, `useState` are in the react import.

(b) Add a module-level map near the top (after `MAX_ENDPOINTS_SHOWN`-style constants / before the component):

```typescript
// The dataset each grain estimates over (mirrors the page's GRAINS list); the
// narrative endpoint wants the grain word, the result carries the dataset.
const DATASET_GRAIN: Record<string, string> = {
  patient_journeys: 'patient',
  hcp_adoption: 'hcp',
  nba_triggers: 'trigger',
};
```

(c) Inside the component, after the existing `const clinicalContext = useClinicalContext(...)` line:

```typescript
  // LLM narrative for THIS analysis: auto-fire once the clinical context AND
  // the result are both in (keyed so one distinct analysis fires exactly once,
  // not on every re-render). The scope tag suppresses a late response from a
  // previous analysis (mirrors the page's manualScope stale-scope guard).
  const narrativeInsight = useClinicalNarrativeInsight();
  const { mutate: generateNarrative, reset: resetNarrative } = narrativeInsight;
  const narrativeKeyRef = useRef<string | null>(null);
  const [narrativeScope, setNarrativeScope] = useState<string | null>(null);
  const narrativeKey = `${brand ?? ''}|${result.dataset}|${result.treatment_var}|${result.outcome_var}|${result.ate ?? 'null'}`;
  useEffect(() => {
    if (!brand || !clinicalContext.data) return;
    if (narrativeKeyRef.current === narrativeKey) return;
    narrativeKeyRef.current = narrativeKey;
    setNarrativeScope(narrativeKey);
    resetNarrative();
    generateNarrative({
      brand,
      grain: DATASET_GRAIN[result.dataset] ?? result.dataset,
      treatment: result.treatment_var,
      outcome: result.outcome_var,
      ate: result.ate ?? null,
      ate_ci_lower: result.ate_ci_lower ?? null,
      ate_ci_upper: result.ate_ci_upper ?? null,
      gate_decision: result.refutation.gate_decision ?? null,
    });
  }, [brand, clinicalContext.data, narrativeKey, result, generateNarrative, resetNarrative]);
  const narrativeInScope = narrativeScope === narrativeKey;
  const narrative = narrativeInScope ? narrativeInsight.data ?? null : null;
  const narrativeLoading = narrativeInScope && narrativeInsight.isPending;
```

(d) Change the panel render (line ~387):

```tsx
      {clinicalContext.data && (
        <ClinicalContextPanel
          context={clinicalContext.data}
          narrative={narrative}
          narrativeLoading={narrativeLoading}
        />
      )}
```

- [ ] **Step 4: Run to verify everything passes**

```bash
npx vitest run src/components/causal/CausalAnalysisDetail.test.tsx src/components/causal/ClinicalContextPanel.test.tsx
npm run typecheck
```

Expected: ALL tests PASS; typecheck clean.

- [ ] **Step 5: Commit**

```bash
cd /home/enunez/Projects/e2i_causal_analytics
git branch --show-current   # MUST print feat/clinical-narrative-distillation
git add frontend/src/components/causal/CausalAnalysisDetail.tsx \
        frontend/src/components/causal/CausalAnalysisDetail.test.tsx
git commit -m "feat(frontend): auto-fire clinical narrative on the causal drill-down

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 9: Verification sweep + PR

- [ ] **Step 1: Scoped backend checks (NOT whole-tree — CI is the arbiter)**

```bash
cd /home/enunez/Projects/e2i_causal_analytics
./.venv/bin/ruff check src/insights/clinical_narrative.py src/api/routes/insights_strategic.py
./.venv/bin/mypy --config-file pyproject.toml src/insights/clinical_narrative.py src/api/routes/insights_strategic.py
./.venv/bin/pytest tests/insights/test_clinical_narrative.py tests/api/test_insights_strategic_routes.py -q
```

Expected: clean / all PASS.

- [ ] **Step 2: Scoped frontend checks**

```bash
cd frontend
npm run typecheck
npx vitest run src/api/insights.test.ts \
    src/components/causal/ClinicalContextPanel.test.tsx \
    src/components/causal/CausalAnalysisDetail.test.tsx
npx eslint src/components/causal/ClinicalContextPanel.tsx \
    src/components/causal/CausalAnalysisDetail.tsx \
    src/api/insights.ts src/hooks/api/use-insights.ts src/types/insights.ts
cd ..
```

Expected: clean / all PASS.

- [ ] **Step 3: Push and open the PR (do NOT merge — CI + user review first)**

```bash
git branch --show-current   # MUST print feat/clinical-narrative-distillation
git push -u origin feat/clinical-narrative-distillation
gh pr create \
  --title "feat: clinical-context narrative distillation on the causal drill-down" \
  --body "$(cat <<'EOF'
## Summary
- One LLM-distilled narrative on the causal-analysis drill-down that reads the causal result (signed ATE, CI, robustness gate) through the brand's clinical and competitive context (spec: docs/superpowers/specs/2026-08-24-clinical-narrative-distillation-design.md)
- New src/insights/clinical_narrative.py (mirrors causal_discovery.py: signature -> build_grounding -> fabrication guard -> honest fallback) + POST /insights/clinical-narrative (facts fetched SERVER-side from ClinicalContextService; Redis-cached per grounding content)
- ClinicalContextPanel: narrative leads with an "LLM-synthesized" chip; today's fragment sections collapse into "Sources & provenance"; the synthetic/real honesty label stays always-visible outside the collapse; fallback/absent narrative renders the fragments expanded exactly as before (no regression path)
- The fact layer (src/services/clinical_context/) and the causal math are untouched

## Test plan
- [ ] tests/insights/test_clinical_narrative.py — grounding permutations (kinds, mapped/unmapped, RWE presence/absence, label read-vs-unreadable), guard rejections, fallback shape
- [ ] tests/api/test_insights_strategic_routes.py — server-fetched grounding pinned in the fallback text, unknown brand 404, fetch failure -> result-only fallback
- [ ] Frontend: panel narrative/fallback/no-narrative/loading tests; detail auto-fire keyed + stale-scope tests; insights API timeout test
- [ ] Post-merge live cert: open /causal-analysis (Remibrutinib, HCP grain), drill into an effect, verify the narrative leads, sources collapse, chip + honesty label present; verify a REAL narrative (is_fallback=false) lands with OPENAI/DSPy configured

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

- [ ] **Step 4: Report**

Report to the user: PR URL, test counts, and the reminder that the merge policy is merge-commit or rebase (never squash), and that post-merge live certification (drill-down on eznomics.site with the deployed image) is a separate step that must verify a NON-fallback narrative at least once, plus the fallback rendering by temporarily unsetting the LM key is NOT required (the unit tests cover it).

---

## Self-review notes (already applied)

- **Spec coverage:** LLM distillation via `src/insights/` (Tasks 2–4), result included (`_result_sentence`, Task 2), fragments collapse + honesty label outside (Task 7), server-side facts + 404 + result-only fallback + grounding-content cache key (Task 4), wire types via regenerated OpenAPI baseline (Task 5), hook auto-fire with stale-scope guard (Task 8), cheapest-disproof gate (Task 1). The spec's "Zod api-schemas + parse test" item was investigated and found VACUOUS for this feature: no insight endpoint is Zod-parsed (`frontend/src/lib/api-schemas.ts` contains no insight schema; insights use plain typed axios), the response reuses the existing `StrategicInsightResponse` wire shape unchanged, and the `ClinicalContext` GET is untouched — there is no schema to declare and no strip-risk. The spec's "MSW handler" concern maps to the REAL run-killer in this codebase: the `vi.mock('@/hooks/api')` factory in `CausalAnalysisDetail.test.tsx` (Task 8 Step 1 handles it first).
- **Type consistency:** `ClinicalNarrativeRequest` fields are identical (names, order, optionality) in the Pydantic model (Task 4), the TS interface (Task 6), the mutate payload (Task 8), and the route tests. `fallback` is public (the route calls it); `_fabricated_identifiers`, `_result_sentence`, `_GATE_PHRASES` are private. Grounding dict keys (`analysis`, `result`, `clinical_position`, `competitive_position`, `trial_endpoints`, `evidence`, `grounding`, `context_unavailable`) are consistent across Tasks 2/3/4 and the prototype.
- **Test-fidelity discipline:** every route/unit assertion pins a DERIVED string (the exact composed sentence), never `is not None`; the panel's pre-existing tests act as the broken-state check for the no-narrative path (they must keep passing untouched).
