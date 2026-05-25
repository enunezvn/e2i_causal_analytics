# #502 golden-label adjudication (per #242 multi-model A/B)

Adjudicator: opus research agent (2026-05-25). READ-ONLY analysis; no JSON mutated.
Inputs: `tests/fixtures/causal_role_golden_set.json` + `/tmp/242_ab.json` (non-contaminated rows).

## Summary verdict table

| # | feature | gt | models | verdict | corrected role |
|---|---------|----|--------|---------|----------------|
| 1 | `ctdna_clearance_90d_flag_given_baseline_positive` | collider | mediator | **LABEL-CORRECT** | — |
| 2 | `best_recist_response_180d` | mediator | descendant | **LABEL-CORRECT** | — |
| 3 | `switch_ai_to_fulvestrant_within_365d_flag` | descendant | collider | **GENUINELY-AMBIGUOUS** | descendant defensible; collider defensible under explicit shared-cause DAG |
| 4 | `post_index_thrombocytopenia_event_180d_flag` | descendant | mediator | **LABEL-CORRECT** | — |
| 5 | `dose_reduced_after_grade3_neutropenia_flag` | collider | mediator | **LABEL-CORRECT** | — |
| 6 | `hepatotoxicity_grade3_post_index_flag` | descendant | mediator | **LABEL-CORRECT** | — |
| 7 | `qtc_prolongation_grade2_post_index_flag` | descendant | mediator | **LABEL-CORRECT** | — |
| 8 | `post_index_switch_to_alternative_biologic_180d_flag` | descendant | collider | **GENUINELY-AMBIGUOUS** | descendant defensible; collider defensible under explicit shared-cause DAG |

Cases 1–4 were the named candidates; cases 5–8 were found by scanning `/tmp/242_ab.json` for all non-contaminated rows where sonnet==opus==gpt5 != gt.

## The bottom line

**6 LABEL-CORRECT, 0 LABEL-WRONG, 2 GENUINELY-AMBIGUOUS. No golden label is wrong.** The "30% correlated-failure" rate is **real model failure, not label noise** — directly answering the experiment's open question.

All three frontier models share two systematic, compile-set-fixable blind spots:
- **(i) temporal-downstream → descendant collapse** (case 2): calling an on-path early surrogate a descendant because it is measured later.
- **(ii) treatment-caused-intermediate → mediator collapse** (cases 1, 4, 5, 6, 7): calling off-path AEs / conditioned / M-structure variables mediators without enumerating the second parent or applying the on-path-to-THIS-outcome test.

Because all three fail *correlated*, agreement-or-escalate ensembling ratifies the wrong answer via majority vote. **This is exactly why multi-vendor diversity (adding GPT-5) did not help** — the blind spot is task-structural, shared across vendors, not vendor-specific.

## Per-case reasoning (condensed)

- **Case 1 (ctDNA clearance | baseline positive → collider).** The name encodes a conditioning restriction (`_given_baseline_positive`). Clearance is a common effect of treatment effectiveness and tumor-shedding biology; conditioning within the baseline-positive stratum selects on that common effect → collider (Greenland-Pearl-Robins 1999, PMID 9888278). Models judged the raw construct ("clearance is downstream of treatment effect → mediator") and ignored the conditioning set. Provenance: PADA-1, PMID 35241469.
- **Case 2 (best RECIST response 180d → mediator).** Early objective response is on the path treatment→regression→delayed progression→PFS — it transmits the treatment effect. Models conflated temporal downstream-ness with off-path-ness → descendant. Provenance: MONALEESA-2, PMID 35263519.
- **Case 4 (thrombocytopenia → descendant).** BTK-inhibitor class AE; no mechanism transmits anti-urticaria effect to UAS7 → off-path → descendant. Provenance: PMID 38492772.
- **Case 5 (dose reduction after G3 neutropenia → collider).** Clean M-structure: treatment→neutropenia, and (oncologist style + frailty)→dose-reduction; conditioning opens a frailty/style backdoor → collider. Models gave the partial exposure-reduction mediator story and missed the second parent.
- **Cases 6, 7 (hepatotoxicity, QTc prolongation → descendant).** Ribociclib off-target AEs (CYP3A4 / hERG; PMID 35440873). Off the path to PFS efficacy → descendant. Same treatment-caused-intermediate→mediator slip.
- **Cases 3, 8 (treatment-switch features → descendant vs collider): GENUINELY-AMBIGUOUS.** Descendant reading: T→switch←Y_nonresponse. Collider reading: switch is a common effect of disease trajectory AND prescriber/payer/access, opening an M-structure on conditioning. The golden set itself endorses the M-structure logic for case 5 (dose-reduction=collider) yet labels the at-least-as-management-driven switch features descendant — an **internal inconsistency**. Practical pipeline impact is nil (descendant and collider both route to exclude-as-predictor), so no JSON change proposed — only flagged to #502.

## Recommendations

1. **#502 (golden set):** no corrections needed for correctness; resolve the internal descendant-vs-collider inconsistency for treatment-switch features (cases 3, 8) vs the dose-reduction collider precedent (case 5). Low priority — no pipeline impact.
2. **Compile-set hardening (the high-value output):** add discriminator examples that teach
   - (a) conditioning encoded in the name (`_given_baseline_positive`) ⇒ collider-conditioning,
   - (b) early on-path surrogates (RECIST / biomarker response) ⇒ mediator despite being measured later,
   - (c) the on-path-to-THIS-specific-outcome test so off-path AEs (hepatotox, QTc, thrombocytopenia) are reliably classed descendant, not mediator.
3. **#240 AC3.5 / #242:** the correlated failure is intrinsic and shared across vendors → multi-vendor agreement cannot be the gate's independence signal for these case types. This strengthens the case for a non-LLM independent check (structural/temporal prior) — pending the zero-shot de-confound run, which tests whether the shared *prompt* contributed on top of the shared blind spot.
