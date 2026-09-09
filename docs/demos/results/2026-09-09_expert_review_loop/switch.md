# Lane 1 — the enforcement switch (Task 15 Step 4)

**Step 1 produced no REVIEW row**, so the switch (`CAUSAL_IMPACT_REQUIRE_DAG_APPROVAL=true`) was **not** flipped
on production: it only changes behaviour for a REVIEW-band estimate whose structure holds no active approval, and
there was no such run to exercise it against. The `.env` on the droplet is unchanged; the api container was not
recreated for this step.

## Impact table (same 11 questions, baseline image `f30e9e9df` vs deployed image `989eec83d`)

| question | baseline band | impact band | data_subset | bootstrap | sensitivity | review decision |
|---|---|---|---|---|---|---|
| copay_support → adherent_180d | proceed | proceed | skipped → passed (5 effects) | skipped → passed (20 effects) | warning | None |
| copay_support → low_gap_180d | proceed | proceed | skipped → passed (5 effects) | skipped → passed (20 effects) | warning | None |
| copay_support → persistent_180d | block | block | skipped → passed (5 effects) | skipped → passed (20 effects) | failed | pending_review |
| psp_enrolled → adherent_180d | block | block | skipped → passed (5 effects) | skipped → passed (20 effects) | failed | pending_review |
| psp_enrolled → persistent_180d | block | block | skipped → passed (5 effects) | skipped → passed (20 effects) | failed | pending_review |
| rep_detailing_high → treatment_initiated | block | block | skipped → passed (5 effects) | skipped → passed (20 effects) | failed | pending_review |
| sample_dropped → treatment_initiated | block | block | skipped → passed (5 effects) | skipped → passed (20 effects) | failed | pending_review |
| treatment_arm → persistent_180d | block | block | skipped → failed (5 effects) | skipped → passed (20 effects) | failed | pending_review |
| treatment_arm → treatment_initiated | proceed | proceed | skipped → passed (5 effects) | skipped → passed (20 effects) | passed | None |
| trigger_accepted → treatment_initiated | proceed | proceed | skipped → passed (5 effects) | skipped → passed (20 effects) | warning | None |
| urticaria_severity_uas7 → persistent_180d | proceed | proceed | skipped → passed (5 effects) | skipped → passed (20 effects) | warning | None |

REVIEW rows: []

Bands are identical to the baseline (5 PROCEED / 6 BLOCK / 0 REVIEW). What changed is the EVIDENCE: both
non-critical tests now score on 11/11 rows (`data_subset` 5 recorded effects per row, `bootstrap` 20 per row) instead
of being SKIPPED with 0 effects on 11/11. One `data_subset` FAILED (40 % CI coverage on `treatment_arm → persistent_180d`)
on a row that was already BLOCK on a critical sensitivity failure, so no band moved.

## The switch's executable demonstration

`tests/unit/test_agents/test_causal_impact/test_refutation_expert_review_enforcement_1971.py` (CI-green on merge
`989eec83d`): with the switch ON, a REVIEW-band structure with no active approval is withheld
(`status failed`, `current_phase awaiting_expert_review`, the `Estimate withheld: CAUSAL_IMPACT_REQUIRE_DAG_APPROVAL=true …`
message naming the review id and `POST /expert-reviews/{id}/resolve`); a rejection halts on every band it can halt on.

## The band-semantics question for the owner (spec §2, §9)

REVIEW is now reachable but requires, on one run: no critical FAILED, a sensitivity WARNING, and BOTH non-critical
tests FAILED (0.65 / 0.675 / 0.625). On the 11 live questions the PROCEED rows all had both non-critical tests PASSED,
so none came near it. The alternative rule — a sensitivity WARNING alone puts a run in REVIEW — would have moved
**34 of 49** historical PROCEED runs (spec §2) and, on this table, 4 of the 5 PROCEED rows (all except
`treatment_arm → treatment_initiated`, whose sensitivity PASSED). That is the decision the impact data was gathered
for; it is filed as issue (a) with this table and is not taken in this lane.
