# Lane 1 — live adjudications through the UI linked-review card (Task 15 Steps 2–3)

Operator: `admin@e2i.local` (role `admin`, no `user_metadata.name`), logged in on the deployed bundle
`ghcr.io/enunezvn/e2i-frontend:989eec83d…`, page `https://eznomics.site/expert-reviews?review=<id>`.
The chrome-devtools MCP server dropped mid-session (memory pressure on the droplet); the clicks were
driven through the same headless Chrome over raw CDP (`Runtime.evaluate` on the rendered React form —
checkbox clicks, the textarea's native value setter + `input` event, the Approve / Reject button click).
Screenshots: `linked_card_pending_4eab7033.png`, `linked_card_approved_4eab7033.png`, `linked_card_rejected_2f79f11f.png`.

## Step 2 — approve the probe row, re-run its pair

Review `4eab7033-7422-422d-83f6-659c9c3b9987` (brand-less, `treatment_arm → persistent_180d`, hash `475f8939b9fd…`,
created 2026-09-08 by the plan's adversarial-review probe). Card before: `pending`, `no brand`, agent assessment
present (3 × supports, 1 unclear, 2 no_evidence), history row with no reviewer.

Ticked the three checklist items the assessment marked `supports` (confounders included, no forbidden edges,
intermediates positioned); comment `lane-1 live verification: structural approval`; **Approve** at 20:14:21Z.

- Card after: `approved` · Reviewer `admin@e2i.local` · Decided `2026-09-09` · Valid until `2026-12-08` · the
  comment · "This review is resolved. Its approval applies until it expires or a newer review rejects the structure."
  History row: `approved`, reviewer `admin@e2i.local`.
- Row (`public.expert_reviews`): `approval_status=approved`, `reviewer_name=admin@e2i.local` (name → email fallback
  because the operator has no `user_metadata.name`), `reviewer_email=admin@e2i.local`,
  `resolved_at = approved_at = updated_at = 2026-09-09 20:14:21.205443+00`, `reviewer_id` unchanged
  (`674e0396-…`, the requester breadcrumb), `checklist_json = {"conf_complete": true, "no_forbidden": true,
  "mediators_correct": true}`, `comments_json = {"note": "lane-1 live verification: structural approval"}`.
  Both JSON columns landed as JSON **strings** (`jsonb_typeof = string`) — the writer pattern issue (e) documents.
- Re-run (`rerun_after_approve_treatment_arm.json`, 20:14:51Z → 20:17:02Z): analysis `591a45e9-e59c-47b9-91bf-d10cf05f6da1`,
  `status: failed`, `gate: block`, **`decision: proceed`**, **`review_id: 4eab7033-…`**, `discovered_dag_id
  e1678e1e-…`, ate 0.0856; tests: unobserved_common_cause FAILED (critical), placebo / random_common_cause /
  data_subset / bootstrap PASSED. Approval is structural: the estimate still fails sensitivity, the band stays BLOCK.
- Expected "a caveat naming the approval": **not present** in the API record. `warnings` =
  `["Refutation gate BLOCKED — the estimate did not survive robustness checks."]`; `executive_summary`, `narrative`,
  `key_insights`, `recommendations` empty (interpretation does not run on a failed band). The node DOES build the
  sentence ("The DAG structure was expert-approved by … (valid until …); that approval covers the DAG structure,
  not this estimate's statistical robustness", `nodes/refutation.py` `_review_note`) into the state field
  `review_caveat`, but that field has no consumer outside the node and the state schema (`grep -rn review_caveat src/`):
  `AgentCausalAnalysisResponse` has no such field and the route never appends it to `warnings`. Pre-existing since
  #1971 (PR #1985); surfaced by this verification; filed as a follow-up issue. The drill-down's **Structure approved**
  block does not need it (it renders from `expert_review_decision` + `expert_review_id`, both present).

## Step 3 — reject another brand-less row, re-run its pair

Cheapest disproof first: the pair `treatment_initiated → persistent_180d` had TWO brand-less pending rows with
different hashes (`2f79f11f` / `b5884ee9592d…` of 2026-09-03 and `07b67a55` / `38c538b2d8a7…` of 2026-09-02), so a
pre-run (`prerun_before_reject_treatment_initiated.json`, 20:17:47Z → 20:20:58Z, analysis `33ca2f21-…`) established
which row today's discovery maps to: `decision: pending_review`, `review_id: 2f79f11f-…` — no new row minted.

Review `2f79f11f-63e3-4075-ab84-07c3eef3cd52`: no assessment existed (card offered "Generate agent assessment"; not
generated), no checklist ticks, comment `lane-1 live verification: rejected to exercise the halt`; **Reject** at 20:21:43Z.

- Card after: `rejected` · Reviewer `admin@e2i.local` · Decided `2026-09-09` · Valid until `no expiry recorded` ·
  "This review is resolved. The rejection holds until a newer pending review of the same structure reopens it."
- Row: `approval_status=rejected`, `reviewer_name/email=admin@e2i.local`, `resolved_at=2026-09-09 20:21:43.2706+00`,
  `approved_at` NULL, `comments_json={"note": …}` (string-shaped, as above). Pending rows for the hash: **0**.
- Re-run (`rerun_after_reject_treatment_initiated.json`, 20:22:13Z → 20:24:22Z): analysis `e9dc7a0c-3cf9-4516-a402-950956231bce`,
  `status: failed`, `gate: block`, **`decision: rejected`**, **`review_id: 2f79f11f-…`**, `discovered_dag_id 33596e7a-…`;
  pending rows for the hash after the re-run: **0** (the rejection is durable; nothing was re-queued).
- Expected `warnings` text `Estimate withheld: a domain expert REJECTED this DAG structure`: **not present**, and by
  construction on this pair: `_expert_review_halt_reason` is only evaluated on the REVIEW band and, on PROCEED, when
  the read-only probe found a rejection (`nodes/refutation.py` ~1700–1745); on the **BLOCK** band the estimate is
  already withheld by the statistical gate (`error_message = _format_block_reason(suite)`), `expert_review_halt` is
  never set, and the route's `Estimate withheld…` line is gated on that flag (`routes/causal.py:3867`). Every
  brand-less pending row in the live queue is a BLOCK-band pair, so the halt sentence cannot be observed live without a
  PROCEED/REVIEW-band structure that a human then rejects. The executable demonstration of the PROCEED-band halt is
  `tests/unit/test_agents/test_causal_impact/test_refutation_expert_review_enforcement_1971.py` (CI-green on this merge).
  What IS observable on BLOCK: `expert_review_decision: rejected` + the review id in the record, and the queue not
  re-opening; the drill-down renders **Structure rejected** with the link from those two fields (the reason text is
  read from the halt warning, so it is absent on BLOCK-band runs — same follow-up issue as the approval caveat).

## Live DB after both adjudications

`expert_reviews`: pending 37 / approved 1 / rejected 2 (was 39 / 0 / 1 before the lane; the two resolved rows are the
ones above). `resolved_at` is populated on both new resolutions and NULL on the 37 pending rows and on the pre-lane
rejected row (no backfill by design).

## Drill-down on the deployed bundle (plan Step 2, "Open the run in the Causal Analysis page")

The Causal Analysis page has no deep link to a stored run, so the pair was run from the page itself:
"Pose your own question" → defaults `treatment_arm → persistent_180d`, scope "All brands · Patient grain" →
**Run analysis** (20:29:49Z; result at 20:31:59Z). The detail panel rendered: ATE 0.0856 · 95% CI [0.019, 0.152] ·
p = 0.0117 · LinearDML · 1,500 rows · **Blocked** · Significant · Refutation 4 / 5 passed · **Review status:
Structure approved** with the **Open review** link (`/expert-reviews?review=4eab7033-…`) and the sentence "A reviewer
approved this DAG structure. Approval covers the structure only; the estimate still stands or falls on its own
robustness checks." · Durable discovery record `a2bed973-2e52-4165-a5aa-f9f998ea4081`. Screenshot
`drilldown_structure_approved.png`. This is the same approved structure the API re-run reported (`decision: proceed`,
`review_id: 4eab7033-…`), rendered from those two fields.
