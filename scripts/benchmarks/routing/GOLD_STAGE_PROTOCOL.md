# Gold-judge stage protocol (#1337 Step 0, next increment)

Status: protocol notes, accreting during the 2026-07-29 interactive verdict review.
The gold-judge stage labels the 337-query benchmark set (`data/benchmark_queries.jsonl`)
using the contract registry (`data/agent_contracts.json`) as SSOT, the #1341 judge
protocol, and the human verdicts recorded in `review/proposed_verdicts.json` (`final`
fields) as calibration anchors.

## Axis 1 — routing gold (the original axis)

Which route SHOULD own the query, per contract covers/does_not_cover ownership.
Normative: classifier reach, resolver operability, and dispatch budgets never
constrain gold (see the five ambiguity resolutions in
`review/tool_composer_corrections.md`). Follow-up-shaped queries are labeled
in-context (prev-turn context travels with the row).

## Axis 2 — answer quality (added 2026-07-29, user directive)

Grading a surface's actual response is a SEPARATE axis from routing gold, with
three ordered layers. Each layer presumes the previous one.

1. **Hallucinated?** — Do the response's claims/numbers exist in a real data
   source at all, or were they invented? (By-construction pass for registry-lookup
   tools; must be checked for LLM-synthesized narrative claims.)
2. **Faithfully retrieved?** — Do the quoted values match the source row(s)
   byte-for-byte, and is the cited source the one actually read?
   Worked example (q05, 2026-07-29): live answer quoted effect 0.166 /
   confidence 0.796 / lag 77d / backdoor.linear_regression for Kisqali
   rep-detailing→TRx; `causal_paths` row `scp_c696e9a45437` matches 6/6 fields.
3. **Accurately computed and business-appropriate?** — Is the upstream value
   methodologically sound (provenance: causal-engine method, confounders,
   refutation/validation status, `is_synthetic`), and does the narrative use the
   business framing the data supports?
   Known probes from the q05 spot-check: `causal_paths.validation_status`
   semantics need pinning (the q07 live answer claimed "no refutation-test flag
   in the registry" while rows carry validation_status='validated'); the registry's
   `business_impact_estimate` (38,244.12 on the q05 row) was NOT surfaced in the
   live answer — completeness gap, not an accuracy error.

Rubric sources for layer 3: per-tier "what good looks like" criteria in
`docs/demos/COPILOT_CHAT_DEMO_SCENARIOS.md` (content semantics owned by #1345).

## Scope notes

- Axis 2 applies to the surfaces we can actually elicit answers from (the AG-UI
  brain today; orchestrator dispatch routes once #1350/#1351 are resolved).
- Axis-2 grades never feed back into axis-1 labels (a wrong answer on the right
  route is a quality finding, not a routing correction).
