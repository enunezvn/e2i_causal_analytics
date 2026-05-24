# Layer-4 Audit-Evaluator Gate Promotion — Design Proposal

- **Issue:** [#240](https://github.com/enunezvn/e2i_causal_analytics/issues/240)
- **Status:** PROPOSAL (not yet approved) — design doc only; no production behavior change in this PR
- **Author:** Claude
- **Date:** 2026-05-24
- **Predecessor plans (gitignored archives):**
  - `.claude/plans/archive/15_layer4_evaluator_audit_signal_DONE_710058e0-13570eb8.md` (producer)
  - `.claude/plans/archive/14_layer4_evaluator_audit_consumer_DONE_dd222f0a-c6313e5b.md` (curation consumer)
- **Empirical basis:** PR [#477](https://github.com/enunezvn/e2i_causal_analytics/pull/477) — MIPROv2 gated A/B at compile-set n=240, golden-set n=91, executed with `--enable-evaluator` (`ADAPTIVE_VALIDITY_EVALUATOR_ENABLED=1`), artifact `artifacts/dspy/ac3_verdict_n200.json`.

---

## 1. Context: today's audit-only contract

The Layer-4 Haiku evaluator (`src/data/causal_role_evaluator.py`) currently runs under a **strictly read-only contract**. After the worker (`CausalRoleClassifier`, DSPy + Claude Sonnet) emits an `LLMVerdict`, the evaluator inspects that verdict and emits an `LLMEvaluatorAudit` sidecar with five audit fields:

| Field | Type | Semantics |
|-------|------|-----------|
| `satisfied` | `bool` | Evaluator's overall pass/fail against criteria text |
| `rationale_complete` | `bool` | Worker cited temporal filter + Pearl arrowheads |
| `missed_considerations` | `tuple[str, ...]` | ≤5 short labels (≤80 chars) of axes worker missed |
| `notes` | `str` | Free-text rationale, truncated to 500 chars |
| `evaluator_model` | `str` | Pinned to `anthropic/claude-haiku-4-5-20251001` |

Plus four telemetry fields (issue #241): `latency_ms`, `input_tokens`, `output_tokens`, `cost_usd`.

The producer attaches the audit onto `LLMVerdict.evaluator_audit`. The `EnsembleVoter` (`src/data/kg/ensemble_voter.py:440`) and the issue-#212 cap path **do not read these fields**. They flow only into the audit-trail sidecar via `_ensemble_to_legacy_dict` (`src/agents/ml_foundation/data_preparer/nodes/adaptive_validity_check.py:1268-1286`) and are persisted to the `adaptive_validity_verdicts` table (`database/migrations/040_adaptive_validity_verdicts.sql`).

Two downstream consumers exist today:

1. **Curation flow** (shipped 2026-05-15): `src/data/audit_sidecar_reader.py` + `src/data/audit_candidate_formatter.py` + `scripts/curate_compile_set_candidates.py` produce a markdown report and JSON manifest for human engineer review. The reviewer hand-merges accepted candidates into `build_compile_set()` for the next classifier compile.
2. **Precision A/B harness** (shipped today via #477): `scripts/measure_layer4_precision.py --enable-evaluator` reads `verdict.evaluator_audit.satisfied` and uses it as a **filter** (gate=true subset) for the AC3 comparison. The gate here is a *measurement-time* filter, not a runtime decision-mutation.

**The promotion direction this proposal addresses:** allow the evaluator's verdict to *modulate* the worker's `severity` / `recommended_remediation` inside the orchestrator, closing the audit→action loop without requiring a human curation cycle.

---

## 2. Empirical signal from #477's gated A/B

### What was measured

PR #477 ran `scripts/measure_layer4_precision.py --enable-evaluator --golden-set <91 entries>` against two compiled classifier artifacts (Bootstrap and MIPROv2). The `--enable-evaluator` flag sets `ADAPTIVE_VALIDITY_EVALUATOR_ENABLED=1` *before* loader import, so each per-feature classifier call also invokes the Haiku evaluator and stores its audit on the returned `LLMVerdict.evaluator_audit`.

The script's `_compute_metrics_for_gate` function (`scripts/measure_layer4_precision.py:138-247`) bins each entry into:

- **`gate=false` (ungated)**: every entry where the classifier returned a verdict.
- **`gate=true` (gated)**: subset where `verdict.evaluator_audit.satisfied is True`.

Entries with `evaluator_audit is None` or `satisfied is False` are counted as `n_skipped_no_eval` in the gated subset.

### Aggregate result from the committed artifact

The only committed artifact from #477 is `artifacts/dspy/ac3_verdict_n200.json` (38 lines, schema_version=1). It contains the AC3 verdict and **per-cohort gated precision_instrument floats only** — no TP/FP/FN counts, no skipped-entry counts, no per-entry payload. The verified contents:

| field | value |
|-------|-------|
| `compile_n_examples` | 240 |
| `compile_seed` | 42 |
| `golden_set_n_entries` | 91 |
| `miprov2_wins` | true |
| `overall_ok` | true |
| `cohort_ok` | true |
| `cohort_regressions` | `[]` |

| cohort (`rows[].cohort`) | `bootstrap_gated_precision_instrument` | `miprov2_gated_precision_instrument` | `miprov2_meets_or_exceeds_bootstrap` |
|--------------------------|----------------------------------------|--------------------------------------|--------------------------------------|
| OVERALL | 1.0 | 1.0 | true |
| CSU_remibrutinib | null | null | null |
| PNH_fabhalta | 1.0 | 1.0 | true |
| BC_kisqali | 1.0 | 1.0 | true |

The CSU `null/null/null` row signals that the gated CSU subset contained zero instrument predictions for either optimizer — `precision_instrument` is undefined when both the gated cohort denominators are empty.

The TP/FP/FN counts reported in PR #477's body table (`OVERALL TP=3 / TP=6`, `BC TP=2 / TP=4`, `PNH TP=1 / TP=2`) were produced by the precision-report run *during* PR #477 but are **not** present in any committed artifact in this repo. They were preserved as local files at `/tmp/{bootstrap,miprov2}_n200_gated_v2.json` per the PR #477 body. This proposal therefore cites only the AC3 verdict JSON's actual fields and treats per-entry / per-TP counts as a known artifact-coverage gap to be closed by the Stage-1 shadow column (§3).

### What is and is not in the committed artifact

- **In the artifact**: the AC3 verdict JSON with the seven scalar fields and four per-cohort precision rows shown above. This is the basis for the optimizer-promotion decision in #477; it is **insufficient** for evaluating gate-promotion rules in this proposal because precision is the only metric and is aggregated to the gate level.
- **Not in the artifact (gap)**: per-cohort TP / FP / FN counts (only the PR body table has them; not on disk).
- **Not in the artifact (gap)**: per-entry rows recording `(worker_severity, evaluator_satisfied, worker_role, ground_truth_role, missed_considerations)`. The script can emit a per-entry disagreement file (`--disagreements-path`, see `scripts/measure_layer4_precision.py:355-433`) but only logs *role* mismatches, not evaluator-vs-worker disagreement.
- **Not in the artifact (gap)**: counts of `evaluator_satisfied=False` and what role/severity those entries would carry if escalated.

**Conclusion for this proposal:** the n=240 / golden-n=91 + `--enable-evaluator` run *did execute the evaluator* end-to-end and produced per-entry audit data at run time, but the per-entry payload was not persisted to a committed artifact. **A Stage-0 (data collection) step is therefore the first pre-requisite of any gate promotion** — see §3 Stage 1.

### What the aggregate signal *does* tell us

1. **The gate is precision-preserving where the gated subset is non-empty.** All three measurable cohorts (OVERALL, PNH, BC) report `precision_instrument=1.000` on the gated subset for both optimizers (verified directly in the artifact). This is consistent with the evaluator's design intent but does not establish that severity-modulation rules built on the same signal would have the same property — precision-preservation as a *filter* and as a *severity-modulator* are distinct claims.
2. **The CSU cohort produced an undefined gated metric** (`null` for both optimizers). Any severity-modulation rule must explicitly handle the empty-gated-cohort case (fail-open) and the Stage-3 AC must accept `null → null` as a non-regression.
3. **The PR-body TP delta (bootstrap=3 vs miprov2=6 OVERALL) suggests a 2× recall difference at equal gated precision**, but this number cannot be re-derived from the on-disk artifact and is reported here only as a Stage-1 hypothesis to verify with the shadow column data, not as committed evidence.

---

## 3. Proposed gate-promotion design

The promotion is staged across four levels of increasing intrusiveness. **Each stage's acceptance criteria gate entry to the next stage.** Stages 1–2 collect data; Stages 3–4 mutate decisions.

### Stage 1 — Shadow mode (instrument; no behavior change)

**Goal:** persist, on every Layer-4 invocation, the (worker_severity, worker_remediation, worker_role, evaluator_satisfied, missed_considerations, would-be-promotion-decision) tuple so we can build evidence tables independently of the curation flow.

**Mechanism (proposed):**

- Add three new nullable columns to `adaptive_validity_verdicts` (migration 041), one per rule so each rule's firing is observable independently:
  - `would_promote_severity` (text, nullable) — set by R1 to the proposed escalated severity, `NULL` otherwise.
  - `would_flag_for_review` (boolean, nullable) — set by R2 to `true` when its trigger fires, `NULL` otherwise.
  - `rationale_incomplete_flag` (boolean, nullable) — set by R3 to `true` when its trigger fires, `NULL` otherwise.
- New file `src/data/evaluator_promotion_rules.py` containing one pure function per rule with signatures:
  - `evaluate_r1(worker_verdict, evaluator_audit) -> Optional[str]` — returns proposed severity or `None`.
  - `evaluate_r2(worker_verdict, evaluator_audit) -> Optional[bool]` — returns `True` when trigger fires or `None`.
  - `evaluate_r3(worker_verdict, evaluator_audit) -> Optional[bool]` — returns `True` when trigger fires or `None`.
  Plus a `PROMOTION_RULES` registry tuple `((rule_id, fn), ...)` so a single call site can iterate. **The voter does not read these columns or call these functions at Stage 1.** They are written for analytics only.
- Extend `_ensemble_to_legacy_dict` to call all three rule functions and surface the three flags into the sidecar row. All `None`-fallback paths (`adversarial_only`, `info_only`, `short_circuit`) leave the three flags as `NULL` per the existing nullability pattern at `adaptive_validity_check.py:1351-1355 / 1420-1424 / 1484-1488`.

**Acceptance criteria (gate to Stage 2):**

- AC1.1 — Migration 041 deployed; column populated on every Layer-4 run for 14 calendar days, **N ≥ 500** runs accumulated.
- AC1.2 — A new test asserts that for **every** run where any of R1 / R2 / R3 fires (i.e., any of the three flag columns is non-NULL), `verdict.severity` and `verdict.recommended_remediation` are byte-identical to the equivalent run with `evaluator_promotion_rules` mocked to return `None` for every rule. "≥99%" is rejected as too weak — Stage 1's invariant is exact byte-identity (shadow truly shadows); any deviation is a bug, not noise.
- AC1.3 — Per-rule firing-rate table reproduced from the new column (see §4 rules R1-R3 below).
- AC1.4 — Cost telemetry confirms no regression in `evaluator_latency_ms` p95 or `evaluator_cost_usd` total (the rule is pure and adds no LM calls).

**Out of scope at Stage 1:** any change to the voter; any change to severity downstream; any change to remediation; any UI surface.

### Stage 2 — Audit-with-promotion-data (enrich manifests; still no auto-decision change)

**Goal:** surface the shadow-mode `would_promote_severity` column into the existing curation flow and the `scripts/query_audit_trail.py` reports so engineers can manually accept/reject promotion decisions during compile-set review.

**Mechanism:**

- Extend `src/data/audit_sidecar_reader.DisagreementEvent` with `would_promote_severity` and the input fields that drove it.
- `audit_candidate_formatter.format_markdown_report` adds a "Promotion candidate" section per row when `would_promote_severity is not None`.
- New CLI flag on `scripts/curate_compile_set_candidates.py`: `--filter-promotion-rule R1` to focus reviewers on one rule at a time.

**Acceptance criteria (gate to Stage 3):**

- AC2.1 — On 90+ days of accumulated shadow data, the **observed-FP-rate of each rule** (rule fired AND human reviewer confirmed worker was correct, evaluator wrong) is **< 10%** for any rule promoted to Stage 3.
- AC2.2 — **Inter-rater check**: 2 reviewers blind-label a sample of 100 rule-firing cases; Cohen's κ ≥ 0.6 (substantial agreement) on the "evaluator-was-right" judgment.
- AC2.3 — Per-rule cost-benefit: `(prevented_FP_count × cost_per_FP_dollars) > (introduced_FN_count × cost_per_FN_dollars)` using best-available estimates (stakeholder input required for the cost-per-X numbers).

### Stage 3 — Soft-gate severity modulation (limited, reversible, fail-open)

**Goal:** allow ONE rule (the strongest from Stage 2) to actually modulate `severity` inside `EnsembleVoter.vote`, behind a kill-switch env var and limited to the `severity=moderate → severity=high` direction.

**Mechanism:**

- New env var `ADAPTIVE_VALIDITY_EVALUATOR_GATE_ENABLED=0|1` (default `0`). When `0`, voter behavior is byte-identical to today.
- When `1`, after the voter computes its candidate severity, the voter calls `evaluate_promotion_rule(...)` exactly once. If the rule fires AND the candidate severity equals the rule's precondition, the voter substitutes the rule's proposed severity AND appends to `evidence` a structured tag `"evaluator_gate:R1:moderate→high"`. The `decided_by` field gains a new value `"evaluator_gate"` ONLY when the gate actually flipped a decision (not when the voter independently reached the same severity). **Schema prerequisite:** the `EnsembleDecidedBy` literal in `src/data/kg/types.py` (~line 28) must be widened to include `"evaluator_gate"` as part of the Stage 3 PR; otherwise the typed dataclass will reject the new value at construction time.
- Remediation override is computed deterministically from the new severity by the existing `_remediation_for_severity` helper. **Remediation is not separately mutated by the gate** at Stage 3 — only severity is, and remediation follows mechanically.
- A new field `gate_rule_fired: Optional[str]` is added to `EnsembleVerdict` (and surfaced to the sidecar) recording which rule fired (or None).

**Fail-open default:** when the evaluator is disabled (`evaluator_audit is None`) OR when the evaluator call raised an exception (recorded as `satisfied=None` by the runner), the gate **does not fire**. Worker verdict passes through unchanged. This preserves the audit-only contract's invariant that evaluator failure cannot harm a healthy worker.

**Rollback story:**

- Operator: `unset ADAPTIVE_VALIDITY_EVALUATOR_GATE_ENABLED` → next process invocation reverts. No data migration required; the new sidecar column simply ceases to populate `gate_rule_fired`.
- Code: revert the voter diff (one PR). The shadow-mode column from Stage 1 is preserved (it remains useful for offline analysis).

**Acceptance criteria (gate to Stage 4):**

- AC3.1 — At Stage 3 enabled for 30 days against a non-production validation cohort, `precision_instrument` on the gated subset of the golden set is **strictly ≥** the corresponding **gated baseline** measured in #477 (the artifact's `overall.gated.precision_instrument` was 1.0 at n=240 — must remain 1.0 OR a previously-`null` cohort must reach a defined value ≥1.0). The comparison is gated-vs-gated; un-gated precision is not a Stage-3 success condition because Stage 3 only mutates verdicts that survive the evaluator filter.
- AC3.2 — Per-cohort regression check: no cohort drops more than 1 instrument TP or gains any FP vs the comparable Stage-1 shadow run. (Strict superset of #477's AC3 cohort-regression rule.)
- AC3.3 — Operator runbook documents the kill-switch + the exact verdict-row query to identify gate-flipped decisions for rollback.
- AC3.4 — Multi-vendor concern (codex Gate-1 rejection of Anthropic-only correlated failures) addressed: either #242 multi-model evaluator ensemble shipped, OR an explicit written decision from stakeholders that single-vendor risk is acceptable for the specific Stage-3 rule. This is a HARD prerequisite per the producer plan's failure-mode analysis.

### Stage 4 — Hard-gate severity AND routing modulation (long-horizon)

**Goal:** allow the evaluator to (a) override `recommended_remediation` independent of severity, and (b) gate Layer-4-downstream consumers (`role_attribution`, KG mirror) on `evaluator_satisfied=True`.

**Mechanism (proposed sketch — to be refined in a separate plan after Stage 3 operational data is available):**

- Extend `evaluate_promotion_rule(...)` (Stage 3 voter call) to additionally consult R2 for remediation override (currently R2 is curation-only).
- New env var `ADAPTIVE_VALIDITY_EVALUATOR_ROUTING_GATE_ENABLED` (default `0`) controls whether downstream consumers (`role_attribution`, KG mirror, `compile_set_curation`) filter on `evaluator_satisfied=True`.

**Acceptance criteria (placeholders; concrete thresholds require Stage 3 operational data):**

- AC4.1 — 90 days of Stage 3 operational data show R1 (and any newly-promoted rule) holding AC3.1 + AC3.2 consistently with no rollback events.
- AC4.2 — Per-consumer (`role_attribution`, KG mirror) impact analysis on the Stage-3 shadow data confirms the routing-gate's blast radius is bounded and reversible.
- AC4.3 — `EnsembleDecidedBy` literal in `src/data/kg/types.py` widened to include `"evaluator_gate"` (already a Stage-3 prerequisite — see §5 R-5 mitigation note).
- AC4.4 — Stakeholder sign-off on remediation-override semantics (a strictly larger blast radius than severity-modulation alone).

**Status in this proposal:** the mechanism above is a sketch, not a committed design. Stage 4's concrete acceptance thresholds (firing rates, error budgets, cost ceilings) require Stage-3 operational data we do not have today. This proposal commits to *not closing the door* on Stage 4: the Stage 1–3 instrumentation (shadow column, kill-switch, `gate_rule_fired` field, three independent rule flags) is a strict subset of what Stage 4 would need, so no rework is required to advance.

---

## 4. Proposed promotion rules (Stage 1 starting set)

Each rule below has:
- **Trigger**: the exact condition on (worker_verdict, evaluator_audit).
- **Stage-1 action**: what the shadow column records.
- **Stage-3 action**: what (if anything) the voter does when this rule is the chosen one.

Initial firing-rate estimates marked `TBD-via-Stage-1` are placeholders for the §3 AC1.3 table.

### R1 — Moderate→High escalation on dissatisfied evaluator

- **Trigger:** `worker_verdict.severity == "moderate" AND evaluator_audit.satisfied == False AND len(evaluator_audit.missed_considerations) >= 1`.
- **Stage-1 action:** record `would_promote_severity = "high"`.
- **Stage-3 action:** voter substitutes `severity="high"`, `remediation="drop"` (via deterministic helper), appends `"evaluator_gate:R1:moderate→high"` to evidence.
- **Rationale:** This is the canonical "the worker said maybe-problematic, the evaluator independently flagged specific missed considerations, escalate to definitely-problematic." It is the rule the issue body explicitly cites as the most defensible.
- **Expected firing rate:** TBD-via-Stage-1. Conservative prior from the #477 gated subset: roughly 17 of 24 ungated instrument predictions are excluded by `evaluator_satisfied=False`, suggesting evaluator dissatisfaction is *frequent* in the instrument-prediction pool. Translation to the moderate-severity pool requires Stage-1 data.
- **False-positive risk:** evaluator-wrong-worker-right cases. Stage-2 inter-rater check (AC2.2) is designed to bound this. The §3 fail-open default ensures evaluator outages don't trigger.

### R2 — Flag-for-review on missed considerations (no severity change)

- **Trigger:** `evaluator_audit.satisfied == False AND len(evaluator_audit.missed_considerations) >= 2`.
- **Stage-1 action:** set `would_flag_for_review = True` (the dedicated R2 column declared in §3 Stage 1 Mechanism). R2 does not touch `would_promote_severity`.
- **Stage-3 action:** **not promoted to Stage 3 in this proposal.** R2 is intentionally a curation-flow accelerator, not a voter rule. It's listed here so the Stage-1 column captures both rules from one schema.
- **Rationale:** Many missed considerations on a moderate or info verdict don't warrant escalation but do warrant human review. This rule formalizes "make sure curation sees this row" without changing semantics.

### R3 — Rationale-incomplete soft-flag (audit-only forever)

- **Trigger:** `evaluator_audit.rationale_complete == False`.
- **Stage-1 action:** set `rationale_incomplete_flag = True` (the dedicated R3 column declared in §3 Stage 1 Mechanism).
- **Stage-3 action:** **deliberately never promoted.** This is a documentation-quality signal, not a correctness signal. It supports compile-set curation but does not warrant runtime mutation.

### Rules explicitly **not** included

- **"`evaluator_satisfied=True` → demote severity"** (worker said high, evaluator said fine, demote to moderate). This is **rejected** because: (a) the worker's high-severity vetoes are typically driven by `decided_by ∈ {layer_1, adversarial}`, which the LLM-evaluator was never designed to second-guess; (b) demotion failure modes are far costlier than escalation failure modes (false negatives on real-leak features ship to production).
- **"Evaluator-overrides-role"** (replace `causal_role` based on the evaluator's verdict). The evaluator does not produce a causal_role; it audits the worker's. Adding a competing causal_role would require redesigning the evaluator prompt and is out of scope. Stage 4's "routing modulation" can gate consumers on `satisfied` without requiring a competing role.

---

## 5. Risks

### R-1 — False-positive escalations (worker right, evaluator wrong)

**Severity:** HIGH (the dominant risk).

The Haiku evaluator is a smaller model than the Sonnet worker. There is no a-priori reason to assume the evaluator is more accurate. Cases where the evaluator overrides a correct worker decision would (a) drop a useful feature, (b) propagate the drop into downstream KG mirroring, (c) require human intervention to recover.

**Mitigations:**
- §3 Stage 2 AC2.1 caps FP rate at 10% before any Stage-3 promotion.
- §3 Stage 3 fail-open default + kill-switch.
- §3 Stage 3 AC3.2 per-cohort regression check.
- §4 R1's scope-limit to moderate→high only (the lowest-stakes severity transition).

### R-2 — Evaluator fragility / correlated failures

**Severity:** MEDIUM-HIGH.

The producer plan's codex Gate-1 review (per issue body) flagged that two Anthropic-family models (Sonnet worker + Haiku evaluator) may correlate in their failure modes. A confident-wrong worker on a feature that the evaluator also misunderstands → the evaluator certifies the wrong answer; the gate would be a false confidence boost.

**Mitigations:**
- §3 Stage 3 AC3.4 — multi-vendor evaluator (#242) is a HARD prerequisite OR an explicit stakeholder-signed risk acceptance.
- The audit-only contract is preserved through Stage 2; correlated failure manifests only as suboptimal curation candidates, not as wrong decisions.

### R-3 — Cost (per-classification Haiku call)

**Severity:** LOW-MEDIUM.

Today the evaluator is operator-opt-in (env-var-gated). Stage 3 would imply running the evaluator on every Layer-4 invocation for the cohorts where the gate is enabled.

**Mitigations:**
- The evaluator is already wired into PR #477's `--enable-evaluator` path with token / cost telemetry (issue #241 fields: `input_tokens`, `output_tokens`, `cost_usd`).
- Stage 1 acceptance includes a cost-baseline (AC1.4) so cost regressions are caught before Stage 3.
- Haiku-class pricing (`HAIKU_INPUT_USD_PER_MTOK` constants in `src/data/causal_role_evaluator.py`) is approximately 1/15th the Sonnet worker cost; the marginal cost is bounded.

### R-4 — Audit-loop coupling (evaluator influencing data the next training run consumes)

**Severity:** MEDIUM.

If the evaluator's verdict is allowed to mutate `severity`, and `severity` is fed back into compile-set curation (it is, via the curation flow), then the evaluator is now indirectly training the worker it audits. This is a known-bad pattern in LLM-evaluator-LLM-worker setups.

**Mitigations:**
- §3 Stage 1 schema preserves `worker_severity` and `effective_severity` as SEPARATE columns. Compile-set curation MUST be configured to use `worker_severity` (not `effective_severity`) when building demonstrations. This is a documented invariant, enforced by a test.
- §3 Stage 2 manifests display the `worker_severity` as the primary value with `would_promote_severity` shown adjacent for context.

### R-5 — Schema breakage downstream

**Severity:** LOW.

Adding `gate_rule_fired` and `would_promote_severity` to the sidecar / `EnsembleVerdict` could break naive consumers that assume a fixed key set.

**Mitigations:**
- Both new fields are nullable / `Optional` and added at the end of the dataclass. The existing 25/25 producer-plan tests should remain green (the issue acceptance criterion).
- Migration 041 follows the additive-column pattern used by 040.

---

## 6. Stage progression as strict subset

Each stage's instrumentation is a strict subset of the next. This is a design constraint, not a documentation flourish:

- **Stage 1** writes shadow columns. Stage 2 reads them. Stage 3 reads them AND mutates one downstream field. Stage 4 reads them AND mutates more.
- A rollback from Stage N to Stage N-1 is at most a code revert + env-var unset; no data migration is required. Forward-compatible nullable columns + env-var-gated voter behavior achieve this.
- At no stage is the audit-only contract on the `LLMEvaluatorAudit` struct itself violated. The 5 existing audit fields' write paths in `adaptive_validity_check.py` — verified on current `main` at lines `:1272-1276` (Layer-4 path with audit) plus three `None`-fallback blocks at `:1351-1355`, `:1420-1424`, `:1484-1488` (adversarial-only, info-only, short-circuit bypass paths) — remain untouched. The gate is **additive**, surfaced through new fields, never by mutating the existing five.

---

## 7. Rejected alternatives

### A-1 — Hard-gate immediately (skip Stages 1–3)

**Rejected:** the producer plan explicitly held the "plausibility ≠ verification" invariant; codex Gate-1 rejected verdict-replacement framings. Skipping data collection violates REASON-BEFORE-RULES (acting on a pattern match without an intent investigation).

### A-2 — Multi-model ensemble first, gate second (block on #242)

**Considered, partially accepted:** the multi-vendor evaluator is a HARD prerequisite for Stage 3 per AC3.4. However, blocking Stages 1–2 on it would prevent us from accumulating the very data needed to scope #242 (which rules' FPs would multi-vendor reduce by how much?). Decision: Stages 1–2 proceed; Stage 3 blocks on #242 OR stakeholder risk acceptance.

### A-3 — Pure-human gate (defeat the cost-saving point)

**Rejected:** the curation flow IS a pure-human gate. The "audit-evaluator" promotion's value is precisely that it can act on the high-volume audit signal without scaling humans linearly with feature count. If we ship pure-human, #240 is closed as `not_planned` and we just keep the curation flow.

### A-4 — Train a discriminator model on shadow data (replace the rule-based gate)

**Considered, deferred:** worth revisiting after Stage 1 produces a labeled corpus. Not appropriate as the v1 promotion design — adds a model to debug.

---

## 8. Open questions for stakeholders

The following decisions are out of Claude's authority and must be answered before Stage 1 implementation work begins:

1. **Cost-per-FP-vs-cost-per-FN ratios** (for §3 AC2.3). What is the operating cost of incorrectly dropping a useful feature vs incorrectly retaining a leaky feature? Both have downstream consequences in the commercial-analytics pipeline; the ratio drives which rules pass the cost-benefit gate.
2. **Multi-vendor evaluator (#242) timeline.** Is shipping a non-Anthropic evaluator a 2026-H2 commitment, a 2026-H1 commitment, or speculative? AC3.4 blocks on either it or an explicit risk-acceptance.
3. **Rule prioritization.** Stages 1–2 collect data on R1, R2, R3 simultaneously. Which rule does the team want to promote to Stage 3 first if all three pass AC2 thresholds?
4. **Shadow-data retention.** How long should the shadow column rows be retained? 90 days for Stage-2 analysis is the assumed default; PII / log-retention policies may constrain this.
5. **Acceptable per-cohort regression at Stage 3.** AC3.2 says "no cohort drops more than 1 instrument TP." Is that the right threshold? At the n=91 golden-set scale, 1 TP is roughly 1 percentage point of cohort precision — large enough to matter, small enough to be noise.
6. **Promotion-as-codification of an empirical pattern, or as a *change*?** If Stage-1 data shows R1 would fire on >50% of moderate-severity verdicts, R1 is no longer a "soft promotion" — it's a significant policy shift. Is there a firing-rate ceiling above which a different governance review is required?

---

## 9. Acceptance criteria summary (consolidated)

| Stage | AC | Description |
|-------|----|-------------|
| 1 | AC1.1 | Migration 041 deployed; N ≥ 500 runs over 14 days |
| 1 | AC1.2 | Shadow-mode invariant test: severity AND remediation byte-identical when any rule flag fires |
| 1 | AC1.3 | Per-rule firing-rate table built from shadow column |
| 1 | AC1.4 | No regression in evaluator latency p95 or cost |
| 2 | AC2.1 | Each Stage-3-candidate rule has FP-rate < 10% on human-reviewed sample |
| 2 | AC2.2 | Inter-rater Cohen's κ ≥ 0.6 on 100-sample blind label |
| 2 | AC2.3 | Per-rule cost-benefit positive |
| 3 | AC3.1 | Gated `precision_instrument` ≥ #477 baseline (1.000) |
| 3 | AC3.2 | Per-cohort: no cohort loses >1 instrument TP or gains any FP |
| 3 | AC3.3 | Kill-switch + rollback query documented in operator runbook |
| 3 | AC3.4 | #242 multi-vendor evaluator shipped OR signed stakeholder risk acceptance |
| 4 | AC4.1 | 90 days of Stage-3 operational data with no rollback events |
| 4 | AC4.2 | Per-consumer routing-gate impact analysis bounded |
| 4 | AC4.3 | `EnsembleDecidedBy` literal widened to include `"evaluator_gate"` |
| 4 | AC4.4 | Stakeholder sign-off on remediation-override semantics |

The original issue body's three "decision document" requirements are addressed as: (a) what gate semantics → §3 + §4; (b) multi-vendor dependency → R-2 + AC3.4; (c) cost/latency budget → R-3 + AC1.4; (d) rollback story → §3 Stage 3 "Rollback story" + AC3.3.

The original issue's two acceptance constraints (ADDITIVE not REPLACEMENT; no regression on 25/25 producer tests) are upheld by §6's strict-subset design and §5 R-5 mitigation.

---

## 10. Out of scope for this proposal

- Any code change to `EnsembleVoter`, `LLMEvaluatorAudit`, or the audit producer in this PR. This proposal is design only.
- The shadow-mode prototype itself (would be a follow-up PR; see §11).
- Resolution of #242 (multi-model ensemble).
- HITL-queue framings — explicitly excluded by the issue body.

---

## 11. Next steps (post-approval)

1. **Stakeholder review** of this proposal — answers to §8 questions.
2. **Stage-1 implementation PR** (separate plan): migration 041 + `evaluator_promotion_rules.py` + sidecar wiring + shadow-mode test (AC1.2). Estimated 1-2 day single-session executor scope.
3. **14-day data accumulation** against the existing live evaluator workflows.
4. **Stage-1 firing-rate report** appended to this doc.
5. **Decision point**: proceed to Stage 2, revise rules, or close #240 as `not_planned` if data does not support promotion.
