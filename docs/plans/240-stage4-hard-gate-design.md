# Layer-4 Audit-Evaluator — Stage 4 (Hard-Gate Severity + Routing Modulation) Design Proposal

- **Issue:** [#240](https://github.com/enunezvn/e2i_causal_analytics/issues/240) (multi-stage tracker)
- **Status:** PROPOSAL — design only; **NOT implementable yet**. Concrete acceptance thresholds require **Stage-3 operational data we do not have today**. This document exists so the door is not closed and so the work is scoped the moment its prerequisites are met.
- **Author:** Claude
- **Date:** 2026-05-24
- **Predecessor:** `docs/plans/240-audit-evaluator-gate-promotion.md` (the parent design; §3 Stage 4 was a sketch — this refines it).
- **Predecessor PRs:** Stage 1 (#492, shadow mode, MERGED), Stage 2 (curation surfacing), Stage 3 (soft-gate severity, env-gated default-OFF).

> **Updated 2026-05-26:** the **#242 multi-vendor-ensemble prerequisite** named below (the AC3.5 inheritance in §1, the R-C mitigation in §5, the recommended-path note in §6) is **RETIRED** — refuted in #242 (closed not-planned: a Sonnet 4.6 + Opus 4.7 + GPT-5 ensemble did **not** decorrelate frontier failures; correlated failure was 30% on hard cases and *rose* to 40% under a per-vendor zero-shot de-confound, i.e. the correlation is intrinsic to the frontier models, not a prompt artifact). The single-vendor correlated-failure RISK is unchanged; what changed is the **independence signal** — it is now a deterministic, **non-LLM** structural/statistical cross-check shipped in SHADOW mode: **#508** (merge `5a767ad0`, leakage × role cross-check) + **#515** (merge `e39d8e20`, M-structure role disambiguation), not more LLM vendors. See the #240 RE-SCOPE UPDATE.

---

## 1. Why this is a proposal, not an implementation

The parent design (§3 Stage 4, §11) states plainly that Stage 4 *"is a sketch, not a committed design. Stage 4's concrete acceptance thresholds (firing rates, error budgets, cost ceilings) require Stage-3 operational data we do not have today."* Building Stage 4 code now would mean inventing thresholds and a blast-radius model with no empirical basis — exactly the "lazy programming / act on a pattern without intent investigation" failure mode the project's REASON-BEFORE-RULES directive forbids.

**What gates Stage 4 (all currently unmet):**

1. **90 days of Stage-3 operational data** showing R1 holding AC3.1–AC3.3 with zero rollback events (AC4.1). Stage 3 ships default-OFF; the clock has not started.
2. **Per-consumer blast-radius analysis** (AC4.2) on Stage-3 shadow data — `role_attribution`, KG mirror, `compile_set_curation` impact must be bounded and reversible.
3. **Stakeholder sign-off** on remediation-override semantics (AC4.4) — a strictly larger blast radius than severity-modulation alone; out of Claude's authority.
4. **Independence signal for single-vendor correlated-failure risk** — already a Stage-3 hard prerequisite (AC3.5); Stage 4 inherits it and raises the stakes. The multi-vendor-ensemble form of this dependency is **RETIRED** (refuted in #242, closed not-planned — multi-vendor LLM agreement does not decorrelate frontier failures); the inherited prerequisite is now the deterministic non-LLM cross-check (#508 leak × role, merge `5a767ad0`; #515 M-structure, merge `e39d8e20`), shipped in shadow, OR a signed single-vendor risk acceptance.

Until 1–4 hold, the responsible action is: keep the Stage 1–3 instrumentation (which is a strict subset of Stage 4's needs — §6 of the parent), accumulate data, and revisit.

---

## 2. Scope (what Stage 4 would add over Stage 3)

Stage 3 modulates **severity only** (R1: `moderate → high`), env-gated and reversible. Stage 4 adds two strictly larger capabilities:

### 2.1 Remediation override independent of severity (R2 promotion)

- Today R2 (`evaluator dissatisfied AND ≥2 missed considerations`) is curation-only (Stage 2). Stage 4 would let R2 override `recommended_remediation` directly in the voter, **decoupled from severity** — e.g. force `remediation="drop"` on a verdict the worker marked `keep_with_caveat`, without changing severity.
- This is a larger blast radius than Stage 3 because severity-driven remediation is mechanical and bounded (`_remediation_for_severity`), whereas an independent remediation override can contradict the severity label.

### 2.2 Routing gate on downstream Layer-4 consumers

- Gate `role_attribution`, the KG mirror, and `compile_set_curation` on `evaluator_satisfied=True` — i.e. let the evaluator's verdict decide whether a feature's role propagates downstream at all, not just how severe it is.
- This does **not** require a competing `causal_role` from the evaluator (the evaluator audits the worker's role; it does not produce one). It only filters on the existing `satisfied` boolean.

---

## 3. Proposed mechanism (sketch — to be finalized against Stage-3 data)

- **New env var** `ADAPTIVE_VALIDITY_EVALUATOR_ROUTING_GATE_ENABLED` (default `0`). Independent of the Stage-3 `ADAPTIVE_VALIDITY_EVALUATOR_GATE_ENABLED` flag so severity-modulation and routing-modulation can be enabled/rolled back separately.
- **Remediation override:** extend the Stage-3 voter call site to additionally consult R2. When the routing flag is `1` and R2 fires, substitute `recommended_remediation` and append an evidence tag `"evaluator_gate:R2:remediation_override"`. Record the pre-override remediation in a new nullable column `worker_remediation_pre_gate` (migration, additive — mirrors Stage-3's `worker_severity_pre_gate`).
- **Routing gate:** the three downstream consumers filter on `evaluator_satisfied=True` only when the routing flag is `1`. Each consumer's filter is individually feature-flagged at first (per-consumer rollout) so blast radius is staged.
- **Fail-open (inherited from Stage 3):** evaluator disabled / errored ⇒ no override, no routing filter. Evaluator failure can never harm a healthy worker verdict.
- **Strict-subset preservation:** Stage 4 reads the same shadow columns + adds two (remediation-pre-gate + a routing-decision audit field). No Stage 1–3 column or contract is mutated.

---

## 4. Acceptance criteria (placeholders — thresholds require Stage-3 data)

| AC | Description | Blocked on |
|----|-------------|-----------|
| AC4.1 | 90 days of Stage-3 operational data; R1 holds AC3.1–AC3.3; zero rollback events | Stage 3 enabled in a validation cohort for 90 days |
| AC4.2 | Per-consumer (`role_attribution`, KG mirror, `compile_set_curation`) routing-gate blast-radius bounded + reversible on Stage-3 shadow data | Stage-3 shadow data + per-consumer impact study |
| AC4.3 | `EnsembleDecidedBy` widened to include `"evaluator_gate"` | **Satisfied by Stage 3** (already shipped in the Stage-3 PR) |
| AC4.4 | Stakeholder sign-off on remediation-override semantics | Stakeholder decision (out of Claude's authority) |
| AC4.5 (new) | R2 promotion-precision on the routing-gated subset measured against the golden set ≥ a threshold TBD from Stage-3 R1 data | Stage-3 data to calibrate the threshold |

---

## 5. Risks (delta over Stage 3)

- **R-A — Remediation/severity contradiction.** An independent remediation override can produce `severity=moderate, remediation=drop`, which downstream consumers may not expect. Mitigation: a voter-level invariant test + a consumer-compatibility audit before enabling.
- **R-B — Routing-gate cascade.** Filtering downstream consumers on `satisfied` can silently drop a feature from the KG entirely. Mitigation: per-consumer flags + the `worker_*_pre_gate` columns so every dropped feature is recoverable from the audit trail; staged per-consumer rollout.
- **R-C — Compounded single-vendor risk.** Routing decisions on a single Anthropic evaluator multiply the correlated-failure risk (parent §5 R-2). Mitigation: a deterministic, **non-LLM** independence cross-check is a hard prerequisite, not just a recommendation, at Stage 4 — **#508** (leak × role, merge `5a767ad0`) + **#515** (M-structure role disambiguation, merge `e39d8e20`), both shipped in shadow. (The previously-named #242 multi-vendor-ensemble mitigation is RETIRED: #242, closed not-planned, proved multi-vendor LLM agreement does not decorrelate frontier failures — correlation *rose* to 40% under a per-vendor de-confound. Adding LLM vendors does not reduce this risk; a non-LLM signal does.)

---

## 6. Recommended path

1. **Do not implement now.** Ship Stages 1–3, enable Stage 3 only after AC3.5 (the deterministic non-LLM independence cross-check — #508 leak × role + #515 M-structure, shipped in shadow — OR a signed single-vendor risk acceptance; the #242 multi-vendor-ensemble form is RETIRED) + Stage-1/2 data.
2. Accumulate 90 days of Stage-3 operational data.
3. Run the per-consumer blast-radius study (AC4.2) on that data.
4. Bring the calibrated thresholds + remediation-override semantics to stakeholders (AC4.4).
5. **Then** convert this proposal into an implementation plan and build it behind the two independent default-OFF flags.

Until step 4 completes, #240 remains the open tracker for this work.
