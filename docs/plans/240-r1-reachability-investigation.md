# Issue #240 R1 Promotion Rule — Reachability Investigation

**Scope:** Independent verification of the claim that promotion rule R1
(`docs/plans/240-audit-evaluator-gate-promotion.md` §4) can never fire in
production, plus feasibility of the proposed Option 2 / Option 3 fixes.
**Repo state:** `main`, Stage-1 of #240 merged (commit `1e8ca0df`).
**Method:** static trace of the production path + an empirical reproduction
run against the real `EnsembleVoter` and `evaluate_r1`. No source or tests
were modified.

**Verdict: the finding is CONFIRMED.** R1 is dead in production. R2 and R3
are reachable. Option 2 is conceptually problematic (it asks the evaluator
to audit reasoning that does not exist). Option 3 as the doc frames it is
impossible (no worker-level "moderate" severity exists). The cleanest fix is
a different one — see §6.

---

## §1 Plain-English explanation (for a non-expert)

Layer 4 has two LLM actors:

1. A **worker** (Claude Sonnet, the `CausalRoleClassifier`). It looks at a
   feature and assigns a *causal role* — one of six labels: `instrument`,
   `confounder`, `ancestor`, `mediator`, `collider`, `descendant`. It does
   **not** assign a severity.
2. An **evaluator** (Claude Haiku). It reads the worker's role + rationale
   and writes an audit: "did the worker cite the temporal filter and the
   Pearl arrowheads? satisfied yes/no, here are the axes it missed."

A *separate*, non-LLM, purely statistical layer — **Layer 3 (adversarial)**
— is the only thing in the system that ever produces the severity word
**"moderate"**. "Moderate" means "a permutation test found a 3σ–5σ signal:
suspicious but not a clear leak."

The final `severity` that R1 reads is the **ensemble** severity, produced by
combining all layers in the `EnsembleVoter`. Here is the trap, in three
steps:

- **The worker's role only ever maps to `high` or `info`.** A "leak" role
  (mediator/collider/descendant) → `high`. An "accept" role
  (ancestor/confounder/instrument) → `info`. There is no role that maps to
  `moderate`. (`ensemble_voter._llm_severity`.)
- **`moderate` only comes out of the voter's branch 6** — "adversarial said
  moderate and *nothing else* spoke." That branch is reached *only when the
  LLM verdict is absent* (`sanitised_llm is None`).
- **The evaluator audit only ever rides along with a valid-role worker
  verdict.** When the worker returns a role outside the six-word vocabulary,
  the loader (`classify_feature`) bails out and returns `None` **before** it
  even calls the evaluator. So an audit can never be attached to a
  "no valid role" verdict.

Put the three together: an audit is present **iff** the worker produced a
valid role **iff** the ensemble severity is `high` or `info` — **never
`moderate`**. R1 fires only on (`moderate` AND audit-present). That
combination is structurally impossible on the production path. R1 is dead
code. Its shadow column `would_promote_severity` will be `NULL` on every
real row, forever.

The reason this slipped through: R1's *logic* is correct and is unit-tested
by hand-feeding it `("moderate", audit)` pairs. But the pipeline can never
hand it that pair. The bug is not in `evaluate_r1`; it is in the assumption
(baked into §4 of the design) that "worker severity" is a thing that
co-exists with the audit at the `moderate` level. It is not.

---

## §2 Verified evidence (file:line)

### 2.1 The worker has no severity; its role maps only to high/info

- `src/data/kg/types.py:319-349` — `LLMVerdict` fields are `causal_role`,
  `mechanism`, `recommended_remediation`, `cited_pmids`, `evaluator_audit`.
  **No `severity` field.** The worker does not emit a severity at all.
- `src/data/kg/ensemble_voter.py:433-437` — `_llm_severity(role)` returns
  `"high"` for leak roles, `"info"` otherwise. There is **no path to
  `"moderate"`** from a role.
- Empirical confirmation (live `EnsembleVoter`, adversarial=moderate +
  audit-bearing valid-role LLM verdict):

  | worker role | `_llm_severity` | ensemble `severity` |
  |---|---|---|
  | mediator / collider / descendant | high | **high** |
  | ancestor / confounder / instrument | info | **info** |

  No valid role yields `moderate`.

### 2.2 `moderate` is produced only by adversarial-alone branches (no LLM)

- `src/data/kg/ensemble_voter.py:866-885` — branch 6
  ("Adversarial moderate alone"): `severity="moderate"`,
  `decided_by="adversarial"`, `final_role=None`. Reached only after
  `sanitised_llm is None` checks above it have all fallen through; the
  branch fires when adversarial moderate is the only signal.
- `src/data/kg/ensemble_voter.py:611-617` — `sanitised_llm` is set to `None`
  whenever `llm_verdict.causal_role not in VALID_LLM_ROLES`.
- The voter's LLM path (`ensemble_voter.py:772-837`, rule 4) is the **only**
  branch that consumes a present `llm_verdict`, and it sets
  `severity = _llm_severity(...)` → high/info, never moderate.

### 2.3 The audit is attached only to valid-role worker verdicts

- `src/data/causal_role_classifier_loader.py:532-539` — in `classify_feature`,
  if `_coerce_role(...)` returns `None` (role outside vocabulary), the
  function logs and **returns `None` immediately**.
- `src/data/causal_role_classifier_loader.py:578-598` — the `LLMVerdict` is
  constructed (line 578) and the evaluator runs (`_build_evaluator` /
  `_run_evaluator`, lines 585-595) **only after** the valid-role gate. The
  audit is attached via `dataclasses.replace(worker_verdict,
  evaluator_audit=audit)` at line 598.
- Net: a verdict carrying an `evaluator_audit` **always** has a valid role.
  An invalid-role verdict never reaches the voter from production (it is
  `None`), so the voter's `sanitised_llm is None` sanitiser at line 612 is a
  defense-in-depth guard for a case the loader already prevents.

### 2.4 The orchestrator wiring closes the loop

- `src/agents/.../adaptive_validity_check.py:3008-3012` — Layer 4 fires when
  `adv_severity_pre == "moderate"` OR (`"high"` AND `layer_1_declared_safe`).
  So when adversarial is moderate, the LLM **is** invoked
  (`classify_feature`, lines 3020-3025) and forwarded to the voter
  (`_compose_legacy_verdict(..., llm_verdict=llm_verdict)`, line 3048).
- Two production sub-cases when adversarial is moderate:
  - **Worker returns a valid role** → `llm_verdict` non-None →
    voter rule 4 → severity `high`/`info`, audit present → **R1 cannot fire
    (severity not moderate).**
  - **Worker returns an invalid role / LM not configured / call raised** →
    `classify_feature` returns `None` → `llm_verdict is None`:
    - If no KG signal: `_compose_legacy_verdict:1695-1701` **bypasses the
      voter** entirely → `_legacy_adversarial_alone_verdict`
      (`:1394-1401`) **hardcodes all three shadow flags to `None`.**
    - If KG signal present: voter branch 6 → severity `moderate`,
      `llm_input is None` → audit absent → **R1 cannot fire (no audit).**

### 2.5 The shadow call site reads ensemble severity, not worker severity

- `src/agents/.../adaptive_validity_check.py:1233-1235` — the shadow loop
  calls `_rule_fn(verdict.severity, llm_audit)` where `verdict` is the
  `EnsembleVerdict` and `llm_audit = verdict.llm_input.evaluator_audit`.
  Confirming: the input pair fed to R1 is `(ensemble_severity, audit)` — the
  exact pair §1 shows can never be `(moderate, present)`.

### 2.6 Empirical reproduction (live voter + live `evaluate_r1`)

| scenario | ensemble severity | audit present | R1 | R2 | R3 |
|---|---|---|---|---|---|
| A: valid role + audit + adv-moderate (real prod path) | `info` | yes | **None** | True | True |
| B: no LLM, adv-moderate alone (real prod path) | `moderate` | no | None | None | None |
| C: invalid role **forced directly into voter** w/ audit | `moderate` | yes | **'high'** | True | True |

Scenario C is the *only* way to make R1 fire — and it requires injecting an
invalid-role `LLMVerdict` that still carries an audit **directly into the
voter**, bypassing the loader. The production loader never produces such an
object (§2.3). So C is unreachable; A and B are the real paths, and R1 is
`None` in both.

### 2.7 Test coverage does not catch this

- `tests/unit/test_data/test_evaluator_promotion_rules.py:41-43` — asserts
  `evaluate_r1("moderate", audit) == "high"`. Correct rule logic, but the
  `("moderate", audit)` pair is hand-built; nothing proves the pipeline can
  produce it.
- `tests/integration/test_audit_evaluator_shadow_byte_identity.py:91,129-146`
  — constructs an `EnsembleVerdict` directly with
  `llm_input=_llm_verdict_with_audit(audit)` and `severity="moderate"`. This
  bypasses `EnsembleVoter.vote`, so it too exercises the impossible pairing.
  No test routes a moderate-adversarial feature through the real voter +
  loader and asserts whether R1 can fire end-to-end.

**Conclusion §2: the finding is fully verified.** No production path produces
`(ensemble_severity == "moderate" AND evaluator_audit present)`.

---

## §3 Which of R1 / R2 / R3 are reachable

The key discriminator: **does the rule gate on `severity == "moderate"`?**

| rule | trigger | gates on severity? | reachable in prod? |
|---|---|---|---|
| **R1** | `severity=="moderate"` AND `satisfied==False` AND `≥1 missed` | **yes (moderate)** | **NO — dead** |
| **R2** | `satisfied==False` AND `≥2 missed` | no | **YES** |
| **R3** | `rationale_complete==False` | no | **YES** |

- **R1 is dead** for the reasons in §1–§2: it requires `moderate` + audit,
  an impossible conjunction.
- **R2 and R3 are reachable.** They ignore `worker_severity` entirely
  (`evaluator_promotion_rules.py:74-100` and `:108-128`; `del
  worker_severity`). They fire on any audit-bearing verdict that meets their
  audit-field conditions. Audit-bearing verdicts always have ensemble
  severity `high` or `info` (§2.1) — and R2/R3 fire on exactly those. The
  reproduction (§2.6 scenario A: severity `info`, audit present) shows R2 and
  R3 both fire. So `would_flag_for_review` and `rationale_incomplete_flag`
  will populate with real data; `would_promote_severity` will not.

Practical impact on Stage 1 acceptance: AC1.3 ("per-rule firing-rate table
built from the shadow column") will show **R1 firing rate = 0 by
construction**, not because the empirical pattern is rare. The Stage-1 data
collection cannot inform the R1 decision at all. R2/R3 firing rates will be
genuine.

---

## §4 Option 2 — "run the evaluator on the adversarial-moderate-alone path
too" — deep feasibility

### 4a. What the evaluator needs as input

`CausalRoleEvaluator.evaluate` (`src/data/causal_role_evaluator.py:223-256`)
requires: `feature_name`, `derivation_pseudocode`, `dataset_context`, and a
`worker_verdict: LLMVerdict` — from which it reads
`worker_verdict.causal_role`, `.mechanism`, `.recommended_remediation`. Its
criteria text (`:31-62`) audits **the worker's causal-role reasoning**:
"did the mechanism cite the temporal filter? did it name the Pearl
arrowheads (ancestor edge / fork / collider / instrument constraint)? does
the remediation match the role?"

### 4b. Can it produce a meaningful verdict for adversarial-moderate-alone?

**No — not without inventing the very thing it is supposed to audit.** The
adversarial-moderate-alone case (`ensemble_voter.py:866`) has, by
definition, **no causal-role verdict**: `final_role=None`, no mechanism, no
remediation. Every one of the evaluator's four criteria references a worker
artifact that does not exist here:

1. "temporal filter cited in the mechanism" — there is no mechanism.
2. "Pearl arrowheads identified" — no role was claimed, so there are no
   arrowheads to check.
3. "remediation matches the role" — no role, remediation is `review`.
4. "no leakage red flags missed in the rationale" — no rationale.

To make the evaluator emit a `satisfied`/`missed_considerations`, you would
have to feed it a fabricated `worker_verdict`. Its `satisfied=False` would
then mean "the (fabricated/empty) rationale was inadequate" — which is
**trivially and uninformatively true for every adversarial-only feature**.
R1's `satisfied==False` would be noise, not signal. This is a labeling-shaped
fix to a functional gap: it makes the column populate without making the
column *mean* anything. Per REASON-BEFORE-RULES this is the wrong move.

A semantically-honest Option 2 would require a *different* evaluator that
audits the **adversarial/statistical** evidence ("is this 3σ–5σ signal a
plausible leak given the feature's derivation?") rather than the worker's
causal-role reasoning. That is a new prompt, new criteria, arguably a new
model role — effectively #242-adjacent net-new work, not "run the existing
evaluator on one more path."

### 4c. What code would change

Minimal mechanical version (the one that produces noise):
- `adaptive_validity_check.py` ~`:3008-3036`: when `llm_verdict is None` and
  `adv_severity_pre == "moderate"`, synthesize a placeholder `LLMVerdict`
  (role=? mechanism=""), call `_build_evaluator()` + `_run_evaluator()`
  directly, attach the audit, and force routing through the voter
  (skip the `_legacy_adversarial_alone_verdict` bypass at `:1695-1701`).
- This is ~30–60 LoC plus test churn, but it is the **noise** version.

Semantically-honest version:
- New evaluator module (criteria auditing statistical evidence, not
  causal-role reasoning), new signature, new prompt, model selection.
- Decide what "satisfied" means for a feature with only a z-score.
- Re-baseline the entire Stage-1/2/3 AC framing because the audit semantics
  changed. This is a multi-day design + build, overlapping #242.

### 4d. Does it make R1's `satisfied==False` meaningful?

The mechanical version: **no** — it is noise (4b). The honest version: yes,
but only by redefining the evaluator's job, at which point R1's precondition
("worker said maybe-problematic, evaluator independently flagged missed
considerations") no longer describes what is happening (there is no worker
verdict to independently corroborate).

**Option 2 size/risk:** mechanical = small code / **high semantic risk**
(populates a meaningless column, plus a real cost increase: a Haiku call now
fires on every adversarial-moderate feature where Layer 4 previously
produced nothing — violating the §5 R-3 cost posture and AC1.4). Honest =
large effort, overlaps #242, changes the whole AC framing.

---

## §5 Option 3 — "redefine R1 to read a severity source that co-occurs with
an audit"

The doc's framing presumes a **worker-level "moderate"** severity that is
distinct from the ensemble severity and is paired with an audited
causal-role verdict.

**No such severity exists.** Verified:
- The worker (`LLMVerdict`) has no severity field at all
  (`types.py:319-349`).
- The only `"moderate"` producer in the codebase that touches the
  audit-bearing path is the **Layer-3 adversarial** statistical severity
  (`adaptive_validity_check.py:573,626`, surfaced as `severity` /
  `severity_pre_joint_check`). That is a permutation-test band, not a
  worker causal-role judgment, and the audit does not describe it.
- A grep of all `"moderate"` literals in `src/` (excluding tests/comments)
  finds only: the adversarial ladder, the ensemble branch 6, unrelated
  subsystems (`sensitivity.py`, `interpretation.py`, `detect_class_imbalance`,
  chatbot confidence). None is a worker-level causal-role severity.

So Option 3 *as written* cannot be implemented — there is nothing to point
R1 at. A near-cousin would be: **redefine R1 to read `severity_pre_joint_check`
== "moderate"** (the Layer-3 z-band, available on `adversarial_input`). On a
feature that fired Layer 4, that band is moderate *and* an audit exists
(scenario A). But this changes R1's meaning from "the worker's moderate
verdict was escalated" to "a statistically-ambiguous feature whose worker
gave a valid role got a dissatisfied audit." That is a defensible *new* rule
("escalate ambiguous-signal features whose worker reasoning the evaluator
distrusts"), but it is **not** the R1 the design defends as "the most
defensible," and the escalation target (`moderate→high`) no longer maps to a
worker severity that was `moderate`. It would need its own AC2 reviewer
labeling to establish it is not noise.

---

## §6 Recommendation

**Primary recommendation: do NOT pursue Option 2 (mechanical) and do NOT
pursue Option 3 as written. Reframe R1 instead, and back the reframing with a
reachability test the current suite lacks.**

Reasoning (intent → harm → fix):

- **Intent of R1** (design §4): "the worker said *maybe*-problematic, the
  evaluator independently flagged specific missed considerations → escalate
  to *definitely* problematic." This intent presupposes a worker "maybe"
  level. The architecture has no such level — the worker speaks in roles, and
  roles are binary (leak→high / accept→info). The premise is void.
- **Harm right now:** none user-facing. Stage 1 is shadow-only; the dead R1
  column is silently always-NULL. The harm is *epistemic*: Stage-1 data
  collection (AC1.3) cannot inform any R1 decision, and Stage 3 would gate on
  a rule that can never fire, so the gate would be a no-op the team might
  ship believing it does something.
- **The actual question** #240 is trying to answer is: *"when should the
  evaluator's dissatisfaction escalate a feature's disposition?"* The honest
  answer given this architecture:
  - The audit only ever sits on `high`/`info` (valid-role) verdicts.
  - An `info` (accept-role) verdict with a *dissatisfied* audit and missed
    considerations is the **real** "worker said keep, evaluator distrusts the
    reasoning" case — and it is **reachable** (scenario A produced exactly
    `info` + dissatisfied audit, and R2/R3 fired on it).
  - So the defensible escalation rule is **`info → moderate` (or → flag)** on
    a dissatisfied audit with missed considerations — *not* `moderate → high`.
    This matches the empirically reachable data and matches the design's
    stated intent ("evaluator distrust escalates disposition") far better
    than the impossible `moderate → high`.

**Concrete path:**

1. **Escalate to the user / issue #240 as a design correction**, not a code
   bug. R1's `moderate → high` precondition is architecturally unreachable;
   the design doc §4 conflates "worker severity" with "ensemble severity" and
   assumes a worker `moderate` that does not exist. This needs a product
   decision, because changing R1's direction changes the gate's semantics.
2. **Preferred redefinition (low risk, reachable, intent-preserving):**
   R1' = `worker produced an accept-role (ensemble `info`) AND
   `evaluator_audit.satisfied==False` AND `≥1 missed_consideration`` →
   propose escalation `info → moderate` (route to review), not `high`. This
   is exactly the signal R2 already partially captures; R1' would be the
   stricter "escalate disposition" variant. It reads only data that exists on
   the audit-bearing path. Update `evaluate_r1` precondition + the migration
   042 comment + design §4.
3. **Add the missing end-to-end reachability test** regardless of which
   redefinition is chosen: route a moderate-adversarial feature through the
   real `EnsembleVoter` + a stubbed `classify_feature` returning a valid-role
   audit-bearing verdict, and assert which shadow flags can fire. This test
   would have caught the dead-R1 gap and will prevent re-introduction.
4. **If the team insists on a `moderate → high` escalation**, the only honest
   way is the semantically-correct Option 2 (a *new* evaluator that audits the
   statistical/adversarial evidence for adversarial-moderate-alone features) —
   which is net-new design overlapping #242 and should be scoped as its own
   issue, not a tweak to Stage 1.

**On Option 2 vs Option 3 directly:** Option 3-as-written is impossible (no
worker moderate severity). Option 2-mechanical populates a meaningless column
and adds cost. Option 2-honest is large and #242-shaped. The redefinition in
(2) above is cleaner than either: it is reachable, cheap (pure function
change, no new LM calls), preserves the design's intent, and aligns R1 with
the data the pipeline actually produces.

**Bottom line:** the finding is correct. R1 is dead on arrival. R2/R3 are
fine. The fix is not "run the evaluator somewhere new" — it is "point R1 at
the severity transition that is actually reachable and audited (`info →
moderate`)," escalated to the user because it is a semantic change to a
defended acceptance criterion.
