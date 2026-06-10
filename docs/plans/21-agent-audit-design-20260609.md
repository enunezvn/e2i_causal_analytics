# 21-Agent Audit — Design Spec (2026-06-09)

**Status:** design, awaiting user review → `writing-plans`
**Type:** audit-only (NO code changes, NO remediation, NO deploy in this phase)
**Owner:** enunezvn

## 1. Goal

Produce one evidence-backed verdict report covering all 21 production agents across
**four dimensions**, using a **layered, OOM-safe, cheapest-disproof-first** method that
front-loads cheap static/SQL checks so that any end-to-end run is a *meaningful* test, not
a surprise failure.

The two hard constraints, in the user's words:

1. **Do not trigger OOM kills** on the prod/dev droplet.
2. **Do not let e2e fail because a cheaper layer should have caught it.**

## 2. Scope

All 21 agents, treated **identically** (no shortcut for recently-merged agents — a merged
PR proves the code was right at merge time on the premise as then understood; it does not
prove HEAD hasn't drifted or that the premise still holds):

| # | Agent | Dir | OOM weight |
|---|-------|-----|-----------|
| 1 | scope_definer | `ml_foundation/scope_definer` | heavy |
| 2 | cohort_constructor | `cohort_constructor` | heavy |
| 3 | model_selector | `ml_foundation/model_selector` | heavy |
| 4 | model_deployer | `ml_foundation/model_deployer` | heavy |
| 5 | observability_connector | `ml_foundation/observability_connector` | heavy |
| 6 | drift_monitor | `drift_monitor` | medium |
| 7 | resource_optimizer | `resource_optimizer` | light |
| 8 | prediction_synthesizer | `prediction_synthesizer` | heavy |
| 9 | gap_analyzer | `gap_analyzer` | medium |
| 10 | heterogeneous_optimizer | `heterogeneous_optimizer` | heavy |
| 11 | health_score | `health_score` | medium |
| 12 | experiment_monitor | `experiment_monitor` | light |
| 13 | orchestrator | `orchestrator` | light |
| 14 | tool_composer | `tool_composer` | medium |
| 15 | causal_impact | `causal_impact` | medium |
| 16 | explainer | `explainer` | light |
| 17 | data_preparer | `ml_foundation/data_preparer` | heavy |
| 18 | feature_analyzer | `ml_foundation/feature_analyzer` | heavy |
| 19 | model_trainer | `ml_foundation/model_trainer` | heavy (loky 5.9 GB observed) |
| 20 | experiment_designer | `experiment_designer` | medium |
| 21 | feedback_learner | `feedback_learner` | medium |

OOM weight governs **execution scheduling only** (serial vs. batched, memory ceiling),
**not** audit depth. Depth is uniform.

## 3. Non-goals (this phase)

- No remediation / PRs / migrations / deploy. Findings only.
- No whole-tree `mypy src/` or full `pytest` on the droplet — CI is the arbiter
  (CLAUDE.md). Any code-level check is scoped to the relevant files.
- No shipping of synthetic data as real. Fixtures are test-time only (see §6).

## 4. The four dimensions — cheapest disproof per dimension

For each agent and each dimension we answer with the cheapest faithful evidence first, and
escalate only on residual doubt.

### D1 — Functional integrity (no silent mocks)
- **Question:** does the agent produce its output from real computation on a real data
  path, or return plausible-fake / hardcoded / random / all-default values?
- **Static (free):** mock-signal scan over the agent's nodes —
  `random.uniform`, `np.random`, hardcoded `ate=`/`confidence=`/`p_value=`/`0.12`/`0.85`,
  `# Placeholder|Mock|Stub|TODO: real`, "actual X would go here" docstrings,
  structured returns with all-default/all-zero fields. Then **read the terminal output
  node** and trace whether the returned value is computed or constant.
- **Probe (cheap):** invoke just the agent's entry on a tiny faithful input; assert the
  output varies with input (a constant-return mock won't).
- **Verdict cells:** `PASS-real` / `PASS-synthetic` / `SILENT-MOCK` / `PARTIAL-MOCK` / `UNVERIFIED`.
- **REASON-BEFORE-RULES caveat:** a mock is not automatically a defect. Before classifying,
  investigate intent (git log / PR / linked issue / inline docstring) and harm
  (prod-reachable? plausible-wrong? user-visible?). Classify per the 4-way framework
  (HARMFUL-NOW / REWIRE / KEEP-AS-INTENTIONAL-PLACEHOLDER / DELETE), not on pattern-match.

### D2 — Wiring & reachability
- **Question:** is the agent registered in the factory/registry, routed to by the
  orchestrator, and consumed?
- **Static (free):** membership in `factory.py` / `agent_registry.py`; presence in the
  dispatcher method-map and router/intent-classifier; downstream consumers via grep.
- **Caveat (from orchestrator-classifier + FE/BE audits):** "registered" ≠ "reachable."
  Route-ORDER shadows and degraded-only reachability hide behind a green registry. Where
  reachability depends on runtime order, confirm with the live matcher / dispatcher, not
  just a grep.
- **Verdict cells:** `WIRED` / `UNWIRED` / `SHADOWED` / `DEGRADED-ONLY`.

### D3 — Data-substrate validity
- **Question:** even if the code is correct, does the underlying data support it?
- **Static (free):** what table/columns/grain the agent reads (from its repository/query).
- **Probe (cheap SQL):** `COUNT(*)` (is prod populated?), positive-rate / null-rate
  (degenerate outcome? — e.g. the 94.7%-positive `treatment_initiated`), and a
  leakage-safe single-feature AUC ceiling where an outcome model is involved. These are
  cheap aggregates, low memory — NOT full e2e.
- **Empty-prod path:** if the substrate is absent, escalate to a **faithful synthetic
  fixture** (§6) to prove the *code path*, and mark `PASS-synthetic` (never `PASS-real`).
- **Verdict cells:** `REAL-OK` / `DEGENERATE` / `LEAKY` / `EMPTY-PROD→synthetic` / `UNKNOWN`.

### D4 — Resource / OOM behavior
- **Question:** does the agent spike memory in a way that would kill the droplet?
- **Static (free, predicts without triggering):** sklearn `n_jobs=-1`/loky without cap;
  full-width parquet reads; unbounded frame loads; `joblib.Parallel`; threadpool fan-out.
- **Probe (guarded):** only if static is ambiguous, run under the §5 harness with RSS
  monitoring and record peak.
- **Verdict cells:** `SAFE` / `SPIKE-RISK (quantified)` / `UNBOUNDED`.

## 5. OOM-safety harness

Core realization: **fan-out is safe; execution is not.** Subagents are LLM processes that
do not import torch/sklearn. Memory only spikes when a `Bash` step *runs agent code or
loads parquet*. So parallelism is gated on whether the step executes agent code — not on
agent count.

- **Static analysis (D1/D2/D4-static, D3-static): fully parallel.** Read/grep/git-log only.
- **Execution (D1 probe, D3 SQL+fixture, D4 probe, any e2e): serialized, concurrency 1**,
  never fanned out, each wrapped in:
  - Env caps: `LOKY_MAX_CPU_COUNT=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
    MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1` (observed: collapses the loky 5.9 GB spike to
    ~1.1 GB peak).
  - Hard memory ceiling: `systemd-run --user -p MemoryMax=2G -p MemorySwapMax=0 …`; fall
    back to an RSS-polling watchdog (SIGTERM on breach) if no user systemd session. A
    runaway kills *itself*, not the box.
  - Parquet: read ≤9 columns; use the **sampled** marts (50K rows), never full 814K width.
  - `free -m` pre-flight before each execution step; if free memory is under a floor,
    wait/abort rather than race the box.
- **No whole-tree `mypy`/`pytest` on the droplet.** Scope to relevant files; CI is arbiter.

## 6. Faithful synthetic fixture policy

Used only when D3 finds the prod substrate empty/absent, to prove the **code path** works.

- **Provenance order:** (1) reuse an existing faithful generator in the repo
  (twin `synthetic_uplift_v1`, optum synthetic-claim generator/plan, converters) →
  (2) sample+anonymize a faithful schema slice if any real rows exist →
  (3) hand-roll a schema-matched fixture only if neither exists.
- **Fidelity requirements:** correct columns + dtypes, correct PG ENUM values (e.g.
  `brand_type`, `agent_name`, `event_type` — arbitrary strings 22P02-fail, which is itself
  a faithfulness check), correct grain, and **non-degenerate** distributions (a fixture
  that accidentally reproduces a 94.7%-positive outcome proves nothing).
- **Truth labeling:** an agent verified only on a fixture is `PASS-synthetic`, never
  `PASS-real`. The report states the fixture's provenance and what real-data question
  remains open.
- **Disposal:** fixtures are test-time artifacts; no fixture row is left in a prod table
  (the dspy-loop live-DB-pollution lesson → clean up / use an offline guard).

## 7. Phased pipeline

- **Phase 0 — Inventory + harness (cheap, once).** For each agent build an *audit card*:
  dir, entry point, terminal output node(s), declared data dependency (table/cols/grain),
  registry key, dispatcher route, sklearn/parquet footprint. Confirm the §5 harness works
  on this box (systemd-run availability; a `free -m` baseline).
- **Phase 1 — Static screen, all 21 in parallel + adversarial refute.** Each agent gets a
  static verdict across D1/D2/D4 and D3-static, **plus an explicit claim-list** of what
  static could not settle (these are the things that surprise-fail e2e). Then an
  **independent agent tries to REFUTE each PASS** (ralph-loop discipline) — a clean verdict
  is not accepted until a second pass fails to find the silent mock / unwired path.
- **Phase 2 — Targeted probes for survivors (serial, guarded).** Probes aimed at the
  claim-list, not "run the whole thing": cheap SQL for D3, single-agent invocation for D1,
  guarded peak-RSS run for ambiguous D4. Under the §5 harness.
- **Phase 3 — Fixture / e2e for the residue only (serial, guarded).** Faithful synthetic
  fixture for empty-prod agents; full orchestrator→agent→DB e2e reserved for the few where
  invocation is the only faithful test and static+probe left genuine doubt.
- **Phase 4 — Synthesize report.** Per-agent verdict table + evidence links + explicit
  "not verified & why" per agent. Lands in `docs/reports/`.

## 8. Report format

One row per agent, four dimension columns, each cell an evidence link (file:line or probe
output):

| Agent | D1 functional | D2 wiring | D3 data | D4 OOM | Overall | Not-verified & why |
|-------|--------------|-----------|---------|--------|---------|--------------------|

Verdict vocab per §4. "Overall" rolls up to one of: `CLEAR-real`, `CLEAR-synthetic`,
`FINDING` (with severity), `FAIL-CLOSED-UNVERIFIED`.

## 9. Execution model (ultracode → Workflow)

- Phase 1 static = `parallel`/`pipeline` fan-out of read-only subagents (one per agent),
  schema-constrained structured output (audit card + claim-list + verdict).
- Adversarial refute = a second subagent per PASS, prompted to refute; majority/independent
  verdict gates acceptance.
- Phase 2/3 execution = a **serial** stage (concurrency 1) under the §5 harness — explicitly
  NOT fanned out, regardless of the workflow's default concurrency cap.
- Phase 4 = a synthesis agent over the structured results.

## 10. Risks & mitigations

| Risk | Mitigation |
|------|-----------|
| False-green static (the thing the user fears) | claim-list + adversarial refute before any PASS is accepted |
| OOM kill on the droplet | fan-out only for read-only static; execution serialized + env-capped + memory-ceilinged + parquet-narrow + `free -m` pre-flight |
| Non-faithful fixture → false green (#504) | schema/ENUM/grain fidelity + non-degenerate distribution + reuse existing generators + `PASS-synthetic` ≠ `PASS-real` |
| Re-deriving settled findings | not avoided by design choice — user chose "treat identically"; recent-PR context is used only as a *lead to refute*, not as a pass |
| Pattern-matching a mock as a defect | REASON-BEFORE-RULES intent+harm investigation precedes every D1 classification |

## 11. Deliverable

`docs/reports/21-agent-audit-20260609.md` + a `docs/reports/21-agent-audit-20260609-repro/`
directory for any fixture/probe scripts, mirroring the existing `*-repro/` convention.
