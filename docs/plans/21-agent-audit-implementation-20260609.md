# 21-Agent Audit — Implementation Plan

> **For agentic workers:** This is an **audit** (read-only report), not a feature build. There is no production-code TDD/commit cycle. The executable artifacts are: (a) a serial OOM-safe probe harness, (b) a Phase-1 static-screen Workflow, (c) per-claim probe/fixture scripts, (d) a synthesis step. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce one evidence-backed verdict report over all 21 agents × 4 dimensions (functional / wiring / data-substrate / OOM), layered and OOM-safe, with adversarial refutation of every PASS so e2e never surprise-fails.

**Architecture:** Read-only static screen fans out in parallel (subagents never import agent code); everything that executes agent code or loads parquet is serialized under a `systemd-run` memory ceiling. Data-substrate checks run on Supabase (server-side, zero droplet memory). Empty-prod agents are proven on faithful synthetic fixtures and labeled `PASS-synthetic`, never `PASS-real`.

**Tech Stack:** Python 3.12, Workflow tool (subagent fan-out), Supabase MCP `execute_sql`, `systemd-run --user`, pandas/pyarrow (column-narrowed), existing repo synthetic generators.

**Design spec:** `docs/plans/21-agent-audit-design-20260609.md`

---

## Grounded facts (verified 2026-06-09, do not re-derive)

- **OOM ceiling works:** `systemd-run --user --scope -p MemoryMax=2G -p MemorySwapMax=0 -- <cmd>` → exit 0. User manager running, `XDG_RUNTIME_DIR=/run/user/1000`.
- **Memory is tight:** free ≈ 4.3 GB / 16 GB; **swap 99.9% full**. OOM = hard kill. Pre-flight floor before every probe: require **≥ 3 GB available** (`free -m` → "available" column) or wait/abort.
- **Registry source of truth:** `src/agents/factory.py::AGENT_REGISTRY_CONFIG` (21 entries, each with `class_name`). DB mirror: `agent_registry` table.
- **Functional tiers (execution order):**
  - T0: scope_definer, cohort_constructor, data_preparer, feature_analyzer, model_selector, model_trainer, model_deployer, observability_connector
  - T1: orchestrator, tool_composer
  - T2: causal_impact, gap_analyzer, heterogeneous_optimizer
  - T3: drift_monitor, experiment_designer, experiment_monitor, health_score
  - T4: prediction_synthesizer, resource_optimizer
  - T5: explainer, feedback_learner
- **Static fan-out cap:** `nproc=8` → ≤ 6 concurrent read-only subagents.
- **Dispatcher map:** `src/agents/orchestrator/nodes/dispatcher.py` + `tests/.../test_dispatcher_method_map.py` (consumer grep must NOT be `head`-truncated — prior miss).

---

## Phase 0 — Harness + shared context

### Task 0.1: Write the reusable OOM-safe execution wrapper

**Files:**
- Create: `docs/reports/21-agent-audit-20260609-repro/oom_run.sh`

- [ ] **Step 1: Write the wrapper**

```bash
#!/usr/bin/env bash
# oom_run.sh — run a memory-bounded, single-threaded probe with a pre-flight floor.
# Usage: ./oom_run.sh <max_gib> <label> -- <command...>
set -euo pipefail
MAX_GIB="$1"; LABEL="$2"; shift 2
[ "$1" = "--" ] && shift
FLOOR_MB=3000
AVAIL_MB=$(free -m | awk '/^Mem:/{print $7}')
if [ "$AVAIL_MB" -lt "$FLOOR_MB" ]; then
  echo "ABORT [$LABEL]: only ${AVAIL_MB}MB available (<${FLOOR_MB}MB floor)"; exit 99
fi
echo "RUN [$LABEL]: cap=${MAX_GIB}G avail=${AVAIL_MB}MB"
exec systemd-run --user --scope --quiet \
  -p MemoryMax="${MAX_GIB}G" -p MemorySwapMax=0 \
  env LOKY_MAX_CPU_COUNT=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
      MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  "$@"
```

- [ ] **Step 2: Verify it caps and pre-flights**

Run: `chmod +x docs/reports/21-agent-audit-20260609-repro/oom_run.sh && ./docs/reports/21-agent-audit-20260609-repro/oom_run.sh 2 selftest -- python -c "print('ok')"`
Expected: prints `RUN [selftest]: cap=2G avail=…MB` then `ok`. (A `MemoryError`/kill on breach hits the child only, never the box.)

### Task 0.2: Snapshot the registry + dispatcher (shared static context for Phase 1)

- [ ] **Step 1: Capture the registry roster and dispatcher map once**

Run:
```bash
sed -n '26,200p' src/agents/factory.py > /tmp/audit_registry.txt
grep -n "agent_name\|method_map\|_dispatch\|allow_mock\|_mock_agent_execution" \
  src/agents/orchestrator/nodes/dispatcher.py > /tmp/audit_dispatch.txt
wc -l /tmp/audit_registry.txt /tmp/audit_dispatch.txt
```
Expected: both files non-empty. These are passed to Phase-1 subagents as shared context so each doesn't re-read the whole registry.

---

## Phase 1 — Static screen (parallel) + adversarial refute  → Workflow

### Task 1.1: Run the static-screen Workflow

**File:** Create `docs/reports/21-agent-audit-20260609-repro/phase1_static.workflow.js`, then invoke via the Workflow tool (`scriptPath`).

- [ ] **Step 1: Write the Workflow script**

```javascript
export const meta = {
  name: '21-agent-static-screen',
  description: 'Static screen of all 21 agents across 4 dimensions + adversarial refute of every PASS',
  phases: [
    { title: 'StaticCard', detail: 'read-only static audit card per agent' },
    { title: 'Refute', detail: 'independent skeptic tries to break each PASS' },
  ],
}

const AGENTS = [
  // {name, dir, tier}
  {name:'scope_definer', dir:'src/agents/ml_foundation/scope_definer', tier:0},
  {name:'cohort_constructor', dir:'src/agents/cohort_constructor', tier:0},
  {name:'data_preparer', dir:'src/agents/ml_foundation/data_preparer', tier:0},
  {name:'feature_analyzer', dir:'src/agents/ml_foundation/feature_analyzer', tier:0},
  {name:'model_selector', dir:'src/agents/ml_foundation/model_selector', tier:0},
  {name:'model_trainer', dir:'src/agents/ml_foundation/model_trainer', tier:0},
  {name:'model_deployer', dir:'src/agents/ml_foundation/model_deployer', tier:0},
  {name:'observability_connector', dir:'src/agents/ml_foundation/observability_connector', tier:0},
  {name:'orchestrator', dir:'src/agents/orchestrator', tier:1},
  {name:'tool_composer', dir:'src/agents/tool_composer', tier:1},
  {name:'causal_impact', dir:'src/agents/causal_impact', tier:2},
  {name:'gap_analyzer', dir:'src/agents/gap_analyzer', tier:2},
  {name:'heterogeneous_optimizer', dir:'src/agents/heterogeneous_optimizer', tier:2},
  {name:'drift_monitor', dir:'src/agents/drift_monitor', tier:3},
  {name:'experiment_designer', dir:'src/agents/experiment_designer', tier:3},
  {name:'experiment_monitor', dir:'src/agents/experiment_monitor', tier:3},
  {name:'health_score', dir:'src/agents/health_score', tier:3},
  {name:'prediction_synthesizer', dir:'src/agents/prediction_synthesizer', tier:4},
  {name:'resource_optimizer', dir:'src/agents/resource_optimizer', tier:4},
  {name:'explainer', dir:'src/agents/explainer', tier:5},
  {name:'feedback_learner', dir:'src/agents/feedback_learner', tier:5},
]

const CARD_SCHEMA = {
  type:'object', additionalProperties:false,
  required:['agent','d1','d2','d3','d4','overall','probe_plan'],
  properties:{
    agent:{type:'string'},
    d1:{type:'object', additionalProperties:false,
      required:['verdict','terminal_node','mock_signals','computed_or_constant','intent_note','claims'],
      properties:{
        verdict:{enum:['PASS-real-candidate','SILENT-MOCK','PARTIAL-MOCK','UNVERIFIED']},
        terminal_node:{type:'string'},            // file:line producing the agent output
        mock_signals:{type:'array', items:{type:'string'}}, // "file:line — signal"
        computed_or_constant:{enum:['computed','constant','mixed','unclear']},
        intent_note:{type:'string'},              // REASON-BEFORE-RULES: if a mock exists, why (git/PR/docstring) + harm
        claims:{type:'array', items:{type:'string'}}, // what static CANNOT settle
      }},
    d2:{type:'object', additionalProperties:false,
      required:['verdict','in_registry','dispatch_route','consumers','claims'],
      properties:{
        verdict:{enum:['WIRED','UNWIRED','SHADOWED','DEGRADED-ONLY']},
        in_registry:{type:'boolean'},
        dispatch_route:{type:'string'},
        consumers:{type:'array', items:{type:'string'}},
        claims:{type:'array', items:{type:'string'}},
      }},
    d3:{type:'object', additionalProperties:false,
      required:['verdict','tables','columns','grain','degenerate_risk','claims'],
      properties:{
        verdict:{enum:['DECLARED','UNKNOWN','NO-DATA-DEP']},
        tables:{type:'array', items:{type:'string'}},
        columns:{type:'array', items:{type:'string'}},
        grain:{type:'string'},
        degenerate_risk:{type:'string'},          // outcome/leakage risk to SQL-probe in Phase 2
        claims:{type:'array', items:{type:'string'}}, // "COUNT(*) of X", "positive-rate of Y", etc.
      }},
    d4:{type:'object', additionalProperties:false,
      required:['verdict','signals','claims'],
      properties:{
        verdict:{enum:['SAFE','SPIKE-RISK','UNBOUNDED']},
        signals:{type:'array', items:{type:'string'}}, // "file:line — n_jobs=-1 / full parquet / unbounded frame"
        claims:{type:'array', items:{type:'string'}},
      }},
    overall:{enum:['CLEAR-pending-probe','FINDING','NEEDS-PROBE']},
    probe_plan:{type:'array', items:{type:'string'}}, // concrete Phase-2 probes
  }
}

const REFUTE_SCHEMA = {
  type:'object', additionalProperties:false,
  required:['agent','attacked','refuted','evidence','revised'],
  properties:{
    agent:{type:'string'},
    attacked:{type:'array', items:{type:'string'}}, // which PASS verdicts were challenged
    refuted:{type:'boolean'},                       // true = found a real defect the card missed
    evidence:{type:'array', items:{type:'string'}}, // file:line proof
    revised:{type:'array', items:{type:'string'}},  // corrected verdicts, if any
  }
}

const cardPrompt = (a) => `READ-ONLY static audit of the '${a.name}' agent at ${a.dir}.
You MUST NOT execute any Python, import agent code, or load any parquet/DB. Read + grep + git log only.

Produce a 4-dimension audit card:

D1 FUNCTIONAL INTEGRITY (no silent mocks):
- Find the agent entry (agent.py / graph.py) and the TERMINAL node(s) that build the agent's output. Give file:line.
- Grep the agent dir for mock signals: random.uniform, np.random, hardcoded ate=/confidence=/p_value=/0.12/0.85, "# Placeholder|Mock|Stub|TODO: real", "actual X would go here", all-default/all-zero structured returns.
- Decide whether the returned values are COMPUTED from inputs or CONSTANT.
- REASON-BEFORE-RULES: if any mock/placeholder exists, investigate INTENT (git log --diff-filter=A on the file, PR body, linked issue, inline docstring) and HARM (prod-reachable? plausible-wrong? user-visible?). A scaffolded placeholder for requested functionality is NOT a defect — say so in intent_note. Do not classify on pattern-match.
- List any claim you CANNOT settle statically (these drive Phase 2).

D2 WIRING & REACHABILITY:
- Is '${a.name}' in src/agents/factory.py AGENT_REGISTRY_CONFIG? (shared snapshot: /tmp/audit_registry.txt)
- How is it dispatched? (shared snapshot: /tmp/audit_dispatch.txt; full file src/agents/orchestrator/nodes/dispatcher.py). Watch for route-ORDER shadows and degraded-only reachability.
- Who consumes its output? grep src/ — do NOT head-truncate the grep.

D3 DATA-SUBSTRATE (static only):
- What table(s)/column(s)/grain does it read? From its repository/query code. If no data dependency, verdict NO-DATA-DEP.
- Note degenerate/leakage risk to SQL-probe later (e.g. an outcome that may be ~all-positive).
- List the exact SQL checks Phase 2 should run (COUNT, positive-rate, null-rate, distinct).

D4 OOM:
- Grep for n_jobs=-1 / loky / joblib.Parallel without a cap, full-width parquet reads, unbounded frame loads. Give file:line. Verdict SAFE/SPIKE-RISK/UNBOUNDED.

overall = FINDING if any dimension is already a clear defect; NEEDS-PROBE if claims remain; CLEAR-pending-probe if static is clean AND no claims remain.
probe_plan = the concrete probes Phase 2 must run (empty if none).`

const refutePrompt = (a, card) => `You are an adversarial skeptic. The static screen of '${a.name}' produced these PASS/clean verdicts:
${JSON.stringify({d1:card.d1.verdict, d2:card.d2.verdict, d3:card.d3.verdict, d4:card.d4.verdict, terminal:card.d1.terminal_node}, null, 2)}

Try to REFUTE them with file:line evidence (READ-ONLY, no execution). Specifically hunt:
- a silent mock / constant return the first pass mistook for computed (re-read the terminal node and everything it calls),
- a registry/route that looks wired but is shadowed or only reachable in a degraded/partial registry,
- a data dependency that is actually unpopulated/degenerate,
- an OOM spike the first pass missed.
Default to refuted=false ONLY if you genuinely cannot break it. Report what you attacked, whether you refuted, evidence, and revised verdicts.`

phase('StaticCard')
const results = await pipeline(
  AGENTS,
  (a) => agent(cardPrompt(a), {label:`card:${a.name}`, phase:'StaticCard', schema:CARD_SCHEMA}),
  (card, a) => {
    if (!card) return {agent:a.name, card:null, refute:null}
    const hasPass = card.d1.verdict==='PASS-real-candidate' || card.d2.verdict==='WIRED'
                 || card.d3.verdict==='DECLARED' || card.d4.verdict==='SAFE'
    if (!hasPass) return {agent:a.name, card, refute:null}
    return agent(refutePrompt(a, card), {label:`refute:${a.name}`, phase:'Refute', schema:REFUTE_SCHEMA})
      .then(r => ({agent:a.name, card, refute:r}))
  }
)
return results.filter(Boolean)
```

- [ ] **Step 2: Invoke the Workflow**

Invoke the Workflow tool with `{scriptPath: 'docs/reports/21-agent-audit-20260609-repro/phase1_static.workflow.js'}`.
Expected: 21 cards + refutations returned as structured JSON. Save the raw result to `docs/reports/21-agent-audit-20260609-repro/phase1_results.json`.

- [ ] **Step 3: Triage the output into the Phase-2 probe queue**

From the results, build a probe queue: every `probe_plan` entry from cards with `overall ∈ {NEEDS-PROBE, FINDING}` OR any card whose refutation set `refuted=true`. Group by probe type (SQL / invocation / fixture / OOM-measure). Record to `docs/reports/21-agent-audit-20260609-repro/probe_queue.md`.

---

## Phase 2 — Targeted probes for survivors (SERIAL, guarded)

> Probes are NOT fanned out. Run them one at a time. SQL probes go to Supabase (zero droplet memory); invocation probes go through `oom_run.sh`.

### Task 2.1: Data-substrate SQL probes (Supabase MCP — zero droplet memory)

- [ ] **Step 1: For each D3 claim, run a read-only aggregate via Supabase `execute_sql`**

Templates (substitute table/column from the card):
```sql
-- populated?
SELECT count(*) AS n FROM <table>;
-- degenerate outcome?
SELECT avg((<outcome_col>)::int)::numeric(6,4) AS positive_rate, count(*) AS n FROM <table>;
-- null/leakage smell?
SELECT count(*) FILTER (WHERE <col> IS NULL)::numeric / nullif(count(*),0) AS null_rate FROM <table>;
-- enum domain (faithfulness input for fixtures)?
SELECT <enum_col>, count(*) FROM <table> GROUP BY 1 ORDER BY 2 DESC LIMIT 20;
```
Expected: record each result + verdict (`REAL-OK` / `DEGENERATE` / `LEAKY` / `EMPTY-PROD`) into the agent's row. `EMPTY-PROD` → escalates to Task 3.1 (fixture).

### Task 2.2: Functional invocation probes (local, serial, memory-capped)

- [ ] **Step 1: Write a generic single-agent invocation probe**

**File:** Create `docs/reports/21-agent-audit-20260609-repro/probe_invoke.py`
```python
"""Invoke ONE agent's terminal logic on two distinct tiny inputs; a real path varies, a constant mock does not.
Usage: python probe_invoke.py <agent_name>  (run ONLY via oom_run.sh)."""
import sys, json
from src.agents.factory import AGENT_REGISTRY_CONFIG, create_agent_registry

def main(name: str) -> None:
    assert name in AGENT_REGISTRY_CONFIG, f"{name} not in registry"
    reg = create_agent_registry()                      # real wiring, no mock
    agent = reg[name] if isinstance(reg, dict) else reg.get(name)
    # Two distinct minimal states; the probe asserts output is input-sensitive.
    out_a = agent.run_probe({"_probe": "A"}) if hasattr(agent, "run_probe") else None
    out_b = agent.run_probe({"_probe": "B"}) if hasattr(agent, "run_probe") else None
    print(json.dumps({"agent": name, "has_probe_hook": out_a is not None,
                      "varies": out_a != out_b, "a": str(out_a)[:200], "b": str(out_b)[:200]}))

if __name__ == "__main__":
    main(sys.argv[1])
```
> Note: if an agent has no `run_probe` hook, the probe instead drives the agent's documented entry (from the card's `terminal_node`) with two minimal states — the per-agent invocation line is filled from the card, NOT guessed here.

- [ ] **Step 2: Run it under the harness, one agent at a time**

Run: `./docs/reports/21-agent-audit-20260609-repro/oom_run.sh 2 invoke-<name> -- python docs/reports/21-agent-audit-20260609-repro/probe_invoke.py <name>`
Expected: JSON with `varies=true` for a real computed path; `varies=false` is a SILENT-MOCK signal to confirm. Record per agent.

### Task 2.3: OOM measurement for ambiguous D4 only

- [ ] **Step 1: For any `SPIKE-RISK` the static pass couldn't quantify, measure peak RSS under a hard cap**

Run: `./docs/reports/21-agent-audit-20260609-repro/oom_run.sh 2 oom-<name> -- /usr/bin/time -v python docs/reports/21-agent-audit-20260609-repro/probe_invoke.py <name> 2>&1 | grep "Maximum resident"`
Expected: a peak-RSS number; if the run is killed by the 2 G cap, that itself is the `UNBOUNDED`/`SPIKE-RISK` confirmation (the box survives — only the child dies).

---

## Phase 3 — Faithful synthetic fixtures + e2e for the residue (SERIAL, guarded)

### Task 3.1: Build a faithful fixture for each `EMPTY-PROD` agent

- [ ] **Step 1: Resolve fixture provenance (reuse before hand-roll)**

Order: (1) an existing repo generator (twin `synthetic_uplift_v1`, optum synthetic-claim generator/plan, the `convert_*` scripts) → (2) sample+anonymize a faithful schema slice if any real rows exist → (3) hand-roll a schema-matched fixture. Record which was used.

- [ ] **Step 2: Assert fixture fidelity BEFORE using it**

The fixture loader MUST assert: correct columns + dtypes; PG ENUM values drawn from the Task-2.1 enum-domain query (arbitrary strings 22P02-fail = a faithfulness check); correct grain; and a **non-degenerate** target distribution (reject a fixture that reproduces a ~all-positive outcome — that proves nothing). Fail the fixture build if any assertion fails.

- [ ] **Step 3: Run the agent against the fixture under the harness**

Run via `oom_run.sh`. A clean run earns `PASS-synthetic` (never `PASS-real`). Record fixture provenance + the open real-data question in the agent's "not-verified & why" cell. **Clean up:** no fixture row persisted to a prod table (offline guard / explicit delete — the dspy-loop live-DB-pollution lesson).

### Task 3.2: Full e2e ONLY for genuine residue

- [ ] **Step 1: Identify the residue**

An agent gets full orchestrator→agent→DB e2e ONLY if invocation is the sole faithful test AND static+probe left genuine doubt. Most agents should NOT reach here. List them explicitly; if the list is empty, say so (no silent skip).

- [ ] **Step 2: Run each e2e serially under the harness**

Run via `oom_run.sh` with a 3 G cap, concurrency 1, `free -m` pre-flight. Record outcome.

---

## Phase 4 — Synthesize the report

### Task 4.1: Assemble the per-agent verdict report

**Files:** Create `docs/reports/21-agent-audit-20260609.md`

- [ ] **Step 1: Merge Phase 1–3 evidence into the report table**

One row per agent × four dimension columns, each cell an evidence link (file:line or probe output), verdict vocab per the design spec §8. Add the `Overall` rollup (`CLEAR-real` / `CLEAR-synthetic` / `FINDING(sev)` / `FAIL-CLOSED-UNVERIFIED`) and a per-agent "not-verified & why" line. Where a refutation flipped a verdict, cite the refutation evidence.

- [ ] **Step 2: Honest-coverage footer**

State explicitly: which agents are `PASS-real` vs `PASS-synthetic`, which are `FAIL-CLOSED-UNVERIFIED` and why, and any probe that was dropped/capped (no silent truncation). List counts.

- [ ] **Step 3: REASON-BEFORE-RULES verification of findings**

Before publishing any `SILENT-MOCK`/`UNWIRED` finding, the dispatcher independently verifies the central claim (intent investigation + harm + consumer count) against source — a refutation verdict is not authoritative until the claim is re-checked. Downgrade any finding that fails verification.

---

## Self-review (spec coverage)

- D1 functional → Tasks 1.1 (static+refute), 2.2 (invocation). ✓
- D2 wiring → Task 1.1 (registry/dispatch/consumers + refute). ✓
- D3 data-substrate → Tasks 1.1 (static), 2.1 (SQL), 3.1 (fixture). ✓
- D4 OOM → Tasks 1.1 (static), 2.3 (measure). ✓
- OOM harness (§5 spec) → Task 0.1 wrapper + serial Phase 2/3, fan-out only Phase 1. ✓
- Faithful fixtures (§6 spec) → Task 3.1 (provenance + fidelity asserts + PASS-synthetic + cleanup). ✓
- Adversarial refute (the anti-surprise-e2e mechanism) → Task 1.1 stage 2 + Task 4.1 step 3. ✓
- Report (§8 spec) → Task 4.1. ✓
- Treat-identically → AGENTS array has all 21, no tier shortcut; tier only orders Phase 2/3 execution. ✓
