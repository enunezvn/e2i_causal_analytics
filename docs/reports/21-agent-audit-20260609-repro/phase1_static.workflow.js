export const meta = {
  name: '21-agent-static-screen',
  description: 'Static screen of all 21 agents across 4 dimensions + adversarial refute of every PASS',
  phases: [
    { title: 'StaticCard', detail: 'read-only static audit card per agent' },
    { title: 'Refute', detail: 'independent skeptic tries to break each PASS' },
  ],
}

const AGENTS = [
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
        terminal_node:{type:'string'},
        mock_signals:{type:'array', items:{type:'string'}},
        computed_or_constant:{enum:['computed','constant','mixed','unclear']},
        intent_note:{type:'string'},
        claims:{type:'array', items:{type:'string'}},
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
        degenerate_risk:{type:'string'},
        claims:{type:'array', items:{type:'string'}},
      }},
    d4:{type:'object', additionalProperties:false,
      required:['verdict','signals','claims'],
      properties:{
        verdict:{enum:['SAFE','SPIKE-RISK','UNBOUNDED']},
        signals:{type:'array', items:{type:'string'}},
        claims:{type:'array', items:{type:'string'}},
      }},
    overall:{enum:['CLEAR-pending-probe','FINDING','NEEDS-PROBE']},
    probe_plan:{type:'array', items:{type:'string'}},
  }
}

const REFUTE_SCHEMA = {
  type:'object', additionalProperties:false,
  required:['agent','attacked','refuted','evidence','revised'],
  properties:{
    agent:{type:'string'},
    attacked:{type:'array', items:{type:'string'}},
    refuted:{type:'boolean'},
    evidence:{type:'array', items:{type:'string'}},
    revised:{type:'array', items:{type:'string'}},
  }
}

const cardPrompt = (a) => `READ-ONLY static audit of the '${a.name}' agent at ${a.dir} (repo root /home/enunez/Projects/e2i_causal_analytics).

HARD CONSTRAINT: You MUST NOT execute Python, import agent code, run the agent, or load any parquet/DB. Use ONLY Read + Grep + Bash for grep/git-log/sed. No edits, no execution of project code. Static analysis genuinely does not need to run anything — running heavy imports would OOM a memory-pressured box.

Produce a 4-dimension audit card (return via the StructuredOutput tool):

D1 FUNCTIONAL INTEGRITY (no silent mocks):
- Find the agent entry (agent.py / graph.py) and the TERMINAL node(s) that build the agent's final output. Give file:line for terminal_node. READ those nodes fully — do not skim.
- Grep the agent dir for mock signals: random.uniform, np.random, hardcoded ate=/confidence=/p_value=/0.12/0.85, "# Placeholder|Mock|Stub|TODO: real", "actual X would go here", structured returns with all-default/all-zero fields.
- Judge: are the returned values COMPUTED from inputs, or CONSTANT/hardcoded? (computed_or_constant)
- REASON-BEFORE-RULES: if any mock/placeholder exists, investigate INTENT (run: git log --diff-filter=A --follow <file> | head; read PR/issue refs in comments; read docstrings) and HARM (prod-reachable? plausible-wrong values? user-visible?). A scaffolded placeholder for requested functionality is NOT a defect — record that in intent_note. Do NOT classify on pattern-match alone.
- claims[]: list anything you CANNOT settle statically (drives Phase 2 probes).

D2 WIRING & REACHABILITY:
- Is '${a.name}' in src/agents/factory.py AGENT_REGISTRY_CONFIG? (shared snapshot at /tmp/audit_registry.txt). Set in_registry.
- How is it dispatched/routed? (shared snapshot /tmp/audit_dispatch.txt; full file src/agents/orchestrator/nodes/dispatcher.py). Watch for route-ORDER shadows and degraded-only reachability (e.g. only reachable when registry is partial). Set dispatch_route.
- Who consumes its output? grep src/ broadly — do NOT head-truncate the grep (a prior audit missed a consumer that way). Set consumers[].
- verdict: WIRED / UNWIRED / SHADOWED / DEGRADED-ONLY.

D3 DATA-SUBSTRATE (static only):
- What table(s)/column(s)/grain does it read? From its repository/query code. If no data dependency, verdict NO-DATA-DEP.
- degenerate_risk: note any outcome that may be ~all-positive / leaky / unpopulated, to SQL-probe later.
- claims[]: the exact SQL checks Phase 2 should run (COUNT(*), positive-rate, null-rate, distinct enum).

D4 OOM:
- Grep for n_jobs=-1 / loky / joblib.Parallel without a cap, full-width parquet reads (pd.read_parquet without columns=), unbounded frame loads. Give file:line in signals[]. verdict: SAFE / SPIKE-RISK / UNBOUNDED.

overall: FINDING if any dimension is already a clear defect; NEEDS-PROBE if claims remain; CLEAR-pending-probe if static is clean AND no claims remain.
probe_plan[]: concrete probes Phase 2 must run (empty if none).`

const refutePrompt = (a, card) => `You are an adversarial skeptic auditing the '${a.name}' agent at ${a.dir}. READ-ONLY: no execution, no edits, grep/read/git-log only.

The first static pass produced these verdicts:
${JSON.stringify({d1:card.d1.verdict, computed:card.d1.computed_or_constant, terminal:card.d1.terminal_node, d2:card.d2.verdict, d3:card.d3.verdict, d4:card.d4.verdict}, null, 2)}

Try HARD to REFUTE them with file:line evidence. Specifically hunt:
- a silent mock / constant return the first pass mistook for "computed" — re-read terminal_node AND every function it calls; follow the data back to its source.
- a registry/route that looks WIRED but is actually shadowed or only reachable in a degraded/partial registry.
- a data dependency marked DECLARED that is actually unpopulated/degenerate.
- an OOM spike marked SAFE that the first pass missed.
Set refuted=true ONLY if you found a real defect the card missed (with evidence). attacked[]=which verdicts you challenged. revised[]=corrected verdicts. Be a genuine skeptic; do not rubber-stamp.`

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
