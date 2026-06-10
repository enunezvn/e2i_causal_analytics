export const meta = {
  name: '21-agent-reverify-findings',
  description: 'Re-verify each audit finding against CURRENT main HEAD (read-only) to confirm it is still live before remediation',
  phases: [
    { title: 'Reverify', detail: 'one read-only agent per finding confirms still-live vs resolved on current main' },
  ],
}

// Findings from docs/reports/21-agent-audit-20260609.md §4, F5 already resolved by #814 (excluded).
const FINDINGS = [
  {id:'F1', sev:'CRITICAL', agent:'health_score',
   claim:"Route routes/health_score.py:~789 constructs HealthScoreAgent() with all stores None, so each node defaults to 1.0 -> composer yields invariant 100.0/grade-A tagged data_provenance='measured'. A real SupabaseHealthClient (health_client.py) is never injected. F1b: REST endpoints /components,/models,/pipelines,/agents return hardcoded _get_mock_*_health (e.g. accuracy=0.89, auc_roc=0.92) while _resolve_health_provenance tags MEASURED merely because the agent imports.",
   anchors:['src/agents/health_score/','src/api/routes/health_score.py']},
  {id:'F2', sev:'HIGH', agent:'gap_analyzer',
   claim:"Real path crashes (segment-column mismatch over 4667 real business_metrics rows), broad except swallows it, empty-branches (roi_calculator.py:~106, prioritizer.py:~65) + formatter overwrite status failed->completed, and gaps.py:~663-666 maps to HTTP 200 'No significant performance gaps'. Should preserve terminal-failure status and set AnalysisStatus.FAILED when errors non-empty.",
   anchors:['src/agents/gap_analyzer/','src/api/routes/gaps.py']},
  {id:'F3', sev:'HIGH', agent:'observability_connector',
   claim:"Repo built with wrong client= kwarg -> TypeError -> swallowed -> _get_mock_spans path taken 100% of time even though ml_observability_spans holds 5313 real rows. Fix is passing supabase_client= at metrics_aggregator.py:~30 + agent.py:~91.",
   anchors:['src/agents/ml_foundation/observability_connector/']},
  {id:'F4', sev:'HIGH', agent:'model_deployer',
   claim:"Returns registration_successful/promotion_successful=True with model_version=1 when MLflow fails (default model_uri='simulated://model' guarantees failure), fabricates random-uuid rollback id, and _store_to_database short-circuits -> ml_deployments/ml_model_registry stay 0 rows despite status='completed'. Should set *_successful=False on simulated/failed registration.",
   anchors:['src/agents/ml_foundation/model_deployer/']},
  {id:'F6', sev:'HIGH', agent:'tool_composer',
   claim:"Despite #813 'honest tools', the BUILD-RESULT block still maps to success=True and synthesizes a ~0.8-confidence answer with no anti-fabrication guard when all tool outputs FAILED. Should gate on tools_succeeded==0 -> status=FAILED, success=False + no-fabrication synthesis instruction. NOTE: tool_composer was changed by recent commits (#810 dec84d35); verify current state.",
   anchors:['src/agents/tool_composer/']},
  {id:'F7', sev:'HIGH', agent:'experiment_monitor',
   claim:"Every DB node does `await get_supabase_client()` but that factory is SYNCHRONOUS -> TypeError on every call -> terminal computes over empty state ('0 experiments checked' while 621 run). An async get_async_supabase_client() exists. NOTE: experiment_monitor nodes (health_checker.py, alert_generator.py) were changed by recent commits 69b51f7b/09609ac2; verify current state of the await pattern.",
   anchors:['src/agents/experiment_monitor/']},
  {id:'F8', sev:'HIGH', agent:'feature_analyzer',
   claim:"When X_sample absent, SHAP silently uses np.random synthetic background (comment 'In production this would come from Feast'), and the full graph never bridges X_train_selected into SHAP -> importances can be synthetic and unlabeled. Should fail-closed skip when no real sample; bridge X_train_selected; gate synthetic behind flag stamping data_provenance='synthetic'.",
   anchors:['src/agents/ml_foundation/feature_analyzer/']},
  {id:'F9', sev:'MEDIUM', agent:'model_selector',
   claim:"MLDataLoader.execute_query does not exist -> AttributeError silenced (type:ignore + bare except) -> ~40% of selection score is a frozen constant; author intended real history (commit 209f9cff).",
   anchors:['src/agents/ml_foundation/model_selector/']},
  {id:'F10', sev:'MEDIUM', agent:'experiment_designer',
   claim:"MockKnowledgeStore (unmarked, non-flag-gated) seeds the design LLM prompt with fabricated org context on every prod run; a scaffold its own docstring says to replace.",
   anchors:['src/agents/experiment_designer/']},
  {id:'F11', sev:'MEDIUM', agent:'drift_monitor',
   claim:"The V4.4 structural_drift node writes results but the input plumbing + alert_aggregator drop them, so DAG drift can read 0.0/'NO DRIFT'. Fails closed (understated). Should thread DAG fields into input + fold structural_drift_results into aggregator/summary.",
   anchors:['src/agents/drift_monitor/']},
  {id:'F12', sev:'MEDIUM', agent:'heterogeneous_optimizer',
   claim:"Dispatcher calls the agent's real entry-point but supplies no inputs (no input_model bridge) -> ValueError/TypeError -> fails closed; feature dead via chat (works via REST/tier-0). Same root as #260 _coerce_to_input_model pattern, not applied here.",
   anchors:['src/agents/heterogeneous_optimizer/','src/agents/orchestrator/nodes/dispatcher.py']},
  {id:'F13', sev:'MEDIUM', agent:'resource_optimizer',
   claim:"Same dispatch-reachability defect: no input_model bridge -> fails closed via chat. NOTE: resource_optimizer files changed by recent commits db75cb33/9cd19d73 (dspy recipient work); verify the dispatch bridge is still absent.",
   anchors:['src/agents/resource_optimizer/','src/agents/orchestrator/nodes/dispatcher.py']},
  {id:'F14', sev:'MEDIUM', agent:'prediction_synthesizer',
   claim:"Same dispatch-reachability defect: no input_model bridge -> fails closed via chat.",
   anchors:['src/agents/prediction_synthesizer/','src/agents/orchestrator/nodes/dispatcher.py']},
  {id:'F15', sev:'MEDIUM', agent:'feedback_learner',
   claim:"Constructed with empty feedback_store/knowledge_stores={} in all prod sites -> update_effectiveness pinned 0.0; documented starved state. NOTE: feedback_learner was heavily changed by recent #811 commits (recipient substrate B0-B5); verify whether prod construction still passes empty stores.",
   anchors:['src/agents/feedback_learner/']},
  {id:'F16', sev:'MEDIUM', agent:'data_preparer',
   claim:"data_loader.py:~196 uses DataFrame.append (removed in pandas 2.x; repo pins 2.3.3) -> crash on the Supabase entity-split path. Fails closed. Fix: pd.concat([...], ignore_index=True).",
   anchors:['src/agents/ml_foundation/data_preparer/']},
  {id:'F17', sev:'MEDIUM', agent:'cohort_constructor',
   claim:"tier0_integration.py:~24 (and ~471) `from cohort_constructor import ...` targets a non-existent top-level package -> ModuleNotFoundError on the data_source branch (reorg artifact). Fix: from src.agents.cohort_constructor import ...",
   anchors:['src/agents/cohort_constructor/']},
  {id:'F18', sev:'LOW', agent:'causal_impact',
   claim:"Orchestrator route fails closed (ValueError) because causal inputs aren't supplied at agent.py:~177-181 -- correct, safe behavior; only a feature-completeness gap. NOTE: causal_impact/nodes/interpretation.py changed recently. Verify it remains fail-closed-correct (no fabrication).",
   anchors:['src/agents/causal_impact/']},
  {id:'F19', sev:'LOW', agent:'model_trainer',
   claim:"Intentional 5-fold CV + 200-perm + 1000-bootstrap with no in-agent thread cap -> observed 5.9 GB spike, mitigated only by external LOKY_MAX_CPU_COUNT=1. No user-facing harm. Fix: wrap per-fold refit in threadpoolctl.threadpool_limits(1) + clone with n_jobs=1.",
   anchors:['src/agents/ml_foundation/model_trainer/']},
]

const SCHEMA = {
  type:'object', additionalProperties:false,
  required:['id','agent','still_live','status','evidence','recent_commit_impact','fix_locations','notes'],
  properties:{
    id:{type:'string'},
    agent:{type:'string'},
    still_live:{type:'boolean'},                 // true = the defect is still present on current main HEAD
    status:{enum:['LIVE-CONFIRMED','RESOLVED','PARTIALLY-RESOLVED','REFRAMED','NOT-FOUND']},
    evidence:{type:'array', items:{type:'string'}}, // file:line excerpts proving current state
    recent_commit_impact:{type:'string'},        // did a recent commit change the cited code? what did it do?
    fix_locations:{type:'array', items:{type:'string'}}, // exact file:line(s) a fix must touch (current line numbers)
    notes:{type:'string'},
  }
}

const prompt = (f) => `READ-ONLY re-verification of audit finding ${f.id} (${f.sev}) on the '${f.agent}' agent, against the CURRENT working tree (git branch main, HEAD).
You MUST NOT execute Python, import agent code, or load parquet/DB. Use Read + Grep + Bash(git log/git show) only.

The audit (run on a stale branch) claimed:
"${f.claim}"

Relevant locations to inspect (line numbers may have drifted — find the CURRENT lines): ${JSON.stringify(f.anchors)}

Your job: determine whether this defect is STILL PRESENT on current main HEAD.
1. Read the actual current source at the cited locations. Find the terminal/output path and the exact lines implicated.
2. Run \`git log --oneline -5 -- <relevant_file>\` to see if a recent commit touched it; if so, \`git show\` the relevant hunk to judge whether the defect was fixed, partially fixed, or untouched.
3. Decide still_live (true if the defect is reproducible on current main) and status.
   - LIVE-CONFIRMED: defect present, unchanged.
   - PARTIALLY-RESOLVED: a recent commit addressed part but the core defect remains.
   - RESOLVED: the defect is gone on current main (e.g., a later PR fixed it). Prove it with the current code.
   - REFRAMED: the defect exists but the audit's description/line is wrong; give the corrected description.
   - NOT-FOUND: the cited code does not exist (audit error).
4. fix_locations: the EXACT current file:line(s) a remediation must touch (so the implementer doesn't re-derive them).

Be skeptical and precise. Cite file:line for every claim. Do not pattern-match the audit text — read the real current code.`

phase('Reverify')
const results = await parallel(FINDINGS.map(f => () =>
  agent(prompt(f), {label:`reverify:${f.id}-${f.agent}`, phase:'Reverify', schema:SCHEMA, agentType:'Explore'})
))
return results.filter(Boolean)
