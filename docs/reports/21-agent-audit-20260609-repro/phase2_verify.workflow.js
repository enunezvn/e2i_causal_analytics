export const meta = {
  name: '21-agent-finding-verification',
  description: 'Independently verify each Phase-1 candidate finding against cited source (CONFIRMED/REFUTED/PARTIAL + severity + fix)',
  phases: [{ title: 'Verify', detail: 'one skeptical source-verifier per finding' }],
}

// Cited line numbers are from the branch feat/dspy-loop-real-results.
// For findings with read_on_main=true, the file differs on main → read `git show main:<path>` (line numbers may differ; find by content).
const FINDINGS = [
  {id:'health_score-composer', agent:'health_score', read_on_main:false,
   claim:"Composer emits an invariant 100.0/grade-A health score because all four component inputs are mock-pinned to 1.0 (never real).",
   locations:["src/agents/health_score/nodes/component_health.py:83 (_create_mock_status)","src/agents/health_score/nodes/model_health.py:94","src/agents/health_score/nodes/pipeline_health.py:93","src/agents/health_score/nodes/agent_health.py:98","src/agents/health_score/nodes/score_composer.py:67"]},
  {id:'health_score-rest', agent:'health_score', read_on_main:false,
   claim:"REST endpoints /components,/models,/pipelines,/agents serve hardcoded mock health tagged data_provenance='measured' (user-facing fake).",
   locations:["src/api/routes/health_score.py:480","src/api/routes/health_score.py:521","src/api/routes/health_score.py:898","src/api/routes/health_score.py:940"]},
  {id:'gap_analyzer-launder', agent:'gap_analyzer', read_on_main:false,
   claim:"Real path raises KeyError('region') (BenchmarkStore lacks the segment column), is swallowed by a broad except, the formatter upgrades status 'failed'->'completed' and emits 'No significant performance gaps' as HTTP 200.",
   locations:["src/agents/gap_analyzer/connectors/benchmark_store.py:187-229","src/agents/gap_analyzer/nodes/gap_detector.py:558-562","src/agents/gap_analyzer/nodes/gap_detector.py:222","src/agents/gap_analyzer/nodes/formatter.py:98","src/agents/gap_analyzer/nodes/formatter.py:260-264","src/api/routes/gaps.py:664-666"]},
  {id:'observability_connector-mock', agent:'observability_connector', read_on_main:false,
   claim:"_get_span_repository() always TypeErrors on a bad client= kwarg (swallowed), so aggregate_metrics()/get_quality_metrics() always return fabricated _get_mock_spans metrics, unmarked.",
   locations:["src/agents/ml_foundation/observability_connector/ (aggregate_metrics, _get_span_repository, _get_mock_spans — find by content)"]},
  {id:'model_deployer-simulated', agent:'model_deployer', read_on_main:false,
   claim:"register_model/promote_stage emit registration_successful/promotion_successful=True with hardcoded model_version=1 when MLflow registration fails (default model_uri='simulated://model' guarantees failure); check_rollback_availability fabricates a random-uuid previous_deployment_id; _store_to_database short-circuits so NO ml_deployments/ml_model_registry rows are written despite status='completed'.",
   locations:["src/agents/ml_foundation/model_deployer/agent.py:375","src/agents/ml_foundation/model_deployer/ (register_model, promote_stage, check_rollback_availability nodes)"]},
  {id:'model_selector-frozen', agent:'model_selector', read_on_main:false,
   claim:"MLDataLoader.execute_query does not exist anywhere in src/, so the historical-performance node always raises AttributeError (silenced by type:ignore + bare except) and falls back to a hardcoded constant table; ~40% of the selection score is a frozen prior, and historical_data_available/historical_experiments_count are hardwired False/0.",
   locations:["src/agents/ml_foundation/model_selector/ (the node calling MLDataLoader.execute_query + the constant fallback table — find by content)"]},
  {id:'experiment_designer-mock', agent:'experiment_designer', read_on_main:false,
   claim:"context_loader.MockKnowledgeStore is an UNMARKED, non-flag-gated mock reachable on every normal prod run (the real store lacks the methods), silently seeding the design LLM prompt with fabricated organizational context.",
   locations:["src/agents/experiment_designer/ (context_loader.py MockKnowledgeStore, organizational_defaults/domain_knowledge — find by content)"]},
  {id:'drift_monitor-structural', agent:'drift_monitor', read_on_main:false,
   claim:"The structural_drift (causal-DAG) node is orphaned at both boundaries: DriftMonitorInput omits the DAG fields and AlertAggregatorNode drops state['structural_drift_results'], so overall_drift_score/drift_summary exclude the structural dimension and a critical DAG drift can surface as drift_score=0.0 / 'NO SIGNIFICANT DRIFT'.",
   locations:["src/agents/drift_monitor/agent.py:54-97 (DriftMonitorInput)","src/agents/drift_monitor/graph.py:64","src/agents/drift_monitor/graph.py:73-74","src/agents/drift_monitor/nodes/alert_aggregator.py:142-165"]},
  {id:'orchestrator-mock', agent:'orchestrator', read_on_main:true,
   claim:"On MAIN, the dispatcher reaches _mock_agent_execution UNCONDITIONALLY when a routed agent is absent from the registry (no allow_mock guard merged), fabricating ATE=0.12/$2.5M as success=True; a partial registry is prod-reachable via create_agent_registry(fail_on_import_error=False) silently dropping un-instantiable agents.",
   locations:["main: src/agents/orchestrator/nodes/dispatcher.py (around the _mock_agent_execution call ~line 411 and def ~line 529)","src/agents/factory.py (create_agent_registry, fail_on_import_error)"]},
  {id:'causal_impact-dispatch', agent:'causal_impact', read_on_main:false,
   claim:"Via the orchestrator route the agent fails input validation (ValueError, success=False) because required causal inputs are never supplied; _get_data fail-closes without estimation_data or data_source=='synthetic'; only direct/harness invocation computes. (Degraded-registry path serves the ATE=0.12 mock — that part is the orchestrator dispatcher on MAIN.)",
   locations:["src/agents/causal_impact/ (_validate_input / required inputs)","src/agents/causal_impact/nodes/estimation.py:703-714 (_get_data fail-close)","src/agents/causal_impact/router.py:40-48","main: src/agents/orchestrator/nodes/dispatcher.py (input prep for causal_impact)"]},
  {id:'heterogeneous_optimizer-dispatch', agent:'heterogeneous_optimizer', read_on_main:false,
   claim:"The orchestrator dispatch path supplies none of the 6 required input fields (treatment_var/outcome_var/segment_vars/effect_modifiers/data_source...), so every orchestrator-routed call fails closed at _validate_input with ValueError and never runs CausalForestDML; only the /segments REST route and the tier0 harness are functionally wired.",
   locations:["src/agents/heterogeneous_optimizer/ (_validate_input, required fields)","src/agents/factory.py:102","main: src/agents/orchestrator/nodes/dispatcher.py (+ _agent_method_map for heterogeneous_optimizer)"]},
  {id:'prediction_synthesizer-dispatch', agent:'prediction_synthesizer', read_on_main:false,
   claim:"synthesize() requires entity_id/prediction_target which _prepare_agent_input never supplies and rejects the splatted dispatch kwargs (no input_model coercion), so the orchestrator route raises TypeError on every call; the ensemble math is real but unreachable via the only wired prod route.",
   locations:["src/agents/prediction_synthesizer/ (synthesize() signature, ensemble_combiner)","main: src/agents/orchestrator/nodes/dispatcher.py (_prepare_agent_input for prediction_synthesizer)"]},
  {id:'resource_optimizer-dispatch', agent:'resource_optimizer', read_on_main:false,
   claim:"optimize() requires positional allocation_targets/constraints and takes no **kwargs, so the orchestrator dispatcher's optimize(**agent_input) raises TypeError (success=False, terminal node never reached); it is functionally wired ONLY via the REST route which populates allocation_targets from the HTTP body.",
   locations:["src/agents/resource_optimizer/agent.py:147 (optimize signature)","main: src/agents/orchestrator/nodes/dispatcher.py:438 (optimize(**agent_input) call)","src/agents/orchestrator/nodes/router.py:85-93","src/api/routes/resource_optimizer.py:557","src/api/routes/resource_optimizer.py:600"]},
  {id:'experiment_monitor-typeerror', agent:'experiment_monitor', read_on_main:false,
   claim:"Every DB-touching node fails at `await get_supabase_client()` with TypeError (awaiting a sync client), so the terminal alert_generator always computes over EMPTY state -> a degenerate 'Experiments checked: 0' summary with no alerts. Confirm get_supabase_client is synchronous (not awaitable).",
   locations:["src/agents/experiment_monitor/nodes/health_checker.py:45","src/agents/experiment_monitor/nodes/srm_detector.py:49","src/agents/experiment_monitor/nodes/interim_analyzer.py:47","src/agents/experiment_monitor/nodes/fidelity_checker.py:41","src/ (definition of get_supabase_client — is it async?)"]},
  {id:'feedback_learner-starved', agent:'feedback_learner', read_on_main:false,
   claim:"feedback_store is unpopulated in all prod construction sites and knowledge_stores={} pins update_effectiveness to 0.0 (dead reward term); the beat path omits focus_agents so even the implicit-feedback fallback is empty -> feedback_count/patterns/recs structurally 0.",
   locations:["src/tasks/dspy_optimization_tasks.py:170","src/api/routes/feedback.py:1347","src/agents/factory.py:261","src/agents/feedback_learner/ (knowledge_stores -> update_effectiveness reward term)"]},
  {id:'data_preparer-append', agent:'data_preparer', read_on_main:false,
   claim:"On the reachable Supabase-table + entity_column path, load_data crashes because pandas 2.x removed DataFrame.append (data_loader.py:196), so the terminal node never computes for entity-split Supabase cohorts.",
   locations:["src/agents/ml_foundation/data_preparer/nodes/data_loader.py:196"]},
  {id:'cohort_constructor-import', agent:'cohort_constructor', read_on_main:false,
   claim:"tier0_integration.py:24 `from cohort_constructor import ...` targets a non-existent TOP-LEVEL package and raises ModuleNotFoundError; this is reachable from the user-suppliable data_source field via cohort_resolution._resolve_via_data_source. Separately, _track_patient_assignments/_determine_failed_criteria use iterrows + per-row pd.Series at 100K-patient scale (OOM spike).",
   locations:["src/agents/cohort_constructor/tier0_integration.py:24","src/agents/cohort_constructor/ (cohort_resolution._resolve_via_data_source; _track_patient_assignments iterrows)"]},
  {id:'tool_composer-launder', agent:'tool_composer', read_on_main:true,
   claim:"On MAIN, the agent emits success=True even when 0/N tools succeeded and a confidence~0.8 from an LLM synthesis prompt with no anti-fabrication guard, so a confident terminal answer can be returned with no successful computation behind it. NOTE: main has #813/#810 commits ('honest tools — real compute or fail-close', F3/F4/F7) — verify whether those fixes already close this on main.",
   locations:["main: src/agents/tool_composer/composer.py (success aggregation ~331-349)","main: src/agents/tool_composer/models/composition_models.py (~250-252)","main: src/agents/tool_composer/synthesizer.py (prompt ~30-62, confidence ~126)","main: src/agents/tool_composer/agent.py (terminal output ~406-418)"]},
  {id:'feature_analyzer-degrade', agent:'feature_analyzer', read_on_main:false,
   claim:"When X_sample is absent, the SHAP background silently degrades to an np.random synthetic generator (_generate_domain_aware_background), and the full-pipeline graph never bridges select_features' X_train_selected into SHAP; additionally the V4.4 causal branch (rank_causal_drivers) is DEAD (0 graph consumers, has_causal_results always False). Assess prod reachability of the synthetic-background fallback.",
   locations:["src/agents/ml_foundation/feature_analyzer/nodes/shap_computer.py:328-335","src/agents/ml_foundation/feature_analyzer/ (rank_causal_drivers consumers; X_train_selected bridge)"]},
  {id:'model_trainer-oom', agent:'model_trainer', read_on_main:false,
   claim:"Default binary-classification path runs unconditional 5-fold CV (full-data clone+refit over pd.concat of train+val+test) + 200-shuffle permutation test + 1000-sample bootstrap with NO in-agent loky/OpenMP cap (n_jobs=1 bounds only joblib, not libgomp/loky) -> the measured 5.9GB spike, mitigated only by an EXTERNAL LOKY_MAX_CPU_COUNT=1. Confirm there is no in-agent thread cap and the CV/permutation/bootstrap are on the default path.",
   locations:["src/agents/ml_foundation/model_trainer/ (evaluator.py ~947-977: cross_val, permutation, bootstrap; grep for n_jobs / LOKY / threadpool cap)"]},
]

const VERDICT_SCHEMA = {
  type:'object', additionalProperties:false,
  required:['finding_id','agent','verdict','evidence_quotes','reachability','severity','recommended_fix','intent_note'],
  properties:{
    finding_id:{type:'string'},
    agent:{type:'string'},
    verdict:{enum:['CONFIRMED','REFUTED','PARTIAL']},
    evidence_quotes:{type:'array', items:{type:'string'}}, // "file:line — <verbatim quoted code>"
    reachability:{type:'object', additionalProperties:false,
      required:['prod_reachable','user_facing','fail_closed','path'],
      properties:{
        prod_reachable:{type:'boolean'},
        user_facing:{type:'boolean'},
        fail_closed:{type:'boolean'},  // true = fails safe (no fabrication); false = fabricates/launders plausible-wrong values
        path:{type:'string'},          // the exact reachable path (or why unreachable)
      }},
    severity:{enum:['CRITICAL','HIGH','MEDIUM','LOW','NONE']},
    recommended_fix:{type:'string'},
    intent_note:{type:'string'},       // REASON-BEFORE-RULES: intentional scaffold? git/PR/docstring evidence; HARMFUL-NOW vs KEEP-PLACEHOLDER
  }
}

const verifyPrompt = (f) => `You are an independent, SKEPTICAL source-verifier (repo root /home/enunez/Projects/e2i_causal_analytics). READ-ONLY: use Read, Grep, and Bash only for grep / sed / git show / git log. Do NOT execute Python, import agent code, run the agent, or load parquet/DB.

A Phase-1 adversarial screen produced this CANDIDATE finding for the '${f.agent}' agent (finding_id=${f.id}):
"${f.claim}"

Cited locations (line numbers are from the branch; on main they may shift — find by content):
${f.locations.map(l => '  - '+l).join('\n')}

${f.read_on_main
  ? "IMPORTANT: This finding concerns code that DIFFERS on production-bound `main`. Read the MAIN version: `git show main:<path>` (NOT the working tree). Verify the claim against MAIN."
  : "Read the working-tree version (this file is identical on HEAD and main)."}

Your job is to CONFIRM or REFUTE the claim against the actual source — not to rubber-stamp it. Read the cited code verbatim and quote it. The screen may be wrong, overstated, or right.

Decide:
- verdict: CONFIRMED (claim holds as stated), PARTIAL (core holds but scope/reachability is narrower/wider than claimed), or REFUTED (claim does not hold against source).
- evidence_quotes[]: 1-5 entries, each "file:line — <verbatim code>" that proves your verdict. No paraphrase — quote the real lines.
- reachability: is the defective path prod_reachable? user_facing (chat/report/dashboard)? does it fail_closed (true=fails safe/honest; false=fabricates or launders plausible-wrong values)? path = the exact reachable call path, or why it cannot be reached.
- severity: CRITICAL (user-facing plausible-wrong values in prod) / HIGH (prod-reachable fabrication or laundered failure, not yet user-facing) / MEDIUM (degraded/unreachable feature, fails closed) / LOW (cosmetic/dead-code/conditional) / NONE (refuted).
- recommended_fix: one concrete sentence (the actual code change), no placeholders.
- intent_note: REASON-BEFORE-RULES. Investigate INTENT (git log --diff-filter=A --follow on the file, PR/issue refs in comments, docstrings) and HARM. Is this an INTENTIONAL test scaffold / flag-gated dev path / documented placeholder (→ KEEP, lower severity), or genuinely harmful-now? State which and why. Do not classify a legitimate scaffold as a defect.`

phase('Verify')
const verdicts = await parallel(FINDINGS.map(f => () =>
  agent(verifyPrompt(f), {label:`verify:${f.id}`, phase:'Verify', schema:VERDICT_SCHEMA})
))
return verdicts.filter(Boolean)
