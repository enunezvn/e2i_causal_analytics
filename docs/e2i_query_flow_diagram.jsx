import React, { useState } from 'react';

const E2IQueryFlowDiagram = () => {
  const [activeStep, setActiveStep] = useState(null);
  const [activeLayer, setActiveLayer] = useState(null);

  const layers = [
    {
      id: 'layer1',
      name: 'Layer 1: Conversational Interface',
      color: '#8b5cf6',
      components: [
        { id: 'query_input', name: 'User Query', type: 'input', desc: 'Natural language question from dashboard chat' },
        { id: 'query_processor', name: 'Query Processor', type: 'process', desc: 'Main NL query processing pipeline' },
        { id: 'intent_classifier', name: 'Intent Classifier', type: 'process', desc: '5 intent types → 18 agent routing' },
        { id: 'entity_extractor', name: 'Entity Extractor', type: 'process', desc: 'Domain-only extraction (NO medical NER)' },
        { id: 'parsed_query', name: 'ParsedQuery', type: 'output', desc: 'Intent + Entities + Rewritten Query' },
      ]
    },
    {
      id: 'tier1',
      name: 'Tier 1: Orchestrator',
      color: '#a855f7',
      components: [
        { id: 'router', name: 'Dynamic Router', type: 'process', desc: 'Maps intent to agent(s) by tier priority' },
        { id: 'planner', name: 'Multi-Step Planner', type: 'process', desc: 'Plans execution sequence for complex queries' },
        { id: 'agent_dispatch', name: 'Agent Dispatch', type: 'output', desc: 'Dispatches to Tier 0-5 agents' },
      ]
    },
    {
      id: 'tier0',
      name: 'Tier 0: ML Foundation (7 Agents)',
      color: '#ec4899',
      isNew: true,
      components: [
        { id: 'scope_definer', name: 'Scope Definer', type: 'agent', desc: 'Problem scope + success criteria' },
        { id: 'data_preparer', name: 'Data Preparer', type: 'agent', desc: 'QC gating (Great Expectations)', hasGate: true },
        { id: 'model_selector', name: 'Model Selector', type: 'agent', desc: 'Algorithm registry + constraints' },
        { id: 'model_trainer', name: 'Model Trainer', type: 'agent', desc: 'Split enforcement + Optuna tuning' },
        { id: 'feature_analyzer', name: 'Feature Analyzer', type: 'agent', desc: 'SHAP + LLM interpretation', isHybrid: true },
        { id: 'model_deployer', name: 'Model Deployer', type: 'agent', desc: 'BentoML serving' },
        { id: 'observability', name: 'Observability Connector', type: 'agent', desc: 'Opik spans + telemetry' },
      ]
    },
    {
      id: 'tier2',
      name: 'Tier 2: Causal Analytics (Core E2I)',
      color: '#3b82f6',
      components: [
        { id: 'causal_impact', name: 'Causal Impact Agent', type: 'agent', desc: '5-node: Graph→Estimate→Refute→Sensitivity→Interpret', isHybrid: true },
        { id: 'gap_analyzer', name: 'Gap Analyzer', type: 'agent', desc: 'ROI opportunity detection' },
        { id: 'heterogeneous', name: 'Heterogeneous Optimizer', type: 'agent', desc: 'CATE by segment' },
      ]
    },
    {
      id: 'layer2',
      name: 'Layer 2: Causal Engine & RAG',
      color: '#06b6d4',
      components: [
        { id: 'dag_builder', name: 'DAG Builder', type: 'process', desc: 'NetworkX graph construction' },
        { id: 'effect_estimator', name: 'Effect Estimator', type: 'process', desc: 'DoWhy/EconML ATE/CATE' },
        { id: 'refutation', name: 'Refutation Runner', type: 'process', desc: '5 tests → causal_validations table', hasGate: true },
        { id: 'causal_rag', name: 'CausalRAG', type: 'process', desc: 'Hybrid: dense + sparse + graph retrieval' },
      ]
    },
    {
      id: 'tier3-5',
      name: 'Tiers 3-5: Monitoring & Learning',
      color: '#10b981',
      components: [
        { id: 'drift_monitor', name: 'Drift Monitor', type: 'agent', desc: 'PSI + degradation detection' },
        { id: 'experiment_designer', name: 'Experiment Designer', type: 'agent', desc: 'A/B test power analysis', isHybrid: true },
        { id: 'health_score', name: 'Health Score', type: 'agent', desc: 'Pareto composite metrics' },
        { id: 'prediction_synth', name: 'Prediction Synthesizer', type: 'agent', desc: 'Ensemble ML predictions' },
        { id: 'explainer', name: 'Explainer', type: 'agent', desc: 'NL narrative generation' },
        { id: 'feedback_learner', name: 'Feedback Learner', type: 'agent', desc: 'Self-improvement loop' },
      ]
    },
    {
      id: 'synthesis',
      name: 'Response Synthesis',
      color: '#f59e0b',
      components: [
        { id: 'synthesizer', name: 'Response Synthesizer', type: 'process', desc: 'Merges multi-agent outputs' },
        { id: 'verification', name: 'Verification Node', type: 'process', desc: 'Confidence + compliance + hallucination check' },
        { id: 'viz_selector', name: 'Chart Selector', type: 'process', desc: 'Rules-based visualization mapping' },
      ]
    },
    {
      id: 'dashboard',
      name: 'Dashboard End States',
      color: '#ef4444',
      components: [
        { id: 'chat_response', name: 'Chat Response', type: 'output', desc: 'Streaming text + agent badges' },
        { id: 'causal_dag', name: 'Causal DAG Viz', type: 'output', desc: 'D3.js interactive graph' },
        { id: 'kpi_cards', name: '46 KPI Cards', type: 'output', desc: 'Metrics with causal insights' },
        { id: 'heatmaps', name: 'CATE Heatmaps', type: 'output', desc: 'Heterogeneous effects by segment' },
        { id: 'sankey', name: 'Resource Sankey', type: 'output', desc: 'Budget allocation flows' },
        { id: 'validation_badge', name: 'Validation Badge', type: 'output', desc: 'Refutation status (V4.1)' },
      ]
    }
  ];

  const dataFlows = [
    { from: 'query_input', to: 'query_processor', data: 'Raw NL string' },
    { from: 'query_processor', to: 'intent_classifier', data: 'Cleaned query' },
    { from: 'query_processor', to: 'entity_extractor', data: 'Cleaned query' },
    { from: 'intent_classifier', to: 'parsed_query', data: 'IntentType enum' },
    { from: 'entity_extractor', to: 'parsed_query', data: 'ExtractedEntities' },
    { from: 'parsed_query', to: 'router', data: 'ParsedQuery object' },
    { from: 'router', to: 'planner', data: 'Agent list + priority' },
    { from: 'planner', to: 'agent_dispatch', data: 'Execution plan' },
    { from: 'agent_dispatch', to: 'scope_definer', data: 'ML_SCOPE intent' },
    { from: 'agent_dispatch', to: 'causal_impact', data: 'CAUSAL intent' },
    { from: 'agent_dispatch', to: 'gap_analyzer', data: 'GAP intent' },
    { from: 'agent_dispatch', to: 'drift_monitor', data: 'DRIFT intent' },
    { from: 'data_preparer', to: 'model_trainer', data: 'QC pass/block gate' },
    { from: 'causal_impact', to: 'dag_builder', data: 'Variables + constraints' },
    { from: 'dag_builder', to: 'effect_estimator', data: 'NetworkX DAG' },
    { from: 'effect_estimator', to: 'refutation', data: 'Effect estimate' },
    { from: 'refutation', to: 'causal_impact', data: 'ValidationGate decision' },
    { from: 'causal_rag', to: 'causal_impact', data: 'Retrieved context' },
    { from: 'feature_analyzer', to: 'synthesizer', data: 'SHAP values + interpretation' },
    { from: 'causal_impact', to: 'synthesizer', data: 'CausalResult' },
    { from: 'gap_analyzer', to: 'synthesizer', data: 'GapAnalysis' },
    { from: 'health_score', to: 'synthesizer', data: 'HealthMetrics' },
    { from: 'synthesizer', to: 'verification', data: 'Merged response' },
    { from: 'verification', to: 'viz_selector', data: 'Verified response' },
    { from: 'viz_selector', to: 'chat_response', data: 'Text content' },
    { from: 'viz_selector', to: 'causal_dag', data: 'DAG spec' },
    { from: 'viz_selector', to: 'kpi_cards', data: 'KPI data' },
    { from: 'viz_selector', to: 'heatmaps', data: 'CATE matrix' },
    { from: 'refutation', to: 'validation_badge', data: 'RefutationSuite' },
  ];

  const databases = [
    { id: 'causal_validations', name: 'causal_validations', desc: 'Refutation test results (V4.1)', color: '#ec4899' },
    { id: 'ml_experiments', name: 'ml_experiments', desc: 'MLflow experiment tracking', color: '#ec4899' },
    { id: 'causal_paths', name: 'causal_paths', desc: 'Discovered causal relationships', color: '#3b82f6' },
    { id: 'business_metrics', name: 'business_metrics', desc: 'KPI time series data', color: '#10b981' },
    { id: 'agent_activities', name: 'agent_activities', desc: 'Agent analysis outputs (RAG indexed)', color: '#f59e0b' },
  ];

  const getComponentStyle = (comp) => {
    const base = "px-3 py-2 rounded-lg text-xs font-medium transition-all duration-200 cursor-pointer border-2";
    const typeStyles = {
      input: "bg-gradient-to-r from-violet-500/20 to-purple-500/20 border-violet-400/50 text-violet-100",
      process: "bg-slate-800/80 border-slate-600/50 text-slate-200",
      agent: "bg-gradient-to-r from-blue-500/20 to-cyan-500/20 border-blue-400/50 text-blue-100",
      output: "bg-gradient-to-r from-amber-500/20 to-orange-500/20 border-amber-400/50 text-amber-100",
    };
    const active = activeStep === comp.id ? "ring-2 ring-white/50 scale-105 shadow-lg shadow-white/10" : "hover:scale-102 hover:brightness-110";
    return `${base} ${typeStyles[comp.type]} ${active}`;
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-950 via-slate-900 to-slate-950 p-6 font-sans">
      {/* Header */}
      <div className="max-w-7xl mx-auto mb-8">
        <div className="flex items-center gap-4 mb-2">
          <div className="w-12 h-12 rounded-xl bg-gradient-to-br from-pink-500 to-violet-600 flex items-center justify-center text-white font-bold text-lg shadow-lg shadow-pink-500/30">
            E2I
          </div>
          <div>
            <h1 className="text-2xl font-bold bg-gradient-to-r from-pink-400 via-violet-400 to-cyan-400 bg-clip-text text-transparent">
              Query Processing Flow Diagram
            </h1>
            <p className="text-slate-400 text-sm">V4.1 • 18 Agents • 6 Tiers • NLP → Dashboard End States</p>
          </div>
        </div>
        
        {/* Legend */}
        <div className="flex flex-wrap gap-4 mt-4 text-xs">
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 rounded bg-gradient-to-r from-violet-500 to-purple-500"></div>
            <span className="text-slate-400">Input</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 rounded bg-slate-600"></div>
            <span className="text-slate-400">Process</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 rounded bg-gradient-to-r from-blue-500 to-cyan-500"></div>
            <span className="text-slate-400">Agent</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 rounded bg-gradient-to-r from-amber-500 to-orange-500"></div>
            <span className="text-slate-400">Output</span>
          </div>
          <div className="flex items-center gap-2">
            <span className="px-1.5 py-0.5 rounded bg-pink-500/30 text-pink-300 text-[10px] font-bold">NEW</span>
            <span className="text-slate-400">V4 Addition</span>
          </div>
          <div className="flex items-center gap-2">
            <span className="px-1.5 py-0.5 rounded bg-purple-500/30 text-purple-300 text-[10px] font-bold">HYBRID</span>
            <span className="text-slate-400">LLM + Code</span>
          </div>
          <div className="flex items-center gap-2">
            <span className="px-1.5 py-0.5 rounded bg-red-500/30 text-red-300 text-[10px] font-bold">GATE</span>
            <span className="text-slate-400">Blocking Check</span>
          </div>
        </div>
      </div>

      {/* Main Flow Diagram */}
      <div className="max-w-7xl mx-auto grid grid-cols-1 gap-4">
        {layers.map((layer, idx) => (
          <div 
            key={layer.id}
            className={`relative rounded-2xl border transition-all duration-300 ${
              activeLayer === layer.id 
                ? 'border-white/30 bg-white/5 shadow-xl' 
                : 'border-slate-700/50 bg-slate-900/50 hover:border-slate-600/50'
            }`}
            onMouseEnter={() => setActiveLayer(layer.id)}
            onMouseLeave={() => setActiveLayer(null)}
          >
            {/* Layer Header */}
            <div 
              className="px-4 py-2 border-b border-slate-700/50 flex items-center gap-3"
              style={{ borderLeftColor: layer.color, borderLeftWidth: '4px' }}
            >
              <div 
                className="w-2 h-2 rounded-full"
                style={{ backgroundColor: layer.color }}
              ></div>
              <span className="font-semibold text-sm text-slate-200">{layer.name}</span>
              {layer.isNew && (
                <span className="px-1.5 py-0.5 rounded bg-pink-500/30 text-pink-300 text-[10px] font-bold">NEW IN V4</span>
              )}
              {idx < layers.length - 1 && (
                <div className="ml-auto text-slate-500 text-xs">↓ Data flows down</div>
              )}
            </div>
            
            {/* Components */}
            <div className="p-4 flex flex-wrap gap-3">
              {layer.components.map((comp) => (
                <div
                  key={comp.id}
                  className={getComponentStyle(comp)}
                  onClick={() => setActiveStep(activeStep === comp.id ? null : comp.id)}
                >
                  <div className="flex items-center gap-2">
                    <span className="font-semibold">{comp.name}</span>
                    {comp.isHybrid && (
                      <span className="px-1 py-0.5 rounded bg-purple-500/30 text-purple-300 text-[8px] font-bold">HYBRID</span>
                    )}
                    {comp.hasGate && (
                      <span className="px-1 py-0.5 rounded bg-red-500/30 text-red-300 text-[8px] font-bold">GATE</span>
                    )}
                  </div>
                  {activeStep === comp.id && (
                    <div className="mt-2 pt-2 border-t border-current/20 text-[10px] opacity-80">
                      {comp.desc}
                    </div>
                  )}
                </div>
              ))}
            </div>
          </div>
        ))}
      </div>

      {/* Database Tables */}
      <div className="max-w-7xl mx-auto mt-6">
        <div className="rounded-2xl border border-slate-700/50 bg-slate-900/50 p-4">
          <div className="flex items-center gap-2 mb-4">
            <span className="text-slate-400 text-sm font-semibold">📊 Database Tables (28 Total)</span>
            <span className="text-slate-500 text-xs">→ Indexed by RAG for retrieval</span>
          </div>
          <div className="flex flex-wrap gap-2">
            {databases.map((db) => (
              <div 
                key={db.id}
                className="px-3 py-2 rounded-lg bg-slate-800/80 border border-slate-600/50 text-xs cursor-pointer hover:bg-slate-700/80 transition-all"
                style={{ borderLeftColor: db.color, borderLeftWidth: '3px' }}
              >
                <div className="font-mono text-slate-200">{db.name}</div>
                <div className="text-slate-400 text-[10px] mt-1">{db.desc}</div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Data Flow Summary */}
      <div className="max-w-7xl mx-auto mt-6 grid grid-cols-1 md:grid-cols-3 gap-4">
        <div className="rounded-xl border border-violet-500/30 bg-violet-500/5 p-4">
          <h3 className="text-violet-300 font-semibold text-sm mb-2">🔤 NLP Layer Outputs</h3>
          <ul className="text-xs text-slate-400 space-y-1">
            <li>• <code className="text-violet-300">IntentType</code>: CAUSAL | GAP | DRIFT | ML_SCOPE | VALIDATION</li>
            <li>• <code className="text-violet-300">ExtractedEntities</code>: brands, regions, KPIs, time_periods</li>
            <li>• <code className="text-violet-300">RewrittenQuery</code>: Optimized for causal retrieval</li>
          </ul>
        </div>
        
        <div className="rounded-xl border border-cyan-500/30 bg-cyan-500/5 p-4">
          <h3 className="text-cyan-300 font-semibold text-sm mb-2">⚡ Causal Engine Outputs</h3>
          <ul className="text-xs text-slate-400 space-y-1">
            <li>• <code className="text-cyan-300">CausalEffect</code>: ATE + CI + p-value</li>
            <li>• <code className="text-cyan-300">RefutationSuite</code>: 5 tests + gate_decision</li>
            <li>• <code className="text-cyan-300">DAGSpec</code>: Nodes + edges for D3.js</li>
          </ul>
        </div>
        
        <div className="rounded-xl border border-amber-500/30 bg-amber-500/5 p-4">
          <h3 className="text-amber-300 font-semibold text-sm mb-2">📈 Dashboard End States</h3>
          <ul className="text-xs text-slate-400 space-y-1">
            <li>• <code className="text-amber-300">ChatResponse</code>: Streaming text + badges</li>
            <li>• <code className="text-amber-300">Visualizations</code>: DAG, heatmaps, Sankey</li>
            <li>• <code className="text-amber-300">ValidationBadge</code>: Refutation status (V4.1)</li>
          </ul>
        </div>
      </div>

      {/* Critical Constraints */}
      <div className="max-w-7xl mx-auto mt-6 rounded-xl border border-red-500/30 bg-red-500/5 p-4">
        <h3 className="text-red-300 font-semibold text-sm mb-3">⚠️ Critical Constraints in Flow</h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-xs">
          <div className="flex items-start gap-2">
            <span className="text-red-400">1.</span>
            <span className="text-slate-400"><strong className="text-red-300">NO Medical NER</strong> - Entity Extractor uses domain_vocabulary.yaml only</span>
          </div>
          <div className="flex items-start gap-2">
            <span className="text-red-400">2.</span>
            <span className="text-slate-400"><strong className="text-red-300">QC Gate Blocking</strong> - Data Preparer must pass before Model Trainer</span>
          </div>
          <div className="flex items-start gap-2">
            <span className="text-red-400">3.</span>
            <span className="text-slate-400"><strong className="text-red-300">Refutation Required</strong> - All causal effects need 5 DoWhy tests</span>
          </div>
          <div className="flex items-start gap-2">
            <span className="text-red-400">4.</span>
            <span className="text-slate-400"><strong className="text-red-300">ML Split Enforcement</strong> - Same patient always in same split</span>
          </div>
        </div>
      </div>

      {/* Footer */}
      <div className="max-w-7xl mx-auto mt-8 text-center text-slate-500 text-xs">
        E2I Causal Analytics V4.1 • 18 Agents • 6 Tiers • 28 Tables • 46 KPIs
      </div>
    </div>
  );
};

export default E2IQueryFlowDiagram;
