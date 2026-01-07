/**
 * AgenticMethodology Component
 * ============================
 *
 * Displays the E2I Agentic Methodology with:
 * - System Architecture Overview (stats)
 * - 4-Layer Processing Architecture
 * - 6-Tier Agent Architecture
 *
 * @module components/kpi/AgenticMethodology
 */

import React from 'react';

// ============================================================================
// Types
// ============================================================================

interface ArchitectureStat {
  value: string;
  label: string;
}

interface ProcessingLayer {
  emoji: string;
  title: string;
  description: string;
  colorClass: string;
  borderColor: string;
}

interface Agent {
  emoji: string;
  name: string;
  type: 'Standard' | 'Hybrid' | 'Deep';
  sla?: string;
  description: string;
}

interface Tier {
  number: number;
  title: string;
  agentCount: string;
  bgColor: string;
  borderColor: string;
  badgeColor: string;
  badge?: string;
  agents: Agent[];
}

export interface AgenticMethodologyProps {
  /** Optional className for custom styling */
  className?: string;
}

// ============================================================================
// Data Constants
// ============================================================================

const ARCHITECTURE_STATS: ArchitectureStat[] = [
  { value: '4', label: 'Processing Layers' },
  { value: '6', label: 'Agent Tiers' },
  { value: '18', label: 'Specialized Agents' },
  { value: '28', label: 'Database Tables' },
];

const PROCESSING_LAYERS: ProcessingLayer[] = [
  {
    emoji: '🗣️',
    title: 'Layer 1: Conversational Interface',
    description: 'Domain-specific NLP • Intent classification • Entity extraction • Query rewriting • NO medical NER',
    colorClass: 'bg-blue-50',
    borderColor: 'border-blue-500',
  },
  {
    emoji: '🔍',
    title: 'Layer 2: Causal Reasoning & RAG',
    description: 'DoWhy/EconML inference • NetworkX DAG • Graph-enhanced retrieval • Hybrid search (dense + sparse + graph)',
    colorClass: 'bg-green-50',
    borderColor: 'border-green-500',
  },
  {
    emoji: '🤖',
    title: 'Layer 3: 18-Agent 6-Tier Orchestration',
    description: 'LangGraph state management • Priority-based routing • 13 Standard + 3 Hybrid + 2 Deep agents',
    colorClass: 'bg-purple-50',
    borderColor: 'border-purple-500',
  },
  {
    emoji: '🎓',
    title: 'Layer 4: Self-Improvement & Learning',
    description: 'Feedback collection • Quality scoring • DSPy prompt optimization • Continuous learning loops',
    colorClass: 'bg-amber-50',
    borderColor: 'border-amber-500',
  },
];

const TIERS: Tier[] = [
  {
    number: 0,
    title: 'ML Foundation',
    agentCount: '7 Agents',
    bgColor: 'bg-pink-50',
    borderColor: 'border-pink-500',
    badgeColor: 'bg-gradient-to-r from-pink-500 to-rose-500',
    badge: 'NEW IN V4',
    agents: [
      { emoji: '📋', name: 'scope_definer', type: 'Standard', sla: '5s', description: 'Problem scope, success criteria' },
      { emoji: '🔍', name: 'data_preparer', type: 'Standard', sla: '60s', description: 'QC gating, Great Expectations' },
      { emoji: '🎯', name: 'model_selector', type: 'Standard', sla: '120s', description: 'Algorithm evaluation, MLflow' },
      { emoji: '⚙️', name: 'model_trainer', type: 'Standard', description: 'Split enforcement, Optuna' },
      { emoji: '📊', name: 'feature_analyzer', type: 'Hybrid', sla: '120s', description: 'SHAP values + LLM interpretation' },
      { emoji: '🚀', name: 'model_deployer', type: 'Standard', sla: '30s', description: 'BentoML serving, stage lifecycle' },
      { emoji: '👁️', name: 'observability_connector', type: 'Standard', sla: '100ms', description: 'Opik spans, cross-tier telemetry' },
    ],
  },
  {
    number: 1,
    title: 'Coordination',
    agentCount: '1 Agent • Routes & Synthesizes',
    bgColor: 'bg-violet-50',
    borderColor: 'border-violet-500',
    badgeColor: 'bg-violet-500',
    agents: [
      { emoji: '🎯', name: 'orchestrator', type: 'Standard', description: 'Never performs analysis. Coordinates all agents, dynamic routing, multi-step planning, response synthesis. Entry point for all queries.' },
    ],
  },
  {
    number: 2,
    title: 'Causal Analytics',
    agentCount: '3 Agents • Core E2I Mission',
    bgColor: 'bg-blue-50',
    borderColor: 'border-blue-500',
    badgeColor: 'bg-blue-500',
    agents: [
      { emoji: '⚡', name: 'causal_impact', type: 'Hybrid', description: '5-node workflow: GraphBuilder → Estimation → Refutation → Sensitivity → Interpretation' },
      { emoji: '📈', name: 'gap_analyzer', type: 'Standard', description: 'Gap detection, ROI calculation, opportunity identification' },
      { emoji: '🎨', name: 'heterogeneous_optimizer', type: 'Standard', description: 'Segment-level treatment effects, personalization' },
    ],
  },
  {
    number: 3,
    title: 'Monitoring & Experimentation',
    agentCount: '3 Agents',
    bgColor: 'bg-emerald-50',
    borderColor: 'border-emerald-500',
    badgeColor: 'bg-emerald-500',
    agents: [
      { emoji: '⚠️', name: 'drift_monitor', type: 'Standard', description: 'Feature drift detection, degradation alerts' },
      { emoji: '🧪', name: 'experiment_designer', type: 'Hybrid', description: 'Power analysis, test design, experimental learning' },
      { emoji: '💚', name: 'health_score', type: 'Standard', description: 'System health metrics, Pareto scoring' },
    ],
  },
  {
    number: 4,
    title: 'ML Predictions',
    agentCount: '2 Agents',
    bgColor: 'bg-amber-50',
    borderColor: 'border-amber-500',
    badgeColor: 'bg-amber-500',
    agents: [
      { emoji: '🔮', name: 'prediction_synthesizer', type: 'Standard', description: 'Ensemble predictions, confidence calibration' },
      { emoji: '💰', name: 'resource_optimizer', type: 'Standard', description: 'Resource allocation, constraint optimization' },
    ],
  },
  {
    number: 5,
    title: 'Self-Improvement',
    agentCount: '2 Agents • NLV + RAG',
    bgColor: 'bg-slate-50',
    borderColor: 'border-slate-400',
    badgeColor: 'bg-slate-500',
    agents: [
      { emoji: '📝', name: 'explainer', type: 'Deep', description: 'Natural language narratives, visualization explanations' },
      { emoji: '🎓', name: 'feedback_learner', type: 'Deep', description: 'Prompt optimization, pattern learning, continuous improvement' },
    ],
  },
];

// ============================================================================
// Sub-components
// ============================================================================

const AgentTypeTag: React.FC<{ type: Agent['type'] }> = ({ type }) => {
  const colorMap = {
    Standard: 'text-gray-600',
    Hybrid: 'text-purple-600',
    Deep: 'text-indigo-600',
  };

  return <span className={`text-xs font-medium ${colorMap[type]}`}>{type}</span>;
};

const TierCard: React.FC<{ tier: Tier }> = ({ tier }) => (
  <div className={`${tier.bgColor} border-2 ${tier.borderColor} rounded-xl p-5 mb-4`}>
    <div className="flex items-center gap-3 mb-4">
      <span className={`${tier.badgeColor} text-white px-4 py-2 rounded-lg font-bold text-sm`}>
        TIER {tier.number}
      </span>
      <strong className="text-lg">{tier.title}</strong>
      {tier.badge && (
        <span className="bg-rose-100 text-rose-700 px-3 py-1 rounded-full text-xs font-semibold ml-auto">
          {tier.badge}
        </span>
      )}
      {!tier.badge && (
        <span className="text-sm text-gray-500 ml-auto">{tier.agentCount}</span>
      )}
    </div>
    <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-3">
      {tier.agents.map((agent, idx) => (
        <div
          key={idx}
          className={`bg-white p-3 rounded-lg border-l-4 ${
            agent.type === 'Hybrid' ? 'border-purple-500' :
            agent.type === 'Deep' ? 'border-indigo-500' :
            tier.borderColor
          }`}
        >
          <div className="font-semibold text-sm flex items-center gap-1">
            <span>{agent.emoji}</span>
            <span>{agent.name}</span>
          </div>
          <div className="flex items-center gap-2 mt-1">
            <AgentTypeTag type={agent.type} />
            {agent.sla && <span className="text-xs text-gray-400">• SLA: {agent.sla}</span>}
          </div>
          <div className="text-xs text-gray-500 mt-2">{agent.description}</div>
        </div>
      ))}
    </div>
  </div>
);

// ============================================================================
// Main Component
// ============================================================================

export const AgenticMethodology: React.FC<AgenticMethodologyProps> = ({ className = '' }) => {
  return (
    <div className={`space-y-8 ${className}`}>
      {/* Header */}
      <div className="text-center">
        <h2 className="text-xl font-bold text-gray-800 flex items-center justify-center gap-2">
          <span>Agentic Methodology</span>
          <span className="bg-indigo-100 text-indigo-700 px-3 py-1 rounded-full text-sm font-medium">
            V4.1 Architecture
          </span>
        </h2>
        <p className="text-sm text-gray-500 mt-2 max-w-2xl mx-auto">
          Complete 18-agent, 6-tier architecture with ML Foundation, Tri-Memory system, causal validation workflow, and MLOps integration.
        </p>
      </div>

      {/* Architecture Overview Stats */}
      <div className="bg-gradient-to-r from-gray-50 to-gray-100 p-6 rounded-xl">
        <h3 className="text-base font-semibold text-gray-800 mb-4 flex items-center gap-2">
          <span>🏗️</span>
          <span>System Architecture Overview</span>
        </h3>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-6 text-center">
          {ARCHITECTURE_STATS.map((stat, idx) => (
            <div key={idx}>
              <div className="text-3xl font-bold text-indigo-600">{stat.value}</div>
              <div className="text-sm text-gray-500">{stat.label}</div>
            </div>
          ))}
        </div>
      </div>

      {/* 4-Layer Processing Architecture */}
      <div>
        <h3 className="text-base font-semibold text-gray-800 mb-4 flex items-center gap-2">
          <span>📊</span>
          <span>4-Layer Processing Architecture</span>
        </h3>
        <div className="space-y-3">
          {PROCESSING_LAYERS.map((layer, idx) => (
            <div
              key={idx}
              className={`flex items-center gap-4 p-4 ${layer.colorClass} border-l-4 ${layer.borderColor} rounded-lg`}
            >
              <div className="text-2xl">{layer.emoji}</div>
              <div className="flex-1">
                <div className="font-semibold text-gray-800">{layer.title}</div>
                <div className="text-sm text-gray-600">{layer.description}</div>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* 6-Tier Agent Architecture */}
      <div>
        <h3 className="text-base font-semibold text-gray-800 mb-4 flex items-center gap-2">
          <span>🎯</span>
          <span>Complete 6-Tier Agent Architecture</span>
        </h3>
        {TIERS.map((tier) => (
          <TierCard key={tier.number} tier={tier} />
        ))}
      </div>

      {/* Agent Type Legend */}
      <div className="bg-gray-50 rounded-lg p-4">
        <div className="text-sm font-medium text-gray-700 mb-2">Agent Types:</div>
        <div className="flex flex-wrap gap-4 text-sm">
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 rounded bg-gray-400"></div>
            <span><strong>Standard:</strong> Rule-based, deterministic (13 agents)</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 rounded bg-purple-500"></div>
            <span><strong>Hybrid:</strong> ML + LLM integration (3 agents)</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 rounded bg-indigo-500"></div>
            <span><strong>Deep:</strong> Full LLM-powered (2 agents)</span>
          </div>
        </div>
      </div>
    </div>
  );
};

export default AgenticMethodology;
