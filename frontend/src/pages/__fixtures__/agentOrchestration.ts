/**
 * AgentOrchestration Test/Storybook Fixtures
 * ===========================================
 *
 * Hardcoded fixture data formerly inlined in AgentOrchestration.tsx as
 * SAMPLE_ACTIVITIES. Moved here so production rendering paths cannot
 * reach them (F-002). Tests and Storybook stories may import from here.
 *
 * @module pages/__fixtures__/agentOrchestration
 */

interface AgentActivityFixture {
  id: string;
  agentId: string;
  agentName: string;
  tier: 0 | 1 | 2 | 3 | 4 | 5;
  action: string;
  timestamp: string;
  duration?: number;
  status: 'completed' | 'in_progress' | 'failed';
  details?: string;
}

export const FIXTURE_ACTIVITIES: AgentActivityFixture[] = [
  {
    id: 'act-1',
    agentId: 'orchestrator',
    agentName: 'Orchestrator',
    tier: 1,
    action: 'Routed query to Causal Impact agent',
    timestamp: new Date(Date.now() - 2 * 60000).toISOString(),
    duration: 145,
    status: 'completed',
    details: 'Query: "What drove Remibrutinib growth in Q4?"',
  },
  {
    id: 'act-2',
    agentId: 'causal-impact',
    agentName: 'Causal Impact',
    tier: 2,
    action: 'Traced causal chain for HCP engagement → TRx',
    timestamp: new Date(Date.now() - 5 * 60000).toISOString(),
    duration: 3200,
    status: 'completed',
    details: 'Found 3 significant causal paths with ATE = 0.23',
  },
  {
    id: 'act-3',
    agentId: 'drift-monitor',
    agentName: 'Drift Monitor',
    tier: 3,
    action: 'Data drift check scheduled',
    timestamp: new Date(Date.now() - 8 * 60000).toISOString(),
    status: 'in_progress',
    details: 'Checking conversion_model features',
  },
  {
    id: 'act-4',
    agentId: 'explainer',
    agentName: 'Explainer',
    tier: 5,
    action: 'Generated SHAP narrative for prediction',
    timestamp: new Date(Date.now() - 12 * 60000).toISOString(),
    duration: 890,
    status: 'completed',
    details: 'Patient journey explanation with 5 key factors',
  },
  {
    id: 'act-5',
    agentId: 'gap-analyzer',
    agentName: 'Gap Analyzer',
    tier: 2,
    action: 'ROI opportunity detection',
    timestamp: new Date(Date.now() - 15 * 60000).toISOString(),
    status: 'failed',
    details: 'Timeout waiting for feature store response',
  },
];
