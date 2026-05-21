/**
 * SystemHealth Test/Storybook Fixtures
 * =====================================
 *
 * Hardcoded fixture data formerly inlined in SystemHealth.tsx as SAMPLE_*
 * constants. Moved here so production rendering paths cannot reach them
 * (F-002). Tests and Storybook stories may import from here.
 *
 * @module pages/__fixtures__/systemHealth
 */

import {
  Server,
  Database,
  HardDrive,
  Activity,
  Cpu,
} from 'lucide-react';
import type { AgentHealth, PipelineHealth as PipelineHealthType } from '@/types/health-score';
import { PipelineStatus } from '@/types/health-score';

interface ServiceStatusFixture {
  name: string;
  status: 'healthy' | 'warning' | 'error' | 'unknown';
  latencyMs?: number;
  lastCheck?: Date;
  icon: React.ElementType;
}

interface ModelHealthFixture {
  modelId: string;
  name: string;
  healthScore: number;
  status: 'healthy' | 'warning' | 'critical';
  driftScore: number;
  activeAlerts: number;
  lastRetrained?: Date;
  performanceTrend: 'improving' | 'stable' | 'degrading';
}

export const FIXTURE_SERVICES: ServiceStatusFixture[] = [
  { name: 'API Gateway', status: 'healthy', latencyMs: 45, lastCheck: new Date(), icon: Server },
  { name: 'PostgreSQL', status: 'healthy', latencyMs: 12, lastCheck: new Date(), icon: Database },
  { name: 'Redis Cache', status: 'healthy', latencyMs: 3, lastCheck: new Date(), icon: HardDrive },
  { name: 'FalkorDB', status: 'healthy', latencyMs: 28, lastCheck: new Date(), icon: Activity },
  { name: 'BentoML', status: 'healthy', latencyMs: 156, lastCheck: new Date(), icon: Cpu },
];

export const FIXTURE_MODELS: ModelHealthFixture[] = [
  {
    modelId: 'propensity_v2.1.0',
    name: 'Propensity Model',
    healthScore: 92,
    status: 'healthy',
    driftScore: 0.15,
    activeAlerts: 0,
    lastRetrained: new Date(Date.now() - 7 * 24 * 60 * 60 * 1000),
    performanceTrend: 'stable',
  },
  {
    modelId: 'churn_v1.5.2',
    name: 'Churn Prediction',
    healthScore: 78,
    status: 'warning',
    driftScore: 0.42,
    activeAlerts: 2,
    lastRetrained: new Date(Date.now() - 21 * 24 * 60 * 60 * 1000),
    performanceTrend: 'degrading',
  },
  {
    modelId: 'conversion_v3.0.1',
    name: 'Conversion Model',
    healthScore: 88,
    status: 'healthy',
    driftScore: 0.22,
    activeAlerts: 1,
    lastRetrained: new Date(Date.now() - 14 * 24 * 60 * 60 * 1000),
    performanceTrend: 'improving',
  },
];

export const FIXTURE_HISTORY = [
  { timestamp: new Date(Date.now() - 6 * 24 * 60 * 60 * 1000).toISOString(), overall_health_score: 85, health_grade: 'B' },
  { timestamp: new Date(Date.now() - 5 * 24 * 60 * 60 * 1000).toISOString(), overall_health_score: 82, health_grade: 'B' },
  { timestamp: new Date(Date.now() - 4 * 24 * 60 * 60 * 1000).toISOString(), overall_health_score: 88, health_grade: 'B' },
  { timestamp: new Date(Date.now() - 3 * 24 * 60 * 60 * 1000).toISOString(), overall_health_score: 91, health_grade: 'A' },
  { timestamp: new Date(Date.now() - 2 * 24 * 60 * 60 * 1000).toISOString(), overall_health_score: 89, health_grade: 'B' },
  { timestamp: new Date(Date.now() - 1 * 24 * 60 * 60 * 1000).toISOString(), overall_health_score: 92, health_grade: 'A' },
  { timestamp: new Date().toISOString(), overall_health_score: 94, health_grade: 'A' },
];

export const FIXTURE_AGENT_HEALTH: AgentHealth[] = [
  { agent_name: 'Orchestrator', tier: 1, available: true, avg_latency_ms: 120, success_rate: 0.98, invocations_24h: 245 },
  { agent_name: 'ToolComposer', tier: 1, available: true, avg_latency_ms: 85, success_rate: 0.99, invocations_24h: 180 },
  { agent_name: 'CausalImpact', tier: 2, available: true, avg_latency_ms: 450, success_rate: 0.95, invocations_24h: 67 },
  { agent_name: 'GapAnalyzer', tier: 2, available: true, avg_latency_ms: 320, success_rate: 0.97, invocations_24h: 89 },
  { agent_name: 'HeterogeneousOptimizer', tier: 2, available: true, avg_latency_ms: 380, success_rate: 0.94, invocations_24h: 45 },
  { agent_name: 'DriftMonitor', tier: 3, available: true, avg_latency_ms: 200, success_rate: 0.99, invocations_24h: 156 },
  { agent_name: 'ExperimentDesigner', tier: 3, available: true, avg_latency_ms: 280, success_rate: 0.96, invocations_24h: 34 },
  { agent_name: 'HealthScore', tier: 3, available: true, avg_latency_ms: 150, success_rate: 0.99, invocations_24h: 312 },
  { agent_name: 'PredictionSynthesizer', tier: 4, available: true, avg_latency_ms: 520, success_rate: 0.97, invocations_24h: 23 },
  { agent_name: 'ResourceOptimizer', tier: 4, available: true, avg_latency_ms: 410, success_rate: 0.95, invocations_24h: 56 },
  { agent_name: 'Explainer', tier: 5, available: true, avg_latency_ms: 680, success_rate: 0.93, invocations_24h: 112 },
  { agent_name: 'FeedbackLearner', tier: 5, available: true, avg_latency_ms: 340, success_rate: 0.97, invocations_24h: 78 },
];

export const FIXTURE_PIPELINE_HEALTH: PipelineHealthType[] = [
  { pipeline_name: 'TRx Data Ingestion', last_run: new Date().toISOString(), last_success: new Date().toISOString(), rows_processed: 125000, freshness_hours: 2.5, status: PipelineStatus.HEALTHY },
  { pipeline_name: 'Feature Store Sync', last_run: new Date().toISOString(), last_success: new Date().toISOString(), rows_processed: 45000, freshness_hours: 1.2, status: PipelineStatus.HEALTHY },
  { pipeline_name: 'Model Retraining', last_run: new Date(Date.now() - 24 * 60 * 60 * 1000).toISOString(), last_success: new Date(Date.now() - 24 * 60 * 60 * 1000).toISOString(), rows_processed: 8500, freshness_hours: 24, status: PipelineStatus.STALE },
  { pipeline_name: 'Causal Graph Update', last_run: new Date().toISOString(), last_success: new Date().toISOString(), rows_processed: 12000, freshness_hours: 4.5, status: PipelineStatus.HEALTHY },
];
