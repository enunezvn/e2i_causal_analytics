import { post } from '@/lib/api-client';
import type {
  StrategicInsightResponse,
  KGInsightRequest,
  ModelPerfInsightRequest,
  CausalInsightRequest,
  PredictiveInsightRequest,
  ResourceInsightRequest,
  TreatmentEffectInsightRequest,
} from '@/types/insights';

const BASE = '/insights';

export const getKnowledgeGraphInsight = (r: KGInsightRequest) =>
  post<StrategicInsightResponse, KGInsightRequest>(`${BASE}/knowledge-graph`, r);

export const getModelPerformanceInsight = (r: ModelPerfInsightRequest) =>
  post<StrategicInsightResponse, ModelPerfInsightRequest>(`${BASE}/model-performance`, r);

export const getCausalDiscoveryInsight = (r: CausalInsightRequest) =>
  post<StrategicInsightResponse, CausalInsightRequest>(`${BASE}/causal-discovery`, r);

export const getPredictiveCohortInsight = (r: PredictiveInsightRequest) =>
  post<StrategicInsightResponse, PredictiveInsightRequest>(`${BASE}/predictive-cohort`, r);

export const getResourceOptimizationInsight = (r: ResourceInsightRequest) =>
  post<StrategicInsightResponse, ResourceInsightRequest>(`${BASE}/resource-optimization`, r);

export const getTreatmentEffectInsight = (r: TreatmentEffectInsightRequest) =>
  post<StrategicInsightResponse, TreatmentEffectInsightRequest>(`${BASE}/treatment-effect`, r);
