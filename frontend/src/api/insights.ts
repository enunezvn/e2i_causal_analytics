import { post } from '@/lib/api-client';
import type {
  StrategicInsightResponse,
  KGInsightRequest,
  ModelPerfInsightRequest,
  CausalInsightRequest,
  PredictiveInsightRequest,
  ResourceInsightRequest,
  TreatmentEffectInsightRequest,
  ExecutiveBriefInsightRequest,
  HTEInsightRequest,
  FeedbackLearningInsightRequest,
  DigitalTwinInsightRequest,
  HomeKpiInsightRequest,
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

export const getExecutiveBriefInsight = (r: ExecutiveBriefInsightRequest) =>
  post<StrategicInsightResponse, ExecutiveBriefInsightRequest>(`${BASE}/executive-brief`, r);

export const getHTEInsight = (r: HTEInsightRequest) =>
  post<StrategicInsightResponse, HTEInsightRequest>(`${BASE}/hte`, r);

export const getFeedbackLearningInsight = (r: FeedbackLearningInsightRequest) =>
  post<StrategicInsightResponse, FeedbackLearningInsightRequest>(`${BASE}/feedback-learning`, r);

export const getDigitalTwinInsight = (r: DigitalTwinInsightRequest) =>
  post<StrategicInsightResponse, DigitalTwinInsightRequest>(`${BASE}/digital-twin`, r);

export const getHomeKpiInsight = (r: HomeKpiInsightRequest) =>
  post<StrategicInsightResponse, HomeKpiInsightRequest>(
    `${BASE}/home-kpis`,
    r,
    // Server recomputes the full KPI batch (~6-10s) then runs the LM (~8-23s
    // measured cold) — a cold scope can exceed the 30s client default. Redis
    // caches the payload per scope, so repeats are fast; nginx allows 120s.
    { timeout: 95000 }
  );
