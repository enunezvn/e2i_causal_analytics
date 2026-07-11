import { useMutation } from '@tanstack/react-query';
import { ApiError } from '@/lib/api-client';
import {
  getKnowledgeGraphInsight,
  getModelPerformanceInsight,
  getCausalDiscoveryInsight,
  getPredictiveCohortInsight,
  getPredictiveWhatIfInsight,
  getResourceOptimizationInsight,
  getTreatmentEffectInsight,
  getExecutiveBriefInsight,
  getHTEInsight,
  getFeedbackLearningInsight,
  getDigitalTwinInsight,
  getHomeKpiInsight,
  getExperimentsInsight,
} from '@/api/insights';
import type {
  StrategicInsightResponse,
  KGInsightRequest,
  ModelPerfInsightRequest,
  CausalInsightRequest,
  PredictiveInsightRequest,
  PredictiveWhatIfInsightRequest,
  ResourceInsightRequest,
  TreatmentEffectInsightRequest,
  ExecutiveBriefInsightRequest,
  HTEInsightRequest,
  FeedbackLearningInsightRequest,
  DigitalTwinInsightRequest,
  HomeKpiInsightRequest,
  ExperimentsInsightRequest,
} from '@/types/insights';

export const useKnowledgeGraphInsight = () =>
  useMutation<StrategicInsightResponse, ApiError, KGInsightRequest>({
    mutationFn: getKnowledgeGraphInsight,
  });

export const useModelPerformanceInsight = () =>
  useMutation<StrategicInsightResponse, ApiError, ModelPerfInsightRequest>({
    mutationFn: getModelPerformanceInsight,
  });

export const useCausalDiscoveryInsight = () =>
  useMutation<StrategicInsightResponse, ApiError, CausalInsightRequest>({
    mutationFn: getCausalDiscoveryInsight,
  });

export const usePredictiveCohortInsight = () =>
  useMutation<StrategicInsightResponse, ApiError, PredictiveInsightRequest>({
    mutationFn: getPredictiveCohortInsight,
  });

export const usePredictiveWhatIfInsight = () =>
  useMutation<StrategicInsightResponse, ApiError, PredictiveWhatIfInsightRequest>({
    mutationFn: getPredictiveWhatIfInsight,
  });

export const useResourceOptimizationInsight = () =>
  useMutation<StrategicInsightResponse, ApiError, ResourceInsightRequest>({
    mutationFn: getResourceOptimizationInsight,
  });

export const useTreatmentEffectInsight = () =>
  useMutation<StrategicInsightResponse, ApiError, TreatmentEffectInsightRequest>({
    mutationFn: getTreatmentEffectInsight,
  });

export const useExecutiveBriefInsight = () =>
  useMutation<StrategicInsightResponse, ApiError, ExecutiveBriefInsightRequest>({
    mutationFn: getExecutiveBriefInsight,
  });

export const useHTEInsight = () =>
  useMutation<StrategicInsightResponse, ApiError, HTEInsightRequest>({
    mutationFn: getHTEInsight,
  });

export const useFeedbackLearningInsight = () =>
  useMutation<StrategicInsightResponse, ApiError, FeedbackLearningInsightRequest>({
    mutationFn: getFeedbackLearningInsight,
  });

export const useDigitalTwinInsight = () =>
  useMutation<StrategicInsightResponse, ApiError, DigitalTwinInsightRequest>({
    mutationFn: getDigitalTwinInsight,
  });

export const useHomeKpiInsight = () =>
  useMutation<StrategicInsightResponse, ApiError, HomeKpiInsightRequest>({
    mutationFn: getHomeKpiInsight,
  });

export const useExperimentsInsight = () =>
  useMutation<StrategicInsightResponse, ApiError, ExperimentsInsightRequest>({
    mutationFn: getExperimentsInsight,
  });
