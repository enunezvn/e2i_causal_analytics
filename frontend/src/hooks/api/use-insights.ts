import { useMutation } from '@tanstack/react-query';
import { ApiError } from '@/lib/api-client';
import {
  getKnowledgeGraphInsight,
  getModelPerformanceInsight,
  getCausalDiscoveryInsight,
  getPredictiveCohortInsight,
  getResourceOptimizationInsight,
} from '@/api/insights';
import type {
  StrategicInsightResponse,
  KGInsightRequest,
  ModelPerfInsightRequest,
  CausalInsightRequest,
  PredictiveInsightRequest,
  ResourceInsightRequest,
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

export const useResourceOptimizationInsight = () =>
  useMutation<StrategicInsightResponse, ApiError, ResourceInsightRequest>({
    mutationFn: getResourceOptimizationInsight,
  });
