/**
 * API Hooks Index
 * ===============
 *
 * Central export for all API-related React Query hooks.
 *
 * @module hooks/api
 */

// =============================================================================
// GAP ANALYSIS HOOKS
// =============================================================================

export {
  useGapAnalysis,
  useOpportunities,
  useGapHealth,
  useRunGapAnalysis,
  useRunGapAnalysisAndWait,
  useQuickWins,
  useStrategicBets,
  usePollGapAnalysis,
} from './use-gaps';

// =============================================================================
// EXPERIMENTS / A/B TESTING HOOKS
// =============================================================================

export {
  useAssignments,
  useEnrollmentStats,
  useInterimAnalyses,
  useExperimentResults,
  useSegmentResults,
  useSRMChecks,
  useFidelityComparisons,
  useExperimentHealth,
  useExperimentAlerts,
  useRandomizeUnits,
  useEnrollUnit,
  useWithdrawUnit,
  useTriggerInterimAnalysis,
  useRunSRMCheck,
  useUpdateFidelityComparison,
  useTriggerMonitoring,
} from './use-experiments';

// =============================================================================
// CAUSAL INFERENCE HOOKS
// =============================================================================

export {
  useHierarchicalAnalysis,
  useEstimators,
  useCausalHealth,
  useRunHierarchicalAnalysis,
  useRunHierarchicalAnalysisAndWait,
  useRouteQuery,
  useRunSequentialPipeline,
  useRunParallelPipeline,
  useRunCrossValidation,
  useRouteAndRunAnalysis,
  useQuickEffectEstimate,
  useFullCausalAnalysis,
  usePollHierarchicalAnalysis,
} from './use-causal';

// =============================================================================
// RESOURCE OPTIMIZATION HOOKS
// =============================================================================

export {
  useOptimization,
  useScenarios,
  useResourceHealth,
  useRunOptimization,
  useRunOptimizationAndWait,
  useOptimizeBudget,
  useOptimizeWithScenarios,
  usePollOptimization,
} from './use-resources';

// =============================================================================
// SEGMENT ANALYSIS HOOKS
// =============================================================================

export {
  useSegmentAnalysis,
  usePolicies,
  useSegmentHealth,
  useRunSegmentAnalysis,
  useRunSegmentAnalysisAndWait,
  useGetHighResponders,
  useGetOptimalPolicy,
  usePollSegmentAnalysis,
} from './use-segments';

// =============================================================================
// HEALTH SCORE HOOKS
// =============================================================================

export {
  useQuickHealthCheck,
  useFullHealthCheck,
  useScopedHealthCheck,
  useComponentHealth,
  useModelHealth,
  usePipelineHealth,
  useAgentHealth,
  useHealthHistory,
  useHealthServiceStatus,
  useComprehensiveHealth,
  useHealthDashboard,
  useRunHealthCheck,
  useHealthMonitor,
} from './use-health-score';

// Re-export composite types
export type { ComprehensiveHealthData, DashboardHealthData } from './use-health-score';
