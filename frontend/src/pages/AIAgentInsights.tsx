/**
 * AI Agent Insights Page
 * =======================
 *
 * Main page for AI-powered insights including executive briefs,
 * priority actions, predictive alerts, and more.
 *
 * Brand and model identifiers are sourced from the active dashboard
 * context (`E2ICopilotProvider`) with optional URL-query override.
 * Each insight is wrapped in an error boundary so a single failing
 * component does not blank the whole page (issue #304).
 *
 * @module pages/AIAgentInsights
 */

import { Brain, Sparkles } from 'lucide-react';
import { useSearchParams } from 'react-router-dom';
import {
  ExecutiveAIBrief,
  PriorityActionsROI,
  PredictiveAlerts,
  ActiveCausalChains,
  ExperimentRecommendations,
  HeterogeneousTreatmentEffects,
  SystemHealthScore,
} from '@/components/insights';
import { Badge } from '@/components/ui/badge';
import { ErrorBoundary } from '@/components/ui/error-boundary';
import { useE2ICopilot } from '@/providers/E2ICopilotProvider';

// =============================================================================
// MAIN COMPONENT
// =============================================================================

export function AIAgentInsights() {
  const [searchParams] = useSearchParams();
  const { filters } = useE2ICopilot();

  // Brand: URL query takes precedence, then dashboard filter context.
  // When neither is set we hand `undefined` to the child so its own
  // documented default kicks in — the page no longer hard-codes a brand.
  const brandFromUrl = searchParams.get('brand')?.trim();
  const brand = brandFromUrl || filters?.brand || undefined;

  // Model id: URL query takes precedence, then a deploy-time env override
  // (`VITE_DEFAULT_MODEL_ID`). When neither is set we hand `undefined`
  // to `SystemHealthScore` and let its own documented default kick in —
  // no model identifier is hard-coded on this page (issue #304).
  const modelIdFromUrl = searchParams.get('modelId')?.trim();
  const modelIdFromEnv =
    typeof import.meta !== 'undefined'
      ? (import.meta.env?.VITE_DEFAULT_MODEL_ID as string | undefined)?.trim()
      : undefined;
  const modelId = modelIdFromUrl || modelIdFromEnv || undefined;

  return (
    <div className="space-y-6">
      {/* Page Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="p-3 rounded-xl bg-purple-500/10">
            <Brain className="h-7 w-7 text-purple-500" />
          </div>
          <div>
            <h1 className="text-2xl font-bold text-[var(--color-foreground)]">
              AI Agent Insights
            </h1>
            <p className="text-sm text-[var(--color-muted-foreground)]">
              GPT-powered executive summaries, recommendations, and predictive alerts
            </p>
          </div>
        </div>
        <Badge variant="outline" className="text-sm">
          <Sparkles className="h-4 w-4 mr-1" />
          21 Agents Active
        </Badge>
      </div>

      {/* Main Grid Layout */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Executive Brief - Full Width */}
        <div className="lg:col-span-2">
          <ErrorBoundary sectionName="Executive AI Brief">
            <ExecutiveAIBrief brand={brand} />
          </ErrorBoundary>
        </div>

        {/* Priority Actions */}
        <div className="lg:col-span-1">
          <ErrorBoundary sectionName="Priority Actions">
            <PriorityActionsROI brand={brand} />
          </ErrorBoundary>
        </div>

        {/* Predictive Alerts */}
        <div className="lg:col-span-1">
          <ErrorBoundary sectionName="Predictive Alerts">
            <PredictiveAlerts />
          </ErrorBoundary>
        </div>

        {/* Active Causal Chains - Full Width */}
        <div className="lg:col-span-2">
          <ErrorBoundary sectionName="Active Causal Chains">
            <ActiveCausalChains />
          </ErrorBoundary>
        </div>

        {/* Experiment Recommendations */}
        <div className="lg:col-span-1">
          <ErrorBoundary sectionName="Experiment Recommendations">
            <ExperimentRecommendations />
          </ErrorBoundary>
        </div>

        {/* Heterogeneous Treatment Effects */}
        <div className="lg:col-span-1">
          <ErrorBoundary sectionName="Heterogeneous Treatment Effects">
            <HeterogeneousTreatmentEffects />
          </ErrorBoundary>
        </div>

        {/* System Health Score - Full Width */}
        <div className="lg:col-span-2">
          <ErrorBoundary sectionName="System Health Score">
            <SystemHealthScore modelId={modelId} />
          </ErrorBoundary>
        </div>
      </div>
    </div>
  );
}

export default AIAgentInsights;
