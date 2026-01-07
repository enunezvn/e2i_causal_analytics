/**
 * AI Agent Insights Page
 * =======================
 *
 * Main page for AI-powered insights including executive briefs,
 * priority actions, predictive alerts, and more.
 *
 * @module pages/AIAgentInsights
 */

import { Brain, Sparkles } from 'lucide-react';
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

// =============================================================================
// MAIN COMPONENT
// =============================================================================

export function AIAgentInsights() {
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
          18 Agents Active
        </Badge>
      </div>

      {/* Main Grid Layout */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Executive Brief - Full Width */}
        <div className="lg:col-span-2">
          <ExecutiveAIBrief brand="Remibrutinib" />
        </div>

        {/* Priority Actions */}
        <div className="lg:col-span-1">
          <PriorityActionsROI />
        </div>

        {/* Predictive Alerts */}
        <div className="lg:col-span-1">
          <PredictiveAlerts />
        </div>

        {/* Active Causal Chains - Full Width */}
        <div className="lg:col-span-2">
          <ActiveCausalChains />
        </div>

        {/* Experiment Recommendations */}
        <div className="lg:col-span-1">
          <ExperimentRecommendations />
        </div>

        {/* Heterogeneous Treatment Effects */}
        <div className="lg:col-span-1">
          <HeterogeneousTreatmentEffects />
        </div>

        {/* System Health Score - Full Width */}
        <div className="lg:col-span-2">
          <SystemHealthScore modelId="propensity_v2.1.0" />
        </div>
      </div>
    </div>
  );
}

export default AIAgentInsights;
