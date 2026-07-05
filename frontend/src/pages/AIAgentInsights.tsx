/**
 * AI Agent Insights Page
 * =======================
 *
 * Main page for AI-powered insights including executive briefs,
 * priority actions, predictive alerts, and more.
 *
 * The brand is sourced from the active dashboard context
 * (`E2ICopilotProvider`) with optional URL-query override.
 * Each insight is wrapped in an error boundary so a single failing
 * component does not blank the whole page (issue #304).
 *
 * System health lives on the dedicated /system-health page — the
 * duplicate System Health Score card this page carried was consolidated
 * there (same `useFullHealthCheck` source, one home). The card's
 * `?modelId=` deep link (#304's URL-addressable per-model drift route)
 * moved with it: `/monitoring?modelId=<id>` pre-selects that model's
 * drift/health view.
 *
 * @module pages/AIAgentInsights
 */

import { useState } from 'react';
import { Brain, Sparkles } from 'lucide-react';
import { useSearchParams } from 'react-router-dom';
import {
  ExecutiveAIBrief,
  PriorityActionsROI,
  PredictiveAlerts,
  ActiveCausalChains,
  ExperimentRecommendations,
  HeterogeneousTreatmentEffects,
} from '@/components/insights';
import { Badge } from '@/components/ui/badge';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import { ErrorBoundary } from '@/components/ui/error-boundary';
import { useAgentHealth } from '@/hooks/api';
import { isTrustedProvenance } from '@/lib/provenance';
import { useE2ICopilot } from '@/providers/E2ICopilotProvider';
import { GOLDSTD_BRANDS } from '@/types/explain';

// Sentinel for the "All brands" selection — handed to children as `undefined`
// so each one's own documented default kicks in (all-brand listing for
// PriorityActionsROI; per-brand default for ExecutiveAIBrief).
const ALL_BRANDS = 'All';

// =============================================================================
// MAIN COMPONENT
// =============================================================================

export function AIAgentInsights() {
  const [searchParams] = useSearchParams();
  const { filters } = useE2ICopilot();

  // Brand: seed the selector from the URL query, then the dashboard filter
  // context; the user can override it via the header dropdown. "All brands"
  // hands `undefined` to each child so its own documented default kicks in —
  // the page never hard-codes a brand.
  const brandFromUrl = searchParams.get('brand')?.trim();
  // Local override: `null` means "follow URL/context" so issue #304's
  // precedence stays REACTIVE if the dashboard filter context changes after
  // mount; once the user picks from the in-page selector their choice wins.
  // We derive (rather than snapshot into state) to avoid a stale selection.
  const [brandOverride, setBrandOverride] = useState<string | null>(null);
  const rawBrand = brandOverride || brandFromUrl || filters?.brand || ALL_BRANDS;
  // Normalize to a representable value: an unknown brand (e.g. a stale ?brand=
  // link or a non-gold context brand) must not silently flow into the children's
  // RAG prompt / opportunities API filter while the Select shows only a
  // placeholder. Coerce anything outside the gold-standard set to "All brands".
  const selectedBrand = (GOLDSTD_BRANDS as readonly string[]).includes(rawBrand)
    ? rawBrand
    : ALL_BRANDS;
  const brand = selectedBrand === ALL_BRANDS ? undefined : selectedBrand;

  // Real agent availability for the header badge. When the health-score
  // service hasn't answered — or answered with untrusted (placeholder/unknown)
  // provenance, i.e. sample data rather than a live probe — the badge is
  // simply absent: no invented count (codex PR-4 round 4).
  const { data: agentHealth } = useAgentHealth();
  const trustedAgentHealth =
    agentHealth && isTrustedProvenance(agentHealth.data_provenance) ? agentHealth : null;

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
        <div className="flex items-center gap-3">
          <Select value={selectedBrand} onValueChange={setBrandOverride}>
            <SelectTrigger aria-label="Brand" className="w-[160px]">
              <SelectValue placeholder="All brands" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value={ALL_BRANDS}>All brands</SelectItem>
              {GOLDSTD_BRANDS.map((b) => (
                <SelectItem key={b} value={b}>
                  {b}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
          {trustedAgentHealth && (
            <Badge variant="outline" className="text-sm">
              <Sparkles className="h-4 w-4 mr-1" />
              {trustedAgentHealth.available_count}/{trustedAgentHealth.total_agents} Agents Active
            </Badge>
          )}
        </div>
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

        {/* Experiment Health Monitor */}
        <div className="lg:col-span-1">
          <ErrorBoundary sectionName="Experiment Health Monitor">
            <ExperimentRecommendations />
          </ErrorBoundary>
        </div>

        {/* Heterogeneous Treatment Effects */}
        <div className="lg:col-span-1">
          <ErrorBoundary sectionName="Heterogeneous Treatment Effects">
            <HeterogeneousTreatmentEffects brand={brand} />
          </ErrorBoundary>
        </div>
      </div>
    </div>
  );
}

export default AIAgentInsights;
