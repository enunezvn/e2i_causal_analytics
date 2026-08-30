/**
 * Gap Analysis Page
 * =================
 *
 * Dashboard for ROI opportunity detection and performance gap analysis.
 * Integrates with the Tier 2 Gap Analyzer agent.
 *
 * @module pages/GapAnalysis
 */

import { useState, useMemo } from 'react';
import {
  Target,
  RefreshCw,
  Download,
  Search,
  DollarSign,
  Clock,
  Filter,
  BarChart3,
  ArrowUpRight,
  ArrowDownRight,
  Loader2,
  Play,
} from 'lucide-react';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  CartesianGrid,
  Cell,
  PieChart,
  Pie,
  Legend,
} from 'recharts';

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Input } from '@/components/ui/input';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import { KPICard, StatusBadge } from '@/components/visualizations';
import { usePageChatContext } from '@/providers/E2ICopilotProvider';
import { WarningBanner } from '@/components/ui/WarningBanner';
import { EmptyState } from '@/components/ui/EmptyState';
import { LabelGateBadge } from '@/components/insights/LabelGateBadge';
import { CompetitorDensityBadge } from '@/components/insights/CompetitorDensityBadge';
import { OpportunityDrilldownDialog } from '@/components/gaps/OpportunityDrilldownDialog';
import {
  useOpportunities,
  useGapHealth,
  useRunGapAnalysisAndWait,
} from '@/hooks/api';
import type { PrioritizedOpportunity } from '@/types/gaps';
import { BUCKET_META, bucketMeta } from '@/lib/gaps/interpret';

// =============================================================================
// CONSTANTS
// =============================================================================

// Brand `value` is sent verbatim to the API as the `brand` filter and must use
// the system's canonical CAPITALIZED casing — it matches the Supabase
// `brand_type` ENUM, the synthetic `Brand` enum, and the copilot provider's
// `VALID_BRANDS`. The grounded `gap_analyses` rows are stored capitalized, so
// sending lowercase here previously returned nothing and emptied the page.
// `'all'` is a sentinel that maps to *no* brand filter on the API: the
// /gaps/opportunities endpoint treats an absent brand as "every brand" and
// returns the latest run per brand. It is NOT a real brand, so "Run Analysis"
// (which needs a concrete brand) is disabled while it is selected.
const ALL_BRANDS = 'all';

/**
 * Poll ceiling (ms) for the durable async gap-analysis run. The backend
 * persists the analysis record and the run completes server-side whether or
 * not the page is still polling, so this only decides how long the page waits
 * before giving up on a run that may be genuinely stuck.
 *
 * Measured on prod from the durable `gap_analyses` rows (n=28 completed,
 * 2026-06-08..2026-08-26): `total_latency_ms` avg 4.5 s, p95 10.8 s, worst
 * 35.3 s (a single 2026-06-19 outlier); end-to-end submit->complete
 * (`updated_at - created_at`) worst 12.6 s. 120 s is >3x the worst sample, so
 * it is kept — a ceiling is a measurement, not a constant (#1839). The hook no
 * longer retries a timed-out poll (use-gaps.ts), so reaching this ceiling can
 * not re-submit the analysis.
 */
const GAP_ANALYSIS_POLL_CEILING_MS = 120_000;

const BRANDS = [
  { value: ALL_BRANDS, label: 'All Brands' },
  { value: 'Kisqali', label: 'Kisqali' },
  { value: 'Fabhalta', label: 'Fabhalta' },
  { value: 'Remibrutinib', label: 'Remibrutinib' },
];

const DIFFICULTY_COLORS: Record<string, string> = {
  low: '#10b981',
  medium: '#f59e0b',
  high: '#ef4444',
};

// Implementation-EFFORT labels — SHORT forms, because effort is now folded in as
// a SECONDARY attribute (rendered with an "Effort:" prefix), not the primary
// label. The primary label is the curated Quick Win / Strategic Bet category
// (see CATEGORY_* below). The page previously badged every high-difficulty
// opportunity "High Effort" as the headline label, conflating effort with the
// curated category and making it look like there were far more strategic bets
// than the KPI reported.
const DIFFICULTY_LABELS: Record<string, string> = {
  low: 'Low',
  medium: 'Medium',
  high: 'High',
};

// Curated list-view category — the 3-bucket scheme (Quick Win / Steady Play /
// Strategic Bet) set by the list endpoint via the shared classification SSOT.
// This is the PRIMARY card label. There is NO residual "other": every surfaced
// opportunity is exactly one of the three. Labels/colors come from BUCKET_META
// (lib/gaps/interpret) so the page, the drill-down, and the charts agree.
// Default bucket when a (pre-T6) row carries no category.
const DEFAULT_CATEGORY = 'steady_play';

// =============================================================================
// HELPER FUNCTIONS
// =============================================================================

function formatCurrency(value: number): string {
  if (value >= 1000000) return `$${(value / 1000000).toFixed(1)}M`;
  if (value >= 1000) return `$${(value / 1000).toFixed(1)}K`;
  return `$${value.toFixed(0)}`;
}

// Primary curated-category badge (Quick Win / Steady Play / Strategic Bet).
function getCategoryBadge(category?: string) {
  const meta = bucketMeta(category ?? DEFAULT_CATEGORY);
  const color = meta.color;
  return (
    <Badge style={{ backgroundColor: `${color}20`, color, borderColor: color }} variant="outline">
      {meta.label}
    </Badge>
  );
}

// Secondary, folded-in implementation-effort badge. Prefixed "Effort:" so it
// reads as an attribute of the opportunity, not its primary classification.
function getDifficultyBadge(difficulty: string) {
  const color = DIFFICULTY_COLORS[difficulty] || '#6b7280';
  const label = DIFFICULTY_LABELS[difficulty] ?? difficulty;

  return (
    <Badge
      style={{
        backgroundColor: `${color}20`,
        color: color,
        borderColor: color,
      }}
      variant="outline"
    >
      {`Effort: ${label}`}
    </Badge>
  );
}
// =============================================================================
// COMPONENT
// =============================================================================

function GapAnalysis() {
  const [selectedBrand, setSelectedBrand] = useState<string>('Kisqali');
  const [searchQuery, setSearchQuery] = useState('');
  const [categoryFilter, setCategoryFilter] = useState<string>('all');
  const [isRefreshing, setIsRefreshing] = useState(false);
  // T6 drill-down: the opportunity whose "why" drawer is open (null = closed).
  const [drillOpp, setDrillOpp] = useState<PrioritizedOpportunity | null>(null);

  const isAllBrands = selectedBrand === ALL_BRANDS;

  // API hooks
  const { data: opportunitiesData, isLoading: opportunitiesLoading, refetch: refetchOpportunities } = useOpportunities({
    // 'all' → omit the brand filter so the endpoint returns the latest run per brand.
    brand: isAllBrands ? undefined : selectedBrand,
    limit: 50,
  });
  const { data: healthData, isLoading: _healthLoading } = useGapHealth();
  // Poll-to-completion variant: keeps `isPending` true until the background
  // analysis reaches a terminal state, then invalidates the opportunities query
  // so the new results render. See `handleRunAnalysis` for the why.
  const runGapAnalysisMutation = useRunGapAnalysisAndWait();

  // F-002 fix: no fabricated `SAMPLE_OPPORTUNITIES` fallback. Data comes
  // strictly from API; absence renders empty state below.
  const opportunities = opportunitiesData?.opportunities ?? [];
  const totalAddressableValue = opportunitiesData?.total_addressable_value ?? 0;
  const quickWinsCount = opportunitiesData?.quick_wins_count ?? 0;
  const steadyPlaysCount = opportunitiesData?.steady_plays_count ?? 0;
  const strategicBetsCount = opportunitiesData?.strategic_bets_count ?? 0;
  // T6 transparency: how many low-value (ROI <= break-even) opportunities the
  // backend hid from this run, so a short/empty list never looks broken.
  const suppressedCount = opportunitiesData?.suppressed_count ?? 0;

  // F-010: surface backend-reported warnings from the analysis mutation
  // (the opportunities-list endpoint does not currently return warnings —
  // only the analysis-run endpoint does). The mutation now waits for the
  // background analysis to complete before resolving, so `.data` is the
  // COMPLETED analysis and `.error` carries a genuine failure/timeout — both
  // surfaced below instead of being silently dropped.
  const apiWarnings = runGapAnalysisMutation.data?.warnings ?? [];
  const runError = runGapAnalysisMutation.error;

  // Calculate metrics
  const metrics = useMemo(() => {
    const totalGaps = opportunities.length;
    const avgROI = opportunities.reduce((sum, opp) => sum + opp.roi_estimate.expected_roi, 0) / totalGaps || 0;
    const avgConfidence = opportunities.reduce((sum, opp) => sum + opp.roi_estimate.confidence, 0) / totalGaps || 0;

    return {
      totalGaps,
      avgROI: avgROI.toFixed(1),
      avgConfidence: (avgConfidence * 100).toFixed(0),
      quickWins: quickWinsCount,
      steadyPlays: steadyPlaysCount,
      strategicBets: strategicBetsCount,
    };
  }, [opportunities, quickWinsCount, steadyPlaysCount, strategicBetsCount]);

  // Filter opportunities
  const filteredOpportunities = useMemo(() => {
    return opportunities.filter((opp) => {
      const matchesCategory = categoryFilter === 'all' || (opp.category ?? DEFAULT_CATEGORY) === categoryFilter;
      const matchesSearch =
        searchQuery === '' ||
        opp.recommended_action.toLowerCase().includes(searchQuery.toLowerCase()) ||
        opp.gap.metric.toLowerCase().includes(searchQuery.toLowerCase()) ||
        opp.gap.segment_value.toLowerCase().includes(searchQuery.toLowerCase());
      return matchesCategory && matchesSearch;
    });
  }, [opportunities, categoryFilter, searchQuery]);

  // Chart data — average ROI grouped by curated opportunity TYPE (the Quick Win
  // / Strategic Bet framework), not by implementation effort.
  const roiByTypeData = useMemo(() => {
    const grouped: Record<string, { total: number; count: number }> = {};
    opportunities.forEach((opp) => {
      const cat = opp.category ?? DEFAULT_CATEGORY;
      if (!grouped[cat]) grouped[cat] = { total: 0, count: 0 };
      grouped[cat].total += opp.roi_estimate.expected_roi;
      grouped[cat].count += 1;
    });
    return Object.entries(grouped).map(([category, { total, count }]) => ({
      type: bucketMeta(category).label,
      avgROI: total / count,
      color: bucketMeta(category).color,
    }));
  }, [opportunities]);

  const gapsByMetricData = useMemo(() => {
    const grouped: Record<string, number> = {};
    opportunities.forEach((opp) => {
      const metric = opp.gap.metric;
      grouped[metric] = (grouped[metric] || 0) + opp.roi_estimate.estimated_revenue_impact;
    });
    return Object.entries(grouped).map(([metric, value]) => ({
      metric,
      value,
    }));
  }, [opportunities]);

  const handleRefresh = async () => {
    setIsRefreshing(true);
    await refetchOpportunities();
    setIsRefreshing(false);
  };

  const handleRunAnalysis = () => {
    // "All Brands" is a cross-brand VIEW; the analyze endpoint needs one concrete
    // brand. The button is disabled in this state — guard defensively too.
    if (isAllBrands) return;
    // The backend runs the gap analysis as a background task (~8s on the live
    // cohort). The previous fire-and-forget call (async mode, no polling)
    // refetched opportunities *immediately* — racing ahead of the still-running
    // job — so the page never showed the new result and the button appeared to
    // "do nothing". Polling to completion keeps the button in its running state
    // and refreshes the opportunities list once the analysis finishes.
    runGapAnalysisMutation.mutate({
      request: {
        query: `Identify performance gaps for ${selectedBrand}`,
        brand: selectedBrand,
        metrics: ['trx', 'nrx', 'market_share'],
        segments: ['region', 'specialty', 'account_type'],
      },
      pollIntervalMs: 3000,
      // Durable record; see the ceiling constant for the measured basis.
      maxWaitMs: GAP_ANALYSIS_POLL_CEILING_MS,
    });
  };

  const handleExport = () => {
    // "all" is a cross-brand view, not a brand named "all" — label the export
    // accordingly in both the body and the filename.
    const exportBrandLabel = isAllBrands ? 'all-brands' : selectedBrand;
    const report = {
      generatedAt: new Date().toISOString(),
      brand: exportBrandLabel,
      totalOpportunities: opportunities.length,
      totalAddressableValue,
      opportunities,
    };
    const blob = new Blob([JSON.stringify(report, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `gap-analysis-${exportBrandLabel}-${new Date().toISOString().split('T')[0]}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  // Publish a compact on-screen data summary so the chat pane can generate
  // opener pills grounded in what this page is showing (usePageChatContext →
  // POST /chat/suggestions page_context).
  const pageChatSummary = useMemo(() => {
    const lines: string[] = [
      `Gap Analysis page. Brand filter: ${isAllBrands ? 'All brands' : selectedBrand}.`,
    ];
    if (opportunities.length > 0) {
      lines.push(
        `Opportunities on screen: ${opportunities.length} (${quickWinsCount} quick wins, ${steadyPlaysCount} steady plays, ${strategicBetsCount} strategic bets); total addressable value ${totalAddressableValue}.`
      );
      const top = opportunities[0];
      lines.push(
        `Top opportunity: ${top.gap.metric} gap in ${top.gap.segment_value} (gap size ${top.gap.gap_size}, expected ROI ${top.roi_estimate.expected_roi}) — ${top.recommended_action}.`
      );
    }
    return lines.join('\n');
  }, [
    isAllBrands,
    selectedBrand,
    opportunities,
    quickWinsCount,
    steadyPlaysCount,
    strategicBetsCount,
    totalAddressableValue,
  ]);
  usePageChatContext(pageChatSummary);

  return (
    <div className="container mx-auto px-4 py-8">
      {/* Header */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 mb-8">
        <div>
          <h1 className="text-3xl font-bold mb-2 flex items-center gap-3">
            <Target className="h-8 w-8" />
            Gap Analysis
          </h1>
          <p className="text-muted-foreground">
            ROI opportunity detection and performance gap prioritization powered by the Tier 2 Gap Analyzer agent.
          </p>
          <p className="text-xs text-muted-foreground mt-1">
            {isAllBrands
              ? 'Showing the most recent analysis for each brand.'
              : `Showing the most recent analysis for ${selectedBrand}.`}
          </p>
        </div>
        <div className="flex items-center gap-2">
          <Select value={selectedBrand} onValueChange={setSelectedBrand}>
            <SelectTrigger className="w-40">
              <SelectValue placeholder="Brand" />
            </SelectTrigger>
            <SelectContent>
              {BRANDS.map((brand) => (
                <SelectItem key={brand.value} value={brand.value}>
                  {brand.label}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
          <Button
            variant="outline"
            onClick={handleRunAnalysis}
            disabled={runGapAnalysisMutation.isPending || isAllBrands}
            title={isAllBrands ? 'Select a specific brand to run a new analysis' : undefined}
          >
            {runGapAnalysisMutation.isPending ? (
              <Loader2 className="h-4 w-4 mr-2 animate-spin" />
            ) : (
              <Play className="h-4 w-4 mr-2" />
            )}
            Run Analysis
          </Button>
          <Button variant="outline" onClick={handleRefresh} disabled={isRefreshing}>
            <RefreshCw className={`h-4 w-4 mr-2 ${isRefreshing ? 'animate-spin' : ''}`} />
            Refresh
          </Button>
          <Button variant="outline" onClick={handleExport}>
            <Download className="h-4 w-4 mr-2" />
            Export
          </Button>
        </div>
      </div>

      {/* Run-in-progress indicator — the background analysis takes a few
          seconds; keep the user informed instead of showing a silent spinner
          that returns to idle with no visible change. */}
      {runGapAnalysisMutation.isPending && (
        <div className="mb-6">
          <Card className="border-blue-200">
            <CardContent className="py-3">
              <div className="flex items-center gap-2 text-sm text-muted-foreground">
                <Loader2 className="h-4 w-4 animate-spin text-blue-500" />
                Running gap analysis for {selectedBrand}… results will appear here when it completes.
              </div>
            </CardContent>
          </Card>
        </div>
      )}

      {/* Analysis failure/timeout — the run mutation now waits for completion,
          so a rejection here is a genuine failed/timed-out analysis. Surface it
          rather than swallowing it. */}
      {runError && (
        <div className="mb-6">
          <WarningBanner
            title="Gap analysis failed"
            messages={[runError.message || 'The gap analysis did not complete. Please try again.']}
          />
        </div>
      )}

      {/* API-reported warnings (F-010) — surfaced prominently so users
          see when the backend fell through to mock or degraded mode. */}
      {apiWarnings.length > 0 && (
        <div className="mb-6">
          <WarningBanner messages={apiWarnings} />
        </div>
      )}

      {/* Service Health Banner */}
      {healthData && (
        <div className="mb-6">
          <Card className={healthData.agent_available ? 'border-emerald-200' : 'border-amber-200'}>
            <CardContent className="py-3">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <StatusBadge
                    status={healthData.agent_available ? 'healthy' : 'warning'}
                    showIcon
                    pulse={!healthData.agent_available}
                  />
                  <span className="text-sm">
                    Gap Analyzer Agent {healthData.agent_available ? 'Available' : 'Unavailable'}
                  </span>
                </div>
                <div className="text-sm text-muted-foreground">
                  {healthData.analyses_24h} analyses in last 24h
                </div>
              </div>
            </CardContent>
          </Card>
        </div>
      )}

      {/* Overview Metrics */}
      <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-7 gap-4 mb-8">
        <KPICard
          title="Total Addressable"
          value={formatCurrency(totalAddressableValue)}
          status="healthy"
          description="Total revenue opportunity"
          size="sm"
        />
        <KPICard
          title="Opportunities"
          value={metrics.totalGaps}
          status="healthy"
          description="Identified gaps"
          size="sm"
        />
        <KPICard
          title="Avg ROI"
          value={parseFloat(metrics.avgROI)}
          unit="x"
          status={parseFloat(metrics.avgROI) > 3 ? 'healthy' : 'warning'}
          description="Expected return"
          size="sm"
        />
        <KPICard
          title="Quick Wins"
          value={metrics.quickWins}
          status="healthy"
          description="Low effort, high ROI"
          size="sm"
        />
        <KPICard
          title="Steady Plays"
          value={metrics.steadyPlays}
          status="healthy"
          description="Dependable middle ground"
          size="sm"
        />
        <KPICard
          title="Strategic Bets"
          value={metrics.strategicBets}
          status="warning"
          description="High effort, high impact"
          size="sm"
        />
        <KPICard
          title="Confidence"
          value={parseInt(metrics.avgConfidence)}
          unit="%"
          status={parseInt(metrics.avgConfidence) > 80 ? 'healthy' : 'warning'}
          description="Estimate confidence"
          size="sm"
        />
      </div>

      {/* Tabs — only show when opportunities are loaded (F-002). */}
      {opportunities.length === 0 && !opportunitiesLoading ? (
        suppressedCount > 0 ? (
          // Every candidate was found and then suppressed as value-destroying —
          // an economic conclusion, NOT a missing or failed analysis. Without
          // this branch the generic "Click Run Analysis" copy implied nothing
          // ran (the live Fabhalta case), and the suppressed-notice inside the
          // Tabs was unreachable in exactly the state it was built for.
          <div data-testid="all-suppressed-empty-state">
            <EmptyState
              title="No opportunities above break-even"
              description={`The latest analysis identified ${suppressedCount} candidate gap${
                suppressedCount === 1 ? '' : 's'
              }, but ${
                suppressedCount === 1 ? 'it falls' : 'all fall'
              } at or below the break-even ROI threshold — closing ${
                suppressedCount === 1 ? 'it' : 'each'
              } would cost at least as much as it returns, so no action is recommended.`}
            />
          </div>
        ) : (
        <EmptyState
          title="No gap opportunities available"
          description="Click Run Analysis to identify ROI-prioritized performance gaps for the selected brand."
        />
        )
      ) : (
      <Tabs defaultValue="opportunities" className="space-y-4">
        <TabsList>
          <TabsTrigger value="opportunities" className="flex items-center gap-2">
            <Target className="h-4 w-4" />
            Opportunities
          </TabsTrigger>
          <TabsTrigger value="charts" className="flex items-center gap-2">
            <BarChart3 className="h-4 w-4" />
            Analysis
          </TabsTrigger>
        </TabsList>

        {/* Opportunities Tab */}
        <TabsContent value="opportunities" className="space-y-4">
          {/* Filters */}
          <div className="flex flex-col md:flex-row gap-4">
            <div className="relative flex-1">
              <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-muted-foreground" />
              <Input
                placeholder="Filter loaded opportunities by action, metric, or segment…"
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="pl-9"
              />
            </div>
            <Select value={categoryFilter} onValueChange={setCategoryFilter}>
              <SelectTrigger className="w-48">
                <Filter className="h-4 w-4 mr-2" />
                <SelectValue placeholder="Opportunity type" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="all">All Opportunities</SelectItem>
                <SelectItem value="quick_win">{BUCKET_META.quick_win.label}s</SelectItem>
                <SelectItem value="steady_play">{BUCKET_META.steady_play.label}s</SelectItem>
                <SelectItem value="strategic_bet">{BUCKET_META.strategic_bet.label}s</SelectItem>
              </SelectContent>
            </Select>
          </div>

          {/* T6 transparency: low-value opportunities the backend suppressed
              (ROI at or below break-even) — shown so a short/empty list is
              never mistaken for a broken page. */}
          {suppressedCount > 0 && (
            <p className="text-xs text-muted-foreground" data-testid="suppressed-notice">
              {suppressedCount} low-value{' '}
              {suppressedCount === 1 ? 'opportunity' : 'opportunities'} hidden (ROI at or below
              break-even — revenue would not cover the cost to close).
            </p>
          )}

          {/* Opportunity Cards */}
          {opportunitiesLoading ? (
            <div className="flex items-center justify-center py-12">
              <Loader2 className="h-8 w-8 animate-spin text-muted-foreground" />
            </div>
          ) : (
            <div className="space-y-4">
              {filteredOpportunities.map((opp) => (
                <Card
                  key={opp.gap.gap_id}
                  className="hover:shadow-md transition-shadow cursor-pointer"
                  role="button"
                  tabIndex={0}
                  aria-label={`View details for ${opp.recommended_action}`}
                  onClick={() => setDrillOpp(opp)}
                  onKeyDown={(e) => {
                    if (e.key === 'Enter' || e.key === ' ') {
                      e.preventDefault();
                      setDrillOpp(opp);
                    }
                  }}
                >
                  <CardContent className="py-4">
                    <div className="flex flex-col lg:flex-row lg:items-center gap-4">
                      {/* Rank */}
                      <div className="flex-shrink-0 w-12 h-12 bg-primary/10 rounded-full flex items-center justify-center">
                        <span className="text-lg font-bold text-primary">#{opp.rank}</span>
                      </div>

                      {/* Main Content */}
                      <div className="flex-1">
                        <div className="flex flex-wrap items-center gap-2 mb-1">
                          <h3 className="font-semibold">{opp.recommended_action}</h3>
                          {getCategoryBadge(opp.category)}
                          {getDifficultyBadge(opp.implementation_difficulty)}
                          <LabelGateBadge
                            label_verdict={opp.roi_estimate.label_verdict}
                            off_label={opp.roi_estimate.off_label}
                            off_label_reason={opp.roi_estimate.off_label_reason}
                            label_evidence_confirmed={opp.roi_estimate.label_evidence_confirmed}
                          />
                        </div>
                        <div className="flex flex-wrap items-center gap-4 text-sm text-muted-foreground">
                          <span className="flex items-center gap-1">
                            <Badge variant="outline">{opp.gap.metric}</Badge>
                          </span>
                          <span>
                            {opp.gap.segment}: {opp.gap.segment_value}
                          </span>
                          <span className="flex items-center gap-1">
                            {opp.gap.gap_percentage > 0 ? (
                              <ArrowDownRight className="h-3 w-3 text-rose-500" />
                            ) : (
                              <ArrowUpRight className="h-3 w-3 text-emerald-500" />
                            )}
                            {opp.gap.gap_percentage.toFixed(1)}% gap
                          </span>
                          <span className="flex items-center gap-1">
                            <Clock className="h-3 w-3" />
                            {opp.time_to_impact}
                          </span>
                        </div>
                        {/* Market landscape — surface-only competitor density (#1056);
                            never affects ROI or ranking. Honest empty when 0/unknown. */}
                        <CompetitorDensityBadge
                          competitor_products_count={opp.roi_estimate.competitor_products_count}
                          competitor_density_label={opp.roi_estimate.competitor_density_label}
                          competitor_drug_names={opp.roi_estimate.competitor_drug_names}
                          className="mt-2"
                        />
                      </div>

                      {/* ROI Metrics */}
                      <div className="flex items-center gap-6 lg:gap-8">
                        <div className="text-center">
                          <p className="text-sm text-muted-foreground">Revenue Impact</p>
                          <p className="text-lg font-bold text-emerald-600">
                            {formatCurrency(opp.roi_estimate.estimated_revenue_impact)}
                          </p>
                        </div>
                        <div className="text-center">
                          <p className="text-sm text-muted-foreground">Investment</p>
                          <p className="text-lg font-bold text-amber-600">
                            {formatCurrency(opp.roi_estimate.estimated_cost_to_close)}
                          </p>
                        </div>
                        <div className="text-center">
                          <p className="text-sm text-muted-foreground">ROI</p>
                          <p className="text-lg font-bold text-primary">
                            {opp.roi_estimate.expected_roi.toFixed(1)}x
                          </p>
                        </div>
                        <div className="text-center">
                          <p className="text-sm text-muted-foreground">Confidence</p>
                          <p className="text-lg font-bold">
                            {(opp.roi_estimate.confidence * 100).toFixed(0)}%
                          </p>
                        </div>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              ))}
            </div>
          )}
        </TabsContent>

        {/* Charts Tab */}
        <TabsContent value="charts" className="space-y-4">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            {/* ROI by Opportunity Type */}
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <BarChart3 className="h-5 w-5" />
                  Average ROI by Opportunity Type
                </CardTitle>
                <CardDescription>Expected returns by opportunity type</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="h-64">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={roiByTypeData} layout="vertical">
                      <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
                      <XAxis type="number" stroke="var(--muted-foreground)" fontSize={12} />
                      <YAxis dataKey="type" type="category" stroke="var(--muted-foreground)" fontSize={12} width={80} />
                      <Tooltip
                        contentStyle={{
                          backgroundColor: 'var(--card)',
                          border: '1px solid var(--border)',
                          borderRadius: '8px',
                        }}
                        formatter={(value) => [`${(value as number)?.toFixed(1) ?? 0}x`, 'Avg ROI']}
                      />
                      <Bar dataKey="avgROI" radius={[0, 4, 4, 0]}>
                        {roiByTypeData.map((entry, index) => (
                          <Cell key={index} fill={entry.color} />
                        ))}
                      </Bar>
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </CardContent>
            </Card>

            {/* Value by Metric */}
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <DollarSign className="h-5 w-5" />
                  Revenue Opportunity by Metric
                </CardTitle>
                <CardDescription>Total addressable value by KPI</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="h-64">
                  <ResponsiveContainer width="100%" height="100%">
                    <PieChart>
                      <Pie
                        data={gapsByMetricData}
                        dataKey="value"
                        nameKey="metric"
                        cx="50%"
                        cy="50%"
                        outerRadius={80}
                        label={({ name, value }) => `${name}: ${formatCurrency(value as number)}`}
                      >
                        {gapsByMetricData.map((_, index) => (
                          <Cell
                            key={index}
                            fill={['#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6'][index % 5]}
                          />
                        ))}
                      </Pie>
                      <Tooltip
                        contentStyle={{
                          backgroundColor: 'var(--card)',
                          border: '1px solid var(--border)',
                          borderRadius: '8px',
                        }}
                        formatter={(value) => [formatCurrency(value as number), 'Value']}
                      />
                      <Legend />
                    </PieChart>
                  </ResponsiveContainer>
                </div>
              </CardContent>
            </Card>
          </div>

          {/* Top Opportunities Table */}
          <Card>
            <CardHeader>
              <CardTitle>Top Opportunities Summary</CardTitle>
              <CardDescription>Highest ROI opportunities across all metrics</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="overflow-x-auto">
                <table className="w-full">
                  <thead>
                    <tr className="border-b border-border">
                      <th className="text-left py-3 px-4 font-medium text-muted-foreground">Rank</th>
                      <th className="text-left py-3 px-4 font-medium text-muted-foreground">Metric</th>
                      <th className="text-left py-3 px-4 font-medium text-muted-foreground">Segment</th>
                      <th className="text-right py-3 px-4 font-medium text-muted-foreground">Gap</th>
                      <th className="text-right py-3 px-4 font-medium text-muted-foreground">Revenue</th>
                      <th className="text-right py-3 px-4 font-medium text-muted-foreground">Cost</th>
                      <th className="text-right py-3 px-4 font-medium text-muted-foreground">ROI</th>
                      <th className="text-center py-3 px-4 font-medium text-muted-foreground">Type</th>
                    </tr>
                  </thead>
                  <tbody>
                    {opportunities.slice(0, 10).map((opp) => (
                      <tr key={opp.gap.gap_id} className="border-b border-border hover:bg-muted/50">
                        <td className="py-3 px-4 font-bold text-primary">#{opp.rank}</td>
                        <td className="py-3 px-4">
                          <Badge variant="outline">{opp.gap.metric}</Badge>
                        </td>
                        <td className="py-3 px-4 text-sm">
                          <span className="inline-flex flex-wrap items-center gap-1">
                            {opp.gap.segment_value}
                            {(opp.roi_estimate.label_verdict === 'off_label' ||
                              opp.roi_estimate.label_verdict === 'mixed') && (
                              <Badge
                                variant="warning"
                                className="text-xs"
                                title={
                                  opp.roi_estimate.off_label_reason
                                    ? `${opp.roi_estimate.off_label_reason} — de-prioritized in ranking.`
                                    : 'Off-label — de-prioritized in ranking.'
                                }
                              >
                                {opp.roi_estimate.label_verdict === 'mixed'
                                  ? 'Partly off-label'
                                  : 'Off-label'}
                              </Badge>
                            )}
                          </span>
                        </td>
                        <td className="py-3 px-4 text-right text-rose-500">
                          {opp.gap.gap_percentage.toFixed(1)}%
                        </td>
                        <td className="py-3 px-4 text-right font-medium text-emerald-600">
                          {formatCurrency(opp.roi_estimate.estimated_revenue_impact)}
                        </td>
                        <td className="py-3 px-4 text-right text-amber-600">
                          {formatCurrency(opp.roi_estimate.estimated_cost_to_close)}
                        </td>
                        <td className="py-3 px-4 text-right font-bold">
                          {opp.roi_estimate.expected_roi.toFixed(1)}x
                        </td>
                        <td className="py-3 px-4 text-center">
                          {getCategoryBadge(opp.category)}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
      )}

      {/* T6 drill-down (extracted to a shared component in T7 so the AI-Insights
          Priority-Actions card surfaces the identical "why" drawer): click an
          opportunity card to understand WHY it's ranked, bucketed, and timed the
          way it is — plus the full ROI rationale. */}
      <OpportunityDrilldownDialog
        opp={drillOpp}
        allOpps={opportunities}
        onClose={() => setDrillOpp(null)}
      />
    </div>
  );
}

export default GapAnalysis;
