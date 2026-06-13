/**
 * Home Page - KPI Executive Dashboard
 * ====================================
 *
 * Main landing page for E2I Causal Analytics.
 * Displays key performance indicators, agent insights, and quick actions.
 *
 * Features:
 * - KPI dashboard with 46+ metrics organized by category
 * - Brand selector (Remibrutinib, Fabhalta, Kisqali)
 * - Recent agent insights feed
 * - System health summary (from API)
 * - Quick action navigation
 *
 * @module pages/Home
 */

import { useState, useMemo, useCallback, useEffect } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { useQuery, useQueryClient } from '@tanstack/react-query';

// API Hooks
import { useKPIList, useKPIHealth, useBatchCalculateKPIs, useKPIValue } from '@/hooks/api/use-kpi';
import { useGraphStats } from '@/hooks/api/use-graph';
import { useAlerts } from '@/hooks/api/use-monitoring';
import { AlertStatus } from '@/types/monitoring';
import { useQuickHealthCheck } from '@/hooks/api/use-health-score';
import { useKpiSummary, useActiveExperimentCount } from '@/hooks/api/use-home-stats';
import { useHomeExecutiveInsights } from '@/hooks/api/use-home-executive-insights';
import { useOpportunities } from '@/hooks/api/use-gaps';
import { getValidated } from '@/lib/api-client';
import { AgentStatusResponseSchema } from '@/lib/api-schemas';
import { Progress } from '@/components/ui/progress';
import {
  Activity,
  Users,
  Target,
  DollarSign,
  BarChart3,
  Brain,
  Zap,
  AlertCircle,
  CheckCircle2,
  Clock,
  ArrowRight,
  Pill,
  MapPin,
  CalendarDays,
  Sparkles,
  RefreshCw,
  ExternalLink,
} from 'lucide-react';
import { cn } from '@/lib/utils';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import {
  KPICard,
} from '@/components/visualizations/dashboard';
import { EmptyState } from '@/components/ui/EmptyState';
import { ExecutiveSummary } from '@/components/dashboard/ExecutiveSummary';
import { CausalValueChains } from '@/components/dashboard/CausalValueChains';
import { getNavigationRoutes } from '@/router/routes';

// =============================================================================
// TYPES
// =============================================================================

type Brand = 'All' | 'Remibrutinib' | 'Fabhalta' | 'Kisqali';
type Region = 'All US' | 'Northeast' | 'Southeast' | 'Midwest' | 'West' | 'Southwest';
type DateRange = 'Q4 2025' | 'Q3 2025' | 'Q2 2025' | 'Q1 2025' | 'YTD 2025' | 'Last 12 Months';

interface KPIMetric {
  id: string;
  name: string;
  category: string;
  value: number;
  previousValue?: number;
  target?: number;
  unit?: string;
  prefix?: string;
  description: string;
  trend: 'up' | 'down' | 'stable';
  status: 'healthy' | 'warning' | 'critical' | 'neutral';
  sparkline?: number[];
}

interface AgentInsight {
  id: string;
  agentName: string;
  agentTier: number;
  type: 'recommendation' | 'alert' | 'opportunity' | 'insight';
  title: string;
  summary: string;
  impact: 'high' | 'medium' | 'low';
  timestamp: string;
  actionable: boolean;
  relatedKPIs: string[];
}

// =============================================================================
// SAMPLE DATA
// =============================================================================

const BRANDS: { value: Brand; label: string; indication: string; color: string }[] = [
  { value: 'All', label: 'All Brands', indication: 'Combined Portfolio', color: 'bg-slate-500' },
  { value: 'Remibrutinib', label: 'Remibrutinib', indication: 'CSU', color: 'bg-blue-500' },
  { value: 'Fabhalta', label: 'Fabhalta', indication: 'PNH', color: 'bg-purple-500' },
  { value: 'Kisqali', label: 'Kisqali', indication: 'HR+/HER2- BC', color: 'bg-rose-500' },
];

const REGIONS: { value: Region; label: string }[] = [
  { value: 'All US', label: 'All US Regions' },
  { value: 'Northeast', label: 'Northeast' },
  { value: 'Southeast', label: 'Southeast' },
  { value: 'Midwest', label: 'Midwest' },
  { value: 'West', label: 'West' },
  { value: 'Southwest', label: 'Southwest' },
];

const DATE_RANGES: { value: DateRange; label: string; description: string }[] = [
  { value: 'Q4 2025', label: 'Q4 2025', description: 'Oct - Dec 2025' },
  { value: 'Q3 2025', label: 'Q3 2025', description: 'Jul - Sep 2025' },
  { value: 'Q2 2025', label: 'Q2 2025', description: 'Apr - Jun 2025' },
  { value: 'Q1 2025', label: 'Q1 2025', description: 'Jan - Mar 2025' },
  { value: 'YTD 2025', label: 'Year to Date', description: 'Jan - Dec 2025' },
  { value: 'Last 12 Months', label: 'Last 12 Months', description: 'Rolling 12 months' },
];

// Demo-mode (SAMPLE_KPIS) categories — only used when the API is offline.
const KPI_CATEGORIES = [
  { id: 'commercial', label: 'Commercial', icon: DollarSign },
  { id: 'hcp', label: 'HCP Engagement', icon: Users },
  { id: 'patient', label: 'Patient Journey', icon: Activity },
  { id: 'market', label: 'Market Share', icon: Target },
  { id: 'causal', label: 'Causal Metrics', icon: Brain },
];

// REAL backend workstreams (src/kpi/models.py Workstream enum, populated from
// config/kpi_definitions.yaml). Live tabs derive from these so every real KPI
// is visible under its true workstream — the old keyword mapper
// (includes('commercial')/...) matched NONE of the real values and dumped all
// live KPIs into 'causal' while other tabs claimed to be empty.
const WORKSTREAM_META: Record<string, { label: string; icon: typeof DollarSign }> = {
  ws3_business: { label: 'Business', icon: DollarSign },
  brand_specific: { label: 'Brand', icon: Pill },
  causal_metrics: { label: 'Causal Metrics', icon: Brain },
  ws2_triggers: { label: 'Triggers', icon: Zap },
  ws1_model_performance: { label: 'Model Performance', icon: Target },
  ws1_data_quality: { label: 'Data Quality', icon: BarChart3 },
};
const WORKSTREAM_ORDER = [
  'ws3_business',
  'brand_specific',
  'causal_metrics',
  'ws2_triggers',
  'ws1_model_performance',
  'ws1_data_quality',
];

const SAMPLE_KPIS: Record<Brand, KPIMetric[]> = {
  All: [
    { id: 'trx_total', name: 'Total TRx', category: 'commercial', value: 125430, previousValue: 118250, target: 130000, description: 'Total prescriptions across all brands', trend: 'up', status: 'healthy', sparkline: [100, 108, 112, 115, 118, 122, 125] },
    { id: 'nrx_total', name: 'New TRx', category: 'commercial', value: 28540, previousValue: 26890, target: 30000, description: 'New prescriptions this period', trend: 'up', status: 'healthy', sparkline: [22, 24, 25, 26, 27, 28, 28.5] },
    { id: 'revenue', name: 'Net Revenue', category: 'commercial', value: 425000000, previousValue: 398000000, target: 450000000, prefix: '$', description: 'Net revenue across portfolio', trend: 'up', status: 'healthy', sparkline: [350, 370, 385, 398, 410, 420, 425] },
    { id: 'market_share', name: 'Market Share', category: 'market', value: 28.5, previousValue: 26.8, target: 32, unit: '%', description: 'Combined market share', trend: 'up', status: 'healthy', sparkline: [24, 25, 26, 26.5, 27, 28, 28.5] },
    { id: 'hcp_reach', name: 'HCP Reach', category: 'hcp', value: 12450, previousValue: 11800, target: 15000, description: 'HCPs engaged this quarter', trend: 'up', status: 'warning', sparkline: [10, 10.5, 11, 11.5, 12, 12.2, 12.4] },
    { id: 'conversion_rate', name: 'Conversion Rate', category: 'hcp', value: 18.5, previousValue: 17.2, target: 22, unit: '%', description: 'HCP to prescription conversion', trend: 'up', status: 'warning', sparkline: [15, 16, 16.5, 17, 17.5, 18, 18.5] },
    { id: 'patient_starts', name: 'Patient Starts', category: 'patient', value: 8920, previousValue: 8340, target: 10000, description: 'New patient starts', trend: 'up', status: 'healthy', sparkline: [7, 7.5, 8, 8.2, 8.5, 8.8, 8.9] },
    { id: 'adherence', name: 'Adherence Rate', category: 'patient', value: 78.5, previousValue: 76.2, target: 85, unit: '%', description: 'Patient medication adherence', trend: 'up', status: 'warning', sparkline: [72, 73, 74, 75, 76, 77, 78.5] },
    { id: 'ate_trx', name: 'ATE on TRx', category: 'causal', value: 0.156, previousValue: 0.142, target: 0.18, description: 'Average treatment effect on prescriptions', trend: 'up', status: 'healthy', sparkline: [0.12, 0.13, 0.14, 0.145, 0.15, 0.155, 0.156] },
    { id: 'roi', name: 'Campaign ROI', category: 'causal', value: 3.8, previousValue: 3.2, target: 4.5, unit: 'x', description: 'Return on marketing investment', trend: 'up', status: 'healthy', sparkline: [2.8, 3, 3.2, 3.4, 3.5, 3.7, 3.8] },
  ],
  Remibrutinib: [
    { id: 'remi_trx', name: 'TRx', category: 'commercial', value: 45230, previousValue: 42150, target: 50000, description: 'Total Remibrutinib prescriptions', trend: 'up', status: 'healthy', sparkline: [38, 40, 41, 42, 43, 44, 45] },
    { id: 'remi_nrx', name: 'New TRx', category: 'commercial', value: 12340, previousValue: 11200, target: 14000, description: 'New Remibrutinib prescriptions', trend: 'up', status: 'healthy', sparkline: [9, 10, 10.5, 11, 11.5, 12, 12.3] },
    { id: 'remi_share', name: 'CSU Market Share', category: 'market', value: 18.2, previousValue: 15.8, target: 25, unit: '%', description: 'Share in CSU market', trend: 'up', status: 'warning', sparkline: [12, 13, 14, 15, 16, 17, 18.2] },
    { id: 'remi_hcp', name: 'Allergists Reached', category: 'hcp', value: 4520, previousValue: 4100, target: 5500, description: 'Allergists engaged', trend: 'up', status: 'warning', sparkline: [3.5, 3.8, 4, 4.1, 4.2, 4.4, 4.5] },
    { id: 'remi_ate', name: 'ATE (HCP Visit)', category: 'causal', value: 0.182, previousValue: 0.165, target: 0.20, description: 'Effect of HCP visits on TRx', trend: 'up', status: 'healthy', sparkline: [0.14, 0.15, 0.16, 0.165, 0.17, 0.18, 0.182] },
  ],
  Fabhalta: [
    { id: 'fab_trx', name: 'TRx', category: 'commercial', value: 28450, previousValue: 26800, target: 32000, description: 'Total Fabhalta prescriptions', trend: 'up', status: 'healthy', sparkline: [24, 25, 26, 26.5, 27, 28, 28.4] },
    { id: 'fab_nrx', name: 'New TRx', category: 'commercial', value: 6890, previousValue: 6200, target: 8000, description: 'New Fabhalta prescriptions', trend: 'up', status: 'warning', sparkline: [5, 5.5, 6, 6.2, 6.4, 6.7, 6.9] },
    { id: 'fab_share', name: 'PNH Market Share', category: 'market', value: 22.5, previousValue: 20.1, target: 28, unit: '%', description: 'Share in PNH market', trend: 'up', status: 'healthy', sparkline: [16, 17, 18, 19, 20, 21, 22.5] },
    { id: 'fab_hcp', name: 'Hematologists Reached', category: 'hcp', value: 2890, previousValue: 2650, target: 3500, description: 'Hematologists engaged', trend: 'up', status: 'warning', sparkline: [2.2, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9] },
    { id: 'fab_ate', name: 'ATE (Speaker Program)', category: 'causal', value: 0.245, previousValue: 0.218, target: 0.28, description: 'Effect of speaker programs', trend: 'up', status: 'healthy', sparkline: [0.18, 0.19, 0.20, 0.21, 0.22, 0.24, 0.245] },
  ],
  Kisqali: [
    { id: 'kis_trx', name: 'TRx', category: 'commercial', value: 51750, previousValue: 49300, target: 55000, description: 'Total Kisqali prescriptions', trend: 'up', status: 'healthy', sparkline: [45, 46, 47, 48, 49, 50, 51.7] },
    { id: 'kis_nrx', name: 'New TRx', category: 'commercial', value: 9310, previousValue: 9490, target: 10000, description: 'New Kisqali prescriptions', trend: 'down', status: 'warning', sparkline: [8.5, 9, 9.5, 9.6, 9.5, 9.4, 9.3] },
    { id: 'kis_share', name: 'CDK4/6 Market Share', category: 'market', value: 38.2, previousValue: 37.5, target: 42, unit: '%', description: 'Share in CDK4/6 market', trend: 'up', status: 'healthy', sparkline: [35, 36, 36.5, 37, 37.5, 38, 38.2] },
    { id: 'kis_hcp', name: 'Oncologists Reached', category: 'hcp', value: 5040, previousValue: 5050, target: 6000, description: 'Oncologists engaged', trend: 'stable', status: 'warning', sparkline: [4.8, 4.9, 5, 5.05, 5.02, 5.04, 5.04] },
    { id: 'kis_ate', name: 'ATE (Digital)', category: 'causal', value: 0.128, previousValue: 0.135, target: 0.16, description: 'Effect of digital campaigns', trend: 'down', status: 'warning', sparkline: [0.14, 0.14, 0.135, 0.13, 0.128, 0.128, 0.128] },
  ],
};

// NOTE: the former SAMPLE_INSIGHTS hardcoded array was removed — the Agent
// Insights tile now renders the REAL dual-source feed (executive insights + gap
// opportunities) with honest loading/empty/error states.

// =============================================================================
// HELPER FUNCTIONS
// =============================================================================

function getInsightIcon(type: AgentInsight['type']) {
  switch (type) {
    case 'opportunity':
      return <Target className="h-4 w-4 text-emerald-500" />;
    case 'alert':
      return <AlertCircle className="h-4 w-4 text-amber-500" />;
    case 'recommendation':
      return <Sparkles className="h-4 w-4 text-blue-500" />;
    case 'insight':
      return <Brain className="h-4 w-4 text-purple-500" />;
    default:
      return <Zap className="h-4 w-4" />;
  }
}

function getImpactBadge(impact: AgentInsight['impact']) {
  const colors = {
    high: 'bg-rose-100 text-rose-700 dark:bg-rose-900/30 dark:text-rose-400',
    medium: 'bg-amber-100 text-amber-700 dark:bg-amber-900/30 dark:text-amber-400',
    low: 'bg-slate-100 text-slate-700 dark:bg-slate-900/30 dark:text-slate-400',
  };
  return <Badge className={cn('text-xs', colors[impact])}>{impact}</Badge>;
}

/** Map a backend KPI status string to the KPICard status enum. */
function mapKpiStatus(
  status: string | undefined
): 'healthy' | 'warning' | 'critical' | 'neutral' {
  switch ((status ?? '').toLowerCase()) {
    case 'good':
    case 'healthy':
      return 'healthy';
    case 'warning':
    case 'degraded':
      return 'warning';
    case 'critical':
    case 'unhealthy':
      return 'critical';
    default:
      return 'neutral';
  }
}

/** Map a 0-1 health dimension score to a status color class. */
function healthScoreClass(score: number): string {
  if (score >= 0.9) return 'text-emerald-600 dark:text-emerald-400';
  if (score >= 0.7) return 'text-amber-600 dark:text-amber-400';
  return 'text-rose-600 dark:text-rose-400';
}

/** Format a large integer with thousands separators (honest empty -> '—'). */
function formatStat(value: number | null | undefined): string {
  if (value === null || value === undefined || Number.isNaN(value)) return '—';
  return Math.round(value).toLocaleString();
}

/** Format an ISO `YYYY-MM-DD` data-coverage date as e.g. "Dec 2025" (dynamic,
 *  never hardcoded). Returns null for a missing/invalid date. */
function formatDataThrough(iso: string | null | undefined): string | null {
  if (!iso) return null;
  const d = new Date(`${iso}T00:00:00`);
  if (Number.isNaN(d.getTime())) return null;
  return d.toLocaleDateString('en-US', { month: 'short', year: 'numeric' });
}

/** Display for a KPI tile: the real value, or an honest "No recent activity —
 *  data through <date>" when the metric is 0/null (no recent data, NOT a
 *  fabrication). The date comes from the backend (`data_through`), so it is
 *  dynamic and advances when fresh data lands. */
function kpiTileDisplay(
  value: number | null | undefined,
  summary: { data_through?: string | null } | undefined,
): { display: string; muted: boolean } {
  const noActivity = !!summary && (value === 0 || value === null || value === undefined);
  if (!noActivity) return { display: formatStat(value), muted: false };
  const through = formatDataThrough(summary?.data_through);
  return {
    display: through ? `No recent activity — data through ${through}` : 'No recent activity',
    muted: true,
  };
}

const AGENT_TIER_META: { tier: number; name: string }[] = [
  { tier: 0, name: 'ML Foundation' },
  { tier: 1, name: 'Orchestration' },
  { tier: 2, name: 'Causal Analytics' },
  { tier: 3, name: 'Monitoring' },
  { tier: 4, name: 'ML Predictions' },
  { tier: 5, name: 'Self-Improvement' },
];

interface QuickStatTileProps {
  label: string;
  icon: React.ReactNode;
  /** Pre-formatted display string ('—' for honest unavailable). */
  display: string;
  loading?: boolean;
  error?: boolean;
  /** Provenance chip for the value's source, omitted (null) when it is real DB
   *  data: 'synthetic' = computed over synthetic-gold rows (demo/review mode);
   *  'sample' = a non-DB fallback. Distinct labels so a reviewer is never misled
   *  into reading synthetic figures as real-world data. */
  provenanceBadge?: 'synthetic' | 'sample' | null;
  /** Render `display` as smaller muted prose (e.g. "No recent activity — data
   *  through Dec 2025") instead of a large stat number. */
  muted?: boolean;
}

function QuickStatTile({
  label,
  icon,
  display,
  loading,
  error,
  provenanceBadge,
  muted,
}: QuickStatTileProps) {
  return (
    <Card>
      <CardContent className="py-4">
        <div className="flex items-center gap-3">
          <div className="p-2 rounded-lg bg-muted/50">{icon}</div>
          <div className="min-w-0">
            <div className="text-xs text-muted-foreground flex items-center gap-1.5">
              {label}
              {provenanceBadge && (
                <Badge variant="outline" className="text-[10px] px-1 py-0">
                  {provenanceBadge === 'synthetic' ? 'synthetic data' : 'sample data'}
                </Badge>
              )}
            </div>
            <div
              className={
                muted ? 'text-sm font-normal text-muted-foreground' : 'text-xl font-semibold truncate'
              }
            >
              {loading ? (
                <span className="text-muted-foreground text-sm">Loading…</span>
              ) : error ? (
                <span className="text-muted-foreground text-sm">Unavailable</span>
              ) : (
                display
              )}
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}

// =============================================================================
// COMPONENT
// =============================================================================

// NOTE: the former hardcoded ACTIVE_ALERTS fallback (a fabricated critical
// "Claims data feed delayed by 4 hours" etc., added pre-API in cdda27e1) was
// removed. Alerts render EXCLUSIVELY from the monitoring API: real alerts,
// honest empty state, or a labeled degraded state on query error.

function Home() {
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const [selectedBrand, setSelectedBrand] = useState<Brand>('All');
  const [selectedRegion, setSelectedRegion] = useState<Region>('All US');
  const [selectedDateRange, setSelectedDateRange] = useState<DateRange>('Q4 2025');
  const [selectedCategory, setSelectedCategory] = useState('commercial');
  const [isRefreshing, setIsRefreshing] = useState(false);
  const [dismissedAlerts, setDismissedAlerts] = useState<string[]>([]);

  // ==========================================================================
  // API HOOKS - Wire up to real backend data
  // ==========================================================================

  // KPI data from API
  const {
    data: kpiListData,
    isLoading: kpisLoading,
    error: kpisError
  } = useKPIList();

  // KPI system health
  const { data: kpiHealthData } = useKPIHealth();

  // Graph statistics for causal metrics
  const { data: graphStatsData } = useGraphStats();

  // System alerts from monitoring — the ONLY alert source (no fabricated
  // fallback). Error state is surfaced as a labeled degraded notice.
  // status=active keeps the "Active Alerts" header honest (acknowledged/
  // resolved alerts are not "active").
  const { data: alertsData, error: alertsError } = useAlerts({
    status: AlertStatus.ACTIVE,
  });

  // QUICK_STATS: real business_metrics rollup (Total TRx (MTD), HCPs Reached).
  const {
    data: kpiSummary,
    isLoading: summaryLoading,
    error: summaryError,
  } = useKpiSummary(selectedBrand);

  // Honest tile display: the real value, or "No recent activity — data through
  // <date>" when 0/null (no recent data, not a fabrication).
  const trxTile = kpiTileDisplay(kpiSummary?.metrics?.trx_volume, kpiSummary);
  const hcpReachTile = kpiTileDisplay(kpiSummary?.metrics?.hcp_reach, kpiSummary);

  // KPI provenance chip: real DB values get no badge; 'synthetic' = demo/review
  // mode (figures over synthetic-gold rows); anything else non-DB = 'sample'.
  const kpiProvenance: 'synthetic' | 'sample' | null =
    !kpiSummary || kpiSummary.data_source === 'database'
      ? null
      : kpiSummary.data_source === 'synthetic'
        ? 'synthetic'
        : 'sample';

  // QUICK_STATS: Active Campaigns = count of running experiments.
  const { data: activeExp, isLoading: activeExpLoading } = useActiveExperimentCount();

  // QUICK_STATS: Model Accuracy = real ROC-AUC (WS1-MP-001, ml_predictions.model_auc).
  const {
    data: rocAucResult,
    isLoading: rocAucLoading,
  } = useKPIValue('WS1-MP-001', selectedBrand !== 'All' ? selectedBrand : undefined);

  // System Health card: real agent-computed aggregate scores.
  const {
    data: health,
    isLoading: healthLoading,
    error: healthError,
  } = useQuickHealthCheck({ refetchInterval: 30000 });

  // Agent Status card: real agent roster.
  const { data: agentStatus, isLoading: agentsLoading } = useQuery({
    queryKey: ['agent-status'],
    queryFn: () => getValidated(AgentStatusResponseSchema, '/agents/status'),
    refetchInterval: 30000,
    retry: false,
  });

  // AI Insights tile — dual source: executive insights + gap opportunities.
  const {
    data: execInsights,
    isLoading: execInsightsLoading,
    error: execInsightsError,
  } = useHomeExecutiveInsights(selectedBrand, { enabled: selectedBrand !== 'All' });
  const {
    data: opportunities,
    isLoading: opportunitiesLoading,
    error: opportunitiesError,
  } = useOpportunities(
    selectedBrand !== 'All' ? { limit: 5 } : { limit: 5 },
    { retry: false }
  );

  // Transform API KPIs to local KPIMetric format. The category IS the real
  // workstream value — no keyword guessing (which silently mis-binned every
  // real KPI), no invented taxonomy.
  const apiKPIs = useMemo((): KPIMetric[] => {
    if (!kpiListData?.kpis) return [];

    return kpiListData.kpis.map((kpi) => ({
      id: kpi.id,
      name: kpi.name,
      category: kpi.workstream ?? 'other',
      value: 0, // Value comes from separate calculation endpoint
      description: kpi.definition || '',
      trend: 'stable' as const,
      status: 'neutral' as const,
      unit: kpi.unit,
    }));
  }, [kpiListData]);

  // Use API metadata when available; fetch the real numeric VALUES via the
  // batch endpoint (POST /api/kpis/batch). View-backed KPIs return a real float;
  // the rest return value:null + error → rendered as an honest "Not yet computed"
  // PER CARD (the real backend state, NOT a fabrication). Fall back to SAMPLE_KPIS
  // ONLY when the API gave no response at all (Demo Mode / offline — the header
  // badge announces it). A SUCCESSFUL response with zero KPIs is real data and
  // renders an honest empty state: the badge must never be green over samples.
  const effectiveKPIs = useMemo(() => {
    if (apiKPIs.length > 0) {
      return apiKPIs;
    }
    if (kpiListData) {
      // Connected but the backend genuinely has no KPI definitions.
      return [];
    }
    if (kpisLoading) {
      // Query in flight — the badge says "Loading...", so no sample values
      // may render beneath it (codex iter-1 HIGH-1).
      return [];
    }
    return SAMPLE_KPIS[selectedBrand];
  }, [apiKPIs, kpiListData, kpisLoading, selectedBrand]);

  // True only when we are rendering live metadata (real ids, real values fetched
  // separately via the batch endpoint below).
  const liveKpiMode = apiKPIs.length > 0;

  // Fetch real KPI values for the live ids via the batch calculation endpoint.
  const {
    mutate: batchCalc,
    data: batchData,
    isError: batchError,
  } = useBatchCalculateKPIs();
  // A failed batch REQUEST is a degraded state (labeled below) — distinct
  // from the backend honestly answering value:null ("Not yet computed").
  const batchFailed = batchError && !batchData;
  const kpiListIds = useMemo(
    () => kpiListData?.kpis?.map((k) => k.id) ?? [],
    [kpiListData]
  );
  useEffect(() => {
    if (kpiListIds.length > 0) {
      batchCalc({
        kpi_ids: kpiListIds,
        use_cache: true,
        context: selectedBrand !== 'All' ? { brand: selectedBrand } : undefined,
      });
    }
    // batchCalc is a stable mutation fn from react-query.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [kpiListIds, selectedBrand]);
  // Map kpi_id → { value, status, error, dataSource } from the batch result.
  // dataSource carries provenance ('synthetic' in demo mode) so the grid can
  // badge synthetic-sourced values rather than passing them off as real.
  const valueByKpiId = useMemo(() => {
    const m = new Map<
      string,
      { value: number | null; status: string; error: string | null; dataSource: string }
    >();
    batchData?.results?.forEach((r) => {
      m.set(r.kpi_id, {
        value: r.value ?? null,
        status: r.status,
        error: r.error ?? null,
        dataSource: r.data_source ?? 'database',
      });
    });
    return m;
  }, [batchData]);

  // Page-level synthetic disclosure: true when ANY KPI surface on this page was
  // computed in E2I_KPI_INCLUDE_SYNTHETIC demo mode — the Home tiles (summary),
  // the Model Accuracy tile (roc-auc), or the KPI grid (batch). Drives the banner
  // so the whole dashboard (incl. the grid) is labelled, never read as real data.
  const isSyntheticKpis =
    kpiSummary?.data_source === 'synthetic' ||
    rocAucResult?.data_source === 'synthetic' ||
    (batchData?.results?.some((r) => r.data_source === 'synthetic') ?? false);

  // Get navigation routes for quick actions
  const navRoutes = getNavigationRoutes().filter((route) => route.path !== '/');

  // Category tabs: in live mode, derived from the REAL workstreams present in
  // the API response (every KPI visible under its true workstream by
  // construction); the fixed demo categories apply only to SAMPLE_KPIS.
  const kpiCategories = useMemo(() => {
    if (!liveKpiMode) return KPI_CATEGORIES;
    const present = new Set(effectiveKPIs.map((k) => k.category));
    const ordered = WORKSTREAM_ORDER.filter((ws) => present.has(ws));
    const extras = [...present]
      .filter((ws) => !WORKSTREAM_ORDER.includes(ws))
      .sort();
    return [...ordered, ...extras].map((ws) => ({
      id: ws,
      label: WORKSTREAM_META[ws]?.label ?? (ws === 'other' ? 'Other' : ws),
      icon: WORKSTREAM_META[ws]?.icon ?? BarChart3,
    }));
  }, [liveKpiMode, effectiveKPIs]);

  // Keep the active tab valid across demo/live category sets.
  const activeCategory = useMemo(() => {
    if (kpiCategories.some((c) => c.id === selectedCategory)) return selectedCategory;
    return kpiCategories[0]?.id ?? selectedCategory;
  }, [kpiCategories, selectedCategory]);

  // Filter KPIs by category and brand (uses API data when available)
  const filteredKPIs = useMemo(() => {
    if (activeCategory === 'all') return effectiveKPIs;
    return effectiveKPIs.filter((kpi) => kpi.category === activeCategory);
  }, [effectiveKPIs, activeCategory]);

  // Calculate summary stats (uses API data when available)
  const summaryStats = useMemo(() => {
    const healthyCount = effectiveKPIs.filter((k) => k.status === 'healthy').length;
    const warningCount = effectiveKPIs.filter((k) => k.status === 'warning').length;
    const criticalCount = effectiveKPIs.filter((k) => k.status === 'critical').length;

    // Enhance with API health data if available
    const apiHealthy = kpiHealthData?.status === 'healthy';

    return {
      total: effectiveKPIs.length,
      healthy: healthyCount,
      warning: warningCount,
      critical: criticalCount,
      apiConnected: !!kpiListData,
      apiHealthy,
    };
  }, [effectiveKPIs, kpiHealthData, kpiListData]);

  // Visible alerts: REAL monitoring alerts only (no fabricated fallback).
  // Stable string ids (the old `Number(a.id) || Math.random()` broke
  // dismissal for non-numeric ids and produced unstable keys).
  const visibleAlerts = useMemo(() => {
    if (!alertsData?.alerts) return [];
    return alertsData.alerts
      .filter((a) => !dismissedAlerts.includes(String(a.id)))
      .map((a) => ({
        id: String(a.id),
        severity: (a.severity === 'critical' || a.severity === 'high'
          ? 'critical'
          : a.severity === 'warning' || a.severity === 'medium'
            ? 'warning'
            : 'info') as 'critical' | 'warning' | 'info',
        title: a.title || a.alert_type || 'Alert',
        message: a.description || '',
        // Honest timestamp: only what the API reports — never an invented
        // 'recently'. null hides the timestamp (and its separator) entirely.
        time: a.triggered_at ? new Date(a.triggered_at).toLocaleString() : null,
      }));
  }, [alertsData, dismissedAlerts]);

  // Derive per-tier agent counts from the real roster (never hardcoded 15/21).
  const agentTierStats = useMemo(() => {
    const agents = agentStatus?.agents ?? [];
    return AGENT_TIER_META.map((t) => {
      const inTier = agents.filter((a) => a.tier === t.tier);
      return {
        ...t,
        total: inTier.length,
        active: inTier.filter((a) => a.status === 'active').length,
      };
    });
  }, [agentStatus]);
  const totalAgents = agentStatus?.agents?.length ?? 0;
  const activeAgents = useMemo(
    () => (agentStatus?.agents ?? []).filter((a) => a.status === 'active').length,
    [agentStatus]
  );

  // Merge the two AI-insight sources into one deduped, ranked list.
  const mergedInsights = useMemo((): AgentInsight[] => {
    const impactRank = { high: 3, medium: 2, low: 1 } as const;
    const items: (AgentInsight & { _rank: number })[] = [];

    // Source A — executive insights.
    (execInsights ?? []).forEach((ins) => {
      const impact: AgentInsight['impact'] =
        ins.effect_size != null && Math.abs(ins.effect_size) >= 0.2
          ? 'high'
          : ins.effect_size != null && Math.abs(ins.effect_size) >= 0.1
            ? 'medium'
            : 'low';
      items.push({
        id: `exec:${ins.insight_id}`,
        agentName: 'Executive Brief',
        agentTier: 5,
        type: 'insight',
        title: ins.title,
        summary: ins.narrative || ins.recommended_next_analysis || '',
        impact,
        timestamp: ins.crystallized_at
          ? new Date(ins.crystallized_at).toLocaleDateString()
          : '',
        actionable: !!ins.recommended_next_analysis,
        relatedKPIs: ins.kpi ? [ins.kpi] : [],
        _rank: impactRank[impact],
      });
    });

    // Source B — gap opportunities.
    (opportunities?.opportunities ?? []).forEach((opp) => {
      const impact: AgentInsight['impact'] =
        opp.implementation_difficulty === 'low'
          ? 'high'
          : opp.implementation_difficulty === 'high'
            ? 'low'
            : 'medium';
      items.push({
        id: `gap:${opp.gap.gap_id}`,
        agentName: 'Gap Analyzer',
        agentTier: 2,
        type: 'opportunity',
        title: `${opp.gap.segment_value} ${opp.gap.metric}`,
        summary: opp.recommended_action,
        impact,
        timestamp: opp.time_to_impact || '',
        actionable: true,
        relatedKPIs: [opp.gap.metric],
        _rank: impactRank[impact],
      });
    });

    // Dedup by composite id, then by (title+summary) defensively.
    const byId = new Map<string, (typeof items)[number]>();
    items.forEach((it) => {
      if (!byId.has(it.id)) byId.set(it.id, it);
    });
    const seen = new Set<string>();
    const deduped = [...byId.values()].filter((it) => {
      const k = `${it.title}|${it.summary}`;
      if (seen.has(k)) return false;
      seen.add(k);
      return true;
    });

    // Order by impact desc, then deterministic tie-break on id for stable keys.
    deduped.sort((a, b) => b._rank - a._rank || a.id.localeCompare(b.id));
    return deduped.map(({ _rank, ...rest }) => {
      void _rank;
      return rest;
    });
  }, [execInsights, opportunities]);

  const insightsLoading = execInsightsLoading || opportunitiesLoading;

  // Dismiss alert handler
  const handleDismissAlert = (id: string) => {
    setDismissedAlerts(prev => [...prev, id]);
  };

  // Refresh all API data
  const handleRefresh = useCallback(async () => {
    setIsRefreshing(true);
    try {
      // Invalidate all KPI and monitoring queries to trigger refetch
      await Promise.all([
        queryClient.invalidateQueries({ queryKey: ['kpi'] }),
        queryClient.invalidateQueries({ queryKey: ['graph'] }),
        queryClient.invalidateQueries({ queryKey: ['monitoring'] }),
      ]);
    } finally {
      // Give a brief visual feedback even if queries complete quickly
      setTimeout(() => setIsRefreshing(false), 500);
    }
  }, [queryClient]);

  // Get brand color
  const selectedBrandInfo = BRANDS.find((b) => b.value === selectedBrand);

  return (
    <div className="space-y-6">
      {/* Synthetic demo-data banner: shown only when the backend reports the
          KPI summary was computed in E2I_KPI_INCLUDE_SYNTHETIC mode, so a
          reviewer is never misled into reading synthetic figures as real-world
          market data. Absent in true production (data_source='database'). */}
      {isSyntheticKpis && (
        <div
          role="status"
          className="rounded-md border border-amber-300 bg-amber-50 px-3 py-2 text-xs text-amber-800 dark:border-amber-700 dark:bg-amber-950/40 dark:text-amber-200"
        >
          Showing <strong>synthetic demo data</strong> — KPI figures are computed on a
          synthetic dataset for review, not real-world market data.
        </div>
      )}
      {/* Header */}
      <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4">
        <div>
          <h1 className="text-3xl font-bold text-[var(--color-foreground)]">
            E2I Executive Dashboard
          </h1>
          <p className="text-[var(--color-muted-foreground)] mt-1">
            Causal Analytics for Commercial Operations
          </p>
          {/* API Connection Status */}
          <div className="flex items-center gap-2 mt-2">
            {kpisLoading ? (
              <Badge variant="secondary" className="text-xs">
                <RefreshCw className="h-3 w-3 mr-1 animate-spin" />
                Loading...
              </Badge>
            ) : kpisError ? (
              <Badge variant="destructive" className="text-xs">
                <AlertCircle className="h-3 w-3 mr-1" />
                API Offline (using sample data)
              </Badge>
            ) : summaryStats.apiConnected ? (
              <Badge variant="outline" className="text-xs border-emerald-500 text-emerald-600">
                <CheckCircle2 className="h-3 w-3 mr-1" />
                API Connected ({kpiListData?.total || 0} KPIs)
              </Badge>
            ) : (
              <Badge variant="outline" className="text-xs">
                <Clock className="h-3 w-3 mr-1" />
                Demo Mode
              </Badge>
            )}
            {graphStatsData && (
              <Badge variant="outline" className="text-xs">
                <Brain className="h-3 w-3 mr-1" />
                {graphStatsData.total_nodes || 0} graph nodes
              </Badge>
            )}
          </div>
        </div>

        <div className="flex flex-wrap items-center gap-3">
          {/* Brand Selector */}
          <Select value={selectedBrand} onValueChange={(v) => setSelectedBrand(v as Brand)}>
            <SelectTrigger className="w-[180px]">
              <div className="flex items-center gap-2">
                <Pill className="h-4 w-4" />
                <SelectValue placeholder="Select Brand" />
              </div>
            </SelectTrigger>
            <SelectContent>
              {BRANDS.map((brand) => (
                <SelectItem key={brand.value} value={brand.value}>
                  <div className="flex items-center gap-2">
                    <div className={cn('w-2 h-2 rounded-full', brand.color)} />
                    <span>{brand.label}</span>
                    <span className="text-xs text-muted-foreground">({brand.indication})</span>
                  </div>
                </SelectItem>
              ))}
            </SelectContent>
          </Select>

          {/* Region Selector */}
          <Select value={selectedRegion} onValueChange={(v) => setSelectedRegion(v as Region)}>
            <SelectTrigger className="w-[160px]">
              <div className="flex items-center gap-2">
                <MapPin className="h-4 w-4" />
                <SelectValue placeholder="Select Region" />
              </div>
            </SelectTrigger>
            <SelectContent>
              {REGIONS.map((region) => (
                <SelectItem key={region.value} value={region.value}>
                  {region.label}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>

          {/* Date Range Selector */}
          <Select value={selectedDateRange} onValueChange={(v) => setSelectedDateRange(v as DateRange)}>
            <SelectTrigger className="w-[160px]">
              <div className="flex items-center gap-2">
                <CalendarDays className="h-4 w-4" />
                <SelectValue placeholder="Select Period" />
              </div>
            </SelectTrigger>
            <SelectContent>
              {DATE_RANGES.map((range) => (
                <SelectItem key={range.value} value={range.value}>
                  <div className="flex flex-col">
                    <span>{range.label}</span>
                    <span className="text-xs text-muted-foreground">{range.description}</span>
                  </div>
                </SelectItem>
              ))}
            </SelectContent>
          </Select>

          {/* Refresh Button */}
          <Button variant="outline" size="icon" onClick={handleRefresh} disabled={isRefreshing}>
            <RefreshCw className={cn('h-4 w-4', isRefreshing && 'animate-spin')} />
          </Button>
        </div>
      </div>

      {/* Executive Intelligence Summary */}
      <ExecutiveSummary />

      {/* Primary Causal Value Chains */}
      <CausalValueChains />

      {/* Quick Stats Bar — REAL data: Total TRx (MTD) + HCPs Reached from the
          business_metrics rollup; Active Campaigns = running experiments; Model
          Accuracy = real ROC-AUC (ml_predictions.model_auc). Honest loading /
          '—' / "sample data" badge where appropriate — never a fabricated value. */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <QuickStatTile
          label="Total TRx (MTD)"
          icon={<Pill className="h-4 w-4 text-blue-500" />}
          loading={summaryLoading}
          error={!!summaryError}
          display={trxTile.display}
          muted={trxTile.muted}
          provenanceBadge={kpiProvenance}
        />
        <QuickStatTile
          label="Active Campaigns"
          icon={<Activity className="h-4 w-4 text-purple-500" />}
          loading={activeExpLoading}
          display={formatStat(activeExp?.active_count)}
        />
        <QuickStatTile
          label="HCPs Reached"
          icon={<Users className="h-4 w-4 text-emerald-500" />}
          loading={summaryLoading}
          error={!!summaryError}
          display={hcpReachTile.display}
          muted={hcpReachTile.muted}
          provenanceBadge={kpiProvenance}
        />
        <QuickStatTile
          label="Model Accuracy"
          icon={<Brain className="h-4 w-4 text-rose-500" />}
          loading={rocAucLoading}
          display={
            rocAucResult?.value != null
              ? `${(rocAucResult.value * 100).toFixed(1)}%`
              : '—'
          }
          provenanceBadge={rocAucResult?.data_source === 'synthetic' ? 'synthetic' : null}
        />
      </div>

      {/* Active Alerts — real monitoring data | honest empty | labeled error */}
      <div className="space-y-2">
        <h3 className="text-sm font-medium text-[var(--color-muted-foreground)] flex items-center gap-2">
          <AlertCircle className="h-4 w-4" />
          Active Alerts ({visibleAlerts.length})
        </h3>
        {alertsError ? (
          <div
            role="alert"
            className="flex items-center gap-3 rounded-lg border border-amber-500/40 bg-amber-500/10 p-3 text-sm text-amber-700 dark:text-amber-400"
          >
            <AlertCircle className="h-4 w-4 flex-shrink-0" aria-hidden="true" />
            <span>
              Alerts unavailable — the monitoring service could not be reached.
              Live alert data cannot be displayed.
            </span>
          </div>
        ) : !alertsData ? (
          /* Query pending / unsettled — no claim either way. */
          <p className="text-sm text-muted-foreground">Checking alerts…</p>
        ) : visibleAlerts.length === 0 ? (
          <EmptyState
            title="No active alerts"
            description="Monitoring is connected and no alerts are currently firing."
            className="p-6"
          />
        ) : (
          <div className="space-y-2">
            {visibleAlerts.map((alert) => (
              <div
                key={alert.id}
                className={cn(
                  'flex items-center justify-between p-3 rounded-lg border',
                  alert.severity === 'critical' && 'bg-rose-50 border-rose-200 dark:bg-rose-900/20 dark:border-rose-800',
                  alert.severity === 'warning' && 'bg-amber-50 border-amber-200 dark:bg-amber-900/20 dark:border-amber-800',
                  alert.severity === 'info' && 'bg-blue-50 border-blue-200 dark:bg-blue-900/20 dark:border-blue-800'
                )}
              >
                <div className="flex items-center gap-3">
                  <div className={cn(
                    'p-1.5 rounded-full',
                    alert.severity === 'critical' && 'bg-rose-500',
                    alert.severity === 'warning' && 'bg-amber-500',
                    alert.severity === 'info' && 'bg-blue-500'
                  )}>
                    <AlertCircle className="h-3 w-3 text-white" />
                  </div>
                  <div>
                    <p className={cn(
                      'font-medium text-sm',
                      alert.severity === 'critical' && 'text-rose-700 dark:text-rose-300',
                      alert.severity === 'warning' && 'text-amber-700 dark:text-amber-300',
                      alert.severity === 'info' && 'text-blue-700 dark:text-blue-300'
                    )}>
                      {alert.title}
                    </p>
                    <p className="text-xs text-[var(--color-muted-foreground)]">
                      {alert.message}
                      {alert.time ? ` • ${alert.time}` : ''}
                    </p>
                  </div>
                </div>
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => handleDismissAlert(alert.id)}
                  className="text-[var(--color-muted-foreground)] hover:text-[var(--color-foreground)]"
                >
                  Dismiss
                </Button>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Brand Context Card */}
      {selectedBrand !== 'All' && (
        <Card className={cn('border-l-4', selectedBrandInfo?.color.replace('bg-', 'border-l-'))}>
          <CardContent className="py-4">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-4">
                <div className={cn('p-3 rounded-lg', selectedBrandInfo?.color, 'bg-opacity-20')}>
                  <Pill className={cn('h-6 w-6', selectedBrandInfo?.color.replace('bg-', 'text-'))} />
                </div>
                <div>
                  <h2 className="text-xl font-semibold">{selectedBrandInfo?.label}</h2>
                  <p className="text-sm text-muted-foreground">{selectedBrandInfo?.indication}</p>
                </div>
              </div>
              <div className="flex items-center gap-6 text-sm">
                <div className="text-center">
                  <div className="font-semibold text-lg">{summaryStats.total}</div>
                  <div className="text-muted-foreground">KPIs</div>
                </div>
                <div className="text-center">
                  <div className="font-semibold text-lg text-emerald-500">{summaryStats.healthy}</div>
                  <div className="text-muted-foreground">On Track</div>
                </div>
                <div className="text-center">
                  <div className="font-semibold text-lg text-amber-500">{summaryStats.warning}</div>
                  <div className="text-muted-foreground">Attention</div>
                </div>
              </div>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Main Content Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* KPI Dashboard - 2 columns */}
        <div className="lg:col-span-2 space-y-4">
          <Card>
            <CardHeader className="pb-2">
              <div className="flex items-center justify-between">
                <div>
                  <CardTitle className="flex items-center gap-2">
                    <BarChart3 className="h-5 w-5" />
                    Key Performance Indicators
                  </CardTitle>
                  <CardDescription>
                    {selectedBrand === 'All' ? 'Portfolio-wide' : selectedBrand} metrics
                  </CardDescription>
                </div>
              </div>
            </CardHeader>
            <CardContent>
              {/* Labeled degraded notice: the batch VALUES request failed —
                  distinct from the backend honestly returning value:null. */}
              {liveKpiMode && batchFailed && (
                <p
                  role="alert"
                  className="mb-3 text-xs text-amber-600 dark:text-amber-400"
                >
                  KPI values unavailable — the batch calculation request
                  failed. Cards show metadata only.
                </p>
              )}
              {/* Category Tabs — live mode: the REAL workstreams present */}
              <Tabs value={activeCategory} onValueChange={setSelectedCategory} className="space-y-4">
                <TabsList className="flex flex-wrap">
                  {kpiCategories.map((cat) => (
                    <TabsTrigger key={cat.id} value={cat.id} className="flex items-center gap-1.5">
                      <cat.icon className="h-3.5 w-3.5" />
                      <span className="hidden sm:inline">{cat.label}</span>
                    </TabsTrigger>
                  ))}
                </TabsList>

                {kpiCategories.map((cat) => (
                  <TabsContent key={cat.id} value={cat.id} className="mt-4">
                    {filteredKPIs.length === 0 ? (
                      kpisLoading ? (
                        <p className="text-sm text-muted-foreground py-4">Loading KPIs…</p>
                      ) : (
                        <EmptyState
                          title="No KPIs available"
                          description="The API returned no KPI definitions for this category yet."
                          className="p-6"
                        />
                      )
                    ) : (
                    <div className="grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-3 gap-4">
                      {filteredKPIs.map((kpi) => {
                        // In live mode, read the REAL value fetched via the batch
                        // endpoint. View-backed KPIs return a float; the rest
                        // return value:null/error → honest "Not yet computed".
                        if (liveKpiMode) {
                          const r = valueByKpiId.get(kpi.id);
                          const hasValue = r != null && r.value != null && !r.error;
                          return (
                            <KPICard
                              key={kpi.id}
                              title={kpi.name}
                              value={
                                hasValue
                                  ? (r!.value as number)
                                  : batchFailed
                                    ? 'Unavailable'
                                    : 'Not yet computed'
                              }
                              unit={hasValue ? kpi.unit : undefined}
                              target={hasValue ? kpi.target : undefined}
                              status={hasValue ? mapKpiStatus(r!.status) : 'neutral'}
                              description={kpi.description}
                              size="sm"
                              onClick={() => navigate('/model-performance')}
                            />
                          );
                        }
                        // Demo Mode (API offline): SAMPLE_KPIS, header badge announces it.
                        return (
                          <KPICard
                            key={kpi.id}
                            title={kpi.name}
                            value={kpi.value}
                            unit={kpi.unit}
                            prefix={kpi.prefix}
                            previousValue={kpi.previousValue}
                            target={kpi.target}
                            sparklineData={kpi.sparkline}
                            status={kpi.status}
                            description={kpi.description}
                            higherIsBetter={kpi.trend !== 'down' || kpi.status === 'healthy'}
                            size="sm"
                            onClick={() => navigate('/model-performance')}
                          />
                        );
                      })}
                    </div>
                    )}
                  </TabsContent>
                ))}
              </Tabs>
            </CardContent>
          </Card>

          {/* Agent Insights Feed */}
          <Card>
            <CardHeader className="pb-2">
              <div className="flex items-center justify-between">
                <div>
                  <CardTitle className="flex items-center gap-2">
                    <Sparkles className="h-5 w-5" />
                    Agent Insights
                  </CardTitle>
                  <CardDescription>Executive insights &amp; gap opportunities</CardDescription>
                </div>
                <Button variant="ghost" size="sm" onClick={() => navigate('/monitoring')}>
                  View All
                  <ArrowRight className="h-4 w-4 ml-1" />
                </Button>
              </div>
            </CardHeader>
            <CardContent>
              {/* Real, dual-source feed. No SAMPLE_INSIGHTS fallback — honest
                  loading / error / empty when the substrate is genuinely empty. */}
              {insightsLoading ? (
                <p className="text-sm text-muted-foreground py-4">Loading insights…</p>
              ) : mergedInsights.length === 0 ? (
                <div className="py-4 space-y-2">
                  <p className="text-sm text-muted-foreground">
                    No insights yet — run a gap analysis or trigger insight crystallization.
                  </p>
                  {opportunitiesError && (
                    <p className="text-xs text-amber-600">Opportunities temporarily unavailable.</p>
                  )}
                  {execInsightsError && selectedBrand !== 'All' && (
                    <p className="text-xs text-amber-600">Executive insights temporarily unavailable.</p>
                  )}
                </div>
              ) : (
                <div className="space-y-3">
                  {mergedInsights.slice(0, 4).map((insight) => (
                    <div
                      key={insight.id}
                      className="flex items-start gap-3 p-3 rounded-lg border bg-[var(--color-card)] hover:bg-muted/50 transition-colors cursor-pointer"
                      onClick={() => navigate('/causal-discovery')}
                    >
                      <div className="mt-0.5">{getInsightIcon(insight.type)}</div>
                      <div className="flex-1 min-w-0">
                        <div className="flex items-center gap-2 mb-1">
                          <span className="font-medium text-sm">{insight.title}</span>
                          {getImpactBadge(insight.impact)}
                        </div>
                        <p className="text-sm text-muted-foreground line-clamp-2">{insight.summary}</p>
                        <div className="flex items-center gap-2 mt-2 text-xs text-muted-foreground">
                          <Badge variant="outline" className="text-xs">
                            Tier {insight.agentTier}: {insight.agentName}
                          </Badge>
                          {insight.timestamp && (
                            <>
                              <span>•</span>
                              <Clock className="h-3 w-3" />
                              <span>{insight.timestamp}</span>
                            </>
                          )}
                        </div>
                      </div>
                      {insight.actionable && (
                        <Button variant="outline" size="sm">
                          Act
                        </Button>
                      )}
                    </div>
                  ))}
                </div>
              )}
            </CardContent>
          </Card>
        </div>

        {/* Right Sidebar - 1 column */}
        <div className="space-y-4">
          {/* System Health */}
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="flex items-center gap-2 text-base">
                <Activity className="h-4 w-4" />
                System Health
              </CardTitle>
            </CardHeader>
            <CardContent>
              {healthLoading ? (
                <p className="text-sm text-muted-foreground py-2">Checking system health…</p>
              ) : healthError || !health ? (
                <p className="text-sm text-muted-foreground py-2">Health check unavailable</p>
              ) : (
                <div className="space-y-2">
                  <div className="flex items-center justify-between">
                    <span className="text-sm">Overall</span>
                    <span className="font-semibold">
                      {Math.round(health.overall_health_score)} ({health.health_grade})
                    </span>
                  </div>
                  {([
                    ['Components', health.component_health_score],
                    ['Models', health.model_health_score],
                    ['Pipelines', health.pipeline_health_score],
                    ['Agents', health.agent_health_score],
                  ] as [string, number][]).map(([label, score]) => (
                    <div key={label} className="flex items-center justify-between text-sm">
                      <span className="text-muted-foreground">{label}</span>
                      <span className={cn('font-medium', healthScoreClass(score))}>
                        {Math.round(score * 100)}%
                      </span>
                    </div>
                  ))}
                  {health.warnings?.some((w) => w.toLowerCase().includes('mock data')) && (
                    <p className="text-xs text-amber-600">Awaiting Health Score agent</p>
                  )}
                </div>
              )}
              <Button
                variant="ghost"
                size="sm"
                className="w-full mt-3"
                onClick={() => navigate('/system-health')}
              >
                View Details
                <ArrowRight className="h-4 w-4 ml-1" />
              </Button>
            </CardContent>
          </Card>

          {/* Agent Tier Summary */}
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="flex items-center gap-2 text-base">
                <Brain className="h-4 w-4" />
                Agent Status
              </CardTitle>
            </CardHeader>
            <CardContent>
              {agentsLoading ? (
                <p className="text-sm text-muted-foreground py-2">Loading agents…</p>
              ) : totalAgents === 0 ? (
                <p className="text-sm text-muted-foreground py-2">Agent roster unavailable</p>
              ) : (
                <div className="space-y-2">
                  {agentTierStats
                    .filter((t) => t.total > 0)
                    .map((t) => (
                      <div key={t.tier} className="space-y-1">
                        <div className="flex items-center justify-between text-sm">
                          <span className="text-muted-foreground">
                            Tier {t.tier}: {t.name}
                          </span>
                          <span className="font-medium">
                            {t.active}/{t.total}
                          </span>
                        </div>
                        <Progress value={t.total > 0 ? (t.active / t.total) * 100 : 0} className="h-1.5" />
                      </div>
                    ))}
                  <p className="text-xs text-muted-foreground pt-1">
                    {activeAgents}/{totalAgents} agents active
                  </p>
                </div>
              )}
              <Button
                variant="ghost"
                size="sm"
                className="w-full mt-3"
                onClick={() => navigate('/agent-orchestration')}
              >
                View Agents
                <ArrowRight className="h-4 w-4 ml-1" />
              </Button>
            </CardContent>
          </Card>

          {/* Quick Actions */}
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="flex items-center gap-2 text-base">
                <Zap className="h-4 w-4" />
                Quick Actions
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-2 gap-2">
                {navRoutes.slice(0, 6).map((route) => (
                  <Link
                    key={route.path}
                    to={route.path}
                    className="flex items-center gap-2 p-2 rounded-lg border hover:bg-muted/50 transition-colors text-sm"
                  >
                    <ExternalLink className="h-3.5 w-3.5 text-muted-foreground" />
                    <span className="truncate">{route.title}</span>
                  </Link>
                ))}
              </div>
            </CardContent>
          </Card>

          {/* Filter Summary */}
          <Card>
            <CardContent className="py-4">
              <div className="flex items-center gap-3">
                <CalendarDays className="h-5 w-5 text-muted-foreground" />
                <div>
                  <div className="text-sm font-medium">Reporting Period</div>
                  <div className="text-xs text-muted-foreground">
                    {DATE_RANGES.find((r) => r.value === selectedDateRange)?.description || selectedDateRange}
                  </div>
                </div>
              </div>
              <div className="flex items-center gap-3 mt-3 pt-3 border-t">
                <MapPin className="h-5 w-5 text-muted-foreground" />
                <div>
                  <div className="text-sm font-medium">Territory</div>
                  <div className="text-xs text-muted-foreground">
                    {REGIONS.find((r) => r.value === selectedRegion)?.label || selectedRegion}
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
}

export default Home;
