/**
 * Segment Analysis Page — Clinical HTE, Agent-Driven
 * ==================================================
 *
 * Heterogeneous treatment effect (HTE) dashboard. The agent estimates CATE
 * across ALL clinical segments of the gold-standard patient_journeys cohort,
 * identifies high / mid / low responder segments, surfaces uplift, and learns
 * a targeting policy with a strategic narrative.
 *
 * The page is AGENT-DRIVEN: the user picks a brand cohort + a curated
 * treatment/outcome pair; the backend FIXES the clinical contract server-side
 * (segment vars / effect modifiers / confounders / substrate) and loads the
 * gold-standard frame. The FE only sends { query, brand?, treatment_var?,
 * outcome_var? }.
 *
 * Features:
 * - CATE estimation across every clinical segment dimension (ordered by spread)
 * - Multi-covariate feature importance for treatment-effect heterogeneity
 * - High / Mid / Low responder identification with per-card drill-down
 * - Strategic targeting policy (who / why / expected lift)
 * - Uplift metrics (honest empty-state when a run genuinely lacks them)
 * - Strategic interpretation + between-segment heterogeneity (I^2)
 *
 * @module pages/SegmentAnalysis
 */

import { useEffect, useMemo, useState } from 'react';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Cell,
  ScatterChart,
  Scatter,
  ReferenceLine,
  ErrorBar,
} from 'recharts';
import { Download } from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog';
import { KPICard } from '@/components/visualizations';
import { QueryErrorState } from '@/components/ui/query-error-state';
import { WarningBanner } from '@/components/ui/WarningBanner';
import { EmptyState } from '@/components/ui/EmptyState';
import { LabelGateBadge } from '@/components/insights/LabelGateBadge';
import { usePageChatContext } from '@/providers/E2ICopilotProvider';
import {
  useSegmentHealth,
  useSegmentDatasets,
  useRunSegmentAnalysisAndWait,
  usePolicies,
  useCausalVariables,
} from '@/hooks/api';
import { useQueryErrorToast, useMutationError } from '@/hooks/use-query-error';
import { SegmentAnalysisTimeoutError } from '@/api/segments';
import type {
  CATEResult,
  SegmentProfile,
  PolicyRecommendation,
  UpliftMetrics,
  CrossLibraryValidationDetail,
} from '@/types/segments';
import { ResponderType } from '@/types/segments';


// =============================================================================
// CONFIG DEFAULTS
// =============================================================================

// "All brands" sentinel — the Select needs a non-empty value; `undefined` is
// sent to the API (cohort spans all brands).
const ALL_BRANDS = '__all__';

/**
 * Poll ceilings (ms) for the durable async run. The backend persists the
 * analysis record and the run completes server-side whether or not the FE is
 * still polling, so these only decide how long the page waits before giving up
 * on a run that may be genuinely stuck (the backend puts no budget on the graph).
 *
 * Measured on prod (2026-08-29/30): the single-brand Fabhalta
 * copay_support -> persistent_180d run completed in 73 s, 125 s, 153 s and
 * 208 s — the hierarchical CausalML uplift forest is ~80% of it and its
 * duration scales with host load. The previous 120 s single-brand cap rested on
 * "~30 s" runs and threw "timed out" on runs that then completed unseen; with
 * the app's mutation retry that timeout even re-submitted the analysis. The
 * all-brands run scans the full cohort (~2.8x the rows), so it gets a
 * proportionally larger ceiling. When it expires the record's analysis_id is
 * kept and the page offers Keep waiting (re-attach by GET, no second POST —
 * #1841); cancelling the server-side run has no endpoint and stays out of scope.
 */
const SINGLE_BRAND_POLL_CEILING_MS = 300_000;
const ALL_BRANDS_POLL_CEILING_MS = 600_000;

// Curated defaults (match the backend's patient_journeys allowlist defaults).
// These are fallbacks only — the real options come from GET /segments/datasets.
const DEFAULT_TREATMENT = 'treatment_arm';
const DEFAULT_OUTCOME = 'persistent_180d';
// The HTE run's dataset is FIXED server-side (segments.py _SEGMENT_HTE_DATASET);
// used here only to fetch the clinical-biomarker union for display grouping.
const SEGMENT_DATASET = 'patient_journeys';

const titleCase = (s: string) =>
  s.replace(/_/g, ' ').replace(/\b\w/g, (l) => l.toUpperCase());

// =============================================================================
// CHART COLORS
// =============================================================================

const COLORS = {
  primary: '#3b82f6',
  secondary: '#8b5cf6',
  success: '#10b981',
  warning: '#f59e0b',
  danger: '#ef4444',
  muted: '#6b7280',
};

// =============================================================================
// HETEROGENEITY ORDERING
// =============================================================================

/**
 * Within-segment heterogeneity = the spread (max-min) of the CATE estimates
 * across the values of a single segment dimension. The dimension whose effect
 * varies the MOST is the most interesting, so we surface it first.
 */
function segmentSpread(results: CATEResult[]): number {
  if (!results || results.length === 0) return 0;
  const estimates = results.map((r) => r.cate_estimate);
  return Math.max(...estimates) - Math.min(...estimates);
}

// =============================================================================
// CATE CHART WITH ERROR BARS
// =============================================================================

interface CATEChartProps {
  cateResults: CATEResult[];
  segmentName: string;
}

function CATEBarChart({ cateResults, segmentName: _segmentName }: CATEChartProps) {
  const chartData = cateResults.map((r) => ({
    name: r.segment_value,
    cate: r.cate_estimate,
    ci_lower: r.cate_estimate - r.cate_ci_lower,
    ci_upper: r.cate_ci_upper - r.cate_estimate,
    significant: r.statistical_significance,
    sample_size: r.sample_size,
  }));

  return (
    <ResponsiveContainer width="100%" height={300}>
      <BarChart data={chartData} margin={{ top: 20, right: 30, left: 20, bottom: 5 }}>
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis dataKey="name" />
        <YAxis label={{ value: 'CATE', angle: -90, position: 'insideLeft' }} />
        <Tooltip
          content={({ payload }) => {
            if (!payload || payload.length === 0) return null;
            const data = payload[0].payload;
            return (
              <div className="bg-background border rounded-lg p-3 shadow-lg">
                <p className="font-medium">{data.name}</p>
                <p className="text-sm">CATE: {data.cate.toFixed(3)}</p>
                <p className="text-sm text-muted-foreground">
                  CI: [{(data.cate - data.ci_lower).toFixed(3)}, {(data.cate + data.ci_upper).toFixed(3)}]
                </p>
                <p className="text-sm text-muted-foreground">n = {data.sample_size}</p>
                <p className={`text-sm ${data.significant ? 'text-green-600' : 'text-yellow-600'}`}>
                  {data.significant ? 'Significant' : 'Not significant'}
                </p>
              </div>
            );
          }}
        />
        <ReferenceLine y={0} stroke={COLORS.muted} strokeDasharray="3 3" />
        <Bar dataKey="cate" name="CATE">
          {chartData.map((entry, index) => (
            <Cell
              key={`cell-${index}`}
              fill={entry.significant ? COLORS.primary : COLORS.muted}
            />
          ))}
          <ErrorBar dataKey="ci_upper" direction="y" stroke={COLORS.secondary} />
          <ErrorBar dataKey="ci_lower" direction="y" stroke={COLORS.secondary} />
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}

// =============================================================================
// FEATURE IMPORTANCE CHART
// =============================================================================

interface FeatureImportanceChartProps {
  importance: Record<string, number>;
  /** Indication-specific biomarker columns (union across brands, from the
   *  causal-variables SSOT). Bars for these are colored apart from the generic
   *  cross-brand confounders; omitted/empty = ungrouped rendering. */
  biomarkers?: ReadonlySet<string>;
}

function FeatureImportanceChart({ importance, biomarkers }: FeatureImportanceChartProps) {
  const chartData = Object.entries(importance)
    .sort(([, a], [, b]) => b - a)
    .map(([feature, value]) => ({
      name: titleCase(feature),
      importance: value * 100,
      isBiomarker: biomarkers?.has(feature) ?? false,
    }));
  const hasBiomarker = chartData.some((d) => d.isBiomarker);

  return (
    <div>
      <ResponsiveContainer width="100%" height={Math.max(280, chartData.length * 36 + 40)}>
        <BarChart data={chartData} layout="vertical" margin={{ top: 5, right: 30, left: 16, bottom: 24 }}>
          <CartesianGrid strokeDasharray="3 3" />
          {/* A bare 'dataMax' domain makes recharts render the exact data max as the
              last tick (e.g. "79.66264674939283"); ceiling to the next multiple of 10
              keeps every tick a round number at its true position. */}
          <XAxis
            type="number"
            domain={[0, (dataMax: number) => Math.ceil(dataMax / 10) * 10]}
            label={{ value: 'Importance (%)', position: 'bottom' }}
          />

          {/* interval={0} forces a tick+label for EVERY feature — recharts otherwise
              auto-skips category ticks under limited height (the "8 bars, 4 labels"
              bug). width=180 + a longer-name tickFormatter stop the left-edge clip. */}
          <YAxis
            type="category"
            dataKey="name"
            width={180}
            interval={0}
            tick={{ fontSize: 12 }}
            tickFormatter={(v: string) => (v.length > 26 ? `${v.slice(0, 24)}…` : v)}
          />
          <Tooltip formatter={(value) => [`${Number(value ?? 0).toFixed(1)}%`, 'Importance']} />
          {/* Generic cross-brand confounders vs the brand's own indication
              biomarkers — the shared modifiers (severity/age/academic) are NOT
              disease-specific "clinical indicators" and must not read as such. */}
          <Bar dataKey="importance" fill={COLORS.secondary}>
            {chartData.map((entry) => (
              <Cell
                key={entry.name}
                fill={entry.isBiomarker ? COLORS.primary : COLORS.secondary}
              />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
      {hasBiomarker && (
        <p className="mt-1 text-xs text-muted-foreground">
          <span style={{ color: COLORS.secondary }}>■</span> Generic confounder (all brands)
          <span className="ml-3" style={{ color: COLORS.primary }}>■</span> Indication-specific
          biomarker
        </p>
      )}
    </div>
  );
}

// =============================================================================
// UPLIFT METRICS VISUALIZATION
// =============================================================================

interface UpliftMetricsChartProps {
  metrics: UpliftMetrics;
}

function UpliftMetricsChart({ metrics }: UpliftMetricsChartProps) {
  const chartData = [
    { name: 'AUUC', value: metrics.overall_auuc * 100, max: 100 },
    { name: 'Qini', value: metrics.overall_qini * 100, max: 100 },
    { name: 'Targeting Efficiency', value: metrics.targeting_efficiency * 100, max: 100 },
  ];

  return (
    <div className="space-y-4">
      {chartData.map((item) => (
        <div key={item.name} className="space-y-1">
          <div className="flex justify-between text-sm">
            <span>{item.name}</span>
            <span className="font-medium">{item.value.toFixed(1)}%</span>
          </div>
          <div className="h-2 bg-muted rounded-full overflow-hidden">
            <div
              className="h-full bg-primary rounded-full transition-all"
              style={{ width: `${Math.max(0, Math.min(100, item.value))}%` }}
            />
          </div>
        </div>
      ))}
      <p className="text-sm text-muted-foreground">
        Model: {metrics.model_type_used.replace(/_/g, ' ')}
      </p>
    </div>
  );
}

// =============================================================================
// POLICY SCATTER CHART
// =============================================================================

interface PolicyChartProps {
  policies: PolicyRecommendation[];
}

function PolicyScatterChart({ policies }: PolicyChartProps) {
  const chartData = policies.map((p) => ({
    name: p.segment,
    current: p.current_treatment_rate * 100,
    recommended: p.recommended_treatment_rate * 100,
    impact: p.expected_incremental_outcome,
    confidence: p.confidence,
    change: (p.recommended_treatment_rate - p.current_treatment_rate) * 100,
  }));

  return (
    <ResponsiveContainer width="100%" height={300}>
      <ScatterChart margin={{ top: 20, right: 30, left: 20, bottom: 20 }}>
        <CartesianGrid strokeDasharray="3 3" />
        {/* Rates are percentages: both axes MUST be numeric with a fixed 0-100
            domain. Without type="number" the XAxis is a category axis, so runs
            where every segment shares one current rate (the common all-maintain
            case) render N duplicate "50" slots and the (0,0)->(100,100)
            no-change diagonal can't resolve at all. */}
        <XAxis
          dataKey="current"
          type="number"
          domain={[0, 100]}
          ticks={[0, 25, 50, 75, 100]}
          name="Current Rate"
          fontSize={12}
          label={{ value: 'Current Treatment Rate (%)', position: 'bottom', offset: 0 }}
        />
        <YAxis
          dataKey="recommended"
          type="number"
          domain={[0, 100]}
          ticks={[0, 25, 50, 75, 100]}
          name="Recommended Rate"
          fontSize={12}
          label={{ value: 'Recommended Rate (%)', angle: -90, position: 'left', offset: 0, style: { textAnchor: 'middle' } }}
        />
        <Tooltip
          content={({ payload }) => {
            if (!payload || payload.length === 0) return null;
            const data = payload[0].payload;
            return (
              <div className="bg-background border rounded-lg p-3 shadow-lg">
                <p className="font-medium">{data.name}</p>
                <p className="text-sm">Current: {data.current.toFixed(0)}%</p>
                <p className="text-sm">Recommended: {data.recommended.toFixed(0)}%</p>
                <p className={`text-sm ${data.impact >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                  Impact: {data.impact >= 0 ? '+' : ''}{data.impact.toFixed(2)} outcomes
                </p>
                <p className="text-sm text-muted-foreground">
                  Confidence: {(data.confidence * 100).toFixed(0)}%
                </p>
              </div>
            );
          }}
        />
        <ReferenceLine x={50} stroke={COLORS.muted} strokeDasharray="3 3" />
        <ReferenceLine y={50} stroke={COLORS.muted} strokeDasharray="3 3" />
        {/* Diagonal line for no-change reference */}
        <ReferenceLine
          segment={[{ x: 0, y: 0 }, { x: 100, y: 100 }]}
          stroke={COLORS.muted}
          strokeDasharray="5 5"
        />
        <Scatter
          data={chartData}
          fill={COLORS.primary}
          shape={(props: unknown) => {
            const { cx, cy, payload } = props as { cx: number; cy: number; payload: { impact: number } };
            return (
              <circle
                cx={cx}
                cy={cy}
                r={8}
                fill={payload.impact >= 0 ? COLORS.success : COLORS.danger}
                stroke={COLORS.primary}
                strokeWidth={2}
              />
            );
          }}
        />
      </ScatterChart>
    </ResponsiveContainer>
  );
}

// =============================================================================
// RESPONDER PROFILE CARD (clickable -> drill-down)
// =============================================================================

type ResponderTier = 'high' | 'mid' | 'low';

const TIER_STYLES: Record<ResponderTier, { border: string; badgeVariant: 'default' | 'secondary' | 'destructive'; label: string }> = {
  high: { border: 'border-green-200 bg-green-50/50 hover:bg-green-50', badgeVariant: 'default', label: 'Above-Average Responder' },
  mid: { border: 'border-amber-200 bg-amber-50/50 hover:bg-amber-50', badgeVariant: 'secondary', label: 'Average Responder' },
  low: { border: 'border-red-200 bg-red-50/50 hover:bg-red-50', badgeVariant: 'destructive', label: 'Harmful Responder' },
};

function tierForProfile(profile: SegmentProfile): ResponderTier {
  if (profile.responder_type === ResponderType.HIGH) return 'high';
  if (profile.responder_type === ResponderType.LOW) return 'low';
  return 'mid';
}

interface ResponderCardProps {
  profile: SegmentProfile;
  onSelect: (profile: SegmentProfile) => void;
}

function ResponderCard({ profile, onSelect }: ResponderCardProps) {
  const tier = tierForProfile(profile);
  const style = TIER_STYLES[tier];

  return (
    <button
      type="button"
      onClick={() => onSelect(profile)}
      className={`w-full text-left p-4 border rounded-lg transition-colors cursor-pointer focus:outline-none focus:ring-2 focus:ring-ring ${style.border}`}
      aria-label={`View details for ${profile.segment_id}`}
    >
      <div className="flex items-center justify-between mb-2">
        <Badge variant={style.badgeVariant}>{style.label}</Badge>
        <span className="text-sm font-medium">
          CATE: {profile.cate_estimate.toFixed(3)}
        </span>
      </div>
      <div className="space-y-2">
        <div className="flex flex-wrap gap-1">
          {profile.defining_features.slice(0, 4).map((feature, idx) => {
            const entry = Object.entries(feature)[0];
            if (!entry) return null;
            const [key, value] = entry;
            return (
              <Badge key={idx} variant="outline" className="text-xs">
                {key}: {String(value)}
              </Badge>
            );
          })}
        </div>
        <div className="text-sm text-muted-foreground">
          Size: {profile.size} ({profile.size_percentage.toFixed(1)}% of total)
        </div>
        <p className="text-xs text-primary mt-1">Click for details &rarr;</p>
      </div>
    </button>
  );
}

// =============================================================================
// RESPONDER COLUMN (one bucket)
// =============================================================================

interface ResponderColumnProps {
  title: string;
  titleClass: string;
  description: string;
  emptyText: string;
  profiles: SegmentProfile[];
  onSelect: (profile: SegmentProfile) => void;
}

function ResponderColumn({ title, titleClass, description, emptyText, profiles, onSelect }: ResponderColumnProps) {
  return (
    <Card>
      <CardHeader>
        <CardTitle className={titleClass}>
          {title} <span className="text-muted-foreground font-normal">({profiles.length})</span>
        </CardTitle>
        <CardDescription>{description}</CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        {profiles.length === 0 ? (
          <p className="text-sm text-muted-foreground italic py-6 text-center">{emptyText}</p>
        ) : (
          profiles.map((profile) => (
            <ResponderCard key={profile.segment_id} profile={profile} onSelect={onSelect} />
          ))
        )}
      </CardContent>
    </Card>
  );
}

// =============================================================================
// RESPONDER DRILL-DOWN DIALOG
// =============================================================================

interface ResponderDetailDialogProps {
  profile: SegmentProfile | null;
  onClose: () => void;
}

function ResponderDetailDialog({ profile, onClose }: ResponderDetailDialogProps) {
  const tier = profile ? tierForProfile(profile) : 'mid';
  const style = TIER_STYLES[tier];

  return (
    <Dialog open={profile !== null} onOpenChange={(open) => { if (!open) onClose(); }}>
      <DialogContent className="max-w-lg">
        {profile && (
          <>
            <DialogHeader>
              <DialogTitle className="flex items-center gap-3">
                <Badge variant={style.badgeVariant}>{style.label}</Badge>
                <span className="font-mono text-sm">{profile.segment_id}</span>
              </DialogTitle>
              <DialogDescription>
                Conditional treatment effect and targeting recommendation for this segment.
              </DialogDescription>
            </DialogHeader>

            <div className="space-y-4">
              {/* CATE + sample */}
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <p className="text-xs text-muted-foreground">CATE</p>
                  <p className="text-lg font-semibold">{profile.cate_estimate.toFixed(3)}</p>
                </div>
                <div>
                  <p className="text-xs text-muted-foreground">Segment size</p>
                  <p className="text-lg font-semibold">
                    {profile.size}{' '}
                    <span className="text-sm font-normal text-muted-foreground">
                      ({profile.size_percentage.toFixed(1)}%)
                    </span>
                  </p>
                </div>
              </div>

              {/* Defining features as key:value badges */}
              <div>
                <p className="text-xs text-muted-foreground mb-2">Defining features</p>
                {profile.defining_features.length === 0 ? (
                  <p className="text-sm text-muted-foreground italic">No defining features reported.</p>
                ) : (
                  <div className="flex flex-wrap gap-1">
                    {profile.defining_features.map((feature, idx) => {
                      const entry = Object.entries(feature)[0];
                      if (!entry) return null;
                      const [key, value] = entry;
                      return (
                        <Badge key={idx} variant="outline" className="text-xs">
                          {key}: {String(value)}
                        </Badge>
                      );
                    })}
                  </div>
                )}
              </div>

              {/* Recommendation */}
              <div>
                <p className="text-xs text-muted-foreground mb-1">Recommendation</p>
                <p className="text-sm whitespace-pre-line">{profile.recommendation}</p>
              </div>
            </div>
          </>
        )}
      </DialogContent>
    </Dialog>
  );
}

// =============================================================================
// MAIN PAGE COMPONENT
// =============================================================================

/**
 * Ordering-agreement row of the Library Validation card. State-driven so the
 * copy never claims "no distinguishable pairs" when pairs existed but the
 * ordering was not scored, and names the chance floor when the ordering fell
 * below it (validation fails on that floor even when the composite passes).
 */
function formatOrderingAgreement(detail: CrossLibraryValidationDetail): string {
  const ordering = detail.ordering_agreement;
  const pairs = detail.n_distinguishable_pairs;
  if (ordering == null) {
    // A reported ordering is the ONLY "testable" signal; the pair count just
    // qualifies it (both come from the same backend branch, but the wire type
    // declares them independently, so never let a missing count erase a
    // reported ordering — codex wave-54 iter-2).
    return pairs != null && pairs > 0
      ? 'Not testable'
      : 'Not testable (no distinguishable pairs)';
  }
  const pct = `${(ordering * 100).toFixed(0)}%`;
  const base =
    pairs != null && pairs > 0
      ? `${pct} of ${pairs} distinguishable pairs`
      : `${pct} of distinguishable pairs (count not reported)`;
  const floor = detail.ordering_floor;
  if (floor != null && ordering < floor) {
    return `${base} (below the ${(floor * 100).toFixed(0)}% chance floor)`;
  }
  return base;
}

export default function SegmentAnalysis() {
  const [activeTab, setActiveTab] = useState('cate');

  // Agent-driven config (data-driven dropdowns). Brand is a cohort FILTER;
  // treatment/outcome are curated allowlist choices. The backend FIXES the
  // clinical contract (segment vars / effect modifiers / confounders / substrate)
  // server-side, so we only ever send { query, brand?, treatment_var?, outcome_var? }.
  const [selectedBrand, setSelectedBrand] = useState<string>(ALL_BRANDS);
  const [selectedTreatment, setSelectedTreatment] = useState<string>(DEFAULT_TREATMENT);
  const [selectedOutcome, setSelectedOutcome] = useState<string>(DEFAULT_OUTCOME);

  // Responder drill-down state.
  const [selectedProfile, setSelectedProfile] = useState<SegmentProfile | null>(null);

  // API hooks
  const { data: healthData, isLoading: healthLoading, error: healthError, refetch: refetchHealth, isRefetching: isRefetchingHealth } = useSegmentHealth();
  // Options are BRAND-SCOPED (causal_paths SSOT pairs): a brand-distinct clinical
  // axis is offered only on its own cohort, and each treatment lists only the
  // outcomes it has a modeled causal edge to. Before this the flat allowlists
  // offered Fabhalta's axis on a Remibrutinib cohort (503 "No usable rows"),
  // listed treatment_initiated in BOTH dropdowns, and let a user pose a pair
  // with no modeled effect whose cross-library check then honestly FAILED.
  const brandArg = selectedBrand === ALL_BRANDS ? undefined : selectedBrand;
  const { data: datasets, isError: datasetsError } = useSegmentDatasets({ brand: brandArg });
  const { data: _policiesData, error: policiesError } = usePolicies({ limit: 10 });
  const onMutationError = useMutationError({ context: 'running segment analysis' });
  // Use the polling variant: async_mode=true returns a PENDING stub first and
  // the result is computed in a background task. This polls the analysis_id to
  // COMPLETED before exposing cate_by_segment.
  const runAnalysis = useRunSegmentAnalysisAndWait({
    onError: (error) => {
      // A poll-ceiling expiry is not a failure: the record is durable and
      // usually still running (or already done). It gets the "Still running"
      // state with Keep waiting below, not a destructive toast (#1841).
      if (error instanceof SegmentAnalysisTimeoutError) return;
      onMutationError(error);
    },
  });
  // The ceiling expired on a still-running record: its analysis_id lives on
  // the typed error until the user re-attaches (Keep waiting) or runs anew.
  const stillRunning =
    runAnalysis.error instanceof SegmentAnalysisTimeoutError ? runAnalysis.error : null;

  // Automatic error toasts for query errors
  useQueryErrorToast(healthError, { context: 'loading segment health' });
  useQueryErrorToast(policiesError, { context: 'loading policies' });

  // Only render real API results. No fabricated fallback. Surface API warnings.
  const analysisResult = runAnalysis.data;
  const apiWarnings = analysisResult?.warnings ?? [];

  // Indication-biomarker union from the causal-variables SSOT (brand-INDEPENDENT,
  // so a post-run brand-dropdown change can never misclassify a bar). Splits the
  // feature-importance bars into generic cross-brand confounders vs the brand's
  // own biomarkers. Graceful fallback: while loading/on error the chart simply
  // renders ungrouped (no fabricated classification).
  const { data: causalVariables } = useCausalVariables(SEGMENT_DATASET);
  const biomarkerSet = useMemo(
    () => new Set(causalVariables?.clinical_biomarkers ?? []),
    [causalVariables]
  );

  // Default-safe responder buckets (mid_responders may be absent on older runs).
  const highResponders = analysisResult?.high_responders ?? [];
  const midResponders = analysisResult?.mid_responders ?? [];
  const lowResponders = analysisResult?.low_responders ?? [];

  // Health status
  const isHealthy =
    healthData?.agent_available &&
    (healthData?.econml_available || healthData?.causalml_available);

  // Dropdown options (fall back to the curated defaults if /datasets is loading).
  const treatmentOptions = useMemo(
    () => (datasets?.treatments?.length ? datasets.treatments : [DEFAULT_TREATMENT]),
    [datasets]
  );
  // Outcomes scoped to the selected treatment when the backend provides the SSOT
  // pairing; the flat list only on the curated fallback (options_source tells).
  const outcomeOptions = useMemo(() => {
    const scoped = datasets?.outcomes_by_treatment?.[selectedTreatment];
    if (scoped?.length) return scoped;
    return datasets?.outcomes?.length ? datasets.outcomes : [DEFAULT_OUTCOME];
  }, [datasets, selectedTreatment]);
  const brandOptions = datasets?.brands ?? [];
  const optionsFellBack = datasets?.options_source === 'curated_fallback';

  // Keep the selection valid as the scope changes (brand -> treatments,
  // treatment -> outcomes): a choice the new options no longer offer is reset to
  // the curated default when offered, else the first option — so the run can
  // never send a stale off-scope column. Only acts once real options exist.
  useEffect(() => {
    if (!datasets) return;
    if (!treatmentOptions.includes(selectedTreatment)) {
      setSelectedTreatment(
        treatmentOptions.includes(DEFAULT_TREATMENT) ? DEFAULT_TREATMENT : treatmentOptions[0]
      );
    }
  }, [datasets, treatmentOptions, selectedTreatment]);
  useEffect(() => {
    if (!datasets) return;
    if (!outcomeOptions.includes(selectedOutcome)) {
      setSelectedOutcome(
        outcomeOptions.includes(DEFAULT_OUTCOME) ? DEFAULT_OUTCOME : outcomeOptions[0]
      );
    }
  }, [datasets, outcomeOptions, selectedOutcome]);
  // Render the backend's human-readable label (e.g. low_gap_180d -> "Low refill gap
  // (≤30d)") when present; otherwise fall back to a title-cased column name. The
  // dropdowns previously always title-cased the raw column, so the curated labels
  // GET /segments/datasets returns never reached the user.
  const labelFor = (col: string) => datasets?.labels?.[col] ?? titleCase(col);

  // Policy Details shows only actionable recommendations. Zero-lift entries are
  // segments the backend's above-ATE significance gate kept at their current
  // rate (rate_change = 0 exactly), so lift > 0 doubles as the significance
  // filter — anything positive already cleared that gate.
  const actionablePolicies = (analysisResult?.policy_recommendations ?? []).filter(
    (p) => p.expected_incremental_outcome > 0,
  );
  const maintainedPolicyCount =
    (analysisResult?.policy_recommendations.length ?? 0) - actionablePolicies.length;

  // Order CATE dimensions by within-segment heterogeneity (most heterogeneous
  // dimension first) so the most interesting breakdown leads.
  const orderedCateSegments = useMemo(() => {
    if (!analysisResult) return [] as Array<[string, CATEResult[]]>;
    return Object.entries(analysisResult.cate_by_segment).sort(
      ([, a], [, b]) => segmentSpread(b) - segmentSpread(a),
    );
  }, [analysisResult]);

  // Handle analysis run. AGENT-DRIVEN: only the cohort filter + curated pair.
  // The backend fixes the clinical contract + substrate server-side.
  const handleRunAnalysis = () => {
    runAnalysis.mutate({
      request: {
        query: `Treatment effect heterogeneity of ${selectedTreatment} on ${selectedOutcome} across clinical segments`,
        brand: brandArg,
        treatment_var: selectedTreatment,
        outcome_var: selectedOutcome,
      },
      // Durable record; see the ceiling constants for the measured basis.
      maxWaitMs: brandArg ? SINGLE_BRAND_POLL_CEILING_MS : ALL_BRANDS_POLL_CEILING_MS,
    });
  };

  // Keep waiting: re-attach to the SAME durable record with GET polling for
  // another ceiling window. Never a second POST — the duplicate would land on
  // the worker whose single heavy-compute slot the original still holds.
  const handleKeepWaiting = () => {
    if (!stillRunning) return;
    runAnalysis.mutate({
      resumeAnalysisId: stillRunning.analysisId,
      maxWaitMs: stillRunning.maxWaitMs,
    });
  };

  // Export — JSON report for CRM integration. No-op when there's no result.
  const handleExport = () => {
    if (!analysisResult) return;
    const mapProfile = (r: SegmentProfile) => ({
      segment_id: r.segment_id,
      responder_type: r.responder_type,
      cate_estimate: r.cate_estimate,
      defining_features: r.defining_features,
      size: r.size,
      size_percentage: r.size_percentage,
      recommendation: r.recommendation,
    });
    const report = {
      generatedAt: new Date().toISOString(),
      brand: brandArg ?? 'All brands',
      treatment: selectedTreatment,
      outcome: selectedOutcome,
      overall_ate: analysisResult.overall_ate,
      heterogeneity_score: analysisResult.heterogeneity_score,
      segment_heterogeneity: analysisResult.segment_heterogeneity,
      n_segments_analyzed: analysisResult.n_segments_analyzed,
      segmentation_method_used: analysisResult.segmentation_method_used,
      expected_total_lift: analysisResult.expected_total_lift,
      high_responders: highResponders.map(mapProfile),
      mid_responders: midResponders.map(mapProfile),
      low_responders: lowResponders.map(mapProfile),
      cate_by_segment: analysisResult.cate_by_segment,
      policy_recommendations: analysisResult.policy_recommendations,
      optimal_allocation_summary: analysisResult.optimal_allocation_summary,
      uplift_metrics: analysisResult.uplift_metrics,
      uplift_by_segment: analysisResult.uplift_by_segment,
      strategic_interpretation: analysisResult.strategic_interpretation,
      executive_summary: analysisResult.executive_summary,
      key_insights: analysisResult.key_insights,
    };
    const blob = new Blob([JSON.stringify(report, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `segment-analysis-${selectedTreatment}-${selectedOutcome}-${new Date().toISOString().split('T')[0]}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  // Publish a compact on-screen data summary so the chat pane can generate
  // opener pills grounded in what this page is showing (usePageChatContext →
  // POST /chat/suggestions page_context).
  const pageChatSummary = useMemo(() => {
    const brandLabel = selectedBrand === ALL_BRANDS ? 'All brands' : selectedBrand;
    const lines: string[] = [
      `Segment Analysis page. Brand filter: ${brandLabel}; treatment: ${selectedTreatment}; outcome: ${selectedOutcome}.`,
    ];
    if (analysisResult) {
      lines.push(
        `Analysis result on screen: overall ATE ${analysisResult.overall_ate?.toFixed(4) ?? 'n/a'}, heterogeneity score ${analysisResult.heterogeneity_score?.toFixed(2) ?? 'n/a'}, ${highResponders.length} high-responder segments.`
      );
      const topHigh = highResponders[0];
      if (topHigh) {
        lines.push(
          `Top high-responder segment: ${topHigh.segment_id} (CATE ${topHigh.cate_estimate.toFixed(4)}, ${topHigh.size_percentage.toFixed(1)}% of cohort).`
        );
      }
      const firstInsight = analysisResult.key_insights?.[0];
      if (firstInsight) lines.push(`Key insight: ${firstInsight}`);
    } else {
      lines.push('No analysis has been run yet on this visit.');
    }
    return lines.join('\n');
  }, [selectedBrand, selectedTreatment, selectedOutcome, analysisResult, highResponders]);
  usePageChatContext(pageChatSummary);

  return (
    <div className="container mx-auto py-6 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold">Segment Analysis</h1>
          <p className="text-muted-foreground">
            Agent estimates CATE across all clinical segments for the selected cohort.
          </p>
        </div>
        <div className="flex items-center gap-4">
          <Badge variant={isHealthy ? 'default' : 'destructive'}>
            {healthLoading ? 'Checking...' : isHealthy ? 'Agents Ready' : 'Agents Unavailable'}
          </Badge>
          {healthData?.analyses_24h !== undefined && (
            <Badge variant="outline">{healthData.analyses_24h} analyses today</Badge>
          )}
          <Button variant="outline" onClick={handleExport} disabled={!analysisResult}>
            <Download className="h-4 w-4 mr-2" />
            Export
          </Button>
        </div>
      </div>

      {/* API-reported warnings — surfaced prominently so users see when the
          backend falls through to mock or degraded mode (F-010). */}
      <WarningBanner messages={apiWarnings} />

      {/* Error States */}
      <QueryErrorState
        error={healthError}
        onRetry={refetchHealth}
        isRetrying={isRefetchingHealth}
        title="Failed to load segment health"
        size="sm"
      />

      {/* KPI Summary — rendered only when a real analysis result is loaded. */}
      {analysisResult && (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-5 gap-4">
          <KPICard
            title="Overall ATE"
            value={analysisResult.overall_ate?.toFixed(3) || 'N/A'}
            description="Average Treatment Effect"
          />
          <KPICard
            title="Heterogeneity"
            value={`${((analysisResult.heterogeneity_score || 0) * 100).toFixed(0)}%`}
            description="Effect variation across segments"
          />
          <KPICard
            title="Above-Average Responders"
            value={highResponders.length.toString()}
            description="segments significantly above the ATE (the targetable set)"
          />
          <KPICard
            title="Expected Lift"
            value={
              analysisResult.expected_lift_pp != null
                ? `+${(analysisResult.expected_lift_pp * 100).toFixed(1)} pp`
                : 'N/A'
            }
            description={
              analysisResult.expected_total_lift != null
                ? `outcome-rate lift, best axis (~${Math.round(analysisResult.expected_total_lift).toLocaleString()} patients)`
                : 'outcome-rate lift from targeting'
            }
          />
          <KPICard
            title="Confidence"
            value={`${(analysisResult.confidence * 100).toFixed(0)}%`}
            description="analysis reliability"
          />
        </div>
      )}

      {/* Configuration Panel — agent-driven. Pick a cohort (brand filter) and a
          curated treatment/outcome pair; the agent fixes the clinical contract
          server-side and estimates CATE across every clinical segment. */}
      <Card>
        <CardHeader>
          <CardTitle>Analysis Configuration</CardTitle>
          <CardDescription>
            Agent estimates CATE across all clinical segments for the selected cohort.
          </CardDescription>
        </CardHeader>
        <CardContent>
          {/* When GET /segments/datasets fails, the dropdowns fall back to a
              single curated default each (and an empty brand list). Surface
              that honestly instead of letting the single defaults masquerade
              as the full, data-driven option set. */}
          {datasetsError && (
            <div
              data-testid="datasets-degraded-notice"
              role="status"
              className="mb-4 rounded-md border border-amber-300 bg-amber-50 px-3 py-2 text-sm text-amber-800 dark:border-amber-800 dark:bg-amber-950/30 dark:text-amber-300"
            >
              Couldn&apos;t load the full analysis options from the segment
              service — showing defaults. Brand, treatment, and outcome choices
              may be incomplete.
            </div>
          )}
          {/* The registry of modeled causal questions was unavailable, so the
              backend returned the flat curated allowlists: treatments are still
              brand-gated, but outcomes are NOT paired to the selected treatment.
              Say so rather than let an unmodeled pair look endorsed. */}
          {!datasetsError && optionsFellBack && (
            <div
              data-testid="datasets-fallback-notice"
              role="status"
              className="mb-4 rounded-md border border-amber-300 bg-amber-50 px-3 py-2 text-sm text-amber-800 dark:border-amber-800 dark:bg-amber-950/30 dark:text-amber-300"
            >
              The causal-question registry is unavailable — showing the full curated
              treatment and outcome lists. Outcomes aren&apos;t paired to the selected
              treatment right now, so some combinations may have no modeled causal
              effect.
            </div>
          )}
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
            <div>
              <label htmlFor="brand-select" className="text-sm font-medium mb-2 block">Brand cohort</label>
              <Select value={selectedBrand} onValueChange={setSelectedBrand} disabled={runAnalysis.isPending}>
                <SelectTrigger id="brand-select" aria-label="Brand">
                  <SelectValue placeholder="All brands" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value={ALL_BRANDS}>All brands</SelectItem>
                  {brandOptions.map((b) => (
                    <SelectItem key={b} value={b}>
                      {b}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
            <div>
              <label htmlFor="treatment-select" className="text-sm font-medium mb-2 block">Treatment</label>
              <Select value={selectedTreatment} onValueChange={setSelectedTreatment} disabled={runAnalysis.isPending}>
                <SelectTrigger id="treatment-select" aria-label="Treatment variable">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {treatmentOptions.map((t) => (
                    <SelectItem key={t} value={t}>
                      {labelFor(t)}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
            <div>
              <label htmlFor="outcome-select" className="text-sm font-medium mb-2 block">Outcome</label>
              <Select value={selectedOutcome} onValueChange={setSelectedOutcome} disabled={runAnalysis.isPending}>
                <SelectTrigger id="outcome-select" aria-label="Outcome variable">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {outcomeOptions.map((o) => (
                    <SelectItem key={o} value={o}>
                      {labelFor(o)}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
            <div className="flex items-end">
              <Button
                onClick={handleRunAnalysis}
                disabled={runAnalysis.isPending}
                className="w-full"
              >
                {runAnalysis.isPending ? 'Analyzing...' : 'Run Analysis'}
              </Button>
            </div>
          </div>
          <p className="text-xs text-muted-foreground mt-3">
            Segmented across the clinical covariate set on the gold-standard
            patient cohort. The agent fixes the segment variables, effect
            modifiers, and confounders server-side.
          </p>
          {/* Poll ceiling expired on a still-running durable record (#1841):
              non-destructive, keeps the analysis_id, offers Keep waiting. */}
          {stillRunning && (
            <div
              role="status"
              data-testid="segment-analysis-still-running"
              className="mt-4 rounded-lg border border-border bg-muted/40 p-4"
            >
              <p className="text-sm font-medium">Still running</p>
              <p className="mt-1 text-sm text-muted-foreground">
                The analysis has not finished after{' '}
                {Math.round(stillRunning.maxWaitMs / 1000)} s, but it keeps computing on
                the server under <span className="font-mono">{stillRunning.analysisId}</span>.
                Keep waiting to re-attach to it; Run Analysis would start a second run.
              </p>
              <Button
                variant="outline"
                size="sm"
                className="mt-3"
                onClick={handleKeepWaiting}
                disabled={runAnalysis.isPending}
              >
                Keep waiting
              </Button>
            </div>
          )}
          {/* Mutation Error Display */}
          <QueryErrorState
            error={stillRunning ? null : runAnalysis.error}
            onRetry={handleRunAnalysis}
            isRetrying={runAnalysis.isPending}
            title="Analysis failed"
            size="sm"
            className="mt-4"
          />
        </CardContent>
      </Card>

      {/* Main Content Tabs — only rendered when API has returned a result. */}
      {!analysisResult ? (
        <EmptyState
          title="No segment analysis available"
          description="Configure the cohort above and click Run Analysis to estimate CATE across clinical segments, identify high/mid/low responders, and surface targeting policies."
        />
      ) : (
      <Tabs value={activeTab} onValueChange={setActiveTab}>
        <TabsList className="grid w-full grid-cols-5">
          <TabsTrigger value="cate">CATE by Segment</TabsTrigger>
          <TabsTrigger value="responders">Responders</TabsTrigger>
          <TabsTrigger value="policies">Policies</TabsTrigger>
          <TabsTrigger value="uplift">Uplift Metrics</TabsTrigger>
          <TabsTrigger value="insights">Insights</TabsTrigger>
        </TabsList>

        {/* CATE Tab */}
        <TabsContent value="cate" className="space-y-4">
          {orderedCateSegments.length === 0 ? (
            <EmptyState
              title="No segment CATE estimates"
              description="The agent did not return any segmented CATE breakdowns for this run."
            />
          ) : (
            orderedCateSegments.map(([segmentName, results]) => (
              <Card key={segmentName}>
                <CardHeader>
                  <CardTitle>CATE by {titleCase(segmentName)}</CardTitle>
                  <CardDescription>
                    Conditional Average Treatment Effect with{' '}
                    {analysisResult.confidence_level != null
                      ? `${(analysisResult.confidence_level * 100).toFixed(0)}%`
                      : '95%'}{' '}
                    confidence intervals
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <CATEBarChart cateResults={results} segmentName={segmentName} />
                </CardContent>
              </Card>
            ))
          )}

          {/* Feature Importance — multi-covariate (clinical) drivers of HTE. */}
          {analysisResult.feature_importance && Object.keys(analysisResult.feature_importance).length > 0 && (
            <Card>
              <CardHeader>
                <CardTitle>Feature Importance for CATE</CardTitle>
                <CardDescription>
                  Which covariates drive treatment-effect heterogeneity — generic confounders
                  shared by every brand vs the brand&apos;s own indication-specific biomarkers
                </CardDescription>
              </CardHeader>
              <CardContent>
                <FeatureImportanceChart
                  importance={analysisResult.feature_importance}
                  biomarkers={biomarkerSet}
                />
              </CardContent>
            </Card>
          )}
        </TabsContent>

        {/* Responders Tab — High / Mid with per-card drill-down. Harmful
            (low) responders are intentionally NOT shown here: adverse-event
            handling has its own dedicated workflow. They remain in the JSON
            export (low_responders) and in Policies-tab DECREASE
            recommendations when they occur. */}
        <TabsContent value="responders" className="space-y-4">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            <ResponderColumn
              title="Above-Average Responders"
              titleClass="text-green-700"
              description="CI lies entirely above the overall ATE — significantly above average (the targetable set)"
              emptyText="No segment is statistically above the average effect — a uniform effect has no targeting opportunity"
              profiles={highResponders}
              onSelect={setSelectedProfile}
            />
            <ResponderColumn
              title="Average Responders"
              titleClass="text-amber-700"
              description="CI overlaps the ATE — statistically indistinguishable from the average effect"
              emptyText="No average-band segments for this run"
              profiles={midResponders}
              onSelect={setSelectedProfile}
            />
          </div>
        </TabsContent>

        {/* Policies Tab — strategic framing + optimal allocation. */}
        <TabsContent value="policies" className="space-y-4">
          {/* Optimal Allocation Summary — prominent, the "what to do". */}
          {(analysisResult.optimal_allocation_summary || analysisResult.expected_total_lift != null) && (
            <Card className="border-primary/40">
              <CardHeader>
                <CardTitle>Optimal Allocation</CardTitle>
                <CardDescription>Recommended targeting strategy and its expected payoff</CardDescription>
              </CardHeader>
              <CardContent>
                {analysisResult.optimal_allocation_summary && (
                  <p className="text-lg whitespace-pre-line">{analysisResult.optimal_allocation_summary}</p>
                )}
                {analysisResult.expected_lift_pp != null && (
                  <div className="mt-4 flex items-center gap-4">
                    <Badge variant="default" className="text-lg px-4 py-1">
                      Expected Lift: +{(analysisResult.expected_lift_pp * 100).toFixed(1)} pp
                      {analysisResult.expected_total_lift != null
                        ? ` (~${Math.round(analysisResult.expected_total_lift).toLocaleString()} patients, best axis)`
                        : ''}
                    </Badge>
                  </div>
                )}
              </CardContent>
            </Card>
          )}

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            <Card>
              <CardHeader>
                <CardTitle>Targeting Recommendations</CardTitle>
                <CardDescription>
                  Current vs recommended treatment rates
                </CardDescription>
              </CardHeader>
              <CardContent>
                {analysisResult.policy_recommendations.length === 0 ? (
                  <p className="text-sm text-muted-foreground italic py-6 text-center">
                    No policy recommendations were produced for this run.
                  </p>
                ) : (
                  <>
                    <PolicyScatterChart policies={analysisResult.policy_recommendations} />
                    <p className="text-sm text-muted-foreground mt-2">
                      Points above the diagonal line = increase targeting; below = decrease
                    </p>
                  </>
                )}
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Policy Details</CardTitle>
                <CardDescription>
                  Segments where changing the targeting rate has a significant expected payoff
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-3">
                  {/* Zero-lift entries are segments the above-ATE significance
                      gate kept at their current rate — no action, so no card.
                      Anything with lift > 0 already passed that gate. */}
                  {actionablePolicies.map((policy, idx) => (
                    <div key={idx} className="p-3 border rounded-lg">
                      <div className="flex items-center justify-between mb-2">
                        <span className="font-medium">{policy.segment}</span>
                        <Badge variant="default">
                          +{policy.expected_incremental_outcome.toFixed(2)} outcomes
                        </Badge>
                      </div>
                      <div className="flex items-center gap-4 text-sm">
                        <span className="text-muted-foreground">
                          Current: {(policy.current_treatment_rate * 100).toFixed(0)}%
                        </span>
                        <span>→</span>
                        <span className="text-green-600">
                          Recommended: {(policy.recommended_treatment_rate * 100).toFixed(0)}%
                        </span>
                      </div>
                      <div className="text-xs text-muted-foreground mt-1">
                        Why: moving targeting from{' '}
                        {(policy.current_treatment_rate * 100).toFixed(0)}% to{' '}
                        {(policy.recommended_treatment_rate * 100).toFixed(0)}% of this segment is
                        expected to add ~{policy.expected_incremental_outcome.toFixed(2)} positive
                        outcomes (patients) · Confidence: {(policy.confidence * 100).toFixed(0)}%
                      </div>
                      <LabelGateBadge
                        label_verdict={policy.label_verdict}
                        off_label={policy.off_label}
                        off_label_reason={policy.off_label_reason}
                        label_evidence_confirmed={policy.label_evidence_confirmed}
                        className="mt-2"
                      />
                    </div>
                  ))}
                  {actionablePolicies.length === 0 &&
                    analysisResult.policy_recommendations.length > 0 && (
                      <p className="text-sm text-muted-foreground italic py-6 text-center">
                        No segment shows a treatment effect significantly above average — all
                        segments stay at their current targeting rate.
                      </p>
                    )}
                  {maintainedPolicyCount > 0 && actionablePolicies.length > 0 && (
                    <p className="text-xs text-muted-foreground italic">
                      {maintainedPolicyCount} other segment
                      {maintainedPolicyCount === 1 ? '' : 's'} showed no statistically
                      significant above-average effect and stay at the current rate (not shown).
                    </p>
                  )}
                </div>
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        {/* Uplift Tab — real metrics or an honest empty state (no fake bars). */}
        <TabsContent value="uplift" className="space-y-4">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            <Card>
              <CardHeader>
                <CardTitle>Uplift Model Performance</CardTitle>
                <CardDescription>
                  How well the model identifies treatment responders
                </CardDescription>
              </CardHeader>
              <CardContent>
                {analysisResult.uplift_metrics ? (
                  <UpliftMetricsChart metrics={analysisResult.uplift_metrics} />
                ) : (
                  <p className="text-sm text-muted-foreground italic py-6 text-center">
                    Uplift metrics unavailable for this run.
                  </p>
                )}
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Library Validation</CardTitle>
                <CardDescription>
                  Cross-validation across causal inference libraries
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  <div className="flex items-center justify-between p-3 bg-muted/50 rounded-lg">
                    <span>Libraries Used</span>
                    <div className="flex gap-2">
                      {analysisResult.libraries_used?.length ? (
                        analysisResult.libraries_used.map((lib) => (
                          <Badge key={lib} variant="outline">
                            {lib}
                          </Badge>
                        ))
                      ) : (
                        <span className="text-sm text-muted-foreground italic">Not reported</span>
                      )}
                    </div>
                  </div>
                  <div className="flex items-center justify-between p-3 bg-muted/50 rounded-lg">
                    <span>Agreement Score</span>
                    {analysisResult.library_agreement_score != null ? (
                      // Colour follows the backend VERDICT, never a re-derived
                      // score threshold: since wave 54 a 0.70 score can still
                      // fail on the ordering chance floor, and a branded badge
                      // beside a red "Failed" would contradict itself.
                      <Badge variant={analysisResult.validation_passed ? 'default' : 'secondary'}>
                        {(analysisResult.library_agreement_score * 100).toFixed(0)}%
                      </Badge>
                    ) : (
                      <Badge variant="outline">Not computed</Badge>
                    )}
                  </div>
                  <div className="flex items-center justify-between p-3 bg-muted/50 rounded-lg">
                    <span>Validation Status</span>
                    {analysisResult.validation_passed != null ? (
                      <Badge variant={analysisResult.validation_passed ? 'default' : 'destructive'}>
                        {analysisResult.validation_passed ? 'Passed' : 'Failed'}
                      </Badge>
                    ) : (
                      <Badge variant="outline">Not run</Badge>
                    )}
                  </div>
                  {/* Wave 54: the components behind the score, so a direction
                      disagreement is distinguishable from an ordering one and
                      the user sees how many pairs the ordering check could
                      actually test (ordering among statistically
                      indistinguishable segments is noise, not evidence). */}
                  {analysisResult.cross_library_validation?.computed && (
                    <>
                      <div className="flex items-center justify-between p-3 bg-muted/50 rounded-lg">
                        <span>Direction agreement</span>
                        <span className="text-sm">
                          {analysisResult.cross_library_validation.sign_agreement != null
                            ? `${(analysisResult.cross_library_validation.sign_agreement * 100).toFixed(0)}% of ${analysisResult.cross_library_validation.n_segments_compared ?? 0} segments`
                            : 'Not reported'}
                        </span>
                      </div>
                      <div className="flex items-center justify-between p-3 bg-muted/50 rounded-lg">
                        <span>Ordering agreement</span>
                        <span className="text-sm">
                          {formatOrderingAgreement(analysisResult.cross_library_validation)}
                        </span>
                      </div>
                      <p className="text-xs text-muted-foreground">
                        Ordering is compared only across segment pairs on the same axis whose
                        CATE confidence intervals do not overlap. Segments with statistically
                        indistinguishable effects have no true ordering to reproduce.
                      </p>
                    </>
                  )}
                  {analysisResult.cross_library_validation &&
                    !analysisResult.cross_library_validation.computed &&
                    analysisResult.cross_library_validation.reason && (
                      <p className="text-xs text-muted-foreground">
                        Not computed: {analysisResult.cross_library_validation.reason}
                      </p>
                    )}
                </div>
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        {/* Insights Tab — strategic interpretation + summary + I^2. */}
        <TabsContent value="insights" className="space-y-4">
          {/* Strategic Interpretation — the 3-tier narrative, leading. */}
          {analysisResult.strategic_interpretation && (
            <Card className="border-primary/40">
              <CardHeader>
                <CardTitle>Strategic Interpretation</CardTitle>
                <CardDescription>
                  Who responds, why, and what it means for targeting
                </CardDescription>
              </CardHeader>
              <CardContent>
                <p className="whitespace-pre-line leading-relaxed">
                  {analysisResult.strategic_interpretation}
                </p>
              </CardContent>
            </Card>
          )}

          {/* Heterogeneity panel — surfaced when the hierarchical analyzer ran. */}
          {(analysisResult.segment_heterogeneity != null ||
            analysisResult.n_segments_analyzed != null ||
            analysisResult.segmentation_method_used) && (
            <Card>
              <CardHeader>
                <CardTitle>Heterogeneity Diagnostics</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-sm">
                  {analysisResult.segment_heterogeneity != null && (
                    <div>
                      <p className="text-muted-foreground">Between-segment heterogeneity (I²)</p>
                      <p className="text-lg font-semibold">
                        {analysisResult.segment_heterogeneity.toFixed(1)}%
                      </p>
                    </div>
                  )}
                  {analysisResult.n_segments_analyzed != null && (
                    <div>
                      <p className="text-muted-foreground">Segments analyzed</p>
                      <p className="text-lg font-semibold">{analysisResult.n_segments_analyzed}</p>
                    </div>
                  )}
                  {analysisResult.segmentation_method_used && (
                    <div>
                      <p className="text-muted-foreground">Segmentation method</p>
                      <p className="text-lg font-semibold">
                        {titleCase(analysisResult.segmentation_method_used)}
                      </p>
                    </div>
                  )}
                </div>
              </CardContent>
            </Card>
          )}

          {analysisResult.executive_summary && (
            <Card>
              <CardHeader>
                <CardTitle>Executive Summary</CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-lg whitespace-pre-line">{analysisResult.executive_summary}</p>
              </CardContent>
            </Card>
          )}

          <Card>
            <CardHeader>
              <CardTitle>Key Insights</CardTitle>
              <CardDescription>
                AI-generated findings from segment analysis
              </CardDescription>
            </CardHeader>
            <CardContent>
              {analysisResult.key_insights.length === 0 ? (
                <p className="text-sm text-muted-foreground italic">No key insights were produced for this run.</p>
              ) : (
                <div className="space-y-3">
                  {analysisResult.key_insights.map((insight, idx) => (
                    <div
                      key={idx}
                      className="flex items-start gap-3 p-3 bg-muted/50 rounded-lg"
                    >
                      <div className="flex-shrink-0 w-6 h-6 rounded-full bg-primary text-primary-foreground flex items-center justify-center text-sm font-medium">
                        {idx + 1}
                      </div>
                      <p>{insight}</p>
                    </div>
                  ))}
                </div>
              )}
            </CardContent>
          </Card>

          {/* Analysis Metadata */}
          <Card>
            <CardHeader>
              <CardTitle>Analysis Metadata</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                <div>
                  <p className="text-muted-foreground">Analysis ID</p>
                  <p className="font-mono">{analysisResult.analysis_id}</p>
                </div>
                <div>
                  <p className="text-muted-foreground">Estimation Time</p>
                  <p>{analysisResult.estimation_latency_ms}ms</p>
                </div>
                <div>
                  <p className="text-muted-foreground">Analysis Time</p>
                  <p>{analysisResult.analysis_latency_ms}ms</p>
                </div>
                <div>
                  <p className="text-muted-foreground">Total Time</p>
                  <p>{analysisResult.total_latency_ms}ms</p>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
      )}

      {/* Responder drill-down dialog (shared across columns). */}
      <ResponderDetailDialog profile={selectedProfile} onClose={() => setSelectedProfile(null)} />
    </div>
  );
}
