/**
 * KPI Dictionary Page
 * ===================
 *
 * Comprehensive reference for all 46 KPIs in the E2I Causal Analytics system.
 * Organized by workstream with definitions, formulas, and threshold information.
 *
 * Features:
 * - Search functionality across all KPIs
 * - Workstream category tabs
 * - Detailed KPI cards with formulas
 * - Threshold visualization (target/warning/critical)
 *
 * @module pages/KPIDictionary
 */

import * as React from 'react';
import { useState, useMemo } from 'react';
import {
  Search,
  BookOpen,
  Calculator,
  Database,
  Clock,
  AlertTriangle,
  CheckCircle,
  XCircle,
  Filter,
  Info,
  Target,
  BarChart3,
  Activity,
  TrendingUp,
  Beaker,
  Users,
} from 'lucide-react';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from '@/components/ui/tooltip';
import { useKPIList, useWorkstreams, useKPIHealth } from '@/hooks/api/use-kpi';
import type { KPIMetadata, KPIThreshold, WorkstreamInfo } from '@/types/kpi';
import { Workstream } from '@/types/kpi';

// =============================================================================
// TYPES
// =============================================================================

interface KPICardDetailedProps {
  kpi: KPIMetadata;
}

// =============================================================================
// CONSTANTS
// =============================================================================

const WORKSTREAM_DISPLAY: Record<string, { name: string; icon: React.ReactNode; color: string }> = {
  ws1_data_quality: {
    name: 'WS1: Data Quality',
    icon: <Database className="h-4 w-4" />,
    color: 'text-blue-500',
  },
  ws1_model_performance: {
    name: 'WS1: Model Performance',
    icon: <BarChart3 className="h-4 w-4" />,
    color: 'text-purple-500',
  },
  ws2_triggers: {
    name: 'WS2: Trigger Performance',
    icon: <Activity className="h-4 w-4" />,
    color: 'text-orange-500',
  },
  ws3_business: {
    name: 'WS3: Business Impact',
    icon: <TrendingUp className="h-4 w-4" />,
    color: 'text-green-500',
  },
  brand_specific: {
    name: 'Brand-Specific',
    icon: <Beaker className="h-4 w-4" />,
    color: 'text-pink-500',
  },
  causal_metrics: {
    name: 'Causal Metrics',
    icon: <Calculator className="h-4 w-4" />,
    color: 'text-indigo-500',
  },
};

// Sample KPIs when API is not available
const SAMPLE_KPIS: KPIMetadata[] = [
  // WS1: Data Quality
  {
    id: 'WS1-DQ-001',
    name: 'Source Coverage - Patients',
    definition: 'Percentage of eligible patients present in source vs reference universe',
    formula: 'covered_patients / reference_patients',
    calculation_type: 'direct',
    workstream: 'ws1_data_quality',
    tables: ['patient_journeys', 'reference_universe'],
    columns: ['patient_id', 'coverage_status'],
    threshold: { target: 85, warning: 70, critical: 50 },
    unit: '%',
    frequency: 'daily',
    primary_causal_library: 'none',
  },
  {
    id: 'WS1-DQ-002',
    name: 'Source Coverage - HCPs',
    definition: 'Percentage of priority HCPs present in source vs universe',
    formula: 'covered_hcps / reference_hcps',
    calculation_type: 'direct',
    workstream: 'ws1_data_quality',
    tables: ['hcp_profiles', 'reference_universe'],
    columns: ['hcp_id', 'coverage_status'],
    threshold: { target: 80, warning: 65, critical: 45 },
    unit: '%',
    frequency: 'daily',
    primary_causal_library: 'none',
  },
  {
    id: 'WS1-DQ-003',
    name: 'Cross-Source Match Rate',
    definition: 'Percentage of entities linkable across data sources',
    formula: 'records_matched / total_records',
    calculation_type: 'direct',
    workstream: 'ws1_data_quality',
    tables: ['data_source_tracking'],
    columns: ['match_rate_vs_claims', 'match_rate_vs_ehr'],
    view: 'v_kpi_cross_source_match',
    threshold: { target: 75, warning: 60, critical: 40 },
    unit: '%',
    frequency: 'daily',
    primary_causal_library: 'none',
    note: 'NEW in V3',
  },
  {
    id: 'WS1-DQ-004',
    name: 'Stacking Lift',
    definition: 'Incremental value from combining multiple data sources',
    formula: '(stacked_value - baseline) / baseline',
    calculation_type: 'derived',
    workstream: 'ws1_data_quality',
    tables: ['data_source_tracking'],
    columns: ['stacking_lift_percentage'],
    view: 'v_kpi_stacking_lift',
    threshold: { target: 15, warning: 10, critical: 5 },
    unit: '%',
    frequency: 'daily',
    primary_causal_library: 'none',
    note: 'NEW in V3',
  },
  {
    id: 'WS1-DQ-005',
    name: 'Completeness Pass Rate',
    definition: 'Percentage of records with complete critical fields',
    formula: '1 - (null_critical / total_records)',
    calculation_type: 'direct',
    workstream: 'ws1_data_quality',
    tables: ['patient_journeys'],
    columns: [],
    threshold: { target: 95, warning: 90, critical: 80 },
    unit: '%',
    frequency: 'daily',
    primary_causal_library: 'none',
  },
  // WS1: Model Performance
  {
    id: 'WS1-MP-001',
    name: 'ROC-AUC',
    definition: 'Area Under the ROC Curve',
    formula: '∫TPR d(FPR)',
    calculation_type: 'direct',
    workstream: 'ws1_model_performance',
    tables: ['ml_predictions'],
    columns: ['model_auc'],
    threshold: { target: 0.80, warning: 0.70, critical: 0.60 },
    frequency: 'daily',
    primary_causal_library: 'none',
  },
  {
    id: 'WS1-MP-002',
    name: 'PR-AUC',
    definition: 'Area under the Precision-Recall curve',
    formula: '∫Precision d(Recall)',
    calculation_type: 'direct',
    workstream: 'ws1_model_performance',
    tables: ['ml_predictions'],
    columns: ['model_pr_auc'],
    threshold: { target: 0.70, warning: 0.55, critical: 0.40 },
    frequency: 'daily',
    primary_causal_library: 'none',
    note: 'NEW in V3',
  },
  {
    id: 'WS1-MP-003',
    name: 'F1 Score',
    definition: 'Harmonic mean of precision and recall',
    formula: '2 * (precision * recall) / (precision + recall)',
    calculation_type: 'derived',
    workstream: 'ws1_model_performance',
    tables: ['ml_predictions'],
    columns: ['model_precision', 'model_recall'],
    threshold: { target: 0.75, warning: 0.60, critical: 0.45 },
    frequency: 'daily',
    primary_causal_library: 'none',
  },
  {
    id: 'WS1-MP-004',
    name: 'Recall@Top-K',
    definition: 'Recall achieved when selecting top K predictions',
    formula: 'TP_at_K / total_positives',
    calculation_type: 'direct',
    workstream: 'ws1_model_performance',
    tables: ['ml_predictions'],
    columns: ['rank_metrics'],
    threshold: { target: 0.60, warning: 0.45, critical: 0.30 },
    frequency: 'daily',
    primary_causal_library: 'none',
    note: 'NEW in V3',
  },
  // WS2: Triggers
  {
    id: 'WS2-TR-001',
    name: 'Trigger Precision',
    definition: 'Percentage of fired triggers resulting in positive outcome',
    formula: 'true_positives / (true_positives + false_positives)',
    calculation_type: 'direct',
    workstream: 'ws2_triggers',
    tables: ['triggers'],
    columns: ['trigger_status'],
    threshold: { target: 70, warning: 55, critical: 40 },
    unit: '%',
    frequency: 'daily',
    primary_causal_library: 'dowhy',
  },
  {
    id: 'WS2-TR-002',
    name: 'Trigger Recall',
    definition: 'Percentage of positive outcomes preceded by a trigger',
    formula: 'true_positives / (true_positives + false_negatives)',
    calculation_type: 'direct',
    workstream: 'ws2_triggers',
    tables: ['triggers', 'treatment_events'],
    columns: [],
    threshold: { target: 60, warning: 45, critical: 30 },
    unit: '%',
    frequency: 'daily',
    primary_causal_library: 'dowhy',
  },
  {
    id: 'WS2-TR-003',
    name: 'Action Rate Uplift',
    definition: 'Incremental action rate vs control group',
    formula: '(action_rate_treatment - action_rate_control) / action_rate_control',
    calculation_type: 'derived',
    workstream: 'ws2_triggers',
    tables: ['triggers'],
    columns: [],
    threshold: { target: 15, warning: 10, critical: 5 },
    unit: '%',
    frequency: 'weekly',
    primary_causal_library: 'econml',
  },
  // WS3: Business
  {
    id: 'WS3-BI-001',
    name: 'Monthly Active Users (MAU)',
    definition: 'Unique users with at least one session in past 30 days',
    formula: 'count(distinct user_id) where session_start >= now() - 30 days',
    calculation_type: 'direct',
    workstream: 'ws3_business',
    tables: ['user_sessions'],
    columns: ['user_id', 'session_start'],
    view: 'v_kpi_active_users',
    threshold: { target: 2000, warning: 1500, critical: 1000 },
    frequency: 'daily',
    primary_causal_library: 'none',
    note: 'NEW in V3',
  },
  {
    id: 'WS3-BI-005',
    name: 'Total Prescriptions (TRx)',
    definition: 'Total prescription volume',
    formula: "count(event_type = 'prescription')",
    calculation_type: 'direct',
    workstream: 'ws3_business',
    tables: ['treatment_events'],
    columns: ['event_type'],
    frequency: 'daily',
    primary_causal_library: 'dowhy',
  },
  {
    id: 'WS3-BI-010',
    name: 'Return on Investment (ROI)',
    definition: 'Value generated per dollar invested',
    formula: 'value_captured / cost_invested',
    calculation_type: 'derived',
    workstream: 'ws3_business',
    tables: ['business_metrics', 'agent_activities'],
    columns: ['roi', 'roi_estimate'],
    threshold: { target: 3.0, warning: 2.0, critical: 1.0 },
    unit: 'x',
    frequency: 'monthly',
    primary_causal_library: 'econml',
  },
  // Brand Specific
  {
    id: 'BR-001',
    name: 'Remi - AH Uncontrolled %',
    definition: 'Percentage of antihistamine patients with uncontrolled symptoms',
    formula: 'uncontrolled_patients / ah_patients',
    calculation_type: 'direct',
    workstream: 'brand_specific',
    tables: ['patient_journeys', 'treatment_events'],
    columns: [],
    threshold: { target: 40, warning: 50, critical: 60 },
    unit: '%',
    frequency: 'weekly',
    primary_causal_library: 'none',
    brand: 'remibrutinib',
  },
  {
    id: 'BR-003',
    name: 'Fabhalta - % PNH Tested',
    definition: 'Percentage of eligible patients tested for PNH',
    formula: 'pnh_tested / eligible_patients',
    calculation_type: 'direct',
    workstream: 'brand_specific',
    tables: ['treatment_events'],
    columns: [],
    threshold: { target: 60, warning: 45, critical: 30 },
    unit: '%',
    frequency: 'weekly',
    primary_causal_library: 'none',
    brand: 'fabhalta',
  },
  // Causal Metrics
  {
    id: 'CM-001',
    name: 'Average Treatment Effect (ATE)',
    definition: 'Average causal effect of treatment on outcome',
    formula: 'E[Y(1) - Y(0)]',
    calculation_type: 'derived',
    workstream: 'causal_metrics',
    tables: ['ml_predictions'],
    columns: ['treatment_effect_estimate'],
    frequency: 'weekly',
    primary_causal_library: 'dowhy',
  },
  {
    id: 'CM-002',
    name: 'Conditional ATE (CATE)',
    definition: 'Treatment effect conditioned on segment',
    formula: 'E[Y(1) - Y(0) | X=x]',
    calculation_type: 'derived',
    workstream: 'causal_metrics',
    tables: ['ml_predictions'],
    columns: ['heterogeneous_effect', 'segment_assignment'],
    frequency: 'weekly',
    primary_causal_library: 'econml',
  },
  {
    id: 'CM-003',
    name: 'Causal Impact',
    definition: 'Estimated causal effect size on causal paths',
    formula: 'Computed by causal_impact agent using DoWhy/EconML',
    calculation_type: 'derived',
    workstream: 'causal_metrics',
    tables: ['causal_paths'],
    columns: ['causal_effect_size', 'confidence_level'],
    frequency: 'on-demand',
    primary_causal_library: 'dowhy',
  },
];

// =============================================================================
// SUBCOMPONENTS
// =============================================================================

/**
 * Threshold indicator component
 */
function ThresholdIndicator({ threshold }: { threshold?: KPIThreshold }) {
  if (!threshold) return null;

  return (
    <div className="flex items-center gap-4 text-xs">
      {threshold.target !== undefined && (
        <div className="flex items-center gap-1">
          <Target className="h-3 w-3 text-emerald-500" />
          <span className="text-[var(--color-muted-foreground)]">Target: </span>
          <span className="font-medium text-emerald-600">{threshold.target}</span>
        </div>
      )}
      {threshold.warning !== undefined && (
        <div className="flex items-center gap-1">
          <AlertTriangle className="h-3 w-3 text-amber-500" />
          <span className="text-[var(--color-muted-foreground)]">Warning: </span>
          <span className="font-medium text-amber-600">{threshold.warning}</span>
        </div>
      )}
      {threshold.critical !== undefined && (
        <div className="flex items-center gap-1">
          <XCircle className="h-3 w-3 text-rose-500" />
          <span className="text-[var(--color-muted-foreground)]">Critical: </span>
          <span className="font-medium text-rose-600">{threshold.critical}</span>
        </div>
      )}
    </div>
  );
}

/**
 * Detailed KPI Card component
 */
function KPICardDetailed({ kpi }: KPICardDetailedProps) {
  const workstreamInfo = WORKSTREAM_DISPLAY[kpi.workstream] || {
    name: kpi.workstream,
    icon: <Database className="h-4 w-4" />,
    color: 'text-gray-500',
  };

  return (
    <div className="bg-[var(--color-card)] rounded-lg border border-[var(--color-border)] p-5 hover:shadow-md transition-shadow">
      {/* Header */}
      <div className="flex items-start justify-between gap-3 mb-3">
        <div className="flex-1">
          <div className="flex items-center gap-2 mb-1">
            <span className="text-xs font-mono bg-[var(--color-muted)] px-2 py-0.5 rounded">
              {kpi.id}
            </span>
            {kpi.note && (
              <span className="text-xs font-medium text-indigo-600 bg-indigo-100 px-2 py-0.5 rounded">
                {kpi.note}
              </span>
            )}
            {kpi.brand && (
              <span className="text-xs font-medium text-pink-600 bg-pink-100 px-2 py-0.5 rounded capitalize">
                {kpi.brand}
              </span>
            )}
          </div>
          <h3 className="font-semibold text-[var(--color-foreground)]">{kpi.name}</h3>
        </div>
        <div className={`flex items-center gap-1 ${workstreamInfo.color}`}>
          {workstreamInfo.icon}
        </div>
      </div>

      {/* Definition */}
      <p className="text-sm text-[var(--color-muted-foreground)] mb-3">{kpi.definition}</p>

      {/* Formula */}
      <div className="bg-[var(--color-muted)] rounded-md p-3 mb-3">
        <div className="flex items-center gap-2 mb-1">
          <Calculator className="h-3.5 w-3.5 text-[var(--color-muted-foreground)]" />
          <span className="text-xs font-medium text-[var(--color-muted-foreground)]">Formula</span>
        </div>
        <code className="text-sm font-mono text-[var(--color-foreground)]">{kpi.formula}</code>
      </div>

      {/* Thresholds */}
      {kpi.threshold && (
        <div className="mb-3">
          <ThresholdIndicator threshold={kpi.threshold} />
        </div>
      )}

      {/* Metadata grid */}
      <div className="grid grid-cols-2 gap-3 text-xs">
        {/* Tables */}
        <div className="flex items-start gap-1">
          <Database className="h-3.5 w-3.5 text-[var(--color-muted-foreground)] mt-0.5 flex-shrink-0" />
          <div>
            <span className="text-[var(--color-muted-foreground)]">Tables: </span>
            <span className="text-[var(--color-foreground)]">
              {kpi.tables.length > 0 ? kpi.tables.join(', ') : 'N/A'}
            </span>
          </div>
        </div>

        {/* Frequency */}
        <div className="flex items-center gap-1">
          <Clock className="h-3.5 w-3.5 text-[var(--color-muted-foreground)] flex-shrink-0" />
          <span className="text-[var(--color-muted-foreground)]">Frequency: </span>
          <span className="text-[var(--color-foreground)] capitalize">{kpi.frequency}</span>
        </div>

        {/* Calculation type */}
        <div className="flex items-center gap-1">
          <Activity className="h-3.5 w-3.5 text-[var(--color-muted-foreground)] flex-shrink-0" />
          <span className="text-[var(--color-muted-foreground)]">Type: </span>
          <span className="text-[var(--color-foreground)] capitalize">{kpi.calculation_type}</span>
        </div>

        {/* Causal Library */}
        <div className="flex items-center gap-1">
          <Beaker className="h-3.5 w-3.5 text-[var(--color-muted-foreground)] flex-shrink-0" />
          <span className="text-[var(--color-muted-foreground)]">Library: </span>
          <span className="text-[var(--color-foreground)] capitalize">{kpi.primary_causal_library}</span>
        </div>
      </div>

      {/* View if exists */}
      {kpi.view && (
        <div className="mt-2 flex items-center gap-1 text-xs">
          <Info className="h-3.5 w-3.5 text-indigo-500" />
          <span className="text-[var(--color-muted-foreground)]">View: </span>
          <code className="font-mono text-indigo-600">{kpi.view}</code>
        </div>
      )}
    </div>
  );
}

/**
 * Stats card component
 */
function StatCard({
  title,
  value,
  icon,
  description,
  variant = 'default',
}: {
  title: string;
  value: string | number;
  icon: React.ReactNode;
  description?: string;
  variant?: 'default' | 'success' | 'warning' | 'error';
}) {
  const variantStyles = {
    default: 'text-[var(--color-foreground)]',
    success: 'text-emerald-600',
    warning: 'text-amber-600',
    error: 'text-rose-600',
  };

  return (
    <div className="bg-[var(--color-card)] rounded-lg border border-[var(--color-border)] p-4">
      <div className="flex items-center gap-2 text-[var(--color-muted-foreground)] mb-2">
        {icon}
        <span className="text-sm font-medium">{title}</span>
      </div>
      <div className={`text-2xl font-bold ${variantStyles[variant]}`}>{value}</div>
      {description && (
        <p className="text-xs text-[var(--color-muted-foreground)] mt-1">{description}</p>
      )}
    </div>
  );
}

// =============================================================================
// MAIN COMPONENT
// =============================================================================

export default function KPIDictionary() {
  // State
  const [searchQuery, setSearchQuery] = useState('');
  const [activeWorkstream, setActiveWorkstream] = useState<string>('all');

  // Data fetching
  const { data: kpiData, isLoading: isLoadingKPIs } = useKPIList();
  const { data: workstreamsData } = useWorkstreams();
  const { data: health } = useKPIHealth();

  // Use sample data if API not available
  const kpis = kpiData?.kpis ?? SAMPLE_KPIS;
  const workstreams = workstreamsData?.workstreams ?? Object.keys(WORKSTREAM_DISPLAY).map((id) => ({
    id,
    name: WORKSTREAM_DISPLAY[id]?.name ?? id,
    kpi_count: SAMPLE_KPIS.filter((k) => k.workstream === id).length,
  }));

  // Filter KPIs based on search and workstream
  const filteredKPIs = useMemo(() => {
    let filtered = kpis;

    // Filter by workstream
    if (activeWorkstream !== 'all') {
      filtered = filtered.filter((kpi) => kpi.workstream === activeWorkstream);
    }

    // Filter by search query
    if (searchQuery) {
      const query = searchQuery.toLowerCase();
      filtered = filtered.filter(
        (kpi) =>
          kpi.id.toLowerCase().includes(query) ||
          kpi.name.toLowerCase().includes(query) ||
          kpi.definition.toLowerCase().includes(query) ||
          kpi.formula.toLowerCase().includes(query)
      );
    }

    return filtered;
  }, [kpis, activeWorkstream, searchQuery]);

  // Calculate statistics
  const stats = useMemo(() => {
    const byWorkstream: Record<string, number> = {};
    const byLibrary: Record<string, number> = {};

    kpis.forEach((kpi) => {
      byWorkstream[kpi.workstream] = (byWorkstream[kpi.workstream] || 0) + 1;
      byLibrary[kpi.primary_causal_library] = (byLibrary[kpi.primary_causal_library] || 0) + 1;
    });

    return {
      total: kpis.length,
      byWorkstream,
      byLibrary,
      causalEnabled: kpis.filter((k) => k.primary_causal_library !== 'none').length,
    };
  }, [kpis]);

  return (
    <div className="p-6 max-w-7xl mx-auto">
      {/* Page Header */}
      <div className="mb-6">
        <div className="flex items-center gap-3 mb-2">
          <div className="p-2 bg-primary/10 rounded-lg">
            <BookOpen className="h-6 w-6 text-primary" />
          </div>
          <div>
            <h1 className="text-2xl font-bold text-[var(--color-foreground)]">KPI Dictionary</h1>
            <p className="text-[var(--color-muted-foreground)]">
              Complete reference of all {stats.total} KPIs with definitions, formulas, and thresholds
            </p>
          </div>
        </div>
      </div>

      {/* Stats Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
        <StatCard
          title="Total KPIs"
          value={stats.total}
          icon={<BarChart3 className="h-4 w-4" />}
          description="Across all workstreams"
        />
        <StatCard
          title="Workstreams"
          value={workstreams.length}
          icon={<Database className="h-4 w-4" />}
          description="Category groups"
        />
        <StatCard
          title="Causal KPIs"
          value={stats.causalEnabled}
          icon={<Calculator className="h-4 w-4" />}
          description="Using DoWhy/EconML"
          variant="success"
        />
        <StatCard
          title="System Status"
          value={health?.status === 'healthy' ? 'Healthy' : health?.status ?? 'Unknown'}
          icon={<Activity className="h-4 w-4" />}
          description={`Registry: ${health?.registry_loaded ? 'Loaded' : 'Unknown'}`}
          variant={health?.status === 'healthy' ? 'success' : health?.status === 'degraded' ? 'warning' : 'default'}
        />
      </div>

      {/* Search and Filter */}
      <div className="flex flex-col sm:flex-row gap-4 mb-6">
        <div className="relative flex-1">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-[var(--color-muted-foreground)]" />
          <input
            type="text"
            placeholder="Search KPIs by ID, name, definition, or formula..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="w-full pl-10 pr-4 py-2 bg-[var(--color-background)] border border-[var(--color-border)] rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-primary"
          />
        </div>
        <div className="flex items-center gap-2 text-sm text-[var(--color-muted-foreground)]">
          <Filter className="h-4 w-4" />
          <span>Showing {filteredKPIs.length} of {stats.total} KPIs</span>
        </div>
      </div>

      {/* Workstream Tabs */}
      <Tabs value={activeWorkstream} onValueChange={setActiveWorkstream} className="w-full">
        <TabsList className="w-full flex-wrap h-auto gap-1 p-1 mb-6">
          <TabsTrigger value="all" className="flex items-center gap-2">
            <BarChart3 className="h-4 w-4" />
            <span>All KPIs</span>
            <span className="ml-1 text-xs bg-[var(--color-muted)] px-1.5 py-0.5 rounded">{stats.total}</span>
          </TabsTrigger>
          {workstreams.map((ws) => {
            const display = WORKSTREAM_DISPLAY[ws.id];
            return (
              <TabsTrigger key={ws.id} value={ws.id} className="flex items-center gap-2">
                {display?.icon}
                <span>{display?.name ?? ws.name}</span>
                <span className="ml-1 text-xs bg-[var(--color-muted)] px-1.5 py-0.5 rounded">
                  {ws.kpi_count}
                </span>
              </TabsTrigger>
            );
          })}
        </TabsList>

        {/* KPI Grid */}
        <TabsContent value={activeWorkstream} className="mt-0">
          {isLoadingKPIs ? (
            <div className="flex items-center justify-center py-12">
              <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary" />
            </div>
          ) : filteredKPIs.length === 0 ? (
            <div className="text-center py-12">
              <Search className="h-12 w-12 text-[var(--color-muted-foreground)] mx-auto mb-4" />
              <h3 className="text-lg font-medium text-[var(--color-foreground)] mb-2">
                No KPIs found
              </h3>
              <p className="text-[var(--color-muted-foreground)]">
                Try adjusting your search or filter criteria
              </p>
            </div>
          ) : (
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
              {filteredKPIs.map((kpi) => (
                <KPICardDetailed key={kpi.id} kpi={kpi} />
              ))}
            </div>
          )}
        </TabsContent>
      </Tabs>

      {/* Footer info */}
      <div className="mt-8 p-4 bg-[var(--color-muted)] rounded-lg">
        <div className="flex items-start gap-3">
          <Info className="h-5 w-5 text-[var(--color-muted-foreground)] mt-0.5 flex-shrink-0" />
          <div className="text-sm text-[var(--color-muted-foreground)]">
            <p className="font-medium mb-1">About KPI Thresholds</p>
            <p>
              Each KPI has configurable thresholds: <span className="text-emerald-600 font-medium">Target</span> (ideal performance),
              <span className="text-amber-600 font-medium"> Warning</span> (needs attention), and
              <span className="text-rose-600 font-medium"> Critical</span> (requires immediate action).
              Thresholds are used for alerting and dashboard status indicators.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
