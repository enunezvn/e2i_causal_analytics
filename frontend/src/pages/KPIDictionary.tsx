/**
 * KPI Dictionary Page
 * ===================
 *
 * Comprehensive reference for all defined KPIs in the E2I Causal Analytics system.
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
  XCircle,
  Filter,
  Info,
  Target,
  BarChart3,
  Activity,
  TrendingUp,
  Beaker,
  Gauge,
  Lightbulb,
} from 'lucide-react';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { useKPIList, useWorkstreams, useKPIHealth } from '@/hooks/api/use-kpi';
import type { KPIMetadata, KPIThreshold } from '@/types/kpi';
import { StatusLegend } from '@/components/visualizations/dashboard/StatusLegend';
import { KeyConcepts } from '@/components/kpi/KeyConcepts';
import { EmptyState } from '@/components/ui/EmptyState';

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

  // Live /api/kpis is the source of truth. No hardcoded literal fallback —
  // absence renders loading/empty/error states instead of fabricated KPIs.
  // Memoised so the empty-array fallback keeps a stable reference across renders.
  const kpis = useMemo(() => kpiData?.kpis ?? [], [kpiData]);
  const hasError = !!(kpiData === undefined && !isLoadingKPIs);
  const workstreams = workstreamsData?.workstreams ?? Object.keys(WORKSTREAM_DISPLAY).map((id) => ({
    id,
    name: WORKSTREAM_DISPLAY[id]?.name ?? id,
    kpi_count: kpis.filter((k) => k.workstream === id).length,
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

  // State for main section tabs
  const [activeSection, setActiveSection] = useState<string>('tables');

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

      {/* Main Section Tabs */}
      <Tabs value={activeSection} onValueChange={setActiveSection} className="w-full mb-6">
        <TabsList className="w-full flex h-auto gap-1 p-1 bg-[var(--color-muted)] rounded-lg">
          <TabsTrigger value="tables" className="flex items-center gap-2 flex-1 py-2.5">
            <BarChart3 className="h-4 w-4" />
            <span>KPI Cards</span>
          </TabsTrigger>
          <TabsTrigger value="legend" className="flex items-center gap-2 flex-1 py-2.5">
            <Gauge className="h-4 w-4" />
            <span>Status Legend</span>
          </TabsTrigger>
          <TabsTrigger value="concepts" className="flex items-center gap-2 flex-1 py-2.5">
            <Lightbulb className="h-4 w-4" />
            <span>Key Concepts</span>
          </TabsTrigger>
        </TabsList>

        {/* Tables Tab Content */}
        <TabsContent value="tables" className="mt-6">
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
          <div className="mb-6">
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
                ) : kpis.length === 0 ? (
                  // Live catalog itself is empty (or the API errored) — surface
                  // the absence honestly instead of falling back to a literal.
                  <EmptyState
                    title="No KPIs available"
                    description={hasError
                      ? 'Failed to load the KPI catalog from the API. Try refreshing.'
                      : 'The KPI catalog is empty. Once /api/kpis returns metadata, KPIs appear here.'}
                  />
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
          </div>

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
        </TabsContent>

        {/* Status Legend Tab Content */}
        <TabsContent value="legend" className="mt-6">
          <StatusLegend />
        </TabsContent>

        {/* Key Concepts Tab Content */}
        <TabsContent value="concepts" className="mt-6">
          <KeyConcepts />
        </TabsContent>
      </Tabs>
    </div>
  );
}
