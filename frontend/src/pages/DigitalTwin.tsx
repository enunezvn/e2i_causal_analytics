/**
 * Digital Twin Page
 * =================
 *
 * E2I Digital Twin simulation interface for intervention pre-screening.
 * Allows running simulations, browsing history, and viewing results.
 *
 * Honesty contract (#705 H1/H2): this page renders ONLY what the backend
 * actually returns — the flat `SimulationResponse` / `SimulationDetailResponse`
 * shape — and shows honest empty / loading / error states. It never fabricates
 * outcomes (no `SAMPLE_SIMULATION` / `SAMPLE_HISTORY`, no static stat cards).
 * Sections with no backend data source (TRx/NRx/ROI lift, multi-axis fidelity
 * breakdown, evidence/risk-factor lists, sensitivity, projections) are NOT
 * rendered rather than filled with plausible-but-fake values.
 *
 * @module pages/DigitalTwin
 */

import { useState, useEffect, useMemo } from 'react';
import {
  FlaskConical,
  Play,
  History,
  AlertTriangle,
  CheckCircle,
  CheckCircle2,
  XCircle,
  RefreshCw,
  BarChart3,
  Settings2,
  Gauge,
  TrendingUp,
  ChevronDown,
  ChevronRight,
} from 'lucide-react';
import {
  useDigitalTwinHealth,
  useSimulationHistory,
  useRunSimulation,
  useSimulation,
  useInterventionTypes,
} from '@/hooks/api/use-digital-twin';
import { toast } from '@/hooks/use-toast';
import { useDataFreshness } from '@/hooks/use-data-freshness';
import { DataFreshnessIndicator } from '@/components/ui/data-freshness-indicator';
import {
  InterventionType,
  SimulationStatus,
  FALLBACK_INTERVENTION_TYPES,
  type SimulationResponse,
  type SimulationDetailResponse,
} from '@/types/digital-twin';
import { groupSimulationsByInterventionBrand } from '@/lib/digital-twin-history';

/** Title-case an intervention_type ("digital_engagement" → "Digital Engagement"). */
function formatIntervention(interventionType: string): string {
  return interventionType.replace(/_/g, ' ').replace(/\b\w/g, (l) => l.toUpperCase());
}

/** Brands offered by the history filter (mirrors the Brand enum). */
const HISTORY_BRAND_OPTIONS = ['Remibrutinib', 'Fabhalta', 'Kisqali'] as const;

// =============================================================================
// TYPES
// =============================================================================

interface StatCardProps {
  title: string;
  value: string | number;
  subtext?: string;
  icon: React.ReactNode;
  trend?: 'up' | 'down' | 'neutral';
}

/** Either shape returned by the simulate / simulation-detail endpoints. */
type AnySimulation = SimulationResponse | SimulationDetailResponse;

// =============================================================================
// HELPER COMPONENTS
// =============================================================================

function StatusBadge({ status }: { status: string }) {
  const styles: Record<string, string> = {
    healthy: 'bg-green-100 text-green-800 dark:bg-green-900/30 dark:text-green-400',
    degraded: 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900/30 dark:text-yellow-400',
    error: 'bg-red-100 text-red-800 dark:bg-red-900/30 dark:text-red-400',
    unknown: 'bg-gray-100 text-gray-800 dark:bg-gray-900/30 dark:text-gray-400',
  };

  return (
    <span className={`px-2 py-1 rounded-full text-xs font-medium ${styles[status] || styles.unknown}`}>
      {status.charAt(0).toUpperCase() + status.slice(1)}
    </span>
  );
}

/**
 * Recommendation badge. Accepts the raw backend recommendation string
 * (`deploy` | `skip` | `refine`) — also tolerates `analyze` for legacy
 * history rows — and never throws on an unknown value.
 */
function RecommendationBadge({ recommendation }: { recommendation: string }) {
  const config: Record<string, { icon: typeof CheckCircle; className: string }> = {
    deploy: { icon: CheckCircle, className: 'bg-green-100 text-green-800 dark:bg-green-900/30 dark:text-green-400' },
    skip: { icon: XCircle, className: 'bg-red-100 text-red-800 dark:bg-red-900/30 dark:text-red-400' },
    refine: { icon: Settings2, className: 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900/30 dark:text-yellow-400' },
    analyze: { icon: BarChart3, className: 'bg-blue-100 text-blue-800 dark:bg-blue-900/30 dark:text-blue-400' },
  };

  const key = String(recommendation).toLowerCase();
  const { icon: Icon, className } = config[key] ?? {
    icon: BarChart3,
    className: 'bg-gray-100 text-gray-800 dark:bg-gray-900/30 dark:text-gray-400',
  };

  return (
    <span className={`inline-flex items-center gap-1.5 px-3 py-1.5 rounded-full text-sm font-medium ${className}`}>
      <Icon className="h-4 w-4" />
      {key.charAt(0).toUpperCase() + key.slice(1)}
    </span>
  );
}

function StatCard({ title, value, subtext, icon, trend }: StatCardProps) {
  const trendColors = {
    up: 'text-green-600 dark:text-green-400',
    down: 'text-red-600 dark:text-red-400',
    neutral: 'text-gray-600 dark:text-gray-400',
  };

  return (
    <div className="bg-[var(--color-card)] rounded-lg border border-[var(--color-border)] p-4">
      <div className="flex items-center justify-between mb-2">
        <span className="text-sm text-[var(--color-text-secondary)]">{title}</span>
        <div className="p-1.5 rounded bg-[var(--color-primary)]/10 text-[var(--color-primary)]">
          {icon}
        </div>
      </div>
      <div className={`text-2xl font-bold ${trend ? trendColors[trend] : 'text-[var(--color-text-primary)]'}`}>
        {value}
      </div>
      {subtext && <p className="text-xs text-[var(--color-text-tertiary)] mt-1">{subtext}</p>}
    </div>
  );
}

function FidelityGauge({ score, label }: { score: number; label: string }) {
  const percentage = score * 100;
  const color = percentage >= 80 ? 'bg-green-500' : percentage >= 60 ? 'bg-yellow-500' : 'bg-red-500';

  return (
    <div className="flex flex-col gap-1">
      <div className="flex items-center justify-between">
        <span className="text-xs text-[var(--color-text-secondary)]">{label}</span>
        <span className="text-xs font-medium text-[var(--color-text-primary)]">{percentage.toFixed(0)}%</span>
      </div>
      <div className="h-2 bg-[var(--color-border)] rounded-full overflow-hidden">
        <div className={`h-full ${color} transition-all`} style={{ width: `${percentage}%` }} />
      </div>
    </div>
  );
}

/** A single metric tile inside the results panel. */
function Metric({ label, value, hint }: { label: string; value: string; hint?: string }) {
  return (
    <div className="flex flex-col">
      <span className="text-sm text-[var(--color-text-secondary)]">{label}</span>
      <span className="text-xl font-bold text-[var(--color-text-primary)]">{value}</span>
      {hint && <span className="text-xs text-[var(--color-text-tertiary)]">{hint}</span>}
    </div>
  );
}

function SimulationForm({
  onSubmit,
  isLoading,
}: {
  onSubmit: (data: { interventionType: InterventionType; brand: string; sampleSize: number; durationDays: number }) => void;
  isLoading: boolean;
}) {
  const [interventionType, setInterventionType] = useState<InterventionType>(InterventionType.EMAIL_CAMPAIGN);
  const [brand, setBrand] = useState('Remibrutinib');
  const [sampleSize, setSampleSize] = useState(1000);
  const [durationDays, setDurationDays] = useState(90);

  // Phase 1b: the intervention dropdown is driven by the backend's canonical
  // /digital-twin/intervention-types endpoint (brand-aware availability), so
  // FE/BE can never drift and the menu exposes only interventions that can
  // actually be simulated for the selected brand (a trained twin model exists).
  // Folding `brand` into the query key refetches on brand change.
  const {
    data: typesData,
    isLoading: typesLoading,
    isError: typesError,
  } = useInterventionTypes({ brand });

  const availableInterventions = useMemo(() => {
    if (typesError) {
      // Endpoint unreachable → degrade to the full canonical fallback so the
      // form stays usable; /simulate remains the authoritative availability gate.
      return FALLBACK_INTERVENTION_TYPES.map((i) => ({
        value: i.value as string,
        label: i.label,
        effect_basis: 'synthetic',
      }));
    }
    return (typesData?.interventions ?? [])
      // Expose only interventions whose effect is IDENTIFIED in the cohort — a trained
      // model alone is not enough; non-identified types 422 at /simulate (no fabrication).
      .filter((i) => i.available && i.available_for_effect)
      .map((i) => ({ value: i.value, label: i.label, effect_basis: i.effect_basis }));
  }, [typesData, typesError]);

  const noneAvailable =
    !typesLoading && !typesError && availableInterventions.length === 0;

  // Phase 2: surface HOW the selected intervention's effect is computed —
  // "cohort_estimated" (brand/intervention-specific, estimated from the
  // synthetic-gold cohort) vs the uniform synthetic uplift.
  const selectedBasis = availableInterventions.find(
    (i) => i.value === interventionType
  )?.effect_basis;

  // Keep the selected intervention valid as availability changes (e.g. the user
  // switches to a brand with a different available set).
  useEffect(() => {
    if (availableInterventions.length === 0) return;
    if (!availableInterventions.some((i) => i.value === interventionType)) {
      setInterventionType(availableInterventions[0].value as InterventionType);
    }
  }, [availableInterventions, interventionType]);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    onSubmit({ interventionType, brand, sampleSize, durationDays });
  };

  return (
    <form onSubmit={handleSubmit} className="space-y-4">
      <div>
        <label className="block text-sm font-medium text-[var(--color-text-secondary)] mb-1">
          Intervention Type
        </label>
        <select
          value={interventionType}
          onChange={(e) => setInterventionType(e.target.value as InterventionType)}
          disabled={typesLoading || noneAvailable}
          className="w-full px-3 py-2 bg-[var(--color-background)] border border-[var(--color-border)] rounded-lg text-[var(--color-text-primary)] disabled:opacity-50"
        >
          {availableInterventions.map((i) => (
            <option key={i.value} value={i.value}>
              {i.label}
            </option>
          ))}
        </select>
        {typesLoading && (
          <p className="mt-1 text-xs text-[var(--color-text-tertiary)]">
            Loading available interventions…
          </p>
        )}
        {typesError && (
          <p className="mt-1 text-xs text-amber-600 dark:text-amber-400">
            Could not verify availability — showing all interventions.
          </p>
        )}
        {noneAvailable && (
          <p className="mt-1 text-xs text-amber-600 dark:text-amber-400">
            No trained twin model for {brand} yet — simulations are unavailable for this brand.
          </p>
        )}
        {!typesLoading && !noneAvailable && selectedBasis === 'cohort_estimated' && (
          <p className="mt-1 text-xs text-[var(--color-text-tertiary)]">
            Effect basis: <span className="font-medium">brand cohort–estimated</span> — the
            ATE is estimated per brand from the synthetic-gold cohort (not a uniform assumption).
          </p>
        )}
        {!typesLoading && !noneAvailable && selectedBasis === 'synthetic' && (
          <p className="mt-1 text-xs text-[var(--color-text-tertiary)]">
            Effect basis: uniform synthetic uplift (not brand-specific).
          </p>
        )}
      </div>

      <div>
        <label className="block text-sm font-medium text-[var(--color-text-secondary)] mb-1">
          Brand
        </label>
        <select
          value={brand}
          onChange={(e) => setBrand(e.target.value)}
          className="w-full px-3 py-2 bg-[var(--color-background)] border border-[var(--color-border)] rounded-lg text-[var(--color-text-primary)]"
        >
          <option value="Remibrutinib">Remibrutinib</option>
          <option value="Fabhalta">Fabhalta</option>
          <option value="Kisqali">Kisqali</option>
        </select>
      </div>

      <div className="grid grid-cols-2 gap-4">
        <div>
          <label className="block text-sm font-medium text-[var(--color-text-secondary)] mb-1">
            Sample Size
          </label>
          <input
            type="number"
            value={sampleSize}
            onChange={(e) => setSampleSize(parseInt(e.target.value) || 0)}
            className="w-full px-3 py-2 bg-[var(--color-background)] border border-[var(--color-border)] rounded-lg text-[var(--color-text-primary)]"
            min={100}
            max={10000}
          />
        </div>
        <div>
          <label className="block text-sm font-medium text-[var(--color-text-secondary)] mb-1">
            Duration (days)
          </label>
          <input
            type="number"
            value={durationDays}
            onChange={(e) => setDurationDays(parseInt(e.target.value) || 0)}
            className="w-full px-3 py-2 bg-[var(--color-background)] border border-[var(--color-border)] rounded-lg text-[var(--color-text-primary)]"
            min={30}
            max={365}
          />
        </div>
      </div>

      <button
        type="submit"
        disabled={isLoading || typesLoading || noneAvailable}
        className="w-full flex items-center justify-center gap-2 px-4 py-2 bg-[var(--color-primary)] text-white rounded-lg hover:bg-[var(--color-primary-hover)] transition-colors disabled:opacity-50"
      >
        {isLoading ? (
          <RefreshCw className="h-4 w-4 animate-spin" />
        ) : (
          <Play className="h-4 w-4" />
        )}
        Run Simulation
      </button>
    </form>
  );
}

/**
 * Friendly label for the backend `data_provenance` marker — honest about the
 * effect basis. Both values are synthetic data (the SYNTHETIC badge stays); the
 * cohort one is a brand/intervention-ESTIMATED effect, the other a flat uniform.
 */
function provenanceLabel(provenance: string): string {
  switch (provenance) {
    case 'synthetic_uplift_v1':
      return 'synthetic uplift model (v1 — uniform, not brand-specific)';
    case 'cohort_estimated_synthetic_gold_v1':
      return 'brand cohort–estimated (synthetic-gold; not real-world data)';
    default:
      return provenance;
  }
}

/**
 * Results panel for a single real simulation (SimulationResponse shape).
 * Renders only fields the backend returns.
 */
function SimulationResultPanel({ simulation }: { simulation: AnySimulation }) {
  const fmt = (n: number) => n.toFixed(3);

  // Supporting evidence — plain-English summary of the signals behind the
  // recommendation (ported from the retired Intervention Impact page, T10).
  // Three of the four bullets are conditional, so an array + .map() reads
  // cleaner than four conditional <li> blocks. Values are DERIVED from the
  // simulation fields (not constants).
  const evidence: string[] = [];
  if (simulation.is_significant) {
    evidence.push(`Effect is statistically significant (ATE: ${fmt(simulation.simulated_ate)})`);
  }
  if (simulation.effect_size_cohens_d != null) {
    evidence.push(`Effect size (Cohen's d): ${simulation.effect_size_cohens_d.toFixed(2)}`);
  }
  if (simulation.statistical_power != null) {
    evidence.push(`Statistical power: ${(simulation.statistical_power * 100).toFixed(0)}%`);
  }
  evidence.push(`95% CI: [${fmt(simulation.simulated_ci_lower)}, ${fmt(simulation.simulated_ci_upper)}]`);

  return (
    <div className="space-y-6">
      {/* Title — identifies WHAT this simulation is (intervention · brand), so an
          opened result card is never anonymous. */}
      <div className="flex flex-wrap items-center justify-between gap-2 border-b border-[var(--color-border)] pb-3">
        <div>
          <h3 className="text-lg font-semibold text-[var(--color-text-primary)]">
            {formatIntervention(simulation.intervention_type)} · {simulation.brand}
          </h3>
          <p className="text-xs text-[var(--color-text-tertiary)]">
            {simulation.twin_type} twin · {simulation.twin_count.toLocaleString()} twins
            {simulation.created_at ? ` · ${new Date(simulation.created_at).toLocaleString()}` : ''}
          </p>
        </div>
        <span className="px-2 py-1 rounded-full text-xs font-medium bg-[var(--color-primary)]/10 text-[var(--color-primary)] capitalize">
          {String(simulation.status)}
        </span>
      </div>

      {/* Recommendation + rationale */}
      <div className="flex items-start justify-between p-4 bg-[var(--color-background)] rounded-lg border border-[var(--color-border)]">
        <div>
          <div className="flex items-center gap-3 mb-2">
            <RecommendationBadge recommendation={simulation.recommendation} />
            {simulation.data_provenance?.includes('synthetic') && (
              <span
                className="inline-flex items-center rounded-full bg-amber-100 dark:bg-amber-900/30 px-2 py-0.5 text-xs font-semibold text-amber-800 dark:text-amber-300"
                title="This estimate comes from synthetic data, not a real-world feed."
              >
                SYNTHETIC
              </span>
            )}
            <span className="text-xs text-[var(--color-text-tertiary)]">
              Confidence: {(simulation.simulation_confidence * 100).toFixed(0)}%
            </span>
          </div>
          <p className="text-sm text-[var(--color-text-primary)]">
            {simulation.recommendation_rationale}
          </p>
          {simulation.data_provenance && (
            <p className="mt-1 text-xs text-[var(--color-text-tertiary)]">
              Estimate source: {provenanceLabel(simulation.data_provenance)}
            </p>
          )}
        </div>
      </div>

      {/* Supporting evidence (derived above from the simulation fields). */}
      <div>
        <h4 className="text-sm font-medium text-[var(--color-text-secondary)] mb-2">
          Supporting Evidence
        </h4>
        <ul className="space-y-1">
          {evidence.map((point) => (
            <li
              key={point}
              className="flex items-start gap-2 text-sm text-[var(--color-text-primary)]"
            >
              <CheckCircle2 className="h-4 w-4 mt-0.5 text-emerald-600 flex-shrink-0" />
              <span>{point}</span>
            </li>
          ))}
        </ul>
      </div>

      {/* Fidelity warning (only when the backend flags one) */}
      {simulation.fidelity_warning && (
        <div className="flex items-start gap-2 p-3 bg-yellow-50 dark:bg-yellow-900/20 border border-yellow-200 dark:border-yellow-800 rounded-lg">
          <AlertTriangle className="h-4 w-4 text-yellow-600 dark:text-yellow-400 mt-0.5" />
          <p className="text-xs text-yellow-800 dark:text-yellow-300">
            {simulation.fidelity_warning_reason || 'Model fidelity is low for this simulation; interpret with caution.'}
          </p>
        </div>
      )}

      {/* Core outcome metrics (exactly what the backend returns) */}
      <div>
        <h4 className="text-sm font-medium text-[var(--color-text-secondary)] mb-3">Estimated Effect</h4>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <Metric
            label="ATE"
            value={fmt(simulation.simulated_ate)}
            hint={`95% CI: [${fmt(simulation.simulated_ci_lower)}, ${fmt(simulation.simulated_ci_upper)}]`}
          />
          <Metric
            label="Std. Error"
            value={fmt(simulation.simulated_std_error)}
            hint={simulation.is_significant ? 'Significant' : 'Not significant'}
          />
          <Metric
            label="Effect Direction"
            value={String(simulation.effect_direction)}
            hint={
              simulation.effect_size_cohens_d != null
                ? `Cohen's d: ${simulation.effect_size_cohens_d.toFixed(2)}`
                : undefined
            }
          />
          <Metric
            label="Twins Simulated"
            value={simulation.twin_count.toLocaleString()}
            hint={
              simulation.statistical_power != null
                ? `Power: ${(simulation.statistical_power * 100).toFixed(0)}%`
                : undefined
            }
          />
        </div>
      </div>

      {/* Model fidelity (single backend score — no fabricated breakdown) */}
      {simulation.model_fidelity_score != null && (
        <div>
          <h4 className="text-sm font-medium text-[var(--color-text-secondary)] mb-3">Model Fidelity</h4>
          <div className="max-w-xs">
            <FidelityGauge score={simulation.model_fidelity_score} label="Overall fidelity score" />
          </div>
        </div>
      )}

      {/* Recommended parameters, when present */}
      {(simulation.recommended_sample_size != null || simulation.recommended_duration_weeks != null) && (
        <div>
          <h4 className="text-sm font-medium text-[var(--color-text-secondary)] mb-3">Recommended Parameters</h4>
          <div className="grid grid-cols-2 gap-4">
            {simulation.recommended_sample_size != null && (
              <Metric label="Sample Size" value={simulation.recommended_sample_size.toLocaleString()} />
            )}
            {simulation.recommended_duration_weeks != null && (
              <Metric label="Duration" value={`${simulation.recommended_duration_weeks} weeks`} />
            )}
          </div>
        </div>
      )}

      {/* Execution metadata */}
      <div className="pt-4 border-t border-[var(--color-border)] flex items-center justify-between text-xs text-[var(--color-text-tertiary)]">
        <span>Simulation ID: {simulation.simulation_id}</span>
        <span>Executed in {simulation.execution_time_ms}ms</span>
      </div>
    </div>
  );
}

// =============================================================================
// MAIN PAGE
// =============================================================================

export default function DigitalTwin() {
  const [activeTab, setActiveTab] = useState<'results' | 'history'>('results');
  // The simulation_id of a history item the user clicked to inspect (if any).
  const [selectedId, setSelectedId] = useState<string | null>(null);
  // History brand filter ('all' = every brand the caller may see).
  const [historyBrand, setHistoryBrand] = useState<string>('all');
  // Expanded (brand, intervention) groups in the deduped history list.
  const [expandedGroups, setExpandedGroups] = useState<Set<string>>(new Set());

  const { data: healthData } = useDigitalTwinHealth();
  const {
    data: historyData,
    refetch: refetchHistory,
    dataUpdatedAt: historyUpdatedAt,
    isFetching: isHistoryFetching,
  } = useSimulationHistory({
    brand: historyBrand === 'all' ? undefined : historyBrand,
    limit: 25,
  });
  const historyFreshness = useDataFreshness(historyUpdatedAt);

  const {
    mutate: runSim,
    isPending: isRunning,
    data: runResult,
    isError: isRunError,
    error: runError,
  } = useRunSimulation({
    onSuccess: (data) => {
      // Show the fresh run result (clear any history selection), refresh list.
      setSelectedId(null);
      setActiveTab('results');
      refetchHistory();
      // Defense-in-depth: the backend now gates FAILED results to 422 (N1), but a
      // clicked history row could still carry status='failed' — never toast it as
      // a success.
      if (data.status === SimulationStatus.FAILED) {
        toast({
          variant: 'destructive',
          title: 'Simulation Did Not Complete',
          description: data.error_message || 'The simulation failed; no result to show.',
        });
        return;
      }
      toast({
        title: 'Simulation Complete',
        description: 'Your simulation has been processed successfully.',
      });
    },
    onError: (error) => {
      const errorMessage = error.message || 'An unexpected error occurred';
      const isTimeout = errorMessage.toLowerCase().includes('timeout');
      const isNetworkError = error.isNetworkError;

      toast({
        variant: 'destructive',
        title: 'Simulation Failed',
        description: isTimeout
          ? 'The simulation took too long. Try reducing the sample size or duration, then try again.'
          : isNetworkError
            ? 'Unable to reach the server. Please check your connection and try again.'
            : `${errorMessage}. Please try again or contact support if the issue persists.`,
      });
    },
  });

  // Detail for a clicked history item (enabled only when one is selected).
  const {
    data: selectedDetail,
    isLoading: isDetailLoading,
    isError: isDetailError,
  } = useSimulation(selectedId ?? '', { enabled: !!selectedId });

  // What the Results tab shows: the inspected history detail takes priority,
  // otherwise the latest run result. Never a fabricated default.
  const displayed: AnySimulation | null = selectedId
    ? (selectedDetail ?? null)
    : (runResult ?? null);

  // Mutually-exclusive results-panel sub-states.
  const detailLoading = !!selectedId && isDetailLoading;
  const detailError = !!selectedId && isDetailError && !isDetailLoading;

  const health = healthData ?? {
    status: 'unknown',
    service: 'digital-twin',
    models_available: 0,
    simulations_pending: 0,
    last_simulation_at: undefined,
  };

  const historyItems = useMemo(() => historyData?.simulations ?? [], [historyData]);
  // Collapse repeated runs of the same (brand, intervention) into one row each
  // (latest + count + expandable run list) so near-identical re-runs no longer
  // read as duplicates — without dropping any run.
  const historyGroups = useMemo(
    () => groupSimulationsByInterventionBrand(historyItems),
    [historyItems],
  );
  const toggleGroup = (key: string) =>
    setExpandedGroups((prev) => {
      const next = new Set(prev);
      if (next.has(key)) next.delete(key);
      else next.add(key);
      return next;
    });
  const openSimulation = (id: string) => {
    setSelectedId(id);
    setActiveTab('results');
  };
  const deployCount = historyItems.filter(
    (s) => String(s.recommendation_type).toLowerCase() === 'deploy'
  ).length;
  const deployRate = historyItems.length > 0 ? Math.round((deployCount / historyItems.length) * 100) : null;
  const fidelityPct =
    displayed?.model_fidelity_score != null ? Math.round(displayed.model_fidelity_score * 100) : null;

  const handleRunSimulation = (formData: { interventionType: InterventionType; brand: string; sampleSize: number; durationDays: number }) => {
    runSim({
      intervention: {
        intervention_type: formData.interventionType,
        duration_weeks: Math.ceil(formData.durationDays / 7),
      },
      brand: formData.brand,
      twin_count: formData.sampleSize,
    });
  };

  return (
    <div className="p-6 space-y-6">
      {/* Header */}
      <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold text-[var(--color-text-primary)] flex items-center gap-3">
            <FlaskConical className="h-7 w-7 text-[var(--color-primary)]" />
            Digital Twin
          </h1>
          <p className="text-[var(--color-text-secondary)] mt-1">
            Intervention pre-screening and scenario analysis
          </p>
        </div>
        <div className="flex items-center gap-3">
          <StatusBadge status={health.status} />
          <span className="text-xs text-[var(--color-text-tertiary)]">
            {health.models_available} model{health.models_available !== 1 ? 's' : ''} available
          </span>
        </div>
      </div>

      {/* Stats Cards — derived from real data; honest "—" when unavailable */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <StatCard
          title="Simulations"
          value={historyItems.length}
          subtext="In recent history"
          icon={<FlaskConical className="h-4 w-4" />}
        />
        <StatCard
          title="Deploy Rate"
          value={deployRate != null ? `${deployRate}%` : '—'}
          subtext="Of recent runs"
          icon={<CheckCircle className="h-4 w-4" />}
          trend={deployRate != null ? 'up' : undefined}
        />
        <StatCard
          title="Models Available"
          value={health.models_available}
          subtext="Trained twin models"
          icon={<Gauge className="h-4 w-4" />}
        />
        <StatCard
          title="Last Run Fidelity"
          value={fidelityPct != null ? `${fidelityPct}%` : '—'}
          subtext="Model fidelity score"
          icon={<TrendingUp className="h-4 w-4" />}
        />
      </div>

      {/* Main Content */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Simulation Form */}
        <div className="bg-[var(--color-card)] rounded-lg border border-[var(--color-border)] p-6">
          <h3 className="text-lg font-semibold text-[var(--color-text-primary)] mb-4 flex items-center gap-2">
            <Settings2 className="h-5 w-5 text-[var(--color-primary)]" />
            Configure Simulation
          </h3>
          <SimulationForm onSubmit={handleRunSimulation} isLoading={isRunning} />
        </div>

        {/* Results / History Panel */}
        <div className="lg:col-span-2 bg-[var(--color-card)] rounded-lg border border-[var(--color-border)] p-6">
          {/* Tabs */}
          <div className="flex items-center gap-4 mb-6 border-b border-[var(--color-border)] pb-4">
            <button
              onClick={() => setActiveTab('results')}
              className={`flex items-center gap-2 px-4 py-2 rounded-lg transition-colors ${
                activeTab === 'results'
                  ? 'bg-[var(--color-primary)] text-white'
                  : 'text-[var(--color-text-secondary)] hover:bg-[var(--color-border)]'
              }`}
            >
              <BarChart3 className="h-4 w-4" />
              Results
            </button>
            <button
              onClick={() => setActiveTab('history')}
              className={`flex items-center gap-2 px-4 py-2 rounded-lg transition-colors ${
                activeTab === 'history'
                  ? 'bg-[var(--color-primary)] text-white'
                  : 'text-[var(--color-text-secondary)] hover:bg-[var(--color-border)]'
              }`}
            >
              <History className="h-4 w-4" />
              History
            </button>
          </div>

          {/* Results Tab — mutually-exclusive states, no stale/fabricated data */}
          {activeTab === 'results' && (
            <>
              {/* Loading: a run is in flight, or a selected history detail is fetching */}
              {(isRunning || detailLoading) && (
                <div className="text-center py-12">
                  <RefreshCw className="h-10 w-10 text-[var(--color-text-tertiary)] mx-auto mb-4 animate-spin" />
                  <p className="text-[var(--color-text-secondary)]">
                    {isRunning ? 'Running simulation…' : 'Loading simulation…'}
                  </p>
                </div>
              )}

              {/* Error: a selected history detail failed to load */}
              {!isRunning && detailError && (
                <div className="text-center py-12">
                  <XCircle className="h-12 w-12 text-red-500/70 mx-auto mb-4" />
                  <p className="text-[var(--color-text-secondary)]">
                    This simulation could not be loaded. Please try again.
                  </p>
                </div>
              )}

              {/* Error: the run failed and there is nothing to show */}
              {!isRunning && !detailError && isRunError && !displayed && (
                <div className="text-center py-12">
                  <XCircle className="h-12 w-12 text-red-500/70 mx-auto mb-4" />
                  <p className="text-[var(--color-text-secondary)]">
                    {runError?.message || 'The simulation could not be completed. Please try again.'}
                  </p>
                </div>
              )}

              {/* Result */}
              {!isRunning && !detailLoading && !detailError && displayed && (
                <SimulationResultPanel simulation={displayed} />
              )}

              {/* Honest empty state */}
              {!isRunning && !detailLoading && !detailError && !displayed && !isRunError && (
                <div className="text-center py-12">
                  <FlaskConical className="h-12 w-12 text-[var(--color-text-tertiary)] mx-auto mb-4" />
                  <p className="text-[var(--color-text-secondary)]">Run a simulation to see results</p>
                </div>
              )}
            </>
          )}

          {/* History Tab */}
          {activeTab === 'history' && (
            <div className="space-y-3">
              {/* Header: brand filter + freshness. The filter lets you scope the
                  history to one brand or view all. */}
              <div className="flex flex-wrap items-center justify-between gap-3 mb-2">
                <div className="flex items-center gap-2">
                  <label
                    htmlFor="dt-history-brand"
                    className="text-xs text-[var(--color-text-secondary)]"
                  >
                    Brand
                  </label>
                  <select
                    id="dt-history-brand"
                    aria-label="Filter history by brand"
                    value={historyBrand}
                    onChange={(e) => setHistoryBrand(e.target.value)}
                    className="px-2 py-1 text-sm bg-[var(--color-background)] border border-[var(--color-border)] rounded-md text-[var(--color-text-primary)]"
                  >
                    <option value="all">All brands</option>
                    {HISTORY_BRAND_OPTIONS.map((b) => (
                      <option key={b} value={b}>
                        {b}
                      </option>
                    ))}
                  </select>
                </div>
                <DataFreshnessIndicator
                  {...historyFreshness}
                  showRefreshButton
                  onRefresh={() => refetchHistory()}
                  isRefreshing={isHistoryFetching}
                />
              </div>

              {historyItems.length === 0 ? (
                <div className="text-center py-12">
                  <History className="h-12 w-12 text-[var(--color-text-tertiary)] mx-auto mb-4" />
                  <p className="text-[var(--color-text-secondary)]">No simulations yet</p>
                  <p className="text-xs text-[var(--color-text-tertiary)] mt-1">
                    {historyBrand === 'all'
                      ? 'Run a simulation to start building history.'
                      : `No simulations for ${historyBrand} yet.`}
                  </p>
                </div>
              ) : (
                historyGroups.map((group) => {
                  const sim = group.latest;
                  const expanded = expandedGroups.has(group.key);
                  return (
                    <div key={group.key}>
                      <div
                        onClick={() => openSimulation(sim.simulation_id)}
                        className="flex items-center justify-between p-4 bg-[var(--color-background)] rounded-lg border border-[var(--color-border)] cursor-pointer hover:border-[var(--color-primary)] transition-colors"
                      >
                        <div className="flex items-center gap-4">
                          <div className="p-2 rounded-lg bg-[var(--color-primary)]/10 text-[var(--color-primary)]">
                            <FlaskConical className="h-4 w-4" />
                          </div>
                          <div>
                            <p className="text-sm font-medium text-[var(--color-text-primary)] flex items-center gap-2">
                              {formatIntervention(sim.intervention_type)}
                              {group.count > 1 && (
                                <button
                                  type="button"
                                  onClick={(e) => {
                                    e.stopPropagation();
                                    toggleGroup(group.key);
                                  }}
                                  aria-label={`${group.count} runs — show all`}
                                  className="inline-flex items-center gap-1 rounded-full bg-[var(--color-border)] px-2 py-0.5 text-xs text-[var(--color-text-secondary)] hover:text-[var(--color-text-primary)]"
                                >
                                  {group.count} runs
                                  {expanded ? (
                                    <ChevronDown className="h-3 w-3" />
                                  ) : (
                                    <ChevronRight className="h-3 w-3" />
                                  )}
                                </button>
                              )}
                            </p>
                            <p className="text-xs text-[var(--color-text-tertiary)]">
                              {sim.brand} - {new Date(sim.created_at).toLocaleString()}
                              {group.count > 1 ? ' · latest' : ''}
                            </p>
                          </div>
                        </div>
                        <div className="flex items-center gap-4">
                          <div className="text-right">
                            <p className="text-sm font-medium text-[var(--color-text-primary)]">
                              ATE: {sim.ate_estimate.toFixed(2)}
                            </p>
                          </div>
                          <RecommendationBadge recommendation={sim.recommendation_type} />
                        </div>
                      </div>

                      {group.count > 1 && expanded && (
                        <div className="ml-6 mt-1 space-y-1 border-l-2 border-[var(--color-border)] pl-3">
                          {group.runs.map((run) => (
                            <div
                              key={run.simulation_id}
                              onClick={() => openSimulation(run.simulation_id)}
                              className="flex items-center justify-between p-2 rounded-md hover:bg-[var(--color-background)] cursor-pointer"
                            >
                              <span className="text-xs text-[var(--color-text-tertiary)]">
                                {new Date(run.created_at).toLocaleString()}
                              </span>
                              <div className="flex items-center gap-3">
                                <span className="text-xs font-medium text-[var(--color-text-primary)]">
                                  ATE: {run.ate_estimate.toFixed(2)}
                                </span>
                                <RecommendationBadge recommendation={run.recommendation_type} />
                              </div>
                            </div>
                          ))}
                        </div>
                      )}
                    </div>
                  );
                })
              )}
            </div>
          )}
        </div>
      </div>

      {/* Info Footer */}
      <div className="bg-[var(--color-card)] rounded-lg border border-[var(--color-border)] p-6">
        <h3 className="text-lg font-semibold text-[var(--color-text-primary)] mb-4">
          About the Digital Twin
        </h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 text-sm text-[var(--color-text-secondary)]">
          <div>
            <h4 className="font-medium text-[var(--color-text-primary)] mb-2">Intervention Types</h4>
            <ul className="list-disc list-inside space-y-1">
              <li><strong>HCP Engagement</strong> - Field force interactions with physicians</li>
              <li><strong>Patient Support</strong> - Hub services and adherence programs</li>
              <li><strong>Digital Marketing</strong> - Online campaigns and content</li>
              <li><strong>Rep Training</strong> - Sales force education programs</li>
            </ul>
          </div>
          <div>
            <h4 className="font-medium text-[var(--color-text-primary)] mb-2">How It Works</h4>
            <p>
              The Digital Twin uses causal models trained on historical data to simulate the
              counterfactual outcomes of interventions. It estimates the Average Treatment Effect (ATE)
              and provides confidence intervals to quantify uncertainty.
            </p>
          </div>
        </div>
        <p className="text-xs text-[var(--color-text-tertiary)] mt-4">
          Last simulation: {health.last_simulation_at ?? 'Never'}
        </p>
      </div>
    </div>
  );
}
