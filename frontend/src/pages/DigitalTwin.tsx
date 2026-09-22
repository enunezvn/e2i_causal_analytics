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
  useTwinModels,
} from '@/hooks/api/use-digital-twin';
import { useDigitalTwinInsight } from '@/hooks/api';
import { StrategicInsightCard } from '@/components/insights';
import { toast } from '@/hooks/use-toast';
import { useDataFreshness } from '@/hooks/use-data-freshness';
import { DataFreshnessIndicator } from '@/components/ui/data-freshness-indicator';
import {
  EstimateScope,
  InterventionType,
  SimulationStatus,
  FALLBACK_INTERVENTION_TYPES,
  type SimulationResponse,
  type SimulationDetailResponse,
} from '@/types/digital-twin';
import { groupSimulationsByInterventionBrand } from '@/lib/digital-twin-history';
import {
  allModelsUnvalidated,
  describeModelCensus,
  explainModelCensus,
  fidelityStatusLabel,
} from '@/lib/digital-twin-models';

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
  /** Fuller explanation, shown as the card's native tooltip. */
  tooltip?: string;
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
/**
 * The experiment a stored simulation is linked to (#2206): written by /simulate when
 * given experiment_design_id, or by the proposed-experiments draft action. Absent
 * when the run is still a proposal — nothing is shown rather than a placeholder.
 */
function LinkedExperimentChip({ experimentId }: { experimentId?: string | null }) {
  if (!experimentId) return null;
  return (
    <span
      className="ml-2 inline-flex items-center rounded-full bg-purple-500/10 px-2 py-0.5 text-[10px] font-medium text-purple-600"
      title={`Linked to experiment ${experimentId}`}
    >
      linked · {experimentId.slice(0, 8)}
    </span>
  );
}

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

function StatCard({ title, value, subtext, tooltip, icon, trend }: StatCardProps) {
  const trendColors = {
    up: 'text-green-600 dark:text-green-400',
    down: 'text-red-600 dark:text-red-400',
    neutral: 'text-gray-600 dark:text-gray-400',
  };

  return (
    <div
      className="bg-[var(--color-card)] rounded-lg border border-[var(--color-border)] p-4"
      title={tooltip}
    >
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
  brand,
  onBrandChange,
}: {
  onSubmit: (data: { interventionType: InterventionType; brand: string; sampleSize: number; durationDays: number }) => void;
  isLoading: boolean;
  // Brand state is lifted to the page so the Strategic Interpretation card and
  // this form always describe the same brand's twin program.
  brand: string;
  onBrandChange: (brand: string) => void;
}) {
  const [interventionType, setInterventionType] = useState<InterventionType>(InterventionType.EMAIL_CAMPAIGN);
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
  // Two different gates empty the menu, and they need different remedies: no trained
  // model (train one) vs a model whose cohort identifies no intervention effect
  // (restore the cohort's treatment data). Naming the wrong one sends the reader
  // looking for a model that already exists.
  const modelExists = (typesData?.interventions ?? []).some((i) => i.available);
  // The backend says whether an empty set is a FINDING or a lookup that did not complete. A
  // repository outage must not read as "no trained model", and errored cohort probes must not
  // read as "your cohort data is gone" (that message recommends a production restore).
  const lookupDidNotComplete =
    noneAvailable &&
    (typesData?.model_resolution === 'unavailable' ||
      (modelExists && typesData?.effect_availability_status === 'unmeasured'));
  const noModel = noneAvailable && !modelExists && !lookupDidNotComplete;
  const modelExistsWithoutEffectData = noneAvailable && modelExists && !lookupDidNotComplete;

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
        {noModel && (
          <p className="mt-1 text-xs text-amber-600 dark:text-amber-400">
            No trained twin model for {brand} yet — simulations are unavailable for this brand.
          </p>
        )}
        {lookupDidNotComplete && (
          <p className="mt-1 text-xs text-amber-600 dark:text-amber-400">
            Intervention availability for {brand} could not be verified just now — the model or
            cohort lookup did not complete. It is retried automatically; simulations stay disabled
            until it succeeds.
          </p>
        )}
        {modelExistsWithoutEffectData && (
          <p className="mt-1 text-xs text-amber-600 dark:text-amber-400">
            A trained twin model exists for {brand}, but its cohort has no usable treatment data
            to estimate an intervention effect from — simulations are unavailable until the
            cohort data is restored.
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
          onChange={(e) => onBrandChange(e.target.value)}
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
 * Title for the confidence badge (#2104). Confidence blends the rows the estimator fit on,
 * the interval's precision and model fidelity.
 *
 * A STORED simulation is told apart from a fresh run by the detail-only
 * `population_filters` field (never a `created_at` cutoff): its score was persisted when
 * it ran, by the heuristic in force at that time — for a `twin_weighted_legacy` row one
 * that scored on the generated twin count — so current-heuristic sentences would be false.
 * A fresh run's sentence is selected by `data_provenance`: on the cohort path the rows are
 * the brand cohort, so generating more twins cannot raise it; on the synthetic path the
 * training frame is drawn from the twins, so it follows the twin sample. Unknown provenance
 * gets the neutral sentence only.
 */
function confidenceTitle(simulation: AnySimulation): string {
  const base =
    'Confidence blends the evidence behind this estimate: the rows the estimator fit on and the precision of the 95% interval, plus model fidelity only once it has been measured. An unvalidated model is scored on its evidence alone.';
  if ('population_filters' in simulation) {
    // Its own opening: a legacy row's evidence term WAS the generated twin count, so the
    // shared "rows the estimator fit on" sentence would be false before the qualification.
    const stored =
      'Confidence is the score stored when this simulation ran, computed by the confidence heuristic in force at that time: its evidence, the precision of the 95% interval, and model fidelity (earlier heuristics imputed 0.7 for an unvalidated model; the current one scores such a run on its evidence alone).';
    return simulation.subgroups_basis === 'twin_weighted_legacy'
      ? `${stored} That heuristic scored the evidence on the generated twin count.`
      : stored;
  }
  switch (simulation.data_provenance) {
    case 'cohort_estimated_synthetic_gold_v1':
      return `${base} Here those rows are the brand cohort rows the estimator fit on. Generating more twins does not raise it.`;
    case 'synthetic_uplift_v1':
      return `${base} Here the training frame is drawn from the twins the estimator fit on, so it follows the twin sample rather than a cohort.`;
    default:
      return base;
  }
}

/** Regions a stored simulation was filtered to, read from its detail payload's population_filters. */
function filteredRegions(simulation: AnySimulation): string[] {
  const regions = 'population_filters' in simulation ? simulation.population_filters?.regions : undefined;
  return Array.isArray(regions) ? regions.filter((r): r is string => typeof r === 'string') : [];
}

/**
 * Scope note for the headline effect (#2053); null when there is nothing worth saying.
 * 'regions' names them. A cohort-wide effect stays a plain ATE. 'unknown' is a stored
 * simulation from before the scope was recorded: it gets a note only when it was filtered
 * to regions, because only then might its effect cover just those regions.
 */
function estimateScopeNote(
  scope: EstimateScope | undefined,
  targetRegions: string[],
  filterRegions: string[]
): string | null {
  if (scope === EstimateScope.REGIONS && targetRegions.length > 0) {
    return `estimated on ${targetRegions.join(', ')}`;
  }
  if (scope === EstimateScope.UNKNOWN && filterRegions.length > 0) {
    return `scope not recorded — this effect may cover only ${filterRegions.join(', ')}`;
  }
  return null;
}

/**
 * Scope qualifier for a history row's ATE, by the same rule as the detail note
 * (estimateScopeNote): a region-scoped row names its regions; an unknown-scope row notes the
 * unrecorded scope only when its stored filter named regions (#2079); otherwise nothing.
 */
function HistoryScopeQualifier({
  scope,
  regions,
  filterRegions,
}: {
  scope?: EstimateScope;
  regions?: string[];
  filterRegions?: string[];
}) {
  const note = estimateScopeNote(scope, regions ?? [], filterRegions ?? []);
  if (note === null) return null;
  const label =
    scope === EstimateScope.REGIONS
      ? (regions ?? []).join(', ')
      : `scope not recorded — may cover only ${(filterRegions ?? []).join(', ')}`;
  return (
    <span className="ml-1 text-xs font-normal text-[var(--color-text-tertiary)]" title={note}>
      · {label}
    </span>
  );
}

/**
 * Results panel for a single real simulation (SimulationResponse shape).
 * Renders only fields the backend returns.
 */
function SimulationResultPanel({ simulation }: { simulation: AnySimulation }) {
  const fmt = (n: number) => n.toFixed(3);
  const specialtyEffects = simulation.effect_heterogeneity?.by_specialty ?? {};
  const specialtyProvenance =
    simulation.effect_heterogeneity?.axis_provenance?.specialty;

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

  // Region scope (#2023). A region filter narrows the estimate itself, so the headline
  // numbers above describe those regions, not the whole cohort. Say which, and keep the
  // cohort-wide effect visible so neither number is lost.
  // The backend states the scope (#2053); a stored simulation whose scope was never
  // recorded says 'unknown' and is noted when it was region-filtered. A response without
  // the field falls back to its regions.
  const targetRegions = simulation.target_regions ?? [];
  const scope =
    simulation.estimate_scope ?? (targetRegions.length > 0 ? EstimateScope.REGIONS : undefined);
  const scopedToRegions = scope === EstimateScope.REGIONS && targetRegions.length > 0;
  const scopeNote = estimateScopeNote(scope, targetRegions, filteredRegions(simulation));
  if (scopedToRegions && simulation.cohort_effect != null) {
    evidence.push(
      `Cohort-wide effect (all regions): ${fmt(simulation.cohort_effect)}` +
        (simulation.cohort_ci_lower != null && simulation.cohort_ci_upper != null
          ? ` [${fmt(simulation.cohort_ci_lower)}, ${fmt(simulation.cohort_ci_upper)}]`
          : '')
    );
  }

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
            <span
              className="text-xs text-[var(--color-text-tertiary)]"
              title={confidenceTitle(simulation)}
            >
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

      {/* Cohort-derived specialty effects (#2162).  The provenance text makes the
          publication floor and scoring-only fallback visible beside the numbers. */}
      {specialtyProvenance && (
        <div>
          <h4 className="text-sm font-medium text-[var(--color-text-secondary)] mb-2">
            Specialty Effects
          </h4>
          <p className="mb-3 text-xs text-[var(--color-text-tertiary)]">
              Observed-region-mix CATE from {specialtyProvenance.source}, supported by dated
              cohort rows (not distinct HCPs). Published with at least{' '}
              {specialtyProvenance.min_group_rows} rows
              {specialtyProvenance.min_treated_rows != null
                ? `, ${specialtyProvenance.min_treated_rows} treated`
                : ''}
              {specialtyProvenance.min_control_rows != null
                ? `, and ${specialtyProvenance.min_control_rows} control`
                : ''}
              . Unsupported specialties fall back to region, then cohort for twin scoring;
              fallback values are not published as specialty effects.
          </p>
          {Object.keys(specialtyEffects).length > 0 ? (
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
              {Object.entries(specialtyEffects).map(([specialty, stats]) => (
                <div
                  key={specialty}
                  className="flex items-center justify-between rounded-lg border border-[var(--color-border)] bg-[var(--color-background)] px-3 py-2"
                >
                  <span className="text-sm text-[var(--color-text-primary)]">{specialty}</span>
                  <span className="text-right text-sm font-medium text-[var(--color-text-primary)]">
                    {fmt(Number(stats.ate))}
                    <span className="block text-xs font-normal text-[var(--color-text-tertiary)]">
                      {Number(stats.n).toLocaleString()} cohort rows
                    </span>
                  </span>
                </div>
              ))}
            </div>
          ) : (
            <p className="text-xs text-[var(--color-text-tertiary)]">
              No specialty effects met the publication floor for this cohort scope.
            </p>
          )}
          {Object.keys(specialtyProvenance.suppressed_groups).length > 0 && (
            <div className="mt-2 text-xs text-amber-700 dark:text-amber-300">
              {Object.entries(specialtyProvenance.suppressed_groups).map(([group, reason]) => (
                <p key={group}>
                  {group}:{' '}
                  {reason === 'source_value_missing'
                    ? 'source specialty is missing.'
                    : 'suppressed for insufficient support.'}
                </p>
              ))}
            </div>
          )}
        </div>
      )}

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
        <h4 className="text-sm font-medium text-[var(--color-text-secondary)] mb-3">
          Estimated Effect
          {scopeNote && (
            <span className="ml-2 font-normal text-[var(--color-text-tertiary)]">
              · {scopeNote}
            </span>
          )}
        </h4>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <Metric
            label={scopedToRegions ? `ATE · ${targetRegions.join(', ')}` : 'ATE'}
            value={fmt(simulation.simulated_ate)}
            hint={
              `95% CI: [${fmt(simulation.simulated_ci_lower)}, ${fmt(simulation.simulated_ci_upper)}]` +
              (scopedToRegions && simulation.cohort_effect != null
                ? ` · cohort-wide: ${fmt(simulation.cohort_effect)}`
                : '')
            }
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

      {/* Model fidelity (#2206): the explicit backend state. A NULL score is
          'unvalidated' — shown as such, never as a blank that reads like a pass. */}
      <div>
        <h4 className="text-sm font-medium text-[var(--color-text-secondary)] mb-3">Model Fidelity</h4>
        <div className="max-w-xs space-y-2">
          <p className="text-xs text-[var(--color-text-secondary)]" data-testid="fidelity-status">
            Status:{' '}
            <span className="font-medium text-[var(--color-text-primary)]">
              {fidelityStatusLabel(simulation.fidelity_status)}
            </span>
          </p>
          {simulation.model_fidelity_score != null ? (
            <FidelityGauge score={simulation.model_fidelity_score} label="Overall fidelity score" />
          ) : (
            <p className="text-xs text-[var(--color-text-tertiary)]">
              No experiment outcome has been compared against this model yet, so its
              prediction accuracy is unknown.
            </p>
          )}
        </div>
      </div>

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
  // Brand under configuration (lifted from the form so the Strategic
  // Interpretation card describes the same brand's twin program).
  const [brand, setBrand] = useState('Remibrutinib');
  // History brand filter ('all' = every brand the caller may see).
  const [historyBrand, setHistoryBrand] = useState<string>('all');
  // Expanded (brand, intervention) groups in the deduped history list.
  const [expandedGroups, setExpandedGroups] = useState<Set<string>>(new Set());

  const { data: healthData } = useDigitalTwinHealth();
  // Trained-model census (#2206): shared-fit + fidelity honesty fields per brand row.
  const { data: modelsData } = useTwinModels();
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

  // Strategic Interpretation — LLM-grounded read of the brand's twin program
  // (models, simulation evidence, intervention coverage); server-derived
  // grounding, honest deterministic fallback when the LLM is unavailable.
  const twinInsight = useDigitalTwinInsight();

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
  const modelRows = useMemo(() => modelsData?.models ?? [], [modelsData]);
  // With no run displayed yet, the card still must not read as a blank pass: when
  // every trained model is unvalidated, say so (codex r1 #2).
  const lastRunUnvalidated = displayed
    ? displayed.fidelity_status === 'unvalidated'
    : allModelsUnvalidated(modelRows);
  const modelCensusText = useMemo(() => describeModelCensus(modelRows), [modelRows]);
  const modelCensusWhy = useMemo(() => explainModelCensus(modelRows), [modelRows]);

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
          subtext={modelCensusText ?? 'Trained twin models'}
          tooltip={modelCensusWhy ?? undefined}
          icon={<Gauge className="h-4 w-4" />}
        />
        <StatCard
          title="Last Run Fidelity"
          value={fidelityPct != null ? `${fidelityPct}%` : lastRunUnvalidated ? 'Unvalidated' : '—'}
          subtext={
            lastRunUnvalidated
              ? displayed
                ? 'No experiment outcome compared against this model yet'
                : 'No experiment outcome compared against any twin model yet'
              : 'Model fidelity score'
          }
          icon={<TrendingUp className="h-4 w-4" />}
        />
      </div>

      {/* Strategic Interpretation — what the twin evidence means for {brand} */}
      <StrategicInsightCard
        description={`Agentic read of ${brand}'s twin simulation program, grounded in its models, run history and intervention coverage`}
        onGenerate={() => twinInsight.mutate({ brand })}
        isLoading={twinInsight.isPending}
        error={twinInsight.error?.message ?? null}
        insight={twinInsight.data?.insight}
        keyTakeaways={twinInsight.data?.key_takeaways}
        grounding={twinInsight.data?.grounding}
        isFallback={twinInsight.data?.is_fallback}
        provenance={twinInsight.data?.provenance}
        generatedAt={twinInsight.data?.generated_at}
      />

      {/* Main Content */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Simulation Form */}
        <div className="bg-[var(--color-card)] rounded-lg border border-[var(--color-border)] p-6">
          <h3 className="text-lg font-semibold text-[var(--color-text-primary)] mb-4 flex items-center gap-2">
            <Settings2 className="h-5 w-5 text-[var(--color-primary)]" />
            Configure Simulation
          </h3>
          <SimulationForm
            onSubmit={handleRunSimulation}
            isLoading={isRunning}
            brand={brand}
            onBrandChange={setBrand}
          />
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
                              <LinkedExperimentChip experimentId={sim.experiment_design_id} />
                            </p>
                          </div>
                        </div>
                        <div className="flex items-center gap-4">
                          <div className="text-right">
                            <p className="text-sm font-medium text-[var(--color-text-primary)]">
                              ATE: {sim.ate_estimate.toFixed(2)}
                              <HistoryScopeQualifier
                                scope={sim.estimate_scope}
                                regions={sim.target_regions}
                                filterRegions={sim.filter_regions}
                              />
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
                                <LinkedExperimentChip experimentId={run.experiment_design_id} />
                              </span>
                              <div className="flex items-center gap-3">
                                <span className="text-xs font-medium text-[var(--color-text-primary)]">
                                  ATE: {run.ate_estimate.toFixed(2)}
                                  <HistoryScopeQualifier
                                    scope={run.estimate_scope}
                                    regions={run.target_regions}
                                    filterRegions={run.filter_regions}
                                  />
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
