/**
 * Feature Importance Page
 * =======================
 *
 * SHAP feature importance for the deployed gold-standard cohort models (#39).
 *
 * Two modes:
 * - **Cohort (global)** — mean |SHAP| aggregated over a sample of real cohort
 *   entities (`/api/explain/global`). Auto-loads on arrival; no entity needed.
 *   This is the canonical "feature importance" view.
 * - **Individual** — local SHAP for one selected real entity
 *   (`/api/explain/predict`), picked from a dropdown of real IDs
 *   (`/api/explain/sample-entities`) — patients for the patient cohorts, HCPs
 *   for hcp_adoption.
 *
 * Brand selector exposes all 12 per-brand models (4 cohorts × 3 brands).
 *
 * @module pages/FeatureImportance
 */

import { useState, useMemo, useCallback, useEffect } from 'react';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Input } from '@/components/ui/input';
import {
  BarChart3,
  RefreshCw,
  Download,
  Search,
  Info,
  TrendingUp,
  TrendingDown,
  Minus,
  AlertCircle,
  Loader2,
  Users,
  User,
  ChevronDown,
  ChevronRight,
} from 'lucide-react';
import {
  SHAPBarChart,
  SHAPBeeswarm,
  SHAPWaterfall,
  type BeeswarmDataPoint,
} from '@/components/visualizations';
import {
  ExplanationFormat,
  ModelType,
  GOLDSTD_BRANDS,
  GOLD_STANDARD_COHORTS,
  type ExplainableModelInfo,
  type FeatureContribution,
  type GlobalImportanceFeature,
  type GlobalImportancePoint,
  type GoldStandardBrand,
} from '@/types/explain';
import {
  useExplain,
  useExplainableModels,
  useExplanationHistory,
  useGlobalFeatureImportance,
  useSampleEntities,
} from '@/hooks/api/use-explain';
import { groupByCovariate, type CovariateGroup } from '@/lib/shap-covariates';
import { interpretGlobalImportance } from '@/lib/feature-importance/interpret';
import { cn } from '@/lib/utils';

// =============================================================================
// CONSTANTS
// =============================================================================

const DEFAULT_TOP_K = 10;
// Keep the cold-compute (sequential SHAP) comfortably under the 30s client
// timeout; the result is cached/warmed so repeat loads are instant regardless.
const COHORT_SAMPLE_SIZE = 25;

/** Friendly cohort labels (no version numbers). */
const COHORT_LABELS: Record<string, string> = {
  initiation: 'Initiation',
  persistence: 'Persistence',
  discontinuation: 'Discontinuation',
  hcp_adoption: 'HCP Adoption',
};

type ViewMode = 'cohort' | 'individual';

// =============================================================================
// HELPERS
// =============================================================================

/** Human-readable label for a model_type value (no version number). */
function formatModelLabel(model: ExplainableModelInfo): string {
  const raw = String(model.model_type);
  return (
    COHORT_LABELS[raw] ??
    raw
      .split('_')
      .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
      .join(' ')
  );
}

/**
 * Map an aggregated global feature into the `FeatureContribution` shape the
 * bar chart / ranking list consume. Magnitude is the global importance
 * (`mean_abs_shap`, always ≥ 0); the SIGN carries the net direction
 * (`mean_shap`), so the bar is sized by importance and colored by direction.
 */
function globalToContribution(f: GlobalImportanceFeature): FeatureContribution {
  const magnitude = f.mean_abs_shap;
  const signed = f.mean_shap >= 0 ? magnitude : -magnitude;
  return {
    feature_name: f.feature_name,
    feature_value: f.mean_feature_value,
    shap_value: signed,
    contribution_direction: f.mean_shap >= 0 ? 'positive' : 'negative',
    contribution_rank: f.contribution_rank,
  };
}

/** Project a feature to the {name, abs, signed} the covariate grouping needs. */
function featureGroupable(f: FeatureContribution) {
  return { name: f.feature_name, abs: Math.abs(f.shap_value), signed: f.shap_value };
}

/**
 * Collapse a covariate group into one FeatureContribution for the bar chart and
 * details panel (per-category detail stays in the group's `categories`).
 * Magnitude shown = net signed effect (sum of children); ranking uses the
 * group's summed importance. feature_value is surfaced only for a bare
 * (single-child) covariate — a multi-category covariate has no single value.
 */
function groupToContribution(group: CovariateGroup<FeatureContribution>): FeatureContribution {
  const single = group.categories.length === 1 ? group.categories[0] : null;
  return {
    feature_name: group.covariate,
    feature_value: single ? single.feature_value : null,
    shap_value: group.signed,
    contribution_direction: group.direction,
    contribution_rank: group.rank,
  };
}

/**
 * Build a REAL beeswarm distribution from per-entity global points: one dot per
 * (top feature, entity). Color axis (`featureValue`, normalized [0,1]) is the
 * per-feature min-max-normalized feature value — the canonical SHAP beeswarm
 * coloring (high feature value = red, low = blue).
 */
function buildGlobalBeeswarm(
  points: GlobalImportancePoint[],
  features: GlobalImportanceFeature[],
  maxFeatures = 8
): BeeswarmDataPoint[] {
  const top = features.slice(0, maxFeatures).map((f) => f.feature_name);
  const topSet = new Set(top);

  const byFeature = new Map<string, GlobalImportancePoint[]>();
  for (const p of points) {
    if (!topSet.has(p.feature_name)) continue;
    const list = byFeature.get(p.feature_name);
    if (list) list.push(p);
    else byFeature.set(p.feature_name, [p]);
  }

  const out: BeeswarmDataPoint[] = [];
  for (const fname of top) {
    const pts = byFeature.get(fname) ?? [];
    const numeric = pts
      .map((p) => p.feature_value)
      .filter((v): v is number => typeof v === 'number');
    const min = numeric.length ? Math.min(...numeric) : 0;
    const max = numeric.length ? Math.max(...numeric) : 1;
    const range = max - min || 1;
    pts.forEach((p, i) => {
      const norm =
        typeof p.feature_value === 'number' ? (p.feature_value - min) / range : 0.5;
      out.push({
        feature: fname,
        shapValue: p.shap_value,
        featureValue: Math.max(0, Math.min(1, norm)),
        originalValue: p.feature_value,
        instanceId: `${fname}-${i}`,
      });
    });
  }
  return out;
}

/**
 * Local beeswarm for a single explanation: one dot per top feature (no
 * fabricated distribution). Color axis derived from SHAP magnitude.
 */
function buildLocalBeeswarm(
  features: FeatureContribution[],
  instanceId: string
): BeeswarmDataPoint[] {
  const maxAbsShap =
    features.reduce((acc, f) => Math.max(acc, Math.abs(f.shap_value)), 0) || 1;
  return features.slice(0, 8).map((f) => {
    const normalized = 0.5 + (f.shap_value / maxAbsShap) * 0.5;
    return {
      feature: f.feature_name,
      shapValue: f.shap_value,
      featureValue: Math.max(0, Math.min(1, normalized)),
      originalValue: f.feature_value,
      instanceId: instanceId || 'current',
    };
  });
}

type HistoryRowLike = {
  explanation_id?: string | null;
  request_timestamp?: string | null;
  model_type?: string | null;
  model_version_id?: string | null;
  prediction_class?: string | null;
  prediction_probability?: number | null;
};

function formatHistoryProbability(p: HistoryRowLike['prediction_probability']): string {
  return typeof p === 'number' && Number.isFinite(p) ? p.toFixed(3) : '—';
}

function formatHistoryTimestamp(ts: HistoryRowLike['request_timestamp']): string {
  if (!ts) return 'Unknown time';
  const d = new Date(ts);
  return Number.isNaN(d.getTime()) ? String(ts) : d.toLocaleString();
}

/**
 * Show a base value, or "—" when it was never computed — never a fabricated 0.0
 * (anti-fabrication: a synthesized zero baseline would mislead).
 */
function formatBaseValue(v: number | null | undefined): string {
  return v === null || v === undefined ? '—' : v.toFixed(3);
}

// =============================================================================
// HELPER COMPONENTS
// =============================================================================

function FeatureRow({
  feature,
  isSelected,
  onClick,
}: {
  feature: FeatureContribution;
  isSelected: boolean;
  onClick: () => void;
}) {
  const TrendIcon =
    feature.shap_value > 0.02 ? TrendingUp : feature.shap_value < -0.02 ? TrendingDown : Minus;
  const trendColor =
    feature.shap_value > 0.02
      ? 'text-emerald-600'
      : feature.shap_value < -0.02
        ? 'text-rose-600'
        : 'text-gray-500';

  return (
    <div
      className={cn(
        'flex items-center justify-between p-3 rounded-lg cursor-pointer transition-colors',
        isSelected ? 'bg-primary/10 border border-primary/20' : 'bg-muted/50 hover:bg-muted'
      )}
      onClick={onClick}
    >
      <div className="flex items-center gap-3 flex-1 min-w-0">
        <Badge
          variant="outline"
          className="w-8 h-8 rounded-full flex items-center justify-center text-xs"
        >
          {feature.contribution_rank}
        </Badge>
        <div className="flex-1 min-w-0">
          <div className="font-medium truncate">{feature.feature_name.replace(/_/g, ' ')}</div>
          <div className="text-xs text-muted-foreground">
            Value: {String(feature.feature_value ?? '—')}
          </div>
        </div>
      </div>
      <div className="flex items-center gap-2">
        <span className={cn('font-mono text-sm', trendColor)}>
          {feature.shap_value >= 0 ? '+' : ''}
          {feature.shap_value.toFixed(4)}
        </span>
        <TrendIcon className={cn('h-4 w-4', trendColor)} />
      </div>
    </div>
  );
}

/**
 * One ranking row at the raw-covariate level. SHAP runs over encoded columns
 * (one-hot `geographic_region_west`, missingness `disease_severity__isna`), so a
 * single covariate would otherwise read as many duplicate rows. A grouped
 * covariate shows a count badge + chevron and expands to its encoded
 * categories; a bare covariate behaves like a plain selectable row.
 */
function CovariateGroupRow({
  group,
  expanded,
  selectedName,
  onToggle,
  onSelectLeaf,
}: {
  group: CovariateGroup<FeatureContribution>;
  expanded: boolean;
  selectedName: string | null;
  onToggle: () => void;
  onSelectLeaf: (f: FeatureContribution) => void;
}) {
  const isSelected = selectedName === group.covariate;
  const value = group.signed;
  const TrendIcon = value > 0.02 ? TrendingUp : value < -0.02 ? TrendingDown : Minus;
  const trendColor =
    value > 0.02 ? 'text-emerald-600' : value < -0.02 ? 'text-rose-600' : 'text-gray-500';

  return (
    <div>
      <div
        className={cn(
          'flex items-center justify-between p-3 rounded-lg cursor-pointer transition-colors',
          isSelected ? 'bg-primary/10 border border-primary/20' : 'bg-muted/50 hover:bg-muted'
        )}
        onClick={() => (group.isGrouped ? onToggle() : onSelectLeaf(group.categories[0]))}
      >
        <div className="flex items-center gap-3 flex-1 min-w-0">
          <Badge
            variant="outline"
            className="w-8 h-8 rounded-full flex items-center justify-center text-xs"
          >
            {group.rank}
          </Badge>
          <div className="flex-1 min-w-0">
            <div className="font-medium truncate flex items-center gap-1.5">
              {group.covariate.replace(/_/g, ' ')}
              {group.isGrouped && (
                <Badge variant="secondary" className="text-[10px] px-1.5 py-0 leading-4">
                  {group.categories.length}
                </Badge>
              )}
            </div>
            <div className="text-xs text-muted-foreground">
              {group.isGrouped ? `${group.categories.length} encoded features` : 'importance'}
            </div>
          </div>
        </div>
        <div className="flex items-center gap-2">
          <span className={cn('font-mono text-sm', trendColor)}>
            {value >= 0 ? '+' : ''}
            {value.toFixed(4)}
          </span>
          <TrendIcon className={cn('h-4 w-4', trendColor)} />
          {group.isGrouped &&
            (expanded ? (
              <ChevronDown className="h-4 w-4 text-muted-foreground" />
            ) : (
              <ChevronRight className="h-4 w-4 text-muted-foreground" />
            ))}
        </div>
      </div>
      {group.isGrouped && expanded && (
        <div className="ml-6 mt-1 space-y-1 border-l-2 border-muted/70 pl-2">
          {group.categories.map((cat) => (
            <FeatureRow
              key={cat.feature_name}
              feature={cat}
              isSelected={selectedName === cat.feature_name}
              onClick={() => onSelectLeaf(cat)}
            />
          ))}
        </div>
      )}
    </div>
  );
}

// =============================================================================
// MAIN COMPONENT
// =============================================================================

function FeatureImportance() {
  // -- Models --------------------------------------------------------------
  const {
    data: modelsData,
    isLoading: isLoadingModels,
    isError: isModelsError,
  } = useExplainableModels();

  // Only the real deployed gold-standard cohorts are explainable today; the
  // legacy demo types have no deployed model (and 503 on /predict), so they
  // would make the page look broken. Filter them out.
  const cohortModels = useMemo(() => {
    const all = modelsData?.supported_models ?? [];
    const gold = all.filter((m) => m.is_gold_standard);
    return gold.length > 0
      ? gold
      : all.filter((m) => GOLD_STANDARD_COHORTS.includes(m.model_type as ModelType));
  }, [modelsData]);

  // -- Selection state -----------------------------------------------------
  const [viewMode, setViewMode] = useState<ViewMode>('cohort');
  const [selectedModelType, setSelectedModelType] = useState<string>('');
  const [selectedBrand, setSelectedBrand] = useState<GoldStandardBrand>(GOLDSTD_BRANDS[0]);
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedFeature, setSelectedFeature] = useState<FeatureContribution | null>(null);

  const effectiveModelType =
    selectedModelType ||
    (cohortModels[0]?.model_type as string | undefined) ||
    ModelType.INITIATION;
  const isHcpCohort = effectiveModelType === ModelType.HCP_ADOPTION;

  const selectedModelInfo = useMemo(
    () => cohortModels.find((m) => String(m.model_type) === effectiveModelType),
    [cohortModels, effectiveModelType]
  );

  // ========================================================================
  // COHORT (GLOBAL) MODE
  // ========================================================================
  const {
    data: global,
    isLoading: isLoadingGlobal,
    isFetching: isFetchingGlobal,
    isError: isGlobalError,
    error: globalError,
    refetch: refetchGlobal,
  } = useGlobalFeatureImportance(effectiveModelType, selectedBrand, COHORT_SAMPLE_SIZE, {
    enabled: viewMode === 'cohort' && !!effectiveModelType,
  });

  const globalFeatures: FeatureContribution[] = useMemo(
    () => (global?.features ?? []).map(globalToContribution),
    [global]
  );
  // Denoise: drop encoded columns with zero mean |SHAP| (inactive __isna /
  // unobserved one-hot categories) so the beeswarm shows only features that
  // actually move the prediction — not a dense stack of dots on the zero line.
  const globalBeeswarm = useMemo(
    () =>
      buildGlobalBeeswarm(
        global?.points ?? [],
        (global?.features ?? []).filter((f) => f.mean_abs_shap > 0)
      ),
    [global]
  );

  // ========================================================================
  // INDIVIDUAL MODE
  // ========================================================================
  const { data: sampleEntities, isLoading: isLoadingEntities } = useSampleEntities(
    effectiveModelType,
    25,
    { enabled: viewMode === 'individual' && !!effectiveModelType }
  );
  const entityIds = useMemo(() => sampleEntities?.entities ?? [], [sampleEntities]);
  // Grain word for prose/placeholders ("hcp"/"patient"); the HCP-grain cohort
  // keys on hcp_id, every other cohort on patient_id.
  const grainLabel = isHcpCohort ? 'HCP' : 'Patient';
  // Field label for the ID picker — the HCP cohort reads "HCP ID" (#967).
  const entityLabel = isHcpCohort ? 'HCP ID' : 'Patient ID';

  const [selectedEntityId, setSelectedEntityId] = useState<string>('');

  // Default the picker to the first real ID once the list arrives / cohort changes.
  useEffect(() => {
    if (viewMode !== 'individual') return;
    if (entityIds.length === 0) return;
    if (!selectedEntityId || !entityIds.includes(selectedEntityId)) {
      setSelectedEntityId(entityIds[0]);
    }
  }, [viewMode, entityIds, selectedEntityId]);

  const {
    mutate: runExplain,
    data: explanation,
    isPending: isExplaining,
    isError: isExplainError,
    error: explainError,
    reset: resetExplain,
  } = useExplain();

  const doExplain = useCallback(
    (entityId: string) => {
      if (!entityId || !effectiveModelType) return;
      resetExplain();
      setSelectedFeature(null);
      runExplain({
        patient_id: entityId,
        hcp_id: isHcpCohort ? entityId : undefined,
        model_type: effectiveModelType as ModelType,
        brand: selectedBrand,
        format: ExplanationFormat.TOP_K,
        top_k: DEFAULT_TOP_K,
      });
    },
    [effectiveModelType, isHcpCohort, selectedBrand, runExplain, resetExplain]
  );

  // Auto-run the individual explanation whenever the entity / model / brand
  // changes so the page is never blank in individual mode.
  useEffect(() => {
    if (viewMode !== 'individual') return;
    if (!selectedEntityId) return;
    doExplain(selectedEntityId);
    // doExplain is stable per (model, brand, hcp) — re-run on those + entity.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [viewMode, selectedEntityId, effectiveModelType, selectedBrand]);

  const localFeatures: FeatureContribution[] = useMemo(
    () => explanation?.top_features ?? [],
    [explanation]
  );
  const localBaseValue = explanation?.base_value ?? 0;
  const localBeeswarm = useMemo(
    () =>
      buildLocalBeeswarm(
        localFeatures.filter((f) => f.shap_value !== 0),
        explanation?.patient_id ?? ''
      ),
    [localFeatures, explanation?.patient_id]
  );

  const {
    data: historyData,
    isLoading: isLoadingHistory,
    isError: isHistoryError,
  } = useExplanationHistory(selectedEntityId, undefined, 10, {
    enabled: viewMode === 'individual' && !!selectedEntityId && !isHcpCohort,
  });
  const historyExplanations = historyData?.explanations ?? [];

  // ========================================================================
  // SHARED DERIVED STATE
  // ========================================================================
  const features = viewMode === 'cohort' ? globalFeatures : localFeatures;
  const beeswarmData = viewMode === 'cohort' ? globalBeeswarm : localBeeswarm;

  const filteredFeatures = useMemo(() => {
    if (!searchQuery) return features;
    const query = searchQuery.toLowerCase();
    return features.filter((f) => f.feature_name.toLowerCase().includes(query));
  }, [features, searchQuery]);

  // Group the ENCODED SHAP features back to their parent raw covariate so the
  // ranking shows one row per covariate (expandable to its encoded categories)
  // instead of "geographic region" ×5 + `X`/`X__isna` twins. keep_columns comes
  // from the model info; when absent the helper passes through to the flat list.
  const keepColumns = selectedModelInfo?.keep_columns ?? null;
  const covariateGroups = useMemo(
    () => groupByCovariate(filteredFeatures, keepColumns, featureGroupable),
    [filteredFeatures, keepColumns]
  );
  // Covariate-level features for the bar chart (consistent with the ranking).
  const barFeatures = useMemo(() => covariateGroups.map(groupToContribution), [covariateGroups]);

  // Strategic interpretation of the cohort importance (rule-based, NO LLM).
  // Cohort mode only — the canonical "feature importance" view the review
  // targeted; null in individual mode / before data so the panel stays hidden.
  const insight = useMemo(
    () =>
      viewMode === 'cohort' && global
        ? interpretGlobalImportance(covariateGroups, {
            modelLabel: selectedModelInfo
              ? formatModelLabel(selectedModelInfo)
              : effectiveModelType,
            brand: selectedBrand,
            sampleSize: global.sample_size,
            grain: isHcpCohort ? 'hcp' : 'patient',
          })
        : null,
    [
      viewMode,
      global,
      covariateGroups,
      selectedModelInfo,
      effectiveModelType,
      selectedBrand,
      isHcpCohort,
    ]
  );

  const [expandedCovariates, setExpandedCovariates] = useState<Set<string>>(new Set());
  const toggleCovariate = useCallback((name: string) => {
    setExpandedCovariates((prev) => {
      const next = new Set(prev);
      if (next.has(name)) next.delete(name);
      else next.add(name);
      return next;
    });
  }, []);
  const selectLeaf = useCallback((f: FeatureContribution) => {
    setSelectedFeature((prev) => (prev?.feature_name === f.feature_name ? null : f));
  }, []);

  const hasData = viewMode === 'cohort' ? !!global : !!explanation;
  const isBusy = viewMode === 'cohort' ? isLoadingGlobal || isFetchingGlobal : isExplaining;
  const errorMessage =
    viewMode === 'cohort'
      ? ((globalError as { message?: string } | null)?.message ??
        'Failed to load cohort importance')
      : ((explainError as { message?: string } | null)?.message ??
        'Failed to compute explanation');

  // -- Handlers ------------------------------------------------------------
  const handleRefresh = useCallback(() => {
    setSelectedFeature(null);
    if (viewMode === 'cohort') refetchGlobal();
    else if (selectedEntityId) doExplain(selectedEntityId);
  }, [viewMode, refetchGlobal, selectedEntityId, doExplain]);

  const handleExport = useCallback(() => {
    if ((viewMode === 'cohort' && !global) || (viewMode === 'individual' && !explanation)) return;
    const payload =
      viewMode === 'cohort'
        ? { mode: 'cohort', model_type: effectiveModelType, brand: selectedBrand, global }
        : { mode: 'individual', model_type: effectiveModelType, brand: selectedBrand, explanation };
    const blob = new Blob([JSON.stringify(payload, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    const id = viewMode === 'cohort' ? selectedBrand : explanation?.patient_id;
    link.download = `${effectiveModelType}-${id}-shap.json`;
    link.href = url;
    link.click();
    URL.revokeObjectURL(url);
  }, [viewMode, effectiveModelType, selectedBrand, global, explanation]);

  // -- Render --------------------------------------------------------------
  return (
    <div className="container mx-auto px-4 py-8">
      {/* Header */}
      <div className="flex flex-col md:flex-row md:items-start md:justify-between gap-4 mb-6">
        <div>
          <h1 className="text-3xl font-bold mb-2">Feature Importance</h1>
          <p className="text-muted-foreground">
            SHAP feature importance for the gold-standard cohort models — cohort-average
            importance and per-entity explanations, grouped by covariate.
          </p>
        </div>

        <div className="flex flex-wrap items-center gap-3">
          {/* Cohort model */}
          <Select
            value={effectiveModelType}
            onValueChange={(v) => {
              setSelectedModelType(v);
              setSelectedFeature(null);
            }}
            disabled={isLoadingModels || cohortModels.length === 0}
          >
            <SelectTrigger className="w-[190px]">
              <SelectValue placeholder={isLoadingModels ? 'Loading models...' : 'Select cohort'} />
            </SelectTrigger>
            <SelectContent>
              {cohortModels.map((model) => (
                <SelectItem key={String(model.model_type)} value={String(model.model_type)}>
                  {formatModelLabel(model)}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>

          {/* Brand */}
          <Select
            value={selectedBrand}
            onValueChange={(v) => {
              setSelectedBrand(v as GoldStandardBrand);
              setSelectedFeature(null);
            }}
          >
            <SelectTrigger className="w-[150px]">
              <SelectValue placeholder="Select brand" />
            </SelectTrigger>
            <SelectContent>
              {GOLDSTD_BRANDS.map((b) => (
                <SelectItem key={b} value={b}>
                  {b}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>

          <Button variant="outline" size="icon" onClick={handleRefresh} disabled={isBusy}>
            <RefreshCw className={`h-4 w-4 ${isBusy ? 'animate-spin' : ''}`} />
          </Button>

          <Button variant="outline" onClick={handleExport} disabled={!hasData}>
            <Download className="h-4 w-4 mr-2" />
            Export
          </Button>
        </div>
      </div>

      {isModelsError && (
        <div role="alert" className="mb-4 flex items-center gap-2 text-sm text-rose-600">
          <AlertCircle className="h-4 w-4" />
          Failed to load model list
        </div>
      )}

      {/* Mode toggle */}
      <Tabs
        value={viewMode}
        onValueChange={(v) => {
          setViewMode(v as ViewMode);
          setSelectedFeature(null);
        }}
        className="mb-6"
      >
        <TabsList>
          <TabsTrigger value="cohort" className="gap-2">
            <Users className="h-4 w-4" /> Cohort average
          </TabsTrigger>
          <TabsTrigger value="individual" className="gap-2">
            <User className="h-4 w-4" /> Individual
          </TabsTrigger>
        </TabsList>
      </Tabs>

      {/* Individual mode: entity picker */}
      {viewMode === 'individual' && (
        <Card className="mb-6">
          <CardContent className="pt-6">
            <div className="flex flex-wrap items-end gap-3">
              <div className="flex-1 min-w-[260px]">
                <label className="block text-sm font-medium mb-1">{entityLabel}</label>
                <Select
                  value={selectedEntityId}
                  onValueChange={(v) => setSelectedEntityId(v)}
                  disabled={isLoadingEntities || entityIds.length === 0}
                >
                  <SelectTrigger>
                    <SelectValue
                      placeholder={
                        isLoadingEntities
                          ? `Loading ${grainLabel.toLowerCase()}s...`
                          : `Select a ${grainLabel.toLowerCase()}`
                      }
                    />
                  </SelectTrigger>
                  <SelectContent className="max-h-[320px]">
                    {entityIds.map((id) => (
                      <SelectItem key={id} value={id}>
                        {id}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
              <p className="text-xs text-muted-foreground pb-2 max-w-sm">
                Pick a real {grainLabel.toLowerCase()} ID — the explanation updates automatically.
              </p>
            </div>

            {isExplainError && (
              <div role="alert" className="mt-3 flex items-center gap-2 text-sm text-rose-600">
                <AlertCircle className="h-4 w-4" />
                Error: {errorMessage}
              </div>
            )}
            {isExplaining && (
              <div className="mt-3 flex items-center gap-2 text-sm text-muted-foreground">
                <Loader2 className="h-4 w-4 animate-spin" />
                Computing explanation...
              </div>
            )}
          </CardContent>
        </Card>
      )}

      {/* Cohort mode loading / error */}
      {viewMode === 'cohort' && isBusy && !global && (
        <Card className="mb-6">
          <CardContent className="pt-6">
            <div className="flex flex-col items-center justify-center py-12 text-center">
              <Loader2 className="h-8 w-8 text-muted-foreground mb-3 animate-spin" />
              <p className="text-sm text-muted-foreground max-w-md">
                Computing cohort feature importance (mean |SHAP| over a sample of real{' '}
                {isHcpCohort ? 'HCPs' : 'patients'})…
              </p>
            </div>
          </CardContent>
        </Card>
      )}
      {viewMode === 'cohort' && isGlobalError && (
        <Card className="mb-6">
          <CardContent className="pt-6">
            <div role="alert" className="flex items-center gap-2 text-sm text-rose-600">
              <AlertCircle className="h-4 w-4" />
              {errorMessage}
            </div>
          </CardContent>
        </Card>
      )}

      {/* Summary card */}
      {hasData && (
        <Card className="mb-6">
          <CardContent className="pt-6">
            <div className="flex flex-wrap items-center justify-between gap-4">
              <div className="flex items-center gap-4">
                <div className="p-3 rounded-lg bg-primary/10">
                  <BarChart3 className="h-6 w-6 text-primary" />
                </div>
                <div>
                  <h2 className="text-xl font-semibold">
                    {selectedModelInfo ? formatModelLabel(selectedModelInfo) : effectiveModelType} ·{' '}
                    {selectedBrand}
                  </h2>
                  <div className="flex flex-wrap items-center gap-x-4 gap-y-1 text-sm text-muted-foreground mt-1">
                    {viewMode === 'cohort' && global ? (
                      <>
                        <span className="font-mono text-xs">{global.model_name}</span>
                        <span>•</span>
                        <span>
                          {covariateGroups.length} covariate
                          {covariateGroups.length === 1 ? '' : 's'} · {global.features.length}{' '}
                          encoded
                        </span>
                        <span>•</span>
                        <span
                          title={`Averaged over a ${global.sample_size}-${
                            isHcpCohort ? 'HCP' : 'patient'
                          } sample of the cohort — not the cohort size.`}
                        >
                          n = {global.sample_size} sampled {isHcpCohort ? 'HCPs' : 'patients'}
                        </span>
                        <span>•</span>
                        <span>{global.cached ? 'cached' : 'freshly computed'}</span>
                      </>
                    ) : (
                      explanation && (
                        <>
                          <span>{localFeatures.length} features</span>
                          <span>•</span>
                          <span>
                            {grainLabel} {explanation.patient_id}
                          </span>
                          <span>•</span>
                          <span>
                            {explanation.prediction_class} (p ={' '}
                            {explanation.prediction_probability.toFixed(3)})
                          </span>
                        </>
                      )
                    )}
                  </div>
                </div>
              </div>
              <div className="flex items-center gap-6">
                <div className="text-center">
                  <div className="text-sm text-muted-foreground">Base Value</div>
                  <div className="text-2xl font-bold">
                    {formatBaseValue(
                      viewMode === 'cohort' ? global?.base_value : explanation?.base_value
                    )}
                  </div>
                </div>
                <div className="text-center">
                  <div className="text-sm text-muted-foreground">Top Feature</div>
                  <div className="text-lg font-semibold">
                    {covariateGroups[0]?.covariate?.replace(/_/g, ' ') ?? '—'}
                  </div>
                </div>
              </div>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Strategic Interpretation (cohort mode) — rule-based, no LLM. Adds the
          "so what": which covariate drives the prediction, how concentrated the
          importance is, what's negligible, and honest caveats. */}
      {viewMode === 'cohort' && global && insight?.available && (
        <Card className="mb-6">
          <CardHeader className="pb-3">
            <CardTitle className="text-base flex items-center gap-2">
              <Info className="h-4 w-4" />
              Strategic Interpretation
            </CardTitle>
            <CardDescription>{insight.headline}</CardDescription>
          </CardHeader>
          <CardContent className="space-y-3">
            <ul className="space-y-1.5">
              {insight.statements.map((s, i) => (
                <li key={i} className="flex gap-2 text-sm">
                  <span className="text-primary mt-0.5">•</span>
                  <span>{s}</span>
                </li>
              ))}
            </ul>
            {/* Provenance — attributed documentation, NOT a live metric (so it
                cannot drift): explains WHY this model has a small covariate set. */}
            <p className="text-xs text-muted-foreground border-t pt-2">
              This deployed model uses an empirically-locked, leakage-safe covariate set; a
              feature-selection experiment found additional columns did not improve held-out
              performance — a small covariate set is by design, not an omission.
            </p>
            <ul className="space-y-1">
              {insight.caveats.map((c, i) => (
                <li key={i} className="text-xs text-muted-foreground">
                  {c}
                </li>
              ))}
            </ul>
          </CardContent>
        </Card>
      )}

      {/* Main Content */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Left: Feature List */}
        <div className="lg:col-span-1">
          <Card className="h-full">
            <CardHeader className="pb-3">
              <CardTitle className="text-base flex items-center gap-2">
                Feature Rankings
                <Badge variant="secondary">{covariateGroups.length}</Badge>
              </CardTitle>
              <div className="relative">
                <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-muted-foreground" />
                <Input
                  placeholder="Search features..."
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  className="pl-10"
                />
              </div>
            </CardHeader>
            <CardContent
              className="space-y-2 max-h-[600px] overflow-y-auto"
              data-testid="feature-rankings"
            >
              {covariateGroups.map((group) => (
                <CovariateGroupRow
                  key={group.covariate}
                  group={group}
                  expanded={expandedCovariates.has(group.covariate)}
                  selectedName={selectedFeature?.feature_name ?? null}
                  onToggle={() => toggleCovariate(group.covariate)}
                  onSelectLeaf={selectLeaf}
                />
              ))}
              {covariateGroups.length === 0 && searchQuery && (
                <div className="text-center py-8 text-muted-foreground">
                  No features match your search
                </div>
              )}
              {covariateGroups.length === 0 && !searchQuery && (
                <div className="text-center py-8 text-muted-foreground text-sm">
                  {isBusy ? 'Loading…' : 'No feature contributions to show.'}
                </div>
              )}
            </CardContent>
          </Card>
        </div>

        {/* Right: Visualizations */}
        <div className="lg:col-span-2">
          {/* Always-visible note: pre-empts the "are the beeswarm vertical
              groupings normal?" question regardless of the active tab. */}
          {viewMode === 'cohort' && (
            <p className="text-xs text-muted-foreground mb-2 flex items-start gap-1.5">
              <Info className="h-3.5 w-3.5 mt-0.5 shrink-0" />
              <span>
                Binary/one-hot features take only a few SHAP values under the linear model, so
                their dots line up in vertical bands in the beeswarm — expected, not an artifact.
              </span>
            </p>
          )}
          <Tabs defaultValue="bar" className="space-y-4">
            <TabsList>
              <TabsTrigger value="bar">Bar Chart</TabsTrigger>
              <TabsTrigger value="beeswarm">Beeswarm</TabsTrigger>
              {viewMode === 'individual' && <TabsTrigger value="waterfall">Waterfall</TabsTrigger>}
              {viewMode === 'individual' && <TabsTrigger value="history">History</TabsTrigger>}
            </TabsList>

            <TabsContent value="bar">
              <Card>
                <CardHeader>
                  <CardTitle>
                    {viewMode === 'cohort'
                      ? 'Average Feature Importance'
                      : `Feature Contributions (this ${grainLabel.toLowerCase()})`}
                  </CardTitle>
                  <CardDescription>
                    {viewMode === 'cohort'
                      ? `Mean |SHAP| over ${global?.sample_size ?? COHORT_SAMPLE_SIZE} real ${
                          isHcpCohort ? 'HCPs' : 'patients'
                        }, grouped by covariate. Bar length = importance; color = net direction (green raises, red lowers the prediction).`
                      : `Signed SHAP contributions for this ${grainLabel.toLowerCase()}, grouped by covariate. Positive values push the prediction higher.`}
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <SHAPBarChart
                    features={barFeatures}
                    maxFeatures={10}
                    height={400}
                    showValues
                    onBarClick={(f) => setSelectedFeature(f)}
                  />
                </CardContent>
              </Card>
            </TabsContent>

            <TabsContent value="beeswarm">
              <Card>
                <CardHeader>
                  <CardTitle>
                    {viewMode === 'cohort'
                      ? 'SHAP Distribution Across the Cohort'
                      : 'Per-Feature SHAP Contributions'}
                  </CardTitle>
                  <CardDescription>
                    {viewMode === 'cohort'
                      ? 'One dot per sampled entity for each top feature. X-axis = SHAP value; color = feature value (red = high, blue = low). Zero-importance encoded columns are hidden. (See the note above on vertical bands.)'
                      : 'One dot per top feature for this entity. X-axis = SHAP value; color reflects SHAP direction.'}
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <SHAPBeeswarm
                    data={beeswarmData}
                    maxFeatures={8}
                    height={450}
                    showLegend={viewMode === 'cohort'}
                    showReferenceLine
                    onPointClick={(point) => {
                      const feature = features.find((f) => f.feature_name === point.feature);
                      if (feature) setSelectedFeature(feature);
                    }}
                  />
                </CardContent>
              </Card>
            </TabsContent>

            {viewMode === 'individual' && (
              <TabsContent value="waterfall">
                <Card>
                  <CardHeader>
                    <CardTitle>Individual Prediction Explanation</CardTitle>
                    <CardDescription>
                      Waterfall showing how features move the prediction from base value to final.
                      {selectedFeature && (
                        <span className="text-primary ml-2">
                          Highlighting: {selectedFeature.feature_name.replace(/_/g, ' ')}
                        </span>
                      )}
                    </CardDescription>
                  </CardHeader>
                  <CardContent>
                    <SHAPWaterfall
                      baseValue={localBaseValue}
                      features={localFeatures}
                      maxFeatures={10}
                      height={450}
                      onBarClick={(f) => setSelectedFeature(f)}
                    />
                  </CardContent>
                </Card>
              </TabsContent>
            )}

            {viewMode === 'individual' && (
              <TabsContent value="history">
                <Card>
                  <CardHeader>
                    <CardTitle>Explanation History</CardTitle>
                    <CardDescription>
                      Past SHAP explanations for{' '}
                      {selectedEntityId
                        ? `${grainLabel.toLowerCase()} ${selectedEntityId}`
                        : 'the selected entity'}
                      .
                    </CardDescription>
                  </CardHeader>
                  <CardContent>
                    {!selectedEntityId && (
                      <div className="text-sm text-muted-foreground">
                        Select an entity to view history.
                      </div>
                    )}
                    {selectedEntityId && isHcpCohort && (
                      <div className="text-sm text-muted-foreground">
                        History is recorded per patient; not available for the HCP cohort.
                      </div>
                    )}
                    {selectedEntityId && !isHcpCohort && isLoadingHistory && (
                      <div className="flex items-center gap-2 text-sm text-muted-foreground">
                        <Loader2 className="h-4 w-4 animate-spin" />
                        Loading history...
                      </div>
                    )}
                    {selectedEntityId && !isHcpCohort && isHistoryError && (
                      <div role="alert" className="flex items-center gap-2 text-sm text-rose-600">
                        <AlertCircle className="h-4 w-4" />
                        Failed to load explanation history
                      </div>
                    )}
                    {selectedEntityId &&
                      !isHcpCohort &&
                      !isLoadingHistory &&
                      !isHistoryError &&
                      historyExplanations.length === 0 && (
                        <div className="text-sm text-muted-foreground">
                          No historical explanations found for this entity.
                        </div>
                      )}
                    {selectedEntityId && !isHcpCohort && historyExplanations.length > 0 && (
                      <ul className="space-y-2">
                        {(historyExplanations as HistoryRowLike[]).map((h, idx) => (
                          <li
                            key={h.explanation_id ?? `history-${idx}`}
                            className="flex items-center justify-between p-3 rounded-lg bg-muted/50"
                          >
                            <div className="flex flex-col">
                              <span className="text-sm font-medium">
                                {h.model_type ?? 'unknown model'}
                              </span>
                              <span className="text-xs text-muted-foreground">
                                {formatHistoryTimestamp(h.request_timestamp)}
                                {h.model_version_id ? ` • ${h.model_version_id}` : ''}
                              </span>
                            </div>
                            <div className="text-right">
                              <div className="text-sm font-mono">{h.prediction_class ?? '—'}</div>
                              <div className="text-xs text-muted-foreground">
                                p = {formatHistoryProbability(h.prediction_probability)}
                              </div>
                            </div>
                          </li>
                        ))}
                      </ul>
                    )}
                  </CardContent>
                </Card>
              </TabsContent>
            )}
          </Tabs>

          {/* Selected Feature Details */}
          {selectedFeature && (
            <Card className="mt-6">
              <CardHeader>
                <CardTitle className="text-base flex items-center gap-2">
                  <Info className="h-4 w-4" />
                  Feature Details: {selectedFeature.feature_name.replace(/_/g, ' ')}
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                  <div className="bg-muted rounded-lg p-3">
                    <div className="text-xs text-muted-foreground">Rank</div>
                    <div className="text-lg font-semibold">#{selectedFeature.contribution_rank}</div>
                  </div>
                  <div className="bg-muted rounded-lg p-3">
                    <div className="text-xs text-muted-foreground">
                      {viewMode === 'cohort' ? 'Mean Value' : 'Current Value'}
                    </div>
                    <div className="text-lg font-semibold">
                      {typeof selectedFeature.feature_value === 'number'
                        ? selectedFeature.feature_value.toFixed(3)
                        : String(selectedFeature.feature_value ?? '—')}
                    </div>
                  </div>
                  <div className="bg-muted rounded-lg p-3">
                    <div className="text-xs text-muted-foreground">
                      {viewMode === 'cohort' ? 'Mean |SHAP|' : 'SHAP Value'}
                    </div>
                    <div
                      className={cn(
                        'text-lg font-semibold',
                        selectedFeature.shap_value >= 0 ? 'text-emerald-600' : 'text-rose-600'
                      )}
                    >
                      {selectedFeature.shap_value >= 0 ? '+' : ''}
                      {selectedFeature.shap_value.toFixed(4)}
                    </div>
                  </div>
                  <div className="bg-muted rounded-lg p-3">
                    <div className="text-xs text-muted-foreground">Direction</div>
                    <div
                      className={cn(
                        'text-lg font-semibold capitalize',
                        selectedFeature.contribution_direction === 'positive'
                          ? 'text-emerald-600'
                          : 'text-rose-600'
                      )}
                    >
                      {selectedFeature.contribution_direction}
                    </div>
                  </div>
                </div>
                <div className="mt-4 p-3 bg-muted/50 rounded-lg">
                  <h4 className="text-sm font-medium mb-2">Interpretation</h4>
                  <p className="text-sm text-muted-foreground">
                    {viewMode === 'cohort'
                      ? `Across the sampled cohort, "${selectedFeature.feature_name.replace(/_/g, ' ')}" is ranked #${selectedFeature.contribution_rank} by mean |SHAP|, with a net ${selectedFeature.contribution_direction} effect on the prediction.`
                      : `This feature has a ${selectedFeature.contribution_direction} impact on the prediction.${
                          selectedFeature.shap_value > 0
                            ? ` Higher values of "${selectedFeature.feature_name.replace(/_/g, ' ')}" tend to increase the predicted outcome.`
                            : ` Higher values of "${selectedFeature.feature_name.replace(/_/g, ' ')}" tend to decrease the predicted outcome.`
                        } It is ranked #${selectedFeature.contribution_rank} in importance for this prediction.`}
                  </p>
                </div>
              </CardContent>
            </Card>
          )}
        </div>
      </div>
    </div>
  );
}

export default FeatureImportance;
