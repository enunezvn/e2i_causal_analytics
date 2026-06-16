/**
 * Feature Importance Page
 * =======================
 *
 * Dashboard for analyzing live feature importance and SHAP explanations
 * by calling the backend `/api/explain/predict` endpoint.
 *
 * Features:
 * - Model selector dropdown populated from `useExplainableModels`
 * - Patient ID input that drives `useExplain` mutation
 * - Real SHAP feature contributions from the response
 * - SHAP Beeswarm + Waterfall + Bar visualizations
 * - Loading / error / empty states
 *
 * @module pages/FeatureImportance
 */

import { useState, useMemo, useCallback } from 'react';
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
  type ExplainRequest,
  type ExplainableModelInfo,
  type FeatureContribution,
} from '@/types/explain';
import {
  useExplain,
  useExplainableModels,
  useExplanationHistory,
} from '@/hooks/api/use-explain';
import { cn } from '@/lib/utils';

// =============================================================================
// CONSTANTS
// =============================================================================

const DEFAULT_TOP_K = 10;

/**
 * Gold-standard cohort families (#39) served by the REAL per-brand
 * `*_goldstd_lr_v1` registry models. For these cohorts the page offers a brand
 * selector so all 12 (4 cohorts × 3 brands) serving bundles are reachable
 * (#967). Mirrors `GOLDSTD_COHORT_MODEL_TYPES` in src/api/routes/explain.py.
 * Used only as a fallback when the backend `/explain/models` response omits the
 * authoritative `is_gold_standard` flag.
 */
const GOLDSTD_COHORTS = new Set<string>([
  'initiation',
  'persistence',
  'discontinuation',
  'hcp_adoption',
]);

/** The HCP-grain gold-standard cohort — keyed on hcp_id, not patient_id. */
const HCP_COHORT = 'hcp_adoption';

/**
 * Brands the gold-standard models are registered for. Mirrors `GOLDSTD_BRANDS`
 * in src/api/routes/explain.py; the backend resolves the serving name as
 * `f"{cohort}_{brand.toLowerCase()}_goldstd_lr_v1"` and FAILS CLOSED (422) on
 * an unknown brand.
 */
const GOLDSTD_BRANDS = ['Remibrutinib', 'Fabhalta', 'Kisqali'] as const;
const DEFAULT_BRAND: (typeof GOLDSTD_BRANDS)[number] = 'Remibrutinib';

// =============================================================================
// HELPERS
// =============================================================================

/** Word fragments that should render as acronyms, not Title-cased. */
const LABEL_ACRONYMS: Record<string, string> = { hcp: 'HCP' };

/**
 * Human-readable label for a backend model_type enum value.
 * e.g. `hcp_adoption` → "HCP Adoption" (matches the cohort labels in
 * TimeSeries.tsx), `initiation` → "Initiation".
 */
function formatModelLabel(model: ExplainableModelInfo): string {
  const raw = String(model.model_type);
  return raw
    .split('_')
    .map((part) => LABEL_ACRONYMS[part] ?? part.charAt(0).toUpperCase() + part.slice(1))
    .join(' ');
}

/**
 * Convert each live feature contribution into a single beeswarm point.
 *
 * The `/api/explain/predict` endpoint returns one explanation per call, so the
 * beeswarm only shows one dot per feature corresponding to *this* patient's
 * SHAP value. A true distribution view would require batch endpoints; until
 * then we honor "real SHAP values + feature names" by NOT fabricating extra
 * points. `instanceId` falls back to `patient_id` so each dot stays uniquely
 * keyed.
 *
 * `featureValue` is the *normalized* coloring axis expected by SHAPBeeswarm
 * (`[0, 1]`), so we derive it from each feature's SHAP magnitude relative to
 * the max so values don't saturate the high-color band. The original domain
 * value is preserved on `originalValue` for tooltips.
 */
function buildBeeswarmData(
  features: FeatureContribution[],
  patientId: string
): BeeswarmDataPoint[] {
  const maxAbsShap =
    features.reduce((acc, f) => Math.max(acc, Math.abs(f.shap_value)), 0) || 1;

  return features.slice(0, 8).map((f) => {
    // Map SHAP value to [0, 1] symmetrically around 0.5 for the coloring axis
    const normalized = 0.5 + (f.shap_value / maxAbsShap) * 0.5;
    return {
      feature: f.feature_name,
      shapValue: f.shap_value,
      featureValue: Math.max(0, Math.min(1, normalized)),
      // Preserve the real domain value for tooltips
      originalValue: f.feature_value,
      instanceId: patientId || 'current',
    };
  });
}

/**
 * Defensive accessor: history rows come straight from `ml_shap_analyses`
 * (see `/api/explain/history/{patient_id}` in src/api/routes/explain.py),
 * which is a superset of `ExplainResponse` with optional columns. Treat
 * everything as unknown and coerce.
 */
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
    feature.shap_value > 0.02 ? TrendingUp :
    feature.shap_value < -0.02 ? TrendingDown : Minus;

  const trendColor =
    feature.shap_value > 0.02 ? 'text-emerald-600' :
    feature.shap_value < -0.02 ? 'text-rose-600' : 'text-gray-500';

  return (
    <div
      className={cn(
        'flex items-center justify-between p-3 rounded-lg cursor-pointer transition-colors',
        isSelected
          ? 'bg-primary/10 border border-primary/20'
          : 'bg-muted/50 hover:bg-muted'
      )}
      onClick={onClick}
    >
      <div className="flex items-center gap-3 flex-1 min-w-0">
        <Badge variant="outline" className="w-8 h-8 rounded-full flex items-center justify-center text-xs">
          {feature.contribution_rank}
        </Badge>
        <div className="flex-1 min-w-0">
          <div className="font-medium truncate">
            {feature.feature_name.replace(/_/g, ' ')}
          </div>
          <div className="text-xs text-muted-foreground">
            Value: {String(feature.feature_value)}
          </div>
        </div>
      </div>
      <div className="flex items-center gap-2">
        <span className={cn('font-mono text-sm', trendColor)}>
          {feature.shap_value >= 0 ? '+' : ''}{feature.shap_value.toFixed(4)}
        </span>
        <TrendIcon className={cn('h-4 w-4', trendColor)} />
      </div>
    </div>
  );
}

// =============================================================================
// MAIN COMPONENT
// =============================================================================

function FeatureImportance() {
  // -- Live backend state --------------------------------------------------
  const {
    data: modelsData,
    isLoading: isLoadingModels,
    isError: isModelsError,
  } = useExplainableModels();

  const {
    mutate: runExplain,
    data: explanation,
    isPending: isExplaining,
    isError: isExplainError,
    error: explainError,
    reset: resetExplain,
  } = useExplain();

  // -- Form state ----------------------------------------------------------
  const supportedModels = useMemo(
    () => modelsData?.supported_models ?? [],
    [modelsData]
  );
  const [selectedModelType, setSelectedModelType] = useState<string>('');
  const [brand, setBrand] = useState<string>(DEFAULT_BRAND);
  const [patientId, setPatientId] = useState<string>('');
  const [submittedPatientId, setSubmittedPatientId] = useState<string>('');
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedFeature, setSelectedFeature] = useState<FeatureContribution | null>(null);

  // -- Historical explanations for the submitted patient ------------------
  const {
    data: historyData,
    isLoading: isLoadingHistory,
    isError: isHistoryError,
  } = useExplanationHistory(submittedPatientId, undefined, 10, {
    enabled: !!submittedPatientId,
  });
  const historyExplanations = historyData?.explanations ?? [];

  // Default to first available model once list arrives
  const effectiveModelType =
    selectedModelType || (supportedModels[0]?.model_type as string | undefined) || '';

  const selectedModelInfo = useMemo(
    () => supportedModels.find((m) => String(m.model_type) === effectiveModelType),
    [supportedModels, effectiveModelType]
  );

  // Gold-standard cohorts (#967) get a brand selector so all per-brand serving
  // bundles are reachable. Trust the backend's authoritative `is_gold_standard`
  // flag when present; fall back to the known cohort set otherwise.
  const isGoldStandard = useMemo(() => {
    if (typeof selectedModelInfo?.is_gold_standard === 'boolean') {
      return selectedModelInfo.is_gold_standard;
    }
    return GOLDSTD_COHORTS.has(effectiveModelType);
  }, [selectedModelInfo, effectiveModelType]);

  // The HCP-grain cohort keys on hcp_id; every other cohort keys on patient_id.
  const isHcpCohort = effectiveModelType === HCP_COHORT;
  const entityLabel = isHcpCohort ? 'HCP ID' : 'Patient ID';
  const entityPlaceholder = isHcpCohort
    ? 'Enter HCP ID (e.g. HCP-NE-5678)'
    : 'Enter patient ID (e.g. patient_123)';

  // -- Derived data from live response -------------------------------------
  const features: FeatureContribution[] = useMemo(
    () => explanation?.top_features ?? [],
    [explanation]
  );
  const baseValue: number = explanation?.base_value ?? 0;

  const beeswarmData = useMemo(
    () => buildBeeswarmData(features, explanation?.patient_id ?? ''),
    [features, explanation?.patient_id]
  );

  const filteredFeatures = useMemo(() => {
    if (!searchQuery) return features;
    const query = searchQuery.toLowerCase();
    return features.filter((f) =>
      f.feature_name.toLowerCase().includes(query)
    );
  }, [features, searchQuery]);

  // -- Handlers ------------------------------------------------------------
  const handleExplain = useCallback(() => {
    const trimmed = patientId.trim();
    if (!trimmed || !effectiveModelType) return;

    resetExplain();
    // Drop any previously selected feature so the details card doesn't show
    // stale info from a prior patient/model while the new explanation loads.
    setSelectedFeature(null);
    setSubmittedPatientId(trimmed);
    const request: ExplainRequest = {
      patient_id: trimmed,
      model_type: effectiveModelType as ModelType,
      format: ExplanationFormat.TOP_K,
      top_k: DEFAULT_TOP_K,
    };
    // HCP-grain cohort: the entered value IS an hcp_id — send it explicitly so
    // Feast keys the lookup on hcp_id (backend falls back to patient_id only if
    // hcp_id is absent). See resolve_features() in src/api/routes/explain.py.
    if (isHcpCohort) {
      request.hcp_id = trimmed;
    }
    // Gold-standard cohorts select one of 3 per-brand serving bundles; legacy
    // single-model cohorts ignore brand, so only send it when it matters.
    if (isGoldStandard) {
      request.brand = brand;
    }
    runExplain(request);
  }, [patientId, effectiveModelType, isHcpCohort, isGoldStandard, brand, runExplain, resetExplain]);

  const handleRefresh = useCallback(() => {
    if (!patientId.trim()) return;
    handleExplain();
  }, [patientId, handleExplain]);

  const handleExport = useCallback(() => {
    if (!explanation) return;
    const exportData = {
      model_type: effectiveModelType,
      explanation,
      exportedAt: new Date().toISOString(),
    };
    const blob = new Blob([JSON.stringify(exportData, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.download = `${effectiveModelType || 'shap'}-${explanation.patient_id}-shap.json`;
    link.href = url;
    link.click();
    URL.revokeObjectURL(url);
  }, [explanation, effectiveModelType]);

  // -- Render --------------------------------------------------------------
  const hasExplanation = !!explanation;
  const errorMessage = explainError
    ? (explainError as { message?: string }).message || 'Failed to compute explanation'
    : null;

  return (
    <div className="container mx-auto px-4 py-8">
      {/* Header */}
      <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4 mb-6">
        <div>
          <h1 className="text-3xl font-bold mb-2">Feature Importance</h1>
          <p className="text-muted-foreground">
            SHAP values, feature importance bar charts, beeswarm plots, and force plots.
          </p>
        </div>

        <div className="flex items-center gap-3">
          <Select
            value={effectiveModelType}
            onValueChange={(v) => {
              setSelectedModelType(v);
              setSelectedFeature(null);
              resetExplain();
            }}
            disabled={isLoadingModels || supportedModels.length === 0}
          >
            <SelectTrigger className="w-[280px]">
              <SelectValue placeholder={isLoadingModels ? 'Loading models...' : 'Select a model'} />
            </SelectTrigger>
            <SelectContent>
              {supportedModels.map((model) => (
                <SelectItem key={String(model.model_type)} value={String(model.model_type)}>
                  <div className="flex items-center gap-2">
                    <span>{formatModelLabel(model)}</span>
                    {model.latest_version && (
                      <span className="text-xs text-muted-foreground">{model.latest_version}</span>
                    )}
                  </div>
                </SelectItem>
              ))}
            </SelectContent>
          </Select>

          {/* Brand selector — only the gold-standard cohorts have per-brand
              serving bundles (#967). Native <select> keeps it accessible
              (aria-label drives the name) and aligned with the model Select.
              The chosen brand PERSISTS across cohort changes (mirrors
              TimeSeries.tsx's handleCohortChange) so a user can explore one
              brand across cohorts; the selector always shows the active brand,
              so there is no hidden/stale state. */}
          {isGoldStandard && (
            <select
              aria-label="Brand"
              value={brand}
              onChange={(e) => {
                setBrand(e.target.value);
                setSelectedFeature(null);
                resetExplain();
              }}
              className="h-10 px-3 border rounded-md text-sm bg-background"
            >
              {GOLDSTD_BRANDS.map((b) => (
                <option key={b} value={b}>
                  {b}
                </option>
              ))}
            </select>
          )}

          <Button variant="outline" size="icon" onClick={handleRefresh} disabled={isExplaining || !patientId.trim()}>
            <RefreshCw className={`h-4 w-4 ${isExplaining ? 'animate-spin' : ''}`} />
          </Button>

          <Button variant="outline" onClick={handleExport} disabled={!hasExplanation}>
            <Download className="h-4 w-4 mr-2" />
            Export
          </Button>
        </div>
      </div>

      {/* Patient ID + Explain action */}
      <Card className="mb-6">
        <CardContent className="pt-6">
          <div className="flex flex-wrap items-end gap-3">
            <div className="flex-1 min-w-[240px]">
              <label
                htmlFor="patient-id-input"
                className="block text-sm font-medium mb-1"
              >
                {entityLabel}
              </label>
              <Input
                id="patient-id-input"
                placeholder={entityPlaceholder}
                value={patientId}
                onChange={(e) => setPatientId(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === 'Enter') handleExplain();
                }}
              />
            </div>
            <Button
              onClick={handleExplain}
              disabled={!patientId.trim() || !effectiveModelType || isExplaining}
            >
              {isExplaining ? (
                <>
                  <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                  Loading...
                </>
              ) : (
                'Explain'
              )}
            </Button>
          </div>

          {isModelsError && (
            <div
              role="alert"
              className="mt-3 flex items-center gap-2 text-sm text-rose-600"
            >
              <AlertCircle className="h-4 w-4" />
              Failed to load model list
            </div>
          )}

          {isExplainError && (
            <div
              role="alert"
              className="mt-3 flex items-center gap-2 text-sm text-rose-600"
            >
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

      {/* Empty state when nothing has been explained yet */}
      {!hasExplanation && !isExplaining && !isExplainError && (
        <Card className="mb-6">
          <CardContent className="pt-6">
            <div className="flex flex-col items-center justify-center py-12 text-center">
              <BarChart3 className="h-12 w-12 text-muted-foreground mb-3" />
              <h2 className="text-lg font-semibold mb-1">
                No explanation yet
              </h2>
              <p className="text-sm text-muted-foreground max-w-md">
                Enter {isHcpCohort ? 'an HCP ID' : 'a patient ID'} and click{' '}
                <strong>Explain</strong> to compute live SHAP feature
                contributions from the model.
              </p>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Model Info — only rendered once we have an explanation */}
      {hasExplanation && selectedModelInfo && (
        <Card className="mb-6">
          <CardContent className="pt-6">
            <div className="flex flex-wrap items-center justify-between gap-4">
              <div className="flex items-center gap-4">
                <div className="p-3 rounded-lg bg-primary/10">
                  <BarChart3 className="h-6 w-6 text-primary" />
                </div>
                <div>
                  <h2 className="text-xl font-semibold">{formatModelLabel(selectedModelInfo)}</h2>
                  <div className="flex items-center gap-4 text-sm text-muted-foreground mt-1">
                    {selectedModelInfo.latest_version && (
                      <>
                        <span>{selectedModelInfo.latest_version}</span>
                        <span>•</span>
                      </>
                    )}
                    <span>{features.length} features</span>
                    <span>•</span>
                    <span>Patient {explanation.patient_id}</span>
                  </div>
                </div>
              </div>
              <div className="flex items-center gap-6">
                <div className="text-center">
                  <div className="text-sm text-muted-foreground">Base Value</div>
                  <div className="text-2xl font-bold">{baseValue.toFixed(3)}</div>
                </div>
                <div className="text-center">
                  <div className="text-sm text-muted-foreground">Top Feature</div>
                  <div className="text-lg font-semibold">
                    {features[0]?.feature_name.replace(/_/g, ' ') ?? '—'}
                  </div>
                </div>
              </div>
            </div>
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
                <Badge variant="secondary">{features.length}</Badge>
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
            <CardContent className="space-y-2 max-h-[600px] overflow-y-auto">
              {filteredFeatures.map((feature) => (
                <FeatureRow
                  key={feature.feature_name}
                  feature={feature}
                  isSelected={selectedFeature?.feature_name === feature.feature_name}
                  onClick={() => setSelectedFeature(
                    selectedFeature?.feature_name === feature.feature_name ? null : feature
                  )}
                />
              ))}
              {filteredFeatures.length === 0 && searchQuery && (
                <div className="text-center py-8 text-muted-foreground">
                  No features match your search
                </div>
              )}
              {filteredFeatures.length === 0 && !searchQuery && (
                <div className="text-center py-8 text-muted-foreground text-sm">
                  Run an explanation to see feature contributions.
                </div>
              )}
            </CardContent>
          </Card>
        </div>

        {/* Right: Visualizations */}
        <div className="lg:col-span-2">
          <Tabs defaultValue="bar" className="space-y-4">
            <TabsList>
              <TabsTrigger value="bar">Bar Chart</TabsTrigger>
              <TabsTrigger value="beeswarm">Beeswarm</TabsTrigger>
              <TabsTrigger value="waterfall">Waterfall</TabsTrigger>
              <TabsTrigger value="history">History</TabsTrigger>
            </TabsList>

            <TabsContent value="bar">
              <Card>
                <CardHeader>
                  <CardTitle>Global Feature Importance</CardTitle>
                  <CardDescription>
                    Mean absolute SHAP values showing overall feature importance.
                    Positive values push the prediction higher.
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <SHAPBarChart
                    features={features}
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
                  <CardTitle>Per-Feature SHAP Contributions</CardTitle>
                  <CardDescription>
                    One dot per top feature for this patient. Color reflects the
                    direction and magnitude of the SHAP contribution
                    (blue = negative impact, red = positive impact). Position
                    along the x-axis shows the raw SHAP value.
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <SHAPBeeswarm
                    data={beeswarmData}
                    maxFeatures={8}
                    height={450}
                    // Built-in legend reads "Feature Value" which doesn't match
                    // our SHAP-direction coloring. Hide it; the CardDescription
                    // above documents what the coloring means.
                    showLegend={false}
                    showReferenceLine
                    onPointClick={(point) => {
                      const feature = features.find((f) => f.feature_name === point.feature);
                      if (feature) setSelectedFeature(feature);
                    }}
                  />
                </CardContent>
              </Card>
            </TabsContent>

            <TabsContent value="waterfall">
              <Card>
                <CardHeader>
                  <CardTitle>Individual Prediction Explanation</CardTitle>
                  <CardDescription>
                    Waterfall showing how features contribute from base value to final prediction.
                    {selectedFeature && (
                      <span className="text-primary ml-2">
                        Highlighting: {selectedFeature.feature_name.replace(/_/g, ' ')}
                      </span>
                    )}
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <SHAPWaterfall
                    baseValue={baseValue}
                    features={features}
                    maxFeatures={10}
                    height={450}
                    onBarClick={(f) => setSelectedFeature(f)}
                  />
                </CardContent>
              </Card>
            </TabsContent>

            <TabsContent value="history">
              <Card>
                <CardHeader>
                  <CardTitle>Explanation History</CardTitle>
                  <CardDescription>
                    Past SHAP explanations for {submittedPatientId
                      ? `patient ${submittedPatientId}`
                      : 'the selected patient'}.
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  {!submittedPatientId && (
                    <div className="text-sm text-muted-foreground">
                      Enter a patient ID and run an explanation to view history.
                    </div>
                  )}
                  {submittedPatientId && isLoadingHistory && (
                    <div className="flex items-center gap-2 text-sm text-muted-foreground">
                      <Loader2 className="h-4 w-4 animate-spin" />
                      Loading history...
                    </div>
                  )}
                  {submittedPatientId && isHistoryError && (
                    <div role="alert" className="flex items-center gap-2 text-sm text-rose-600">
                      <AlertCircle className="h-4 w-4" />
                      Failed to load explanation history
                    </div>
                  )}
                  {submittedPatientId &&
                    !isLoadingHistory &&
                    !isHistoryError &&
                    historyExplanations.length === 0 && (
                      <div className="text-sm text-muted-foreground">
                        No historical explanations found for this patient.
                      </div>
                    )}
                  {submittedPatientId && historyExplanations.length > 0 && (
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
                            <div className="text-sm font-mono">
                              {h.prediction_class ?? '—'}
                            </div>
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
                    <div className="text-lg font-semibold">
                      #{selectedFeature.contribution_rank}
                    </div>
                  </div>
                  <div className="bg-muted rounded-lg p-3">
                    <div className="text-xs text-muted-foreground">Current Value</div>
                    <div className="text-lg font-semibold">
                      {String(selectedFeature.feature_value)}
                    </div>
                  </div>
                  <div className="bg-muted rounded-lg p-3">
                    <div className="text-xs text-muted-foreground">SHAP Value</div>
                    <div className={cn(
                      'text-lg font-semibold',
                      selectedFeature.shap_value >= 0 ? 'text-emerald-600' : 'text-rose-600'
                    )}>
                      {selectedFeature.shap_value >= 0 ? '+' : ''}
                      {selectedFeature.shap_value.toFixed(4)}
                    </div>
                  </div>
                  <div className="bg-muted rounded-lg p-3">
                    <div className="text-xs text-muted-foreground">Direction</div>
                    <div className={cn(
                      'text-lg font-semibold capitalize',
                      selectedFeature.contribution_direction === 'positive'
                        ? 'text-emerald-600'
                        : 'text-rose-600'
                    )}>
                      {selectedFeature.contribution_direction}
                    </div>
                  </div>
                </div>
                <div className="mt-4 p-3 bg-muted/50 rounded-lg">
                  <h4 className="text-sm font-medium mb-2">Interpretation</h4>
                  <p className="text-sm text-muted-foreground">
                    This feature has a {selectedFeature.contribution_direction} impact on the model's prediction.
                    {selectedFeature.shap_value > 0
                      ? ` Higher values of "${selectedFeature.feature_name.replace(/_/g, ' ')}" tend to increase the predicted outcome.`
                      : ` Higher values of "${selectedFeature.feature_name.replace(/_/g, ' ')}" tend to decrease the predicted outcome.`
                    }
                    {' '}It is ranked #{selectedFeature.contribution_rank} in terms of importance for this prediction.
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
