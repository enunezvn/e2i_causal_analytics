/**
 * PredictiveAnalytics Page
 * ========================
 *
 * DATA-DRIVEN population view. Instead of asking the user to hand-type one
 * feature row, the page scores the selected model's OWN out-of-sample holdout
 * cohort (loaded server-side from the model's FeatureBuilder) and presents a
 * RANKED list of targets + a probability distribution, labeled by what the
 * rows ARE (patients vs prescribers, from the model's cohort). Clicking a
 * ranked row drills down into that real target's per-feature SHAP
 * contributions. A what-if tool scores a hypothetical profile and gets its
 * own strategic interpretation (inputs, score vs cohort mean, how to use it).
 *
 * Backed by the predictions hooks (`useModelsStatus`, `useModelInfo`,
 * `useScoreCohort` + `usePollCohortScore`, `usePredict`) — no synthetic data
 * fabricated in the UI; provenance (synthetic holdout, out-of-sample) is labeled.
 *
 * @module pages/PredictiveAnalytics
 */

import * as React from 'react';
import { Brain, Sparkles, TrendingUp, Loader2, Users, Target } from 'lucide-react';

import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import { Progress } from '@/components/ui/progress';
import { QueryErrorState } from '@/components/ui/query-error-state';
import { StatusBadge } from '@/components/visualizations/dashboard/StatusBadge';
import { StrategicInsightCard } from '@/components/insights';
import { usePageChatContext } from '@/providers/E2ICopilotProvider';
import {
  useModelsStatus,
  useModelInfo,
  usePredict,
  useScoreCohort,
  usePollCohortScore,
} from '@/hooks/api/use-predictions';
import { usePredictiveCohortInsight, usePredictiveWhatIfInsight } from '@/hooks/api';
import type {
  CohortScoredRow,
  ModelEndpointHealth,
  ModelInfoResponse,
  PredictionRequest,
} from '@/types/predictions';

// =============================================================================
// COHORT FACETS (what the scored rows ARE + what the model predicts)
// =============================================================================

// The raw entity ids (scvpt_*/scvhcp_*) don't say whether targets are patients
// or prescribers — the model name's cohort prefix does. Mirrors the backend's
// insights/predictive_cohort facets.
interface CohortFacets {
  singular: string;
  plural: string;
  outcome: string;
}

const COHORT_FACETS: Record<string, CohortFacets> = {
  hcp_adoption: {
    singular: 'HCP',
    plural: 'HCPs (prescribers)',
    outcome: 'adopting the brand (intent to prescribe)',
  },
  persistence: {
    singular: 'patient',
    plural: 'patients',
    outcome: 'staying on therapy at 180 days',
  },
  initiation: { singular: 'patient', plural: 'patients', outcome: 'starting treatment' },
  discontinuation: {
    singular: 'patient',
    plural: 'patients',
    outcome: 'discontinuing therapy within 180 days',
  },
};

const NEUTRAL_FACETS: CohortFacets = {
  singular: 'entity',
  plural: 'entities',
  outcome: 'the targeted outcome',
};

function facetsForModel(modelName: string, cohort?: string | null): CohortFacets {
  const key =
    (cohort && cohort in COHORT_FACETS ? cohort : undefined) ??
    Object.keys(COHORT_FACETS).find((c) => modelName.toLowerCase().startsWith(c));
  return key ? COHORT_FACETS[key] : NEUTRAL_FACETS;
}

// =============================================================================
// MODEL SELECTOR (page-local; not exported)
// =============================================================================

interface PredictiveAnalyticsModelSelectorProps {
  models: ModelEndpointHealth[];
  value: string;
  onChange: (modelName: string) => void;
  disabled?: boolean;
}

function PredictiveAnalyticsModelSelector({
  models,
  value,
  onChange,
  disabled,
}: PredictiveAnalyticsModelSelectorProps) {
  return (
    <Select value={value} onValueChange={onChange} disabled={disabled}>
      <SelectTrigger className="w-[260px]" aria-label="Model">
        <SelectValue placeholder="Select a model" />
      </SelectTrigger>
      <SelectContent>
        {models.map((model) => (
          <SelectItem key={model.model_name} value={model.model_name}>
            {model.model_name}
          </SelectItem>
        ))}
      </SelectContent>
    </Select>
  );
}

// =============================================================================
// FORM FIELD DERIVATION (Advanced what-if)
// =============================================================================

type FormFieldType = 'number' | 'string' | 'category';

interface FormField {
  name: string;
  type: FormFieldType;
  /** Allowed values for a categorical field (one-hot-encoded covariate). */
  options?: string[];
  /** DGP-grounded numeric guidance from the backend's curated input_fields. */
  min?: number;
  max?: number;
  step?: number;
  hint?: string;
}

/**
 * Derive the what-if input form from the model's metadata.
 *
 * PRIORITY 0: the backend's curated, brand/cohort-appropriate `input_fields`
 * (routes/predictions.build_curated_input_fields) — the SSOT for clinically
 * coherent features + choices (e.g. Kisqali -> oncology). Falls back to the
 * gold-standard raw covariates (`keep_columns`), then the legacy coarse schema,
 * then the raw encoded columns.
 */
function deriveFormFields(
  info: ModelInfoResponse | undefined,
  legacySchema: Record<string, 'number' | 'string' | 'unknown'>
): FormField[] {
  if (!info) return [];
  const bag = info as unknown as Record<string, unknown>;

  const asStringList = (value: unknown): string[] =>
    Array.isArray(value) ? value.filter((x): x is string => typeof x === 'string') : [];

  const asFiniteNumber = (value: unknown): number | undefined =>
    typeof value === 'number' && Number.isFinite(value) ? value : undefined;

  const curatedFields = Array.isArray(bag['input_fields'])
    ? (bag['input_fields'] as Array<Record<string, unknown>>)
        .filter((f) => typeof f?.['name'] === 'string')
        .map((f): FormField => {
          const name = f['name'] as string;
          const choices = asStringList(f['choices']);
          if (f['type'] === 'category' && choices.length > 0) {
            return { name, type: 'category', options: choices };
          }
          return {
            name,
            type: f['type'] === 'number' ? 'number' : 'string',
            min: asFiniteNumber(f['min']),
            max: asFiniteNumber(f['max']),
            step: asFiniteNumber(f['step']),
            hint: typeof f['hint'] === 'string' ? f['hint'] : undefined,
          };
        })
    : [];
  if (curatedFields.length > 0) return curatedFields;

  const featureColumns = asStringList(bag['feature_columns']);
  const keepColumns = asStringList(bag['keep_columns']);

  if (keepColumns.length > 0) {
    return keepColumns.map((kc): FormField => {
      const singlePrefix = `${kc}_`;
      const doublePrefix = `${kc}__`;
      const options = featureColumns
        .filter((f) => f.startsWith(singlePrefix) && !f.startsWith(doublePrefix))
        .map((f) => f.slice(singlePrefix.length))
        .filter((opt) => opt !== '' && opt !== 'nan');
      return options.length > 0
        ? { name: kc, type: 'category', options }
        : { name: kc, type: 'number' };
    });
  }

  const legacyKeys = Object.keys(legacySchema);
  if (legacyKeys.length > 0) {
    return legacyKeys.map(
      (name): FormField => ({
        name,
        type: legacySchema[name] === 'number' ? 'number' : 'string',
      })
    );
  }

  if (featureColumns.length > 0) {
    return featureColumns.map((name): FormField => ({ name, type: 'number' }));
  }

  return [];
}

// =============================================================================
// FEATURE CONTRIBUTIONS (shared SHAP rendering)
// =============================================================================

function FeatureContributions({ importance }: { importance: Record<string, number> }) {
  const contributions = Object.entries(importance).sort(
    ([, a], [, b]) => Math.abs(b) - Math.abs(a)
  );
  if (contributions.length === 0) return null;
  const maxAbs = Math.max(...contributions.map(([, v]) => Math.abs(v)), Number.EPSILON);
  return (
    <div>
      <div className="flex items-center gap-2 mb-1">
        <TrendingUp className="h-4 w-4 text-emerald-600" />
        <p className="text-sm font-medium">Feature Contributions</p>
      </div>
      <p className="text-xs text-muted-foreground mb-2">
        Signed SHAP contributions (log-odds) to this prediction
      </p>
      <div className="space-y-2">
        {contributions.map(([feature, impact]) => (
          <div key={feature} className="space-y-1">
            <div className="flex items-center justify-between text-xs">
              <span className="font-mono">{feature}</span>
              <span className={impact >= 0 ? 'text-emerald-600 font-medium' : 'text-rose-600 font-medium'}>
                {impact >= 0 ? '+' : ''}
                {impact.toFixed(3)}
              </span>
            </div>
            <Progress value={(Math.abs(impact) / maxAbs) * 100} className="h-1.5" />
          </div>
        ))}
      </div>
    </div>
  );
}

// =============================================================================
// COHORT DISTRIBUTION (simple histogram bars)
// =============================================================================

function CohortDistribution({
  binEdges,
  binCounts,
  mean,
}: {
  binEdges: number[];
  binCounts: number[];
  mean: number;
}) {
  const maxCount = Math.max(...binCounts, 1);
  return (
    <div>
      <div className="flex items-center justify-between text-xs mb-2">
        <span className="text-muted-foreground">Predicted probability distribution</span>
        <span className="font-medium">mean {(mean * 100).toFixed(1)}%</span>
      </div>
      <div className="flex items-end gap-1 h-24" role="img" aria-label="Probability distribution">
        {binCounts.map((count, i) => (
          <div
            key={i}
            className="flex-1 flex flex-col items-center justify-end"
            title={`${(binEdges[i] * 100).toFixed(0)}–${(binEdges[i + 1] * 100).toFixed(0)}%: ${count}`}
          >
            <div
              className="w-full bg-blue-500/70 rounded-t"
              style={{ height: `${(count / maxCount) * 100}%` }}
            />
          </div>
        ))}
      </div>
      <div className="flex justify-between text-[10px] text-muted-foreground mt-1">
        <span>0%</span>
        <span>50%</span>
        <span>100%</span>
      </div>
    </div>
  );
}

// =============================================================================
// MAIN COMPONENT
// =============================================================================

function PredictiveAnalytics() {
  const modelsStatusQuery = useModelsStatus();
  const models = React.useMemo<ModelEndpointHealth[]>(
    () => modelsStatusQuery.data?.models ?? [],
    [modelsStatusQuery.data]
  );

  const [selectedModel, setSelectedModel] = React.useState<string>('');
  React.useEffect(() => {
    if (!selectedModel && models.length > 0) {
      setSelectedModel(models[0].model_name);
    }
  }, [models, selectedModel]);

  const modelInfoQuery = useModelInfo(selectedModel);

  // ---------------------------------------------------------------------------
  // Cohort scoring (primary, data-driven flow)
  // ---------------------------------------------------------------------------
  const scoreCohortMutation = useScoreCohort();
  const [cohortJobId, setCohortJobId] = React.useState<string | null>(null);
  const cohortQuery = usePollCohortScore(selectedModel, cohortJobId);
  const cohort = cohortQuery.data;

  const [selectedRow, setSelectedRow] = React.useState<CohortScoredRow | null>(null);
  const predictMutation = usePredict();

  // Strategic interpretation (agentic read of the scored cohort).
  const predInsight = usePredictiveCohortInsight();

  // What-if tool (hypothetical row) + its own per-row interpretation.
  const [showAdvanced, setShowAdvanced] = React.useState(false);
  const [featureValues, setFeatureValues] = React.useState<Record<string, string>>({});
  const [lastRunWasWhatIf, setLastRunWasWhatIf] = React.useState(false);
  const whatIfInsight = usePredictiveWhatIfInsight();

  // Reset everything when the model changes.
  React.useEffect(() => {
    setCohortJobId(null);
    setSelectedRow(null);
    setFeatureValues({});
    setShowAdvanced(false);
    setLastRunWasWhatIf(false);
    scoreCohortMutation.reset();
    predictMutation.reset();
    predInsight.reset();
    whatIfInsight.reset();
    // mutations excluded — new identity each render would loop.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedModel]);

  // What the ranked rows ARE (patients vs prescribers) + the predicted outcome,
  // from the completed job's cohort or the model name.
  const facets = facetsForModel(selectedModel, cohort?.cohort ?? null);
  const singularTitle = facets.singular.charAt(0).toUpperCase() + facets.singular.slice(1);

  const handleScoreCohort = () => {
    if (!selectedModel) return;
    setSelectedRow(null);
    setLastRunWasWhatIf(false);
    predictMutation.reset();
    // The interpretations are grounded in the previous scoring run — clear them.
    predInsight.reset();
    whatIfInsight.reset();
    scoreCohortMutation.mutate(
      { modelName: selectedModel, topN: 100 },
      { onSuccess: (data) => setCohortJobId(data.job_id) }
    );
  };

  // Generate the strategic interpretation from the REAL scored cohort:
  // top targets, distribution mean, and the backend's cohort-level drivers
  // (mean |SHAP| over the top-ranked rows). No drivers are fabricated when
  // the driver aggregation was unavailable.
  const handleGenerateInsight = () => {
    const rows = cohort?.top_rows;
    if (!rows?.length) return;
    predInsight.mutate({
      model_version: selectedModel,
      n_scored: cohort?.n_scored ?? rows.length,
      mean_prob: cohort?.distribution?.mean ?? 0,
      top_targets: rows
        .slice(0, 5)
        .map((row) => ({ entity_id: row.entity_id, probability: row.probability })),
      top_drivers: (cohort?.top_drivers ?? []).map((d) => ({
        feature: d.feature,
        importance: d.importance,
      })),
    });
  };

  // Drill into a ranked row: re-score that real entity to surface SHAP.
  const handleSelectRow = (row: CohortScoredRow) => {
    setSelectedRow(row);
    setShowAdvanced(false);
    setLastRunWasWhatIf(false);
    whatIfInsight.reset();
    predictMutation.mutate({
      modelName: selectedModel,
      request: {
        features: row.covariates,
        return_probabilities: true,
        return_feature_importance: true,
      },
    });
  };

  const featureSchema = React.useMemo<Record<string, 'number' | 'string' | 'unknown'>>(() => {
    const info = modelInfoQuery.data;
    if (!info) return {};
    const meta = (info.metadata ?? {}) as Record<string, unknown>;
    const infoBag = info as unknown as Record<string, unknown>;
    const classify = (raw: unknown): 'number' | 'string' | 'unknown' => {
      if (typeof raw !== 'string') return 'unknown';
      const t = raw.toLowerCase();
      if (t === 'number' || t === 'float' || t === 'int' || t === 'integer') return 'number';
      if (t === 'string' || t === 'str' || t === 'text') return 'string';
      return 'unknown';
    };
    const fromRecord = (rec: Record<string, unknown>) => {
      const out: Record<string, 'number' | 'string' | 'unknown'> = {};
      for (const key of Object.keys(rec)) out[key] = classify(rec[key]);
      return out;
    };
    const fromList = (list: unknown[]) => {
      const out: Record<string, 'number' | 'string' | 'unknown'> = {};
      for (const name of list) if (typeof name === 'string') out[name] = 'unknown';
      return out;
    };
    if (info.input_schema && typeof info.input_schema === 'object' && !Array.isArray(info.input_schema)) {
      return fromRecord(info.input_schema as Record<string, unknown>);
    }
    const metaInputSchema = meta['input_schema'];
    if (metaInputSchema && typeof metaInputSchema === 'object' && !Array.isArray(metaInputSchema)) {
      return fromRecord(metaInputSchema as Record<string, unknown>);
    }
    const topFeatures = infoBag['features'];
    if (Array.isArray(topFeatures)) return fromList(topFeatures);
    if (topFeatures && typeof topFeatures === 'object' && !Array.isArray(topFeatures)) {
      return fromRecord(topFeatures as Record<string, unknown>);
    }
    const featureNames = meta['feature_names'];
    if (Array.isArray(featureNames)) return fromList(featureNames);
    const metaFeatures = meta['features'];
    if (Array.isArray(metaFeatures)) return fromList(metaFeatures);
    if (metaFeatures && typeof metaFeatures === 'object' && !Array.isArray(metaFeatures)) {
      return fromRecord(metaFeatures as Record<string, unknown>);
    }
    return {};
  }, [modelInfoQuery.data]);

  const formFields = React.useMemo<FormField[]>(
    () => deriveFormFields(modelInfoQuery.data, featureSchema),
    [modelInfoQuery.data, featureSchema]
  );

  const handleFeatureChange = (key: string, value: string) => {
    setFeatureValues((prev) => ({ ...prev, [key]: value }));
  };

  const allFieldsFilled =
    formFields.length > 0 && formFields.every((f) => (featureValues[f.name] ?? '') !== '');

  const handleRunWhatIf = () => {
    if (!selectedModel || !allFieldsFilled) return;
    setSelectedRow(null);
    setLastRunWasWhatIf(true);
    whatIfInsight.reset();
    const features: Record<string, unknown> = {};
    for (const field of formFields) {
      const raw = featureValues[field.name];
      if (raw === undefined || raw === '') continue;
      if (field.type === 'number') {
        const asNumber = Number(raw);
        features[field.name] = Number.isFinite(asNumber) ? asNumber : raw;
      } else {
        features[field.name] = raw;
      }
    }
    const request: PredictionRequest = {
      features,
      return_probabilities: true,
      return_feature_importance: true,
    };
    predictMutation.mutate(
      { modelName: selectedModel, request },
      {
        // Auto-generate the what-if interpretation from THIS result — the
        // cohort-level Strategic Interpretation deliberately does not cover
        // hypothetical rows. Grounded only in the returned score/SHAP; no
        // probability -> no insight (never a read on an unknown score).
        onSuccess: (data) => {
          const probability =
            data.probabilities?.positive_class ??
            (typeof data.prediction === 'number' &&
            data.prediction >= 0 &&
            data.prediction <= 1
              ? data.prediction
              : null);
          if (probability == null) return;
          whatIfInsight.mutate({
            model_version: selectedModel,
            features,
            probability,
            confidence: data.confidence ?? null,
            cohort_mean: cohort?.distribution?.mean ?? null,
            n_scored: cohort?.n_scored ?? null,
            top_drivers: Object.entries(data.feature_importance ?? {})
              .sort(([, a], [, b]) => Math.abs(b) - Math.abs(a))
              .slice(0, 8)
              .map(([feature, value]) => ({ feature, importance: value })),
          });
        },
      }
    );
  };

  // Seed the what-if form from the selected ranked row so "change one
  // attribute and compare" is one edit away instead of full manual re-entry.
  const handlePrefillFromSelected = () => {
    if (!selectedRow) return;
    const next: Record<string, string> = {};
    for (const field of formFields) {
      const value = selectedRow.covariates[field.name];
      if (value !== undefined && value !== null) next[field.name] = String(value);
    }
    setFeatureValues(next);
  };

  // ---------------------------------------------------------------------------
  // Render
  // ---------------------------------------------------------------------------
  const isLoadingModels = modelsStatusQuery.isLoading;
  const modelsError = modelsStatusQuery.error;
  const prediction = predictMutation.data;
  const isPredicting = predictMutation.isPending;
  const predictionError = predictMutation.isError ? predictMutation.error : null;

  const cohortStatus = cohort?.status;
  const isScoring =
    scoreCohortMutation.isPending ||
    cohortStatus === 'pending' ||
    cohortStatus === 'running' ||
    (!!cohortJobId && !cohort);

  // Publish a compact on-screen data summary so the chat pane can generate
  // opener pills grounded in what this page is showing (usePageChatContext →
  // POST /chat/suggestions page_context).
  const pageChatSummary = React.useMemo(() => {
    const lines: string[] = [
      `Predictive Analytics page. Model: ${selectedModel || 'none selected'}.`,
    ];
    if (cohort?.status === 'completed') {
      lines.push(
        `Scored cohort on screen: ${cohort.n_scored} ${facets.plural} for ${cohort.brand}, showing the top ${cohort.top_rows?.length ?? 0} ranked by probability of ending up ${facets.outcome}.`
      );
      if (cohort.distribution?.mean != null) {
        lines.push(`Mean predicted probability: ${(cohort.distribution.mean * 100).toFixed(1)}%.`);
      }
      const drivers = cohort.top_drivers?.slice(0, 3).map((d) => d.feature);
      if (drivers?.length) lines.push(`Top cohort drivers (mean |SHAP|): ${drivers.join(', ')}.`);
    } else {
      lines.push('No cohort scored yet on this visit.');
    }
    return lines.join('\n');
  }, [selectedModel, cohort, facets]);
  usePageChatContext(pageChatSummary);

  return (
    <div className="container mx-auto px-4 py-8">
      {/* Header */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 mb-6">
        <div>
          <h1 className="text-3xl font-bold mb-2">Predictive Analytics</h1>
          <p className="text-muted-foreground">
            Score a model&apos;s real holdout cohort, rank the {facets.plural} most likely
            to end up {facets.outcome}, and drill into any {facets.singular}&apos;s feature
            contributions.
          </p>
        </div>

        <div className="flex items-center gap-2">
          {isLoadingModels && (
            <div className="flex items-center gap-2 text-sm text-muted-foreground">
              <Loader2 className="h-4 w-4 animate-spin" />
              <span>Loading models...</span>
            </div>
          )}
          {!isLoadingModels && models.length > 0 && (
            <PredictiveAnalyticsModelSelector
              models={models}
              value={selectedModel}
              onChange={setSelectedModel}
              disabled={isScoring}
            />
          )}
        </div>
      </div>

      {modelsError && (
        <div className="mb-6">
          <QueryErrorState
            error={modelsError}
            onRetry={() => modelsStatusQuery.refetch()}
            isRetrying={modelsStatusQuery.isFetching}
          />
        </div>
      )}

      {!isLoadingModels && !modelsError && models.length === 0 && (
        <Card>
          <CardContent className="p-8 text-center">
            <Brain className="h-10 w-10 mx-auto mb-3 text-muted-foreground" />
            <p className="font-medium">No models available</p>
            <p className="text-sm text-muted-foreground mt-1">
              No prediction models are currently registered. Deploy a model to get started.
            </p>
          </CardContent>
        </Card>
      )}

      {/* Selected model summary + Score cohort */}
      {selectedModel && (
        <Card className="mb-6 bg-gradient-to-r from-blue-50 to-indigo-50 dark:from-blue-900/20 dark:to-indigo-900/20">
          <CardContent className="p-4">
            <div className="flex items-center justify-between gap-4 flex-wrap">
              <div className="flex items-center gap-3">
                <div className="p-2 rounded-lg bg-blue-100 dark:bg-blue-900/30">
                  <Brain className="h-6 w-6 text-blue-600" />
                </div>
                <div>
                  <p className="text-sm text-muted-foreground">Active Model</p>
                  <p className="font-semibold">{selectedModel}</p>
                  {modelInfoQuery.data?.version && (
                    <p className="text-xs text-muted-foreground">v{modelInfoQuery.data.version}</p>
                  )}
                </div>
              </div>
              <div className="flex items-center gap-3">
                {(() => {
                  const modelHealth = models.find((m) => m.model_name === selectedModel);
                  const healthStatus =
                    modelHealth?.status === 'healthy'
                      ? 'healthy'
                      : modelHealth?.status === 'unhealthy'
                        ? 'critical'
                        : 'warning';
                  return <StatusBadge status={healthStatus} label={modelHealth?.status ?? 'unknown'} />;
                })()}
                <Button onClick={handleScoreCohort} disabled={isScoring}>
                  {isScoring ? (
                    <>
                      <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                      Scoring cohort...
                    </>
                  ) : (
                    <>
                      <Users className="h-4 w-4 mr-2" />
                      Score holdout cohort
                    </>
                  )}
                </Button>
              </div>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Cohort scoring errors */}
      {scoreCohortMutation.isError && (
        <div className="mb-6">
          <QueryErrorState error={scoreCohortMutation.error} onRetry={handleScoreCohort} isRetrying={isScoring} />
        </div>
      )}
      {cohortStatus === 'failed' && (
        <Card className="mb-6 border-rose-300">
          <CardContent className="p-4 text-sm text-rose-700">
            Cohort scoring failed: {cohort?.error ?? 'unknown error'}
          </CardContent>
        </Card>
      )}

      {/* Strategic interpretation (agentic read of the scored cohort) */}
      <div className="mb-6">
        <StrategicInsightCard
          description={`Agentic read of the scored holdout cohort of ${facets.plural} — grounded in the ranked targets, probability distribution, and cohort-level SHAP drivers. Covers the cohort only; drill-down and what-if results are per-${facets.singular} and do not change this read.`}
          insight={predInsight.data?.insight}
          keyTakeaways={predInsight.data?.key_takeaways}
          grounding={predInsight.data?.grounding}
          isLoading={predInsight.isPending}
          error={predInsight.error?.message ?? null}
          onGenerate={handleGenerateInsight}
          isFallback={predInsight.data?.is_fallback}
          provenance={predInsight.data?.provenance}
          generatedAt={predInsight.data?.generated_at}
        />
      </div>

      {/* Results: ranked targets + distribution + drill-down */}
      {selectedModel && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Ranked targets */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Target className="h-5 w-5" /> Ranked Targets
              </CardTitle>
              <CardDescription>
                {cohort?.cohort && cohort?.brand
                  ? `${cohort.brand} ${cohort.cohort.replace(/_/g, ' ')} holdout cohort — ${facets.plural} ranked by probability of ${facets.outcome}`
                  : `Score the cohort to rank ${facets.plural} by predicted probability`}
              </CardDescription>
            </CardHeader>
            <CardContent>
              {isScoring && (
                <div className="flex items-center gap-2 text-sm text-muted-foreground py-4">
                  <Loader2 className="h-4 w-4 animate-spin" />
                  <span>Scoring the holdout cohort...</span>
                </div>
              )}
              {!isScoring && !cohort && (
                <p className="text-sm text-muted-foreground py-4">
                  Click &ldquo;Score holdout cohort&rdquo; to rank this model&apos;s real
                  out-of-sample {facets.plural}.
                </p>
              )}
              {cohortStatus === 'completed' && cohort && (
                <div className="space-y-4">
                  {/* Provenance banner */}
                  <div className="rounded-md bg-amber-50 dark:bg-amber-900/20 border border-amber-200 dark:border-amber-800 px-3 py-2 text-xs text-amber-800 dark:text-amber-200">
                    Scored <span className="font-semibold">{cohort.n_scored.toLocaleString()}</span>{' '}
                    {facets.plural} · {cohort.split} split ·{' '}
                    {cohort.out_of_sample ? 'out-of-sample' : 'in-sample'} ·{' '}
                    {cohort.feature_source === 'holdout_synthetic' ? 'synthetic data' : cohort.feature_source}
                  </div>

                  {cohort.distribution && (
                    <CohortDistribution
                      binEdges={cohort.distribution.bin_edges}
                      binCounts={cohort.distribution.bin_counts}
                      mean={cohort.distribution.mean}
                    />
                  )}

                  {/* Ranked table (already sorted desc by the backend) */}
                  <div className="border rounded-md divide-y max-h-80 overflow-auto">
                    <div className="flex items-center justify-between px-3 py-2 text-xs font-medium text-muted-foreground bg-muted/50">
                      <span>
                        {singularTitle} (top {cohort.top_rows.length})
                      </span>
                      <span>Probability</span>
                    </div>
                    {cohort.top_rows.map((row) => {
                      const active = selectedRow?.entity_id === row.entity_id;
                      const meanProb = cohort.distribution?.mean;
                      return (
                        <button
                          key={row.entity_id}
                          type="button"
                          onClick={() => handleSelectRow(row)}
                          className={`w-full flex items-center justify-between gap-3 px-3 py-2 text-sm text-left hover:bg-accent ${active ? 'bg-accent' : ''}`}
                        >
                          <span className="font-mono truncate min-w-0 flex-1">{row.entity_id}</span>
                          <span className="flex items-center gap-2 shrink-0">
                            <span
                              className="relative h-2 w-24 rounded-sm bg-muted overflow-hidden"
                              aria-hidden="true"
                            >
                              <span
                                className="absolute inset-y-0 left-0 rounded-sm bg-blue-500/70"
                                style={{ width: `${row.probability * 100}%` }}
                              />
                              {typeof meanProb === 'number' && (
                                <span
                                  className="absolute inset-y-0 w-px bg-foreground/70"
                                  style={{ left: `${meanProb * 100}%` }}
                                />
                              )}
                            </span>
                            <span className="font-medium tabular-nums w-14 text-right">
                              {(row.probability * 100).toFixed(1)}%
                            </span>
                          </span>
                        </button>
                      );
                    })}
                  </div>
                  {typeof cohort.distribution?.mean === 'number' && (
                    <p className="text-[10px] text-muted-foreground">
                      Row bars show each {facets.singular}&apos;s predicted probability; the
                      tick marks the cohort mean ({(cohort.distribution.mean * 100).toFixed(1)}
                      %).
                      {(cohort.drivers_from_top_n ?? 0) > 0 && (
                        <>
                          {' '}
                          Cohort drivers for the interpretation = mean |SHAP| over the top{' '}
                          {cohort.drivers_from_top_n} targets.
                        </>
                      )}
                    </p>
                  )}
                </div>
              )}
            </CardContent>
          </Card>

          {/* Drill-down / what-if */}
          <Card>
            <CardHeader>
              <CardTitle>
                {selectedRow
                  ? `${singularTitle} ${selectedRow.entity_id}`
                  : lastRunWasWhatIf && prediction
                    ? `What-if result (hypothetical ${facets.singular})`
                    : 'Prediction Detail'}
              </CardTitle>
              <CardDescription>
                {selectedRow
                  ? `Real ${facets.singular} from the cohort — prediction + feature contributions`
                  : lastRunWasWhatIf && prediction
                    ? 'Model score for the profile you entered below'
                    : `Select a ranked ${facets.singular}, or score a hypothetical one with the what-if tool`}
              </CardDescription>
            </CardHeader>
            <CardContent>
              {predictionError && (
                <QueryErrorState
                  error={predictionError}
                  onRetry={() => selectedRow && handleSelectRow(selectedRow)}
                  isRetrying={isPredicting}
                />
              )}
              {isPredicting && (
                <div className="flex items-center gap-2 text-sm text-muted-foreground py-4">
                  <Loader2 className="h-4 w-4 animate-spin" />
                  <span>Scoring entity...</span>
                </div>
              )}
              {!predictionError && !isPredicting && !prediction && !showAdvanced && (
                <p className="text-sm text-muted-foreground py-4">
                  Click a ranked {facets.singular} on the left to see its prediction and feature
                  contributions.
                </p>
              )}
              {!predictionError && !isPredicting && prediction && (
                <div className="space-y-4">
                  <div>
                    <p className="text-xs text-muted-foreground mb-1">Prediction</p>
                    <p className="text-2xl font-bold">{formatPrediction(prediction.prediction)}</p>
                    {typeof prediction.probabilities?.positive_class === 'number' && (
                      <p className="text-xs text-muted-foreground mt-1">
                        {(prediction.probabilities.positive_class * 100).toFixed(1)}% probability
                        of {facets.outcome}
                        {typeof cohort?.distribution?.mean === 'number' && (
                          <> · cohort mean {(cohort.distribution.mean * 100).toFixed(1)}%</>
                        )}
                      </p>
                    )}
                  </div>
                  {typeof prediction.confidence === 'number' && (
                    <div>
                      <div className="flex items-center justify-between text-xs mb-1">
                        <span className="text-muted-foreground">Confidence</span>
                        <span className="font-medium">{(prediction.confidence * 100).toFixed(1)}%</span>
                      </div>
                      <Progress value={prediction.confidence * 100} className="h-2" />
                    </div>
                  )}
                  {prediction.feature_importance &&
                    Object.keys(prediction.feature_importance).length > 0 && (
                      <FeatureContributions importance={prediction.feature_importance} />
                    )}
                </div>
              )}

              {/* What-if tool: score a hypothetical profile */}
              <div className="mt-4 pt-4 border-t">
                <button
                  type="button"
                  className="text-xs text-muted-foreground hover:text-foreground underline"
                  onClick={() => setShowAdvanced((v) => !v)}
                >
                  {showAdvanced
                    ? 'Hide what-if tool'
                    : `What-if: score a hypothetical ${facets.singular}`}
                </button>
                {showAdvanced && (
                  <div className="rounded-md bg-muted/50 border px-3 py-2 text-xs text-muted-foreground space-y-1 mt-3">
                    <p>
                      <span className="font-medium text-foreground">Inputs</span> — the
                      attributes the model was trained on. Fill them in to describe a
                      hypothetical {facets.singular}
                      {selectedRow
                        ? `, or start from ${selectedRow.entity_id} and change one attribute to compare`
                        : ''}
                      .
                    </p>
                    <p>
                      <span className="font-medium text-foreground">Output</span> — the
                      model&apos;s predicted probability that this profile ends up{' '}
                      {facets.outcome}, its SHAP feature contributions, and a strategic
                      interpretation of how to use the result.
                    </p>
                    <p>
                      This is a prediction, not a causal estimate: changing an input shows how
                      the score responds, not what an intervention would achieve.
                    </p>
                  </div>
                )}
                {showAdvanced && selectedRow && formFields.length > 0 && (
                  <Button
                    type="button"
                    variant="outline"
                    size="sm"
                    className="mt-3"
                    onClick={handlePrefillFromSelected}
                  >
                    Start from {selectedRow.entity_id}
                  </Button>
                )}
                {showAdvanced && formFields.length > 0 && (
                  <form
                    className="space-y-3 mt-3"
                    onSubmit={(e) => {
                      e.preventDefault();
                      handleRunWhatIf();
                    }}
                  >
                    {formFields.map((field) => (
                      <div key={field.name} className="space-y-1">
                        <Label htmlFor={`feature-${field.name}`}>{field.name}</Label>
                        {field.type === 'category' ? (
                          <Select
                            value={featureValues[field.name] ?? ''}
                            onValueChange={(value) => handleFeatureChange(field.name, value)}
                            disabled={isPredicting}
                          >
                            <SelectTrigger id={`feature-${field.name}`} aria-label={field.name}>
                              <SelectValue placeholder={`Select ${field.name}`} />
                            </SelectTrigger>
                            <SelectContent>
                              {field.options?.map((opt) => (
                                <SelectItem key={opt} value={opt}>
                                  {opt}
                                </SelectItem>
                              ))}
                            </SelectContent>
                          </Select>
                        ) : (
                          <Input
                            id={`feature-${field.name}`}
                            type={field.type === 'number' ? 'number' : 'text'}
                            value={featureValues[field.name] ?? ''}
                            onChange={(e) => handleFeatureChange(field.name, e.target.value)}
                            placeholder={
                              field.min !== undefined && field.max !== undefined
                                ? `${field.min}–${field.max}`
                                : `Enter ${field.name}`
                            }
                            min={field.min}
                            max={field.max}
                            step={field.step}
                            disabled={isPredicting}
                          />
                        )}
                        {field.hint && (
                          <p className="text-xs text-muted-foreground">{field.hint}</p>
                        )}
                      </div>
                    ))}
                    <Button type="submit" className="w-full" disabled={isPredicting || !allFieldsFilled}>
                      <Sparkles className="h-4 w-4 mr-2" />
                      Run what-if
                    </Button>
                  </form>
                )}
                {/* Per-row interpretation of the LAST what-if run (auto-generated).
                    Separate from the cohort card above, which never covers
                    hypothetical rows. */}
                {lastRunWasWhatIf &&
                  (whatIfInsight.isPending || whatIfInsight.data || whatIfInsight.error) && (
                    <div className="mt-4">
                      <StrategicInsightCard
                        title="What-If Interpretation"
                        description={`How to read and use this hypothetical ${facets.singular}'s score`}
                        insight={whatIfInsight.data?.insight}
                        keyTakeaways={whatIfInsight.data?.key_takeaways}
                        grounding={whatIfInsight.data?.grounding}
                        isLoading={whatIfInsight.isPending}
                        error={whatIfInsight.error?.message ?? null}
                        isFallback={whatIfInsight.data?.is_fallback}
                        provenance={whatIfInsight.data?.provenance}
                        generatedAt={whatIfInsight.data?.generated_at}
                      />
                    </div>
                  )}
              </div>
            </CardContent>
          </Card>
        </div>
      )}
    </div>
  );
}

/**
 * Format the prediction payload — backend returns `unknown`, so we cover
 * the typical shapes (string, number, boolean, object).
 */
function formatPrediction(prediction: unknown): string {
  if (prediction === null || prediction === undefined) return '-';
  if (typeof prediction === 'object') return JSON.stringify(prediction);
  return String(prediction);
}

export default PredictiveAnalytics;
