/**
 * PredictiveAnalytics Page
 * ========================
 *
 * DATA-DRIVEN population view. Instead of asking the user to hand-type one
 * feature row, the page scores the selected model's OWN out-of-sample holdout
 * cohort (loaded server-side from the model's FeatureBuilder) and presents a
 * RANKED list of targets + a probability distribution. Clicking a ranked row
 * drills down into that real entity's per-feature SHAP contributions. A custom
 * "Advanced what-if" row is still available for hypotheticals.
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
import {
  useModelsStatus,
  useModelInfo,
  usePredict,
  useScoreCohort,
  usePollCohortScore,
} from '@/hooks/api/use-predictions';
import { usePredictiveCohortInsight } from '@/hooks/api';
import type {
  CohortScoredRow,
  ModelEndpointHealth,
  ModelInfoResponse,
  PredictionRequest,
} from '@/types/predictions';

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

  const curatedFields = Array.isArray(bag['input_fields'])
    ? (bag['input_fields'] as Array<Record<string, unknown>>)
        .filter((f) => typeof f?.['name'] === 'string')
        .map((f): FormField => {
          const name = f['name'] as string;
          const choices = asStringList(f['choices']);
          if (f['type'] === 'category' && choices.length > 0) {
            return { name, type: 'category', options: choices };
          }
          return { name, type: f['type'] === 'number' ? 'number' : 'string' };
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

  // Advanced what-if (manual row — preserved for hypotheticals)
  const [showAdvanced, setShowAdvanced] = React.useState(false);
  const [featureValues, setFeatureValues] = React.useState<Record<string, string>>({});

  // Reset everything when the model changes.
  React.useEffect(() => {
    setCohortJobId(null);
    setSelectedRow(null);
    setFeatureValues({});
    setShowAdvanced(false);
    scoreCohortMutation.reset();
    predictMutation.reset();
    // mutations excluded — new identity each render would loop.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedModel]);

  const handleScoreCohort = () => {
    if (!selectedModel) return;
    setSelectedRow(null);
    predictMutation.reset();
    scoreCohortMutation.mutate(
      { modelName: selectedModel, topN: 100 },
      { onSuccess: (data) => setCohortJobId(data.job_id) }
    );
  };

  // Generate the strategic interpretation from the REAL scored cohort.
  // Grounded only in data already on the page — top targets, distribution mean,
  // and (when a row has been drilled into) that entity's SHAP contributions.
  // No drivers are fabricated when SHAP is unavailable.
  const handleGenerateInsight = () => {
    const rows = cohort?.top_rows;
    if (!rows?.length) return;
    const importance = predictMutation.data?.feature_importance;
    const topDrivers = importance
      ? Object.entries(importance).map(([feature, value]) => ({
          feature,
          importance: value,
        }))
      : [];
    predInsight.mutate({
      model_version: selectedModel,
      n_scored: cohort?.n_scored ?? rows.length,
      mean_prob: cohort?.distribution?.mean ?? 0,
      top_targets: rows
        .slice(0, 5)
        .map((row) => ({ entity_id: row.entity_id, probability: row.probability })),
      top_drivers: topDrivers,
    });
  };

  // Drill into a ranked row: re-score that real entity to surface SHAP.
  const handleSelectRow = (row: CohortScoredRow) => {
    setSelectedRow(row);
    setShowAdvanced(false);
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
    predictMutation.mutate({ modelName: selectedModel, request });
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

  return (
    <div className="container mx-auto px-4 py-8">
      {/* Header */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 mb-6">
        <div>
          <h1 className="text-3xl font-bold mb-2">Predictive Analytics</h1>
          <p className="text-muted-foreground">
            Score a model&apos;s real holdout cohort, rank the top targets, and drill into
            any entity&apos;s feature contributions.
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
          description="Agentic read of the scored holdout cohort, grounded in the ranked targets and probability distribution"
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
                  ? `${cohort.brand} ${cohort.cohort} holdout cohort`
                  : 'Score the cohort to rank targets by predicted probability'}
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
                  out-of-sample entities.
                </p>
              )}
              {cohortStatus === 'completed' && cohort && (
                <div className="space-y-4">
                  {/* Provenance banner */}
                  <div className="rounded-md bg-amber-50 dark:bg-amber-900/20 border border-amber-200 dark:border-amber-800 px-3 py-2 text-xs text-amber-800 dark:text-amber-200">
                    Scored <span className="font-semibold">{cohort.n_scored.toLocaleString()}</span>{' '}
                    entities · {cohort.split} split ·{' '}
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
                      <span>Entity (top {cohort.top_rows.length})</span>
                      <span>Probability</span>
                    </div>
                    {cohort.top_rows.map((row) => {
                      const active = selectedRow?.entity_id === row.entity_id;
                      return (
                        <button
                          key={row.entity_id}
                          type="button"
                          onClick={() => handleSelectRow(row)}
                          className={`w-full flex items-center justify-between px-3 py-2 text-sm text-left hover:bg-accent ${active ? 'bg-accent' : ''}`}
                        >
                          <span className="font-mono truncate">{row.entity_id}</span>
                          <span className="font-medium">{(row.probability * 100).toFixed(1)}%</span>
                        </button>
                      );
                    })}
                  </div>
                </div>
              )}
            </CardContent>
          </Card>

          {/* Drill-down / what-if */}
          <Card>
            <CardHeader>
              <CardTitle>{selectedRow ? `Entity ${selectedRow.entity_id}` : 'Prediction Detail'}</CardTitle>
              <CardDescription>
                {selectedRow
                  ? 'Real entity from the cohort — prediction + feature contributions'
                  : 'Select a ranked entity, or open Advanced what-if'}
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
                  Click a ranked entity on the left to see its prediction and feature contributions.
                </p>
              )}
              {!predictionError && !isPredicting && prediction && (
                <div className="space-y-4">
                  <div>
                    <p className="text-xs text-muted-foreground mb-1">Prediction</p>
                    <p className="text-2xl font-bold">{formatPrediction(prediction.prediction)}</p>
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

              {/* Advanced what-if toggle */}
              <div className="mt-4 pt-4 border-t">
                <button
                  type="button"
                  className="text-xs text-muted-foreground hover:text-foreground underline"
                  onClick={() => setShowAdvanced((v) => !v)}
                >
                  {showAdvanced ? 'Hide advanced what-if' : 'Advanced: score a custom row (what-if)'}
                </button>
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
                            placeholder={`Enter ${field.name}`}
                            disabled={isPredicting}
                          />
                        )}
                      </div>
                    ))}
                    <Button type="submit" className="w-full" disabled={isPredicting || !allFieldsFilled}>
                      <Sparkles className="h-4 w-4 mr-2" />
                      Run what-if
                    </Button>
                  </form>
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
