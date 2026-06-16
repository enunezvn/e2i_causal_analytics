/**
 * PredictiveAnalytics Page
 * ========================
 *
 * Live-data dashboard for invoking deployed prediction models via
 * `/api/models/predict/{model_name}` and rendering the prediction,
 * confidence, and feature contributions returned by the backend.
 *
 * Backed by the predictions hooks (`useModelsStatus`, `useModelInfo`,
 * `usePredict`) — no synthetic data.
 *
 * @module pages/PredictiveAnalytics
 */

import * as React from 'react';
import { Brain, Sparkles, TrendingUp, Loader2 } from 'lucide-react';

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
import {
  useModelsStatus,
  useModelInfo,
  usePredict,
} from '@/hooks/api/use-predictions';
import type {
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
// FORM FIELD DERIVATION
// =============================================================================

type FormFieldType = 'number' | 'string' | 'category';

interface FormField {
  name: string;
  type: FormFieldType;
  /** Allowed values for a categorical field (one-hot-encoded covariate). */
  options?: string[];
}

/**
 * Derive the input form from the model's metadata.
 *
 * PRIMARY (gold-standard raw-covariate path): the live BentoML `/model_info`
 * exposes `keep_columns` (the model's RAW human inputs — e.g. disease_severity,
 * academic_hcp, geographic_region) alongside `feature_columns` (the ENCODED
 * columns the model actually scores). The prediction route forwards the raw
 * covariates as `raw_features` and the bundled FeatureBuilder one-hot-encodes
 * them server-side, so the form must collect the RAW inputs — NOT the
 * engineered columns (`geographic_region_south`, `academic_hcp__isna`, …),
 * which a human cannot meaningfully fill.
 *
 * A `keep_column` is CATEGORICAL when the encoded `feature_columns` contain
 * one-hot expansions `${kc}_<value>` (single underscore). The `${kc}__isna`
 * missingness flag uses a DOUBLE underscore and is excluded, as is the `nan`
 * placeholder the encoder emits for a missing categorical.
 *
 * FALLBACK (legacy / non-gold-standard models): use the coarse `legacySchema`
 * extracted from input_schema / metadata / features / feature_names; then the
 * raw `feature_columns` names as numeric fields; else an empty form.
 */
function deriveFormFields(
  info: ModelInfoResponse | undefined,
  legacySchema: Record<string, 'number' | 'string' | 'unknown'>
): FormField[] {
  if (!info) return [];
  const bag = info as unknown as Record<string, unknown>;

  const asStringList = (value: unknown): string[] =>
    Array.isArray(value) ? value.filter((x): x is string => typeof x === 'string') : [];

  const featureColumns = asStringList(bag['feature_columns']);
  const keepColumns = asStringList(bag['keep_columns']);

  // PRIMARY — gold-standard raw covariate inputs.
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

  // FALLBACK — legacy coarse schema (input_schema / metadata / features).
  const legacyKeys = Object.keys(legacySchema);
  if (legacyKeys.length > 0) {
    return legacyKeys.map(
      (name): FormField => ({
        // 'unknown' → string input (never coerce an ID-like value to a number).
        name,
        type: legacySchema[name] === 'number' ? 'number' : 'string',
      })
    );
  }

  // LAST RESORT — raw encoded column names as numeric inputs.
  if (featureColumns.length > 0) {
    return featureColumns.map((name): FormField => ({ name, type: 'number' }));
  }

  return [];
}

// =============================================================================
// MAIN COMPONENT
// =============================================================================

function PredictiveAnalytics() {
  // ---------------------------------------------------------------------------
  // Model status (drives the selector)
  // ---------------------------------------------------------------------------
  const modelsStatusQuery = useModelsStatus();
  const models = React.useMemo<ModelEndpointHealth[]>(
    () => modelsStatusQuery.data?.models ?? [],
    [modelsStatusQuery.data]
  );

  // Select the first available model once data arrives
  const [selectedModel, setSelectedModel] = React.useState<string>('');
  React.useEffect(() => {
    if (!selectedModel && models.length > 0) {
      setSelectedModel(models[0].model_name);
    }
  }, [models, selectedModel]);

  // ---------------------------------------------------------------------------
  // Model info (drives the form fields)
  // ---------------------------------------------------------------------------
  const modelInfoQuery = useModelInfo(selectedModel);

  /**
   * Extract the typed feature schema for the selected model.
   *
   * `GET /api/models/{name}/info` returns raw BentoML model metadata which
   * is service-specific. Try the most likely shapes in priority order:
   *   1. `info.input_schema`             -> Record<string, type>
   *   2. `info.metadata.input_schema`    -> Record<string, type>
   *   3. `info.features`                 -> string[] | Record<string, type>
   *   4. `info.metadata.feature_names`   -> string[]
   *   5. `info.metadata.features`        -> string[] | Record<string, type>
   *
   * Returns a Record keyed by feature name with a coarse type tag
   * ('number' | 'string' | 'unknown'). Empty record if nothing usable.
   */
  const featureSchema = React.useMemo<
    Record<string, 'number' | 'string' | 'unknown'>
  >(() => {
    const info = modelInfoQuery.data;
    if (!info) return {};

    const meta = (info.metadata ?? {}) as Record<string, unknown>;
    // `info` is typed as ModelInfoResponse, but backend forwards raw
    // metadata so top-level `features` may exist too. Cast to record for
    // safe property probing without losing the structured type elsewhere.
    const infoBag = info as unknown as Record<string, unknown>;

    const classify = (raw: unknown): 'number' | 'string' | 'unknown' => {
      if (typeof raw !== 'string') return 'unknown';
      const t = raw.toLowerCase();
      if (t === 'number' || t === 'float' || t === 'int' || t === 'integer') {
        return 'number';
      }
      if (t === 'string' || t === 'str' || t === 'text') {
        return 'string';
      }
      return 'unknown';
    };

    const fromRecord = (
      rec: Record<string, unknown>
    ): Record<string, 'number' | 'string' | 'unknown'> => {
      const out: Record<string, 'number' | 'string' | 'unknown'> = {};
      for (const key of Object.keys(rec)) {
        out[key] = classify(rec[key]);
      }
      return out;
    };

    const fromList = (
      list: unknown[]
    ): Record<string, 'number' | 'string' | 'unknown'> => {
      const out: Record<string, 'number' | 'string' | 'unknown'> = {};
      for (const name of list) {
        if (typeof name === 'string') out[name] = 'unknown';
      }
      return out;
    };

    // 1) Top-level info.input_schema
    if (
      info.input_schema &&
      typeof info.input_schema === 'object' &&
      !Array.isArray(info.input_schema)
    ) {
      return fromRecord(info.input_schema as Record<string, unknown>);
    }

    // 2) metadata.input_schema
    const metaInputSchema = meta['input_schema'];
    if (
      metaInputSchema &&
      typeof metaInputSchema === 'object' &&
      !Array.isArray(metaInputSchema)
    ) {
      return fromRecord(metaInputSchema as Record<string, unknown>);
    }

    // 3) Top-level info.features (string[] OR record) — used by the
    //    backend test fixture and BentoML model_info passthrough.
    const topFeatures = infoBag['features'];
    if (Array.isArray(topFeatures)) {
      return fromList(topFeatures);
    }
    if (
      topFeatures &&
      typeof topFeatures === 'object' &&
      !Array.isArray(topFeatures)
    ) {
      return fromRecord(topFeatures as Record<string, unknown>);
    }

    // 4) metadata.feature_names (string[])
    const featureNames = meta['feature_names'];
    if (Array.isArray(featureNames)) {
      return fromList(featureNames);
    }

    // 5) metadata.features (string[] OR record)
    const metaFeatures = meta['features'];
    if (Array.isArray(metaFeatures)) {
      return fromList(metaFeatures);
    }
    if (
      metaFeatures &&
      typeof metaFeatures === 'object' &&
      !Array.isArray(metaFeatures)
    ) {
      return fromRecord(metaFeatures as Record<string, unknown>);
    }

    return {};
  }, [modelInfoQuery.data]);

  // The typed input form. Prefers the gold-standard raw covariates
  // (keep_columns), falling back to the legacy coarse schema.
  const formFields = React.useMemo<FormField[]>(
    () => deriveFormFields(modelInfoQuery.data, featureSchema),
    [modelInfoQuery.data, featureSchema]
  );

  // Feature values keyed by feature name
  const [featureValues, setFeatureValues] = React.useState<Record<string, string>>({});

  // ---------------------------------------------------------------------------
  // Prediction mutation
  // ---------------------------------------------------------------------------
  const predictMutation = usePredict();

  // Reset feature inputs AND any stale prediction when the model changes,
  // so the previous model's result is not surfaced under the new model.
  React.useEffect(() => {
    setFeatureValues({});
    predictMutation.reset();
    // predictMutation is intentionally excluded — React Query returns a new
    // mutation object identity on every render which would cause a loop.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedModel]);

  const handleFeatureChange = (key: string, value: string) => {
    setFeatureValues((prev) => ({ ...prev, [key]: value }));
  };

  // The gold-standard raw-covariate path (and the legacy positional path) BOTH
  // require every declared input to be present — the served FeatureBuilder
  // rejects a row that omits a keep_column. Gate the Run button on completeness
  // so we never send an incomplete row that the service would reject.
  const allFieldsFilled =
    formFields.length > 0 &&
    formFields.every((field) => (featureValues[field.name] ?? '') !== '');

  const handleRunPrediction = () => {
    if (!selectedModel || !allFieldsFilled) return;

    // Build the raw covariate dict by declared field type. Categorical/string
    // values are sent verbatim (the server-side FeatureBuilder one-hot-encodes
    // them); numeric fields are coerced to numbers. An empty field is omitted
    // (the completeness gate above means this only trims accidental blanks).
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
      // Populate "Feature Contributions" with REAL per-prediction SHAP — the
      // backend delegates to the BentoML /shap endpoint for this exact input.
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

  return (
    <div className="container mx-auto px-4 py-8">
      {/* Header */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 mb-6">
        <div>
          <h1 className="text-3xl font-bold mb-2">Predictive Analytics</h1>
          <p className="text-muted-foreground">
            Run live predictions against deployed models and inspect prediction,
            confidence, and feature contributions.
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
              disabled={isPredicting}
            />
          )}
        </div>
      </div>

      {/* Models error */}
      {modelsError && (
        <div className="mb-6">
          <QueryErrorState
            error={modelsError}
            onRetry={() => modelsStatusQuery.refetch()}
            isRetrying={modelsStatusQuery.isFetching}
          />
        </div>
      )}

      {/* Empty state */}
      {!isLoadingModels && !modelsError && models.length === 0 && (
        <Card>
          <CardContent className="p-8 text-center">
            <Brain className="h-10 w-10 mx-auto mb-3 text-muted-foreground" />
            <p className="font-medium">No models available</p>
            <p className="text-sm text-muted-foreground mt-1">
              No prediction models are currently registered. Deploy a model to
              get started.
            </p>
          </CardContent>
        </Card>
      )}

      {/* Selected model summary */}
      {selectedModel && (
        <Card className="mb-6 bg-gradient-to-r from-blue-50 to-indigo-50 dark:from-blue-900/20 dark:to-indigo-900/20">
          <CardContent className="p-4">
            <div className="flex items-center justify-between gap-4">
              <div className="flex items-center gap-3">
                <div className="p-2 rounded-lg bg-blue-100 dark:bg-blue-900/30">
                  <Brain className="h-6 w-6 text-blue-600" />
                </div>
                <div>
                  <p className="text-sm text-muted-foreground">Active Model</p>
                  <p className="font-semibold">{selectedModel}</p>
                  {modelInfoQuery.data?.version && (
                    <p className="text-xs text-muted-foreground">
                      v{modelInfoQuery.data.version}
                    </p>
                  )}
                </div>
              </div>
              {(() => {
                const modelHealth = models.find(
                  (m) => m.model_name === selectedModel
                );
                const healthStatus =
                  modelHealth?.status === 'healthy'
                    ? 'healthy'
                    : modelHealth?.status === 'unhealthy'
                      ? 'critical'
                      : 'warning';
                return (
                  <StatusBadge
                    status={healthStatus}
                    label={modelHealth?.status ?? 'unknown'}
                  />
                );
              })()}
            </div>
          </CardContent>
        </Card>
      )}

      {/* Form + Prediction Result */}
      {selectedModel && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Feature input form */}
          <Card>
            <CardHeader>
              <CardTitle>Input Features</CardTitle>
              <CardDescription>
                Fields derived from the model&apos;s declared input schema
              </CardDescription>
            </CardHeader>
            <CardContent>
              {modelInfoQuery.isLoading && (
                <div className="flex items-center gap-2 text-sm text-muted-foreground py-4">
                  <Loader2 className="h-4 w-4 animate-spin" />
                  <span>Loading model schema...</span>
                </div>
              )}
              {modelInfoQuery.error && (
                <QueryErrorState
                  error={modelInfoQuery.error}
                  onRetry={() => modelInfoQuery.refetch()}
                  isRetrying={modelInfoQuery.isFetching}
                />
              )}
              {!modelInfoQuery.isLoading &&
                !modelInfoQuery.error &&
                formFields.length === 0 && (
                  <p className="text-sm text-muted-foreground py-4">
                    No input schema available for this model.
                  </p>
                )}
              {formFields.length > 0 && (
                <form
                  className="space-y-3"
                  onSubmit={(e) => {
                    e.preventDefault();
                    handleRunPrediction();
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
                          <SelectTrigger
                            id={`feature-${field.name}`}
                            aria-label={field.name}
                          >
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

                  <Button
                    type="submit"
                    className="w-full mt-4"
                    disabled={isPredicting || !selectedModel || !allFieldsFilled}
                  >
                    {isPredicting ? (
                      <>
                        <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                        Running...
                      </>
                    ) : (
                      <>
                        <Sparkles className="h-4 w-4 mr-2" />
                        Run Prediction
                      </>
                    )}
                  </Button>
                </form>
              )}
            </CardContent>
          </Card>

          {/* Prediction result */}
          <Card>
            <CardHeader>
              <CardTitle>Prediction Result</CardTitle>
              <CardDescription>
                Live output from /api/models/predict/{selectedModel}
              </CardDescription>
            </CardHeader>
            <CardContent>
              {predictionError && (
                <QueryErrorState
                  error={predictionError}
                  onRetry={handleRunPrediction}
                  isRetrying={isPredicting}
                />
              )}
              {/* Empty state — only when there's no error AND no prior result */}
              {!predictionError && !prediction && (
                <p className="text-sm text-muted-foreground py-4">
                  Submit features above to run a prediction.
                </p>
              )}
              {/* Suppress stale result if a fresh attempt errored — */}
              {/* otherwise the user sees an error AND a prior result */}
              {!predictionError && prediction && (
                <div className="space-y-4">
                  <div>
                    <p className="text-xs text-muted-foreground mb-1">
                      Prediction
                    </p>
                    <p className="text-2xl font-bold">
                      {formatPrediction(prediction.prediction)}
                    </p>
                  </div>

                  {typeof prediction.confidence === 'number' && (
                    <div>
                      <div className="flex items-center justify-between text-xs mb-1">
                        <span className="text-muted-foreground">Confidence</span>
                        <span className="font-medium">
                          {(prediction.confidence * 100).toFixed(1)}%
                        </span>
                      </div>
                      <Progress
                        value={prediction.confidence * 100}
                        className="h-2"
                      />
                    </div>
                  )}

                  {prediction.feature_importance &&
                    Object.keys(prediction.feature_importance).length > 0 &&
                    (() => {
                      // These are signed SHAP contributions (log-odds), NOT
                      // percentages — render them as raw decimals (matching the
                      // app's SHAP components) so they are not misread against the
                      // Confidence % above. The bar shows each contribution's
                      // MAGNITUDE relative to the largest in this prediction (no
                      // *100, no hard cap that would truncate values > 1).
                      const contributions = Object.entries(
                        prediction.feature_importance
                      ).sort(([, a], [, b]) => Math.abs(b) - Math.abs(a));
                      const maxAbs = Math.max(
                        ...contributions.map(([, v]) => Math.abs(v)),
                        Number.EPSILON
                      );
                      return (
                        <div>
                          <div className="flex items-center gap-2 mb-1">
                            <TrendingUp className="h-4 w-4 text-emerald-600" />
                            <p className="text-sm font-medium">
                              Feature Contributions
                            </p>
                          </div>
                          <p className="text-xs text-muted-foreground mb-2">
                            Signed SHAP contributions (log-odds) to this prediction
                          </p>
                          <div className="space-y-2">
                            {contributions.map(([feature, impact]) => (
                              <div key={feature} className="space-y-1">
                                <div className="flex items-center justify-between text-xs">
                                  <span className="font-mono">{feature}</span>
                                  <span
                                    className={
                                      impact >= 0
                                        ? 'text-emerald-600 font-medium'
                                        : 'text-rose-600 font-medium'
                                    }
                                  >
                                    {impact >= 0 ? '+' : ''}
                                    {impact.toFixed(3)}
                                  </span>
                                </div>
                                <Progress
                                  value={(Math.abs(impact) / maxAbs) * 100}
                                  className="h-1.5"
                                />
                              </div>
                            ))}
                          </div>
                        </div>
                      );
                    })()}

                  <div className="grid grid-cols-2 gap-3 pt-2 border-t text-xs">
                    <div>
                      <p className="text-muted-foreground">Latency</p>
                      <p className="font-medium">
                        {prediction.latency_ms.toFixed(0)} ms
                      </p>
                    </div>
                    {prediction.model_version && (
                      <div>
                        <p className="text-muted-foreground">Version</p>
                        <p className="font-medium">{prediction.model_version}</p>
                      </div>
                    )}
                  </div>
                </div>
              )}
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
  if (prediction === null || prediction === undefined) {
    return '-';
  }
  if (typeof prediction === 'object') {
    return JSON.stringify(prediction);
  }
  return String(prediction);
}

export default PredictiveAnalytics;
