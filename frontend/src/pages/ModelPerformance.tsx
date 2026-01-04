/**
 * Model Performance Page
 * ======================
 *
 * Dashboard for analyzing ML model performance with metrics,
 * confusion matrix, ROC curves, and performance trends.
 *
 * Features:
 * - Model selector dropdown
 * - Key performance metrics cards (Accuracy, Precision, Recall, F1, AUC)
 * - Confusion matrix heatmap
 * - ROC curve comparison
 * - Performance trend over time
 *
 * @module pages/ModelPerformance
 */

import { useState, useMemo } from 'react';
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
import {
  Target,
  Activity,
  RefreshCw,
  Download,
  Calendar,
  Clock,
} from 'lucide-react';
import {
  ConfusionMatrix,
  ROCCurve,
  MetricTrend,
  type ConfusionMatrixData,
  type ROCCurveData,
  type MetricDataPoint,
} from '@/components/visualizations';
import { KPICard } from '@/components/visualizations/dashboard';

// =============================================================================
// TYPES
// =============================================================================

interface ModelInfo {
  id: string;
  name: string;
  version: string;
  type: 'classification' | 'regression';
  status: 'production' | 'staging' | 'retired';
  lastTrained: string;
  lastEvaluated: string;
}

interface ModelMetrics {
  accuracy: number;
  precision: number;
  recall: number;
  f1Score: number;
  auc: number;
  logloss?: number;
  samplesEvaluated: number;
}

// =============================================================================
// SAMPLE DATA
// =============================================================================

const SAMPLE_MODELS: ModelInfo[] = [
  {
    id: 'churn-v3',
    name: 'Patient Churn Predictor',
    version: 'v3.2.1',
    type: 'classification',
    status: 'production',
    lastTrained: '2024-03-15',
    lastEvaluated: '2024-03-20',
  },
  {
    id: 'hcp-tier',
    name: 'HCP Tier Classifier',
    version: 'v2.1.0',
    type: 'classification',
    status: 'production',
    lastTrained: '2024-03-10',
    lastEvaluated: '2024-03-18',
  },
  {
    id: 'conversion-v2',
    name: 'Conversion Predictor',
    version: 'v2.0.5',
    type: 'classification',
    status: 'staging',
    lastTrained: '2024-03-18',
    lastEvaluated: '2024-03-19',
  },
  {
    id: 'adherence-v1',
    name: 'Adherence Risk Model',
    version: 'v1.4.2',
    type: 'classification',
    status: 'production',
    lastTrained: '2024-02-28',
    lastEvaluated: '2024-03-15',
  },
];

const SAMPLE_METRICS: Record<string, ModelMetrics> = {
  'churn-v3': {
    accuracy: 0.912,
    precision: 0.895,
    recall: 0.878,
    f1Score: 0.886,
    auc: 0.945,
    logloss: 0.234,
    samplesEvaluated: 15420,
  },
  'hcp-tier': {
    accuracy: 0.847,
    precision: 0.823,
    recall: 0.856,
    f1Score: 0.839,
    auc: 0.912,
    logloss: 0.312,
    samplesEvaluated: 8750,
  },
  'conversion-v2': {
    accuracy: 0.876,
    precision: 0.861,
    recall: 0.842,
    f1Score: 0.851,
    auc: 0.928,
    logloss: 0.278,
    samplesEvaluated: 12300,
  },
  'adherence-v1': {
    accuracy: 0.834,
    precision: 0.812,
    recall: 0.798,
    f1Score: 0.805,
    auc: 0.891,
    logloss: 0.345,
    samplesEvaluated: 9840,
  },
};

const SAMPLE_CONFUSION_MATRICES: Record<string, ConfusionMatrixData> = {
  'churn-v3': {
    matrix: [
      [4250, 380],
      [420, 3870],
    ],
    labels: ['Retained', 'Churned'],
  },
  'hcp-tier': {
    matrix: [
      [2150, 180, 120],
      [210, 1980, 160],
      [140, 190, 1820],
    ],
    labels: ['Tier 1', 'Tier 2', 'Tier 3'],
  },
  'conversion-v2': {
    matrix: [
      [5120, 620],
      [580, 4780],
    ],
    labels: ['Non-Converter', 'Converter'],
  },
  'adherence-v1': {
    matrix: [
      [2850, 380, 270],
      [320, 2680, 350],
      [290, 310, 2590],
    ],
    labels: ['Low Risk', 'Medium Risk', 'High Risk'],
  },
};

function generateROCPoints(baseAUC: number): { fpr: number; tpr: number }[] {
  const points: { fpr: number; tpr: number }[] = [{ fpr: 0, tpr: 0 }];
  const steps = 20;

  for (let i = 1; i < steps; i++) {
    const fpr = i / steps;
    // Generate TPR based on desired AUC
    const baseTpr = fpr + (baseAUC - 0.5) * 2 * Math.sqrt(fpr * (1 - fpr));
    const tpr = Math.min(1, Math.max(fpr, baseTpr + (Math.random() * 0.05 - 0.025)));
    points.push({ fpr, tpr });
  }

  points.push({ fpr: 1, tpr: 1 });
  return points;
}

const SAMPLE_ROC_CURVES: Record<string, ROCCurveData[]> = {
  'churn-v3': [
    { name: 'v3.2.1 (Current)', points: generateROCPoints(0.945), color: 'hsl(var(--chart-1))' },
    { name: 'v3.1.0 (Previous)', points: generateROCPoints(0.92), color: 'hsl(var(--chart-2))' },
  ],
  'hcp-tier': [
    { name: 'v2.1.0 (Current)', points: generateROCPoints(0.912), color: 'hsl(var(--chart-1))' },
    { name: 'v2.0.0 (Previous)', points: generateROCPoints(0.88), color: 'hsl(var(--chart-2))' },
  ],
  'conversion-v2': [
    { name: 'v2.0.5 (Current)', points: generateROCPoints(0.928), color: 'hsl(var(--chart-1))' },
    { name: 'v1.5.0 (Previous)', points: generateROCPoints(0.89), color: 'hsl(var(--chart-2))' },
  ],
  'adherence-v1': [
    { name: 'v1.4.2 (Current)', points: generateROCPoints(0.891), color: 'hsl(var(--chart-1))' },
    { name: 'v1.3.0 (Previous)', points: generateROCPoints(0.86), color: 'hsl(var(--chart-2))' },
  ],
};

function generateMetricHistory(): MetricDataPoint[] {
  const points: MetricDataPoint[] = [];
  const baseDate = new Date('2024-01-01');
  let value = 0.82 + Math.random() * 0.05;

  for (let week = 0; week < 12; week++) {
    const date = new Date(baseDate);
    date.setDate(date.getDate() + week * 7);

    // Add slight trend upward with noise
    value = Math.min(0.98, Math.max(0.75, value + (Math.random() * 0.04 - 0.015)));

    const annotation = week === 7 ? 'Model retrained' : undefined;
    if (week === 7) value += 0.02; // Bump after retraining

    points.push({
      timestamp: date.toISOString().split('T')[0],
      value,
      annotation,
    });
  }

  return points;
}

// =============================================================================
// HELPER COMPONENTS
// =============================================================================

function StatusBadge({ status }: { status: ModelInfo['status'] }) {
  const config = {
    production: { label: 'Production', variant: 'default' as const, className: 'bg-emerald-500' },
    staging: { label: 'Staging', variant: 'secondary' as const, className: 'bg-amber-500' },
    retired: { label: 'Retired', variant: 'outline' as const, className: 'bg-gray-500' },
  }[status];

  return (
    <Badge variant={config.variant} className={status === 'production' ? config.className : ''}>
      {config.label}
    </Badge>
  );
}

// =============================================================================
// MAIN COMPONENT
// =============================================================================

function ModelPerformance() {
  const [selectedModelId, setSelectedModelId] = useState<string>(SAMPLE_MODELS[0].id);
  const [isRefreshing, setIsRefreshing] = useState(false);

  // Get selected model data
  const selectedModel = useMemo(
    () => SAMPLE_MODELS.find((m) => m.id === selectedModelId) ?? SAMPLE_MODELS[0],
    [selectedModelId]
  );

  const metrics = useMemo(
    () => SAMPLE_METRICS[selectedModelId] ?? SAMPLE_METRICS['churn-v3'],
    [selectedModelId]
  );

  const confusionMatrix = useMemo(
    () => SAMPLE_CONFUSION_MATRICES[selectedModelId] ?? SAMPLE_CONFUSION_MATRICES['churn-v3'],
    [selectedModelId]
  );

  const rocCurves = useMemo(
    () => SAMPLE_ROC_CURVES[selectedModelId] ?? SAMPLE_ROC_CURVES['churn-v3'],
    [selectedModelId]
  );

  const accuracyHistory = useMemo(() => generateMetricHistory(), [selectedModelId]);

  const handleRefresh = async () => {
    setIsRefreshing(true);
    // Simulate API refresh
    await new Promise((resolve) => setTimeout(resolve, 1000));
    setIsRefreshing(false);
  };

  const handleExport = () => {
    const exportData = {
      model: selectedModel,
      metrics,
      confusionMatrix,
      exportedAt: new Date().toISOString(),
    };
    const blob = new Blob([JSON.stringify(exportData, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.download = `${selectedModel.id}-performance.json`;
    link.href = url;
    link.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="container mx-auto px-4 py-8">
      {/* Header */}
      <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4 mb-6">
        <div>
          <h1 className="text-3xl font-bold mb-2">Model Performance</h1>
          <p className="text-muted-foreground">
            View model metrics, confusion matrix, ROC curves, and performance trends.
          </p>
        </div>

        <div className="flex items-center gap-3">
          <Select value={selectedModelId} onValueChange={setSelectedModelId}>
            <SelectTrigger className="w-[280px]">
              <SelectValue placeholder="Select a model" />
            </SelectTrigger>
            <SelectContent>
              {SAMPLE_MODELS.map((model) => (
                <SelectItem key={model.id} value={model.id}>
                  <div className="flex items-center gap-2">
                    <span>{model.name}</span>
                    <span className="text-xs text-muted-foreground">{model.version}</span>
                  </div>
                </SelectItem>
              ))}
            </SelectContent>
          </Select>

          <Button variant="outline" size="icon" onClick={handleRefresh} disabled={isRefreshing}>
            <RefreshCw className={`h-4 w-4 ${isRefreshing ? 'animate-spin' : ''}`} />
          </Button>

          <Button variant="outline" onClick={handleExport}>
            <Download className="h-4 w-4 mr-2" />
            Export
          </Button>
        </div>
      </div>

      {/* Model Info Card */}
      <Card className="mb-6">
        <CardContent className="pt-6">
          <div className="flex flex-wrap items-center justify-between gap-4">
            <div className="flex items-center gap-4">
              <div className="p-3 rounded-lg bg-primary/10">
                <Target className="h-6 w-6 text-primary" />
              </div>
              <div>
                <div className="flex items-center gap-2">
                  <h2 className="text-xl font-semibold">{selectedModel.name}</h2>
                  <StatusBadge status={selectedModel.status} />
                </div>
                <div className="flex items-center gap-4 text-sm text-muted-foreground mt-1">
                  <span className="flex items-center gap-1">
                    <Activity className="h-4 w-4" />
                    {selectedModel.version}
                  </span>
                  <span className="flex items-center gap-1">
                    <Calendar className="h-4 w-4" />
                    Trained: {selectedModel.lastTrained}
                  </span>
                  <span className="flex items-center gap-1">
                    <Clock className="h-4 w-4" />
                    Evaluated: {selectedModel.lastEvaluated}
                  </span>
                </div>
              </div>
            </div>
            <div className="text-right">
              <div className="text-sm text-muted-foreground">Samples Evaluated</div>
              <div className="text-2xl font-bold">{metrics.samplesEvaluated.toLocaleString()}</div>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Metrics Cards */}
      <div className="grid grid-cols-2 md:grid-cols-5 gap-4 mb-6">
        <KPICard
          title="Accuracy"
          value={(metrics.accuracy * 100).toFixed(1)}
          unit="%"
          status="healthy"
        />
        <KPICard
          title="Precision"
          value={(metrics.precision * 100).toFixed(1)}
          unit="%"
          status="healthy"
        />
        <KPICard
          title="Recall"
          value={(metrics.recall * 100).toFixed(1)}
          unit="%"
          status="healthy"
        />
        <KPICard
          title="F1 Score"
          value={(metrics.f1Score * 100).toFixed(1)}
          unit="%"
          status="healthy"
        />
        <KPICard
          title="AUC-ROC"
          value={metrics.auc.toFixed(3)}
          status="healthy"
        />
      </div>

      {/* Main Visualizations */}
      <Tabs defaultValue="confusion" className="space-y-6">
        <TabsList>
          <TabsTrigger value="confusion">Confusion Matrix</TabsTrigger>
          <TabsTrigger value="roc">ROC Curve</TabsTrigger>
          <TabsTrigger value="trend">Performance Trend</TabsTrigger>
        </TabsList>

        <TabsContent value="confusion">
          <Card>
            <CardHeader>
              <CardTitle>Confusion Matrix</CardTitle>
              <CardDescription>
                Classification results showing predicted vs actual labels.
                Diagonal cells represent correct predictions.
              </CardDescription>
            </CardHeader>
            <CardContent>
              <ConfusionMatrix
                data={confusionMatrix}
                title=""
                showPercentages={false}
                cellSize={90}
                onCellClick={(actual, predicted, value) => {
                  console.log(`Cell clicked: Actual=${actual}, Predicted=${predicted}, Count=${value}`);
                }}
              />
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="roc">
          <Card>
            <CardHeader>
              <CardTitle>ROC Curve Comparison</CardTitle>
              <CardDescription>
                Receiver Operating Characteristic curve showing trade-off between
                true positive rate and false positive rate at various thresholds.
              </CardDescription>
            </CardHeader>
            <CardContent>
              <ROCCurve
                curves={rocCurves}
                height={450}
                showAUC
                showArea
                showDiagonal
              />
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="trend">
          <Card>
            <CardHeader>
              <CardTitle>Accuracy Over Time</CardTitle>
              <CardDescription>
                Weekly model accuracy trend with target threshold. Annotations show
                significant events like model retraining.
              </CardDescription>
            </CardHeader>
            <CardContent>
              <MetricTrend
                name="Model Accuracy"
                data={accuracyHistory}
                unit=""
                height={350}
                showHeader={false}
                thresholds={[
                  { value: 0.90, label: 'Target', type: 'target', color: '#22c55e' },
                  { value: 0.80, label: 'Minimum', type: 'lower', color: '#ef4444' },
                ]}
                valueFormatter={(v) => (v * 100).toFixed(1) + '%'}
                timestampFormatter={(ts) => {
                  const date = new Date(ts);
                  return date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
                }}
              />
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>

      {/* Additional Details */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mt-6">
        <Card>
          <CardHeader>
            <CardTitle className="text-base">Model Configuration</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-3 text-sm">
              <div className="flex justify-between">
                <span className="text-muted-foreground">Model Type</span>
                <span className="font-medium capitalize">{selectedModel.type}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-muted-foreground">Algorithm</span>
                <span className="font-medium">XGBoost Classifier</span>
              </div>
              <div className="flex justify-between">
                <span className="text-muted-foreground">Features Used</span>
                <span className="font-medium">47</span>
              </div>
              <div className="flex justify-between">
                <span className="text-muted-foreground">Training Samples</span>
                <span className="font-medium">125,000</span>
              </div>
              <div className="flex justify-between">
                <span className="text-muted-foreground">Cross-Validation</span>
                <span className="font-medium">5-fold Stratified</span>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="text-base">Threshold Settings</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-3 text-sm">
              <div className="flex justify-between">
                <span className="text-muted-foreground">Classification Threshold</span>
                <span className="font-medium">0.50</span>
              </div>
              <div className="flex justify-between">
                <span className="text-muted-foreground">High Confidence Threshold</span>
                <span className="font-medium">0.85</span>
              </div>
              <div className="flex justify-between">
                <span className="text-muted-foreground">Low Confidence Flag</span>
                <span className="font-medium">&lt; 0.60</span>
              </div>
              <div className="flex justify-between">
                <span className="text-muted-foreground">Accuracy Target</span>
                <span className="font-medium text-emerald-600">90%</span>
              </div>
              <div className="flex justify-between">
                <span className="text-muted-foreground">Minimum Acceptable</span>
                <span className="font-medium text-rose-600">80%</span>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}

export default ModelPerformance;
