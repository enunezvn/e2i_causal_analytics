/**
 * Feature Importance Page
 * =======================
 *
 * Dashboard for analyzing feature importance and SHAP explanations
 * with multiple visualization types.
 *
 * Features:
 * - Model selector dropdown
 * - Global feature importance bar chart
 * - SHAP Beeswarm plot (distribution analysis)
 * - Individual prediction waterfall explanation
 * - Feature details table
 *
 * @module pages/FeatureImportance
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
} from 'lucide-react';
import {
  SHAPBarChart,
  SHAPBeeswarm,
  SHAPWaterfall,
  type BeeswarmDataPoint,
} from '@/components/visualizations';
import type { FeatureContribution } from '@/types/explain';
import { cn } from '@/lib/utils';

// =============================================================================
// TYPES
// =============================================================================

interface ModelInfo {
  id: string;
  name: string;
  version: string;
  featureCount: number;
}

// =============================================================================
// SAMPLE DATA
// =============================================================================

const SAMPLE_MODELS: ModelInfo[] = [
  { id: 'churn-v3', name: 'Patient Churn Predictor', version: 'v3.2.1', featureCount: 47 },
  { id: 'hcp-tier', name: 'HCP Tier Classifier', version: 'v2.1.0', featureCount: 35 },
  { id: 'conversion-v2', name: 'Conversion Predictor', version: 'v2.0.5', featureCount: 42 },
];

const SAMPLE_FEATURES: Record<string, FeatureContribution[]> = {
  'churn-v3': [
    { feature_name: 'days_since_last_visit', feature_value: 45, shap_value: 0.35, contribution_direction: 'positive', contribution_rank: 1 },
    { feature_name: 'total_prescriptions_ytd', feature_value: 12, shap_value: -0.28, contribution_direction: 'negative', contribution_rank: 2 },
    { feature_name: 'territory_revenue', feature_value: 150000, shap_value: 0.22, contribution_direction: 'positive', contribution_rank: 3 },
    { feature_name: 'specialty_oncology', feature_value: 1, shap_value: 0.18, contribution_direction: 'positive', contribution_rank: 4 },
    { feature_name: 'recent_engagement_count', feature_value: 3, shap_value: -0.15, contribution_direction: 'negative', contribution_rank: 5 },
    { feature_name: 'competitor_market_share', feature_value: 0.35, shap_value: -0.12, contribution_direction: 'negative', contribution_rank: 6 },
    { feature_name: 'formulary_status_preferred', feature_value: 1, shap_value: 0.10, contribution_direction: 'positive', contribution_rank: 7 },
    { feature_name: 'hcp_experience_years', feature_value: 15, shap_value: 0.08, contribution_direction: 'positive', contribution_rank: 8 },
    { feature_name: 'patient_volume', feature_value: 250, shap_value: 0.07, contribution_direction: 'positive', contribution_rank: 9 },
    { feature_name: 'payer_mix_commercial', feature_value: 0.65, shap_value: -0.05, contribution_direction: 'negative', contribution_rank: 10 },
  ],
  'hcp-tier': [
    { feature_name: 'prescription_volume', feature_value: 85, shap_value: 0.42, contribution_direction: 'positive', contribution_rank: 1 },
    { feature_name: 'patient_count', feature_value: 120, shap_value: 0.35, contribution_direction: 'positive', contribution_rank: 2 },
    { feature_name: 'influence_score', feature_value: 0.78, shap_value: 0.25, contribution_direction: 'positive', contribution_rank: 3 },
    { feature_name: 'kol_status', feature_value: 1, shap_value: 0.18, contribution_direction: 'positive', contribution_rank: 4 },
    { feature_name: 'academic_affiliation', feature_value: 1, shap_value: 0.12, contribution_direction: 'positive', contribution_rank: 5 },
    { feature_name: 'practice_size', feature_value: 'large', shap_value: 0.09, contribution_direction: 'positive', contribution_rank: 6 },
    { feature_name: 'geography_urban', feature_value: 1, shap_value: -0.06, contribution_direction: 'negative', contribution_rank: 7 },
    { feature_name: 'tenure_years', feature_value: 12, shap_value: 0.05, contribution_direction: 'positive', contribution_rank: 8 },
  ],
  'conversion-v2': [
    { feature_name: 'engagement_frequency', feature_value: 8, shap_value: 0.38, contribution_direction: 'positive', contribution_rank: 1 },
    { feature_name: 'sample_requests', feature_value: 4, shap_value: 0.32, contribution_direction: 'positive', contribution_rank: 2 },
    { feature_name: 'content_interactions', feature_value: 15, shap_value: 0.28, contribution_direction: 'positive', contribution_rank: 3 },
    { feature_name: 'webinar_attendance', feature_value: 2, shap_value: 0.15, contribution_direction: 'positive', contribution_rank: 4 },
    { feature_name: 'competitor_loyalty', feature_value: 0.6, shap_value: -0.22, contribution_direction: 'negative', contribution_rank: 5 },
    { feature_name: 'time_since_switch', feature_value: 180, shap_value: -0.18, contribution_direction: 'negative', contribution_rank: 6 },
    { feature_name: 'peer_influence', feature_value: 0.45, shap_value: 0.12, contribution_direction: 'positive', contribution_rank: 7 },
    { feature_name: 'price_sensitivity', feature_value: 0.35, shap_value: -0.08, contribution_direction: 'negative', contribution_rank: 8 },
  ],
};

const SAMPLE_BASE_VALUES: Record<string, number> = {
  'churn-v3': 0.35,
  'hcp-tier': 0.48,
  'conversion-v2': 0.42,
};

function generateBeeswarmData(features: FeatureContribution[]): BeeswarmDataPoint[] {
  const data: BeeswarmDataPoint[] = [];
  const random = (min: number, max: number) => Math.random() * (max - min) + min;

  features.slice(0, 8).forEach((f) => {
    // Generate 25 sample points per feature
    for (let i = 0; i < 25; i++) {
      const featureValue = Math.random();
      // SHAP values correlate with feature direction
      const baseShap = f.shap_value * (0.5 + featureValue);
      const noise = random(-Math.abs(f.shap_value) * 0.3, Math.abs(f.shap_value) * 0.3);
      data.push({
        feature: f.feature_name,
        shapValue: baseShap + noise,
        featureValue,
        originalValue: Math.round(featureValue * 100),
        instanceId: `instance_${i}`,
      });
    }
  });

  return data;
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
  const [selectedModelId, setSelectedModelId] = useState<string>(SAMPLE_MODELS[0].id);
  const [selectedFeature, setSelectedFeature] = useState<FeatureContribution | null>(null);
  const [searchQuery, setSearchQuery] = useState('');
  const [isRefreshing, setIsRefreshing] = useState(false);

  // Get selected model data
  const selectedModel = useMemo(
    () => SAMPLE_MODELS.find((m) => m.id === selectedModelId) ?? SAMPLE_MODELS[0],
    [selectedModelId]
  );

  const features = useMemo(
    () => SAMPLE_FEATURES[selectedModelId] ?? SAMPLE_FEATURES['churn-v3'],
    [selectedModelId]
  );

  const baseValue = useMemo(
    () => SAMPLE_BASE_VALUES[selectedModelId] ?? 0.35,
    [selectedModelId]
  );

  const beeswarmData = useMemo(
    () => generateBeeswarmData(features),
    [features]
  );

  const filteredFeatures = useMemo(() => {
    if (!searchQuery) return features;
    const query = searchQuery.toLowerCase();
    return features.filter((f) =>
      f.feature_name.toLowerCase().includes(query)
    );
  }, [features, searchQuery]);

  const handleRefresh = async () => {
    setIsRefreshing(true);
    await new Promise((resolve) => setTimeout(resolve, 1000));
    setIsRefreshing(false);
  };

  const handleExport = () => {
    const exportData = {
      model: selectedModel,
      features,
      baseValue,
      exportedAt: new Date().toISOString(),
    };
    const blob = new Blob([JSON.stringify(exportData, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.download = `${selectedModel.id}-shap-values.json`;
    link.href = url;
    link.click();
    URL.revokeObjectURL(url);
  };

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

      {/* Model Info */}
      <Card className="mb-6">
        <CardContent className="pt-6">
          <div className="flex flex-wrap items-center justify-between gap-4">
            <div className="flex items-center gap-4">
              <div className="p-3 rounded-lg bg-primary/10">
                <BarChart3 className="h-6 w-6 text-primary" />
              </div>
              <div>
                <h2 className="text-xl font-semibold">{selectedModel.name}</h2>
                <div className="flex items-center gap-4 text-sm text-muted-foreground mt-1">
                  <span>{selectedModel.version}</span>
                  <span>•</span>
                  <span>{selectedModel.featureCount} features</span>
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
                  {features[0]?.feature_name.replace(/_/g, ' ')}
                </div>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

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
              {filteredFeatures.length === 0 && (
                <div className="text-center py-8 text-muted-foreground">
                  No features match your search
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
                  <CardTitle>Feature Value Distribution</CardTitle>
                  <CardDescription>
                    Each dot represents one sample. Color shows feature value (blue=low, red=high).
                    Position shows SHAP impact on prediction.
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <SHAPBeeswarm
                    data={beeswarmData}
                    maxFeatures={8}
                    height={450}
                    showLegend
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
