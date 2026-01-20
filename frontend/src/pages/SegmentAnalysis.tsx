/**
 * Segment Analysis Page
 * =====================
 *
 * Heterogeneous treatment effect analysis dashboard for identifying
 * high/low responder segments using Causal Forest and uplift modeling.
 * Uses Tier 2 Heterogeneous Optimizer agent with EconML/CausalML backends.
 *
 * Features:
 * - CATE (Conditional Average Treatment Effect) estimation by segment
 * - High/low responder identification
 * - Policy recommendations for optimal targeting
 * - Uplift metrics visualization
 * - Feature importance analysis
 *
 * @module pages/SegmentAnalysis
 */

import { useState } from 'react';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Cell,
  ScatterChart,
  Scatter,
  ReferenceLine,
  ErrorBar,
} from 'recharts';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { KPICard } from '@/components/visualizations';
import {
  useSegmentHealth,
  useRunSegmentAnalysis,
  usePolicies,
} from '@/hooks/api';
import type {
  SegmentAnalysisResponse,
  CATEResult,
  SegmentProfile,
  PolicyRecommendation,
  UpliftMetrics,
} from '@/types/segments';

// =============================================================================
// SAMPLE DATA FOR DEVELOPMENT
// =============================================================================

const sampleAnalysisResult: SegmentAnalysisResponse = {
  analysis_id: 'seg_xyz789',
  status: 'completed' as never,
  question_type: 'effect_heterogeneity' as never,
  cate_by_segment: {
    region: [
      {
        segment_name: 'region',
        segment_value: 'Northeast',
        cate_estimate: 0.45,
        cate_ci_lower: 0.32,
        cate_ci_upper: 0.58,
        sample_size: 1250,
        statistical_significance: true,
      },
      {
        segment_name: 'region',
        segment_value: 'Southeast',
        cate_estimate: 0.22,
        cate_ci_lower: 0.08,
        cate_ci_upper: 0.36,
        sample_size: 980,
        statistical_significance: true,
      },
      {
        segment_name: 'region',
        segment_value: 'Midwest',
        cate_estimate: 0.35,
        cate_ci_lower: 0.21,
        cate_ci_upper: 0.49,
        sample_size: 1100,
        statistical_significance: true,
      },
      {
        segment_name: 'region',
        segment_value: 'West',
        cate_estimate: 0.15,
        cate_ci_lower: -0.02,
        cate_ci_upper: 0.32,
        sample_size: 850,
        statistical_significance: false,
      },
    ],
    specialty: [
      {
        segment_name: 'specialty',
        segment_value: 'Cardiology',
        cate_estimate: 0.52,
        cate_ci_lower: 0.38,
        cate_ci_upper: 0.66,
        sample_size: 620,
        statistical_significance: true,
      },
      {
        segment_name: 'specialty',
        segment_value: 'Oncology',
        cate_estimate: 0.48,
        cate_ci_lower: 0.33,
        cate_ci_upper: 0.63,
        sample_size: 540,
        statistical_significance: true,
      },
      {
        segment_name: 'specialty',
        segment_value: 'Primary Care',
        cate_estimate: 0.18,
        cate_ci_lower: 0.05,
        cate_ci_upper: 0.31,
        sample_size: 1800,
        statistical_significance: true,
      },
      {
        segment_name: 'specialty',
        segment_value: 'Dermatology',
        cate_estimate: 0.08,
        cate_ci_lower: -0.08,
        cate_ci_upper: 0.24,
        sample_size: 420,
        statistical_significance: false,
      },
    ],
  },
  overall_ate: 0.28,
  heterogeneity_score: 0.72,
  feature_importance: {
    specialty: 0.35,
    region: 0.22,
    practice_size: 0.18,
    years_experience: 0.12,
    patient_volume: 0.08,
    digital_engagement: 0.05,
  },
  uplift_metrics: {
    overall_auuc: 0.68,
    overall_qini: 0.42,
    targeting_efficiency: 0.75,
    model_type_used: 'causal_forest',
  },
  high_responders: [
    {
      segment_id: 'seg_001',
      responder_type: 'high' as never,
      cate_estimate: 0.58,
      defining_features: [
        { specialty: 'Cardiology' },
        { region: 'Northeast' },
        { practice_size: 'large' },
      ],
      size: 245,
      size_percentage: 5.8,
      recommendation: 'Increase rep visit frequency to 2x per month',
    },
    {
      segment_id: 'seg_002',
      responder_type: 'high' as never,
      cate_estimate: 0.52,
      defining_features: [
        { specialty: 'Oncology' },
        { digital_engagement: 'high' },
      ],
      size: 180,
      size_percentage: 4.3,
      recommendation: 'Combine rep visits with digital outreach',
    },
    {
      segment_id: 'seg_003',
      responder_type: 'high' as never,
      cate_estimate: 0.47,
      defining_features: [
        { region: 'Midwest' },
        { years_experience: '10+' },
      ],
      size: 320,
      size_percentage: 7.6,
      recommendation: 'Focus on peer-reviewed clinical data',
    },
  ],
  low_responders: [
    {
      segment_id: 'seg_010',
      responder_type: 'low' as never,
      cate_estimate: 0.05,
      defining_features: [
        { specialty: 'Dermatology' },
        { digital_engagement: 'low' },
      ],
      size: 150,
      size_percentage: 3.6,
      recommendation: 'Reduce visit frequency, focus on digital channels',
    },
    {
      segment_id: 'seg_011',
      responder_type: 'low' as never,
      cate_estimate: 0.08,
      defining_features: [
        { region: 'West' },
        { practice_size: 'small' },
      ],
      size: 280,
      size_percentage: 6.7,
      recommendation: 'Shift resources to higher-response segments',
    },
  ],
  policy_recommendations: [
    {
      segment: 'Cardiology - Northeast',
      current_treatment_rate: 0.45,
      recommended_treatment_rate: 0.75,
      expected_incremental_outcome: 125,
      confidence: 0.92,
    },
    {
      segment: 'Oncology - High Digital',
      current_treatment_rate: 0.38,
      recommended_treatment_rate: 0.65,
      expected_incremental_outcome: 98,
      confidence: 0.88,
    },
    {
      segment: 'Primary Care - Midwest',
      current_treatment_rate: 0.55,
      recommended_treatment_rate: 0.50,
      expected_incremental_outcome: -15,
      confidence: 0.85,
    },
    {
      segment: 'Dermatology - All',
      current_treatment_rate: 0.40,
      recommended_treatment_rate: 0.20,
      expected_incremental_outcome: -45,
      confidence: 0.78,
    },
  ],
  expected_total_lift: 163,
  optimal_allocation_summary:
    'Optimal targeting suggests reallocating 20% of resources from low-response segments (Dermatology, West/Small) to high-response segments (Cardiology-Northeast, Oncology-Digital) for an expected 163 incremental TRx lift.',
  executive_summary:
    'Analysis reveals significant treatment effect heterogeneity (score: 0.72) across segments. Cardiology and Oncology specialists show 2-3x higher response to rep visits compared to Primary Care. Geographic effects are secondary to specialty, with Northeast showing strongest response. Recommend targeted reallocation of sales effort.',
  key_insights: [
    'Cardiology specialists in Northeast show highest CATE (0.52-0.58)',
    'Digital engagement amplifies treatment effect in Oncology segment',
    'Dermatology segment shows minimal response - consider resource reallocation',
    'Specialty is the strongest predictor of treatment response (35% importance)',
  ],
  libraries_used: ['econml', 'causalml'],
  library_agreement_score: 0.89,
  validation_passed: true,
  estimation_latency_ms: 2500,
  analysis_latency_ms: 1800,
  total_latency_ms: 4500,
  timestamp: new Date().toISOString(),
  warnings: [],
  confidence: 0.87,
};

// =============================================================================
// CHART COLORS
// =============================================================================

const COLORS = {
  primary: '#3b82f6',
  secondary: '#8b5cf6',
  success: '#10b981',
  warning: '#f59e0b',
  danger: '#ef4444',
  muted: '#6b7280',
};

const PIE_COLORS = ['#3b82f6', '#8b5cf6', '#10b981', '#f59e0b', '#ef4444', '#ec4899'];

// =============================================================================
// CATE CHART WITH ERROR BARS
// =============================================================================

interface CATEChartProps {
  cateResults: CATEResult[];
  segmentName: string;
}

function CATEBarChart({ cateResults, segmentName }: CATEChartProps) {
  const chartData = cateResults.map((r) => ({
    name: r.segment_value,
    cate: r.cate_estimate,
    ci_lower: r.cate_estimate - r.cate_ci_lower,
    ci_upper: r.cate_ci_upper - r.cate_estimate,
    significant: r.statistical_significance,
    sample_size: r.sample_size,
  }));

  return (
    <ResponsiveContainer width="100%" height={300}>
      <BarChart data={chartData} margin={{ top: 20, right: 30, left: 20, bottom: 5 }}>
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis dataKey="name" />
        <YAxis label={{ value: 'CATE', angle: -90, position: 'insideLeft' }} />
        <Tooltip
          content={({ payload }) => {
            if (!payload || payload.length === 0) return null;
            const data = payload[0].payload;
            return (
              <div className="bg-background border rounded-lg p-3 shadow-lg">
                <p className="font-medium">{data.name}</p>
                <p className="text-sm">CATE: {data.cate.toFixed(3)}</p>
                <p className="text-sm text-muted-foreground">
                  95% CI: [{(data.cate - data.ci_lower).toFixed(3)}, {(data.cate + data.ci_upper).toFixed(3)}]
                </p>
                <p className="text-sm text-muted-foreground">n = {data.sample_size}</p>
                <p className={`text-sm ${data.significant ? 'text-green-600' : 'text-yellow-600'}`}>
                  {data.significant ? 'Significant' : 'Not significant'}
                </p>
              </div>
            );
          }}
        />
        <ReferenceLine y={0} stroke={COLORS.muted} strokeDasharray="3 3" />
        <Bar dataKey="cate" name="CATE">
          {chartData.map((entry, index) => (
            <Cell
              key={`cell-${index}`}
              fill={entry.significant ? COLORS.primary : COLORS.muted}
            />
          ))}
          <ErrorBar dataKey="ci_upper" direction="y" stroke={COLORS.secondary} />
          <ErrorBar dataKey="ci_lower" direction="y" stroke={COLORS.secondary} />
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}

// =============================================================================
// FEATURE IMPORTANCE CHART
// =============================================================================

interface FeatureImportanceChartProps {
  importance: Record<string, number>;
}

function FeatureImportanceChart({ importance }: FeatureImportanceChartProps) {
  const chartData = Object.entries(importance)
    .sort(([, a], [, b]) => b - a)
    .map(([feature, value]) => ({
      name: feature.replace(/_/g, ' ').replace(/\b\w/g, (l) => l.toUpperCase()),
      importance: value * 100,
    }));

  return (
    <ResponsiveContainer width="100%" height={250}>
      <BarChart data={chartData} layout="vertical" margin={{ top: 5, right: 30, left: 100, bottom: 5 }}>
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis type="number" domain={[0, 40]} label={{ value: 'Importance (%)', position: 'bottom' }} />
        <YAxis type="category" dataKey="name" />
        <Tooltip formatter={(value) => [`${Number(value ?? 0).toFixed(1)}%`, 'Importance']} />
        <Bar dataKey="importance" fill={COLORS.secondary} />
      </BarChart>
    </ResponsiveContainer>
  );
}

// =============================================================================
// UPLIFT METRICS VISUALIZATION
// =============================================================================

interface UpliftMetricsChartProps {
  metrics: UpliftMetrics;
}

function UpliftMetricsChart({ metrics }: UpliftMetricsChartProps) {
  const chartData = [
    { name: 'AUUC', value: metrics.overall_auuc * 100, max: 100 },
    { name: 'Qini', value: metrics.overall_qini * 100, max: 100 },
    { name: 'Targeting Efficiency', value: metrics.targeting_efficiency * 100, max: 100 },
  ];

  return (
    <div className="space-y-4">
      {chartData.map((item) => (
        <div key={item.name} className="space-y-1">
          <div className="flex justify-between text-sm">
            <span>{item.name}</span>
            <span className="font-medium">{item.value.toFixed(1)}%</span>
          </div>
          <div className="h-2 bg-muted rounded-full overflow-hidden">
            <div
              className="h-full bg-primary rounded-full transition-all"
              style={{ width: `${item.value}%` }}
            />
          </div>
        </div>
      ))}
      <p className="text-sm text-muted-foreground">
        Model: {metrics.model_type_used.replace(/_/g, ' ')}
      </p>
    </div>
  );
}

// =============================================================================
// POLICY SCATTER CHART
// =============================================================================

interface PolicyChartProps {
  policies: PolicyRecommendation[];
}

function PolicyScatterChart({ policies }: PolicyChartProps) {
  const chartData = policies.map((p) => ({
    name: p.segment,
    current: p.current_treatment_rate * 100,
    recommended: p.recommended_treatment_rate * 100,
    impact: p.expected_incremental_outcome,
    confidence: p.confidence,
    change: (p.recommended_treatment_rate - p.current_treatment_rate) * 100,
  }));

  return (
    <ResponsiveContainer width="100%" height={300}>
      <ScatterChart margin={{ top: 20, right: 30, left: 20, bottom: 5 }}>
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis
          dataKey="current"
          name="Current Rate"
          label={{ value: 'Current Treatment Rate (%)', position: 'bottom' }}
        />
        <YAxis
          dataKey="recommended"
          name="Recommended Rate"
          label={{ value: 'Recommended Rate (%)', angle: -90, position: 'insideLeft' }}
        />
        <Tooltip
          content={({ payload }) => {
            if (!payload || payload.length === 0) return null;
            const data = payload[0].payload;
            return (
              <div className="bg-background border rounded-lg p-3 shadow-lg">
                <p className="font-medium">{data.name}</p>
                <p className="text-sm">Current: {data.current.toFixed(0)}%</p>
                <p className="text-sm">Recommended: {data.recommended.toFixed(0)}%</p>
                <p className={`text-sm ${data.impact >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                  Impact: {data.impact >= 0 ? '+' : ''}{data.impact}
                </p>
                <p className="text-sm text-muted-foreground">
                  Confidence: {(data.confidence * 100).toFixed(0)}%
                </p>
              </div>
            );
          }}
        />
        <ReferenceLine x={50} stroke={COLORS.muted} strokeDasharray="3 3" />
        <ReferenceLine y={50} stroke={COLORS.muted} strokeDasharray="3 3" />
        {/* Diagonal line for no-change reference */}
        <ReferenceLine
          segment={[{ x: 0, y: 0 }, { x: 100, y: 100 }]}
          stroke={COLORS.muted}
          strokeDasharray="5 5"
        />
        <Scatter
          data={chartData}
          fill={COLORS.primary}
          shape={(props: { cx: number; cy: number; payload: { impact: number } }) => {
            const { cx, cy, payload } = props;
            return (
              <circle
                cx={cx}
                cy={cy}
                r={8}
                fill={payload.impact >= 0 ? COLORS.success : COLORS.danger}
                stroke={COLORS.primary}
                strokeWidth={2}
              />
            );
          }}
        />
      </ScatterChart>
    </ResponsiveContainer>
  );
}

// =============================================================================
// RESPONDER PROFILE CARD
// =============================================================================

interface ResponderCardProps {
  profile: SegmentProfile;
}

function ResponderCard({ profile }: ResponderCardProps) {
  const isHigh = profile.responder_type === 'high';

  return (
    <div className={`p-4 border rounded-lg ${isHigh ? 'border-green-200 bg-green-50/50' : 'border-red-200 bg-red-50/50'}`}>
      <div className="flex items-center justify-between mb-2">
        <Badge variant={isHigh ? 'default' : 'destructive'}>
          {isHigh ? 'High Responder' : 'Low Responder'}
        </Badge>
        <span className="text-sm font-medium">
          CATE: {profile.cate_estimate.toFixed(3)}
        </span>
      </div>
      <div className="space-y-2">
        <div className="flex flex-wrap gap-1">
          {profile.defining_features.map((feature, idx) => {
            const [key, value] = Object.entries(feature)[0];
            return (
              <Badge key={idx} variant="outline" className="text-xs">
                {key}: {String(value)}
              </Badge>
            );
          })}
        </div>
        <div className="text-sm text-muted-foreground">
          Size: {profile.size} ({profile.size_percentage.toFixed(1)}% of total)
        </div>
        <p className="text-sm mt-2">{profile.recommendation}</p>
      </div>
    </div>
  );
}

// =============================================================================
// MAIN PAGE COMPONENT
// =============================================================================

export default function SegmentAnalysis() {
  const [activeTab, setActiveTab] = useState('cate');
  const [selectedTreatment, setSelectedTreatment] = useState('rep_visits');
  const [selectedOutcome, setSelectedOutcome] = useState('trx');

  // API hooks
  const { data: healthData, isLoading: healthLoading } = useSegmentHealth();
  const { data: _policiesData } = usePolicies({ limit: 10 });
  const runAnalysis = useRunSegmentAnalysis();

  // Use sample data for now (API may not be available)
  const analysisResult = sampleAnalysisResult;

  // Health status
  const isHealthy =
    healthData?.agent_available &&
    (healthData?.econml_available || healthData?.causalml_available);

  // Handle analysis run
  const handleRunAnalysis = () => {
    runAnalysis.mutate({
      request: {
        query: `Analyze treatment effect heterogeneity for ${selectedTreatment} on ${selectedOutcome}`,
        treatment_var: selectedTreatment,
        outcome_var: selectedOutcome,
        segment_vars: ['region', 'specialty'],
        effect_modifiers: ['practice_size', 'years_experience', 'digital_engagement'],
      },
    });
  };

  return (
    <div className="container mx-auto py-6 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold">Segment Analysis</h1>
          <p className="text-muted-foreground">
            Heterogeneous treatment effect analysis and targeting optimization
          </p>
        </div>
        <div className="flex items-center gap-4">
          <Badge variant={isHealthy ? 'default' : 'destructive'}>
            {healthLoading ? 'Checking...' : isHealthy ? 'Agents Ready' : 'Agents Unavailable'}
          </Badge>
          {healthData?.analyses_24h !== undefined && (
            <Badge variant="outline">{healthData.analyses_24h} analyses today</Badge>
          )}
        </div>
      </div>

      {/* KPI Summary */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-5 gap-4">
        <KPICard
          title="Overall ATE"
          value={analysisResult.overall_ate?.toFixed(3) || 'N/A'}
          subtitle="Average Treatment Effect"
        />
        <KPICard
          title="Heterogeneity"
          value={`${((analysisResult.heterogeneity_score || 0) * 100).toFixed(0)}%`}
          trend={{ value: 15, direction: 'up' }}
          subtitle="Effect variation across segments"
        />
        <KPICard
          title="High Responders"
          value={analysisResult.high_responders.length.toString()}
          subtitle="segments identified"
        />
        <KPICard
          title="Expected Lift"
          value={`+${analysisResult.expected_total_lift}`}
          trend={{ value: 8, direction: 'up' }}
          subtitle="from optimal targeting"
        />
        <KPICard
          title="Confidence"
          value={`${(analysisResult.confidence * 100).toFixed(0)}%`}
          subtitle="analysis reliability"
        />
      </div>

      {/* Configuration Panel */}
      <Card>
        <CardHeader>
          <CardTitle>Analysis Configuration</CardTitle>
          <CardDescription>
            Select treatment and outcome variables for heterogeneity analysis
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div>
              <label className="text-sm font-medium mb-2 block">Treatment Variable</label>
              <select
                className="w-full p-2 border rounded-md"
                value={selectedTreatment}
                onChange={(e) => setSelectedTreatment(e.target.value)}
              >
                <option value="rep_visits">Rep Visits</option>
                <option value="email_campaigns">Email Campaigns</option>
                <option value="samples">Samples</option>
                <option value="speaker_programs">Speaker Programs</option>
              </select>
            </div>
            <div>
              <label className="text-sm font-medium mb-2 block">Outcome Variable</label>
              <select
                className="w-full p-2 border rounded-md"
                value={selectedOutcome}
                onChange={(e) => setSelectedOutcome(e.target.value)}
              >
                <option value="trx">TRx (Total Prescriptions)</option>
                <option value="nrx">NRx (New Prescriptions)</option>
                <option value="conversion">Conversion Rate</option>
                <option value="revenue">Revenue</option>
              </select>
            </div>
            <div className="flex items-end">
              <Button
                onClick={handleRunAnalysis}
                disabled={runAnalysis.isPending}
                className="w-full"
              >
                {runAnalysis.isPending ? 'Analyzing...' : 'Run Analysis'}
              </Button>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Main Content Tabs */}
      <Tabs value={activeTab} onValueChange={setActiveTab}>
        <TabsList className="grid w-full grid-cols-5">
          <TabsTrigger value="cate">CATE by Segment</TabsTrigger>
          <TabsTrigger value="responders">Responders</TabsTrigger>
          <TabsTrigger value="policies">Policies</TabsTrigger>
          <TabsTrigger value="uplift">Uplift Metrics</TabsTrigger>
          <TabsTrigger value="insights">Insights</TabsTrigger>
        </TabsList>

        {/* CATE Tab */}
        <TabsContent value="cate" className="space-y-4">
          {Object.entries(analysisResult.cate_by_segment).map(([segmentName, results]) => (
            <Card key={segmentName}>
              <CardHeader>
                <CardTitle>CATE by {segmentName.replace(/\b\w/g, (l) => l.toUpperCase())}</CardTitle>
                <CardDescription>
                  Conditional Average Treatment Effect with 95% confidence intervals
                </CardDescription>
              </CardHeader>
              <CardContent>
                <CATEBarChart cateResults={results} segmentName={segmentName} />
              </CardContent>
            </Card>
          ))}

          {/* Feature Importance */}
          {analysisResult.feature_importance && (
            <Card>
              <CardHeader>
                <CardTitle>Feature Importance for CATE</CardTitle>
                <CardDescription>
                  Which variables most influence treatment effect heterogeneity
                </CardDescription>
              </CardHeader>
              <CardContent>
                <FeatureImportanceChart importance={analysisResult.feature_importance} />
              </CardContent>
            </Card>
          )}
        </TabsContent>

        {/* Responders Tab */}
        <TabsContent value="responders" className="space-y-4">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            <Card>
              <CardHeader>
                <CardTitle className="text-green-700">High Responders</CardTitle>
                <CardDescription>
                  Segments with above-average treatment response
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                {analysisResult.high_responders.map((profile) => (
                  <ResponderCard key={profile.segment_id} profile={profile} />
                ))}
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle className="text-red-700">Low Responders</CardTitle>
                <CardDescription>
                  Segments with below-average treatment response
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                {analysisResult.low_responders.map((profile) => (
                  <ResponderCard key={profile.segment_id} profile={profile} />
                ))}
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        {/* Policies Tab */}
        <TabsContent value="policies" className="space-y-4">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            <Card>
              <CardHeader>
                <CardTitle>Targeting Recommendations</CardTitle>
                <CardDescription>
                  Current vs recommended treatment rates
                </CardDescription>
              </CardHeader>
              <CardContent>
                <PolicyScatterChart policies={analysisResult.policy_recommendations} />
                <p className="text-sm text-muted-foreground mt-2">
                  Points above the diagonal line = increase targeting; below = decrease
                </p>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Policy Details</CardTitle>
                <CardDescription>
                  Individual segment recommendations
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-3">
                  {analysisResult.policy_recommendations.map((policy, idx) => (
                    <div key={idx} className="p-3 border rounded-lg">
                      <div className="flex items-center justify-between mb-2">
                        <span className="font-medium">{policy.segment}</span>
                        <Badge
                          variant={
                            policy.expected_incremental_outcome >= 0 ? 'default' : 'outline'
                          }
                        >
                          {policy.expected_incremental_outcome >= 0 ? '+' : ''}
                          {policy.expected_incremental_outcome} impact
                        </Badge>
                      </div>
                      <div className="flex items-center gap-4 text-sm">
                        <span className="text-muted-foreground">
                          Current: {(policy.current_treatment_rate * 100).toFixed(0)}%
                        </span>
                        <span>→</span>
                        <span className={policy.expected_incremental_outcome >= 0 ? 'text-green-600' : 'text-red-600'}>
                          Recommended: {(policy.recommended_treatment_rate * 100).toFixed(0)}%
                        </span>
                      </div>
                      <div className="text-xs text-muted-foreground mt-1">
                        Confidence: {(policy.confidence * 100).toFixed(0)}%
                      </div>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          </div>

          {/* Optimal Allocation Summary */}
          <Card>
            <CardHeader>
              <CardTitle>Optimal Allocation Summary</CardTitle>
            </CardHeader>
            <CardContent>
              <p className="text-lg">{analysisResult.optimal_allocation_summary}</p>
              <div className="mt-4 flex items-center gap-4">
                <Badge variant="default" className="text-lg px-4 py-1">
                  Expected Total Lift: +{analysisResult.expected_total_lift}
                </Badge>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        {/* Uplift Tab */}
        <TabsContent value="uplift" className="space-y-4">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            <Card>
              <CardHeader>
                <CardTitle>Uplift Model Performance</CardTitle>
                <CardDescription>
                  How well the model identifies treatment responders
                </CardDescription>
              </CardHeader>
              <CardContent>
                {analysisResult.uplift_metrics && (
                  <UpliftMetricsChart metrics={analysisResult.uplift_metrics} />
                )}
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Library Validation</CardTitle>
                <CardDescription>
                  Cross-validation across causal inference libraries
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  <div className="flex items-center justify-between p-3 bg-muted/50 rounded-lg">
                    <span>Libraries Used</span>
                    <div className="flex gap-2">
                      {analysisResult.libraries_used?.map((lib) => (
                        <Badge key={lib} variant="outline">
                          {lib}
                        </Badge>
                      ))}
                    </div>
                  </div>
                  <div className="flex items-center justify-between p-3 bg-muted/50 rounded-lg">
                    <span>Agreement Score</span>
                    <Badge variant={analysisResult.library_agreement_score! >= 0.8 ? 'default' : 'secondary'}>
                      {((analysisResult.library_agreement_score || 0) * 100).toFixed(0)}%
                    </Badge>
                  </div>
                  <div className="flex items-center justify-between p-3 bg-muted/50 rounded-lg">
                    <span>Validation Status</span>
                    <Badge variant={analysisResult.validation_passed ? 'default' : 'destructive'}>
                      {analysisResult.validation_passed ? 'Passed' : 'Failed'}
                    </Badge>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        {/* Insights Tab */}
        <TabsContent value="insights" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Executive Summary</CardTitle>
            </CardHeader>
            <CardContent>
              <p className="text-lg">{analysisResult.executive_summary}</p>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Key Insights</CardTitle>
              <CardDescription>
                AI-generated findings from segment analysis
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-3">
                {analysisResult.key_insights.map((insight, idx) => (
                  <div
                    key={idx}
                    className="flex items-start gap-3 p-3 bg-muted/50 rounded-lg"
                  >
                    <div className="flex-shrink-0 w-6 h-6 rounded-full bg-primary text-primary-foreground flex items-center justify-center text-sm font-medium">
                      {idx + 1}
                    </div>
                    <p>{insight}</p>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>

          {/* Analysis Metadata */}
          <Card>
            <CardHeader>
              <CardTitle>Analysis Metadata</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                <div>
                  <p className="text-muted-foreground">Analysis ID</p>
                  <p className="font-mono">{analysisResult.analysis_id}</p>
                </div>
                <div>
                  <p className="text-muted-foreground">Estimation Time</p>
                  <p>{analysisResult.estimation_latency_ms}ms</p>
                </div>
                <div>
                  <p className="text-muted-foreground">Analysis Time</p>
                  <p>{analysisResult.analysis_latency_ms}ms</p>
                </div>
                <div>
                  <p className="text-muted-foreground">Total Time</p>
                  <p>{analysisResult.total_latency_ms}ms</p>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
}
