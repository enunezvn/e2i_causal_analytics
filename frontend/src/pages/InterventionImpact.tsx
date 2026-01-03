/**
 * Intervention Impact Page
 * ========================
 *
 * Comprehensive intervention analysis dashboard with:
 * - Before/after comparisons
 * - Causal treatment effect estimation
 * - Counterfactual analysis
 * - A/B test result visualization
 *
 * @module pages/InterventionImpact
 */

import { useState, useMemo } from 'react';
import {
  ComposedChart,
  LineChart,
  BarChart,
  Line,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  ReferenceLine,
  Area,
} from 'recharts';
import {
  TrendingUp,
  TrendingDown,
  RefreshCw,
  Download,
  Target,
  Activity,
  Zap,
  Beaker,
  GitBranch,
  ArrowRight,
  CheckCircle2,
  XCircle,
  AlertTriangle,
  Info,
  FlaskConical,
} from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';
import { cn } from '@/lib/utils';
import { KPICard, StatusBadge } from '@/components/visualizations';
import { SimulationPanel, ScenarioResults, RecommendationCards } from '@/components/digital-twin';
import { useRunSimulation } from '@/hooks/api/use-digital-twin';
import type { SimulationRequest, SimulationResponse } from '@/types/digital-twin';

// =============================================================================
// TYPES
// =============================================================================

interface Intervention {
  id: string;
  name: string;
  type: 'campaign' | 'pricing' | 'product' | 'promotional' | 'training';
  status: 'active' | 'completed' | 'planned';
  startDate: string;
  endDate?: string;
  targetMetric: string;
  description: string;
}

interface ImpactData {
  date: string;
  actual: number;
  counterfactual: number;
  upperBound: number;
  lowerBound: number;
}

interface TreatmentEffect {
  id: string;
  intervention: string;
  metric: string;
  ate: number; // Average Treatment Effect
  ci: [number, number]; // Confidence Interval
  pValue: number;
  isSignificant: boolean;
  sampleSize: number;
  effectSize: 'small' | 'medium' | 'large';
}

interface BeforeAfterComparison {
  metric: string;
  beforeMean: number;
  afterMean: number;
  change: number;
  changePercent: number;
  isPositive: boolean;
}

interface SegmentEffect {
  segment: string;
  sampleSize: number;
  effect: number;
  ci: [number, number];
  isSignificant: boolean;
}

// =============================================================================
// SAMPLE DATA
// =============================================================================

const INTERVENTIONS: Intervention[] = [
  {
    id: 'int-001',
    name: 'Q1 2024 HCP Engagement Campaign',
    type: 'campaign',
    status: 'completed',
    startDate: '2024-01-15',
    endDate: '2024-03-15',
    targetMetric: 'TRx Volume',
    description: 'Targeted engagement campaign for high-potential HCPs in the Northeast region',
  },
  {
    id: 'int-002',
    name: 'Digital Rep Training Program',
    type: 'training',
    status: 'completed',
    startDate: '2024-02-01',
    endDate: '2024-04-01',
    targetMetric: 'Conversion Rate',
    description: 'Enhanced training program for digital engagement techniques',
  },
  {
    id: 'int-003',
    name: 'Kisqali Patient Support Enhancement',
    type: 'promotional',
    status: 'active',
    startDate: '2024-03-01',
    targetMetric: 'Patient Retention',
    description: 'Improved patient support program with additional touchpoints',
  },
  {
    id: 'int-004',
    name: 'Remibrutinib Launch Preparation',
    type: 'product',
    status: 'planned',
    startDate: '2024-06-01',
    targetMetric: 'Market Awareness',
    description: 'Pre-launch awareness campaign for Remibrutinib',
  },
];

// Generate causal impact data
const generateImpactData = (interventionStart: Date): ImpactData[] => {
  const data: ImpactData[] = [];
  const startDate = new Date(interventionStart);
  startDate.setDate(startDate.getDate() - 30); // 30 days before intervention

  for (let i = 0; i < 90; i++) {
    const date = new Date(startDate);
    date.setDate(startDate.getDate() + i);

    const isPostIntervention = i >= 30;
    const baseValue = 1000 + i * 0.5 + Math.sin(i / 7) * 20;
    const interventionEffect = isPostIntervention ? 80 + (i - 30) * 0.5 : 0;
    const noise = (Math.random() - 0.5) * 40;

    const actual = baseValue + interventionEffect + noise;
    const counterfactual = baseValue + noise * 0.8;
    const uncertainty = 25 + (isPostIntervention ? (i - 30) * 0.3 : 0);

    data.push({
      date: date.toISOString().split('T')[0],
      actual: Math.round(actual),
      counterfactual: Math.round(counterfactual),
      upperBound: Math.round(counterfactual + uncertainty),
      lowerBound: Math.round(counterfactual - uncertainty),
    });
  }

  return data;
};

const SAMPLE_IMPACT_DATA = generateImpactData(new Date('2024-01-15'));

const SAMPLE_TREATMENT_EFFECTS: TreatmentEffect[] = [
  {
    id: 'te-001',
    intervention: 'Q1 2024 HCP Engagement Campaign',
    metric: 'TRx Volume',
    ate: 85.3,
    ci: [62.1, 108.5],
    pValue: 0.0012,
    isSignificant: true,
    sampleSize: 1250,
    effectSize: 'large',
  },
  {
    id: 'te-002',
    intervention: 'Q1 2024 HCP Engagement Campaign',
    metric: 'NRx Volume',
    ate: 24.7,
    ci: [18.2, 31.2],
    pValue: 0.0034,
    isSignificant: true,
    sampleSize: 1250,
    effectSize: 'medium',
  },
  {
    id: 'te-003',
    intervention: 'Digital Rep Training Program',
    metric: 'Conversion Rate',
    ate: 3.2,
    ci: [1.8, 4.6],
    pValue: 0.0078,
    isSignificant: true,
    sampleSize: 450,
    effectSize: 'medium',
  },
  {
    id: 'te-004',
    intervention: 'Digital Rep Training Program',
    metric: 'HCP Satisfaction',
    ate: 0.8,
    ci: [-0.2, 1.8],
    pValue: 0.1245,
    isSignificant: false,
    sampleSize: 450,
    effectSize: 'small',
  },
];

const SAMPLE_BEFORE_AFTER: BeforeAfterComparison[] = [
  { metric: 'TRx Volume', beforeMean: 1024, afterMean: 1109, change: 85, changePercent: 8.3, isPositive: true },
  { metric: 'NRx Volume', beforeMean: 312, afterMean: 337, change: 25, changePercent: 8.0, isPositive: true },
  { metric: 'Market Share', beforeMean: 23.4, afterMean: 24.8, change: 1.4, changePercent: 6.0, isPositive: true },
  { metric: 'HCP Reach', beforeMean: 856, afterMean: 912, change: 56, changePercent: 6.5, isPositive: true },
  { metric: 'Cost per TRx', beforeMean: 42.5, afterMean: 38.2, change: -4.3, changePercent: -10.1, isPositive: true },
];

const SAMPLE_SEGMENT_EFFECTS: SegmentEffect[] = [
  { segment: 'High-Volume HCPs', sampleSize: 245, effect: 112.5, ci: [78.3, 146.7], isSignificant: true },
  { segment: 'Medium-Volume HCPs', sampleSize: 520, effect: 78.2, ci: [52.1, 104.3], isSignificant: true },
  { segment: 'Low-Volume HCPs', sampleSize: 485, effect: 45.8, ci: [18.9, 72.7], isSignificant: true },
  { segment: 'Northeast Region', sampleSize: 380, effect: 95.3, ci: [68.2, 122.4], isSignificant: true },
  { segment: 'Southeast Region', sampleSize: 350, effect: 72.1, ci: [42.5, 101.7], isSignificant: true },
  { segment: 'Midwest Region', sampleSize: 290, effect: 68.4, ci: [35.2, 101.6], isSignificant: true },
  { segment: 'West Region', sampleSize: 230, effect: 52.6, ci: [12.8, 92.4], isSignificant: true },
];

// =============================================================================
// HELPER FUNCTIONS
// =============================================================================

function formatDate(dateStr: string): string {
  const date = new Date(dateStr);
  return date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
}

function getInterventionTypeColor(type: Intervention['type']): string {
  switch (type) {
    case 'campaign':
      return 'bg-blue-100 text-blue-700 dark:bg-blue-900/30 dark:text-blue-400';
    case 'pricing':
      return 'bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-400';
    case 'product':
      return 'bg-purple-100 text-purple-700 dark:bg-purple-900/30 dark:text-purple-400';
    case 'promotional':
      return 'bg-amber-100 text-amber-700 dark:bg-amber-900/30 dark:text-amber-400';
    case 'training':
      return 'bg-rose-100 text-rose-700 dark:bg-rose-900/30 dark:text-rose-400';
  }
}

function getStatusColor(status: Intervention['status']): string {
  switch (status) {
    case 'active':
      return 'bg-emerald-100 text-emerald-700 dark:bg-emerald-900/30 dark:text-emerald-400';
    case 'completed':
      return 'bg-slate-100 text-slate-700 dark:bg-slate-900/30 dark:text-slate-400';
    case 'planned':
      return 'bg-blue-100 text-blue-700 dark:bg-blue-900/30 dark:text-blue-400';
  }
}

function getEffectSizeLabel(size: TreatmentEffect['effectSize']): string {
  switch (size) {
    case 'small':
      return "Cohen's d < 0.2";
    case 'medium':
      return "Cohen's d 0.2-0.8";
    case 'large':
      return "Cohen's d > 0.8";
  }
}

// =============================================================================
// COMPONENT
// =============================================================================

function InterventionImpact() {
  const [selectedIntervention, setSelectedIntervention] = useState<string>(INTERVENTIONS[0].id);
  const [isRefreshing, setIsRefreshing] = useState(false);
  const [simulationResults, setSimulationResults] = useState<SimulationResponse | null>(null);

  // Digital Twin simulation mutation
  const { mutate: runSimulation, isPending: isSimulating } = useRunSimulation({
    onSuccess: (data) => {
      setSimulationResults(data);
    },
  });

  // Handle simulation request
  const handleSimulate = (request: SimulationRequest) => {
    runSimulation(request);
  };

  // Get selected intervention details
  const currentIntervention = useMemo(
    () => INTERVENTIONS.find((i) => i.id === selectedIntervention) || INTERVENTIONS[0],
    [selectedIntervention]
  );

  // Calculate summary metrics
  const summaryMetrics = useMemo(() => {
    const impactData = SAMPLE_IMPACT_DATA;
    const preInterventionData = impactData.slice(0, 30);
    const postInterventionData = impactData.slice(30);

    const avgActual = postInterventionData.reduce((a, b) => a + b.actual, 0) / postInterventionData.length;
    const avgCounterfactual = postInterventionData.reduce((a, b) => a + b.counterfactual, 0) / postInterventionData.length;
    const cumulativeEffect = postInterventionData.reduce((a, b) => a + (b.actual - b.counterfactual), 0);

    const significantEffects = SAMPLE_TREATMENT_EFFECTS.filter((e) => e.isSignificant).length;
    const totalEffects = SAMPLE_TREATMENT_EFFECTS.length;

    return {
      avgLift: ((avgActual - avgCounterfactual) / avgCounterfactual) * 100,
      cumulativeEffect,
      significantEffects,
      totalEffects,
      avgATE: SAMPLE_TREATMENT_EFFECTS.filter((e) => e.isSignificant).reduce((a, b) => a + b.ate, 0) / significantEffects,
    };
  }, []);

  const handleRefresh = () => {
    setIsRefreshing(true);
    setTimeout(() => setIsRefreshing(false), 1000);
  };

  const handleExport = () => {
    const exportData = {
      intervention: currentIntervention,
      impactData: SAMPLE_IMPACT_DATA,
      treatmentEffects: SAMPLE_TREATMENT_EFFECTS,
      beforeAfter: SAMPLE_BEFORE_AFTER,
      segmentEffects: SAMPLE_SEGMENT_EFFECTS,
    };
    const blob = new Blob([JSON.stringify(exportData, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'intervention-impact-analysis.json';
    a.click();
    URL.revokeObjectURL(url);
  };

  // Custom tooltip for causal impact chart
  const ImpactTooltip = ({ active, payload, label }: { active?: boolean; payload?: Array<Record<string, unknown>>; label?: string }) => {
    if (!active || !payload || !payload.length) return null;

    const dataPoint = payload[0].payload as ImpactData;
    const causalEffect = dataPoint.actual - dataPoint.counterfactual;

    return (
      <div className="bg-[var(--color-popover)] border border-[var(--color-border)] rounded-md shadow-lg p-3">
        <p className="font-medium text-[var(--color-foreground)] mb-2">
          {formatDate(label || '')}
        </p>
        <div className="space-y-1 text-sm">
          <div className="flex items-center justify-between gap-4">
            <span className="text-[var(--color-muted-foreground)]">Actual:</span>
            <span className="font-medium">{dataPoint.actual.toLocaleString()}</span>
          </div>
          <div className="flex items-center justify-between gap-4">
            <span className="text-[var(--color-muted-foreground)]">Counterfactual:</span>
            <span className="font-medium">{dataPoint.counterfactual.toLocaleString()}</span>
          </div>
          <div className="flex items-center justify-between gap-4 pt-1 border-t border-border">
            <span className="text-[var(--color-muted-foreground)]">Causal Effect:</span>
            <span className={cn('font-bold', causalEffect > 0 ? 'text-emerald-600' : 'text-rose-600')}>
              {causalEffect > 0 ? '+' : ''}{causalEffect.toLocaleString()}
            </span>
          </div>
        </div>
      </div>
    );
  };

  return (
    <div className="container mx-auto px-4 py-8">
      {/* Header */}
      <div className="flex items-center justify-between mb-8">
        <div>
          <h1 className="text-3xl font-bold mb-2">Intervention Impact</h1>
          <p className="text-muted-foreground">
            Before/after comparisons, treatment effects, and counterfactual analysis.
          </p>
        </div>
        <div className="flex items-center gap-3">
          <Select value={selectedIntervention} onValueChange={setSelectedIntervention}>
            <SelectTrigger className="w-[300px]">
              <SelectValue placeholder="Select intervention" />
            </SelectTrigger>
            <SelectContent>
              {INTERVENTIONS.map((intervention) => (
                <SelectItem key={intervention.id} value={intervention.id}>
                  {intervention.name}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
          <Button variant="outline" size="icon" onClick={handleRefresh} disabled={isRefreshing}>
            <RefreshCw className={cn('h-4 w-4', isRefreshing && 'animate-spin')} />
          </Button>
          <Button variant="outline" size="icon" onClick={handleExport}>
            <Download className="h-4 w-4" />
          </Button>
        </div>
      </div>

      {/* Intervention Summary Card */}
      <Card className="mb-8">
        <CardContent className="pt-6">
          <div className="flex items-start justify-between">
            <div className="space-y-2">
              <div className="flex items-center gap-2">
                <h2 className="text-xl font-semibold">{currentIntervention.name}</h2>
                <Badge variant="outline" className={getInterventionTypeColor(currentIntervention.type)}>
                  {currentIntervention.type}
                </Badge>
                <Badge variant="outline" className={getStatusColor(currentIntervention.status)}>
                  {currentIntervention.status}
                </Badge>
              </div>
              <p className="text-sm text-muted-foreground max-w-2xl">
                {currentIntervention.description}
              </p>
              <div className="flex items-center gap-4 text-sm">
                <span className="text-muted-foreground">
                  Start: <strong>{formatDate(currentIntervention.startDate)}</strong>
                </span>
                {currentIntervention.endDate && (
                  <span className="text-muted-foreground">
                    End: <strong>{formatDate(currentIntervention.endDate)}</strong>
                  </span>
                )}
                <span className="text-muted-foreground">
                  Target: <strong>{currentIntervention.targetMetric}</strong>
                </span>
              </div>
            </div>
            <div className="flex items-center gap-4">
              <div className="text-center">
                <p className="text-2xl font-bold text-emerald-600">
                  +{summaryMetrics.avgLift.toFixed(1)}%
                </p>
                <p className="text-xs text-muted-foreground">Avg. Lift</p>
              </div>
              <div className="text-center">
                <p className="text-2xl font-bold">
                  {summaryMetrics.cumulativeEffect.toLocaleString()}
                </p>
                <p className="text-xs text-muted-foreground">Cumulative Effect</p>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* KPI Summary */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-8">
        <KPICard
          title="Average Treatment Effect"
          value={`+${summaryMetrics.avgATE.toFixed(1)}`}
          icon={Target}
          status="success"
          subtitle="Across significant outcomes"
        />
        <KPICard
          title="Significant Effects"
          value={`${summaryMetrics.significantEffects}/${summaryMetrics.totalEffects}`}
          icon={CheckCircle2}
          status={summaryMetrics.significantEffects > summaryMetrics.totalEffects / 2 ? 'success' : 'warning'}
          subtitle="p < 0.05"
        />
        <KPICard
          title="Cumulative Impact"
          value={`+${(summaryMetrics.cumulativeEffect / 1000).toFixed(1)}K`}
          icon={TrendingUp}
          status="success"
          subtitle="Total incremental units"
        />
        <KPICard
          title="ROI Estimate"
          value="3.2x"
          icon={Zap}
          status="success"
          subtitle="Return on investment"
        />
      </div>

      {/* Main Content Tabs */}
      <Tabs defaultValue="causal" className="space-y-6">
        <TabsList>
          <TabsTrigger value="causal" className="gap-2">
            <Activity className="h-4 w-4" />
            Causal Impact
          </TabsTrigger>
          <TabsTrigger value="beforeafter" className="gap-2">
            <ArrowRight className="h-4 w-4" />
            Before/After
          </TabsTrigger>
          <TabsTrigger value="effects" className="gap-2">
            <Beaker className="h-4 w-4" />
            Treatment Effects
          </TabsTrigger>
          <TabsTrigger value="segments" className="gap-2">
            <GitBranch className="h-4 w-4" />
            Segment Analysis
          </TabsTrigger>
          <TabsTrigger value="digital-twin" className="gap-2">
            <FlaskConical className="h-4 w-4" />
            Digital Twin
          </TabsTrigger>
        </TabsList>

        {/* Causal Impact Tab */}
        <TabsContent value="causal" className="space-y-6">
          <Card>
            <CardHeader>
              <div className="flex items-center justify-between">
                <div>
                  <CardTitle>Causal Impact Analysis</CardTitle>
                  <CardDescription>
                    Comparison of actual outcomes vs. counterfactual (what would have happened without intervention)
                  </CardDescription>
                </div>
                <div className="flex items-center gap-2">
                  <div className="flex items-center gap-1">
                    <div className="w-3 h-3 rounded-full bg-blue-500" />
                    <span className="text-xs">Actual</span>
                  </div>
                  <div className="flex items-center gap-1">
                    <div className="w-3 h-3 rounded-full bg-slate-400" />
                    <span className="text-xs">Counterfactual</span>
                  </div>
                  <div className="flex items-center gap-1">
                    <div className="w-3 h-1 bg-slate-300" />
                    <span className="text-xs">95% CI</span>
                  </div>
                </div>
              </div>
            </CardHeader>
            <CardContent>
              <ResponsiveContainer width="100%" height={400}>
                <ComposedChart data={SAMPLE_IMPACT_DATA} margin={{ top: 20, right: 30, left: 20, bottom: 5 }}>
                  <CartesianGrid strokeDasharray="3 3" opacity={0.3} />
                  <XAxis
                    dataKey="date"
                    tickFormatter={formatDate}
                    fontSize={12}
                    tickLine={false}
                  />
                  <YAxis fontSize={12} tickLine={false} axisLine={false} />
                  <Tooltip content={<ImpactTooltip />} />
                  <Legend />

                  {/* Confidence interval area */}
                  <Area
                    type="monotone"
                    dataKey="upperBound"
                    stroke="none"
                    fill="hsl(var(--muted))"
                    fillOpacity={0.5}
                    name="Upper CI"
                  />
                  <Area
                    type="monotone"
                    dataKey="lowerBound"
                    stroke="none"
                    fill="white"
                    fillOpacity={1}
                    name="Lower CI"
                  />

                  {/* Counterfactual line */}
                  <Line
                    type="monotone"
                    dataKey="counterfactual"
                    stroke="hsl(var(--muted-foreground))"
                    strokeWidth={2}
                    strokeDasharray="5 5"
                    dot={false}
                    name="Counterfactual"
                  />

                  {/* Actual line */}
                  <Line
                    type="monotone"
                    dataKey="actual"
                    stroke="hsl(var(--chart-1))"
                    strokeWidth={2}
                    dot={false}
                    name="Actual"
                  />

                  {/* Intervention start line */}
                  <ReferenceLine
                    x={SAMPLE_IMPACT_DATA[30].date}
                    stroke="hsl(var(--destructive))"
                    strokeDasharray="3 3"
                    label={{ value: 'Intervention Start', position: 'top', fontSize: 10, fill: 'hsl(var(--destructive))' }}
                  />
                </ComposedChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>

          {/* Impact Interpretation */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Info className="h-5 w-5" />
                Impact Interpretation
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                <div className="space-y-2">
                  <h4 className="font-medium text-emerald-600">Positive Impact Detected</h4>
                  <p className="text-sm text-muted-foreground">
                    The intervention shows a statistically significant positive effect. The actual outcomes
                    consistently exceed the counterfactual prediction, with the gap widening over time.
                  </p>
                </div>
                <div className="space-y-2">
                  <h4 className="font-medium">Confidence Level: 95%</h4>
                  <p className="text-sm text-muted-foreground">
                    The shaded area represents the 95% confidence interval for the counterfactual.
                    Since actual values fall above this range, we can be confident the effect is real.
                  </p>
                </div>
                <div className="space-y-2">
                  <h4 className="font-medium">Methodology: CausalImpact</h4>
                  <p className="text-sm text-muted-foreground">
                    Using Bayesian structural time-series model to estimate what would have happened
                    without the intervention, controlling for trends and seasonality.
                  </p>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        {/* Before/After Tab */}
        <TabsContent value="beforeafter" className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle>Before/After Comparison</CardTitle>
              <CardDescription>
                Metric changes comparing pre-intervention and post-intervention periods
              </CardDescription>
            </CardHeader>
            <CardContent>
              <ResponsiveContainer width="100%" height={300}>
                <BarChart
                  data={SAMPLE_BEFORE_AFTER}
                  layout="vertical"
                  margin={{ top: 5, right: 30, left: 100, bottom: 5 }}
                >
                  <CartesianGrid strokeDasharray="3 3" opacity={0.3} />
                  <XAxis type="number" fontSize={12} tickLine={false} />
                  <YAxis dataKey="metric" type="category" fontSize={12} tickLine={false} width={90} />
                  <Tooltip />
                  <Legend />
                  <Bar dataKey="beforeMean" fill="hsl(var(--muted-foreground))" name="Before" />
                  <Bar dataKey="afterMean" fill="hsl(var(--chart-1))" name="After" />
                </BarChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>

          {/* Before/After Details Table */}
          <Card>
            <CardHeader>
              <CardTitle>Detailed Comparison</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="overflow-x-auto">
                <table className="w-full">
                  <thead>
                    <tr className="border-b border-border">
                      <th className="text-left py-3 px-4 font-medium">Metric</th>
                      <th className="text-right py-3 px-4 font-medium">Before</th>
                      <th className="text-right py-3 px-4 font-medium">After</th>
                      <th className="text-right py-3 px-4 font-medium">Change</th>
                      <th className="text-right py-3 px-4 font-medium">% Change</th>
                      <th className="text-center py-3 px-4 font-medium">Status</th>
                    </tr>
                  </thead>
                  <tbody>
                    {SAMPLE_BEFORE_AFTER.map((comparison) => (
                      <tr key={comparison.metric} className="border-b border-border hover:bg-muted/50">
                        <td className="py-3 px-4 font-medium">{comparison.metric}</td>
                        <td className="py-3 px-4 text-right text-muted-foreground">
                          {comparison.beforeMean.toLocaleString()}
                        </td>
                        <td className="py-3 px-4 text-right font-medium">
                          {comparison.afterMean.toLocaleString()}
                        </td>
                        <td className={cn(
                          'py-3 px-4 text-right font-medium',
                          comparison.isPositive ? 'text-emerald-600' : 'text-rose-600'
                        )}>
                          {comparison.change > 0 ? '+' : ''}{comparison.change.toLocaleString()}
                        </td>
                        <td className={cn(
                          'py-3 px-4 text-right font-medium',
                          comparison.isPositive ? 'text-emerald-600' : 'text-rose-600'
                        )}>
                          {comparison.changePercent > 0 ? '+' : ''}{comparison.changePercent.toFixed(1)}%
                        </td>
                        <td className="py-3 px-4 text-center">
                          {comparison.isPositive ? (
                            <CheckCircle2 className="h-5 w-5 text-emerald-500 inline" />
                          ) : (
                            <XCircle className="h-5 w-5 text-rose-500 inline" />
                          )}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        {/* Treatment Effects Tab */}
        <TabsContent value="effects" className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle>Treatment Effect Estimates</CardTitle>
              <CardDescription>
                Statistical estimates of causal effects with confidence intervals and significance tests
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-6">
                {SAMPLE_TREATMENT_EFFECTS.map((effect) => (
                  <div
                    key={effect.id}
                    className={cn(
                      'p-4 rounded-lg border',
                      effect.isSignificant ? 'border-emerald-200 bg-emerald-50/50 dark:border-emerald-900 dark:bg-emerald-950/20' : 'border-border'
                    )}
                  >
                    <div className="flex items-start justify-between mb-3">
                      <div>
                        <h4 className="font-medium">{effect.metric}</h4>
                        <p className="text-sm text-muted-foreground">{effect.intervention}</p>
                      </div>
                      <div className="flex items-center gap-2">
                        <Badge variant="outline" className={cn(
                          effect.effectSize === 'large' ? 'bg-purple-100 text-purple-700' :
                          effect.effectSize === 'medium' ? 'bg-blue-100 text-blue-700' :
                          'bg-slate-100 text-slate-700'
                        )}>
                          {effect.effectSize} effect
                        </Badge>
                        {effect.isSignificant ? (
                          <Badge variant="outline" className="bg-emerald-100 text-emerald-700">
                            Significant
                          </Badge>
                        ) : (
                          <Badge variant="outline" className="bg-amber-100 text-amber-700">
                            Not Significant
                          </Badge>
                        )}
                      </div>
                    </div>

                    <div className="grid grid-cols-4 gap-4 mb-3">
                      <div>
                        <p className="text-xs text-muted-foreground">ATE (Average Treatment Effect)</p>
                        <p className={cn(
                          'text-lg font-bold',
                          effect.ate > 0 ? 'text-emerald-600' : 'text-rose-600'
                        )}>
                          {effect.ate > 0 ? '+' : ''}{effect.ate.toFixed(1)}
                        </p>
                      </div>
                      <div>
                        <p className="text-xs text-muted-foreground">95% Confidence Interval</p>
                        <p className="text-lg font-medium">
                          [{effect.ci[0].toFixed(1)}, {effect.ci[1].toFixed(1)}]
                        </p>
                      </div>
                      <div>
                        <p className="text-xs text-muted-foreground">P-Value</p>
                        <p className={cn(
                          'text-lg font-medium',
                          effect.pValue < 0.05 ? 'text-emerald-600' : 'text-muted-foreground'
                        )}>
                          {effect.pValue < 0.001 ? '< 0.001' : effect.pValue.toFixed(4)}
                        </p>
                      </div>
                      <div>
                        <p className="text-xs text-muted-foreground">Sample Size</p>
                        <p className="text-lg font-medium">{effect.sampleSize.toLocaleString()}</p>
                      </div>
                    </div>

                    {/* Visual CI representation */}
                    <div className="relative h-8 bg-muted rounded-full overflow-hidden">
                      <div
                        className="absolute h-full bg-blue-200 dark:bg-blue-800"
                        style={{
                          left: `${((effect.ci[0] / (effect.ci[1] * 1.5)) * 50) + 25}%`,
                          width: `${((effect.ci[1] - effect.ci[0]) / (effect.ci[1] * 1.5)) * 50}%`,
                        }}
                      />
                      <div
                        className="absolute h-full w-1 bg-blue-600 dark:bg-blue-400"
                        style={{
                          left: `${((effect.ate / (effect.ci[1] * 1.5)) * 50) + 25}%`,
                        }}
                      />
                      <div
                        className="absolute h-full w-0.5 bg-slate-400"
                        style={{ left: '25%' }}
                      />
                    </div>
                    <div className="flex justify-between text-xs text-muted-foreground mt-1">
                      <span>0</span>
                      <span className="text-xs text-muted-foreground">
                        {getEffectSizeLabel(effect.effectSize)}
                      </span>
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        {/* Segment Analysis Tab */}
        <TabsContent value="segments" className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle>Heterogeneous Treatment Effects</CardTitle>
              <CardDescription>
                How the intervention impact varies across different segments
              </CardDescription>
            </CardHeader>
            <CardContent>
              <ResponsiveContainer width="100%" height={350}>
                <BarChart
                  data={SAMPLE_SEGMENT_EFFECTS}
                  layout="vertical"
                  margin={{ top: 5, right: 30, left: 120, bottom: 5 }}
                >
                  <CartesianGrid strokeDasharray="3 3" opacity={0.3} />
                  <XAxis type="number" fontSize={12} tickLine={false} />
                  <YAxis dataKey="segment" type="category" fontSize={12} tickLine={false} width={110} />
                  <Tooltip
                    formatter={(value: number, name: string) => {
                      if (name === 'effect') return [`+${value.toFixed(1)}`, 'Treatment Effect'];
                      return [value, name];
                    }}
                  />
                  <Bar dataKey="effect" fill="hsl(var(--chart-1))" name="effect" radius={[0, 4, 4, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>

          {/* Segment Details */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {SAMPLE_SEGMENT_EFFECTS.map((segment) => (
              <Card key={segment.segment}>
                <CardContent className="pt-4">
                  <div className="flex items-center justify-between mb-2">
                    <h4 className="font-medium">{segment.segment}</h4>
                    {segment.isSignificant ? (
                      <CheckCircle2 className="h-4 w-4 text-emerald-500" />
                    ) : (
                      <AlertTriangle className="h-4 w-4 text-amber-500" />
                    )}
                  </div>
                  <div className="space-y-2">
                    <div className="flex justify-between text-sm">
                      <span className="text-muted-foreground">Treatment Effect</span>
                      <span className="font-medium text-emerald-600">+{segment.effect.toFixed(1)}</span>
                    </div>
                    <div className="flex justify-between text-sm">
                      <span className="text-muted-foreground">95% CI</span>
                      <span className="font-medium">[{segment.ci[0].toFixed(1)}, {segment.ci[1].toFixed(1)}]</span>
                    </div>
                    <div className="flex justify-between text-sm">
                      <span className="text-muted-foreground">Sample Size</span>
                      <span className="font-medium">{segment.sampleSize}</span>
                    </div>
                    <Progress
                      value={(segment.effect / Math.max(...SAMPLE_SEGMENT_EFFECTS.map(s => s.effect))) * 100}
                      className="h-2"
                    />
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>

          {/* Segment Insights */}
          <Card>
            <CardHeader>
              <CardTitle>Key Insights</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div className="space-y-2">
                  <h4 className="font-medium flex items-center gap-2">
                    <TrendingUp className="h-4 w-4 text-emerald-500" />
                    Highest Impact Segment
                  </h4>
                  <p className="text-sm text-muted-foreground">
                    <strong>High-Volume HCPs</strong> show the strongest response to the intervention
                    with an average treatment effect of +112.5 units. This segment should be prioritized
                    for similar future interventions.
                  </p>
                </div>
                <div className="space-y-2">
                  <h4 className="font-medium flex items-center gap-2">
                    <Target className="h-4 w-4 text-blue-500" />
                    Regional Performance
                  </h4>
                  <p className="text-sm text-muted-foreground">
                    The <strong>Northeast Region</strong> outperforms other regions with a +95.3 treatment effect,
                    suggesting the intervention design may be particularly effective for this market.
                  </p>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        {/* Digital Twin Tab */}
        <TabsContent value="digital-twin" className="space-y-6">
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            {/* Simulation Panel - Left Side */}
            <div className="lg:col-span-1">
              <SimulationPanel
                onSimulate={handleSimulate}
                isSimulating={isSimulating}
                initialBrand="Remibrutinib"
                brands={['Remibrutinib', 'Fabhalta', 'Kisqali']}
              />
            </div>

            {/* Results and Recommendations - Right Side */}
            <div className="lg:col-span-2 space-y-6">
              <ScenarioResults
                results={simulationResults}
                isLoading={isSimulating}
              />

              <RecommendationCards
                recommendation={simulationResults?.recommendation ?? null}
                onAccept={() => {
                  // TODO: Implement deployment flow
                  console.log('Accepting recommendation');
                }}
                onRefine={() => {
                  // TODO: Scroll back to simulation panel or open refinement dialog
                  console.log('Refining parameters');
                }}
                onAnalyze={() => {
                  // TODO: Trigger deeper analysis
                  console.log('Running additional analysis');
                }}
              />
            </div>
          </div>

          {/* Digital Twin Context Card */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Info className="h-5 w-5" />
                About Digital Twin Simulation
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                <div className="space-y-2">
                  <h4 className="font-medium text-blue-600">Pre-Screen Interventions</h4>
                  <p className="text-sm text-muted-foreground">
                    Test intervention scenarios virtually before committing real resources.
                    The digital twin models HCP behavior and market dynamics to predict outcomes.
                  </p>
                </div>
                <div className="space-y-2">
                  <h4 className="font-medium text-emerald-600">Causal Inference Engine</h4>
                  <p className="text-sm text-muted-foreground">
                    Powered by DoWhy and EconML, the simulation uses causal models trained
                    on historical data to estimate treatment effects and confidence intervals.
                  </p>
                </div>
                <div className="space-y-2">
                  <h4 className="font-medium text-amber-600">Fidelity Metrics</h4>
                  <p className="text-sm text-muted-foreground">
                    Each simulation includes fidelity scores indicating how well the model
                    represents your specific market conditions and data coverage.
                  </p>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
}

export default InterventionImpact;
