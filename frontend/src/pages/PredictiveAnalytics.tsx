/**
 * PredictiveAnalytics Page
 * ========================
 *
 * Comprehensive predictive analytics dashboard showing:
 * - Risk scores and probability distributions
 * - Uplift models and targeting recommendations
 * - Prediction confidence and model outputs
 * - AI-generated recommendations
 *
 * @module pages/PredictiveAnalytics
 */

import * as React from 'react';
import {
  AreaChart,
  Area,
  BarChart,
  Bar,
  ComposedChart,
  Line,
  ScatterChart,
  Scatter,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  ReferenceLine,
  Cell,
} from 'recharts';
import {
  TrendingUp,
  TrendingDown,
  Target,
  Users,
  AlertTriangle,
  CheckCircle2,
  Sparkles,
  ArrowUpRight,
  ArrowDownRight,
  Filter,
  Download,
  RefreshCw,
  Zap,
  BarChart3,
  Activity,
  Brain,
  Lightbulb,
} from 'lucide-react';

import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Progress } from '@/components/ui/progress';
import { KPICard } from '@/components/visualizations/dashboard/KPICard';
import { StatusBadge } from '@/components/visualizations/dashboard/StatusBadge';

// =============================================================================
// TYPES
// =============================================================================

interface RiskScore {
  id: string;
  entity: string;
  entityType: 'hcp' | 'account' | 'patient' | 'territory';
  riskCategory: 'churn' | 'adoption' | 'conversion' | 'engagement';
  score: number;
  probability: number;
  confidence: number;
  factors: { name: string; impact: number }[];
  trend: 'increasing' | 'decreasing' | 'stable';
  lastUpdated: string;
}

interface ProbabilityDistribution {
  bucket: string;
  count: number;
  cumulative: number;
  avgScore: number;
}

interface UpliftSegment {
  id: string;
  name: string;
  size: number;
  uplift: number;
  baselineConversion: number;
  predictedConversion: number;
  roi: number;
  recommendedAction: string;
}

interface Recommendation {
  id: string;
  priority: 'high' | 'medium' | 'low';
  type: 'targeting' | 'timing' | 'channel' | 'messaging';
  title: string;
  description: string;
  impact: string;
  confidence: number;
  actionable: boolean;
}

interface ModelMetrics {
  accuracy: number;
  precision: number;
  recall: number;
  f1Score: number;
  auc: number;
  lastTrained: string;
  trainingSize: number;
}

// =============================================================================
// SAMPLE DATA GENERATORS
// =============================================================================

function generateRiskScores(): RiskScore[] {
  const entities = [
    { name: 'Dr. Sarah Chen', type: 'hcp' },
    { name: 'Dr. Michael Roberts', type: 'hcp' },
    { name: 'Memorial Hospital', type: 'account' },
    { name: 'Pacific Medical Center', type: 'account' },
    { name: 'Northeast Territory', type: 'territory' },
    { name: 'Dr. Emily Watson', type: 'hcp' },
    { name: 'City Health Network', type: 'account' },
    { name: 'Dr. James Liu', type: 'hcp' },
    { name: 'Southwest Region', type: 'territory' },
    { name: 'Dr. Amanda Foster', type: 'hcp' },
  ];

  const categories: RiskScore['riskCategory'][] = ['churn', 'adoption', 'conversion', 'engagement'];
  const trends: RiskScore['trend'][] = ['increasing', 'decreasing', 'stable'];

  return entities.map((entity, i) => ({
    id: `risk-${i}`,
    entity: entity.name,
    entityType: entity.type as RiskScore['entityType'],
    riskCategory: categories[i % categories.length],
    score: Math.round(30 + Math.random() * 60),
    probability: Math.round((0.3 + Math.random() * 0.5) * 100) / 100,
    confidence: Math.round((0.7 + Math.random() * 0.25) * 100) / 100,
    factors: [
      { name: 'Recent Activity', impact: Math.round((-0.3 + Math.random() * 0.6) * 100) / 100 },
      { name: 'Historical Pattern', impact: Math.round((-0.3 + Math.random() * 0.6) * 100) / 100 },
      { name: 'Engagement Score', impact: Math.round((-0.3 + Math.random() * 0.6) * 100) / 100 },
    ],
    trend: trends[Math.floor(Math.random() * trends.length)],
    lastUpdated: new Date(Date.now() - Math.random() * 7 * 24 * 60 * 60 * 1000).toISOString(),
  }));
}

function generateProbabilityDistribution(): ProbabilityDistribution[] {
  const buckets = ['0-10%', '10-20%', '20-30%', '30-40%', '40-50%', '50-60%', '60-70%', '70-80%', '80-90%', '90-100%'];
  let cumulative = 0;

  return buckets.map((bucket, i) => {
    // Create a roughly normal distribution
    const mean = 5;
    const stdDev = 2;
    const z = (i - mean) / stdDev;
    const count = Math.max(5, Math.round(100 * Math.exp(-0.5 * z * z)));
    cumulative += count;

    return {
      bucket,
      count,
      cumulative,
      avgScore: (i + 0.5) * 10,
    };
  });
}

function generateUpliftSegments(): UpliftSegment[] {
  return [
    {
      id: 'seg-1',
      name: 'High-Value Responders',
      size: 1250,
      uplift: 0.35,
      baselineConversion: 0.12,
      predictedConversion: 0.47,
      roi: 4.2,
      recommendedAction: 'Prioritize for intensive engagement',
    },
    {
      id: 'seg-2',
      name: 'Persuadables',
      size: 3400,
      uplift: 0.22,
      baselineConversion: 0.08,
      predictedConversion: 0.30,
      roi: 2.8,
      recommendedAction: 'Target with personalized messaging',
    },
    {
      id: 'seg-3',
      name: 'Sure Things',
      size: 890,
      uplift: 0.05,
      baselineConversion: 0.65,
      predictedConversion: 0.70,
      roi: 0.5,
      recommendedAction: 'Maintain light touch engagement',
    },
    {
      id: 'seg-4',
      name: 'Lost Causes',
      size: 2100,
      uplift: 0.02,
      baselineConversion: 0.03,
      predictedConversion: 0.05,
      roi: -0.2,
      recommendedAction: 'Deprioritize - low ROI potential',
    },
    {
      id: 'seg-5',
      name: 'Sleeping Giants',
      size: 780,
      uplift: 0.28,
      baselineConversion: 0.05,
      predictedConversion: 0.33,
      roi: 3.5,
      recommendedAction: 'Increase touchpoint frequency',
    },
  ];
}

function generateRecommendations(): Recommendation[] {
  return [
    {
      id: 'rec-1',
      priority: 'high',
      type: 'targeting',
      title: 'Focus on High-Value Responders Segment',
      description: 'Analysis shows 35% uplift potential in this segment. Allocating 40% more resources could yield 2.3x ROI improvement.',
      impact: '+$2.4M projected impact',
      confidence: 0.89,
      actionable: true,
    },
    {
      id: 'rec-2',
      priority: 'high',
      type: 'timing',
      title: 'Optimal Engagement Window Detected',
      description: 'Predictive model indicates Tuesday-Thursday, 10am-2pm shows 28% higher response rates for target HCPs.',
      impact: '+18% engagement rate',
      confidence: 0.92,
      actionable: true,
    },
    {
      id: 'rec-3',
      priority: 'medium',
      type: 'channel',
      title: 'Channel Mix Optimization',
      description: 'Shift 15% budget from email to virtual meetings for oncology specialists based on response patterns.',
      impact: '+12% conversion rate',
      confidence: 0.78,
      actionable: true,
    },
    {
      id: 'rec-4',
      priority: 'medium',
      type: 'messaging',
      title: 'Personalize Clinical Data Presentation',
      description: 'HCPs with research background show 45% higher engagement with detailed efficacy data visualizations.',
      impact: '+22% content engagement',
      confidence: 0.84,
      actionable: true,
    },
    {
      id: 'rec-5',
      priority: 'low',
      type: 'targeting',
      title: 'Expand to Adjacent Specialties',
      description: 'Early signals suggest opportunity in rheumatology crossover patients. Recommend pilot program.',
      impact: 'New market potential',
      confidence: 0.65,
      actionable: false,
    },
  ];
}

function generateModelMetrics(): ModelMetrics {
  return {
    accuracy: 0.87,
    precision: 0.84,
    recall: 0.81,
    f1Score: 0.82,
    auc: 0.91,
    lastTrained: new Date(Date.now() - 2 * 24 * 60 * 60 * 1000).toISOString(),
    trainingSize: 45000,
  };
}

function generateScoreDistributionData(): { score: number; actual: number; predicted: number }[] {
  return Array.from({ length: 20 }, (_, i) => {
    const score = (i + 1) * 5;
    return {
      score,
      actual: Math.round(10 + Math.random() * 30 + (score > 50 ? score * 0.5 : 0)),
      predicted: Math.round(15 + Math.random() * 25 + (score > 50 ? score * 0.45 : 0)),
    };
  });
}

// =============================================================================
// CUSTOM COMPONENTS
// =============================================================================

interface RiskScoreCardProps {
  score: RiskScore;
}

function RiskScoreCard({ score }: RiskScoreCardProps) {
  const getRiskColor = (value: number) => {
    if (value >= 70) return 'text-rose-600';
    if (value >= 40) return 'text-amber-600';
    return 'text-emerald-600';
  };

  const getRiskBg = (value: number) => {
    if (value >= 70) return 'bg-rose-100 dark:bg-rose-900/30';
    if (value >= 40) return 'bg-amber-100 dark:bg-amber-900/30';
    return 'bg-emerald-100 dark:bg-emerald-900/30';
  };

  const getCategoryLabel = (category: RiskScore['riskCategory']) => {
    switch (category) {
      case 'churn': return 'Churn Risk';
      case 'adoption': return 'Adoption Likelihood';
      case 'conversion': return 'Conversion Probability';
      case 'engagement': return 'Engagement Score';
      default: return category;
    }
  };

  return (
    <Card className="hover:shadow-md transition-shadow">
      <CardContent className="p-4">
        <div className="flex items-start justify-between mb-3">
          <div className="flex-1 min-w-0">
            <p className="font-medium text-sm truncate">{score.entity}</p>
            <div className="flex items-center gap-2 mt-1">
              <Badge variant="outline" className="text-xs">
                {score.entityType.toUpperCase()}
              </Badge>
              <span className="text-xs text-muted-foreground">
                {getCategoryLabel(score.riskCategory)}
              </span>
            </div>
          </div>
          <div className={`text-2xl font-bold ${getRiskColor(score.score)}`}>
            {score.score}
          </div>
        </div>

        <div className="space-y-2">
          <div className="flex items-center justify-between text-xs">
            <span className="text-muted-foreground">Probability</span>
            <span className="font-medium">{(score.probability * 100).toFixed(0)}%</span>
          </div>
          <Progress value={score.probability * 100} className="h-1.5" />

          <div className="flex items-center justify-between text-xs mt-2">
            <span className="text-muted-foreground">Confidence</span>
            <span className="font-medium">{(score.confidence * 100).toFixed(0)}%</span>
          </div>

          <div className="flex items-center justify-between text-xs mt-2">
            <span className="text-muted-foreground">Trend</span>
            <div className="flex items-center gap-1">
              {score.trend === 'increasing' && (
                <>
                  <ArrowUpRight className="h-3 w-3 text-rose-500" />
                  <span className="text-rose-600">Increasing</span>
                </>
              )}
              {score.trend === 'decreasing' && (
                <>
                  <ArrowDownRight className="h-3 w-3 text-emerald-500" />
                  <span className="text-emerald-600">Decreasing</span>
                </>
              )}
              {score.trend === 'stable' && (
                <span className="text-muted-foreground">Stable</span>
              )}
            </div>
          </div>
        </div>

        <div className="mt-3 pt-3 border-t">
          <p className="text-xs text-muted-foreground mb-2">Key Factors</p>
          <div className="space-y-1">
            {score.factors.slice(0, 3).map((factor, i) => (
              <div key={i} className="flex items-center justify-between text-xs">
                <span className="truncate">{factor.name}</span>
                <span className={factor.impact > 0 ? 'text-emerald-600' : 'text-rose-600'}>
                  {factor.impact > 0 ? '+' : ''}{(factor.impact * 100).toFixed(0)}%
                </span>
              </div>
            ))}
          </div>
        </div>
      </CardContent>
    </Card>
  );
}

interface UpliftSegmentCardProps {
  segment: UpliftSegment;
}

function UpliftSegmentCard({ segment }: UpliftSegmentCardProps) {
  const getROIColor = (roi: number) => {
    if (roi >= 3) return 'text-emerald-600';
    if (roi >= 1) return 'text-blue-600';
    if (roi >= 0) return 'text-amber-600';
    return 'text-rose-600';
  };

  return (
    <Card className="hover:shadow-md transition-shadow">
      <CardContent className="p-4">
        <div className="flex items-start justify-between mb-3">
          <div>
            <p className="font-semibold">{segment.name}</p>
            <p className="text-sm text-muted-foreground">
              {segment.size.toLocaleString()} entities
            </p>
          </div>
          <Badge
            variant="outline"
            className={getROIColor(segment.roi)}
          >
            {segment.roi >= 0 ? '+' : ''}{segment.roi.toFixed(1)}x ROI
          </Badge>
        </div>

        <div className="grid grid-cols-3 gap-4 mb-4">
          <div className="text-center">
            <p className="text-xs text-muted-foreground">Baseline</p>
            <p className="font-semibold">{(segment.baselineConversion * 100).toFixed(0)}%</p>
          </div>
          <div className="text-center">
            <p className="text-xs text-muted-foreground">Predicted</p>
            <p className="font-semibold text-blue-600">
              {(segment.predictedConversion * 100).toFixed(0)}%
            </p>
          </div>
          <div className="text-center">
            <p className="text-xs text-muted-foreground">Uplift</p>
            <p className="font-semibold text-emerald-600">
              +{(segment.uplift * 100).toFixed(0)}%
            </p>
          </div>
        </div>

        <div className="p-2 rounded-lg bg-muted/50">
          <div className="flex items-start gap-2">
            <Lightbulb className="h-4 w-4 text-amber-500 shrink-0 mt-0.5" />
            <p className="text-xs">{segment.recommendedAction}</p>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}

interface RecommendationCardProps {
  recommendation: Recommendation;
}

function RecommendationCard({ recommendation }: RecommendationCardProps) {
  const getPriorityColor = (priority: Recommendation['priority']) => {
    switch (priority) {
      case 'high': return 'bg-rose-100 text-rose-700 dark:bg-rose-900/30 dark:text-rose-400';
      case 'medium': return 'bg-amber-100 text-amber-700 dark:bg-amber-900/30 dark:text-amber-400';
      case 'low': return 'bg-slate-100 text-slate-700 dark:bg-slate-800 dark:text-slate-400';
    }
  };

  const getTypeIcon = (type: Recommendation['type']) => {
    switch (type) {
      case 'targeting': return Target;
      case 'timing': return Activity;
      case 'channel': return Zap;
      case 'messaging': return Sparkles;
      default: return Lightbulb;
    }
  };

  const TypeIcon = getTypeIcon(recommendation.type);

  return (
    <Card className={`hover:shadow-md transition-shadow ${!recommendation.actionable ? 'opacity-75' : ''}`}>
      <CardContent className="p-4">
        <div className="flex items-start gap-3">
          <div className="p-2 rounded-lg bg-blue-100 dark:bg-blue-900/30">
            <TypeIcon className="h-5 w-5 text-blue-600" />
          </div>

          <div className="flex-1 min-w-0">
            <div className="flex items-center gap-2 mb-1">
              <Badge className={getPriorityColor(recommendation.priority)}>
                {recommendation.priority.toUpperCase()}
              </Badge>
              <Badge variant="outline" className="text-xs">
                {recommendation.type}
              </Badge>
              {!recommendation.actionable && (
                <Badge variant="secondary" className="text-xs">
                  Needs Validation
                </Badge>
              )}
            </div>

            <h4 className="font-semibold text-sm mb-1">{recommendation.title}</h4>
            <p className="text-sm text-muted-foreground mb-2">
              {recommendation.description}
            </p>

            <div className="flex items-center justify-between">
              <div className="flex items-center gap-1 text-emerald-600">
                <TrendingUp className="h-4 w-4" />
                <span className="text-sm font-medium">{recommendation.impact}</span>
              </div>
              <div className="text-xs text-muted-foreground">
                {(recommendation.confidence * 100).toFixed(0)}% confidence
              </div>
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}

// Custom tooltip for probability distribution
function ProbabilityTooltip({ active, payload, label }: { active?: boolean; payload?: any[]; label?: string }) {
  if (!active || !payload || payload.length === 0) return null;

  return (
    <div className="bg-popover border rounded-lg shadow-lg p-3">
      <p className="font-semibold mb-2">{label}</p>
      {payload.map((entry: any, index: number) => (
        <div key={index} className="flex items-center justify-between gap-4 text-sm">
          <span style={{ color: entry.color }}>{entry.name}</span>
          <span className="font-medium">{entry.value.toLocaleString()}</span>
        </div>
      ))}
    </div>
  );
}

// =============================================================================
// MAIN COMPONENT
// =============================================================================

function PredictiveAnalytics() {
  // State
  const [selectedModel, setSelectedModel] = React.useState('churn');
  const [selectedTimeframe, setSelectedTimeframe] = React.useState('30d');
  const [activeTab, setActiveTab] = React.useState('risk-scores');

  // Generate sample data
  const riskScores = React.useMemo(() => generateRiskScores(), []);
  const probabilityDist = React.useMemo(() => generateProbabilityDistribution(), []);
  const upliftSegments = React.useMemo(() => generateUpliftSegments(), []);
  const recommendations = React.useMemo(() => generateRecommendations(), []);
  const modelMetrics = React.useMemo(() => generateModelMetrics(), []);
  const scoreDistribution = React.useMemo(() => generateScoreDistributionData(), []);

  // Calculate summary metrics
  const summaryMetrics = React.useMemo(() => {
    const highRisk = riskScores.filter(s => s.score >= 70).length;
    const avgConfidence = riskScores.reduce((sum, s) => sum + s.confidence, 0) / riskScores.length;
    const totalUplift = upliftSegments.reduce((sum, s) => sum + s.uplift * s.size, 0);
    const totalSize = upliftSegments.reduce((sum, s) => sum + s.size, 0);

    return {
      highRiskCount: highRisk,
      avgConfidence: avgConfidence,
      avgUplift: totalUplift / totalSize,
      topRecommendations: recommendations.filter(r => r.priority === 'high').length,
    };
  }, [riskScores, upliftSegments, recommendations]);

  return (
    <div className="container mx-auto px-4 py-8">
      {/* Header */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 mb-6">
        <div>
          <h1 className="text-3xl font-bold mb-2">Predictive Analytics</h1>
          <p className="text-muted-foreground">
            Risk scores, probability distributions, uplift models, and AI-powered recommendations.
          </p>
        </div>

        <div className="flex items-center gap-2">
          <Select value={selectedModel} onValueChange={setSelectedModel}>
            <SelectTrigger className="w-[160px]">
              <SelectValue placeholder="Select model" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="churn">Churn Model</SelectItem>
              <SelectItem value="adoption">Adoption Model</SelectItem>
              <SelectItem value="conversion">Conversion Model</SelectItem>
              <SelectItem value="engagement">Engagement Model</SelectItem>
            </SelectContent>
          </Select>

          <Select value={selectedTimeframe} onValueChange={setSelectedTimeframe}>
            <SelectTrigger className="w-[120px]">
              <SelectValue placeholder="Timeframe" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="7d">7 days</SelectItem>
              <SelectItem value="30d">30 days</SelectItem>
              <SelectItem value="90d">90 days</SelectItem>
            </SelectContent>
          </Select>

          <Button variant="outline" size="icon">
            <RefreshCw className="h-4 w-4" />
          </Button>
          <Button variant="outline" size="icon">
            <Download className="h-4 w-4" />
          </Button>
        </div>
      </div>

      {/* Model Performance Summary */}
      <Card className="mb-6 bg-gradient-to-r from-blue-50 to-indigo-50 dark:from-blue-900/20 dark:to-indigo-900/20">
        <CardContent className="p-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="p-2 rounded-lg bg-blue-100 dark:bg-blue-900/30">
                <Brain className="h-6 w-6 text-blue-600" />
              </div>
              <div>
                <p className="text-sm text-muted-foreground">Active Model</p>
                <p className="font-semibold">
                  {selectedModel.charAt(0).toUpperCase() + selectedModel.slice(1)} Prediction Model
                </p>
              </div>
            </div>

            <div className="flex items-center gap-8">
              <div className="text-center">
                <p className="text-xs text-muted-foreground">AUC-ROC</p>
                <p className="text-xl font-bold text-blue-600">{(modelMetrics.auc * 100).toFixed(1)}%</p>
              </div>
              <div className="text-center">
                <p className="text-xs text-muted-foreground">Accuracy</p>
                <p className="text-xl font-bold">{(modelMetrics.accuracy * 100).toFixed(1)}%</p>
              </div>
              <div className="text-center">
                <p className="text-xs text-muted-foreground">F1 Score</p>
                <p className="text-xl font-bold">{(modelMetrics.f1Score * 100).toFixed(1)}%</p>
              </div>
              <div className="text-center">
                <p className="text-xs text-muted-foreground">Last Trained</p>
                <p className="text-sm font-medium">
                  {new Date(modelMetrics.lastTrained).toLocaleDateString()}
                </p>
              </div>
              <div>
                <StatusBadge status="healthy" label="Model Healthy" />
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* KPI Cards */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-6">
        <KPICard
          title="High Risk Entities"
          value={summaryMetrics.highRiskCount}
          suffix=" entities"
          change={-2}
          changeLabel="vs last week"
          icon={<AlertTriangle className="h-5 w-5" />}
          status="warning"
        />
        <KPICard
          title="Avg Model Confidence"
          value={summaryMetrics.avgConfidence * 100}
          suffix="%"
          change={3.5}
          changeLabel="vs last week"
          icon={<Target className="h-5 w-5" />}
          status="healthy"
        />
        <KPICard
          title="Avg Uplift Potential"
          value={summaryMetrics.avgUplift * 100}
          suffix="%"
          change={1.2}
          changeLabel="vs last week"
          icon={<TrendingUp className="h-5 w-5" />}
          status="healthy"
        />
        <KPICard
          title="High Priority Actions"
          value={summaryMetrics.topRecommendations}
          suffix=" actions"
          icon={<Sparkles className="h-5 w-5" />}
          status="warning"
        />
      </div>

      {/* Main Content Tabs */}
      <Tabs value={activeTab} onValueChange={setActiveTab}>
        <TabsList className="mb-6">
          <TabsTrigger value="risk-scores" className="flex items-center gap-2">
            <AlertTriangle className="h-4 w-4" />
            Risk Scores
          </TabsTrigger>
          <TabsTrigger value="distributions" className="flex items-center gap-2">
            <BarChart3 className="h-4 w-4" />
            Distributions
          </TabsTrigger>
          <TabsTrigger value="uplift" className="flex items-center gap-2">
            <TrendingUp className="h-4 w-4" />
            Uplift Models
          </TabsTrigger>
          <TabsTrigger value="recommendations" className="flex items-center gap-2">
            <Sparkles className="h-4 w-4" />
            Recommendations
          </TabsTrigger>
        </TabsList>

        {/* Risk Scores Tab */}
        <TabsContent value="risk-scores" className="space-y-6">
          <div className="flex items-center justify-between">
            <div>
              <h2 className="text-xl font-semibold">Entity Risk Scores</h2>
              <p className="text-sm text-muted-foreground">
                Individual risk assessments with contributing factors
              </p>
            </div>
            <Button variant="outline" size="sm">
              <Filter className="h-4 w-4 mr-2" />
              Filter
            </Button>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
            {riskScores.map((score) => (
              <RiskScoreCard key={score.id} score={score} />
            ))}
          </div>
        </TabsContent>

        {/* Distributions Tab */}
        <TabsContent value="distributions" className="space-y-6">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Probability Distribution */}
            <Card>
              <CardHeader>
                <CardTitle>Score Probability Distribution</CardTitle>
                <CardDescription>
                  Distribution of prediction scores across all entities
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="h-[300px]">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={probabilityDist}>
                      <CartesianGrid strokeDasharray="3 3" opacity={0.3} />
                      <XAxis
                        dataKey="bucket"
                        tick={{ fontSize: 11 }}
                        angle={-45}
                        textAnchor="end"
                        height={60}
                      />
                      <YAxis />
                      <Tooltip content={<ProbabilityTooltip />} />
                      <Bar dataKey="count" name="Entity Count" fill="#3b82f6" radius={[4, 4, 0, 0]}>
                        {probabilityDist.map((entry, index) => (
                          <Cell
                            key={`cell-${index}`}
                            fill={entry.avgScore >= 70 ? '#ef4444' : entry.avgScore >= 40 ? '#f59e0b' : '#22c55e'}
                          />
                        ))}
                      </Bar>
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </CardContent>
            </Card>

            {/* Calibration Plot */}
            <Card>
              <CardHeader>
                <CardTitle>Model Calibration</CardTitle>
                <CardDescription>
                  Predicted vs actual outcome rates
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="h-[300px]">
                  <ResponsiveContainer width="100%" height="100%">
                    <ComposedChart data={scoreDistribution}>
                      <CartesianGrid strokeDasharray="3 3" opacity={0.3} />
                      <XAxis dataKey="score" label={{ value: 'Predicted Score', position: 'bottom', offset: -5 }} />
                      <YAxis label={{ value: 'Outcome Rate (%)', angle: -90, position: 'insideLeft' }} />
                      <Tooltip />
                      <Legend />
                      <Line
                        type="monotone"
                        dataKey="actual"
                        name="Actual"
                        stroke="#22c55e"
                        strokeWidth={2}
                        dot={{ fill: '#22c55e', r: 4 }}
                      />
                      <Line
                        type="monotone"
                        dataKey="predicted"
                        name="Predicted"
                        stroke="#3b82f6"
                        strokeWidth={2}
                        strokeDasharray="5 5"
                        dot={{ fill: '#3b82f6', r: 4 }}
                      />
                      <ReferenceLine
                        stroke="#888"
                        strokeDasharray="3 3"
                        segment={[{ x: 0, y: 0 }, { x: 100, y: 100 }]}
                      />
                    </ComposedChart>
                  </ResponsiveContainer>
                </div>
              </CardContent>
            </Card>

            {/* Cumulative Distribution */}
            <Card className="lg:col-span-2">
              <CardHeader>
                <CardTitle>Cumulative Score Distribution</CardTitle>
                <CardDescription>
                  Running total of entities at each score threshold
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="h-[300px]">
                  <ResponsiveContainer width="100%" height="100%">
                    <AreaChart data={probabilityDist}>
                      <CartesianGrid strokeDasharray="3 3" opacity={0.3} />
                      <XAxis dataKey="bucket" />
                      <YAxis />
                      <Tooltip />
                      <Area
                        type="monotone"
                        dataKey="cumulative"
                        name="Cumulative Count"
                        stroke="#6366f1"
                        fill="#6366f1"
                        fillOpacity={0.3}
                      />
                    </AreaChart>
                  </ResponsiveContainer>
                </div>
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        {/* Uplift Models Tab */}
        <TabsContent value="uplift" className="space-y-6">
          <div className="flex items-center justify-between">
            <div>
              <h2 className="text-xl font-semibold">Uplift Model Segments</h2>
              <p className="text-sm text-muted-foreground">
                Identify high-impact segments for targeted interventions
              </p>
            </div>
          </div>

          {/* Uplift Visualization */}
          <Card>
            <CardHeader>
              <CardTitle>Segment Uplift Analysis</CardTitle>
              <CardDescription>
                Comparing baseline vs predicted conversion with uplift potential
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="h-[350px]">
                <ResponsiveContainer width="100%" height="100%">
                  <ComposedChart data={upliftSegments} layout="vertical">
                    <CartesianGrid strokeDasharray="3 3" opacity={0.3} />
                    <XAxis type="number" domain={[0, 1]} tickFormatter={(v) => `${(v * 100).toFixed(0)}%`} />
                    <YAxis type="category" dataKey="name" width={150} tick={{ fontSize: 12 }} />
                    <Tooltip
                      formatter={(value: number) => `${(value * 100).toFixed(1)}%`}
                    />
                    <Legend />
                    <Bar
                      dataKey="baselineConversion"
                      name="Baseline Conversion"
                      fill="#94a3b8"
                      barSize={20}
                      radius={[0, 4, 4, 0]}
                    />
                    <Bar
                      dataKey="predictedConversion"
                      name="Predicted Conversion"
                      fill="#3b82f6"
                      barSize={20}
                      radius={[0, 4, 4, 0]}
                    />
                  </ComposedChart>
                </ResponsiveContainer>
              </div>
            </CardContent>
          </Card>

          {/* Segment Cards */}
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {upliftSegments.map((segment) => (
              <UpliftSegmentCard key={segment.id} segment={segment} />
            ))}
          </div>

          {/* ROI Summary */}
          <Card>
            <CardHeader>
              <CardTitle>Segment ROI Comparison</CardTitle>
              <CardDescription>
                Expected return on investment by segment
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="h-[250px]">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={upliftSegments}>
                    <CartesianGrid strokeDasharray="3 3" opacity={0.3} />
                    <XAxis dataKey="name" tick={{ fontSize: 11 }} angle={-15} textAnchor="end" height={80} />
                    <YAxis tickFormatter={(v) => `${v}x`} />
                    <Tooltip formatter={(value: number) => `${value.toFixed(1)}x`} />
                    <Bar dataKey="roi" name="ROI Multiple" radius={[4, 4, 0, 0]}>
                      {upliftSegments.map((entry, index) => (
                        <Cell
                          key={`cell-${index}`}
                          fill={entry.roi >= 3 ? '#22c55e' : entry.roi >= 1 ? '#3b82f6' : entry.roi >= 0 ? '#f59e0b' : '#ef4444'}
                        />
                      ))}
                    </Bar>
                    <ReferenceLine y={1} stroke="#888" strokeDasharray="3 3" label="Break-even" />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        {/* Recommendations Tab */}
        <TabsContent value="recommendations" className="space-y-6">
          <div className="flex items-center justify-between">
            <div>
              <h2 className="text-xl font-semibold">AI-Powered Recommendations</h2>
              <p className="text-sm text-muted-foreground">
                Actionable insights derived from predictive models
              </p>
            </div>
            <div className="flex items-center gap-2">
              <Badge variant="outline">
                {recommendations.filter(r => r.priority === 'high').length} High Priority
              </Badge>
              <Badge variant="outline">
                {recommendations.filter(r => r.actionable).length} Actionable
              </Badge>
            </div>
          </div>

          <div className="space-y-4">
            {recommendations.map((rec) => (
              <RecommendationCard key={rec.id} recommendation={rec} />
            ))}
          </div>

          {/* Action Summary */}
          <Card className="bg-gradient-to-r from-emerald-50 to-teal-50 dark:from-emerald-900/20 dark:to-teal-900/20">
            <CardContent className="p-6">
              <div className="flex items-center gap-4">
                <div className="p-3 rounded-full bg-emerald-100 dark:bg-emerald-900/30">
                  <CheckCircle2 className="h-8 w-8 text-emerald-600" />
                </div>
                <div className="flex-1">
                  <h3 className="font-semibold text-lg mb-1">Summary Impact</h3>
                  <p className="text-muted-foreground">
                    Implementing all high-priority recommendations could yield estimated
                    <span className="font-bold text-emerald-600"> +$4.8M </span>
                    in additional revenue with
                    <span className="font-bold"> 85% </span>
                    average confidence.
                  </p>
                </div>
                <Button>
                  <Sparkles className="h-4 w-4 mr-2" />
                  Generate Action Plan
                </Button>
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
}

export default PredictiveAnalytics;
