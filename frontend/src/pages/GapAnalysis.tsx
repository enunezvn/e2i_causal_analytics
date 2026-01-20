/**
 * Gap Analysis Page
 * =================
 *
 * Dashboard for ROI opportunity detection and performance gap analysis.
 * Integrates with the Tier 2 Gap Analyzer agent.
 *
 * @module pages/GapAnalysis
 */

import { useState, useMemo } from 'react';
import {
  Target,
  RefreshCw,
  Download,
  Search,
  DollarSign,
  Clock,
  Filter,
  BarChart3,
  ArrowUpRight,
  ArrowDownRight,
  Loader2,
  Play,
} from 'lucide-react';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  CartesianGrid,
  Cell,
  PieChart,
  Pie,
  Legend,
} from 'recharts';

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Input } from '@/components/ui/input';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import { KPICard, StatusBadge } from '@/components/visualizations';
import {
  useOpportunities,
  useGapHealth,
  useRunGapAnalysis,
} from '@/hooks/api';
import type {
  PrioritizedOpportunity,
  ImplementationDifficulty,
} from '@/types/gaps';

// =============================================================================
// CONSTANTS
// =============================================================================

const BRANDS = [
  { value: 'kisqali', label: 'Kisqali' },
  { value: 'fabhalta', label: 'Fabhalta' },
  { value: 'remibrutinib', label: 'Remibrutinib' },
];

const DIFFICULTY_COLORS: Record<string, string> = {
  low: '#10b981',
  medium: '#f59e0b',
  high: '#ef4444',
};

const DIFFICULTY_LABELS: Record<string, string> = {
  low: 'Quick Win',
  medium: 'Moderate',
  high: 'Strategic Bet',
};

// =============================================================================
// HELPER FUNCTIONS
// =============================================================================

function formatCurrency(value: number): string {
  if (value >= 1000000) return `$${(value / 1000000).toFixed(1)}M`;
  if (value >= 1000) return `$${(value / 1000).toFixed(1)}K`;
  return `$${value.toFixed(0)}`;
}

function getDifficultyBadge(difficulty: string) {
  const color = DIFFICULTY_COLORS[difficulty] || '#6b7280';
  const label = DIFFICULTY_LABELS[difficulty] || difficulty;

  return (
    <Badge
      style={{
        backgroundColor: `${color}20`,
        color: color,
        borderColor: color,
      }}
      variant="outline"
    >
      {label}
    </Badge>
  );
}

// =============================================================================
// SAMPLE DATA (fallback when API unavailable)
// =============================================================================

const SAMPLE_OPPORTUNITIES: PrioritizedOpportunity[] = [
  {
    rank: 1,
    gap: {
      gap_id: 'gap_001',
      metric: 'TRx',
      segment: 'region',
      segment_value: 'Northeast',
      current_value: 4200,
      target_value: 5500,
      gap_size: 1300,
      gap_percentage: 23.6,
      gap_type: 'vs_target',
    },
    roi_estimate: {
      gap_id: 'gap_001',
      estimated_revenue_impact: 2500000,
      estimated_cost_to_close: 350000,
      expected_roi: 7.1,
      risk_adjusted_roi: 5.2,
      payback_period_months: 4,
      attribution_level: 'high',
      attribution_rate: 0.85,
      confidence: 0.92,
    },
    recommended_action: 'Increase rep coverage in underperforming territories',
    implementation_difficulty: 'low' as ImplementationDifficulty,
    time_to_impact: '2-3 months',
  },
  {
    rank: 2,
    gap: {
      gap_id: 'gap_002',
      metric: 'Market Share',
      segment: 'specialty',
      segment_value: 'Oncology',
      current_value: 18.5,
      target_value: 25.0,
      gap_size: 6.5,
      gap_percentage: 26.0,
      gap_type: 'vs_benchmark',
    },
    roi_estimate: {
      gap_id: 'gap_002',
      estimated_revenue_impact: 4200000,
      estimated_cost_to_close: 850000,
      expected_roi: 4.9,
      risk_adjusted_roi: 3.8,
      payback_period_months: 6,
      attribution_level: 'medium',
      attribution_rate: 0.72,
      confidence: 0.85,
    },
    recommended_action: 'Launch targeted speaker program for key oncologists',
    implementation_difficulty: 'medium' as ImplementationDifficulty,
    time_to_impact: '4-6 months',
  },
  {
    rank: 3,
    gap: {
      gap_id: 'gap_003',
      metric: 'NRx',
      segment: 'account_type',
      segment_value: 'Academic Medical Centers',
      current_value: 890,
      target_value: 1400,
      gap_size: 510,
      gap_percentage: 36.4,
      gap_type: 'vs_potential',
    },
    roi_estimate: {
      gap_id: 'gap_003',
      estimated_revenue_impact: 1800000,
      estimated_cost_to_close: 420000,
      expected_roi: 4.3,
      risk_adjusted_roi: 3.2,
      payback_period_months: 5,
      attribution_level: 'medium',
      attribution_rate: 0.68,
      confidence: 0.78,
    },
    recommended_action: 'Establish medical liaison partnerships at top AMCs',
    implementation_difficulty: 'medium' as ImplementationDifficulty,
    time_to_impact: '3-5 months',
  },
  {
    rank: 4,
    gap: {
      gap_id: 'gap_004',
      metric: 'Conversion Rate',
      segment: 'channel',
      segment_value: 'Digital',
      current_value: 12.3,
      target_value: 18.0,
      gap_size: 5.7,
      gap_percentage: 31.7,
      gap_type: 'vs_benchmark',
    },
    roi_estimate: {
      gap_id: 'gap_004',
      estimated_revenue_impact: 3100000,
      estimated_cost_to_close: 1200000,
      expected_roi: 2.6,
      risk_adjusted_roi: 2.1,
      payback_period_months: 9,
      attribution_level: 'low',
      attribution_rate: 0.55,
      confidence: 0.72,
    },
    recommended_action: 'Implement AI-powered HCP engagement platform',
    implementation_difficulty: 'high' as ImplementationDifficulty,
    time_to_impact: '6-9 months',
  },
  {
    rank: 5,
    gap: {
      gap_id: 'gap_005',
      metric: 'TRx',
      segment: 'region',
      segment_value: 'West',
      current_value: 3800,
      target_value: 4800,
      gap_size: 1000,
      gap_percentage: 20.8,
      gap_type: 'vs_target',
    },
    roi_estimate: {
      gap_id: 'gap_005',
      estimated_revenue_impact: 1950000,
      estimated_cost_to_close: 280000,
      expected_roi: 7.0,
      risk_adjusted_roi: 5.4,
      payback_period_months: 3,
      attribution_level: 'high',
      attribution_rate: 0.88,
      confidence: 0.91,
    },
    recommended_action: 'Add 2 field reps to high-potential territories',
    implementation_difficulty: 'low' as ImplementationDifficulty,
    time_to_impact: '1-2 months',
  },
];

// =============================================================================
// COMPONENT
// =============================================================================

function GapAnalysis() {
  const [selectedBrand, setSelectedBrand] = useState<string>('kisqali');
  const [searchQuery, setSearchQuery] = useState('');
  const [difficultyFilter, setDifficultyFilter] = useState<string>('all');
  const [isRefreshing, setIsRefreshing] = useState(false);

  // API hooks
  const { data: opportunitiesData, isLoading: opportunitiesLoading, refetch: refetchOpportunities } = useOpportunities({
    brand: selectedBrand,
    limit: 50,
  });
  const { data: healthData, isLoading: _healthLoading } = useGapHealth();
  const runGapAnalysisMutation = useRunGapAnalysis();

  // Use sample data as fallback
  const opportunities = opportunitiesData?.opportunities || SAMPLE_OPPORTUNITIES;
  const totalAddressableValue = opportunitiesData?.total_addressable_value || 13550000;
  const quickWinsCount = opportunitiesData?.quick_wins_count || 2;
  const strategicBetsCount = opportunitiesData?.strategic_bets_count || 1;

  // Calculate metrics
  const metrics = useMemo(() => {
    const totalGaps = opportunities.length;
    const avgROI = opportunities.reduce((sum, opp) => sum + opp.roi_estimate.expected_roi, 0) / totalGaps || 0;
    const avgConfidence = opportunities.reduce((sum, opp) => sum + opp.roi_estimate.confidence, 0) / totalGaps || 0;

    return {
      totalGaps,
      avgROI: avgROI.toFixed(1),
      avgConfidence: (avgConfidence * 100).toFixed(0),
      quickWins: quickWinsCount,
      strategicBets: strategicBetsCount,
    };
  }, [opportunities, quickWinsCount, strategicBetsCount]);

  // Filter opportunities
  const filteredOpportunities = useMemo(() => {
    return opportunities.filter((opp) => {
      const matchesDifficulty = difficultyFilter === 'all' || opp.implementation_difficulty === difficultyFilter;
      const matchesSearch =
        searchQuery === '' ||
        opp.recommended_action.toLowerCase().includes(searchQuery.toLowerCase()) ||
        opp.gap.metric.toLowerCase().includes(searchQuery.toLowerCase()) ||
        opp.gap.segment_value.toLowerCase().includes(searchQuery.toLowerCase());
      return matchesDifficulty && matchesSearch;
    });
  }, [opportunities, difficultyFilter, searchQuery]);

  // Chart data
  const roiByDifficultyData = useMemo(() => {
    const grouped: Record<string, { total: number; count: number }> = {};
    opportunities.forEach((opp) => {
      const diff = opp.implementation_difficulty;
      if (!grouped[diff]) grouped[diff] = { total: 0, count: 0 };
      grouped[diff].total += opp.roi_estimate.expected_roi;
      grouped[diff].count += 1;
    });
    return Object.entries(grouped).map(([difficulty, { total, count }]) => ({
      difficulty: DIFFICULTY_LABELS[difficulty] || difficulty,
      avgROI: total / count,
      color: DIFFICULTY_COLORS[difficulty],
    }));
  }, [opportunities]);

  const gapsByMetricData = useMemo(() => {
    const grouped: Record<string, number> = {};
    opportunities.forEach((opp) => {
      const metric = opp.gap.metric;
      grouped[metric] = (grouped[metric] || 0) + opp.roi_estimate.estimated_revenue_impact;
    });
    return Object.entries(grouped).map(([metric, value]) => ({
      metric,
      value,
    }));
  }, [opportunities]);

  const handleRefresh = async () => {
    setIsRefreshing(true);
    await refetchOpportunities();
    setIsRefreshing(false);
  };

  const handleRunAnalysis = () => {
    runGapAnalysisMutation.mutate({
      request: {
        query: `Identify performance gaps for ${selectedBrand}`,
        brand: selectedBrand,
        metrics: ['trx', 'nrx', 'market_share'],
        segments: ['region', 'specialty', 'account_type'],
      },
      asyncMode: true,
    });
  };

  const handleExport = () => {
    const report = {
      generatedAt: new Date().toISOString(),
      brand: selectedBrand,
      totalOpportunities: opportunities.length,
      totalAddressableValue,
      opportunities,
    };
    const blob = new Blob([JSON.stringify(report, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `gap-analysis-${selectedBrand}-${new Date().toISOString().split('T')[0]}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="container mx-auto px-4 py-8">
      {/* Header */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 mb-8">
        <div>
          <h1 className="text-3xl font-bold mb-2 flex items-center gap-3">
            <Target className="h-8 w-8" />
            Gap Analysis
          </h1>
          <p className="text-muted-foreground">
            ROI opportunity detection and performance gap prioritization powered by the Tier 2 Gap Analyzer agent.
          </p>
        </div>
        <div className="flex items-center gap-2">
          <Select value={selectedBrand} onValueChange={setSelectedBrand}>
            <SelectTrigger className="w-40">
              <SelectValue placeholder="Brand" />
            </SelectTrigger>
            <SelectContent>
              {BRANDS.map((brand) => (
                <SelectItem key={brand.value} value={brand.value}>
                  {brand.label}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
          <Button
            variant="outline"
            onClick={handleRunAnalysis}
            disabled={runGapAnalysisMutation.isPending}
          >
            {runGapAnalysisMutation.isPending ? (
              <Loader2 className="h-4 w-4 mr-2 animate-spin" />
            ) : (
              <Play className="h-4 w-4 mr-2" />
            )}
            Run Analysis
          </Button>
          <Button variant="outline" onClick={handleRefresh} disabled={isRefreshing}>
            <RefreshCw className={`h-4 w-4 mr-2 ${isRefreshing ? 'animate-spin' : ''}`} />
            Refresh
          </Button>
          <Button variant="outline" onClick={handleExport}>
            <Download className="h-4 w-4 mr-2" />
            Export
          </Button>
        </div>
      </div>

      {/* Service Health Banner */}
      {healthData && (
        <div className="mb-6">
          <Card className={healthData.agent_available ? 'border-emerald-200' : 'border-amber-200'}>
            <CardContent className="py-3">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <StatusBadge
                    status={healthData.agent_available ? 'healthy' : 'warning'}
                    showIcon
                    pulse={!healthData.agent_available}
                  />
                  <span className="text-sm">
                    Gap Analyzer Agent {healthData.agent_available ? 'Available' : 'Unavailable'}
                  </span>
                </div>
                <div className="text-sm text-muted-foreground">
                  {healthData.analyses_24h} analyses in last 24h
                </div>
              </div>
            </CardContent>
          </Card>
        </div>
      )}

      {/* Overview Metrics */}
      <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-4 mb-8">
        <KPICard
          title="Total Addressable"
          value={formatCurrency(totalAddressableValue)}
          status="healthy"
          description="Total revenue opportunity"
          size="sm"
        />
        <KPICard
          title="Opportunities"
          value={metrics.totalGaps}
          status="healthy"
          description="Identified gaps"
          size="sm"
        />
        <KPICard
          title="Avg ROI"
          value={parseFloat(metrics.avgROI)}
          unit="x"
          status={parseFloat(metrics.avgROI) > 3 ? 'healthy' : 'warning'}
          description="Expected return"
          size="sm"
        />
        <KPICard
          title="Quick Wins"
          value={metrics.quickWins}
          status="healthy"
          description="Low effort, high ROI"
          size="sm"
        />
        <KPICard
          title="Strategic Bets"
          value={metrics.strategicBets}
          status="warning"
          description="High effort, high impact"
          size="sm"
        />
        <KPICard
          title="Confidence"
          value={parseInt(metrics.avgConfidence)}
          unit="%"
          status={parseInt(metrics.avgConfidence) > 80 ? 'healthy' : 'warning'}
          description="Estimate confidence"
          size="sm"
        />
      </div>

      {/* Tabs */}
      <Tabs defaultValue="opportunities" className="space-y-4">
        <TabsList>
          <TabsTrigger value="opportunities" className="flex items-center gap-2">
            <Target className="h-4 w-4" />
            Opportunities
          </TabsTrigger>
          <TabsTrigger value="charts" className="flex items-center gap-2">
            <BarChart3 className="h-4 w-4" />
            Analysis
          </TabsTrigger>
        </TabsList>

        {/* Opportunities Tab */}
        <TabsContent value="opportunities" className="space-y-4">
          {/* Filters */}
          <div className="flex flex-col md:flex-row gap-4">
            <div className="relative flex-1">
              <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-muted-foreground" />
              <Input
                placeholder="Search opportunities..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="pl-9"
              />
            </div>
            <Select value={difficultyFilter} onValueChange={setDifficultyFilter}>
              <SelectTrigger className="w-48">
                <Filter className="h-4 w-4 mr-2" />
                <SelectValue placeholder="Difficulty" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="all">All Difficulties</SelectItem>
                <SelectItem value="low">Quick Wins</SelectItem>
                <SelectItem value="medium">Moderate</SelectItem>
                <SelectItem value="high">Strategic Bets</SelectItem>
              </SelectContent>
            </Select>
          </div>

          {/* Opportunity Cards */}
          {opportunitiesLoading ? (
            <div className="flex items-center justify-center py-12">
              <Loader2 className="h-8 w-8 animate-spin text-muted-foreground" />
            </div>
          ) : (
            <div className="space-y-4">
              {filteredOpportunities.map((opp) => (
                <Card key={opp.gap.gap_id} className="hover:shadow-md transition-shadow">
                  <CardContent className="py-4">
                    <div className="flex flex-col lg:flex-row lg:items-center gap-4">
                      {/* Rank */}
                      <div className="flex-shrink-0 w-12 h-12 bg-primary/10 rounded-full flex items-center justify-center">
                        <span className="text-lg font-bold text-primary">#{opp.rank}</span>
                      </div>

                      {/* Main Content */}
                      <div className="flex-1">
                        <div className="flex items-center gap-2 mb-1">
                          <h3 className="font-semibold">{opp.recommended_action}</h3>
                          {getDifficultyBadge(opp.implementation_difficulty)}
                        </div>
                        <div className="flex flex-wrap items-center gap-4 text-sm text-muted-foreground">
                          <span className="flex items-center gap-1">
                            <Badge variant="outline">{opp.gap.metric}</Badge>
                          </span>
                          <span>
                            {opp.gap.segment}: {opp.gap.segment_value}
                          </span>
                          <span className="flex items-center gap-1">
                            {opp.gap.gap_percentage > 0 ? (
                              <ArrowDownRight className="h-3 w-3 text-rose-500" />
                            ) : (
                              <ArrowUpRight className="h-3 w-3 text-emerald-500" />
                            )}
                            {opp.gap.gap_percentage.toFixed(1)}% gap
                          </span>
                          <span className="flex items-center gap-1">
                            <Clock className="h-3 w-3" />
                            {opp.time_to_impact}
                          </span>
                        </div>
                      </div>

                      {/* ROI Metrics */}
                      <div className="flex items-center gap-6 lg:gap-8">
                        <div className="text-center">
                          <p className="text-sm text-muted-foreground">Revenue Impact</p>
                          <p className="text-lg font-bold text-emerald-600">
                            {formatCurrency(opp.roi_estimate.estimated_revenue_impact)}
                          </p>
                        </div>
                        <div className="text-center">
                          <p className="text-sm text-muted-foreground">Investment</p>
                          <p className="text-lg font-bold text-amber-600">
                            {formatCurrency(opp.roi_estimate.estimated_cost_to_close)}
                          </p>
                        </div>
                        <div className="text-center">
                          <p className="text-sm text-muted-foreground">ROI</p>
                          <p className="text-lg font-bold text-primary">
                            {opp.roi_estimate.expected_roi.toFixed(1)}x
                          </p>
                        </div>
                        <div className="text-center">
                          <p className="text-sm text-muted-foreground">Confidence</p>
                          <p className="text-lg font-bold">
                            {(opp.roi_estimate.confidence * 100).toFixed(0)}%
                          </p>
                        </div>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              ))}
            </div>
          )}
        </TabsContent>

        {/* Charts Tab */}
        <TabsContent value="charts" className="space-y-4">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            {/* ROI by Difficulty */}
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <BarChart3 className="h-5 w-5" />
                  Average ROI by Difficulty
                </CardTitle>
                <CardDescription>Expected returns by implementation effort</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="h-64">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={roiByDifficultyData} layout="vertical">
                      <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
                      <XAxis type="number" stroke="var(--muted-foreground)" fontSize={12} />
                      <YAxis dataKey="difficulty" type="category" stroke="var(--muted-foreground)" fontSize={12} width={80} />
                      <Tooltip
                        contentStyle={{
                          backgroundColor: 'var(--card)',
                          border: '1px solid var(--border)',
                          borderRadius: '8px',
                        }}
                        formatter={(value) => [`${(value as number)?.toFixed(1) ?? 0}x`, 'Avg ROI']}
                      />
                      <Bar dataKey="avgROI" radius={[0, 4, 4, 0]}>
                        {roiByDifficultyData.map((entry, index) => (
                          <Cell key={index} fill={entry.color} />
                        ))}
                      </Bar>
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </CardContent>
            </Card>

            {/* Value by Metric */}
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <DollarSign className="h-5 w-5" />
                  Revenue Opportunity by Metric
                </CardTitle>
                <CardDescription>Total addressable value by KPI</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="h-64">
                  <ResponsiveContainer width="100%" height="100%">
                    <PieChart>
                      <Pie
                        data={gapsByMetricData}
                        dataKey="value"
                        nameKey="metric"
                        cx="50%"
                        cy="50%"
                        outerRadius={80}
                        label={({ name, value }) => `${name}: ${formatCurrency(value as number)}`}
                      >
                        {gapsByMetricData.map((_, index) => (
                          <Cell
                            key={index}
                            fill={['#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6'][index % 5]}
                          />
                        ))}
                      </Pie>
                      <Tooltip
                        contentStyle={{
                          backgroundColor: 'var(--card)',
                          border: '1px solid var(--border)',
                          borderRadius: '8px',
                        }}
                        formatter={(value) => [formatCurrency(value as number), 'Value']}
                      />
                      <Legend />
                    </PieChart>
                  </ResponsiveContainer>
                </div>
              </CardContent>
            </Card>
          </div>

          {/* Top Opportunities Table */}
          <Card>
            <CardHeader>
              <CardTitle>Top Opportunities Summary</CardTitle>
              <CardDescription>Highest ROI opportunities across all metrics</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="overflow-x-auto">
                <table className="w-full">
                  <thead>
                    <tr className="border-b border-border">
                      <th className="text-left py-3 px-4 font-medium text-muted-foreground">Rank</th>
                      <th className="text-left py-3 px-4 font-medium text-muted-foreground">Metric</th>
                      <th className="text-left py-3 px-4 font-medium text-muted-foreground">Segment</th>
                      <th className="text-right py-3 px-4 font-medium text-muted-foreground">Gap</th>
                      <th className="text-right py-3 px-4 font-medium text-muted-foreground">Revenue</th>
                      <th className="text-right py-3 px-4 font-medium text-muted-foreground">Cost</th>
                      <th className="text-right py-3 px-4 font-medium text-muted-foreground">ROI</th>
                      <th className="text-center py-3 px-4 font-medium text-muted-foreground">Difficulty</th>
                    </tr>
                  </thead>
                  <tbody>
                    {opportunities.slice(0, 10).map((opp) => (
                      <tr key={opp.gap.gap_id} className="border-b border-border hover:bg-muted/50">
                        <td className="py-3 px-4 font-bold text-primary">#{opp.rank}</td>
                        <td className="py-3 px-4">
                          <Badge variant="outline">{opp.gap.metric}</Badge>
                        </td>
                        <td className="py-3 px-4 text-sm">{opp.gap.segment_value}</td>
                        <td className="py-3 px-4 text-right text-rose-500">
                          {opp.gap.gap_percentage.toFixed(1)}%
                        </td>
                        <td className="py-3 px-4 text-right font-medium text-emerald-600">
                          {formatCurrency(opp.roi_estimate.estimated_revenue_impact)}
                        </td>
                        <td className="py-3 px-4 text-right text-amber-600">
                          {formatCurrency(opp.roi_estimate.estimated_cost_to_close)}
                        </td>
                        <td className="py-3 px-4 text-right font-bold">
                          {opp.roi_estimate.expected_roi.toFixed(1)}x
                        </td>
                        <td className="py-3 px-4 text-center">
                          {getDifficultyBadge(opp.implementation_difficulty)}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
}

export default GapAnalysis;
