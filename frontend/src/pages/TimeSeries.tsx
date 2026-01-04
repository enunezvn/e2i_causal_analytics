/**
 * Time Series Analysis Page
 * =========================
 *
 * Comprehensive time series analytics dashboard with:
 * - Multi-metric trend visualization
 * - Seasonality decomposition
 * - Forecasting with confidence intervals
 * - Anomaly detection and highlighting
 *
 * @module pages/TimeSeries
 */

import { useState, useMemo } from 'react';
import {
  ComposedChart,
  LineChart,
  AreaChart,
  BarChart,
  Line,
  Area,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  ReferenceLine,
  Scatter,
} from 'recharts';
import {
  Calendar,
  RefreshCw,
  Download,
  AlertTriangle,
  Activity,
  BarChart3,
  Waves,
  Clock,
} from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Badge } from '@/components/ui/badge';
import { cn } from '@/lib/utils';
import { KPICard, StatusBadge } from '@/components/visualizations';

// =============================================================================
// TYPES
// =============================================================================

interface TimeSeriesDataPoint {
  date: string;
  value: number;
  forecast?: number;
  upperBound?: number;
  lowerBound?: number;
  isAnomaly?: boolean;
  anomalyScore?: number;
}

interface SeasonalityComponent {
  date: string;
  trend: number;
  seasonal: number;
  residual: number;
  original: number;
}

interface MetricConfig {
  id: string;
  name: string;
  color: string;
  unit: string;
}

interface ForecastMetrics {
  mape: number;
  rmse: number;
  mae: number;
  r2: number;
}

interface AnomalyEvent {
  id: string;
  date: string;
  metric: string;
  value: number;
  expected: number;
  deviation: number;
  severity: 'low' | 'medium' | 'high' | 'critical';
  description: string;
}

// =============================================================================
// SAMPLE DATA
// =============================================================================

const METRICS: MetricConfig[] = [
  { id: 'trx', name: 'TRx Volume', color: 'hsl(var(--chart-1))', unit: 'scripts' },
  { id: 'nrx', name: 'NRx Volume', color: 'hsl(var(--chart-2))', unit: 'scripts' },
  { id: 'marketShare', name: 'Market Share', color: 'hsl(var(--chart-3))', unit: '%' },
  { id: 'revenue', name: 'Revenue', color: 'hsl(var(--chart-4))', unit: '$K' },
];

// Generate time series data with trend, seasonality, and anomalies
const generateTimeSeriesData = (): TimeSeriesDataPoint[] => {
  const data: TimeSeriesDataPoint[] = [];
  const startDate = new Date('2024-01-01');

  for (let i = 0; i < 90; i++) {
    const date = new Date(startDate);
    date.setDate(startDate.getDate() + i);

    // Base trend (gradual increase)
    const trend = 1000 + i * 2;

    // Weekly seasonality (lower on weekends)
    const dayOfWeek = date.getDay();
    const seasonal = dayOfWeek === 0 || dayOfWeek === 6 ? -50 : 20;

    // Random noise
    const noise = (Math.random() - 0.5) * 100;

    // Combine components
    let value = trend + seasonal + noise;

    // Add some anomalies
    let isAnomaly = false;
    let anomalyScore = 0;
    if (i === 25 || i === 52 || i === 78) {
      value = value * 1.3; // Spike
      isAnomaly = true;
      anomalyScore = 0.85;
    }
    if (i === 40) {
      value = value * 0.6; // Drop
      isAnomaly = true;
      anomalyScore = 0.92;
    }

    data.push({
      date: date.toISOString().split('T')[0],
      value: Math.round(value),
      isAnomaly,
      anomalyScore,
    });
  }

  return data;
};

// Generate forecast data (extends from actual data)
const generateForecastData = (historicalData: TimeSeriesDataPoint[]): TimeSeriesDataPoint[] => {
  const lastDate = new Date(historicalData[historicalData.length - 1].date);
  const lastValue = historicalData[historicalData.length - 1].value;
  const forecastData: TimeSeriesDataPoint[] = [];

  for (let i = 1; i <= 30; i++) {
    const date = new Date(lastDate);
    date.setDate(lastDate.getDate() + i);

    // Simple forecast with trend
    const forecast = lastValue + i * 2.5 + (Math.random() - 0.5) * 20;
    const uncertainty = 30 + i * 2;

    forecastData.push({
      date: date.toISOString().split('T')[0],
      value: 0, // No actual value
      forecast: Math.round(forecast),
      upperBound: Math.round(forecast + uncertainty),
      lowerBound: Math.round(forecast - uncertainty),
    });
  }

  return forecastData;
};

// Generate seasonality decomposition data
const generateSeasonalityData = (): SeasonalityComponent[] => {
  const data: SeasonalityComponent[] = [];
  const startDate = new Date('2024-01-01');

  for (let i = 0; i < 90; i++) {
    const date = new Date(startDate);
    date.setDate(startDate.getDate() + i);

    const trend = 1000 + i * 2;
    const dayOfWeek = date.getDay();
    const seasonal = Math.sin((dayOfWeek / 7) * Math.PI * 2) * 50;
    const residual = (Math.random() - 0.5) * 40;
    const original = trend + seasonal + residual;

    data.push({
      date: date.toISOString().split('T')[0],
      trend,
      seasonal,
      residual,
      original,
    });
  }

  return data;
};

const SAMPLE_TIME_SERIES = generateTimeSeriesData();
const SAMPLE_FORECAST = generateForecastData(SAMPLE_TIME_SERIES);
const SAMPLE_SEASONALITY = generateSeasonalityData();

const SAMPLE_FORECAST_METRICS: ForecastMetrics = {
  mape: 3.2,
  rmse: 45.6,
  mae: 38.2,
  r2: 0.94,
};

const SAMPLE_ANOMALIES: AnomalyEvent[] = [
  {
    id: 'a1',
    date: '2024-01-26',
    metric: 'TRx Volume',
    value: 1452,
    expected: 1098,
    deviation: 32.2,
    severity: 'high',
    description: 'Unusual spike in TRx volume, potentially due to promotional campaign',
  },
  {
    id: 'a2',
    date: '2024-02-10',
    metric: 'TRx Volume',
    value: 678,
    expected: 1120,
    deviation: -39.5,
    severity: 'critical',
    description: 'Sharp decline in volume, investigate supply chain or competitor activity',
  },
  {
    id: 'a3',
    date: '2024-02-22',
    metric: 'TRx Volume',
    value: 1389,
    expected: 1145,
    deviation: 21.3,
    severity: 'medium',
    description: 'Moderate increase, correlates with HCP engagement event',
  },
  {
    id: 'a4',
    date: '2024-03-19',
    metric: 'TRx Volume',
    value: 1498,
    expected: 1180,
    deviation: 26.9,
    severity: 'medium',
    description: 'Above-average volume, seasonal pattern detected',
  },
];

// =============================================================================
// HELPER FUNCTIONS
// =============================================================================

function formatDate(dateStr: string): string {
  const date = new Date(dateStr);
  return date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
}

function getSeverityColor(severity: AnomalyEvent['severity']): string {
  switch (severity) {
    case 'critical':
      return 'bg-red-100 text-red-700 dark:bg-red-900/30 dark:text-red-400';
    case 'high':
      return 'bg-orange-100 text-orange-700 dark:bg-orange-900/30 dark:text-orange-400';
    case 'medium':
      return 'bg-amber-100 text-amber-700 dark:bg-amber-900/30 dark:text-amber-400';
    case 'low':
      return 'bg-blue-100 text-blue-700 dark:bg-blue-900/30 dark:text-blue-400';
  }
}

// =============================================================================
// COMPONENT
// =============================================================================

function TimeSeries() {
  const [selectedMetric, setSelectedMetric] = useState<string>('trx');
  const [timeRange, setTimeRange] = useState<string>('90d');
  const [forecastHorizon, setForecastHorizon] = useState<string>('30d');
  const [isRefreshing, setIsRefreshing] = useState(false);
  const [showConfidenceInterval, setShowConfidenceInterval] = useState(true);

  // Combine historical and forecast data for visualization
  const combinedData = useMemo(() => {
    const historical = SAMPLE_TIME_SERIES.map((d) => ({
      ...d,
      type: 'historical' as const,
    }));
    const forecast = SAMPLE_FORECAST.map((d) => ({
      ...d,
      type: 'forecast' as const,
    }));
    return [...historical, ...forecast];
  }, []);

  // Calculate summary metrics
  const summaryMetrics = useMemo(() => {
    const values = SAMPLE_TIME_SERIES.map((d) => d.value);
    const lastValue = values[values.length - 1];
    const prevValue = values[values.length - 8]; // 7 days ago
    const change = ((lastValue - prevValue) / prevValue) * 100;
    const avg = values.reduce((a, b) => a + b, 0) / values.length;
    const max = Math.max(...values);
    const min = Math.min(...values);
    const anomalyCount = SAMPLE_ANOMALIES.length;

    return {
      current: lastValue,
      change,
      average: avg,
      max,
      min,
      anomalyCount,
      forecastAccuracy: SAMPLE_FORECAST_METRICS.mape,
    };
  }, []);

  const handleRefresh = () => {
    setIsRefreshing(true);
    setTimeout(() => setIsRefreshing(false), 1000);
  };

  const handleExport = () => {
    const exportData = {
      historical: SAMPLE_TIME_SERIES,
      forecast: SAMPLE_FORECAST,
      seasonality: SAMPLE_SEASONALITY,
      anomalies: SAMPLE_ANOMALIES,
      metrics: SAMPLE_FORECAST_METRICS,
    };
    const blob = new Blob([JSON.stringify(exportData, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'time-series-analysis.json';
    a.click();
    URL.revokeObjectURL(url);
  };

  // Custom tooltip for combined chart
  const CombinedTooltip = ({ active, payload, label }: { active?: boolean; payload?: Array<Record<string, unknown>>; label?: string }) => {
    if (!active || !payload || !payload.length) return null;

    const dataPoint = payload[0].payload as TimeSeriesDataPoint & { type: string };

    return (
      <div className="bg-[var(--color-popover)] border border-[var(--color-border)] rounded-md shadow-lg p-3">
        <p className="font-medium text-[var(--color-foreground)] mb-2">
          {formatDate(label || '')}
        </p>
        {dataPoint.type === 'historical' ? (
          <>
            <div className="flex items-center gap-2 text-sm">
              <div className="w-3 h-3 rounded-full bg-blue-500" />
              <span className="text-[var(--color-muted-foreground)]">Actual:</span>
              <span className="font-medium">{dataPoint.value.toLocaleString()}</span>
            </div>
            {dataPoint.isAnomaly && (
              <div className="flex items-center gap-1 mt-1 text-xs text-amber-600">
                <AlertTriangle className="h-3 w-3" />
                <span>Anomaly detected (score: {dataPoint.anomalyScore?.toFixed(2)})</span>
              </div>
            )}
          </>
        ) : (
          <>
            <div className="flex items-center gap-2 text-sm">
              <div className="w-3 h-3 rounded-full bg-violet-500" />
              <span className="text-[var(--color-muted-foreground)]">Forecast:</span>
              <span className="font-medium">{dataPoint.forecast?.toLocaleString()}</span>
            </div>
            {showConfidenceInterval && (
              <div className="text-xs text-[var(--color-muted-foreground)] mt-1">
                95% CI: {dataPoint.lowerBound?.toLocaleString()} - {dataPoint.upperBound?.toLocaleString()}
              </div>
            )}
          </>
        )}
      </div>
    );
  };

  return (
    <div className="container mx-auto px-4 py-8">
      {/* Header */}
      <div className="flex items-center justify-between mb-8">
        <div>
          <h1 className="text-3xl font-bold mb-2">Time Series Analysis</h1>
          <p className="text-muted-foreground">
            Time series trends, forecasting, seasonality decomposition, and anomaly detection.
          </p>
        </div>
        <div className="flex items-center gap-3">
          <Select value={selectedMetric} onValueChange={setSelectedMetric}>
            <SelectTrigger className="w-[180px]">
              <SelectValue placeholder="Select metric" />
            </SelectTrigger>
            <SelectContent>
              {METRICS.map((metric) => (
                <SelectItem key={metric.id} value={metric.id}>
                  {metric.name}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
          <Select value={timeRange} onValueChange={setTimeRange}>
            <SelectTrigger className="w-[120px]">
              <SelectValue placeholder="Time range" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="30d">30 Days</SelectItem>
              <SelectItem value="60d">60 Days</SelectItem>
              <SelectItem value="90d">90 Days</SelectItem>
              <SelectItem value="180d">6 Months</SelectItem>
              <SelectItem value="365d">1 Year</SelectItem>
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

      {/* KPI Summary */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 xl:grid-cols-7 gap-4 mb-8">
        <KPICard
          title="Current Value"
          value={summaryMetrics.current.toLocaleString()}
          description="vs 7 days ago"
        />
        <KPICard
          title="Average"
          value={Math.round(summaryMetrics.average).toLocaleString()}
          description="Over period"
        />
        <KPICard
          title="Maximum"
          value={summaryMetrics.max.toLocaleString()}
          status="healthy"
          description="Peak value"
        />
        <KPICard
          title="Minimum"
          value={summaryMetrics.min.toLocaleString()}
          description="Lowest value"
        />
        <KPICard
          title="Anomalies"
          value={summaryMetrics.anomalyCount}
          status={summaryMetrics.anomalyCount > 3 ? 'warning' : 'healthy'}
          description="Detected"
        />
        <KPICard
          title="Forecast MAPE"
          value={`${summaryMetrics.forecastAccuracy.toFixed(1)}%`}
          status={summaryMetrics.forecastAccuracy < 5 ? 'healthy' : 'warning'}
          description="Accuracy"
        />
        <KPICard
          title="Forecast R²"
          value={SAMPLE_FORECAST_METRICS.r2.toFixed(2)}
          status={SAMPLE_FORECAST_METRICS.r2 > 0.9 ? 'healthy' : 'warning'}
          description="Model fit"
        />
      </div>

      {/* Main Content Tabs */}
      <Tabs defaultValue="trends" className="space-y-6">
        <TabsList>
          <TabsTrigger value="trends" className="gap-2">
            <Activity className="h-4 w-4" />
            Trends & Forecast
          </TabsTrigger>
          <TabsTrigger value="seasonality" className="gap-2">
            <Waves className="h-4 w-4" />
            Seasonality
          </TabsTrigger>
          <TabsTrigger value="anomalies" className="gap-2">
            <AlertTriangle className="h-4 w-4" />
            Anomalies
          </TabsTrigger>
          <TabsTrigger value="comparison" className="gap-2">
            <BarChart3 className="h-4 w-4" />
            Period Comparison
          </TabsTrigger>
        </TabsList>

        {/* Trends & Forecast Tab */}
        <TabsContent value="trends" className="space-y-6">
          <Card>
            <CardHeader>
              <div className="flex items-center justify-between">
                <div>
                  <CardTitle>Time Series with Forecast</CardTitle>
                  <CardDescription>
                    Historical data with {forecastHorizon} forecast and confidence intervals
                  </CardDescription>
                </div>
                <div className="flex items-center gap-3">
                  <Select value={forecastHorizon} onValueChange={setForecastHorizon}>
                    <SelectTrigger className="w-[140px]">
                      <SelectValue placeholder="Forecast" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="7d">7 Days</SelectItem>
                      <SelectItem value="14d">14 Days</SelectItem>
                      <SelectItem value="30d">30 Days</SelectItem>
                      <SelectItem value="60d">60 Days</SelectItem>
                    </SelectContent>
                  </Select>
                  <Button
                    variant={showConfidenceInterval ? 'default' : 'outline'}
                    size="sm"
                    onClick={() => setShowConfidenceInterval(!showConfidenceInterval)}
                  >
                    95% CI
                  </Button>
                </div>
              </div>
            </CardHeader>
            <CardContent>
              <ResponsiveContainer width="100%" height={400}>
                <ComposedChart data={combinedData} margin={{ top: 20, right: 30, left: 20, bottom: 5 }}>
                  <CartesianGrid strokeDasharray="3 3" opacity={0.3} />
                  <XAxis
                    dataKey="date"
                    tickFormatter={formatDate}
                    fontSize={12}
                    tickLine={false}
                  />
                  <YAxis fontSize={12} tickLine={false} axisLine={false} />
                  <Tooltip content={<CombinedTooltip />} />
                  <Legend />

                  {/* Confidence interval area */}
                  {showConfidenceInterval && (
                    <Area
                      type="monotone"
                      dataKey="upperBound"
                      stackId="ci"
                      stroke="none"
                      fill="hsl(var(--chart-3))"
                      fillOpacity={0.2}
                      name="Upper Bound"
                    />
                  )}
                  {showConfidenceInterval && (
                    <Area
                      type="monotone"
                      dataKey="lowerBound"
                      stackId="ci"
                      stroke="none"
                      fill="white"
                      fillOpacity={1}
                      name="Lower Bound"
                    />
                  )}

                  {/* Historical line */}
                  <Line
                    type="monotone"
                    dataKey="value"
                    stroke="hsl(var(--chart-1))"
                    strokeWidth={2}
                    dot={false}
                    name="Actual"
                    connectNulls={false}
                  />

                  {/* Forecast line */}
                  <Line
                    type="monotone"
                    dataKey="forecast"
                    stroke="hsl(var(--chart-3))"
                    strokeWidth={2}
                    strokeDasharray="5 5"
                    dot={false}
                    name="Forecast"
                  />

                  {/* Anomaly markers */}
                  <Scatter
                    dataKey="value"
                    fill="hsl(var(--destructive))"
                    shape={(props: unknown) => {
                      const { cx, cy, payload } = props as { cx: number; cy: number; payload: TimeSeriesDataPoint };
                      if (payload?.isAnomaly) {
                        return (
                          <circle
                            cx={cx}
                            cy={cy}
                            r={6}
                            fill="hsl(var(--destructive))"
                            stroke="white"
                            strokeWidth={2}
                          />
                        );
                      }
                      return <circle cx={0} cy={0} r={0} />;
                    }}
                    name="Anomalies"
                  />

                  {/* Reference line at forecast start */}
                  <ReferenceLine
                    x={SAMPLE_TIME_SERIES[SAMPLE_TIME_SERIES.length - 1].date}
                    stroke="var(--border)"
                    strokeDasharray="3 3"
                    label={{ value: 'Forecast Start', position: 'top', fontSize: 10 }}
                  />
                </ComposedChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>

          {/* Forecast Metrics */}
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-sm font-medium text-muted-foreground">
                  MAPE
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="flex items-end gap-2">
                  <span className="text-2xl font-bold">{SAMPLE_FORECAST_METRICS.mape.toFixed(2)}%</span>
                  <StatusBadge status="success" size="sm" label="Excellent" />
                </div>
                <p className="text-xs text-muted-foreground mt-1">Mean Absolute Percentage Error</p>
              </CardContent>
            </Card>
            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-sm font-medium text-muted-foreground">
                  RMSE
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="flex items-end gap-2">
                  <span className="text-2xl font-bold">{SAMPLE_FORECAST_METRICS.rmse.toFixed(2)}</span>
                  <StatusBadge status="success" size="sm" label="Good" />
                </div>
                <p className="text-xs text-muted-foreground mt-1">Root Mean Square Error</p>
              </CardContent>
            </Card>
            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-sm font-medium text-muted-foreground">
                  MAE
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="flex items-end gap-2">
                  <span className="text-2xl font-bold">{SAMPLE_FORECAST_METRICS.mae.toFixed(2)}</span>
                  <StatusBadge status="success" size="sm" label="Good" />
                </div>
                <p className="text-xs text-muted-foreground mt-1">Mean Absolute Error</p>
              </CardContent>
            </Card>
            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-sm font-medium text-muted-foreground">
                  R² Score
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="flex items-end gap-2">
                  <span className="text-2xl font-bold">{SAMPLE_FORECAST_METRICS.r2.toFixed(3)}</span>
                  <StatusBadge status="success" size="sm" label="Excellent" />
                </div>
                <p className="text-xs text-muted-foreground mt-1">Coefficient of Determination</p>
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        {/* Seasonality Tab */}
        <TabsContent value="seasonality" className="space-y-6">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Trend Component */}
            <Card>
              <CardHeader>
                <CardTitle>Trend Component</CardTitle>
                <CardDescription>Long-term direction of the time series</CardDescription>
              </CardHeader>
              <CardContent>
                <ResponsiveContainer width="100%" height={200}>
                  <LineChart data={SAMPLE_SEASONALITY}>
                    <CartesianGrid strokeDasharray="3 3" opacity={0.3} />
                    <XAxis dataKey="date" tickFormatter={formatDate} fontSize={10} tickLine={false} />
                    <YAxis fontSize={10} tickLine={false} axisLine={false} />
                    <Tooltip
                      formatter={(value) => value !== undefined ? Number(value).toFixed(0) : '-'}
                      labelFormatter={formatDate}
                    />
                    <Line
                      type="monotone"
                      dataKey="trend"
                      stroke="hsl(var(--chart-1))"
                      strokeWidth={2}
                      dot={false}
                      name="Trend"
                    />
                  </LineChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>

            {/* Seasonal Component */}
            <Card>
              <CardHeader>
                <CardTitle>Seasonal Component</CardTitle>
                <CardDescription>Repeating patterns in the data</CardDescription>
              </CardHeader>
              <CardContent>
                <ResponsiveContainer width="100%" height={200}>
                  <AreaChart data={SAMPLE_SEASONALITY}>
                    <CartesianGrid strokeDasharray="3 3" opacity={0.3} />
                    <XAxis dataKey="date" tickFormatter={formatDate} fontSize={10} tickLine={false} />
                    <YAxis fontSize={10} tickLine={false} axisLine={false} />
                    <Tooltip
                      formatter={(value) => value !== undefined ? Number(value).toFixed(2) : '-'}
                      labelFormatter={formatDate}
                    />
                    <Area
                      type="monotone"
                      dataKey="seasonal"
                      stroke="hsl(var(--chart-2))"
                      fill="hsl(var(--chart-2))"
                      fillOpacity={0.3}
                      name="Seasonal"
                    />
                    <ReferenceLine y={0} stroke="var(--border)" />
                  </AreaChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>

            {/* Residual Component */}
            <Card>
              <CardHeader>
                <CardTitle>Residual Component</CardTitle>
                <CardDescription>Random variation not explained by trend or seasonality</CardDescription>
              </CardHeader>
              <CardContent>
                <ResponsiveContainer width="100%" height={200}>
                  <BarChart data={SAMPLE_SEASONALITY}>
                    <CartesianGrid strokeDasharray="3 3" opacity={0.3} />
                    <XAxis dataKey="date" tickFormatter={formatDate} fontSize={10} tickLine={false} />
                    <YAxis fontSize={10} tickLine={false} axisLine={false} />
                    <Tooltip
                      formatter={(value) => value !== undefined ? Number(value).toFixed(2) : '-'}
                      labelFormatter={formatDate}
                    />
                    <Bar
                      dataKey="residual"
                      fill="hsl(var(--chart-4))"
                      name="Residual"
                    />
                    <ReferenceLine y={0} stroke="var(--border)" />
                  </BarChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>

            {/* Original vs Reconstructed */}
            <Card>
              <CardHeader>
                <CardTitle>Original vs Components</CardTitle>
                <CardDescription>Compare original data with decomposed components</CardDescription>
              </CardHeader>
              <CardContent>
                <ResponsiveContainer width="100%" height={200}>
                  <LineChart data={SAMPLE_SEASONALITY}>
                    <CartesianGrid strokeDasharray="3 3" opacity={0.3} />
                    <XAxis dataKey="date" tickFormatter={formatDate} fontSize={10} tickLine={false} />
                    <YAxis fontSize={10} tickLine={false} axisLine={false} />
                    <Tooltip
                      formatter={(value) => value !== undefined ? Number(value).toFixed(0) : '-'}
                      labelFormatter={formatDate}
                    />
                    <Legend />
                    <Line
                      type="monotone"
                      dataKey="original"
                      stroke="hsl(var(--chart-1))"
                      strokeWidth={2}
                      dot={false}
                      name="Original"
                    />
                    <Line
                      type="monotone"
                      dataKey="trend"
                      stroke="hsl(var(--chart-3))"
                      strokeWidth={1}
                      strokeDasharray="5 5"
                      dot={false}
                      name="Trend"
                    />
                  </LineChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>
          </div>

          {/* Seasonality Summary */}
          <Card>
            <CardHeader>
              <CardTitle>Seasonality Summary</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                <div className="space-y-2">
                  <h4 className="font-medium flex items-center gap-2">
                    <Calendar className="h-4 w-4" />
                    Weekly Pattern
                  </h4>
                  <p className="text-sm text-muted-foreground">
                    Strong weekly seasonality detected with peaks on Tuesdays and Wednesdays,
                    and lower values on weekends (-15% average).
                  </p>
                </div>
                <div className="space-y-2">
                  <h4 className="font-medium flex items-center gap-2">
                    <Clock className="h-4 w-4" />
                    Trend Direction
                  </h4>
                  <p className="text-sm text-muted-foreground">
                    Consistent upward trend of approximately +2 units per day,
                    indicating steady growth over the analysis period.
                  </p>
                </div>
                <div className="space-y-2">
                  <h4 className="font-medium flex items-center gap-2">
                    <Activity className="h-4 w-4" />
                    Residual Analysis
                  </h4>
                  <p className="text-sm text-muted-foreground">
                    Residuals are approximately normally distributed with mean near zero,
                    indicating a good decomposition fit.
                  </p>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        {/* Anomalies Tab */}
        <TabsContent value="anomalies" className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle>Anomaly Detection</CardTitle>
              <CardDescription>
                Identified unusual patterns in the time series data
              </CardDescription>
            </CardHeader>
            <CardContent>
              <ResponsiveContainer width="100%" height={300}>
                <ComposedChart data={SAMPLE_TIME_SERIES}>
                  <CartesianGrid strokeDasharray="3 3" opacity={0.3} />
                  <XAxis dataKey="date" tickFormatter={formatDate} fontSize={12} tickLine={false} />
                  <YAxis fontSize={12} tickLine={false} axisLine={false} />
                  <Tooltip
                    formatter={(value) => value !== undefined ? Number(value).toLocaleString() : '-'}
                    labelFormatter={formatDate}
                  />
                  <Line
                    type="monotone"
                    dataKey="value"
                    stroke="hsl(var(--chart-1))"
                    strokeWidth={2}
                    dot={false}
                    name="Value"
                  />
                  <Scatter
                    dataKey="value"
                    fill="hsl(var(--destructive))"
                    shape={(props: unknown) => {
                      const { cx, cy, payload } = props as { cx: number; cy: number; payload: TimeSeriesDataPoint };
                      if (payload?.isAnomaly) {
                        return (
                          <g>
                            <circle
                              cx={cx}
                              cy={cy}
                              r={8}
                              fill="hsl(var(--destructive))"
                              fillOpacity={0.3}
                            />
                            <circle
                              cx={cx}
                              cy={cy}
                              r={5}
                              fill="hsl(var(--destructive))"
                              stroke="white"
                              strokeWidth={2}
                            />
                          </g>
                        );
                      }
                      return <circle cx={0} cy={0} r={0} />;
                    }}
                    name="Anomalies"
                  />
                </ComposedChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>

          {/* Anomaly Events List */}
          <Card>
            <CardHeader>
              <CardTitle>Detected Anomalies</CardTitle>
              <CardDescription>
                {SAMPLE_ANOMALIES.length} anomalies detected in the selected time range
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                {SAMPLE_ANOMALIES.map((anomaly) => (
                  <div
                    key={anomaly.id}
                    className="flex items-start gap-4 p-4 rounded-lg border border-border"
                  >
                    <div className={cn('p-2 rounded-lg', getSeverityColor(anomaly.severity))}>
                      <AlertTriangle className="h-5 w-5" />
                    </div>
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center gap-2 mb-1">
                        <span className="font-medium">{anomaly.metric}</span>
                        <Badge variant="outline" className={getSeverityColor(anomaly.severity)}>
                          {anomaly.severity.toUpperCase()}
                        </Badge>
                        <span className="text-sm text-muted-foreground">
                          {formatDate(anomaly.date)}
                        </span>
                      </div>
                      <p className="text-sm text-muted-foreground mb-2">
                        {anomaly.description}
                      </p>
                      <div className="flex items-center gap-4 text-sm">
                        <span>
                          Actual: <strong>{anomaly.value.toLocaleString()}</strong>
                        </span>
                        <span>
                          Expected: <strong>{anomaly.expected.toLocaleString()}</strong>
                        </span>
                        <span className={anomaly.deviation > 0 ? 'text-emerald-600' : 'text-rose-600'}>
                          {anomaly.deviation > 0 ? '+' : ''}
                          {anomaly.deviation.toFixed(1)}% deviation
                        </span>
                      </div>
                    </div>
                    <Button variant="outline" size="sm">
                      Investigate
                    </Button>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        {/* Period Comparison Tab */}
        <TabsContent value="comparison" className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle>Period-over-Period Comparison</CardTitle>
              <CardDescription>
                Compare current period with previous periods
              </CardDescription>
            </CardHeader>
            <CardContent>
              <ResponsiveContainer width="100%" height={400}>
                <LineChart margin={{ top: 20, right: 30, left: 20, bottom: 5 }}>
                  <CartesianGrid strokeDasharray="3 3" opacity={0.3} />
                  <XAxis
                    dataKey="dayOfPeriod"
                    type="number"
                    domain={[1, 30]}
                    fontSize={12}
                    tickLine={false}
                    label={{ value: 'Day of Period', position: 'bottom', offset: -5 }}
                  />
                  <YAxis fontSize={12} tickLine={false} axisLine={false} />
                  <Tooltip />
                  <Legend />
                  <Line
                    data={SAMPLE_TIME_SERIES.slice(0, 30).map((d, i) => ({
                      dayOfPeriod: i + 1,
                      value: d.value,
                    }))}
                    type="monotone"
                    dataKey="value"
                    stroke="hsl(var(--chart-1))"
                    strokeWidth={2}
                    dot={false}
                    name="Current Period"
                  />
                  <Line
                    data={SAMPLE_TIME_SERIES.slice(30, 60).map((d, i) => ({
                      dayOfPeriod: i + 1,
                      value: d.value,
                    }))}
                    type="monotone"
                    dataKey="value"
                    stroke="hsl(var(--chart-2))"
                    strokeWidth={2}
                    dot={false}
                    name="Previous Period"
                    strokeDasharray="5 5"
                  />
                  <Line
                    data={SAMPLE_TIME_SERIES.slice(60, 90).map((d, i) => ({
                      dayOfPeriod: i + 1,
                      value: d.value,
                    }))}
                    type="monotone"
                    dataKey="value"
                    stroke="hsl(var(--chart-3))"
                    strokeWidth={2}
                    dot={false}
                    name="Two Periods Ago"
                    strokeDasharray="2 2"
                  />
                </LineChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>

          {/* Period Statistics */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-sm font-medium text-muted-foreground">
                  Current Period (Days 1-30)
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-2">
                  <div className="flex justify-between">
                    <span className="text-sm text-muted-foreground">Average</span>
                    <span className="font-medium">
                      {Math.round(
                        SAMPLE_TIME_SERIES.slice(0, 30).reduce((a, b) => a + b.value, 0) / 30
                      ).toLocaleString()}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-sm text-muted-foreground">Total</span>
                    <span className="font-medium">
                      {SAMPLE_TIME_SERIES.slice(0, 30)
                        .reduce((a, b) => a + b.value, 0)
                        .toLocaleString()}
                    </span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-sm text-muted-foreground">vs Previous</span>
                    <Badge variant="outline" className="bg-emerald-100 text-emerald-700">
                      +5.2%
                    </Badge>
                  </div>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-sm font-medium text-muted-foreground">
                  Previous Period (Days 31-60)
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-2">
                  <div className="flex justify-between">
                    <span className="text-sm text-muted-foreground">Average</span>
                    <span className="font-medium">
                      {Math.round(
                        SAMPLE_TIME_SERIES.slice(30, 60).reduce((a, b) => a + b.value, 0) / 30
                      ).toLocaleString()}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-sm text-muted-foreground">Total</span>
                    <span className="font-medium">
                      {SAMPLE_TIME_SERIES.slice(30, 60)
                        .reduce((a, b) => a + b.value, 0)
                        .toLocaleString()}
                    </span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-sm text-muted-foreground">vs Prior</span>
                    <Badge variant="outline" className="bg-emerald-100 text-emerald-700">
                      +3.8%
                    </Badge>
                  </div>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-sm font-medium text-muted-foreground">
                  Two Periods Ago (Days 61-90)
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-2">
                  <div className="flex justify-between">
                    <span className="text-sm text-muted-foreground">Average</span>
                    <span className="font-medium">
                      {Math.round(
                        SAMPLE_TIME_SERIES.slice(60, 90).reduce((a, b) => a + b.value, 0) / 30
                      ).toLocaleString()}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-sm text-muted-foreground">Total</span>
                    <span className="font-medium">
                      {SAMPLE_TIME_SERIES.slice(60, 90)
                        .reduce((a, b) => a + b.value, 0)
                        .toLocaleString()}
                    </span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-sm text-muted-foreground">Baseline</span>
                    <Badge variant="outline">Reference</Badge>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>
        </TabsContent>
      </Tabs>
    </div>
  );
}

export default TimeSeries;
