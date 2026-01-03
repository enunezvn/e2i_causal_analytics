/**
 * Monitoring Page
 * ===============
 *
 * Dashboard for user activity logs, API usage statistics,
 * error tracking, and performance metrics.
 *
 * @module pages/Monitoring
 */

import { useState, useMemo } from 'react';
import {
  Activity,
  RefreshCw,
  Download,
  Clock,
  Users,
  Zap,
  AlertCircle,
  CheckCircle2,
  XCircle,
  Search,
  Filter,
  TrendingUp,
  TrendingDown,
  Server,
  Globe,
  BarChart3,
  List,
} from 'lucide-react';
import { LineChart, Line, BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid, Legend, AreaChart, Area } from 'recharts';

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
import { KPICard, StatusBadge, AlertList } from '@/components/visualizations';

// =============================================================================
// TYPES
// =============================================================================

interface ApiMetric {
  timestamp: string;
  requests: number;
  errors: number;
  latencyP50: number;
  latencyP99: number;
}

interface EndpointStats {
  endpoint: string;
  method: 'GET' | 'POST' | 'PUT' | 'DELETE';
  requests: number;
  errors: number;
  avgLatency: number;
  status: 'healthy' | 'warning' | 'error';
}

interface UserActivity {
  id: string;
  userId: string;
  userName: string;
  action: string;
  resource: string;
  timestamp: string;
  ipAddress: string;
  userAgent: string;
  status: 'success' | 'failure';
}

interface ErrorLog {
  id: string;
  timestamp: string;
  level: 'error' | 'warning' | 'critical';
  message: string;
  endpoint: string;
  stackTrace?: string;
  userId?: string;
  requestId: string;
  count: number;
}

interface SystemMetric {
  name: string;
  value: number;
  unit: string;
  status: 'healthy' | 'warning' | 'critical';
  trend: 'up' | 'down' | 'stable';
  trendValue: number;
}

// =============================================================================
// SAMPLE DATA
// =============================================================================

const SAMPLE_API_METRICS: ApiMetric[] = Array.from({ length: 24 }, (_, i) => ({
  timestamp: `${String(i).padStart(2, '0')}:00`,
  requests: Math.floor(Math.random() * 5000) + 2000,
  errors: Math.floor(Math.random() * 50) + 5,
  latencyP50: Math.floor(Math.random() * 50) + 20,
  latencyP99: Math.floor(Math.random() * 200) + 100,
}));

const SAMPLE_ENDPOINT_STATS: EndpointStats[] = [
  { endpoint: '/api/v1/query', method: 'POST', requests: 45230, errors: 23, avgLatency: 245, status: 'healthy' },
  { endpoint: '/api/v1/agents/orchestrate', method: 'POST', requests: 12450, errors: 156, avgLatency: 1250, status: 'warning' },
  { endpoint: '/api/v1/kpis', method: 'GET', requests: 28900, errors: 5, avgLatency: 45, status: 'healthy' },
  { endpoint: '/api/v1/predictions', method: 'GET', requests: 15600, errors: 12, avgLatency: 320, status: 'healthy' },
  { endpoint: '/api/v1/explain', method: 'POST', requests: 8200, errors: 89, avgLatency: 890, status: 'warning' },
  { endpoint: '/api/v1/causal/discover', method: 'POST', requests: 3400, errors: 245, avgLatency: 2150, status: 'error' },
  { endpoint: '/api/v1/health', method: 'GET', requests: 86400, errors: 0, avgLatency: 12, status: 'healthy' },
  { endpoint: '/api/v1/users', method: 'GET', requests: 5600, errors: 2, avgLatency: 35, status: 'healthy' },
];

const SAMPLE_USER_ACTIVITIES: UserActivity[] = [
  { id: 'ua-001', userId: 'usr-001', userName: 'John Smith', action: 'Query', resource: 'HCP Targeting', timestamp: '2026-01-02T08:45:23Z', ipAddress: '10.0.0.15', userAgent: 'Chrome/120', status: 'success' },
  { id: 'ua-002', userId: 'usr-002', userName: 'Sarah Johnson', action: 'Export', resource: 'TRx Report', timestamp: '2026-01-02T08:42:10Z', ipAddress: '10.0.0.22', userAgent: 'Firefox/122', status: 'success' },
  { id: 'ua-003', userId: 'usr-003', userName: 'Mike Chen', action: 'Query', resource: 'Causal Discovery', timestamp: '2026-01-02T08:40:55Z', ipAddress: '10.0.0.8', userAgent: 'Chrome/120', status: 'failure' },
  { id: 'ua-004', userId: 'usr-001', userName: 'John Smith', action: 'View', resource: 'Dashboard', timestamp: '2026-01-02T08:38:00Z', ipAddress: '10.0.0.15', userAgent: 'Chrome/120', status: 'success' },
  { id: 'ua-005', userId: 'usr-004', userName: 'Emily Davis', action: 'Update', resource: 'Model Config', timestamp: '2026-01-02T08:35:42Z', ipAddress: '10.0.0.31', userAgent: 'Safari/17', status: 'success' },
  { id: 'ua-006', userId: 'usr-002', userName: 'Sarah Johnson', action: 'Query', resource: 'Gap Analysis', timestamp: '2026-01-02T08:32:18Z', ipAddress: '10.0.0.22', userAgent: 'Firefox/122', status: 'success' },
  { id: 'ua-007', userId: 'usr-005', userName: 'David Wilson', action: 'Login', resource: 'Session', timestamp: '2026-01-02T08:30:00Z', ipAddress: '10.0.0.45', userAgent: 'Edge/120', status: 'success' },
  { id: 'ua-008', userId: 'usr-003', userName: 'Mike Chen', action: 'Query', resource: 'Feature Importance', timestamp: '2026-01-02T08:28:15Z', ipAddress: '10.0.0.8', userAgent: 'Chrome/120', status: 'success' },
];

const SAMPLE_ERROR_LOGS: ErrorLog[] = [
  { id: 'err-001', timestamp: '2026-01-02T08:40:55Z', level: 'error', message: 'Causal discovery timeout: operation exceeded 30s limit', endpoint: '/api/v1/causal/discover', requestId: 'req-abc123', count: 45 },
  { id: 'err-002', timestamp: '2026-01-02T08:35:12Z', level: 'critical', message: 'Database connection pool exhausted', endpoint: '/api/v1/query', requestId: 'req-def456', count: 12 },
  { id: 'err-003', timestamp: '2026-01-02T08:30:00Z', level: 'warning', message: 'Rate limit exceeded for user usr-006', endpoint: '/api/v1/agents/orchestrate', userId: 'usr-006', requestId: 'req-ghi789', count: 156 },
  { id: 'err-004', timestamp: '2026-01-02T08:25:30Z', level: 'error', message: 'Model inference failed: insufficient memory', endpoint: '/api/v1/explain', requestId: 'req-jkl012', count: 23 },
  { id: 'err-005', timestamp: '2026-01-02T08:20:00Z', level: 'warning', message: 'Slow query detected: 2.5s response time', endpoint: '/api/v1/predictions', requestId: 'req-mno345', count: 89 },
  { id: 'err-006', timestamp: '2026-01-02T08:15:45Z', level: 'error', message: 'Authentication token expired', endpoint: '/api/v1/users', userId: 'usr-007', requestId: 'req-pqr678', count: 5 },
];

const SAMPLE_SYSTEM_METRICS: SystemMetric[] = [
  { name: 'CPU Usage', value: 42, unit: '%', status: 'healthy', trend: 'up', trendValue: 5 },
  { name: 'Memory Usage', value: 68, unit: '%', status: 'warning', trend: 'up', trendValue: 12 },
  { name: 'Disk I/O', value: 125, unit: 'MB/s', status: 'healthy', trend: 'stable', trendValue: 0 },
  { name: 'Network In', value: 450, unit: 'Mbps', status: 'healthy', trend: 'down', trendValue: -8 },
  { name: 'Network Out', value: 320, unit: 'Mbps', status: 'healthy', trend: 'up', trendValue: 3 },
  { name: 'Active Connections', value: 1250, unit: '', status: 'healthy', trend: 'up', trendValue: 15 },
];

// =============================================================================
// HELPER FUNCTIONS
// =============================================================================

function formatNumber(num: number): string {
  if (num >= 1000000) return `${(num / 1000000).toFixed(1)}M`;
  if (num >= 1000) return `${(num / 1000).toFixed(1)}K`;
  return num.toString();
}

function formatTimestamp(timestamp: string): string {
  const date = new Date(timestamp);
  return date.toLocaleString();
}

function getMethodColor(method: string): string {
  switch (method) {
    case 'GET':
      return 'bg-blue-100 text-blue-700 dark:bg-blue-900/30 dark:text-blue-400';
    case 'POST':
      return 'bg-emerald-100 text-emerald-700 dark:bg-emerald-900/30 dark:text-emerald-400';
    case 'PUT':
      return 'bg-amber-100 text-amber-700 dark:bg-amber-900/30 dark:text-amber-400';
    case 'DELETE':
      return 'bg-rose-100 text-rose-700 dark:bg-rose-900/30 dark:text-rose-400';
    default:
      return 'bg-gray-100 text-gray-700';
  }
}

function getErrorLevelStyle(level: string): { bg: string; text: string } {
  switch (level) {
    case 'critical':
      return { bg: 'bg-red-100 dark:bg-red-900/30', text: 'text-red-700 dark:text-red-400' };
    case 'error':
      return { bg: 'bg-rose-100 dark:bg-rose-900/30', text: 'text-rose-700 dark:text-rose-400' };
    case 'warning':
      return { bg: 'bg-amber-100 dark:bg-amber-900/30', text: 'text-amber-700 dark:text-amber-400' };
    default:
      return { bg: 'bg-gray-100', text: 'text-gray-700' };
  }
}

// =============================================================================
// COMPONENT
// =============================================================================

function Monitoring() {
  const [timeRange, setTimeRange] = useState<string>('24h');
  const [searchQuery, setSearchQuery] = useState('');
  const [isRefreshing, setIsRefreshing] = useState(false);
  const [errorLevelFilter, setErrorLevelFilter] = useState<string>('all');

  // Calculate overview metrics
  const overviewMetrics = useMemo(() => {
    const totalRequests = SAMPLE_API_METRICS.reduce((sum, m) => sum + m.requests, 0);
    const totalErrors = SAMPLE_API_METRICS.reduce((sum, m) => sum + m.errors, 0);
    const avgLatency = SAMPLE_API_METRICS.reduce((sum, m) => sum + m.latencyP50, 0) / SAMPLE_API_METRICS.length;
    const errorRate = (totalErrors / totalRequests) * 100;

    return {
      totalRequests,
      totalErrors,
      avgLatency: Math.round(avgLatency),
      errorRate: errorRate.toFixed(2),
      activeUsers: 48,
      uptime: 99.97,
    };
  }, []);

  // Filter error logs
  const filteredErrors = useMemo(() => {
    return SAMPLE_ERROR_LOGS.filter((error) => {
      const matchesLevel = errorLevelFilter === 'all' || error.level === errorLevelFilter;
      const matchesSearch =
        searchQuery === '' ||
        error.message.toLowerCase().includes(searchQuery.toLowerCase()) ||
        error.endpoint.toLowerCase().includes(searchQuery.toLowerCase());
      return matchesLevel && matchesSearch;
    });
  }, [errorLevelFilter, searchQuery]);

  // Filter user activities
  const filteredActivities = useMemo(() => {
    if (searchQuery === '') return SAMPLE_USER_ACTIVITIES;
    return SAMPLE_USER_ACTIVITIES.filter(
      (activity) =>
        activity.userName.toLowerCase().includes(searchQuery.toLowerCase()) ||
        activity.action.toLowerCase().includes(searchQuery.toLowerCase()) ||
        activity.resource.toLowerCase().includes(searchQuery.toLowerCase())
    );
  }, [searchQuery]);

  const handleRefresh = () => {
    setIsRefreshing(true);
    setTimeout(() => setIsRefreshing(false), 2000);
  };

  const handleExport = () => {
    const report = {
      generatedAt: new Date().toISOString(),
      timeRange,
      overview: overviewMetrics,
      apiMetrics: SAMPLE_API_METRICS,
      endpointStats: SAMPLE_ENDPOINT_STATS,
      errors: SAMPLE_ERROR_LOGS,
    };
    const blob = new Blob([JSON.stringify(report, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `monitoring-report-${new Date().toISOString().split('T')[0]}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="container mx-auto px-4 py-8">
      {/* Header */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 mb-8">
        <div>
          <h1 className="text-3xl font-bold mb-2 flex items-center gap-3">
            <Activity className="h-8 w-8" />
            Monitoring
          </h1>
          <p className="text-muted-foreground">
            User activity logs, API usage statistics, error tracking, and performance metrics.
          </p>
        </div>
        <div className="flex items-center gap-2">
          <Select value={timeRange} onValueChange={setTimeRange}>
            <SelectTrigger className="w-32">
              <Clock className="h-4 w-4 mr-2" />
              <SelectValue placeholder="Time Range" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="1h">Last Hour</SelectItem>
              <SelectItem value="6h">Last 6 Hours</SelectItem>
              <SelectItem value="24h">Last 24 Hours</SelectItem>
              <SelectItem value="7d">Last 7 Days</SelectItem>
              <SelectItem value="30d">Last 30 Days</SelectItem>
            </SelectContent>
          </Select>
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

      {/* Overview Metrics */}
      <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-4 mb-8">
        <KPICard
          title="Total Requests"
          value={overviewMetrics.totalRequests}
          status="healthy"
          description="API requests in selected period"
          sparklineData={SAMPLE_API_METRICS.map((m) => m.requests)}
          size="sm"
        />
        <KPICard
          title="Error Rate"
          value={parseFloat(overviewMetrics.errorRate)}
          unit="%"
          status={parseFloat(overviewMetrics.errorRate) < 1 ? 'healthy' : 'warning'}
          description="Percentage of failed requests"
          sparklineData={SAMPLE_API_METRICS.map((m) => (m.errors / m.requests) * 100)}
          higherIsBetter={false}
          size="sm"
        />
        <KPICard
          title="Avg Latency"
          value={overviewMetrics.avgLatency}
          unit="ms"
          status={overviewMetrics.avgLatency < 100 ? 'healthy' : 'warning'}
          description="P50 response time"
          sparklineData={SAMPLE_API_METRICS.map((m) => m.latencyP50)}
          higherIsBetter={false}
          size="sm"
        />
        <KPICard
          title="Active Users"
          value={overviewMetrics.activeUsers}
          status="healthy"
          description="Currently active sessions"
          sparklineData={[35, 38, 42, 45, 48, 52, 50, 48, 45, overviewMetrics.activeUsers]}
          size="sm"
        />
        <KPICard
          title="Total Errors"
          value={overviewMetrics.totalErrors}
          status={overviewMetrics.totalErrors < 500 ? 'healthy' : 'critical'}
          description="Failed requests count"
          sparklineData={SAMPLE_API_METRICS.map((m) => m.errors)}
          higherIsBetter={false}
          size="sm"
        />
        <KPICard
          title="Uptime"
          value={overviewMetrics.uptime}
          unit="%"
          status="healthy"
          description="Service availability"
          sparklineData={[99.95, 99.97, 99.98, 99.99, 99.97, 99.98, 99.99, 99.97, 99.98, overviewMetrics.uptime]}
          size="sm"
        />
      </div>

      {/* Tabs for different views */}
      <Tabs defaultValue="api" className="space-y-4">
        <TabsList>
          <TabsTrigger value="api" className="flex items-center gap-2">
            <Globe className="h-4 w-4" />
            API Usage
          </TabsTrigger>
          <TabsTrigger value="activity" className="flex items-center gap-2">
            <Users className="h-4 w-4" />
            User Activity
          </TabsTrigger>
          <TabsTrigger value="errors" className="flex items-center gap-2">
            <AlertCircle className="h-4 w-4" />
            Errors
            <Badge variant="destructive" className="ml-1">
              {SAMPLE_ERROR_LOGS.filter((e) => e.level === 'critical' || e.level === 'error').length}
            </Badge>
          </TabsTrigger>
          <TabsTrigger value="system" className="flex items-center gap-2">
            <Server className="h-4 w-4" />
            System
          </TabsTrigger>
        </TabsList>

        {/* API Usage Tab */}
        <TabsContent value="api" className="space-y-4">
          {/* Request Volume Chart */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <BarChart3 className="h-5 w-5" />
                Request Volume & Errors
              </CardTitle>
              <CardDescription>API requests and errors over time</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="h-64">
                <ResponsiveContainer width="100%" height="100%">
                  <AreaChart data={SAMPLE_API_METRICS}>
                    <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
                    <XAxis dataKey="timestamp" stroke="var(--muted-foreground)" fontSize={12} />
                    <YAxis stroke="var(--muted-foreground)" fontSize={12} />
                    <Tooltip
                      contentStyle={{
                        backgroundColor: 'var(--card)',
                        border: '1px solid var(--border)',
                        borderRadius: '8px',
                      }}
                    />
                    <Legend />
                    <Area
                      type="monotone"
                      dataKey="requests"
                      stroke="#10b981"
                      fill="#10b981"
                      fillOpacity={0.2}
                      name="Requests"
                    />
                    <Area
                      type="monotone"
                      dataKey="errors"
                      stroke="#ef4444"
                      fill="#ef4444"
                      fillOpacity={0.3}
                      name="Errors"
                    />
                  </AreaChart>
                </ResponsiveContainer>
              </div>
            </CardContent>
          </Card>

          {/* Latency Chart */}
          <Card>
            <CardHeader>
              <CardTitle>Response Latency</CardTitle>
              <CardDescription>P50 and P99 response times</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="h-48">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={SAMPLE_API_METRICS}>
                    <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
                    <XAxis dataKey="timestamp" stroke="var(--muted-foreground)" fontSize={12} />
                    <YAxis stroke="var(--muted-foreground)" fontSize={12} unit="ms" />
                    <Tooltip
                      contentStyle={{
                        backgroundColor: 'var(--card)',
                        border: '1px solid var(--border)',
                        borderRadius: '8px',
                      }}
                    />
                    <Legend />
                    <Line type="monotone" dataKey="latencyP50" stroke="#3b82f6" name="P50" strokeWidth={2} dot={false} />
                    <Line type="monotone" dataKey="latencyP99" stroke="#f59e0b" name="P99" strokeWidth={2} dot={false} />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </CardContent>
          </Card>

          {/* Endpoint Stats */}
          <Card>
            <CardHeader>
              <CardTitle>Endpoint Statistics</CardTitle>
              <CardDescription>Performance breakdown by endpoint</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="overflow-x-auto">
                <table className="w-full">
                  <thead>
                    <tr className="border-b border-border">
                      <th className="text-left py-3 px-4 font-medium text-muted-foreground">Endpoint</th>
                      <th className="text-left py-3 px-4 font-medium text-muted-foreground">Method</th>
                      <th className="text-right py-3 px-4 font-medium text-muted-foreground">Requests</th>
                      <th className="text-right py-3 px-4 font-medium text-muted-foreground">Errors</th>
                      <th className="text-right py-3 px-4 font-medium text-muted-foreground">Error Rate</th>
                      <th className="text-right py-3 px-4 font-medium text-muted-foreground">Avg Latency</th>
                      <th className="text-center py-3 px-4 font-medium text-muted-foreground">Status</th>
                    </tr>
                  </thead>
                  <tbody>
                    {SAMPLE_ENDPOINT_STATS.map((endpoint, i) => (
                      <tr key={i} className="border-b border-border hover:bg-muted/50">
                        <td className="py-3 px-4">
                          <code className="text-sm">{endpoint.endpoint}</code>
                        </td>
                        <td className="py-3 px-4">
                          <Badge className={getMethodColor(endpoint.method)}>{endpoint.method}</Badge>
                        </td>
                        <td className="py-3 px-4 text-right font-medium">{formatNumber(endpoint.requests)}</td>
                        <td className="py-3 px-4 text-right">
                          {endpoint.errors > 0 ? (
                            <span className="text-rose-500">{formatNumber(endpoint.errors)}</span>
                          ) : (
                            <span className="text-muted-foreground">0</span>
                          )}
                        </td>
                        <td className="py-3 px-4 text-right">
                          <span
                            className={
                              (endpoint.errors / endpoint.requests) * 100 > 1
                                ? 'text-rose-500'
                                : 'text-muted-foreground'
                            }
                          >
                            {((endpoint.errors / endpoint.requests) * 100).toFixed(2)}%
                          </span>
                        </td>
                        <td className="py-3 px-4 text-right">
                          <span className={endpoint.avgLatency > 500 ? 'text-amber-500' : ''}>
                            {endpoint.avgLatency}ms
                          </span>
                        </td>
                        <td className="py-3 px-4 text-center">
                          <StatusBadge status={endpoint.status} size="sm" />
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        {/* User Activity Tab */}
        <TabsContent value="activity">
          <Card>
            <CardHeader>
              <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
                <div>
                  <CardTitle className="flex items-center gap-2">
                    <List className="h-5 w-5" />
                    User Activity Log
                  </CardTitle>
                  <CardDescription>Recent user actions and events</CardDescription>
                </div>
                <div className="relative">
                  <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-muted-foreground" />
                  <Input
                    placeholder="Search activities..."
                    value={searchQuery}
                    onChange={(e) => setSearchQuery(e.target.value)}
                    className="pl-9 w-64"
                  />
                </div>
              </div>
            </CardHeader>
            <CardContent>
              <div className="overflow-x-auto">
                <table className="w-full">
                  <thead>
                    <tr className="border-b border-border">
                      <th className="text-left py-3 px-4 font-medium text-muted-foreground">Timestamp</th>
                      <th className="text-left py-3 px-4 font-medium text-muted-foreground">User</th>
                      <th className="text-left py-3 px-4 font-medium text-muted-foreground">Action</th>
                      <th className="text-left py-3 px-4 font-medium text-muted-foreground">Resource</th>
                      <th className="text-left py-3 px-4 font-medium text-muted-foreground">IP Address</th>
                      <th className="text-center py-3 px-4 font-medium text-muted-foreground">Status</th>
                    </tr>
                  </thead>
                  <tbody>
                    {filteredActivities.map((activity) => (
                      <tr key={activity.id} className="border-b border-border hover:bg-muted/50">
                        <td className="py-3 px-4 text-sm text-muted-foreground">
                          {formatTimestamp(activity.timestamp)}
                        </td>
                        <td className="py-3 px-4">
                          <div>
                            <p className="font-medium">{activity.userName}</p>
                            <p className="text-xs text-muted-foreground">{activity.userId}</p>
                          </div>
                        </td>
                        <td className="py-3 px-4">
                          <Badge variant="outline">{activity.action}</Badge>
                        </td>
                        <td className="py-3 px-4 text-sm">{activity.resource}</td>
                        <td className="py-3 px-4">
                          <code className="text-xs bg-muted px-1.5 py-0.5 rounded">{activity.ipAddress}</code>
                        </td>
                        <td className="py-3 px-4 text-center">
                          {activity.status === 'success' ? (
                            <CheckCircle2 className="h-4 w-4 text-emerald-500 mx-auto" />
                          ) : (
                            <XCircle className="h-4 w-4 text-rose-500 mx-auto" />
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

        {/* Errors Tab */}
        <TabsContent value="errors">
          <Card>
            <CardHeader>
              <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
                <div>
                  <CardTitle className="flex items-center gap-2">
                    <AlertCircle className="h-5 w-5" />
                    Error Logs
                  </CardTitle>
                  <CardDescription>Application errors and warnings</CardDescription>
                </div>
                <div className="flex items-center gap-2">
                  <div className="relative">
                    <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-muted-foreground" />
                    <Input
                      placeholder="Search errors..."
                      value={searchQuery}
                      onChange={(e) => setSearchQuery(e.target.value)}
                      className="pl-9 w-64"
                    />
                  </div>
                  <Select value={errorLevelFilter} onValueChange={setErrorLevelFilter}>
                    <SelectTrigger className="w-36">
                      <Filter className="h-4 w-4 mr-2" />
                      <SelectValue placeholder="Level" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="all">All Levels</SelectItem>
                      <SelectItem value="critical">Critical</SelectItem>
                      <SelectItem value="error">Error</SelectItem>
                      <SelectItem value="warning">Warning</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
              </div>
            </CardHeader>
            <CardContent>
              <div className="space-y-3">
                {filteredErrors.map((error) => {
                  const levelStyle = getErrorLevelStyle(error.level);
                  return (
                    <div
                      key={error.id}
                      className={`p-4 rounded-lg border border-border ${levelStyle.bg}`}
                    >
                      <div className="flex items-start justify-between gap-4">
                        <div className="flex-1">
                          <div className="flex items-center gap-2 mb-1">
                            <Badge className={`${levelStyle.bg} ${levelStyle.text} uppercase text-xs`}>
                              {error.level}
                            </Badge>
                            <code className="text-xs bg-muted px-1.5 py-0.5 rounded">{error.endpoint}</code>
                            <span className="text-xs text-muted-foreground">
                              {error.count > 1 && `(${error.count} occurrences)`}
                            </span>
                          </div>
                          <p className={`font-medium ${levelStyle.text}`}>{error.message}</p>
                          <div className="flex items-center gap-4 mt-2 text-xs text-muted-foreground">
                            <span className="flex items-center gap-1">
                              <Clock className="h-3 w-3" />
                              {formatTimestamp(error.timestamp)}
                            </span>
                            <span>Request ID: {error.requestId}</span>
                            {error.userId && <span>User: {error.userId}</span>}
                          </div>
                        </div>
                        <Button variant="outline" size="sm">
                          View Details
                        </Button>
                      </div>
                    </div>
                  );
                })}
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        {/* System Tab */}
        <TabsContent value="system">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            {/* System Metrics */}
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Server className="h-5 w-5" />
                  System Resources
                </CardTitle>
                <CardDescription>Server resource utilization</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  {SAMPLE_SYSTEM_METRICS.map((metric, i) => (
                    <div key={i} className="flex items-center justify-between">
                      <div className="flex items-center gap-3">
                        <div
                          className={`w-2 h-8 rounded-full ${
                            metric.status === 'healthy'
                              ? 'bg-emerald-500'
                              : metric.status === 'warning'
                              ? 'bg-amber-500'
                              : 'bg-rose-500'
                          }`}
                        />
                        <div>
                          <p className="font-medium">{metric.name}</p>
                          <div className="flex items-center gap-1 text-sm">
                            {metric.trend === 'up' && <TrendingUp className="h-3 w-3 text-rose-500" />}
                            {metric.trend === 'down' && <TrendingDown className="h-3 w-3 text-emerald-500" />}
                            <span className="text-muted-foreground">
                              {metric.trendValue > 0 ? '+' : ''}
                              {metric.trendValue}%
                            </span>
                          </div>
                        </div>
                      </div>
                      <div className="text-right">
                        <p className="text-2xl font-bold">
                          {metric.value}
                          <span className="text-sm font-normal text-muted-foreground ml-1">{metric.unit}</span>
                        </p>
                      </div>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>

            {/* Service Health */}
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Zap className="h-5 w-5" />
                  Service Health
                </CardTitle>
                <CardDescription>Status of dependent services</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-3">
                  {[
                    { name: 'API Gateway', status: 'healthy', latency: '12ms' },
                    { name: 'PostgreSQL', status: 'healthy', latency: '3ms' },
                    { name: 'Redis Cache', status: 'healthy', latency: '1ms' },
                    { name: 'ML Inference', status: 'warning', latency: '890ms' },
                    { name: 'Causal Engine', status: 'warning', latency: '2.1s' },
                    { name: 'Vector Store', status: 'healthy', latency: '45ms' },
                    { name: 'Message Queue', status: 'healthy', latency: '5ms' },
                    { name: 'External APIs', status: 'healthy', latency: '125ms' },
                  ].map((service, i) => (
                    <div
                      key={i}
                      className="flex items-center justify-between p-3 rounded-lg border border-border"
                    >
                      <div className="flex items-center gap-3">
                        <StatusBadge status={service.status as 'healthy' | 'warning' | 'error'} showIcon pulse={service.status !== 'healthy'} />
                        <span className="font-medium">{service.name}</span>
                      </div>
                      <div className="text-sm text-muted-foreground">{service.latency}</div>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          </div>
        </TabsContent>
      </Tabs>
    </div>
  );
}

export default Monitoring;
