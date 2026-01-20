/**
 * Experiments Page
 * ================
 *
 * A/B Testing & Experiments management dashboard.
 * Displays experiment health, enrollment stats, SRM checks,
 * interim analyses, and Digital Twin fidelity tracking.
 *
 * @module pages/Experiments
 */

import { useState, useMemo } from 'react';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell,
  LineChart,
  Line,
} from 'recharts';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Badge } from '@/components/ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert';
import {
  useEnrollmentStats,
  useSRMChecks,
  useExperimentHealth,
  useExperimentAlerts,
  useTriggerMonitoring,
} from '@/hooks/api';
import {
  AlertSeverity,
  ExperimentHealthStatus,
  StoppingDecision,
  MonitorAlert,
} from '@/types/experiments';
import { KPICard } from '@/components/visualizations';
import {
  RefreshCw,
  Play,
  AlertTriangle,
  CheckCircle,
  XCircle,
  Activity,
  Target,
  Beaker,
  Shield,
  Clock,
} from 'lucide-react';

// =============================================================================
// LOCAL TYPES
// =============================================================================

interface LocalExperiment {
  experiment_id: string;
  experiment_name: string;
  health_status: ExperimentHealthStatus;
  total_enrolled: number;
  enrollment_rate_per_day: number;
  current_information_fraction: number;
  has_srm: boolean;
  active_alerts: number;
  last_checked: string;
  variant_breakdown: Record<string, number>;
  start_date: string;
  primary_metric: string;
}

// =============================================================================
// SAMPLE DATA (when API is unavailable)
// =============================================================================

const SAMPLE_EXPERIMENTS: LocalExperiment[] = [
  {
    experiment_id: 'exp_kisqali_hcp_outreach_001',
    experiment_name: 'Kisqali HCP Outreach Campaign',
    health_status: ExperimentHealthStatus.HEALTHY,
    total_enrolled: 1250,
    enrollment_rate_per_day: 25.5,
    current_information_fraction: 0.65,
    has_srm: false,
    active_alerts: 0,
    last_checked: new Date().toISOString(),
    variant_breakdown: { control: 625, treatment: 625 },
    start_date: '2026-01-01',
    primary_metric: 'trx_conversion_rate',
  },
  {
    experiment_id: 'exp_fabhalta_digital_001',
    experiment_name: 'Fabhalta Digital Engagement',
    health_status: ExperimentHealthStatus.WARNING,
    total_enrolled: 850,
    enrollment_rate_per_day: 18.2,
    current_information_fraction: 0.42,
    has_srm: false,
    active_alerts: 2,
    last_checked: new Date().toISOString(),
    variant_breakdown: { control: 420, treatment: 430 },
    start_date: '2026-01-05',
    primary_metric: 'nrx_rate',
  },
  {
    experiment_id: 'exp_remibrutinib_rep_001',
    experiment_name: 'Remibrutinib Rep Training',
    health_status: ExperimentHealthStatus.CRITICAL,
    total_enrolled: 320,
    enrollment_rate_per_day: 8.5,
    current_information_fraction: 0.22,
    has_srm: true,
    active_alerts: 5,
    last_checked: new Date().toISOString(),
    variant_breakdown: { control: 180, treatment: 140 },
    start_date: '2026-01-10',
    primary_metric: 'call_quality_score',
  },
  {
    experiment_id: 'exp_multi_brand_001',
    experiment_name: 'Multi-brand Messaging Test',
    health_status: ExperimentHealthStatus.HEALTHY,
    total_enrolled: 2100,
    enrollment_rate_per_day: 42.0,
    current_information_fraction: 0.78,
    has_srm: false,
    active_alerts: 0,
    last_checked: new Date().toISOString(),
    variant_breakdown: { control: 700, treatment_a: 700, treatment_b: 700 },
    start_date: '2025-12-15',
    primary_metric: 'engagement_rate',
  },
];

const SAMPLE_ALERTS: MonitorAlert[] = [
  {
    alert_id: 'alert_001',
    alert_type: 'enrollment_slow',
    severity: AlertSeverity.WARNING,
    experiment_id: 'exp_fabhalta_digital_001',
    experiment_name: 'Fabhalta Digital Engagement',
    message: 'Enrollment rate 27% below target',
    details: { target_rate: 25, actual_rate: 18.2 },
    recommended_action: 'Consider expanding eligibility criteria or extending recruitment timeline',
    timestamp: new Date(Date.now() - 2 * 60 * 60 * 1000).toISOString(),
  },
  {
    alert_id: 'alert_002',
    alert_type: 'srm_detected',
    severity: AlertSeverity.CRITICAL,
    experiment_id: 'exp_remibrutinib_rep_001',
    experiment_name: 'Remibrutinib Rep Training',
    message: 'Sample Ratio Mismatch detected (p < 0.001)',
    details: { expected_ratio: '50:50', actual_ratio: '56:44', chi_squared: 12.5 },
    recommended_action: 'Investigate randomization process immediately. Check for systematic exclusions.',
    timestamp: new Date(Date.now() - 30 * 60 * 1000).toISOString(),
  },
  {
    alert_id: 'alert_003',
    alert_type: 'fidelity_low',
    severity: AlertSeverity.WARNING,
    experiment_id: 'exp_remibrutinib_rep_001',
    experiment_name: 'Remibrutinib Rep Training',
    message: 'Digital Twin fidelity score dropped below threshold',
    details: { current_score: 0.72, threshold: 0.85 },
    recommended_action: 'Review Digital Twin calibration and update model parameters',
    timestamp: new Date(Date.now() - 4 * 60 * 60 * 1000).toISOString(),
  },
];

const SAMPLE_INTERIM_ANALYSES = [
  {
    analysis_id: 'ia_001',
    experiment_id: 'exp_multi_brand_001',
    analysis_number: 3,
    performed_at: new Date(Date.now() - 24 * 60 * 60 * 1000).toISOString(),
    information_fraction: 0.78,
    p_value: 0.018,
    decision: StoppingDecision.CONTINUE,
    alpha_spent: 0.0234,
    adjusted_alpha: 0.0212,
  },
  {
    analysis_id: 'ia_002',
    experiment_id: 'exp_kisqali_hcp_outreach_001',
    analysis_number: 2,
    performed_at: new Date(Date.now() - 3 * 24 * 60 * 60 * 1000).toISOString(),
    information_fraction: 0.65,
    p_value: 0.042,
    decision: StoppingDecision.CONTINUE,
    alpha_spent: 0.0156,
    adjusted_alpha: 0.0252,
  },
];

const SAMPLE_FIDELITY = [
  { date: '2026-01-10', score: 0.92 },
  { date: '2026-01-12', score: 0.89 },
  { date: '2026-01-14', score: 0.85 },
  { date: '2026-01-16', score: 0.78 },
  { date: '2026-01-18', score: 0.72 },
  { date: '2026-01-20', score: 0.74 },
];

// =============================================================================
// CONSTANTS
// =============================================================================

const COLORS = {
  healthy: '#10b981',
  warning: '#f59e0b',
  critical: '#ef4444',
  primary: '#3b82f6',
  secondary: '#8b5cf6',
  tertiary: '#06b6d4',
};

const PIE_COLORS = [COLORS.primary, COLORS.secondary, COLORS.tertiary, '#f97316'];

// =============================================================================
// HELPER FUNCTIONS
// =============================================================================

function getHealthIcon(status: ExperimentHealthStatus) {
  switch (status) {
    case ExperimentHealthStatus.HEALTHY:
      return <CheckCircle className="h-4 w-4 text-green-500" />;
    case ExperimentHealthStatus.WARNING:
      return <AlertTriangle className="h-4 w-4 text-yellow-500" />;
    case ExperimentHealthStatus.CRITICAL:
      return <XCircle className="h-4 w-4 text-red-500" />;
  }
}

function getHealthBadge(status: ExperimentHealthStatus) {
  const variants: Record<ExperimentHealthStatus, 'default' | 'secondary' | 'destructive'> = {
    [ExperimentHealthStatus.HEALTHY]: 'default',
    [ExperimentHealthStatus.WARNING]: 'secondary',
    [ExperimentHealthStatus.CRITICAL]: 'destructive',
  };
  return (
    <Badge variant={variants[status]} className="capitalize">
      {status}
    </Badge>
  );
}

function getSeverityIcon(severity: AlertSeverity) {
  switch (severity) {
    case AlertSeverity.CRITICAL:
      return <XCircle className="h-4 w-4 text-red-500" />;
    case AlertSeverity.WARNING:
      return <AlertTriangle className="h-4 w-4 text-yellow-500" />;
    case AlertSeverity.INFO:
      return <Activity className="h-4 w-4 text-blue-500" />;
  }
}

function formatTimestamp(timestamp: string): string {
  const date = new Date(timestamp);
  return date.toLocaleString();
}

function formatTimeAgo(timestamp: string): string {
  const seconds = Math.floor((Date.now() - new Date(timestamp).getTime()) / 1000);
  if (seconds < 60) return `${seconds}s ago`;
  const minutes = Math.floor(seconds / 60);
  if (minutes < 60) return `${minutes}m ago`;
  const hours = Math.floor(minutes / 60);
  if (hours < 24) return `${hours}h ago`;
  const days = Math.floor(hours / 24);
  return `${days}d ago`;
}

// =============================================================================
// MAIN COMPONENT
// =============================================================================

export default function Experiments() {
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedExperiment, setSelectedExperiment] = useState<string | null>(null);

  // API hooks with fallback to sample data
  const { data: monitorData, isLoading: isLoadingMonitor, refetch: refetchMonitor } = useTriggerMonitoring();

  // Derive experiments from monitor data or use sample data
  const experiments = useMemo(() => {
    if (monitorData?.experiments?.length) {
      return monitorData.experiments;
    }
    return SAMPLE_EXPERIMENTS;
  }, [monitorData]);

  const alerts = useMemo(() => {
    if (monitorData?.alerts?.length) {
      return monitorData.alerts;
    }
    return SAMPLE_ALERTS;
  }, [monitorData]);

  // Filter experiments based on search
  const filteredExperiments = useMemo(() => {
    if (!searchQuery) return experiments;
    const query = searchQuery.toLowerCase();
    return experiments.filter(
      (exp) =>
        exp.experiment_name.toLowerCase().includes(query) ||
        exp.experiment_id.toLowerCase().includes(query)
    );
  }, [experiments, searchQuery]);

  // Calculate overview metrics
  const overviewMetrics = useMemo(() => {
    const total = experiments.length;
    const healthy = experiments.filter((e) => e.health_status === ExperimentHealthStatus.HEALTHY).length;
    const warning = experiments.filter((e) => e.health_status === ExperimentHealthStatus.WARNING).length;
    const critical = experiments.filter((e) => e.health_status === ExperimentHealthStatus.CRITICAL).length;
    const totalEnrolled = experiments.reduce((sum, e) => sum + e.total_enrolled, 0);
    const avgEnrollmentRate = experiments.reduce((sum, e) => sum + e.enrollment_rate_per_day, 0) / total;
    const srmCount = experiments.filter((e) => e.has_srm).length;
    const totalAlerts = alerts.length;
    const criticalAlerts = alerts.filter((a) => a.severity === AlertSeverity.CRITICAL).length;

    return {
      total,
      healthy,
      warning,
      critical,
      totalEnrolled,
      avgEnrollmentRate: avgEnrollmentRate.toFixed(1),
      srmCount,
      totalAlerts,
      criticalAlerts,
    };
  }, [experiments, alerts]);

  // Enrollment trend data
  const enrollmentTrendData = useMemo(() => {
    return [
      { week: 'W1', enrolled: 450, target: 500 },
      { week: 'W2', enrolled: 920, target: 1000 },
      { week: 'W3', enrolled: 1380, target: 1500 },
      { week: 'W4', enrolled: 1820, target: 2000 },
      { week: 'W5', enrolled: 2180, target: 2500 },
      { week: 'W6', enrolled: 2520, target: 3000 },
    ];
  }, []);

  // Health distribution data
  const healthDistributionData = useMemo(() => {
    return [
      { name: 'Healthy', value: overviewMetrics.healthy, color: COLORS.healthy },
      { name: 'Warning', value: overviewMetrics.warning, color: COLORS.warning },
      { name: 'Critical', value: overviewMetrics.critical, color: COLORS.critical },
    ].filter((d) => d.value > 0);
  }, [overviewMetrics]);

  const handleRunMonitoring = async () => {
    try {
      await refetchMonitor();
    } catch (error) {
      console.error('Failed to run monitoring:', error);
    }
  };

  return (
    <div className="container mx-auto px-4 py-8">
      {/* Header */}
      <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4 mb-8">
        <div>
          <h1 className="text-3xl font-bold text-foreground flex items-center gap-2">
            <Beaker className="h-8 w-8" />
            A/B Testing & Experiments
          </h1>
          <p className="text-muted-foreground mt-1">
            Monitor experiment health, enrollment, and statistical analysis
          </p>
        </div>
        <div className="flex gap-2">
          <Button onClick={handleRunMonitoring} disabled={isLoadingMonitor}>
            <Play className="mr-2 h-4 w-4" />
            Run Monitoring
          </Button>
          <Button variant="outline" onClick={() => refetchMonitor()}>
            <RefreshCw className={`mr-2 h-4 w-4 ${isLoadingMonitor ? 'animate-spin' : ''}`} />
            Refresh
          </Button>
        </div>
      </div>

      {/* Critical Alerts Banner */}
      {overviewMetrics.criticalAlerts > 0 && (
        <Alert variant="destructive" className="mb-6">
          <XCircle className="h-4 w-4" />
          <AlertTitle>Critical Alerts</AlertTitle>
          <AlertDescription>
            {overviewMetrics.criticalAlerts} experiment(s) require immediate attention.
            {alerts
              .filter((a) => a.severity === AlertSeverity.CRITICAL)
              .slice(0, 2)
              .map((a) => (
                <div key={a.alert_id} className="mt-1 text-sm">
                  <strong>{a.experiment_name}:</strong> {a.message}
                </div>
              ))}
          </AlertDescription>
        </Alert>
      )}

      {/* Overview Metrics */}
      <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-4 mb-8">
        <KPICard
          title="Active Experiments"
          value={overviewMetrics.total}
          description="Currently running"
        />
        <KPICard
          title="Healthy"
          value={overviewMetrics.healthy}
          status="healthy"
          description="Passing all checks"
        />
        <KPICard
          title="Total Enrolled"
          value={overviewMetrics.totalEnrolled.toLocaleString()}
          description="Participants enrolled"
        />
        <KPICard
          title="Avg Enrollment/Day"
          value={overviewMetrics.avgEnrollmentRate}
          description="Daily enrollment rate"
        />
        <KPICard
          title="SRM Detected"
          value={overviewMetrics.srmCount}
          status={overviewMetrics.srmCount > 0 ? 'critical' : 'healthy'}
          description="Sample ratio mismatch"
        />
        <KPICard
          title="Active Alerts"
          value={overviewMetrics.totalAlerts}
          status={overviewMetrics.criticalAlerts > 0 ? 'critical' : 'warning'}
          description="Requiring attention"
        />
      </div>

      {/* Main Content Tabs */}
      <Tabs defaultValue="experiments" className="space-y-6">
        <TabsList className="grid w-full grid-cols-4">
          <TabsTrigger value="experiments">Experiments</TabsTrigger>
          <TabsTrigger value="alerts">Alerts ({alerts.length})</TabsTrigger>
          <TabsTrigger value="analytics">Analytics</TabsTrigger>
          <TabsTrigger value="fidelity">Digital Twin</TabsTrigger>
        </TabsList>

        {/* Experiments Tab */}
        <TabsContent value="experiments" className="space-y-6">
          {/* Search */}
          <div className="flex gap-4">
            <Input
              placeholder="Search experiments..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="max-w-md"
            />
          </div>

          {/* Experiment Cards */}
          <div className="grid gap-4">
            {filteredExperiments.map((experiment) => (
              <Card
                key={experiment.experiment_id}
                className={`cursor-pointer transition-all hover:shadow-md ${
                  selectedExperiment === experiment.experiment_id ? 'ring-2 ring-primary' : ''
                }`}
                onClick={() =>
                  setSelectedExperiment(
                    selectedExperiment === experiment.experiment_id ? null : experiment.experiment_id
                  )
                }
              >
                <CardHeader className="pb-2">
                  <div className="flex justify-between items-start">
                    <div className="flex items-center gap-2">
                      {getHealthIcon(experiment.health_status)}
                      <CardTitle className="text-lg">{experiment.experiment_name}</CardTitle>
                    </div>
                    {getHealthBadge(experiment.health_status)}
                  </div>
                  <CardDescription>{experiment.experiment_id}</CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-4 text-sm">
                    <div>
                      <span className="text-muted-foreground">Enrolled</span>
                      <p className="font-semibold">{experiment.total_enrolled.toLocaleString()}</p>
                    </div>
                    <div>
                      <span className="text-muted-foreground">Rate/Day</span>
                      <p className="font-semibold">{experiment.enrollment_rate_per_day.toFixed(1)}</p>
                    </div>
                    <div>
                      <span className="text-muted-foreground">Info Fraction</span>
                      <p className="font-semibold">
                        {(experiment.current_information_fraction * 100).toFixed(0)}%
                      </p>
                    </div>
                    <div>
                      <span className="text-muted-foreground">SRM</span>
                      <p className={`font-semibold ${experiment.has_srm ? 'text-red-600' : 'text-green-600'}`}>
                        {experiment.has_srm ? 'Detected' : 'None'}
                      </p>
                    </div>
                    <div>
                      <span className="text-muted-foreground">Alerts</span>
                      <p className={`font-semibold ${experiment.active_alerts > 0 ? 'text-yellow-600' : ''}`}>
                        {experiment.active_alerts}
                      </p>
                    </div>
                    <div>
                      <span className="text-muted-foreground">Last Check</span>
                      <p className="font-semibold">{formatTimeAgo(experiment.last_checked)}</p>
                    </div>
                  </div>

                  {/* Variant Breakdown Progress Bar */}
                  <div className="mt-4">
                    <div className="flex justify-between text-xs text-muted-foreground mb-1">
                      <span>Variant Distribution</span>
                      <span>
                        {Object.entries(experiment.variant_breakdown)
                          .map(([k, v]) => `${k}: ${v}`)
                          .join(' | ')}
                      </span>
                    </div>
                    <div className="flex h-2 rounded-full overflow-hidden">
                      {Object.entries(experiment.variant_breakdown).map(([variant, countValue], idx) => {
                        const count = countValue as number;
                        const total = (Object.values(experiment.variant_breakdown) as number[]).reduce((a, b) => a + b, 0);
                        const percent = (count / total) * 100;
                        return (
                          <div
                            key={variant}
                            style={{
                              width: `${percent}%`,
                              backgroundColor: PIE_COLORS[idx % PIE_COLORS.length],
                            }}
                          />
                        );
                      })}
                    </div>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        </TabsContent>

        {/* Alerts Tab */}
        <TabsContent value="alerts" className="space-y-4">
          {alerts.length === 0 ? (
            <Card>
              <CardContent className="py-12 text-center">
                <CheckCircle className="h-12 w-12 text-green-500 mx-auto mb-4" />
                <h3 className="text-lg font-semibold">No Active Alerts</h3>
                <p className="text-muted-foreground">All experiments are running smoothly.</p>
              </CardContent>
            </Card>
          ) : (
            alerts.map((alert: MonitorAlert) => (
              <Card
                key={alert.alert_id}
                className={`border-l-4 ${
                  alert.severity === AlertSeverity.CRITICAL
                    ? 'border-l-red-500'
                    : alert.severity === AlertSeverity.WARNING
                    ? 'border-l-yellow-500'
                    : 'border-l-blue-500'
                }`}
              >
                <CardHeader className="pb-2">
                  <div className="flex justify-between items-start">
                    <div className="flex items-center gap-2">
                      {getSeverityIcon(alert.severity)}
                      <CardTitle className="text-base">{alert.message}</CardTitle>
                    </div>
                    <Badge
                      variant={
                        alert.severity === AlertSeverity.CRITICAL
                          ? 'destructive'
                          : alert.severity === AlertSeverity.WARNING
                          ? 'secondary'
                          : 'default'
                      }
                    >
                      {alert.severity}
                    </Badge>
                  </div>
                  <CardDescription>
                    {alert.experiment_name} &middot; {formatTimeAgo(alert.timestamp)}
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="space-y-2">
                    <div className="flex items-center gap-2 text-sm">
                      <Target className="h-4 w-4 text-muted-foreground" />
                      <span className="font-medium">Recommended Action:</span>
                      <span className="text-muted-foreground">{alert.recommended_action}</span>
                    </div>
                    {Object.keys(alert.details).length > 0 && (
                      <div className="bg-muted/50 rounded p-2 text-xs font-mono">
                        {JSON.stringify(alert.details, null, 2)}
                      </div>
                    )}
                  </div>
                </CardContent>
              </Card>
            ))
          )}
        </TabsContent>

        {/* Analytics Tab */}
        <TabsContent value="analytics" className="space-y-6">
          <div className="grid md:grid-cols-2 gap-6">
            {/* Health Distribution */}
            <Card>
              <CardHeader>
                <CardTitle>Experiment Health Distribution</CardTitle>
                <CardDescription>Current status of all active experiments</CardDescription>
              </CardHeader>
              <CardContent>
                <ResponsiveContainer width="100%" height={250}>
                  <PieChart>
                    <Pie
                      data={healthDistributionData}
                      cx="50%"
                      cy="50%"
                      innerRadius={60}
                      outerRadius={90}
                      paddingAngle={5}
                      dataKey="value"
                      label={({ name, value }) => `${name}: ${value}`}
                    >
                      {healthDistributionData.map((entry, index) => (
                        <Cell key={`cell-${index}`} fill={entry.color} />
                      ))}
                    </Pie>
                    <Tooltip />
                  </PieChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>

            {/* Enrollment Trend */}
            <Card>
              <CardHeader>
                <CardTitle>Enrollment Progress</CardTitle>
                <CardDescription>Weekly enrollment vs. target</CardDescription>
              </CardHeader>
              <CardContent>
                <ResponsiveContainer width="100%" height={250}>
                  <BarChart data={enrollmentTrendData}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="week" />
                    <YAxis />
                    <Tooltip />
                    <Legend />
                    <Bar dataKey="enrolled" fill={COLORS.primary} name="Enrolled" />
                    <Bar dataKey="target" fill={COLORS.secondary} name="Target" opacity={0.5} />
                  </BarChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>
          </div>

          {/* Interim Analyses */}
          <Card>
            <CardHeader>
              <CardTitle>Recent Interim Analyses</CardTitle>
              <CardDescription>Statistical stopping decisions with alpha spending</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="border-b">
                      <th className="text-left py-2 px-4">Experiment</th>
                      <th className="text-left py-2 px-4">Analysis #</th>
                      <th className="text-left py-2 px-4">Info Fraction</th>
                      <th className="text-left py-2 px-4">P-Value</th>
                      <th className="text-left py-2 px-4">Adj. Alpha</th>
                      <th className="text-left py-2 px-4">Decision</th>
                      <th className="text-left py-2 px-4">Date</th>
                    </tr>
                  </thead>
                  <tbody>
                    {SAMPLE_INTERIM_ANALYSES.map((analysis) => (
                      <tr key={analysis.analysis_id} className="border-b hover:bg-muted/50">
                        <td className="py-2 px-4">{analysis.experiment_id}</td>
                        <td className="py-2 px-4">{analysis.analysis_number}</td>
                        <td className="py-2 px-4">{(analysis.information_fraction * 100).toFixed(0)}%</td>
                        <td className="py-2 px-4 font-mono">{analysis.p_value.toFixed(4)}</td>
                        <td className="py-2 px-4 font-mono">{analysis.adjusted_alpha.toFixed(4)}</td>
                        <td className="py-2 px-4">
                          <Badge
                            variant={
                              analysis.decision === StoppingDecision.CONTINUE
                                ? 'secondary'
                                : analysis.decision === StoppingDecision.STOP_EFFICACY
                                ? 'default'
                                : 'destructive'
                            }
                          >
                            {analysis.decision}
                          </Badge>
                        </td>
                        <td className="py-2 px-4 text-muted-foreground">
                          {formatTimestamp(analysis.performed_at)}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        {/* Digital Twin Fidelity Tab */}
        <TabsContent value="fidelity" className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Shield className="h-5 w-5" />
                Digital Twin Fidelity Tracking
              </CardTitle>
              <CardDescription>
                Comparing Digital Twin predictions with actual experiment outcomes
              </CardDescription>
            </CardHeader>
            <CardContent>
              <ResponsiveContainer width="100%" height={300}>
                <LineChart data={SAMPLE_FIDELITY}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="date" />
                  <YAxis domain={[0.5, 1]} tickFormatter={(v) => `${(v * 100).toFixed(0)}%`} />
                  <Tooltip formatter={(value) => `${((value as number) * 100).toFixed(1)}%`} />
                  <Legend />
                  <Line
                    type="monotone"
                    dataKey="score"
                    stroke={COLORS.primary}
                    strokeWidth={2}
                    name="Fidelity Score"
                    dot={{ r: 4 }}
                  />
                  {/* Threshold line */}
                  <Line
                    type="monotone"
                    dataKey={() => 0.85}
                    stroke={COLORS.warning}
                    strokeDasharray="5 5"
                    name="Threshold"
                    dot={false}
                  />
                </LineChart>
              </ResponsiveContainer>

              <div className="mt-6 grid md:grid-cols-3 gap-4">
                <Card className="bg-muted/50">
                  <CardContent className="pt-4">
                    <div className="text-center">
                      <div className="text-2xl font-bold">92%</div>
                      <div className="text-sm text-muted-foreground">Initial Fidelity</div>
                    </div>
                  </CardContent>
                </Card>
                <Card className="bg-muted/50">
                  <CardContent className="pt-4">
                    <div className="text-center">
                      <div className="text-2xl font-bold text-yellow-600">74%</div>
                      <div className="text-sm text-muted-foreground">Current Fidelity</div>
                    </div>
                  </CardContent>
                </Card>
                <Card className="bg-muted/50">
                  <CardContent className="pt-4">
                    <div className="text-center">
                      <div className="text-2xl font-bold text-red-600">-18%</div>
                      <div className="text-sm text-muted-foreground">Change</div>
                    </div>
                  </CardContent>
                </Card>
              </div>

              <Alert className="mt-6">
                <AlertTriangle className="h-4 w-4" />
                <AlertTitle>Calibration Recommended</AlertTitle>
                <AlertDescription>
                  Digital Twin fidelity has dropped below the 85% threshold. Consider updating the model
                  parameters based on the latest experiment observations to improve prediction accuracy.
                </AlertDescription>
              </Alert>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
}
