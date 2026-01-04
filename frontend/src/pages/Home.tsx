/**
 * Home Page - KPI Executive Dashboard
 * ====================================
 *
 * Main landing page for E2I Causal Analytics.
 * Displays key performance indicators, agent insights, and quick actions.
 *
 * Features:
 * - KPI dashboard with 46+ metrics organized by category
 * - Brand selector (Remibrutinib, Fabhalta, Kisqali)
 * - Recent agent insights feed
 * - System health summary
 * - Quick action navigation
 *
 * @module pages/Home
 */

import { useState, useMemo } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import {
  TrendingUp,
  TrendingDown,
  Activity,
  Users,
  Target,
  DollarSign,
  BarChart3,
  Brain,
  Zap,
  AlertCircle,
  CheckCircle2,
  Clock,
  ArrowRight,
  Pill,
  MapPin,
  CalendarDays,
  Sparkles,
  RefreshCw,
  ExternalLink,
} from 'lucide-react';
import { cn } from '@/lib/utils';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Progress } from '@/components/ui/progress';
import {
  KPICard,
  StatusBadge,
} from '@/components/visualizations/dashboard';
import { getNavigationRoutes } from '@/router/routes';

// =============================================================================
// TYPES
// =============================================================================

type Brand = 'All' | 'Remibrutinib' | 'Fabhalta' | 'Kisqali';

interface KPIMetric {
  id: string;
  name: string;
  category: string;
  value: number;
  previousValue?: number;
  target?: number;
  unit?: string;
  prefix?: string;
  description: string;
  trend: 'up' | 'down' | 'stable';
  status: 'healthy' | 'warning' | 'critical' | 'neutral';
  sparkline?: number[];
}

interface AgentInsight {
  id: string;
  agentName: string;
  agentTier: number;
  type: 'recommendation' | 'alert' | 'opportunity' | 'insight';
  title: string;
  summary: string;
  impact: 'high' | 'medium' | 'low';
  timestamp: string;
  actionable: boolean;
  relatedKPIs: string[];
}

interface SystemStatus {
  service: string;
  status: 'healthy' | 'warning' | 'error' | 'loading';
  latency?: number;
  lastCheck: string;
}

// =============================================================================
// SAMPLE DATA
// =============================================================================

const BRANDS: { value: Brand; label: string; indication: string; color: string }[] = [
  { value: 'All', label: 'All Brands', indication: 'Combined Portfolio', color: 'bg-slate-500' },
  { value: 'Remibrutinib', label: 'Remibrutinib', indication: 'CSU', color: 'bg-blue-500' },
  { value: 'Fabhalta', label: 'Fabhalta', indication: 'PNH', color: 'bg-purple-500' },
  { value: 'Kisqali', label: 'Kisqali', indication: 'HR+/HER2- BC', color: 'bg-rose-500' },
];

const KPI_CATEGORIES = [
  { id: 'commercial', label: 'Commercial', icon: DollarSign },
  { id: 'hcp', label: 'HCP Engagement', icon: Users },
  { id: 'patient', label: 'Patient Journey', icon: Activity },
  { id: 'market', label: 'Market Share', icon: Target },
  { id: 'causal', label: 'Causal Metrics', icon: Brain },
];

const SAMPLE_KPIS: Record<Brand, KPIMetric[]> = {
  All: [
    { id: 'trx_total', name: 'Total TRx', category: 'commercial', value: 125430, previousValue: 118250, target: 130000, description: 'Total prescriptions across all brands', trend: 'up', status: 'healthy', sparkline: [100, 108, 112, 115, 118, 122, 125] },
    { id: 'nrx_total', name: 'New TRx', category: 'commercial', value: 28540, previousValue: 26890, target: 30000, description: 'New prescriptions this period', trend: 'up', status: 'healthy', sparkline: [22, 24, 25, 26, 27, 28, 28.5] },
    { id: 'revenue', name: 'Net Revenue', category: 'commercial', value: 425000000, previousValue: 398000000, target: 450000000, prefix: '$', description: 'Net revenue across portfolio', trend: 'up', status: 'healthy', sparkline: [350, 370, 385, 398, 410, 420, 425] },
    { id: 'market_share', name: 'Market Share', category: 'market', value: 28.5, previousValue: 26.8, target: 32, unit: '%', description: 'Combined market share', trend: 'up', status: 'healthy', sparkline: [24, 25, 26, 26.5, 27, 28, 28.5] },
    { id: 'hcp_reach', name: 'HCP Reach', category: 'hcp', value: 12450, previousValue: 11800, target: 15000, description: 'HCPs engaged this quarter', trend: 'up', status: 'warning', sparkline: [10, 10.5, 11, 11.5, 12, 12.2, 12.4] },
    { id: 'conversion_rate', name: 'Conversion Rate', category: 'hcp', value: 18.5, previousValue: 17.2, target: 22, unit: '%', description: 'HCP to prescription conversion', trend: 'up', status: 'warning', sparkline: [15, 16, 16.5, 17, 17.5, 18, 18.5] },
    { id: 'patient_starts', name: 'Patient Starts', category: 'patient', value: 8920, previousValue: 8340, target: 10000, description: 'New patient starts', trend: 'up', status: 'healthy', sparkline: [7, 7.5, 8, 8.2, 8.5, 8.8, 8.9] },
    { id: 'adherence', name: 'Adherence Rate', category: 'patient', value: 78.5, previousValue: 76.2, target: 85, unit: '%', description: 'Patient medication adherence', trend: 'up', status: 'warning', sparkline: [72, 73, 74, 75, 76, 77, 78.5] },
    { id: 'ate_trx', name: 'ATE on TRx', category: 'causal', value: 0.156, previousValue: 0.142, target: 0.18, description: 'Average treatment effect on prescriptions', trend: 'up', status: 'healthy', sparkline: [0.12, 0.13, 0.14, 0.145, 0.15, 0.155, 0.156] },
    { id: 'roi', name: 'Campaign ROI', category: 'causal', value: 3.8, previousValue: 3.2, target: 4.5, unit: 'x', description: 'Return on marketing investment', trend: 'up', status: 'healthy', sparkline: [2.8, 3, 3.2, 3.4, 3.5, 3.7, 3.8] },
  ],
  Remibrutinib: [
    { id: 'remi_trx', name: 'TRx', category: 'commercial', value: 45230, previousValue: 42150, target: 50000, description: 'Total Remibrutinib prescriptions', trend: 'up', status: 'healthy', sparkline: [38, 40, 41, 42, 43, 44, 45] },
    { id: 'remi_nrx', name: 'New TRx', category: 'commercial', value: 12340, previousValue: 11200, target: 14000, description: 'New Remibrutinib prescriptions', trend: 'up', status: 'healthy', sparkline: [9, 10, 10.5, 11, 11.5, 12, 12.3] },
    { id: 'remi_share', name: 'CSU Market Share', category: 'market', value: 18.2, previousValue: 15.8, target: 25, unit: '%', description: 'Share in CSU market', trend: 'up', status: 'warning', sparkline: [12, 13, 14, 15, 16, 17, 18.2] },
    { id: 'remi_hcp', name: 'Allergists Reached', category: 'hcp', value: 4520, previousValue: 4100, target: 5500, description: 'Allergists engaged', trend: 'up', status: 'warning', sparkline: [3.5, 3.8, 4, 4.1, 4.2, 4.4, 4.5] },
    { id: 'remi_ate', name: 'ATE (HCP Visit)', category: 'causal', value: 0.182, previousValue: 0.165, target: 0.20, description: 'Effect of HCP visits on TRx', trend: 'up', status: 'healthy', sparkline: [0.14, 0.15, 0.16, 0.165, 0.17, 0.18, 0.182] },
  ],
  Fabhalta: [
    { id: 'fab_trx', name: 'TRx', category: 'commercial', value: 28450, previousValue: 26800, target: 32000, description: 'Total Fabhalta prescriptions', trend: 'up', status: 'healthy', sparkline: [24, 25, 26, 26.5, 27, 28, 28.4] },
    { id: 'fab_nrx', name: 'New TRx', category: 'commercial', value: 6890, previousValue: 6200, target: 8000, description: 'New Fabhalta prescriptions', trend: 'up', status: 'warning', sparkline: [5, 5.5, 6, 6.2, 6.4, 6.7, 6.9] },
    { id: 'fab_share', name: 'PNH Market Share', category: 'market', value: 22.5, previousValue: 20.1, target: 28, unit: '%', description: 'Share in PNH market', trend: 'up', status: 'healthy', sparkline: [16, 17, 18, 19, 20, 21, 22.5] },
    { id: 'fab_hcp', name: 'Hematologists Reached', category: 'hcp', value: 2890, previousValue: 2650, target: 3500, description: 'Hematologists engaged', trend: 'up', status: 'warning', sparkline: [2.2, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9] },
    { id: 'fab_ate', name: 'ATE (Speaker Program)', category: 'causal', value: 0.245, previousValue: 0.218, target: 0.28, description: 'Effect of speaker programs', trend: 'up', status: 'healthy', sparkline: [0.18, 0.19, 0.20, 0.21, 0.22, 0.24, 0.245] },
  ],
  Kisqali: [
    { id: 'kis_trx', name: 'TRx', category: 'commercial', value: 51750, previousValue: 49300, target: 55000, description: 'Total Kisqali prescriptions', trend: 'up', status: 'healthy', sparkline: [45, 46, 47, 48, 49, 50, 51.7] },
    { id: 'kis_nrx', name: 'New TRx', category: 'commercial', value: 9310, previousValue: 9490, target: 10000, description: 'New Kisqali prescriptions', trend: 'down', status: 'warning', sparkline: [8.5, 9, 9.5, 9.6, 9.5, 9.4, 9.3] },
    { id: 'kis_share', name: 'CDK4/6 Market Share', category: 'market', value: 38.2, previousValue: 37.5, target: 42, unit: '%', description: 'Share in CDK4/6 market', trend: 'up', status: 'healthy', sparkline: [35, 36, 36.5, 37, 37.5, 38, 38.2] },
    { id: 'kis_hcp', name: 'Oncologists Reached', category: 'hcp', value: 5040, previousValue: 5050, target: 6000, description: 'Oncologists engaged', trend: 'stable', status: 'warning', sparkline: [4.8, 4.9, 5, 5.05, 5.02, 5.04, 5.04] },
    { id: 'kis_ate', name: 'ATE (Digital)', category: 'causal', value: 0.128, previousValue: 0.135, target: 0.16, description: 'Effect of digital campaigns', trend: 'down', status: 'warning', sparkline: [0.14, 0.14, 0.135, 0.13, 0.128, 0.128, 0.128] },
  ],
};

const SAMPLE_INSIGHTS: AgentInsight[] = [
  {
    id: '1',
    agentName: 'Gap Analyzer',
    agentTier: 2,
    type: 'opportunity',
    title: 'High-Value Territory Opportunity',
    summary: 'Northeast region shows 23% untapped potential in CSU market. Recommended: Increase Remibrutinib HCP engagement by 40%.',
    impact: 'high',
    timestamp: '2 hours ago',
    actionable: true,
    relatedKPIs: ['remi_share', 'remi_hcp'],
  },
  {
    id: '2',
    agentName: 'Drift Monitor',
    agentTier: 3,
    type: 'alert',
    title: 'Model Performance Drift Detected',
    summary: 'Kisqali conversion model showing 8% accuracy degradation. Retraining recommended within 7 days.',
    impact: 'medium',
    timestamp: '5 hours ago',
    actionable: true,
    relatedKPIs: ['kis_nrx', 'conversion_rate'],
  },
  {
    id: '3',
    agentName: 'Causal Impact',
    agentTier: 2,
    type: 'insight',
    title: 'Speaker Programs Outperforming Digital',
    summary: 'Fabhalta speaker programs show 2.1x higher ATE than digital campaigns. Consider budget reallocation.',
    impact: 'high',
    timestamp: '1 day ago',
    actionable: true,
    relatedKPIs: ['fab_ate', 'fab_share'],
  },
  {
    id: '4',
    agentName: 'Health Score',
    agentTier: 3,
    type: 'recommendation',
    title: 'Patient Adherence Improvement',
    summary: 'Implementing 3-month refill reminders could increase adherence by 12% based on causal analysis.',
    impact: 'medium',
    timestamp: '1 day ago',
    actionable: true,
    relatedKPIs: ['adherence', 'patient_starts'],
  },
];

const SYSTEM_STATUS: SystemStatus[] = [
  { service: 'API Gateway', status: 'healthy', latency: 45, lastCheck: '1m ago' },
  { service: 'PostgreSQL', status: 'healthy', latency: 12, lastCheck: '1m ago' },
  { service: 'FalkorDB', status: 'healthy', latency: 28, lastCheck: '1m ago' },
  { service: 'BentoML', status: 'healthy', latency: 156, lastCheck: '1m ago' },
  { service: 'Celery Workers', status: 'warning', latency: 320, lastCheck: '2m ago' },
  { service: 'Redis Cache', status: 'healthy', latency: 8, lastCheck: '1m ago' },
];

const AGENT_TIER_STATS = [
  { tier: 0, name: 'ML Foundation', agents: 7, active: 5, color: 'bg-slate-500' },
  { tier: 1, name: 'Orchestration', agents: 2, active: 2, color: 'bg-purple-500' },
  { tier: 2, name: 'Causal Analytics', agents: 3, active: 3, color: 'bg-blue-500' },
  { tier: 3, name: 'Monitoring', agents: 3, active: 3, color: 'bg-amber-500' },
  { tier: 4, name: 'ML Predictions', agents: 2, active: 1, color: 'bg-emerald-500' },
  { tier: 5, name: 'Self-Improvement', agents: 2, active: 1, color: 'bg-rose-500' },
];

// =============================================================================
// HELPER FUNCTIONS
// =============================================================================

function getInsightIcon(type: AgentInsight['type']) {
  switch (type) {
    case 'opportunity':
      return <Target className="h-4 w-4 text-emerald-500" />;
    case 'alert':
      return <AlertCircle className="h-4 w-4 text-amber-500" />;
    case 'recommendation':
      return <Sparkles className="h-4 w-4 text-blue-500" />;
    case 'insight':
      return <Brain className="h-4 w-4 text-purple-500" />;
    default:
      return <Zap className="h-4 w-4" />;
  }
}

function getImpactBadge(impact: AgentInsight['impact']) {
  const colors = {
    high: 'bg-rose-100 text-rose-700 dark:bg-rose-900/30 dark:text-rose-400',
    medium: 'bg-amber-100 text-amber-700 dark:bg-amber-900/30 dark:text-amber-400',
    low: 'bg-slate-100 text-slate-700 dark:bg-slate-900/30 dark:text-slate-400',
  };
  return <Badge className={cn('text-xs', colors[impact])}>{impact}</Badge>;
}

// =============================================================================
// COMPONENT
// =============================================================================

// Quick Stats Data
const QUICK_STATS = [
  { label: 'Total TRx (MTD)', value: '125,430', change: '+6.1%', isPositive: true },
  { label: 'Active Campaigns', value: '24', change: '+3', isPositive: true },
  { label: 'HCPs Reached', value: '12,450', change: '-2.3%', isPositive: false },
  { label: 'Model Accuracy', value: '94.2%', change: '+0.8%', isPositive: true },
];

// Active Alerts
const ACTIVE_ALERTS = [
  { id: 1, severity: 'critical' as const, title: 'Data Pipeline Delay', message: 'Claims data feed delayed by 4 hours', time: '15 min ago' },
  { id: 2, severity: 'warning' as const, title: 'Model Drift Detected', message: 'Kisqali conversion model requires retraining', time: '2 hours ago' },
  { id: 3, severity: 'info' as const, title: 'New Insights Available', message: '3 new Gap Analyzer recommendations ready', time: '4 hours ago' },
];

function Home() {
  const navigate = useNavigate();
  const [selectedBrand, setSelectedBrand] = useState<Brand>('All');
  const [selectedCategory, setSelectedCategory] = useState('commercial');
  const [isRefreshing, setIsRefreshing] = useState(false);
  const [dismissedAlerts, setDismissedAlerts] = useState<number[]>([]);

  // Get navigation routes for quick actions
  const navRoutes = getNavigationRoutes().filter((route) => route.path !== '/');

  // Filter KPIs by category and brand
  const filteredKPIs = useMemo(() => {
    const brandKPIs = SAMPLE_KPIS[selectedBrand];
    if (selectedCategory === 'all') return brandKPIs;
    return brandKPIs.filter((kpi) => kpi.category === selectedCategory);
  }, [selectedBrand, selectedCategory]);

  // Calculate summary stats
  const summaryStats = useMemo(() => {
    const kpis = SAMPLE_KPIS[selectedBrand];
    const healthyCount = kpis.filter((k) => k.status === 'healthy').length;
    const warningCount = kpis.filter((k) => k.status === 'warning').length;
    const criticalCount = kpis.filter((k) => k.status === 'critical').length;
    return { total: kpis.length, healthy: healthyCount, warning: warningCount, critical: criticalCount };
  }, [selectedBrand]);

  // Filter visible alerts
  const visibleAlerts = useMemo(() =>
    ACTIVE_ALERTS.filter(a => !dismissedAlerts.includes(a.id)),
    [dismissedAlerts]
  );

  // Dismiss alert handler
  const handleDismissAlert = (id: number) => {
    setDismissedAlerts(prev => [...prev, id]);
  };

  // Simulate refresh
  const handleRefresh = () => {
    setIsRefreshing(true);
    setTimeout(() => setIsRefreshing(false), 1500);
  };

  // Get brand color
  const selectedBrandInfo = BRANDS.find((b) => b.value === selectedBrand);

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4">
        <div>
          <h1 className="text-3xl font-bold text-[var(--color-foreground)]">
            E2I Executive Dashboard
          </h1>
          <p className="text-[var(--color-muted-foreground)] mt-1">
            Causal Analytics for Commercial Operations
          </p>
        </div>

        <div className="flex items-center gap-3">
          {/* Brand Selector */}
          <Select value={selectedBrand} onValueChange={(v) => setSelectedBrand(v as Brand)}>
            <SelectTrigger className="w-[200px]">
              <div className="flex items-center gap-2">
                <Pill className="h-4 w-4" />
                <SelectValue placeholder="Select Brand" />
              </div>
            </SelectTrigger>
            <SelectContent>
              {BRANDS.map((brand) => (
                <SelectItem key={brand.value} value={brand.value}>
                  <div className="flex items-center gap-2">
                    <div className={cn('w-2 h-2 rounded-full', brand.color)} />
                    <span>{brand.label}</span>
                    <span className="text-xs text-muted-foreground">({brand.indication})</span>
                  </div>
                </SelectItem>
              ))}
            </SelectContent>
          </Select>

          {/* Refresh Button */}
          <Button variant="outline" size="icon" onClick={handleRefresh} disabled={isRefreshing}>
            <RefreshCw className={cn('h-4 w-4', isRefreshing && 'animate-spin')} />
          </Button>
        </div>
      </div>

      {/* Quick Stats Bar */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        {QUICK_STATS.map((stat, idx) => (
          <div
            key={idx}
            className="bg-[var(--color-card)] border border-[var(--color-border)] rounded-lg p-4 flex items-center justify-between"
          >
            <div>
              <p className="text-xs text-[var(--color-muted-foreground)]">{stat.label}</p>
              <p className="text-xl font-bold text-[var(--color-foreground)]">{stat.value}</p>
            </div>
            <div className={cn(
              'text-sm font-medium flex items-center gap-1',
              stat.isPositive ? 'text-emerald-500' : 'text-rose-500'
            )}>
              {stat.isPositive ? (
                <TrendingUp className="h-4 w-4" />
              ) : (
                <TrendingDown className="h-4 w-4" />
              )}
              {stat.change}
            </div>
          </div>
        ))}
      </div>

      {/* Active Alerts */}
      {visibleAlerts.length > 0 && (
        <div className="space-y-2">
          <h3 className="text-sm font-medium text-[var(--color-muted-foreground)] flex items-center gap-2">
            <AlertCircle className="h-4 w-4" />
            Active Alerts ({visibleAlerts.length})
          </h3>
          <div className="space-y-2">
            {visibleAlerts.map((alert) => (
              <div
                key={alert.id}
                className={cn(
                  'flex items-center justify-between p-3 rounded-lg border',
                  alert.severity === 'critical' && 'bg-rose-50 border-rose-200 dark:bg-rose-900/20 dark:border-rose-800',
                  alert.severity === 'warning' && 'bg-amber-50 border-amber-200 dark:bg-amber-900/20 dark:border-amber-800',
                  alert.severity === 'info' && 'bg-blue-50 border-blue-200 dark:bg-blue-900/20 dark:border-blue-800'
                )}
              >
                <div className="flex items-center gap-3">
                  <div className={cn(
                    'p-1.5 rounded-full',
                    alert.severity === 'critical' && 'bg-rose-500',
                    alert.severity === 'warning' && 'bg-amber-500',
                    alert.severity === 'info' && 'bg-blue-500'
                  )}>
                    <AlertCircle className="h-3 w-3 text-white" />
                  </div>
                  <div>
                    <p className={cn(
                      'font-medium text-sm',
                      alert.severity === 'critical' && 'text-rose-700 dark:text-rose-300',
                      alert.severity === 'warning' && 'text-amber-700 dark:text-amber-300',
                      alert.severity === 'info' && 'text-blue-700 dark:text-blue-300'
                    )}>
                      {alert.title}
                    </p>
                    <p className="text-xs text-[var(--color-muted-foreground)]">
                      {alert.message} • {alert.time}
                    </p>
                  </div>
                </div>
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => handleDismissAlert(alert.id)}
                  className="text-[var(--color-muted-foreground)] hover:text-[var(--color-foreground)]"
                >
                  Dismiss
                </Button>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Brand Context Card */}
      {selectedBrand !== 'All' && (
        <Card className={cn('border-l-4', selectedBrandInfo?.color.replace('bg-', 'border-l-'))}>
          <CardContent className="py-4">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-4">
                <div className={cn('p-3 rounded-lg', selectedBrandInfo?.color, 'bg-opacity-20')}>
                  <Pill className={cn('h-6 w-6', selectedBrandInfo?.color.replace('bg-', 'text-'))} />
                </div>
                <div>
                  <h2 className="text-xl font-semibold">{selectedBrandInfo?.label}</h2>
                  <p className="text-sm text-muted-foreground">{selectedBrandInfo?.indication}</p>
                </div>
              </div>
              <div className="flex items-center gap-6 text-sm">
                <div className="text-center">
                  <div className="font-semibold text-lg">{summaryStats.total}</div>
                  <div className="text-muted-foreground">KPIs</div>
                </div>
                <div className="text-center">
                  <div className="font-semibold text-lg text-emerald-500">{summaryStats.healthy}</div>
                  <div className="text-muted-foreground">On Track</div>
                </div>
                <div className="text-center">
                  <div className="font-semibold text-lg text-amber-500">{summaryStats.warning}</div>
                  <div className="text-muted-foreground">Attention</div>
                </div>
              </div>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Main Content Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* KPI Dashboard - 2 columns */}
        <div className="lg:col-span-2 space-y-4">
          <Card>
            <CardHeader className="pb-2">
              <div className="flex items-center justify-between">
                <div>
                  <CardTitle className="flex items-center gap-2">
                    <BarChart3 className="h-5 w-5" />
                    Key Performance Indicators
                  </CardTitle>
                  <CardDescription>
                    {selectedBrand === 'All' ? 'Portfolio-wide' : selectedBrand} metrics
                  </CardDescription>
                </div>
              </div>
            </CardHeader>
            <CardContent>
              {/* Category Tabs */}
              <Tabs value={selectedCategory} onValueChange={setSelectedCategory} className="space-y-4">
                <TabsList className="flex flex-wrap">
                  {KPI_CATEGORIES.map((cat) => (
                    <TabsTrigger key={cat.id} value={cat.id} className="flex items-center gap-1.5">
                      <cat.icon className="h-3.5 w-3.5" />
                      <span className="hidden sm:inline">{cat.label}</span>
                    </TabsTrigger>
                  ))}
                </TabsList>

                {KPI_CATEGORIES.map((cat) => (
                  <TabsContent key={cat.id} value={cat.id} className="mt-4">
                    <div className="grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-3 gap-4">
                      {filteredKPIs.map((kpi) => (
                        <KPICard
                          key={kpi.id}
                          title={kpi.name}
                          value={kpi.value}
                          unit={kpi.unit}
                          prefix={kpi.prefix}
                          previousValue={kpi.previousValue}
                          target={kpi.target}
                          sparklineData={kpi.sparkline}
                          status={kpi.status}
                          description={kpi.description}
                          higherIsBetter={kpi.trend !== 'down' || kpi.status === 'healthy'}
                          size="sm"
                          onClick={() => navigate('/model-performance')}
                        />
                      ))}
                    </div>
                  </TabsContent>
                ))}
              </Tabs>
            </CardContent>
          </Card>

          {/* Agent Insights Feed */}
          <Card>
            <CardHeader className="pb-2">
              <div className="flex items-center justify-between">
                <div>
                  <CardTitle className="flex items-center gap-2">
                    <Sparkles className="h-5 w-5" />
                    Agent Insights
                  </CardTitle>
                  <CardDescription>Recent recommendations from the 18-agent system</CardDescription>
                </div>
                <Button variant="ghost" size="sm" onClick={() => navigate('/monitoring')}>
                  View All
                  <ArrowRight className="h-4 w-4 ml-1" />
                </Button>
              </div>
            </CardHeader>
            <CardContent>
              <div className="space-y-3">
                {SAMPLE_INSIGHTS.slice(0, 4).map((insight) => (
                  <div
                    key={insight.id}
                    className="flex items-start gap-3 p-3 rounded-lg border bg-[var(--color-card)] hover:bg-muted/50 transition-colors cursor-pointer"
                    onClick={() => navigate('/causal-discovery')}
                  >
                    <div className="mt-0.5">{getInsightIcon(insight.type)}</div>
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center gap-2 mb-1">
                        <span className="font-medium text-sm">{insight.title}</span>
                        {getImpactBadge(insight.impact)}
                      </div>
                      <p className="text-sm text-muted-foreground line-clamp-2">{insight.summary}</p>
                      <div className="flex items-center gap-2 mt-2 text-xs text-muted-foreground">
                        <Badge variant="outline" className="text-xs">
                          Tier {insight.agentTier}: {insight.agentName}
                        </Badge>
                        <span>•</span>
                        <Clock className="h-3 w-3" />
                        <span>{insight.timestamp}</span>
                      </div>
                    </div>
                    {insight.actionable && (
                      <Button variant="outline" size="sm">
                        Act
                      </Button>
                    )}
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Right Sidebar - 1 column */}
        <div className="space-y-4">
          {/* System Health */}
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="flex items-center gap-2 text-base">
                <Activity className="h-4 w-4" />
                System Health
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-2">
                {SYSTEM_STATUS.map((service) => (
                  <div
                    key={service.service}
                    className="flex items-center justify-between py-1.5 border-b last:border-0"
                  >
                    <div className="flex items-center gap-2">
                      <StatusBadge status={service.status} size="sm" showIcon={false} />
                      <span className="text-sm">{service.service}</span>
                    </div>
                    <div className="text-xs text-muted-foreground">
                      {service.latency}ms
                    </div>
                  </div>
                ))}
              </div>
              <Button
                variant="ghost"
                size="sm"
                className="w-full mt-3"
                onClick={() => navigate('/system-health')}
              >
                View Details
                <ArrowRight className="h-4 w-4 ml-1" />
              </Button>
            </CardContent>
          </Card>

          {/* Agent Tier Summary */}
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="flex items-center gap-2 text-base">
                <Brain className="h-4 w-4" />
                Agent Status
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-3">
                {AGENT_TIER_STATS.map((tier) => (
                  <div key={tier.tier} className="space-y-1">
                    <div className="flex items-center justify-between text-sm">
                      <div className="flex items-center gap-2">
                        <div className={cn('w-2 h-2 rounded-full', tier.color)} />
                        <span>Tier {tier.tier}</span>
                      </div>
                      <span className="text-muted-foreground">
                        {tier.active}/{tier.agents} active
                      </span>
                    </div>
                    <Progress value={(tier.active / tier.agents) * 100} className="h-1.5" />
                  </div>
                ))}
              </div>
              <div className="flex items-center justify-center gap-2 mt-4 pt-3 border-t text-sm">
                <CheckCircle2 className="h-4 w-4 text-emerald-500" />
                <span>15/19 agents active</span>
              </div>
            </CardContent>
          </Card>

          {/* Quick Actions */}
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="flex items-center gap-2 text-base">
                <Zap className="h-4 w-4" />
                Quick Actions
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-2 gap-2">
                {navRoutes.slice(0, 6).map((route) => (
                  <Link
                    key={route.path}
                    to={route.path}
                    className="flex items-center gap-2 p-2 rounded-lg border hover:bg-muted/50 transition-colors text-sm"
                  >
                    <ExternalLink className="h-3.5 w-3.5 text-muted-foreground" />
                    <span className="truncate">{route.title}</span>
                  </Link>
                ))}
              </div>
            </CardContent>
          </Card>

          {/* Date Context */}
          <Card>
            <CardContent className="py-4">
              <div className="flex items-center gap-3">
                <CalendarDays className="h-5 w-5 text-muted-foreground" />
                <div>
                  <div className="text-sm font-medium">Reporting Period</div>
                  <div className="text-xs text-muted-foreground">Q4 2025 (Oct - Dec)</div>
                </div>
              </div>
              <div className="flex items-center gap-3 mt-3 pt-3 border-t">
                <MapPin className="h-5 w-5 text-muted-foreground" />
                <div>
                  <div className="text-sm font-medium">Territory</div>
                  <div className="text-xs text-muted-foreground">All US Regions</div>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
}

export default Home;
