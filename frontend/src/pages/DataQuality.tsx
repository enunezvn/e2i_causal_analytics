/**
 * Data Quality Page
 * =================
 *
 * Dashboard for data profiling, completeness metrics, accuracy checks,
 * and validation rule monitoring.
 *
 * @module pages/DataQuality
 */

import { useState, useMemo } from 'react';
import {
  Database,
  RefreshCw,
  Download,
  CheckCircle2,
  XCircle,
  AlertTriangle,
  Clock,
  Search,
  Filter,
  Table as TableIcon,
  BarChart3,
  FileText,
} from 'lucide-react';

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
import {
  KPICard,
  ProgressRing,
  StatusBadge,
  AlertCard,
  AlertList,
} from '@/components/visualizations';

// =============================================================================
// TYPES
// =============================================================================

interface DataSource {
  id: string;
  name: string;
  type: 'database' | 'api' | 'file';
  status: 'healthy' | 'warning' | 'error';
  lastSynced: string;
  rowCount: number;
  completeness: number;
}

interface ValidationRule {
  id: string;
  name: string;
  description: string;
  targetField: string;
  dataSource: string;
  ruleType: 'not_null' | 'range' | 'regex' | 'unique' | 'foreign_key' | 'custom';
  status: 'pass' | 'fail' | 'warning';
  lastChecked: string;
  violationCount: number;
  totalRows: number;
}

interface ColumnProfile {
  name: string;
  dataType: string;
  completeness: number;
  uniqueness: number;
  nullCount: number;
  distinctValues: number;
  sampleValues: string[];
  minValue?: string | number;
  maxValue?: string | number;
  meanValue?: number;
}

interface QualityIssue {
  id: string;
  severity: 'info' | 'warning' | 'error' | 'critical';
  title: string;
  message: string;
  dataSource: string;
  field?: string;
  detectedAt: string;
  affectedRows: number;
}

// =============================================================================
// SAMPLE DATA
// =============================================================================

const SAMPLE_DATA_SOURCES: DataSource[] = [
  {
    id: 'ds-001',
    name: 'HCP Master',
    type: 'database',
    status: 'healthy',
    lastSynced: '2026-01-02T08:30:00Z',
    rowCount: 125420,
    completeness: 98.5,
  },
  {
    id: 'ds-002',
    name: 'Sales Transactions',
    type: 'database',
    status: 'healthy',
    lastSynced: '2026-01-02T08:25:00Z',
    rowCount: 2450000,
    completeness: 99.2,
  },
  {
    id: 'ds-003',
    name: 'Prescriptions (TRx)',
    type: 'api',
    status: 'warning',
    lastSynced: '2026-01-02T07:45:00Z',
    rowCount: 890000,
    completeness: 94.8,
  },
  {
    id: 'ds-004',
    name: 'Territory Mapping',
    type: 'file',
    status: 'healthy',
    lastSynced: '2026-01-02T06:00:00Z',
    rowCount: 5200,
    completeness: 100,
  },
  {
    id: 'ds-005',
    name: 'Market Access Data',
    type: 'api',
    status: 'error',
    lastSynced: '2026-01-01T22:15:00Z',
    rowCount: 45000,
    completeness: 87.3,
  },
];

const SAMPLE_VALIDATION_RULES: ValidationRule[] = [
  {
    id: 'vr-001',
    name: 'HCP ID Not Null',
    description: 'Healthcare provider ID must be present',
    targetField: 'hcp_id',
    dataSource: 'HCP Master',
    ruleType: 'not_null',
    status: 'pass',
    lastChecked: '2026-01-02T08:30:00Z',
    violationCount: 0,
    totalRows: 125420,
  },
  {
    id: 'vr-002',
    name: 'Valid NPI Format',
    description: 'NPI must be 10 digits',
    targetField: 'npi',
    dataSource: 'HCP Master',
    ruleType: 'regex',
    status: 'warning',
    lastChecked: '2026-01-02T08:30:00Z',
    violationCount: 234,
    totalRows: 125420,
  },
  {
    id: 'vr-003',
    name: 'Sales Amount Range',
    description: 'Sales amount between $0 and $10M',
    targetField: 'amount',
    dataSource: 'Sales Transactions',
    ruleType: 'range',
    status: 'pass',
    lastChecked: '2026-01-02T08:25:00Z',
    violationCount: 0,
    totalRows: 2450000,
  },
  {
    id: 'vr-004',
    name: 'Unique Transaction ID',
    description: 'Transaction IDs must be unique',
    targetField: 'transaction_id',
    dataSource: 'Sales Transactions',
    ruleType: 'unique',
    status: 'pass',
    lastChecked: '2026-01-02T08:25:00Z',
    violationCount: 0,
    totalRows: 2450000,
  },
  {
    id: 'vr-005',
    name: 'Valid HCP Reference',
    description: 'HCP ID must exist in HCP Master',
    targetField: 'hcp_id',
    dataSource: 'Prescriptions (TRx)',
    ruleType: 'foreign_key',
    status: 'fail',
    lastChecked: '2026-01-02T07:45:00Z',
    violationCount: 1523,
    totalRows: 890000,
  },
  {
    id: 'vr-006',
    name: 'Prescription Date Valid',
    description: 'Prescription date must not be in future',
    targetField: 'rx_date',
    dataSource: 'Prescriptions (TRx)',
    ruleType: 'custom',
    status: 'pass',
    lastChecked: '2026-01-02T07:45:00Z',
    violationCount: 0,
    totalRows: 890000,
  },
  {
    id: 'vr-007',
    name: 'Territory Code Format',
    description: 'Territory code must match pattern',
    targetField: 'territory_code',
    dataSource: 'Territory Mapping',
    ruleType: 'regex',
    status: 'pass',
    lastChecked: '2026-01-02T06:00:00Z',
    violationCount: 0,
    totalRows: 5200,
  },
  {
    id: 'vr-008',
    name: 'Market Access Date Range',
    description: 'Coverage dates must be valid',
    targetField: 'effective_date',
    dataSource: 'Market Access Data',
    ruleType: 'range',
    status: 'fail',
    lastChecked: '2026-01-01T22:15:00Z',
    violationCount: 892,
    totalRows: 45000,
  },
];

const SAMPLE_COLUMN_PROFILES: ColumnProfile[] = [
  {
    name: 'hcp_id',
    dataType: 'VARCHAR(36)',
    completeness: 100,
    uniqueness: 100,
    nullCount: 0,
    distinctValues: 125420,
    sampleValues: ['HCP-001234', 'HCP-005678', 'HCP-009012'],
  },
  {
    name: 'first_name',
    dataType: 'VARCHAR(100)',
    completeness: 99.8,
    uniqueness: 12.5,
    nullCount: 251,
    distinctValues: 15680,
    sampleValues: ['John', 'Sarah', 'Michael'],
  },
  {
    name: 'last_name',
    dataType: 'VARCHAR(100)',
    completeness: 99.9,
    uniqueness: 28.4,
    nullCount: 125,
    distinctValues: 35640,
    sampleValues: ['Smith', 'Johnson', 'Williams'],
  },
  {
    name: 'npi',
    dataType: 'VARCHAR(10)',
    completeness: 98.5,
    uniqueness: 99.8,
    nullCount: 1882,
    distinctValues: 125170,
    sampleValues: ['1234567890', '9876543210'],
  },
  {
    name: 'specialty',
    dataType: 'VARCHAR(50)',
    completeness: 97.2,
    uniqueness: 0.15,
    nullCount: 3513,
    distinctValues: 185,
    sampleValues: ['Oncology', 'Cardiology', 'Internal Medicine'],
  },
  {
    name: 'trx_volume',
    dataType: 'INTEGER',
    completeness: 95.5,
    uniqueness: 8.2,
    nullCount: 5644,
    distinctValues: 10285,
    sampleValues: ['150', '320', '85'],
    minValue: 0,
    maxValue: 2450,
    meanValue: 187.5,
  },
  {
    name: 'territory_id',
    dataType: 'VARCHAR(20)',
    completeness: 100,
    uniqueness: 0.8,
    nullCount: 0,
    distinctValues: 1024,
    sampleValues: ['TERR-NE-001', 'TERR-SW-042'],
  },
  {
    name: 'created_at',
    dataType: 'TIMESTAMP',
    completeness: 100,
    uniqueness: 98.5,
    nullCount: 0,
    distinctValues: 123540,
    sampleValues: ['2025-01-15 10:30:00', '2025-06-22 14:45:00'],
    minValue: '2020-01-01',
    maxValue: '2026-01-02',
  },
];

const SAMPLE_QUALITY_ISSUES: QualityIssue[] = [
  {
    id: 'qi-001',
    severity: 'critical',
    title: 'Market Access Sync Failed',
    message: 'API connection timeout. Data has not been refreshed in over 10 hours.',
    dataSource: 'Market Access Data',
    detectedAt: '2026-01-02T08:00:00Z',
    affectedRows: 45000,
  },
  {
    id: 'qi-002',
    severity: 'error',
    title: 'Foreign Key Violations Detected',
    message: '1,523 prescription records reference non-existent HCP IDs.',
    dataSource: 'Prescriptions (TRx)',
    field: 'hcp_id',
    detectedAt: '2026-01-02T07:45:00Z',
    affectedRows: 1523,
  },
  {
    id: 'qi-003',
    severity: 'warning',
    title: 'NPI Format Violations',
    message: '234 records have invalid NPI format (expected 10 digits).',
    dataSource: 'HCP Master',
    field: 'npi',
    detectedAt: '2026-01-02T08:30:00Z',
    affectedRows: 234,
  },
  {
    id: 'qi-004',
    severity: 'warning',
    title: 'Missing Specialty Data',
    message: '3,513 HCP records are missing specialty information.',
    dataSource: 'HCP Master',
    field: 'specialty',
    detectedAt: '2026-01-02T08:30:00Z',
    affectedRows: 3513,
  },
  {
    id: 'qi-005',
    severity: 'info',
    title: 'TRx Data Delay',
    message: 'Prescription data is 45 minutes behind real-time.',
    dataSource: 'Prescriptions (TRx)',
    detectedAt: '2026-01-02T08:15:00Z',
    affectedRows: 0,
  },
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

function getStatusFromScore(score: number): 'healthy' | 'warning' | 'critical' {
  if (score >= 95) return 'healthy';
  if (score >= 85) return 'warning';
  return 'critical';
}

function getRuleStatusIcon(status: 'pass' | 'fail' | 'warning') {
  switch (status) {
    case 'pass':
      return <CheckCircle2 className="h-4 w-4 text-emerald-500" />;
    case 'fail':
      return <XCircle className="h-4 w-4 text-rose-500" />;
    case 'warning':
      return <AlertTriangle className="h-4 w-4 text-amber-500" />;
  }
}

// =============================================================================
// COMPONENT
// =============================================================================

function DataQuality() {
  const [selectedDataSource, setSelectedDataSource] = useState<string>('all');
  const [searchQuery, setSearchQuery] = useState('');
  const [isRefreshing, setIsRefreshing] = useState(false);
  const [ruleStatusFilter, setRuleStatusFilter] = useState<string>('all');

  // Calculate overall quality scores
  const qualityScores = useMemo(() => {
    const completeness = SAMPLE_DATA_SOURCES.reduce((acc, ds) => acc + ds.completeness, 0) / SAMPLE_DATA_SOURCES.length;
    const validationPassRate = (SAMPLE_VALIDATION_RULES.filter((r) => r.status === 'pass').length / SAMPLE_VALIDATION_RULES.length) * 100;
    const healthySourcesRate = (SAMPLE_DATA_SOURCES.filter((ds) => ds.status === 'healthy').length / SAMPLE_DATA_SOURCES.length) * 100;

    return {
      completeness,
      accuracy: validationPassRate,
      consistency: 96.2, // Simulated
      timeliness: healthySourcesRate,
      overall: (completeness + validationPassRate + 96.2 + healthySourcesRate) / 4,
    };
  }, []);

  // Filter validation rules
  const filteredRules = useMemo(() => {
    return SAMPLE_VALIDATION_RULES.filter((rule) => {
      const matchesSource = selectedDataSource === 'all' || rule.dataSource === selectedDataSource;
      const matchesStatus = ruleStatusFilter === 'all' || rule.status === ruleStatusFilter;
      const matchesSearch =
        searchQuery === '' ||
        rule.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
        rule.targetField.toLowerCase().includes(searchQuery.toLowerCase());
      return matchesSource && matchesStatus && matchesSearch;
    });
  }, [selectedDataSource, ruleStatusFilter, searchQuery]);

  // Filter column profiles based on selected data source
  const filteredColumns = useMemo(() => {
    if (searchQuery === '') return SAMPLE_COLUMN_PROFILES;
    return SAMPLE_COLUMN_PROFILES.filter(
      (col) =>
        col.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
        col.dataType.toLowerCase().includes(searchQuery.toLowerCase())
    );
  }, [searchQuery]);

  const handleRefresh = () => {
    setIsRefreshing(true);
    setTimeout(() => setIsRefreshing(false), 2000);
  };

  const handleExport = () => {
    const report = {
      generatedAt: new Date().toISOString(),
      qualityScores,
      dataSources: SAMPLE_DATA_SOURCES,
      validationRules: SAMPLE_VALIDATION_RULES,
      issues: SAMPLE_QUALITY_ISSUES,
    };
    const blob = new Blob([JSON.stringify(report, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `data-quality-report-${new Date().toISOString().split('T')[0]}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="container mx-auto px-4 py-8">
      {/* Header */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 mb-8">
        <div>
          <h1 className="text-3xl font-bold mb-2 flex items-center gap-3">
            <Database className="h-8 w-8" />
            Data Quality
          </h1>
          <p className="text-muted-foreground">
            Data profiling, completeness metrics, accuracy checks, and validation rules.
          </p>
        </div>
        <div className="flex items-center gap-2">
          <Button variant="outline" onClick={handleRefresh} disabled={isRefreshing}>
            <RefreshCw className={`h-4 w-4 mr-2 ${isRefreshing ? 'animate-spin' : ''}`} />
            Refresh
          </Button>
          <Button variant="outline" onClick={handleExport}>
            <Download className="h-4 w-4 mr-2" />
            Export Report
          </Button>
        </div>
      </div>

      {/* Quality Score Overview */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-5 gap-4 mb-8">
        <KPICard
          title="Overall Quality"
          value={qualityScores.overall}
          unit="%"
          status={getStatusFromScore(qualityScores.overall)}
          description="Composite score of all quality dimensions"
          sparklineData={[92, 93, 94, 93, 95, 94, 96, 95, 95, qualityScores.overall]}
          higherIsBetter
        />
        <KPICard
          title="Completeness"
          value={qualityScores.completeness}
          unit="%"
          status={getStatusFromScore(qualityScores.completeness)}
          description="Percentage of non-null values across all fields"
          sparklineData={[96, 97, 97, 98, 97, 98, 98, 99, 98, qualityScores.completeness]}
          higherIsBetter
        />
        <KPICard
          title="Accuracy"
          value={qualityScores.accuracy}
          unit="%"
          status={getStatusFromScore(qualityScores.accuracy)}
          description="Validation rule pass rate"
          sparklineData={[88, 90, 89, 91, 92, 91, 90, 92, 91, qualityScores.accuracy]}
          higherIsBetter
        />
        <KPICard
          title="Consistency"
          value={qualityScores.consistency}
          unit="%"
          status={getStatusFromScore(qualityScores.consistency)}
          description="Cross-source data consistency"
          sparklineData={[94, 95, 95, 96, 95, 96, 96, 97, 96, qualityScores.consistency]}
          higherIsBetter
        />
        <KPICard
          title="Timeliness"
          value={qualityScores.timeliness}
          unit="%"
          status={getStatusFromScore(qualityScores.timeliness)}
          description="Data freshness and sync status"
          sparklineData={[85, 88, 90, 88, 92, 90, 85, 88, 90, qualityScores.timeliness]}
          higherIsBetter
        />
      </div>

      {/* Data Sources */}
      <Card className="mb-8">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <TableIcon className="h-5 w-5" />
            Data Sources
          </CardTitle>
          <CardDescription>Health status and completeness of connected data sources</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {SAMPLE_DATA_SOURCES.map((source) => (
              <div
                key={source.id}
                className="p-4 rounded-lg border border-border bg-card hover:shadow-sm transition-shadow"
              >
                <div className="flex items-start justify-between mb-3">
                  <div>
                    <h4 className="font-medium">{source.name}</h4>
                    <p className="text-sm text-muted-foreground capitalize">{source.type}</p>
                  </div>
                  <StatusBadge
                    status={source.status}
                    size="sm"
                  />
                </div>
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-3">
                    <ProgressRing
                      value={source.completeness}
                      size={48}
                      strokeWidth={5}
                      status={getStatusFromScore(source.completeness)}
                    />
                    <div>
                      <p className="text-sm font-medium">{formatNumber(source.rowCount)} rows</p>
                      <p className="text-xs text-muted-foreground flex items-center gap-1">
                        <Clock className="h-3 w-3" />
                        {formatTimestamp(source.lastSynced)}
                      </p>
                    </div>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>

      {/* Tabs for Rules, Profiling, and Issues */}
      <Tabs defaultValue="rules" className="space-y-4">
        <TabsList>
          <TabsTrigger value="rules" className="flex items-center gap-2">
            <CheckCircle2 className="h-4 w-4" />
            Validation Rules
          </TabsTrigger>
          <TabsTrigger value="profiling" className="flex items-center gap-2">
            <BarChart3 className="h-4 w-4" />
            Data Profiling
          </TabsTrigger>
          <TabsTrigger value="issues" className="flex items-center gap-2">
            <AlertTriangle className="h-4 w-4" />
            Quality Issues
            <Badge variant="destructive" className="ml-1">
              {SAMPLE_QUALITY_ISSUES.filter((i) => i.severity === 'critical' || i.severity === 'error').length}
            </Badge>
          </TabsTrigger>
        </TabsList>

        {/* Validation Rules Tab */}
        <TabsContent value="rules">
          <Card>
            <CardHeader>
              <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
                <div>
                  <CardTitle>Validation Rules</CardTitle>
                  <CardDescription>Data quality rules and their current status</CardDescription>
                </div>
                <div className="flex items-center gap-2">
                  <div className="relative">
                    <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-muted-foreground" />
                    <Input
                      placeholder="Search rules..."
                      value={searchQuery}
                      onChange={(e) => setSearchQuery(e.target.value)}
                      className="pl-9 w-64"
                    />
                  </div>
                  <Select value={selectedDataSource} onValueChange={setSelectedDataSource}>
                    <SelectTrigger className="w-48">
                      <SelectValue placeholder="Data Source" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="all">All Sources</SelectItem>
                      {SAMPLE_DATA_SOURCES.map((ds) => (
                        <SelectItem key={ds.id} value={ds.name}>
                          {ds.name}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                  <Select value={ruleStatusFilter} onValueChange={setRuleStatusFilter}>
                    <SelectTrigger className="w-32">
                      <Filter className="h-4 w-4 mr-2" />
                      <SelectValue placeholder="Status" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="all">All</SelectItem>
                      <SelectItem value="pass">Pass</SelectItem>
                      <SelectItem value="warning">Warning</SelectItem>
                      <SelectItem value="fail">Fail</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
              </div>
            </CardHeader>
            <CardContent>
              <div className="overflow-x-auto">
                <table className="w-full">
                  <thead>
                    <tr className="border-b border-border">
                      <th className="text-left py-3 px-4 font-medium text-muted-foreground">Status</th>
                      <th className="text-left py-3 px-4 font-medium text-muted-foreground">Rule Name</th>
                      <th className="text-left py-3 px-4 font-medium text-muted-foreground">Data Source</th>
                      <th className="text-left py-3 px-4 font-medium text-muted-foreground">Target Field</th>
                      <th className="text-left py-3 px-4 font-medium text-muted-foreground">Type</th>
                      <th className="text-right py-3 px-4 font-medium text-muted-foreground">Violations</th>
                      <th className="text-left py-3 px-4 font-medium text-muted-foreground">Last Checked</th>
                    </tr>
                  </thead>
                  <tbody>
                    {filteredRules.map((rule) => (
                      <tr key={rule.id} className="border-b border-border hover:bg-muted/50">
                        <td className="py-3 px-4">{getRuleStatusIcon(rule.status)}</td>
                        <td className="py-3 px-4">
                          <div>
                            <p className="font-medium">{rule.name}</p>
                            <p className="text-xs text-muted-foreground">{rule.description}</p>
                          </div>
                        </td>
                        <td className="py-3 px-4 text-sm">{rule.dataSource}</td>
                        <td className="py-3 px-4">
                          <code className="text-sm bg-muted px-1.5 py-0.5 rounded">{rule.targetField}</code>
                        </td>
                        <td className="py-3 px-4">
                          <Badge variant="outline" className="capitalize">
                            {rule.ruleType.replace('_', ' ')}
                          </Badge>
                        </td>
                        <td className="py-3 px-4 text-right">
                          {rule.violationCount > 0 ? (
                            <span className="text-rose-500 font-medium">
                              {formatNumber(rule.violationCount)}
                            </span>
                          ) : (
                            <span className="text-emerald-500">0</span>
                          )}
                          <span className="text-muted-foreground text-sm"> / {formatNumber(rule.totalRows)}</span>
                        </td>
                        <td className="py-3 px-4 text-sm text-muted-foreground">
                          {formatTimestamp(rule.lastChecked)}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              {filteredRules.length === 0 && (
                <div className="text-center py-8 text-muted-foreground">
                  <FileText className="h-8 w-8 mx-auto mb-2 opacity-50" />
                  <p>No validation rules match your filters</p>
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>

        {/* Data Profiling Tab */}
        <TabsContent value="profiling">
          <Card>
            <CardHeader>
              <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
                <div>
                  <CardTitle>Column Profiling</CardTitle>
                  <CardDescription>Statistical analysis of data columns</CardDescription>
                </div>
                <div className="relative">
                  <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-muted-foreground" />
                  <Input
                    placeholder="Search columns..."
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
                      <th className="text-left py-3 px-4 font-medium text-muted-foreground">Column</th>
                      <th className="text-left py-3 px-4 font-medium text-muted-foreground">Data Type</th>
                      <th className="text-center py-3 px-4 font-medium text-muted-foreground">Completeness</th>
                      <th className="text-center py-3 px-4 font-medium text-muted-foreground">Uniqueness</th>
                      <th className="text-right py-3 px-4 font-medium text-muted-foreground">Nulls</th>
                      <th className="text-right py-3 px-4 font-medium text-muted-foreground">Distinct</th>
                      <th className="text-left py-3 px-4 font-medium text-muted-foreground">Sample Values</th>
                    </tr>
                  </thead>
                  <tbody>
                    {filteredColumns.map((col) => (
                      <tr key={col.name} className="border-b border-border hover:bg-muted/50">
                        <td className="py-3 px-4">
                          <code className="text-sm font-medium">{col.name}</code>
                        </td>
                        <td className="py-3 px-4">
                          <Badge variant="secondary" className="font-mono text-xs">
                            {col.dataType}
                          </Badge>
                        </td>
                        <td className="py-3 px-4">
                          <div className="flex items-center justify-center gap-2">
                            <ProgressRing
                              value={col.completeness}
                              size={32}
                              strokeWidth={3}
                              status={getStatusFromScore(col.completeness)}
                              showLabel={false}
                            />
                            <span className="text-sm">{col.completeness.toFixed(1)}%</span>
                          </div>
                        </td>
                        <td className="py-3 px-4 text-center text-sm">{col.uniqueness.toFixed(1)}%</td>
                        <td className="py-3 px-4 text-right text-sm">
                          {col.nullCount > 0 ? (
                            <span className="text-amber-500">{formatNumber(col.nullCount)}</span>
                          ) : (
                            <span className="text-muted-foreground">0</span>
                          )}
                        </td>
                        <td className="py-3 px-4 text-right text-sm">{formatNumber(col.distinctValues)}</td>
                        <td className="py-3 px-4">
                          <div className="flex gap-1 flex-wrap">
                            {col.sampleValues.slice(0, 2).map((val, i) => (
                              <code key={i} className="text-xs bg-muted px-1.5 py-0.5 rounded">
                                {val}
                              </code>
                            ))}
                            {col.sampleValues.length > 2 && (
                              <span className="text-xs text-muted-foreground">
                                +{col.sampleValues.length - 2} more
                              </span>
                            )}
                          </div>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        {/* Quality Issues Tab */}
        <TabsContent value="issues">
          <Card>
            <CardHeader>
              <CardTitle>Quality Issues</CardTitle>
              <CardDescription>Active data quality alerts and anomalies</CardDescription>
            </CardHeader>
            <CardContent>
              <AlertList
                alerts={SAMPLE_QUALITY_ISSUES.map((issue) => ({
                  severity: issue.severity,
                  title: issue.title,
                  message: issue.message,
                  timestamp: issue.detectedAt,
                  source: issue.dataSource,
                  dismissible: true,
                  actions: [
                    {
                      label: 'Investigate',
                      onClick: () => console.log('Investigating:', issue.id),
                      primary: true,
                    },
                  ],
                }))}
              />
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
}

export default DataQuality;
