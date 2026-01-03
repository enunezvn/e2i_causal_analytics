/**
 * Scenario Results Component
 * ==========================
 *
 * Displays digital twin simulation results including outcomes,
 * projections, and fidelity metrics.
 *
 * @module components/digital-twin/ScenarioResults
 */

import { useMemo } from 'react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  ReferenceLine,
  Area,
  ComposedChart,
} from 'recharts';
import {
  TrendingUp,
  TrendingDown,
  AlertTriangle,
  CheckCircle2,
  Clock,
  Activity,
  Target,
  DollarSign,
  BarChart3,
  ShieldCheck,
} from 'lucide-react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';
import { cn } from '@/lib/utils';
import {
  type SimulationResponse,
  type ConfidenceInterval,
  type FidelityMetrics,
  type SensitivityResult,
  ConfidenceLevel,
} from '@/types/digital-twin';

// =============================================================================
// TYPES
// =============================================================================

export interface ScenarioResultsProps {
  /** Simulation results to display */
  results: SimulationResponse | null;
  /** Whether results are loading */
  isLoading?: boolean;
  /** Additional CSS classes */
  className?: string;
}

// =============================================================================
// HELPER FUNCTIONS
// =============================================================================

function formatDate(dateStr: string): string {
  const date = new Date(dateStr);
  return date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
}

function formatConfidenceInterval(ci: ConfidenceInterval): string {
  return `[${ci.lower.toFixed(1)}, ${ci.upper.toFixed(1)}]`;
}

function getConfidenceBadgeColor(level: ConfidenceLevel): string {
  switch (level) {
    case ConfidenceLevel.HIGH:
      return 'bg-emerald-100 text-emerald-700 dark:bg-emerald-900/30 dark:text-emerald-400';
    case ConfidenceLevel.MEDIUM:
      return 'bg-amber-100 text-amber-700 dark:bg-amber-900/30 dark:text-amber-400';
    case ConfidenceLevel.LOW:
      return 'bg-rose-100 text-rose-700 dark:bg-rose-900/30 dark:text-rose-400';
  }
}

function getFidelityScoreColor(score: number): string {
  if (score >= 0.8) return 'text-emerald-600';
  if (score >= 0.6) return 'text-amber-600';
  return 'text-rose-600';
}

// =============================================================================
// SUB-COMPONENTS
// =============================================================================

interface OutcomeCardProps {
  title: string;
  value: ConfidenceInterval;
  icon: React.ElementType;
  prefix?: string;
  suffix?: string;
  isPercentage?: boolean;
}

function OutcomeCard({ title, value, icon: Icon, prefix = '', suffix = '', isPercentage = false }: OutcomeCardProps) {
  const displayValue = isPercentage ? `${value.estimate.toFixed(1)}%` : value.estimate.toFixed(1);
  const isPositive = value.estimate > 0;

  return (
    <Card>
      <CardContent className="pt-4">
        <div className="flex items-center justify-between mb-2">
          <span className="text-sm text-muted-foreground">{title}</span>
          <Icon className="h-4 w-4 text-muted-foreground" />
        </div>
        <div className="space-y-1">
          <p className={cn(
            'text-2xl font-bold',
            isPositive ? 'text-emerald-600' : 'text-rose-600'
          )}>
            {prefix}{isPositive && '+'}{displayValue}{suffix}
          </p>
          <p className="text-xs text-muted-foreground">
            95% CI: {formatConfidenceInterval(value)}
          </p>
        </div>
      </CardContent>
    </Card>
  );
}

interface FidelityMeterProps {
  label: string;
  value: number;
}

function FidelityMeter({ label, value }: FidelityMeterProps) {
  return (
    <div className="space-y-1">
      <div className="flex justify-between text-sm">
        <span className="text-muted-foreground">{label}</span>
        <span className={cn('font-medium', getFidelityScoreColor(value))}>
          {(value * 100).toFixed(0)}%
        </span>
      </div>
      <Progress value={value * 100} className="h-2" />
    </div>
  );
}

// =============================================================================
// COMPONENT
// =============================================================================

export function ScenarioResults({
  results,
  isLoading = false,
  className = '',
}: ScenarioResultsProps) {
  // Custom tooltip for projections chart
  const ProjectionTooltip = ({ active, payload, label }: { active?: boolean; payload?: Array<Record<string, unknown>>; label?: string }) => {
    if (!active || !payload || !payload.length) return null;

    const dataPoint = payload[0].payload as {
      date: string;
      with_intervention: number;
      without_intervention: number;
      lower_bound: number;
      upper_bound: number;
    };
    const effect = dataPoint.with_intervention - dataPoint.without_intervention;

    return (
      <div className="bg-[var(--color-popover)] border border-[var(--color-border)] rounded-md shadow-lg p-3">
        <p className="font-medium text-[var(--color-foreground)] mb-2">
          {formatDate(label || '')}
        </p>
        <div className="space-y-1 text-sm">
          <div className="flex items-center justify-between gap-4">
            <span className="text-[var(--color-muted-foreground)]">With Intervention:</span>
            <span className="font-medium">{dataPoint.with_intervention.toLocaleString()}</span>
          </div>
          <div className="flex items-center justify-between gap-4">
            <span className="text-[var(--color-muted-foreground)]">Without:</span>
            <span className="font-medium">{dataPoint.without_intervention.toLocaleString()}</span>
          </div>
          <div className="flex items-center justify-between gap-4 pt-1 border-t border-border">
            <span className="text-[var(--color-muted-foreground)]">Expected Effect:</span>
            <span className={cn('font-bold', effect > 0 ? 'text-emerald-600' : 'text-rose-600')}>
              {effect > 0 ? '+' : ''}{effect.toLocaleString()}
            </span>
          </div>
        </div>
      </div>
    );
  };

  if (isLoading) {
    return (
      <Card className={className}>
        <CardContent className="pt-6">
          <div className="flex flex-col items-center justify-center py-12">
            <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary" />
            <p className="mt-4 text-muted-foreground">Running simulation...</p>
            <p className="text-sm text-muted-foreground">This may take a few moments</p>
          </div>
        </CardContent>
      </Card>
    );
  }

  if (!results) {
    return (
      <Card className={className}>
        <CardContent className="pt-6">
          <div className="flex flex-col items-center justify-center py-12 text-center">
            <Activity className="h-12 w-12 text-muted-foreground mb-4" />
            <h3 className="text-lg font-medium mb-2">No Simulation Results</h3>
            <p className="text-muted-foreground max-w-md">
              Configure and run a simulation to see predicted intervention outcomes,
              ROI projections, and recommendations.
            </p>
          </div>
        </CardContent>
      </Card>
    );
  }

  const { outcomes, fidelity, sensitivity, projections, execution_time_ms } = results;

  return (
    <div className={cn('space-y-6', className)}>
      {/* Outcome Summary Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <OutcomeCard
          title="Average Treatment Effect"
          value={outcomes.ate}
          icon={Target}
          prefix=""
        />
        <OutcomeCard
          title="TRx Lift"
          value={outcomes.trx_lift}
          icon={TrendingUp}
        />
        <OutcomeCard
          title="Market Share Change"
          value={outcomes.market_share_change}
          icon={BarChart3}
          isPercentage
        />
        <OutcomeCard
          title="ROI Projection"
          value={outcomes.roi}
          icon={DollarSign}
          suffix="x"
        />
      </div>

      {/* Projections Chart */}
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <div>
              <CardTitle>Projected Outcomes</CardTitle>
              <CardDescription>
                Time series projection comparing with and without intervention scenarios
              </CardDescription>
            </div>
            <div className="flex items-center gap-2 text-sm text-muted-foreground">
              <Clock className="h-4 w-4" />
              <span>Computed in {execution_time_ms.toLocaleString()}ms</span>
            </div>
          </div>
        </CardHeader>
        <CardContent>
          <ResponsiveContainer width="100%" height={350}>
            <ComposedChart data={projections} margin={{ top: 20, right: 30, left: 20, bottom: 5 }}>
              <CartesianGrid strokeDasharray="3 3" opacity={0.3} />
              <XAxis
                dataKey="date"
                tickFormatter={formatDate}
                fontSize={12}
                tickLine={false}
              />
              <YAxis fontSize={12} tickLine={false} axisLine={false} />
              <Tooltip content={<ProjectionTooltip />} />
              <Legend />

              {/* Uncertainty band */}
              <Area
                type="monotone"
                dataKey="upper_bound"
                stroke="none"
                fill="hsl(var(--chart-1))"
                fillOpacity={0.2}
                name="Upper Bound"
              />
              <Area
                type="monotone"
                dataKey="lower_bound"
                stroke="none"
                fill="white"
                fillOpacity={1}
                name="Lower Bound"
              />

              {/* Without intervention (counterfactual) */}
              <Line
                type="monotone"
                dataKey="without_intervention"
                stroke="hsl(var(--muted-foreground))"
                strokeWidth={2}
                strokeDasharray="5 5"
                dot={false}
                name="Without Intervention"
              />

              {/* With intervention */}
              <Line
                type="monotone"
                dataKey="with_intervention"
                stroke="hsl(var(--chart-1))"
                strokeWidth={2}
                dot={false}
                name="With Intervention"
              />
            </ComposedChart>
          </ResponsiveContainer>
        </CardContent>
      </Card>

      {/* Fidelity Metrics and Sensitivity Analysis */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Fidelity Metrics */}
        <Card>
          <CardHeader>
            <div className="flex items-center justify-between">
              <div>
                <CardTitle className="flex items-center gap-2">
                  <ShieldCheck className="h-5 w-5" />
                  Model Fidelity
                </CardTitle>
                <CardDescription>
                  Confidence in simulation accuracy
                </CardDescription>
              </div>
              <Badge variant="outline" className={getConfidenceBadgeColor(fidelity.confidence_level)}>
                {fidelity.confidence_level} confidence
              </Badge>
            </div>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="text-center mb-4">
              <p className={cn(
                'text-4xl font-bold',
                getFidelityScoreColor(fidelity.overall_score)
              )}>
                {(fidelity.overall_score * 100).toFixed(0)}%
              </p>
              <p className="text-sm text-muted-foreground">Overall Fidelity Score</p>
            </div>

            <div className="space-y-3">
              <FidelityMeter label="Data Coverage" value={fidelity.data_coverage} />
              <FidelityMeter label="Model Calibration" value={fidelity.calibration} />
              <FidelityMeter label="Temporal Alignment" value={fidelity.temporal_alignment} />
              <FidelityMeter label="Feature Completeness" value={fidelity.feature_completeness} />
            </div>

            {fidelity.warnings && fidelity.warnings.length > 0 && (
              <div className="pt-4 border-t border-border">
                <p className="text-sm font-medium text-amber-600 mb-2 flex items-center gap-2">
                  <AlertTriangle className="h-4 w-4" />
                  Limitations
                </p>
                <ul className="text-sm text-muted-foreground space-y-1">
                  {fidelity.warnings.map((warning, idx) => (
                    <li key={idx}>• {warning}</li>
                  ))}
                </ul>
              </div>
            )}
          </CardContent>
        </Card>

        {/* Sensitivity Analysis */}
        <Card>
          <CardHeader>
            <CardTitle>Sensitivity Analysis</CardTitle>
            <CardDescription>
              How parameter changes affect outcomes
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              {sensitivity.slice(0, 5).map((param) => (
                <div key={param.parameter} className="space-y-2">
                  <div className="flex justify-between text-sm">
                    <span className="font-medium">{param.parameter}</span>
                    <span className="text-muted-foreground">
                      Sensitivity: {(param.sensitivity_score * 100).toFixed(0)}%
                    </span>
                  </div>
                  <div className="flex items-center gap-2 text-xs">
                    <span className="text-muted-foreground w-16">
                      Low ({param.low_value.toFixed(0)})
                    </span>
                    <div className="flex-1 relative h-6">
                      <div className="absolute inset-0 bg-muted rounded-full" />
                      <div
                        className="absolute h-6 bg-blue-200 dark:bg-blue-800 rounded-full"
                        style={{
                          left: `${Math.min(50, 50 + (param.ate_at_low / param.base_value) * 50)}%`,
                          right: `${Math.min(50, 50 - (param.ate_at_high / param.base_value) * 50)}%`,
                        }}
                      />
                      <div
                        className="absolute h-6 w-1 bg-primary"
                        style={{ left: '50%', transform: 'translateX(-50%)' }}
                      />
                    </div>
                    <span className="text-muted-foreground w-16 text-right">
                      High ({param.high_value.toFixed(0)})
                    </span>
                  </div>
                  <div className="flex justify-between text-xs text-muted-foreground">
                    <span>ATE: {param.ate_at_low.toFixed(1)}</span>
                    <span>ATE: {param.ate_at_high.toFixed(1)}</span>
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}

export default ScenarioResults;
