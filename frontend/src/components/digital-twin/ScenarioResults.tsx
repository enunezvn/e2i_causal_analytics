/**
 * Scenario Results Component
 * ==========================
 *
 * Displays digital twin simulation results from the REAL
 * `POST /api/digital-twin/simulate` response (SimulationResponse):
 * simulated ATE + CI, significance, effect size, statistical power,
 * simulation confidence, fidelity warnings, and data provenance.
 *
 * The previous version rendered `LegacySimulationResponse` — a UI-shaped
 * type (TRx/NRx lift, ROI, projections time-series, fidelity meters,
 * sensitivity analysis) the backend never returns. That shape mismatch
 * forced the consumer to pass `results={null}` forever, and wiring it
 * would have required fabricating every legacy field. The legacy sections
 * were deleted rather than faked.
 *
 * @module components/digital-twin/ScenarioResults
 */

import {
  Activity,
  AlertTriangle,
  BarChart3,
  Clock,
  Target,
  Users,
  ShieldCheck,
  CheckCircle2,
  MinusCircle,
} from 'lucide-react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';
import { cn } from '@/lib/utils';
import type { SimulationResponse } from '@/types/digital-twin';
import type { ApiError } from '@/lib/api-client';

// =============================================================================
// TYPES
// =============================================================================

export interface ScenarioResultsProps {
  /** Real simulation results to display (null until a run completes) */
  results: SimulationResponse | null;
  /** Whether results are loading */
  isLoading?: boolean;
  /**
   * The error from a failed simulation run, if any. When present (and not
   * loading), an HONEST error card is shown instead of the neutral
   * "No Simulation Results" empty state — so a 503/408/500 is never silently
   * indistinguishable from "never ran".
   */
  error?: ApiError | null;
  /** Additional CSS classes */
  className?: string;
}

// =============================================================================
// SUB-COMPONENTS
// =============================================================================

function StatCard({
  title,
  value,
  detail,
  icon: Icon,
  valueClassName,
}: {
  title: string;
  value: string;
  detail?: string;
  icon: React.ElementType;
  valueClassName?: string;
}) {
  return (
    <Card>
      <CardContent className="pt-4">
        <div className="flex items-center justify-between mb-2">
          <span className="text-sm text-muted-foreground">{title}</span>
          <Icon className="h-4 w-4 text-muted-foreground" />
        </div>
        <div className="space-y-1">
          <p className={cn('text-2xl font-bold', valueClassName)}>{value}</p>
          {detail && <p className="text-xs text-muted-foreground">{detail}</p>}
        </div>
      </CardContent>
    </Card>
  );
}

// =============================================================================
// COMPONENT
// =============================================================================

export function ScenarioResults({
  results,
  isLoading = false,
  error = null,
  className = '',
}: ScenarioResultsProps) {
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

  // Honest error state — a failed run must NOT look like "never ran".
  // The backend detail is surfaced verbatim; no fabricated numbers.
  if (error) {
    const detail = error.data?.message ?? error.message;
    let title = 'Simulation failed';
    let hint =
      'The simulation could not be completed. See the details below and try again.';
    if (error.status === 503) {
      title = 'No trained twin model is available';
      hint =
        'No trained digital-twin model is available for this brand/cohort, or the model registry is temporarily unavailable. Try a brand with a trained twin, or retry shortly.';
    } else if (error.status === 408) {
      title = 'Simulation timed out';
      hint = 'The simulation took too long. Try again with a smaller twin count.';
    } else if (error.isNetworkError) {
      title = 'Could not reach the simulation service';
      hint = 'A network error occurred. Check your connection and try again.';
    } else if (error.isServerError) {
      title = 'Simulation failed';
      hint = 'An unexpected server error occurred while running the simulation.';
    }
    return (
      <Card className={cn('border-destructive/40', className)}>
        <CardContent className="pt-6">
          <div className="flex flex-col items-center justify-center py-12 text-center">
            <AlertTriangle className="h-12 w-12 text-destructive mb-4" />
            <h3 className="text-lg font-medium mb-2">{title}</h3>
            <p className="text-muted-foreground max-w-md">{hint}</p>
            {detail && (
              <p className="mt-3 text-sm text-destructive/90 max-w-md break-words">
                {detail}
              </p>
            )}
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
              Configure and run a simulation to see the predicted treatment
              effect, confidence interval, and deployment recommendation.
            </p>
          </div>
        </CardContent>
      </Card>
    );
  }

  const isPositive = results.simulated_ate > 0;
  const ciLabel = `[${results.simulated_ci_lower.toFixed(3)}, ${results.simulated_ci_upper.toFixed(3)}]`;

  return (
    <div className={cn('space-y-6', className)}>
      {/* Fidelity warning — surfaced prominently, never hidden */}
      {results.fidelity_warning && (
        <div className="flex items-start gap-2 rounded-lg border border-amber-500/30 bg-amber-500/5 p-3">
          <AlertTriangle className="h-4 w-4 text-amber-500 mt-0.5" />
          <div className="text-sm text-muted-foreground">
            <span className="font-medium text-amber-600">Fidelity warning:</span>{' '}
            {results.fidelity_warning_reason ??
              'The twin model flagged reduced fidelity for this simulation.'}
          </div>
        </div>
      )}

      {/* Real outcome cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <StatCard
          title="Simulated ATE"
          value={results.simulated_ate.toFixed(3)}
          detail={`95% CI: ${ciLabel}`}
          icon={Target}
          valueClassName={isPositive ? 'text-emerald-600' : 'text-rose-600'}
        />
        <StatCard
          title="Effect Size (Cohen's d)"
          value={
            typeof results.effect_size_cohens_d === 'number'
              ? results.effect_size_cohens_d.toFixed(2)
              : '—'
          }
          detail={`Std. error ${results.simulated_std_error.toFixed(3)}`}
          icon={BarChart3}
        />
        <StatCard
          title="Statistical Power"
          value={
            typeof results.statistical_power === 'number'
              ? `${Math.round(results.statistical_power * 100)}%`
              : '—'
          }
          detail={
            results.recommended_sample_size
              ? `Recommended n: ${results.recommended_sample_size.toLocaleString()}`
              : undefined
          }
          icon={ShieldCheck}
        />
        <StatCard
          title="Twins Simulated"
          value={results.twin_count.toLocaleString()}
          detail={`${results.twin_type} twins`}
          icon={Users}
        />
      </div>

      {/* Significance + run metadata */}
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <div>
              <CardTitle>Simulation Summary</CardTitle>
              <CardDescription>
                {results.brand} — {results.intervention_type}
              </CardDescription>
            </div>
            <Badge
              variant="outline"
              className={cn(
                results.is_significant
                  ? 'bg-emerald-100 text-emerald-700 dark:bg-emerald-900/30 dark:text-emerald-400'
                  : 'bg-amber-100 text-amber-700 dark:bg-amber-900/30 dark:text-amber-400'
              )}
            >
              {results.is_significant ? (
                <CheckCircle2 className="h-3 w-3 mr-1" />
              ) : (
                <MinusCircle className="h-3 w-3 mr-1" />
              )}
              {results.is_significant ? 'Significant' : 'Not significant'}
            </Badge>
          </div>
        </CardHeader>
        <CardContent className="space-y-4">
          <div>
            <div className="flex justify-between text-sm mb-1">
              <span className="text-muted-foreground">Simulation confidence</span>
              <span className="font-medium">
                {Math.round(results.simulation_confidence * 100)}%
              </span>
            </div>
            <Progress value={results.simulation_confidence * 100} className="h-2" />
          </div>

          {typeof results.model_fidelity_score === 'number' && (
            <div>
              <div className="flex justify-between text-sm mb-1">
                <span className="text-muted-foreground">Model fidelity score</span>
                <span className="font-medium">
                  {Math.round(results.model_fidelity_score * 100)}%
                </span>
              </div>
              <Progress value={results.model_fidelity_score * 100} className="h-2" />
            </div>
          )}

          <div className="flex flex-wrap items-center gap-x-6 gap-y-2 text-xs text-muted-foreground pt-2 border-t border-border">
            <span className="flex items-center gap-1">
              <Clock className="h-3 w-3" />
              Computed in {results.execution_time_ms.toLocaleString()} ms
            </span>
            <span>Effect direction: {results.effect_direction}</span>
            <span>Model: {results.model_id}</span>
            {results.data_provenance && (
              <span>Provenance: {results.data_provenance}</span>
            )}
          </div>

          {results.error_message && (
            <div className="flex items-start gap-2 rounded-lg border border-rose-500/30 bg-rose-500/5 p-3 text-sm text-muted-foreground">
              <AlertTriangle className="h-4 w-4 text-rose-500 mt-0.5" />
              <span>
                <span className="font-medium text-rose-600">Simulation error:</span>{' '}
                {results.error_message}
              </span>
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}

export default ScenarioResults;
