/**
 * Experiment Recommendations Component
 * =====================================
 *
 * Displays A/B test suggestions with Digital Twin pre-screening results.
 * Shows recommended experiments based on causal analysis and gap detection.
 *
 * @module components/insights/ExperimentRecommendations
 */

import { FlaskConical, TrendingUp, Users, Clock, CheckCircle2, AlertCircle } from 'lucide-react';
import { cn } from '@/lib/utils';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Progress } from '@/components/ui/progress';

// =============================================================================
// TYPES
// =============================================================================

interface ExperimentRecommendationsProps {
  className?: string;
}

interface Experiment {
  id: string;
  title: string;
  hypothesis: string;
  primaryMetric: string;
  expectedLift: number;
  confidence: number;
  sampleSize: number;
  duration: string;
  status: 'recommended' | 'simulated' | 'approved' | 'running';
  digitalTwinScore: number;
  riskLevel: 'low' | 'medium' | 'high';
}

// =============================================================================
// SAMPLE DATA
// =============================================================================

const SAMPLE_EXPERIMENTS: Experiment[] = [
  {
    id: 'exp-1',
    title: 'Increased Call Frequency - NE Region',
    hypothesis: 'Increasing rep call frequency from 2x to 3x monthly will improve TRx by 15%+',
    primaryMetric: 'TRx Volume',
    expectedLift: 18.5,
    confidence: 0.87,
    sampleSize: 450,
    duration: '8 weeks',
    status: 'recommended',
    digitalTwinScore: 0.92,
    riskLevel: 'low',
  },
  {
    id: 'exp-2',
    title: 'Digital-First Engagement Pilot',
    hypothesis: 'Replacing 1 in-person call with digital touchpoint maintains engagement at lower cost',
    primaryMetric: 'Engagement Score',
    expectedLift: -2.1,
    confidence: 0.78,
    sampleSize: 320,
    duration: '6 weeks',
    status: 'simulated',
    digitalTwinScore: 0.76,
    riskLevel: 'medium',
  },
  {
    id: 'exp-3',
    title: 'Sample Distribution Optimization',
    hypothesis: 'Targeted sample distribution to high-potential HCPs increases conversion 20%+',
    primaryMetric: 'Conversion Rate',
    expectedLift: 23.2,
    confidence: 0.82,
    sampleSize: 280,
    duration: '10 weeks',
    status: 'approved',
    digitalTwinScore: 0.88,
    riskLevel: 'low',
  },
];

// =============================================================================
// HELPERS
// =============================================================================

function getStatusConfig(status: Experiment['status']) {
  const config = {
    recommended: {
      label: 'Recommended',
      className: 'bg-blue-500/10 text-blue-600 border-blue-500/20',
      icon: FlaskConical,
    },
    simulated: {
      label: 'Simulated',
      className: 'bg-purple-500/10 text-purple-600 border-purple-500/20',
      icon: TrendingUp,
    },
    approved: {
      label: 'Approved',
      className: 'bg-emerald-500/10 text-emerald-600 border-emerald-500/20',
      icon: CheckCircle2,
    },
    running: {
      label: 'Running',
      className: 'bg-amber-500/10 text-amber-600 border-amber-500/20',
      icon: Clock,
    },
  };
  return config[status];
}

function getRiskConfig(risk: Experiment['riskLevel']) {
  const config = {
    low: { label: 'Low Risk', className: 'text-emerald-600' },
    medium: { label: 'Medium Risk', className: 'text-amber-600' },
    high: { label: 'High Risk', className: 'text-rose-600' },
  };
  return config[risk];
}

// =============================================================================
// SUB-COMPONENTS
// =============================================================================

function ExperimentCard({ experiment }: { experiment: Experiment }) {
  const statusConfig = getStatusConfig(experiment.status);
  const riskConfig = getRiskConfig(experiment.riskLevel);
  const StatusIcon = statusConfig.icon;

  return (
    <div className="p-4 rounded-lg border border-[var(--color-border)] bg-[var(--color-card)]">
      {/* Header */}
      <div className="flex items-start justify-between gap-2 mb-3">
        <div className="flex items-center gap-2">
          <div className="p-1.5 rounded bg-purple-500/10">
            <FlaskConical className="h-4 w-4 text-purple-500" />
          </div>
          <h4 className="text-sm font-medium text-[var(--color-foreground)]">
            {experiment.title}
          </h4>
        </div>
        <Badge variant="outline" className={cn('text-xs', statusConfig.className)}>
          <StatusIcon className="h-3 w-3 mr-1" />
          {statusConfig.label}
        </Badge>
      </div>

      {/* Hypothesis */}
      <p className="text-sm text-[var(--color-muted-foreground)] mb-3">
        {experiment.hypothesis}
      </p>

      {/* Metrics Grid */}
      <div className="grid grid-cols-2 gap-3 mb-3">
        <div className="p-2 rounded bg-[var(--color-muted)]/30">
          <div className="text-xs text-[var(--color-muted-foreground)]">Expected Lift</div>
          <div className={cn(
            'text-lg font-bold',
            experiment.expectedLift >= 0 ? 'text-emerald-600' : 'text-rose-600'
          )}>
            {experiment.expectedLift >= 0 ? '+' : ''}{experiment.expectedLift.toFixed(1)}%
          </div>
        </div>
        <div className="p-2 rounded bg-[var(--color-muted)]/30">
          <div className="text-xs text-[var(--color-muted-foreground)]">Digital Twin Score</div>
          <div className="text-lg font-bold text-purple-600">
            {(experiment.digitalTwinScore * 100).toFixed(0)}%
          </div>
        </div>
      </div>

      {/* Confidence Bar */}
      <div className="mb-3">
        <div className="flex items-center justify-between text-xs mb-1">
          <span className="text-[var(--color-muted-foreground)]">Confidence</span>
          <span className="font-medium">{(experiment.confidence * 100).toFixed(0)}%</span>
        </div>
        <Progress value={experiment.confidence * 100} className="h-1.5" />
      </div>

      {/* Footer */}
      <div className="flex items-center justify-between pt-3 border-t border-[var(--color-border)]">
        <div className="flex items-center gap-3 text-xs text-[var(--color-muted-foreground)]">
          <div className="flex items-center gap-1">
            <Users className="h-3 w-3" />
            <span>n={experiment.sampleSize}</span>
          </div>
          <div className="flex items-center gap-1">
            <Clock className="h-3 w-3" />
            <span>{experiment.duration}</span>
          </div>
          <span className={riskConfig.className}>{riskConfig.label}</span>
        </div>
        {experiment.status === 'recommended' && (
          <Button size="sm" variant="outline" className="h-7 text-xs">
            Simulate
          </Button>
        )}
        {experiment.status === 'simulated' && (
          <Button size="sm" className="h-7 text-xs">
            Approve
          </Button>
        )}
      </div>
    </div>
  );
}

// =============================================================================
// MAIN COMPONENT
// =============================================================================

export function ExperimentRecommendations({ className }: ExperimentRecommendationsProps) {
  const recommendedCount = SAMPLE_EXPERIMENTS.filter(e => e.status === 'recommended').length;
  const simulatedCount = SAMPLE_EXPERIMENTS.filter(e => e.status === 'simulated').length;

  return (
    <Card className={cn('bg-[var(--color-card)] border-[var(--color-border)]', className)}>
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <div className="p-2 rounded-lg bg-purple-500/10">
              <FlaskConical className="h-5 w-5 text-purple-500" />
            </div>
            <div>
              <CardTitle className="text-base font-semibold">Experiment Recommendations</CardTitle>
              <p className="text-xs text-[var(--color-muted-foreground)]">
                A/B tests with Digital Twin pre-screening
              </p>
            </div>
          </div>
          <div className="flex items-center gap-2">
            {recommendedCount > 0 && (
              <Badge variant="outline" className="text-xs bg-blue-500/10 text-blue-600">
                {recommendedCount} New
              </Badge>
            )}
            {simulatedCount > 0 && (
              <Badge variant="outline" className="text-xs bg-purple-500/10 text-purple-600">
                {simulatedCount} Simulated
              </Badge>
            )}
          </div>
        </div>
      </CardHeader>
      <CardContent className="space-y-3">
        {SAMPLE_EXPERIMENTS.map((experiment) => (
          <ExperimentCard key={experiment.id} experiment={experiment} />
        ))}

        {/* Info Banner */}
        <div className="flex items-start gap-2 p-3 rounded-lg bg-purple-500/5 border border-purple-500/20">
          <AlertCircle className="h-4 w-4 text-purple-500 mt-0.5" />
          <div className="text-xs text-[var(--color-muted-foreground)]">
            <span className="font-medium text-purple-600">Digital Twin Pre-Screening:</span> All
            experiments are simulated using our Digital Twin environment before live deployment to
            estimate outcomes and minimize risk.
          </div>
        </div>
      </CardContent>
    </Card>
  );
}

export default ExperimentRecommendations;
