/**
 * Heterogeneous Treatment Effects Component
 * ==========================================
 *
 * Displays segment-level Conditional Average Treatment Effects (CATE) analysis.
 * Shows how treatment effects vary across different patient/HCP segments.
 *
 * @module components/insights/HeterogeneousTreatmentEffects
 */

import { useState, useEffect } from 'react';
import { Users, TrendingUp, TrendingDown, BarChart3, RefreshCw, Info } from 'lucide-react';
import { cn } from '@/lib/utils';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Progress } from '@/components/ui/progress';
import { useBatchExplain } from '@/hooks/api/use-explain';
import { ModelType } from '@/types/explain';

// =============================================================================
// TYPES
// =============================================================================

interface HeterogeneousTreatmentEffectsProps {
  className?: string;
}

interface SegmentEffect {
  id: string;
  segment: string;
  description: string;
  sampleSize: number;
  treatmentEffect: number;
  confidence: number;
  pValue: number;
  isSignificant: boolean;
  topDrivers: { feature: string; impact: number }[];
}

// =============================================================================
// SAMPLE DATA
// =============================================================================

const SAMPLE_SEGMENTS: SegmentEffect[] = [
  {
    id: 'seg-1',
    segment: 'High-Volume Specialists',
    description: 'HCPs with >100 patients/month in specialty areas',
    sampleSize: 1247,
    treatmentEffect: 0.23,
    confidence: 0.94,
    pValue: 0.001,
    isSignificant: true,
    topDrivers: [
      { feature: 'Prior Auth Volume', impact: 0.18 },
      { feature: 'Payer Mix Index', impact: 0.12 },
      { feature: 'Practice Size', impact: 0.08 },
    ],
  },
  {
    id: 'seg-2',
    segment: 'Academic Medical Centers',
    description: 'Physicians affiliated with teaching hospitals',
    sampleSize: 523,
    treatmentEffect: 0.15,
    confidence: 0.87,
    pValue: 0.012,
    isSignificant: true,
    topDrivers: [
      { feature: 'Research Activity', impact: 0.14 },
      { feature: 'Specialty Mix', impact: 0.09 },
      { feature: 'Patient Complexity', impact: 0.07 },
    ],
  },
  {
    id: 'seg-3',
    segment: 'Community Practice',
    description: 'Independent community-based practices',
    sampleSize: 2156,
    treatmentEffect: 0.08,
    confidence: 0.72,
    pValue: 0.089,
    isSignificant: false,
    topDrivers: [
      { feature: 'Geographic Access', impact: 0.06 },
      { feature: 'Payer Distribution', impact: 0.04 },
      { feature: 'Competition Density', impact: 0.03 },
    ],
  },
  {
    id: 'seg-4',
    segment: 'Early Adopters',
    description: 'HCPs who rapidly adopt new therapies',
    sampleSize: 412,
    treatmentEffect: 0.31,
    confidence: 0.91,
    pValue: 0.002,
    isSignificant: true,
    topDrivers: [
      { feature: 'Innovation Score', impact: 0.22 },
      { feature: 'Conference Attendance', impact: 0.11 },
      { feature: 'Digital Engagement', impact: 0.09 },
    ],
  },
];

// =============================================================================
// SUB-COMPONENTS
// =============================================================================

function SegmentCard({ segment }: { segment: SegmentEffect }) {
  const effectPercent = (segment.treatmentEffect * 100).toFixed(1);
  const isPositive = segment.treatmentEffect >= 0;

  return (
    <div
      className={cn(
        'p-4 rounded-lg border',
        segment.isSignificant
          ? 'border-emerald-500/30 bg-emerald-500/5'
          : 'border-[var(--color-border)] bg-[var(--color-card)]'
      )}
    >
      {/* Header */}
      <div className="flex items-start justify-between gap-2 mb-2">
        <div>
          <h4 className="text-sm font-medium text-[var(--color-foreground)]">{segment.segment}</h4>
          <p className="text-xs text-[var(--color-muted-foreground)]">{segment.description}</p>
        </div>
        <Badge
          variant="outline"
          className={cn(
            'text-xs',
            segment.isSignificant
              ? 'bg-emerald-500/10 text-emerald-600 border-emerald-500/20'
              : 'bg-gray-500/10 text-gray-600 border-gray-500/20'
          )}
        >
          {segment.isSignificant ? 'Significant' : 'Not Significant'}
        </Badge>
      </div>

      {/* Treatment Effect */}
      <div className="grid grid-cols-3 gap-3 mb-3">
        <div className="p-2 rounded bg-[var(--color-muted)]/30">
          <div className="text-xs text-[var(--color-muted-foreground)]">CATE</div>
          <div
            className={cn(
              'text-lg font-bold flex items-center gap-1',
              isPositive ? 'text-emerald-600' : 'text-rose-600'
            )}
          >
            {isPositive ? <TrendingUp className="h-4 w-4" /> : <TrendingDown className="h-4 w-4" />}
            {isPositive ? '+' : ''}
            {effectPercent}%
          </div>
        </div>
        <div className="p-2 rounded bg-[var(--color-muted)]/30">
          <div className="text-xs text-[var(--color-muted-foreground)]">Confidence</div>
          <div className="text-lg font-bold text-blue-600">
            {(segment.confidence * 100).toFixed(0)}%
          </div>
        </div>
        <div className="p-2 rounded bg-[var(--color-muted)]/30">
          <div className="text-xs text-[var(--color-muted-foreground)]">Sample</div>
          <div className="text-lg font-bold text-[var(--color-foreground)]">
            {segment.sampleSize.toLocaleString()}
          </div>
        </div>
      </div>

      {/* Top Drivers */}
      <div className="space-y-2">
        <div className="text-xs text-[var(--color-muted-foreground)] font-medium">Top Drivers</div>
        {segment.topDrivers.map((driver, idx) => (
          <div key={idx} className="flex items-center gap-2">
            <span className="text-xs text-[var(--color-muted-foreground)] w-28 truncate">
              {driver.feature}
            </span>
            <Progress value={driver.impact * 100} className="flex-1 h-1.5" />
            <span className="text-xs font-medium w-10 text-right">
              {(driver.impact * 100).toFixed(0)}%
            </span>
          </div>
        ))}
      </div>

      {/* P-value footer */}
      <div className="mt-3 pt-2 border-t border-[var(--color-border)]">
        <span className="text-xs text-[var(--color-muted-foreground)]">
          p-value: {segment.pValue.toFixed(3)}
        </span>
      </div>
    </div>
  );
}

// =============================================================================
// MAIN COMPONENT
// =============================================================================

export function HeterogeneousTreatmentEffects({ className }: HeterogeneousTreatmentEffectsProps) {
  const [segments] = useState<SegmentEffect[]>(SAMPLE_SEGMENTS);

  // Use batch explain for SHAP-based segment analysis
  const { mutate: explainBatch, data: batchResponse, isPending } = useBatchExplain();

  // Fetch on mount
  useEffect(() => {
    explainBatch({
      requests: [
        { patient_id: 'segment_high_volume', model_type: ModelType.PROPENSITY },
        { patient_id: 'segment_academic', model_type: ModelType.PROPENSITY },
        { patient_id: 'segment_community', model_type: ModelType.PROPENSITY },
        { patient_id: 'segment_early_adopter', model_type: ModelType.PROPENSITY },
      ],
      parallel: true,
    });
  }, [explainBatch]);

  // Transform batch response if available (simplified - real impl would parse properly)
  useEffect(() => {
    if (batchResponse?.successful && batchResponse.successful > 0) {
      // In a real implementation, we would transform batchResponse.explanations
      // to SegmentEffect format. For now, keep sample data as fallback.
      // Future: setSegments(transformedSegments);
    }
  }, [batchResponse]);

  const handleRefresh = () => {
    explainBatch({
      requests: [
        { patient_id: 'segment_high_volume', model_type: ModelType.PROPENSITY },
        { patient_id: 'segment_academic', model_type: ModelType.PROPENSITY },
        { patient_id: 'segment_community', model_type: ModelType.PROPENSITY },
        { patient_id: 'segment_early_adopter', model_type: ModelType.PROPENSITY },
      ],
      parallel: true,
    });
  };

  const significantCount = segments.filter((s) => s.isSignificant).length;
  const avgEffect =
    segments.reduce((sum, s) => sum + s.treatmentEffect, 0) / segments.length;

  return (
    <Card className={cn('bg-[var(--color-card)] border-[var(--color-border)]', className)}>
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <div className="p-2 rounded-lg bg-indigo-500/10">
              <BarChart3 className="h-5 w-5 text-indigo-500" />
            </div>
            <div>
              <CardTitle className="text-base font-semibold">
                Heterogeneous Treatment Effects
              </CardTitle>
              <p className="text-xs text-[var(--color-muted-foreground)]">
                Segment-level CATE analysis
              </p>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <Badge variant="outline" className="text-xs bg-indigo-500/10 text-indigo-600">
              {significantCount}/{segments.length} Significant
            </Badge>
            <Button
              variant="ghost"
              size="icon"
              onClick={handleRefresh}
              disabled={isPending}
              className="h-8 w-8"
            >
              <RefreshCw className={cn('h-4 w-4', isPending && 'animate-spin')} />
            </Button>
          </div>
        </div>
      </CardHeader>
      <CardContent className="space-y-3">
        {/* Summary Stats */}
        <div className="grid grid-cols-2 gap-3 mb-4">
          <div className="p-3 rounded-lg bg-[var(--color-muted)]/30 border border-[var(--color-border)]">
            <div className="flex items-center gap-2 mb-1">
              <Users className="h-4 w-4 text-[var(--color-muted-foreground)]" />
              <span className="text-xs text-[var(--color-muted-foreground)]">Avg. Treatment Effect</span>
            </div>
            <div className="text-xl font-bold text-emerald-600">
              +{(avgEffect * 100).toFixed(1)}%
            </div>
          </div>
          <div className="p-3 rounded-lg bg-[var(--color-muted)]/30 border border-[var(--color-border)]">
            <div className="flex items-center gap-2 mb-1">
              <TrendingUp className="h-4 w-4 text-[var(--color-muted-foreground)]" />
              <span className="text-xs text-[var(--color-muted-foreground)]">Total Sample Size</span>
            </div>
            <div className="text-xl font-bold text-[var(--color-foreground)]">
              {segments.reduce((sum, s) => sum + s.sampleSize, 0).toLocaleString()}
            </div>
          </div>
        </div>

        {/* Loading State */}
        {isPending && (
          <div className="flex items-center justify-center py-8">
            <div className="flex items-center gap-3 text-[var(--color-muted-foreground)]">
              <RefreshCw className="h-5 w-5 animate-spin" />
              <span className="text-sm">Analyzing segment effects...</span>
            </div>
          </div>
        )}

        {/* Segment Cards */}
        {!isPending && (
          <div className="grid gap-3">
            {segments.map((segment) => (
              <SegmentCard key={segment.id} segment={segment} />
            ))}
          </div>
        )}

        {/* Info Banner */}
        <div className="flex items-start gap-2 p-3 rounded-lg bg-indigo-500/5 border border-indigo-500/20">
          <Info className="h-4 w-4 text-indigo-500 mt-0.5" />
          <div className="text-xs text-[var(--color-muted-foreground)]">
            <span className="font-medium text-indigo-600">CATE Analysis:</span> Conditional Average
            Treatment Effects show how intervention impact varies across segments. Significant
            effects (p &lt; 0.05) indicate reliable segment-level targeting opportunities.
          </div>
        </div>
      </CardContent>
    </Card>
  );
}

export default HeterogeneousTreatmentEffects;
