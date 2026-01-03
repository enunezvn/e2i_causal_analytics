/**
 * Recommendation Cards Component
 * ==============================
 *
 * Displays digital twin simulation recommendations with
 * actionable insights and risk factors.
 *
 * @module components/digital-twin/RecommendationCards
 */

import {
  CheckCircle2,
  XCircle,
  AlertTriangle,
  Lightbulb,
  ArrowRight,
  Shield,
  TrendingUp,
  Settings2,
  Search,
} from 'lucide-react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { cn } from '@/lib/utils';
import {
  type SimulationRecommendation,
  RecommendationType,
  ConfidenceLevel,
} from '@/types/digital-twin';

// =============================================================================
// TYPES
// =============================================================================

export interface RecommendationCardsProps {
  /** Simulation recommendation */
  recommendation: SimulationRecommendation | null;
  /** Callback when user accepts recommendation */
  onAccept?: () => void;
  /** Callback when user wants to refine */
  onRefine?: () => void;
  /** Callback when user wants more analysis */
  onAnalyze?: () => void;
  /** Additional CSS classes */
  className?: string;
}

// =============================================================================
// HELPER FUNCTIONS
// =============================================================================

interface RecommendationStyle {
  icon: React.ElementType;
  bgColor: string;
  textColor: string;
  borderColor: string;
  label: string;
  actionLabel: string;
}

function getRecommendationStyle(type: RecommendationType): RecommendationStyle {
  switch (type) {
    case RecommendationType.DEPLOY:
      return {
        icon: CheckCircle2,
        bgColor: 'bg-emerald-50 dark:bg-emerald-950/20',
        textColor: 'text-emerald-700 dark:text-emerald-400',
        borderColor: 'border-emerald-200 dark:border-emerald-900',
        label: 'Deploy',
        actionLabel: 'Proceed with Deployment',
      };
    case RecommendationType.SKIP:
      return {
        icon: XCircle,
        bgColor: 'bg-rose-50 dark:bg-rose-950/20',
        textColor: 'text-rose-700 dark:text-rose-400',
        borderColor: 'border-rose-200 dark:border-rose-900',
        label: 'Skip',
        actionLabel: 'Skip Intervention',
      };
    case RecommendationType.REFINE:
      return {
        icon: Settings2,
        bgColor: 'bg-amber-50 dark:bg-amber-950/20',
        textColor: 'text-amber-700 dark:text-amber-400',
        borderColor: 'border-amber-200 dark:border-amber-900',
        label: 'Refine',
        actionLabel: 'Adjust Parameters',
      };
    case RecommendationType.ANALYZE:
      return {
        icon: Search,
        bgColor: 'bg-blue-50 dark:bg-blue-950/20',
        textColor: 'text-blue-700 dark:text-blue-400',
        borderColor: 'border-blue-200 dark:border-blue-900',
        label: 'Analyze',
        actionLabel: 'Run More Analysis',
      };
  }
}

function getConfidenceStyle(level: ConfidenceLevel): { color: string; label: string } {
  switch (level) {
    case ConfidenceLevel.HIGH:
      return { color: 'bg-emerald-100 text-emerald-700', label: 'High Confidence' };
    case ConfidenceLevel.MEDIUM:
      return { color: 'bg-amber-100 text-amber-700', label: 'Medium Confidence' };
    case ConfidenceLevel.LOW:
      return { color: 'bg-rose-100 text-rose-700', label: 'Low Confidence' };
  }
}

// =============================================================================
// COMPONENT
// =============================================================================

export function RecommendationCards({
  recommendation,
  onAccept,
  onRefine,
  onAnalyze,
  className = '',
}: RecommendationCardsProps) {
  if (!recommendation) {
    return (
      <Card className={className}>
        <CardContent className="pt-6">
          <div className="flex flex-col items-center justify-center py-8 text-center">
            <Lightbulb className="h-10 w-10 text-muted-foreground mb-3" />
            <h3 className="text-lg font-medium mb-1">No Recommendation Yet</h3>
            <p className="text-sm text-muted-foreground">
              Run a simulation to receive AI-powered recommendations
            </p>
          </div>
        </CardContent>
      </Card>
    );
  }

  const style = getRecommendationStyle(recommendation.type);
  const confidenceStyle = getConfidenceStyle(recommendation.confidence);
  const Icon = style.icon;

  return (
    <div className={cn('space-y-4', className)}>
      {/* Main Recommendation Card */}
      <Card className={cn('border-2', style.borderColor, style.bgColor)}>
        <CardHeader>
          <div className="flex items-start justify-between">
            <div className="flex items-center gap-3">
              <div className={cn('p-3 rounded-full', style.bgColor)}>
                <Icon className={cn('h-6 w-6', style.textColor)} />
              </div>
              <div>
                <CardTitle className={cn('text-xl', style.textColor)}>
                  Recommendation: {style.label}
                </CardTitle>
                <CardDescription className="mt-1">
                  {recommendation.rationale}
                </CardDescription>
              </div>
            </div>
            <Badge variant="outline" className={confidenceStyle.color}>
              {confidenceStyle.label}
            </Badge>
          </div>
        </CardHeader>
        <CardContent className="space-y-4">
          {/* Expected Value */}
          {recommendation.expected_value !== undefined && (
            <div className="flex items-center gap-2 p-3 bg-white/50 dark:bg-black/20 rounded-lg">
              <TrendingUp className="h-5 w-5 text-emerald-600" />
              <span className="text-sm text-muted-foreground">Expected Value:</span>
              <span className="font-bold text-lg text-emerald-600">
                ${recommendation.expected_value.toLocaleString()}
              </span>
            </div>
          )}

          {/* Evidence Points */}
          <div>
            <h4 className="text-sm font-medium mb-2 flex items-center gap-2">
              <CheckCircle2 className="h-4 w-4 text-emerald-600" />
              Supporting Evidence
            </h4>
            <ul className="space-y-2">
              {recommendation.evidence.map((point, idx) => (
                <li key={idx} className="flex items-start gap-2 text-sm">
                  <ArrowRight className="h-4 w-4 mt-0.5 text-muted-foreground flex-shrink-0" />
                  <span>{point}</span>
                </li>
              ))}
            </ul>
          </div>

          {/* Action Buttons */}
          <div className="flex gap-3 pt-2">
            {recommendation.type === RecommendationType.DEPLOY && onAccept && (
              <Button onClick={onAccept} className="flex-1">
                <CheckCircle2 className="h-4 w-4 mr-2" />
                {style.actionLabel}
              </Button>
            )}
            {recommendation.type === RecommendationType.REFINE && onRefine && (
              <Button onClick={onRefine} variant="outline" className="flex-1">
                <Settings2 className="h-4 w-4 mr-2" />
                {style.actionLabel}
              </Button>
            )}
            {recommendation.type === RecommendationType.ANALYZE && onAnalyze && (
              <Button onClick={onAnalyze} variant="outline" className="flex-1">
                <Search className="h-4 w-4 mr-2" />
                {style.actionLabel}
              </Button>
            )}
            {recommendation.type === RecommendationType.SKIP && (
              <Button variant="destructive" className="flex-1" disabled>
                <XCircle className="h-4 w-4 mr-2" />
                Not Recommended
              </Button>
            )}
          </div>
        </CardContent>
      </Card>

      {/* Risk Factors Card */}
      {recommendation.risk_factors && recommendation.risk_factors.length > 0 && (
        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="text-base flex items-center gap-2">
              <Shield className="h-5 w-5 text-amber-600" />
              Risk Factors to Consider
            </CardTitle>
          </CardHeader>
          <CardContent>
            <ul className="space-y-2">
              {recommendation.risk_factors.map((risk, idx) => (
                <li key={idx} className="flex items-start gap-2 text-sm">
                  <AlertTriangle className="h-4 w-4 mt-0.5 text-amber-500 flex-shrink-0" />
                  <span className="text-muted-foreground">{risk}</span>
                </li>
              ))}
            </ul>
          </CardContent>
        </Card>
      )}

      {/* Suggested Refinements Card */}
      {recommendation.type === RecommendationType.REFINE &&
        recommendation.suggested_refinements &&
        Object.keys(recommendation.suggested_refinements).length > 0 && (
        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="text-base flex items-center gap-2">
              <Settings2 className="h-5 w-5 text-blue-600" />
              Suggested Refinements
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {Object.entries(recommendation.suggested_refinements).map(([key, value]) => (
                <div
                  key={key}
                  className="flex items-center justify-between p-3 bg-muted/50 rounded-lg"
                >
                  <span className="text-sm font-medium capitalize">
                    {key.replace(/_/g, ' ')}
                  </span>
                  <Badge variant="outline" className="bg-blue-50 text-blue-700">
                    {String(value)}
                  </Badge>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
}

export default RecommendationCards;
