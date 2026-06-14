/**
 * Executive AI Brief Component
 * ============================
 *
 * Displays an AI-generated executive summary of key insights.
 *
 * Data sources, in order of preference (real data only):
 * 1. Crystallized executive insights for the brand
 *    (`GET /api/executive-insights`, M5 REWIRE).
 * 2. The live cognitive-RAG response (`POST /api/cognitive/rag`).
 *
 * Honest-state contract: real sections, an explicit empty state, or a
 * labeled error. SAMPLE_BRIEF (fabricated $2.3M / 847-HCP / beta=0.42
 * sections with invented confidence badges) was DELETED — its removal was
 * the deferred "SAMPLE_* phase" flagged in commit 9a6c9404. Confidence
 * badges render only when a real confidence value exists; none is invented.
 *
 * @module components/insights/ExecutiveAIBrief
 */

import { useEffect, useState } from 'react';
import { Brain, RefreshCw, Sparkles, Clock, CheckCircle2, AlertTriangle } from 'lucide-react';
import { cn } from '@/lib/utils';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { EmptyState } from '@/components/ui/EmptyState';
import { useCognitiveRAG } from '@/hooks/api/use-cognitive';
import { useExecutiveInsights } from '@/hooks/api/use-executive-insights';

// =============================================================================
// TYPES
// =============================================================================

interface ExecutiveAIBriefProps {
  className?: string;
  brand?: string;
}

interface BriefSection {
  title: string;
  content: string;
  /** Real source metadata when available (e.g. crystallization source count). */
  sourceLabel?: string;
}

// =============================================================================
// MAIN COMPONENT
// =============================================================================

export function ExecutiveAIBrief({ className, brand = 'Remibrutinib' }: ExecutiveAIBriefProps) {
  const [lastUpdated, setLastUpdated] = useState<Date | null>(null);

  // AI-powered brief via cognitive RAG (real backend response).
  const {
    mutate: generateBrief,
    data: briefResponse,
    error: briefError,
    isPending: isGenerating,
  } = useCognitiveRAG();

  // Real crystallized insights for this brand (M5 REWIRE). When present,
  // these take precedence over the cognitive-RAG path.
  const { data: crystallized } = useExecutiveInsights(brand);

  const crystallizedSections: BriefSection[] | null =
    crystallized && crystallized.length > 0
      ? crystallized.slice(0, 3).map((ins) => ({
          title: ins.title,
          content: ins.narrative,
          sourceLabel: `${ins.source_count} source${ins.source_count === 1 ? '' : 's'}`,
        }))
      : null;

  // The cognitive-RAG endpoint reports failures IN-BAND: an HTTP 200 whose
  // payload carries a non-empty `error` (with `response` holding the error
  // STRING, `hop_count === 0`, and `evidence === []`). Rendering that
  // `response` as an insight would surface a backend error string to the user
  // as if it were genuine AI content (the #932/#939 "error-as-data" class).
  // Treat the error flag — and the zero-hop / zero-evidence degenerate shape —
  // as "no real brief" so the honest error/empty states below take over.
  const ragHasError = !!briefResponse?.error;
  const ragIsDegenerate =
    !!briefResponse &&
    briefResponse.hop_count === 0 &&
    (briefResponse.evidence?.length ?? 0) === 0;
  const ragHasRealAnswer =
    !!briefResponse?.response && !ragHasError && !ragIsDegenerate;

  // The RAG response becomes a single real section — nothing is spliced in,
  // and only when it is a genuine grounded answer (not an error payload and
  // not a zero-hop / zero-evidence degenerate result).
  const ragSections: BriefSection[] | null = ragHasRealAnswer
    ? [
        {
          title: 'AI-Generated Insight',
          content: briefResponse.response,
          sourceLabel:
            briefResponse.hop_count > 0
              ? `${briefResponse.hop_count} retrieval hop${briefResponse.hop_count === 1 ? '' : 's'}`
              : undefined,
        },
      ]
    : null;

  const sections: BriefSection[] = crystallizedSections ?? ragSections ?? [];

  // An in-band error (HTTP 200 with `error` set) is a real failure even
  // though TanStack's transport-level `briefError` is null. Fold it into the
  // error state so the user sees an honest "unable to generate" message rather
  // than the raw error string masquerading as an insight.
  const ragErrorMessage = ragHasError ? briefResponse?.error ?? null : null;

  // Generate initial brief on mount
  useEffect(() => {
    generateBrief({
      query: `Generate an executive brief summary for ${brand}. Include key performance trends, emerging opportunities, and risk alerts.`,
    });
  }, [brand, generateBrief]);

  // Track when a real (non-error) response arrives. A 200 carrying an in-band
  // error is NOT a successful update, so it must not stamp "Last updated".
  // Depend on the response OBJECT (not just the derived boolean) so a SECOND
  // real refresh after an earlier real answer still re-stamps the timestamp —
  // the boolean alone would stay `true` across refreshes and skip the update.
  useEffect(() => {
    if (ragHasRealAnswer) {
      setLastUpdated(new Date());
    }
  }, [briefResponse, ragHasRealAnswer]);

  const handleRefresh = () => {
    generateBrief({
      query: `Generate an executive brief summary for ${brand}. Include key performance trends, emerging opportunities, and risk alerts.`,
    });
  };

  const hasAnyError = !!briefError || ragHasError;
  const showError = !isGenerating && sections.length === 0 && hasAnyError;
  const showEmpty = !isGenerating && sections.length === 0 && !hasAnyError;

  return (
    <Card className={cn('bg-[var(--color-card)] border-[var(--color-border)]', className)}>
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <div className="p-2 rounded-lg bg-purple-500/10">
              <Brain className="h-5 w-5 text-purple-500" />
            </div>
            <div>
              <CardTitle className="text-base font-semibold">Executive AI Brief</CardTitle>
              <p className="text-xs text-[var(--color-muted-foreground)]">
                Powered by E2I Cognitive Engine
              </p>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <Badge variant="outline" className="text-xs">
              <Sparkles className="h-3 w-3 mr-1" />
              AI-Generated
            </Badge>
            <Button
              variant="ghost"
              size="icon"
              onClick={handleRefresh}
              disabled={isGenerating}
              className="h-8 w-8"
            >
              <RefreshCw className={cn('h-4 w-4', isGenerating && 'animate-spin')} />
            </Button>
          </div>
        </div>
      </CardHeader>
      <CardContent className="space-y-4">
        {/* Loading State */}
        {isGenerating && (
          <div className="flex items-center justify-center py-8">
            <div className="flex items-center gap-3 text-[var(--color-muted-foreground)]">
              <RefreshCw className="h-5 w-5 animate-spin" />
              <span className="text-sm">Generating AI brief...</span>
            </div>
          </div>
        )}

        {/* Error State — labeled, never replaced with fabricated sections */}
        {showError && (
          <div className="flex items-start gap-2 p-3 rounded-lg bg-rose-500/5 border border-rose-500/20">
            <AlertTriangle className="h-4 w-4 text-rose-500 mt-0.5" />
            <div className="text-xs text-[var(--color-muted-foreground)]">
              <span className="font-medium text-rose-600">Unable to generate brief:</span>{' '}
              {briefError?.message ?? ragErrorMessage ?? 'Cognitive engine unavailable'}
            </div>
          </div>
        )}

        {/* Honest empty state */}
        {showEmpty && (
          <EmptyState
            title="No executive brief available"
            description={`No crystallized insights exist for ${brand} yet and the cognitive engine has not returned a brief. Use refresh to try again.`}
          />
        )}

        {/* Brief Sections — real content only */}
        {!isGenerating && sections.length > 0 && (
          <div className="space-y-4">
            {sections.map((section, idx) => (
              <div
                key={idx}
                className="p-3 rounded-lg bg-[var(--color-muted)]/30 border border-[var(--color-border)]"
              >
                <div className="flex items-center justify-between mb-2">
                  <h4 className="text-sm font-medium text-[var(--color-foreground)]">
                    {section.title}
                  </h4>
                  {section.sourceLabel && (
                    <Badge variant="outline" className="text-xs">
                      {section.sourceLabel}
                    </Badge>
                  )}
                </div>
                <p className="text-sm text-[var(--color-muted-foreground)] leading-relaxed">
                  {section.content}
                </p>
              </div>
            ))}
          </div>
        )}

        {/* Footer */}
        <div className="flex items-center justify-between pt-2 border-t border-[var(--color-border)]">
          <div className="flex items-center gap-1 text-xs text-[var(--color-muted-foreground)]">
            <Clock className="h-3 w-3" />
            <span>
              {lastUpdated
                ? `Last updated: ${lastUpdated.toLocaleTimeString()}`
                : 'Not yet generated'}
            </span>
          </div>
          <div
            className={cn(
              'flex items-center gap-1 text-xs',
              sections.length > 0
                ? 'text-emerald-600'
                : 'text-[var(--color-muted-foreground)]'
            )}
          >
            <CheckCircle2 className="h-3 w-3" />
            <span>
              {sections.length} insight{sections.length === 1 ? '' : 's'} generated
            </span>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}

export default ExecutiveAIBrief;
