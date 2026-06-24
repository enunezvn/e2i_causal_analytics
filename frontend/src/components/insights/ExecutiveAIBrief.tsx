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

import { useEffect, useMemo, useRef, useState } from 'react';
import { Brain, RefreshCw, Sparkles, Clock, CheckCircle2, AlertTriangle } from 'lucide-react';
import { cn } from '@/lib/utils';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { EmptyState } from '@/components/ui/EmptyState';
import { useCognitiveRAG } from '@/hooks/api/use-cognitive';
import { useExecutiveInsights } from '@/hooks/api/use-executive-insights';
import { useOpportunities } from '@/hooks/api';
import { buildExecutiveBriefQuery } from '@/lib/insights/brief-query';

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
    reset: resetBrief,
    data: briefResponse,
    error: briefError,
    isPending: isGenerating,
  } = useCognitiveRAG();

  // Real crystallized insights for this brand (M5 REWIRE). When present,
  // these take precedence over the cognitive-RAG path.
  const { data: crystallized } = useExecutiveInsights(brand);

  // T7a: the cognitive-RAG fallback was STARVED of context — its query carried
  // no KPI/ROI/gap numbers, so the brief read generic. Pull the brand's real
  // gap-analysis figures (the same `/gaps/opportunities` feed the sibling
  // Priority-Actions card uses) and ground the query in them. The RAG
  // `user_query` is the primary synthesis input, so this materially enriches the
  // brief with real data — no fabrication.
  const { data: oppData, isLoading: oppLoading } = useOpportunities({
    brand,
    limit: 5,
  });

  // The grounded query (or the basic prompt when no real context is available).
  const briefQuery = useMemo(
    () => buildExecutiveBriefQuery(brand, oppData),
    [brand, oppData]
  );

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

  // Synchronous (render-time) detection of a brand switch. The reset below is a
  // passive effect that runs AFTER paint, so on a CACHED-opportunities switch
  // the prior brand's brief/footer would paint for one frame before it clears.
  // `brandChanged` folds into `isBusy` so that frame shows the busy state
  // instead of stale content (codex round-2 HIGH).
  const prevBrandRef = useRef(brand);
  const brandChanged = prevBrandRef.current !== brand;
  useEffect(() => {
    prevBrandRef.current = brand;
  }, [brand]);

  // On a brand CHANGE (not the initial mount — reset-on-mount is a no-op since
  // no brief has been generated yet), clear the previous brand's RAG result AND
  // its "last updated" stamp immediately so neither can be displayed under the
  // new brand while it (re)generates (the grounded fire below is gated on the
  // opportunities feed settling). Without this, brand A's brief + footer would
  // linger on screen — a stale-attribution honest-state violation
  // (codex round-1 HIGH for the body, round-2 HIGH for the footer).
  const didMountRef = useRef(false);
  useEffect(() => {
    if (!didMountRef.current) {
      didMountRef.current = true;
      return;
    }
    resetBrief();
    setLastUpdated(null);
  }, [brand, resetBrief]);

  // Generate the brief once the opportunity context has SETTLED, so the first
  // request is grounded in real figures rather than fired context-free. The
  // effect re-runs whenever `briefQuery` changes (brand switch or the figures
  // load), firing exactly one grounded request per distinct query. On an
  // error/empty feed the query falls back to the basic prompt — honest
  // degradation, never fabricated numbers.
  useEffect(() => {
    if (oppLoading) return;
    generateBrief({ query: briefQuery });
  }, [briefQuery, oppLoading, generateBrief]);

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
    generateBrief({ query: briefQuery });
  };

  // "Busy" covers the RAG call in flight, the opportunities feed still loading
  // (grounded fire deferred), AND the single render after a brand switch before
  // the reset effect clears the prior brief — in all three we show the loading
  // state, never stale content or a premature empty/error state.
  const isBusy = isGenerating || oppLoading || brandChanged;

  // What the user actually sees. When busy, nothing real is shown yet, so the
  // body AND footer must not surface the prior brand's sections/count.
  const displaySections: BriefSection[] = isBusy ? [] : sections;

  const hasAnyError = !!briefError || ragHasError;
  const showError = !isBusy && displaySections.length === 0 && hasAnyError;
  const showEmpty = !isBusy && displaySections.length === 0 && !hasAnyError;

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
              disabled={isBusy}
              className="h-8 w-8"
            >
              <RefreshCw className={cn('h-4 w-4', isBusy && 'animate-spin')} />
            </Button>
          </div>
        </div>
      </CardHeader>
      <CardContent className="space-y-4">
        {/* Loading State */}
        {isBusy && (
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
        {!isBusy && displaySections.length > 0 && (
          <div className="space-y-4">
            {displaySections.map((section, idx) => (
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

        {/* Footer — reflects only the CURRENT brand's displayed brief. While
            busy (incl. the brand-switch frame) the stamp/count are not shown so
            the prior brand's state can never leak through. */}
        <div className="flex items-center justify-between pt-2 border-t border-[var(--color-border)]">
          <div className="flex items-center gap-1 text-xs text-[var(--color-muted-foreground)]">
            <Clock className="h-3 w-3" />
            <span>
              {!isBusy && lastUpdated
                ? `Last updated: ${lastUpdated.toLocaleTimeString()}`
                : 'Not yet generated'}
            </span>
          </div>
          <div
            className={cn(
              'flex items-center gap-1 text-xs',
              displaySections.length > 0
                ? 'text-emerald-600'
                : 'text-[var(--color-muted-foreground)]'
            )}
          >
            <CheckCircle2 className="h-3 w-3" />
            <span>
              {displaySections.length} insight{displaySections.length === 1 ? '' : 's'} generated
            </span>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}

export default ExecutiveAIBrief;
