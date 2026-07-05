/**
 * Executive AI Brief Component
 * ============================
 *
 * Displays an AI-generated executive brief of key insights.
 *
 * Data sources, in order of preference (real data only):
 * 1. Crystallized executive insights for the brand
 *    (`GET /api/executive-insights`, M5 REWIRE).
 * 2. The DSPy strategic distillation (`POST /api/insights/executive-brief`),
 *    grounded server-side in the brand's REAL gap-analysis figures — the same
 *    `/gaps/opportunities` feed the sibling Priority-Actions card renders.
 *
 * PR-5 rewire (review finding 1: the brief read as a *description*, not a
 * strategic distillation): the previous fallback posted a client-assembled
 * prompt to `POST /api/cognitive/rag`. That endpoint is intentionally KEPT —
 * it is the cognitive engine's general query surface (chatbot / ad-hoc RAG) —
 * but this card now uses the dedicated insights endpoint, which structures the
 * output as a decision aid (highest-impact decision, quantified stakes, ranked
 * action sequence, actionability judgment, suppression caveat) with an honest
 * deterministic fallback when the LM is unavailable.
 *
 * Honest-state contract: real sections, an explicit empty state, or a labeled
 * error. When the opportunities feed has NO real signal (nothing surfaced and
 * nothing suppressed) the endpoint is not called at all — an LLM riff over
 * zero figures would be fabrication. SAMPLE_BRIEF (fabricated $2.3M / 847-HCP
 * sections) was deleted long ago and must never return.
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
import { useExecutiveBriefInsight } from '@/hooks/api/use-insights';
import { useExecutiveInsights } from '@/hooks/api/use-executive-insights';
import { useOpportunities } from '@/hooks/api';
import { buildExecutiveBriefRequest } from '@/lib/insights/brief-request';

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
  /** Grounded, decision-ready takeaways (DSPy path only). */
  takeaways?: string[];
}

// =============================================================================
// MAIN COMPONENT
// =============================================================================

export function ExecutiveAIBrief({ className, brand = 'Remibrutinib' }: ExecutiveAIBriefProps) {
  const [lastUpdated, setLastUpdated] = useState<Date | null>(null);

  // Strategic distillation from the dedicated insights endpoint (real backend
  // response; server-side grounding + honest deterministic fallback).
  const {
    mutate: generateBrief,
    reset: resetBrief,
    data: briefResponse,
    variables: briefVariables,
    error: briefRawError,
    isPending: isGenerating,
  } = useExecutiveBriefInsight();

  // Attribution guard (codex PR-5 round 1 HIGH): reset() does NOT cancel an
  // in-flight mutation. If brand A's request resolves after a switch to brand
  // B — and B never fires a new request (e.g. no signal) — the hook still
  // observes A's mutation and its late data/error would render under B. Only
  // consume a result whose REQUEST brand matches the brand on screen.
  const briefInsight =
    briefResponse && briefVariables?.brand === brand ? briefResponse : null;
  const briefError =
    briefRawError && briefVariables?.brand === brand ? briefRawError : null;

  // Real crystallized insights for this brand (M5 REWIRE). When present,
  // these take precedence over the insights-endpoint path.
  const { data: crystallized } = useExecutiveInsights(brand);

  // The brand's real gap-analysis figures (the same `/gaps/opportunities` feed
  // the sibling Priority-Actions card uses) — the ONLY grounding the brief
  // request carries; no fabrication.
  const { data: oppData, isLoading: oppLoading } = useOpportunities({
    brand,
    limit: 5,
  });

  // The grounded request, or null when there is no real signal to distill —
  // in which case the endpoint is never called and the honest empty state
  // renders instead.
  const briefRequest = useMemo(
    () => buildExecutiveBriefRequest(brand, oppData),
    [brand, oppData]
  );
  // Content-stable key so the generate effect fires exactly once per DISTINCT
  // request (mirrors the previous string-query dependency semantics).
  const briefRequestKey = useMemo(
    () => (briefRequest ? JSON.stringify(briefRequest) : null),
    [briefRequest]
  );

  const crystallizedSections: BriefSection[] | null =
    crystallized && crystallized.length > 0
      ? crystallized.slice(0, 3).map((ins) => ({
          title: ins.title,
          content: ins.narrative,
          sourceLabel: `${ins.source_count} source${ins.source_count === 1 ? '' : 's'}`,
        }))
      : null;

  // The insights endpoint reports degradation IN-BAND but HONESTLY: a 200
  // always carries a real, grounded `insight` (LLM distillation or the
  // labelled deterministic fallback) — never an error string dressed as
  // content. `is_fallback` drives the source label so the user can tell them
  // apart at a glance.
  const briefSections: BriefSection[] | null = briefInsight?.insight
    ? [
        {
          title: 'Strategic Brief',
          content: briefInsight.insight,
          sourceLabel: briefInsight.is_fallback
            ? 'Factual summary (LLM unavailable)'
            : 'AI distillation of live gap-analysis figures',
          takeaways: briefInsight.key_takeaways,
        },
      ]
    : null;

  const sections: BriefSection[] = crystallizedSections ?? briefSections ?? [];

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
  // no brief has been generated yet), clear the previous brand's brief AND its
  // "last updated" stamp immediately so neither can be displayed under the new
  // brand while it (re)generates (the grounded fire below is gated on the
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

  // Generate the brief once the opportunity context has SETTLED, so the
  // request is grounded in real figures. Fires exactly once per distinct
  // request (the key encodes every figure). No signal -> no call: the honest
  // empty state below is the truthful answer, not an ungrounded LLM riff.
  useEffect(() => {
    if (oppLoading || !briefRequestKey) return;
    generateBrief(JSON.parse(briefRequestKey));
  }, [briefRequestKey, oppLoading, generateBrief]);

  // Track when a real response arrives. Depend on the response OBJECT (not a
  // derived boolean) so a SECOND refresh after an earlier answer still
  // re-stamps the timestamp.
  useEffect(() => {
    if (briefInsight?.insight) {
      setLastUpdated(new Date());
    }
  }, [briefInsight]);

  const handleRefresh = () => {
    if (briefRequestKey) {
      generateBrief(JSON.parse(briefRequestKey));
    }
  };

  // "Busy" covers the insight call in flight, the opportunities feed still
  // loading (grounded fire deferred), AND the single render after a brand
  // switch before the reset effect clears the prior brief — in all three we
  // show the loading state, never stale content or a premature empty/error
  // state.
  const isBusy = isGenerating || oppLoading || brandChanged;

  // What the user actually sees. When busy, nothing real is shown yet, so the
  // body AND footer must not surface the prior brand's sections/count.
  const displaySections: BriefSection[] = isBusy ? [] : sections;

  const hasAnyError = !!briefError;
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
                Strategic distillation grounded in live gap-analysis figures
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
              {briefError?.message ?? 'Insights service unavailable'}
            </div>
          </div>
        )}

        {/* Honest empty state — no crystallized insights and no gap-analysis
            signal to distill (the endpoint is not called without signal). */}
        {showEmpty && (
          <EmptyState
            title="No executive brief available"
            description={`No crystallized insights exist for ${brand} yet and there is no gap-analysis signal to distill — run a gap analysis to generate a brief.`}
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
                {section.takeaways && section.takeaways.length > 0 && (
                  <ul className="mt-2 space-y-1">
                    {section.takeaways.map((t, ti) => (
                      <li
                        key={ti}
                        className="flex items-start gap-1.5 text-sm text-[var(--color-muted-foreground)]"
                      >
                        <CheckCircle2 className="h-3.5 w-3.5 mt-0.5 shrink-0 text-purple-500" />
                        <span>{t}</span>
                      </li>
                    ))}
                  </ul>
                )}
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
