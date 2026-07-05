/**
 * Executive-brief request builder (T7 → PR-5 rewire).
 *
 * History: T7a fixed a context-starved RAG query by interpolating the brand's
 * real `/gaps/opportunities` figures into the `POST /api/cognitive/rag` prompt.
 * PR-5 replaces that client-assembled prompt with a dedicated server-side DSPy
 * endpoint (`POST /api/insights/executive-brief`) whose grounding, caveats and
 * fallback are built server-side from the SAME figures — the strategic
 * distillation the /ai-insights review asked for (finding 1: the brief read as
 * a description, not a decision aid). This pure helper now maps the live
 * opportunities feed onto that endpoint's request shape.
 *
 * Honest-state contract unchanged: when there is NO real signal (no surfaced
 * opportunities AND nothing suppressed) it returns `null` and the caller must
 * not call the endpoint at all — an LLM riff over zero figures is fabrication.
 * A brand whose opportunities were ALL suppressed (hidden below break-even)
 * still carries real signal — the honest brief is "don't invest now" (mirrors
 * the T6 gap-analyzer narrative).
 *
 * `/api/cognitive/rag` itself is intentionally KEPT: it is the cognitive
 * engine's general query endpoint (chatbot / ad-hoc RAG); this card simply no
 * longer consumes it.
 *
 * @module lib/insights/brief-request
 */

import type { OpportunityListResponse } from '@/types/gaps';
import type { ExecutiveBriefInsightRequest } from '@/types/insights';

/**
 * Map the live `/gaps/opportunities` response onto the executive-brief
 * insight request, or return `null` when there is no real signal to distill.
 *
 * Free-text bounding (action/segment caps) and the top-5 rank cap happen
 * server-side in `src/insights/executive_brief.py`, so this mapping stays a
 * faithful 1:1 of what the feed returned.
 */
export function buildExecutiveBriefRequest(
  brand: string,
  context?: OpportunityListResponse
): ExecutiveBriefInsightRequest | null {
  const opps = context?.opportunities ?? [];
  const suppressed = context?.suppressed_count ?? 0;
  if (!context || (opps.length === 0 && suppressed === 0)) {
    return null;
  }
  return {
    brand,
    total_addressable_value: context.total_addressable_value ?? null,
    quick_wins_count: context.quick_wins_count ?? 0,
    steady_plays_count: context.steady_plays_count ?? 0,
    strategic_bets_count: context.strategic_bets_count ?? 0,
    suppressed_count: suppressed,
    opportunities: opps.map((o) => ({
      rank: o.rank,
      recommended_action: o.recommended_action,
      expected_roi: o.roi_estimate.expected_roi,
      revenue_impact: o.roi_estimate.estimated_revenue_impact,
      gap_metric: o.gap.metric,
      gap_percentage: o.gap.gap_percentage,
      segment_value: o.gap.segment_value,
      implementation_difficulty: o.implementation_difficulty,
    })),
  };
}

export default buildExecutiveBriefRequest;
