/**
 * Executive-brief query builder (T7).
 *
 * The Executive AI Brief was STARVED of context: its RAG query was literally
 * "Generate an executive brief summary for ${brand}. Include key performance
 * trends, emerging opportunities, and risk alerts." with no KPI/ROI/gap numbers
 * — so the cognitive engine had nothing concrete to ground the brief in and the
 * result read generic. The page already holds the real ROI/gap figures (the
 * sibling Priority-Actions card loads them from `/gaps/opportunities`).
 *
 * This pure helper interpolates those REAL figures into the query. The RAG
 * `user_query` is the primary synthesis input (it drives memory retrieval and
 * is fed verbatim to `EvidenceSynthesisSignature.user_query`), so grounding it
 * in real numbers materially enriches the brief — no fabrication, no mocking.
 *
 * When no real context is available (loading error, empty feed) it falls back
 * to the basic prompt rather than inventing numbers — the honest-state contract.
 *
 * @module lib/insights/brief-query
 */

import type { OpportunityListResponse, PrioritizedOpportunity } from '@/types/gaps';

// The cognitive-RAG endpoint validates `query` at 1-5000 chars
// (frontend/src/types/cognitive.ts). `recommended_action` / `segment_value`
// are unbounded backend free-text, so cap the assembled query defensively with
// headroom to avoid a 422 on a verbose opportunities payload.
const MAX_QUERY_CHARS = 4800;

/** Compact USD formatting, mirroring the gap drill-down ($5.0M / $300K / $42). */
function money(value: number): string {
  if (!Number.isFinite(value)) return '$0';
  if (Math.abs(value) >= 1_000_000) return `$${(value / 1_000_000).toFixed(1)}M`;
  if (Math.abs(value) >= 1_000) return `$${(value / 1_000).toFixed(0)}K`;
  return `$${Math.round(value)}`;
}

function plural(n: number, word: string): string {
  return `${n} ${word}${n === 1 ? '' : 's'}`;
}

/** The context-free fallback — preserves the original brief intent, no numbers. */
function basicQuery(brand: string): string {
  return (
    `Generate an executive brief summary for ${brand}. ` +
    `Include key performance trends, emerging opportunities, and risk alerts.`
  );
}

/** One human-readable line describing a single opportunity for the prompt. */
function opportunityLine(opp: PrioritizedOpportunity): string {
  const roi = opp.roi_estimate.expected_roi.toFixed(1);
  const rev = money(opp.roi_estimate.estimated_revenue_impact);
  const gapPct = opp.gap.gap_percentage.toFixed(0);
  const metric = opp.gap.metric.toUpperCase();
  const seg = opp.gap.segment_value;
  const effort = opp.implementation_difficulty;
  return (
    `${opp.rank}. ${opp.recommended_action} — ${roi}× ROI, ${rev} revenue impact, ` +
    `closing a ${gapPct}% ${metric} gap in ${seg} (${effort} effort).`
  );
}

/**
 * Build the RAG query for the executive brief, grounded in the brand's real
 * gap-analysis figures when available, otherwise the basic prompt.
 *
 * @param brand   The brand the brief is for.
 * @param context The live `/gaps/opportunities` response (or undefined).
 */
export function buildExecutiveBriefQuery(
  brand: string,
  context?: OpportunityListResponse
): string {
  const opps = context?.opportunities ?? [];
  const suppressed = context?.suppressed_count ?? 0;
  // No real signal at all → do not fabricate; fall back to the basic prompt.
  // A brand whose opportunities were ALL suppressed (none surfaced, but some
  // hidden below break-even) still carries real signal worth grounding on —
  // mirrors the T6 gap-analyzer honest narrative ("all below break-even").
  if (!context || (opps.length === 0 && suppressed === 0)) {
    return basicQuery(brand);
  }

  const top = [...opps].sort((a, b) => a.rank - b.rank).slice(0, 3);
  const mix = [
    plural(context.quick_wins_count ?? 0, 'quick win'),
    plural(context.steady_plays_count ?? 0, 'steady play'),
    plural(context.strategic_bets_count ?? 0, 'strategic bet'),
  ].join(', ');

  const lines: string[] = [
    basicQuery(brand),
    '',
    `Ground the brief in these real gap-analysis figures for ${brand}:`,
    `- Total addressable opportunity value: ${money(context.total_addressable_value ?? 0)}.`,
    `- Opportunity mix: ${mix}.`,
  ];

  if (suppressed > 0) {
    const noun = suppressed === 1 ? 'opportunity was' : 'opportunities were';
    lines.push(
      `- ${suppressed} low-value ${noun} suppressed (below break-even) and excluded.`
    );
  }

  // Only enumerate surfaced opportunities — never fabricate a list when all
  // were suppressed.
  if (top.length > 0) {
    lines.push('- Top opportunities by ROI:');
    for (const opp of top) {
      lines.push(`  ${opportunityLine(opp)}`);
    }
  }

  const query = lines.join('\n');
  return query.length > MAX_QUERY_CHARS ? query.slice(0, MAX_QUERY_CHARS) : query;
}

export default buildExecutiveBriefQuery;
