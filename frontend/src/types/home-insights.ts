/**
 * Home Insights Types
 * ===================
 *
 * Minimal, Home-LOCAL type for the executive-insights source consumed by the
 * Home page's "Agent Insights" tile. Kept SEPARATE from
 * `src/types/executive-insights.ts` (which is owned by PR #798) to avoid a
 * merge conflict — do NOT import from or edit that file here.
 *
 * Mirrors the fields of the backend `ExecutiveInsightResponse`
 * (src/api/routes/executive_insights.py) that the Home tile actually renders.
 *
 * @module types/home-insights
 */

/** A crystallized cross-agent executive insight (Home-local subset). */
export interface HomeExecutiveInsight {
  insight_id: string;
  title: string;
  narrative: string;
  brand: string;
  region?: string | null;
  kpi?: string | null;
  /** ISO 8601 timestamp the insight was crystallized. */
  crystallized_at: string;
  source_count?: number;
  effect_size?: number | null;
  /** 'positive' | 'negative' | 'null' */
  effect_direction?: string | null;
  recommended_next_analysis?: string | null;
}
