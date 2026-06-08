/**
 * Executive Insights types — mirror the backend ExecutiveInsightResponse
 * (src/api/routes/executive_insights.py). UI consumes the subset below;
 * extra backend fields are tolerated by the api-client (no .strict()).
 *
 * @module types/executive-insights
 */

export interface ExecutiveInsight {
  insight_id: string;
  title: string;
  narrative: string;
  brand: string;
  region?: string | null;
  kpi?: string | null;
  crystallized_at: string;
  source_count: number;
  effect_size?: number | null;
  effect_direction?: 'positive' | 'negative' | 'null' | null;
  limitations?: string | null;
  recommended_next_analysis?: string | null;
}

export interface ListExecutiveInsightsParams {
  brand?: string;
  region?: string;
  include_recalled?: boolean;
  limit?: number;
}
