export interface GroundingChip {
  label: string;
  value: string;
}

export interface StrategicInsightResponse {
  insight: string;
  key_takeaways: string[];
  grounding: GroundingChip[];
  is_fallback: boolean;
  generated_at: string;
  provenance: string;
}

export interface KGInsightRequest {
  brand: string;
  curated_only?: boolean;
}

export interface ModelPerfInsightRequest {
  model_version: string;
}

export interface CausalInsightRequest {
  brand: string;
  grain: string;
  effects: Array<{
    treatment: string;
    outcome: string;
    ate: number;
    ate_ci_lower?: number;
    ate_ci_upper?: number;
    status?: string;
    selected_estimator?: string;
  }>;
}

export interface PredictiveInsightRequest {
  model_version: string;
  n_scored: number;
  mean_prob: number;
  top_targets: Array<{ entity_id: string; probability: number }>;
  top_drivers: Array<{ feature: string; importance: number }>;
}

export interface ResourceInsightRequest {
  optimization_summary: string;
  recommendations: string[];
  projected_lift_pct?: number | null;
  solver_status?: string | null;
}
