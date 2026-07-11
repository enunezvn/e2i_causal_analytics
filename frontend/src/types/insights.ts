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

/** One hypothetical what-if row: the entered profile + the model's score. */
export interface PredictiveWhatIfInsightRequest {
  model_version: string;
  features: Record<string, unknown>;
  probability: number;
  confidence?: number | null;
  cohort_mean?: number | null;
  n_scored?: number | null;
  top_drivers: Array<{ feature: string; importance: number }>;
}

export interface AllocationMove {
  entity_id: string;
  change_percentage?: number | null;
  change?: number | null;
}

export interface ResourceInsightRequest {
  optimization_summary: string;
  recommendations: string[];
  projected_lift_pct?: number | null;
  solver_status?: string | null;
  objective?: string | null;
  brand?: string | null;
  resource_type?: string | null;
  entity_count?: number | null;
  total_budget?: number | null;
  /** Actual optimized spend — maximize_roi can intentionally deploy less than
   *  the budget (marginal return below hurdle) */
  total_spend?: number | null;
  top_increases?: AllocationMove[];
  top_decreases?: AllocationMove[];
  synthetic?: boolean;
}

export interface TreatmentEffectInsightRequest {
  cohort: string;
  brand: string;
  treatment_var: string;
  outcome_var: string;
  confounders: string[];
  ate: number;
  ci_lower?: number | null;
  ci_upper?: number | null;
  p_value?: number | null;
  n: number;
  estimator?: string | null;
}

export interface ExecutiveBriefInsightRequest {
  /**
   * Brand only: the grounding figures are derived SERVER-SIDE from the latest
   * completed gap analysis (same read path as GET /gaps/opportunities).
   * Caller-posted figures are not accepted — they would let any authenticated
   * caller mint a grounded-looking brief from arbitrary numbers (codex PR-5
   * round 3).
   */
  brand: string;
}

export interface HTEInsightRequest {
  /**
   * analysis_id only: the grounding figures are derived SERVER-SIDE from the
   * persisted segment-analysis record (same trust boundary as
   * ExecutiveBriefInsightRequest — caller-posted figures are not accepted).
   */
  analysis_id: string;
}

export interface FeedbackLearningInsightRequest {
  /**
   * Days only: all grounding (persisted cycles/patterns/updates + real
   * feedback inflow) is derived SERVER-SIDE — caller-posted figures are not
   * accepted (same trust boundary as ExecutiveBriefInsightRequest).
   */
  days?: number;
}

export interface DigitalTwinInsightRequest {
  /**
   * Brand (+ twin type) only: the grounding (twin-model inventory, simulation
   * history, per-intervention effect coverage) is derived SERVER-SIDE —
   * caller-posted figures are not accepted (same trust boundary as
   * ExecutiveBriefInsightRequest).
   */
  brand: string;
  twin_type?: string;
}

export interface HomeKpiInsightRequest {
  /**
   * Scope only: the KPI figures are recomputed SERVER-SIDE under the same
   * brand/region context the dashboard's batch endpoint uses — caller-posted
   * figures are not accepted (same trust boundary as
   * ExecutiveBriefInsightRequest).
   */
  brand: string;
  /** Lowercase US-census region, or null/omitted for the All-US portfolio view. */
  region?: string | null;
}

export interface ExperimentsInsightRequest {
  /**
   * Scope only: the per-channel A/B effects are recomputed SERVER-SIDE from
   * ml_experiments × ab_experiment_results — caller-posted figures are not
   * accepted (same trust boundary as ExecutiveBriefInsightRequest).
   */
  brand: string;
  /** Provenance opt-in mirroring the monitor sweep's (#894). */
  include_synthetic?: boolean;
}
