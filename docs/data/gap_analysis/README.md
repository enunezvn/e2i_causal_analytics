# Optum-RWD Gap-Feature Q&A

Annotates the data scientist's questions about deriving E2I fields from Optum
Real-World Data with codebase-grounded responses.

## Files

| File | Purpose |
|------|---------|
| `build_gap_feature_responses.py` | Reads the original workbook, adds column `G` (Claude analysis), writes the annotated workbook. Idempotent — re-running overwrites. |
| `Questions_around_Gap_features_annotated.xlsx` | Output workbook with the new `G` column. |

## What changed

The original workbook had 5 columns plus an orphan `F7` note:

| Col | Original | After |
|-----|----------|-------|
| A   | E2I Field                  | (unchanged) |
| B   | E2I Purpose                | (unchanged) |
| C   | Notes / Transform Required | (unchanged) |
| D   | Questions                  | (unchanged) |
| E   | Comments                   | (unchanged) |
| F   | _(orphan F7 only)_         | Header now reads "Additional Comments" so the stranded `F7` engagement_score weighting note is identifiable. |
| G   | _(new)_                    | "Claude Code Analysis & Recommendations" — per-row response. |

Each `G` cell answers the data scientist's specific question or comment using
direct references to:

- The Optum converter (`scripts/convert_optum_rwd.py`) and its constants
  (`LOOKBACK_DAYS=180`, `BIOLOGIC_DISCONT_GAP_DAYS=90`, `CSU_BIOLOGIC_*`,
  `NON_TARGET_DRUG_CLASSES`, etc.).
- The canonical schema
  (`database/core/e2i_ml_complete_v3_schema.sql`) and its enums + check
  constraints — e.g. flagging the 5-stage `journey_stage_type` mismatch
  against the proposed 7-stage funnel, and the `priority_tier` 1-5 range
  vs the proposed 3-tier mapping.
- The conversion guide (`docs/OPTUM_CONVERSION.md`) for known approximations
  (Charlson/Elixhauser).
- The KPI calculators (`src/kpi/calculators/brand_specific.py`) for the
  canonical `treatment_response` vocabulary.
- The feedback-loop wiring (`database/migrations/006_feedback_loop_infrastructure.sql`,
  `src/tasks/feedback_loop_tasks.py`) for predictions vs ground truth.

## Per-row response summary

| # | E2I Field | Verdict |
|---|-----------|---------|
| 2 | `patient_journeys.journey_stage` | Parameters OK with caveats (use 90d gap for biologics). Schema enum needs extension (5 → 7 stages). |
| 3 | `patient_journeys.risk_score` | Use existing §7 features + therapy-related target (`discontinued_180d`). Replace Charlson/Elixhauser approximations with Quan mapping for production. |
| 4 | `patient_journeys.data_quality_score` | Hybrid claim-level + patient-rollup. Recommended weights: dx 0.40 / proc 0.25 / cost 0.20 / enrollment 0.15. |
| 5 | `hcp_profiles.priority_tier` | Schema is 1-5; map 3-tier proposal onto 5-bin storage. Use ZIP3 (not ZIP5) + 12-month rolling. |
| 6 | `hcp_profiles.adoption_category` | Filter on NDC prefix + HCPCS, not brand string. Xolair launch 2014-03-21 (CSU). Dupixent not approved for CSU. Use Rogers cumulative-share thresholds, not equal quartiles. |
| 7 | `hcp_profiles.engagement_score` | No Veeva data in Optum. Adopt user's weighting for the Veeva pathway; leave NULL for Tier-1 Optum-only. |
| 8 | `triggers.*` | Table exists with full delivery/acceptance tracking. No production generator yet — leave empty in cohort build. |
| 9 | `business_metrics.market_share` | Numerator = brand TRx, denominator = full CSU basket already enumerated in converter. Optum is single-payer, not all-payer — document the limitation. |
| 10 | `treatment_events.treatment_response` | No lab biomarker for CSU. Use claim-pattern proxies (steroid rescue, ED visit, treatment switch). |
| 11 | `hcp_profiles.influence_network` | Build shared-patient clique network from Optum (claims-derivable). Replace with vendor KOL data in Tier 2. |
| 12 | `patient_journeys.source_timestamp` | Optum `extract_ym` is month-granular; derive `source_timestamp` from `LAST_DAY(extract_ym)`. |
| 13 | `hcp_intent_surveys.*` | Not in Optum. Leave empty until market-research integration ships. |
| 14 | `user_sessions.*` | Platform telemetry, requires frontend instrumentation. Out of Optum scope. |
| 15 | `causal_paths.*` | Output of `causal_impact` Tier-2 agent. Downstream of Optum conversion. |
| 16 | `ml_predictions.*` | Output of `prediction_synthesizer` Tier-4 agent. Confirms feedback-loop wiring with Optum targets. |
| 17 | `patient_journeys.payer_category` | Derive from `bus + product + health_exch + lis_dual`. Specialty pharmacy channel requires NPI taxonomy lookup against NPPES — not in standard Optum extract. |

## How to regenerate

```bash
python docs/data/gap_analysis/build_gap_feature_responses.py
```

The script reads from the upload path
(`/root/.claude/uploads/.../Questions_around_Gap_features.xlsx`) and writes the
annotated copy to `Questions_around_Gap_features_annotated.xlsx` in this
directory.
