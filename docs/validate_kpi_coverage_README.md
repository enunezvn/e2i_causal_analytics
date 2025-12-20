# E2I KPI Coverage Validator

**Version:** 3.0.0  
**Purpose:** Validates that all 46 E2I KPIs are calculable from the V3 schema

---

## Overview

The `validate_kpi_coverage.py` script ensures 100% KPI calculability by verifying that all required tables, columns, and views exist in the E2I V3 schema. It can run in two modes:

1. **Dry-run mode** — Schema-only validation (no database connection required)
2. **Live mode** — Validates against a live Supabase database

---

## Quick Start

```bash
# Basic validation (dry-run)
python validate_kpi_coverage.py --dry-run

# Verbose output showing each KPI
python validate_kpi_coverage.py --dry-run --verbose

# Generate markdown report
python validate_kpi_coverage.py --dry-run --output kpi_report.md

# Live database validation
export SUPABASE_URL="https://your-project.supabase.co"
export SUPABASE_KEY="your-service-key"
python validate_kpi_coverage.py --verbose
```

---

## Installation

### Requirements

- Python 3.8+
- No external dependencies for dry-run mode

### Optional Dependencies

For live database validation:

```bash
pip install supabase
```

### Environment Variables (for live mode)

| Variable | Description |
|----------|-------------|
| `SUPABASE_URL` | Your Supabase project URL |
| `SUPABASE_KEY` | Service role key (or `SUPABASE_SERVICE_KEY`) |

---

## Command Line Options

| Option | Short | Description |
|--------|-------|-------------|
| `--dry-run` | | Schema check only, no database connection |
| `--verbose` | `-v` | Show detailed output for each KPI |
| `--output FILE` | `-o` | Save markdown report to file |

---

## Output

### Console Output (Verbose Mode)

```
======================================================================
E2I Causal Analytics - KPI Coverage Validator V3.0
======================================================================

Validating 46 KPIs...
----------------------------------------------------------------------
  ✅ PASS [WS1-DQ-001] Source Coverage - Patients
       └─ V3 NEW: V3: reference_universe table
  ✅ PASS [WS1-DQ-003] Cross-source Match Rate
       └─ V3 NEW: V3: NEW data_source_tracking table
  ...
----------------------------------------------------------------------

SUMMARY
======================================================================
  Total KPIs:     46
  ✅ Passed:      46
  ❌ Failed:      0
  ⚠️  Warnings:    0
  🆕 V3 New:      14

  Coverage: 46/46 (100.0%)

  🎉 100% KPI COVERAGE ACHIEVED!
======================================================================
```

### Validation Statuses

| Status | Meaning |
|--------|---------|
| ✅ PASS | All tables, columns, and views exist |
| ❌ FAIL | Required table or column missing |
| ⚠️ WARN | View not found (may need creation) |
| ⏭️ SKIP | Validation skipped |

### Exit Codes

| Code | Meaning |
|------|---------|
| `0` | All KPIs validated successfully |
| `1` | One or more KPIs failed validation |

---

## KPI Coverage Summary

### By Workstream

| Workstream | Category | KPIs | V3 New |
|------------|----------|------|--------|
| WS1 | Data Quality | 9 | 6 |
| WS1 | Model Performance | 9 | 3 |
| WS2 | Trigger Performance | 8 | 1 |
| WS3 | Business Impact | 10 | 2 |
| Brand | Brand-Specific | 5 | 1 |
| Causal | Causal Metrics | 5 | 0 |
| **Total** | | **46** | **13** |

### V3 New Tables

| Table | Enables KPIs |
|-------|--------------|
| `user_sessions` | MAU, WAU, DAU |
| `data_source_tracking` | Cross-source Match Rate, Stacking Lift |
| `ml_annotations` | Label Quality (IAA) |
| `etl_pipeline_metrics` | Time-to-Release (TTR) |
| `hcp_intent_surveys` | Intent-to-Prescribe Δ |
| `reference_universe` | Source Coverage calculations |
| `agent_registry` | 11-agent routing |

### V3 New Fields

| Table | New Fields | Enables KPIs |
|-------|------------|--------------|
| `patient_journeys` | `data_lag_hours`, `source_timestamp`, `ingestion_timestamp` | Data Lag |
| `ml_predictions` | `model_pr_auc`, `rank_metrics`, `brier_score` | PR-AUC, Recall@K, Brier |
| `triggers` | `change_type`, `change_failed`, `change_outcome_delta` | Change-Fail Rate |

### KPI Helper Views

| View | Purpose |
|------|---------|
| `v_kpi_cross_source_match` | Daily match rates by source |
| `v_kpi_stacking_lift` | Stacking eligible vs applied |
| `v_kpi_data_lag` | Avg/median/p95 lag hours |
| `v_kpi_label_quality` | IAA metrics by annotation type |
| `v_kpi_time_to_release` | TTR hours by pipeline |
| `v_kpi_change_fail_rate` | Change success/failure rates |
| `v_kpi_active_users` | MAU/WAU/DAU counts |
| `v_kpi_intent_to_prescribe` | Intent scores by brand/month |

---

## Markdown Report

When using `--output`, the script generates a detailed markdown report:

```markdown
# E2I KPI Coverage Validation Report
Generated: 2025-11-28 16:56:09

## Summary
- **Total KPIs**: 46
- **Passed**: 46 ✅
- **Failed**: 0 ❌
- **Warnings**: 0 ⚠️
- **V3 New**: 14

## By Workstream

### WS1 (18/18)
| ID | KPI | Status | V3 New | Notes |
|---|---|---|---|---|
| WS1-DQ-001 | Source Coverage - Patients | ✅ | ✓ | V3: reference_universe table |
...
```

---

## Integration

### CI/CD Pipeline

```yaml
# GitHub Actions example
- name: Validate KPI Coverage
  run: |
    python validate_kpi_coverage.py --dry-run
    if [ $? -ne 0 ]; then
      echo "KPI coverage validation failed!"
      exit 1
    fi
```

### Pre-commit Hook

```bash
#!/bin/bash
# .git/hooks/pre-commit
python validate_kpi_coverage.py --dry-run
```

### With Supabase Migrations

```bash
# After applying migrations
supabase db push
python validate_kpi_coverage.py --verbose
```

---

## Extending

### Adding New KPIs

Edit the `KPI_DEFINITIONS` list in `validate_kpi_coverage.py`:

```python
KPIDefinition(
    id="WS3-BI-011",
    name="New KPI Name",
    workstream="WS3",
    category="Business Impact",
    calculation_type=CalculationType.DERIVED,
    tables=["table1", "table2"],
    columns=["table1.column1", "table2.column2"],
    view="v_kpi_new_view",  # Optional
    is_v3_new=True,
    note="Description of new addition"
)
```

### Adding New Tables

Update the `V3_SCHEMA` dictionary:

```python
"new_table": {
    "columns": ["col1", "col2", "col3"],
    "is_new": True
}
```

### Adding New Views

Add to the `views` list in `V3_SCHEMA`:

```python
"views": [
    ...existing views...,
    "v_kpi_new_view"
]
```

---

## Troubleshooting

### "Supabase client not available"

```bash
pip install supabase
```

### "SUPABASE_URL/SUPABASE_KEY not set"

```bash
export SUPABASE_URL="https://your-project.supabase.co"
export SUPABASE_KEY="your-service-role-key"
```

### View warnings in dry-run mode

Views are validated by name only in dry-run mode. To fully validate views, run against a live database with the views created.

### Column not found errors

Ensure your database schema matches V3. Run migrations:
- `006_update_agent_enum.sql`
- `007_kpi_gap_tables.sql`
- `008_kpi_helper_views.sql`

Or use the complete schema: `e2i_ml_complete_v3_schema.sql`

---

## Related Files

| File | Description |
|------|-------------|
| `e2i_ml_complete_v3_schema.sql` | Complete V3 database schema |
| `kpi_definitions.yaml` | KPI metadata and thresholds |
| `domain_vocabulary.yaml` | Agent and KPI vocabularies |
| `E2I_KPI_Verification_V3.md` | Detailed KPI documentation |

---

## License

Internal use only - Novartis E2I Causal Analytics Platform

---

## Changelog

### V3.0.0 (2025-11-28)
- Initial release
- 46 KPIs validated
- 6 new tables, 8 new views
- Dry-run and live validation modes
- Markdown report generation
