# CSV Templates for E2I Data Conversion

These templates provide the exact column structure expected by the E2I Causal Analytics pipeline. Each file contains 5 example rows with realistic pharmaceutical domain values.

## Files

| Template | Rows | Description | Required For |
|----------|------|-------------|--------------|
| `hcp_profiles.csv` | 5 | HCP master data — NPI, specialty, tier, engagement | All tiers |
| `patient_journeys.csv` | 5 | Patient journeys with causal variables | Tier 0 ML pipeline |
| `treatment_events.csv` | 5 | Treatment/prescription/lab events | Tier 0 + Tier 2 causal |
| `business_metrics.csv` | 5 | KPI snapshots by brand/region | Tier 3 + Tier 4 |
| `triggers.csv` | 5 | NBA trigger events with outcomes | Tier 2 + Tier 4 |

## Usage

### 1. Copy and populate

```bash
cp docs/data/templates/patient_journeys.csv data/input/patient_journeys.csv
# Edit with your real data, keeping the header row intact
```

### 2. Validate with Pandera

```bash
.venv/bin/python -c "
import pandas as pd
from src.mlops.pandera_schemas import validate_dataframe

df = pd.read_csv('data/input/patient_journeys.csv')
result = validate_dataframe(df, 'patient_journeys')
print(f'Status: {result[\"status\"]}')
if result['errors']:
    for e in result['errors']:
        print(f'  Column: {e.get(\"column\")}, Check: {e.get(\"check\")}')
"
```

### 3. Load into Supabase

```sql
\copy patient_journeys FROM 'data/input/patient_journeys.csv' WITH (FORMAT csv, HEADER true);
```

## Column Notes

### ID Formats

| Column | Pattern | Example |
|--------|---------|---------|
| `hcp_id` | `HCP` + 8 digits | `HCP00000001` |
| `patient_id` | `PAT` + 8 digits | `PAT00000001` |
| `patient_journey_id` | `PJ` + 16 digits | `PJ00000000000001` |
| `treatment_event_id` | `TE` + 25 digits | `TE000000000000000000000001` |
| `trigger_id` | `TRG` + 25 digits | `TRG000000000000000000000001` |
| `metric_id` | Free-form string | `BM-2025W47-REMI-NE-TRX` |
| `npi` | 10 digits (Luhn checksum) | `1234567890` |

### Enum Values

| Column | Valid Values |
|--------|-------------|
| `brand` | `Remibrutinib`, `Fabhalta`, `Kisqali`, `competitor`, `other` |
| `geographic_region` | `northeast`, `south`, `midwest`, `west` |
| `journey_stage` | `diagnosis`, `initial_treatment`, `treatment_optimization`, `maintenance`, `treatment_switch` |
| `journey_status` | `active`, `stable`, `transitioning`, `completed` |
| `event_type` | `diagnosis`, `prescription`, `lab_test`, `procedure`, `consultation`, `hospitalization` |
| `priority` | `critical`, `high`, `medium`, `low` |
| `data_split` | `train`, `validation`, `test`, `holdout`, `unassigned` |
| `insurance_type` | `commercial`, `medicare`, `medicaid`, `uninsured`, `other` |
| `adoption_category` | `innovator`, `early_adopter`, `early_majority`, `late_majority`, `laggard` |

### Nullable vs Required

- **Required**: `hcp_id`, `patient_journey_id`, `patient_id`, `journey_start_date`, `event_date`, `trigger_id`, `trigger_timestamp`, `metric_id`, `metric_date`
- **Nullable but recommended**: `brand`, `geographic_region`, `hcp_id` (on patient_journeys), `data_source`
- **Causal variables** (patient_journeys): `disease_severity` (0-10), `engagement_score` (0-10), `treatment_initiated` (0 or 1), `academic_hcp` (0 or 1)

### Data Split Assignment

The `data_split` column controls which ML split each record belongs to. For initial loading, set to `unassigned` — the pipeline assigns splits via `assign_patient_split()` based on chronological boundaries. See [01-DATA-CONVERSION-GUIDE.md](../01-DATA-CONVERSION-GUIDE.md) for split strategy details.

### Minimum Data Volumes

| Dataset | Minimum (Tier 0 basic) | Recommended (full pipeline) |
|---------|------------------------|-----------------------------|
| Patients | 30 | 1,500+ |
| HCPs | 10 | 200+ |
| Treatment events | 50 | 5,000+ |
| Triggers | 20 | 1,000+ |
| Business metrics | 10 | 500+ |
