# Lane A — causal cohort export run (2026-09-22)

**Verdict: MATCHES the persistence-definition disproof's faithful replication** — n = 15,209; XOLAIR 11,009 / DUPIXENT 4,200; `persistent_at_180d_g28` 0.734 / 0.744.

## Command (worktree code at `750024ffd`, main checkout's data; `src` asserted to resolve to the worktree)

```
cd /home/enunez/Projects/e2i_causal_analytics/.worktrees/real-data-causal
/home/enunez/Projects/e2i_causal_analytics/.venv/bin/python -m scripts.convert_optum_mart \
  --cohort persistence_causal \
  --input  /home/enunez/Projects/e2i_causal_analytics/data/rwd/Optum_Parquet/Optum_enriched.parquet \
  --output /home/enunez/Projects/e2i_causal_analytics/data/rwd/mart/persistence_causal --verbose
```

| measure | value |
|---|---|
| wall | 19.82 s |
| max RSS | 622,052 KB |
| exit | 0 |

## Summary printed by the converter

```
{'cohort': 'persistence_causal', 'patients': 15209, 'positives': 11206, 'prevalence': 0.7368,
 'splits': {'train': 9125, 'validation': 3041, 'test': 2281, 'holdout': 762},
 'output_dir': '/home/enunez/Projects/e2i_causal_analytics/data/rwd/mart/persistence_causal',
 'arms': {'XOLAIR': 11009, 'DUPIXENT': 4200}}
```

## Attrition (`export_attrition_report.csv`)

| step | count |
|---|---|
| patient_panel | 814,587 |
| input_patients (initiators pushed down at read) | 24,429 |
| initiators | 24,429 |
| quality_filter (claim_record_count ≥ 2) | 22,601 |
| followup_observable (≥ 180 d) | 15,209 |
| coverage_end_observable | 15,209 |
| two_arm_contrast | 15,209 |
| target_positives (`persistent_at_180d_g28`) | 11,206 |
| arm_dupixent | 4,200 |

No `excluded_arm:<BRAND>` step was emitted: the drop holds exactly the two observed brands (remibrutinib absent, as the spec §1 records).

## Exported frame

Shape (15,209 × 81); `is_synthetic` = False on every row; `patient_id` unique; `treatment_dupixent` agrees with `index_biologic_brand` on every row; the only NULLs are `geographic_region` (8.11 %, the ZIP→region map's unmapped prefixes — same as the prediction frame).

| brand | n | persistent_at_180d_g28 | discontinued_180d | biologic_switch_180d_flag | persistent_at_180d (shipped) |
|---|---|---|---|---|---|
| XOLAIR | 11,009 | 0.734 | 0.106 | 0.010 | 0.523 |
| DUPIXENT | 4,200 | 0.744 | 0.122 | 0.012 | 0.347 |

Reference (`docs/demos/results/2026-09-22_persistence_definition_disproof/README.md`): grace-28 persistence 0.734 / 0.744; discontinuation D = 90 0.106 / 0.122; shipped persistence 0.523 / 0.347. All three reproduce. The shipped column's −17.6 pp gap is the days-supply artefact and is carried only to be reported alongside the grace sweep.

Files: `export_attrition_report.csv`, `export_data_dictionary.csv` (71 entries: 64 features, the g28 target, the six causal columns typed treatment / anchor / outcome).
