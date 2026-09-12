# #2031 leaf × seed sweep — n=5000, 6 seeds, 11 planted truths, production-mirrored LinearDML

## Per leaf (aggregated over 11 pairs × seeds)

| leaf | coverage of planted truth | mean \|bias\| | median seed-SD / mean-SE | mean CI half-width | max \|bias\| pair |
|---|---|---|---|---|---|
| 5 | 0.80 | 0.0187 | 0.66 | 0.0306 | copay_support->low_gap_180d 0.052 |
| 50 | 0.82 | 0.0169 | 0.20 | 0.0306 | copay_support->low_gap_180d 0.044 |
| 100 | 0.82 | 0.0163 | 0.17 | 0.0306 | copay_support->low_gap_180d 0.043 |

## Per pair × leaf: mean ATE (seed SD) / mean SE / coverage

| pair | truth | leaf 5 | leaf 50 | leaf 100 |
|---|---|---|---|---|
| treatment_arm->treatment_initiated | 0.164 | 0.149 (0.010) / 0.019 / 1.00 | 0.158 (0.004) / 0.019 / 1.00 | 0.159 (0.002) / 0.019 / 1.00 |
| treatment_arm->adherent_180d | 0.246 | 0.216 (0.010) / 0.021 / 0.83 | 0.215 (0.003) / 0.021 / 1.00 | 0.215 (0.003) / 0.021 / 1.00 |
| treatment_arm->low_gap_180d | 0.241 | 0.213 (0.011) / 0.020 / 1.00 | 0.215 (0.003) / 0.020 / 1.00 | 0.216 (0.004) / 0.020 / 1.00 |
| copay_support->adherent_180d | 0.120 | 0.073 (0.010) / 0.014 / 0.00 | 0.080 (0.005) / 0.014 / 0.00 | 0.081 (0.002) / 0.014 / 0.00 |
| copay_support->low_gap_180d | 0.115 | 0.062 (0.007) / 0.014 / 0.00 | 0.071 (0.004) / 0.014 / 0.00 | 0.072 (0.003) / 0.014 / 0.00 |
| copay_support->persistent_180d | 0.089 | 0.097 (0.013) / 0.014 / 1.00 | 0.096 (0.003) / 0.014 / 1.00 | 0.094 (0.001) / 0.014 / 1.00 |
| psp_enrolled->adherent_180d | 0.099 | 0.100 (0.010) / 0.014 / 1.00 | 0.094 (0.002) / 0.014 / 1.00 | 0.095 (0.001) / 0.014 / 1.00 |
| psp_enrolled->persistent_180d | 0.077 | 0.073 (0.004) / 0.014 / 1.00 | 0.080 (0.002) / 0.014 / 1.00 | 0.080 (0.001) / 0.014 / 1.00 |
| rep_detailing_high->treatment_initiated | 0.062 | 0.069 (0.009) / 0.013 / 1.00 | 0.073 (0.006) / 0.013 / 1.00 | 0.074 (0.003) / 0.013 / 1.00 |
| sample_dropped->treatment_initiated | 0.041 | 0.037 (0.009) / 0.014 / 1.00 | 0.044 (0.006) / 0.014 / 1.00 | 0.044 (0.004) / 0.014 / 1.00 |
| trigger_accepted->treatment_initiated | 0.068 | 0.077 (0.009) / 0.013 / 1.00 | 0.077 (0.003) / 0.013 / 1.00 | 0.076 (0.003) / 0.013 / 1.00 |
