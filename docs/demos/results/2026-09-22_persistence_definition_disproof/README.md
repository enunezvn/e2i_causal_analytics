# Is `persistent_at_180d` dosing-interval sensitive per brand? (2026-09-22)

**Ask (owner):** ascertain with the cheapest disproof before the outcome is used in
real-data causal estimation.

**Method:** `run_sweep.py` replicates `scripts/convert_optum_mart.py`'s persistence and
discontinuation cohorts from the raw enriched drop (initiators, `claim_record_count >= 2`,
≥180 d follow-up, recorded coverage end) and sweeps (a) the internal-gap threshold
`PERSIST_GAP_DAYS`, (b) a grace on the "covered through day 180" condition, (c) the
discontinuation gap `DISCONT_GAP_DAYS`. Every number below is quoted from
`sweep_run.txt`.

**Replication:** n = 15,209 (XOLAIR 11,009 / DUPIXENT 4,200), and at G = 60 the rates
are 0.523 / 0.347 — identical to the mart frame, so the replication is faithful.

## Result: the gap rule is not the driver; the days-supply is

| Definition | XOLAIR | DUPIXENT | gap (pp) |
|---|---|---|---|
| persistence, G = 30 | 0.399 | 0.268 | 13.1 |
| persistence, G = 60 (**shipped**) | 0.523 | 0.347 | 17.6 |
| persistence, G = 120 | 0.568 | 0.371 | 19.7 |
| persistence, no gap rule | 0.574 | 0.374 | 20.0 |
| — covered through day 180 alone | 0.574 | 0.374 | 20.0 |
| — no internal gap > 60 alone | 0.910 | 0.937 | −2.7 |
| persistence G = 60, grace 14 d | 0.679 | 0.644 | 3.5 |
| persistence G = 60, grace 28 d | 0.734 | 0.744 | −1.0 |
| persistence G = 60, grace 45 d | 0.768 | 0.779 | −1.1 |
| persistence G = 60, grace 60 d | 0.787 | 0.800 | −1.3 |
| discontinuation, D = 60 | 0.164 | 0.176 | −1.2 |
| discontinuation, D = 90 (**shipped**) | 0.106 | 0.122 | −1.6 |
| discontinuation, D = 120 | 0.068 | 0.087 | −1.9 |

Mechanism: Dupixent fills carry a 14-day supply (`max_consecutive_biologic_coverage_days`
is 14 at every quartile) and Xolair fills 28–45 days, so the last fill in the window ends
coverage before day 180 far more often on Dupixent — 43.0 % of Dupixent patients have
`cov_to_end` in [150, 180) vs 24.1 % of Xolair (`sweep_run.txt`, last line). The
shipped definition therefore penalises the 14-day-supply product mechanically; the
brand gap collapses with a one-interval grace and inverts from a 28-day grace on.

## Decision for Lane A (recommended; reversible)

- Primary outcome: `persistent_at_180d_g28` = covered through day 180 − 28 AND no internal
  gap > 60 — brand-invariant across 28–60 d grace; a NEW column in the causal export only
  (the prediction target `persistent_at_180d` is untouched).
- Secondary: `discontinued_180d` as shipped (D = 90), already brand-robust.
- Tertiary: `persistent_at_180d` as shipped, reported WITH this table so the −17.6 pp raw
  gap is read as a measurement artefact, never as an effect.
- Open data-quality item: the 14-day Dupixent supply is what the aggregated drop
  records; whether the raw fills are 14-day pens or 28-day packs recorded as 14 needs the
  claim-level feed. Until then the grace definition is the honest one.
