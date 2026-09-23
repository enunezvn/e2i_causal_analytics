## Owner-GO load command (prod write -- the owner runs it, never an agent)

Backing (gitignored data/, built 2026-09-23 01:04 from the worktree at e0468f0de, byte-stable across two builds):

| file | sha256 |
|---|---|
| data/rwd/synthetic_CSU/csu_escalation_causal/csu_escalation_causal_synthetic.parquet | 0b551ee172d82a1f93891b29eb295465f4a912e490e6eee8e1e44601814f6a97 |
| data/rwd/synthetic_CSU/csu_escalation_causal/ground_truth.json | 0dd5a9dcf8126ef5a67286f3e58e967d0a865c499042f0aaf575ab2521ae7bfe |
| data/rwd/synthetic_CSU/csu_escalation_causal/build_summary.json | edb96c68ca2f412ccf4fdaa5094a9129004c42336e645d4a5c9f353508e7713d |

Rebuild (identical bytes): `python -m scripts.build_csu_escalation_synthetic_cohort --n 3000 --seed 20260922 --out-dir /home/enunez/Projects/e2i_causal_analytics/data/rwd/synthetic_CSU/csu_escalation_causal`

Before merge (the loader lives on the lane branch), from the lane worktree:

```
cd /home/enunez/Projects/e2i_causal_analytics/.worktrees/lane-c-load && \
/home/enunez/Projects/e2i_causal_analytics/.venv/bin/python -m scripts.load_csu_escalation_cohort \
  --input /home/enunez/Projects/e2i_causal_analytics/data/rwd/synthetic_CSU/csu_escalation_causal/csu_escalation_causal_synthetic.parquet \
  --execute
```

After merge, from the main checkout: `python -m scripts.load_csu_escalation_cohort --execute` (the default `--input` is the path above, relative to the checkout).

Expected output: `EXECUTE: upserted 3000 rows`, `LIVE AFTER: n=3000 arms={'XOLAIR': 1220, 'DUPIXENT': 506, 'RHAPSIDO': 1274} treatment={'0': 1726, '1': 1274}`, `LIVE AFTER provenance: is_synthetic counts {'true': 3000, 'false': 0}`, `ROW VERIFICATION: compared 3000 exported rows against 3000 live rows.`, `VERIFIED: ...`, exit 0. Anything else is MISMATCH / exit 1 -- stop and report.
