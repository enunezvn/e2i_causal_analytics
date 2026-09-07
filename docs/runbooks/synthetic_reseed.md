# Runbook — synthetic substrate reseed and the manual data jobs

**Scope:** the weekly Monday 03:00 reseed chain (`scripts/reseed_synthetic.sh` →
`scripts/retrain_goldstd.sh` → hcp_adoption champion re-promotion), and the
one-off data jobs the CI deploy never runs.

`DEPLOYMENT.md` says **"Not part of the CI deploy: FalkorDB seeding (manual) and
synthetic data reseeds."** This runbook is what that sentence points at.

**Verified 2026-09-07** by reading each script and the live user crontab
(read-only). No job below was executed while writing this.

---

## 0. The one rule

**The synthetic substrate is FROZEN.** The gold-standard models were trained on
it. A full reseed rewrites every timestamp *and*, because the anchored and
calendar paths consume different RNG draw counts, every attribute value under the
same primary keys — invalidating those models and re-firing an honest drift storm.

So: the weekly job **appends at the frontier**; a full reseed is **disaster
recovery only**. Every targeted seed script below is written to be idempotent
against the frozen substrate for the same reason.

---

## 1. The weekly chain

Live user crontab (`crontab -l` as `enunez`, confirmed 2026-09-07):

```
0 3 * * 1 /home/enunez/Projects/e2i_causal_analytics/scripts/reseed_synthetic.sh >> /home/enunez/logs/e2i-reseed.log 2>&1
```

Note the log path: `/var/log` is **not** writable by the cron user on this host,
and a root-owned redirect target kills the job with `Permission denied` *before*
the script runs — silently no-oping it. Logs go under `$HOME/logs`.

This entry lives in the **user** crontab, not `/etc/cron.d/e2i-maintenance`, so
the freshness alarm in [`maintenance-cron.md`](maintenance-cron.md) does **not**
watch it. Nothing does.

### `scripts/reseed_synthetic.sh`

| Invocation | Mode passed to the loader | Meaning |
|---|---|---|
| *(no args)* — what cron runs | `--append-frontier` | Grow the substrate with the trailing deterministic weekly cohorts + downstream events, refresh `user_sessions` (MAU/WAU), **without rewriting history** |
| `--full` | `--anchor-to-now` | **Recovery only.** Legacy full-size destructive reseed; rewrites every synthetic timestamp and value |
| `--skip-retrain` | *(consumed by the wrapper)* | Skip the gold-standard retrain stage; anywhere in the args, never forwarded to the loader |
| `--dry-run` | *(detected **and** forwarded)* | Loader and A/B stages validate without writing (they parse the flag); stages 2, 3 and 4 are **skipped** — see below |

Any other argument is **forwarded verbatim** to `scripts/load_synthetic_data.py`
(§3) — i.e. to stages 1 and 5 only, the two that *are* that script.

#### `--dry-run` reaches only the two loader stages — the other three are skipped

`--dry-run` is an argparse flag of `load_synthetic_data.py`. Stages 2, 3 and 4
are different entrypoints and **cannot be handed it** (#1930):

- `src/kpi/history_backfill.py` reads `sys.argv[1:]` as KPI ids, so `--dry-run`
  would become a KPI id;
- `src/kpi/history_capture.py` recognises only `--purge` and drops every other
  `--`-prefixed arg, so `--dry-run` would be **silently swallowed and the stage
  would write anyway** — a false green, worse than no flag at all;
- `scripts/retrain_goldstd.sh` reads no arguments.

So the wrapper **withholds those three stages** under `--dry-run` and says so in
the log:

```
=== kpi_history backfill SKIPPED (--dry-run) <ts> ===
=== kpi_history weekly capture SKIPPED (--dry-run) <ts> ===
=== goldstd retrain SKIPPED (--dry-run) <ts> ===
```

Before #1930 they ran regardless: `--dry-run` printed `DRY RUN` and then wrote
`kpi_history` twice and the model registry once, and `--full --dry-run` ran the
**irrecoverable** `history_capture --purge` first.

Environment gotchas the wrapper encodes — do not "simplify" them away:

- `dotenv` is **not** on the bare shell `PATH`; only `.venv/bin/dotenv` works.
- `PYTHONPATH` must be the repo root for `src.*` imports.
- `LOKY_MAX_CPU_COUNT=1` keeps joblib from over-forking on the droplet.
- With `--full`, full size is **required**: `--small` yields ~250 users and the
  MAU/WAU KPI targets (2000/1200) can never reach GOOD.

A preflight (not a stage) fails loud if `.venv/bin/dotenv` or `.venv/bin/python`
is missing — nothing can run without the venv.

### Stages, and why a failure does not stop the run

Every stage runs through `reseed_run_stage` in `scripts/lib/reseed_stages.sh`
(#1577). A failed stage prints a `FAILED` marker, **the remaining stages still
run**, the final done line is always reached with a status summary, and the exit
code is nonzero iff any stage failed.

This exists because the wrapper used to run under bare `set -euo pipefail`: the
loader exiting 1 on *partial* failure killed the wrapper before any later stage,
so the kpi backfill, capture, retrain and A/B stages **never executed via cron
between 2026-07-06 and 2026-08-10** — zero stage markers in the log, no run ever
printed the done line. The loader's own exit semantics are deliberately
unchanged: partial failure still fails its stage, loudly.

| Order | Stage name in the log | What it does |
|---|---|---|
| 1 | `loader` | `load_synthetic_data.py $MODE …` |
| 2 | `kpi_history backfill` | `python -m src.kpi.history_backfill` — rebuilds `kpi_history` with replace semantics (delete per `(kpi_id, source)`, then upsert) |
| 3 | `kpi_history weekly capture` | `python -m src.kpi.history_capture` — records this week's live reading of the present-state KPIs (coverage/eligibility) that cannot be backfilled honestly. In `--full` mode only, a `--purge` runs first and the `&&` chain means a failed purge **skips** the capture |
| 4 | `goldstd retrain` | `scripts/retrain_goldstd.sh` (skipped by `--skip-retrain`) |
| 5 | `A/B substrate refresh` | `load_synthetic_data.py --refresh-ab …`; **append mode only** — the full path already rebuilds it |

Reading the log:

```bash
grep -E '=== .* (start|done|FAILED)' /home/enunez/logs/e2i-reseed.log | tail -40
```

`=== reseed_synthetic done <ts> (all stages OK) ===` or
`… (FAILED stages: <names>) ===` is always the last line. **Its absence means the
run died outside the stage runner** — that is the #1577 shape, and it is the one
thing to look for first.

### `scripts/retrain_goldstd.sh`

Retrains the **12** gold-standard **staging** models on the current substrate and
re-records their walk-forward + holdout metric trends:
`{initiation, persistence, discontinuation} × {Remibrutinib, Fabhalta, Kisqali}`
at patient grain (`-m src.mlops.gold_standard_eval.run_patient_cohorts`), plus
`hcp_adoption × 3` brands at HCP grain (`… .run_hcp_cohorts`).

Measured cost on this droplet (2026-07-04, per the script header): **~43 s wall /
~570 MiB peak RSS** for one slot, **~9 min** for all 12 sequentially.

Idempotent by construction: registry rows UPSERT on `(model_name, model_version)`
preserving row ids (and every RESTRICT FK from `ml_performance_metrics` /
`ml_drift_history` / `ml_monitoring_alerts`); metric rows are
delete-by-`(model_id, source)`-then-insert. All 12 register at `stage='staging'`
— `cohort_deployer` hard-refuses `'production'`, so the retrain never touches the
serving ensemble directly.

**Not included** (heavier, service-restarting choreography — run by hand when the
serving layer should catch up): the SHAP serving-bundle re-materialize, the
bentoml restart, and the SHAP cache refresh. See
`scripts/sync_goldstd_serving.py`.

Standalone: `./scripts/retrain_goldstd.sh >> /home/enunez/logs/e2i-reseed.log 2>&1`

### Champion re-promotion — the fail-closed leg

The HCP-slot UPSERT resets the three `hcp_adoption` rows to `stage='staging'`,
`is_champion=false`, **demoting** the owner-ruled production champions
(#1354/#1384) that chat propensity serves from. So `retrain_goldstd.sh` ends with:

```
.venv/bin/python scripts/promote_hcp_adoption_champions.py --execute
```

`--execute` is the write phase; **the default is a dry run** that prints the exact
intended updates. The script re-scores each retrained artifact against its fresh
holdout metrics and **HOLDs per brand** on:

- a **faithfulness** miss — recomputed `auc_roc`/`accuracy` must reproduce the
  stored `source='holdout'` values within tolerance, so a divergent artifact is
  never promoted on stale numbers; or
- a **pathology** gate — `calibration_slope` unfittable or outside `[0.5, 2.0]`,
  or `brier_score >= prevalence * (1 - prevalence)` (no probabilistic skill over
  predicting the base rate).

A held brand stays `staging` and **chat propensity for that brand fails closed** —
`resolve_hcp_adoption_champion` refuses anything but a production champion
(#1690). That is designed behaviour, not an outage to patch around.

A promotion failure does **not** abort the wrapper — the retrain itself succeeded
and the downstream reseed stages must still run — so it warns instead:

```
WARNING: hcp_adoption champion re-promotion FAILED — chat propensity fails closed
until scripts/promote_hcp_adoption_champions.py --execute succeeds (#1690)
```

**Grep for that line after every weekly run.** Recover by re-running the promote
script by hand (dry run first, no flag).

---

## 2. Manual / one-off data jobs the deploy never runs

None of these are in `deploy.yml` or any crontab. Run them deliberately, and read
each script's own header first — the summaries below are orientation, not a
substitute.

**Chat-RAG chunk corpus** — `scripts/rag/ingest_chunk_corpus.py` (#1373/#1374).
Renders real `business_metrics` rows as prose (values verbatim from the fact
table) into `rag_document_chunks`, embedded in the `text-embedding-3-small` space
the chat `HybridRetriever` queries. Run **in-container against prod after merge +
deploy** — the container carries `SUPABASE_*` and `OPENAI_API_KEY`. Idempotent:
a re-run embeds only new/changed snapshots (content-hash dedup).

```bash
docker exec e2i_api python -m scripts.rag.ingest_chunk_corpus --health-only          # report size, no writes, no spend
docker exec e2i_api python -m scripts.rag.ingest_chunk_corpus --brands Kisqali --limit 10   # smoke
docker exec e2i_api python -m scripts.rag.ingest_chunk_corpus --latest-per-combo     # full production sync
```

**Targeted `causal_paths` seeds** (#1321/#1325) — each upserts only its own grain
with content-addressed `path_ids`, so re-runs are no-ops apart from freshness
stamps, and a future disaster-recovery reseed emits the same rows and upserts over
them harmlessly. All take `--dry-run`:

```bash
docker exec e2i_api python scripts/seed_commercial_causal_paths.py --dry-run       # commercial-KPI grain
docker exec e2i_api python scripts/seed_brand_clinical_causal_paths.py --dry-run   # brand-distinct clinical axis
docker exec e2i_api python scripts/seed_comm_arm_causal_paths.py --dry-run         # patient-grain commercial arms
docker exec e2i_api python scripts/sync_causal_paths_to_falkordb.py --dry-run      # Supabase SSOT -> FalkorDB graph
```

The Knowledge-Graph page and graph-stats read FalkorDB, so the sync is what makes
seeded chains visible there. The dashboard's "Primary Causal Value Chains" reads
`causal_paths` directly (`GET /api/causal/value-chains`) and does **not** depend
on the sync.

**FalkorDB graph seeding** — `scripts/seed_falkordb_all.sh`. Seeds the
`e2i_causal` graph if it is empty. Needs `FALKORDB_PASSWORD` (read from `.env` if
unset); `FALKORDB_HOST` defaults to `localhost`, `FALKORDB_PORT` to **6381** on
the host, **6379** with `--docker`. The `e2i_semantic` seeding step was retired
(#890): no runtime reader uses it, and even a read-only count probe re-created the
empty graph shell, since FalkorDB creates a graph key on any `GRAPH.QUERY`.

```bash
./scripts/seed_falkordb_all.sh            # seed if empty (host port 6381)
./scripts/seed_falkordb_all.sh --force    # clear and re-seed e2i_causal
./scripts/seed_falkordb_all.sh --docker   # internal port 6379
```

**Single-table backfill** — `load_synthetic_data.py --only-tables` (#1387): see §3.

---

## 3. `scripts/load_synthetic_data.py` flags

Read from the script's own `argparse` on 2026-09-07. The wrapper supplies the
mode; these are what you can add to a `reseed_synthetic.sh` invocation or pass
directly.

| Flag | Default | What it does |
|---|---|---|
| `--dry-run` | off | Validate without loading |
| `--small` | off | Smaller dataset (**~250 users — cannot reach the MAU/WAU KPI targets**) |
| `--verbose` | off | Verbose logging |
| `--dgp NAME` | `confounded` | One of `simple_linear`, `confounded`, `heterogeneous`, `time_series`, `selection_bias` |
| `--parquet-out DIR` | `None` | Also write each dataset to `<dir>/<table>.parquet` + `manifest.json` |
| `--parquet-only` | off | Write parquet only; **skip** the Supabase load (pollution-free) |
| `--tag PREFIX` | `scv` | Entity-id namespace prefix |
| `--anchor-to-now` | off | Remap all generated dates onto a rolling window ending today so windowed KPIs read non-zero — **this is what `--full` selects** |
| `--anchor-ref YYYY-MM-DD` | `None` | Reference date for `--anchor-to-now` |
| `--append-frontier` | off | Frontier-append mode (`src/ml/synthetic/frontier_append.py`) — **the weekly default** |
| `--frontier-ref YYYY-MM-DD` | `None` (today) | Frontier date for `--append-frontier` |
| `--only-tables a,b,c` | `None` | Generate the **full** dataset graph (so cross-table draws stay coherent) but **load only these tables**. Errors on unknown names |
| `--refresh-ab` | off | Refresh only the Shard-09 A/B substrate (`ml_experiments` + `ab_experiment_assignments`/`enrollments`/`results`) in place via deterministic `uuid5` ids. **Ignores `--small`** — the purge is all-or-nothing, so the reload must be full-size |

`--only-tables` was built for the #1387 triggers-only view-stage backfill: with
the same `--tag`/`--dgp`/seed and a pinned `--anchor-ref`, the generators
reproduce the frozen substrate byte-identically, so upserting one table cannot
decohere the rest. It **must not** be combined with `--refresh-ab` or the whole-
substrate modes, whose loads are only coherent as a whole; the script errors.

Without `--refresh-ab`, the frontier append does not touch the A/B tables at all —
which is why stage 5 exists: otherwise `experiment_monitor` staleness alerts
alarm on a frozen substrate.

---

## 4. Running one by hand

Always from the repo root, always through the venv's `dotenv` — see the
environment gotchas in §1.

```bash
cd /home/enunez/Projects/e2i_causal_analytics

# Weekly job, by hand (same thing cron runs)
./scripts/reseed_synthetic.sh >> /home/enunez/logs/e2i-reseed.log 2>&1

# Append without the ~9-min retrain
./scripts/reseed_synthetic.sh --skip-retrain

# Whole-chain dry run (#1930). `--dry-run` is forwarded to the two
# load_synthetic_data.py stages (1 loader, 5 A/B), which validate without
# writing; the kpi backfill, the weekly capture and the retrain cannot take the
# flag, so the wrapper SKIPS them and prints a SKIPPED marker for each. Nothing
# in the chain writes.
./scripts/reseed_synthetic.sh --dry-run

# Note --full must come FIRST when combined (mode detection reads $1):
./scripts/reseed_synthetic.sh --full --dry-run

# DISASTER RECOVERY ONLY — invalidates the frozen substrate the gold-standard
# models were trained on and re-fires a drift storm. Read section 0 first.
./scripts/reseed_synthetic.sh --full
```

A bare `-m src.kpi.*` invocation on this box needs the venv's dotenv wrapper —
`.venv/bin/dotenv -f .env run -- .venv/bin/python -m src.kpi.history_backfill` —
which is exactly what the stage functions do.

These are long jobs. Run them detached (`setsid nohup … &`) and poll the log
rather than holding a foreground shell.
