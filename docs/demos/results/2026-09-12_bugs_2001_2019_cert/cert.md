# Live certification — #2001 (PR #2037 → `87d5e26eb`) and #2019 (PR #2038 → `dde03e0b9`), 2026-09-12

Two worktree-isolated lanes, merged serially with a full deploy between them so a failure could be attributed to one change. Both merged with merge commits (no squash). Containers read from `docker ps`, never from the deploy job conclusion (see `cert_container.txt`).

| step | result |
|---|---|
| `f16979a51` deploy (#2006, another session) | completed success — waited for it before merging, to avoid a partial image |
| merge #2037 → `87d5e26eb` | deploy run 34668032053 success; `e2i_api` tag flipped to `87d5e26eb`; api health 200 |
| #2001 live cert on `87d5e26eb` | **GREEN** |
| merge #2038 → `dde03e0b9` | deploy run 34670414476; all e2i containers on `dde03e0b9`; api health 200 |
| #2001 **re-cert** on `dde03e0b9` | **GREEN** (guards against a partial second image) |
| #2019 live cert on `dde03e0b9` | **GREEN** |

## 1. Container content

`cert_container.txt`. The #2001 guard is at `/app/src/causal_engine/evalue.py:578`. For #2019, `project_tool_output` is at `synthesizer.py:178`, the `max(n, scalar_floor(x))` bound is documented at `:191`, and the old byte cut `output_str[:1000]` occurs **0** times. Positive control so the marker check is not vacuous: the same grep against the pre-merge tree `84f74b691` returns **1**.

## 2. #2001 — a degenerate frame returns the empty benchmark silently

Probe: `covariate_bias_factors` on four degenerate frames under `warnings.catch_warnings`, run inside `e2i_api`.

| input | before (measured on `main` pre-merge) | live on `dde03e0b9` |
|---|---|---|
| all-NaN treatment | `{}` + `RuntimeWarning: All-NaN slice encountered` | `{}`, no warning |
| all-NaN outcome | `{}` + `RuntimeWarning: Degrees of freedom <= 0 for slice.` | `{}`, no warning |
| empty frame | `{}` + `RuntimeWarning: Mean of empty slice` | `{}`, no warning |
| all-NaN covariate | `{}`, no warning (already guarded) | `{}`, no warning |
| **POSITIVE CONTROL** healthy frame | `{'c': 1.0}` | **`{'c': 1.0}`** — non-empty |

The positive control is load-bearing: without it, "no warnings" would pass trivially if the guard had made every frame return empty.

No returned value changed. Pre-merge, a differential harness over **83 frames** (degenerate probes, fixtures from `test_evalue.py`, 60 randomized frames with partial-NaN injection, 8 long-tail categorical) was identical before and after, with a positive control of 56 non-empty factor dicts / 1 preserved `ValueError` / 26 empty.

Two corrections to the issue's own report, found while reproducing it: the empty-frame case warns `Mean of empty slice`, not `All-NaN slice encountered`; and the warning `lineno` values point into numpy's `_nanfunctions_impl.py`, not `evalue.py`.

## 3. #2019 — tool outputs are projected, not cut at 1,000 chars

Production path: `tool_composer_tool.ainvoke({...})` (the chat entry point, which resolves a real cohort frame into `context["estimation_data"]`), real Kisqali cohort **8,730 × 85**, real LLM.

Query: *"For Kisqali, build the cohort of patients eligible for a copay support program and score each patient's risk of discontinuation, then summarise who is highest risk."*

| step | raw output | projected | verdict |
|---|---|---|---|
| step_1 `cohort_builder` | **61,090** chars | **1,999** | OVER-CAP |
| step_2 | **24,703** chars | **1,988** | OVER-CAP |

Keys the **old** 1,000-char cut would have removed entirely, and which the **new** projection keeps: `total_evaluated`, `total_eligible`, `eligibility_rate`, `criteria_breakdown`, `execution_time_ms` — all five, on both steps.

The decisive evidence is in the answer text itself:

> Out of **8,730** patients evaluated, **3,018 (34.6%)** met the general program inclusion criteria … the cohort narrowed to **1,207 patients (13.8%)** who currently meet all eligibility criteria.

Every one of those numbers lives in a key that begins past character 1,000 of the serialized output. Under the pre-#2019 cut the synthesis prompt contained raw patient IDs and **none** of the counts, so the model could not have stated them. Both projections also land within the 2,000-char budget (1,999 / 1,988), confirming live that the budget covers the whole string including the footer.

### Attempts that did NOT certify (recorded deliberately)

1. `compose_query` called **without** context → every tool refused ("requires a real DataFrame … does not fabricate"), no outputs at all → **INCONCLUSIVE**. The probe was unfaithful to production, which resolves the frame first.
2. Payer-tier and territory queries on the live cohort → largest output **489** chars (3 payer tiers), then **254** chars (4 territories) → **INCONCLUSIVE**. Live data is smaller than the constructed frames the lane measured (16-150 segments); these queries genuinely cannot cross the cap.

The cert reports INCONCLUSIVE rather than GREEN when nothing exceeds 1,000 chars — a cert that cannot fail proves nothing.

## 4. Gates

Both PRs merged green. #2037: 6/6 (Backend Tests, Lifecycle State Guard, Security Scanning, Synthetic Benchmarks, Tier 1-5 Agent Harness, Verify OpenAPI Types). #2038: 5/5 — Synthetic Benchmarks is path-filtered to `src/causal_engine/**` and `tests/synthetic/**`, so it correctly did not trigger for a `tool_composer`-only change; that is a complete set, not a skipped gate.

The `87d5e26eb` deploy's `Type Check (MyPy)` job passed, resolving #2001's locally-deferred mypy (the scoped single-file run reached 1.98 GiB RSS against 2 GiB available on the droplet and was killed rather than risk OOM-ing concurrent sessions).

## 5. Cost, unverified

#2019 adds **+5,668 chars (~+1,400 input tokens)** to the synthesis call, worst realistic 6-step block 11,042 chars. This is a **size** measurement, not a latency one — no end-to-end latency was measured. No prompt-size budget exists in `docs/demos/results/*copilot_chat_perf*` (those carry end-to-end latency budgets only: T3 ≤ 15 s / 22.4 s), which is why the 2,000 budget is pinned to the existing `composer._MAX_FAILURE_REASON_CHARS` rather than to an invented number. Whether the token delta is immaterial against T3 remains a **hypothesis**.

## 6. Issues

- #2001 CLOSED by PR #2037; #2019 CLOSED by PR #2038.
- **#2044 filed** — `risk_scorer` crashes on NaN feature values (raw sklearn `Input X contains NaN`); observed live in the cert run, killing the composition and skipping 3 downstream steps. Pre-existing; not caused by either change.
- **#2045 filed** — a missing required tool argument raises `TypeError` and is retried 3× as a transient failure (opening the circuit breaker) instead of failing fast as a plan defect. Observed live (`gap_calculator() missing 3 required positional arguments`); planner-nondeterministic, and the same query later planned 4 steps that all succeeded.
