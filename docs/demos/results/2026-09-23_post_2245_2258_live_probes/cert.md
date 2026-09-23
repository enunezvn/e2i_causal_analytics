# Post-#2245 / #2258 live probes on deployed main `4974774db` — 2026-09-23

**PASS on both probes.** Default path: the `structural prior not applied` warning is gone (PR #2245 declared `feature_manifest_source=optum_mart` on `optum_biologic_persistence`), the estimate is bit-identical to the pre-B cert, and PR #2258's bounded adjustment-set search reports itself once. Discovery-on (the shape that killed the gunicorn worker in the Lane A cert and left the job `running` forever): status **completed**, `dag_source=augmented`, 675.6 s wall under the 900 s cap, the container did not restart, the estimate is bit-identical.

Deployed image: `ghcr.io/enunezvn/e2i-api:4974774db…` (StartedAt 2026-09-23T08:35:40Z; main head carrying #2237, #2240, #2243, #2246 (migration 152), #2239, #2250, #2252, #2245, #2258). Both probes are the platform's normal job through the deployed API (read-only against the repo; no paid LLM call on this path).

## Probe 1 — default path (`default_path/`, script = the post-B probe, same request: treatment_dupixent → discontinued_180d, limit 20000)

| field | pre-B cert (2026-09-22) | post-#2237 (2ccf2f9a5) | post-#2245/#2258 (4974774db) |
|---|---|---|---|
| `structural prior not applied` occurrences | — | 1 | **0** |
| `warnings` | — | 2 | 2: bounded-search line (#2258) + the null-finding line |
| `ate` / `ate_ci` / `p_value` | 0.007056123869614513 / [-0.00537280…, 0.01948505…] / 0.26583454… | identical | identical |
| `data_source` / `dag_source` | synthetic (showcase flag) / — | database / domain_knowledge | database / domain_knowledge |
| `structural_prior` field | — | absent | absent (no APPROVED optum_mart review yet; two PENDING `initial_dag` reviews exist: b54d7c32, 2fa5a4a5) |

The #2258 bound fired on the default path: "61 candidate covariates exceed the cap of 40; the minimal-set enumeration was skipped and the full admissible candidate set was used instead" — the same full-candidate fallback the unbounded search reached on every manual-DAG path (no size≤3 set exists), which is why the estimate did not move.

## Probe 2 — discovery on (`discovery_on/`, `run_cert.py --probe-discovery`, auto_discover=true, same estimand)

| field | Lane A cert probe (814eaa959, 2026-09-22) | post-#2258 (4974774db) |
|---|---|---|
| job outcome | worker killed (gunicorn code 134), job left `running` | **completed** |
| `dag_source` | — | augmented |
| wall | > 900 s cap never fired | 675.6 s (44 polls) |
| `ate` / `ate_ci` | — | 0.007056123869614513 / [-0.00537280…, 0.01948505…] (identical to probe 1) |
| refutation / gate / E-value | — | None / None | gate None | evalue None |
| container before / after | — | identical (`container_before.txt`, `container_after.txt`) |

Warnings (4): the bounded-search line; discovery pre-flight (77 offered, 20 handed to structure learning, 16 exactly collinear dropped); "Discovery did not draw the estimand edge … removed in PC's skeleton phase"; the null-finding line. `discovered_confounders=[]`.

What this does and does not prove: it proves the #2258 mechanism (search off the event loop, bounded, FCI diagnostic abandoned deterministically, orphan read-repair) keeps the worker alive and the job terminal on the real cohort at production size on a loaded box. It does not measure the GIL-contention case in isolation (the in-process measurement is in PR #2258's body: 34.9 s alone, 87.8 s with one contending thread).

Raw responses: `default_path/raw_discontinued_180d.json`, `discovery_on/raw_discontinued_180d.json` (image tags shortened to 9 hex for the secret scanner). Logs: `*/probe.log`. Scripts: `default_path/probe.py`, `discovery_on/run_cert.py` (copies of the post-B probe and the Lane A cert runner).
