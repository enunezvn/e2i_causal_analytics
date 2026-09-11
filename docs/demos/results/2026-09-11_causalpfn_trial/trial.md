# #1990 — CausalPFN one-day scratch trial (spec §11)

**Date:** 2026-09-11. **Spec:** `docs/superpowers/specs/2026-09-08-expert-review-loop-closure-design.md` §11.
**Decision it feeds:** #1991 debt 1 (retire or refactor the refutation node's reconstruction path).

## Setup (faithful to the spec)

| item | value |
|---|---|
| image | `ghcr.io/enunezvn/e2i-api:f4940c97a…` (main HEAD, the deployed image); see `container.txt` |
| container | `docker run` scratch container, `--cpus=2 --memory=2560m --memory-swap=2560m --user 1000:1000`; never the live `e2i_api` |
| package | `causalpfn==0.1.4` + `faiss-cpu==1.15.0`, `pip install --no-deps --target` into a mounted site dir (image `/app` is read-only; torch 2.9.1+cpu, huggingface_hub, tqdm already present) |
| weights | `vdblm/causalpfn` `causalpfn_v0.pt`, 71 MB, from the Hugging Face hub |
| estimator | `CATEEstimator(device="cpu")`, `fit(X, t, y)`, `estimate_cate(X)` → ATE = mean CATE; 95 % interval from `_estimate_ate_cate_CI(X, alpha=0.05, n_samples=10_000)`; `torch.set_num_threads(2)` |
| DGP frames | 3 brands × 20 seeds (21, 7, 99, 123, then 1..29 minus those), `n_records=3000`, heterogeneous DGP, `treatment_arm → treatment_initiated`, X = `ARM_CONFOUNDERS` (`disease_severity`, `academic_hcp`), truth = `df.attrs["true_ate"]`, segment map = `df.attrs["cate_by_segment"]` |
| live frames | the two spec §2 pairs, 1,500 rows each, pulled by `_load_agent_estimation_frame` with the route's brand-scoped default covariates (8 and 10 numeric columns) |
| comparator | LinearDML with the production RandomForest nuisances (`recovery_probe.py` configuration), `ate_inference(X).conf_int_mean()` |
| scripts | `prep_frames.py` (host, frames → parquet + truth json), `run_trial.py` (container), `summarize.py` |

## Findings before the sweep

1. **`causalpfn 0.1.4` interval wrapper is broken.** `CATEEstimator.estimate_ate_CI` returns `output["ate"]`, a key `_estimate_ate_cate_CI` never produces → `KeyError: 'ate'` on every call. The trial calls the helper directly (`ate_lower_bound` / `ate_upper_bound`). Recorded in `smoke_*.jsonl` first attempt.
2. **Temperature calibration collapses to the grid floor on a binary outcome.** With `calibrate=True` (the paper's OOD correction, 3-fold, 500 temperatures in [0.001, 10]) the chosen temperature on the seed-21 Remibrutinib frame is 0.00102 — the search minimum — and the 95 % ATE interval degenerates to width 0.001 ([0.230, 0.231]) that contains neither the planted truth (0.166) nor the model's own point estimate (0.138). Calibration also costs 418 s per frame.

## Smoke (one frame, `dgp_Remibrutinib_s21`, truth 0.1655)

| variant | ATE | |err| | 95 % interval | covers | segment order | t_total | ru_maxrss |
|---|---|---|---|---|---|---|---|
| CausalPFN, T = 1 (uncalibrated) | 0.138 | 0.027 | [0.114, 0.163] | no (misses by 0.003) | high > med > low ✓ | 60 s | 1.9 GiB |
| CausalPFN, calibrated (T = 0.001) | 0.138 | 0.027 | [0.230, 0.231] | no | ✓ | 470 s | 1.8 GiB |
| LinearDML (production config) | 0.146 | 0.020 | [0.101, 0.190] | yes | n/a | 4–7 s | — |

The 60 s uncalibrated cost splits as fit 1.3 s (weak learner), CATE forward passes 23 s, interval sampling 36 s. The 1.9 GiB peak is the interval's sample array (6,000 queries × 10,000 draws).

## Sweep 1 — the spec's criteria on 60 DGP frames (uncalibrated, T = 1; `sweep_cal0.jsonl`)

| brand | seeds | max \|ATE − truth\| pfn / dml | segment order ok | 95 % coverage pfn / dml | median interval width pfn / dml | median t_total | max ru_maxrss |
|---|---|---|---|---|---|---|---|
| Fabhalta | 20 | 0.063 / 0.057 | 18/20 | **10/20** / 18/20 | 0.052 / 0.099 | 61 s | 2.4 GiB |
| Kisqali | 20 | 0.063 / 0.070 | **11/20** | **13/20** / 16/20 | 0.050 / 0.096 | 60 s | 2.4 GiB |
| Remibrutinib | 20 | 0.067 / 0.051 | 17/20 | **11/20** / 19/20 | 0.051 / 0.098 | 58 s | 2.4 GiB |

Against spec §11 criterion 3 (all required):

| criterion | result | verdict |
|---|---|---|
| \|ATE − truth\| < 0.15 on every frame | 60/60 (max 0.067) | PASS |
| segment CATE order high > medium > low on every brand | 46/60; Kisqali 11/20 | **FAIL** |
| 95 % interval covers truth in ≥ 18/20 seeds | 10, 13, 11 of 20 (34/60) | **FAIL** |
| < 10 s per frame on 2 CPUs | 0/60 (median 60 s: 1 s weak learner, ~23 s CATE forward passes, ~36 s interval draws) | **FAIL** |
| peak RSS < 1 GiB | 0/60 (2.4 GiB; the 6,000 × 10,000 interval sample array) | **FAIL** |

Reading. The point estimate is fine (signed bias +0.016, sd 0.025, vs LinearDML +0.006) but the posterior
is **overconfident on a binary outcome**: its interval is half the width of the DML analytic interval
(0.05 vs 0.10) and misses are one-sided on the high side (e.g. `Remibrutinib_s17` truth 0.162, pfn
0.229 [0.203, 0.254], dml 0.213 [0.163, 0.264]). That is the coverage collapse spec §11 named as the
one fact that would reverse the recommendation. The ordering misses are a compression of the
high-vs-medium gap (Kisqali medium 0.26–0.34 vs high 0.23–0.27 while truth is 0.27 vs 0.29–0.32): the
segment map's top two levels are ~0.03 apart for Kisqali and the amortized CATE cannot resolve them,
where CausalForestDML in the repo's probe does (`tests/integration/test_dgp_recovery_probe.py`).

## Sweep 2 — the two live spec §2 pairs (1,500 rows, the route's brand-scoped covariates; `live_cal0.jsonl`, `live_cal1.jsonl`)

No planted truth here; the reference is the platform's reported interval (spec §2) and LinearDML refit on the same frame.

| pair | n treated | CausalPFN T = 1: ATE [95 %] | CausalPFN calibrated: T, ATE [95 %] | LinearDML refit: ATE [95 %] | reported CI (spec §2) | t_total T = 1 / calibrated |
|---|---|---|---|---|---|---|
| all brands, `treatment_arm → persistent_180d` | 247 | 0.062 [0.025, 0.099] | T = 0.00102, 0.062 **[0.168, 0.170]** | 0.094 [0.028, 0.161] | [0.019, 0.152] | 27 s / 213 s |
| Remibrutinib, `treatment_arm → treatment_initiated` | 258 | 0.156 [0.122, 0.190] | T = 0.00159, 0.156 **[0.201, 0.203]** | 0.174 [0.114, 0.233] | [0.117, 0.236] | 24 s / 222 s |

Reading. Uncalibrated, both point estimates sit inside the platform's reported intervals and the
CausalPFN interval is again about half the width of the analytic one (0.07 vs 0.13; 0.07 vs 0.12).
Calibrated, the temperature hits the search floor on both live frames exactly as on the DGP frame,
and the interval degenerates to a 0.002-wide band that does not contain the estimator's own point
estimate. Three calibrated frames (one DGP, two live), three identical collapses; peak RSS 1.2–1.4 GiB
at n = 1,500.

## Sweep 3 — calibrated (the paper's temperature scaling), seed 21 on each brand (`sweep_cal1_subset.jsonl`)

The full 12-frame calibrated subset was cut after the three seed-21 frames: the collapse was identical
on every calibrated frame run (six of six, DGP and live), and at 7.5 min per frame the variant is already
outside criterion 4 by 45×.

| frame | truth | chosen T | ATE | 95 % interval | covers | t_total |
|---|---|---|---|---|---|---|
| Fabhalta s21 | 0.125 | 0.00159 | 0.133 | [0.265, 0.267] | no | 480 s |
| Kisqali s21 | 0.212 | 0.00125 | 0.211 | [0.349, 0.351] | no | 469 s |
| Remibrutinib s21 | 0.166 | 0.00102 | 0.138 | [0.230, 0.231] | no | 452 s |

The 3-fold ICE calibration selects the smallest temperature in its grid (0.001) every time; the sampled
posterior then concentrates on a value ~0.1 above the point estimate and the interval is ~0.002 wide.
Whether that is a package defect (the samples use `self.temperature` while the point estimate uses
`prediction_temperature = 1.0`) or the documented OOD behaviour on a binary outcome, the prescribed
correction does not produce a usable interval here.

## Verdict — FAIL (spec §11 item 5)

| criterion | uncalibrated | calibrated |
|---|---|---|
| \|ATE − truth\| < 0.15, every frame | PASS 60/60 | PASS 3/3 |
| segment order, every brand | FAIL 46/60 (Kisqali 11/20) | 2/3 |
| coverage ≥ 18/20 per brand | FAIL 10 / 13 / 11 | FAIL 0/3 |
| < 10 s per frame, 2 CPUs | FAIL 0/60 (median 60 s) | FAIL (≈ 470 s) |
| peak RSS < 1 GiB | FAIL 0/60 (2.4 GiB) | FAIL (1.8–2.0 GiB) |

Only the point-estimate criterion passes, and LinearDML already passes it. The fact spec §11 named as
the one that would reverse the recommendation — coverage collapse on binary outcomes — is what the
sweep measured (34/60, one-sided high misses, intervals half the analytic width).

**Decision for #1991 debt 1.** The amortized-estimator route to removing the refutation node's
reconstruction machinery is closed by this evidence. Debt 1 is a *refactor* question (one fitted object
shared by estimation and refutation, or a cheaper refit), not a *replacement* question. Not integrated,
not a shadow candidate; `causalpfn` is not added to the image.

**What would reopen it.** A continuous outcome on this platform (none today), or a causalpfn release
whose calibration yields non-degenerate intervals on a binary outcome — re-run `run_trial.py` as is.

## Reproduce

```
# host, venv, repo .env exported: frames → scratch/trial/frames
python prep_frames.py <frames_dir> all
# scratch container from the deployed image (never the live e2i_api), packages in a mounted site dir
docker run --rm --user 1000:1000 --memory=2560m --memory-swap=2560m --cpus=2 -v <trial_dir>:/trial \
  -e HOME=/tmp -e HF_HOME=/trial/hf -e PYTHONPATH=/trial/site -w /app ghcr.io/enunezvn/e2i-api:<tag> \
  sh -c 'pip install --no-deps --target /trial/site causalpfn==0.1.4 faiss-cpu && python /trial/run_trial.py --tag sweep_cal0 --calibrate 0 --glob "dgp_*.json"'
python summarize.py <results_dir> sweep_cal0 live_cal0 live_cal1 sweep_cal1_subset
```
