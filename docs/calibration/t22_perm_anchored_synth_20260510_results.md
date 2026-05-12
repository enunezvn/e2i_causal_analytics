# T2.2 Perm-Anchored AUC Buffer — Calibration Sweep Results

**Generated:** 2026-05-12T13:43:58+00:00
**Source:** `scripts/calibration/aggregate_t22_sweep.py`
**Backlog:** #135 — close-out of `t22_perm_anchored_synth_20260510.md` §3.

## 1. Recommended buffer

Two readings of the §2.3 threshold-fit logic are surfaced:

- **Mechanical** (all non-degenerate targets): raw=`-0.1447` floored=`-0.1500` recommended=`0.0000` (limiting target_auc=`0.55`, clamped to 0=True, n_eligible=7)
- **Well-conditioned** (cells where P5 margin > 0, i.e., even worst seed beats perm null): raw=`0.0597` floored=`0.0500` recommended=`0.0400` (limiting target_auc=`0.7`, clamped to 0=False, n_eligible=4)

**Why two readings:** the mechanical reading enforces the spec's exact words
("the buffer must pass at every target point") and clamps to 0.0 when any
low-signal target cell has a negative P5 margin — i.e., when the regime can
produce a nominal AUC the model cannot reliably exceed the perm-null p99 at.
At small n (≈ 400 held-out), that floor is empirically ≈ 0.55-0.60 AUC, so any
target cell below ≈ 0.65 risks the model not separating from noise.

The well-conditioned reading restricts the fit to target cells where the
regime is producing genuine signal (mean margin > 0). This is the practical
interpretation: how much above noise must a model be for the advisory to
clear it, given that the lowest target cells are below the perm-null floor
by construction at this n.

**Recommendation:** adopt the **well-conditioned** buffer (`0.0400`) — the mechanical reading clamps to 0 not because
the calibration says "no buffer needed" but because the regime spans target AUCs
that are sub-noise-floor at the sweep's sample size. The mechanical 0.0 reading is
a tautology (any buffer ≥ 0 fires on those cells); the well-conditioned reading is
the empirically meaningful floor for the cells where the regime produces signal.

## 2. Per-target summary

| Target AUC | n | Realized (mean ± std) | Perm null p99 (mean ± std) | Margin (mean ± std) | Margin (P5 = min) | Drift vs target | Flagged |
| ---------- | - | --------------------- | -------------------------- | -------------------- | ----------------- | --------------- | ------- |
| 0.55 | 5 | 0.5514 ± 0.0505 | 0.6012 ± 0.0044 | -0.0499 ± 0.0514 | -0.1447 | +0.0014 | no |
| 0.60 | 5 | 0.6005 ± 0.0265 | 0.5876 ± 0.0024 | +0.0129 ± 0.0260 | -0.0091 | +0.0005 | no |
| 0.65 | 5 | 0.6111 ± 0.0469 | 0.5865 ± 0.0044 | +0.0246 ± 0.0451 | -0.0143 | -0.0389 | YES |
| 0.70 | 5 | 0.6766 ± 0.0363 | 0.5781 ± 0.0069 | +0.0985 ± 0.0342 | +0.0597 | -0.0234 | YES |
| 0.75 | 5 | 0.7606 ± 0.0159 | 0.5748 ± 0.0116 | +0.1858 ± 0.0131 | +0.1759 | +0.0106 | no |
| 0.80 | 5 | 0.7991 ± 0.0229 | 0.5635 ± 0.0067 | +0.2356 ± 0.0242 | +0.2066 | -0.0009 | no |
| 0.85 | 5 | 0.8672 ± 0.0121 | 0.5620 ± 0.0059 | +0.3051 ± 0.0136 | +0.2910 | +0.0172 | no |

**Total cells:** 35

## 3. Acceptance criteria (`docs/calibration/t22_perm_anchored_synth_20260510.md` §5)

- All cells produced a non-error pipeline run: YES
- Per-target realized AUC within ±0.02 of target (mean across seeds): NO (2 flagged — see table)
- Recommended buffer passes the synthetic_rwd_realistic regime's pinned [0.62, 0.68] cell: see the integration pin test (`tests/integration/test_t22_perm_anchored_auc_advisory.py`). When `buffer_recommended` ≤ the margin at signal_scale=1.0, the advisory does NOT fire on the pin.
- Recommended buffer rejects a pure-noise (signal_scale=0) cell: ensured by construction — at signal_scale=0 the realized AUC ≈ 0.50, the perm null p99 ≈ 0.55-0.58, and the margin is negative; any non-negative buffer fires the advisory.

## 4. Reproduction

```bash
for seed in 0 1 2 3 4; do
  for auc in 0.55 0.60 0.65 0.70 0.75 0.80 0.85; do
    PYTHONPATH=. python scripts/calibration/run_t22_synth_sweep.py \
      --seed "$seed" --target-auc "$auc" \
      --output-jsonl "calibration_runs/t22_synth_seed${seed}_auc${auc}.jsonl"
  done
done
PYTHONPATH=. python scripts/calibration/aggregate_t22_sweep.py \
  --input-glob 'calibration_runs/t22_synth_*.jsonl' \
  --output-md docs/calibration/t22_perm_anchored_synth_20260510_results.md
```
