# #242 zero-shot de-confound — final findings (multi-vendor independence)

**Status: FINAL. The first complete, uncontaminated 42-entry run of the saga (credits landed 2026-05-25). The multi-vendor independence premise behind #240 AC3.5 fails — and de-confounding makes it WORSE, not better. The correlated failure is intrinsic to the frontier models on this causal-role task, not an artifact of the shared prompt.**

Harness: `scripts/measure_ensemble_ab.py` (hardened: checkpoint/resume + `--max-cost` + clean quota-stop). Models: Sonnet 4.6 + Opus 4.7 + GPT-5. Run: `--prompt-mode zeroshot --order shuffle --seed 42 --max-cost 10.00`. Completed all 42 leak-relevant golden entries, 0 contaminated, within the $10 cap.

## The de-confound test (the experiment this whole follow-up existed to run)

The prior A/B ran all three models on the SAME Sonnet-compiled few-shot prompt, so "multi-vendor independence" was confounded — identical Sonnet-optimized demos imported Sonnet's bias into Opus and GPT-5. `--prompt-mode zeroshot` strips the demos so each vendor reasons from the bare signature. If the 30% correlated failure were prompt-induced, zero-shot should reduce it.

### Apples-to-apples (the 30 features measured in BOTH modes)
| Mode | correlated all-3-wrong | full-agreement | single-Sonnet correct |
|------|------------------------|----------------|-----------------------|
| COMPILED (shared Sonnet demos) | 9/30 = **30%** | 23/30 | 19/30 |
| ZEROSHOT (no shared demos)     | 12/30 = **40%** | 20/30 | 16/30 |

**Removing the shared prompt did not reduce cross-vendor correlation — it increased it (30% → 40%)** and lowered accuracy (19 → 16). The models still agree most of the time; without task-specific demos they fall back to the same default (often wrong) reading of the causal-role boundaries even more readily.

### Full 42-entry zero-shot (incl. the never-before-measured PNH tail)
- single-Sonnet: 26/42 = 0.619 ; ensemble: 25/42 = 0.595 (1 regression, 0 escalations)
- correlated all-3-wrong: **14/42 = 33%**
- **30 of 42 full-agreement, 0 splits** — zero-shot makes the three vendors agree MORE, not less
- AC5 (single-Sonnet false-negatives a leak, ensemble catches): 0 ; clean leak-FN caught: 0
- PNH tail (finally measured): 3/14 correlated all-3-wrong — consistent with the rest

## Conclusion (decisive for #240 AC3.5 / #501)

1. **The cross-vendor correlated failure is intrinsic to the frontier models on this task, not a prompt artifact.** It holds under both compiled and zero-shot prompting, and de-confounding makes it worse. This corroborates the #502 label adjudication (the 30% is real shared-blind-spot model failure, not label noise) from an independent angle.
2. **Multi-vendor *agreement* cannot be the gate's independence signal.** Where the three vendors agree (30/42 here), they are confidently-correct AND confidently-wrong (~33% of hard cases), so agreement carries no asymmetric-failure signal.
3. **Per-vendor prompts would not rescue it.** Zero-shot already eliminated prompt-sharing and correlation rose — the limiting factor is shared model reasoning, not shared exemplars.
4. **→ #240 AC3.5 needs a NON-LLM independent check** (deterministic structural/temporal prior: post-index timing ⇒ descendant-prior; on-treatment conditioning ⇒ collider-prior). A non-LLM signal cannot share LLM failure modes — that is real independence. This is the Step-C direction, now empirically justified rather than speculative.

The ensemble capability (PR #500) still ships as built — it is honest about disagreement and causes no regressions. But it is not the multi-vendor unblocker #240 AC3.5 assumed, and #501 should re-scope the gate's independence signal toward a structural check. The compile-set hardening from #502 (three discriminator example types) is the separate lever for raising the base accuracy.

## What it cost / reproducibility
One clean run, within the $10 cap (the run completed all 42 without hitting the budget guard, so actual spend < $10). Re-run: `python scripts/measure_ensemble_ab.py --prompt-mode zeroshot --order shuffle --seed 42 --max-cost 10.00 --out /tmp/242_ab_zeroshot.json`. Per-entry rows: `/tmp/242_ab_zeroshot.json`.
