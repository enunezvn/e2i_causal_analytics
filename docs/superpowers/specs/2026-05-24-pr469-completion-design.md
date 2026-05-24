# Design: Complete PR #469 — n=200 compile-set + gated A/B + AC3 close on #239

**Status**: Approved 2026-05-24
**Author**: brainstormed with E. Nuñez via Claude Code
**Linked issues**: #239 (MIPROv2 upgrade), #468 (gated A/B successor), PR #469 (Branch B partial-ship)

---

## 1. Goal

Resolve #239 by extending the MIPROv2 vs BootstrapFewShot comparison from underpowered n=50 (current state, tied ungated) to decision-quality n≈200, then running the gated A/B with the Layer-4 Haiku audit evaluator enabled, then making the AC3 decision per a strengthened rule.

## 2. Sequencing (stacked PRs)

```
Step 1 (today, ~5 min)         Step 2 (~5-7 days)
─────────────────────          ─────────────────────────────────────────
merge PR #469 (Branch B)       new branch off main: feat/239-miprov2-n200
        │                               │
        ▼                               ▼
   #239 stays open               monolithic PR with bucket-staged commits
   #468 stays open               (9 commits, codex audit between buckets)
                                        │
                                        ▼
                                 AC3 verdict commit
                                        │
                                        ├── MIPROv2 wins → Closes #239 + #468
                                        └── MIPROv2 fails → enhanced compile-set
                                                            ships, #239 stays open
                                                            with rich rationale
```

PR #469 is iter-1 ACCEPT'd as Branch B. Merging it first locks in already-validated MIPROv2 wiring + 17 differentiated compile-set entries. The new PR is the close-vehicle for #239 if MIPROv2 wins the strengthened AC3 rule.

## 3. AC3 rule (strengthened from plan §7.1)

```
MIPROv2 wins iff
    mipro.overall.gated.precision_instrument >= boot.overall.gated.precision_instrument
    AND
    ∀ cohort c ∈ {CSU, PNH, BC}:
        mipro[c].gated.precision_instrument >= boot[c].gated.precision_instrument
```

Rationale: at n=50, the plan's STRICT `≥` rule lets ties pass — but the ungated A/B showed per-cohort divergence (MIPROv2 +1 BC, −1 PNH). Flipping the default for a production-routing classifier on aggregate-tie + per-cohort regression would not be data-driven. At n≈200 the per-cohort no-regression rule should be reachable if MIPROv2 has real signal.

## 4. Components touched (new PR)

| File | Change | Size |
|---|---|---|
| `src/data/causal_role_classifier.py` | +150-160 `dspy.Example()` entries (total ~200-210); normalize cohort tags on existing 50 ({csu, CSU}→CSU; {bc, HR+_BC, HR+_BreastCancer}→BC; {pnh}→PNH) | +~1800 LoC |
| `scripts/compile_causal_role_classifier.py` | Bump `max_labeled_demos` default 40→210 | +1 line |
| `scripts/measure_layer4_precision.py` | Document `ADAPTIVE_VALIDITY_EVALUATOR_ENABLED=1`; add `--enable-evaluator` convenience flag that sets env in-process | +~20 LoC |
| `tests/unit/test_data/test_causal_role_classifier.py` | TDD red-first tests (size + cohort floors + canonical tags + disjointness) | +~150 LoC |
| `tests/integration/test_measure_layer4_precision_audit_eval.py` | New: audit-eval populates `evaluator_audit` when env enabled | +~80 LoC |
| `artifacts/dspy/causal_role_classifier.json` | Replaced post-compile (winning artifact) | regenerated |
| `artifacts/dspy/ac3_verdict_n200.json` | New: per-cohort gated metrics + AC3 verdict | new |
| PR body | AC3 verdict + per-cohort metrics table + decision rationale | (PR description) |

Existing 50-entry composition (for reference):
- Roles: confounder 11, descendant 10, instrument 9, collider 8, mediator 7, ancestor 5
- Cohorts (pre-normalization): CSU 8 + csu 3 = 11 CSU; bc 6 + HR+_BC 2 = 8 BC; pnh 5; synthetic_a1/a2/a4 = 6 synthetic; hypertension 2; remainder untagged

## 5. Compile-set growth plan (50 → 200, balanced cohort floors)

**Buckets** (4 buckets × ~35-40 entries each):

| Bucket | Target | Sources | Notes |
|---|---|---|---|
| 1: PNH expansion | +45 (30 lit + 15 adversarial/edge) | PubMed PNH/Fabhalta/iptacopan; Hernan/Brookhart short-term IV designs | Hardest bucket — current floor 5; aim PNH ≥60 (over-floor) to buffer AC3 risk |
| 2: BC expansion | +45 (40 lit + 5 adversarial/edge) | PubMed BC/Kisqali/ribociclib; CDK4/6 trial-mimicking observational designs | Moderate — current floor 8 normalized |
| 3: CSU expansion | +40 (35 lit + 5 adversarial/edge) | PubMed CSU/Remibrutinib/biologics; chronic-urticaria registry | Easiest — current floor 11 normalized |
| 4: Cross-cohort + synthetic | +30 | Adversarial worker-evaluator boundary + edge-case + synthetic-DGP | Diversifies away from cohort skew |

**Per-entry curation contract** (carried forward from plan-239 §3.0):
- `nearest_neighbor` (feature name + role + cohort) against {existing 50 + golden 91 + prior-bucket-new entries}
- `distance_assessment` (lexical edit distance + role + cohort + derivation signature)
- `why_not_duplicate` justification
- Provenance: PMID/DOI for lit entries; adversarial-design rationale for boundary entries; DGP spec for synthetic

**Cohort tag normalization** (commit 2, before any growth):
- Map: `{csu, CSU} → CSU`; `{bc, HR+_BC, HR+_BreastCancer} → BC`; `{pnh} → PNH`; `synthetic_*` preserved.
- Helper `_canonical_cohort(example)` in `causal_role_classifier.py`; raises on unknown.
- Tests in commit 1 (RED) lock the canonical set; commit 2 (GREEN) normalizes.

## 6. Audit-eval invocation (minimal change)

The Layer-4 Haiku audit evaluator is **already wired** into `classify_feature` (`src/data/causal_role_classifier_loader.py:585-598`). It's gated by env var `ADAPTIVE_VALIDITY_EVALUATOR_ENABLED=1` (`:336`). The §6.8 gated subset was empty *not* because wiring is missing — the env var simply wasn't set at script invocation time.

**Change to `scripts/measure_layer4_precision.py`**:
- Add `--enable-evaluator` flag that sets `os.environ["ADAPTIVE_VALIDITY_EVALUATOR_ENABLED"] = "1"` before importing the loader.
- Update docstring to surface the env-var contract for direct invocations.
- No change to `classify_feature` or `_build_evaluator` themselves.

## 7. Data flow (gated A/B execution, commit 8)

```
ADAPTIVE_VALIDITY_EVALUATOR_ENABLED=1
ANTHROPIC_API_KEY=sk-ant-...
        │
        ▼
scripts/measure_layer4_precision.py --classifier-artifact cr_bootstrap_n200.json --evaluator-gate true
        │                                          ──┘
        ▼                                          (also run for cr_miprov2_n200.json)
For each of 91 golden entries:
  classify_feature(name, pseudocode, ctx)
    ├── worker: DSPy classifier → verdict.causal_role
    └── audit:  Haiku evaluator → verdict.evaluator_audit.satisfied (True/False)
        │
        ▼
gated subset = {entries where verdict.evaluator_audit.satisfied == True}
        │
        ▼
overall.gated.precision_instrument = TP_instrument / (TP_instrument + FP_instrument) on gated subset
per-cohort breakdown: same metric scoped to {CSU, PNH, BC}
        │
        ▼
AC3 verdict = (mipro.overall.gated >= boot.overall.gated)
              AND ALL cohort c: (mipro[c].gated >= boot[c].gated)
        │
        ├── True  → flip default artifact → Closes #239 + #468 in PR body
        └── False → enhanced compile-set ships; #239 stays open with full metrics table
```

## 8. TDD test plan

**Commit 1: RED-first** (`tests/unit/test_data/test_causal_role_classifier.py`):

```python
def test_compile_set_size_meets_dspy_miprov2_floor():
    """MIPROv2 documented floor is ~200; we target ≥200 for AC3 power."""
    assert len(build_compile_set()) >= 200

def test_compile_set_cohort_floors():
    """No-per-cohort-regression AC3 requires balanced cohort representation."""
    floors = {"CSU": 50, "PNH": 50, "BC": 50}
    counts = Counter(_canonical_cohort(x) for x in build_compile_set())
    for cohort, floor in floors.items():
        assert counts[cohort] >= floor, f"{cohort}: {counts[cohort]} < {floor}"
    assert counts["synthetic_or_other"] >= 50

def test_compile_set_cohort_tags_canonical():
    """No silent skew from CSU/csu, bc/HR+_BC, pnh inconsistency."""
    CANONICAL = {"CSU", "PNH", "BC", "synthetic_a1", "synthetic_a2",
                 "synthetic_a3", "synthetic_a4", "hypertension"}
    for x in build_compile_set():
        assert _canonical_cohort(x) in CANONICAL

def test_compile_set_disjoint_from_golden():
    """Compile-set entries must not appear in the 91-entry literature golden set."""
    golden_keys = {_entry_key(g) for g in _load_golden()}
    for x in build_compile_set():
        assert _entry_key(x) not in golden_keys
```

**Mid-build integration test** (after audit-eval wiring commit):
```python
# tests/integration/test_measure_layer4_precision_audit_eval.py
@pytest.mark.live_lm
def test_evaluator_audit_populates_when_env_enabled(monkeypatch, tmp_path):
    """ADAPTIVE_VALIDITY_EVALUATOR_ENABLED=1 + ANTHROPIC_API_KEY → evaluator_audit non-None."""
    monkeypatch.setenv("ADAPTIVE_VALIDITY_EVALUATOR_ENABLED", "1")
    # invoke measure script on a 3-entry subset; assert gated subset > 0
```

**Final AC3-decision test** (commit 8, deterministic):
```python
def test_ac3_decision_rule_strict_ge_and_no_cohort_regression():
    """MIPROv2 wins iff overall AND every cohort tie-or-beats Bootstrap."""
    # mock metrics tables; assert decision is "miprov2_wins" / "bootstrap_holds"
```

## 9. Error handling

| Phase | Failure mode | Response |
|---|---|---|
| Cohort normalization | Existing entry has untagged or non-canonical cohort | `_canonical_cohort()` helper raises on unknown; PR commit 2 must add explicit tags |
| PNH lit pool exhaustion | <30 plausible PNH lit entries findable | Backfill PNH adversarial/edge to hit floor 50; document in PR body |
| Disjointness collision | A new entry's nearest neighbor distance < threshold | Reject in ralph-loop, request rework before commit |
| Compile cost overrun | MIPROv2 at n=200 costs > $20 | Lower `num_trials`; document deviation |
| Audit eval rate-limit | Haiku rate-limit during 91×2 A/B | Retry with backoff (already in `_run_evaluator`); fall through to `evaluator_audit=None` on permanent failure — entry drops from gated subset, ungated still scored |
| AC3 fails | MIPROv2 regresses some cohort | PR ships enhanced compile-set; #239 stays open; PR body documents which cohort regressed + recommended next step |
| Live LM unavailable in CI | CI can't run gated A/B | A/B is **local-only** (live-LM tier); CI runs unit tests + dry-run mode; AC3 verdict committed as a JSON artifact + PR body table from local run |

## 10. Top risks

1. **AC3 fails at n=200** (~40% probability per pre-mortem): per-cohort regression most likely failure shape since PNH lit pool is thin. Mitigation: aim PNH ≥60 (over-floor by 10).
2. **Curation calendar overrun**: 150 entries × multi-bucket codex rounds could stretch to 7-10 days. Mitigation: hard time-box buckets at 2 days each; if a bucket can't reach floor cleanly, document the gap and ship with PR-body acknowledgment.
3. **Disjointness regression**: with ~300 entries to check against, semantic-neighbor table becomes noisy. Mitigation: automated lexical-distance pre-filter (Levenshtein ratio ≥ 0.85 on feature_name **OR** identical (role, cohort, target) triple → flag for manual review). Threshold set in planning phase; tunable per bucket if false-positive rate too high.
4. **MIPROv2 finds shortcut on cohort-imbalanced data**: if growth ends up skewed despite floors, optimizer may overfit majority cohort. Mitigation: AC3 per-cohort rule catches this empirically.

## 11. Tool deployment

| Tool | When | Scope | Output |
|---|---|---|---|
| `superpowers:test-driven-development` | Commit 1 (start) + after each major contract change | Write tests first, watch RED, implement to GREEN | RED→GREEN log in commit messages |
| `ralph-wiggum:ralph-loop` | Each curation bucket (4 buckets) | Iterate: propose ~10-entry batch → check disjointness → codex review → revise → commit batch. Stops at bucket floor met. | Per-bucket commit on branch |
| `codex:rescue` (codex-rescue agent) | Per-bucket curation review (~5-10 min/bucket) + final pre-merge spec audit + when ralph-loop gets stuck | Bucket-scoped diff review; final whole-PR audit; root-cause when iteration stalls | Codex memo committed to gitignored `.claude/plans/` |
| Live LM (Anthropic Sonnet + Haiku) | Compile steps (commit 7) + gated A/B (commit 8) | Two compile runs + 182 Haiku audit calls | `cr_bootstrap_n200.json` + `cr_miprov2_n200.json` + AC3 JSON report |

## 12. Costs

- Compile MIPROv2 at n≈200: ~$8-15
- Compile BootstrapFewShot at n≈200: ~$2-3
- Gated A/B (91 entries × 2 artifacts × Haiku audit): ~$2-5
- Codex review per bucket (× 4 buckets): ~$5-10
- **Total live-LM**: ~$20-35

## 13. Exit conditions (new PR ready to land)

1. All TDD tests GREEN.
2. `mypy --config-file pyproject.toml src/` strictly non-increasing vs main baseline.
3. `ruff check src/ scripts/ tests/` clean on changed files.
4. Codex final pre-merge audit returns 0 HIGH findings (iter-ACCEPT).
5. AC3 verdict JSON committed to `artifacts/dspy/ac3_verdict_n200.json` with full per-cohort table.
6. PR body has explicit `Closes #239` + `Closes #468` (if MIPROv2 wins) or `Refs #239 — AC3 not met, ships enhanced compile-set + measurement infra` (if MIPROv2 loses).

## 14. Out-of-scope

- Audit-evaluator promotion from audit-only to severity-modulating gate (#240 tracker, separate plan)
- Multi-model ensemble (Sonnet + Opus + GPT-5) for Layer-4 (#242 backlog)
- Compile-set growth to >300 (deferred follow-up if AC3 fails at 200)
- Cohort tag normalization in the 91-entry golden set (read-only here; if needed, separate chore PR)
