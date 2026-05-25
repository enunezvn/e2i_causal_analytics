# Plan #242 — Layer-4 multi-model worker ensemble (Sonnet 4.6 + Opus 4.7 + GPT-5)

**Issue:** #242 — Layer-4 Phase 4.3 multi-model ensemble, agreement-or-escalate.
**Branch:** `feat/242-multi-model-ensemble` (worktree `.claude/worktrees/242-ensemble`, off main `147765e3`).
**Goal of this PR:** ship the ensemble harness that satisfies all five #242 acceptance criteria → `Closes #242`.

> **Reason-before-rules note.** #242 was filed as a *deferred tracker* ("activate when a trigger fires"). The user has chosen to prioritize it now (their call as stakeholder). It is also the named HARD prerequisite (#240 plan AC3.5) for the #240 severity-gate. We build it **offline-first**: the live `adaptive_validity_check` node (`adaptive_validity_check.py:3054`) stays single-Sonnet. Wiring a 3× LLM call into a live LangGraph path now would (a) triple cost on every ambiguous feature and (b) feed a fused role of unproven calibration into the voter + KG mirror — plausible-but-unvalidated values in a production-reachable path, which the project's anti-mocking discipline forbids. The ensemble's first job is to *produce the evidence* (offline A/B + the known-leak test) that the #240 gate needs before it gates anything live.

## #242 Acceptance Criteria → how this PR meets them

| AC | Requirement | This PR |
|----|-------------|---------|
| AC1 | 3-model harness in `src/data/causal_role_classifier_ensemble.py`: Sonnet 4.6 + Opus 4.7 + GPT-5 in parallel | `classify_feature_ensemble()` runs the existing `CausalRoleClassifier` DSPy module 3× via `dspy.context(lm=...)`, parallel via `ThreadPoolExecutor` |
| AC2 | Agreement: 3/3 → auto-verdict; 2/3 → majority + "split" audit; 1/3 (all-disagree) → `unknown` | `_fuse_votes()` pure function; agreement on `causal_role` |
| AC3 | Disagreement routing: configurable consumer (curation CLI vs gate path — see #240) | Fused result adapts to `LLMVerdict` + an `LLMEvaluatorAudit`-shaped trust signal the existing voter/#240-gate already consume; curation CLI surfaces the split. Consumer selected by config, default = offline curation/harness |
| AC4 | Cost + latency telemetry per provider (mirrors #241) | New `EnsembleModelVote` carries per-provider latency/tokens/cost; reuses `_extract_lm_usage` dual OpenAI/Anthropic usage extraction; new per-provider rate constants |
| AC5 | Integration test plants a known leak single-Sonnet false-negatives on; assert ensemble catches it | `tests/integration/test_ensemble_catches_single_sonnet_miss.py`, live (skips without all 3 keys). **Empirically gated — see §7.** |

## 1. Design (approved)

Build the **3-model worker ensemble whose agreement-state doubles as the multi-vendor trust signal #240 needs**. One artifact satisfies #242's literal worker ACs *and* #240's AC3.5 dependency.

Flow: `classify_feature_ensemble(feature_name, derivation_pseudocode, dataset_context)`:
1. For each of 3 models, run `CausalRoleClassifier()(...)` inside `with dspy.context(lm=dspy.LM(model_string)):` — parallel via `ThreadPoolExecutor(max_workers=3)`.
2. Collect per-model `EnsembleModelVote` (role, mechanism, remediation, telemetry, error-or-None).
3. `_fuse_votes()` → `EnsembleClassification` (fused role, agreement level, votes, aggregate telemetry).
4. Adapt to the downstream contract: a fused `LLMVerdict` (majority role/mechanism/remediation) + an `LLMEvaluatorAudit` (`satisfied = (agreement == FULL)`, `missed_considerations = per-model dissent labels`, `notes = split summary`, `evaluator_model = "ensemble:<sonnet>+<opus>+<gpt>"`).

**Degrade-to-healthy (user decision):** a model error/timeout = non-vote. 2 healthy agree → majority; ≤1 healthy → escalate `unknown`. Per-provider failure recorded in its `EnsembleModelVote.error`.

**Agreement levels:**
- `FULL` (3/3 healthy agree, or 2/2 when 1 errored): fused role = that role, `satisfied=True`.
- `MAJORITY` (2/3 agree, 3rd dissents): fused role = majority role, `satisfied=False`, split audit lists the dissenter.
- `SPLIT` (all 3 disagree, or ≤1 healthy vote): fused role = `unknown` sentinel, `satisfied=False`, escalate.

## 2. New types (`src/data/kg/types.py`, additive/frozen)

```python
EnsembleAgreement = Literal["full", "majority", "split"]

@dataclass(frozen=True)
class EnsembleModelVote:
    model: str                      # provider-prefixed, e.g. "openai/gpt-5"
    causal_role: Optional[CausalRole]   # None when this model errored
    mechanism: str = ""
    recommended_remediation: Optional[Remediation] = None
    latency_ms: Optional[float] = None
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    cost_usd: Optional[float] = None
    error: Optional[str] = None     # short error label; None when healthy

@dataclass(frozen=True)
class EnsembleClassification:
    feature_name: str
    agreement: EnsembleAgreement
    fused_role: Optional[CausalRole]    # None ⇒ escalate (split)
    fused_mechanism: str
    fused_remediation: Optional[Remediation]
    votes: tuple[EnsembleModelVote, ...]
    healthy_votes: int
    total_cost_usd: Optional[float]
    max_latency_ms: Optional[float]     # ensemble wall-time ≈ slowest model
```

## 3. New module `src/data/causal_role_classifier_ensemble.py`

- `ENSEMBLE_MODELS` default from env (read at **call time**, not import):
  - `ENSEMBLE_SONNET_MODEL` default `anthropic/claude-sonnet-4-6`
  - `ENSEMBLE_OPUS_MODEL` default `anthropic/claude-opus-4-7`
  - `ENSEMBLE_GPT_MODEL` default `openai/gpt-5`
- `classify_feature_ensemble(...) -> Optional[LLMVerdict]` — public entry; returns `LLMVerdict` (with the `LLMEvaluatorAudit` trust sidecar attached) or `None` when no model is reachable.
- `run_ensemble_classification(...) -> EnsembleClassification` — richer result for the harness/curation consumers.
- `_classify_one(model, ...) -> EnsembleModelVote` — single-model run under `dspy.context`, catches per-model exceptions → `error`.
- `_fuse_votes(feature_name, votes) -> EnsembleClassification` — **pure**, no I/O; the heart of AC2/degrade logic.
- `_ensemble_to_llm_verdict(classification) -> LLMVerdict` — adapter (AC3).
- `_preflight_models(models) -> None` — loud check that each provider's key is present (reuses loader `_PROVIDER_TO_ENV_VARS`); raises a clear error naming the missing key. No silent fallback.

### Telemetry (AC4) — per-provider rate constants
Add documented per-MTok constants (mirroring `HAIKU_*_USD_PER_MTOK`): `SONNET_*`, `OPUS_*`, `GPT5_*`. `_cost_for(model, in_tok, out_tok)` maps model→rates. Reuse loader `_extract_lm_usage` (handles both `prompt_tokens/completion_tokens` and `input_tokens/output_tokens`). `None`-stamp when usage absent.

## 4. Consumers wired this PR

- **`scripts/measure_layer4_precision.py`**: add `--ensemble` flag. When set, the overridable `_classify_feature` indirection (line 418) resolves to `classify_feature_ensemble`. Enables offline single-Sonnet vs ensemble A/B. (No behavior change when flag absent.)
- **Curation surfacing**: `src/data/audit_candidate_formatter.py` already renders `evaluator_satisfied` + `missed_considerations`; the ensemble's `LLMEvaluatorAudit` sidecar flows through unchanged. Add the per-provider split breakdown to the formatter's "Promotion candidate" section (additive).
- **NOT wired:** `adaptive_validity_check.py:3054` stays `classify_feature` (single-Sonnet). Add a documented `ADAPTIVE_VALIDITY_ENSEMBLE_ENABLED` env stub (default OFF) marking the future live seam — **read but not acted on** in this PR beyond a logged "not yet activated" notice, so the wiring point is explicit without flipping production. (If even the stub risks confusion, drop it and document the seam in this plan only — decide during codex review.)

## 5. Files changed

| File | Change |
|------|--------|
| `src/data/kg/types.py` | +`EnsembleAgreement`, +`EnsembleModelVote`, +`EnsembleClassification` (additive) |
| `src/data/causal_role_classifier_ensemble.py` | **new** module |
| `scripts/measure_layer4_precision.py` | +`--ensemble` flag → `_classify_feature` routing |
| `src/data/audit_candidate_formatter.py` | +per-provider split breakdown in promotion section (additive) |
| `tests/unit/test_data/test_causal_role_classifier_ensemble.py` | **new** — fuse logic, degrade, telemetry, adapter, preflight |
| `tests/unit/test_data/test_kg/test_types.py` | +construction/frozen tests for new types |
| `tests/integration/test_ensemble_catches_single_sonnet_miss.py` | **new** — live, skippable (AC5) |
| `docs/plans/242-multi-model-ensemble.md` | this plan |

## 6. Phased TDD plan (red-first; each phase: write failing test → run RED → implement → GREEN)

- **P0** New types (P-types): construct + frozen + defaults. RED→GREEN.
- **P1** `_fuse_votes` pure logic — the bulk:
  - 3/3 agree → `full`, fused=role, satisfied path.
  - 2/3 agree, 1 dissent → `majority`, fused=majority, dissenter in split.
  - 2/2 agree, 1 errored (non-vote) → `full`, healthy_votes=2.
  - 1 healthy only → `split`/escalate, fused_role=None.
  - all 3 distinct roles → `split`.
  - all 3 errored → `split`, healthy_votes=0, fused_role=None.
  - tie-break policy when 2 distinct roles each have 1 vote + 1 error (i.e., 1-1): treated as `split` (no majority). Documented.
- **P2** `_cost_for` + telemetry aggregation: per-model rates, `total_cost_usd` sums healthy votes, `None`-stamp when usage missing; `max_latency_ms` = slowest.
- **P3** `_ensemble_to_llm_verdict` adapter: full→satisfied=True empty missed; majority/split→satisfied=False + dissent labels; sidecar `evaluator_model` string shape.
- **P4** `_classify_one` + `classify_feature_ensemble` with **stubbed `dspy.context`/classifier** (monkeypatch — no live API): healthy path, single-model exception → `error` vote + degrade, all-fail → None-or-split. Mirror the `dspy.context` stub pattern from `test_causal_role_classifier.py:439`.
- **P5** `_preflight_models`: missing `OPENAI_API_KEY` (monkeypatched absent) → raises naming the key; all present → passes.
- **P6** `measure_layer4_precision.py --ensemble`: flag routes `_classify_feature` (unit test patches the module symbol, asserts ensemble entry selected; no live call).
- **P7** `audit_candidate_formatter` split breakdown: additive render test.
- **P8** (live, skippable) AC5 known-leak integration — see §7.

## 7. AC5 — the one empirically-gated criterion (HONEST RISK)

AC5 requires a real feature where the **ensemble catches a leak single-Sonnet mislabels**. This is verified by a *live* run with `.env` keys, not assumed.

Procedure:
1. Construct candidate leak features (adversarial/edge: e.g. a post-index-window aggregate that single-Sonnet has historically scored as `ancestor`/benign but is a `descendant` leak). Seed from the plan-239 adversarial bucket + the `causal_role_golden_set.json` edge cases.
2. Run all 3 models live; find a case where single-Sonnet's role is wrong AND ≥2 of {Sonnet,Opus,GPT-5} (or the majority) get it right, so the ensemble's fused role/escalation differs from single-Sonnet and is correct.
3. Pin that case as the integration fixture; test asserts: single-Sonnet alone mislabels, ensemble catches (correct majority OR escalates to `unknown`/`split` for review).
4. Test `skipif` when any of the 3 keys absent (RAGAS pattern) — CI stays green; the live proof is run once locally and recorded in the PR body.

**If no such case is found** after a reasonable search, AC5 is not met → we do **not** assert `Closes #242`; instead ship as `Refs #242` with the harness + the negative finding documented (i.e., "single-Sonnet and ensemble agree on all tested cases at current calibration"), and the user decides. This is the gating decision before claiming the close.

## 8. Out of scope (belongs to #240, data/human-gated — NOT #242 ACs)

Live-path gate activation; FP-rate AC2.1 (<10%); inter-rater κ AC2.2 (≥0.6); stakeholder sign-off; production routing toggle. The ensemble is the *input* to those; they are measured downstream with operational data.

## 9. Verification / definition of done

- All new unit tests GREEN; targeted existing suites (`test_kg`, `test_causal_role_classifier`, `test_evaluator_telemetry`, measure-script unit tests) still GREEN.
- `ruff check` + `ruff format --check` + `mypy` clean on changed files (CI parity).
- AC5 live run executed once with `.env` keys; result recorded (close vs Refs decision per §7).
- codex:codex-rescue audit → fixed point (ACCEPT), with the design-pushback brief (anti-mocking / labeling-vs-functional / intent-investigation).
- PR body: `Closes #242` only if AC5 met; otherwise `Refs #242`. No closing-keyword bigram for #240 (it stays OPEN). Merge `--merge` (never squash).
