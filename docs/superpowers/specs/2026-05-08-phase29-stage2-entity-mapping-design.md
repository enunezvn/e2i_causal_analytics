# Phase 2.9 Stage 2 Entity-Mapping Design

**Date:** 2026-05-08
**Status:** Design — pending approval
**Prerequisite:** `2026-05-08-kg-predicate-reconciliation-design.md` (PR-0) MUST land first
**Plan reference:** `.claude/plans/adaptive_temporal_validity_redesign.md` §Phase 2.9

## Goal

Wire `KnowledgeGraphQuerier` (PR #86) output into Layer 5's `adaptive_validity_check` via the `EnsembleVoter` (PR #88) so per-feature verdicts can carry `layer="2"` KG signals. The blocker has been feature → entity-code mapping: the voter accepts `kg_edges`, `feature_entity_ids`, `target_entity_ids`, but no upstream component populates them per feature. This design closes that gap.

## Disease-agnostic posture

The `e2i` plan thesis is "intelligent and adaptive to disease or brand." This design separates universal pipeline infrastructure (built once) from disease-specific declarative content (authored per indication):

**Universal pipeline (built once, never changes per disease):**

- `FeatureContract.kg_entity_codes` — schema field on the existing dataclass.
- `scope_spec["target_entity_codes"]` — runner contract, set per-cohort.
- `scripts/build_kg_cache.py --manifest-module {path}` — parameterized; reads any registered manifest from `MANIFEST_SOURCES` (`src/data/manifests/__init__.py:48`).
- `_compose_legacy_verdict` accepts `kg_edges` from a cache loader; reads `scope_spec["kg_cache_path"]`. **No cohort string in pipeline code.**
- Cache filename: `{manifest_fingerprint_sha8}__{target_codes_fingerprint_sha8}.json`. **No cohort name in path.**
- Shadow-mode promotion criteria: applied per `(manifest, target)` tuple independently.

**Disease-specific declarative layer (per-disease authorship):**

- Per-disease manifest module (`csu_feature_manifest.py`, `optum_feature_manifest.py`, future `{disease}_feature_manifest.py`).
- One-line entry in `MANIFEST_SOURCES`.
- Runner sets `scope_spec["target_entity_codes"]`.

**Adding a new disease/brand:**

1. Author a manifest module + register in `MANIFEST_SOURCES`.
2. Set `scope_spec["target_entity_codes"]` in the cohort runner.
3. Run cache builder.
4. Pipeline picks up via `scope_spec["kg_cache_path"]`. **Zero pipeline code changes.**

A new BRAND within an existing disease is even simpler: just change `scope_spec["target_entity_codes"]`. No new manifest needed. The cache fingerprint changes (because `target_codes_fingerprint` changes), so the builder regenerates only the affected cache.

## Architecture

```
                ┌─────────────────────────────────┐
                │  Manifest module (per disease)  │
                │  FeatureContract(                │
                │    name="...",                   │
                │    kg_entity_codes=(             │
                │      ("ICD10CM", "L20.9"),       │
                │      ("UMLS",    "C0011615"),    │
                │    ),                            │
                │    ...                           │
                │  )                               │
                └────────────────┬────────────────┘
                                 │ (build-time)
                                 ▼
                ┌─────────────────────────────────┐
                │  scripts/build_kg_cache.py       │
                │  --manifest-module {path}        │
                │  Validates entity codes via      │
                │  EntityLinker; queries KG;       │
                │  writes per-feature provenance.  │
                └────────────────┬────────────────┘
                                 │ (artifact: gitignored)
                                 ▼
              data/kg_cache/{mf_sha8}__{tc_sha8}.json
                                 │
                                 │ (run-time)
                                 ▼
                ┌─────────────────────────────────┐
                │  scope_spec["kg_cache_path"]    │
                │           ▼                     │
                │  adaptive_validity_check        │
                │           ▼                     │
                │  _compose_legacy_verdict(       │
                │    kg_edges=..., target_ids=...)│
                │           ▼                     │
                │  EnsembleVoter.vote             │
                └─────────────────────────────────┘

Generated summary report (committed to git):
  data/kg_cache/{mf_sha8}__{tc_sha8}.summary.md
  Per-feature: status + edge count + KG signal classification.
  Reviewable in PRs.
```

## Components

### 1. `FeatureContract.kg_entity_codes` field

Add to `src/data/feature_contract.py:107`:

```python
@dataclass(frozen=True)
class FeatureContract:
    ...
    kg_entity_codes: tuple[tuple[CodeSystem, str], ...] = ()
```

`CodeSystem` is the existing literal at `src/data/kg/types.py:103` (`ICD10CM` / `RXNORM` / `LOINC` / `CPT` / `HCPCS` / `SNOMEDCT_US` / `MESH`). Default `()` makes the field backward-compatible — manifests without entity codes work unchanged.

Validation in `__post_init__`: every `(system, code)` tuple must have a non-empty code; system must be in `CodeSystem`. Empty tuple is a valid declaration that "this feature has no KG-mappable entities" (e.g., demographics).

### 2. `scope_spec["target_entity_codes"]`

Runner-supplied. List of `(CodeSystem, code)` tuples. Examples:

- CSU bio_initiation target → `[("RXNORM", "479158"), ("RXNORM", "1011295"), ...]` (omalizumab + dupilumab + future biologics)
- Optum bio_initiation → same RxCUIs
- A future Dupixent-specific brand prediction → `[("RXNORM", "1011295")]` (dupilumab only)

The cohort runner (e.g., `scripts/run_data_preparer.py` for CSU) sets this in scope_spec before invoking `data_preparer`. Adding a new brand-specific target in the same disease changes only this list.

### 3. `scripts/build_kg_cache.py`

Parameterized:

```bash
python scripts/build_kg_cache.py \
    --manifest-module src.data.manifests.optum_feature_manifest \
    --target-entity-codes RXNORM:479158,RXNORM:1011295 \
    --out data/kg_cache
```

Builder responsibilities:

- **Validate** every `kg_entity_codes` tuple in the manifest via `EntityLinker.resolve_code` BEFORE querying KG. Unknown codes → hard failure with manifest line number. (Codex H2 from prior review: "If an ICD-10 typo enters the manifest, the builder may produce empty edges and the runtime will happily consume that as no-signal.")
- **Query** `KnowledgeGraphQuerier` per feature (`query_drug_disease_edges` / `query_disease_hierarchy` / `query_concept_relations`) given the feature's entity codes paired with the target's entity codes.
- **Emit** a per-feature provenance record (codex H1: distinguish "queried, returned nothing" from "never queried"):

```json
{
  "feature_name": "has_atopic_dermatitis",
  "manifest_fingerprint_sha8": "a3f9c2b1",
  "target_codes_fingerprint_sha8": "5e2d8f04",
  "queried_at": "2026-05-08T12:30:00Z",
  "feature_entity_codes": [["ICD10CM", "L20.9"], ["UMLS", "C0011615"]],
  "target_entity_codes": [["RXNORM", "479158"], ["RXNORM", "1011295"]],
  "sources_attempted": ["umls_uts", "open_targets"],
  "status": "ok",
  "edges": [...KGEdge JSON...],
  "errors": []
}
```

`status` ∈ `{"ok", "queried_no_edges", "entity_unresolved", "source_error"}`. A missing key → `cache_missing` audit event at run time, **not** the same as `queried_no_edges`.

- **Write** atomically: write to a temp file, fsync, rename. Deterministic JSON sorting (sort keys) so concurrent regenerations produce identical bytes.
- **Generate** a companion summary report (`data/kg_cache/{mf_sha8}__{tc_sha8}.summary.md`) with a per-feature table: `status | n_edges | kg_signal_class`. **Committed to git** for PR review (raw cache JSON gitignored under existing `.gitignore:86` `data/cache/` rule).

### 4. Cache reader + pipeline integration

In `_compose_legacy_verdict` (`src/agents/ml_foundation/data_preparer/nodes/adaptive_validity_check.py:449`):

- Add `kg_edges: Iterable[KGEdge] = ()` and `target_entity_ids: Iterable[str] = ()` keyword args.
- Pass through to `voter.vote(...)`.
- Caller (in the main scoring loop) reads `scope_spec["kg_cache_path"]`, loads the cache, looks up per-feature `edges` by `feature_name`, derives `feature_entity_ids` from the matching `kg_entity_codes` (cross-walked to UMLS CUIs), and passes both to `_compose_legacy_verdict`.

Cache load is one-shot at node entry, not per-feature — `O(1)` per-feature lookup against an in-memory dict.

### 5. CI freshness check

A new pytest fixture computes the manifest fingerprint at test time and asserts the cache file matching the current fingerprint exists in `data/kg_cache/`. If absent, CI fails with:

```
manifest fingerprint changed; regenerate cache via:
  python scripts/build_kg_cache.py --manifest-module src.data.manifests.optum_feature_manifest --target-entity-codes ...
```

The cache file itself is gitignored, but the `summary.md` companion is committed — reviewers see signal-class changes per feature in PR diffs.

### 6. Failure policy

| Mode | Cache missing | Cache stale (TTL) | Entity unresolved at build time |
|---|---|---|---|
| **Production / promoted** | Pipeline fails loud | Pipeline fails loud | Builder fails loud |
| **Shadow mode** | `decided_by="abstain"` audit + warning log; do not block | `decided_by="abstain"` + warning | Builder fails loud |
| **CI** | Fail before merge (fingerprint mismatch) | Fail before merge | Fail at builder step |

No mid-run TTL check — startup pin only, tolerate version drift across a long-running training job (codex M3).

### 7. Shadow-mode promotion criteria

Shadow mode emits `decided_by="kg"` audit fields with `severity="info"` only — KG verdicts never change pipeline outcomes. Exit criteria (codex M4):

- ≥95% of entity-bearing features produce non-`abstain` KG verdicts in 3 consecutive runs **AND**
- disagreement-rate (KG vs adversarial) ≤ 5%

Calendar-based exit (e.g., "1 week") is explicitly **rejected** — UMLS releases 2× yearly, Open Targets quarterly; calendar gives false confidence.

A minimum-cohort-size guard prevents promotion on small-n cohorts where 95% non-`abstain` is statistically unreliable: require N ≥ 200 patients for promotion.

## PR sequence

| PR | Scope | LOC | Depends on |
|---|---|---|---|
| **PR-0** | KG predicate reconciliation (companion spec) | ~150 | — |
| **PR-A** | `FeatureContract.kg_entity_codes` + `scope_spec["target_entity_codes"]` schema | ~150 | PR-0 |
| **PR-B** | Optum manifest population (~70 entity-bearing features) + CSU `primary_diagnosis_code` | ~300 | PR-A |
| **PR-C** | `scripts/build_kg_cache.py` + summary report generator + CI freshness fixture | ~300 | PR-B |
| **PR-D** | `_compose_legacy_verdict` integration; cache reader; shadow-mode opt-in flag | ~250 | PR-C |
| **PR-E** | Promotion gate (criteria check, opt-out flag, audit telemetry) | ~150 | PR-D |

Total: ~1100 LOC across 6 PRs (PR-0 + 5 Stage 2 PRs). Each independently mergeable in sequence.

## Cache identity scheme

**Cache key:** `{manifest_fingerprint_sha8}__{target_codes_fingerprint_sha8}.json` plus `.summary.md` companion.

- **Manifest fingerprint:** SHA-256 over a deterministic serialization of every `FeatureContract` in the manifest module — name, knowable_at, source, derivation_inputs, aggregation, window_days, kg_entity_codes. Changes when ANY contract is edited; coarse but explicit.
- **Target codes fingerprint:** SHA-256 over the sorted `scope_spec["target_entity_codes"]` tuple list. Changes when the target's RxCUIs are edited (e.g., adding a new biologic to the bio_initiation target class).
- **Cohort name NOT in path:** two cohorts with identical `(manifest, target)` legitimately share a cache. The pipeline reads only `scope_spec["kg_cache_path"]`; the cohort identity is communicated through the manifest+target contents, not a string label.

**Invalidation grain trade-off:** manifest-level fingerprint means adding ONE new feature with entity codes invalidates the entire cache. Per-feature fingerprinting was considered but rejected for v1 because (a) it complicates the freshness check, (b) Open Targets/UMLS API rate limits are not the bottleneck on a ~70-feature manifest (~10s of HTTP calls total), (c) regen frequency is low (manifest changes are typically multi-feature). Defer per-feature caching to v2 if regen becomes painful.

## Acceptance criteria

1. `FeatureContract.kg_entity_codes` field added; default `()`; backward-compatible.
2. `scope_spec["target_entity_codes"]` documented; runners populate.
3. Optum manifest populated with entity codes for ~70 entity-bearing features.
4. `scripts/build_kg_cache.py` validates codes via `EntityLinker`, queries KG, emits provenance JSON + summary md, supports `--manifest-module {path}` parameterization.
5. CI freshness check fires on manifest-fingerprint mismatch.
6. `_compose_legacy_verdict` plumbs `kg_edges` + `target_entity_ids` into `voter.vote(...)`.
7. Shadow-mode opt-in flag in `scope_spec`; promotion gate enforces 95% non-abstain + ≤5% disagreement criteria.
8. Disease-agnostic verification: a synthetic regime test demonstrates a "new cohort" can be added with only manifest authorship + scope_spec changes — zero pipeline code changes.
9. mypy clean; ruff clean; full kg + data_preparer test suites pass.

## Open questions for implementation phase

These are deferred to the writing-plans skill (which produces the per-PR implementation plans):

- Specific Optum entity-code resolution (e.g., `has_atopic_dermatitis` → which UMLS CUI? `L20.9` is one option; `L20` covers the family).
- Cache file size estimate (how big does the JSON get for a 70-feature × multiple-edges manifest?). Determines whether we need to gzip.
- Builder concurrency: do we anticipate 2+ devs regenerating simultaneously? If yes, lockfile or atomic-rename + content-hash dedup.
- Runner integration: which runners under `scripts/run_*.py` need to be updated to set `scope_spec["target_entity_codes"]` and `scope_spec["kg_cache_path"]`?

## Risk assessment

- **CI dependency on UMLS_UTS_API_KEY:** the cache builder requires it, but CI does NOT have it (per PR #85 procurement). Builder runs locally only; cache file is the artifact CI checks. Builder execution is a manual ceremony pre-merge, like the existing Feast materialization pattern.
- **Stale cache in production:** mitigated by startup pin + fail-loud policy in promoted mode; mitigated by abstain-and-warn in shadow mode.
- **First-time-cohort scenario:** CI catches missing fingerprint before merge. Doc step in cohort-onboarding playbook: "register manifest → set scope_spec → run build_kg_cache.py."
- **Entity-code drift:** EntityLinker validation at build time catches typos. UMLS / Open Targets schema drift detected by builder failures.
- **Codex critique gaps addressed:** every concern from the codex pressure-tests on the Stage 2 design is mapped to a component above (provenance records → §3 builder; source-vocab validation → §3 builder; cache identity → §"Cache identity scheme"; cache-miss policy → §6; shadow exit → §7; review summaries → §3+§5).

## Companion spec

For the predicate-mismatch fix that this design depends on, see `2026-05-08-kg-predicate-reconciliation-design.md`.
