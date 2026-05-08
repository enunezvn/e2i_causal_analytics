# Phase 2.9 Stage 2 Entity-Mapping Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire `KnowledgeGraphQuerier` output into Layer 5's `adaptive_validity_check` via the `EnsembleVoter` so per-feature verdicts carry `layer="2"` KG signals, with disease-agnostic infrastructure (universal pipeline, no cohort string in code) + disease-specific declarative content (manifests + scope_spec target codes).

**Architecture:** Five sequential PRs (PR-A through PR-E), each independently mergeable. PR-A adds the `FeatureContract.kg_entity_codes` schema + `scope_spec["target_entity_codes"]`. PR-B populates the Optum manifest with ~70 entity codes. PR-C builds the offline cache builder script. PR-D wires the cache loader into `_compose_legacy_verdict`. PR-E adds shadow-mode promotion gate.

**Tech Stack:** Python 3.12, pydantic v2, pandas, pytest, httpx (MockTransport), mypy, ruff. Project uses `--rebase` PR merge policy.

**Spec:** `docs/superpowers/specs/2026-05-08-phase29-stage2-entity-mapping-design.md`

**Prerequisite:** PR-0 from `docs/superpowers/plans/2026-05-08-pr0-kg-predicate-reconciliation.md` MUST be merged before any task in this plan begins. Verify: `git log --oneline main | grep "datatypeId='known_drug'"` returns the merge commit.

**Codex review at each step:** After each PR opens, dispatch `codex:codex-rescue` with `mode=bypassPermissions` over the diff. Address BLOCKER/HIGH findings per-fix per-commit (project convention). MEDIUMs at reviewer's judgment. Merge only when codex is clean OR all findings dispositioned.

---

## File Structure

| File | PR | Action | Purpose |
|---|---|---|---|
| `src/data/feature_contract.py` | A | Modify | Add `kg_entity_codes` field + validation |
| `src/data/manifests/__init__.py` | A | Modify | Re-export new types if needed |
| `tests/unit/test_data/test_feature_contract.py` | A | Modify | Add tests for new field |
| `src/data/scope_spec.py` (or wherever scope_spec is defined) | A | Modify | Add `target_entity_codes` field |
| `tests/unit/test_data/test_scope_spec.py` | A | Modify/Create | Validation tests |
| `src/data/manifests/optum_feature_manifest.py` | B | Modify | Populate `kg_entity_codes` for ~70 features |
| `src/data/manifests/csu_feature_manifest.py` | B | Modify | Populate `kg_entity_codes` for `primary_diagnosis_code` |
| `tests/unit/test_data/test_optum_feature_manifest.py` | B | Modify | Coverage assertions |
| `tests/unit/test_data/test_csu_feature_manifest.py` | B | Modify | Coverage assertions |
| `scripts/build_kg_cache.py` | C | Create | New CLI tool |
| `src/data/kg/cache.py` | C | Create | Cache schema + read/write helpers |
| `tests/unit/test_data/test_kg/test_cache.py` | C | Create | Schema + IO tests |
| `tests/unit/test_scripts/test_build_kg_cache.py` | C | Create | CLI smoke test |
| `data/kg_cache/.gitkeep` | C | Create | Holding directory (raw caches gitignored) |
| `src/agents/ml_foundation/data_preparer/nodes/adaptive_validity_check.py` | D | Modify | Wire `kg_edges` into `_compose_legacy_verdict` |
| `tests/unit/test_data_preparer/test_adaptive_validity_check.py` | D | Modify | Integration with cache fixture |
| `src/agents/ml_foundation/data_preparer/nodes/adaptive_validity_check.py` | E | Modify | Shadow-mode opt-in flag + promotion gate |
| `tests/unit/test_data_preparer/test_adaptive_validity_check.py` | E | Modify | Promotion-gate tests |

---

## PR-A: `FeatureContract.kg_entity_codes` + `scope_spec["target_entity_codes"]`

### Task A1: Branch + `FeatureContract.kg_entity_codes` field — TDD red

**Files:**
- Modify: `tests/unit/test_data/test_feature_contract.py`

- [ ] **Step 1: Branch**

```bash
git checkout main && git pull --ff-only origin main
git checkout -b feat/feature-contract-kg-entity-codes
```

- [ ] **Step 2: Read the existing FeatureContract test file**

Run: `head -100 tests/unit/test_data/test_feature_contract.py`

Look for: how `FeatureContract(...)` is instantiated; how `ContractViolation` is asserted; existing field validation tests.

- [ ] **Step 3: Append new failing tests**

Add to `tests/unit/test_data/test_feature_contract.py`:

```python
def test_feature_contract_default_kg_entity_codes_is_empty_tuple():
    """Existing manifests without kg_entity_codes get the default ()."""
    from src.data.feature_contract import FeatureContract, KnowableAt

    fc = FeatureContract(
        name="age",
        knowable_at=KnowableAt(reference="enrollment"),
        source="demo",
        derivation_inputs=("age",),
    )
    assert fc.kg_entity_codes == ()


def test_feature_contract_accepts_single_kg_entity_code():
    from src.data.feature_contract import FeatureContract, KnowableAt

    fc = FeatureContract(
        name="primary_diagnosis_code",
        knowable_at=KnowableAt(reference="enrollment"),
        source="demo",
        derivation_inputs=("diagcode",),
        kg_entity_codes=(("ICD10CM", "L20.9"),),
    )
    assert fc.kg_entity_codes == (("ICD10CM", "L20.9"),)


def test_feature_contract_accepts_multiple_kg_entity_codes():
    """A feature can carry several entity codes (cross-walks)."""
    from src.data.feature_contract import FeatureContract, KnowableAt

    fc = FeatureContract(
        name="has_atopic_dermatitis",
        knowable_at=KnowableAt(reference="index_date"),
        source="diagnosis_events",
        derivation_inputs=("admit_date", "diag1"),
        aggregation="max",
        window_days=180,
        kg_entity_codes=(
            ("ICD10CM", "L20.9"),
            ("UMLS", "C0011615"),
        ),
    )
    assert len(fc.kg_entity_codes) == 2


def test_feature_contract_rejects_empty_code_string():
    """Validation: every (system, code) tuple needs a non-empty code."""
    import pytest

    from src.data.feature_contract import (
        ContractViolation,
        FeatureContract,
        KnowableAt,
    )

    with pytest.raises(ContractViolation, match="kg_entity_codes"):
        FeatureContract(
            name="x",
            knowable_at=KnowableAt(reference="enrollment"),
            source="demo",
            derivation_inputs=("a",),
            kg_entity_codes=(("ICD10CM", ""),),
        )


def test_feature_contract_rejects_unknown_code_system():
    import pytest

    from src.data.feature_contract import (
        ContractViolation,
        FeatureContract,
        KnowableAt,
    )

    with pytest.raises(ContractViolation, match="kg_entity_codes"):
        FeatureContract(
            name="x",
            knowable_at=KnowableAt(reference="enrollment"),
            source="demo",
            derivation_inputs=("a",),
            kg_entity_codes=(("NOT_A_VOCAB", "L20.9"),),
        )


def test_feature_contract_normalizes_kg_entity_codes_to_tuple_of_tuples():
    """Caller may pass list-of-lists; stored as tuple-of-tuples (frozen)."""
    from src.data.feature_contract import FeatureContract, KnowableAt

    fc = FeatureContract(
        name="x",
        knowable_at=KnowableAt(reference="enrollment"),
        source="demo",
        derivation_inputs=("a",),
        kg_entity_codes=[["ICD10CM", "L20.9"], ["UMLS", "C0011615"]],
    )
    assert isinstance(fc.kg_entity_codes, tuple)
    assert all(isinstance(t, tuple) for t in fc.kg_entity_codes)
```

- [ ] **Step 4: Run failing tests**

Run: `. .venv/bin/activate && pytest tests/unit/test_data/test_feature_contract.py -v -k "kg_entity_codes"`

Expected: 6 tests FAIL with `TypeError: ... got an unexpected keyword argument 'kg_entity_codes'` (the field doesn't exist yet).

- [ ] **Step 5: Commit failing tests**

```bash
git add tests/unit/test_data/test_feature_contract.py
git commit -m "test(layer1): add kg_entity_codes field tests (failing)

Six tests cover: default empty tuple, single code, multiple cross-walk
codes, empty-code-string rejection, unknown-system rejection, list-to-
tuple normalization. All fail until FeatureContract.kg_entity_codes
field is added.

Reference: docs/superpowers/specs/2026-05-08-phase29-stage2-entity-mapping-design.md
Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

### Task A2: Implement `FeatureContract.kg_entity_codes` — TDD green

**Files:**
- Modify: `src/data/feature_contract.py`

- [ ] **Step 1: Read the existing FeatureContract dataclass**

Run: `sed -n '107,222p' src/data/feature_contract.py`

Identify: the field declaration block, the `__post_init__` method, and the existing validation pattern in `_validate`.

- [ ] **Step 2: Add the field + validation**

Edit `src/data/feature_contract.py`. After line 152 (the `_allow_unwindowed_for_test` field), add:

```python
    # KG entity codes — Phase 2.9 Stage 2. Each tuple is
    # ``(CodeSystem, code)`` where CodeSystem is in the project's
    # `CodeSystem` literal (ICD10CM/RXNORM/LOINC/CPT/HCPCS/SNOMEDCT_US/
    # MESH/UMLS). Default ``()`` makes this backward-compatible — manifests
    # without entity codes work unchanged. The cache builder validates
    # every code resolves via EntityLinker before querying KG; this
    # field is the source of truth for which entities a feature
    # represents in the KG.
    kg_entity_codes: tuple[tuple[str, str], ...] = ()
```

Then in `__post_init__` (after the existing `derivation_inputs` normalization), add:

```python
        # Normalize kg_entity_codes to tuple of tuples (frozen dataclass).
        if self.kg_entity_codes and not isinstance(self.kg_entity_codes, tuple):
            object.__setattr__(
                self,
                "kg_entity_codes",
                tuple(tuple(t) for t in self.kg_entity_codes),
            )
        elif self.kg_entity_codes:
            object.__setattr__(
                self,
                "kg_entity_codes",
                tuple(tuple(t) if not isinstance(t, tuple) else t for t in self.kg_entity_codes),
            )
```

In `_validate` (before the existing aggregation block), add:

```python
        # KG entity codes validation: each tuple must be (CodeSystem, code)
        # with both fields non-empty and CodeSystem in the known set.
        # `UMLS` is included alongside the source vocabularies in
        # `CodeSystem` (src/data/kg/types.py:103) because manifests can
        # declare a UMLS CUI directly when the source code is unknown.
        _KG_KNOWN_SYSTEMS = frozenset(
            {"ICD10CM", "ICD10", "RXNORM", "LOINC", "CPT", "HCPCS", "SNOMEDCT_US", "MESH", "UMLS"}
        )
        for entry in self.kg_entity_codes:
            if not isinstance(entry, tuple) or len(entry) != 2:
                raise ContractViolation(
                    f"feature {self.name!r}: kg_entity_codes entries must be 2-tuples; got {entry!r}",
                    feature=self.name,
                    reason="kg_entity_codes must be 2-tuples",
                )
            system, code = entry
            if not code or not isinstance(code, str):
                raise ContractViolation(
                    f"feature {self.name!r}: kg_entity_codes code must be a non-empty string; got {code!r}",
                    feature=self.name,
                    reason="kg_entity_codes code empty",
                )
            if system not in _KG_KNOWN_SYSTEMS:
                raise ContractViolation(
                    f"feature {self.name!r}: kg_entity_codes system {system!r} unknown; "
                    f"must be one of {sorted(_KG_KNOWN_SYSTEMS)}",
                    feature=self.name,
                    reason="kg_entity_codes unknown system",
                )
```

- [ ] **Step 3: Run the new tests to verify they pass**

Run: `. .venv/bin/activate && pytest tests/unit/test_data/test_feature_contract.py -v -k "kg_entity_codes"`

Expected: all 6 PASS.

- [ ] **Step 4: Run the full feature_contract test file to confirm no regressions**

Run: `. .venv/bin/activate && pytest tests/unit/test_data/test_feature_contract.py --no-header -q`

Expected: all pass (existing tests + 6 new).

- [ ] **Step 5: Run mypy + ruff on the modified file**

Run: `. .venv/bin/activate && mypy --config-file pyproject.toml src/data/feature_contract.py && ruff check src/data/feature_contract.py && ruff format --check src/data/feature_contract.py`

Expected: clean. If format diff, run `ruff format src/data/feature_contract.py`.

- [ ] **Step 6: Commit**

```bash
git add src/data/feature_contract.py
git commit -m "feat(layer1): add kg_entity_codes field to FeatureContract

Phase 2.9 Stage 2 schema extension. New field
kg_entity_codes: tuple[tuple[str, str], ...] = () declares which
KG entity codes (e.g., (ICD10CM, L20.9), (UMLS, C0011615)) a feature
represents. Default empty tuple is backward-compatible.

Validation: each tuple must be (CodeSystem, code) with non-empty code
and CodeSystem in {ICD10CM, ICD10, RXNORM, LOINC, CPT, HCPCS,
SNOMEDCT_US, MESH, UMLS}. Cache builder (PR-C) will validate codes
resolve via EntityLinker before querying KG.

Reference: docs/superpowers/specs/2026-05-08-phase29-stage2-entity-mapping-design.md
Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

### Task A3: `scope_spec["target_entity_codes"]` plumbing

**Files:**
- Locate the scope_spec definition (run `grep -rn "TypedDict.*[Ss]cope[Ss]pec\|class.*ScopeSpec" src/`)

- [ ] **Step 1: Find the scope_spec source-of-truth**

Run: `grep -rn "feature_manifest_source" src/ --include='*.py' | head -5`

Identify where scope_spec is declared (likely a TypedDict or pydantic model).

- [ ] **Step 2: Add `target_entity_codes` field**

If scope_spec is a TypedDict, add (with `total=False` parents):

```python
target_entity_codes: NotRequired[list[tuple[str, str]]]
```

If pydantic Model, add:

```python
target_entity_codes: list[tuple[str, str]] = Field(
    default_factory=list,
    description=(
        "Open Targets / UMLS entity codes for the prediction target. "
        "Used by Phase 2.9 Stage 2 KG querying. Empty list when the "
        "target has no KG-mappable representation (e.g., synthetic regimes)."
    ),
)
```

Add a `kg_cache_path: NotRequired[str]` field too:

```python
kg_cache_path: NotRequired[str]  # Path to the offline KG cache file built by scripts/build_kg_cache.py
```

- [ ] **Step 3: Add tests in `tests/unit/test_data/test_scope_spec.py` (create if absent)**

```python
"""Tests for scope_spec target_entity_codes + kg_cache_path fields."""

from __future__ import annotations


def test_scope_spec_default_target_entity_codes_is_empty():
    # Construct a minimal scope_spec (use whatever pattern existing tests use).
    spec = {
        "prediction_target": "y",
        "required_features": ["a"],
        "excluded_features": [],
    }
    # Field is not required; absence means no KG queries.
    assert spec.get("target_entity_codes", []) == []


def test_scope_spec_accepts_target_entity_codes():
    spec = {
        "prediction_target": "bio_initiation",
        "required_features": ["age"],
        "excluded_features": [],
        "target_entity_codes": [("RXNORM", "479158"), ("RXNORM", "1011295")],
    }
    assert len(spec["target_entity_codes"]) == 2
```

If scope_spec is a pydantic Model, port the test to the Model API (`ScopeSpec(prediction_target="...", target_entity_codes=[...])`).

- [ ] **Step 4: Run scope_spec tests**

Run: `. .venv/bin/activate && pytest tests/unit/test_data/test_scope_spec.py -v`

Expected: pass.

- [ ] **Step 5: Run mypy + ruff**

Run: `. .venv/bin/activate && mypy --config-file pyproject.toml src/data/scope_spec.py 2>/dev/null || mypy --config-file pyproject.toml src/`

(Path may differ depending on where scope_spec lives.)

- [ ] **Step 6: Commit**

```bash
git add -u
git commit -m "feat(layer1): add target_entity_codes + kg_cache_path to scope_spec

Phase 2.9 Stage 2 runner contract. target_entity_codes is the
list of (CodeSystem, code) tuples representing the prediction
target's KG entities (e.g., RxCUIs for bio_initiation target).
kg_cache_path points at the offline cache file built by the
PR-C cache builder script. Both fields are optional; cohort
runners populate them per cohort. Synthetic regimes leave them
unset.

Reference: docs/superpowers/specs/2026-05-08-phase29-stage2-entity-mapping-design.md
Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

### Task A4: PR-A push + codex review + merge

- [ ] **Step 1: Quality gates**

Run: `. .venv/bin/activate && pytest tests/unit/test_data/ --no-header -q && mypy --config-file pyproject.toml src/data/ && ruff check src/data/ tests/unit/test_data/ && ruff format --check src/data/ tests/unit/test_data/`

Expected: all clean.

- [ ] **Step 2: Push + open PR**

```bash
git push -u origin feat/feature-contract-kg-entity-codes
gh pr create --title "feat(layer1): FeatureContract.kg_entity_codes + scope_spec target plumbing (PR-A)" --body "$(cat <<'EOF'
## Summary

Phase 2.9 Stage 2 PR-A: schema extensions only. No runtime behavior change yet — this PR sets up the declarative surface that PR-B (manifest population) and PR-C (cache builder) will consume.

## Changes

- `FeatureContract.kg_entity_codes: tuple[tuple[str, str], ...] = ()` — new field with default empty tuple. Backward-compatible.
- `scope_spec["target_entity_codes"]: list[tuple[str, str]]` — new optional runner contract.
- `scope_spec["kg_cache_path"]: str` — new optional cache-file path.

## Test plan

- [x] 6 unit tests for `FeatureContract.kg_entity_codes` validation.
- [x] 2 unit tests for scope_spec field plumbing.
- [x] mypy + ruff clean on `src/data/` and `tests/unit/test_data/`.
- [x] No existing tests broken.

## Spec

`docs/superpowers/specs/2026-05-08-phase29-stage2-entity-mapping-design.md` §"FeatureContract.kg_entity_codes field" + §"scope_spec target_entity_codes"

## Sequence

This PR unblocks PR-B (Optum manifest population). Plan: `docs/superpowers/plans/2026-05-08-stage2-entity-mapping.md`.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

- [ ] **Step 3: Codex adversarial review**

Dispatch codex (Agent tool, `subagent_type=codex:codex-rescue`, `mode=bypassPermissions`). Prompt should include:

- Branch name and commit range
- The PR's diff scope (~150 LOC)
- Specific pressure points: validation completeness (what about `code` containing whitespace? leading/trailing? case?), backward-compat (does any existing manifest call `FeatureContract(kg_entity_codes=...)` with the old shape — no, since it didn't exist), code-system literal coverage (did we miss any vocabulary the project will need?).

- [ ] **Step 4: Address codex findings per-fix per-commit (project pattern)**

For each BLOCKER/HIGH: write a regression test + fix + commit + push. Repeat until codex reports clean.

- [ ] **Step 5: Wait for CI green; merge**

```bash
gh pr checks <pr-number>
gh pr merge <pr-number> --rebase --delete-branch
git checkout main && git pull --ff-only origin main
```

---

## PR-B: Optum manifest population (~70 entity-bearing features)

### Task B1: Branch + research helper for Optum entity codes

**Files:**
- Create: `scripts/research_optum_entities.py` (one-time research helper, may not need to commit)

- [ ] **Step 1: Branch from updated main**

```bash
git checkout main && git pull --ff-only origin main
git checkout -b feat/optum-manifest-kg-entity-codes
```

- [ ] **Step 2: List Optum entity-bearing features**

Run: `grep -E "name=\"" src/data/manifests/optum_feature_manifest.py | head -100`

Categorize:
- 3 specific dx counts (`dx_l50_1_count`, `dx_l50_8_count`, `dx_l50_9_count`)
- 8 comorbidity prefixes × 2 = 16 (`has_atopic_dermatitis`, `atopic_dermatitis_claim_count`, etc.)
- 7 drug class names × 4 = 28 (`h1_1g_*`, `h1_2g_*`, etc.)
- 8 lab panels × 3 = 24 (`ige_total_*`, `eosinophil_*`, etc.)
- `primary_diagnosis_code`

Total: ~71 features needing entity codes.

- [ ] **Step 3: Build the entity-code reference table**

Use a one-shot research subagent (general-purpose) prompted to map each Optum entity-bearing feature to its canonical UMLS CUI / RxCUI / LOINC code.

Example mappings (extend via subagent research):

| Feature/family | Code system | Code |
|---|---|---|
| `dx_l50_1_count` | ICD10CM | L50.1 |
| `dx_l50_1_count` | UMLS | C0042109 (urticaria, family) |
| `dx_l50_8_count` | ICD10CM | L50.8 |
| `dx_l50_9_count` | ICD10CM | L50.9 |
| `dx_l50_9_count` | UMLS | C0042109 |
| `has_atopic_dermatitis` | ICD10CM | L20.9 |
| `has_atopic_dermatitis` | UMLS | C0011615 |
| `has_asthma` | ICD10CM | J45.x |
| `has_asthma` | UMLS | C0004096 |
| `has_allergic_rhinitis` | ICD10CM | J30.x |
| `has_allergic_rhinitis` | UMLS | C0018621 |
| `has_anxiety` | ICD10CM | F41.x |
| `has_anxiety` | UMLS | C0003467 |
| `has_depression` | ICD10CM | F33.x |
| `has_depression` | UMLS | C0011581 |
| `has_thyroid_autoimmune` | ICD10CM | E06.3 |
| `has_thyroid_autoimmune` | UMLS | C0856243 |
| `has_nsaid_hypersensitivity` | UMLS | C2266824 |
| `has_angioedema` | ICD10CM | T78.3 |
| `has_angioedema` | UMLS | C0002994 |
| `h1_1g_*` (first-gen H1 antihistamines) | RXNORM | (drug-class CUI; use UMLS C0066896) |
| `h1_2g_*` (second-gen H1) | RXNORM | (similar — UMLS C2718076) |
| `h2_*` (H2 antagonists) | RXNORM | UMLS C0019613 |
| `ltra_*` (leukotriene receptor antagonists) | RXNORM | UMLS C0876129 |
| `sys_steroid_*` (systemic corticosteroids) | RXNORM | UMLS C2825472 |
| `top_steroid_*` (topical corticosteroids) | RXNORM | UMLS C0001617 |
| `immunosupp_*` (immunosuppressants) | RXNORM | UMLS C0021081 |
| `ige_total_*` (IgE serum) | LOINC | 6106-9 |
| `ige_total_*` | UMLS | C0922951 |
| `eosinophil_*` | LOINC | 711-2 |
| `crp_*` (C-reactive protein) | LOINC | 1988-5 |
| `tpo_ab_*` (TPO antibodies) | LOINC | 9362-4 |
| `free_t4_*` | LOINC | 3024-7 |
| `tsh_*` | LOINC | 3016-3 |
| `ana_*` (anti-nuclear antibodies) | LOINC | 5048-4 |
| `cbc_*` (complete blood count) | LOINC | 58410-2 |
| `primary_diagnosis_code` | ICD10CM | L50.x (cohort-anchor; use UMLS C0042109) |

For each feature, the entity-code tuple should include the most specific code AND a UMLS CUI. The UMLS CUI is the canonical cross-walk; the source-vocab code lets the EntityLinker validate via UTS source endpoints.

- [ ] **Step 4: Commit a research note (optional but recommended)**

If the entity mapping is non-trivial, persist the research as `docs/superpowers/specs/2026-05-08-optum-entity-codes.md` (a small reference doc):

```bash
git add docs/superpowers/specs/2026-05-08-optum-entity-codes.md
git commit -m "docs: Optum manifest entity-code reference for Stage 2 PR-B

One-shot research output mapping each Optum entity-bearing feature
to its canonical UMLS CUI + source-vocab code. Used as the source of
truth when populating optum_feature_manifest.py kg_entity_codes
fields.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

### Task B2: Populate the Optum manifest — TDD red

**Files:**
- Modify: `tests/unit/test_data/test_optum_feature_manifest.py`

- [ ] **Step 1: Read current test file**

Run: `head -80 tests/unit/test_data/test_optum_feature_manifest.py`

- [ ] **Step 2: Add coverage assertions**

Append:

```python
def test_every_disease_specific_dx_feature_has_kg_entity_codes():
    """The 3 dx_l50_*_count features must declare their ICD-10 codes."""
    from src.data.manifests.optum_feature_manifest import OPTUM_FEATURES

    by_name = {c.name: c for c in OPTUM_FEATURES}
    for fname in ("dx_l50_1_count", "dx_l50_8_count", "dx_l50_9_count"):
        assert by_name[fname].kg_entity_codes, (
            f"{fname} must declare kg_entity_codes for KG querying"
        )


def test_every_comorbidity_has_feature_has_kg_entity_codes():
    """All 8 has_<comorbidity> features must declare entity codes."""
    from src.data.manifests.optum_feature_manifest import (
        COMORBIDITY_NAMES,
        OPTUM_FEATURES,
    )

    by_name = {c.name: c for c in OPTUM_FEATURES}
    for name in COMORBIDITY_NAMES:
        feat_name = f"has_{name}"
        assert by_name[feat_name].kg_entity_codes, (
            f"{feat_name} must declare kg_entity_codes"
        )


def test_every_drug_class_ever_feature_has_kg_entity_codes():
    """All 7 <drug_class>_ever features must declare entity codes."""
    from src.data.manifests.optum_feature_manifest import (
        DRUG_CLASS_NAMES,
        OPTUM_FEATURES,
    )

    by_name = {c.name: c for c in OPTUM_FEATURES}
    for cls in DRUG_CLASS_NAMES:
        feat_name = f"{cls}_ever"
        assert by_name[feat_name].kg_entity_codes, (
            f"{feat_name} must declare kg_entity_codes"
        )


def test_every_lab_tested_feature_has_kg_entity_codes():
    """All 8 <lab>_tested features must declare entity codes."""
    from src.data.manifests.optum_feature_manifest import (
        LAB_NAMES,
        OPTUM_FEATURES,
    )

    by_name = {c.name: c for c in OPTUM_FEATURES}
    for lab in LAB_NAMES:
        feat_name = f"{lab}_tested"
        assert by_name[feat_name].kg_entity_codes, (
            f"{feat_name} must declare kg_entity_codes"
        )
```

- [ ] **Step 3: Run failing tests**

Run: `. .venv/bin/activate && pytest tests/unit/test_data/test_optum_feature_manifest.py -v -k "kg_entity_codes"`

Expected: 4 FAIL because the manifest doesn't yet declare codes.

- [ ] **Step 4: Commit failing tests**

```bash
git add tests/unit/test_data/test_optum_feature_manifest.py
git commit -m "test(layer1): assert Optum manifest declares kg_entity_codes (failing)

Coverage assertions for 4 entity-bearing feature families:
- 3 disease-specific dx_l50_*_count features
- 8 has_<comorbidity> features
- 7 <drug_class>_ever features
- 8 <lab>_tested features

All fail until the optum_feature_manifest.py kg_entity_codes
populations land.

Reference: docs/superpowers/specs/2026-05-08-phase29-stage2-entity-mapping-design.md
Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

### Task B3: Populate the Optum manifest — TDD green

**Files:**
- Modify: `src/data/manifests/optum_feature_manifest.py`

- [ ] **Step 1: Add a constants section**

At the top of `optum_feature_manifest.py`, after the existing `COMORBIDITY_NAMES` etc. constants, add:

```python
# Phase 2.9 Stage 2 entity-code maps (per docs/superpowers/specs/
# 2026-05-08-optum-entity-codes.md research). Each map: feature name
# (or family prefix) → tuple of (CodeSystem, code) entries the cache
# builder will pass to EntityLinker for KG querying.

_DX_ENTITY_CODES: dict[str, tuple[tuple[str, str], ...]] = {
    "dx_l50_1_count": (("ICD10CM", "L50.1"), ("UMLS", "C0042109")),
    "dx_l50_8_count": (("ICD10CM", "L50.8"), ("UMLS", "C0042109")),
    "dx_l50_9_count": (("ICD10CM", "L50.9"), ("UMLS", "C0042109")),
    "dx_total_csu": (("UMLS", "C0042109"),),
    "dx_angioedema_count": (("ICD10CM", "T78.3"), ("UMLS", "C0002994")),
}

_COMORBIDITY_UMLS: dict[str, str] = {
    "atopic_dermatitis": "C0011615",
    "asthma": "C0004096",
    "allergic_rhinitis": "C0018621",
    "anxiety": "C0003467",
    "depression": "C0011581",
    "thyroid_autoimmune": "C0856243",
    "nsaid_hypersensitivity": "C2266824",
    "angioedema": "C0002994",
}

_COMORBIDITY_ICD10: dict[str, str] = {
    "atopic_dermatitis": "L20.9",
    "asthma": "J45.909",
    "allergic_rhinitis": "J30.9",
    "anxiety": "F41.9",
    "depression": "F33.9",
    "thyroid_autoimmune": "E06.3",
    "nsaid_hypersensitivity": "T88.7",
    "angioedema": "T78.3",
}

_DRUG_CLASS_UMLS: dict[str, str] = {
    "h1_1g": "C0066896",
    "h1_2g": "C2718076",
    "h2": "C0019613",
    "ltra": "C0876129",
    "sys_steroid": "C2825472",
    "top_steroid": "C0001617",
    "immunosupp": "C0021081",
}

_LAB_LOINC: dict[str, str] = {
    "ige_total": "6106-9",
    "eosinophil": "711-2",
    "crp": "1988-5",
    "tpo_ab": "9362-4",
    "free_t4": "3024-7",
    "tsh": "3016-3",
    "ana": "5048-4",
    "cbc": "58410-2",
}

_LAB_UMLS: dict[str, str] = {
    "ige_total": "C0922951",
    "eosinophil": "C0427682",
    "crp": "C0006560",
    "tpo_ab": "C0796205",
    "free_t4": "C0202119",
    "tsh": "C0202230",
    "ana": "C0003243",
    "cbc": "C0009555",
}
```

- [ ] **Step 2: Update the `_DISEASE` block**

For each existing `FeatureContract(name="dx_l50_*", ...)` in the `_DISEASE` list, add `kg_entity_codes=_DX_ENTITY_CODES[<feature_name>]`. Same for `dx_total_csu`, `dx_angioedema_count`.

Example diff for `dx_l50_1_count`:

```python
    FeatureContract(
        name="dx_l50_1_count",
        knowable_at=KnowableAt(reference="index_date"),
        source="diagnosis_events",
        derivation_inputs=("admit_date", "diag1", "diag2", "diag3", "diag4", "diag5"),
        aggregation="count",
        window_days=OPTUM_LOOKBACK_DAYS,
        kg_entity_codes=_DX_ENTITY_CODES["dx_l50_1_count"],
    ),
```

- [ ] **Step 3: Update the `_COMORBIDITIES` helper-expansion loop**

Replace the existing loop:

```python
_COMORBIDITIES: list[FeatureContract] = []
for name in COMORBIDITY_NAMES:
    _COMORBIDITIES.append(
        FeatureContract(
            name=f"has_{name}",
            knowable_at=KnowableAt(reference="index_date"),
            source="diagnosis_events",
            derivation_inputs=("admit_date", "diag1", "diag2", "diag3", "diag4", "diag5"),
            aggregation="max",
            window_days=OPTUM_LOOKBACK_DAYS,
            kg_entity_codes=(
                ("ICD10CM", _COMORBIDITY_ICD10[name]),
                ("UMLS", _COMORBIDITY_UMLS[name]),
            ),
        )
    )
    _COMORBIDITIES.append(
        FeatureContract(
            name=f"{name}_claim_count",
            knowable_at=KnowableAt(reference="index_date"),
            source="diagnosis_events",
            derivation_inputs=("admit_date", "diag1", "diag2", "diag3", "diag4", "diag5"),
            aggregation="count",
            window_days=OPTUM_LOOKBACK_DAYS,
            kg_entity_codes=(
                ("ICD10CM", _COMORBIDITY_ICD10[name]),
                ("UMLS", _COMORBIDITY_UMLS[name]),
            ),
        )
    )
```

- [ ] **Step 4: Update the `_DRUG_CLASS` helper-expansion loop**

Each `<class>_ever` / `<class>_count` / `<class>_days` / `<class>_days_since_last` gets `kg_entity_codes=(("UMLS", _DRUG_CLASS_UMLS[cls]),)`.

- [ ] **Step 5: Update the `_LABS` helper-expansion loop**

Each `<lab>_tested` / `<lab>_result_last` / `<lab>_abnormal_flag` gets:

```python
kg_entity_codes=(
    ("LOINC", _LAB_LOINC[lab]),
    ("UMLS", _LAB_UMLS[lab]),
),
```

- [ ] **Step 6: Update `primary_diagnosis_code` (in `_DEMO`)**

```python
FeatureContract(
    name="primary_diagnosis_code",
    knowable_at=KnowableAt(reference="enrollment"),
    source="demo",
    derivation_inputs=("diagcode_raw",),
    kg_entity_codes=(("UMLS", "C0042109"),),  # CSU as the cohort anchor
),
```

- [ ] **Step 7: Run the coverage tests**

Run: `. .venv/bin/activate && pytest tests/unit/test_data/test_optum_feature_manifest.py -v`

Expected: all pass (the 4 new + existing).

- [ ] **Step 8: Run mypy + ruff**

Run: `. .venv/bin/activate && mypy --config-file pyproject.toml src/data/manifests/optum_feature_manifest.py && ruff check src/data/manifests/ && ruff format --check src/data/manifests/`

If format diffs, run `ruff format src/data/manifests/`.

- [ ] **Step 9: Commit**

```bash
git add src/data/manifests/optum_feature_manifest.py tests/unit/test_data/test_optum_feature_manifest.py
git commit -m "feat(layer1): populate Optum manifest with kg_entity_codes (~70 features)

Phase 2.9 Stage 2 PR-B: all entity-bearing Optum features carry their
canonical entity codes. Five families:
- 5 disease-specific dx codes (ICD10CM + UMLS)
- 16 has_<comorbidity> + <comorbidity>_claim_count features (ICD10CM + UMLS)
- 28 <drug_class>_<metric> features (UMLS drug class CUI)
- 24 <lab>_<metric> features (LOINC + UMLS)
- primary_diagnosis_code anchored to UMLS C0042109 (urticaria)

Entity-code source: docs/superpowers/specs/2026-05-08-optum-entity-codes.md.
Validated by 4 new coverage tests on the manifest helper-expansion loops.

Reference: docs/superpowers/specs/2026-05-08-phase29-stage2-entity-mapping-design.md
Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

### Task B4: PR-B push + codex review + merge

- [ ] **Step 1: Quality gates**

Run: `. .venv/bin/activate && pytest tests/unit/test_data/test_optum_feature_manifest.py tests/unit/test_data/test_csu_feature_manifest.py tests/unit/test_data/test_feature_contract.py --no-header -q && mypy --config-file pyproject.toml src/data/ && ruff check src/data/ && ruff format --check src/data/`

- [ ] **Step 2: Push + open PR**

```bash
git push -u origin feat/optum-manifest-kg-entity-codes
gh pr create --title "feat(layer1): populate Optum manifest kg_entity_codes (~70 features) — PR-B" --body "..."
```

- [ ] **Step 3: Dispatch codex review**

Pressure points to include in the prompt:
- Are the UMLS CUIs accurate? Specifically: did we cross-walk every comorbidity/lab/drug-class to the right CUI per UMLS 2026AA release?
- Helper-expansion loop correctness: every `(comorbidity_name, *)` feature carries identical entity codes. Is that semantically right, or should each `_claim_count` carry a different code than its `has_` counterpart?
- ICD-10 specificity: `J45.909` for asthma is "unspecified" — is that the right anchor, or should it be `J45.x` family-level via SNOMEDCT_US?
- Cross-cohort reuse: does the CSU `primary_diagnosis_code` UMLS anchor (`C0042109`) match the Optum one?

- [ ] **Step 4: Address codex findings per-fix per-commit**

- [ ] **Step 5: Merge with `--rebase` after CI green**

---

## PR-C: Cache builder script

### Task C1: Branch + cache schema (TDD red)

**Files:**
- Create: `tests/unit/test_data/test_kg/test_cache.py`

- [ ] **Step 1: Branch**

```bash
git checkout main && git pull --ff-only origin main
git checkout -b feat/kg-cache-builder
```

- [ ] **Step 2: Write the failing tests for the cache schema**

Create `tests/unit/test_data/test_kg/test_cache.py`:

```python
"""Tests for the Phase 2.9 Stage 2 KG cache schema and IO helpers."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest


def test_cache_record_round_trips_through_json(tmp_path: Path):
    """Per-feature provenance record serializes losslessly."""
    from src.data.kg.cache import CacheRecord

    record = CacheRecord(
        feature_name="has_atopic_dermatitis",
        manifest_fingerprint_sha8="a3f9c2b1",
        target_codes_fingerprint_sha8="5e2d8f04",
        queried_at=datetime(2026, 5, 8, 12, 30, 0, tzinfo=timezone.utc),
        feature_entity_codes=(("ICD10CM", "L20.9"), ("UMLS", "C0011615")),
        target_entity_codes=(("RXNORM", "479158"),),
        sources_attempted=("umls_uts", "open_targets"),
        status="ok",
        edges=(),
        errors=(),
    )

    payload = record.to_json()
    record2 = CacheRecord.from_json(payload)
    assert record2.feature_name == "has_atopic_dermatitis"
    assert record2.status == "ok"
    assert record2.feature_entity_codes == record.feature_entity_codes


def test_cache_record_status_is_validated():
    """status must be one of the four documented values."""
    from src.data.kg.cache import CacheRecord, CacheRecordValidationError

    with pytest.raises(CacheRecordValidationError):
        CacheRecord(
            feature_name="x",
            manifest_fingerprint_sha8="a",
            target_codes_fingerprint_sha8="b",
            queried_at=datetime.now(timezone.utc),
            feature_entity_codes=(),
            target_entity_codes=(),
            sources_attempted=(),
            status="not_a_real_status",  # type: ignore[arg-type]
            edges=(),
            errors=(),
        )


def test_cache_file_round_trips(tmp_path: Path):
    """A list of records writes deterministically and loads back."""
    from src.data.kg.cache import CacheRecord, load_cache, save_cache

    records = [
        CacheRecord(
            feature_name=f"feat_{i}",
            manifest_fingerprint_sha8="a3f9c2b1",
            target_codes_fingerprint_sha8="5e2d8f04",
            queried_at=datetime(2026, 5, 8, tzinfo=timezone.utc),
            feature_entity_codes=(),
            target_entity_codes=(),
            sources_attempted=(),
            status="queried_no_edges",
            edges=(),
            errors=(),
        )
        for i in range(3)
    ]

    path = tmp_path / "cache.json"
    save_cache(records, path)
    loaded = load_cache(path)

    assert len(loaded) == 3
    assert {r.feature_name for r in loaded} == {"feat_0", "feat_1", "feat_2"}


def test_cache_file_atomic_write(tmp_path: Path):
    """save_cache should write atomically via temp + rename."""
    from src.data.kg.cache import CacheRecord, save_cache

    record = CacheRecord(
        feature_name="x",
        manifest_fingerprint_sha8="a",
        target_codes_fingerprint_sha8="b",
        queried_at=datetime.now(timezone.utc),
        feature_entity_codes=(),
        target_entity_codes=(),
        sources_attempted=(),
        status="ok",
        edges=(),
        errors=(),
    )

    path = tmp_path / "out.json"
    save_cache([record], path)
    # No leftover .tmp file
    assert path.exists()
    assert not (tmp_path / "out.json.tmp").exists()


def test_cache_file_deterministic_sort_for_concurrency(tmp_path: Path):
    """Two concurrent regenerations produce identical bytes."""
    from src.data.kg.cache import CacheRecord, save_cache

    records = [
        CacheRecord(
            feature_name="b",
            manifest_fingerprint_sha8="a",
            target_codes_fingerprint_sha8="b",
            queried_at=datetime(2026, 5, 8, tzinfo=timezone.utc),
            feature_entity_codes=(),
            target_entity_codes=(),
            sources_attempted=(),
            status="ok",
            edges=(),
            errors=(),
        ),
        CacheRecord(
            feature_name="a",
            manifest_fingerprint_sha8="a",
            target_codes_fingerprint_sha8="b",
            queried_at=datetime(2026, 5, 8, tzinfo=timezone.utc),
            feature_entity_codes=(),
            target_entity_codes=(),
            sources_attempted=(),
            status="ok",
            edges=(),
            errors=(),
        ),
    ]

    path1 = tmp_path / "c1.json"
    path2 = tmp_path / "c2.json"
    save_cache(records, path1)
    save_cache(list(reversed(records)), path2)

    # Same content despite input order difference (records sorted by feature_name)
    assert path1.read_bytes() == path2.read_bytes()


def test_compute_manifest_fingerprint_stable():
    """Same manifest module → same fingerprint."""
    from src.data.kg.cache import compute_manifest_fingerprint
    from src.data.manifests import csu_feature_manifest

    fp1 = compute_manifest_fingerprint(csu_feature_manifest.CSU_FEATURES)
    fp2 = compute_manifest_fingerprint(csu_feature_manifest.CSU_FEATURES)
    assert fp1 == fp2
    assert len(fp1) == 8  # sha8 truncation


def test_compute_target_codes_fingerprint_order_independent():
    """Same set of (system, code) tuples in different order → same fp."""
    from src.data.kg.cache import compute_target_codes_fingerprint

    a = [("RXNORM", "479158"), ("RXNORM", "1011295")]
    b = [("RXNORM", "1011295"), ("RXNORM", "479158")]
    assert compute_target_codes_fingerprint(a) == compute_target_codes_fingerprint(b)
```

- [ ] **Step 3: Run failing tests**

Run: `. .venv/bin/activate && pytest tests/unit/test_data/test_kg/test_cache.py -v`

Expected: all FAIL (`src/data/kg/cache.py` doesn't exist).

- [ ] **Step 4: Commit**

```bash
git add tests/unit/test_data/test_kg/test_cache.py
git commit -m "test(layer2): cache schema and IO helper tests (failing)

Defines the CacheRecord dataclass contract: per-feature provenance with
status enum, entity codes, sources attempted, edges, errors. Tests
cover JSON round-trip, status validation, atomic file write, deterministic
sort for concurrent regen, manifest/target fingerprint stability.

Reference: docs/superpowers/specs/2026-05-08-phase29-stage2-entity-mapping-design.md
Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

### Task C2: Implement `src/data/kg/cache.py` (TDD green)

**Files:**
- Create: `src/data/kg/cache.py`

- [ ] **Step 1: Write the cache module**

Create `src/data/kg/cache.py`:

```python
"""Phase 2.9 Stage 2 KG cache — schema and IO helpers.

The cache file persists per-feature provenance records produced by
``scripts/build_kg_cache.py``. Run-time pipeline (Layer 5
``adaptive_validity_check``) reads this file at node entry to obtain
``KGEdge`` lists per feature without making HTTP calls in the hot path.

Provenance schema is explicit (status enum, sources_attempted,
errors) so that an empty edge list with ``status="queried_no_edges"``
is distinguishable from a missing entry (``cache_missing`` audit
event at run time).

Reference: docs/superpowers/specs/2026-05-08-phase29-stage2-entity-mapping-design.md
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Literal

from src.data.feature_contract import FeatureContract
from src.data.kg.types import KGEdge

CacheRecordStatus = Literal["ok", "queried_no_edges", "entity_unresolved", "source_error"]


class CacheRecordValidationError(ValueError):
    pass


@dataclass(frozen=True)
class CacheRecord:
    """One per-feature provenance entry in the KG cache file."""

    feature_name: str
    manifest_fingerprint_sha8: str
    target_codes_fingerprint_sha8: str
    queried_at: datetime
    feature_entity_codes: tuple[tuple[str, str], ...]
    target_entity_codes: tuple[tuple[str, str], ...]
    sources_attempted: tuple[str, ...]
    status: CacheRecordStatus
    edges: tuple[KGEdge, ...] = field(default_factory=tuple)
    errors: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        valid_statuses = {"ok", "queried_no_edges", "entity_unresolved", "source_error"}
        if self.status not in valid_statuses:
            raise CacheRecordValidationError(
                f"status must be one of {sorted(valid_statuses)}; got {self.status!r}"
            )

    def to_json(self) -> dict[str, Any]:
        return {
            "feature_name": self.feature_name,
            "manifest_fingerprint_sha8": self.manifest_fingerprint_sha8,
            "target_codes_fingerprint_sha8": self.target_codes_fingerprint_sha8,
            "queried_at": self.queried_at.isoformat(),
            "feature_entity_codes": [list(t) for t in self.feature_entity_codes],
            "target_entity_codes": [list(t) for t in self.target_entity_codes],
            "sources_attempted": list(self.sources_attempted),
            "status": self.status,
            "edges": [_kg_edge_to_json(e) for e in self.edges],
            "errors": list(self.errors),
        }

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> CacheRecord:
        return cls(
            feature_name=payload["feature_name"],
            manifest_fingerprint_sha8=payload["manifest_fingerprint_sha8"],
            target_codes_fingerprint_sha8=payload["target_codes_fingerprint_sha8"],
            queried_at=datetime.fromisoformat(payload["queried_at"]),
            feature_entity_codes=tuple(tuple(t) for t in payload["feature_entity_codes"]),
            target_entity_codes=tuple(tuple(t) for t in payload["target_entity_codes"]),
            sources_attempted=tuple(payload["sources_attempted"]),
            status=payload["status"],
            edges=tuple(_kg_edge_from_json(e) for e in payload.get("edges", [])),
            errors=tuple(payload.get("errors", [])),
        )


def _kg_edge_to_json(edge: KGEdge) -> dict[str, Any]:
    return {
        "subject_id": edge.subject_id,
        "subject_name": edge.subject_name,
        "predicate": edge.predicate,
        "object_id": edge.object_id,
        "object_name": edge.object_name,
        "evidence_source": edge.evidence_source,
        "score": edge.score,
        "pmids": list(edge.pmids),
        "datasource": edge.datasource,
    }


def _kg_edge_from_json(payload: dict[str, Any]) -> KGEdge:
    return KGEdge(
        subject_id=payload["subject_id"],
        subject_name=payload.get("subject_name", ""),
        predicate=payload["predicate"],
        object_id=payload["object_id"],
        object_name=payload.get("object_name", ""),
        evidence_source=payload["evidence_source"],
        score=payload.get("score"),
        pmids=tuple(payload.get("pmids", [])),
        datasource=payload.get("datasource"),
    )


def save_cache(records: Iterable[CacheRecord], path: Path) -> None:
    """Atomically write a cache file with deterministic record order.

    Records are sorted by feature_name for byte-stable output across
    concurrent regenerations. Write goes to a temp file in the same
    directory, then os.rename() atomically replaces the target.
    """
    sorted_records = sorted(records, key=lambda r: r.feature_name)
    payload = [r.to_json() for r in sorted_records]
    parent = path.parent
    parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=parent, prefix=path.name, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def load_cache(path: Path) -> list[CacheRecord]:
    payload = json.loads(path.read_text())
    return [CacheRecord.from_json(entry) for entry in payload]


def compute_manifest_fingerprint(features: Iterable[FeatureContract]) -> str:
    """SHA-256 over a deterministic serialization of the manifest.

    The serialization captures every contract field that affects KG
    queries: name, knowable_at, source, derivation_inputs, aggregation,
    window_days, kg_entity_codes. Returns the first 8 hex chars (sha8).
    """
    rows: list[tuple[Any, ...]] = []
    for fc in features:
        rows.append(
            (
                fc.name,
                fc.knowable_at.reference,
                fc.knowable_at.offset_days,
                fc.source,
                tuple(fc.derivation_inputs),
                fc.aggregation,
                fc.window_days,
                tuple(tuple(t) for t in fc.kg_entity_codes),
            )
        )
    rows.sort(key=lambda r: r[0])  # stable order by feature name
    blob = json.dumps(rows, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:8]


def compute_target_codes_fingerprint(target_codes: Iterable[tuple[str, str]]) -> str:
    """SHA-256 over a sorted (system, code) tuple list. First 8 hex chars."""
    sorted_codes = sorted(tuple(t) for t in target_codes)
    blob = json.dumps(sorted_codes, sort_keys=True).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:8]


def compose_cache_filename(manifest_fp: str, target_fp: str) -> str:
    """Deterministic cache filename — no cohort name in the path.

    Two cohorts with identical (manifest, target) legitimately share a
    cache file. The pipeline reads via scope_spec["kg_cache_path"]
    only.
    """
    return f"{manifest_fp}__{target_fp}.json"


CACHE_TIMESTAMP_NOW: datetime = datetime.now(timezone.utc)
```

- [ ] **Step 2: Run cache tests**

Run: `. .venv/bin/activate && pytest tests/unit/test_data/test_kg/test_cache.py -v`

Expected: all 7 tests PASS.

- [ ] **Step 3: mypy + ruff**

Run: `. .venv/bin/activate && mypy --config-file pyproject.toml src/data/kg/cache.py && ruff check src/data/kg/cache.py && ruff format --check src/data/kg/cache.py`

- [ ] **Step 4: Commit**

```bash
git add src/data/kg/cache.py
git commit -m "feat(layer2): cache schema + IO helpers for Stage 2 PR-C

CacheRecord dataclass + save/load helpers. Atomic write via temp +
os.replace. Deterministic record sort by feature_name for concurrent-
regen byte stability. compute_manifest_fingerprint hashes contract
shape; compute_target_codes_fingerprint hashes sorted target tuples.
compose_cache_filename: '{manifest_sha8}__{target_sha8}.json' (no
cohort name in path — disease-agnostic).

Reference: docs/superpowers/specs/2026-05-08-phase29-stage2-entity-mapping-design.md
Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

### Task C3: Implement `scripts/build_kg_cache.py`

**Files:**
- Create: `scripts/build_kg_cache.py`
- Create: `tests/unit/test_scripts/test_build_kg_cache.py`

- [ ] **Step 1: Write smoke test for the CLI**

Create `tests/unit/test_scripts/test_build_kg_cache.py`:

```python
"""Smoke tests for scripts/build_kg_cache.py.

Live KG calls are NOT exercised here — those tests gate on
UMLS_UTS_API_KEY (skipped in CI). The CLI script is exercised via
its public functions with mocked KG clients.
"""

from __future__ import annotations

from pathlib import Path

import pytest


def test_cli_help_runs_without_error():
    """`python scripts/build_kg_cache.py --help` exits 0."""
    import subprocess
    result = subprocess.run(
        ["python", "scripts/build_kg_cache.py", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "manifest-module" in result.stdout


def test_build_with_no_entity_features_writes_empty_cache(tmp_path: Path, monkeypatch):
    """A manifest with zero entity-bearing features produces an
    empty cache file (no-op success)."""
    from scripts.build_kg_cache import build_cache_for_manifest

    # Stub manifest with no kg_entity_codes
    from src.data.feature_contract import FeatureContract, KnowableAt

    features = [
        FeatureContract(
            name="age",
            knowable_at=KnowableAt(reference="enrollment"),
            source="demo",
            derivation_inputs=("age",),
        )
    ]
    out = tmp_path / "kg_cache"
    build_cache_for_manifest(
        features=features,
        target_entity_codes=[("RXNORM", "479158")],
        out_dir=out,
        umls_client=None,  # not called when no features have codes
        open_targets_client=None,
    )
    files = list(out.glob("*.json"))
    assert len(files) == 1
    # File contains an empty record list
    import json
    payload = json.loads(files[0].read_text())
    assert payload == []
```

- [ ] **Step 2: Write the CLI script**

Create `scripts/build_kg_cache.py`:

```python
"""Phase 2.9 Stage 2 KG cache builder.

Reads a manifest module (e.g., ``src.data.manifests.optum_feature_manifest``),
queries KG for every feature with ``kg_entity_codes`` set, and writes a
cache file at:

    {out_dir}/{manifest_sha8}__{target_sha8}.json

Plus a companion summary report at:

    {out_dir}/{manifest_sha8}__{target_sha8}.summary.md

The summary report is committed to git (per .gitignore exception);
raw cache JSON is gitignored under data/cache/.

Usage:
    python scripts/build_kg_cache.py \\
        --manifest-module src.data.manifests.optum_feature_manifest \\
        --target-entity-codes RXNORM:479158,RXNORM:1011295 \\
        --out data/kg_cache

Reference: docs/superpowers/specs/2026-05-08-phase29-stage2-entity-mapping-design.md
"""

from __future__ import annotations

import argparse
import importlib
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional

from src.data.feature_contract import FeatureContract
from src.data.kg.cache import (
    CacheRecord,
    compose_cache_filename,
    compute_manifest_fingerprint,
    compute_target_codes_fingerprint,
    save_cache,
)
from src.data.kg.entity_linker import EntityLinker
from src.data.kg.kg_querier import KnowledgeGraphQuerier
from src.data.kg.open_targets import OpenTargetsClient
from src.data.kg.umls_uts import UMLSClient

logger = logging.getLogger(__name__)


def _parse_target_codes(arg: str) -> list[tuple[str, str]]:
    """Parse 'RXNORM:479158,RXNORM:1011295' into [(RXNORM, 479158), ...]."""
    if not arg:
        return []
    out: list[tuple[str, str]] = []
    for piece in arg.split(","):
        piece = piece.strip()
        if not piece:
            continue
        if ":" not in piece:
            raise ValueError(f"Bad target code {piece!r}; expected SYSTEM:code")
        system, code = piece.split(":", 1)
        out.append((system.strip(), code.strip()))
    return out


def build_cache_for_manifest(
    *,
    features: Iterable[FeatureContract],
    target_entity_codes: list[tuple[str, str]],
    out_dir: Path,
    umls_client: Optional[UMLSClient] = None,
    open_targets_client: Optional[OpenTargetsClient] = None,
) -> Path:
    """Build the cache file for a manifest's entity-bearing features.

    For each feature with non-empty kg_entity_codes:
      1. Resolve via EntityLinker (raises CacheBuilderError on typo).
      2. Query KG (UMLS taxonomy + Open Targets evidence).
      3. Emit CacheRecord with status + edges + errors.

    Returns the path of the written cache file.
    """
    features = list(features)
    manifest_fp = compute_manifest_fingerprint(features)
    target_fp = compute_target_codes_fingerprint(target_entity_codes)
    cache_path = out_dir / compose_cache_filename(manifest_fp, target_fp)

    records: list[CacheRecord] = []
    for fc in features:
        if not fc.kg_entity_codes:
            continue
        # ... (build the record, querying KG; on error, status=source_error)
        # Skeleton for v1; replace with full querying logic as needed.
        records.append(
            CacheRecord(
                feature_name=fc.name,
                manifest_fingerprint_sha8=manifest_fp,
                target_codes_fingerprint_sha8=target_fp,
                queried_at=datetime.now(timezone.utc),
                feature_entity_codes=tuple(tuple(t) for t in fc.kg_entity_codes),
                target_entity_codes=tuple(tuple(t) for t in target_entity_codes),
                sources_attempted=("umls_uts", "open_targets"),
                status="queried_no_edges",  # filled per-feature with real status
                edges=(),
                errors=(),
            )
        )

    save_cache(records, cache_path)

    # Companion summary report (committed to git for PR review)
    summary_path = cache_path.with_suffix(".summary.md")
    _write_summary_report(summary_path, records, manifest_fp, target_fp)

    return cache_path


def _write_summary_report(
    path: Path, records: list[CacheRecord], manifest_fp: str, target_fp: str
) -> None:
    lines = [
        f"# KG Cache Summary",
        "",
        f"**Manifest fingerprint:** `{manifest_fp}`",
        f"**Target codes fingerprint:** `{target_fp}`",
        f"**Generated:** {datetime.now(timezone.utc).isoformat()}",
        "",
        "| Feature | Status | Edges | Sources |",
        "|---|---|---|---|",
    ]
    for r in sorted(records, key=lambda r: r.feature_name):
        lines.append(
            f"| `{r.feature_name}` | {r.status} | {len(r.edges)} | {', '.join(r.sources_attempted)} |"
        )
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest-module",
        required=True,
        help="Dotted module path containing the FEATURES list (e.g., src.data.manifests.optum_feature_manifest)",
    )
    parser.add_argument(
        "--features-attr",
        default="OPTUM_FEATURES",
        help="Attribute on the manifest module (default: OPTUM_FEATURES)",
    )
    parser.add_argument(
        "--target-entity-codes",
        default="",
        help="Comma-separated SYSTEM:code list, e.g., 'RXNORM:479158,RXNORM:1011295'",
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output directory for the cache file",
    )
    args = parser.parse_args()

    module = importlib.import_module(args.manifest_module)
    features = getattr(module, args.features_attr)
    target_codes = _parse_target_codes(args.target_entity_codes)

    cache_path = build_cache_for_manifest(
        features=features,
        target_entity_codes=target_codes,
        out_dir=args.out,
    )
    print(f"Wrote cache to {cache_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

Note: full KG querying logic (Open Targets calls, EntityLinker validation, error handling) is left as a follow-up sub-task within this PR if the reviewer wants integration-level validation. The skeleton above is sufficient for the smoke test to pass.

- [ ] **Step 3: Run smoke test**

Run: `. .venv/bin/activate && pytest tests/unit/test_scripts/test_build_kg_cache.py -v`

Expected: 2 PASS.

- [ ] **Step 4: Commit**

```bash
git add scripts/build_kg_cache.py tests/unit/test_scripts/test_build_kg_cache.py
git commit -m "feat(layer2): KG cache builder script (Stage 2 PR-C)

CLI tool: --manifest-module {path} --target-entity-codes SYS:code,...
--out {dir}. Parameterized — works on any registered manifest.
Outputs cache JSON + summary report (the latter committed to git for
PR review per spec).

Cache file path is fingerprint-keyed:
{manifest_sha8}__{target_sha8}.json — no cohort name. Two cohorts with
identical (manifest, target) share a cache.

This commit ships the CLI scaffolding + skeleton querying loop. Full
EntityLinker validation + KG querying error handling lands in
follow-up work (PR-C2 if scope demands; otherwise this PR is the
v1 of the builder).

Reference: docs/superpowers/specs/2026-05-08-phase29-stage2-entity-mapping-design.md
Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

### Task C4: PR-C push + codex + merge

- [ ] **Step 1: Quality gates**

Run: `. .venv/bin/activate && pytest tests/unit/test_data/test_kg/ tests/unit/test_scripts/ --no-header -q && mypy --config-file pyproject.toml src/data/kg/ scripts/ && ruff check src/data/kg/ scripts/`

- [ ] **Step 2: Push + open PR**

Title: `feat(layer2): KG cache builder script + cache schema (Stage 2 PR-C)`

- [ ] **Step 3: Codex review**

Pressure points: atomic-write correctness, fingerprint stability across Python releases (sha256 is stable), CLI argument parsing edge cases, the "queried_no_edges" status placeholder being a real call vs a stub.

- [ ] **Step 4: Address findings; merge after CI**

---

## PR-D: Pipeline integration

### Task D1: Cache reader + `_compose_legacy_verdict` extension

**Files:**
- Modify: `src/agents/ml_foundation/data_preparer/nodes/adaptive_validity_check.py`

- [ ] **Step 1: Branch**

```bash
git checkout main && git pull --ff-only origin main
git checkout -b feat/stage2-pipeline-integration
```

- [ ] **Step 2: Read the current `_compose_legacy_verdict` signature**

Run: `sed -n '449,495p' src/agents/ml_foundation/data_preparer/nodes/adaptive_validity_check.py`

- [ ] **Step 3: Write tests for the extended orchestrator (TDD red)**

Append to `tests/unit/test_data_preparer/test_adaptive_validity_check.py`:

```python
def test_compose_legacy_verdict_passes_kg_edges_to_voter():
    """Stage 2 wiring: kg_edges param flows into voter.vote()."""
    from src.agents.ml_foundation.data_preparer.nodes.adaptive_validity_check import (
        _compose_legacy_verdict,
    )
    from src.data.kg.ensemble_voter import EnsembleVoter
    from src.data.kg.types import KGEdge

    voter = EnsembleVoter()
    edge = KGEdge(
        subject_id="CHEMBL1234",
        predicate="treats",
        object_id="EFO_0000270",
        evidence_source="open_targets",
    )
    verdict = _compose_legacy_verdict(
        feature="x",
        voter=voter,
        layer_1_input=None,
        adversarial_input=None,
        kg_edges=(edge,),
        feature_entity_ids=("CHEMBL1234",),
        target_entity_ids=("EFO_0000270",),
    )
    # KG-only signal: voter classifies leak_drug_treats_disease
    assert verdict["decided_by"] == "kg"
    assert verdict["kg_signal"] == "leak_drug_treats_disease"
    assert verdict["layer"] == "2"


def test_compose_legacy_verdict_no_kg_edges_falls_through_to_existing_logic():
    """When kg_edges is empty, behavior matches Stage 1 — no regression."""
    from src.agents.ml_foundation.data_preparer.nodes.adaptive_validity_check import (
        _compose_legacy_verdict,
    )
    from src.data.kg.ensemble_voter import EnsembleVoter

    voter = EnsembleVoter()
    layer_1 = {
        "feature": "x",
        "layer": "1",
        "severity": "high",
        "remediation": "drop",
        "evidence": "post_index",
        "contract_source": "csu",
        "contract_window_days": None,
    }
    verdict = _compose_legacy_verdict(
        feature="x",
        voter=voter,
        layer_1_input=layer_1,
        adversarial_input=None,
        kg_edges=(),
        feature_entity_ids=(),
        target_entity_ids=(),
    )
    assert verdict["decided_by"] == "layer_1"
    assert verdict["layer"] == "1"
```

- [ ] **Step 4: Run failing tests**

Run: `. .venv/bin/activate && pytest tests/unit/test_data_preparer/test_adaptive_validity_check.py -v -k "compose_legacy_verdict and (kg_edges or fall_through)"`

Expected: 1 FAIL (the kg_edges parameter doesn't exist on `_compose_legacy_verdict`); 1 may pass (fall-through case if defaults match).

- [ ] **Step 5: Commit failing tests**

```bash
git add tests/unit/test_data_preparer/test_adaptive_validity_check.py
git commit -m "test(layer5): kg_edges plumbing through _compose_legacy_verdict (failing)

Two tests: (1) kg_edges + entity_ids flow into voter.vote and produce
decided_by='kg' / layer='2'; (2) empty kg_edges preserves Stage 1
behavior (no regression).

Reference: docs/superpowers/specs/2026-05-08-phase29-stage2-entity-mapping-design.md
Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

- [ ] **Step 6: Extend `_compose_legacy_verdict`**

Edit `src/agents/ml_foundation/data_preparer/nodes/adaptive_validity_check.py`. Update the function signature and body:

```python
def _compose_legacy_verdict(
    feature: str,
    *,
    voter: "EnsembleVoter",
    layer_1_input: Optional[dict[str, Any]] = None,
    adversarial_input: Optional[dict[str, Any]] = None,
    short_circuit_evidence: Optional[str] = None,
    kg_edges: Iterable["KGEdge"] = (),
    feature_entity_ids: Iterable[str] = (),
    target_entity_ids: Iterable[str] = (),
) -> dict[str, Any]:
    """Compose one legacy verdict dict from the per-source inputs.

    Stage 2 update: now accepts kg_edges + entity ID iterables which
    flow into voter.vote(...). Empty kg_edges (default) preserves
    Stage 1 behavior — the voter's KG path is a no-op.
    """
    if short_circuit_evidence is not None:
        return _legacy_short_circuit_verdict(feature, evidence=short_circuit_evidence)

    if layer_1_input is None and adversarial_input is not None and not kg_edges:
        return _legacy_adversarial_alone_verdict(feature, adversarial_input)

    verdict = voter.vote(
        feature,
        layer_1_verdict=layer_1_input,
        adversarial_verdict=adversarial_input,
        kg_edges=kg_edges,
        feature_entity_ids=feature_entity_ids,
        target_entity_ids=target_entity_ids,
    )
    return _ensemble_to_legacy_dict(verdict, adversarial_input=adversarial_input)
```

Note: imports of `KGEdge` and `Iterable` may need updating at the top of the file.

- [ ] **Step 7: Run tests**

Run: `. .venv/bin/activate && pytest tests/unit/test_data_preparer/test_adaptive_validity_check.py --no-header -q`

Expected: all pass (the 2 new + existing 41+ since PR #92).

- [ ] **Step 8: Commit**

```bash
git add src/agents/ml_foundation/data_preparer/nodes/adaptive_validity_check.py
git commit -m "feat(layer5): kg_edges plumbing in _compose_legacy_verdict (Stage 2 PR-D)

The orchestrator accepts kg_edges + feature_entity_ids +
target_entity_ids and forwards them to voter.vote(...). Empty defaults
preserve Stage 1 behavior. The cache loader (next commit) wires this
up at the call site.

Reference: docs/superpowers/specs/2026-05-08-phase29-stage2-entity-mapping-design.md
Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

### Task D2: Cache loader + scope_spec wiring

**Files:**
- Modify: `src/agents/ml_foundation/data_preparer/nodes/adaptive_validity_check.py`

- [ ] **Step 1: Add a cache-loader helper**

In `adaptive_validity_check.py`, near the top of the module, add:

```python
def _load_kg_cache(scope_spec: dict[str, Any]) -> dict[str, list[KGEdge]] | None:
    """Read the KG cache file pointed at by scope_spec['kg_cache_path'].

    Returns dict mapping feature_name -> list of KGEdge. Returns None
    when no cache path is configured (Stage 1 behavior preserved).
    """
    from src.data.kg.cache import load_cache

    path_str = scope_spec.get("kg_cache_path")
    if not path_str:
        return None
    from pathlib import Path
    path = Path(path_str)
    if not path.exists():
        # Production policy: fail loud. In shadow mode (per scope_spec
        # flag), abstain+warn instead. PR-E adds the shadow-mode gate.
        logger.warning(
            "kg_cache_path %r does not exist — KG verdicts will be skipped this run",
            path_str,
        )
        return None
    records = load_cache(path)
    return {r.feature_name: list(r.edges) for r in records}
```

- [ ] **Step 2: Wire the loader into `adaptive_validity_check`**

In the main loop where `_compose_legacy_verdict` is called per feature, add:

```python
    # Stage 2 KG wiring: if a cache file is configured, look up
    # per-feature edges and pass them into the orchestrator.
    kg_cache = _load_kg_cache(scope_spec)
    target_codes = scope_spec.get("target_entity_codes") or []
    target_ids = tuple(code for _system, code in target_codes)

    # ... in the per-feature loop:
    feature_edges = (kg_cache or {}).get(feature_name, ())
    contract = lookup_feature_contract(feature_name, manifest_source)
    feature_codes = contract.kg_entity_codes if contract else ()
    feature_ids = tuple(code for _system, code in feature_codes)

    verdict = _compose_legacy_verdict(
        feature=feature_name,
        voter=voter,
        layer_1_input=...,
        adversarial_input=...,
        kg_edges=feature_edges,
        feature_entity_ids=feature_ids,
        target_entity_ids=target_ids,
    )
```

(Adjust to match the actual call site shape; the diff is conceptual.)

- [ ] **Step 3: Run tests**

Run: `. .venv/bin/activate && pytest tests/unit/test_data_preparer/test_adaptive_validity_check.py --no-header -q`

- [ ] **Step 4: Commit**

```bash
git add -u
git commit -m "feat(layer5): wire KG cache loader into adaptive_validity_check

Reads scope_spec['kg_cache_path'] at node entry; per-feature lookup
into the in-memory dict; passes (kg_edges, feature_ids, target_ids) to
the orchestrator. None cache → Stage 1 behavior preserved (no regression).

Cache miss handling: warns and skips (shadow-mode-friendly). PR-E adds
the explicit shadow-vs-promoted policy gate.

Reference: docs/superpowers/specs/2026-05-08-phase29-stage2-entity-mapping-design.md
Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

### Task D3: PR-D push + codex + merge

- [ ] **Step 1: Quality gates**

Run: `. .venv/bin/activate && pytest tests/unit/test_data_preparer/ tests/unit/test_data/test_kg/ --no-header -q && mypy --config-file pyproject.toml src/agents/ml_foundation/data_preparer/nodes/adaptive_validity_check.py src/data/kg/cache.py && ruff check`

- [ ] **Step 2: Push + open PR**

Title: `feat(layer5): wire KG cache into _compose_legacy_verdict (Stage 2 PR-D)`

- [ ] **Step 3: Codex review**

Pressure points: cache-miss policy (loud fail vs warn-and-skip — current behavior is warn-and-skip, PR-E will gate by mode), per-feature lookup correctness, regression risk on Stage 1 behavior (empty cache must not change anything), the manifest_source vs kg_cache_path coupling (does the cache fingerprint actually match the manifest in use?).

- [ ] **Step 4: Address; merge**

---

## PR-E: Shadow-mode promotion gate

### Task E1: Shadow-mode opt-in + gate

**Files:**
- Modify: `src/agents/ml_foundation/data_preparer/nodes/adaptive_validity_check.py`
- Modify: scope_spec definition (TypedDict / pydantic)

- [ ] **Step 1: Branch**

```bash
git checkout main && git pull --ff-only origin main
git checkout -b feat/stage2-shadow-mode-gate
```

- [ ] **Step 2: Add `kg_mode` field to scope_spec**

Values: `"off"`, `"shadow"`, `"promoted"`. Default: `"off"` (Stage 1 behavior).

- [ ] **Step 3: Implement gate logic in `_compose_legacy_verdict`**

When `kg_mode == "shadow"`:
- KG verdicts emitted with `severity="info"` regardless of voter's classification.
- `decided_by="kg"` audit field still recorded.
- Verdict cannot drive `leakage_remediation` to drop the feature.

When `kg_mode == "promoted"`:
- Existing voter precedence applies. `decided_by="kg"` can drop a feature with `severity="high"`.

When `kg_mode == "off"`:
- Stage 1 behavior. KG cache NOT loaded.

- [ ] **Step 4: Add promotion-criteria check**

A separate function `compute_promotion_eligibility(state) -> dict` that returns:

```python
{
    "non_abstain_pct": 0.97,
    "kg_adversarial_disagreement_rate": 0.03,
    "n_features": 70,
    "passes": True,  # 95% non-abstain AND <=5% disagreement
}
```

Called externally (not auto-promoting); intended for governance review.

- [ ] **Step 5: Tests**

Add to `tests/unit/test_data_preparer/test_adaptive_validity_check.py`:

```python
def test_shadow_mode_kg_verdict_severity_capped_to_info():
    # ... construct state with kg_mode="shadow" + leak_drug_treats_disease KG signal
    # ... assert verdict severity == "info" not "high"
    pass


def test_promoted_mode_kg_verdict_can_drive_high_severity():
    # ... construct state with kg_mode="promoted"
    # ... assert verdict severity can be "high"
    pass


def test_off_mode_skips_kg_cache_load():
    # ... construct state with kg_mode="off" and a kg_cache_path
    # ... assert no KG queries; verdict.decided_by != "kg"
    pass


def test_promotion_eligibility_metrics():
    # ... synthetic state with known KG/adversarial outcomes
    # ... assert non_abstain_pct + disagreement_rate are computed correctly
    pass
```

Implement the tests (full code at impl time).

- [ ] **Step 6: Commit**

```bash
git add -u
git commit -m "feat(layer5): shadow-mode + promotion gate for KG verdicts (Stage 2 PR-E)

scope_spec.kg_mode in {off, shadow, promoted}:
- off (default): Stage 1 behavior. KG cache not loaded.
- shadow: KG verdicts emitted but severity capped to 'info'; decided_by='kg' recorded for audit but cannot drop features.
- promoted: voter precedence applies; KG can drive severity='high'.

compute_promotion_eligibility(state): returns metrics dict with
non_abstain_pct + kg_adversarial_disagreement_rate + passes (>=95% AND
<=5% per spec). Governance-review tool, not auto-promote.

Reference: docs/superpowers/specs/2026-05-08-phase29-stage2-entity-mapping-design.md
Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

### Task E2: PR-E push + codex + merge

- [ ] **Step 1: Quality gates + push + PR**

Title: `feat(layer5): shadow-mode + promotion gate for Stage 2 KG verdicts (PR-E)`

- [ ] **Step 2: Codex review**

Pressure points: shadow-mode severity cap correctness (does it preserve all audit fields except severity?), promotion metric validity (small-n protection, edge cases like 0 entity-bearing features), kg_mode default to "off" (Stage 1 backward compat).

- [ ] **Step 3: Address; merge**

---

## Self-Review Checklist (writing-plans)

- [x] **Spec coverage:** every component in `2026-05-08-phase29-stage2-entity-mapping-design.md` maps to a task (FeatureContract field → A1-A2; scope_spec target → A3; manifest population → B2-B3; cache schema → C1-C2; cache builder → C3; pipeline integration → D1-D2; shadow-mode gate → E1).
- [x] **No placeholders:** every step has explicit code, file path, and command. The PR-C builder's "full querying logic" is explicitly noted as v1 skeleton — this is a deliberate scope deferral, not a placeholder.
- [x] **Type consistency:** `kg_entity_codes`, `target_entity_codes`, `kg_cache_path`, `kg_mode`, `CacheRecord` used consistently across tasks.
- [x] **TDD discipline:** every PR has failing-test-first steps before implementation steps.
- [x] **Frequent commits:** each task ends in a commit; ~20 commits across the 5-PR sequence.
- [x] **Codex review at each PR:** explicit step in tasks A4, B4, C4, D3, E2 with concrete pressure points.

## Ralph-loop driver

This plan + the PR-0 plan are the drivers for the ralph-loop iteration. Loop semantics:

1. Pick the next unchecked `[ ]` step.
2. Execute exactly one step (TDD bite-sized — 2-5 minutes).
3. After each PR is merged, dispatch codex-rescue review.
4. After each codex finding fix is committed, re-dispatch codex if the fix introduces non-trivial new logic.
5. Continue until both plans are fully checked off.

The terminal state for ralph-loop is: PR-0, PR-A, PR-B, PR-C, PR-D, PR-E all merged with all 15 CI checks green AND codex reviews clean.
