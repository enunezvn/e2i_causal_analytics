# KG Predicate Reconciliation Design (PR-0 prerequisite for Phase 2.9 Stage 2)

**Date:** 2026-05-08
**Status:** Design — pending implementation
**Companion spec:** `2026-05-08-phase29-stage2-entity-mapping-design.md` (depends on this)

## Problem

`KGQuerier.query_drug_disease_edges` (`src/data/kg/kg_querier.py:180`) emits `predicate="associated_with"` for every Open Targets evidence row. `EnsembleVoter.classify_kg_signal` (`src/data/kg/ensemble_voter.py:221`) only matches treats edges when `predicate in TREATS_PREDICATES = {"treats", "indicated_for", "treats_indicates"}`. Net effect: every Open Targets drug-disease edge in production produces `kg_signal="no_signal"` regardless of content. The Layer 2 `decided_by="kg"` precedence path is dead.

The bug is on the merge boundary between PR #86 (querier) and PR #88 (voter). Each was unit-tested in isolation; no integration test fed querier output through the voter.

## Why this blocks Phase 2.9 Stage 2

Stage 2 wires `kg_edges` into `_compose_legacy_verdict` so the voter receives KG signals per feature. With the predicate mismatch in place, that wiring would plumb `no_signal` everywhere — Stage 2 would be a no-op refactor for KG. PR-0 must land before any Stage 2 work has measurable effect.

## Research that grounds the fix

Three parallel research agents (2026-05-08) established the data-driven answer:

1. **Open Targets datasource semantics** (Ochoa 2021 NAR; Open Targets datatype taxonomy):
   - The Open Targets data model has seven canonical `datatypeId` values: `known_drug`, `genetic_association`, `somatic_mutation`, `affected_pathway`, `rna_expression`, `literature`, `animal_model`.
   - **Only `datatypeId="known_drug"` carries "drug treats disease" semantics.** Open Targets explicitly defines this as "existing drug that engages the target and is used to treat the disease."
   - All other datatypes are gene/target-disease association, not drug-disease therapeutic claim.
   - The `datatypeId` IS already pulled by the existing GraphQL query at `src/data/kg/open_targets.py:64` — zero schema-level change required.
   - `datasourceId` is the contributing pipeline (e.g., `chembl`, `clinical_trials`) within a `datatypeId`. Keying on `datasourceId` would silently miss future sources Open Targets adds with `datatypeId="known_drug"` (e.g., `fda_label`, `clinical_trials_v2`).

2. **Test-fixture audit** (Explore agent across `tests/`):
   - 12+ voter tests use a `_kg_treats_edge()` factory with synthetic `predicate="treats"`.
   - Only one test uses `predicate="associated_with"` (`tests/unit/test_data/test_kg/test_kg_querier.py:193`) and it asserts the predicate value but never feeds the edge through `classify_kg_signal`.
   - **Zero end-to-end tests** chain `query_drug_disease_edges → classify_kg_signal`.
   - Bug class: "fixture realism gap" — querier tested with stub Open Targets responses; voter tested with hand-crafted `treats` edges; never chained.

3. **Disease-domain coupling assessment** (Explore agent across plans + manifests):
   - Codebase is firmly immunology (CSU + Optum-CSU sub-cohorts). No oncology/PNH/rare-disease cohort in flight.
   - `TREATS_PREDICATES` and `TAXONOMIC_PREDICATES` are indication-neutral — a CDK4/6 inhibitor "treats" breast cancer with the same vocabulary; trastuzumab → HER2+ breast cancer evidence still lands in `datatypeId="known_drug"`.
   - Externalization to per-domain config is YAGNI for ≥6 months.

## The fix

**One-rule mapping at the querier boundary.** In `KGQuerier.query_drug_disease_edges` (`src/data/kg/kg_querier.py`), replace the unconditional `predicate="associated_with"` with:

```python
datatype_id = row.get("datatypeId") or ""
predicate = "treats" if datatype_id == "known_drug" else "associated_with"
```

The voter (`src/data/kg/ensemble_voter.py:221`) is unchanged. Its existing `evidence_source == "open_targets" AND predicate in TREATS_PREDICATES` guard correctly classifies edges as `leak_drug_treats_disease` once the querier emits the right predicate.

## Why `datatypeId` not `datasourceId`

| Key | Future-proof? | Semantic correctness | API cost |
|---|---|---|---|
| `datasourceId == "chembl"` | NO — silently misses new sources Open Targets adds with `datatypeId="known_drug"` | Approximate; chembl IS today's only `known_drug` source but the relationship is contingent | Same |
| `datatypeId == "known_drug"` | YES — Open Targets data-model invariant per Ochoa 2021 | Exact match to the documented semantic | Same; field already in GraphQL query at `open_targets.py:64` |

`datatypeId` is the stable Open Targets data-model invariant; `datasourceId` is a pipeline detail that may be reorganized between releases. The cost of the safer key is zero.

## Test strategy — closes the fixture-realism gap

The integration test must be **non-tautological**: assert behavior that would have failed under the OLD code AND will fail under any future regression.

### Querier-level parameterized contract test

`tests/unit/test_data/test_kg/test_kg_querier.py` — new parameterized test:

```python
@pytest.mark.parametrize("datatype_id, datasource_id, expected_predicate", [
    ("known_drug", "chembl", "treats"),
    ("known_drug", "clinical_trials", "treats"),    # belongs to known_drug datatype
    ("literature", "europepmc", "associated_with"),
    ("genetic_association", "eva", "associated_with"),
    ("affected_pathway", "progeny", "associated_with"),
    ("animal_model", "phenodigm", "associated_with"),
])
def test_query_drug_disease_edges_predicate_by_datatype(
    datatype_id, datasource_id, expected_predicate
):
    """Each Open Targets datatypeId maps to one semantic predicate.

    Pre-fix the querier emitted "associated_with" for all rows, so the
    "known_drug" rows of this test would have failed. Post-fix the
    "known_drug" rows produce "treats". Regression coverage prevents the
    mapping from drifting back to a single hardcoded predicate.
    """
    transport = httpx.MockTransport(_make_handler({...}))
    edges = client.query_drug_disease_edges(...)
    assert edges[0].predicate == expected_predicate
```

This test would have FAILED under the old code (because all datatype rows produced `"associated_with"`, the `known_drug` parametrize cases would mismatch on the expected value).

### Integration test (querier → voter)

`tests/integration/test_kg/test_querier_voter_integration.py` (new file):

```python
def test_known_drug_row_produces_leak_signal_through_voter():
    """End-to-end: real-shaped Open Targets response → KGQuerier →
    EnsembleVoter.classify_kg_signal. This is the chain that was broken
    in production: PR #86 emitted predicate="associated_with"; PR #88
    matched only "treats". Bug invisible to unit tests.
    """
    # Mock Open Targets with TWO realistic rows — one known_drug, one literature
    transport = httpx.MockTransport(_two_row_handler(...))
    client = OpenTargetsClient(...)
    querier = KnowledgeGraphQuerier(open_targets=client, ...)

    edges = querier.query_drug_disease_edges("CHEMBL_X", "EFO_Y")
    assert len(edges) == 2

    # Now chain through the voter
    voter = EnsembleVoter()
    signal, considered = classify_kg_signal(
        edges,
        feature_entity_ids={"CHEMBL_X"},
        target_entity_ids={"EFO_Y"},
    )
    # Pre-fix: signal == "no_signal" (bug). Post-fix: "leak_drug_treats_disease".
    assert signal == "leak_drug_treats_disease"
    assert len(considered) == 1  # only the known_drug edge counts
```

### Documentation comment at the voter

`src/data/kg/ensemble_voter.py:126` — add a comment block:

```python
# These predicate sets are INDICATION-NEUTRAL (verified 2026-05-08 via
# disease-domain audit). A CDK4/6 inhibitor "treats" breast cancer with
# the same vocabulary as a biologic "treats" CSU. The Open Targets
# `datatypeId="known_drug"` taxonomy applies across diseases; UMLS
# taxonomic relations (isa/par/chd/etc.) are universal.
#
# When a future cohort introduces a non-immunology indication, expand
# the TREATS predicate set ONLY if Open Targets adds new datatypes that
# carry therapeutic semantics (currently `known_drug` is the sole one).
# Externalization to per-domain config is YAGNI until that pressure
# arrives.
TREATS_PREDICATES: frozenset[str] = frozenset({"treats", "indicated_for", "treats_indicates"})
```

## Out of scope (future work)

- **Maxphase refinement.** A natural follow-up: gate "treats" emission on `drug.indications.maxPhaseForIndication >= 4` (regulatory-approved indication) for higher-confidence treats verdicts. The query already pulls this field at `src/data/kg/open_targets.py:53` (`drug.indications.rows[].maxPhaseForIndication`). Defer to a separate PR with its own design note. PR-0 emits binary "treats"/"associated_with" only.
- **Per-domain predicate externalization.** YAGNI per disease-domain audit. Document indication-neutrality in the voter comment block; revisit if/when a non-immunology cohort lands.
- **`query_disease_hierarchy` / `query_concept_relations` predicate audits.** UMLS predicates are pass-through `additionalRelationLabel` strings; live integration test at `test_kg_querier_live.py:60-70` provides real-payload grounding. No equivalent fixture-realism gap. Defer.

## Acceptance criteria

1. `KGQuerier.query_drug_disease_edges` maps `datatypeId="known_drug"` → `predicate="treats"`; all other datatypes → `predicate="associated_with"`.
2. Parameterized querier contract test (above) passes; would have failed under pre-fix code.
3. Querier-voter integration test passes; demonstrates `leak_drug_treats_disease` signal flows end-to-end on a known_drug row.
4. `ensemble_voter.py:126` carries the indication-neutrality comment block.
5. Existing voter unit tests (12+ using synthetic `predicate="treats"`) continue to pass — voter logic unchanged.
6. mypy + ruff clean on touched files; full kg test suite (currently 249 tests on main) passes.
7. PR title: `fix(layer2): map Open Targets datatypeId="known_drug" to "treats" predicate`.

## Risk assessment

- **Backwards compatibility:** SAFE. Source-tree audit found `"associated_with"` only at `kg_querier.py:125` (docstring) and `kg_querier.py:180` (the emission). No downstream consumer branches on the string.
- **Voter-side false positives:** SAFE. UMLS edges (`evidence_source="umls_relations"`) cannot trigger the treats path because the voter requires `evidence_source == "open_targets"` at line 221. UMLS `may_treat` (a real relation per `test_kg_querier.py:373`) is not in `TREATS_PREDICATES` and would not collide.
- **Future Open Targets schema drift:** PARTIAL HEDGE. If Open Targets reorganizes `datatypeId` values in a future release, the mapping rule needs an update. Detection: live integration test would fail, surfacing the drift. Mitigation: keep the mapping rule centralised and short.

## Sequencing

PR-0 ships standalone, before any Phase 2.9 Stage 2 work. Stage 2's `_compose_legacy_verdict` wiring depends on PR-0's `predicate="treats"` emission to produce non-`no_signal` verdicts.

## Companion spec

For the entity-mapping work that consumes the fixed predicate semantics, see `2026-05-08-phase29-stage2-entity-mapping-design.md`.
