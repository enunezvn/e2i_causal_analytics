# PR-0 — KG Predicate Reconciliation Implementation Plan

> **STATUS: COMPLETE 2026-05-08.** Merged as **PR #94** via `--rebase`. All steps `[x]`. See [`phase29_stage2_arc_close_20260508.md`](../../../../.claude/projects/-home-enunez-Projects-e2i-causal-analytics/memory/phase29_stage2_arc_close_20260508.md) for the full arc closure record.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Map Open Targets `datatypeId="known_drug"` rows to `predicate="treats"` at the `KGQuerier` boundary so the dead `EnsembleVoter.classify_kg_signal` treats path comes alive on real Open Targets data.

**Architecture:** One-rule mapping at `KGQuerier.query_drug_disease_edges`. Voter unchanged. Existing 12+ voter tests with synthetic `predicate="treats"` continue passing. New parameterized contract test + querier→voter integration test close the fixture-realism gap that hid the bug.

**Tech Stack:** Python 3.12, pandas, pytest, httpx (MockTransport for unit tests), mypy, ruff. Project uses `--rebase` PR merge policy (per `CLAUDE.md`).

**Spec:** `docs/superpowers/specs/2026-05-08-kg-predicate-reconciliation-design.md`

---

## File Structure

| File | Action | Purpose |
|---|---|---|
| `src/data/kg/kg_querier.py` | Modify | Add `datatypeId → predicate` mapping at line ~180 |
| `src/data/kg/ensemble_voter.py` | Modify | Add indication-neutrality docstring at line 126 |
| `tests/unit/test_data/test_kg/test_kg_querier.py` | Modify | Add parameterized predicate-by-datatype contract test |
| `tests/integration/test_kg/test_querier_voter_integration.py` | Create | New file — end-to-end test feeding querier output through voter |

---

## Task 1: Branch + parameterized querier contract test (TDD red)

**Files:**
- Modify: `tests/unit/test_data/test_kg/test_kg_querier.py`

- [x] **Step 1: Set up branch with proxy bypass**

```bash
git config --global http.https://github.com.proxy ""
git checkout main
git pull --ff-only origin main
git checkout -b fix/kg-predicate-reconciliation
```

- [x] **Step 2: Read existing querier test file to understand fixture pattern**

Run: `head -200 tests/unit/test_data/test_kg/test_kg_querier.py`

Look for: how `httpx.MockTransport` is used; how `_make_handler` constructs Open Targets responses; the existing `test_query_drug_disease_edges_happy_path` at ~line 187.

- [x] **Step 3: Write the failing parameterized contract test**

Append to `tests/unit/test_data/test_kg/test_kg_querier.py`:

```python
@pytest.mark.parametrize(
    "datatype_id, datasource_id, expected_predicate",
    [
        ("known_drug", "chembl", "treats"),
        ("known_drug", "clinical_trials", "treats"),
        ("literature", "europepmc", "associated_with"),
        ("genetic_association", "eva", "associated_with"),
        ("affected_pathway", "progeny", "associated_with"),
        ("animal_model", "phenodigm", "associated_with"),
    ],
)
def test_query_drug_disease_edges_predicate_by_datatype(
    datatype_id: str, datasource_id: str, expected_predicate: str
) -> None:
    """Each Open Targets datatypeId maps to one semantic predicate.

    Pre-fix the querier emitted "associated_with" for ALL rows. The
    "known_drug" parametrize cases would FAIL under pre-fix code,
    proving the dead-signal bug. Post-fix the "known_drug" rows produce
    "treats". Regression coverage prevents the mapping from drifting
    back to a single hardcoded predicate.

    Reference: docs/superpowers/specs/2026-05-08-kg-predicate-reconciliation-design.md
    """

    def _handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "data": {
                    "evidences": {
                        "rows": [
                            {
                                "datatypeId": datatype_id,
                                "datasourceId": datasource_id,
                                "score": 0.85,
                                "drug": {"id": "CHEMBL1234", "name": "drug-x"},
                                "disease": {"id": "EFO_0000270", "name": "disease-y"},
                                "literature": [],
                            }
                        ]
                    }
                }
            },
        )

    transport = httpx.MockTransport(_handler)
    ot = OpenTargetsClient(http=httpx.Client(transport=transport, base_url=OpenTargetsClient.DEFAULT_BASE_URL))
    querier = KnowledgeGraphQuerier(open_targets=ot, umls=_stub_umls())
    edges = querier.query_drug_disease_edges("CHEMBL1234", "EFO_0000270")
    assert len(edges) == 1
    assert edges[0].predicate == expected_predicate
    assert edges[0].evidence_source == "open_targets"
    assert edges[0].datasource == datasource_id
```

If the file's existing imports don't include `httpx`, `pytest`, `OpenTargetsClient`, `KnowledgeGraphQuerier`, add them. Reuse the existing `_stub_umls()` helper if present; otherwise use the simplest stub that returns no edges.

- [x] **Step 4: Run the new test to verify it fails (datatype_id="known_drug" cases)**

Run: `. .venv/bin/activate && pytest tests/unit/test_data/test_kg/test_kg_querier.py::test_query_drug_disease_edges_predicate_by_datatype -v`

Expected: 4 PASS (the `associated_with` parametrize cases) + 2 FAIL (the `treats` parametrize cases for `known_drug`). The failures are the bug — pre-fix code emits `associated_with` for everything, so the `known_drug → treats` assertion fails.

If ALL pass before the fix, something is wrong with the fixture wiring.

- [x] **Step 5: Commit the failing test**

```bash
git add tests/unit/test_data/test_kg/test_kg_querier.py
git commit -m "test(layer2): parameterized predicate-by-datatype contract test (failing)

Pre-fix: KGQuerier emits predicate=\"associated_with\" for every Open
Targets evidence row. The known_drug parametrize cases assert
predicate=\"treats\" and FAIL under main today. This is the test that
would have caught the dead-signal bug between PR #86 querier and
PR #88 voter.

Reference: docs/superpowers/specs/2026-05-08-kg-predicate-reconciliation-design.md
Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 2: Implement the `datatypeId` → predicate mapping (TDD green)

**Files:**
- Modify: `src/data/kg/kg_querier.py:147-189`

- [x] **Step 1: Read the current emission site**

Run: `sed -n '147,189p' src/data/kg/kg_querier.py`

Verify the loop iterates `evidences.rows` and emits `KGEdge(predicate="associated_with", ...)` at line ~180.

- [x] **Step 2: Replace the hardcoded predicate**

Edit `src/data/kg/kg_querier.py`. Find:

```python
            edges.append(
                KGEdge(
                    subject_id=str(drug.get("id") or drug_id),
                    subject_name=str(drug.get("name") or ""),
                    predicate="associated_with",
                    object_id=str(disease.get("id") or disease_id),
                    object_name=str(disease.get("name") or ""),
                    evidence_source="open_targets",
                    score=score,
                    pmids=pmids,
                    datasource=row.get("datasourceId"),
                    raw=row,
                )
            )
```

Replace with:

```python
            # Open Targets datatypeId taxonomy (Ochoa 2021, NAR): the only
            # datatype carrying drug-treats-disease semantics is "known_drug".
            # All other datatypes (literature, genetic_association,
            # affected_pathway, rna_expression, somatic_mutation, animal_model)
            # are gene/target-disease association, not therapeutic claim.
            # See docs/superpowers/specs/2026-05-08-kg-predicate-reconciliation-design.md.
            datatype_id = str(row.get("datatypeId") or "")
            predicate = "treats" if datatype_id == "known_drug" else "associated_with"
            edges.append(
                KGEdge(
                    subject_id=str(drug.get("id") or drug_id),
                    subject_name=str(drug.get("name") or ""),
                    predicate=predicate,
                    object_id=str(disease.get("id") or disease_id),
                    object_name=str(disease.get("name") or ""),
                    evidence_source="open_targets",
                    score=score,
                    pmids=pmids,
                    datasource=row.get("datasourceId"),
                    raw=row,
                )
            )
```

- [x] **Step 3: Update the `query_drug_disease_edges` docstring**

In the same file, find the docstring section listing edge attributes (around line 122-131). Replace:

```
            - predicate    = ``"associated_with"`` (Open Targets does not
                             distinguish causal direction; that is Layer 4's
                             job)
```

with:

```
            - predicate    = ``"treats"`` when the row's
                             ``datatypeId == "known_drug"`` (Open Targets'
                             unique drug-indication datatype, Ochoa 2021),
                             else ``"associated_with"``. The voter's
                             ``classify_kg_signal`` consumes the predicate
                             to drive the ``leak_drug_treats_disease``
                             classification.
```

- [x] **Step 4: Run the parameterized test to verify it passes**

Run: `. .venv/bin/activate && pytest tests/unit/test_data/test_kg/test_kg_querier.py::test_query_drug_disease_edges_predicate_by_datatype -v`

Expected: ALL 6 parametrize cases PASS.

- [x] **Step 5: Run the full kg test suite to confirm no regressions**

Run: `. .venv/bin/activate && pytest tests/unit/test_data/test_kg/ --no-header -q`

Expected: all tests pass (was 249 on main pre-Stage 2; +1 new parameterized test = 250+ now). No regressions.

- [x] **Step 6: Commit the fix**

```bash
git add src/data/kg/kg_querier.py
git commit -m "fix(layer2): map Open Targets datatypeId='known_drug' to predicate='treats'

KGQuerier.query_drug_disease_edges previously emitted
predicate='associated_with' for every Open Targets evidence row. The
EnsembleVoter (PR #88) only matches treats edges when predicate is in
TREATS_PREDICATES = {'treats', 'indicated_for', 'treats_indicates'},
so every Open Targets edge produced kg_signal='no_signal' regardless
of content — the Layer 2 decided_by='kg' precedence path was dead.

Fix at the querier boundary: map row['datatypeId'] == 'known_drug' →
predicate='treats'; all other datatypes keep 'associated_with'.

Keys on the Open Targets data-model invariant (datatypeId, Ochoa 2021)
not the contributing-pipeline detail (datasourceId='chembl'), so future
sources Open Targets adds with datatypeId='known_drug' (e.g.,
fda_label, clinical_trials_v2) are picked up automatically.

Voter unchanged. Existing 12+ voter tests with synthetic
predicate='treats' continue passing.

Reference: docs/superpowers/specs/2026-05-08-kg-predicate-reconciliation-design.md
Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 3: Integration test — querier → voter end-to-end

**Files:**
- Create: `tests/integration/test_kg/test_querier_voter_integration.py`

- [x] **Step 1: Verify the integration test directory exists**

Run: `ls tests/integration/test_kg/`

Expected: existing files like `test_kg_querier_live.py`. Confirm `__init__.py` is present.

- [x] **Step 2: Create the new integration test file**

Write `tests/integration/test_kg/test_querier_voter_integration.py`:

```python
"""End-to-end: KGQuerier → EnsembleVoter integration.

Closes the fixture-realism gap that hid the predicate-mismatch bug
between PR #86 (querier) and PR #88 (voter). Each was unit-tested in
isolation: the querier with stub Open Targets responses, the voter
with hand-crafted ``predicate="treats"`` edges. No test fed querier
output through the voter — so the bug (querier emits
``predicate="associated_with"``; voter matches only TREATS_PREDICATES)
shipped silently.

These tests use realistic Open Targets responses (mixed datatypeId
values) and assert the voter's ``classify_kg_signal`` produces the
right signal class on the chained output.

Reference: docs/superpowers/specs/2026-05-08-kg-predicate-reconciliation-design.md
"""

from __future__ import annotations

import httpx
import pytest

from src.data.kg.ensemble_voter import classify_kg_signal
from src.data.kg.kg_querier import KnowledgeGraphQuerier
from src.data.kg.open_targets import OpenTargetsClient
from src.data.kg.umls_uts import UMLSClient


def _stub_umls() -> UMLSClient:
    """Return a UMLS client whose HTTP transport returns 404 for all calls.

    The drug-disease integration test does not exercise UMLS; the
    KnowledgeGraphQuerier still requires a UMLSClient instance.
    """

    def _handler(_req: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"error": "stubbed"})

    transport = httpx.MockTransport(_handler)
    return UMLSClient(
        api_key="stub",
        http=httpx.Client(transport=transport, base_url=UMLSClient.DEFAULT_BASE_URL),
    )


def _two_row_handler(rows: list[dict]) -> object:
    def _handler(_req: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"data": {"evidences": {"rows": rows}}},
        )

    return _handler


def test_known_drug_row_produces_leak_signal_through_voter() -> None:
    """Real-shaped Open Targets response with one known_drug + one
    literature row → voter classifies leak_drug_treats_disease.

    Pre-fix querier emitted "associated_with" for both rows; voter
    classified "no_signal". Post-fix the known_drug row emits "treats";
    voter classifies "leak_drug_treats_disease" using only that row.
    """
    rows = [
        {
            "datatypeId": "known_drug",
            "datasourceId": "chembl",
            "score": 0.95,
            "drug": {"id": "CHEMBL1234", "name": "drug-x"},
            "disease": {"id": "EFO_0000270", "name": "disease-y"},
            "literature": [],
        },
        {
            "datatypeId": "literature",
            "datasourceId": "europepmc",
            "score": 0.30,
            "drug": {"id": "CHEMBL1234", "name": "drug-x"},
            "disease": {"id": "EFO_0000270", "name": "disease-y"},
            "literature": ["12345"],
        },
    ]
    transport = httpx.MockTransport(_two_row_handler(rows))
    ot = OpenTargetsClient(
        http=httpx.Client(transport=transport, base_url=OpenTargetsClient.DEFAULT_BASE_URL)
    )
    querier = KnowledgeGraphQuerier(open_targets=ot, umls=_stub_umls())

    edges = querier.query_drug_disease_edges("CHEMBL1234", "EFO_0000270")
    assert len(edges) == 2
    predicates = {e.predicate for e in edges}
    assert predicates == {"treats", "associated_with"}

    signal, considered = classify_kg_signal(
        edges,
        feature_entity_ids={"CHEMBL1234"},
        target_entity_ids={"EFO_0000270"},
    )
    assert signal == "leak_drug_treats_disease"
    # Only the known_drug edge counted.
    assert len(considered) == 1
    assert considered[0].predicate == "treats"


def test_only_non_known_drug_rows_produce_no_signal_through_voter() -> None:
    """An Open Targets response with NO known_drug rows produces no
    treats signal. Confirms non-treats datatypes don't accidentally
    promote.
    """
    rows = [
        {
            "datatypeId": "literature",
            "datasourceId": "europepmc",
            "score": 0.30,
            "drug": {"id": "CHEMBL1234", "name": "drug-x"},
            "disease": {"id": "EFO_0000270", "name": "disease-y"},
            "literature": ["12345"],
        },
        {
            "datatypeId": "genetic_association",
            "datasourceId": "eva",
            "score": 0.50,
            "drug": {"id": "CHEMBL1234", "name": "drug-x"},
            "disease": {"id": "EFO_0000270", "name": "disease-y"},
            "literature": [],
        },
    ]
    transport = httpx.MockTransport(_two_row_handler(rows))
    ot = OpenTargetsClient(
        http=httpx.Client(transport=transport, base_url=OpenTargetsClient.DEFAULT_BASE_URL)
    )
    querier = KnowledgeGraphQuerier(open_targets=ot, umls=_stub_umls())

    edges = querier.query_drug_disease_edges("CHEMBL1234", "EFO_0000270")
    assert len(edges) == 2
    assert all(e.predicate == "associated_with" for e in edges)

    signal, considered = classify_kg_signal(
        edges,
        feature_entity_ids={"CHEMBL1234"},
        target_entity_ids={"EFO_0000270"},
    )
    assert signal == "no_signal"
    assert considered == ()


def test_empty_evidence_rows_produce_no_signal() -> None:
    """An Open Targets response with zero rows is the queried-no-edges
    case — distinct from the predicate-mismatch dead-signal bug.
    """
    rows: list[dict] = []
    transport = httpx.MockTransport(_two_row_handler(rows))
    ot = OpenTargetsClient(
        http=httpx.Client(transport=transport, base_url=OpenTargetsClient.DEFAULT_BASE_URL)
    )
    querier = KnowledgeGraphQuerier(open_targets=ot, umls=_stub_umls())

    edges = querier.query_drug_disease_edges("CHEMBL1234", "EFO_0000270")
    assert edges == []

    signal, considered = classify_kg_signal(
        edges,
        feature_entity_ids={"CHEMBL1234"},
        target_entity_ids={"EFO_0000270"},
    )
    assert signal == "no_signal"
    assert considered == ()
```

- [x] **Step 3: Run the integration tests**

Run: `. .venv/bin/activate && pytest tests/integration/test_kg/test_querier_voter_integration.py -v`

Expected: 3 PASS.

- [x] **Step 4: Commit the integration test**

```bash
git add tests/integration/test_kg/test_querier_voter_integration.py
git commit -m "test(layer2): integration test KGQuerier → EnsembleVoter end-to-end

Closes the fixture-realism gap that hid the PR #86/#88 predicate
mismatch. Three tests:
1. known_drug row produces leak_drug_treats_disease through voter
2. non-known_drug rows produce no_signal (non-treats datatypes don't
   accidentally promote)
3. empty evidence rows are the queried-no-edges case (distinct from
   dead-signal bug)

Reference: docs/superpowers/specs/2026-05-08-kg-predicate-reconciliation-design.md
Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 4: Indication-neutrality docstring on the voter

**Files:**
- Modify: `src/data/kg/ensemble_voter.py:123-127`

- [x] **Step 1: Replace the comment block**

In `src/data/kg/ensemble_voter.py`, find:

```python
# KG predicates we recognise as drug→disease "treats" evidence
# (Open Targets) and as taxonomic isa (UMLS relations). Stored as
# lowercase for case-insensitive matching at classification time.
TREATS_PREDICATES: frozenset[str] = frozenset({"treats", "indicated_for", "treats_indicates"})
TAXONOMIC_PREDICATES: frozenset[str] = frozenset({"isa", "inverse_isa", "par", "chd", "rb", "rn"})
```

Replace with:

```python
# KG predicates we recognise as drug→disease "treats" evidence
# (Open Targets) and as taxonomic isa (UMLS relations). Stored as
# lowercase for case-insensitive matching at classification time.
#
# These predicate sets are INDICATION-NEUTRAL (verified 2026-05-08
# disease-domain audit; see docs/superpowers/specs/2026-05-08-kg-
# predicate-reconciliation-design.md §"Disease-domain coupling"). A
# CDK4/6 inhibitor "treats" breast cancer with the same vocabulary as
# a biologic "treats" CSU. The Open Targets ``datatypeId="known_drug"``
# taxonomy applies across diseases; UMLS taxonomic relations
# (isa/par/chd/etc.) are universal medical-ontology vocabulary.
#
# When a future cohort introduces a non-immunology indication, expand
# these sets ONLY if Open Targets adds new datatypes that carry
# therapeutic semantics (currently ``known_drug`` is the sole one) or
# UMLS adds taxonomic relations the project relies on. Externalisation
# to per-domain config is YAGNI until that pressure arrives.
TREATS_PREDICATES: frozenset[str] = frozenset({"treats", "indicated_for", "treats_indicates"})
TAXONOMIC_PREDICATES: frozenset[str] = frozenset({"isa", "inverse_isa", "par", "chd", "rb", "rn"})
```

- [x] **Step 2: Verify no logic change**

Run: `. .venv/bin/activate && pytest tests/unit/test_data/test_kg/test_ensemble_voter.py --no-header -q`

Expected: all voter tests pass (no behavioral change — comment only).

- [x] **Step 3: Commit**

```bash
git add src/data/kg/ensemble_voter.py
git commit -m "docs(layer2): indication-neutrality comment on voter predicate sets

Documents that TREATS_PREDICATES and TAXONOMIC_PREDICATES are universal
medical-ontology vocabulary applying across CSU/Optum/oncology/rare
diseases without modification (per disease-domain audit 2026-05-08).
Defers externalisation to per-domain config until non-immunology
cohort lands.

Reference: docs/superpowers/specs/2026-05-08-kg-predicate-reconciliation-design.md
Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 5: Quality gates + push + open PR

**Files:** none (verification + git ops)

- [x] **Step 1: Run mypy on touched files**

Run: `. .venv/bin/activate && mypy --config-file pyproject.toml src/data/kg/kg_querier.py src/data/kg/ensemble_voter.py`

Expected: no issues.

- [x] **Step 2: Run ruff check + format**

Run: `. .venv/bin/activate && ruff check src/data/kg/ tests/unit/test_data/test_kg/ tests/integration/test_kg/ && ruff format --check src/data/kg/ tests/unit/test_data/test_kg/ tests/integration/test_kg/`

Expected: all checks passed; no format-needed.

If `ruff format --check` reports differences, run `ruff format <files>` and amend the relevant commit (`git add -u && git commit --amend --no-edit`).

- [x] **Step 3: Run full kg unit suite**

Run: `. .venv/bin/activate && pytest tests/unit/test_data/test_kg/ --no-header -q`

Expected: all pass (was 249 on main; +6 parameterized = 255).

- [x] **Step 4: Push the branch**

Run: `git push -u origin fix/kg-predicate-reconciliation`

- [x] **Step 5: Open the PR**

Run:

```bash
gh pr create --title "fix(layer2): map Open Targets datatypeId='known_drug' to 'treats' predicate (PR-0)" --body "$(cat <<'EOF'
## Summary

Fix the predicate-mismatch bug between PR #86 KGQuerier and PR #88 EnsembleVoter that made the Layer 2 `decided_by="kg"` precedence path dead. Map Open Targets `datatypeId="known_drug"` rows to `predicate="treats"` at the querier boundary; voter unchanged.

## Bug

`KGQuerier.query_drug_disease_edges` (`src/data/kg/kg_querier.py:180`) emitted `predicate="associated_with"` for every Open Targets evidence row. `EnsembleVoter.classify_kg_signal` (`src/data/kg/ensemble_voter.py:221`) only matches treats edges when predicate is in `TREATS_PREDICATES = {"treats", "indicated_for", "treats_indicates"}`. Net effect: every Open Targets drug-disease edge produced `kg_signal="no_signal"`.

The bug was on the merge boundary between PRs #86 and #88. Each was unit-tested in isolation; no integration test fed querier output through the voter.

## Fix

Key on the Open Targets data-model invariant — `datatypeId == "known_drug"` (per Ochoa 2021 NAR) — not the contributing-pipeline detail (`datasourceId="chembl"`). Future sources Open Targets adds with `datatypeId="known_drug"` (e.g., `fda_label`, `clinical_trials_v2`) are picked up automatically. The `datatypeId` field is already pulled by the existing GraphQL query at `open_targets.py:64`.

## Test plan

- [x] **Parameterized querier contract test** — 6 cases covering each Open Targets datatypeId. The known_drug cases would fail under pre-fix code (proving the bug exists); they pass post-fix.
- [x] **Querier→voter integration test** (new file `tests/integration/test_kg/test_querier_voter_integration.py`) — closes the fixture-realism gap. Three scenarios: known_drug → `leak_drug_treats_disease`; non-known_drug → `no_signal`; empty rows → `no_signal`.
- [x] **Indication-neutrality comment** — `ensemble_voter.py:123-141` documents that predicate sets apply across CSU/Optum/oncology without modification; defers externalisation.
- [x] mypy + ruff clean.
- [x] Existing voter tests (12+ using synthetic `predicate="treats"`) pass — voter logic unchanged.

## Spec

`docs/superpowers/specs/2026-05-08-kg-predicate-reconciliation-design.md`

## Unblocks

Phase 2.9 Stage 2 entity-mapping (separate plan: `docs/superpowers/plans/2026-05-08-stage2-entity-mapping.md`).

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

- [x] **Step 6: Wait for CI green; merge with `--rebase`**

Watch: `gh pr checks <pr-number>`

When all 15 checks green:

```bash
gh pr merge <pr-number> --rebase --delete-branch
git checkout main && git pull --ff-only origin main
```

---

## Self-Review Checklist (writing-plans)

- [x] **Spec coverage:** every requirement in the PR-0 spec maps to a task: querier mapping (Task 2), parameterized contract test (Task 1), integration test (Task 3), indication-neutrality comment (Task 4), test pattern that would have failed under old code (Task 1 step 4 explicit).
- [x] **No placeholders:** every step has explicit code, exact file path, exact command, expected output.
- [x] **Type consistency:** `predicate`, `datatypeId`, `datasourceId`, `KGEdge`, `classify_kg_signal` used consistently across tasks.
- [x] **TDD discipline:** Task 1 writes failing test FIRST; Task 2 makes it green. Tasks 3+4 add additional coverage and docs.
- [x] **Frequent commits:** each task ends in a commit; six commits total on the branch.

## Codex adversarial review (per ralph-loop directive)

After Task 5 step 6, before declaring complete: dispatch `codex:codex-rescue` with the PR diff (similar pattern to PRs #88, #90, #92). If codex finds BLOCKERs/HIGHs, fix per-fix per-commit, push, re-run CI. Loop until clean. The expected codex pressure points (already considered in spec):

- Whether the rule should also gate on `maxPhaseForIndication >= 4` (deferred — out of scope per spec).
- Whether `clinical_trials` rows (which can be `datatypeId="known_drug"` but Phase I/II) should produce `treats` (current spec: yes; refinement to phase-gate is future work).
- Whether the integration test fixtures match real Open Targets payloads (responsive to fixture-realism critique).
