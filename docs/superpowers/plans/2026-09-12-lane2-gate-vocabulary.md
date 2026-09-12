# Lane 2 — One vocabulary per gate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give the discovery gate its own enum name, fix the chat provenance summarizer that merges two gate vocabularies and falls open to "proceed", type the gate fields as Literals/unions end to end, and put the discovery vocabulary under the enum-sync CI guard.

**Architecture:** Rename only in Python (`DiscoveryGateDecision`) and on the wire (Literals in Pydantic, unions in TS); the DB column names stay. The summarizer reduces to the three refutation tokens and reports `unknown` rather than `proceed` for anything else. The domain-vocabulary YAML gains a discovery section that `scripts/validate_vocabulary_enum_sync.py` checks against the Python enum and the SQL type in `database/ml/026_causal_discovery_tables.sql`.

**Tech Stack:** Python 3.12 / Pydantic v2 `Literal`, React + TypeScript, `openapi-typescript` (regenerates `frontend/src/types/generated/api.ts`, verify-types gate), PyYAML, pytest `-n 0`, vitest.

Spec: `docs/superpowers/specs/2026-09-12-debts-3-4-wave-design.md` §5. Worktree `.worktrees/lane-vocab`, branch `claude/1991-gate-vocabulary`, ONE push at the end.

```bash
cd /home/enunez/Projects/e2i_causal_analytics && git fetch origin
git worktree add -b claude/1991-gate-vocabulary .worktrees/lane-vocab origin/main && cd .worktrees/lane-vocab
```

---

## File map

| file | change |
|---|---|
| `src/causal_engine/discovery/base.py` | `GateDecision` → `DiscoveryGateDecision` |
| `src/causal_engine/discovery/{gate,cache,observability,__init__}.py`, `src/causal_engine/__init__.py`, `src/agents/causal_impact/nodes/graph_builder.py` | import/name updates |
| `tests/unit/test_causal_engine/test_discovery/*.py`, `tests/unit/test_agents/test_causal_impact/test_graph_builder_priors.py`, `tests/unit/test_database/test_migration_ml_036_discovery_to_public.py` | import updates |
| `src/api/routes/chatbot_tools.py` | `_summarize_refutation_rows` |
| `tests/unit/test_api/test_chatbot_causal_validation_provenance.py` | new cases |
| `src/api/schemas/causal.py` | Literals on `gate_decision` (×2) and `expert_review_decision` |
| `src/api/schemas/expert_review.py` | `DagStructure`-shaped `discovery_gate_decision` Literal (new `DagStructureSnapshot` model) |
| `frontend/src/types/causal.ts`, `frontend/src/types/expert-review.ts` | unions |
| `frontend/src/components/causal/CausalAnalysisDetail.tsx`, `ReviewStatusPanel.tsx` | exhaustive rendering |
| `frontend/src/types/generated/api.ts` | regenerated |
| `config/domain_vocabulary.yaml`, `scripts/validate_vocabulary_enum_sync.py` | discovery section + check; `negative_control_outcome` |
| `tests/unit/test_scripts/test_validate_vocabulary_enum_sync_2029.py` | new |

---

### Task 1: Rename the discovery enum

**Files:** listed above (rename block)
**Test:** `tests/unit/test_causal_engine/test_discovery/test_gate.py` (existing, updated import)

- [ ] **Step 1: Write the failing import test**

Create `tests/unit/test_causal_engine/test_discovery/test_gate_enum_name_1991.py`:
```python
"""#1991 debt 4: the discovery gate enum has its own name; the refutation gate keeps GateDecision."""


def test_discovery_enum_is_named_distinctly():
    from src.causal_engine.discovery.base import DiscoveryGateDecision
    from src.causal_engine.refutation_runner import GateDecision

    assert {m.value for m in DiscoveryGateDecision} == {"accept", "review", "reject", "augment"}
    assert {m.value for m in GateDecision} == {"proceed", "review", "block"}
    assert DiscoveryGateDecision.__name__ != GateDecision.__name__


def test_old_name_is_gone_from_discovery():
    import src.causal_engine.discovery.base as base

    assert not hasattr(base, "GateDecision")
```

- [ ] **Step 2: Run it to verify it fails**

Run: `.venv/bin/pytest tests/unit/test_causal_engine/test_discovery/test_gate_enum_name_1991.py -n 0 -q`
Expected: FAIL — `ImportError: cannot import name 'DiscoveryGateDecision'`.

- [ ] **Step 3: Rename everywhere**

```bash
# the class and every reference in the discovery package, its package exports, the graph builder and tests
grep -rl "GateDecision" src/causal_engine/discovery src/causal_engine/__init__.py src/agents/causal_impact/nodes/graph_builder.py tests/unit/test_causal_engine/test_discovery tests/unit/test_agents/test_causal_impact/test_graph_builder_priors.py tests/unit/test_database/test_migration_ml_036_discovery_to_public.py \
 | xargs sed -i -E 's/\bGateDecision\b/DiscoveryGateDecision/g'
```
Then hand-check the two files that mention BOTH enums and revert the refutation one:
- `src/causal_engine/__init__.py`: the import from `.refutation_runner` and its `__all__` entry must stay `GateDecision`; the discovery import becomes `DiscoveryGateDecision` and `__all__` gains it.
- `src/agents/causal_impact/nodes/graph_builder.py`: imports from `src.causal_engine.discovery` only → all become `DiscoveryGateDecision`. Confirm with `grep -n "GateDecision" src/agents/causal_impact/nodes/graph_builder.py`.
Docstring examples in `discovery/__init__.py` and `gate.py` follow the sed.

- [ ] **Step 4: Run the discovery, graph-builder and 036 suites**

Run: `.venv/bin/pytest tests/unit/test_causal_engine/test_discovery tests/unit/test_agents/test_causal_impact/test_graph_builder.py tests/unit/test_agents/test_causal_impact/test_graph_builder_priors.py tests/unit/test_agents/test_causal_impact/test_graph_builder_guarantee.py tests/unit/test_database/test_migration_ml_036_discovery_to_public.py tests/unit/test_causal_engine/test_discovery/test_gate_enum_name_1991.py -n 0 -q`
Expected: all pass; `grep -rn "discovery.*import.*\bGateDecision\b" src tests` → 0 hits.

- [ ] **Step 5: Commit**

```bash
git add -A src/causal_engine src/agents/causal_impact/nodes/graph_builder.py tests/unit/test_causal_engine tests/unit/test_agents/test_causal_impact tests/unit/test_database
git commit -m "refactor(discovery): rename the discovery gate enum to DiscoveryGateDecision (#1991 debt 4)"
```

---

### Task 2: The summarizer speaks one vocabulary and fails closed

**Files:**
- Modify: `src/api/routes/chatbot_tools.py` (`_summarize_refutation_rows`, ~610–660)
- Test: `tests/unit/test_api/test_chatbot_causal_validation_provenance.py`

- [ ] **Step 1: Write the failing tests** (append to the existing file)

```python
@pytest.mark.unit
@pytest.mark.parametrize("token", ["accept", "augment", "reject", None, "bogus"])
def test_summarize_unknown_gate_is_unknown_never_proceed(token):
    rows = _seeded_evidence_rows(n_passed=2)
    for r in rows:
        r["gate_decision"] = token
    summary = _summarize_refutation_rows(rows)
    assert summary["gate_decision"] == "unknown"
    assert summary["gate_unreadable_rows"] == 2


@pytest.mark.unit
def test_summarize_mixed_unknown_and_block_still_blocks():
    rows = _seeded_evidence_rows(n_passed=2)
    rows[0]["gate_decision"] = None
    rows[1]["gate_decision"] = "block"
    summary = _summarize_refutation_rows(rows)
    assert summary["gate_decision"] == "block"
    assert summary["gate_unreadable_rows"] == 1


@pytest.mark.unit
def test_summarize_proceed_only_reads_proceed_and_zero_unreadable():
    summary = _summarize_refutation_rows(_seeded_evidence_rows(n_passed=3))
    assert summary["gate_decision"] == "proceed"
    assert summary["gate_unreadable_rows"] == 0
```

- [ ] **Step 2: Run to verify they fail**

Run: `.venv/bin/pytest tests/unit/test_api/test_chatbot_causal_validation_provenance.py -n 0 -q -k "unknown_gate or mixed_unknown or proceed_only"`
Expected: FAIL — `assert 'proceed' == 'unknown'` and `KeyError: 'gate_unreadable_rows'`.

- [ ] **Step 3: Rewrite the gate reduction**

Replace the docstring sentence "extended over the full ``gate_decision`` enum (reject counts as blocking, augment as review-band, accept as proceed)." with "The column holds ONLY the refutation vocabulary (proceed / review / block); any other value, including NULL, is reported as ``unknown`` and counted in ``gate_unreadable_rows`` — never mapped to proceed (#1991 debt 4)." Replace the reduction block with:

```python
    _REFUTATION_GATES = {"proceed", "review", "block"}
    gates = [r.get("gate_decision") for r in rows]
    known = {g for g in gates if g in _REFUTATION_GATES}
    unreadable = sum(1 for g in gates if g not in _REFUTATION_GATES)
    if "block" in known:
        gate = "block"
    elif "review" in known:
        gate = "review"
    elif known and not unreadable:
        gate = "proceed"
    else:
        # Fail closed: a row we cannot read is not evidence of robustness.
        gate = "unknown"
```
and add `"gate_unreadable_rows": unreadable,` to the returned dict. Keep everything else. (Note: a proceed-only set with one unreadable row reads `unknown`; block/review still win because they are the more conservative readings.)

- [ ] **Step 4: Run the whole file**

Run: `.venv/bin/pytest tests/unit/test_api/test_chatbot_causal_validation_provenance.py -n 0 -q`
Expected: all pass (the existing `block_wins` and `proceed` cases still hold).

- [ ] **Step 5: Commit**

```bash
git add src/api/routes/chatbot_tools.py tests/unit/test_api/test_chatbot_causal_validation_provenance.py
git commit -m "fix(chat): refutation summary speaks one gate vocabulary and fails closed on unreadable rows (#1991 debt 4)"
```

---

### Task 3: Literal types on the wire and exhaustive rendering

**Files:**
- Modify: `src/api/schemas/causal.py` (~592, ~607, ~872), `src/api/schemas/expert_review.py` (`PendingReviewItem.dag_structure_json`)
- Modify: `frontend/src/types/causal.ts` (~244, ~258, ~399), `frontend/src/types/expert-review.ts` (~27)
- Modify: `frontend/src/components/causal/CausalAnalysisDetail.tsx` (`gateBadge`), `frontend/src/components/causal/ReviewStatusPanel.tsx` (`DECISION_COPY`)
- Regenerate: `frontend/src/types/generated/api.ts`
- Test: `tests/unit/test_api/test_causal_schema_gate_literals_1991.py`, `frontend/src/components/causal/__tests__/gateBadge.test.tsx`

- [ ] **Step 1: Write the failing schema test**

```python
"""#1991 debt 4: gate fields are Literals of their own vocabulary."""
import pytest
from pydantic import ValidationError

from src.api.schemas.causal import RefutationSummary


def test_refutation_gate_rejects_discovery_tokens():
    with pytest.raises(ValidationError):
        RefutationSummary(gate_decision="reject")
    assert RefutationSummary(gate_decision="block").gate_decision == "block"


def test_expert_review_decision_is_its_own_vocabulary():
    with pytest.raises(ValidationError):
        RefutationSummary(expert_review_decision="review")
    ok = RefutationSummary(expert_review_decision="pending_review")
    assert ok.expert_review_decision == "pending_review"
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/pytest tests/unit/test_api/test_causal_schema_gate_literals_1991.py -n 0 -q`
Expected: FAIL — no `ValidationError` raised.

- [ ] **Step 3: Add the Literals**

In `src/api/schemas/causal.py` (typing already imports `Literal`), near the top after imports:
```python
RefutationGate = Literal["proceed", "review", "block"]
ExpertReviewDecision = Literal[
    "proceed", "renewal_required", "pending_review", "rejected", "blocked", "unavailable"
]
DiscoveryGate = Literal["accept", "review", "reject", "augment"]
```
Change `RefutationSummary.gate_decision: Optional[str]` → `Optional[RefutationGate]`, `expert_review_decision: Optional[str]` → `Optional[ExpertReviewDecision]`, and the leaderboard effect's `gate_decision: Optional[str]` (~872) → `Optional[RefutationGate]`. Keep every `description=`.

In `src/api/schemas/expert_review.py` add a typed snapshot and use it for `dag_structure_json`:
```python
class DagStructureSnapshot(BaseModel):
    """The sanitized causal-graph snapshot (mig 097) with the DISCOVERY gate typed."""

    model_config = ConfigDict(extra="allow")

    nodes: List[str] = []
    edges: List[List[str]] = []
    treatment_nodes: Optional[List[str]] = None
    outcome_nodes: Optional[List[str]] = None
    adjustment_sets: Optional[List[List[str]]] = None
    augmented_edges: Optional[List[List[str]]] = None
    discovery_gate_decision: Optional[Literal["accept", "review", "reject", "augment"]] = None
    confidence: Optional[float] = None
    dag_version_hash: Optional[str] = None
```
and `dag_structure_json: Optional[DagStructureSnapshot] = None` (the `_parse_json_string` validator still runs `mode="before"`, so string cells keep parsing). Add `Literal` to the typing import.

Frontend `types/causal.ts`:
```ts
export type RefutationGate = 'proceed' | 'review' | 'block';
export type ExpertReviewDecision =
  | 'proceed' | 'renewal_required' | 'pending_review' | 'rejected' | 'blocked' | 'unavailable';
```
use `gate_decision?: RefutationGate | null;` (both places) and `expert_review_decision?: ExpertReviewDecision | null;`. `types/expert-review.ts`: `export type DiscoveryGate = 'accept' | 'review' | 'reject' | 'augment';` and `discovery_gate_decision?: DiscoveryGate | null;`.

`CausalAnalysisDetail.tsx`:
```tsx
const GATE_BADGE: Record<RefutationGate, { label: string; variant: 'default' | 'secondary' | 'destructive' }> = {
  proceed: { label: 'Proceed', variant: 'default' },
  review: { label: 'Review', variant: 'secondary' },
  block: { label: 'Blocked', variant: 'destructive' },
};
function gateBadge(decision?: RefutationGate | null) {
  if (!decision) return <Badge variant="outline">Not gated</Badge>;
  const b = GATE_BADGE[decision];
  return <Badge variant={b.variant}>{b.label}</Badge>;
}
```
`ReviewStatusPanel.tsx`: `const DECISION_COPY: Record<ExpertReviewDecision, {...}>` (same six entries) and the fallback branch `: decision ? (<Badge variant="outline">{decision}</Badge>)` is removed — with the union, `copy` is always defined when `decision` is.

- [ ] **Step 4: Regenerate the client and type-check**

```bash
.venv/bin/python -m scripts.export_openapi --output /tmp/openapi.json
cd frontend && npx openapi-typescript ../openapi.json -o src/types/generated/api.ts && npx tsc --noEmit --strict --skipLibCheck src/types/generated/api.ts && npm run typecheck && cd ..
git diff --stat frontend/src/types/generated/api.ts
```
Expected: `api.ts` diff shows `gate_decision?: "proceed" | "review" | "block" | null` style enums (the ONLY expected diff in this lane); `tsc` and `npm run typecheck` rc=0.

- [ ] **Step 5: Frontend test for the badge**

`frontend/src/components/causal/__tests__/gateBadge.test.tsx`:
```tsx
import { render, screen } from '@testing-library/react';
import { describe, expect, it } from 'vitest';
import { gateBadge } from '../CausalAnalysisDetail';

describe('gateBadge', () => {
  it('renders every gate', () => {
    render(<>{gateBadge('proceed')}{gateBadge('review')}{gateBadge('block')}{gateBadge(null)}</>);
    expect(screen.getByText('Proceed')).toBeInTheDocument();
    expect(screen.getByText('Review')).toBeInTheDocument();
    expect(screen.getByText('Blocked')).toBeInTheDocument();
    expect(screen.getByText('Not gated')).toBeInTheDocument();
  });
});
```
Export `gateBadge` from the component (`export function gateBadge`). Run: `cd frontend && npx vitest run src/components/causal/__tests__/gateBadge.test.tsx && cd ..` → pass.

- [ ] **Step 6: Backend tests, commit**

Run: `.venv/bin/pytest tests/unit/test_api/test_causal_schema_gate_literals_1991.py tests/unit/test_api/test_causal_agent_analyze.py tests/unit/test_api/test_causal_agent_analyze_expert_review_1971.py tests/unit/test_api/test_causal_agent_analyze_review_caveat_1995.py tests/unit/test_api/test_expert_review*.py -n 0 -q` → pass.
```bash
git add src/api/schemas/causal.py src/api/schemas/expert_review.py frontend/src/types frontend/src/components/causal tests/unit/test_api/test_causal_schema_gate_literals_1991.py
git commit -m "feat(api,frontend): gate fields are Literals/unions of their own vocabulary (#1991 debt 4)"
```

---

### Task 4: The discovery vocabulary under the enum-sync guard

**Files:**
- Modify: `config/domain_vocabulary.yaml` (section 4), `scripts/validate_vocabulary_enum_sync.py` (`enum_checks`)
- Test: `tests/unit/test_scripts/test_validate_vocabulary_enum_sync_1991.py`

- [ ] **Step 1: Write the failing tests**

```python
"""#1991 debt 4: the discovery gate vocabulary is under the enum-sync guard, and the guard
fails when any of its three sources drifts (mutation test)."""
from __future__ import annotations

import importlib
import re
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[3]
YAML = REPO / "config" / "domain_vocabulary.yaml"
SQL = REPO / "database" / "ml" / "026_causal_discovery_tables.sql"


def _script():
    return importlib.import_module("scripts.validate_vocabulary_enum_sync")


def test_yaml_has_discovery_section_with_augment():
    vocab = yaml.safe_load(YAML.read_text())
    assert vocab["discovery_gate_decisions"]["values"] == ["accept", "review", "reject", "augment"]
    assert "negative_control_outcome" in vocab["refutation_test_types"]["values"]


def test_sql_extractor_reads_schema_qualified_type():
    values = _script().extract_enum_from_sql(SQL, "ml.gate_decision")
    assert values == ["accept", "review", "reject", "augment"]


def test_python_enum_matches_yaml():
    from src.causal_engine.discovery.base import DiscoveryGateDecision

    vocab = yaml.safe_load(YAML.read_text())
    assert [m.value for m in DiscoveryGateDecision] == vocab["discovery_gate_decisions"]["values"]


def test_guard_fails_on_drift(tmp_path, monkeypatch):
    s = _script()
    drifted = tmp_path / "vocab.yaml"
    vocab = yaml.safe_load(YAML.read_text())
    vocab["discovery_gate_decisions"]["values"].remove("augment")
    drifted.write_text(yaml.safe_dump(vocab))
    monkeypatch.setattr(s, "VOCAB_PATH", drifted, raising=False)
    assert s.validate_enum_sync(vocab_path=drifted) is False
```

- [ ] **Step 2: Run to verify they fail**

Run: `.venv/bin/pytest tests/unit/test_scripts/test_validate_vocabulary_enum_sync_1991.py -n 0 -q`
Expected: FAIL — `KeyError: 'discovery_gate_decisions'`, extractor returns `[]` for the dotted name, `validate_enum_sync() got an unexpected keyword argument`.

- [ ] **Step 3: Implement**

`config/domain_vocabulary.yaml`, section 4: add `- negative_control_outcome   # Negative-control outcome (mig 138)` to `refutation_test_types.values`, and after `gate_decisions`:
```yaml
discovery_gate_decisions:
  description: "Causal-discovery gate decisions (discovered_dags.gate_decision, SQL type discovery_gate_decision)"
  values:
    - accept    # High confidence, use the discovered DAG
    - review    # Medium confidence, requires expert validation
    - reject    # Low confidence, use the manual DAG
    - augment   # Supplement the manual DAG with high-confidence edges
```
`scripts/validate_vocabulary_enum_sync.py`:
- `extract_enum_from_sql`: the regex must accept a schema-qualified name: `pattern = rf"CREATE\s+TYPE\s+{re.escape(enum_name)}\s+AS\s+ENUM\s*\((.*?)\);"` (the `DO $$` wrapper in 026 does not matter; the `CREATE TYPE ml.gate_decision AS ENUM (...)` text is inside it).
- `enum_checks` gains:
```python
        (
            "ml.gate_decision",   # renamed to public.discovery_gate_decision by ml/036 (type OID unchanged)
            project_root / "database" / "ml" / "026_causal_discovery_tables.sql",
            "discovery_gate_decisions",
            "values",
        ),
```
- `validate_enum_sync(vocab_path: Path | None = None)`: read the YAML from `vocab_path or VOCAB_PATH` (introduce `VOCAB_PATH = project_root / "config" / "domain_vocabulary.yaml"` at module level if the function currently builds the path inline).
- Add a Python-side check after the SQL loop: import `DiscoveryGateDecision` and `GateDecision` (refutation) and compare `[m.value for m in enum]` to the YAML sections `discovery_gate_decisions` / `gate_decisions`; append to `errors` on mismatch.

- [ ] **Step 4: Run the tests and the script**

Run: `.venv/bin/pytest tests/unit/test_scripts/test_validate_vocabulary_enum_sync_1991.py -n 0 -q && .venv/bin/python scripts/validate_vocabulary_enum_sync.py`
Expected: 4 passed; script exits 0 and prints the new check among the passed ones.

- [ ] **Step 5: Commit**

```bash
git add config/domain_vocabulary.yaml scripts/validate_vocabulary_enum_sync.py tests/unit/test_scripts/test_validate_vocabulary_enum_sync_1991.py
git commit -m "feat(vocab): discovery gate vocabulary under the enum-sync guard; negative_control_outcome listed (#1991 debt 4)"
```

---

### Task 5: Lane close

- [ ] **Step 1:** ruff `--no-cache` check+format on every touched `.py`; scoped mypy on `src/api/schemas/causal.py`, `src/api/schemas/expert_review.py`, `src/api/routes/chatbot_tools.py`, `scripts/validate_vocabulary_enum_sync.py`.
- [ ] **Step 2:** `.venv/bin/pytest tests/unit/test_api tests/unit/test_causal_engine/test_discovery tests/unit/test_agents/test_causal_impact tests/unit/test_scripts tests/unit/test_database/test_migration_ml_036_discovery_to_public.py -n 0 -q -p no:cacheprovider` → pass; `cd frontend && npm run typecheck && npx vitest run && cd ..` → pass.
- [ ] **Step 3:** codex read-only round with the mandatory pushback paragraph; iterate to `VERDICT: ACCEPT`.
- [ ] **Step 4:** ONE push, PR `feat(gates): one vocabulary per gate — DiscoveryGateDecision, fail-closed chat summary, Literal wire types, enum-sync guard (#1991 debt 4)`; body states the expected `api.ts` diff and that no DB column changes.
- [ ] **Step 5:** Cert after deploy (spec §5): enum-sync green on the merge commit; live chat provenance on a Remibrutinib path with block rows reads `block`; on a path with no persisted rows the answer's provenance reads the unknown caveat (negative control: the pre-lane image reads `proceed` for the same input — capture BEFORE the merge with the scratch container, env built from `docker inspect e2i_api`, never `--env-file .env`). Record in `docs/demos/results/2026-09-12_lane_vocab/cert.md`; comment #1991.
