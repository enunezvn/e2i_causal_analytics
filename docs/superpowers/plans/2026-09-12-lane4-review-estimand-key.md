# Lane 4 — Expert reviews keyed on the estimand Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A review is one per estimand (brand, treatment, outcome), minted only by a REVIEW-band run, updated with a new structure version (and a DAG diff) instead of a sibling row when the structure changes, and the 37 no-op BLOCK-band pending rows are resolved as superseded.

**Architecture:** Migration 140 adds `estimand_key` and moves pending-uniqueness onto it; migration 141 adds `expert_review_versions` (one row per structure the estimand was reviewed under) and backfills one version per existing review. The gate keys its lookups on the estimand, appends versions on REVIEW re-runs, re-opens after an approval on a new hash, and never mints on BLOCK. The detail endpoint returns versions with `get_dag_changes` diffs; the frontend renders the delta.

**Tech Stack:** Postgres (migrations via `scripts/run_migrations.sh` on deploy; rehearsal via `docker exec -i supabase-db psql`), Supabase PostgREST client (`self.client.table(...)`), FastAPI/Pydantic, React/TS, pytest `-n 0`, vitest.

Spec: `docs/superpowers/specs/2026-09-12-debts-3-4-wave-design.md` §7. Worktree `.worktrees/lane-reviews`, branch `claude/1991-review-estimand-key`, ONE push at the end. Runs AFTER lane 3 (the refutation node is unchanged by lane 3, but `agent.py` in the new package carries the review fields — verify with `grep -n expert_review src/api/routes/causal/agent.py`).

```bash
cd /home/enunez/Projects/e2i_causal_analytics && git fetch origin
git worktree add -b claude/1991-review-estimand-key .worktrees/lane-reviews origin/main && cd .worktrees/lane-reviews
```

---

## File map

| file | change |
|---|---|
| `database/migrations/140_expert_reviews_estimand_key.sql` | new: `estimand_key`, backfill, index swap, guarded supersede of BLOCK-band pending rows |
| `database/migrations/141_expert_review_versions.sql` | new: versions table + backfill |
| `src/repositories/expert_review.py` | `estimand_key_for`, `create_review(estimand_key=…)`, `_find_pending_review_id` by estimand, `get_reviews_for_estimand`, `append_version`, `get_versions`, `get_pending_reviews` adds version fields, `get_review_summary` adds `superseded` |
| `src/causal_engine/expert_review_gate.py` | `check_approval` keys on the estimand, appends versions, re-opens after approval |
| `src/agents/causal_impact/nodes/refutation.py` | BLOCK band no longer mints |
| `src/causal_engine/dag_hash.py` | `get_dag_changes` gains `adjustment_sets_added/removed` |
| `src/api/routes/expert_review.py`, `src/api/schemas/expert_review.py` | versions + diffs on detail; `version_count`/`last_changed_at` on pending; `superseded` in summary |
| `frontend/src/types/expert-review.ts`, `components/expert-review/DagPanel.tsx`, `DagDiff.tsx` (new), `pages/ExpertReviews.tsx` | versions column, diff render, summary badge |
| tests: `tests/unit/test_database/test_migration_140_estimand_key.py`, `test_migration_141_review_versions.py`, `tests/unit/test_repositories/test_expert_review_estimand_1991.py`, `tests/unit/test_causal_engine/test_expert_review_gate_estimand_1991.py`, `tests/unit/test_agents/test_causal_impact/test_refutation_block_never_mints_1991.py`, `tests/unit/test_causal_engine/test_dag_hash_changes_1991.py`, `tests/unit/test_api/test_expert_review_versions_1991.py`, `frontend/src/components/expert-review/__tests__/DagDiff.test.tsx` |

---

### Task 1: Migration 140 — `estimand_key`, index swap, supersede the BLOCK-band rows

**Files:**
- Create: `database/migrations/140_expert_reviews_estimand_key.sql`
- Test: `tests/unit/test_database/test_migration_140_estimand_key.py`

- [ ] **Step 1: Write the failing content-lock test**

```python
"""Migration 140 content lock (#1991 debt 3): estimand_key, pending-uniqueness on the estimand,
and the guarded supersede of BLOCK-band pending rows."""
from __future__ import annotations

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
M = REPO / "database" / "migrations" / "140_expert_reviews_estimand_key.sql"


def _sql() -> str:
    return "\n".join(l for l in M.read_text().splitlines() if not l.strip().startswith("--"))


def test_adds_and_backfills_estimand_key():
    s = _sql()
    assert re.search(r"ADD COLUMN IF NOT EXISTS estimand_key TEXT", s)
    assert re.search(r"UPDATE public\.expert_reviews\s+SET estimand_key\s*=", s)
    assert "lower(" in s and "COALESCE(brand, '')" in s
    assert re.search(r"ALTER COLUMN estimand_key SET NOT NULL", s)


def test_swaps_the_pending_unique_index():
    s = _sql()
    assert "DROP INDEX IF EXISTS uq_er_pending_dag_brand" in s
    assert re.search(r"CREATE UNIQUE INDEX IF NOT EXISTS uq_er_pending_estimand\s+ON public\.expert_reviews \(estimand_key\)\s+WHERE approval_status = 'pending'", s)


def test_supersede_is_guarded_by_an_asserted_count():
    s = _sql()
    assert "approval_status = 'superseded'" in s
    assert "gate=block" in s
    assert "RAISE EXCEPTION" in s and "GET DIAGNOSTICS" in s
    assert "resolved_at = now()" in s


def test_no_delete_and_no_own_transaction():
    s = _sql().upper()
    assert "DELETE FROM" not in s
    assert "BEGIN;" not in s and "COMMIT;" not in s
```

- [ ] **Step 2: Run → FAIL (file missing)**

- [ ] **Step 3: Write the migration**

```sql
-- ============================================================================
-- Migration 140: expert_reviews keyed on the ESTIMAND (#1991 debt 3, lane 4)
-- ============================================================================
-- WHAT: (1) estimand_key TEXT = lower(brand):treatment:outcome, backfilled from the
--   three columns every row already carries, then NOT NULL; (2) pending uniqueness
--   moves from (dag_version_hash, brand) [migration 062] to (estimand_key), so a
--   covariate change on the same estimand UPDATES the pending review (a new
--   structure version, migration 141) instead of minting a sibling; (3) every
--   pending row minted by a BLOCK-band run is resolved as 'superseded' -- a BLOCK
--   run is terminal before the review is consulted, so approving it changes
--   nothing (measured 2026-09-12: 37 pending, all gate=block, 27 hashes, 27
--   estimands; no mint since 2026-09-08).
-- WHY the adjustment set is NOT in the key: a covariate change is the event that
--   must update a review, not mint one. The hash and adjustment set are the
--   VERSION (141), not the identity.
-- SAFETY: additive except the index swap; the supersede is a guarded UPDATE that
--   raises (rolling the whole file back under run_migrations.sh's
--   --single-transaction) unless it touches exactly the rows counted at the top
--   of the block. Nothing is deleted. dag_version_hash and its approval indexes
--   are untouched; approval lookups by hash stay valid.
-- ============================================================================

ALTER TABLE public.expert_reviews ADD COLUMN IF NOT EXISTS estimand_key TEXT;

UPDATE public.expert_reviews
SET estimand_key = lower(COALESCE(brand, '')) || ':' || lower(COALESCE(treatment_variable, ''))
                   || ':' || lower(COALESCE(outcome_variable, ''))
WHERE estimand_key IS NULL;

ALTER TABLE public.expert_reviews ALTER COLUMN estimand_key SET NOT NULL;

COMMENT ON COLUMN public.expert_reviews.estimand_key IS
    'Review identity (migration 140, #1991 debt 3): lower(brand):treatment:outcome. '
    'dag_version_hash is the structure VERSION the review currently covers, not its identity.';

-- Supersede the BLOCK-band pending rows BEFORE the new uniqueness lands (several
-- share an estimand -- 4 for Remibrutinib treatment_arm -> persistent_180d).
DO $$
DECLARE
    v_expected INTEGER;
    v_done INTEGER;
BEGIN
    SELECT count(*) INTO v_expected FROM public.expert_reviews
     WHERE approval_status = 'pending' AND analysis_context LIKE '%gate=block%';

    UPDATE public.expert_reviews
       SET approval_status = 'superseded',
           resolved_at = now(),
           comments_json = COALESCE(comments_json, '{}'::jsonb)
               || jsonb_build_object('superseded_reason',
                  'block-band review: the run was terminal before the review was consulted, '
                  'so an approval could not change an outcome (#1991 debt 3, migration 140)')
     WHERE approval_status = 'pending' AND analysis_context LIKE '%gate=block%';
    GET DIAGNOSTICS v_done = ROW_COUNT;

    IF v_done <> v_expected THEN
        RAISE EXCEPTION 'migration 140: superseded % rows, expected %', v_done, v_expected;
    END IF;
END $$;

DROP INDEX IF EXISTS uq_er_pending_dag_brand;

CREATE UNIQUE INDEX IF NOT EXISTS uq_er_pending_estimand
    ON public.expert_reviews (estimand_key)
    WHERE approval_status = 'pending';

CREATE INDEX IF NOT EXISTS idx_er_estimand_created
    ON public.expert_reviews (estimand_key, created_at DESC);

COMMENT ON INDEX uq_er_pending_estimand IS
    'At most one PENDING expert review per estimand (migration 140); a new structure on '
    'the same estimand appends a version (expert_review_versions, migration 141).';
```

- [ ] **Step 4: Run the test → 4 passed. Rehearse on the live DB**

Run: `{ echo "BEGIN;"; cat database/migrations/140_expert_reviews_estimand_key.sql; echo "SELECT approval_status, count(*) FROM public.expert_reviews GROUP BY 1 ORDER BY 1; SELECT count(*) FILTER (WHERE estimand_key IS NULL) FROM public.expert_reviews; ROLLBACK;"; } | docker exec -i supabase-db psql -U postgres -d postgres -v ON_ERROR_STOP=1`
Expected: `approved 1 / rejected 2 / superseded 37 / pending 0`, null-count 0, then ROLLBACK. Record in `docs/demos/results/2026-09-12_lane_reviews/rehearsal_140.txt`.

- [ ] **Step 5: Commit**

---

### Task 2: Migration 141 — `expert_review_versions`

**Files:**
- Create: `database/migrations/141_expert_review_versions.sql`
- Test: `tests/unit/test_database/test_migration_141_review_versions.py`

- [ ] **Step 1: Write the failing content-lock test**

```python
from __future__ import annotations
import re
from pathlib import Path
REPO = Path(__file__).resolve().parents[3]
M = REPO / "database" / "migrations" / "141_expert_review_versions.sql"

def _sql():
    return "\n".join(l for l in M.read_text().splitlines() if not l.strip().startswith("--"))

def test_creates_versions_table_with_fk_and_columns():
    s = _sql()
    assert re.search(r"CREATE TABLE IF NOT EXISTS public\.expert_review_versions", s)
    for col in ("version_id UUID", "review_id UUID NOT NULL REFERENCES public.expert_reviews(review_id)",
                "dag_version_hash VARCHAR(64) NOT NULL", "adjustment_set_hash VARCHAR(64)",
                "dag_structure_json JSONB", "query_id TEXT", "created_at TIMESTAMPTZ NOT NULL DEFAULT now()"):
        assert col in s, col
    assert "UNIQUE (review_id, dag_version_hash)" in s

def test_backfills_one_version_per_existing_review():
    s = _sql()
    assert re.search(r"INSERT INTO public\.expert_review_versions", s)
    assert "SELECT" in s and "FROM public.expert_reviews" in s
    assert "ON CONFLICT (review_id, dag_version_hash) DO NOTHING" in s

def test_grants_match_expert_reviews_pattern():
    s = _sql()
    assert "REVOKE ALL ON public.expert_review_versions FROM PUBLIC, anon, authenticated" in s
    assert "GRANT SELECT, INSERT ON public.expert_review_versions TO service_role" in s
```

- [ ] **Step 2: Run → FAIL. Step 3: Write the migration**

```sql
-- ============================================================================
-- Migration 141: expert_review_versions (#1991 debt 3, lane 4)
-- ============================================================================
-- One row per DAG structure an estimand's review has covered. A REVIEW-band run
-- on an estimand with a pending review APPENDS here when its hash differs from
-- the latest version, and the API renders the diff between consecutive versions
-- (src/causal_engine/dag_hash.py get_dag_changes). Backfill: one version per
-- existing review from its current hash and snapshot, so history starts full.
-- ============================================================================

CREATE TABLE IF NOT EXISTS public.expert_review_versions (
    version_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    review_id UUID NOT NULL REFERENCES public.expert_reviews(review_id) ON DELETE CASCADE,
    dag_version_hash VARCHAR(64) NOT NULL,
    adjustment_set_hash VARCHAR(64),
    dag_structure_json JSONB,
    query_id TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (review_id, dag_version_hash)
);

CREATE INDEX IF NOT EXISTS idx_erv_review_created
    ON public.expert_review_versions (review_id, created_at);

INSERT INTO public.expert_review_versions (review_id, dag_version_hash, dag_structure_json, query_id, created_at)
SELECT review_id, dag_version_hash, dag_structure_json, reviewer_id, created_at
  FROM public.expert_reviews
 WHERE dag_version_hash IS NOT NULL
ON CONFLICT (review_id, dag_version_hash) DO NOTHING;

REVOKE ALL ON public.expert_review_versions FROM PUBLIC, anon, authenticated;
GRANT SELECT, INSERT ON public.expert_review_versions TO service_role;

COMMENT ON TABLE public.expert_review_versions IS
    'Structure versions an expert review has covered (migration 141, #1991 debt 3): the diff '
    'between consecutive rows is what a reviewer sees when a re-run changes the DAG.';
```
(`reviewer_id` holds the REQUESTER query id — see migration 136's note — so it is the right `query_id` for the backfill.)

- [ ] **Step 4: Test → pass; rehearse 140+141 together in BEGIN…ROLLBACK**: `{ echo BEGIN\;; cat database/migrations/140_*.sql database/migrations/141_*.sql; echo "SELECT (SELECT count(*) FROM public.expert_reviews) reviews, (SELECT count(*) FROM public.expert_review_versions) versions; ROLLBACK;"; } | docker exec -i supabase-db psql -U postgres -d postgres -v ON_ERROR_STOP=1` → `reviews == versions` (40 = 40 on 2026-09-12). Record. Commit.

---

### Task 3: Repository — estimand key, versions, history, summary

**Files:**
- Modify: `src/repositories/expert_review.py`
- Test: `tests/unit/test_repositories/test_expert_review_estimand_1991.py`

- [ ] **Step 1: Write the failing tests** (the file's existing tests use a fake PostgREST client; mirror `tests/unit/test_repositories/test_expert_review.py`'s fixture — copy its `_FakeClient` / table builder)

```python
"""#1991 debt 3: reviews are keyed on the estimand; versions append; summary counts superseded."""
from __future__ import annotations

import pytest

from src.repositories.expert_review import ExpertReviewRepository, estimand_key_for


def test_estimand_key_is_lowercase_and_null_safe():
    assert estimand_key_for("Remibrutinib", "treatment_arm", "persistent_180d") == "remibrutinib:treatment_arm:persistent_180d"
    assert estimand_key_for(None, "T", "Y") == ":t:y"


@pytest.mark.asyncio
async def test_create_review_writes_estimand_key(fake_client):
    repo = ExpertReviewRepository(client=fake_client)
    await repo.create_review(reviewer_id="q1", review_type="dag_approval", dag_version_hash="h1",
                             brand="B", treatment_variable="T", outcome_variable="Y")
    row = fake_client.inserted("expert_reviews")[0]
    assert row["estimand_key"] == "b:t:y"


@pytest.mark.asyncio
async def test_append_version_inserts_and_updates_current_hash(fake_client):
    repo = ExpertReviewRepository(client=fake_client)
    ok = await repo.append_version("r1", dag_version_hash="h2", dag_structure={"nodes": ["a"], "edges": []},
                                   adjustment_set_hash="a2", query_id="q2")
    assert ok is True
    v = fake_client.inserted("expert_review_versions")[0]
    assert v["review_id"] == "r1" and v["dag_version_hash"] == "h2"
    upd = fake_client.updated("expert_reviews")[0]
    assert upd["dag_version_hash"] == "h2" and "dag_structure_json" in upd


@pytest.mark.asyncio
async def test_summary_counts_superseded(fake_client):
    fake_client.seed("expert_reviews", [
        {"approval_status": "pending", "valid_until": None},
        {"approval_status": "superseded", "valid_until": None},
        {"approval_status": "superseded", "valid_until": None},
    ])
    repo = ExpertReviewRepository(client=fake_client)
    s = await repo.get_review_summary()
    assert s["pending"] == 1 and s["superseded"] == 2
```

- [ ] **Step 2: Run → FAIL (`estimand_key_for` missing).**

- [ ] **Step 3: Implement**

Module level:
```python
def estimand_key_for(brand: Optional[str], treatment: Optional[str], outcome: Optional[str]) -> str:
    """Review identity (migration 140): lower(brand):treatment:outcome, null-safe."""
    return f"{(brand or '').lower()}:{(treatment or '').lower()}:{(outcome or '').lower()}"
```
`create_review`: add `"estimand_key": estimand_key_for(brand, treatment_variable, outcome_variable),` to `row`; the unique-violation recovery calls `_find_pending_review_id(estimand_key=...)`.
`_find_pending_review_id(self, estimand_key: str)`: `.select("review_id").eq("estimand_key", estimand_key).eq("approval_status", "pending").limit(1)`.
New:
```python
    async def get_reviews_for_estimand(self, estimand_key: str, include_expired: bool = True) -> List[Dict[str, Any]]:
        """Every review of this estimand, newest first (the history the gate and the detail route read)."""
        if not self.client:
            return []
        query = self.client.table(self.table_name).select("*").eq("estimand_key", estimand_key).order("created_at", desc=True)
        if not include_expired:
            query = _apply_active_validity(query)
        result = await query.execute()
        return result.data or []

    async def append_version(self, review_id: str, *, dag_version_hash: str, dag_structure: Optional[Dict[str, Any]],
                             adjustment_set_hash: Optional[str], query_id: Optional[str]) -> bool:
        """Record a new structure version on a pending review and make it the review's current hash."""
        if not self.client:
            return False
        await self.client.table("expert_review_versions").insert({
            "review_id": review_id, "dag_version_hash": dag_version_hash,
            "dag_structure_json": to_plain_json(dag_structure) if dag_structure else None,
            "adjustment_set_hash": adjustment_set_hash, "query_id": query_id,
        }).execute()
        await (self.client.table(self.table_name)
               .update({"dag_version_hash": dag_version_hash,
                        "dag_structure_json": to_plain_json(dag_structure) if dag_structure else None})
               .eq("review_id", review_id).eq("approval_status", "pending").execute())
        return True

    async def get_versions(self, review_id: str) -> List[Dict[str, Any]]:
        if not self.client:
            return []
        result = await (self.client.table("expert_review_versions").select("*")
                        .eq("review_id", review_id).order("created_at", desc=False).execute())
        return result.data or []
```
`get_review_summary`: add `superseded = 0`, count `elif status == "superseded": superseded += 1`, return it. `get_pending_reviews`: unchanged query; the route computes `version_count` from `get_versions` (Task 5).

- [ ] **Step 4: Run new + existing repo tests** `tests/unit/test_repositories/test_expert_review*.py -n 0 -q` → pass. Commit.

---

### Task 4: The gate keys on the estimand; BLOCK never mints

**Files:**
- Modify: `src/causal_engine/expert_review_gate.py` (`check_approval`), `src/agents/causal_impact/nodes/refutation.py` (BLOCK branch ~2229–2245)
- Test: `tests/unit/test_causal_engine/test_expert_review_gate_estimand_1991.py`, `tests/unit/test_agents/test_causal_impact/test_refutation_block_never_mints_1991.py`

- [ ] **Step 1: Write the failing gate tests** (reuse `_CapturingRepo` from `tests/unit/test_causal_engine/test_expert_review_gate.py`, extended with `append_version`, `get_reviews_for_estimand`, `get_versions` capture lists)

```python
@pytest.mark.asyncio
async def test_review_band_new_estimand_mints_once():
    repo = _EstimandRepo(history=[])
    gate = ExpertReviewGate(repository=repo, auto_create_review=True)
    r = await gate.check_approval(dag_hash="h1", brand="B", treatment="T", outcome="Y", requester_id="q",
                                  dag_structure={"nodes": ["T", "Y"], "edges": [["T", "Y"]]})
    assert r.decision == ReviewGateDecision.PENDING_REVIEW
    assert len(repo.created) == 1 and repo.created[0]["estimand_key"] == "b:t:y"


@pytest.mark.asyncio
async def test_review_band_same_estimand_new_hash_appends_version_not_row():
    repo = _EstimandRepo(history=[{"review_id": "r1", "approval_status": "pending", "dag_version_hash": "h1",
                                   "estimand_key": "b:t:y", "created_at": "2026-09-01T00:00:00+00:00"}])
    gate = ExpertReviewGate(repository=repo, auto_create_review=True)
    r = await gate.check_approval(dag_hash="h2", brand="B", treatment="T", outcome="Y", requester_id="q",
                                  dag_structure={"nodes": ["T", "Y", "W"], "edges": [["W", "T"], ["T", "Y"]]})
    assert r.decision == ReviewGateDecision.PENDING_REVIEW and r.review_id == "r1"
    assert repo.created == [] and repo.appended == [("r1", "h2")]


@pytest.mark.asyncio
async def test_same_hash_appends_nothing():
    repo = _EstimandRepo(history=[{"review_id": "r1", "approval_status": "pending", "dag_version_hash": "h1",
                                   "estimand_key": "b:t:y", "created_at": "2026-09-01T00:00:00+00:00"}])
    gate = ExpertReviewGate(repository=repo, auto_create_review=True)
    await gate.check_approval(dag_hash="h1", brand="B", treatment="T", outcome="Y", requester_id="q")
    assert repo.appended == [] and repo.created == []


@pytest.mark.asyncio
async def test_approved_estimand_on_new_hash_reopens_with_supersedes():
    repo = _EstimandRepo(history=[{"review_id": "r0", "approval_status": "approved", "dag_version_hash": "h1",
                                   "estimand_key": "b:t:y", "valid_until": "2099-01-01",
                                   "created_at": "2026-09-01T00:00:00+00:00"}])
    gate = ExpertReviewGate(repository=repo, auto_create_review=True)
    r = await gate.check_approval(dag_hash="h2", brand="B", treatment="T", outcome="Y", requester_id="q")
    assert r.decision == ReviewGateDecision.PENDING_REVIEW
    assert repo.created[0]["supersedes_review_id"] == "r0"


@pytest.mark.asyncio
async def test_rejection_on_the_estimand_is_still_authoritative():
    repo = _EstimandRepo(history=[{"review_id": "r0", "approval_status": "rejected", "dag_version_hash": "h1",
                                   "estimand_key": "b:t:y", "created_at": "2026-09-01T00:00:00+00:00"}])
    gate = ExpertReviewGate(repository=repo, auto_create_review=True)
    r = await gate.check_approval(dag_hash="h1", brand="B", treatment="T", outcome="Y", requester_id="q")
    assert r.decision == ReviewGateDecision.REJECTED and repo.created == []
```
And the node test:
```python
@pytest.mark.asyncio
async def test_block_band_never_consults_or_mints(monkeypatch):
    """A BLOCK run is terminal before the review is consulted (#1991 debt 3)."""
    node = _node_with_block_suite()  # reuse the BLOCK fixture from test_refutation.py
    consulted = []
    monkeypatch.setattr(node, "_consult_review_gate", lambda *a, **k: consulted.append(1) or {})
    result = await node.execute(_state())
    assert consulted == []
    assert result["status"] == "failed" and result.get("expert_review_id") is None
```

- [ ] **Step 2: Run → FAIL.**

- [ ] **Step 3: Implement the gate change**

In `check_approval`, replace the history read:
```python
        estimand_key = estimand_key_for(brand, treatment, outcome)
        history = await self.repository.get_reviews_for_estimand(estimand_key, include_expired=True)
```
(import `estimand_key_for` from `src.repositories.expert_review`; keep `get_reviews_for_dag` for `check_rejection`, which stays hash-keyed — a rejection is recorded against a version's hash). Chronology (`_latest_adjudication`) and the rejection short-circuit are unchanged. Approval lookup: `approval = None if superseded_by_rejection else next((r for r in history if r.get("approval_status") == "approved" and r.get("dag_version_hash") == dag_hash and approval_validity(r, date.today()) != "expired"), None)` — approval is scoped to the hash it was given on. If there is a valid approval on a DIFFERENT hash and no pending row: fall through to minting with `supersedes_review_id=<that approval's review_id>` (add the parameter to `create_review`'s row; it is an existing column).
Pending branch: if `pending` and `pending_row["dag_version_hash"] != dag_hash` → `await self.repository.append_version(review_id, dag_version_hash=dag_hash, dag_structure=sanitize_dag_structure(dag_structure), adjustment_set_hash=compute_adjustment_set_hash((dag_structure or {}).get("adjustment_sets") or []), query_id=requester_id)`; then return `PENDING_REVIEW` with that review id. Same hash → return `PENDING_REVIEW` as today (the 097 backfill branch stays).
Minting: after `create_review(...)` returns an id, call `append_version(...)` for the first version (so every review has version 1; the backfill covers old rows).

Node: in the BLOCK branch of `nodes/refutation.py` replace `review_fields = await self._review_fields_for_band(state, suite, validation_ids, rejection)` with:
```python
                # #1991 debt 3: a BLOCK run is terminal here; a review of it could not
                # change an outcome, so nothing is queued. A rejection already observed
                # by the probe is still surfaced (durable, #1970).
                review_fields = (
                    self._review_fields(suite, ReviewGateDecision.REJECTED.value, rejection)
                    if rejection is not None
                    else {}
                )
```
Update the BLOCK docstring/comment ("Route BLOCKED … to the expert-review queue too") accordingly.

- [ ] **Step 4: Run** `tests/unit/test_causal_engine/test_expert_review_gate*.py tests/unit/test_agents/test_causal_impact/test_refutation*.py tests/unit/test_api/test_causal_agent_analyze_expert_review_1971.py -n 0 -q` → pass (update any existing test asserting a BLOCK-band queue row: it now asserts none). Commit.

---

### Task 5: Diffs and the API surface

**Files:**
- Modify: `src/causal_engine/dag_hash.py` (`get_dag_changes`), `src/api/schemas/expert_review.py`, `src/api/routes/expert_review.py`
- Test: `tests/unit/test_causal_engine/test_dag_hash_changes_1991.py`, `tests/unit/test_api/test_expert_review_versions_1991.py`

- [ ] **Step 1: Failing tests**

```python
def test_get_dag_changes_reports_adjustment_set_delta():
    from src.causal_engine.dag_hash import get_dag_changes
    old = {"nodes": ["T", "Y"], "edges": [["T", "Y"]], "adjustment_sets": [["W"]]}
    new = {"nodes": ["T", "Y", "W"], "edges": [["T", "Y"], ["W", "T"]], "adjustment_sets": [["W", "Z"]]}
    d = get_dag_changes(old, new)
    assert d["nodes_added"] == ["W"] and d["edges_added"] == [["W", "T"]]
    assert d["adjustment_sets_added"] == [["W", "Z"]] and d["adjustment_sets_removed"] == [["W"]]
```
API test (TestClient with the repo dependency overridden, mirroring `tests/unit/test_api/test_expert_review*.py`): `GET /api/expert-reviews/{id}` returns `versions: [{version_id, dag_version_hash, created_at, changes}]` where `versions[0].changes is None` and `versions[1].changes.nodes_added == ["W"]`; `history` entries carry `changes_from_previous`; `GET /pending` items carry `version_count` and `last_changed_at`; `GET /summary` carries `superseded`.

- [ ] **Step 2: Run → FAIL. Step 3: Implement**

`get_dag_changes`: add
```python
    old_adj = {tuple(sorted(a)) for a in old_graph.get("adjustment_sets") or []}
    new_adj = {tuple(sorted(a)) for a in new_graph.get("adjustment_sets") or []}
    ...
        "adjustment_sets_added": [list(a) for a in sorted(new_adj - old_adj)],
        "adjustment_sets_removed": [list(a) for a in sorted(old_adj - new_adj)],
```
and fold it into `is_changed`.
Schemas: `class DagChanges(BaseModel)` (six lists + `is_changed: bool`), `class ReviewVersion(BaseModel)` (`version_id, dag_version_hash, adjustment_set_hash, dag_structure_json: Optional[DagStructureSnapshot], query_id, created_at, changes: Optional[DagChanges]`), `ExpertReviewDetailResponse` gains `versions: List[ReviewVersion]` and `ReviewRecord` gains `changes_from_previous: Optional[DagChanges] = None`; `PendingReviewItem` gains `version_count: int = 1`, `last_changed_at: Optional[datetime] = None`; `ReviewSummaryResponse` gains `superseded: int`.
Route `get_expert_review`: `history_rows = await repo.get_reviews_for_estimand(row["estimand_key"])` (fallback to the hash read when `estimand_key` is absent); `versions = await repo.get_versions(canonical_id)`; compute `changes` for each version after the first with `get_dag_changes(prev["dag_structure_json"] or {}, cur["dag_structure_json"] or {})`; for `history` (newest first) compute `changes_from_previous` between each row's `dag_structure_json` and the next-older row's. Route `list_pending_reviews`: for each row `versions = await repo.get_versions(row["review_id"])`, set `version_count=len(versions) or 1`, `last_changed_at=versions[-1]["created_at"] if versions else row["created_at"]`. Summary: pass `superseded=summary.get("superseded", 0)`.

- [ ] **Step 4: Run the API + engine tests → pass. Regenerate `api.ts`** (`python -m scripts.export_openapi …; npx openapi-typescript …`), expected diff limited to the expert-review schemas. Commit.

---

### Task 6: Frontend — version count, diff panel, superseded badge

**Files:**
- Modify: `frontend/src/types/expert-review.ts`, `frontend/src/pages/ExpertReviews.tsx`, `frontend/src/components/expert-review/DagPanel.tsx`
- Create: `frontend/src/components/expert-review/DagDiff.tsx`, `frontend/src/components/expert-review/__tests__/DagDiff.test.tsx`

- [ ] **Step 1: Failing vitest**

```tsx
import { render, screen } from '@testing-library/react';
import { describe, expect, it } from 'vitest';
import { DagDiff } from '../DagDiff';

describe('DagDiff', () => {
  it('lists added and removed nodes, edges and adjustment sets', () => {
    render(<DagDiff changes={{ nodes_added: ['W'], nodes_removed: [], edges_added: [['W', 'T']], edges_removed: [],
      adjustment_sets_added: [['W', 'Z']], adjustment_sets_removed: [['W']], is_changed: true }} />);
    expect(screen.getByText('+ W')).toBeInTheDocument();
    expect(screen.getByText('+ W → T')).toBeInTheDocument();
    expect(screen.getByText('+ {W, Z}')).toBeInTheDocument();
    expect(screen.getByText('− {W}')).toBeInTheDocument();
  });
  it('says so when nothing changed', () => {
    render(<DagDiff changes={{ nodes_added: [], nodes_removed: [], edges_added: [], edges_removed: [],
      adjustment_sets_added: [], adjustment_sets_removed: [], is_changed: false }} />);
    expect(screen.getByText('No structural change from the previous version.')).toBeInTheDocument();
  });
});
```

- [ ] **Step 2: Run → FAIL. Step 3: Implement**

Types: `export interface DagChanges { nodes_added: string[]; nodes_removed: string[]; edges_added: string[][]; edges_removed: string[][]; adjustment_sets_added: string[][]; adjustment_sets_removed: string[][]; is_changed: boolean; }`, `export interface ReviewVersion { version_id: string; dag_version_hash: string; adjustment_set_hash?: string | null; dag_structure_json?: DagStructure | null; query_id?: string | null; created_at: string; changes?: DagChanges | null; }`; `PendingReviewItem` gains `version_count?: number; last_changed_at?: string | null;`; `ReviewRecord` gains `changes_from_previous?: DagChanges | null;`; `ExpertReviewDetailResponse` gains `versions: ReviewVersion[]`; `ReviewSummaryResponse` gains `superseded: number`.
`DagDiff.tsx`:
```tsx
import type { DagChanges } from '@/types/expert-review';

const edge = ([s, t]: string[]) => `${s} → ${t}`;
const set = (a: string[]) => `{${a.join(', ')}}`;

export function DagDiff({ changes }: { changes: DagChanges }) {
  if (!changes.is_changed) {
    return <p className="text-xs text-[var(--color-muted-foreground)]">No structural change from the previous version.</p>;
  }
  const rows: Array<[string, string[]]> = [
    ['Nodes', [...changes.nodes_added.map((n) => `+ ${n}`), ...changes.nodes_removed.map((n) => `− ${n}`)]],
    ['Edges', [...changes.edges_added.map((e) => `+ ${edge(e)}`), ...changes.edges_removed.map((e) => `− ${edge(e)}`)]],
    ['Adjustment sets', [...changes.adjustment_sets_added.map((a) => `+ ${set(a)}`), ...changes.adjustment_sets_removed.map((a) => `− ${set(a)}`)]],
  ];
  return (
    <dl className="grid gap-1 text-xs">
      {rows.filter(([, items]) => items.length > 0).map(([label, items]) => (
        <div key={label}>
          <dt className="font-medium">{label}</dt>
          {items.map((it) => <dd key={it} className="ml-2 font-mono">{it}</dd>)}
        </div>
      ))}
    </dl>
  );
}
```
`DagPanel.tsx`: accept `versions?: ReviewVersion[]`; when `versions.length > 1` render `<DagDiff changes={versions[versions.length - 1].changes!} />` under the graph with the heading "Changed since the previous version". `ExpertReviews.tsx`: a "Versions" column (`review.version_count ?? 1`), the summary badge `Superseded: {summary.data.superseded}`, and pass `versions` to `DagPanel` from the detail hook when the row is expanded (the pending list has no versions; the expanded row fetches `useExpertReview(review.review_id)`).

- [ ] **Step 4: `cd frontend && npm run typecheck && npx vitest run src/components/expert-review && cd ..` → pass. Commit.**

---

### Task 7: Lane close and cert

- [ ] **Step 1:** ruff `--no-cache`, scoped mypy on the four touched `src` modules; `.venv/bin/pytest tests/unit/test_causal_engine/test_expert_review_gate*.py tests/unit/test_causal_engine/test_dag_hash*.py tests/unit/test_repositories/test_expert_review*.py tests/unit/test_api/test_expert_review*.py tests/unit/test_api/test_causal_agent_analyze*.py tests/unit/test_agents/test_causal_impact tests/unit/test_database/test_migration_14*.py -n 0 -q -p no:cacheprovider` → pass; frontend typecheck + vitest → pass.
- [ ] **Step 2:** codex read-only round with the mandatory pushback paragraph → `VERDICT: ACCEPT`.
- [ ] **Step 3:** ONE push; PR `feat(expert-review): reviews keyed on the estimand, structure versions with diffs, BLOCK never mints (#1991 debt 3)`; body carries the 140/141 rehearsal readings (37 superseded, 40 = 40 versions) and the spec's key deviation note.
- [ ] **Step 4:** Cert after deploy (spec §7): ledger shows 140 and 141; `select approval_status, count(*) from public.expert_reviews group by 1` → pending 0 / superseded 37 / rejected 2 / approved 1; `select count(*) from public.expert_review_versions` == review count; an 11-pair Remibrutinib discovery mints nothing (pending stays 0); `GET /api/expert-reviews/{id}` for one of the four superseded `treatment_arm → persistent_180d` reviews (operator token) returns `versions` (1) and `history` (4 rows) with `changes_from_previous` non-null between different hashes. Negative control: the lane's node test proves the BLOCK mint is gone; the pre-deploy pending count (37) versus post (0) is the live reading. Record in `docs/demos/results/2026-09-12_lane_reviews/cert.md`; comment #1991 with the debt-3 close-out.
