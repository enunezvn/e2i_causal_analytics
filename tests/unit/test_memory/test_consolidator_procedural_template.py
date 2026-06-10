"""Unit tests for procedural-template extraction (issue #389).

These tests pin the V1 contract for ``Consolidator.extract_procedural_templates``
and the pure helper ``_compute_template_signature``.

Design (binding decisions from the #389 brief — mirror the Decision 2 = HYBRID
shape from PR #384):

* **Clustering basis (V1)**: exact-match key-tuples
  ``(brand, event_type, event_subtype, sorted(action_keys))`` — NOT
  embedding similarity. Mirror the cluster-grouping shape used by
  ``_promote_to_semantic`` at consolidator.py:825 — i.e. ``defaultdict(list)``
  keyed by a deterministic tuple.
* **Template body format**: Pydantic schema with ``Optional[str]`` variable
  placeholders + concrete ``shared_action_keys`` (intersection across cluster
  rows). NOT Jinja2 templates (cross-language fragility) and NOT free-form
  ``{var}`` text. Serialized to JSONB at the persistence boundary.
* **Confidence scoring**: cluster cohesion = mean pairwise Jaccard similarity
  over per-row ``action_keys`` sets (deterministic, in [0..1]). When the LLM
  path is active, the symbolic confidence is multiplied by an LLM-rated
  coherence in [0..1]. Documented in code comments.
* **LLM cost gating**: feature flag ``PROCEDURAL_LLM_EXTRACTION_ENABLED``
  (env var, default false). Symbolic path ALWAYS runs first; LLM augments
  confidence when flag is on.
* **Brand boundary preserved** — no cross-brand templates.
* **No revision/versioning in V1** — extract once per cluster.

Out of scope (V2 follow-ups, filed separately):
* Embedding-similarity clustering.
* Template revision/versioning.
* Cross-brand templates.

Forbidden patterns (per [[feedback-test-must-exercise-real-catch-not-mock]]
and [[feedback-codex-audits-within-existing-signature-not-design]]):

* ``monkeypatch.setattr(anthropic_mod, "AsyncAnthropic", ...)`` — bypasses
  the production catch surface and is xdist-fragile.
* ``monkeypatch.setattr(<prod_module>, "<function_under_test>", stub)`` —
  bypasses the real catch boundary.

Instead: inject a fake ``client_factory`` callable via the ``client_factory``
parameter (the parameter-DI pattern adopted by PR #384's narrator refactor).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest

from src.memory.lifecycle.consolidator import (
    PROCEDURAL_LLM_EXTRACTION_ENV_VAR,
    Consolidator,
    ProceduralTemplate,
    _compute_template_signature,
    _jaccard_cohesion,
    _llm_extraction_enabled,
)

# ---------------------------------------------------------------------------
# Fake supabase that supports the procedural-template persistence surface
# (select on episodic_memories for cluster source, insert + on-conflict on
# procedural_templates). Reuses the shape of test_consolidator_dedup.py's
# FakeSupabase but adds the procedural_templates table and ``insert``
# support with partial-unique-index emulation.
# ---------------------------------------------------------------------------


class _UniqueViolationStub(Exception):
    """Stand-in for ``psycopg.errors.UniqueViolation`` shape — both class
    name AND message include 'unique' + 'index' so the consolidator's
    :meth:`_is_unique_violation` detector accepts it."""

    def __str__(self) -> str:  # pragma: no cover - trivial
        return "duplicate key value violates unique index uix_procedural_templates_signature"


class _FakeQuery:
    def __init__(self, store: "FakeSupabase", table: str) -> None:
        self.store = store
        self.table_name = table
        self._select_cols: Optional[str] = None
        self._filters: Dict[str, Any] = {}
        self._is_null_cols: List[str] = []
        self._gte: Dict[str, Any] = {}
        self._update_payload: Dict[str, Any] = {}
        self._insert_payload: Optional[Dict[str, Any]] = None
        self._mode: Optional[str] = None  # 'select' | 'update' | 'insert' | 'delete'
        self._in_filters: Dict[str, List[Any]] = {}
        self._range: Optional[tuple] = None  # (start, end) inclusive, PostgREST-style

    def select(self, cols: str, count: Optional[str] = None) -> "_FakeQuery":
        self._mode = "select"
        self._select_cols = cols
        return self

    def update(self, payload: Dict[str, Any]) -> "_FakeQuery":
        self._mode = "update"
        self._update_payload = payload
        return self

    def insert(self, payload: Dict[str, Any]) -> "_FakeQuery":
        self._mode = "insert"
        self._insert_payload = payload
        return self

    def delete(self) -> "_FakeQuery":
        self._mode = "delete"
        return self

    def eq(self, col: str, val: Any) -> "_FakeQuery":
        self._filters[col] = val
        return self

    def gte(self, col: str, val: Any) -> "_FakeQuery":
        self._gte[col] = val
        return self

    def is_(self, col: str, val: Any) -> "_FakeQuery":
        if val == "null" or val is None:
            self._is_null_cols.append(col)
        else:
            raise NotImplementedError(
                f"FakeSupabase.is_({col}, {val!r}) — only 'null'/None values supported; "
                "raise to prevent silent test-only divergence from PostgREST semantics"
            )
        return self

    def in_(self, col: str, vals: List[Any]) -> "_FakeQuery":
        self._in_filters[col] = list(vals)
        return self

    def range(self, start: int, end: int) -> "_FakeQuery":
        self._range = (start, end)
        return self

    def _match(self) -> List[Dict[str, Any]]:
        rows = list(self.store.rows.get(self.table_name, []))
        for col, want in self._filters.items():
            # Model the schema default: is_synthetic is NOT NULL DEFAULT false
            # (migration 063), so a seeded row that omits it reads as False.
            if col == "is_synthetic":
                rows = [r for r in rows if r.get(col, False) == want]
            else:
                rows = [r for r in rows if r.get(col) == want]
        for col, threshold in self._gte.items():
            rows = [r for r in rows if (r.get(col) or 0) >= threshold]
        for col in self._is_null_cols:
            rows = [r for r in rows if r.get(col) is None]
        for col, vals in self._in_filters.items():
            rows = [r for r in rows if r.get(col) in vals]
        return rows

    def execute(self) -> MagicMock:
        rows = self._match()
        if self._range is not None:
            start, end = self._range
            rows = rows[start : end + 1]
        if self._mode == "update":
            for r in rows:
                for orig in self.store.rows[self.table_name]:
                    if orig is r:
                        orig.update(self._update_payload)
                        break
        elif self._mode == "delete":
            keep = [r for r in self.store.rows[self.table_name] if r not in rows]
            self.store.rows[self.table_name] = keep
            rows = []
        elif self._mode == "insert":
            payload = self._insert_payload or {}
            # Emulate the partial-unique-index on
            # (COALESCE(brand,''), template_signature) WHERE
            # template_signature IS NOT NULL: if a row already exists
            # with the same (brand, signature) AND signature is not None,
            # raise the stand-in.
            sig = payload.get("template_signature")
            brand = payload.get("brand")
            if sig is not None and self.table_name == "procedural_templates":
                for existing in self.store.rows.get(self.table_name, []):
                    if existing.get("template_signature") == sig and existing.get("brand") == brand:
                        raise _UniqueViolationStub()
            self.store.rows.setdefault(self.table_name, []).append(dict(payload))
            rows = [dict(payload)]
        mock = MagicMock()
        mock.data = rows
        return mock


class FakeSupabase:
    def __init__(self) -> None:
        self.rows: Dict[str, List[Dict[str, Any]]] = {
            "causal_paths": [],
            "episodic_memories": [],
            "procedural_memories": [],
            "procedural_templates": [],
        }

    def table(self, name: str) -> _FakeQuery:
        return _FakeQuery(self, name)


@pytest.fixture
def fake_supabase() -> FakeSupabase:
    return FakeSupabase()


@pytest.fixture(autouse=True)
def patch_client(fake_supabase: FakeSupabase):
    with patch(
        "src.memory.lifecycle.consolidator.get_supabase_client",
        return_value=fake_supabase,
    ):
        yield


# Make sure the flag default-off behavior holds regardless of host env.
@pytest.fixture(autouse=True)
def flag_off_by_default(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv(PROCEDURAL_LLM_EXTRACTION_ENV_VAR, raising=False)


# ---------------------------------------------------------------------------
# _compute_template_signature contract
# ---------------------------------------------------------------------------


def test_compute_template_signature_includes_brand_event_subtype_action_keys() -> None:
    """Signature must be a deterministic hash over the four cluster-key
    fields. Same input → same signature."""
    sig_a = _compute_template_signature(
        brand="Kisqali",
        event_type="agent_action",
        event_subtype="ate_estimation",
        action_keys=["plan_run", "estimate", "refute"],
    )
    sig_b = _compute_template_signature(
        brand="Kisqali",
        event_type="agent_action",
        event_subtype="ate_estimation",
        action_keys=["plan_run", "estimate", "refute"],
    )
    assert sig_a is not None
    assert sig_a == sig_b
    assert sig_a.startswith("v1:")


def test_compute_template_signature_normalises_action_key_order() -> None:
    """sorted(action_keys) means the signature is invariant to encounter
    order — Decision §3.4 explicitly calls this out."""
    sig_a = _compute_template_signature(
        brand="Kisqali",
        event_type="agent_action",
        event_subtype="ate_estimation",
        action_keys=["plan_run", "estimate", "refute"],
    )
    sig_b = _compute_template_signature(
        brand="Kisqali",
        event_type="agent_action",
        event_subtype="ate_estimation",
        action_keys=["refute", "plan_run", "estimate"],
    )
    assert sig_a == sig_b


def test_compute_template_signature_brand_boundary() -> None:
    """Same action_keys + event_type + event_subtype under different
    brands → distinct signatures. Defense in depth alongside the DB
    partial-unique-index (which uses COALESCE(brand,'')."""
    sig_kis = _compute_template_signature(
        brand="Kisqali",
        event_type="agent_action",
        event_subtype="x",
        action_keys=["a", "b"],
    )
    sig_fab = _compute_template_signature(
        brand="Fabhalta",
        event_type="agent_action",
        event_subtype="x",
        action_keys=["a", "b"],
    )
    assert sig_kis != sig_fab


def test_compute_template_signature_returns_none_when_required_fields_missing() -> None:
    """Missing brand / event_type / event_subtype / empty action_keys
    → signature is None and the row is not safe to template."""
    assert (
        _compute_template_signature(
            brand=None,
            event_type="x",
            event_subtype="y",
            action_keys=["a"],
        )
        is None
    )
    assert (
        _compute_template_signature(
            brand="b",
            event_type=None,
            event_subtype="y",
            action_keys=["a"],
        )
        is None
    )
    assert (
        _compute_template_signature(
            brand="b",
            event_type="x",
            event_subtype="y",
            action_keys=[],
        )
        is None
    )


# ---------------------------------------------------------------------------
# _jaccard_cohesion contract
# ---------------------------------------------------------------------------


def test_jaccard_cohesion_identical_sets_returns_1() -> None:
    """All pairwise sets equal → cohesion = 1.0 (perfect agreement)."""
    members = [
        {"action_keys": ["plan", "estimate", "refute"]},
        {"action_keys": ["plan", "estimate", "refute"]},
        {"action_keys": ["plan", "estimate", "refute"]},
    ]
    assert _jaccard_cohesion(members) == pytest.approx(1.0)


def test_jaccard_cohesion_disjoint_sets_returns_0() -> None:
    """All pairwise sets disjoint → cohesion = 0.0 (no agreement)."""
    members = [
        {"action_keys": ["a"]},
        {"action_keys": ["b"]},
        {"action_keys": ["c"]},
    ]
    assert _jaccard_cohesion(members) == pytest.approx(0.0)


def test_jaccard_cohesion_partial_overlap_returns_midrange() -> None:
    """Half-overlapping sets land in (0, 1) — used as the confidence
    score for noise-cluster rejection at threshold 0.3."""
    members = [
        {"action_keys": ["a", "b"]},
        {"action_keys": ["a", "c"]},
    ]
    # |intersection| = 1 ({a}); |union| = 3 ({a,b,c}); J = 1/3 ≈ 0.333
    assert _jaccard_cohesion(members) == pytest.approx(1.0 / 3.0)


def test_jaccard_cohesion_singleton_cluster_returns_1() -> None:
    """A single-member cluster trivially has cohesion 1 (no pairs)."""
    members = [{"action_keys": ["a", "b"]}]
    assert _jaccard_cohesion(members) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# extract_procedural_templates: happy-path symbolic extraction
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_n3_cluster_produces_one_procedural_template(
    fake_supabase: FakeSupabase,
):
    """N≥3 cluster of episodic rows sharing the cluster-key tuple
    produces exactly one ``ProceduralTemplate`` row inserted into
    ``procedural_templates``.

    Cohesion 1.0 (identical action_keys) → confidence ≥ 0.8 →
    template promoted."""
    fake_supabase.rows["episodic_memories"].extend(
        [
            {
                "memory_id": f"m{i}",
                "brand": "Kisqali",
                "event_type": "agent_action",
                "event_subtype": "ate_estimation",
                "raw_content": {"action_keys": ["plan_run", "estimate", "refute"]},
                "occurred_at": f"2026-05-2{i}T00:00:00+00:00",
            }
            for i in range(3)
        ]
    )
    consolidator = Consolidator()
    n_templates = await consolidator.extract_procedural_templates(brand="Kisqali")
    assert n_templates == 1
    assert len(fake_supabase.rows["procedural_templates"]) == 1
    row = fake_supabase.rows["procedural_templates"][0]
    assert row["brand"] == "Kisqali"
    assert row["extraction_confidence"] >= 0.8
    assert row["extraction_method"] == "symbolic"


@pytest.mark.asyncio
async def test_cluster_below_n_min_does_not_promote(
    fake_supabase: FakeSupabase,
):
    """N=2 cluster (below default N_MIN=3) does NOT produce a
    template — too few observations to call it a pattern."""
    fake_supabase.rows["episodic_memories"].extend(
        [
            {
                "memory_id": f"m{i}",
                "brand": "Kisqali",
                "event_type": "agent_action",
                "event_subtype": "x",
                "raw_content": {"action_keys": ["a", "b"]},
                "occurred_at": "2026-05-20T00:00:00+00:00",
            }
            for i in range(2)
        ]
    )
    consolidator = Consolidator()
    n_templates = await consolidator.extract_procedural_templates(brand="Kisqali")
    assert n_templates == 0
    assert fake_supabase.rows["procedural_templates"] == []


@pytest.mark.asyncio
async def test_noisy_cluster_below_confidence_threshold_not_promoted(
    fake_supabase: FakeSupabase,
):
    """A 3-row cluster with widely-divergent action_keys yields cohesion
    < 0.3 → NOT promoted. ``noisy_cluster`` here is rows sharing
    (brand, event_type, event_subtype) but differing on action_keys —
    the consolidator must group them by cluster-key (which INCLUDES
    sorted action_keys) so cohesion is technically 1 within each
    smaller subgroup. The N_MIN=3 filter therefore catches noise here
    indirectly (subgroups are size 1)."""
    fake_supabase.rows["episodic_memories"].extend(
        [
            {
                "memory_id": "m0",
                "brand": "Kisqali",
                "event_type": "agent_action",
                "event_subtype": "x",
                "raw_content": {"action_keys": ["a", "b", "c"]},
                "occurred_at": "2026-05-20T00:00:00+00:00",
            },
            {
                "memory_id": "m1",
                "brand": "Kisqali",
                "event_type": "agent_action",
                "event_subtype": "x",
                "raw_content": {"action_keys": ["d", "e", "f"]},
                "occurred_at": "2026-05-20T00:00:00+00:00",
            },
            {
                "memory_id": "m2",
                "brand": "Kisqali",
                "event_type": "agent_action",
                "event_subtype": "x",
                "raw_content": {"action_keys": ["g", "h", "i"]},
                "occurred_at": "2026-05-20T00:00:00+00:00",
            },
        ]
    )
    consolidator = Consolidator()
    n_templates = await consolidator.extract_procedural_templates(brand="Kisqali")
    # All three rows fall into different cluster-key subgroups (because
    # sorted(action_keys) is part of the key). Each subgroup has N=1,
    # so the N_MIN=3 filter rejects all of them. No template produced.
    assert n_templates == 0


@pytest.mark.asyncio
async def test_brand_isolation_same_signature_different_brands(
    fake_supabase: FakeSupabase,
):
    """Same action_keys + event_type + event_subtype under two brands
    → two separate ``procedural_templates`` rows (one per brand)."""
    for brand in ("Kisqali", "Fabhalta"):
        for i in range(3):
            fake_supabase.rows["episodic_memories"].append(
                {
                    "memory_id": f"{brand}-m{i}",
                    "brand": brand,
                    "event_type": "agent_action",
                    "event_subtype": "x",
                    "raw_content": {"action_keys": ["a", "b"]},
                    "occurred_at": "2026-05-20T00:00:00+00:00",
                }
            )
    consolidator = Consolidator()
    n_kis = await consolidator.extract_procedural_templates(brand="Kisqali")
    n_fab = await consolidator.extract_procedural_templates(brand="Fabhalta")
    assert n_kis == 1
    assert n_fab == 1
    brands = sorted(r["brand"] for r in fake_supabase.rows["procedural_templates"])
    assert brands == ["Fabhalta", "Kisqali"]


@pytest.mark.asyncio
async def test_idempotent_re_extraction_does_not_duplicate(
    fake_supabase: FakeSupabase,
):
    """Re-running on the same cluster does NOT add a second template
    row — the partial-unique-index emulated by ``FakeSupabase`` raises
    on the second insert and the consolidator swallows it (idempotency
    contract)."""
    for i in range(3):
        fake_supabase.rows["episodic_memories"].append(
            {
                "memory_id": f"m{i}",
                "brand": "Kisqali",
                "event_type": "agent_action",
                "event_subtype": "x",
                "raw_content": {"action_keys": ["a", "b"]},
                "occurred_at": "2026-05-20T00:00:00+00:00",
            }
        )
    consolidator = Consolidator()
    first_run = await consolidator.extract_procedural_templates(brand="Kisqali")
    second_run = await consolidator.extract_procedural_templates(brand="Kisqali")
    assert first_run == 1
    # Second run: existing row catches the unique-violation; no duplicate.
    assert second_run == 0
    assert len(fake_supabase.rows["procedural_templates"]) == 1


@pytest.mark.asyncio
async def test_variable_substitution_extracts_shared_vs_variable_keys(
    fake_supabase: FakeSupabase,
):
    """When cluster rows share SOME action_keys but differ on others
    (within the SAME cluster-key tuple — i.e. their sorted union is
    identical), the template's ``shared_action_keys`` captures the
    intersection and ``variables`` captures the symmetric-difference.

    NOTE: V1 clustering uses ``sorted(action_keys)`` as part of the
    cluster key, so this test exercises clusters where every row has
    the SAME sorted union. Per-instance variation is captured by
    looking at OTHER fields (e.g. raw_content key names that appear in
    SOME but not all rows of the cluster — e.g. ``hcp_id`` vs
    ``provider_id`` aliases for the same role). Variables in V1 capture
    BOTH:
      * **presence variance** — key appears in some rows but not all
        (e.g. only m1 carries ``region``).
      * **value variance** — key appears in EVERY row but with
        different values (e.g. every row has ``hcp_id`` but each row's
        hcp_id is distinct — HCP-001/002/003).

    Iter-1 codex H2 fix: the original formula handled only presence
    variance; value variance is now also captured to match the
    ProceduralTemplate docstring's own examples (hcp_id / region /
    cohort)."""
    fake_supabase.rows["episodic_memories"].extend(
        [
            {
                "memory_id": "m0",
                "brand": "Kisqali",
                "event_type": "agent_action",
                "event_subtype": "x",
                # Same action_keys (cluster key) but different param names.
                "raw_content": {
                    "action_keys": ["a", "b"],
                    "hcp_id": "HCP-001",
                    "kpi": "trx",
                },
                "occurred_at": "2026-05-20T00:00:00+00:00",
            },
            {
                "memory_id": "m1",
                "brand": "Kisqali",
                "event_type": "agent_action",
                "event_subtype": "x",
                "raw_content": {
                    "action_keys": ["a", "b"],
                    "hcp_id": "HCP-002",
                    "kpi": "trx",
                    "region": "EU",
                },
                "occurred_at": "2026-05-20T00:00:00+00:00",
            },
            {
                "memory_id": "m2",
                "brand": "Kisqali",
                "event_type": "agent_action",
                "event_subtype": "x",
                "raw_content": {
                    "action_keys": ["a", "b"],
                    "hcp_id": "HCP-003",
                    "kpi": "trx",
                },
                "occurred_at": "2026-05-20T00:00:00+00:00",
            },
        ]
    )
    consolidator = Consolidator()
    n_templates = await consolidator.extract_procedural_templates(brand="Kisqali")
    assert n_templates == 1
    template_row = fake_supabase.rows["procedural_templates"][0]
    body = template_row["template_body"]
    # shared_action_keys = intersection of action_keys = {"a", "b"}.
    assert sorted(body["shared_action_keys"]) == ["a", "b"]
    # variables = presence_variance ∪ value_variance (excluding the
    # cluster basis key "action_keys").
    #   presence_variance = {region}     (only m1 has it)
    #   value_variance   = {hcp_id}      (all rows have it but with
    #                                     distinct values HCP-001/002/003;
    #                                     kpi='trx' is identical so NOT
    #                                     in value_variance)
    # → variables = sorted({hcp_id, region}) = ["hcp_id", "region"]
    assert body["variables"] == ["hcp_id", "region"]
    assert sorted(template_row["derived_from_episodic_ids"]) == ["m0", "m1", "m2"]


# ---------------------------------------------------------------------------
# Feature-flag gating + LLM path
# ---------------------------------------------------------------------------


def test_llm_extraction_enabled_default_off(monkeypatch: pytest.MonkeyPatch) -> None:
    """With no env override, ``_llm_extraction_enabled`` is False —
    the symbolic path is the only one that runs in production by
    default."""
    monkeypatch.delenv(PROCEDURAL_LLM_EXTRACTION_ENV_VAR, raising=False)
    assert _llm_extraction_enabled() is False


def test_llm_extraction_enabled_true_when_env_set(monkeypatch: pytest.MonkeyPatch) -> None:
    """Truthy env values (``1``/``true``/``yes``/``on``) opt in."""
    for val in ("1", "true", "TRUE", "yes", "YES", "on", "ON"):
        monkeypatch.setenv(PROCEDURAL_LLM_EXTRACTION_ENV_VAR, val)
        assert _llm_extraction_enabled() is True, f"expected truthy for {val!r}"


@pytest.mark.asyncio
async def test_flag_off_does_not_invoke_client_factory(
    fake_supabase: FakeSupabase,
    monkeypatch: pytest.MonkeyPatch,
):
    """With ``PROCEDURAL_LLM_EXTRACTION_ENABLED=false`` (default), the
    consolidator must NOT invoke the injected ``client_factory`` — the
    symbolic path produces the template alone."""
    monkeypatch.delenv(PROCEDURAL_LLM_EXTRACTION_ENV_VAR, raising=False)
    for i in range(3):
        fake_supabase.rows["episodic_memories"].append(
            {
                "memory_id": f"m{i}",
                "brand": "Kisqali",
                "event_type": "agent_action",
                "event_subtype": "x",
                "raw_content": {"action_keys": ["a", "b"]},
                "occurred_at": "2026-05-20T00:00:00+00:00",
            }
        )

    factory_call_count = 0

    def _spy_factory(api_key: str):  # pragma: no cover - must NOT be called
        nonlocal factory_call_count
        factory_call_count += 1
        raise AssertionError(
            "client_factory invoked despite PROCEDURAL_LLM_EXTRACTION_ENABLED=false"
        )

    consolidator = Consolidator(anthropic_client_factory=_spy_factory)
    n_templates = await consolidator.extract_procedural_templates(brand="Kisqali")
    assert n_templates == 1
    assert factory_call_count == 0
    # Method recorded as 'symbolic'.
    row = fake_supabase.rows["procedural_templates"][0]
    assert row["extraction_method"] == "symbolic"


@pytest.mark.asyncio
async def test_flag_on_invokes_client_factory_and_records_llm_method(
    fake_supabase: FakeSupabase,
    monkeypatch: pytest.MonkeyPatch,
):
    """With the flag ON and a fake ``client_factory`` returning a
    deterministic high-coherence rating, the consolidator MULTIPLIES
    symbolic cohesion by the LLM rating and records
    ``extraction_method='llm_with_fallback'``."""
    monkeypatch.setenv(PROCEDURAL_LLM_EXTRACTION_ENV_VAR, "true")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    for i in range(3):
        fake_supabase.rows["episodic_memories"].append(
            {
                "memory_id": f"m{i}",
                "brand": "Kisqali",
                "event_type": "agent_action",
                "event_subtype": "x",
                "raw_content": {"action_keys": ["a", "b"]},
                "occurred_at": "2026-05-20T00:00:00+00:00",
            }
        )

    class _FakeMessages:
        async def create(self, **kwargs):
            class _R:
                def __init__(self) -> None:
                    class _C:
                        text = '{"coherence": 0.9}'

                    self.content = [_C()]
                    self.usage = None

            return _R()

    class _FakeClient:
        @property
        def messages(self) -> "_FakeMessages":
            return _FakeMessages()

    def _fake_factory(api_key: str) -> "_FakeClient":
        return _FakeClient()

    consolidator = Consolidator(anthropic_client_factory=_fake_factory)
    n_templates = await consolidator.extract_procedural_templates(brand="Kisqali")
    assert n_templates == 1
    row = fake_supabase.rows["procedural_templates"][0]
    assert row["extraction_method"] == "llm_with_fallback"
    # symbolic = 1.0; LLM = 0.9; product = 0.9
    assert row["extraction_confidence"] == pytest.approx(0.9, abs=1e-6)


@pytest.mark.asyncio
async def test_flag_on_llm_exception_falls_back_to_symbolic(
    fake_supabase: FakeSupabase,
    monkeypatch: pytest.MonkeyPatch,
):
    """Real-catch-not-mock: flag ON, ``client_factory`` returns a
    client whose ``messages.create`` raises an
    ``anthropic.APITimeoutError`` shape. The consolidator MUST still
    emit a template (the symbolic path always runs first) with
    ``extraction_method='symbolic'`` and the symbolic-only confidence
    (no LLM multiplier applied)."""
    monkeypatch.setenv(PROCEDURAL_LLM_EXTRACTION_ENV_VAR, "true")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    for i in range(3):
        fake_supabase.rows["episodic_memories"].append(
            {
                "memory_id": f"m{i}",
                "brand": "Kisqali",
                "event_type": "agent_action",
                "event_subtype": "x",
                "raw_content": {"action_keys": ["a", "b"]},
                "occurred_at": "2026-05-20T00:00:00+00:00",
            }
        )

    import anthropic

    class _BoomMessages:
        async def create(self, **kwargs):
            # Use the real anthropic.APITimeoutError class so the
            # consolidator's narrow catch tuple is the boundary under
            # test — NOT a mock-bypass. Per memory
            # [[feedback-test-must-exercise-real-catch-not-mock]].
            raise anthropic.APITimeoutError(request=MagicMock())

    class _BoomClient:
        @property
        def messages(self) -> "_BoomMessages":
            return _BoomMessages()

    def _boom_factory(api_key: str) -> "_BoomClient":
        return _BoomClient()

    consolidator = Consolidator(anthropic_client_factory=_boom_factory)
    n_templates = await consolidator.extract_procedural_templates(brand="Kisqali")
    assert n_templates == 1
    row = fake_supabase.rows["procedural_templates"][0]
    # Symbolic-only fallback: method downgrades from llm_with_fallback
    # to symbolic and confidence = symbolic-only (no multiplier).
    assert row["extraction_method"] == "symbolic"
    assert row["extraction_confidence"] == pytest.approx(1.0, abs=1e-6)


@pytest.mark.asyncio
async def test_programming_errors_propagate_not_swallowed(
    fake_supabase: FakeSupabase,
    monkeypatch: pytest.MonkeyPatch,
):
    """Narrow-catch contract: programming errors (TypeError /
    AttributeError / KeyError) must NOT be silently caught — they
    propagate out of the LLM helper so they surface in CI / DLQ
    instead of being misclassified as 'LLM transient'."""
    monkeypatch.setenv(PROCEDURAL_LLM_EXTRACTION_ENV_VAR, "true")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    for i in range(3):
        fake_supabase.rows["episodic_memories"].append(
            {
                "memory_id": f"m{i}",
                "brand": "Kisqali",
                "event_type": "agent_action",
                "event_subtype": "x",
                "raw_content": {"action_keys": ["a", "b"]},
                "occurred_at": "2026-05-20T00:00:00+00:00",
            }
        )

    def _broken_factory(api_key: str):
        raise TypeError("intentional programming error to verify narrow catch")

    consolidator = Consolidator(anthropic_client_factory=_broken_factory)
    with pytest.raises(TypeError, match="intentional programming error"):
        await consolidator.extract_procedural_templates(brand="Kisqali")


# ---------------------------------------------------------------------------
# ProceduralTemplate Pydantic shape
# ---------------------------------------------------------------------------


def test_procedural_template_model_validates_confidence_bounds() -> None:
    """Confidence must be in [0..1]; out-of-range raises ValidationError."""
    from pydantic import ValidationError

    # In-range values OK.
    ProceduralTemplate(
        brand="Kisqali",
        template_signature="v1:sig",
        event_type="agent_action",
        event_subtype="x",
        shared_action_keys=["a"],
        variables=[],
        derived_from_episodic_ids=["m0"],
        extraction_confidence=0.5,
        extraction_method="symbolic",
    )
    # Out-of-range rejected.
    with pytest.raises(ValidationError):
        ProceduralTemplate(
            brand="Kisqali",
            template_signature="v1:sig",
            event_type="agent_action",
            event_subtype="x",
            shared_action_keys=["a"],
            variables=[],
            derived_from_episodic_ids=["m0"],
            extraction_confidence=1.5,  # > 1
            extraction_method="symbolic",
        )
    with pytest.raises(ValidationError):
        ProceduralTemplate(
            brand="Kisqali",
            template_signature="v1:sig",
            event_type="agent_action",
            event_subtype="x",
            shared_action_keys=["a"],
            variables=[],
            derived_from_episodic_ids=["m0"],
            extraction_confidence=-0.1,  # < 0
            extraction_method="symbolic",
        )


def test_procedural_template_model_validates_extraction_method_literal() -> None:
    """``extraction_method`` is constrained to the two literal values
    'symbolic' / 'llm_with_fallback'."""
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        ProceduralTemplate(
            brand="Kisqali",
            template_signature="v1:sig",
            event_type="agent_action",
            event_subtype="x",
            shared_action_keys=["a"],
            variables=[],
            derived_from_episodic_ids=["m0"],
            extraction_confidence=0.5,
            extraction_method="weather_forecast",  # type: ignore[arg-type]
        )


# ---------------------------------------------------------------------------
# Iter-1 codex M1: postgrest.APIError-with-SQLSTATE-23505 idempotency
# ---------------------------------------------------------------------------


def test_is_unique_violation_or_postgrest_23505_accepts_psycopg_shape() -> None:
    """The widened helper must STILL accept the psycopg/test-stub
    UniqueViolation shape (back-compat with the dedup path's
    detector)."""

    class _UV(Exception):
        """psycopg-like UniqueViolation class name."""

        def __init__(self) -> None:
            super().__init__("duplicate key value violates unique constraint uix_x")

    # Rename class so the type-name check passes.
    _UV.__name__ = "UniqueViolation"
    assert Consolidator._is_unique_violation_or_postgrest_23505(_UV()) is True


def test_is_unique_violation_or_postgrest_23505_accepts_postgrest_apierror() -> None:
    """The widened helper must accept postgrest.APIError with
    SQLSTATE 23505 (supabase-py surfaces unique-violations through
    this class — NOT through psycopg.UniqueViolation — so the dedup
    path's narrower detector would MISS it on the production path)."""

    class _PostgrestAPIError(Exception):
        def __init__(self, code: str) -> None:
            super().__init__("PostgREST APIError")
            self.code = code

    _PostgrestAPIError.__name__ = "APIError"
    assert Consolidator._is_unique_violation_or_postgrest_23505(_PostgrestAPIError("23505")) is True


def test_is_unique_violation_or_postgrest_23505_rejects_non_unique_apierror() -> None:
    """A postgrest APIError with a DIFFERENT SQLSTATE (e.g. 23503 =
    foreign_key_violation, 42P01 = undefined_table) must NOT be
    treated as a unique-violation. False positives here would silently
    swallow real errors."""

    class _PostgrestAPIError(Exception):
        def __init__(self, code: str) -> None:
            super().__init__("PostgREST APIError")
            self.code = code

    _PostgrestAPIError.__name__ = "APIError"
    assert (
        Consolidator._is_unique_violation_or_postgrest_23505(_PostgrestAPIError("23503")) is False
    )
    assert (
        Consolidator._is_unique_violation_or_postgrest_23505(_PostgrestAPIError("42P01")) is False
    )


def test_is_unique_violation_or_postgrest_23505_rejects_unrelated_exceptions() -> None:
    """Random exception shapes (TypeError, ValueError, generic
    Exception) must NOT match — the helper is conservative."""
    assert (
        Consolidator._is_unique_violation_or_postgrest_23505(
            TypeError("unrelated programming error")
        )
        is False
    )
    assert (
        Consolidator._is_unique_violation_or_postgrest_23505(ValueError("unrelated value error"))
        is False
    )
    assert (
        Consolidator._is_unique_violation_or_postgrest_23505(
            Exception("some other error 23505 in message")
        )
        is False
    )


# ---------------------------------------------------------------------------
# Iter-1 codex H1: extract_procedural_templates wired into Consolidator.run()
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_invokes_extract_procedural_templates(
    fake_supabase: FakeSupabase,
):
    """``Consolidator.run()`` must call ``extract_procedural_templates``
    so scheduled Celery passes produce templates without an extra
    wiring step. Iter-1 codex H1 fix: previously the method existed but
    was not wired into ``run()``."""
    # Seed a 3-row cluster that will produce one template.
    for i in range(3):
        fake_supabase.rows["episodic_memories"].append(
            {
                "memory_id": f"m{i}",
                "brand": "Kisqali",
                "event_type": "agent_action",
                "event_subtype": "x",
                "raw_content": {"action_keys": ["a", "b"]},
                "occurred_at": "2026-05-20T00:00:00+00:00",
            }
        )
    consolidator = Consolidator()
    result = await consolidator.run(brand="Kisqali")
    assert result.procedural_templates_extracted == 1
    assert len(fake_supabase.rows["procedural_templates"]) == 1


@pytest.mark.asyncio
async def test_run_does_not_double_extract_on_second_pass(
    fake_supabase: FakeSupabase,
):
    """Running ``Consolidator.run()`` twice on the same data does NOT
    produce duplicate templates — idempotency via the partial-unique-
    index swallow. Second pass returns
    ``procedural_templates_extracted == 0`` while the first row still
    sits in the table."""
    for i in range(3):
        fake_supabase.rows["episodic_memories"].append(
            {
                "memory_id": f"m{i}",
                "brand": "Kisqali",
                "event_type": "agent_action",
                "event_subtype": "x",
                "raw_content": {"action_keys": ["a", "b"]},
                "occurred_at": "2026-05-20T00:00:00+00:00",
            }
        )
    consolidator = Consolidator()
    r1 = await consolidator.run(brand="Kisqali")
    r2 = await consolidator.run(brand="Kisqali")
    assert r1.procedural_templates_extracted == 1
    assert r2.procedural_templates_extracted == 0
    assert len(fake_supabase.rows["procedural_templates"]) == 1


@pytest.mark.asyncio
async def test_run_records_non_unique_insert_failure_on_result_errors(
    fake_supabase: FakeSupabase,
):
    """Iter-2 codex M1 fix: when the procedural-template insert fails
    with a NON-unique-violation exception (e.g. PostgREST APIError
    with SQLSTATE 23503 = foreign-key-violation, 42P01 = undefined-
    table, etc.), the error MUST appear on
    ``ConsolidationResult.errors`` so the consolidator's run summary
    surfaces it.

    Test exercises the failure path by injecting a FakeSupabase that
    raises a non-unique APIError-shaped exception on insert into
    ``procedural_templates``. The cluster must be valid (N≥3 seeded
    rows that survive dedup → effective_cluster_size≥3) so the
    insert is actually attempted."""

    # Build a FakeSupabase subclass that raises a postgrest-style
    # APIError with SQLSTATE 23503 (foreign-key-violation) — NOT the
    # 23505 unique-violation idempotency case.
    class _FailingFakeSupabase(FakeSupabase):
        class _FailingQuery(_FakeQuery):
            def execute(self):  # type: ignore[override]
                if self._mode == "insert" and self.table_name == "procedural_templates":

                    class _APIError(Exception):
                        def __init__(self) -> None:
                            super().__init__("foreign-key-violation")
                            self.code = "23503"

                    _APIError.__name__ = "APIError"
                    raise _APIError()
                return super().execute()

        def table(self, name: str) -> "_FakeQuery":  # type: ignore[override]
            return _FailingFakeSupabase._FailingQuery(self, name)

    failing_supabase = _FailingFakeSupabase()
    # Seed a 3-row cluster where the rows have DIFFERENT causal_path_ids
    # so dedup leaves them distinct AND template-extraction sees them
    # as a single cluster (same action_keys + brand + event_type +
    # event_subtype). Without distinct dedup signatures the rows
    # collapse to 1 row with dedup_counter=3 — that ALSO trips the
    # threshold but covers a different code path; the multi-row form
    # exercises the insert-failure code path more directly.
    for i in range(3):
        failing_supabase.rows["episodic_memories"].append(
            {
                "memory_id": f"m{i}",
                "brand": "Kisqali",
                "event_type": "agent_action",
                "event_subtype": "x",
                "causal_path_id": f"cp-{i}",  # distinct → dedup leaves alone
                "raw_content": {"action_keys": ["a", "b"]},
                "occurred_at": f"2026-05-2{i}T00:00:00+00:00",
            }
        )

    with patch(
        "src.memory.lifecycle.consolidator.get_supabase_client",
        return_value=failing_supabase,
    ):
        consolidator = Consolidator()
        result = await consolidator.run(brand="Kisqali")

    # Wiring envelope: run() completes without raising.
    assert result.procedural_templates_extracted == 0
    # The non-unique APIError MUST appear on result.errors
    # (iter-2 codex M1 contract).
    assert any("procedural-template insert" in err for err in result.errors), (
        f"expected procedural-template insert error in result.errors; got {result.errors!r}"
    )
    assert any("foreign-key-violation" in err for err in result.errors), (
        f"expected the original exception message preserved; got {result.errors!r}"
    )


@pytest.mark.asyncio
async def test_run_does_not_record_unique_violation_on_result_errors(
    fake_supabase: FakeSupabase,
):
    """Idempotency contract: a unique-violation on re-extraction is
    EXPECTED and must NOT appear on ``ConsolidationResult.errors`` —
    only non-idempotent failures do (iter-2 codex M1 contract)."""
    for i in range(3):
        fake_supabase.rows["episodic_memories"].append(
            {
                "memory_id": f"m{i}",
                "brand": "Kisqali",
                "event_type": "agent_action",
                "event_subtype": "x",
                "causal_path_id": f"cp-{i}",
                "raw_content": {"action_keys": ["a", "b"]},
                "occurred_at": f"2026-05-2{i}T00:00:00+00:00",
            }
        )
    consolidator = Consolidator()
    r1 = await consolidator.run(brand="Kisqali")
    r2 = await consolidator.run(brand="Kisqali")
    assert r1.procedural_templates_extracted == 1
    assert r2.procedural_templates_extracted == 0
    # No procedural-template errors on either run — unique-violation
    # is the idempotency contract, not an error.
    assert not any("procedural-template insert" in err for err in r1.errors), (
        f"first run should be clean; got {r1.errors!r}"
    )
    assert not any("procedural-template insert" in err for err in r2.errors), (
        f"second run should be silent on idempotent re-insert; got {r2.errors!r}"
    )
