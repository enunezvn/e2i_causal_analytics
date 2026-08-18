"""#1685: the procedural fallback must target the REAL schema.

``ProceduralMemoryAdapter._execute_procedure_search`` was written (2025-12-20,
19ea4858a) against an assumed schema: an RPC ``search_procedural_memory`` and a
table ``procedural_memory``. Neither has ever existed — prod has the table
``procedural_memories`` (plural) and the embedding-based RPC
``find_relevant_procedures`` (used by the adapter's PRIMARY semantic path).
Measured live (2026-08-18): PGRST202 for the RPC, PGRST205 for the table, so
the fallback returned ``[]`` unconditionally, always.

``_build_procedure_content`` had the same disease: it rendered ``name`` /
``steps`` / ``pattern`` keys that neither the real table nor the real RPC
return — real rows carry ``procedure_name`` / ``tool_sequence`` /
``trigger_pattern``, degrading even the WORKING primary path's content to
"Procedure Type: …; Success Rate: …".

Test design (per the wave-13 discipline):

- a schema-aware fake client that knows ONLY the real identifiers and raises
  for anything else — so the old code's phantom names fail here exactly as
  they fail in prod;
- sync/async client parametrization keeps the #1683 ``_execute_query``
  handling covered on the rewritten call;
- legacy-shape content cases are in-file positive controls that pass before
  AND after the fix.
"""

import asyncio
import json
from typing import Any, Dict, List

import pytest

from src.rag.memory_adapters import ProceduralMemoryAdapter

REAL_ROWS: List[Dict[str, Any]] = [
    {
        "procedure_id": "b1",
        "procedure_name": "adoption_funnel_walkthrough",
        "procedure_type": "tool_sequence",
        "tool_sequence": [
            {"tool": "query_episodic", "args": {"metric": "TRx"}},
            {"tool": "segment_hcps"},
        ],
        "trigger_pattern": "adoption analysis for a brand",
        "success_rate": 0.9,
        "is_active": True,
    },
    {
        "procedure_id": "b2",
        "procedure_name": "hpo_LogisticRegression_binary_classification",
        "procedure_type": "tool_sequence",
        # measured live: HPO rows hold a plain dict, not a list
        "tool_sequence": {"type": "hpo_pattern", "algorithm": "LogisticRegression"},
        "trigger_pattern": "binary classification tuning",
        "success_rate": 1,
        "is_active": True,
    },
]


class _Response:
    def __init__(self, data):
        self.data = data


class _RecordingBuilder:
    """Chainable postgrest builder that records the query it was asked for."""

    def __init__(self, is_async: bool, rows: List[Dict[str, Any]], calls: List[Any]):
        self._is_async = is_async
        self._rows = rows
        self._calls = calls

    def select(self, *a, **k):
        self._calls.append(("select", a, k))
        return self

    def eq(self, *a, **k):
        self._calls.append(("eq", a, k))
        return self

    def order(self, *a, **k):
        self._calls.append(("order", a, k))
        return self

    def limit(self, *a, **k):
        self._calls.append(("limit", a, k))
        return self

    def execute(self):
        if self._is_async:

            async def _run():
                return _Response(self._rows)

            return _run()
        return _Response(self._rows)


class _SchemaAwareClient:
    """Knows ONLY the real schema — phantom identifiers fail like prod does."""

    def __init__(self, is_async: bool):
        self._is_async = is_async
        self.rpc_calls: List[str] = []
        self.builder_calls: List[Any] = []

    def rpc(self, name, _params):
        self.rpc_calls.append(name)
        raise RuntimeError(f"PGRST202: function public.{name} does not exist")

    def table(self, name):
        if name != "procedural_memories":
            raise RuntimeError(f"PGRST205: table public.{name} does not exist")
        return _RecordingBuilder(self._is_async, REAL_ROWS, self.builder_calls)


@pytest.mark.parametrize("is_async", [True, False], ids=["async_client", "sync_client"])
def test_fallback_reads_real_table_for_either_client(is_async):
    """RED before the fix: the phantom names make this return [] against the real schema."""
    client = _SchemaAwareClient(is_async)
    adapter = ProceduralMemoryAdapter(supabase_client=client)

    rows = asyncio.run(adapter._execute_procedure_search("kisqali adoption", 5))

    assert rows == REAL_ROWS, f"expected real rows from procedural_memories, got {rows!r}"


def test_fallback_does_not_call_phantom_rpc():
    """The text-search RPC never existed; the fallback must stop paying for it."""
    client = _SchemaAwareClient(is_async=False)
    adapter = ProceduralMemoryAdapter(supabase_client=client)

    asyncio.run(adapter._execute_procedure_search("kisqali adoption", 5))

    assert client.rpc_calls == [], f"phantom RPC still attempted: {client.rpc_calls}"


def test_fallback_prefers_active_high_success_procedures():
    """Mirror find_relevant_procedures semantics: active rows, best-first, capped.

    (The order matches the partial index idx_procedural_success:
    success_rate DESC WHERE is_active.)
    """
    client = _SchemaAwareClient(is_async=False)
    adapter = ProceduralMemoryAdapter(supabase_client=client)

    asyncio.run(adapter._execute_procedure_search("kisqali adoption", 3))

    calls = {name: (a, k) for name, a, k in client.builder_calls}
    assert calls["eq"][0] == ("is_active", True)
    assert calls["order"][0] == ("success_rate",) and calls["order"][1].get("desc") is True
    assert calls["limit"][0] == (3,)


def test_content_renders_real_schema_columns():
    """RED before the fix: real rows degraded to 'Procedure Type: …; Success Rate: …'."""
    adapter = ProceduralMemoryAdapter(supabase_client=None)

    content = adapter._build_procedure_content(REAL_ROWS[0])

    assert "adoption_funnel_walkthrough" in content, content
    assert "query_episodic" in content, content
    assert "adoption analysis for a brand" in content, content
    assert "90%" in content, content


def test_content_renders_dict_and_double_encoded_tool_sequences():
    """Live HPO rows hold a dict; pre-migration-072 rows hold a JSON string."""
    adapter = ProceduralMemoryAdapter(supabase_client=None)

    dict_content = adapter._build_procedure_content(REAL_ROWS[1])
    assert "hpo_pattern" in dict_content or "LogisticRegression" in dict_content, dict_content

    encoded = dict(REAL_ROWS[0], tool_sequence=json.dumps(REAL_ROWS[0]["tool_sequence"]))
    encoded_content = adapter._build_procedure_content(encoded)
    assert "query_episodic" in encoded_content, encoded_content


def test_content_still_renders_legacy_shape():
    """Positive control — passes before AND after: the imagined shape must keep working."""
    adapter = ProceduralMemoryAdapter(supabase_client=None)

    content = adapter._build_procedure_content(
        {
            "name": "legacy_proc",
            "steps": ["a", "b"],
            "pattern": "legacy pattern",
            "success_rate": 0.5,
        }
    )

    assert "legacy_proc" in content
    assert "a" in content and "b" in content
    assert "legacy pattern" in content
    assert "50%" in content
