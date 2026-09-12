"""The registry sync payload is the running code's tools, and nothing else (spec §4).

``build_sync_payload()`` is what ``sync_tool_registry`` (ml/040) makes the DB registry equal to
at API startup, so it must cover exactly the live tools, carry values the DB constraints admit,
take every field from the live registry, and refuse a partially imported registry rather than
deprecate the tools that failed to import. The real-DB round trip is in
``tests/unit/test_database/learning_loop/test_registry_sync_client.py``.
"""

from __future__ import annotations

import json
import re

import pytest

from src.agents.factory import AGENT_REGISTRY_CONFIG
from src.agents.tool_composer import registry_sync
from src.agents.tool_composer.tool_registry import DEPENDENCY_FIELD_MAPPINGS
from src.tool_registry.registry import get_registry
from tests.unit.test_agents.test_tool_composer.test_registry_schema_drift_2003 import (
    LIVE_TOOLS,
    REPO_ROOT,
)

TOOL_FIELDS = {
    "name",
    "description",
    "category",
    "source_agent",
    "input_schema",
    "output_schema",
    "avg_latency_ms",
    "version",
}


@pytest.fixture(scope="module")
def payload():
    return registry_sync.build_sync_payload()


def _db_tool_categories() -> set:
    ml013 = (REPO_ROOT / "database" / "ml" / "013_tool_composer_tables.sql").read_text()
    enum_body = re.search(r"CREATE TYPE tool_category AS ENUM \((.*?)\);", ml013, re.S)
    assert enum_body, "tool_category enum not found in ml/013"
    values = set(re.findall(r"'([A-Z_]+)'", enum_body.group(1)))
    ml039 = (REPO_ROOT / "database" / "ml" / "039_tool_category_cohort.sql").read_text()
    values |= set(re.findall(r"ADD VALUE IF NOT EXISTS '([A-Z_]+)'", ml039))
    return values


def test_payload_covers_exactly_the_live_tools(payload):
    tools, _ = payload
    names = [t["name"] for t in tools]
    assert sorted(names) == names, "payload order is deterministic (by name)"
    assert set(names) == LIVE_TOOLS
    assert len(names) == len(LIVE_TOOLS) == 20


def test_payload_rows_have_exactly_the_sync_fields(payload):
    tools, deps = payload
    for tool in tools:
        assert set(tool) == TOOL_FIELDS, tool["name"]
    for dep in deps:
        assert set(dep) == {"consumer", "producer", "output_field", "input_field"}


def test_payload_categories_in_db_enum(payload):
    tools, _ = payload
    allowed = _db_tool_categories()
    assert {"CAUSAL", "COHORT"} <= allowed
    assert {t["category"] for t in tools} <= allowed


def test_payload_source_agents_are_known_agents(payload):
    tools, _ = payload
    assert {t["source_agent"] for t in tools} <= set(AGENT_REGISTRY_CONFIG)


def test_payload_fields_come_from_the_live_registry(payload):
    tools, _ = payload
    live = get_registry()
    for tool in tools:
        registered = live.get(tool["name"])
        assert tool["description"] == registered.schema.description
        assert tool["source_agent"] == registered.schema.source_agent
        assert tool["avg_latency_ms"] == float(registered.schema.avg_execution_ms)
        assert tool["version"] == registered.schema.version
        assert tool["output_schema"] == registered.pydantic_output_model.model_json_schema()
        assert set(tool["input_schema"]["properties"]) == {
            p.name for p in registered.schema.input_parameters
        }


def test_payload_dependencies_match_mappings(payload):
    tools, deps = payload
    expected = [
        {"consumer": c, "producer": p, "output_field": out, "input_field": inp}
        for (c, p), (out, inp) in sorted(DEPENDENCY_FIELD_MAPPINGS.items())
    ]
    assert deps == expected
    assert len(deps) == 13
    names = {t["name"] for t in tools}
    assert {d["consumer"] for d in deps} | {d["producer"] for d in deps} <= names


def test_payload_is_json_for_the_rpc(payload):
    tools, deps = payload
    assert json.loads(json.dumps({"p_tools": tools, "p_dependencies": deps})) == {
        "p_tools": tools,
        "p_dependencies": deps,
    }


def test_payload_refuses_partial_registry():
    live = get_registry()
    registry_sync.build_sync_payload()  # every tool imported and registered
    snapshot = live.snapshot()
    missing = "psi_calculator"
    partial = {
        "tools": {k: v for k, v in snapshot["tools"].items() if k != missing},
        "by_agent": {
            a: [n for n in names if n != missing] for a, names in snapshot["by_agent"].items()
        },
        "by_tier": {
            t: [n for n in names if n != missing] for t, names in snapshot["by_tier"].items()
        },
    }
    try:
        live.restore_snapshot(partial)
        with pytest.raises(LookupError, match=missing):
            registry_sync.build_sync_payload()
    finally:
        live.restore_snapshot(snapshot)
    assert live.get(missing) is not None
