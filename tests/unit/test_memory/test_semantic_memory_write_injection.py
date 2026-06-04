"""Write-side Cypher-injection hardening for semantic memory (H2).

Memory-review finding H2: the FalkorDB WRITE methods spliced caller-supplied
structural tokens straight into the query string —

    MERGE (e:{label} {{...}})                      # add_e2i_entity
    MERGE (s)-[r:{rel_type}]->(t)                  # add_e2i_relationship
    ON CREATE SET e += {{{prop_string}}}           # property KEY names

— with no validation, while the READ methods were already allowlisted. The
live vector is ``rag/cognitive_backends.py`` passing an LLM-derived
``relationship_type.upper()``. ``.upper()`` does not neutralise ``]``/``(``.

Fix: validate the relationship type, node label and every property KEY as a
strict Cypher identifier BEFORE it is interpolated, raising ``ValueError`` so a
poisoned token can never reach ``graph.query``. A strict identifier regex
(not the read-side closed allowlist) is used because production agents write
open-ended labels (e.g. ``"SegmentEffect"``, ``"Variable"``) — an allowlist
would break those legitimate writes while adding no extra injection safety.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.memory.episodic_memory import E2IEntityType
from src.memory.semantic_memory import FalkorDBSemanticMemory


def _sm_with_fake_graph() -> FalkorDBSemanticMemory:
    sm = FalkorDBSemanticMemory()
    sm._graph = MagicMock()  # bypass real FalkorDB; observe .query calls
    return sm


# ---------------------------------------------------------------------------
# add_e2i_relationship
# ---------------------------------------------------------------------------


def test_relationship_rejects_poison_rel_type() -> None:
    sm = _sm_with_fake_graph()
    with pytest.raises(ValueError):
        sm.add_e2i_relationship(
            source_type=E2IEntityType.PATIENT,
            source_id="pat_1",
            target_type=E2IEntityType.HCP,
            target_id="hcp_1",
            rel_type="OWNS]->(x) DETACH DELETE x //",
        )
    sm._graph.query.assert_not_called()


def test_relationship_rejects_poison_property_key() -> None:
    sm = _sm_with_fake_graph()
    with pytest.raises(ValueError):
        sm.add_e2i_relationship(
            source_type=E2IEntityType.PATIENT,
            source_id="pat_1",
            target_type=E2IEntityType.HCP,
            target_id="hcp_1",
            rel_type="TREATED_BY",
            properties={"weight} ) DETACH DELETE n //": 1},
        )
    sm._graph.query.assert_not_called()


def test_relationship_accepts_valid_rel_type() -> None:
    sm = _sm_with_fake_graph()
    ok = sm.add_e2i_relationship(
        source_type=E2IEntityType.PATIENT,
        source_id="pat_1",
        target_type=E2IEntityType.HCP,
        target_id="hcp_1",
        rel_type="TREATED_BY",
        properties={"confidence": 0.9},
    )
    assert ok is True
    assert sm._graph.query.called


# ---------------------------------------------------------------------------
# add_e2i_entity
# ---------------------------------------------------------------------------


def test_entity_rejects_poison_string_label() -> None:
    sm = _sm_with_fake_graph()
    with pytest.raises(ValueError):
        sm.add_e2i_entity(
            entity_type="Evil {id:1}) DETACH DELETE n //",
            entity_id="x1",
        )
    sm._graph.query.assert_not_called()


def test_entity_rejects_poison_property_key() -> None:
    sm = _sm_with_fake_graph()
    with pytest.raises(ValueError):
        sm.add_e2i_entity(
            entity_type=E2IEntityType.PATIENT,
            entity_id="pat_1",
            properties={"name} ) DETACH DELETE n //": "x"},
        )
    sm._graph.query.assert_not_called()


def test_entity_accepts_enum_type() -> None:
    sm = _sm_with_fake_graph()
    ok = sm.add_e2i_entity(entity_type=E2IEntityType.PATIENT, entity_id="pat_1")
    assert ok is True
    assert sm._graph.query.called


def test_entity_accepts_open_ended_string_label() -> None:
    """Legitimate agent writes use dynamic labels not in the read-side
    allowlist (e.g. ``SegmentEffect``); the identifier regex must allow them."""
    sm = _sm_with_fake_graph()
    ok = sm.add_e2i_entity(entity_type="SegmentEffect", entity_id="seg_1")
    assert ok is True
    assert sm._graph.query.called
