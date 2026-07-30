"""#1367: the semantic-memory debug log must not emit the full user query.

``CausalImpactAgent.query_semantic_memory`` logged ``'{query}'`` untruncated at
DEBUG (agent.py:769) — the only zero-truncation query log site in the repo.
This caplog test pins that the emitted record is now redacted through
:func:`src.utils.redaction.redact_query`, so a long query no longer lands whole
in the container logs.
"""

import logging
from unittest.mock import MagicMock

import pytest

from src.agents.causal_impact.agent import CausalImpactAgent
from src.utils.redaction import redact_query


@pytest.mark.asyncio
async def test_query_semantic_memory_redacts_query_in_debug_log(caplog):
    # Build the agent without its heavy __init__; the method only touches
    # ``self._semantic_memory`` and the module logger.
    agent = object.__new__(CausalImpactAgent)
    semantic = MagicMock()
    semantic.traverse_causal_chain.return_value = []
    agent._semantic_memory = semantic

    long_query = "brand-x hcp engagement impact on patient conversions " * 5
    assert len(long_query) > 50  # guard: the query must exceed the redaction cap

    with caplog.at_level(logging.DEBUG, logger="src.agents.causal_impact.agent"):
        await agent.query_semantic_memory(long_query)

    records = [r.getMessage() for r in caplog.records if "Semantic memory query" in r.getMessage()]
    assert records, "expected the semantic-memory DEBUG log to be emitted"
    logged = records[0]

    assert long_query not in logged, "the full untruncated query must not be logged"
    assert redact_query(long_query) in logged, "the log must contain the redacted query"
