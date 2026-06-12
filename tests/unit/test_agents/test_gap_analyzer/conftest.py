"""Shared fixtures for gap_analyzer unit tests."""

import pytest


@pytest.fixture(autouse=True)
def _hermetic_memory_layer(monkeypatch):
    """Keep unit tests off the live memory DB (#883).

    Before #883 the gap_analyzer episodic write ALWAYS failed against the live
    enums (22P02 swallowed into ``logger.warning`` -> None), so this suite was
    de-facto hermetic even where agent/graph tests let the formatter's
    contribute_to_memory fire unmocked. Now that the write actually lands, an
    unmocked run from a creds-configured dev box inserts REAL rows into the
    live DB mid-unit-run. Simulate the "memory unavailable" baseline instead:
    the source-module functions raise, the hooks swallow and return None/[],
    matching the suite's historical expectations. Tests exercising a specific
    memory behavior patch these symbols themselves (an inner ``with patch``
    wins over this fixture).
    """

    async def _unavailable(*_args, **_kwargs):
        raise RuntimeError(
            "hermetic unit-test memory layer (#883): patch the memory function "
            "explicitly if the test needs a specific behavior"
        )

    import src.memory.episodic_memory as episodic_memory
    import src.memory.procedural_memory as procedural_memory

    monkeypatch.setattr(episodic_memory, "insert_episodic_memory_with_text", _unavailable)
    monkeypatch.setattr(episodic_memory, "search_episodic_by_text", _unavailable)
    monkeypatch.setattr(procedural_memory, "insert_procedural_memory_with_text", _unavailable)
    monkeypatch.setattr(procedural_memory, "find_relevant_procedures_by_text", _unavailable)
