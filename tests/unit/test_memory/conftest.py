"""Hermetic-unit guard for ``tests/unit/test_memory``.

Production code in this subsystem performs best-effort live-DB side-effects — e.g.
``CognitiveService._persist_cognitive_cycle`` writes the parent ``cognitive_cycles``
row (restored mig 042). Unit tests must NOT touch the real Supabase: a dev box with
Supabase reachable would otherwise have these best-effort writes pollute the live
tables, diverging from CI (where no DB is attached and the writes simply skip).

This autouse fixture stubs the Supabase factory so those best-effort writes skip in
unit tests — making local behaviour match CI. Faithful real-DB coverage of the
producer lives in
``tests/integration/test_memory/test_cognitive_cycle_persistence.py`` (self-cleaning).
"""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _hermetic_no_live_supabase(monkeypatch: pytest.MonkeyPatch) -> None:
    from src.memory.services import factories

    def _disabled(*_args: object, **_kwargs: object) -> object:
        raise factories.ServiceConnectionError(
            "Supabase", "disabled in tests/unit/test_memory (hermetic unit guard)"
        )

    # The cognitive_cycles producer imports get_supabase_client lazily at call time,
    # so patching the module attribute makes its best-effort write skip cleanly.
    monkeypatch.setattr(factories, "get_supabase_client", _disabled, raising=False)
