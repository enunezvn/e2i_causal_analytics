"""Discovery tests provoke latent-diagnostic timeouts on purpose; the abandoned
daemon thread (#2233) must not leak into the next test, where the process-wide
one-outstanding-diagnostic bound would skip its diagnostic."""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _join_outlived_diagnostic_threads():
    yield
    from src.causal_engine.discovery import runner as runner_mod

    runner_mod.join_outlived_diagnostic_threads(timeout=5.0)
