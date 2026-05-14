"""Issue #215 regression test: experiment_designer.graph must NOT
monkey-patch asyncio.run at import time.

Root cause (pinned 2026-05-14): `src/agents/experiment_designer/graph.py`
previously evaluated a module-level singleton

    experiment_designer_graph = create_experiment_designer_graph()

at import time. The factory called ``wrap_async_node`` which in turn
called ``nest_asyncio.apply()`` — a PROCESS-WIDE monkey-patch of
``asyncio.run``. Once any test on an xdist worker imported the module
(directly or via ``src.agents.experiment_designer``), every subsequent
``asyncio.run(coro)`` on that worker routed through ``nest_asyncio.run``,
which calls ``loop.create_task(...)`` on the currently-tracked
pytest-asyncio per-test loop. Between tests that loop closes, so the
next test's ``asyncio.run`` got ``RuntimeError: Event loop is closed``.

Fix (issue #215):
1. Removed the eager module-level singleton.
2. Moved ``nest_asyncio.apply()`` from ``wrap_async_node`` construction
   into the ``loop.is_running()`` branch of ``sync_wrapper``, where it
   is actually needed for ``loop.run_until_complete`` to nest.

This file pins the IMPORT invariant via subprocess-isolation: a fresh
interpreter records ``asyncio.run`` identity BEFORE and AFTER the import.
If the module re-introduces an eager nest_asyncio.apply() at import
time (the actual reported regression), the test fails with a clear
message.

A deeper factory-call invariant test was considered but dropped because
constructing the full graph in subprocess inside xdist workers crashes
the workers (heavy LLM-client instantiation). The import-time test
catches the actually-reported regression class; a factory-call test
would be defense-in-depth but is too expensive to run in the integration
suite.
"""

from __future__ import annotations

import subprocess
import sys

import pytest


@pytest.mark.integration
def test_import_graph_module_does_not_monkey_patch_asyncio_run() -> None:
    """Importing src.agents.experiment_designer.graph must NOT call
    ``nest_asyncio.apply()``.

    Subprocess isolation: a fresh interpreter records the identity of
    ``asyncio.run`` BEFORE the import, then re-checks identity AFTER
    the import. If the module monkey-patched asyncio.run (the issue
    #215 regression), the assertion fails with a clear message.

    Note: the test imports only the bare module (no factory call), so
    no API-key env vars are required. The import-time monkey-patch
    chain runs before any LLM-client constructor.
    """
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import asyncio\n"
                "orig_run = asyncio.run\n"
                "orig_module = asyncio.run.__module__\n"
                "import src.agents.experiment_designer.graph\n"
                "assert asyncio.run is orig_run, (\n"
                "    'issue #215 regression: importing experiment_designer.graph '\n"
                "    f'replaced asyncio.run (was module={orig_module!r}, '\n"
                "    f'now module={asyncio.run.__module__!r})'\n"
                ")\n"
            ),
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, (
        f"issue #215 regression detected. stdout={result.stdout!r} stderr={result.stderr!r}"
    )
