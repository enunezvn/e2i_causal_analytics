"""Issue #215 regression test: experiment_designer.graph must NOT
monkey-patch asyncio.run at import time or graph-construction time.

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

This file pins both halves of the contract via subprocess-isolation:
each assertion runs in a fresh Python interpreter so the global asyncio
module state from prior test imports doesn't mask a regression.
"""

from __future__ import annotations

import os
import subprocess
import sys

import pytest


# create_experiment_designer_graph() instantiates LLM-backed nodes whose
# constructors raise ValueError when OPENAI_API_KEY / ANTHROPIC_API_KEY
# are unset. We supply fake non-empty values in the subprocess env so the
# factory call can proceed past the env-var preconditions and we can
# assert the asyncio.run invariant. The values do not need to be valid;
# nothing in this test actually contacts an LLM provider.
_FAKE_API_ENV = {
    **os.environ,
    "OPENAI_API_KEY": os.environ.get("OPENAI_API_KEY", "sk-fake-issue-215-regression-test"),
    "ANTHROPIC_API_KEY": os.environ.get(
        "ANTHROPIC_API_KEY", "sk-ant-fake-issue-215-regression-test"
    ),
}


@pytest.mark.integration
def test_import_graph_module_does_not_monkey_patch_asyncio_run() -> None:
    """Importing src.agents.experiment_designer.graph must NOT call
    ``nest_asyncio.apply()``.

    Subprocess isolation: a fresh interpreter records the identity of
    ``asyncio.run`` BEFORE the import, then re-checks identity AFTER
    the import. If the module monkey-patched asyncio.run (the issue
    #215 regression), the assertion fails with a clear message.
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
        env=_FAKE_API_ENV,
    )
    assert result.returncode == 0, (
        f"issue #215 regression detected. stdout={result.stdout!r} "
        f"stderr={result.stderr!r}"
    )


@pytest.mark.integration
def test_calling_create_experiment_designer_graph_does_not_monkey_patch_asyncio_run() -> (
    None
):
    """Calling ``create_experiment_designer_graph()`` must NOT trigger
    ``nest_asyncio.apply()`` at graph construction time.

    This is the same defense at a deeper level: even if a caller
    explicitly invokes the factory (rather than just importing the
    module), the resulting graph construction should leave
    ``asyncio.run`` untouched. ``nest_asyncio.apply()`` is deferred
    to the actually-nested execution path inside ``sync_wrapper``.
    """
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import asyncio\n"
                "orig_run = asyncio.run\n"
                "orig_module = asyncio.run.__module__\n"
                "from src.agents.experiment_designer.graph import (\n"
                "    create_experiment_designer_graph,\n"
                ")\n"
                "graph = create_experiment_designer_graph()\n"
                "assert graph is not None\n"
                "assert asyncio.run is orig_run, (\n"
                "    'issue #215 regression: create_experiment_designer_graph '\n"
                "    f'replaced asyncio.run at construction time (was '\n"
                "    f'module={orig_module!r}, now '\n"
                "    f'module={asyncio.run.__module__!r})'\n"
                ")\n"
            ),
        ],
        capture_output=True,
        text=True,
        timeout=120,
        env=_FAKE_API_ENV,
    )
    assert result.returncode == 0, (
        f"issue #215 regression detected. stdout={result.stdout!r} "
        f"stderr={result.stderr!r}"
    )
