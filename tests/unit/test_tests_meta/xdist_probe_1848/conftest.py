"""Induce the #1848 collection-phase worker crash -- only when asked to.

Inert unless :data:`ENV_CRASH` is set (see the package docstring). With
``E2I_1848_PROBE_CRASH=collection``:

* the controller creates the :data:`ENV_SENTINEL` file when it has recorded
  ``gw1``'s collection (``pytest_xdist_node_collection_finished``, which
  ``DSession.worker_collectionfinish`` fires right before it sets
  ``session.testscollected``);
* ``gw0`` finishes its own collection, waits for that file, and then
  ``os._exit(1)`` -- exactly what ``pytest_timeout``'s thread method does to a
  worker -- BEFORE its ``collectionfinish`` event is sent (``tryfirst`` puts
  this hook ahead of xdist's ``WorkerInteractor.pytest_collection_finish``).

That ordering is the one that yielded rc=0 with zero tests run on HEAD:
``testscollected`` is already 2 from the peer, no item was running so xdist
synthesises no failed report, and pytest's ``_main`` returns ``None`` -> 0.
"""

from __future__ import annotations

import os
import sys
import time

import pytest

from . import ENV_CRASH, ENV_SENTINEL, MODE_COLLECTION

_MODE = os.environ.get(ENV_CRASH, "")
_WORKER = os.environ.get("PYTEST_XDIST_WORKER", "")
_SENTINEL = os.environ.get(ENV_SENTINEL, "")

#: Upper bound on the wait for the peer's collection. Past it gw0 crashes
#: anyway (the crash-first ordering, rc=5 on HEAD) rather than hanging.
_PEER_WAIT_SECONDS = 60.0


class _ControllerSentinel:
    """Registered on the controller only, and only when xdist's hookspecs are
    present, so the xdist-specific hook never trips pluggy's unknown-hook
    validation on a ``-p no:xdist`` run of this directory. The hookspec is
    probed rather than ``hasplugin("xdist")`` because the plugin's registered
    name depends on how it was loaded (``"xdist.plugin"`` under
    ``PYTEST_PLUGINS``, which is how the nested sessions load it)."""

    def pytest_xdist_node_collection_finished(self, node: object, ids: object) -> None:
        gateway = getattr(node, "gateway", None)
        if getattr(gateway, "id", None) == "gw1" and _SENTINEL:
            with open(_SENTINEL, "w") as fh:
                fh.write("gw1 collection recorded by the controller\n")


def pytest_configure(config: pytest.Config) -> None:
    if _MODE != MODE_COLLECTION:
        return
    if hasattr(config, "workerinput"):
        return
    if not hasattr(config.pluginmanager.hook, "pytest_xdist_node_collection_finished"):
        return
    config.pluginmanager.register(_ControllerSentinel(), name="e2i-1848-probe-sentinel")


@pytest.hookimpl(tryfirst=True)
def pytest_collection_finish(session: pytest.Session) -> None:
    if _MODE != MODE_COLLECTION or _WORKER != "gw0":
        return
    deadline = time.monotonic() + _PEER_WAIT_SECONDS
    while _SENTINEL and not os.path.exists(_SENTINEL) and time.monotonic() < deadline:
        time.sleep(0.05)
    sys.stderr.write(f"[1848-probe] {_WORKER} dying before it reports its collection\n")
    sys.stderr.flush()
    os._exit(1)
