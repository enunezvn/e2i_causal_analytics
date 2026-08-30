"""Crash probe for the #1848 xdist vacuous-green guard.

This directory is collected by the ``Unit Tests`` lane like any other: two
trivial tests that pass, and a ``conftest.py`` that is INERT unless
:data:`ENV_CRASH` is set. ``test_xdist_vacuous_green_1848.py`` sets it and runs
a *nested* pytest against this directory, so the induced worker crash happens
under the repo's real ``tests/conftest.py`` and the lane's exact xdist shape
(``-n 2 --dist=loadscope --timeout=30``) -- the guard is exercised through the
same wiring CI relies on, not through a copy of it.

The names live here, in a plain package module, so the test module can import
them without importing the conftest.
"""

#: Set to :data:`MODE_COLLECTION` to make worker ``gw0`` die during collection.
ENV_CRASH = "E2I_1848_PROBE_CRASH"

#: Path of a file the CONTROLLER creates the moment it has recorded ``gw1``'s
#: collection. ``gw0`` waits for it before dying, which pins the ordering that
#: produced the vacuous green: the peer's collection is already counted when
#: the crash lands.
ENV_SENTINEL = "E2I_1848_PROBE_SENTINEL"

MODE_COLLECTION = "collection"
