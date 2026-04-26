"""Shared helpers for Feast integration tests.

Centralises a single source of truth for the FEAST_INTEGRATION opt-in
gate and the FV → PushSource naming convention so the three Feast
integration test files (offline/online parity, tier0 auto-register,
Block 6B integration suite) cannot drift.
"""

from __future__ import annotations

import os

# Suffix appended to a FeatureView name to form its auto-generated
# PushSource name, mirroring the construction in
# ``FeastClient.register_feature_view`` (single source of truth: any
# change to the suffix here MUST also change the FeastClient or the
# integration tests will go red).
PUSH_SOURCE_SUFFIX = "_push_source"


def feast_integration_available() -> bool:
    """True iff the caller has opted into the live Feast integration suite.

    The droplet (and only the droplet) sets ``FEAST_INTEGRATION=1`` in its
    environment so these tests run there but stay a no-op everywhere else.
    """
    return os.environ.get("FEAST_INTEGRATION", "").strip().lower() in {
        "1",
        "true",
        "yes",
    }


def push_source_name(fv_name: str) -> str:
    """Return the auto-generated PushSource name for a FeatureView.

    Mirrors the convention applied by
    ``FeastClient.register_feature_view``: ``{fv_name}_push_source``.
    """
    return f"{fv_name}{PUSH_SOURCE_SUFFIX}"
