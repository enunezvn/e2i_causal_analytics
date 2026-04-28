"""Shared helpers for Feast integration tests.

Centralises a single source of truth for the FEAST_INTEGRATION opt-in
gate, the FV → PushSource naming convention, and the schema-deep
proto-byte diff helpers so the Feast integration test files (offline /
online parity, tier0 auto-register, Block 6B integration suite, apply
idempotency) cannot drift.
"""

from __future__ import annotations

import os
from datetime import timedelta
from typing import Any

# Suffix appended to a FeatureView name to form its auto-generated
# PushSource name, mirroring the construction in
# ``FeastClient.register_feature_view`` (single source of truth: any
# change to the suffix here MUST also change the FeastClient or the
# integration tests will go red).
PUSH_SOURCE_SUFFIX = "_push_source"

# Default entity for synthetic FVs built in-memory by the schema-deep
# proto-byte diff tests. Mirrors the live-confirmed registry name on the
# droplet (see ``test_feast_integration_suite`` and
# ``test_feast_tier0_auto_register``) so an in-memory FV, if ever applied,
# would slot in next to the real ones rather than introduce a parallel
# ``hcp`` definition.
DEFAULT_ENTITY_NAME = "hcp"
DEFAULT_ENTITY_JOIN_KEY = "hcp_id"


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


def proto_bytes(fv: Any) -> bytes:
    """Return a stable byte-string for a FeatureView proto.

    Direct ``SerializeToString()`` is sufficient for our drift-detection
    contract: Feast's proto definition is deterministic for the fields we
    perturb (ttl, schema, source name). We do NOT use ``SerializeToString(
    deterministic=True)`` because Feast 0.43's wrapper chain doesn't expose
    that kwarg uniformly across proto versions.

    Shared single source of truth for both the schema-deep idempotency
    suite (``test_feast_integration_suite``) and the apply-idempotency
    gate (``test_feast_apply_idempotent``); a future drift in serialisation
    semantics surfaces in one place.
    """
    return bytes(fv.to_proto().SerializeToString())


def build_minimal_feature_view(
    name: str,
    *,
    ttl: timedelta,
    feature_names: list[str],
    source_name: str | None = None,
    entity_name: str = DEFAULT_ENTITY_NAME,
    entity_join_key: str = DEFAULT_ENTITY_JOIN_KEY,
) -> Any:
    """Build a FeatureView in memory (no apply) for proto-byte comparison.

    Mirrors the construction in ``FeastClient.register_feature_view`` but
    skips the apply call so the caller can compute
    ``to_proto().SerializeToString()`` on a hypothetical-but-not-yet-applied
    FV. Used by both the schema-deep idempotency suite (which exercises
    multiple drift cases) and the apply-idempotency gate's deliberate-
    failure verification (a single TTL flip).

    The ``source_name`` argument perturbs the **outer** ``PushSource``
    name (the source the FV directly attaches to), NOT the inner
    ``FileSource`` (the throwaway stub backing the PushSource). This is
    the layer that matters for the source-rename drift test because Feast
    serialises the PushSource name into the FV proto; the FileSource
    identity rides along but isn't the perturbation surface here. When
    ``source_name`` is ``None`` the PushSource follows the canonical
    ``{name}_push_source`` convention shared with
    ``FeastClient.register_feature_view`` (and the
    ``push_source_name`` helper above).

    All ``Field``s are typed ``Float64``: the schema-deep tests perturb
    schema membership, not dtypes (a dtype-flip test would use a separate
    builder). The ``entity_name`` / ``entity_join_key`` defaults match
    the live registry so callers usually omit them; tests that need to
    perturb entity wiring can override them explicitly.
    """
    from feast import Entity, FeatureView, Field, FileSource, PushSource
    from feast.types import Float64

    schema = [Field(name=fn, dtype=Float64) for fn in feature_names]
    push_src = source_name if source_name is not None else push_source_name(name)

    # We build a stub batch source ref — we do NOT apply this FV, so the
    # batch source identity doesn't have to round-trip through Feast. The
    # proto bytes will differ if and only if the configured stub source name
    # differs, which is the contract we want for the source-rename test.
    batch_source = FileSource(
        name=f"{name}_batch_stub",
        path=f"/tmp/{name}_stub.parquet",
        timestamp_field="event_timestamp",
    )
    push_source = PushSource(name=push_src, batch_source=batch_source)

    entity = Entity(name=entity_name, join_keys=[entity_join_key])

    return FeatureView(
        name=name,
        entities=[entity],
        ttl=ttl,
        schema=schema,
        source=push_source,
        online=True,
    )
