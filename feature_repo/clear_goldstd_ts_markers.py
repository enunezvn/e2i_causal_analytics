"""Clear the Feast Redis dedup markers on the gold-standard ONLINE views (#1296).

THE SAME-DAY-TIE HAZARD THIS CLOSES
-----------------------------------
Feast's Redis online store keeps, per entity hash, a ``_ts:<view>`` field
holding the last-written ``event_timestamp``. On write it SKIPS a column when
the incoming event time is not strictly newer than that marker
(``feast/infra/online_stores/redis.py``: ``if prev_ts.seconds and
event_time_seconds <= prev_ts.seconds: continue`` — ``created_timestamp`` is
discarded, so a fresher write with an equal event time is silently dropped).

``goldstd_cohort_features`` derives ``event_timestamp`` from
``event_date::TIMESTAMPTZ`` (``patient_journeys``) — a DATE cast to midnight,
i.e. DAY-GRANULAR — so TWO reseeds on the SAME calendar day produce
byte-identical event times: the ``<=`` branch fires and the new/changed columns
are NEVER written to the online store, while ``feast materialize`` still exits
0. A same-day re-reseed therefore silently no-ops the serving layer (degenerate
SHAP surface + the #576 null-trap 503). ``goldstd_hcp_cohort_features`` uses
``updated_at`` (``hcp_profiles``) — a real timestamptz, so it ties only when a
reseed rewrites rows WITHOUT bumping ``updated_at``; its marker is cleared here
too because the skip mechanism is identical and the clear is free + idempotent.

THE FIX (proven live for COMM-ARMS Phase 3, 2026-07-20)
-------------------------------------------------------
Surgically ``HDEL`` ONLY the ``_ts:<view>`` dedup marker on each entity hash,
then re-materialize. With the marker gone the next write is unconditionally
laid down (the current, full row). This touches NOTHING else in the shared
Redis: not the feature values, not other views' markers (the HCP hash also
carries ``_ts:hcp_profile_features`` and ``_ts:hcp_features`` — untouched), not
any non-Feast application key. It is non-destructive and idempotent, so it is
safe to run before EVERY FULL materialize.

WHERE THIS RUNS
---------------
Inside the ``e2i_feast`` sidecar only — the app/worker image cannot import feast
(#307). The compose bind mounts ``feature_repo/`` read-only at ``/feast-src`` and
the entrypoint COPIES the repo into the writable ``/feast`` layer (where the
rendered ``feature_store.yaml`` with the real Redis/Postgres secrets lives). So
this file is reachable at ``/feast-src/clear_goldstd_ts_markers.py`` while the
Feast config is read from ``repo_path=/feast``. Invoke::

    # dry-run first (counts markers, deletes nothing):
    docker exec e2i_feast python /feast-src/clear_goldstd_ts_markers.py --dry-run

    # then the real clear (both goldstd online views):
    docker exec e2i_feast python /feast-src/clear_goldstd_ts_markers.py

Entity IDs are pulled in-container straight from the SAME offline Postgres source
each view materializes from (DSN derived from the feast offline config), so the
cleared set matches exactly what the subsequent materialize will re-write. If
in-container DB access is unavailable, ``--ids-file`` supplies the IDs for a
single ``--view`` instead.
"""

from __future__ import annotations

import argparse
import json
import logging
from typing import Any, Callable, List, Sequence, Tuple

logger = logging.getLogger(__name__)

# The gold-standard ONLINE feature views whose markers a reseed must be able to
# clear. This is the deliberate SCOPE of the tool; the join key of each is
# derived from the feast registry at runtime (robust to a view rename) rather
# than hardcoded here.
GOLDSTD_ONLINE_VIEWS = ("goldstd_cohort_features", "goldstd_hcp_cohort_features")

# repo_path holds the rendered feature_store.yaml (secrets substituted in by the
# sidecar entrypoint) — NOT /feast-src (the read-only, un-rendered bind mount).
DEFAULT_REPO_PATH = "/feast"

# Pipeline width. 2000 matches the proven live clear; keeps each round-trip
# bounded while still amortising the per-command overhead across 25k+ keys.
DEFAULT_BATCH_SIZE = 2000


def ts_marker_field(view_name: str) -> bytes:
    """Exact Redis hash field Feast uses as the per-view dedup marker (``_ts:<view>``).

    Returned as bytes so the HDEL targets the precise field and can never be
    widened to the whole hash or another view's marker.
    """
    return f"_ts:{view_name}".encode("utf-8")


def parse_redis_connection_string(connection_string: str) -> Tuple[str, str, dict]:
    """Parse a Feast redis connection string into ``(host, port, kwargs)``.

    Mirrors ``RedisOnlineStore._parse_connection_string`` (feast 0.43.0): comma
    splits the string; a chunk WITHOUT ``=`` is a ``host:port`` node, a chunk
    WITH ``=`` is a client kwarg whose value is JSON-decoded (so ``db=0`` -> int
    ``0``, ``ssl=true`` -> ``True``) and left as the raw string when it is not
    valid JSON (``password=changeme``). We connect a single ``redis.Redis`` to
    the first node with those kwargs, exactly as feast does for a non-cluster
    online store — so we reach the SAME server Feast writes to, never a guess.
    """
    nodes: List[Tuple[str, str]] = []
    params: dict = {}
    for chunk in connection_string.split(","):
        if "=" in chunk:
            key, _, raw = chunk.partition("=")
            try:
                value: Any = json.loads(raw)
            except json.JSONDecodeError:
                value = raw
            params[key] = value
        else:
            host, _, port = chunk.partition(":")
            nodes.append((host, port))
    if not nodes:
        raise ValueError(
            f"no host:port node found in redis connection string: {connection_string!r}"
        )
    host, port = nodes[0]
    return host, port, params


def clear_view_markers(
    client: Any,
    key_for_id: Callable[[str], bytes],
    entity_ids: Sequence[str],
    marker_field: bytes,
    *,
    batch_size: int = DEFAULT_BATCH_SIZE,
    dry_run: bool = False,
) -> Tuple[int, int]:
    """Batched, pipelined HDEL of ONLY ``marker_field`` on each entity's hash.

    Returns ``(hit, absent)``. In a real run ``hit`` is the number of markers
    actually deleted (HDEL returned 1) and ``absent`` the keys that carried no
    marker (HDEL returned 0 — a missing key or an already-cleared field). In
    ``dry_run`` mode no field is deleted: HEXISTS is pipelined instead, so
    ``hit`` counts the markers that EXIST (would be deleted) and ``absent`` the
    rest. Both counts are derived from the pipeline results — never assumed — so
    a truncated or empty run cannot masquerade as a full one.
    """
    hit = 0
    absent = 0
    for start in range(0, len(entity_ids), batch_size):
        chunk = entity_ids[start : start + batch_size]
        pipe = client.pipeline(transaction=False)
        for entity_id in chunk:
            key = key_for_id(entity_id)
            if dry_run:
                pipe.hexists(key, marker_field)
            else:
                pipe.hdel(key, marker_field)
        for result in pipe.execute():
            if int(result):
                hit += 1
            else:
                absent += 1
    return hit, absent


def _load_store(repo_path: str) -> Any:
    """Construct the Feast FeatureStore; fail fast + loud if not in the sidecar.

    A missing feast import (the app image, #307) or a missing repo_path both mean
    this was launched outside e2i_feast — abort cleanly rather than half-run.
    """
    try:
        from feast import FeatureStore
    except ImportError as exc:  # pragma: no cover - exercised only outside the sidecar
        raise SystemExit(
            "clear_goldstd_ts_markers must run INSIDE the e2i_feast sidecar — the "
            "app/worker image cannot import feast (#307). Invoke it as:\n"
            "  docker exec e2i_feast python /feast-src/clear_goldstd_ts_markers.py\n"
            f"(feast import failed: {exc})"
        ) from exc

    import os

    if not os.path.isdir(repo_path):
        raise SystemExit(
            f"feast repo_path {repo_path!r} does not exist. Run inside e2i_feast, "
            "where the entrypoint renders feature_store.yaml into /feast."
        )
    return FeatureStore(repo_path=repo_path)


def _redis_client_from_store(store: Any) -> Any:
    """Build the redis client from ``store.config.online_store`` (no hardcoded creds)."""
    import redis

    connection_string = store.config.online_store.connection_string
    host, port, params = parse_redis_connection_string(connection_string)
    return redis.Redis(host=host, port=port, **params)


def _key_builder(store: Any, join_key: str) -> Callable[[str], bytes]:
    """Return an id -> Feast Redis-key function for ``join_key`` under this project."""
    from feast.infra.online_stores.redis import _redis_key
    from feast.protos.feast.types.EntityKey_pb2 import EntityKey as EntityKeyProto
    from feast.protos.feast.types.Value_pb2 import Value as ValueProto

    project = store.config.project
    eks_version = store.config.entity_key_serialization_version

    def build(entity_id: str) -> bytes:
        entity_key = EntityKeyProto(
            join_keys=[join_key],
            entity_values=[ValueProto(string_val=str(entity_id))],
        )
        # _redis_key is untyped (feast has no stubs) -> pin to bytes for the caller.
        key: bytes = _redis_key(project, entity_key, entity_key_serialization_version=eks_version)
        return key

    return build


def _fetch_ids_from_source(store: Any, feature_view: Any, join_key: str) -> List[str]:
    """DISTINCT join-key values from the SAME offline source the view materializes.

    DSN is derived from ``store.config.offline_store`` (not hardcoded). Wrapping
    the view's own ``get_table_query_string()`` as a subquery guarantees the
    cleared entity set is exactly the population Feast will re-write on the next
    materialize — no drift between the clear and the write.
    """
    import psycopg

    off = store.config.offline_store
    password = getattr(off, "password", None)
    dsn = (
        f"host={off.host} port={off.port} dbname={off.database} "
        f"user={off.user} password={password} sslmode={getattr(off, 'sslmode', 'prefer')}"
    )
    source_sql = feature_view.batch_source.get_table_query_string()
    query = (
        f"SELECT DISTINCT ({join_key})::text AS _id "
        f"FROM {source_sql} AS _src WHERE {join_key} IS NOT NULL"
    )
    with psycopg.connect(dsn, connect_timeout=30) as conn:
        with conn.cursor() as cur:
            cur.execute(query)
            return [row[0] for row in cur.fetchall()]


def _load_ids_file(path: str) -> List[str]:
    """Read newline-delimited entity IDs (blank lines ignored) — the fallback source."""
    with open(path) as handle:
        return [line.strip() for line in handle if line.strip()]


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(
        description=(
            "Clear the Feast Redis _ts:<view> dedup markers on the gold-standard "
            "online views so a same-day re-reseed propagates (#1296). Run inside "
            "the e2i_feast sidecar BEFORE the FULL materialize."
        )
    )
    parser.add_argument(
        "--repo-path",
        default=DEFAULT_REPO_PATH,
        help="Feast repo dir holding the rendered feature_store.yaml (default: /feast).",
    )
    parser.add_argument(
        "--view",
        choices=(*GOLDSTD_ONLINE_VIEWS, "all"),
        default="all",
        help="Which goldstd online view(s) to clear (default: all).",
    )
    parser.add_argument(
        "--ids-file",
        default=None,
        help=(
            "Fallback: newline-delimited entity IDs, used INSTEAD of the offline "
            "Postgres source. Requires a single --view (supplies IDs for one view)."
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Pipeline batch size (default: {DEFAULT_BATCH_SIZE}).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Count markers that exist (would be cleared) without deleting anything.",
    )
    args = parser.parse_args(argv)

    if args.ids_file and args.view == "all":
        parser.error("--ids-file requires a single --view (it supplies IDs for one view only)")

    views = GOLDSTD_ONLINE_VIEWS if args.view == "all" else (args.view,)

    store = _load_store(args.repo_path)
    client = _redis_client_from_store(store)

    total_hit = 0
    total_absent = 0
    had_error = False
    for view in views:
        feature_view = store.get_feature_view(view)
        join_key = feature_view.join_keys[0]
        if args.ids_file:
            entity_ids = _load_ids_file(args.ids_file)
            id_source = f"ids-file {args.ids_file}"
        else:
            entity_ids = _fetch_ids_from_source(store, feature_view, join_key)
            id_source = "offline postgres source"

        # NO SILENT CAPS: a post-reseed clear over 0 entities is never a success
        # (a reseed populates thousands). Surface it and fail rather than let an
        # empty run look clean.
        if not entity_ids:
            logger.error(
                "%s: 0 entity ids from %s — refusing to report success. Check the "
                "source query / ids-file before materializing.",
                view,
                id_source,
            )
            had_error = True
            continue

        marker_field = ts_marker_field(view)
        key_for_id = _key_builder(store, join_key)
        hit, absent = clear_view_markers(
            client,
            key_for_id,
            entity_ids,
            marker_field,
            batch_size=args.batch_size,
            dry_run=args.dry_run,
        )
        verb = "would clear" if args.dry_run else "cleared"
        logger.info(
            "%s [join_key=%s, field=%s, source=%s]: %d entity ids, %s %d markers, %d had no marker",
            view,
            join_key,
            marker_field.decode("utf-8"),
            id_source,
            len(entity_ids),
            verb,
            hit,
            absent,
        )
        total_hit += hit
        total_absent += absent

    verb = "would clear" if args.dry_run else "cleared"
    logger.info(
        "TOTAL across %d view(s): %s %d markers, %d absent%s",
        len(views),
        verb,
        total_hit,
        total_absent,
        "  [DRY RUN — nothing deleted]" if args.dry_run else "",
    )
    return 1 if had_error else 0


if __name__ == "__main__":
    raise SystemExit(main())
