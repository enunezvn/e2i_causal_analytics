"""Clear the Feast Redis dedup marker for ONE online view with any join keys (canonical TRx lane).

Migration 144 renamed hcp_conversion_features' fields. A re-materialize writes the
renamed fields with the SAME event timestamps the old fields were written with, and
Feast's Redis store skips every column of a write whose event time is not newer than
the view's ``_ts:<view>`` marker (see clear_goldstd_ts_markers.py) — ``feast
materialize`` exits 0 while the renamed fields are never written. That tool is scoped
to the two goldstd views and one join key; hcp_conversion_features keys on hcp_id AND
hcp_brand_id. This tool reuses its tested batching and Redis helpers and builds
composite entity keys, for a bounded allowlist of views.

Verified against feast 0.43.0 (``feast/infra/online_stores/redis.py``): the entity
hash key comes from the join keys alone (``_redis_key``), so a FIELD rename leaves it
unchanged; the marker is ``_ts:<view>`` (:285) and the skip is
``event_time_seconds <= prev_ts.seconds`` (:313). Each feature occupies hash field
``_mmh3(f"{view}:{field}")`` (:325), read back under the CURRENT names (:359) — so the
old-named hash fields survive the rename as unreferenced orphans. This tool does NOT
remove them: it clears only the dedup marker, which is what unblocks the write.

Invoke inside the e2i_feast sidecar:
    docker exec e2i_feast python /feast-src/clear_view_ts_markers.py --view hcp_conversion_features --dry-run
    docker exec e2i_feast python /feast-src/clear_view_ts_markers.py --view hcp_conversion_features
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from typing import Any, Callable, List, Optional, Sequence, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from clear_goldstd_ts_markers import (  # noqa: E402
    DEFAULT_BATCH_SIZE,
    DEFAULT_REPO_PATH,
    _load_store,
    _redis_client_from_store,
    clear_view_markers,
    ts_marker_field,
)

logger = logging.getLogger(__name__)

#: Views this lane may clear — bounded on purpose (never "all").
ALLOWED_VIEWS: Tuple[str, ...] = ("hcp_conversion_features",)


def composite_ids_query(join_keys: Sequence[str], source_sql: str) -> str:
    """DISTINCT join-key tuples from the view's own offline source."""
    columns = ", ".join(f"({key})::text" for key in join_keys)
    not_null = " AND ".join(f"{key} IS NOT NULL" for key in join_keys)
    return f"SELECT DISTINCT {columns} FROM {source_sql} AS _src WHERE {not_null}"


def composite_key_builder(
    join_keys: Sequence[str], make_key: Callable[[Sequence[str], Sequence[str]], bytes]
) -> Callable[[Tuple[str, ...]], bytes]:
    keys = list(join_keys)

    def build(values: Tuple[str, ...]) -> bytes:
        if len(values) != len(keys):
            raise ValueError(f"expected {len(keys)} join values, got {len(values)}")
        return make_key(keys, [str(v) for v in values])

    return build


def _feast_key_maker(store: Any) -> Callable[[Sequence[str], Sequence[str]], bytes]:
    from feast.infra.online_stores.redis import _redis_key
    from feast.protos.feast.types.EntityKey_pb2 import EntityKey as EntityKeyProto
    from feast.protos.feast.types.Value_pb2 import Value as ValueProto

    project = store.config.project
    version = store.config.entity_key_serialization_version

    def make(join_keys: Sequence[str], values: Sequence[str]) -> bytes:
        entity_key = EntityKeyProto(
            join_keys=list(join_keys), entity_values=[ValueProto(string_val=v) for v in values]
        )
        key: bytes = _redis_key(project, entity_key, entity_key_serialization_version=version)
        return key

    return make


def _fetch_tuples(store: Any, feature_view: Any, join_keys: Sequence[str]) -> List[Tuple[str, ...]]:
    import psycopg

    off = store.config.offline_store
    dsn = (
        f"host={off.host} port={off.port} dbname={off.database} user={off.user} "
        f"password={getattr(off, 'password', None)} sslmode={getattr(off, 'sslmode', 'prefer')}"
    )
    query = composite_ids_query(join_keys, feature_view.batch_source.get_table_query_string())
    with psycopg.connect(dsn, connect_timeout=30) as conn, conn.cursor() as cur:
        cur.execute(query)
        return [tuple(row) for row in cur.fetchall()]


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument("--repo-path", default=DEFAULT_REPO_PATH)
    parser.add_argument("--view", choices=ALLOWED_VIEWS, required=True)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    store = _load_store(args.repo_path)
    feature_view = store.get_feature_view(args.view)
    join_keys = list(feature_view.join_keys)
    tuples = _fetch_tuples(store, feature_view, join_keys)
    key_for = composite_key_builder(join_keys, _feast_key_maker(store))
    hit, absent = clear_view_markers(
        _redis_client_from_store(store),
        key_for,
        tuples,
        ts_marker_field(args.view),
        batch_size=args.batch_size,
        dry_run=args.dry_run,
    )
    logger.info(
        "%s [join_keys=%s]: %d entity tuples, %s %d markers, %d had no marker",
        args.view,
        join_keys,
        len(tuples),
        "would clear" if args.dry_run else "cleared",
        hit,
        absent,
    )
    return 0 if tuples else 1


if __name__ == "__main__":
    raise SystemExit(main())
