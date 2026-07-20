"""Unit tests for the #1296 Feast _ts marker-clear tooling.

Covers the PURE logic of ``feature_repo/clear_goldstd_ts_markers.py`` — which is
importable in CI WITHOUT feast/redis/psycopg because the module keeps every one
of those imports function-local — and a doc-drift guard on
``scripts/sync_goldstd_serving.py``: the marker-clear step must be sequenced
BEFORE the FULL materialize in both the module docstring's "Usage (in order)"
block and the operator steps ``main()`` prints. All redis behaviour is exercised
through a fake pipeline; no docker, redis, feast server, or DB is touched.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pytest

# feature_repo/ is not a package (no __init__.py); follow the existing
# tests/unit/test_feature_repo convention and put the dir on sys.path so
# ``import clear_goldstd_ts_markers`` resolves. The module's feast/redis/psycopg
# imports are function-local, so this import needs none of them installed.
_REPO_ROOT = Path(__file__).resolve().parents[3]
_FEATURE_REPO = _REPO_ROOT / "feature_repo"
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_FEATURE_REPO) not in sys.path:
    sys.path.insert(0, str(_FEATURE_REPO))

import clear_goldstd_ts_markers as ctm  # type: ignore[import-not-found]  # noqa: E402

import scripts.sync_goldstd_serving as sync  # noqa: E402

MARKER = b"_ts:goldstd_cohort_features"


def _key(entity_id: str) -> bytes:
    """Deterministic fake redis-key builder standing in for the real _redis_key."""
    return f"k:{entity_id}".encode()


class _FakePipeline:
    """Records hdel/hexists ops; ``execute()`` returns per-op 1/0 from a shared marker set.

    hdel additionally discards a present marker so a second run over the same
    fake client observes the idempotent (already-absent) outcome.
    """

    def __init__(self, present: set, transaction: bool):
        self._present = present
        self.transaction = transaction
        self.ops: list[tuple[str, bytes, bytes]] = []

    def hdel(self, key: bytes, field: bytes) -> None:
        self.ops.append(("hdel", key, field))

    def hexists(self, key: bytes, field: bytes) -> None:
        self.ops.append(("hexists", key, field))

    def execute(self) -> list[int]:
        results: list[int] = []
        for op, key, field in self.ops:
            has = (key, field) in self._present
            if op == "hdel" and has:
                self._present.discard((key, field))
            results.append(1 if has else 0)
        return results


class _FakeRedis:
    def __init__(self, present: set):
        self._present = present
        self.pipelines: list[_FakePipeline] = []

    def pipeline(self, transaction: bool = False) -> _FakePipeline:
        pipe = _FakePipeline(self._present, transaction)
        self.pipelines.append(pipe)
        return pipe


# ---------------------------------------------------------------------------
# ts_marker_field — exact per-view field naming
# ---------------------------------------------------------------------------


def test_ts_marker_field_exact_bytes():
    assert ctm.ts_marker_field("goldstd_cohort_features") == b"_ts:goldstd_cohort_features"
    assert ctm.ts_marker_field("goldstd_hcp_cohort_features") == b"_ts:goldstd_hcp_cohort_features"
    assert isinstance(ctm.ts_marker_field("anything"), bytes)


# ---------------------------------------------------------------------------
# parse_redis_connection_string — mirrors feast's parser
# ---------------------------------------------------------------------------


def test_parse_host_port_password():
    assert ctm.parse_redis_connection_string("redis:6379,password=changeme") == (
        "redis",
        "6379",
        {"password": "changeme"},
    )


def test_parse_json_decodes_db_and_ssl():
    host, port, params = ctm.parse_redis_connection_string("redis:6379,db=0,ssl=true,password=x")
    assert (host, port) == ("redis", "6379")
    assert params == {"db": 0, "ssl": True, "password": "x"}


def test_parse_password_with_equals_kept_raw():
    # A password containing '=' is not valid JSON, so it survives verbatim
    # (partition on the FIRST '=' keeps the rest of the value intact).
    _, _, params = ctm.parse_redis_connection_string("redis:6379,password=a=b=c")
    assert params["password"] == "a=b=c"


def test_parse_no_node_raises():
    with pytest.raises(ValueError):
        ctm.parse_redis_connection_string("password=x")


# ---------------------------------------------------------------------------
# clear_view_markers — batching + cleared/absent accounting + dry-run
# ---------------------------------------------------------------------------


def test_counts_cleared_and_absent():
    present = {(_key("a"), MARKER), (_key("c"), MARKER)}
    client = _FakeRedis(present)
    hit, absent = ctm.clear_view_markers(client, _key, ["a", "b", "c"], MARKER, batch_size=10)
    assert (hit, absent) == (2, 1)


def test_batches_by_batch_size():
    client = _FakeRedis(set())
    ids = [str(i) for i in range(5)]
    ctm.clear_view_markers(client, _key, ids, MARKER, batch_size=2)
    # 5 ids @ batch 2 => three pipelines of sizes 2, 2, 1.
    assert [len(p.ops) for p in client.pipelines] == [2, 2, 1]


def test_only_touches_the_marker_field():
    present = {(_key("a"), MARKER)}
    client = _FakeRedis(present)
    ctm.clear_view_markers(client, _key, ["a", "b"], MARKER, batch_size=10)
    ops = [op for pipe in client.pipelines for op in pipe.ops]
    assert ops, "expected recorded ops"
    assert all(kind == "hdel" for kind, _, _ in ops)
    # Never widens beyond the exact _ts marker field.
    assert all(field == MARKER for _, _, field in ops)
    assert {key for _, key, _ in ops} == {_key("a"), _key("b")}


def test_dry_run_uses_hexists_and_deletes_nothing():
    present = {(_key("a"), MARKER), (_key("b"), MARKER)}
    client = _FakeRedis(present)
    hit, absent = ctm.clear_view_markers(
        client, _key, ["a", "b", "c"], MARKER, batch_size=10, dry_run=True
    )
    ops = [op for pipe in client.pipelines for op in pipe.ops]
    assert ops and all(kind == "hexists" for kind, _, _ in ops)  # NO hdel issued
    assert (hit, absent) == (2, 1)
    assert present == {(_key("a"), MARKER), (_key("b"), MARKER)}  # untouched


def test_idempotent_second_run_all_absent():
    present = {(_key("a"), MARKER), (_key("b"), MARKER)}
    client = _FakeRedis(present)
    first = ctm.clear_view_markers(client, _key, ["a", "b"], MARKER, batch_size=10)
    second = ctm.clear_view_markers(client, _key, ["a", "b"], MARKER, batch_size=10)
    assert first == (2, 0)
    assert second == (0, 2)  # markers already gone — safe to re-run


def test_empty_ids_issues_no_pipeline():
    client = _FakeRedis(set())
    hit, absent = ctm.clear_view_markers(client, _key, [], MARKER, batch_size=10)
    assert (hit, absent) == (0, 0)
    assert client.pipelines == []


# ---------------------------------------------------------------------------
# Doc-drift guard on scripts/sync_goldstd_serving.py
# ---------------------------------------------------------------------------


def test_sync_docstring_sequences_clear_before_materialize():
    doc = sync.__doc__ or ""
    usage = doc[doc.index("Usage (in order)") :]
    assert "clear_goldstd_ts_markers.py" in usage
    assert usage.index("clear_goldstd_ts_markers.py") < usage.index(
        "feast --chdir /feast materialize"
    )


def test_sync_docstring_documents_same_day_hazard():
    doc = (sync.__doc__ or "").lower()
    assert "1296" in doc
    assert "same-day" in doc
    assert "_ts:" in doc
    assert "day-granular" in doc


def test_printed_operator_steps_clear_before_materialize(monkeypatch, caplog):
    # --skip-bundles takes main() straight to the printed operator steps with no
    # bundle re-materialize, feast, redis, or DB — a hermetic path.
    monkeypatch.setattr(sys, "argv", ["sync_goldstd_serving", "--skip-bundles"])
    with caplog.at_level(logging.INFO, logger="scripts.sync_goldstd_serving"):
        rc = sync.main()
    assert rc == 0
    msgs = [r.getMessage() for r in caplog.records if "Bundles done" in r.getMessage()]
    assert msgs, "expected the 'Bundles done' operator-steps message"
    msg = msgs[0]
    assert "clear_goldstd_ts_markers.py" in msg
    assert msg.index("clear_goldstd_ts_markers.py") < msg.index("feast --chdir /feast materialize")
