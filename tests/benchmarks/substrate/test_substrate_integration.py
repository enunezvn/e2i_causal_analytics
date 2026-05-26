import asyncio
import os

import pytest

from tests.benchmarks.substrate.direct_sql_connector import DirectSQLMemoryConnector
from tests.benchmarks.substrate.fixture import substrate_ready

pytestmark = pytest.mark.skipif(
    not substrate_ready(),
    reason="local pg substrate not configured (set BENCH_SUBSTRATE/BENCH_PG_DSN)",
)


def test_vector_stream_returns_nonempty():
    conn = DirectSQLMemoryConnector(os.environ["BENCH_PG_DSN"])
    try:
        results = asyncio.run(conn.vector_search_by_text("kisqali trx growth west region q3", k=10))
    finally:
        conn.close()
    assert results, "vector stream returned empty against a seeded substrate"


def test_fulltext_stream_returns_nonempty():
    conn = DirectSQLMemoryConnector(os.environ["BENCH_PG_DSN"])
    try:
        results = asyncio.run(conn.fulltext_search("kisqali trx growth", k=10))
    finally:
        conn.close()
    assert results, "full-text stream returned empty against a seeded substrate"


def test_broken_substrate_fails_loud():
    """A bad DSN must RAISE, never silently return [] (the #403 failure mode)."""
    with pytest.raises(Exception):
        DirectSQLMemoryConnector("postgresql://nobody@127.0.0.1:1/none")
