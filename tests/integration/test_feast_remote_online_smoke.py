"""Faithful integration smoke for FeastClient remote mode (#532 Option 1).

The unit tests in ``tests/unit/test_feature_store/test_feast_client.py`` mock httpx,
so they validate request/response *handling* but never a real round-trip. This test
closes that gap: it spins up a REAL Feast feature server (``feast serve``) at the same
version the ``e2i_feast`` sidecar runs (``feastdev/feature-server:0.43.0``), seeds a
known online value, and exercises the FULL path through the unmodified
:class:`FeastClient` remote mode — request transpose -> real feature-server -> real
``MessageToDict`` response -> ``_remote_response_to_flat``.

Gated by ``pytest.importorskip("feast")``: the production app image does NOT ship feast
(feast 0.43.0 pins ``tenacity<9`` vs prod ``tenacity==9.1.2``), so this is SKIPPED in
the app-image / CI unit environment and only runs where feast is installed (a dev box,
or a feast-capable integration runner). It still does not reproduce the prod
redis-online-store / container / network — that remains a deploy-time sidecar smoke.
"""

from __future__ import annotations

import socket
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx
import pandas as pd
import pytest

feast = pytest.importorskip("feast")

from feast import Entity, FeatureView, Field, FileSource  # noqa: E402
from feast.types import Float32  # noqa: E402

from src.feature_store.feast_client import FeastClient, FeastConfig  # noqa: E402

PROJECT = "e2i_remote_smoke"
ENTITY_VALUE = 1001
EXPECTED_CONV_RATE = 0.42


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _write_feature_store_yaml(repo: Path) -> None:
    (repo / "feature_store.yaml").write_text(
        "\n".join(
            [
                f"project: {PROJECT}",
                "provider: local",
                "registry: data/registry.db",
                "online_store:",
                "    type: sqlite",
                "    path: data/online_store.db",
                "entity_key_serialization_version: 2",
                "",
            ]
        )
    )


@pytest.fixture(scope="module")
def feast_server(tmp_path_factory: pytest.TempPathFactory):
    """Apply a tiny repo, seed one online value, and run `feast serve`; yield base URL."""
    repo = tmp_path_factory.mktemp("feast_remote_smoke")
    (repo / "data").mkdir()
    _write_feature_store_yaml(repo)

    # A FileSource is required structurally; its parquet is never read (we write the
    # online value directly), but create a minimal one so apply/serve are happy.
    parquet = repo / "data" / "driver_stats.parquet"
    pd.DataFrame(
        {
            "driver_id": [ENTITY_VALUE],
            "conv_rate": [EXPECTED_CONV_RATE],
            "event_timestamp": [datetime.now(timezone.utc)],
        }
    ).to_parquet(parquet)

    store = feast.FeatureStore(repo_path=str(repo))
    driver = Entity(name="driver", join_keys=["driver_id"])
    source = FileSource(path=str(parquet), timestamp_field="event_timestamp")
    fv = FeatureView(
        name="driver_hourly",
        entities=[driver],
        ttl=timedelta(days=3650),
        schema=[Field(name="conv_rate", dtype=Float32)],
        online=True,
        source=source,
    )
    store.apply([driver, fv])
    store.write_to_online_store(
        feature_view_name="driver_hourly",
        df=pd.DataFrame(
            {
                "driver_id": [ENTITY_VALUE],
                "conv_rate": [EXPECTED_CONV_RATE],
                "event_timestamp": [datetime.now(timezone.utc)],
            }
        ),
    )

    port = _free_port()
    feast_bin = Path(sys.executable).with_name("feast")
    cmd = [
        str(feast_bin) if feast_bin.exists() else "feast",
        "--chdir",
        str(repo),
        "serve",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--no-access-log",
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    base_url = f"http://127.0.0.1:{port}"
    try:
        deadline = time.monotonic() + 40.0
        healthy = False
        while time.monotonic() < deadline:
            if proc.poll() is not None:  # serve died
                break
            try:
                if httpx.get(f"{base_url}/health", timeout=2.0).status_code == 200:
                    healthy = True
                    break
            except httpx.HTTPError:
                time.sleep(0.5)
        if not healthy:
            out = b""
            if proc.poll() is not None:
                out = proc.stdout.read() if proc.stdout else b""
            pytest.skip(
                "feast serve did not become healthy in this environment; "
                f"exit={proc.poll()} output={out[:500]!r}"
            )
        yield base_url
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()


@pytest.mark.asyncio
async def test_remote_online_features_roundtrip_against_real_feast_server(feast_server):
    """Full round-trip: a real `feast serve` returns the seeded value through remote mode."""
    client = FeastClient(config=FeastConfig(server_url=feast_server))

    result = await client.get_online_features(
        entity_rows=[{"driver_id": ENTITY_VALUE}],
        feature_refs=["driver_hourly:conv_rate"],
        full_feature_names=True,
    )

    # Drop-in with the embedded to_dict() shape: {fully_qualified_name: [value_per_row]}
    assert "driver_hourly__conv_rate" in result
    values = result["driver_hourly__conv_rate"]
    assert len(values) == 1
    assert values[0] == pytest.approx(EXPECTED_CONV_RATE, abs=1e-5)
