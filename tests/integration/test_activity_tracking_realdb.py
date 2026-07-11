"""Middleware flush writes REAL rows via the record_user_activity RPC and the
RPC merges counts additively (no mocks).

    E2I_DB_INTEGRATION=1 .venv/bin/pytest tests/integration/test_activity_tracking_realdb.py -p no:cacheprovider -v
"""

import os
import uuid
from datetime import datetime, timezone

import pytest

pytestmark = pytest.mark.skipif(
    os.getenv("E2I_DB_INTEGRATION") != "1",
    reason="real-DB integration; set E2I_DB_INTEGRATION=1 with docker supabase-db reachable",
)


@pytest.fixture()
def db():
    from supabase import create_client

    client = create_client(
        os.environ["SUPABASE_URL"],
        os.environ.get("SUPABASE_SERVICE_ROLE_KEY") or os.environ["SUPABASE_SERVICE_KEY"],
    )
    test_user = str(uuid.uuid4())
    yield client, test_user
    client.table("user_activity_log").delete().eq("user_id", test_user).execute()


@pytest.mark.asyncio
async def test_flush_rows_persists_and_merges(db):
    from src.api.middleware.activity_tracking import flush_rows

    client, test_user = db
    minute = datetime.now(timezone.utc).replace(second=0, microsecond=0).isoformat()
    row = {
        "user_id": test_user,
        "user_email": "activity-test@example.invalid",
        "endpoint_group": "causal",
        "http_method": "GET",
        "bucket_minute": minute,
        "request_count": 3,
    }
    await flush_rows([row])
    await flush_rows([dict(row, request_count=2)])  # same bucket -> additive merge

    got = (
        client.table("user_activity_log")
        .select("request_count, endpoint_group")
        .eq("user_id", test_user)
        .execute()
    )
    assert len(got.data) == 1
    assert got.data[0]["request_count"] == 5


@pytest.mark.asyncio
async def test_middleware_records_authenticated_api_requests(db):
    """Drive the middleware directly with a minimal ASGI app — the real app's
    JWT layer sets request.state.user; here we set it the same way and assert
    the buffer bucketing + skip rules (non-/api paths, missing user, non-UUID
    test users are all skipped)."""
    from fastapi import FastAPI, Request
    from starlette.testclient import TestClient

    from src.api.middleware.activity_tracking import (
        ActivityBuffer,
        ActivityTrackingMiddleware,
    )

    client, test_user = db
    app = FastAPI()
    buf = ActivityBuffer(flush_interval_s=99999, flush_threshold=99999)

    # add_middleware prepends: ActivityTracking added FIRST = inner; the
    # decorator middleware below (added after) runs OUTER and sets state.user
    # before ActivityTracking sees the request — matching prod where
    # JWTAuthMiddleware is outer to activity tracking.
    app.add_middleware(ActivityTrackingMiddleware, buffer=buf)

    @app.middleware("http")
    async def fake_jwt(request: Request, call_next):
        request.state.user = {"id": test_user, "email": "activity-test@example.invalid"}
        return await call_next(request)

    @app.get("/api/causal/estimate")
    async def causal():
        return {"ok": True}

    @app.get("/healthz")
    async def health():
        return {"ok": True}

    tc = TestClient(app)
    tc.get("/api/causal/estimate")
    tc.get("/api/causal/estimate")
    tc.get("/healthz")  # non-/api -> not recorded

    rows = buf.drain()
    assert len(rows) == 1
    assert rows[0]["endpoint_group"] == "causal"
    assert rows[0]["request_count"] == 2


@pytest.mark.asyncio
async def test_middleware_skips_non_uuid_testing_mode_user(db):
    from fastapi import FastAPI, Request
    from starlette.testclient import TestClient

    from src.api.middleware.activity_tracking import (
        ActivityBuffer,
        ActivityTrackingMiddleware,
    )

    _, _ = db
    app = FastAPI()
    buf = ActivityBuffer(flush_interval_s=99999, flush_threshold=99999)
    app.add_middleware(ActivityTrackingMiddleware, buffer=buf)

    @app.middleware("http")
    async def fake_jwt(request: Request, call_next):
        request.state.user = {"id": "test-user-id", "email": "test@e2i.internal"}
        return await call_next(request)

    @app.get("/api/kpis/list")
    async def kpis():
        return {"ok": True}

    TestClient(app).get("/api/kpis/list")
    assert buf.drain() == []
