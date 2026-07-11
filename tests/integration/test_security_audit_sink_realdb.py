"""Red-first pin for the security_audit_log DB sink (spec: admin feature).

ROOT CAUSE (verified 2026-07-11): get_security_audit_service() imports
``from src.api.deps import get_supabase`` — src.api.deps does not exist
(the real module is src.api.dependencies). The ImportError is swallowed,
self.db stays None, and every security event ever emitted went to
stdout only. security_audit_log has 0 rows in prod despite months of
auth-failure logging.

Real-DB test (no mocks): requires docker supabase-db + service key in env.
    E2I_DB_INTEGRATION=1 .venv/bin/pytest tests/integration/test_security_audit_sink_realdb.py -p no:cacheprovider -v
"""

import os
import uuid

import pytest

pytestmark = pytest.mark.skipif(
    os.getenv("E2I_DB_INTEGRATION") != "1",
    reason="real-DB integration; set E2I_DB_INTEGRATION=1 with docker supabase-db reachable",
)


def test_security_audit_service_persists_to_database():
    from src.api.dependencies import get_supabase
    from src.utils.security_audit import (
        get_security_audit_service,
        reset_security_audit_service,
    )

    reset_security_audit_service()
    try:
        service = get_security_audit_service()
        # The factory must have wired a real client (this is the red assertion:
        # today the broken import leaves service.db None).
        assert service.db is not None, (
            "security audit service has no DB sink — src.api.deps import bug"
        )

        marker = f"sink-test-{uuid.uuid4()}"
        service.log_auth_failure(
            user_email="sink-test@example.invalid",
            client_ip="127.0.0.1",
            reason=marker,
        )

        client = get_supabase()
        rows = (
            client.table("security_audit_log")
            .select("event_id, event_type, error_details")
            .eq("error_details", marker)
            .execute()
        )
        assert len(rows.data) == 1, f"expected 1 persisted row, got {rows.data}"
        assert rows.data[0]["event_type"] == "auth.login.failure"

        # cleanup
        client.table("security_audit_log").delete().eq("error_details", marker).execute()
    finally:
        reset_security_audit_service()
