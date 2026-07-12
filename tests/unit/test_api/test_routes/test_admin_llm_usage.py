"""GET /admin/observability/llm-usage: thin route — admin-gated (Depends,
enforced like every admin.py sibling), runs the aggregation off-thread,
passes days + the user listing through."""

import asyncio

from src.api.routes.admin import llm_usage_overview


class _FakeAdminService:
    def list_users(self):
        return [{"id": "u1", "email": "a@x.com"}]


class _FakeObs:
    def __init__(self):
        self.calls = []

    def llm_usage(self, days, users):
        self.calls.append((days, users))
        return {"summary": {"days": days, "calls": 0}}


def test_llm_usage_overview_wires_days_and_users():
    obs = _FakeObs()
    result = asyncio.run(
        llm_usage_overview(
            days=7,
            admin={"id": "admin-1"},
            service=_FakeAdminService(),
            obs=obs,
        )
    )
    assert obs.calls == [(7, [{"id": "u1", "email": "a@x.com"}])]
    assert result["summary"]["days"] == 7
