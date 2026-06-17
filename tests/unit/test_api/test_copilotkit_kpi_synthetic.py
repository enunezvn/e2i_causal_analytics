"""Hermetic tests for the KPI synthetic-visibility demo mode at the copilotkit
``get_kpi_summary`` surface (the Home landing tiles).

Locks two behaviours under the ``E2I_KPI_INCLUDE_SYNTHETIC`` flag:
1. OFF (production default): base query_ids are used and ``data_source`` is
   ``"database"`` (or ``"unavailable"``) -- the strict migration-066 gate stands.
2. ON (demo/review): the underlying ``kpi_query`` ids swap to their
   ``_include_synthetic`` twins AND ``data_source`` is ``"synthetic"`` so the FE
   badges the figures honestly (never passed off as real-world "database" data).
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.api.routes.copilotkit import get_kpi_summary


class _FakeExec:
    def __init__(self, data):
        self._data = data

    def execute(self):
        return SimpleNamespace(data=self._data)


class _RecordingClient:
    """Records every ``kpi_query`` query_id and returns a row carrying every
    result key the summary reads, so each metric resolves to a real number."""

    def __init__(self):
        self.query_ids: list[str] = []

    def rpc(self, name, payload):
        assert name == "kpi_query"
        self.query_ids.append(payload["query_id"])
        return _FakeExec(
            [
                {
                    "trx": 42642,
                    "nrx": 1234,
                    "conversion_rate": 0.31,
                    "hcp_reach": 321,
                    "nbrx": 99,
                    "data_through": "2026-06-10",
                }
            ]
        )


@pytest.fixture
def recording_client(monkeypatch):
    client = _RecordingClient()
    # get_kpi_summary does `from src.api.dependencies.supabase_client import
    # get_supabase` at call time -> patch the source symbol.
    monkeypatch.setattr(
        "src.api.dependencies.supabase_client.get_supabase",
        lambda: client,
    )
    return client


async def test_flag_off_uses_base_ids_and_database_source(recording_client, monkeypatch):
    monkeypatch.delenv("E2I_KPI_INCLUDE_SYNTHETIC", raising=False)
    result = await get_kpi_summary("All")

    assert result["data_source"] == "database"
    assert recording_client.query_ids, "expected at least one kpi_query call"
    # No twin ids leak while the production gate stands.
    assert all(not q.endswith("_include_synthetic") for q in recording_client.query_ids)
    # The base TRx id was used (and data_through queried).
    assert "business_impact_trx" in recording_client.query_ids
    assert "business_impact_data_through" in recording_client.query_ids


async def test_flag_on_uses_twins_and_synthetic_source(recording_client, monkeypatch):
    monkeypatch.setenv("E2I_KPI_INCLUDE_SYNTHETIC", "true")
    result = await get_kpi_summary("All")

    assert result["data_source"] == "synthetic"
    # The TRx tile + data_through label now read the synthetic-inclusive twins.
    assert "business_impact_trx_include_synthetic" in recording_client.query_ids
    assert "business_impact_data_through_include_synthetic" in recording_client.query_ids
    # And the populated value flows through (the synthetic-gold TRx).
    assert result["metrics"]["trx_volume"] == 42642
    # Honest labelling: synthetic mode is NOT reported as production "database".
    assert result["data_source"] != "database"


def test_fallback_response_is_honest_transient_not_canned_actions():
    """The chat node's LLM-failure fallback (``generate_e2i_response``) must be
    an honest "temporarily unavailable" message -- not the old keyword-canned
    text that dumped internal CopilotKit action names and read like a real
    answer (the exact "not optimal" reply the user reported). It must never
    fabricate a data figure for the question it could not actually answer."""
    from src.api.routes.copilotkit import generate_e2i_response

    msg = generate_e2i_response("what was the Fabhalta NBRx for the past 3 months?")
    low = msg.lower()
    # Honest about a transient failure and invites a retry.
    assert "try again" in low
    # Does NOT masquerade as a working answer / dump internal action names.
    assert "getkpisummary" not in low
    assert "21-agent" not in low
    assert "use the **get" not in low
    # No fabricated metric value for a request it did not actually compute.
    assert "3168" not in msg and "3298" not in msg
