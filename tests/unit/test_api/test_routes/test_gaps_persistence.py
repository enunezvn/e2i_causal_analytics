import pytest

import src.api.routes.gaps as gaps_mod
from src.api.routes.gaps import (
    GapAnalysisResponse,
    GapAnalysisStatus,
    get_gap_analysis,
)


class _Repo:
    def __init__(self):
        self._rows: dict = {}

    async def upsert(self, resp):
        self._rows[resp.analysis_id] = resp

    async def get(self, analysis_id):
        return self._rows.get(analysis_id)

    async def list_completed(self, brand=None):
        return [
            r
            for r in self._rows.values()
            if r.status == GapAnalysisStatus.COMPLETED and (brand is None or r.brand == brand)
        ]

    async def list_all(self):
        return list(self._rows.values())


@pytest.mark.asyncio
async def test_get_reads_from_repo_not_dict(monkeypatch):
    repo = _Repo()
    # Persisted by "another worker" — never written to the process-local dict.
    await repo.upsert(
        GapAnalysisResponse(
            analysis_id="gap_xworker",
            status=GapAnalysisStatus.COMPLETED,
            brand="kisqali",
            metrics_analyzed=["trx"],
            segments_analyzed=2,
        )
    )
    monkeypatch.setattr(gaps_mod, "_get_repo", lambda: repo)
    monkeypatch.setattr(gaps_mod, "_use_inmemory_fallback", lambda: False)
    gaps_mod._analyses_store.clear()  # prove we are NOT reading the dict

    got = await get_gap_analysis("gap_xworker")
    assert got.analysis_id == "gap_xworker"
