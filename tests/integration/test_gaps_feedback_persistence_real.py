"""M2: round-trip gap + feedback persistence against a REAL Supabase.

Skips unless SUPABASE_URL and a service/anon key are configured (faithful only
where infra allows). Proves a row written by one repo instance is readable by a
SECOND independent instance — the cross-worker read the in-memory dict failed.
"""

import os
import uuid

import pytest

_HAVE_SUPABASE = bool(os.environ.get("SUPABASE_URL")) and bool(
    os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
    or os.environ.get("SUPABASE_SERVICE_KEY")
    or os.environ.get("SUPABASE_ANON_KEY")
)

pytestmark = pytest.mark.skipif(not _HAVE_SUPABASE, reason="real Supabase not configured")


@pytest.mark.asyncio
async def test_gap_analysis_roundtrips_across_repo_instances():
    from src.api.repositories.gaps_repository import GapsRepository
    from src.api.routes.gaps import AnalysisStatus, GapAnalysisResponse

    analysis_id = f"gap_it_{uuid.uuid4().hex[:8]}"
    resp = GapAnalysisResponse(
        analysis_id=analysis_id,
        status=AnalysisStatus.COMPLETED,
        brand="kisqali",
        metrics_analyzed=["trx"],
        segments_analyzed=4,
    )
    await GapsRepository().upsert(resp)  # "worker A"
    got = await GapsRepository(client=None).get(analysis_id)  # fresh instance ~ "worker B"
    assert got is not None and got.brand == "kisqali"
    assert got.status == AnalysisStatus.COMPLETED


@pytest.mark.asyncio
async def test_feedback_batch_roundtrips_across_repo_instances():
    from src.api.repositories.feedback_repository import FeedbackRepository
    from src.api.routes.feedback import LearningResponse, LearningStatus

    batch_id = f"fb_it_{uuid.uuid4().hex[:8]}"
    await FeedbackRepository().upsert_batch(
        LearningResponse(batch_id=batch_id, status=LearningStatus.COMPLETED)
    )
    got = await FeedbackRepository().get_batch(batch_id)
    assert got is not None and got.batch_id == batch_id
