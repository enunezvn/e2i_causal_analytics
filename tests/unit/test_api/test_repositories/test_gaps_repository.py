import re

import pytest

from src.api.routes.gaps import GapAnalysisResponse, GapAnalysisStatus


def _ilike_to_regex(pattern: str) -> re.Pattern[str]:
    """Translate a SQL ``ILIKE`` pattern into a case-insensitive anchored regex.

    Mirrors PostgreSQL ``ILIKE`` semantics so the fake client faithfully models
    wildcard behaviour: unescaped ``%`` matches any run of chars, unescaped
    ``_`` matches a single char, and ``\\`` escapes the next metacharacter to a
    literal. This lets the repository tests catch wildcard broadening if the
    ``_escape_like`` guard in the repo were ever removed.
    """
    out: list[str] = []
    i = 0
    while i < len(pattern):
        ch = pattern[i]
        if ch == "\\" and i + 1 < len(pattern):
            out.append(re.escape(pattern[i + 1]))
            i += 2
            continue
        if ch == "%":
            out.append(".*")
        elif ch == "_":
            out.append(".")
        else:
            out.append(re.escape(ch))
        i += 1
    return re.compile("^" + "".join(out) + "$", re.IGNORECASE)


class _FakeExec:
    def __init__(self, rows):
        self._rows = rows

    def execute(self):  # supabase-py .execute() is SYNC; repo offloads via to_thread
        class _R:
            data = self._rows

        return _R()


class _FakeQuery:
    """Records upserts into a shared dict keyed by primary key; supports
    select().eq().order().limit() chaining used by the repo."""

    def __init__(self, store, rows=None):
        self._store = store
        self._rows = rows if rows is not None else list(store.values())

    def upsert(self, row, on_conflict=None):
        self._store[row["analysis_id"]] = row
        return _FakeExec([row])

    def select(self, *_a, **_k):
        return _FakeQuery(self._store, list(self._store.values()))

    def eq(self, col, val):
        return _FakeQuery(self._store, [r for r in self._rows if r.get(col) == val])

    def ilike(self, col, val):
        # PostgREST .ilike is a case-insensitive PATTERN match (NOT a literal
        # equality): supabase-py forwards the value verbatim, so unescaped
        # ``%``/``_`` act as wildcards. Model the real ILIKE semantics here so a
        # missing ``_escape_like`` guard in the repo would surface as wildcard
        # broadening in the brand-filter tests.
        rx = _ilike_to_regex(str(val))
        return _FakeQuery(
            self._store,
            [r for r in self._rows if rx.match(str(r.get(col, "")))],
        )

    def order(self, *_a, **_k):
        return self

    def limit(self, *_a, **_k):
        return self

    def execute(self):
        class _R:
            data = self._rows

        return _R()


class _FakeClient:
    def __init__(self):
        self.gap_analyses: dict = {}

    def table(self, name):
        assert name == "gap_analyses"
        return _FakeQuery(self.gap_analyses)


@pytest.mark.asyncio
async def test_upsert_then_get_roundtrips_across_boundary():
    from src.api.repositories.gaps_repository import GapsRepository

    repo = GapsRepository(client=_FakeClient())
    resp = GapAnalysisResponse(
        analysis_id="gap_deadbeef0001",
        status=GapAnalysisStatus.COMPLETED,
        brand="kisqali",
        metrics_analyzed=["trx"],
        segments_analyzed=3,
    )
    await repo.upsert(resp)
    got = await repo.get("gap_deadbeef0001")
    assert got is not None
    assert got.analysis_id == "gap_deadbeef0001"
    assert got.status == GapAnalysisStatus.COMPLETED
    assert got.brand == "kisqali"


@pytest.mark.asyncio
async def test_get_missing_returns_none():
    from src.api.repositories.gaps_repository import GapsRepository

    repo = GapsRepository(client=_FakeClient())
    assert await repo.get("gap_does_not_exist") is None


@pytest.mark.asyncio
async def test_list_completed_filters_by_brand():
    from src.api.repositories.gaps_repository import GapsRepository

    client = _FakeClient()
    repo = GapsRepository(client=client)
    await repo.upsert(
        GapAnalysisResponse(
            analysis_id="gap_a",
            status=GapAnalysisStatus.COMPLETED,
            brand="kisqali",
            metrics_analyzed=["trx"],
            segments_analyzed=1,
        )
    )
    await repo.upsert(
        GapAnalysisResponse(
            analysis_id="gap_b",
            status=GapAnalysisStatus.PENDING,
            brand="kisqali",
            metrics_analyzed=["trx"],
            segments_analyzed=1,
        )
    )
    await repo.upsert(
        GapAnalysisResponse(
            analysis_id="gap_c",
            status=GapAnalysisStatus.COMPLETED,
            brand="cosentyx",
            metrics_analyzed=["trx"],
            segments_analyzed=1,
        )
    )

    rows = await repo.list_completed(brand="kisqali")
    ids = {r.analysis_id for r in rows}
    assert ids == {"gap_a"}  # only COMPLETED + brand match


@pytest.mark.asyncio
async def test_list_completed_brand_filter_is_case_insensitive():
    """Regression for the GapAnalysis empty-page bug (brand-case mismatch).

    Ground truth verified against the live prod DB: the grounded analyses are
    stored with CAPITALIZED ``gap_analyses.brand`` ("Kisqali", "Fabhalta",
    "Remibrutinib") — matching the canonical ``brand_type`` ENUM and the
    synthetic ``Brand`` enum. The frontend Select previously sent the LOWERCASE
    value ("kisqali"), so the case-sensitive ``.eq("brand", ...)`` filter never
    returned the real rows and the whole page rendered empty.

    The repo brand filter MUST be case-insensitive so canonical-cased requests
    (and any historical casing drift) still surface the grounded analyses.
    """
    from src.api.repositories.gaps_repository import GapsRepository

    client = _FakeClient()
    repo = GapsRepository(client=client)

    # The real grounded analysis, stored capitalized (as in prod).
    await repo.upsert(
        GapAnalysisResponse(
            analysis_id="gap_grounded_kisqali",
            status=GapAnalysisStatus.COMPLETED,
            brand="Kisqali",
            metrics_analyzed=["trx"],
            segments_analyzed=5,
        )
    )

    # Canonical capitalized request returns the grounded analysis.
    rows_canonical = await repo.list_completed(brand="Kisqali")
    assert {r.analysis_id for r in rows_canonical} == {"gap_grounded_kisqali"}

    # A lowercase request must ALSO match the capitalized stored row (the exact
    # mismatch that emptied the page before the fix).
    rows_lower = await repo.list_completed(brand="kisqali")
    assert {r.analysis_id for r in rows_lower} == {"gap_grounded_kisqali"}


@pytest.mark.asyncio
async def test_list_completed_does_not_double_count_legacy_lowercase_rows():
    """Case-insensitive matching must not silently double-count the 2 legacy
    lowercase ``kisqali`` junk rows that exist in prod.

    Those legacy rows carry ZERO ``prioritized_opportunities``; with a
    case-insensitive filter they are returned alongside the real "Kisqali"
    analysis, but contribute no opportunities, so the opportunity list is not
    inflated. This test pins that behaviour: the grounded analysis is present and
    the empty legacy rows add no opportunities.
    """
    from src.api.repositories.gaps_repository import GapsRepository

    client = _FakeClient()
    repo = GapsRepository(client=client)

    # Real grounded analysis (capitalized) carrying opportunities.
    grounded = GapAnalysisResponse(
        analysis_id="gap_grounded_kisqali",
        status=GapAnalysisStatus.COMPLETED,
        brand="Kisqali",
        metrics_analyzed=["trx"],
        segments_analyzed=5,
    )
    await repo.upsert(grounded)

    # Two legacy lowercase junk rows with no opportunities (mirrors prod).
    for legacy_id in ("gap_it_legacy_a", "gap_it_legacy_b"):
        await repo.upsert(
            GapAnalysisResponse(
                analysis_id=legacy_id,
                status=GapAnalysisStatus.COMPLETED,
                brand="kisqali",
                metrics_analyzed=["trx"],
                segments_analyzed=0,
            )
        )

    rows = await repo.list_completed(brand="Kisqali")
    ids = {r.analysis_id for r in rows}
    assert ids == {"gap_grounded_kisqali", "gap_it_legacy_a", "gap_it_legacy_b"}

    # Opportunities come only from the grounded row; legacy rows add none.
    total_opps = sum(len(r.prioritized_opportunities) for r in rows)
    grounded_opps = sum(
        len(r.prioritized_opportunities) for r in rows if r.analysis_id == "gap_grounded_kisqali"
    )
    assert total_opps == grounded_opps


@pytest.mark.asyncio
async def test_list_completed_brand_filter_does_not_treat_wildcards_as_patterns():
    """``.ilike`` is a pattern match, so the brand value MUST be escaped.

    Without escaping, a brand argument of ``"%"`` (or any value containing SQL
    ``LIKE`` metacharacters) would broaden the filter and return analyses for
    EVERY brand — a correctness/contract bug. The repo escapes the value so the
    filter is an exact, whole-string, case-insensitive match. This faithfully
    reproduces the prod ILIKE semantics via the fake client.
    """
    from src.api.repositories.gaps_repository import GapsRepository

    client = _FakeClient()
    repo = GapsRepository(client=client)

    for brand, aid in (
        ("Kisqali", "gap_k"),
        ("Fabhalta", "gap_f"),
        ("Remibrutinib", "gap_r"),
    ):
        await repo.upsert(
            GapAnalysisResponse(
                analysis_id=aid,
                status=GapAnalysisStatus.COMPLETED,
                brand=brand,
                metrics_analyzed=["trx"],
                segments_analyzed=1,
            )
        )

    # A wildcard-only brand must NOT match every brand (no literal "%" brand
    # exists, so the result is empty — not all three analyses).
    assert await repo.list_completed(brand="%") == []

    # An underscore wildcard must not match arbitrary single-char-different
    # brands either (it is escaped to a literal underscore).
    assert await repo.list_completed(brand="_") == []

    # Sanity: an exact (case-insensitive) brand still matches just its own row.
    rows = await repo.list_completed(brand="fabhalta")
    assert {r.analysis_id for r in rows} == {"gap_f"}
