"""Unit tests for the cohort-resolution service (issue #779).

The tool-composer-remediation (PRs #774-777) wired
``chatbot_tools._resolve_cohort_frame`` to the tier0 loader, but that path only
produced a real DataFrame when an explicit ``data_source`` was supplied. This
service is the deferred data-layer follow-up: resolve a ``(brand, region)`` pair
to a real patient-cohort DataFrame from the canonical ``patient_journeys`` table
WITHOUT an explicit ``data_source``.

Anti-mocking discipline: the service NEVER fabricates a synthetic cohort. It
returns ``None`` (fail closed) on unrecognized brand/region or an empty result
so callers honestly proceed without ``estimation_data`` and the composable tools
fail closed in turn.
"""

import types

import pandas as pd

from src.services import cohort_resolution
from src.services.cohort_resolution import resolve_cohort_frame


class _FakeQuery:
    """Records the PostgREST query-builder chain and returns injected rows."""

    def __init__(self, recorder, rows):
        self._recorder = recorder
        self._rows = rows

    def select(self, cols):
        self._recorder["select"] = cols
        return self

    def eq(self, col, val):
        self._recorder["eq"].append((col, val))
        return self

    def limit(self, n):
        self._recorder["limit"] = n
        return self

    def execute(self):
        self._recorder["executed"] = True
        return types.SimpleNamespace(data=self._rows)


class _FakeSupabase:
    def __init__(self, rows):
        self._rows = rows
        self.recorder = {"eq": [], "executed": False, "table": None, "limit": None}

    def table(self, name):
        self.recorder["table"] = name
        return _FakeQuery(self.recorder, self._rows)


def _rows(n=3):
    return [
        {
            "patient_journey_id": f"PJ{i}",
            "brand": "Kisqali",
            "geographic_region": "northeast",
            "engagement_score": 5.0,
            "treatment_initiated": i % 2,
        }
        for i in range(n)
    ]


def test_resolves_patient_journeys_filtered_by_brand_and_region():
    fake = _FakeSupabase(_rows(3))
    frame = resolve_cohort_frame("Kisqali", "Northeast", supabase_client=fake)
    assert isinstance(frame, pd.DataFrame)
    assert len(frame) == 3
    assert fake.recorder["table"] == "patient_journeys"
    assert ("brand", "Kisqali") in fake.recorder["eq"]
    # region normalized to the lowercase region_type enum value
    assert ("geographic_region", "northeast") in fake.recorder["eq"]


def test_normalizes_brand_case_to_canonical_enum():
    fake = _FakeSupabase(_rows(1))
    resolve_cohort_frame("kisqali", "northeast", supabase_client=fake)
    assert ("brand", "Kisqali") in fake.recorder["eq"]


def test_unrecognized_region_returns_none_without_db_call():
    # "US" is NOT a region_type member (enum = northeast/south/midwest/west).
    fake = _FakeSupabase(_rows(3))
    frame = resolve_cohort_frame("Kisqali", "US", supabase_client=fake)
    assert frame is None
    assert fake.recorder["table"] is None  # never issued a query


def test_unrecognized_brand_returns_none_without_db_call():
    fake = _FakeSupabase(_rows(3))
    frame = resolve_cohort_frame("NotABrand", "northeast", supabase_client=fake)
    assert frame is None
    assert fake.recorder["table"] is None


# ---------------------------------------------------------------------------
# #1505 consolidation guardrails → #1517 product decision
#
# The brand/region resolvers live in src/services/enum_labels.py, shared with
# the chat KPI tool. #1505 kept this service strict (allow_synonyms=False) so
# consolidation could not widen a fail-closed contract as a side effect; the
# tests below were the alarm for that. #1517 then made the widening an explicit
# product decision: every production consumer feeding this service (the
# tool_composer chat tool, the cohort_builder composable tool, the orchestrator
# dispatcher) passes chat/LLM-derived or frontend-typed region strings — the
# exact input domain the chat KPI tool already resolves WITH the platform
# synonym table. No consumer passes market/territory identifiers that an alias
# could falsely match. The same ask ("conversion in the Pacific") must not get
# KPI numbers from one chat tool and a fail-closed cohort from the other.
# ---------------------------------------------------------------------------


def test_region_synonyms_resolve_to_canonical_labels():
    # #1517: platform synonyms resolve to their region_type label and FILTER
    # the query — identical semantics to the chat KPI tool on the same input.
    for synonym, expected in (
        ("NE", "northeast"),
        ("new england", "northeast"),
        ("central", "midwest"),
        ("Pacific", "west"),
        ("southern", "south"),
        ("nw", "west"),
        ("Southeast", "south"),
        ("southwest", "south"),
    ):
        fake = _FakeSupabase(_rows(3))
        frame = resolve_cohort_frame("Kisqali", synonym, supabase_client=fake)
        assert isinstance(frame, pd.DataFrame), synonym
        assert ("geographic_region", expected) in fake.recorder["eq"], synonym


def test_region_separator_variants_resolve():
    for variant, expected in (
        ("North East", "northeast"),
        ("north-east", "northeast"),
        ("mid west", "midwest"),
        ("north_east", "northeast"),
    ):
        fake = _FakeSupabase(_rows(3))
        frame = resolve_cohort_frame("Kisqali", variant, supabase_client=fake)
        assert isinstance(frame, pd.DataFrame), variant
        assert ("geographic_region", expected) in fake.recorder["eq"], variant


def test_non_synonym_regions_still_fail_closed():
    # The synonym table is closed: anything outside it (market names, typos)
    # still fails closed WITHOUT a query — never a silently-wrong population.
    for garbage in ("US", "EU", "APAC", "atlantis", "emea"):
        fake = _FakeSupabase(_rows(3))
        frame = resolve_cohort_frame("Kisqali", garbage, supabase_client=fake)
        assert frame is None, garbage
        assert fake.recorder["table"] is None, garbage


def test_brand_aliases_fail_closed():
    # "remi" is an entity-extraction alias, not a brand_type label.
    for alias in ("remi", "btk inhibitor", "ribociclib"):
        fake = _FakeSupabase(_rows(3))
        frame = resolve_cohort_frame(alias, "northeast", supabase_client=fake)
        assert frame is None, alias
        assert fake.recorder["table"] is None, alias


def test_resolvers_are_the_shared_ones():
    from src.services import enum_labels

    assert cohort_resolution._normalize_brand is enum_labels.resolve_brand_label
    # No second copy of the label tables may survive the consolidation.
    assert not hasattr(cohort_resolution, "_BRAND_CANONICAL")
    assert not hasattr(cohort_resolution, "_REGION_CANONICAL")
    for label in enum_labels.REGION_ENUM_LABELS:
        assert cohort_resolution._normalize_region(label.title()) == label
    # #1517: _normalize_region resolves via the SHARED synonym table — a value
    # only the alias table knows must land on its canonical label.
    assert cohort_resolution._normalize_region("new england") == "northeast"


def test_empty_result_returns_none():
    fake = _FakeSupabase([])
    frame = resolve_cohort_frame("Kisqali", "northeast", supabase_client=fake)
    assert frame is None
    assert fake.recorder["executed"] is True


def test_brand_only_applies_single_filter():
    fake = _FakeSupabase(_rows(2))
    resolve_cohort_frame("Kisqali", None, supabase_client=fake)
    assert ("brand", "Kisqali") in fake.recorder["eq"]
    assert all(col != "geographic_region" for col, _ in fake.recorder["eq"])


def test_no_brand_no_region_queries_unfiltered():
    fake = _FakeSupabase(_rows(5))
    frame = resolve_cohort_frame(None, None, supabase_client=fake)
    assert isinstance(frame, pd.DataFrame)
    assert len(frame) == 5
    # Shard 07 R11: no brand/region filter is applied, but the provenance predicate
    # (default-exclude is_synthetic) IS — real-mode never blends synthetic rows.
    assert fake.recorder["eq"] == [("is_synthetic", False)]


def test_no_brand_no_region_include_synthetic_is_unfiltered():
    # Validation opt-in: include_synthetic=True drops even the provenance predicate.
    fake = _FakeSupabase(_rows(5))
    resolve_cohort_frame(None, None, supabase_client=fake, include_synthetic=True)
    assert fake.recorder["eq"] == []


def test_limit_applied_when_supplied():
    fake = _FakeSupabase(_rows(3))
    resolve_cohort_frame("Kisqali", "northeast", supabase_client=fake, limit=500)
    assert fake.recorder["limit"] == 500


def test_empty_string_brand_treated_as_no_filter_not_failclosed():
    # An empty/whitespace brand means "not specified" -> no brand filter (same
    # as None), NOT a fail-closed and NOT a silent unrecognized-value error.
    fake = _FakeSupabase(_rows(2))
    frame = resolve_cohort_frame("   ", "northeast", supabase_client=fake)
    assert isinstance(frame, pd.DataFrame)
    assert all(col != "brand" for col, _ in fake.recorder["eq"])
    assert ("geographic_region", "northeast") in fake.recorder["eq"]


def test_empty_string_region_treated_as_no_filter():
    fake = _FakeSupabase(_rows(2))
    resolve_cohort_frame("Kisqali", "", supabase_client=fake)
    assert ("brand", "Kisqali") in fake.recorder["eq"]
    assert all(col != "geographic_region" for col, _ in fake.recorder["eq"])


def test_competitor_brand_normalizes_to_lowercase_enum():
    # 'competitor'/'other' are the only lowercase brand_type members; a title-case
    # caller input must map to the lowercase canonical spelling.
    fake = _FakeSupabase(_rows(1))
    resolve_cohort_frame("Competitor", "west", supabase_client=fake)
    assert ("brand", "competitor") in fake.recorder["eq"]


# ---------------------------------------------------------------------------
# #1587 — truncation detection must be EXACT, not a heuristic.
#
# The original guard warned whenever an unlimited fetch returned >= 1000 rows,
# on the premise that PostgREST caps a request at its 1000-row default. That
# premise is false for this deployment: a single response measurably carried
# 8,554 rows. Complete focal-brand cohorts are 8,554 / 8,615 / 8,729 rows, so
# the warning fired on EVERY complete cohort and carried no signal either way
# (it could not distinguish "complete fetch > 1000" from "truncated at a real
# cap"). The fix mirrors src/services/kpi_resolution.py: request an explicit
# cap and warn only when the fetch hits the cap we actually asked for.
# ---------------------------------------------------------------------------


def test_complete_large_cohort_logs_no_truncation_warning(caplog):
    # 8,554 rows == the measured COMPLETE Kisqali cohort. Nothing was truncated,
    # so nothing may be warned about. This is the #1587 false positive.
    import logging as _logging

    fake = _FakeSupabase(_rows(8554))
    with caplog.at_level(_logging.WARNING, logger="src.services.cohort_resolution"):
        frame = resolve_cohort_frame("Kisqali", "northeast", supabase_client=fake)
    assert isinstance(frame, pd.DataFrame)
    assert len(frame) == 8554
    assert not any("truncated" in r.message for r in caplog.records)


def test_default_cap_is_requested_on_the_query():
    # Truncation can only be detected exactly if we ASK for a bound. Without an
    # explicit caller limit the service must still send its own default cap.
    fake = _FakeSupabase(_rows(3))
    resolve_cohort_frame("Kisqali", "northeast", supabase_client=fake)
    assert fake.recorder["limit"] == cohort_resolution._MAX_COHORT_ROWS


def test_warns_only_when_the_requested_cap_is_hit_and_names_it(caplog, monkeypatch):
    # Hitting the cap we requested is the ONLY state that means "possibly
    # truncated". The message must name the cap so the reader can act on it.
    import logging as _logging

    monkeypatch.setattr(cohort_resolution, "_MAX_COHORT_ROWS", 5)
    fake = _FakeSupabase(_rows(5))
    with caplog.at_level(_logging.WARNING, logger="src.services.cohort_resolution"):
        frame = resolve_cohort_frame("Kisqali", "northeast", supabase_client=fake)
    assert isinstance(frame, pd.DataFrame)
    warnings = [r.message for r in caplog.records if "truncated" in r.message]
    assert len(warnings) == 1
    assert "5" in warnings[0]


def test_cap_comfortably_exceeds_designed_substrate_volume():
    # Cap derivation, made executable: the widest COMPLETE cohort this service
    # can be asked for is the whole patient_journeys table (brand=None,
    # region=None is a supported call — see test_no_brand_no_region_queries_
    # unfiltered), so the cap must clear the DGP's designed full-table volume
    # with room to spare. If a future volume bump erodes that headroom this
    # test fails LOUDLY rather than silently truncating a complete cohort.
    from src.ml.synthetic.config import EntityVolumes

    designed_full_table = EntityVolumes().total_patients
    assert cohort_resolution._MAX_COHORT_ROWS >= 2 * designed_full_table


def test_no_truncation_warning_when_explicit_limit(caplog):
    import logging as _logging

    fake = _FakeSupabase(_rows(1000))
    with caplog.at_level(_logging.WARNING, logger="src.services.cohort_resolution"):
        resolve_cohort_frame("Kisqali", "northeast", supabase_client=fake, limit=1000)
    assert not any("may be truncated" in r.message for r in caplog.records)


def test_explicit_data_source_uses_tier0_loader(monkeypatch):
    # With an explicit data_source we MUST delegate to the tier0
    # CohortConstructorAgent loader and never touch patient_journeys.
    captured = {}

    class _FakeAgent:
        def run(self, input_data):
            captured["input"] = input_data
            return {"eligible_patients": pd.DataFrame(_rows(4)), "success": True}

    monkeypatch.setattr(cohort_resolution, "_load_tier0_agent", lambda: _FakeAgent())
    fake = _FakeSupabase(_rows(99))
    frame = resolve_cohort_frame(
        "Kisqali",
        "northeast",
        data_source="s3://bucket/cohort.parquet",
        supabase_client=fake,
    )
    assert isinstance(frame, pd.DataFrame)
    assert len(frame) == 4  # from tier0 loader, not patient_journeys
    assert fake.recorder["table"] is None  # patient_journeys path NOT used
    assert captured["input"]["patient_data_source"] == "s3://bucket/cohort.parquet"


def test_explicit_data_source_empty_returns_none(monkeypatch):
    class _FakeAgent:
        def run(self, input_data):
            return {"eligible_patients": None, "success": False}

    monkeypatch.setattr(cohort_resolution, "_load_tier0_agent", lambda: _FakeAgent())
    frame = resolve_cohort_frame(
        "Kisqali", "northeast", data_source="tbl", supabase_client=_FakeSupabase([])
    )
    assert frame is None


def test_chatbot_tools_resolver_delegates_to_service(monkeypatch):
    # The chat caller's _resolve_cohort_frame must delegate to the shared
    # service so the chat path gains the no-data_source resolution too.
    import src.api.routes.chatbot_tools as ct

    calls = {}

    def _spy(brand, region, *, data_source=None, supabase_client=None, limit=None):
        calls["args"] = (brand, region, data_source)
        return pd.DataFrame(_rows(2))

    monkeypatch.setattr(ct.cohort_resolution, "resolve_cohort_frame", _spy)
    frame = ct._resolve_cohort_frame("Kisqali", "Northeast", None)
    assert isinstance(frame, pd.DataFrame)
    assert len(frame) == 2
    assert calls["args"] == ("Kisqali", "Northeast", None)
