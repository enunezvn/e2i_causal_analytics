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


def test_postgrest_cap_logs_truncation_warning(caplog):
    import logging as _logging

    fake = _FakeSupabase(_rows(1000))  # exactly the default PostgREST cap
    with caplog.at_level(_logging.WARNING, logger="src.services.cohort_resolution"):
        frame = resolve_cohort_frame("Kisqali", "northeast", supabase_client=fake)
    assert isinstance(frame, pd.DataFrame)
    assert len(frame) == 1000
    assert any("may be truncated" in r.message for r in caplog.records)


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
