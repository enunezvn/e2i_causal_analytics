"""Unit coverage for the brand-scoped causal discovery.

The discovery page's brand dropdown is data-driven: ``_list_dataset_brands``
returns the distinct brands actually present in the cohort, and
``_load_agent_estimation_frame(brand=...)`` scopes the estimation rows to one
brand via a server-side ``.eq('brand', X)`` filter (brand is a row filter, NOT a
causal variable, so it never enters the estimation columns). These use a fake
supabase client — the end-to-end real-DB path is covered by a faithful check.
"""

from typing import Any, Dict, List

import pytest

import src.api.routes.causal as causal


class _FakeQuery:
    """Records the PostgREST-style builder chain and returns canned rows."""

    def __init__(self, rows: List[Dict[str, Any]], log: Dict[str, Any]):
        self._rows = rows
        self._log = log

    def select(self, cols: str) -> "_FakeQuery":
        self._log["select"] = cols
        return self

    def eq(self, col: str, val: Any) -> "_FakeQuery":
        self._log.setdefault("eq", []).append((col, val))
        return self

    def limit(self, n: int) -> "_FakeQuery":
        self._log["limit"] = n
        return self

    async def execute(self) -> Any:
        return type("_Result", (), {"data": self._rows})()


class _FakeClient:
    def __init__(self, rows: List[Dict[str, Any]], log: Dict[str, Any]):
        self._rows = rows
        self._log = log

    def table(self, name: str) -> _FakeQuery:
        self._log["table"] = name
        return _FakeQuery(self._rows, self._log)


def _patch_client(monkeypatch, rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    log: Dict[str, Any] = {}

    async def _fake_factory():
        return _FakeClient(rows, log)

    # Both helpers import the factory locally from this module path.
    import src.memory.services.factories as factories

    monkeypatch.setattr(factories, "get_async_supabase_client", _fake_factory)
    # Provenance filter is orthogonal to brand scoping — identity in the unit test.
    monkeypatch.setattr(causal, "apply_provenance_filter", lambda q, *a, **k: q)
    return log


@pytest.mark.unit
async def test_list_dataset_brands_distinct_sorted_non_null(monkeypatch):
    _patch_client(
        monkeypatch,
        [
            {"brand": "Kisqali"},
            {"brand": "Remibrutinib"},
            {"brand": "Kisqali"},
            {"brand": None},  # nulls dropped
            {"brand": "Fabhalta"},
            {},  # missing key tolerated
        ],
    )
    brands = await causal._list_dataset_brands("patient_journeys")
    assert brands == ["Fabhalta", "Kisqali", "Remibrutinib"]


@pytest.mark.unit
async def test_list_dataset_brands_empty_when_store_unavailable(monkeypatch):
    import src.memory.services.factories as factories

    async def _none_factory():
        return None

    monkeypatch.setattr(factories, "get_async_supabase_client", _none_factory)
    assert await causal._list_dataset_brands("patient_journeys") == []


@pytest.mark.unit
async def test_load_frame_with_brand_applies_eq_filter_and_excludes_brand_column(monkeypatch):
    # Two rows; brand present in the raw payload but must NOT leak into the frame.
    log = _patch_client(
        monkeypatch,
        [
            {
                "treatment_arm": 1.0,
                "persistent_180d": 1.0,
                "disease_severity": 0.3,
                "brand": "Kisqali",
            },
            {
                "treatment_arm": 0.0,
                "persistent_180d": 0.0,
                "disease_severity": 0.7,
                "brand": "Kisqali",
            },
        ],
    )
    df, select = await causal._load_agent_estimation_frame(
        dataset="patient_journeys",
        treatment_var="treatment_arm",
        outcome_var="persistent_180d",
        covariates=["disease_severity"],
        limit=1500,
        brand="Kisqali",
    )
    # The server-side filter scoped the cohort to the brand.
    assert ("brand", "Kisqali") in log.get("eq", [])
    # brand is a FILTER, not a causal column — it never enters the estimation frame.
    assert "brand" not in df.columns
    assert list(df.columns) == ["treatment_arm", "persistent_180d", "disease_severity"]
    assert len(df) == 2


@pytest.mark.unit
async def test_load_frame_without_brand_does_not_filter(monkeypatch):
    log = _patch_client(
        monkeypatch,
        [{"treatment_arm": 1.0, "persistent_180d": 1.0, "disease_severity": 0.3}],
    )
    df, _select = await causal._load_agent_estimation_frame(
        dataset="patient_journeys",
        treatment_var="treatment_arm",
        outcome_var="persistent_180d",
        covariates=["disease_severity"],
        limit=1500,
    )
    assert "eq" not in log  # no brand -> no .eq filter
    assert "brand" not in df.columns
    assert len(df) == 1
