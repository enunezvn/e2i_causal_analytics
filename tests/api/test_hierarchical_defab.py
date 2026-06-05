"""C1 — /api/causal/hierarchical/analyze must not fabricate input data.

Background (causal-validation-pipeline-review-20260605, finding C1):
    `_execute_hierarchical_analysis` ran the REAL EconML HierarchicalAnalyzer
    over `np.random` data (np.random.seed(42); binomial/normal/randn; a baked-in
    treatment_effect=5.0+modifier*3.0) and returned COMPLETED segment CATEs with
    NO demo gate and NO provenance label — the exact H5/F-005 fabrication shape
    the team already purged from every *sibling* endpoint in this file.

These tests assert the de-fabrication contract (mirrors test_causal_pipeline_no_random):
    - Default (no demo_mode, no inline data) MUST fail-closed with 503.
    - Explicit demo_mode=true returns clearly-labeled pinned-zero values (is_demo=true).
    - Real inline estimation_data_records flow through as DATA (not just names).
    - Source pin: the handler contains no np.random fabrication.
"""

from __future__ import annotations

import inspect
import types

import pytest

from src.api.routes import causal as causal_module

pytestmark = pytest.mark.integration


# =============================================================================
# Static-source regression pin (cheapest assertion: forbid the primitive)
# =============================================================================


class TestNoNpRandomInHierarchicalSource:
    def test_execute_hierarchical_has_no_np_random(self):
        """_execute_hierarchical_analysis must not fabricate input via np.random."""
        source = inspect.getsource(causal_module._execute_hierarchical_analysis)
        assert "np.random" not in source, (
            "C1 regression: np.random fabrication reintroduced in _execute_hierarchical_analysis"
        )
        assert "Generate mock data" not in source, (
            "C1 regression: mock-data block reintroduced in hierarchical handler"
        )

    def test_run_hierarchical_endpoint_exposes_demo_mode(self):
        """The handler must expose a demo_mode gate (sibling-endpoint idiom)."""
        sig = inspect.signature(causal_module.run_hierarchical_analysis)
        assert "demo_mode" in sig.parameters, (
            "C1 regression: run_hierarchical_analysis must expose a demo_mode gate"
        )


# =============================================================================
# Behavior tests
# =============================================================================


@pytest.fixture
def hierarchical_request():
    """Hierarchical request with NO inline data (default-path → 503)."""
    return {
        "treatment_var": "promotion",
        "outcome_var": "trx",
        "effect_modifiers": ["age"],
        "n_segments": 2,
        "min_segment_size": 10,
    }


class TestHierarchicalDefaultPath503:
    def test_default_path_returns_503(self, hierarchical_request):
        """No demo_mode + no inline DataFrame → 503 (honest fail-close)."""
        from fastapi.testclient import TestClient

        from src.api.main import app

        client = TestClient(app)
        response = client.post(
            "/api/causal/hierarchical/analyze",
            json=hierarchical_request,
        )
        assert response.status_code == 503, (
            f"Default path must fail-closed with 503, got {response.status_code}: "
            f"{response.text[:500]}"
        )


class TestHierarchicalAsyncFailsClosed:
    def test_async_non_demo_no_data_returns_503(self, hierarchical_request):
        """async_mode=true + no inline data + non-demo → 503 up-front (C1).

        The submission must be rejected BEFORE it is accepted as pending, so the
        503 fail-close signal is not lost to a generic background FAILED record.
        """
        from fastapi.testclient import TestClient

        from src.api.main import app

        client = TestClient(app)
        response = client.post(
            "/api/causal/hierarchical/analyze?async_mode=true",
            json=hierarchical_request,
        )
        assert response.status_code == 503, (
            f"Async non-demo path must fail-closed with 503, got {response.status_code}"
        )


class TestHierarchicalMissingColumns:
    def test_missing_required_columns_returns_400(self, hierarchical_request):
        """Inline records missing treatment/outcome/modifier columns → 400."""
        from fastapi.testclient import TestClient

        from src.api.main import app

        # Records present but lacking the required 'trx'/'age' columns.
        req = dict(hierarchical_request)
        req["filters"] = {"estimation_data_records": [{"promotion": i % 2} for i in range(8)]}

        client = TestClient(app)
        response = client.post("/api/causal/hierarchical/analyze", json=req)
        assert response.status_code == 400, (
            f"Missing required columns must return 400, got {response.status_code}: "
            f"{response.text[:300]}"
        )


class TestHierarchicalDemoMode:
    def test_demo_mode_returns_pinned_zeros_labeled(self, hierarchical_request):
        """demo_mode=true → 200, is_demo=true, pinned-zero segments, demo warning."""
        from fastapi.testclient import TestClient

        from src.api.main import app

        client = TestClient(app)
        response = client.post(
            "/api/causal/hierarchical/analyze?demo_mode=true",
            json=hierarchical_request,
        )
        assert response.status_code == 200, (
            f"demo_mode=true must succeed, got {response.status_code}: {response.text[:300]}"
        )
        data = response.json()
        assert data.get("is_demo") is True, f"Demo response must set is_demo=true: {data}"
        # Every segment must be a pinned zero — never an RNG-derived value.
        for seg in data.get("segment_results", []):
            assert seg.get("cate_mean") == 0.0, f"Demo segment must be pinned-zero: {seg}"
            assert seg.get("cate_ci_lower") == 0.0
            assert seg.get("cate_ci_upper") == 0.0
        warnings = data.get("warnings", [])
        assert any("demo_mode" in w.lower() or "is_demo" in w.lower() for w in warnings), (
            f"Demo response must include a demo_mode warning, got: {warnings}"
        )


class TestHierarchicalRealDataFlowsThrough:
    def test_real_records_are_read_as_data_not_just_names(self, monkeypatch, hierarchical_request):
        """Inline estimation_data_records flow through to the analyzer as DATA.

        Patches HierarchicalAnalyzer.analyze (avoids heavy EconML compute on the
        low-memory host) and asserts the treatment/outcome columns the analyzer
        receives equal the SUPPLIED records — proving the handler reads columns as
        data, not merely as names against a fabricated frame.
        """
        from fastapi.testclient import TestClient

        from src.api.main import app
        from src.causal_engine.hierarchical import (
            HierarchicalAnalyzer,
            NestedConfidenceInterval,
        )

        captured: dict = {}

        async def fake_analyze(self, X, treatment, outcome):  # noqa: ANN001
            captured["treatment"] = [int(v) for v in treatment]
            captured["outcome"] = [float(v) for v in outcome]
            seg = types.SimpleNamespace(
                segment_id=0,
                segment_name="seg0",
                n_samples=len(captured["treatment"]),
                uplift_range=(0.0, 1.0),
                cate_mean=0.5,
                cate_std=0.1,
                cate_ci_lower=0.3,
                cate_ci_upper=0.7,
                success=True,
                error_message=None,
            )
            return types.SimpleNamespace(
                segment_results=[seg],
                overall_ate=0.5,
                overall_ate_ci_lower=0.3,
                overall_ate_ci_upper=0.7,
                segment_heterogeneity=10.0,
                n_segments=1,
                errors=[],
                warnings=[],
            )

        monkeypatch.setattr(HierarchicalAnalyzer, "analyze", fake_analyze)

        # Decouple from the real nested-CI engine (a single-segment stub trips a
        # pre-existing inf edge case in aggregation — P7/H6 territory, not C1).
        def fake_compute(self, estimates):  # noqa: ANN001
            return types.SimpleNamespace(
                aggregate_ate=0.5,
                aggregate_ci_lower=0.3,
                aggregate_ci_upper=0.7,
                aggregate_std=0.1,
                confidence_level=0.95,
                aggregation_method="variance_weighted",
                segment_contributions={"0": 1.0},
                i_squared=0.0,
                tau_squared=0.0,
                n_segments_included=1,
                total_sample_size=8,
            )

        monkeypatch.setattr(NestedConfidenceInterval, "compute", fake_compute)

        records = [{"promotion": i % 2, "trx": 100.0 + i, "age": 30 + i} for i in range(8)]
        req = dict(hierarchical_request)
        req["filters"] = {"estimation_data_records": records}

        client = TestClient(app)
        response = client.post("/api/causal/hierarchical/analyze", json=req)
        assert response.status_code == 200, (
            f"Real-data path must succeed, got {response.status_code}: {response.text[:300]}"
        )
        data = response.json()
        assert data.get("status") == "completed"
        assert data.get("is_demo") in (False, None), "Real path must NOT be labeled demo"
        # The analyzer received the SUPPLIED treatment/outcome — not RNG data.
        assert captured["treatment"] == [r["promotion"] for r in records]
        assert captured["outcome"] == [r["trx"] for r in records]


class TestNestedCIUsesTrueSE_H6:
    """H6 - the hierarchical API handler must build SegmentEstimate.ate_std from
    the segment ATE's TRUE standard error (cate_se), not the per-unit CATE
    dispersion (cate_std). cate_std does not shrink with n and inflates the
    inverse-variance weights / I^2 / tau^2, making aggregate CIs ~sqrt(n) too wide.
    """

    def test_api_handler_builds_ate_std_from_cate_se(self):
        """Source pin: _execute_hierarchical_analysis must prefer seg.cate_se."""
        source = inspect.getsource(causal_module._execute_hierarchical_analysis)
        # The corrected bridge must reference cate_se as the SE source.
        assert "seg.cate_se" in source, (
            "H6 regression: API hierarchical handler must feed the true SE "
            "(seg.cate_se) into SegmentEstimate.ate_std, not raw cate_std"
        )
        # The corrected, full expression must be present verbatim.
        assert (
            "ate_std=(seg.cate_se if seg.cate_se is not None else (seg.cate_std or 0.01))"
            in source
        ), "H6: API ate_std must be the cate_se-preferring expression"
        # The buggy raw-dispersion bridge must be gone.
        assert "ate_std=seg.cate_std or 0.01" not in source, (
            "H6 regression: API handler still feeds raw cate_std as the standard error"
        )
