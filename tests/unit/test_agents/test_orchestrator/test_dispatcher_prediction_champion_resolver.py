"""prediction_synthesizer resolver: registry-driven champion service (#1351/#1354).

Before this change the resolver failed closed on EVERY chat dispatch with
"no registered champion model" — written when that was measured fact (#F14).
The #1354 ruling promotes the three ``hcp_adoption_{brand}_goldstd_lr_v1``
models to production champions (PR #1384's calibrate+promote script), so the
resolver must now consult the registry (champion lookup = registry query,
NEVER hardcoded model ids) and:

* bind a real (entity_id, prediction_target) when the ask names an adoption
  question, a brand, and a specific HCP entity;
* fail closed HONESTLY otherwise — naming the real champion target that exists
  (and what is missing) instead of claiming no champion is registered;
* keep the pre-champion fail-closed behaviour for asks that match no champion
  family (nothing fabricated, no registry probe wasted).

The registry probe is faked at the seam (``_probe_prediction_champions``): the
matching logic is what these tests pin; the probe's SQL shape mirrors
``MLModelRegistryRepository.get_models_for_target`` (production stage +
loadable artifact + non-synthetic + is_champion) and is exercised against the
real DB by the integration layer, not here.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import pytest

from src.agents.orchestrator.nodes import dispatcher as disp
from src.agents.orchestrator.nodes.dispatcher import NeedsStructuredInput

Q14 = "Predict which HCP segments are most likely to increase Fabhalta prescriptions next quarter"

CHAMPIONS: List[Tuple[str, str]] = [
    ("hcp_adoption_fabhalta_goldstd_lr_v1", "hcp_adoption_fabhalta"),
    ("hcp_adoption_kisqali_goldstd_lr_v1", "hcp_adoption_kisqali"),
    ("hcp_adoption_remibrutinib_goldstd_lr_v1", "hcp_adoption_remibrutinib"),
]


def _agent_input(query: str, *, user_context=None) -> Dict[str, Any]:
    return {
        "query": query,
        "user_context": user_context if user_context is not None else {},
        "session_id": "sess-ps",
        "parsed_query": {},
    }


def _dispatch(params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    return {
        "agent_name": "prediction_synthesizer",
        "priority": "critical",
        "parameters": params or {},
        "timeout_ms": 15000,
        "fallback_agent": None,
        "execution_mode": "parallel",
    }


@pytest.fixture()
def champions(monkeypatch):
    monkeypatch.setattr(disp, "_probe_prediction_champions", lambda: list(CHAMPIONS))
    # Entity-existence probe: default every named entity to "exists".
    monkeypatch.setattr(disp, "_hcp_entity_exists", lambda _e: True)
    return CHAMPIONS


class TestFamilyMatching:
    def test_q14_matches_the_adoption_family(self) -> None:
        assert disp._match_prediction_family(Q14) == "hcp_adoption"

    def test_explicit_adoption_wording_matches(self) -> None:
        assert disp._match_prediction_family("How likely is this HCP to adopt Kisqali?") == (
            "hcp_adoption"
        )
        assert disp._match_prediction_family(
            "which prescribers will start prescribing Fabhalta"
        ) == "hcp_adoption"

    def test_unrelated_forecast_matches_nothing(self) -> None:
        assert disp._match_prediction_family("what's the forecast?") is None
        assert disp._match_prediction_family("forecast Q3 revenue") is None


class TestChampionBinding:
    def test_entity_plus_brand_binds_the_registry_champion(self, champions) -> None:
        resolved = disp.INPUT_RESOLVERS["prediction_synthesizer"](
            _agent_input("How likely is scvhcp_00042 to adopt Fabhalta?"), _dispatch()
        )
        assert isinstance(resolved, dict)
        assert resolved["entity_id"] == "scvhcp_00042"
        assert resolved["prediction_target"] == "hcp_adoption_fabhalta"
        assert resolved["entity_type"] == "hcp"

    def test_brand_binding_is_registry_discovered_not_hardcoded(self, monkeypatch) -> None:
        # A registry with a DIFFERENT target naming still binds — the match is
        # token-driven over what the registry actually serves.
        monkeypatch.setattr(
            disp,
            "_probe_prediction_champions",
            lambda: [("model_x", "kisqali_hcp_adoption_v2")],
        )
        monkeypatch.setattr(disp, "_hcp_entity_exists", lambda _e: True)
        resolved = disp.INPUT_RESOLVERS["prediction_synthesizer"](
            _agent_input("will scvhcp_00007 adopt Kisqali?"), _dispatch()
        )
        assert isinstance(resolved, dict)
        assert resolved["prediction_target"] == "kisqali_hcp_adoption_v2"

    def test_unknown_entity_fails_closed_naming_it(self, champions, monkeypatch) -> None:
        monkeypatch.setattr(disp, "_hcp_entity_exists", lambda _e: False)
        resolved = disp.INPUT_RESOLVERS["prediction_synthesizer"](
            _agent_input("will scvhcp_99999 adopt Fabhalta?"), _dispatch()
        )
        assert isinstance(resolved, NeedsStructuredInput)
        assert "scvhcp_99999" in resolved.reason

    def test_no_entity_fails_closed_naming_the_real_champion(self, champions) -> None:
        resolved = disp.INPUT_RESOLVERS["prediction_synthesizer"](
            _agent_input(Q14), _dispatch()
        )
        assert isinstance(resolved, NeedsStructuredInput)
        # The message must be HONEST about registry state: the champion exists;
        # what's missing is the specific entity.
        assert "hcp_adoption_fabhalta" in resolved.reason
        assert "entity_id" in resolved.missing
        assert "no registered champion" not in resolved.reason.lower()

    def test_no_brand_fails_closed_listing_served_targets(self, champions) -> None:
        resolved = disp.INPUT_RESOLVERS["prediction_synthesizer"](
            _agent_input("which HCPs are most likely to adopt?"), _dispatch()
        )
        assert isinstance(resolved, NeedsStructuredInput)
        # All three served brands surface so the user can scope the re-ask.
        assert "fabhalta" in resolved.reason.lower()
        assert "kisqali" in resolved.reason.lower()
        assert "remibrutinib" in resolved.reason.lower()

    def test_family_match_but_empty_registry_fails_closed_honestly(self, monkeypatch) -> None:
        monkeypatch.setattr(disp, "_probe_prediction_champions", lambda: [])
        resolved = disp.INPUT_RESOLVERS["prediction_synthesizer"](
            _agent_input("will scvhcp_00042 adopt Fabhalta?"), _dispatch()
        )
        assert isinstance(resolved, NeedsStructuredInput)
        assert "production champion" in resolved.reason.lower()

    def test_registry_probe_failure_fails_closed_not_raises(self, monkeypatch) -> None:
        def _boom():
            raise RuntimeError("db down")

        monkeypatch.setattr(disp, "_probe_prediction_champions", _boom)
        resolved = disp.INPUT_RESOLVERS["prediction_synthesizer"](
            _agent_input("will scvhcp_00042 adopt Fabhalta?"), _dispatch()
        )
        assert isinstance(resolved, NeedsStructuredInput)


class TestExistingContractPreserved:
    def test_explicit_params_still_pass_through(self) -> None:
        params = {"entity_id": "HCP-993", "prediction_target": "conversion"}
        resolved = disp.INPUT_RESOLVERS["prediction_synthesizer"](
            _agent_input("predict"), _dispatch(params)
        )
        assert isinstance(resolved, dict)
        assert resolved["entity_id"] == "HCP-993"
        assert resolved["prediction_target"] == "conversion"

    def test_bare_forecast_still_fails_closed_without_probing(self, monkeypatch) -> None:
        def _boom():
            raise AssertionError("registry must not be probed for a non-champion ask")

        monkeypatch.setattr(disp, "_probe_prediction_champions", _boom)
        resolved = disp.INPUT_RESOLVERS["prediction_synthesizer"](
            _agent_input("what's the forecast?"), _dispatch()
        )
        assert isinstance(resolved, NeedsStructuredInput)
        assert "entity_id" in resolved.missing
        assert "fabricat" in resolved.to_error().lower()


class TestTimeHorizon:
    def test_next_quarter_maps_to_90d(self, champions) -> None:
        resolved = disp.INPUT_RESOLVERS["prediction_synthesizer"](
            _agent_input("will scvhcp_00042 adopt Fabhalta next quarter?"), _dispatch()
        )
        assert isinstance(resolved, dict)
        assert resolved["time_horizon"] == "90d"


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-q"])
