"""Tests for F2(a): the dispatcher threads a real cohort DataFrame into the
``tool_composer`` agent's input under ``data`` (which ``ToolComposerAgent.run``
normalizes to ``context["estimation_data"]``).

Context
-------
Two live production paths reach the Tool Composer. The CHAT path
(``chatbot_tools.tool_composer_tool``) already resolves a cohort frame for
``(brand, region)`` and threads it as ``estimation_data``. The ORCHESTRATOR
path did not: ``intent_classifier`` -> ``router.multi_faceted`` (dispatch with
``parameters={}``) -> ``DispatcherNode._prepare_agent_input`` threaded only
``query``/``user_context``/``parameters``/``session_id``/``parsed_query`` -- NO
data. ``ToolComposerAgent.run`` reads ``input_data["data"]`` but the dispatcher
never supplied it, so multi_faceted queries via the orchestrator delivered 0
data and the real causal tools all fail-closed.

F12/F13/F14 generalized this single special-case into the
``dispatcher.INPUT_RESOLVERS`` registry — the tool_composer data resolution now
lives in ``_resolve_tool_composer_input`` and is applied in ``_dispatch_agent``
after ``_prepare_agent_input`` builds the generic payload. These tests pin the
SAME wiring (no network): they patch ``resolve_cohort_frame`` (imported lazily
inside the dispatcher helper) and assert on the resolver's output, including the
fail-closed contract for unrecognized brands and resolver exceptions.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Union

import pandas as pd
import pytest

from src.agents.orchestrator.nodes import dispatcher as disp
from src.agents.orchestrator.nodes.dispatcher import DispatcherNode, NeedsStructuredInput


def _state_with_entities(
    brand: Optional[str] = None,
    region: Optional[str] = None,
    *,
    user_context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Return an OrchestratorState-like dict with brand/region parsed entities."""
    entities = []
    if brand is not None:
        entities.append({"type": "brand", "value": brand, "confidence": 0.95, "source": "exact"})
    if region is not None:
        entities.append({"type": "region", "value": region, "confidence": 0.9, "source": "exact"})
    return {
        "query": "What drives adoption and where are the gaps?",
        "user_context": user_context if user_context is not None else {"user_id": "u1"},
        "session_id": "sess-1",
        "parsed_query": {"intent": "causal_impact", "entities": entities},
    }


def _tool_composer_dispatch() -> Dict[str, Any]:
    return {
        "agent_name": "tool_composer",
        "priority": "high",
        "parameters": {},
        "timeout_ms": 90000,
        "fallback_agent": None,
        "execution_mode": "parallel",
    }


def _other_dispatch(agent_name: str) -> Dict[str, Any]:
    return {
        "agent_name": agent_name,
        "priority": "high",
        "parameters": {},
        "timeout_ms": 30000,
        "fallback_agent": None,
        "execution_mode": "parallel",
    }


def _resolved_input(
    node: DispatcherNode, state: Dict[str, Any], dispatch: Dict[str, Any]
) -> Union[Dict[str, Any], NeedsStructuredInput]:
    """Mirror ``_dispatch_agent``'s flow for a non-kwargs agent: build the generic
    payload, then apply the agent's INPUT_RESOLVER (if any) and merge.

    Returns the merged input dict (what the agent method would receive) or a
    ``NeedsStructuredInput`` when the resolver fails closed. tool_composer never
    fails closed, so for it this always returns a dict.
    """
    prepared = node._prepare_agent_input(state, dispatch)  # type: ignore[arg-type]
    resolver = disp.INPUT_RESOLVERS.get(dispatch["agent_name"])
    if resolver is None:
        return prepared
    resolved = resolver(prepared, dispatch)  # type: ignore[arg-type]
    if isinstance(resolved, NeedsStructuredInput):
        return resolved
    merged = dict(prepared)
    merged.update(resolved)
    return merged


def test_tool_composer_input_carries_real_dataframe(monkeypatch) -> None:
    """A tool_composer dispatch with brand/region entities -> ``data`` is a frame.

    The fake resolver stands in for the real Supabase-backed resolver; the
    assertion is on the WIRING (the dispatcher derives brand/region, calls the
    resolver, and threads the returned frame under ``data``).
    """
    fake_frame = pd.DataFrame({"engagement_score": [1, 2, 3], "treatment_initiated": [0, 1, 1]})
    captured: Dict[str, Any] = {}

    def fake_resolve(brand, region):  # noqa: ANN001
        captured["brand"] = brand
        captured["region"] = region
        return fake_frame

    monkeypatch.setattr("src.services.cohort_resolution.resolve_cohort_frame", fake_resolve)

    node = DispatcherNode()
    prepared = _resolved_input(
        node, _state_with_entities("Kisqali", "Northeast"), _tool_composer_dispatch()
    )

    assert captured == {"brand": "Kisqali", "region": "northeast"}  # normalised (r4 option 3)
    assert "data" in prepared
    assert isinstance(prepared["data"], pd.DataFrame)
    assert len(prepared["data"]) == 3
    # The contract pass-through fields remain intact.
    assert prepared["query"] == "What drives adoption and where are the gaps?"
    assert prepared["session_id"] == "sess-1"


def test_brand_region_fallback_to_user_context(monkeypatch) -> None:
    """When entities are absent, brand/region fall back to ``user_context``."""
    fake_frame = pd.DataFrame({"x": [1]})
    captured: Dict[str, Any] = {}

    def fake_resolve(brand, region):  # noqa: ANN001
        captured["brand"] = brand
        captured["region"] = region
        return fake_frame

    monkeypatch.setattr("src.services.cohort_resolution.resolve_cohort_frame", fake_resolve)

    node = DispatcherNode()
    prepared = _resolved_input(
        node,
        _state_with_entities(user_context={"brand": "Fabhalta", "region": "South"}),
        _tool_composer_dispatch(),
    )

    assert captured == {"brand": "Fabhalta", "region": "south"}  # normalised (r4 option 3)
    assert isinstance(prepared["data"], pd.DataFrame)


def test_unrecognized_brand_proceeds_without_data(monkeypatch) -> None:
    """An unrecognized brand -> resolver returns None -> NO ``data`` key (fail closed).

    The dispatcher must NOT raise and must NOT add a ``data`` key (so
    ``ToolComposerAgent.run`` leaves ``estimation_data`` unset rather than being
    tripped by a ``None`` value). The real resolver returns ``None`` for an
    unrecognized brand; the fake mirrors that.
    """

    def fake_resolve(brand, region):  # noqa: ANN001
        return None

    monkeypatch.setattr("src.services.cohort_resolution.resolve_cohort_frame", fake_resolve)

    node = DispatcherNode()
    prepared = _resolved_input(
        node, _state_with_entities("NotARealBrand", "Northeast"), _tool_composer_dispatch()
    )

    assert "data" not in prepared
    # Pass-through contract still intact.
    assert prepared["query"]


def test_resolver_exception_proceeds_without_data(monkeypatch) -> None:
    """A resolver exception is logged and swallowed -> NO ``data`` key (fail closed)."""

    def fake_resolve(brand, region):  # noqa: ANN001
        raise RuntimeError("supabase down")

    monkeypatch.setattr("src.services.cohort_resolution.resolve_cohort_frame", fake_resolve)

    node = DispatcherNode()
    prepared = _resolved_input(
        node, _state_with_entities("Kisqali", "Northeast"), _tool_composer_dispatch()
    )

    assert "data" not in prepared


def test_no_brand_or_region_skips_resolution(monkeypatch) -> None:
    """With neither brand nor region, the dispatcher skips the resolver call entirely."""
    called = {"n": 0}

    def fake_resolve(brand, region):  # noqa: ANN001
        called["n"] += 1
        return pd.DataFrame({"x": [1]})

    monkeypatch.setattr("src.services.cohort_resolution.resolve_cohort_frame", fake_resolve)

    node = DispatcherNode()
    prepared = _resolved_input(node, _state_with_entities(), _tool_composer_dispatch())

    assert called["n"] == 0
    assert "data" not in prepared


def test_other_agents_have_no_tool_composer_data_resolver() -> None:
    """Scoping: non-tool_composer agents never gain a ``data`` key.

    The registry enforces this structurally — agents that do not declare a
    ``data``-threading resolver simply are not in ``INPUT_RESOLVERS`` (or their
    resolver does not emit ``data``). drift_monitor's wrapped input model would
    TypeError on an undeclared ``data`` kwarg, so it must never receive one.
    """
    for other in ("causal_impact", "gap_analyzer", "drift_monitor"):
        resolver = disp.INPUT_RESOLVERS.get(other)
        if resolver is None:
            continue  # no resolver → can never add a data key
        out = resolver({"query": "q", "session_id": "s"}, _other_dispatch(other))
        if isinstance(out, dict):
            assert "data" not in out, f"{other} resolver unexpectedly emitted a data key"


def test_extract_brand_region_prefers_entities_over_user_context() -> None:
    """Unit-level: parsed_query entities win over user_context for brand/region."""
    payload = {
        "parsed_query": {
            "entities": [
                {"type": "brand", "value": "Kisqali"},
                {"type": "region", "value": "West"},
            ]
        },
        "user_context": {"brand": "Fabhalta", "region": "South"},
    }
    assert disp._extract_brand_region(payload) == ("Kisqali", "west")  # normalised (r4 option 3)


def _kpi_frame(is_truncated: bool):
    """Build a minimal real KpiFrame (no DB) for the truncation-provenance tests."""
    from src.services.kpi_resolution import KpiFrame

    return KpiFrame(
        frame=pd.DataFrame({"accepted": [0, 1], "converted": [0, 1]}),
        outcome_column="converted",
        driver_columns=["accepted"],
        treatment_column="accepted",
        kpi_id="WS3-BI-009",
        kpi_name="Conversion Rate",
        is_truncated=is_truncated,
    )


def test_tool_composer_kpi_truncation_provenance_threaded(monkeypatch) -> None:
    """#810 / codex MED: a TRUNCATED KPI substrate must surface ``kpi_truncated``
    on the orchestrator path (parity with the chatbot path) — never dropped."""
    monkeypatch.setattr("src.services.kpi_resolution.recognize_kpi", lambda _q: object())
    monkeypatch.setattr(
        "src.services.kpi_resolution.resolve_kpi_frame", lambda *a, **k: _kpi_frame(True)
    )

    node = DispatcherNode()
    prepared = _resolved_input(
        node, _state_with_entities("Kisqali", "Northeast"), _tool_composer_dispatch()
    )
    assert isinstance(prepared["data"], pd.DataFrame)
    assert prepared["kpi_outcome"] == "converted"
    assert prepared["kpi_truncated"] is True


def test_tool_composer_kpi_not_truncated_omits_flag(monkeypatch) -> None:
    """A non-truncated KPI substrate must NOT add the ``kpi_truncated`` flag."""
    monkeypatch.setattr("src.services.kpi_resolution.recognize_kpi", lambda _q: object())
    monkeypatch.setattr(
        "src.services.kpi_resolution.resolve_kpi_frame", lambda *a, **k: _kpi_frame(False)
    )

    node = DispatcherNode()
    prepared = _resolved_input(
        node, _state_with_entities("Kisqali", "Northeast"), _tool_composer_dispatch()
    )
    assert prepared["kpi_outcome"] == "converted"
    assert "kpi_truncated" not in prepared


# --------------------------------------------------------------------------
# #2114 codex r4 HIGH-2 — the SEAM this file did not test.
#
# `test_unrecognized_brand_proceeds_without_data` above stubs the resolver with
# a fake that IGNORES its arguments, so it asserts only that the dispatcher
# survives a None result. It passes whether the dispatcher hands over
# 'NotARealBrand' or None, and was therefore blind when b09a3271d started
# erasing unrecognised brands: `_structured_brand` returned None, cohort
# resolution read that as "no brand specified", and the cohort silently widened
# to the whole population -- the exact harm cohort_resolution.py:174-177 exists
# to prevent, by failing closed on a non-empty unrecognised brand.
#
# These tests capture the VALUE the resolver receives, which is the only thing
# that can catch an erasure at this seam.
# --------------------------------------------------------------------------


def _capture_resolver(monkeypatch) -> Dict[str, Any]:
    """Patch the cohort resolver and record exactly what it is handed."""
    captured: Dict[str, Any] = {}

    def fake_resolve(brand, region):  # noqa: ANN001
        captured["brand"] = brand
        captured["region"] = region
        return None  # fail closed, as the real resolver does for a bogus brand

    monkeypatch.setattr("src.services.cohort_resolution.resolve_cohort_frame", fake_resolve)
    return captured


@pytest.mark.parametrize("unserveable", ["Xolair", "NotARealBrand", "Dupixent"])
def test_an_unrecognised_structured_brand_reaches_the_resolver(monkeypatch, unserveable) -> None:
    """A brand the enum cannot serve must still be DELIVERED, so the
    fail-closed downstream can reject it. Erasing it to None reads as "no brand
    specified" and widens the cohort to every patient."""
    captured = _capture_resolver(monkeypatch)

    node = DispatcherNode()
    prepared = _resolved_input(
        node,
        _state_with_entities(user_context={"brand": unserveable, "region": "west"}),
        _tool_composer_dispatch(),
    )

    assert captured["brand"] == unserveable, "an unrecognised brand must not be erased"
    assert "data" not in prepared


def test_a_recognised_brand_is_normalised_on_the_way_to_the_resolver(monkeypatch) -> None:
    """Recognition NORMALISES (the 10e win, kept): padding and casing are
    resolved to the enum label the case-sensitive predicate can match."""
    captured = _capture_resolver(monkeypatch)

    node = DispatcherNode()
    _resolved_input(
        node,
        _state_with_entities(user_context={"brand": " kisqali "}),
        _tool_composer_dispatch(),
    )

    assert captured["brand"] == "Kisqali"


@pytest.mark.parametrize("blank", ["", " ", "​", "﻿", "\x00"])
def test_a_blank_structured_brand_is_still_erased(monkeypatch, blank) -> None:
    """The other half of the distinction: a value naming NOBODY stays None, so
    the text scan may still speak and the clarify gate still fires.

    A region is supplied because the dispatcher short-circuits when brand AND
    region are both None ("nothing to resolve against"), and this test is about
    what the resolver RECEIVES, not about that short-circuit.
    """
    captured = _capture_resolver(monkeypatch)

    node = DispatcherNode()
    _resolved_input(
        node,
        _state_with_entities(user_context={"brand": blank, "region": "west"}),
        _tool_composer_dispatch(),
    )

    assert captured["brand"] is None


def test_an_unresolvable_entity_does_not_hide_a_valid_context_brand(monkeypatch) -> None:
    """HIGH-1 at this seam: source selection must not precede validation."""
    captured = _capture_resolver(monkeypatch)

    node = DispatcherNode()
    state = _state_with_entities(
        "NotARealBrand", user_context={"brand": "competitor", "region": "west"}
    )
    _resolved_input(node, state, _tool_composer_dispatch())

    assert captured["brand"] == "competitor"


# --------------------------------------------------------------------------
# #2114 codex r4 — the REGION half, measured separately rather than assumed.
#
# Region had defect A (an unresolvable candidate masks a servable later one) and
# the blank leak, but NOT the erasure/widening defect: it passed everything
# through raw, so `cohort_resolution`'s region fail-closed still fired. Region is
# therefore returned RAW here — `cohort_resolution._normalize_region` normalises
# it itself (measured: 'West' and ' West ' -> 'west'), so casing is harmless at
# that consumer and normalising would be an unmotivated behaviour change.
# --------------------------------------------------------------------------


def test_an_unresolvable_entity_region_does_not_hide_a_valid_context_region(monkeypatch) -> None:
    """Defect A on the region half."""
    captured = _capture_resolver(monkeypatch)

    node = DispatcherNode()
    state = _state_with_entities(region="NotARegion", user_context={"region": "west"})
    _resolved_input(node, state, _tool_composer_dispatch())

    assert captured["region"] == "west"


@pytest.mark.parametrize("blank", ["", " ", "​", "﻿", "\x00"])
def test_a_blank_structured_region_names_nobody(monkeypatch, blank) -> None:
    """The region blank leak: U+200B survived `.strip()` and read as a decision.

    A brand is supplied so the dispatcher does not short-circuit on
    (None, None) before reaching the resolver.
    """
    captured = _capture_resolver(monkeypatch)

    node = DispatcherNode()
    _resolved_input(
        node,
        _state_with_entities(user_context={"brand": "Kisqali", "region": blank}),
        _tool_composer_dispatch(),
    )

    assert captured["region"] is None


@pytest.mark.parametrize("unserveable", ["NotARegion", "Atlantis", "EMEA"])
def test_an_unrecognised_region_reaches_the_resolver(monkeypatch, unserveable) -> None:
    """Region never had the erasure defect and must not acquire one: the value
    travels so cohort_resolution's own region fail-closed can reject it."""
    captured = _capture_resolver(monkeypatch)

    node = DispatcherNode()
    _resolved_input(
        node,
        _state_with_entities(user_context={"region": unserveable}),
        _tool_composer_dispatch(),
    )

    assert captured["region"] == unserveable


@pytest.mark.parametrize(
    "raw,normalised", [(" West ", "west"), ("WEST", "west"), ("South", "south")]
)
def test_a_recognised_region_is_NORMALISED(monkeypatch, raw, normalised) -> None:
    """REVERSED from this task's own first cut, on measured evidence.

    Option 2 returned region raw on the argument that `cohort_resolution`
    normalises for itself. It does -- but the Branch A KPI path does not: it
    passes `context["region"]` straight into `LOWER(region::text) = LOWER($N)`,
    which folds CASE but never strips WHITESPACE, so a padded ' West ' matched
    no row and the KPI failed closed on a scope the caller gave correctly.
    Normalising here serves every consumer, including the one that cannot.
    """
    captured = _capture_resolver(monkeypatch)

    node = DispatcherNode()
    _resolved_input(
        node,
        _state_with_entities(user_context={"region": raw}),
        _tool_composer_dispatch(),
    )

    assert captured["region"] == normalised


# --------------------------------------------------------------------------
# #2114 codex r5: the entity ingress chose its candidate BEFORE validating it,
# one level below where r4's fix landed. `_entity_value` returned the first
# entity whose value was `.strip()`-truthy and stopped -- and U+200B is not
# whitespace, so a zero-width entity satisfied that test and the valid entity
# BEHIND it was never offered to the vocabulary.
#
# Measured on the pre-fix code, with a positive control:
#     [ZWSP, " Kisqali "]     -> None        (valid entity never seen)
#     ["Xolair", " Kisqali "] -> "Xolair"
#     [" Kisqali "]           -> "Kisqali"   <- control
#
# The fix offers EVERY matching entity, in order, then `user_context`, to the
# one chooser -- so recognition decides across all candidates instead of
# position deciding before recognition. Fifth round on this gate; the point is
# to stop patching shapes.
#
# NOTE ON REACHABILITY, so a future reader weighs these correctly: NOTHING in
# this codebase populates `parsed_query.entities` today -- `ParsedQuery` is a
# TypedDict that is never constructed and `parsed_query` appears nowhere under
# `src/api/`. This ingress is a declared contract awaiting its NLP producer, so
# these tests pin a LATENT defect, not a live one. The `user_context` half of
# the same seam IS live (chatbot_tools.py:1631).
# --------------------------------------------------------------------------


def _state_with_entity_list(
    entities: Any, *, user_context: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """State whose `parsed_query.entities` is EXACTLY what the caller passes --
    several entities of one type, or a malformed container."""
    return {
        "query": "What drives adoption and where are the gaps?",
        "user_context": user_context if user_context is not None else {"user_id": "u1"},
        "session_id": "sess-1",
        "parsed_query": {"intent": "causal_impact", "entities": entities},
    }


def _ent(entity_type: str, value: Any) -> Dict[str, Any]:
    return {"type": entity_type, "value": value, "confidence": 0.95, "source": "exact"}


ZWSP = "​"


def test_one_valid_brand_entity_still_resolves(monkeypatch) -> None:
    """POSITIVE CONTROL for every multi-entity assertion below: the single-entity
    path must work, or 'the second entity was reached' proves nothing."""
    captured = _capture_resolver(monkeypatch)

    node = DispatcherNode()
    _resolved_input(
        node,
        _state_with_entity_list([_ent("brand", " Kisqali ")]),
        _tool_composer_dispatch(),
    )

    assert captured["brand"] == "Kisqali"


@pytest.mark.parametrize("hidden_by", [ZWSP, "﻿", "\x00", "Xolair", "NotARealBrand"])
def test_an_earlier_entity_does_not_hide_a_later_VALID_brand(monkeypatch, hidden_by) -> None:
    """Recognition decides, not position. A junk or unserviceable entity ahead
    of a real one must not consume the slot."""
    captured = _capture_resolver(monkeypatch)

    node = DispatcherNode()
    _resolved_input(
        node,
        _state_with_entity_list([_ent("brand", hidden_by), _ent("brand", " Kisqali ")]),
        _tool_composer_dispatch(),
    )

    assert captured["brand"] == "Kisqali", f"{hidden_by!r} hid the valid entity behind it"


@pytest.mark.parametrize("hidden_by", [ZWSP, "﻿", "NotARegion"])
def test_an_earlier_entity_does_not_hide_a_later_VALID_region(monkeypatch, hidden_by) -> None:
    """The region half of the same defect -- `_entity_value` is shared."""
    captured = _capture_resolver(monkeypatch)

    node = DispatcherNode()
    _resolved_input(
        node,
        _state_with_entity_list([_ent("region", hidden_by), _ent("region", " West ")]),
        _tool_composer_dispatch(),
    )

    assert captured["region"] == "west", f"{hidden_by!r} hid the valid entity behind it"


def test_an_unrecognised_entity_is_still_DELIVERED_when_nothing_resolves(monkeypatch) -> None:
    """Recognition NORMALISES and never DELETES (r4 HIGH-2, kept): when no
    candidate resolves, the first meaningful one is still handed on so the
    fail-closed consumer sees a decision rather than 'no scope given'."""
    captured = _capture_resolver(monkeypatch)

    node = DispatcherNode()
    _resolved_input(
        node,
        _state_with_entity_list(
            [_ent("brand", ZWSP), _ent("brand", "Xolair")],
            user_context={"region": "west"},
        ),
        _tool_composer_dispatch(),
    )

    assert captured["brand"] == "Xolair"


def test_entities_ahead_of_user_context_still_win_when_they_resolve(monkeypatch) -> None:
    """Ordering across SOURCES is unchanged: a resolvable entity outranks the
    stashed context, and only a candidate that names nothing is skipped."""
    captured = _capture_resolver(monkeypatch)

    node = DispatcherNode()
    _resolved_input(
        node,
        _state_with_entity_list(
            [_ent("brand", "Fabhalta")],
            user_context={"brand": "Kisqali", "region": "west"},
        ),
        _tool_composer_dispatch(),
    )

    assert captured["brand"] == "Fabhalta"


# --------------------------------------------------------------------------
# #2114 codex r5 MEDIUM: `(parsed_query.get("entities") ... ) or []` leaves a
# TRUTHY NON-ITERABLE container in place, so `for ent in 1` raised TypeError --
# killing the dispatch even when a perfectly valid `user_context` brand was
# sitting right there. Missing / None / non-dict / non-string entries were all
# handled; this one shape was not. A malformed container means "no structured
# entities", so the context fallback must survive it.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("malformed", [1, True, 3.5, object()])
def test_a_malformed_entities_container_falls_back_to_user_context(monkeypatch, malformed) -> None:
    captured = _capture_resolver(monkeypatch)

    node = DispatcherNode()
    _resolved_input(
        node,
        _state_with_entity_list(malformed, user_context={"brand": "Kisqali", "region": "west"}),
        _tool_composer_dispatch(),
    )

    assert captured["brand"] == "Kisqali"


def test_a_string_entities_container_degrades_to_absent_rather_than_raising(monkeypatch) -> None:
    """Pinned deliberately: a str IS iterable, so this never raised -- each
    character simply fails the dict check and the context fallback wins. Kept so
    a later 'fix' to the container guard cannot turn a working degradation into
    a TypeError."""
    captured = _capture_resolver(monkeypatch)

    node = DispatcherNode()
    _resolved_input(
        node,
        _state_with_entity_list("abc", user_context={"brand": "Kisqali", "region": "west"}),
        _tool_composer_dispatch(),
    )

    assert captured["brand"] == "Kisqali"
