import asyncio

import pytest

from src.insights.common import cache_get, cache_key, cache_set, normalize_list, run_signature


def test_normalize_list_from_str_splits_lines():
    assert normalize_list("- a\n- b\n- c") == ["a", "b", "c"]


def test_normalize_list_from_list_trims_and_caps():
    assert normalize_list([" x ", "", "y"]) == ["x", "y"]


def test_run_signature_returns_none_without_lm(monkeypatch):
    # No OPENAI_API_KEY -> ensure_dspy_configured() is False -> None (caller falls back)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    import dspy

    # Ensure no globally-configured LM leaks in from another test in the same session.
    monkeypatch.setattr(dspy.settings, "lm", None, raising=False)
    assert run_signature(object, foo="bar") is None


def test_cache_key_is_stable_and_input_sensitive():
    k1 = cache_key("knowledge-graph", "Kisqali", {"n": 10})
    k2 = cache_key("knowledge-graph", "Kisqali", {"n": 10})
    k3 = cache_key("knowledge-graph", "Kisqali", {"n": 11})
    assert k1 == k2 and k1 != k3
    assert k1.startswith("insight:knowledge-graph:")


def test_cache_roundtrip_when_redis_available():
    """Prove cache_set/cache_get actually round-trip against a LIVE redis (async
    client). Skips when redis is unavailable (e.g. CI) — no mocking."""

    async def _run():
        import src.memory.services.factories as fac

        # Force a fresh client bound to THIS event loop (redis.asyncio is loop-bound).
        fac._redis_client = None
        try:
            await fac.get_redis_client().ping()
        except Exception:
            return "UNAVAILABLE"
        key = cache_key("test", "roundtrip", {"x": 1})
        await cache_set(key, {"insight": "hi", "is_fallback": True}, ttl_seconds=30)
        return await cache_get(key)

    got = asyncio.run(_run())
    if got == "UNAVAILABLE":
        pytest.skip("redis unavailable")
    assert got is not None and got["insight"] == "hi" and got["is_fallback"] is True


# ---- #1880: enumeration markers are prose structure, not numeric claims ---------
# The LM mints list-index digits ("5.") without grounding; the fail-closed
# numeric guards strip them before extraction instead of rejecting the output.


def test_strip_enum_markers_removes_dot_and_paren_forms():
    from src.insights.common import strip_enum_markers

    assert strip_enum_markers("1. Lift TRx Share first.") == "Lift TRx Share first."
    assert strip_enum_markers("12) Expand coverage.") == "Expand coverage."


def test_strip_enum_markers_multiline_and_indented():
    from src.insights.common import strip_enum_markers

    assert (
        strip_enum_markers("1. First action\n  2. Second action") == "First action\nSecond action"
    )


def test_strip_enum_markers_never_touches_real_figures():
    from src.insights.common import strip_enum_markers

    for s in (
        "5.2% is the observed gap",  # decimal: period followed by a digit
        "1.85 ROI needs attention",
        "-0.02 is the lower CI bound",
        "42 of 45 defined KPIs computed",  # bare integer, no marker punctuation
        "ranked 1. among peers",  # mid-line, not a list marker
        "Gates: proceed=1",
    ):
        assert strip_enum_markers(s) == s


def test_strip_enum_markers_requires_trailing_space():
    from src.insights.common import strip_enum_markers

    # A bare trailing "4." is a sentence-final figure, not a list marker.
    assert strip_enum_markers("4.") == "4."


def test_strip_enum_markers_idempotent_and_empty_safe():
    from src.insights.common import strip_enum_markers

    once = strip_enum_markers("3. Do the thing")
    assert strip_enum_markers(once) == once
    assert strip_enum_markers("") == ""
