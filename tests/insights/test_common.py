from src.insights.common import normalize_list, run_signature, cache_key


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
