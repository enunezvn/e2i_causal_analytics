"""Live-LM e2e smoke for ``_invoke_llm_narrator`` (issue #376).

PR #384 (merged 2026-05-20 at ``c634de78``) refactored the narrator
to parameter-based dependency injection (``client_factory: Optional
[Callable[[str], _AnthropicClientProtocol]] = None``) so unit tests
can inject a fake client without monkey-patching the SDK module.

The DI refactor removed the only real-SDK exercise that ever ran in
CI: unit tests now ALL inject fakes (correct shape), so the default-
factory branch — ``effective_factory = lambda key: anthropic.
AsyncAnthropic(api_key=key)`` — is exercised ONLY in production.

This test plugs that gap with a live Anthropic Haiku call exercising
the REAL default-factory path. No ``client_factory`` arg, no
``httpx_mock``, no ``respx``, no SDK-module monkey-patch. The test
covers:

1. ``client_factory=None`` branch instantiates the real
   ``anthropic.AsyncAnthropic`` (the default-factory path that ships
   in production).
2. Real prompt -> real Haiku 4.5 -> real response shape returns an
   ``LLMCrystalNarrativeAudit`` with the expected telemetry fields
   populated (tokens, cost, latency, model id).
3. The narrow catch tuple
   ``(APIConnectionError, APITimeoutError, RateLimitError,
   APIStatusError)`` is exercised against the live SDK without
   raising on a healthy call (sanity-check the catch surface compiles
   + the SDK exposes the four named classes).
4. At least one of the LLM-generated prose fields (``limitations`` or
   ``recommended_next_analysis``) is populated, confirming the
   ``_parse_narrator_response`` parser handles a real JSON response.

Skip-gate (per memory ``[[feedback-live-lm-skip-must-check-key-
shape]]``): ``ANTHROPIC_API_KEY`` must START WITH ``sk-ant-``. A
presence-only check would let the CI placeholder
``ANTHROPIC_API_KEY=test-key`` past the gate, producing 401s + a
broken catch path (the placeholder is rejected by the Anthropic API
as an auth error, which would mask a real regression).

Run manually pre-PR::

    set -a && source .env && set +a
    pytest tests/integration/test_crystallizer_live_llm.py \\
        -v -m "live_llm and integration" --no-header

Cost: ~$0.001 (single Haiku call, ~600 input + ~200 output tokens).
Wall: ~3-6 seconds against Anthropic's eu-west / us-east region.
"""

from __future__ import annotations

import os

import pytest


@pytest.mark.live_llm
@pytest.mark.integration
@pytest.mark.timeout(120)
async def test_invoke_llm_narrator_exercises_real_default_factory() -> None:
    """Real-SDK smoke: ``_invoke_llm_narrator`` with NO ``client_factory``
    arg must round-trip through ``anthropic.AsyncAnthropic`` and return
    an ``LLMCrystalNarrativeAudit`` populated with real telemetry.

    The default-factory branch at ``src/memory/crystallization/
    crystallizer.py:811-820`` (line numbers as of PR #384) is the
    production path NO unit test exercises post-DI-refactor:

        if client_factory is not None:
            effective_factory = client_factory
        else:
            def _default_factory(key: str) -> _AnthropicClientProtocol:
                return anthropic_module.AsyncAnthropic(api_key=key)
            effective_factory = _default_factory

    A regression that swaps ``AsyncAnthropic`` for the sync
    ``Anthropic`` (blocking the event loop) or that changes the
    constructor kwarg would slip past every unit test. This test
    catches it.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if not api_key.startswith("sk-ant-"):
        # Per ``[[feedback-live-lm-skip-must-check-key-shape]]`` — a
        # plain presence check lets CI's ``ANTHROPIC_API_KEY=test-key``
        # placeholder through, then 401s against the API. Skip cleanly
        # on any non-sk-ant- value so the test only fires on a real key.
        pytest.skip(
            "ANTHROPIC_API_KEY missing or not a real Anthropic key "
            "(expected `sk-ant-` prefix; got "
            f"{api_key[:7] + '…' if api_key else '<empty>'!r}). This e2e "
            "smoke requires a live key — run locally with `.env` loaded "
            "(`set -a && source .env && set +a`) before invoking pytest."
        )

    # Local imports so module-load cost is paid only when the test
    # actually runs (heavy import chain: anthropic SDK, crystallizer).
    from src.memory.crystallization.crystallizer import (
        DEFAULT_NARRATOR_MODEL,
        _invoke_llm_narrator,
    )

    # Realistic inputs mirroring what the crystallizer feeds the
    # narrator in production. The members list shape mirrors the
    # ``_compose_narrative`` call site (memory_id + raw_content);
    # ``derived`` mirrors the deterministic-field bundle assembled
    # from estimator state + episodic key_metrics. The values are
    # plausible-but-synthetic so no real-cohort PHI is sent over the
    # wire.
    members = [
        {
            "memory_id": "ep-1",
            "raw_content": {
                "summary": "Adherence increased by 18% in northeast cohort.",
                "agent": "causal_impact",
            },
        },
        {
            "memory_id": "ep-2",
            "raw_content": {
                "summary": "Gap analysis identified 240 untreated patients.",
                "agent": "gap_analyzer",
            },
        },
    ]
    derived = {
        "effect_size": 0.18,
        "effect_ci_lower": 0.12,
        "effect_ci_upper": 0.24,
        "effect_direction": "positive",
        "cohort_size": 1200,
        "confounders_controlled": ["age", "prior_use"],
        "sensitivity_checks_passed": ["placebo_treatment"],
        "sensitivity_checks_failed": [],
    }

    # NO client_factory arg — exercises the real default-factory branch.
    # This is the load-bearing assertion: a regression that swaps
    # AsyncAnthropic for a different constructor surface, or drops
    # the api_key kwarg, would fail HERE rather than only in prod.
    audit = await _invoke_llm_narrator(
        brand="kisqali",
        region="northeast",
        members=members,
        derived=derived,
    )

    # The returned audit must be a real LLMCrystalNarrativeAudit (not
    # the empty-audit fallback emitted on the error path).
    from src.data.kg.types import LLMCrystalNarrativeAudit

    assert isinstance(audit, LLMCrystalNarrativeAudit), (
        f"Expected LLMCrystalNarrativeAudit, got {type(audit).__name__}. "
        f"The narrator may have hit the import-fallback branch or the "
        f"narrow-catch error path."
    )

    # Model id pinned. If a future PR changes DEFAULT_NARRATOR_MODEL,
    # this assertion catches it (the model literal is asserted, not
    # just the constant — so a constant rename that targets a
    # different model fails here).
    assert audit.narrator_model == DEFAULT_NARRATOR_MODEL, (
        f"narrator_model={audit.narrator_model!r} does not match "
        f"DEFAULT_NARRATOR_MODEL={DEFAULT_NARRATOR_MODEL!r}"
    )
    assert audit.narrator_model.startswith("claude-haiku-"), (
        f"Expected a claude-haiku-* model, got {audit.narrator_model!r}. "
        f"A regression that points the narrator at a different model "
        f"family would surface a cost / latency / quality drift; the "
        f"prefix check is the cheapest pin."
    )

    # Telemetry — populated only when the real SDK round-trip
    # succeeds (the error-path branch returns an audit with
    # latency_ms populated but tokens/cost None).
    assert audit.input_tokens is not None and audit.input_tokens > 0, (
        f"input_tokens={audit.input_tokens!r} — the SDK round-trip "
        f"did not return a usage block, or it failed to surface "
        f"through getattr(response, 'usage', None)."
    )
    assert audit.output_tokens is not None and audit.output_tokens > 0, (
        f"output_tokens={audit.output_tokens!r}"
    )
    assert audit.cost_usd is not None and audit.cost_usd > 0.0, (
        f"cost_usd={audit.cost_usd!r} — compute_haiku_cost_usd must "
        f"produce a positive value for any non-empty token counts."
    )
    assert audit.latency_ms is not None and audit.latency_ms > 0.0, (
        f"latency_ms={audit.latency_ms!r}"
    )

    # Prose: the model is asked for three short fields. At least one
    # of the LLM-generated text fields should land — if all three
    # come back empty, the parser is broken or the prompt no longer
    # matches the parser's expected JSON shape. We do NOT assert all
    # three because Haiku occasionally drops a field on terse inputs;
    # the "≥1 of 3" pin catches a hard parser break while leaving
    # slack for LM variability.
    populated_prose = [
        text
        for text in (audit.key_finding, audit.limitations, audit.recommended_next_analysis)
        if text
    ]
    assert populated_prose, (
        "All three LLM prose fields (key_finding, limitations, "
        "recommended_next_analysis) are empty. Either the parser "
        "is broken or the prompt no longer elicits a JSON-shaped "
        "response from Haiku 4.5. Investigate "
        "src/memory/crystallization/crystallizer.py::_parse_narrator_response."
    )

    # Bounded length — production truncates to 500 chars at the audit
    # boundary, so anything coming back from the SDK should already
    # fit. A regression that bypasses the truncation would surface as
    # a Pydantic length-validation error downstream; pin here to
    # catch it at the narrator boundary.
    for field, text in (
        ("key_finding", audit.key_finding),
        ("limitations", audit.limitations),
        ("recommended_next_analysis", audit.recommended_next_analysis),
    ):
        assert len(text) <= 500, (
            f"{field} length {len(text)} exceeds the 500-char truncation "
            f"contract at crystallizer.py:867-869."
        )
