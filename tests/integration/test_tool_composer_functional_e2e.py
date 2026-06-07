"""Functional end-to-end proof for the ToolComposer pipeline (F1 / Rec#6).

This is the gate that would have caught the original "0/4 tools succeeded on
the canonical query" failure. It runs the REAL ToolComposer pipeline
(real decomposer/planner/executor/synthesizer wiring) on the canonical
multi-faceted query with a REAL ~400-row cohort DataFrame in context, and
asserts that genuine tool execution occurred:

  * tools_succeeded > 0
  * at least one EffectEstimate.ate is a finite float
  * synthesis confidence > 0.3

The LLM is stubbed by a deterministic ``_StubChatLLM`` that returns valid
DECOMPOSE / PLAN / SYNTHESIZE JSON keyed on the system prompt. The stub
stands in for the *planner LLM only* -- every tool runs for real against the
real frame. The plan it returns deliberately OMITS any data/estimation_data
kwarg, so a PASS proves the R2 executor auto-injection and R3 real-column
binding carry the DataFrame to the tool.

A second, opt-in variant (``test_canonical_query_real_llm_end_to_end``) runs
the SAME assertions with a real network LLM, gated on an ANTHROPIC/OPENAI key.

Anti-mock note: the test BUILDS its own pandas DataFrame and passes it in
(allowed). The forbidden anti-pattern is fabricating data INSIDE a tool body.
"""

from __future__ import annotations

import json
import math
import os
from typing import Any, List

import numpy as np
import pandas as pd
import pytest

from src.agents.tool_composer.composer import compose_query

CANONICAL_QUERY = "What drove Kisqali conversion in the Northeast, and which segments respond best?"


# ---------------------------------------------------------------------------
# Real ~400-row cohort frame with a genuine, recoverable treatment signal.
# Mirrors tests/unit/test_agents/test_tool_composer/
# test_registry_tools_real_compute.py::_build_cohort_df.
# ---------------------------------------------------------------------------
def _build_cohort_df(*, n: int = 400, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    regions = rng.choice(["northeast", "south", "midwest", "west"], size=n)
    segment = rng.choice(["high_volume", "mid_volume", "low_volume"], size=n)
    days_on_therapy = rng.integers(30, 400, size=n)
    prior_treatments = rng.integers(0, 5, size=n)
    hcp_visits = rng.integers(1, 20, size=n)
    high_engagement = (hcp_visits >= np.median(hcp_visits)).astype(int)
    # Outcome (conversion) correlated with the binary treatment + a covariate
    # => a real, recoverable ATE.
    logit = -0.5 + 1.2 * high_engagement - 0.01 * days_on_therapy + 0.2 * prior_treatments
    prob = 1.0 / (1.0 + np.exp(-logit))
    converted = (rng.random(n) < prob).astype(int)
    return pd.DataFrame(
        {
            "patient_id": [f"pt-{i:04d}" for i in range(n)],
            "geographic_region": regions,
            "segment": segment,
            "days_on_therapy": days_on_therapy,
            "prior_treatments": prior_treatments,
            "hcp_visits": hcp_visits,
            "high_engagement": high_engagement,
            "converted": converted,
        }
    )


# ---------------------------------------------------------------------------
# Deterministic stub LLM. Routes on the SystemMessage text.
# The PLAN it returns references the REAL causal_effect_estimator tool, binds
# treatment/outcome to REAL columns, and OMITS any data kwarg (proving R2/R3).
# ---------------------------------------------------------------------------
class _AIMsg:
    def __init__(self, content: str) -> None:
        self.content = content


def _system_text(messages: List[Any]) -> str:
    for m in messages:
        content = getattr(m, "content", "")
        if (
            isinstance(content, str)
            and "specialist" in content
            or "synthesizer" in str(getattr(m, "content", ""))
        ):
            return str(content)
    # Fall back to the first message's content.
    return str(getattr(messages[0], "content", ""))


class _StubChatLLM:
    """Minimal LangChain-style chat client: implements async ``ainvoke``."""

    _DECOMPOSE = json.dumps(
        {
            "sub_questions": [
                {
                    "id": "sq_1",
                    "question": "What drove Kisqali conversion in the Northeast?",
                    "intent": "CAUSAL",
                    "depends_on": [],
                },
                {
                    "id": "sq_2",
                    "question": "Which segments respond best?",
                    "intent": "COMPARATIVE",
                    "depends_on": [],
                },
            ]
        }
    )

    # NOTE: input_mapping carries NO data/estimation_data key. The executor's
    # R2 auto-injection + R3 column binding must supply the frame. Both steps
    # route to the real, fail-closed causal_effect_estimator so a genuine ATE
    # is computed (never a fabricated tool output).
    _PLAN = json.dumps(
        {
            "reasoning": "Route both sub-questions to causal_effect_estimator.",
            "tool_mappings": [
                {
                    "sub_question_id": "sq_1",
                    "tool_name": "causal_effect_estimator",
                    "confidence": 0.95,
                    "reasoning": "Causal driver question.",
                },
                {
                    "sub_question_id": "sq_2",
                    "tool_name": "causal_effect_estimator",
                    "confidence": 0.9,
                    "reasoning": "Segment comparison via treatment effect.",
                },
            ],
            "execution_steps": [
                {
                    "step_id": "step_1",
                    "sub_question_id": "sq_1",
                    "tool_name": "causal_effect_estimator",
                    "input_mapping": {
                        "treatment": "high_engagement",
                        "outcome": "converted",
                    },
                    "depends_on_steps": [],
                },
                {
                    "step_id": "step_2",
                    "sub_question_id": "sq_2",
                    "tool_name": "causal_effect_estimator",
                    "input_mapping": {
                        "treatment": "high_engagement",
                        "outcome": "converted",
                    },
                    "depends_on_steps": [],
                },
            ],
            "parallel_groups": [["step_1", "step_2"]],
        }
    )

    _SYNTH = json.dumps(
        {
            "answer": (
                "Higher HCP engagement raised Kisqali conversion; effect varies across segments."
            ),
            "confidence": 0.82,
            "supporting_data": {},
            "citations": [],
            "caveats": [],
            "reasoning": "Both causal estimates returned finite ATEs.",
        }
    )

    async def ainvoke(self, messages: List[Any], **_: Any) -> _AIMsg:
        text = _system_text(messages)
        if "decomposition specialist" in text:
            return _AIMsg(self._DECOMPOSE)
        if "tool planning specialist" in text:
            return _AIMsg(self._PLAN)
        if "response synthesizer" in text:
            return _AIMsg(self._SYNTH)
        # Unknown phase -> empty JSON forces a real, observable failure
        # rather than a silent plausible-but-fake answer.
        return _AIMsg("{}")


def _finite_ates(result: Any) -> List[float]:
    """Collect every finite numeric ``ate`` across all successful tool outputs."""
    ates: List[float] = []
    for output in result.execution.get_all_outputs().values():
        if isinstance(output, dict) and "ate" in output:
            val = output["ate"]
            if isinstance(val, (int, float)) and math.isfinite(float(val)):
                ates.append(float(val))
    return ates


# ---------------------------------------------------------------------------
# T1: stub-planner functional gate (always runs, no network).
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_canonical_query_produces_real_tool_successes_stub_planner():
    df = _build_cohort_df()
    context = {
        "brand": "Kisqali",
        "region": "northeast",
        # Canonical in-context DataFrame key (SHARED CONTRACT). R2 auto-injects
        # this into the tool kwargs; R3 binds the plan args to its columns.
        "estimation_data": df,
    }

    result = await compose_query(
        query=CANONICAL_QUERY,
        llm_client=_StubChatLLM(),
        context=context,
    )

    # The original F1 failure: tools_succeeded == 0. This is the gate.
    assert result.execution.tools_succeeded > 0, (
        "F1 regression: the pipeline produced zero successful tools on the "
        "canonical query. The DataFrame did not reach the tools "
        "(R2 auto-inject / R3 column binding broken)."
    )

    ates = _finite_ates(result)
    assert ates, "No finite EffectEstimate.ate in any successful tool output"
    assert all(math.isfinite(a) for a in ates)

    assert result.response.confidence > 0.3, (
        f"Synthesis confidence too low: {result.response.confidence}"
    )


# ---------------------------------------------------------------------------
# T1 (manual-only opt-in): real LLM end to end. Same assertions as the stub
# gate, but driven by the live network LLM.
#
# Gated behind an EXPLICIT flag (E2I_RUN_REAL_LLM_E2E=1), NOT merely the
# presence of an API key: CI sets OPENAI_API_KEY/ANTHROPIC_API_KEY as secrets
# (backend-tests.yml, tier1-5-test.yml) and pytest loads .env on the dev
# droplet, so a key is almost always present — a real LLM can emit malformed
# planning JSON nondeterministically, and that flakiness must NOT gate CI.
# The DETERMINISTIC stub-planner gate above is the F1 regression arbiter; this
# variant is a deliberate manual smoke test. See CLAUDE.md CHEAPEST-DISPROOF-
# FIRST / incident #504 (real-LLM evals made manual-only for exactly this).
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
@pytest.mark.skipif(
    os.getenv("E2I_RUN_REAL_LLM_E2E") != "1"
    or not (os.getenv("ANTHROPIC_API_KEY") or os.getenv("OPENAI_API_KEY")),
    reason=(
        "manual-only: set E2I_RUN_REAL_LLM_E2E=1 (plus an API key) to run the "
        "real-LLM e2e. Gated off by default so nondeterministic LLM planning "
        "JSON cannot flake CI — the stub-planner gate is the F1 arbiter (#504)."
    ),
)
async def test_canonical_query_real_llm_end_to_end():
    from src.utils.llm_factory import get_chat_llm

    df = _build_cohort_df()
    context = {"brand": "Kisqali", "region": "northeast", "estimation_data": df}

    llm_client = get_chat_llm(model_tier="reasoning", max_tokens=4096)
    result = await compose_query(query=CANONICAL_QUERY, llm_client=llm_client, context=context)

    assert result.execution.tools_succeeded > 0
    ates = _finite_ates(result)
    assert ates and all(math.isfinite(a) for a in ates)
    assert result.response.confidence > 0.3


# ---------------------------------------------------------------------------
# T2 (Rec#1a): the production caller resolves a real frame and threads it into
# the composer context under "estimation_data".
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_compose_tool_threads_resolved_frame_into_context(monkeypatch):
    """When the tier0 loader yields a frame, the compose tool must place it in
    context['estimation_data'] before calling compose_query."""
    import src.api.routes.chatbot_tools as ct

    captured: dict[str, Any] = {}
    resolved_df = _build_cohort_df(n=120)

    # Stub the cohort resolver to yield a real frame for (brand, region).
    # NOTE: this patches ct._resolve_cohort_frame (the caller-facing 3-arg
    # wrapper), NOT the underlying cohort_resolution.resolve_cohort_frame service
    # (whose signature is brand, region, *, data_source=...). The wrapper's name
    # and positional 3-arg contract are preserved precisely so this gate keeps
    # working after #779's delegation refactor.
    def _fake_resolve(brand, region, data_source):  # noqa: ANN001
        captured["resolve_args"] = (brand, region, data_source)
        return resolved_df

    monkeypatch.setattr(ct, "_resolve_cohort_frame", _fake_resolve)

    # Capture the context handed to compose_query (avoid running the full LLM
    # pipeline -- we only assert the caller wiring).
    async def _fake_compose_query(query, llm_client, context):  # noqa: ANN001
        captured["context"] = context
        raise RuntimeError("short-circuit after context build")

    monkeypatch.setattr(ct, "compose_query", _fake_compose_query)
    monkeypatch.setattr(ct, "get_chat_llm", lambda **_: object())  # never actually invoked

    out = await ct.tool_composer_tool.coroutine(
        query=CANONICAL_QUERY,
        brand="Kisqali",
        region="northeast",
        data_source="s3://bucket/kisqali_ne.parquet",
    )

    # The resolver was called with the caller-supplied brand/region/data_source.
    assert captured["resolve_args"] == (
        "Kisqali",
        "northeast",
        "s3://bucket/kisqali_ne.parquet",
    )
    # The resolved frame is in the composer context under the canonical key.
    ctx = captured["context"]
    assert "estimation_data" in ctx
    assert ctx["estimation_data"] is resolved_df
    # Tool still returns a dict (fallback path on the short-circuit RuntimeError).
    assert isinstance(out, dict)


@pytest.mark.asyncio
async def test_compose_tool_proceeds_when_resolver_unavailable(monkeypatch):
    """If the loader raises, the tool logs and proceeds WITHOUT estimation_data
    (tools then fail-closed honestly -- never fabricated)."""
    import src.api.routes.chatbot_tools as ct

    captured: dict[str, Any] = {}

    def _raising_resolve(brand, region, data_source):  # noqa: ANN001
        raise RuntimeError("loader offline")

    async def _fake_compose_query(query, llm_client, context):  # noqa: ANN001
        captured["context"] = context
        raise RuntimeError("short-circuit after context build")

    monkeypatch.setattr(ct, "_resolve_cohort_frame", _raising_resolve)
    monkeypatch.setattr(ct, "compose_query", _fake_compose_query)
    monkeypatch.setattr(ct, "get_chat_llm", lambda **_: object())

    out = await ct.tool_composer_tool.coroutine(
        query=CANONICAL_QUERY, brand="Kisqali", region="northeast"
    )

    assert "estimation_data" not in captured["context"]
    assert isinstance(out, dict)


# ---------------------------------------------------------------------------
# T3 (Rec#6 process): this module is the standard-path functional gate.
# The stub-planner test runs with NO network and NO opt-in flag, so a CI run
# of `pytest tests/integration/` exercises it. This assertion documents that
# the canonical-query gate exists and is collectible.
# ---------------------------------------------------------------------------
def test_functional_gate_is_registered_in_standard_path():
    """The canonical-query functional gate (the test that would have caught the
    original 0/4 failure) is present and runs without a network LLM."""
    import inspect

    import tests.integration.test_tool_composer_functional_e2e as mod

    gate = getattr(mod, "test_canonical_query_produces_real_tool_successes_stub_planner", None)
    assert gate is not None, "canonical-query stub-planner gate is missing"
    # It must be an async test with the asyncio marker (runs under asyncio_mode=auto).
    assert inspect.iscoroutinefunction(gate)
    marks = {m.name for m in getattr(gate, "pytestmark", [])}
    assert "asyncio" in marks, "stub-planner gate must carry @pytest.mark.asyncio"
    # And it must NOT be skipif-gated on a key (always runs in CI).
    assert "skipif" not in marks
