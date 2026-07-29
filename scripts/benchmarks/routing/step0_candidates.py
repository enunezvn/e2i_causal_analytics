"""Step 0 routing-classifier candidates (#1337).

Four candidates, all running REAL code — no mocks anywhere:

- ``legacy``          — the incumbent: the real ``IntentClassifierNode.execute``
                        (pattern-first, real haiku fallback for ambiguous
                        queries) followed by the real ``RouterNode.execute``;
                        (pattern, agents) read off the actual dispatch plan.
- ``pipeline_rules``  — the 4-stage ``ClassificationPipeline`` exactly as
                        shadow-deployed (LLM layer disabled). Free ablation
                        baseline for what the LLM stage buys.
- ``pipeline_llm``    — candidate (a): the 4-stage pipeline plus a prototype
                        LLM stage. When the rules stage is uncertain
                        (confidence < 0.5 — the shadow MIN_ACTIVE_CONFIDENCE
                        floor), one haiku call sees the query, the rule-stage
                        verdict, and the contract cards, and produces the
                        final (pattern, agents). Certain rows never pay for
                        the LLM.
- ``single_llm``      — candidate (b): one haiku call per query (same contract
                        cards, same output schema), no staged rules at all.

Both LLM candidates use the fast-tier model (claude-haiku-4-5) at
temperature 0 so the comparison is architecture vs architecture at the
routing layer's real latency/cost point.
"""

from __future__ import annotations

import asyncio
import contextvars
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, FrozenSet, List, Optional

import anthropic

from scripts.benchmarks.routing.step0_scoring import (
    contract_cards_from_registry,
    derive_legacy_pattern,
    parse_candidate_json,
)

DATA_DIR = Path(__file__).parent / "data"
CONTRACTS_PATH = DATA_DIR / "agent_contracts.json"

FAST_MODEL = "claude-haiku-4-5-20251001"
MAX_TOKENS = 300
ESCALATION_THRESHOLD = 0.5  # shadow MIN_ACTIVE_CONFIDENCE floor (#1330)


@dataclass
class Prediction:
    routing_pattern: str
    target_agents: List[str]
    confidence: float
    latency_ms: float
    llm_used: bool = False
    parse_failed: bool = False
    detail: Dict[str, Any] = field(default_factory=dict)


def load_registry() -> Dict[str, Any]:
    with open(CONTRACTS_PATH) as f:
        return json.load(f)


def known_agents(registry: Dict[str, Any]) -> FrozenSet[str]:
    return frozenset((registry.get("agents") or {}).keys())


def context_to_history(row: Dict[str, Any]) -> Optional[List[Dict[str, str]]]:
    """Benchmark ``context`` ({prev_user, prev_assistant}) -> chat history."""
    ctx = row.get("context") or {}
    history: List[Dict[str, str]] = []
    if ctx.get("prev_user"):
        history.append({"role": "user", "content": ctx["prev_user"]})
    if ctx.get("prev_assistant"):
        history.append({"role": "assistant", "content": ctx["prev_assistant"]})
    return history or None


def _context_block(row: Dict[str, Any]) -> str:
    history = context_to_history(row)
    if not history:
        return ""
    turns = "\n".join(f"{t['role']}: {t['content']}" for t in history)
    return f"Prior conversation turns (this query may be a follow-up):\n{turns}\n\n"


# =============================================================================
# Candidate: legacy (incumbent)
# =============================================================================

# Per-asyncio-task flag: did the wrapped _llm_classify run during this
# predict? ContextVar (not an instance attribute) because run_candidate
# executes predicts concurrently on ONE LegacyCandidate instance; each
# gather task copies the context, so tasks cannot race each other's flag.
_LLM_CALL_SEEN: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "step0_legacy_llm_call_seen", default=False
)


def reset_llm_call_tracking() -> None:
    _LLM_CALL_SEEN.set(False)


def llm_call_seen() -> bool:
    return _LLM_CALL_SEEN.get()


def install_llm_call_tracker(classifier: Any) -> None:
    """Wrap ``classifier._llm_classify`` so llm_used detection is exact.

    The previous heuristic (``confidence < 0.8 or method == "llm"``)
    undercounted: _llm_classify can return confidence >= 0.8 (its
    parse-degradation default is 0.85) and IntentResult never carries a
    ``method`` field, so real haiku fallbacks were booked as rule hits.
    """
    real = classifier._llm_classify

    async def _tracked(*args: Any, **kwargs: Any) -> Any:
        _LLM_CALL_SEEN.set(True)
        return await real(*args, **kwargs)

    classifier._llm_classify = _tracked


class LegacyCandidate:
    """The real incumbent chain: IntentClassifierNode -> RouterNode."""

    name = "legacy"

    def __init__(self) -> None:
        # ORCHESTRATOR_CLASSIFIER_MODE=off must be exported by the runner
        # BEFORE predictions so the node's embedded shadow pipeline (and its
        # classification_logs writer) stays out of the measurement.
        from src.agents.orchestrator.nodes.intent_classifier import IntentClassifierNode
        from src.agents.orchestrator.nodes.router import RouterNode

        self._classifier = IntentClassifierNode()
        self._router = RouterNode()
        install_llm_call_tracker(self._classifier)

    async def predict(self, row: Dict[str, Any]) -> Prediction:
        state: Dict[str, Any] = {"query": row["text"]}
        history = context_to_history(row)
        if history:
            state["conversation_history"] = history

        reset_llm_call_tracking()
        t0 = time.perf_counter()
        classified = await self._classifier.execute(state)  # type: ignore[arg-type]
        routed = await self._router.execute(classified)
        latency_ms = (time.perf_counter() - t0) * 1000

        intent = classified.get("intent") or {}
        agents = [d["agent_name"] for d in routed.get("dispatch_plan") or []]
        pattern = derive_legacy_pattern(intent.get("primary_intent", "general"), agents)
        llm_used = llm_call_seen()
        return Prediction(
            routing_pattern=pattern,
            target_agents=sorted(set(agents)),
            confidence=float(intent.get("confidence", 0.0)),
            latency_ms=latency_ms,
            llm_used=llm_used,
            detail={"primary_intent": intent.get("primary_intent")},
        )


# =============================================================================
# Candidate: pipeline_rules (4-stage, LLM layer off — the shadow deployment)
# =============================================================================


class PipelineRulesCandidate:
    name = "pipeline_rules"

    def __init__(self) -> None:
        from src.agents.orchestrator.classifier import ClassificationPipeline

        self._pipeline = ClassificationPipeline(llm_client=None, enable_llm_layer=False)

    async def _classify(self, row: Dict[str, Any]):
        history = context_to_history(row)
        return await self._pipeline.classify(
            query=row["text"],
            is_followup=bool(history),
            context_source="conversation_history" if history else None,
        )

    async def predict(self, row: Dict[str, Any]) -> Prediction:
        t0 = time.perf_counter()
        result = await self._classify(row)
        latency_ms = (time.perf_counter() - t0) * 1000
        return Prediction(
            routing_pattern=result.routing_pattern.value,
            target_agents=sorted(set(result.target_agents)),
            confidence=result.confidence,
            latency_ms=latency_ms,
        )


# =============================================================================
# LLM prompt shared by both LLM candidates
# =============================================================================

_PATTERNS_BLOCK = """Routing patterns:
- SINGLE_AGENT: one agent's contract covers the whole query. target_agents = [that agent].
- PARALLEL_DELEGATION: 2+ agents' contracts are needed for INDEPENDENT facets (no facet depends on another's output). target_agents = those agents.
- TOOL_COMPOSER: 2+ DISTINCT agent domains AND the facets are dependency-linked (one part needs another's output). target_agents = ["tool_composer"].
- CLARIFICATION_NEEDED: the query is genuinely too ambiguous to route (no contract owns it without guessing the user's intent). target_agents = []."""


def _single_call_prompt(row: Dict[str, Any], cards: str) -> str:
    return f"""You route pharmaceutical-analytics chat queries to specialist agents.

Agent contracts (what each agent covers):
{cards}

{_PATTERNS_BLOCK}

{_context_block(row)}Query: "{row["text"]}"

Respond with ONLY a JSON object:
{{"routing_pattern": "<one of SINGLE_AGENT|PARALLEL_DELEGATION|TOOL_COMPOSER|CLARIFICATION_NEEDED>", "target_agents": ["<agent>", ...], "confidence": <0.0-1.0>}}"""


def _escalation_prompt(row: Dict[str, Any], cards: str, rule_result: Prediction) -> str:
    return f"""You are the LLM escalation stage of a staged routing classifier for pharmaceutical-analytics chat.

The deterministic rule stages were UNCERTAIN about this query. Their verdict:
- rule_pattern: {rule_result.routing_pattern}
- rule_target_agents: {rule_result.target_agents}
- rule_confidence: {rule_result.confidence:.2f}

Agent contracts (what each agent covers):
{cards}

{_PATTERNS_BLOCK}

{_context_block(row)}Query: "{row["text"]}"

Decide the FINAL routing. You may confirm or override the rule verdict.
Respond with ONLY a JSON object:
{{"routing_pattern": "<one of SINGLE_AGENT|PARALLEL_DELEGATION|TOOL_COMPOSER|CLARIFICATION_NEEDED>", "target_agents": ["<agent>", ...], "confidence": <0.0-1.0>}}"""


class _HaikuCaller:
    """Shared async Anthropic caller with bounded retries."""

    def __init__(self, client: anthropic.AsyncAnthropic) -> None:
        self._client = client

    async def call(self, prompt: str) -> str:
        delay = 2.0
        for attempt in range(4):
            try:
                msg = await self._client.messages.create(
                    model=FAST_MODEL,
                    max_tokens=MAX_TOKENS,
                    temperature=0.0,
                    messages=[{"role": "user", "content": prompt}],
                )
                return "".join(
                    block.text for block in msg.content if getattr(block, "type", "") == "text"
                )
            except (anthropic.RateLimitError, anthropic.APIStatusError) as e:
                status = getattr(e, "status_code", 0)
                if isinstance(e, anthropic.APIStatusError) and status < 500 and status != 429:
                    raise
                if attempt == 3:
                    raise
                await asyncio.sleep(delay)
                delay *= 2
        raise RuntimeError("unreachable")


# =============================================================================
# Candidate (a): pipeline + prototype LLM stage
# =============================================================================


class PipelineLLMCandidate(PipelineRulesCandidate):
    name = "pipeline_llm"

    def __init__(self, caller: _HaikuCaller, cards: str, agents: FrozenSet[str]) -> None:
        super().__init__()
        self._caller = caller
        self._cards = cards
        self._agents = agents

    async def predict(self, row: Dict[str, Any]) -> Prediction:
        t0 = time.perf_counter()
        result = await self._classify(row)
        rule_pred = Prediction(
            routing_pattern=result.routing_pattern.value,
            target_agents=sorted(set(result.target_agents)),
            confidence=result.confidence,
            latency_ms=0.0,
        )
        if result.confidence >= ESCALATION_THRESHOLD:
            rule_pred.latency_ms = (time.perf_counter() - t0) * 1000
            return rule_pred

        text = await self._caller.call(_escalation_prompt(row, self._cards, rule_pred))
        latency_ms = (time.perf_counter() - t0) * 1000
        parsed = parse_candidate_json(text, known_agents=self._agents)
        if parsed is None:
            # Unusable LLM reply: fall back to the rule verdict, counted.
            rule_pred.latency_ms = latency_ms
            rule_pred.llm_used = True
            rule_pred.parse_failed = True
            return rule_pred
        return Prediction(
            routing_pattern=parsed["routing_pattern"],
            target_agents=sorted(set(parsed["target_agents"])),
            confidence=parsed["confidence"],
            latency_ms=latency_ms,
            llm_used=True,
            detail={"rule_pattern": rule_pred.routing_pattern},
        )


# =============================================================================
# Candidate (b): single LLM call
# =============================================================================


class SingleLLMCandidate:
    name = "single_llm"

    def __init__(self, caller: _HaikuCaller, cards: str, agents: FrozenSet[str]) -> None:
        self._caller = caller
        self._cards = cards
        self._agents = agents

    async def predict(self, row: Dict[str, Any]) -> Prediction:
        t0 = time.perf_counter()
        text = await self._caller.call(_single_call_prompt(row, self._cards))
        latency_ms = (time.perf_counter() - t0) * 1000
        parsed = parse_candidate_json(text, known_agents=self._agents)
        if parsed is None:
            # Unusable reply scores as a guaranteed miss, counted explicitly.
            return Prediction(
                routing_pattern="PARSE_FAILED",
                target_agents=[],
                confidence=0.0,
                latency_ms=latency_ms,
                llm_used=True,
                parse_failed=True,
            )
        return Prediction(
            routing_pattern=parsed["routing_pattern"],
            target_agents=sorted(set(parsed["target_agents"])),
            confidence=parsed["confidence"],
            latency_ms=latency_ms,
            llm_used=True,
        )


def build_candidates(names: List[str]) -> Dict[str, Any]:
    """Instantiate the requested candidates (env must already be loaded)."""
    registry = load_registry()
    cards = contract_cards_from_registry(registry)
    agents = known_agents(registry)

    needs_llm = bool({"pipeline_llm", "single_llm", "legacy"} & set(names))
    caller: Optional[_HaikuCaller] = None
    if needs_llm:
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise SystemExit(
                "ANTHROPIC_API_KEY not set — refusing to run: candidates would "
                "silently degrade instead of producing real results"
            )
        caller = _HaikuCaller(anthropic.AsyncAnthropic(api_key=api_key))

    out: Dict[str, Any] = {}
    for name in names:
        if name == "legacy":
            out[name] = LegacyCandidate()
        elif name == "pipeline_rules":
            out[name] = PipelineRulesCandidate()
        elif name == "pipeline_llm":
            assert caller is not None
            out[name] = PipelineLLMCandidate(caller, cards, agents)
        elif name == "single_llm":
            assert caller is not None
            out[name] = SingleLLMCandidate(caller, cards, agents)
        else:
            raise SystemExit(f"unknown candidate: {name}")
    return out
