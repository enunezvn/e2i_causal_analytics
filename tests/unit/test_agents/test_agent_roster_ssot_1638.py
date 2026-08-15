"""#1638: the agent roster must come from the registry, not from prose that rots.

Turn 5.2 of the 2026-08-15 eval ("what agents are available") answered with two
agent names plus a list of TOOL names — a category error, the tool surface is not
the agent roster — and called the system "21-agent".

Two separate defects, and the filed issue got the first one wrong:

**The answering prompt had no roster at all.** The issue claimed the full roster
was "present verbatim in the system prompt that turn ran under". It is not: the
roster it cited lives on ``chatbot_dspy.AgentRoutingSignature``, the per-query
ROUTING classifier for a different surface. The eval ran on AG-UI, whose prompt
(``copilotkit.E2I_COPILOT_SYSTEM_PROMPT``) mentioned only "the 21-agent tiered
architecture" and listed no agents. With no roster in context the model called
``agent_routing_tool`` — a router, not a directory — and substituted tool names
when it came back without one. Given what it had, that behaviour is explicable.

**The count was wrong, and inconsistent within single files.** Measured from the
SSOT (``factory.AGENT_REGISTRY_CONFIG``): **22** agents, all enabled, in 6 tiers
(0-5). ``chatbot_tools.py`` said both 21 and 22; ``repositories/agent_registry.py``
said 21 on line 4 and 22 on line 38.

**The frontend roster was genuinely missing an agent** — not a labeling problem.
``documentation/content.ts`` enumerated 21 agents while claiming
AGENT_REGISTRY_CONFIG as its source; ``cohort_profiler`` (a real, dispatched
agent) was absent, so the docs page under-reported the system.

The fix derives the roster and the count from the registry so neither can rot,
and these tests pin that derivation rather than any literal number.
"""

import re
from pathlib import Path

import pytest

from src.agents.factory import AGENT_REGISTRY_CONFIG

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[3]


class TestRegistryIsTheSourceOfTruth:
    def test_registry_has_a_single_unambiguous_count(self):
        assert len(AGENT_REGISTRY_CONFIG) == len(set(AGENT_REGISTRY_CONFIG)), "duplicate ids"
        assert len(AGENT_REGISTRY_CONFIG) > 0

    def test_every_agent_declares_a_tier(self):
        missing = [n for n, c in AGENT_REGISTRY_CONFIG.items() if c.get("tier") is None]
        assert not missing, f"agents with no tier: {missing}"


class TestNoStaleAgentCountStrings:
    """A hardcoded count in prose is a fact that rots the moment an agent is
    added. Any literal count anywhere in src/ must equal the registry's."""

    def test_no_source_file_states_a_wrong_whole_roster_count(self):
        """Only counts that claim to describe the WHOLE architecture.

        A bare ``\\d+ agents`` sweep is too broad and would flag legitimately
        SCOPED counts — ``_agent_method_map.py`` says "13 agents" about the Tier
        1-5 dispatcher contract, deliberately excluding Tier 0, and is pinned by
        its own test. That is a true statement about a subset, not roster drift.
        """
        expected = len(AGENT_REGISTRY_CONFIG)
        whole_roster = re.compile(
            # (?<!tier ) — "Tier 0 Agent Orchestration" names a TIER, not a count.
            r"(?<!tier )(\d+)[- ]agents?\b\s*(?:tiered|architecture|system|orchestrat\w*|roster)"
            r"|(?:all|the full)\s+(\d+)\s+agents\b"
            r"|(\d+)\s+agents\s+(?:organized|with tier)",
            re.IGNORECASE,
        )
        offenders = []
        for path in sorted((REPO_ROOT / "src").rglob("*.py")):
            try:
                text = path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue
            for lineno, line in enumerate(text.splitlines(), start=1):
                for m in whole_roster.finditer(line):
                    count = int(next(g for g in m.groups() if g))
                    if count != expected:
                        rel = path.relative_to(REPO_ROOT)
                        offenders.append(
                            f"{rel}:{lineno}: {m.group(0)!r} (registry has {expected})"
                        )
        assert not offenders, "stale whole-roster agent counts:\n  " + "\n  ".join(offenders)


class TestAnsweringPromptCarriesTheRoster:
    """5.2 must be answerable from context. The AG-UI prompt is the surface the
    eval actually ran on."""

    def test_every_registered_agent_is_named_in_the_agui_prompt(self):
        from src.api.routes.copilotkit import E2I_COPILOT_SYSTEM_PROMPT

        missing = [n for n in AGENT_REGISTRY_CONFIG if n not in E2I_COPILOT_SYSTEM_PROMPT]
        assert not missing, (
            "agents absent from the answering prompt (5.2 cannot enumerate what it "
            f"cannot see): {missing}"
        )

    def test_prompt_states_the_registry_count_and_tier_count(self):
        from src.api.routes.copilotkit import E2I_COPILOT_SYSTEM_PROMPT

        n_agents = len(AGENT_REGISTRY_CONFIG)
        n_tiers = len({c["tier"] for c in AGENT_REGISTRY_CONFIG.values()})
        assert f"{n_agents} agents" in E2I_COPILOT_SYSTEM_PROMPT
        assert f"{n_tiers} tiers" in E2I_COPILOT_SYSTEM_PROMPT

    def test_roster_is_derived_not_transcribed(self):
        """The guarantee that matters: adding an agent to the registry must change
        the prompt with no other edit. A hand-typed list would pass the
        membership test above today and silently rot tomorrow."""
        from src.agents.factory import build_agent_roster_block
        from src.api.routes.copilotkit import E2I_COPILOT_SYSTEM_PROMPT

        block = build_agent_roster_block()
        assert block in E2I_COPILOT_SYSTEM_PROMPT, (
            "the prompt does not embed the generated roster block verbatim, so it "
            "is transcribed rather than derived"
        )

    def test_routing_signature_roster_agrees_with_the_registry(self):
        """The other surface that carries a roster (#1638): the /chat routing
        classifier. It must not drift from the registry either."""
        from src.api.routes import chatbot_dspy

        source = Path(chatbot_dspy.__file__).read_text(encoding="utf-8")
        missing = [n for n in AGENT_REGISTRY_CONFIG if n not in source]
        assert not missing, f"agents absent from the routing signature roster: {missing}"


class TestFrontendRosterMatchesTheRegistry:
    """The docs page enumerated 21 agents and was missing cohort_profiler — a
    real agent, so the page under-reported the system rather than mislabeling it."""

    def _content_ts(self) -> str:
        path = REPO_ROOT / "frontend/src/components/documentation/content.ts"
        return path.read_text(encoding="utf-8")

    def test_every_registered_agent_appears_in_the_docs_roster(self):
        text = self._content_ts()
        ids = set(re.findall(r"id:\s*'([a-z0-9_]+)'", text))
        missing = sorted(set(AGENT_REGISTRY_CONFIG) - ids)
        assert not missing, f"agents missing from the frontend docs roster: {missing}"

    def test_copilot_provider_roster_matches_the_registry(self):
        """The THIRD roster (#1638). ``E2ICopilotProvider.AGENT_REGISTRY`` mirrors
        the backend in kebab-case and claimed to hold "all" the agents while
        missing cohort_profiler, so the provider under-reported the system exactly
        as the docs page did. Pinned here because a Python-side registry change is
        what makes a TS-side roster stale."""
        path = REPO_ROOT / "frontend/src/providers/E2ICopilotProvider.tsx"
        block = path.read_text(encoding="utf-8").split("const AGENT_REGISTRY", 1)[-1]
        ids = {
            m.group(1).replace("-", "_")
            for m in re.finditer(r"id:\s*'([a-z0-9-]+)',\s*\n\s*name:", block)
        }
        missing = sorted(set(AGENT_REGISTRY_CONFIG) - ids)
        extra = sorted(ids - set(AGENT_REGISTRY_CONFIG))
        assert not missing, f"agents missing from E2ICopilotProvider: {missing}"
        assert not extra, f"E2ICopilotProvider names agents the registry lacks: {extra}"

    def test_docs_roster_declares_no_agent_the_registry_lacks(self):
        text = self._content_ts()
        # Only ids inside the AGENT_TIERS array are roster entries — bound the
        # slice at the next top-level export or the tail sweeps in unrelated
        # `id:` fields from other structures in this file.
        block = text.split("export const AGENT_TIERS", 1)[-1]
        block = re.split(r"^export const ", block, maxsplit=1, flags=re.MULTILINE)[0]
        ids = set(re.findall(r"\{\s*id:\s*'([a-z0-9_]+)'", block))
        extra = sorted(ids - set(AGENT_REGISTRY_CONFIG))
        assert not extra, f"frontend roster names agents the registry does not: {extra}"
