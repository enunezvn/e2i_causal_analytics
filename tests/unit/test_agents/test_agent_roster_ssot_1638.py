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
    added. Any literal count in a swept tree must equal the registry's."""

    #: Substrings whose lines the sweep must NOT flag, each with its reason.
    #:
    #: An ALLOWLIST rather than a cleverer regex, deliberately. Separating "this
    #: number describes the roster" from "this number is a mock fixture, a row
    #: count, or a note about something already deleted" is not a lexical
    #: property, and chasing it with lookarounds produced false positives faster
    #: than it removed them. Every entry states why it is not roster drift; a NEW
    #: stale string still trips the sweep because it will not be in this set.
    #:
    #: This sweep SUPPLEMENTS the structural guarantees (three rosters pinned
    #: against the registry, plus prompt derivation) — those are what actually
    #: prevent drift; this only catches prose that repeats a count by hand.
    ALLOWED_SCOPED_COUNTS = {
        # True statement about a SUBSET: the Tier 1-5 dispatcher contract,
        # deliberately excluding Tier 0, pinned by its own test.
        "13 agents",
        # Home.test.tsx asserts a FABRICATED summary does NOT render — a prior
        # guardrail. "Fixing" the number would break the guard it encodes.
        "15/21 agents",
        # DB row counts, not agents.
        "133 of 356 agent",
        # AgentOrchestration.test.tsx renders from a 15-agent MOCK fixture; the
        # assertion pins the component against its mock, not against the roster.
        "All 15 agents",
        # Comments recording values that were DELETED as fabricated (the hardcoded
        # health summary and the always-on badge). Restating them is the point.
        "97.5% / 21 agents",
        '"21 Agents Active"',
    }

    def test_no_source_file_states_a_wrong_whole_roster_count(self):
        """Sweeps src/, frontend/ and scripts/ (codex iter-1, then #1773).

        The first version of this test scanned only ``src/**/*.py`` and required
        the count to sit adjacent to its noun, so it gave FALSE ASSURANCE twice
        over: it could not see the frontend at all, and "21 **specialized**
        agents" walked straight through the adjective gap. A tripwire that misses
        the drift it exists to catch is worse than none, because it is quoted as
        evidence the sweep was complete.

        #1773 found the same failure a third time, in two independent halves:

        1. ``scripts/`` was never swept, and it is not a dev-only tree —
           ``docker/Dockerfile:208`` bakes it into the runtime image because the
           prod rootfs is read-only. Added below.
        2. The number-first matcher could not see the drift #1773 was filed for.
           ``seed_falkordb.py`` said "full roster = 21" with the noun BEFORE the
           number, so widening the roots alone would have left that line green —
           false assurance again, from the widening meant to end it. Hence the
           second pattern, ``roster_claim``.

        ``roster_claim`` is deliberately the narrowest form that catches it,
        because the ALLOWLIST note below is a measured finding, not a slogan. The
        broad noun-first variants were tried over these four roots first:
        ``roster<=24 chars><n>`` flagged 10 lines of which 9 were false (dates in
        "roster (2026-07-11)", the capped experiment roster of 25, and — worst —
        two comments whose whole point is that a count is NOT hardcoded); adding
        ``registry`` took it to 47 hits, 46 false. Requiring an explicit copula
        (``= : is has``) between the noun and the number, on a line that also
        says "agent", flags exactly 1 line across all four roots: the true
        positive. That is the whole justification for its shape — anything looser
        was measured to cry wolf, and a muted tripwire is the failure mode this
        class exists to prevent.

        It still must not flag legitimately SCOPED counts — see
        ``ALLOWED_SCOPED_COUNTS`` for each and why.
        """
        expected = len(AGENT_REGISTRY_CONFIG)
        whole_roster = re.compile(
            # (?<!tier ) — "Tier 0 Agent Orchestration" names a TIER, not a count.
            # (?:\w+\s+){0,2} — tolerate intervening adjectives ("21 specialized
            # agents"), which is exactly what the narrow version missed.
            r"(?<!tier )\b(\d+)[- ](?:\w+\s+){0,2}agents?\b"
            r"\s*(?:tiered|architecture|system|orchestrat\w*|roster|hierarchy)?",
            re.IGNORECASE,
        )
        # The noun-BEFORE-number half (#1773): "full roster = 21". See the
        # docstring for the measurement that fixed its shape.
        # "of" is NOT in the copula set: the ratio guard below ("133 of 356
        # agent rows") would swallow it anyway, and "a roster of 21 agents" is
        # already caught by whole_roster. Listing a branch that can never fire
        # is the same false assurance in miniature.
        roster_claim = re.compile(
            r"\b(?:roster|registry)\b\s*(?:=|:|\bis\b|\bhas\b)\s*(\d+)\b",
            re.IGNORECASE,
        )
        roots = [
            (REPO_ROOT / "src", ("*.py",)),
            (REPO_ROOT / "frontend/src", ("*.ts", "*.tsx")),
            (REPO_ROOT / "frontend/e2e", ("*.ts",)),
            # Shipped in the runtime image (docker/Dockerfile:208), not dev-only.
            # Source files only, matching the roots above: the benchmark .md/.json
            # under scripts/benchmarks/ describe the 14-agent chat CONTRACT
            # registry, a real subset, and sweeping them would buy allowlist
            # entries rather than caught drift.
            (REPO_ROOT / "scripts", ("*.py",)),
        ]
        offenders = []
        for root, globs in roots:
            if not root.exists():
                continue
            for pattern in globs:
                for path in sorted(root.rglob(pattern)):
                    try:
                        text = path.read_text(encoding="utf-8")
                    except (OSError, UnicodeDecodeError):
                        continue
                    for lineno, line in enumerate(text.splitlines(), start=1):
                        # roster_claim only speaks about agents; "registry of 45"
                        # in this repo is as likely to be KPIs or models.
                        claims = (
                            list(roster_claim.finditer(line)) if "agent" in line.lower() else []
                        )
                        for m in list(whole_roster.finditer(line)) + claims:
                            hit = m.group(0).strip()
                            if any(a in line for a in self.ALLOWED_SCOPED_COUNTS):
                                continue
                            before = line[: m.start(1)]
                            # "#876 agents" is an ISSUE REFERENCE; "133 of 356
                            # agent rows" and "46/54 agents-exact" are ratios over
                            # rows or benchmark queries. None are roster claims,
                            # and a tripwire that cries wolf on them gets muted.
                            if before.endswith(("#", "/")):
                                continue
                            if re.search(r"\b(?:of|\d)\s*$", before) or "/" in hit:
                                continue
                            count = int(m.group(1))
                            # Below 10 is ordinary prose ("2-agent handoff"), and
                            # tier subcounts are checked by the roster tests.
                            if count < 10 or count == expected:
                                continue
                            rel = path.relative_to(REPO_ROOT)
                            offenders.append(f"{rel}:{lineno}: {hit!r} (registry has {expected})")
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

    def test_chat_stream_prompt_also_carries_the_roster(self):
        """There are TWO chat brains, and 5.2 can be asked of either.

        codex iter-1 caught this: fixing only the AG-UI prompt left ``/chat``
        answering "what agents are available" from a bare count with no roster in
        context — the same defect on the other surface. Both prompts now
        interpolate the same generated block.
        """
        from src.agents.factory import build_agent_roster_block
        from src.api.routes.chatbot_graph import E2I_CHATBOT_SYSTEM_PROMPT

        assert build_agent_roster_block() in E2I_CHATBOT_SYSTEM_PROMPT
        assert "{agent_roster}" not in E2I_CHATBOT_SYSTEM_PROMPT
        # The other substitution slot must survive untouched — it is filled per
        # request, and _render_system_prompt uses .replace precisely because the
        # prompt contains literal braces that .format would crash on (#1332).
        assert "{context}" in E2I_CHATBOT_SYSTEM_PROMPT

    def test_agent_status_fallback_names_agents_rather_than_a_count(self):
        """The one canned reply for "which agents are there" must be able to
        answer it. It previously asserted a bare count and named no agent — and
        that count had already gone stale once."""
        from src.api.routes.chatbot_graph import _AGENT_STATUS_FALLBACK

        missing = [n for n in AGENT_REGISTRY_CONFIG if n not in _AGENT_STATUS_FALLBACK]
        assert not missing, f"agents absent from the AGENT_STATUS fallback: {missing}"

    def test_episodic_memory_enum_can_represent_every_agent(self):
        """Roster drift with teeth (codex iter-1): ``E2IAgentName`` is an ENUM, so
        an omitted agent is not a stale docstring — episodic memory simply cannot
        represent that agent at all."""
        from src.memory.episodic_memory import E2IAgentName

        values = {e.value for e in E2IAgentName}
        missing = sorted(set(AGENT_REGISTRY_CONFIG) - values)
        assert not missing, f"agents E2IAgentName cannot represent: {missing}"

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
