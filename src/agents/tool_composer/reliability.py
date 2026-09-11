"""Tool reliability: counts in, one verdict word out (spec §7.1).

This module is the single place that decides whether a tool's failures are evidence. The admin
surface reads it, and the planner does too once its experiment gate passes (§7.2).

Two decisions are load-bearing:

- **The denominator excludes refusals.** ``n_health = n_succeeded + n_health_failures``. A refusal
  is the tool declining to answer a question its data cannot support — a correct outcome, not a
  failure — so twenty refusals can never earn a tool a verdict.
- **There is a floor of 20 health runs.** Below it the verdict is ``too_few_runs``, because the
  Wilson interval at n = 5 caveats a healthy (p = 0.05) tool 2.3% of the time. Calibrated on
  planted truth, 20,000 draws per cell: at n >= 20 a healthy tool earns a false caveat at most
  0.2% of the time, while a 30%-failing tool is caught 76% of the time at n = 20 and 94.5% at
  n = 40 (spec §7.1). At today's production volume every tool reads ``too_few_runs``.

Measured latency is shown only once there are 20 successful runs to measure. Below that the
measured fields are ``None`` — the DECLARED latency is a separate, labelled field and is never
substituted into them.
"""

from __future__ import annotations

import asyncio
import logging
import math
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Tuple

from .rpc_port import RpcPort

logger = logging.getLogger(__name__)

#: Health runs required before any verdict other than ``too_few_runs``.
HEALTH_FLOOR = 20

#: Successful runs required before a measured latency is reported at all.
MIN_SUCCESSES_FOR_LATENCY = 20

#: The health-failure rate a caveat claims the tool is above.
CAVEAT_RATE = 0.10

#: Default cache lifetime of one reliability read, in seconds.
CACHE_TTL_S = 300.0

#: Deadline for one reliability read. The Supabase client's own timeout is tens of seconds, which
#: is a sensible ceiling for a request that matters and far too long for one that does not: this
#: read is optional, so planning gives it a short deadline and proceeds without it.
READ_TIMEOUT_S = 5.0

#: Default window, in days, that the planner integration reads.
DEFAULT_DAYS = 30

#: The flag gating the planner caveat (§7.2). Read fresh on every call.
RELIABILITY_IN_PLANNER_ENV = "TOOL_COMPOSER_RELIABILITY_IN_PLANNER"

_TRUTHY = ("1", "true", "yes")


def reliability_in_planner_enabled() -> bool:
    """Whether the planning prompt may carry a reliability caveat (default off)."""
    return os.getenv(RELIABILITY_IN_PLANNER_ENV, "").strip().lower() in _TRUTHY


def wilson(k: int, n: int, z: float = 1.959964) -> Tuple[float, float]:
    """Wilson score interval for a binomial proportion; ``(0, 1)`` when ``n == 0``.

    Mirrors ``scripts/benchmarks/routing/step0_scoring.wilson_ci``, the interval this codebase
    already reports elsewhere.
    """
    if n == 0:
        return (0.0, 1.0)
    phat = k / n
    denom = 1 + z * z / n
    center = (phat + z * z / (2 * n)) / denom
    half = (z / denom) * math.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n))
    lo, hi = center - half, center + half
    if k == 0:
        lo = 0.0
    if k == n:
        hi = 1.0
    return (max(0.0, lo), min(1.0, hi))


def verdict(counts: Mapping[str, Any]) -> str:
    """One of ``no_runs`` / ``too_few_runs`` / ``caveat`` / ``reliable`` / ``inconclusive``.

    ``counts`` needs ``n_invoked``, ``n_health`` and ``n_health_failures`` (a row of
    ``get_tool_reliability``). The interval is of the HEALTH-failure rate, so a tool can only be
    caveated for failing, never for refusing.
    """
    n_invoked = int(counts.get("n_invoked") or 0)
    n_health = int(counts.get("n_health") or 0)
    n_failures = int(counts.get("n_health_failures") or 0)

    if n_invoked <= 0:
        return "no_runs"
    if n_health < HEALTH_FLOOR:
        return "too_few_runs"

    lo, hi = wilson(n_failures, n_health)
    if lo >= CAVEAT_RATE:
        return "caveat"
    if hi < CAVEAT_RATE:
        return "reliable"
    return "inconclusive"


@dataclass(frozen=True)
class ToolReliability:
    """One tool's window of evidence, with the verdict the rule gives it."""

    tool_name: str
    verdict: str
    n_invoked: int
    n_succeeded: int
    n_refused: int
    n_health_failures: int
    n_health: int
    n_retried: int
    n_synthetic: int
    category: Optional[str] = None
    source_agent: Optional[str] = None
    version: Optional[str] = None
    declared_latency_ms: Optional[float] = None
    p50_latency_ms: Optional[float] = None
    p95_latency_ms: Optional[float] = None
    last_executed_at: Optional[str] = None
    most_common_health_error: Optional[str] = None

    @classmethod
    def from_row(cls, row: Mapping[str, Any]) -> "ToolReliability":
        """Build from a ``get_tool_reliability`` row. Raises on a row without a tool name."""
        name = row["tool_name"]
        if not isinstance(name, str) or not name:
            raise ValueError("reliability row has no tool_name")

        def count(field: str) -> int:
            return int(row.get(field) or 0)

        def number(field: str) -> Optional[float]:
            value = row.get(field)
            return float(value) if isinstance(value, (int, float)) else None

        n_succeeded = count("n_succeeded")
        # Measured only when there is something to measure (spec §7.1).
        measured = n_succeeded >= MIN_SUCCESSES_FOR_LATENCY
        counts = {
            "n_invoked": count("n_invoked"),
            "n_succeeded": n_succeeded,
            "n_refused": count("n_refused"),
            "n_health_failures": count("n_health_failures"),
            "n_health": count("n_health"),
            "n_retried": count("n_retried"),
            "n_synthetic": count("n_synthetic"),
        }
        return cls(
            tool_name=name,
            verdict=verdict(counts),
            category=row.get("category"),
            source_agent=row.get("source_agent"),
            version=row.get("version"),
            declared_latency_ms=number("declared_latency_ms"),
            p50_latency_ms=number("p50_latency_ms") if measured else None,
            p95_latency_ms=number("p95_latency_ms") if measured else None,
            last_executed_at=str(row["last_executed_at"])
            if row.get("last_executed_at") is not None
            else None,
            most_common_health_error=row.get("most_common_health_error"),
            **counts,
        )

    @property
    def latency_is_measured(self) -> bool:
        return self.n_succeeded >= MIN_SUCCESSES_FOR_LATENCY


class ToolReliabilityReader:
    """Reads ``get_tool_reliability`` once per window per process lifetime slice.

    Cached for ``ttl_s`` per ``(days, include_synthetic)``: the planner and the admin surface ask
    for different windows, and the provenance flag decides whether synthetic-substrate runs count
    at all, so both belong in the key.

    Only a COMPLETE reading is cached. An error, a timeout, a payload that is not rows, or a
    reading that had to drop a row all return what they have without caching it — otherwise one
    bad response would suppress every caveat for the whole TTL, long after the database recovered.
    """

    def __init__(
        self,
        port: Optional[RpcPort] = None,
        ttl_s: float = CACHE_TTL_S,
        read_timeout_s: float = READ_TIMEOUT_S,
    ) -> None:
        self._port = port
        self._ttl_s = ttl_s
        self._read_timeout_s = read_timeout_s
        self._cache: Dict[Tuple[int, bool], Tuple[float, Dict[str, ToolReliability]]] = {}

    @property
    def port(self) -> RpcPort:
        if self._port is None:
            from .rpc_port import SupabaseRpcPort

            self._port = SupabaseRpcPort()
        return self._port

    async def get(self, days: int = DEFAULT_DAYS) -> Dict[str, ToolReliability]:
        """Every tool's verdict for the window, keyed by tool name. Never raises."""
        from src.repositories.provenance import deployment_includes_synthetic

        include_synthetic = deployment_includes_synthetic()
        key = (int(days), include_synthetic)
        cached = self._cache.get(key)
        now = time.monotonic()
        if cached is not None and cached[0] > now:
            return cached[1]

        try:
            rows = await asyncio.wait_for(
                self.port.call(
                    "get_tool_reliability",
                    {"p_days": int(days), "p_include_synthetic": include_synthetic},
                ),
                timeout=self._read_timeout_s,
            )
        except Exception as e:  # noqa: BLE001 - a missing reading is not a planning failure
            logger.warning(f"Tool reliability read failed ({type(e).__name__}: {e})")
            return {}

        if not isinstance(rows, list):
            logger.warning("Tool reliability read returned no rows; not cached")
            return {}

        verdicts: Dict[str, ToolReliability] = {}
        dropped = False
        for row in rows:
            if not isinstance(row, Mapping):
                dropped = True
                continue
            try:
                tool = ToolReliability.from_row(row)
            except Exception:  # noqa: BLE001 - one unusable row never loses the rest
                logger.warning("skipped an unusable tool reliability row")
                dropped = True
                continue
            verdicts[tool.tool_name] = tool

        # An incomplete reading serves this call but is never held: the next caller reads again.
        if not dropped:
            self._cache[key] = (now + self._ttl_s, verdicts)
        return verdicts


def reliability_line(tool: Optional[ToolReliability]) -> Optional[str]:
    """The one line a caveated tool adds to the planning prompt, or ``None``.

    Only ``caveat`` says anything: the other verdicts are either an absence of evidence or good
    news, and neither belongs in a prompt that is already long. The line states the counts it is
    based on, never a rate, and names the most common health error when there is one.
    """
    if tool is None or tool.verdict != "caveat":
        return None
    line = (
        f"Reliability caveat: {tool.n_health_failures} of {tool.n_health} runs failed "
        "on tool errors"
    )
    if tool.most_common_health_error:
        line += f" ({tool.most_common_health_error})"
    return line


def format_tool_block(
    tool: Mapping[str, Any], reliability: Optional[ToolReliability] = None
) -> List[str]:
    """The prompt lines for one tool, shared by the planner and the DSPy formatter.

    With no reliability — or with the flag off, which is the caller's decision — the lines are
    byte-identical to what both formatters rendered before this module existed. The declared
    "Avg execution" line is never replaced by a measured number: the LLM reading it is the only
    planning consumer of that value, and swapping its meaning silently would be a different
    change from adding a caveat.
    """
    lines = [
        f"### {tool['name']} ({tool['source']})",
        f"Description: {tool['description']}",
        f"Inputs: {', '.join(tool['inputs'])}",
    ]
    output_fields = tool.get("output_fields") or []
    if output_fields:
        lines.append(f"Output: {tool['output']} (fields: {', '.join(output_fields)})")
    else:
        lines.append(f"Output: {tool['output']}")
    lines.append(f"Avg execution: {tool['avg_ms']}ms")

    caveat = reliability_line(reliability)
    if caveat:
        lines.append(caveat)

    lines.append("")
    return lines
