# Copilot Pill Capability Catalog Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the chat sidebar's suggestion pills propose only analyses the E2I assistant can deliver, and let the assistant see the same on-screen page summary the pill generator sees.

**Architecture:** A new service module builds a `CapabilityCatalog` from code and data (KPI registry, history coverage view, segmented-history families, causal-outcome registry, agent roster), renders it into the pill prompt, and post-filters generated pills with a narrow deterministic validator. The frontend tops the pill row back up to four with its existing static pills and publishes each page's `pageChatSummary` as a fifth CopilotKit readable so the agent's ON-SCREEN APP CONTEXT carries it.

**Tech Stack:** Python 3.12 / FastAPI / LangChain fast-tier LLM (Anthropic Haiku 4.5 in prod), pytest (`asyncio_mode=auto`); React 18 / TypeScript / CopilotKit react-core 1.51.2, vitest.

**Spec:** `docs/superpowers/specs/2026-09-05-copilot-pill-capability-catalog-design.md`

---

## Working environment (read first)

- Work ONLY in the worktree `/home/enunez/Projects/e2i_causal_analytics/.worktrees/pill-catalog` on branch `claude/copilot-pill-capability-catalog`. Never `cd` into the main checkout: an active peer session uses it.
- Python: `PY=/home/enunez/Projects/e2i_causal_analytics/.venv/bin/python`. Run pytest from the worktree root so `src` resolves to the worktree (verified: `$PY -c "import src; print(src.__file__)"` prints the worktree path). Always pass `-n 0 -p no:cacheprovider` (no xdist on this box; a nonexistent path under xdist is a SILENT `[0 items]`).
- Do NOT run whole-tree `mypy` or whole-tree `pytest` on this box (memory pressure). CI is the arbiter for both.
- Frontend: `frontend/node_modules` in the worktree is a symlink to the main checkout's `node_modules` (already created). Run vitest as `cd frontend && npx vitest run <files>`; typecheck as `npm run typecheck` (`tsc -b`, not bare `tsc`).
- The evidence directory `/home/enunez/Projects/e2i_causal_analytics/docs/demos/results/2026-09-05_pill_suggestions_review/` is untracked (by convention `docs/demos/results/*` stay untracked). Scratch scripts and re-measurement outputs go there by absolute path.
- `.env` lives only in the main checkout: scripts load it with `load_dotenv('/home/enunez/Projects/e2i_causal_analytics/.env')`. Tests need no `.env` (verified: the existing 18 route tests pass in the worktree without one).
- Before EVERY commit run `git branch --show-current` and confirm `claude/copilot-pill-capability-catalog`.
- Every commit message ends with these two trailer lines:

```
Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01QCr4zzEYNQDCrLaM4gFHGr
```

## File structure

| File | Responsibility |
|---|---|
| `src/services/chat_capability_catalog.py` (new) | `CapabilityCatalog` dataclass; `build_capability_catalog` (loaders injectable); `render_catalog_block`; `ROUTE_HINTS` + `route_hint`; `journey_outcomes` + `filter_unsupported_pills`; TTL cache `get_capability_catalog` / `reset_capability_catalog_cache`. |
| `src/api/routes/chat.py` | Prompt template with `{capability_catalog}` / `{route_hint}` placeholders; `build_system_prompt`; handler fetches the catalog, fills the prompt, runs the validator, logs drops. |
| `src/api/routes/copilotkit.py` | `_readables_context_note` wording covers prose summaries. |
| `frontend/src/providers/E2ICopilotProvider.tsx` | Fifth readable carrying `pageChatContext`. |
| `frontend/src/components/chat/E2IChatSidebar.tsx` | `topUpChatSuggestions`; pill memo uses it; doc comment updated. |
| `tests/api/test_chat_capability_catalog.py` (new) | Unit tests for the service module. |
| `tests/api/test_chat_suggestions.py` | Route tests extended; autouse catalog fake. |
| `frontend/src/providers/E2ICopilotProvider.test.tsx` | Readable count 5; summary readable test. |
| `frontend/src/components/chat/E2IChatSidebar.suggestions.test.tsx` | Top-up tests. |

Task order puts the re-measurement gate (Task 7) BEFORE the route is rewired (Task 8): the catalog and prompt builder exist as pure code first, the faithful prototype runs against them, and only a passing gate unlocks the production wiring, validator plumbing and frontend work.

---

### Task 1: Catalog dataclass and builder

**Files:**
- Create: `src/services/chat_capability_catalog.py`
- Test: `tests/api/test_chat_capability_catalog.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/api/test_chat_capability_catalog.py`:

```python
"""Unit tests for src.services.chat_capability_catalog.

Everything DB-backed is injected through the two loader callables, so these
tests run without Supabase. The KPI registry (YAML) and the agent roster
(factory config) are real: they are code, and the point of the catalog is
that its lists come from code.
"""

from __future__ import annotations

from typing import Any, Dict, List

import pytest

from src.kpi.segmented_history import SEGMENTED_KPI_QUERY_FAMILIES
from src.services import chat_capability_catalog as cat

# =============================================================================
# FIXTURE DATA
# =============================================================================

COVERAGE_ROWS: List[Dict[str, Any]] = [
    {"kpi_id": "WS3-BI-005", "brand": "", "region": "", "points": 24},
    {"kpi_id": "WS3-BI-005", "brand": "Kisqali", "region": "", "points": 24},
    # NBRx: per-brand scopes only, no '' row -> per_brand_only
    {"kpi_id": "WS3-BI-007", "brand": "Kisqali", "region": "", "points": 24},
    {"kpi_id": "WS3-BI-007", "brand": "Fabhalta", "region": "", "points": 24},
    # zero points is not a trend
    {"kpi_id": "WS3-BI-010", "brand": "", "region": "", "points": 0},
    # junk row is skipped
    {"kpi_id": "", "brand": None, "points": "x"},
]

OUTCOMES: List[str] = [
    "treatment_initiated",
    "persistent_180d",
    "trx_volume",
    "nrx_volume",
    "discontinued_180d",
    "roi",
    "adopted",
    "trx_market_share",
    "nbrx_volume",
    "intent_to_prescribe",
    "adherent_180d",
    "action_taken",
    "low_gap_180d",
    "conversion_flag",
]


async def _coverage() -> List[Dict[str, Any]]:
    return list(COVERAGE_ROWS)


async def _outcomes() -> List[str]:
    return list(OUTCOMES)


async def _boom() -> Any:
    raise RuntimeError("db down")


async def _empty() -> list:
    return []


async def make_catalog(coverage=_coverage, outcomes=_outcomes) -> cat.CapabilityCatalog:
    return await cat.build_capability_catalog(coverage_loader=coverage, outcomes_loader=outcomes)


# =============================================================================
# BUILDER
# =============================================================================


async def test_kpis_come_from_the_registry():
    c = await make_catalog()
    ids = {k.id for k in c.kpis}
    assert "WS3-BI-005" in ids
    assert len(ids) >= 40
    assert c.kpi_name("WS3-BI-005") == "Total Prescriptions (TRx)"
    # unknown ids fall back to the id itself (never KeyError in a prompt)
    assert c.kpi_name("NOPE-1") == "NOPE-1"


async def test_trend_sets_from_coverage_rows():
    c = await make_catalog()
    assert c.trend_kpi_ids == frozenset({"WS3-BI-005", "WS3-BI-007"})
    assert c.per_brand_only_trend_ids == frozenset({"WS3-BI-007"})


async def test_axis_kpis_from_segmented_history_families():
    c = await make_catalog()
    assert c.axis_kpi_ids == frozenset(SEGMENTED_KPI_QUERY_FAMILIES)


async def test_outcomes_sorted_deduped_and_roster_present():
    async def dup() -> List[str]:
        return ["roi", "roi", "adopted", ""]

    c = await make_catalog(outcomes=dup)
    assert c.causal_outcomes == ("adopted", "roi")
    assert "The E2I system has" in c.agent_roster
    assert c.degraded == ()


async def test_loader_failure_marks_degraded_and_does_not_raise():
    c = await make_catalog(coverage=_boom, outcomes=_boom)
    assert set(c.degraded) == {"trend_coverage", "causal_outcomes"}
    assert c.trend_kpi_ids == frozenset()
    assert c.causal_outcomes == ()
    # code-derived fields survive a DB outage
    assert len(c.kpis) >= 40
    assert c.axis_kpi_ids == frozenset(SEGMENTED_KPI_QUERY_FAMILIES)


async def test_empty_results_are_degraded_too():
    # KPIHistoryRepository.get_coverage returns [] on error AND when it has no
    # client; an empty coverage view is not a realistic prod state.
    c = await make_catalog(coverage=_empty, outcomes=_empty)
    assert set(c.degraded) == {"trend_coverage", "causal_outcomes"}
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `$PY -m pytest tests/api/test_chat_capability_catalog.py -n 0 -q -p no:cacheprovider 2>&1 | tail -3`
Expected: `ImportError` / `ModuleNotFoundError: No module named 'src.services.chat_capability_catalog'`

- [ ] **Step 3: Write the module**

Create `src/services/chat_capability_catalog.py`:

```python
"""Capability catalog for the chat suggestion pills (``POST /api/chat/suggestions``).

Why this exists
---------------
Measured 2026-09-05 (docs/demos/results/2026-09-05_pill_suggestions_review/):
42% of the live suggestion pills asked for analyses the E2I assistant cannot
deliver -- SHAP recomputation, territory detail, trends of causal-registry
outcomes -- because the pill prompt described the assistant's abilities in one
prose sentence. A prompt carrying a catalog derived from code and data moved
the unanswerable share to 9% in a faithful prototype.

Everything list-shaped here is DERIVED, never transcribed (#1638 roster
pattern): KPI names from the registry, trend coverage from
``v_kpi_history_coverage``, axis-capable KPIs from the segmented-history
families, causal outcomes from the causal-path registry, agents from the
factory. The only hand-written text is the axis/composition RULES (guarded by
a test against ``kpi_calculate_tool``'s signature) and the per-route hints.

Design: docs/superpowers/specs/2026-09-05-copilot-pill-capability-catalog-design.md
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Dict, FrozenSet, List, Optional, Sequence, Tuple

from src.agents.factory import build_agent_roster_block
from src.kpi.registry import get_registry
from src.kpi.segmented_history import SEGMENTED_KPI_QUERY_FAMILIES

logger = logging.getLogger(__name__)

CoverageLoader = Callable[[], Awaitable[List[Dict[str, Any]]]]
OutcomesLoader = Callable[[], Awaitable[List[str]]]


@dataclass(frozen=True)
class KpiEntry:
    """One registry KPI as the prompt needs it."""

    id: str
    name: str
    workstream: str  # Workstream.value, e.g. "ws3_business"
    brand: Optional[str]  # brand-specific KPIs name their brand


@dataclass(frozen=True)
class CapabilityCatalog:
    """What the assistant can answer, as data. Rendered by :func:`render_catalog_block`."""

    kpis: Tuple[KpiEntry, ...]
    trend_kpi_ids: FrozenSet[str]  # have a materialized monthly series
    per_brand_only_trend_ids: FrozenSet[str]  # trend exists only in per-brand scopes
    axis_kpi_ids: FrozenSet[str]  # accept severity / therapy-line splits
    causal_outcomes: Tuple[str, ...]  # distinct end_node names in the causal registry
    agent_roster: str  # prompt-ready roster block from the factory
    degraded: Tuple[str, ...] = ()  # DB-backed fields that failed to load
    loaded_at: float = 0.0  # time.monotonic() at build

    def kpi_name(self, kpi_id: str) -> str:
        for entry in self.kpis:
            if entry.id == kpi_id:
                return entry.name
        return kpi_id


async def _default_coverage_loader() -> List[Dict[str, Any]]:
    from src.repositories.kpi_history import get_kpi_history_repository

    repo = await get_kpi_history_repository()
    return await repo.get_coverage()


async def _default_outcomes_loader() -> List[str]:
    from src.kpi.synthetic_mode import kpi_include_synthetic
    from src.memory.services.factories import get_async_supabase_client
    from src.repositories.causal_path import CausalPathRepository

    client = await get_async_supabase_client()
    repo = CausalPathRepository(client)
    return await repo.get_distinct_outcomes(include_synthetic=kpi_include_synthetic())


def _kpi_entries() -> Tuple[KpiEntry, ...]:
    entries = [
        KpiEntry(id=k.id, name=k.name, workstream=k.workstream.value, brand=k.brand)
        for k in get_registry().get_all()
    ]
    return tuple(sorted(entries, key=lambda e: (e.workstream, e.id)))


def _trend_sets(rows: Sequence[Dict[str, Any]]) -> Tuple[FrozenSet[str], FrozenSet[str]]:
    scopes: Dict[str, set[str]] = {}
    for row in rows:
        kpi_id = str(row.get("kpi_id") or "")
        try:
            points = int(row.get("points") or 0)
        except (TypeError, ValueError):
            points = 0
        if not kpi_id or points <= 0:
            continue
        brand = row.get("brand")
        scopes.setdefault(kpi_id, set()).add("" if brand is None else str(brand))
    trend = frozenset(scopes)
    per_brand_only = frozenset(k for k, brands in scopes.items() if "" not in brands)
    return trend, per_brand_only


async def build_capability_catalog(
    *,
    coverage_loader: Optional[CoverageLoader] = None,
    outcomes_loader: Optional[OutcomesLoader] = None,
) -> CapabilityCatalog:
    """Build the catalog. Never raises for a DB-backed field: it records it in ``degraded``."""
    degraded: List[str] = []

    rows: List[Dict[str, Any]] = []
    try:
        rows = list(await (coverage_loader or _default_coverage_loader)())
    except Exception as exc:  # noqa: BLE001 - degrade, never 502 the pills
        logger.warning("capability catalog: trend coverage unavailable: %s", exc)
    if not rows:
        degraded.append("trend_coverage")

    outcomes: List[str] = []
    try:
        outcomes = [str(o) for o in await (outcomes_loader or _default_outcomes_loader)() if o]
    except Exception as exc:  # noqa: BLE001
        logger.warning("capability catalog: causal outcomes unavailable: %s", exc)
    if not outcomes:
        degraded.append("causal_outcomes")

    trend, per_brand_only = _trend_sets(rows)
    return CapabilityCatalog(
        kpis=_kpi_entries(),
        trend_kpi_ids=trend,
        per_brand_only_trend_ids=per_brand_only,
        axis_kpi_ids=frozenset(SEGMENTED_KPI_QUERY_FAMILIES),
        causal_outcomes=tuple(sorted(set(outcomes))),
        agent_roster=build_agent_roster_block(),
        degraded=tuple(degraded),
        loaded_at=time.monotonic(),
    )
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `$PY -m pytest tests/api/test_chat_capability_catalog.py -n 0 -q -p no:cacheprovider 2>&1 | tail -3`
Expected: `6 passed`

- [ ] **Step 5: Commit**

```bash
git branch --show-current   # must print claude/copilot-pill-capability-catalog
git add src/services/chat_capability_catalog.py tests/api/test_chat_capability_catalog.py
git commit -F - <<'EOF'
feat(chat): capability catalog dataclass and builder for suggestion pills

Derives what the assistant can answer from code and data: KPI names from
the registry, trend coverage from v_kpi_history_coverage, axis-capable KPIs
from the segmented-history families, causal outcomes from the causal-path
registry, agents from the factory. DB-backed fields degrade instead of
raising.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01QCr4zzEYNQDCrLaM4gFHGr
EOF
```

---

### Task 2: Catalog renderer

**Files:**
- Modify: `src/services/chat_capability_catalog.py`
- Test: `tests/api/test_chat_capability_catalog.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/api/test_chat_capability_catalog.py`:

```python
# =============================================================================
# RENDERER
# =============================================================================


async def test_render_lists_registry_kpis_by_area():
    c = await make_catalog()
    block = cat.render_catalog_block(c)
    assert block.startswith("WHAT THE ASSISTANT CAN DO")
    assert "A. KPI values" in block
    assert "- Business impact: " in block
    assert "Total Prescriptions (TRx)" in block


async def test_render_trend_and_axis_kpis_by_name():
    c = await make_catalog()
    block = cat.render_catalog_block(c)
    assert f"{c.kpi_name('WS3-BI-007')} (per brand only)" in block
    assert c.kpi_name("WS3-BI-005") in block
    # axis KPIs are named in the comparison clause
    for kpi_id in SEGMENTED_KPI_QUERY_FAMILIES:
        assert c.kpi_name(kpi_id) in block


async def test_render_causal_outcomes_as_registry_nodes():
    c = await make_catalog()
    block = cat.render_catalog_block(c)
    assert "for these OUTCOMES only: " + ", ".join(c.causal_outcomes) in block
    assert "registry NODES, not KPIs" in block
    assert "NO time, region or segment dimension" in block


async def test_render_roster_never_block_and_letters():
    c = await make_catalog()
    block = cat.render_catalog_block(c)
    assert "The E2I system has" in block
    assert "NEVER PROPOSE" in block
    for letter in "ABCDEFGH":
        assert f"\n{letter}. " in block or block.startswith(f"{letter}. "), letter


async def test_render_degraded_fallbacks_invent_nothing():
    c = await make_catalog(coverage=_boom, outcomes=_boom)
    block = cat.render_catalog_block(c)
    assert "coverage list is unavailable" in block
    assert "outcome list is unavailable" in block
    assert "persistent_180d" not in block
    # the Rx-volume trends are code-derived and stay offered
    assert c.kpi_name("WS3-BI-005") in block


def test_axis_vocabulary_matches_kpi_calculate_tool():
    """The axis RULES are prose; pin their vocabulary to the tool's parameters
    so the prompt can never name an axis the tool does not accept."""
    import inspect

    from src.api.routes.chatbot_tools import kpi_calculate_tool

    params = set(inspect.signature(kpi_calculate_tool).parameters)
    for axis in cat.AXIS_PARAMETER_NAMES:
        assert axis in params, axis
        assert axis in cat.AXIS_RULES, axis
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `$PY -m pytest tests/api/test_chat_capability_catalog.py -n 0 -q -p no:cacheprovider 2>&1 | tail -3`
Expected: 6 failures with `AttributeError: module ... has no attribute 'render_catalog_block'` / `'AXIS_PARAMETER_NAMES'`

- [ ] **Step 3: Add the renderer**

Append to `src/services/chat_capability_catalog.py` (after `build_capability_catalog`). Also add `from src.kpi.models import Workstream` to the imports at the top.

```python
# =============================================================================
# RENDERER
# =============================================================================

_WORKSTREAM_ORDER: Tuple[Tuple[Workstream, str], ...] = (
    (Workstream.WS3_BUSINESS, "Business impact"),
    (Workstream.WS2_TRIGGERS, "Trigger performance"),
    (Workstream.WS1_MODEL_PERFORMANCE, "Model performance"),
    (Workstream.WS1_DATA_QUALITY, "Data quality"),
    (Workstream.BRAND_SPECIFIC, "Brand-specific"),
    (Workstream.CAUSAL_METRICS, "Causal-effect metrics"),
)

# Hand-written RULES (not a list). The axis words are pinned to
# kpi_calculate_tool's parameter names by test_axis_vocabulary_matches_kpi_calculate_tool.
AXIS_PARAMETER_NAMES: Tuple[str, ...] = ("segment", "therapy_line", "region", "biologic", "ige_tier")
AXIS_RULES = (
    "Breakdown axes, AT MOST ONE per ask: segment = patient severity tier (low/medium/high); "
    "therapy_line = line of therapy (0-3); region = US census region (northeast/south/midwest/west); "
    "and - Remibrutinib ONLY - biologic status (naive/experienced) or ige_tier (low/medium/high). "
    'An optional time window ("last 3 months", "Q1 2025", "2025-01-01 to 2025-03-31") composes with '
    "segment/therapy_line but NOT with region/biologic/ige_tier for share, conversion or trigger KPIs. "
    "TRx share is share of the tracked 3-brand portfolio, NOT share versus competitors."
)

NEVER_BLOCK = (
    "NEVER PROPOSE (no tool serves these): named HCP or patient lists / rosters / exports; "
    "territory-level detail; competitor brands' share or volume; TRx/NRx/NBRx \"by HCP segment\" "
    "(patient axes only); trends over time of SHAP values, CATE / treatment effects, predicted "
    "probabilities, gap sizes or optimizer allocations; recomputing, validating, re-deriving or "
    "EXTENDING an on-screen SHAP, optimizer, prediction, gap or CATE result (another segment, more "
    "features, per-territory detail, robustness, thresholds); two breakdown axes at once; causal "
    "drivers scoped to a region, month or segment; drivers OF a driver (unless it is itself a "
    "section-C outcome); thresholds, dose-response or nonlinearity questions; on-demand sensitivity / "
    'subgroup / "controlling for" analyses; live experiment status, lift or results; agent accuracy / '
    "error rates; audit-cycle metrics; data refresh schedules or pipeline latency; campaign-level ROI; "
    'toggling page UI (e.g. nowcast overlay); undefined ratios such as "conversion from NRx to NBRx"; '
    "emails, external data, CRM or any write action; treating a section-C outcome as a KPI (its rate, "
    "value, trend, chart or breakdown)."
)


def _names(catalog: CapabilityCatalog, ids: FrozenSet[str], *, mark_per_brand: bool = False) -> str:
    parts: List[str] = []
    for kpi_id in sorted(ids):
        name = catalog.kpi_name(kpi_id)
        if mark_per_brand and kpi_id in catalog.per_brand_only_trend_ids:
            name += " (per brand only)"
        parts.append(name)
    return ", ".join(parts)


def render_catalog_block(catalog: CapabilityCatalog) -> str:
    """Render the catalog as the prompt's A-H capability sections plus the NEVER list."""
    lines: List[str] = ["WHAT THE ASSISTANT CAN DO (every pill must map to exactly one of A-H):"]

    lines.append(
        "A. KPI values - the current value of any registry KPI, per brand, optionally over a time "
        "window. Registry KPIs by area:"
    )
    for workstream, label in _WORKSTREAM_ORDER:
        names = [
            e.name + (f" ({e.brand} only)" if e.brand else "")
            for e in catalog.kpis
            if e.workstream == workstream.value
        ]
        if names:
            lines.append(f"   - {label}: {'; '.join(names)}")
    lines.append("   " + AXIS_RULES)

    axis_names = _names(catalog, catalog.axis_kpi_ids)
    if "trend_coverage" in catalog.degraded:
        trend_clause = (
            "a monthly trend line for the KPIs with a materialized history (the coverage list is "
            f"unavailable right now - the Rx-volume KPIs {axis_names} always have one; propose trends "
            "only for those)"
        )
    else:
        trend_clause = (
            f"a monthly trend line for {_names(catalog, catalog.trend_kpi_ids, mark_per_brand=True)}"
        )
    lines.append(
        f"B. Charts: {trend_clause}; ONE chart comparing severity tiers or lines of therapy for "
        f"{axis_names}; any other registry KPI as a current-value chart; several KPIs side by side."
    )

    if "causal_outcomes" in catalog.degraded:
        lines.append(
            "C. Causal drivers, causal paths and treatment effects from the causal-path registry, per "
            "brand, with confidence and refutation evidence, for the registry's patient-journey and "
            "commercial outcomes. The outcome list is unavailable right now: propose at most ONE "
            'causal-driver pill, phrased "what drives <the outcome or KPI named on screen> for '
            '<brand>?", and invent no outcome names.'
        )
    else:
        lines.append(
            "C. Causal drivers, causal paths and treatment effects from the causal-path registry, per "
            "brand, with confidence and refutation evidence, for these OUTCOMES only: "
            f"{', '.join(catalog.causal_outcomes)}. These outcomes are registry NODES, not KPIs: they "
            "cannot be computed, trended, charted or broken down by region, segment or month - a "
            'driver question is "what drives <outcome> for <brand>?", nothing finer. The registry has '
            "NO time, region or segment dimension."
        )

    lines.append(
        "D. Segments: KPI breakdowns by ONE of the axes in A; a ranking of HCP segments by predicted "
        "likelihood to prescribe a brand, by specialty OR by geographic region; aggregate HCP / "
        "patient cohort profiles (counts by specialty, tier, severity - never named individuals)."
    )
    lines.append(
        "E. Clinical and regulatory context per brand: FDA-label indications, mechanism of action, "
        "pivotal trial endpoints, real-world evidence, competitor landscape (as context, not as data)."
    )
    lines.append(
        "F. Platform: the agents below and what each does; an agent's recent activity; the system "
        "health score; experiment design, drift checks and gap/ROI opportunity analysis run through "
        "the orchestrator."
    )
    lines.extend("   " + line for line in catalog.agent_roster.splitlines())
    lines.append("G. Internal document / knowledge-base search.")
    lines.append(
        "H. Dashboard actions: navigate to a page, set the brand or region filter, set the date range."
    )
    lines.append("")
    lines.append(NEVER_BLOCK)
    return "\n".join(lines)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `$PY -m pytest tests/api/test_chat_capability_catalog.py -n 0 -q -p no:cacheprovider 2>&1 | tail -3`
Expected: `12 passed`

- [ ] **Step 5: Commit**

```bash
git branch --show-current
git add src/services/chat_capability_catalog.py tests/api/test_chat_capability_catalog.py
git commit -F - <<'EOF'
feat(chat): render the capability catalog as the pill prompt's A-H sections

Sections A-H mirror the prototype prompt measured on 2026-09-05; the axis
rules are pinned to kpi_calculate_tool's parameters by test; degraded
DB-backed fields render an honest fallback and invent no names.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01QCr4zzEYNQDCrLaM4gFHGr
EOF
```

---

### Task 3: Route hints

**Files:**
- Modify: `src/services/chat_capability_catalog.py`
- Test: `tests/api/test_chat_capability_catalog.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/api/test_chat_capability_catalog.py`:

```python
# =============================================================================
# ROUTE HINTS
# =============================================================================

import re as _re


def test_route_hints_are_normalized_paths_with_a_catalog_letter():
    assert "/" in cat.ROUTE_HINTS
    for path, hint in cat.ROUTE_HINTS.items():
        assert path.startswith("/"), path
        assert path == "/" or not path.endswith("/"), path
        assert hint.strip() == hint and hint.endswith("."), path
        # every hint tells the model which catalog letters fit: "(A/B)", "(C)"
        assert _re.search(r"\([A-H](?:/[A-H])*\)", hint), path


def test_route_hint_lookup_tolerates_query_and_trailing_slash():
    expected = cat.ROUTE_HINTS["/kpi-dictionary"]
    assert cat.route_hint("/kpi-dictionary") == expected
    assert cat.route_hint("/kpi-dictionary/") == expected
    assert cat.route_hint("/kpi-dictionary?tab=ws3") == expected
    assert cat.route_hint("/") == cat.ROUTE_HINTS["/"]


def test_route_hint_unknown_or_missing_page_is_empty():
    assert cat.route_hint("/nope") == ""
    assert cat.route_hint(None) == ""
    assert cat.route_hint("") == ""
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `$PY -m pytest tests/api/test_chat_capability_catalog.py -n 0 -q -p no:cacheprovider 2>&1 | tail -3`
Expected: 3 failures, `AttributeError: ... 'ROUTE_HINTS'`

- [ ] **Step 3: Add the hint map**

Append to `src/services/chat_capability_catalog.py`:

```python
# =============================================================================
# ROUTE HINTS - used only when page_content is empty
# =============================================================================
# One sentence per app route (frontend/src/router/routes.tsx, auth routes
# excluded): what the page shows and which catalog letters fit it. A renamed
# route simply falls back to today's behaviour (path + brand only).

ROUTE_HINTS: Dict[str, str] = {
    "/": (
        "Home dashboard: KPI tiles (TRx, market share, HCP reach), active campaigns, model accuracy, "
        "system health and the top gap opportunity; pills should ask for KPI values or trends (A/B), "
        "drivers of the gap's KPI (C) or platform health (F)."
    ),
    "/documentation": (
        "How E2I Works: explains the platform, its agents and analyses; pills should ask what the "
        "assistant can analyse, which agents exist (F) or where a KPI is defined (A)."
    ),
    "/ai-insights": (
        "Executive Insights: brand-level narrative of KPI movements and causal drivers; pills should "
        "ask for KPI values and trends (A/B) and causal drivers (C)."
    ),
    "/knowledge-graph": (
        "Knowledge Graph: causal paths between drivers and outcomes; pills should ask for the drivers "
        "of a registry outcome (C) or the KPIs those outcomes relate to (A)."
    ),
    "/causal-analysis": (
        "Causal Analysis: treatment-effect estimates driver -> outcome with confidence and refutation; "
        "pills should ask for drivers or paths of an outcome (C), never a trend of an effect."
    ),
    "/causal-discovery": (
        "Causal Discovery: discovered causal graphs over the patient-journey data; pills should ask for "
        "drivers of a registry outcome (C) or the related KPIs (A)."
    ),
    "/segment-analysis": (
        "Segment Analysis: KPI and effect differences across patient axes (severity tier, line of "
        "therapy, biologic/IgE for Remibrutinib); pills should ask for KPI breakdowns by ONE axis (A/D)."
    ),
    "/expert-reviews": (
        "Expert Reviews: human review queue for agent outputs; pills should ask about agents and "
        "platform status (F) or the KPIs under review (A)."
    ),
    "/predictive-analytics": (
        "Predictive Analytics: scored cohorts and predicted probabilities from the ML models; pills "
        "should ask for the HCP segment likelihood ranking (D), model-performance KPIs (A) or clinical "
        "context (E), never per-patient predictions."
    ),
    "/model-performance": (
        "Model Performance: ROC-AUC, PR-AUC, F1, calibration and PSI drift KPIs; pills should ask for "
        "those KPI values or charts (A/B)."
    ),
    "/feature-importance": (
        "Feature Importance: SHAP feature rankings for the brand models; pills should turn a feature "
        "into a catalog ask - HCP segment likelihood (D), a KPI breakdown (A) or clinical context (E) - "
        "never SHAP recomputation."
    ),
    "/time-series": (
        "Time Series: monthly KPI history with nowcast; pills should ask for trend charts and period "
        "comparisons of ONE KPI (B) or KPI values over a window (A)."
    ),
    "/intervention-impact": (
        "Intervention Impact: measured effects of interventions on outcomes; pills should ask for "
        "drivers or treatment effects of a registry outcome (C) and the affected KPIs (A/B)."
    ),
    "/digital-twin": (
        "Digital Twin: simulated intervention scenarios; pills should ask for the causal drivers behind "
        "a scenario's outcome (C) or the baseline KPI values (A)."
    ),
    "/gap-analysis": (
        "Gap Analysis: KPI gaps versus target by segment with expected ROI; pills should ask for the "
        "underlying KPI value or breakdown (A), its trend (B) or its drivers (C), never a trend of the "
        "gap itself."
    ),
    "/resource-optimization": (
        "Resource Optimization: recommended field-force allocation by territory; pills should ask for "
        "regional KPI breakdowns (A), ROI (A/B) or causal drivers (C), never territory-level detail."
    ),
    "/experiments": (
        "Experiments: designed A/B tests and experiment proposals; pills should ask for experiment "
        "design through the orchestrator (F) or the KPIs an experiment targets (A), never live lift or "
        "results."
    ),
    "/kpi-dictionary": (
        "KPI Dictionary: registry definitions of every KPI; pills should ask for the value, definition, "
        "chart or drivers of specific KPIs (A/B/C)."
    ),
    "/data-quality": (
        "Data Quality: source coverage, match rate and freshness KPIs; pills should ask for those KPI "
        "values or charts (A/B)."
    ),
    "/system-health": (
        "System Health: platform health score and component status; pills should ask about the health "
        "score and the agents (F)."
    ),
    "/monitoring": (
        "Monitoring: drift and model-monitoring KPIs; pills should ask for PSI/drift and "
        "model-performance KPI values or charts (A/B) or a drift check via the orchestrator (F)."
    ),
    "/analytics": (
        "Analytics: cross-brand KPI overview; pills should ask for KPI values, comparisons and trends "
        "(A/B)."
    ),
    "/agent-orchestration": (
        "Agent Orchestration: the agent tiers and their recent activity; pills should ask which agents "
        "exist, what they do and what they did recently (F)."
    ),
    "/memory-architecture": (
        "Memory Architecture: how the assistant's memory tiers work; pills should ask about the "
        "platform and agents (F) or search internal documents (G)."
    ),
    "/audit-chain": (
        "Audit Chain: provenance of agent decisions; pills should ask about agents and their activity "
        "(F), never audit-cycle metrics."
    ),
    "/feedback-learning": (
        "Feedback Learning: how user feedback improves the agents; pills should ask about agents (F) "
        "or internal documents (G)."
    ),
    "/admin": (
        "Administration: users and settings; pills should stay on platform status and agents (F)."
    ),
}


def route_hint(page: Optional[str]) -> str:
    """Hint for ``page`` ('' when unknown). Tolerates a query string and a trailing slash."""
    if not page:
        return ""
    path = page.split("?", 1)[0].split("#", 1)[0].rstrip("/") or "/"
    return ROUTE_HINTS.get(path, "")
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `$PY -m pytest tests/api/test_chat_capability_catalog.py -n 0 -q -p no:cacheprovider 2>&1 | tail -3`
Expected: `15 passed`

- [ ] **Step 5: Commit**

```bash
git branch --show-current
git add src/services/chat_capability_catalog.py tests/api/test_chat_capability_catalog.py
git commit -F - <<'EOF'
feat(chat): per-route hints for pages that publish no page summary

Grounds opener pills on the 19 routes without a pageChatSummary, which
the prototype showed collapsing onto the same lead pill.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01QCr4zzEYNQDCrLaM4gFHGr
EOF
```

---

### Task 4: Deterministic pill validator

**Files:**
- Modify: `src/services/chat_capability_catalog.py`
- Test: `tests/api/test_chat_capability_catalog.py`

The fixtures below are REAL pills from the 2026-09-05 live sample
(`live_pills_baseline_2026-09-05.json`) that were graded NO, plus synthetic
outcome-as-KPI pills of the shape the prototype still produced.

- [ ] **Step 1: Write the failing tests**

Append to `tests/api/test_chat_capability_catalog.py`:

```python
# =============================================================================
# VALIDATOR
# =============================================================================

from dataclasses import dataclass as _dataclass


@_dataclass
class _Pill:
    title: str
    message: str


# Live pills graded NO on 2026-09-05, by rule family.
DROP_FIXTURES = [
    (
        "gap_recompute",
        "Market share gap trend",
        "Can you chart how Kisqali's market share gap in the midwest has evolved over the past 12 months?",
    ),
    (
        "shap_or_feature_importance",
        "Champion cohort vs. all HCPs",
        "How do feature importances differ between the fabhalta_hcp_adoption_champion cohort and all Fabhalta-prescribing HCPs?",
    ),
    (
        "individual_prediction",
        "Persistence trend by baseline UAS7",
        "Can you chart the predicted 180-day persistence probability for Remibrutinib across baseline UAS7 severity tiers?",
    ),
    (
        "individual_prediction",
        "Persistence by IgE tier",
        "What is the distribution of predicted 180-day persistence probability for Remibrutinib across IgE tiers?",
    ),
    (
        "individual_prediction",
        "Model performance & calibration",
        "What is the validation accuracy of the patient_persistence model for Remibrutinib, and how reliable is the 61.3% mean predicted probability?",
    ),
    (
        "territory_detail",
        "Why T-114 gained the most",
        "What are the key drivers behind the +6 field force increase recommended for territory T-114 in Fabhalta's optimization?",
    ),
    (
        "territory_detail",
        "Impact of T-072 reduction",
        "What is the expected TRx impact on Fabhalta if we reduce field force in territory T-072 by 4 as suggested?",
    ),
    (
        "uplift_by_segment",
        "Why naive > experienced?",
        "Run a causal analysis to identify the drivers behind biologic-naive patients showing +0.16 CATE versus +0.07 for biologic-experienced on Remibrutinib.",
    ),
    (
        "uplift_by_segment",
        "Validate persistence model",
        "Can we run a sensitivity analysis on the uas7_baseline -> persistent_180d treatment effect for Remibrutinib to test robustness across patient subgroups?",
    ),
    (
        "off_platform_action",
        "Email the summary",
        "Email this TRx summary for Kisqali to the brand team.",
    ),
    (
        "competitor_data",
        "Competitor share",
        "What is Kisqali's TRx versus competitors in the Northeast?",
    ),
    # outcome-as-KPI: the residual family the prototype still produced
    (
        "outcome_as_kpi:persistent_180d",
        "Persistence by region",
        "Chart the persistent_180d rate for Remibrutinib by census region.",
    ),
    (
        "outcome_as_kpi:adherent_180d",
        "Adherence trend",
        "What is the adherent 180d trend for Kisqali over the last 6 months?",
    ),
    (
        "outcome_as_kpi:discontinued_180d",
        "Discontinuation level",
        "What is the discontinued_180d percentage for Fabhalta?",
    ),
]

# Pills the assistant CAN answer; every one must survive.
KEEP_FIXTURES = [
    ("Persistence drivers", "What drives persistent_180d for Remibrutinib, and how confident are those paths?"),
    ("Kisqali TRx trend", "Show me the month-over-month trend for Kisqali total TRx."),
    ("TRx by severity", "Chart Fabhalta's TRx trend by severity tier."),
    ("Midwest conversion", "What is Kisqali's conversion rate in the Midwest over the last 3 months?"),
    ("Likely specialties", "Which HCP specialties are most likely to increase Fabhalta prescriptions?"),
    ("ROI trend", "Chart the ROI trend for Remibrutinib."),
    ("Action rate uplift", "What is the action rate uplift for Kisqali?"),
    ("TRx volume drivers", "What are the causal drivers of trx_volume for Kisqali?"),
    ("Active agents", "Which agents are active right now and what are they working on?"),
    ("Competitive landscape", "Give me the competitive landscape context for Fabhalta's PNH indication."),
    ("Effect comparison", "How does the nba_trigger_accepted -> persistent_180d effect for Remibrutinib compare in confidence to the uas7_baseline path?"),
    ("Regional TRx", "What is Fabhalta's TRx by census region?"),
]


async def test_journey_outcomes_exclude_kpi_named_outcomes():
    c = await make_catalog()
    journey = set(cat.journey_outcomes(c))
    assert {"persistent_180d", "discontinued_180d", "adherent_180d", "low_gap_180d", "adopted"} <= journey
    # KPI-named outcomes (a trend of ROI or TRx volume IS answerable) stay out
    for kpi_like in ("roi", "trx_volume", "nrx_volume", "nbrx_volume", "trx_market_share"):
        assert kpi_like not in journey, kpi_like


async def test_treatment_initiated_is_left_to_the_prompt():
    """The KPI recognizer reads 'treatment initiated' as a causal-metric KPI
    mention (CM-001), so the outcome rule does not fire on it; the prompt's
    section C carries that case. Pinned so a recognizer change is visible."""
    c = await make_catalog()
    assert "treatment_initiated" not in cat.journey_outcomes(c)
    kept, dropped = cat.filter_unsupported_pills(
        [_Pill("LoT depth", "What is the treatment_initiated conversion rate for Fabhalta in line-of-therapy 0?")],
        c,
    )
    assert len(kept) == 1 and dropped == []


@pytest.mark.parametrize("rule,title,message", DROP_FIXTURES)
async def test_known_unsupported_pills_are_dropped(rule, title, message):
    c = await make_catalog()
    kept, dropped = cat.filter_unsupported_pills([_Pill(title, message)], c)
    assert kept == []
    assert [r for _, r in dropped] == [rule]


@pytest.mark.parametrize("title,message", KEEP_FIXTURES)
async def test_supported_pills_are_kept(title, message):
    c = await make_catalog()
    kept, dropped = cat.filter_unsupported_pills([_Pill(title, message)], c)
    assert dropped == []
    assert [p.title for p in kept] == [title]


async def test_filter_preserves_order_and_returns_rules():
    c = await make_catalog()
    pills = [
        _Pill("keep-1", "Chart the TRx trend for Kisqali."),
        _Pill("drop", "Which SHAP features matter most for Kisqali adoption?"),
        _Pill("keep-2", "What drives adopted for Kisqali?"),
    ]
    kept, dropped = cat.filter_unsupported_pills(pills, c)
    assert [p.title for p in kept] == ["keep-1", "keep-2"]
    assert [(p.title, r) for p, r in dropped] == [("drop", "shap_or_feature_importance")]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `$PY -m pytest tests/api/test_chat_capability_catalog.py -n 0 -q -p no:cacheprovider 2>&1 | tail -3`
Expected: failures with `AttributeError: ... 'journey_outcomes'` / `'filter_unsupported_pills'`

- [ ] **Step 3: Add the validator**

Add `import re` and `from typing import Protocol, TypeVar` to the module's imports, then append to `src/services/chat_capability_catalog.py`:

```python
# =============================================================================
# VALIDATOR - narrow, deterministic, tuned to the pill families graded NO
# =============================================================================


class SuggestionLike(Protocol):
    @property
    def title(self) -> str: ...

    @property
    def message(self) -> str: ...


P = TypeVar("P", bound=SuggestionLike)

# A time-boxed journey flag (persistent_180d) is never a KPI.
_DURATION_OUTCOME_RE = re.compile(r"_\d+d$")

# "the <outcome> rate / trend / by region ..." - the outcome used as a metric.
_VALUE_ASK_RE = re.compile(
    r"\b(?:rates?|values?|levels?|trends?|chart|plot|graph|over time|monthly|month-over-month|"
    r"quarterly|weekly|by (?:census )?region|by (?:severity )?tier|by segment|by line|breakdown|"
    r"distribution|percentage|how many|count|volume)\b",
    re.I,
)
# ... unless the pill is a causal question, which section C serves.
_CAUSAL_ASK_RE = re.compile(
    r"\b(?:driv\w*|caus\w*|paths?|effects?|why|influenc\w*|factors?|impacts?|refut\w*|confiden\w*)\b",
    re.I,
)

_OFF_PLATFORM_RULES: Tuple[Tuple[str, "re.Pattern[str]"], ...] = (
    (
        "shap_or_feature_importance",
        re.compile(r"\bSHAP\b|\bfeature[- ]importances?\b|\bfeature rankings?\b|\btop(?:-| )?\d* ?features\b", re.I),
    ),
    ("territory_detail", re.compile(r"\bterritor(?:y|ies)\b|\bT-\d{3}\b", re.I)),
    (
        "individual_prediction",
        re.compile(
            r"\bpredicted (?:\d+-day )?(?:[a-z_]+ )?probabilit(?:y|ies)\b|\bmean predicted probability\b|"
            r"\bpropensity scores?\b|\b(?:each|individual|specific) (?:HCP|patient|prescriber)s?\b|"
            r"\b(?:HCP|patient) (?:list|roster)s?\b",
            re.I,
        ),
    ),
    (
        "gap_recompute",
        re.compile(
            r"\bgap\b[^.?]*\b(?:trend|evolv\w*|evolution|over the (?:past|last)|chart|plot|month)\b|"
            r"\b(?:chart|plot|trend of)\b[^.?]*\bgap\b",
            re.I,
        ),
    ),
    (
        "uplift_by_segment",
        re.compile(
            r"\bCATE\b|\bheterogen\w*\b|\btreatment effects? (?:by|across|for) (?:patient )?"
            r"(?:segment|tier|subgroup|cohort)s?\b|\bsubgroup analys\w*|\bsensitivity analys\w*|"
            r"\bcontrolling for\b",
            re.I,
        ),
    ),
    (
        "off_platform_action",
        re.compile(
            r"\be-?mails?\b|\bexport(?:s|ed|ing)?\b|\bCRM\b|\bVeeva\b|"
            r"\bsend (?:a |an )?(?:report|message|alert|email)\b",
            re.I,
        ),
    ),
    (
        "competitor_data",
        re.compile(
            r"\bcompetitors?'?s? (?:market )?(?:share|volume|TRx|NRx|sales)\b|"
            r"\b(?:vs\.?|versus|against) (?:the )?competitors?\b",
            re.I,
        ),
    ),
)


def journey_outcomes(catalog: CapabilityCatalog) -> Tuple[str, ...]:
    """Outcomes with no KPI counterpart - the ones a pill can mistake for a metric.

    Time-boxed journey flags (``persistent_180d``) are never KPIs; anything else
    the KPI recognizer cannot resolve (``adopted``) is treated the same. Outcomes
    the recognizer reads as a KPI mention (``roi``, ``trx_volume``, and also
    ``treatment_initiated`` -> the causal-metric KPI) are left to the prompt.
    """
    from src.services.kpi_resolution import recognize_kpi

    out: List[str] = []
    for outcome in catalog.causal_outcomes:
        if _DURATION_OUTCOME_RE.search(outcome) or recognize_kpi(outcome.replace("_", " ")) is None:
            out.append(outcome)
    return tuple(out)


def match_unsupported_rule(text: str, journey: Sequence[str]) -> Optional[str]:
    """Name of the rule ``text`` violates, or None when the pill is supported."""
    for name, pattern in _OFF_PLATFORM_RULES:
        if pattern.search(text):
            return name
    lowered = text.lower()
    for outcome in journey:
        spaced = outcome.replace("_", " ")
        if (
            re.search(rf"\b{re.escape(outcome)}\b", lowered) is None
            and re.search(rf"\b{re.escape(spaced)}\b", lowered) is None
        ):
            continue
        if _VALUE_ASK_RE.search(lowered) and not _CAUSAL_ASK_RE.search(lowered):
            return f"outcome_as_kpi:{outcome}"
    return None


def filter_unsupported_pills(
    pills: Sequence[P], catalog: CapabilityCatalog
) -> Tuple[List[P], List[Tuple[P, str]]]:
    """Split ``pills`` into (kept, [(dropped, rule), ...]) preserving order."""
    journey = journey_outcomes(catalog)
    kept: List[P] = []
    dropped: List[Tuple[P, str]] = []
    for pill in pills:
        rule = match_unsupported_rule(f"{pill.title} {pill.message}", journey)
        if rule is None:
            kept.append(pill)
        else:
            dropped.append((pill, rule))
    return kept, dropped
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `$PY -m pytest tests/api/test_chat_capability_catalog.py -n 0 -q -p no:cacheprovider 2>&1 | tail -3`
Expected: `44 passed` (15 earlier + 2 + 14 parametrized drops + 12 parametrized keeps + 1)

If a DROP fixture is kept or a KEEP fixture is dropped, fix the regex for that family; do not delete the fixture.

- [ ] **Step 5: Commit**

```bash
git branch --show-current
git add src/services/chat_capability_catalog.py tests/api/test_chat_capability_catalog.py
git commit -F - <<'EOF'
feat(chat): deterministic validator for unsupported suggestion pills

Two narrow families measured NO on 2026-09-05: off-platform asks (SHAP,
territory detail, per-patient predictions, gap trends, CATE/subgroup
analyses, email/CRM, competitor data) and causal-registry outcomes used as
KPIs. Outcome names come from the catalog; KPI-named outcomes are exempt.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01QCr4zzEYNQDCrLaM4gFHGr
EOF
```

---

### Task 5: TTL cache

**Files:**
- Modify: `src/services/chat_capability_catalog.py`
- Test: `tests/api/test_chat_capability_catalog.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/api/test_chat_capability_catalog.py`:

```python
# =============================================================================
# CACHE
# =============================================================================


class _Counting:
    def __init__(self, fn):
        self.fn, self.calls = fn, 0

    async def __call__(self):
        self.calls += 1
        return await self.fn()


async def test_cache_builds_once_within_ttl():
    cov, out = _Counting(_coverage), _Counting(_outcomes)
    cache = cat._CatalogCache()
    first = await cache.get(now=1000.0, coverage_loader=cov, outcomes_loader=out)
    second = await cache.get(now=1000.0 + cat.CATALOG_TTL_SECONDS - 1, coverage_loader=cov, outcomes_loader=out)
    assert second is first
    assert (cov.calls, out.calls) == (1, 1)


async def test_cache_rebuilds_after_ttl():
    cov, out = _Counting(_coverage), _Counting(_outcomes)
    cache = cat._CatalogCache()
    first = await cache.get(now=1000.0, coverage_loader=cov, outcomes_loader=out)
    second = await cache.get(now=1000.0 + cat.CATALOG_TTL_SECONDS + 1, coverage_loader=cov, outcomes_loader=out)
    assert second is not first
    assert (cov.calls, out.calls) == (2, 2)


async def test_degraded_catalog_retries_sooner_and_heals():
    cache = cat._CatalogCache()
    broken = await cache.get(now=0.0, coverage_loader=_boom, outcomes_loader=_boom)
    assert broken.degraded
    # still cached inside the short TTL
    same = await cache.get(now=cat.DEGRADED_TTL_SECONDS - 1, coverage_loader=_coverage, outcomes_loader=_outcomes)
    assert same is broken
    healed = await cache.get(now=cat.DEGRADED_TTL_SECONDS + 1, coverage_loader=_coverage, outcomes_loader=_outcomes)
    assert healed.degraded == ()
    assert healed.causal_outcomes == tuple(sorted(set(OUTCOMES)))


async def test_refresh_failure_keeps_last_good_fields():
    cache = cat._CatalogCache()
    good = await cache.get(now=0.0, coverage_loader=_coverage, outcomes_loader=_outcomes)
    after = await cache.get(now=cat.CATALOG_TTL_SECONDS + 1, coverage_loader=_boom, outcomes_loader=_boom)
    assert after is not good
    assert after.causal_outcomes == good.causal_outcomes
    assert after.trend_kpi_ids == good.trend_kpi_ids
    assert after.per_brand_only_trend_ids == good.per_brand_only_trend_ids
    assert after.degraded == ()  # stale-but-good is not degraded


async def test_module_level_accessor_and_reset(monkeypatch):
    calls = {"n": 0}

    async def fake_build(**kwargs):
        calls["n"] += 1
        return await make_catalog()

    monkeypatch.setattr(cat, "build_capability_catalog", fake_build)
    cat.reset_capability_catalog_cache()
    a = await cat.get_capability_catalog()
    b = await cat.get_capability_catalog()
    assert a is b and calls["n"] == 1
    cat.reset_capability_catalog_cache()
    c = await cat.get_capability_catalog()
    assert c is not a and calls["n"] == 2
    cat.reset_capability_catalog_cache()
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `$PY -m pytest tests/api/test_chat_capability_catalog.py -n 0 -q -p no:cacheprovider 2>&1 | tail -3`
Expected: 5 failures, `AttributeError: ... '_CatalogCache'`

- [ ] **Step 3: Add the cache**

Add `import dataclasses` to the imports and these two constants right after `OutcomesLoader = ...` near the top of the module:

```python
CATALOG_TTL_SECONDS = 600.0
# A degraded catalog (a DB-backed field failed) retries sooner, but not on
# every pill request - a down database must not be hammered.
DEGRADED_TTL_SECONDS = 60.0
```

Append to `src/services/chat_capability_catalog.py`:

```python
# =============================================================================
# CACHE - lazy, in-process, TTL; no startup hook (CI runs TestClient lifespans
# on a 30 s thread timeout, and a lazy cache adds no work there)
# =============================================================================


def _keep_last_good_fields(
    fresh: CapabilityCatalog, previous: Optional[CapabilityCatalog]
) -> CapabilityCatalog:
    """On a degraded refresh, carry the previous catalog's good DB-backed fields forward."""
    if previous is None or not fresh.degraded:
        return fresh
    updates: Dict[str, Any] = {}
    if "trend_coverage" in fresh.degraded and "trend_coverage" not in previous.degraded:
        updates["trend_kpi_ids"] = previous.trend_kpi_ids
        updates["per_brand_only_trend_ids"] = previous.per_brand_only_trend_ids
    if "causal_outcomes" in fresh.degraded and "causal_outcomes" not in previous.degraded:
        updates["causal_outcomes"] = previous.causal_outcomes
    if not updates:
        return fresh
    still_degraded = tuple(d for d in fresh.degraded if d in previous.degraded)
    return dataclasses.replace(fresh, degraded=still_degraded, **updates)


class _CatalogCache:
    def __init__(self) -> None:
        self._catalog: Optional[CapabilityCatalog] = None

    async def get(
        self,
        *,
        now: Optional[float] = None,
        coverage_loader: Optional[CoverageLoader] = None,
        outcomes_loader: Optional[OutcomesLoader] = None,
    ) -> CapabilityCatalog:
        current = time.monotonic() if now is None else now
        cached = self._catalog
        if cached is not None:
            ttl = DEGRADED_TTL_SECONDS if cached.degraded else CATALOG_TTL_SECONDS
            if current - cached.loaded_at < ttl:
                return cached
        fresh = await build_capability_catalog(
            coverage_loader=coverage_loader, outcomes_loader=outcomes_loader
        )
        fresh = _keep_last_good_fields(fresh, cached)
        if now is not None:
            fresh = dataclasses.replace(fresh, loaded_at=now)
        self._catalog = fresh
        return fresh

    def reset(self) -> None:
        self._catalog = None


_cache = _CatalogCache()


async def get_capability_catalog() -> CapabilityCatalog:
    """The process-wide cached catalog (built lazily on first use)."""
    return await _cache.get()


def reset_capability_catalog_cache() -> None:
    """Test hook: forget the cached catalog."""
    _cache.reset()
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `$PY -m pytest tests/api/test_chat_capability_catalog.py -n 0 -q -p no:cacheprovider 2>&1 | tail -3`
Expected: `49 passed`

- [ ] **Step 5: Lint the module and commit**

Run: `/home/enunez/Projects/e2i_causal_analytics/.venv/bin/ruff check src/services/chat_capability_catalog.py tests/api/test_chat_capability_catalog.py && /home/enunez/Projects/e2i_causal_analytics/.venv/bin/ruff format --check src/services/chat_capability_catalog.py tests/api/test_chat_capability_catalog.py`
Expected: `All checks passed!` and `2 files already formatted`. If format fails, run `ruff format` on those two files and re-run the tests.

```bash
git branch --show-current
git add src/services/chat_capability_catalog.py tests/api/test_chat_capability_catalog.py
git commit -F - <<'EOF'
feat(chat): lazy TTL cache for the capability catalog

10-minute TTL, 60 s when a DB-backed field is degraded; a failed refresh
keeps the last good outcomes and trend coverage instead of an honest-but-
empty section.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01QCr4zzEYNQDCrLaM4gFHGr
EOF
```

---

### Task 6: Prompt template and `build_system_prompt` in chat.py (no handler change yet)

**Files:**
- Modify: `src/api/routes/chat.py:60-104` (the `_SYSTEM_PROMPT` constant) and its imports
- Test: `tests/api/test_chat_suggestions.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/api/test_chat_suggestions.py`:

```python
# =============================================================================
# PROMPT TEMPLATE (2026-09-05 capability catalog)
# =============================================================================

import asyncio

from src.services import chat_capability_catalog as catalog_module


async def _fake_coverage():
    return [
        {"kpi_id": "WS3-BI-005", "brand": "", "points": 24},
        {"kpi_id": "WS3-BI-007", "brand": "Kisqali", "points": 24},
    ]


async def _fake_outcomes():
    return ["persistent_180d", "treatment_initiated", "roi", "adopted"]


def make_fake_catalog():
    return asyncio.run(
        catalog_module.build_capability_catalog(
            coverage_loader=_fake_coverage, outcomes_loader=_fake_outcomes
        )
    )


def test_build_system_prompt_interpolates_catalog_and_route_hint():
    catalog = make_fake_catalog()
    prompt = chat_module.build_system_prompt(catalog, "/time-series")
    assert "{capability_catalog}" not in prompt and "{route_hint}" not in prompt
    assert "WHAT THE ASSISTANT CAN DO" in prompt
    assert "Total Prescriptions (TRx)" in prompt
    assert "persistent_180d" in prompt
    assert "The E2I system has" in prompt
    assert "PAGE HINT" in prompt
    assert catalog_module.ROUTE_HINTS["/time-series"] in prompt
    # the JSON output instruction survives the placeholder fill
    assert '{"suggestions": [{"title": "...", "message": "..."}, ...]}' in prompt


def test_build_system_prompt_omits_hint_block_for_unknown_page():
    catalog = make_fake_catalog()
    prompt = chat_module.build_system_prompt(catalog, "/nope")
    assert "PAGE HINT" not in prompt
    assert "\n\n\n" not in prompt


def test_prompt_tells_the_model_the_assistant_sees_page_content():
    """Part C of the design publishes page_content to the assistant, so the
    prompt must say pills MAY read on-screen values and must NOT extend them."""
    prompt = chat_module.build_system_prompt(make_fake_catalog(), "/")
    assert "ALSO shown to the assistant" in prompt
    assert "must NOT ask for anything beyond those literal values" in prompt
    assert "at least two different letters" in prompt
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `$PY -m pytest tests/api/test_chat_suggestions.py -n 0 -q -p no:cacheprovider 2>&1 | tail -3`
Expected: 3 failures, `AttributeError: module 'src.api.routes.chat' has no attribute 'build_system_prompt'`

- [ ] **Step 3: Replace the prompt constant and add the builder**

In `src/api/routes/chat.py`, add to the imports:

```python
from src.services.chat_capability_catalog import (
    CapabilityCatalog,
    filter_unsupported_pills,
    get_capability_catalog,
    render_catalog_block,
    route_hint,
)
```

Replace the whole `_SYSTEM_PROMPT = """..."""` block (lines 60-104) with:

```python
# The prompt is a TEMPLATE: {capability_catalog} and {route_hint} are filled per
# request by build_system_prompt() (str.replace, not str.format - the JSON
# example below has braces). Everything list-shaped comes from
# src.services.chat_capability_catalog, derived from code and data (#1638
# pattern); measured 2026-09-05, the one-sentence capability description this
# replaces let 42% of live pills ask for analyses no tool serves.
_SYSTEM_PROMPT = """You generate suggestion pills for the E2I Assistant, a pharmaceutical \
commercial-analytics chatbot (brands: Remibrutinib, Fabhalta, Kisqali). A pill is a question the \
analyst clicks; the assistant must be able to ANSWER it with the capabilities below. A pill that asks \
for an analysis, grain, axis or metric outside this catalog is a defect.

{capability_catalog}

ON-SCREEN CONTENT: page_content (when present) is ALSO shown to the assistant as on-screen context, \
so a pill MAY ask it to read, compare, rank or summarize values that appear literally in page_content \
(e.g. "Which of the on-screen SHAP features ranks highest for Fabhalta?"). A pill must NOT ask for \
anything beyond those literal values - no recomputing, validating, extending or explaining WHY an \
on-screen SHAP, optimizer, prediction, gap or CATE number is what it is (another segment, more \
features, per-territory detail, robustness, thresholds). Prefer turning an on-screen entity into a \
catalog analysis: an on-screen "sample_drop -> persistent_180d" effect becomes "What drives \
persistent_180d for Remibrutinib, and how confident are those paths?"; an on-screen SHAP feature \
specialty_hematology becomes "Which HCP specialties are most likely to increase Fabhalta \
prescriptions?"; an on-screen territory allocation becomes "What is Fabhalta's TRx by census region?".

{route_hint}

Input is JSON with the current page path, brand filter, page_content (may be empty) and the \
conversation so far (may be empty).

If the conversation is NON-EMPTY, propose exactly 4 follow-ups the analyst most likely wants next - \
deepen or branch from what was discussed, never repeat an answered question, stay with the entities \
in play (brand, KPI, outcome, segment, window).

If the conversation is EMPTY, propose exactly 4 openers grounded in page_content (name the specific \
brand / KPI / outcome / segment on screen); if page_content is empty, ground them in the PAGE HINT, \
the page path and the brand filter. Mix capability letters - the 4 pills must cover at least two \
different letters, and at most two pills may share a letter.

Rules:
- "title": at most 42 characters, imperative or noun phrase; MAY start with one emoji (📈 for a chart).
- "message": the full one-sentence question the pill sends.
- When brand_filter is set (not "All"), every "message" MUST name that brand explicitly, unless the \
pill is deliberately about another named brand or a cross-brand comparison. A pill click sends only \
the message text, and a brand-less question forces the assistant to ask which brand was meant.
- When numeric KPIs are on screen or were discussed, at least one pill asks to chart a trend or \
comparison. NEVER propose comparing, summing or ratio-ing two figures whose sources differ or are \
unstated (#1640): page_content marks each KPI "[from <tables>]" or "[source unstated]"; a trend of ONE \
figure is always safe, a comparison is safe only within one source.
- Before answering, check each pill against A-H and the NEVER list; replace any pill that fails.

Respond with JSON only, no prose: {"suggestions": [{"title": "...", "message": "..."}, ...]}"""


def build_system_prompt(catalog: CapabilityCatalog, page: Optional[str]) -> str:
    """Fill the prompt template with the rendered catalog and the page's route hint."""
    hint = route_hint(page)
    hint_block = (
        f"PAGE HINT (what this route shows and which catalog letters fit it): {hint}" if hint else ""
    )
    prompt = _SYSTEM_PROMPT.replace("{capability_catalog}", render_catalog_block(catalog))
    prompt = prompt.replace("{route_hint}", hint_block)
    while "\n\n\n" in prompt:
        prompt = prompt.replace("\n\n\n", "\n\n")
    return prompt
```

`filter_unsupported_pills` and `get_capability_catalog` are imported now but used in Task 8; ruff will flag them as unused (F401) until then, so for this commit import only `CapabilityCatalog`, `render_catalog_block`, `route_hint` and add the other two in Task 8.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `$PY -m pytest tests/api/test_chat_suggestions.py -n 0 -q -p no:cacheprovider 2>&1 | tail -3`
Expected: `21 passed` (18 existing, incl. `test_prompt_requires_brand_in_pill_messages_when_filter_set` against the template, plus 3 new)

- [ ] **Step 5: Commit**

```bash
git branch --show-current
git add src/api/routes/chat.py tests/api/test_chat_suggestions.py
git commit -F - <<'EOF'
feat(chat): pill prompt becomes a template filled with the capability catalog

build_system_prompt() interpolates the rendered catalog and the page's
route hint. The on-screen rule now tells the model the assistant ALSO sees
page_content (design part C) and may read but not extend it. The handler
still uses the old flow; wiring follows the re-measurement gate.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01QCr4zzEYNQDCrLaM4gFHGr
EOF
```

---

### Task 7: Re-measurement gate (cheapest disproof, faithful environment)

The one assumption the redesign adds over the measured prototype is the revised
on-screen rule. Measure it with the REAL catalog (real Supabase, real fast-tier
model, same 46 request shapes) before the production route changes.

**Files:**
- Create (untracked evidence): `/home/enunez/Projects/e2i_causal_analytics/docs/demos/results/2026-09-05_pill_suggestions_review/proto_v3.py`
- Output: `prototype_v3_pills.json`, `prototype_v3_system_prompt.txt`, `v3_grades.md` in the same directory

- [ ] **Step 1: Write the harness**

Create `proto_v3.py` in the evidence directory:

```python
"""Re-measure the pill prompt with the REAL capability catalog (design 2026-09-05, gate 5.1).

Faithful: same get_fast_llm() as the route, real Supabase for the catalog,
the 46 live request shapes from the baseline sample. Imports the WORKTREE
code (sys.path insert before any src import).
"""

import asyncio
import json
import os
import sys
import time

from dotenv import load_dotenv

load_dotenv("/home/enunez/Projects/e2i_causal_analytics/.env")
WT = "/home/enunez/Projects/e2i_causal_analytics/.worktrees/pill-catalog"
sys.path.insert(0, WT)

from langchain_core.messages import HumanMessage, SystemMessage  # noqa: E402

from src.api.routes.chat import _parse_suggestions, build_system_prompt  # noqa: E402
from src.services.chat_capability_catalog import (  # noqa: E402
    filter_unsupported_pills,
    get_capability_catalog,
)
from src.utils.llm_content import normalize_llm_content  # noqa: E402
from src.utils.llm_factory import get_fast_llm  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sample = json.load(open(os.path.join(HERE, "live_pills_baseline_2026-09-05.json")))
scenarios = [r for r in sample if r["label"].startswith(("opener1", "turn1"))]


async def main() -> None:
    catalog = await get_capability_catalog()
    print("degraded:", catalog.degraded)
    print("outcomes:", catalog.causal_outcomes)
    print("trend ids:", sorted(catalog.trend_kpi_ids), "per-brand-only:", sorted(catalog.per_brand_only_trend_ids))
    assert not catalog.degraded, "a faithful run needs the real catalog; fix the DB access first"

    llm = get_fast_llm(max_tokens=600, timeout=25)
    try:
        llm.callbacks = []  # no prod usage-telemetry rows from a scratch experiment
    except Exception as exc:  # noqa: BLE001
        print("callbacks not cleared:", exc)
    print("model:", getattr(llm, "model", None) or getattr(llm, "model_name", None))

    out = []
    for rec in scenarios:
        body = rec["body"]
        ctx = {
            "page": body["page"],
            "brand_filter": body.get("brand", ""),
            "page_content": body.get("page_context", ""),
            "conversation": body["messages"],
        }
        prompt = build_system_prompt(catalog, body["page"])
        t = time.time()
        reply = await llm.ainvoke(
            [SystemMessage(content=prompt), HumanMessage(content=json.dumps(ctx, ensure_ascii=False))]
        )
        ms = int((time.time() - t) * 1000)
        try:
            pills = _parse_suggestions(normalize_llm_content(reply.content))
            kept, dropped = filter_unsupported_pills(pills, catalog)
            row = {
                "label": rec["label"],
                "ms": ms,
                "pills": [p.model_dump() for p in kept],
                "dropped": [{"rule": rule, **p.model_dump()} for p, rule in dropped],
            }
        except ValueError as exc:
            row = {"label": rec["label"], "ms": ms, "parse_fail": str(exc)}
        out.append(row)
        print(f"{rec['label']:<36} {ms}ms kept={len(row.get('pills', []))} dropped={len(row.get('dropped', []))}")

    json.dump(out, open(os.path.join(HERE, "prototype_v3_pills.json"), "w"), indent=1, ensure_ascii=False)
    open(os.path.join(HERE, "prototype_v3_system_prompt.txt"), "w").write(build_system_prompt(catalog, "/"))
    print("\n=== V3 PILLS ===")
    for row in out:
        print("##", row["label"])
        for p in row.get("pills", []):
            print("  -", p["title"], "::", p["message"])
        for d in row.get("dropped", []):
            print("  x", d["rule"], "::", d["message"])


asyncio.run(main())
```

- [ ] **Step 2: Run it**

Run (from the worktree root; the script inserts the worktree on `sys.path` itself):

```bash
$PY /home/enunez/Projects/e2i_causal_analytics/docs/demos/results/2026-09-05_pill_suggestions_review/proto_v3.py 2>&1 | tee /home/enunez/Projects/e2i_causal_analytics/docs/demos/results/2026-09-05_pill_suggestions_review/proto_v3.log | tail -80
```

Expected: `degraded: ()`, 14 outcomes printed, ~23 scenario lines with latency 2-4 s each, then the pill listing. If `degraded` is non-empty, stop and fix the DB access (the gate must run against the real catalog).

- [ ] **Step 3: Grade every pill**

Rubric (same as the 2026-09-05 baseline and v2 grading):
- **OK**: answerable as phrased by one catalog capability (a bound tool, `renderKpiTrend`/`renderChart`, a dashboard action, or reading a value literally present in that scenario's `page_context`).
- **PARTIAL**: answerable with a caveat or only in part (e.g. asks for a region scope the causal registry ignores; the assistant would answer brand-level and say so).
- **NO**: needs a tool, dimension or recomputation that does not exist.

Write `v3_grades.md` in the evidence directory with: a table `scenario | OK | PARTIAL | NO | notes`, totals overall, totals for scenarios whose body has `page_context`, totals for those without, totals for `turn1` follow-ups, the count of validator drops by rule, the distinct first-pill titles across the no-context openers, and median/max latency.

- [ ] **Step 4: Apply the gate**

Pass iff ALL of: NO <= 10% overall; NO <= 10% on `page_context` scenarios; at least 3 distinct lead-pill titles across the no-context openers; zero `parse_fail` rows.

If the gate fails: adjust the prompt text in `src/api/routes/chat.py` (`_SYSTEM_PROMPT`) or the catalog wording in `render_catalog_block`, re-run Step 2 and Step 3, and record each iteration in `v3_grades.md` (v3a, v3b, ...). Do not proceed to Task 8 until it passes. If it cannot pass in three iterations, stop and report to the user with the numbers.

- [ ] **Step 5: Record the gate result in the repo**

Append a short "Gate 5.1 result" paragraph to the spec's section 5 (`docs/superpowers/specs/2026-09-05-copilot-pill-capability-catalog-design.md`) with the four numbers and the evidence file names, then commit:

```bash
git branch --show-current
git add docs/superpowers/specs/2026-09-05-copilot-pill-capability-catalog-design.md src/api/routes/chat.py src/services/chat_capability_catalog.py
git commit -F - <<'EOF'
docs(specs): record the pill prompt re-measurement gate result

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01QCr4zzEYNQDCrLaM4gFHGr
EOF
```

(`git add` of the two source files is a no-op when the gate passed first time.)

---

### Task 8: Wire the route: catalog, validator, drop logging

**Files:**
- Modify: `src/api/routes/chat.py` (module docstring, handler `generate_chat_suggestions`)
- Test: `tests/api/test_chat_suggestions.py`

- [ ] **Step 1: Write the failing tests and the autouse catalog fake**

In `tests/api/test_chat_suggestions.py`, right after the `_good_reply` helper (before the `# ROUTE CONTRACT` banner), add an autouse fixture so every route test uses a fake catalog and never touches the DB:

```python
@pytest.fixture(autouse=True)
def _fake_catalog(monkeypatch):
    """Route tests never build the real catalog (Supabase). The fake is built
    with the module's own builder and injected loaders."""
    catalog = make_fake_catalog()

    async def _get():
        return catalog

    monkeypatch.setattr(chat_module, "get_capability_catalog", _get)
    return catalog
```

(`make_fake_catalog` is defined lower in the file from Task 6; move the `import asyncio`, `catalog_module` import, `_fake_coverage`, `_fake_outcomes` and `make_fake_catalog` definitions up above this fixture so they are defined before use.)

Then append these route tests at the end of the file:

```python
# =============================================================================
# ROUTE: catalog in the prompt, validator on the output
# =============================================================================


def test_llm_receives_catalog_and_route_hint(test_client, auth_headers, monkeypatch):
    fake = _FakeLLM(content=_good_reply(4))
    monkeypatch.setattr(chat_module, "get_fast_llm", lambda **kwargs: fake)

    resp = test_client.post(
        "/api/chat/suggestions", json=_payload(page="/time-series"), headers=auth_headers
    )

    assert resp.status_code == 200
    system = fake.calls[0][0].content
    assert "WHAT THE ASSISTANT CAN DO" in system
    assert "Total Prescriptions (TRx)" in system
    assert "persistent_180d" in system
    assert "PAGE HINT" in system and "Time Series:" in system
    assert "{capability_catalog}" not in system


def test_unsupported_pills_are_dropped_and_logged(test_client, auth_headers, monkeypatch, caplog):
    reply = json.dumps(
        {
            "suggestions": [
                {"title": "TRx trend", "message": "Chart the TRx trend for Kisqali."},
                {"title": "T-114", "message": "Why did territory T-114 gain field force for Kisqali?"},
                {"title": "Drivers", "message": "What drives persistent_180d for Kisqali?"},
                {"title": "Persistence rate", "message": "Chart the persistent_180d rate for Kisqali by region."},
            ]
        }
    )
    fake = _FakeLLM(content=reply)
    monkeypatch.setattr(chat_module, "get_fast_llm", lambda **kwargs: fake)

    with caplog.at_level("INFO", logger="src.api.routes.chat"):
        resp = test_client.post("/api/chat/suggestions", json=_payload(), headers=auth_headers)

    assert resp.status_code == 200
    assert [s["title"] for s in resp.json()["suggestions"]] == ["TRx trend", "Drivers"]
    dropped = [r.message for r in caplog.records if "chat suggestion dropped" in r.message]
    assert len(dropped) == 2
    assert any("rule=territory_detail" in m for m in dropped)
    assert any("rule=outcome_as_kpi:persistent_180d" in m for m in dropped)


def test_all_pills_dropped_returns_502(test_client, auth_headers, monkeypatch):
    reply = json.dumps(
        {"suggestions": [{"title": "SHAP", "message": "Which SHAP features drive Kisqali adoption?"}]}
    )
    monkeypatch.setattr(chat_module, "get_fast_llm", lambda **kwargs: _FakeLLM(content=reply))

    resp = test_client.post("/api/chat/suggestions", json=_payload(), headers=auth_headers)

    assert resp.status_code == 502
    assert "no supported pills" in resp.json()["detail"]


def test_fast_llm_gets_600_tokens(test_client, auth_headers, monkeypatch):
    seen = {}

    def _factory(**kwargs):
        seen.update(kwargs)
        return _FakeLLM(content=_good_reply(2))

    monkeypatch.setattr(chat_module, "get_fast_llm", _factory)
    test_client.post("/api/chat/suggestions", json=_payload(), headers=auth_headers)
    assert seen == {"max_tokens": 600, "timeout": 8}
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `$PY -m pytest tests/api/test_chat_suggestions.py -n 0 -q -p no:cacheprovider 2>&1 | tail -3`
Expected: the 4 new tests fail (`WHAT THE ASSISTANT CAN DO` not in the old prompt; dropped pills still returned; 200 instead of 502; `max_tokens` 500).

- [ ] **Step 3: Rewire the handler**

In `src/api/routes/chat.py`: extend the catalog import to include `filter_unsupported_pills` and `get_capability_catalog`; replace the module docstring's last paragraph ("Suggestion topics are constrained ...") with:

```python
Suggestion topics are constrained to what the chatbot's bound tools
(``E2I_CHATBOT_TOOLS`` in ``chatbot_tools.py``) and generative-UI actions can
actually answer. Since 2026-09-05 that constraint is a capability CATALOG
interpolated into the prompt from code and data
(``src.services.chat_capability_catalog``: KPI registry, history coverage,
causal outcomes, agent roster) plus a narrow deterministic post-filter that
drops the pill families measured unanswerable (SHAP recomputation, territory
detail, per-patient predictions, causal outcomes used as KPIs, ...). Drops are
logged at INFO with their rule so the production drop rate is measurable.
```

Replace the body of `generate_chat_suggestions` from `context = {` to the end with:

```python
    context = {
        "page": payload.page or "/",
        "brand_filter": payload.brand or "",
        "page_content": payload.page_context or "",
        "conversation": [{"role": m.role, "content": m.content} for m in payload.messages],
    }
    catalog = await get_capability_catalog()
    system_prompt = build_system_prompt(catalog, payload.page)
    # 600 (was 500): catalog-grounded pill messages name a brand and an axis and
    # run a little longer; measured 2.0-3.3 s against the 8 s timeout.
    llm = get_fast_llm(max_tokens=600, timeout=8)
    try:
        reply = await llm.ainvoke(
            [
                SystemMessage(content=system_prompt),
                HumanMessage(content=json.dumps(context, ensure_ascii=False)),
            ]
        )
    except Exception as exc:
        logger.warning("chat suggestion LLM call failed: %s", exc)
        raise HTTPException(status_code=502, detail="suggestion generation failed") from exc

    try:
        # AIMessage.content is str | list of content blocks (#1350)
        suggestions = _parse_suggestions(normalize_llm_content(reply.content))
    except ValueError as exc:
        logger.warning("chat suggestion reply unusable: %s", exc)
        raise HTTPException(
            status_code=502, detail="suggestion generation returned no usable pills"
        ) from exc

    kept, dropped = filter_unsupported_pills(suggestions, catalog)
    for pill, rule in dropped:
        # INFO, not DEBUG: the drop rate is the production measurement of how
        # often the prompt still proposes an unanswerable pill.
        logger.info(
            "chat suggestion dropped rule=%s page=%s title=%r",
            rule,
            payload.page or "/",
            pill.title,
        )
    if not kept:
        raise HTTPException(
            status_code=502, detail="suggestion generation returned no supported pills"
        )
    return SuggestionsResponse(suggestions=kept)
```

Also update the route decorator's `description=` string: after "502 on any generation/parsing failure" add " or when the post-filter drops every pill".

- [ ] **Step 4: Run the tests to verify they pass**

Run: `$PY -m pytest tests/api/test_chat_suggestions.py tests/api/test_chat_capability_catalog.py -n 0 -q -p no:cacheprovider 2>&1 | tail -3`
Expected: `74 passed` (25 route + 49 catalog)

- [ ] **Step 5: Lint and commit**

Run: `/home/enunez/Projects/e2i_causal_analytics/.venv/bin/ruff check src/api/routes/chat.py tests/api/test_chat_suggestions.py && /home/enunez/Projects/e2i_causal_analytics/.venv/bin/ruff format --check src/api/routes/chat.py tests/api/test_chat_suggestions.py`
Expected: `All checks passed!` / `2 files already formatted`

```bash
git branch --show-current
git add src/api/routes/chat.py tests/api/test_chat_suggestions.py
git commit -F - <<'EOF'
feat(chat): ground suggestion pills in the capability catalog and post-filter them

POST /api/chat/suggestions fills the prompt template with the cached
catalog and the page's route hint, drops pills the deterministic validator
flags (logged at INFO with the rule), and 502s only when nothing survives
so the frontend keeps its static fallback.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01QCr4zzEYNQDCrLaM4gFHGr
EOF
```

---

### Task 9: Readable note wording covers prose page summaries

**Files:**
- Modify: `src/api/routes/copilotkit.py:728-745` (`_readables_context_note` return string)
- Test: `tests/api/` (locate with the grep below)

- [ ] **Step 1: Find any test that pins the old wording**

Run: `grep -rn "values are JSON\|ON-SCREEN APP CONTEXT" tests/ | head`
Expected: zero or a few hits. For each hit that asserts the exact phrase `values are JSON`, change the asserted phrase to `values are JSON or short prose summaries` in the same test. If there are no hits, add one test to `tests/api/test_copilotkit_readables.py` if that file exists, else create `tests/api/test_copilotkit_readables_note.py`:

```python
"""_readables_context_note wording: readables can be JSON or prose page summaries (2026-09-05)."""

from src.api.routes.copilotkit import _readables_context_note


def test_note_renders_prose_summary_and_says_how_to_treat_it():
    state = {
        "copilotkit": {
            "context": [
                {
                    "description": "Summary of the data currently visible on the page",
                    "value": "Home dashboard. Brand filter: Kisqali; region: All US.",
                }
            ]
        }
    }
    note = _readables_context_note(state["copilotkit"])
    assert "Home dashboard. Brand filter: Kisqali" in note
    assert "values are JSON or short prose summaries" in note
    assert "not a data table" in note


def test_note_is_empty_without_readables():
    assert _readables_context_note({"context": []}) == ""
    assert _readables_context_note(None) == ""
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `$PY -m pytest tests/api/test_copilotkit_readables_note.py -n 0 -q -p no:cacheprovider 2>&1 | tail -3` (use the file you edited/created)
Expected: 1 failure on `values are JSON or short prose summaries`

- [ ] **Step 3: Change the wording**

In `_readables_context_note`, replace the return statement's two literal fragments:

```python
    return (
        "\n\nON-SCREEN APP CONTEXT (what the user is currently looking at, shared "
        "live by the dashboard via AG-UI readables; values are JSON or short prose "
        "summaries):\n"
        + "\n".join(lines)
        + "\n\nWhen the user asks about 'the data on the page/screen/GUI', 'these "
        "results', or the analysis they are viewing, answer from this context "
        "first — compute counts, ranks and percentages directly from it (a "
        "histogram's bin_counts cover the FULL scored cohort; top_rows are only "
        "the rows shown on screen) and say which on-screen values you used. "
        "A prose page summary is a description of what the page shows, not a "
        "data table: cite it as on-screen context and never present its figures "
        "as the result of a tool you ran. "
        "Call tools only for data that is not on screen."
    )
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `$PY -m pytest tests/api/test_copilotkit_readables_note.py -n 0 -q -p no:cacheprovider 2>&1 | tail -3`
Expected: `2 passed`. Then run the existing copilotkit test files that mention readables so nothing else pinned the old phrase: `$PY -m pytest $(grep -rl "_readables_context_note" tests/ | tr '\n' ' ') -n 0 -q -p no:cacheprovider 2>&1 | tail -3` — expected all passed.

- [ ] **Step 5: Commit**

```bash
git branch --show-current
git add src/api/routes/copilotkit.py tests/api/
git commit -F - <<'EOF'
fix(copilotkit): on-screen context note covers prose page summaries

The dashboard will publish each page's pageChatSummary as a readable; the
note now says values may be prose and that a summary is a description to
cite, not a tool result.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01QCr4zzEYNQDCrLaM4gFHGr
EOF
```

---

### Task 10: Publish the page summary as a fifth readable (frontend)

**Files:**
- Modify: `frontend/src/providers/E2ICopilotProvider.tsx:740-766` (`CopilotHooksInner` readables)
- Test: `frontend/src/providers/E2ICopilotProvider.test.tsx:430-471`

- [ ] **Step 1: Update the readable-count test and add the summary test**

In `frontend/src/providers/E2ICopilotProvider.test.tsx`, add `usePageChatContext` to the `import { ... } from './E2ICopilotProvider'` list. In the test `registers readables when CopilotKit is enabled`, change:

```ts
    // Should register 4 readables: filters, page context, agents, preferences
    expect(mockUseCopilotReadable).toHaveBeenCalledTimes(4);
```

to:

```ts
    // Should register 5 readables: filters, page path, agents, preferences,
    // on-screen page summary (2026-09-05)
    expect(mockUseCopilotReadable).toHaveBeenCalledTimes(5);
```

Then add, inside the same `describe('CopilotHooksConnector', ...)` block after that test:

```tsx
  it('publishes the page summary as a readable only when a page has published one', () => {
    const Publisher: React.FC<{ summary: string | null }> = ({ summary }) => {
      usePageChatContext(summary);
      return null;
    };
    const summaryCalls = () =>
      mockUseCopilotReadable.mock.calls.filter((call) =>
        call[0]?.description?.includes('visible on the page')
      );
    const lastSummaryCall = () => {
      const calls = summaryCalls();
      return calls[calls.length - 1][0];
    };

    const { rerender } = render(
      <CopilotKitWrapper enabled={true}>
        <E2ICopilotProvider>
          <Publisher summary={null} />
          <TestConsumer />
        </E2ICopilotProvider>
      </CopilotKitWrapper>
    );

    // Nothing published: the readable is registered but DISABLED, so the
    // agent prompt stays byte-identical to before on such pages.
    expect(summaryCalls().length).toBeGreaterThan(0);
    expect(lastSummaryCall().available).toBe('disabled');
    expect(lastSummaryCall().value).toBe('');

    rerender(
      <CopilotKitWrapper enabled={true}>
        <E2ICopilotProvider>
          <Publisher summary="Home dashboard. Brand filter: Kisqali; region: All US." />
          <TestConsumer />
        </E2ICopilotProvider>
      </CopilotKitWrapper>
    );

    // Published: enabled, carrying the exact string the pill endpoint receives,
    // sent as raw prose (not JSON.stringify'd with quotes and \n escapes).
    expect(lastSummaryCall().available).toBe('enabled');
    expect(lastSummaryCall().value).toBe('Home dashboard. Brand filter: Kisqali; region: All US.');
    expect(lastSummaryCall().convert('ignored', 'raw text')).toBe('raw text');
  });
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cd frontend && npx vitest run src/providers/E2ICopilotProvider.test.tsx 2>&1 | tail -15`
Expected: 2 failures (`toHaveBeenCalledTimes(5)` received 4; `summaryCalls().length` is 0).

- [ ] **Step 3: Add the readable**

In `frontend/src/providers/E2ICopilotProvider.tsx`, change the readables banner comment from `(4 readables)` to `(5 readables)` and add after the preferences readable (after `// 4. User preferences ... });`):

```tsx
  // 5. On-screen page summary (2026-09-05 pill review). Eight pages publish a
  // prose summary of what they show via usePageChatContext; until now it
  // reached ONLY the suggestion-pill endpoint, so pills asked about SHAP
  // features and gap sizes the agent could not see. Publishing the SAME
  // string here keeps the two channels identical by construction. Disabled
  // (not sent) when nothing is published, so the agent prompt is unchanged
  // on those pages. convert passes the prose through instead of the default
  // JSON.stringify, which would wrap it in quotes and escape newlines.
  const pageSummary = context?.pageChatContext ?? '';
  useCopilotReadable({
    description:
      'Summary of the data currently visible on the page, as published by the page itself (prose; same text used to ground suggestion pills)',
    value: pageSummary,
    convert: passThroughText,
    available: pageSummary ? 'enabled' : 'disabled',
  });
```

Add this module-level helper next to `VALID_BRANDS` (above `CopilotHooksConnector`):

```ts
// Readable converter for prose values: the SDK default is JSON.stringify.
const passThroughText = (_description: string, value: unknown): string => String(value);
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cd frontend && npx vitest run src/providers/E2ICopilotProvider.test.tsx 2>&1 | tail -6`
Expected: `73 passed` (72 existing + 1)

- [ ] **Step 5: Commit**

```bash
git branch --show-current
git add frontend/src/providers/E2ICopilotProvider.tsx frontend/src/providers/E2ICopilotProvider.test.tsx
git commit -F - <<'EOF'
feat(frontend): publish the page's chat summary to the agent as a readable

The same pageChatSummary string the suggestion-pill endpoint receives now
reaches the agent's ON-SCREEN APP CONTEXT; disabled when a page publishes
nothing so those prompts are unchanged.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01QCr4zzEYNQDCrLaM4gFHGr
EOF
```

---

### Task 11: Top the pill row back up to four (frontend)

**Files:**
- Modify: `frontend/src/components/chat/E2IChatSidebar.tsx:76` (type export), `:79-111` (doc comment), `:349` (pill memo), new helper after `buildChatSuggestions`
- Test: `frontend/src/components/chat/E2IChatSidebar.suggestions.test.tsx`

- [ ] **Step 1: Write the failing tests**

In `frontend/src/components/chat/E2IChatSidebar.suggestions.test.tsx`, change the import to `import { buildChatSuggestions, topUpChatSuggestions } from './E2IChatSidebar';` and append:

```ts
describe('topUpChatSuggestions (2026-09-05 validator top-up)', () => {
  it('returns the static pills when no adaptive pills exist', () => {
    expect(topUpChatSuggestions(null, '/', 'Kisqali')).toEqual(buildChatSuggestions('/', 'Kisqali'));
    expect(topUpChatSuggestions([], '/', 'Kisqali')).toEqual(buildChatSuggestions('/', 'Kisqali'));
  });

  it('fills up to four with static pills, adaptive first, no duplicate titles', () => {
    const adaptive = [
      { title: '📈 Chart the TRx trend', message: 'Chart the TRx trend for Kisqali' },
      { title: 'Persistence drivers', message: 'What drives persistent_180d for Kisqali?' },
    ];
    const pills = topUpChatSuggestions(adaptive, '/', 'Kisqali');

    expect(pills).toHaveLength(4);
    expect(pills.slice(0, 2)).toEqual(adaptive);
    const titles = pills.map((p) => p.title.toLowerCase());
    expect(new Set(titles).size).toBe(4);
    // the duplicated static "Chart the TRx trend" was skipped, the others filled in
    expect(pills.some((p) => p.title.includes('Kisqali market share'))).toBe(true);
  });

  it('never exceeds four and keeps a full adaptive set untouched', () => {
    const adaptive = [1, 2, 3, 4, 5].map((i) => ({ title: `Pill ${i}`, message: `Question ${i}?` }));
    expect(topUpChatSuggestions(adaptive, '/', 'All')).toEqual(adaptive.slice(0, 4));
  });
});
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cd frontend && npx vitest run src/components/chat/E2IChatSidebar.suggestions.test.tsx 2>&1 | tail -8`
Expected: 3 failures, `topUpChatSuggestions is not a function`

- [ ] **Step 3: Add the helper and use it**

In `frontend/src/components/chat/E2IChatSidebar.tsx`:

1. Line 76: `type ChatSuggestion = ...` becomes `export type ChatSuggestion = { title: string; message: string };`
2. After the `buildChatSuggestions` function, add:

```ts
/**
 * Top the adaptive pills back up to four with the static route+brand set.
 *
 * The backend post-filters generated pills against the assistant's
 * capability catalog (2026-09-05) and may return fewer than four; the static
 * pills are the guaranteed floor, so they fill the gap. Adaptive pills come
 * first, duplicates (case-insensitive title) are skipped, never more than four.
 * Exported for tests.
 */
export function topUpChatSuggestions(
  adaptive: ChatSuggestion[] | null,
  pathname: string,
  brand: E2IFilters['brand']
): ChatSuggestion[] {
  const statics = buildChatSuggestions(pathname, brand);
  if (!adaptive || adaptive.length === 0) return statics;
  const out = adaptive.slice(0, 4);
  const seen = new Set(out.map((p) => p.title.trim().toLowerCase()));
  for (const pill of statics) {
    if (out.length >= 4) break;
    const key = pill.title.trim().toLowerCase();
    if (seen.has(key)) continue;
    seen.add(key);
    out.push(pill);
  }
  return out;
}
```

3. The pill memo (line ~349) becomes:

```ts
  const chatSuggestions = React.useMemo(() => {
    return topUpChatSuggestions(pillState.adaptive, pathname, filters.brand);
  }, [pillState, pathname, filters.brand]);
```

4. In the architecture doc comment (lines 79-111), replace tier 3's text:

```
 * 3. STATIC FALLBACK: the route+brand template set below shows instantly
 *    while a generation is in flight and whenever generation fails (502) —
 *    never a blank pill row, never invented output.
```

with:

```
 * 3. STATIC FALLBACK + TOP-UP: the route+brand template set below shows
 *    instantly while a generation is in flight and whenever generation fails
 *    (502) — never a blank pill row, never invented output. Since 2026-09-05
 *    the backend post-filters generated pills against the assistant's
 *    capability catalog and may return fewer than four; topUpChatSuggestions
 *    fills the row back up from the same static set.
```

and replace the final paragraph of that comment:

```
 * Keep fallback pill topics inside what the bound backend tools can actually
 * answer (KPIs, causal paths, agents — see E2I_CHATBOT_TOOLS in
 * chatbot_tools.py); the chart pills route through the renderKpiTrend /
 * renderChart generative-UI actions so users discover inline visuals.
```

with:

```
 * Keep fallback pill topics inside what the bound backend tools can actually
 * answer (KPIs, causal paths, agents — see E2I_CHATBOT_TOOLS in
 * chatbot_tools.py and the capability catalog in
 * src/services/chat_capability_catalog.py); the chart pills route through the
 * renderKpiTrend / renderChart generative-UI actions so users discover inline
 * visuals. The page summary a page publishes via usePageChatContext reaches
 * BOTH the pill endpoint (page_context) and the agent (readable #5 in
 * E2ICopilotProvider), so pills may refer to on-screen values.
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cd frontend && npx vitest run src/components/chat/E2IChatSidebar.suggestions.test.tsx src/components/chat/ 2>&1 | tail -6`
Expected: all passed (5 in the suggestions file plus any other chat tests).

- [ ] **Step 5: Commit**

```bash
git branch --show-current
git add frontend/src/components/chat/E2IChatSidebar.tsx frontend/src/components/chat/E2IChatSidebar.suggestions.test.tsx
git commit -F - <<'EOF'
feat(frontend): top the suggestion pill row back up to four with static pills

The backend now drops unsupported pills; the sidebar fills the gap from
its static route+brand set, adaptive first, no duplicate titles.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01QCr4zzEYNQDCrLaM4gFHGr
EOF
```

---

### Task 12: Static checks on the changed files only

**Files:** none new; verification only.

- [ ] **Step 1: Python lint and format**

Run:

```bash
/home/enunez/Projects/e2i_causal_analytics/.venv/bin/ruff check src/services/chat_capability_catalog.py src/api/routes/chat.py src/api/routes/copilotkit.py tests/api/test_chat_capability_catalog.py tests/api/test_chat_suggestions.py
/home/enunez/Projects/e2i_causal_analytics/.venv/bin/ruff format --check src/services/chat_capability_catalog.py src/api/routes/chat.py src/api/routes/copilotkit.py tests/api/test_chat_capability_catalog.py tests/api/test_chat_suggestions.py
```

Expected: `All checks passed!` and `5 files already formatted`. Fix and re-run tests if not.

- [ ] **Step 2: Scoped mypy on the two touched modules only**

Run: `/home/enunez/Projects/e2i_causal_analytics/.venv/bin/mypy --config-file pyproject.toml src/services/chat_capability_catalog.py src/api/routes/chat.py 2>&1 | tail -5`
Expected: `Success: no issues found in 2 source files`. This is scoped (two files); never run whole-tree mypy on this box. If it exceeds ~3 minutes or the box swaps, kill it and rely on CI's `Type Check (MyPy)` gate.

- [ ] **Step 3: Frontend typecheck and lint**

Run:

```bash
cd frontend && npm run typecheck 2>&1 | tail -5
cd frontend && npx eslint src/providers/E2ICopilotProvider.tsx src/providers/E2ICopilotProvider.test.tsx src/components/chat/E2IChatSidebar.tsx src/components/chat/E2IChatSidebar.suggestions.test.tsx 2>&1 | tail -5
```

Expected: typecheck exits 0 with no errors; eslint prints nothing.

- [ ] **Step 4: Full targeted backend run**

Run: `$PY -m pytest tests/api/test_chat_capability_catalog.py tests/api/test_chat_suggestions.py $(grep -rl "_readables_context_note" tests/ | tr '\n' ' ') -n 0 -q -p no:cacheprovider 2>&1 | tail -3`
Expected: all passed, 0 failed.

- [ ] **Step 5: Commit any fixes**

Only if Steps 1-3 changed files:

```bash
git branch --show-current
git add -u
git commit -F - <<'EOF'
chore(chat): lint and type fixes for the capability catalog

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01QCr4zzEYNQDCrLaM4gFHGr
EOF
```

---

### Task 13: Spec status, push, pull request

**Files:**
- Modify: `docs/superpowers/specs/2026-09-05-copilot-pill-capability-catalog-design.md:4` (status line)

- [ ] **Step 1: Mark the spec approved**

Change the status line to: `**Status:** approved by the user 2026-09-05; implemented on branch claude/copilot-pill-capability-catalog`

- [ ] **Step 2: Rebase onto current main and re-run the targeted tests**

The peer session is merging PR #1899 to main during this work.

```bash
git fetch -q origin main && git rebase origin/main
$PY -m pytest tests/api/test_chat_capability_catalog.py tests/api/test_chat_suggestions.py -n 0 -q -p no:cacheprovider 2>&1 | tail -2
cd frontend && npx vitest run src/providers/E2ICopilotProvider.test.tsx src/components/chat/E2IChatSidebar.suggestions.test.tsx 2>&1 | tail -4
```

Expected: rebase clean (if a conflict appears in `copilotkit.py` or the provider, resolve keeping both sides' intent and re-run); all tests passed.

- [ ] **Step 3: Commit the spec status and push**

```bash
git branch --show-current
git add docs/superpowers/specs/2026-09-05-copilot-pill-capability-catalog-design.md docs/superpowers/plans/2026-09-05-copilot-pill-capability-catalog.md
git commit -F - <<'EOF'
docs: mark the pill capability catalog spec approved and add the plan

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01QCr4zzEYNQDCrLaM4gFHGr
EOF
git push -u origin claude/copilot-pill-capability-catalog
```

- [ ] **Step 4: Open the PR**

```bash
gh pr create --base main --head claude/copilot-pill-capability-catalog \
  --title "feat(chat): capability-catalog suggestion pills + page summary readable" \
  --body-file - <<'EOF'
## Why

Live sample 2026-09-05 (46 calls, 92 pills): **42%** of the sidebar's suggestion pills asked for analyses the assistant cannot deliver (63% on pages that publish a page summary). Root causes: a one-sentence capability description in the pill prompt, and a context asymmetry (the pill generator saw page summaries the agent never received). Faithful prototype with a code-derived catalog: **9%**.

Design: `docs/superpowers/specs/2026-09-05-copilot-pill-capability-catalog-design.md`
Plan: `docs/superpowers/plans/2026-09-05-copilot-pill-capability-catalog.md`
Evidence (untracked, on the droplet): `docs/demos/results/2026-09-05_pill_suggestions_review/`

## What

- `src/services/chat_capability_catalog.py` (new): `CapabilityCatalog` built from the KPI registry, `v_kpi_history_coverage`, the segmented-history families, the causal-path registry's distinct outcomes and the agent roster (#1638 pattern, nothing transcribed); rendered as prompt sections A-H + NEVER list; lazy 10-min TTL cache with honest degraded fallbacks; per-route hints; deterministic validator for the two pill families measured NO.
- `POST /api/chat/suggestions`: prompt is a template filled with the catalog and route hint; validator drops flagged pills (INFO log with rule); 502 only when nothing survives.
- Frontend: fifth CopilotKit readable carries the page's `pageChatSummary` to the agent (disabled when a page publishes nothing); the sidebar tops the pill row back up to four from its static set.
- `_readables_context_note`: wording covers prose summaries.

## Gate (re-measured with the real catalog before wiring)

<fill from v3_grades.md: NO % overall / on page_context pages; distinct lead pills on no-context pages; latency>

## Tests

- 49 unit tests for the catalog module (registry-derived names, degraded rendering, TTL/last-good cache, route hints, validator fixtures taken from the live NO pills)
- 7 new route tests (catalog in prompt, drops + logging, all-dropped 502, token budget)
- Frontend: readable enabled/disabled + raw prose convert; top-up helper

🤖 Generated with [Claude Code](https://claude.com/claude-code)

https://claude.ai/code/session_01QCr4zzEYNQDCrLaM4gFHGr
EOF
```

Replace the `<fill from v3_grades.md ...>` line with the actual numbers BEFORE running the command (do not open a PR with a placeholder). Record the PR number.

---

### Task 14: CI and independent review

- [ ] **Step 1: Watch CI (bounded polls)**

```bash
gh pr checks <PR#> 2>&1 | tail -20
```

Repeat every 3-5 minutes (each foreground call under 9.5 minutes; never a background sleep loop on this box) until every check is `pass` or `fail`. If a check fails, read the job log via the REST endpoint (`gh run view --log` can print 0 lines on a real failure): `gh api repos/enunezvn/e2i_causal_analytics/actions/jobs/<job_id>/logs | tail -80`. Fix, commit, push, re-watch. The `Type Check (MyPy)` job is a ceiling gate: read its `mypy-report` artifact for the actual errors.

- [ ] **Step 2: Codex audit of the diff**

Run from the worktree root (`< /dev/null` is mandatory; codex blocks on stdin; the verdict is in the FINAL codex block only):

```bash
git diff origin/main...HEAD --stat > /tmp/pill_catalog_diff_stat.txt
codex exec -m gpt-5.5 "You are auditing PR <PR#> on branch claude/copilot-pill-capability-catalog of this repo (cwd). Read docs/superpowers/specs/2026-09-05-copilot-pill-capability-catalog-design.md, then review the diff against origin/main (git diff origin/main...HEAD). Focus: (1) does the validator over-block legitimate pills (list any regex that could match an answerable question); (2) does the catalog invent any capability the bound tools in src/api/routes/chatbot_tools.py and the generative-UI actions in frontend/src/providers/E2ICopilotProvider.tsx do not have; (3) cache correctness under concurrent first calls and a failing DB; (4) the readable: any way the agent prompt changes on pages that publish nothing; (5) test quality. If a recommendation solves a labeling problem instead of a functional problem, flag it as HIGH finding. If a recommendation preserves code without investigating intent (PR history, linked issues, user-requested functionality), flag it as HIGH finding. If a recommendation deletes code without verifying intent, flag it as HIGH finding. Audit the question being asked, not just the answer given. End with a single verdict line: ACCEPT or REJECT with numbered findings." < /dev/null 2>&1 | tee /tmp/pill_catalog_codex_iter1.log | tail -60
```

Address every HIGH finding with a commit (test first), re-run the audit as iter2 with the same brief plus "Previous findings and how they were addressed: ..." until the final block says ACCEPT. Record the iteration count for the memory note.

---

### Task 15: Merge (user go), deploy verification, live certification

- [ ] **Step 1: Ask the user for the merge go**

Report: CI status, codex verdict, gate numbers. Merging main triggers a prod deploy that force-recreates the api container; do not merge without an explicit go in this session.

- [ ] **Step 2: Merge preserving history**

```bash
gh pr merge <PR#> --merge --delete-branch=false
git fetch -q origin main && git log --oneline -1 origin/main
```

Expected: a merge commit on main whose subject is `Merge pull request #<PR#> ...`. Never `--squash`.

- [ ] **Step 3: Wait for the LAST deploy run to be terminal**

```bash
gh run list --branch main --limit 5 --json databaseId,headSha,status,conclusion,workflowName,createdAt
```

Find the deploy run whose `headSha` starts with the merge sha's prefix; poll (bounded, 3-5 min apart) until `status == completed`. A newer run on main supersedes it: wait for the newest. Then verify container CONTENT rather than the job conclusion:

```bash
API=$(docker ps --format '{{.Names}}' | grep -i api | head -1); echo "$API"
docker exec "$API" grep -c "WHAT THE ASSISTANT CAN DO" /app/src/services/chat_capability_catalog.py
docker exec "$API" grep -c "JSON or short prose summaries" /app/src/routes/copilotkit.py 2>/dev/null || docker exec "$API" grep -rc "JSON or short prose summaries" /app/src/api/routes/copilotkit.py
```

Expected: `1` for each marker. For the frontend, grep the served bundle (the provider is in the eager index chunk; the sidebar may be a lazy chunk, so grep all assets):

```bash
WEB=$(docker ps --format '{{.Names}}' | grep -iE "frontend|web|nginx" | head -1); echo "$WEB"
docker exec "$WEB" sh -c 'grep -l "visible on the page, as published by the page itself" /usr/share/nginx/html/assets/*.js | head -3'
docker exec "$WEB" sh -c 'grep -l "STATIC FALLBACK + TOP-UP\|topUpChatSuggestions\|ground suggestion pills" /usr/share/nginx/html/assets/*.js | head -3'
```

Expected: at least one file each. If the html root differs, find it with `docker exec "$WEB" sh -c 'ls /usr/share/nginx/html || ls /app/dist'`.

- [ ] **Step 4: Live pill probe against prod**

Replay the baseline request shapes through the deployed endpoint. Copy the existing probe (it logs in with the reviewer credentials from `.env` and posts each body to `/api/chat/suggestions`) and point its outputs at post-merge filenames:

```bash
R=/home/enunez/Projects/e2i_causal_analytics/docs/demos/results/2026-09-05_pill_suggestions_review
grep -n "json.dump\|open(" $R/probe_pills.py      # find the two output paths
cp $R/probe_pills.py $R/probe_pills_post_merge.py
# edit the two output paths found above to live_pills_post_merge.json / live_pills_post_merge.log
$PY $R/probe_pills_post_merge.py 2>&1 | tail -60
```

Grade with the Task 7 rubric into `$R/live_grades_post_merge.md`. Pass: NO <= 10% overall; every response 200 (a 502 means all pills were dropped; note how many). Also count validator drops in the container log for the probe window:

```bash
docker logs "$API" --since 30m 2>&1 | grep -c "chat suggestion dropped"
docker logs "$API" --since 30m 2>&1 | grep "chat suggestion dropped" | sed 's/.*rule=\([^ ]*\).*/\1/' | sort | uniq -c
```

- [ ] **Step 5: Agent-side certification (the readable)**

With the reviewer credentials, open `https://eznomics.site/feature-importance` in the browser (Playwright recipe: memory `system_health_score_one_decimal_pr1897_20260905.md`), wait for the page's SHAP table to render, open the chat, type `What is on this page?` and click the send button (Enter does not submit the CopilotKit textbox). Pass: the reply names entities from the page's published summary (the model/brand and at least one on-screen feature or value), says they are on-screen values, and does not claim to have computed SHAP. Save a screenshot and the reply text to `$R/agent_readable_cert_post_merge.md`.

Then open `https://eznomics.site/kpi-dictionary` (publishes no summary), open the chat, and confirm the opener pills are four, brand-appropriate, and not all the same lead pill as on `/time-series`.

- [ ] **Step 6: Memory update**

Update `/home/enunez/.claude/projects/-home-enunez-Projects-e2i-causal-analytics/memory/copilot_pill_suggestions_review_20260905.md`: status shipped (PR number, merge sha, deploy run id, gate numbers, live post-merge NO %, codex iterations), and flip its `MEMORY.md` pointer line from ⏳ to ✅ (edit the pointer by file name; keep the line short).

---

## Self-review against the spec

| Spec section | Task |
|---|---|
| 3.1 catalog fields from registry / coverage / families / outcomes / roster | Task 1 |
| 3.1 sections A-H, NEVER list, outcome-not-KPI sentence, degraded fallbacks | Task 2 |
| 3.1 axis vocabulary pinned to `kpi_calculate_tool` | Task 2 (`test_axis_vocabulary_matches_kpi_calculate_tool`) |
| 3.1 route hints for non-publishing pages; "at least two letters" rule | Tasks 3, 6 |
| 3.1 caching, TTL, last-good on refresh failure, no startup hook | Task 5 |
| 3.1 on-screen rule revised for Part C; `max_tokens` 600 | Tasks 6, 8 |
| 3.2 validator families, INFO log with rule, 502 when none survive | Tasks 4, 8 |
| 3.2 frontend top-up to four | Task 11 |
| 3.3 fifth readable, enabled/disabled, same string, note wording | Tasks 9, 10 |
| 4 error handling table | Tasks 1, 5, 8, 10 |
| 5.1 re-measurement gate before implementation of the route | Task 7 |
| 5.2-5.4 unit, route, frontend tests | Tasks 1-6, 8-11 |
| 5.5 live certification | Task 15 |
| 6 files | all |

Type consistency checked: `CapabilityCatalog` fields (`kpis`, `trend_kpi_ids`, `per_brand_only_trend_ids`, `axis_kpi_ids`, `causal_outcomes`, `agent_roster`, `degraded`, `loaded_at`) are used with those names in Tasks 2, 4, 5, 6, 7, 8; `filter_unsupported_pills(pills, catalog) -> (kept, [(pill, rule)])` is used identically in Tasks 4, 7, 8; `build_system_prompt(catalog, page)` in Tasks 6, 7, 8; `topUpChatSuggestions(adaptive, pathname, brand)` in Task 11; `route_hint(page)` in Tasks 3, 6.
