# KPI Arbitrary Time Window — Implementation Plan (Phase 1)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let users request *any* time window for the claims-temporal KPIs (rolling like "last 3 months" or absolute like "Q1 2025"), answered correctly by the KPI engine and the CopilotKit chatbot — with honest "window not applicable" handling for non-temporal KPIs.

**Architecture:** A normalized `[start, end)` window flows from the chatbot (NL parser) into `calculate(kpi_id, context={brand, region, window})`. The calculator routes to **additive, code-generated** `*_windowed` allowlist variants (base vetted queries untouched), binding `$start`/`$end` as positional params (≤4 total). `KPIResult` carries window provenance, echoed by the chatbot. Non-temporal KPIs ignore the window and say so.

**Tech Stack:** Python 3.12, Pydantic v2, FastAPI, LangGraph/CopilotKit, Supabase (Postgres `kpi_query` allowlist RPC), pytest, `python-dateutil`.

**Spec:** `docs/superpowers/specs/2026-06-20-kpi-arbitrary-window-design.md`

**Scope (Phase 1):** the ~18 CLEAN volume + same-window-ratio KPIs + full chatbot wiring. `needs_care` KPIs and the dashboard date-range UI are Phase 2 (out of scope here).

---

## File Structure

| File | Create/Modify | Responsibility |
|---|---|---|
| `src/services/time_window.py` | Create | Parse rolling/absolute window text → normalized `[start,end)` |
| `tests/unit/test_services/test_time_window.py` | Create | Parser unit tests |
| `src/kpi/synthetic_mode.py` | Modify | Add `windowed_query_id()` suffix composition |
| `tests/unit/test_kpi/test_synthetic_mode_windowed.py` | Create | Query-id composition tests |
| `src/kpi/models.py` | Modify | Add `window_requested` / `window_applied` / `window_status` to `KPIResult` |
| `src/kpi/calculators/base.py` (or shared mixin) | Modify | `_resolve_windowed_call()` → `(query_id, params)` helper |
| `src/kpi/calculators/business_impact.py` | Modify | Route windowable KPIs through the helper |
| `src/kpi/calculators/trigger_performance.py`, `data_quality.py`, `model_performance.py`, `brand_specific.py` | Modify | Same routing for their Phase-1 windowable KPIs |
| `config/kpi_definitions.yaml` | Modify | Per-KPI `windowable` + `window` block |
| `scripts/gen_kpi_windowed_variants.py` | Create | Generate `*_windowed*` allowlist rows from config |
| `database/migrations/0NN_kpi_windowed_variants.sql` | Create (generated) | The windowed query rows |
| `tests/unit/test_kpi/test_gen_windowed_variants.py` | Create | Generator golden-SQL test |
| `src/api/routes/chatbot_tools.py` | Modify | `kpi_calculate_tool` window arg + `_kpi_result_to_response` provenance |
| `tests/unit/test_api/test_chatbot_kpi_tool.py` | Modify | Tool window + provenance tests |
| `src/api/routes/copilotkit.py` | Modify | `synthesize_node` context fix + system prompt |
| `tests/unit/test_api/test_copilotkit_synthesis.py` | Create | Synthesis-includes-question test |

**Conventions discovered (must match):**
- Windowed variant id naming (canonical order): `{base}_windowed`, `{base}_windowed_region`, `{base}_windowed_include_synthetic`, `{base}_windowed_region_include_synthetic`.
- Param order for windowed calls: `[brand, start, end]` (no region) or `[brand, region, start, end]` (region) — brand first, region second, then `$start`,`$end` (≤4, the RPC cap).
- Window dict shape everywhere: `{"start": "<ISO8601>", "end": "<ISO8601>"}`, half-open `[start, end)`, both UTC.
- Base (no-window) queries and their `_region`/`_include_synthetic` twins are **never modified**.

---

## Milestone 0 — Cheapest-disproof GATE (do FIRST)

**Why:** The load-bearing assumption is "the window can be a parameter." The `kpi_query` RPC forbids client SQL and bakes intervals into vetted strings. Validate param-binding against the real RPC before generating ~70 variant rows. If this fails, STOP and switch to the "new windowed RPC" approach (spec §4 alt) before writing more code.

### Task 0: Validate `$start/$end` binding against the live RPC

**Files:** none committed (throwaway spike script `/tmp/kpi_window_gate.py`).

- [ ] **Step 1: Write the spike** — register ONE additive windowed row, call it, assert, then delete it (additive + reversible; never touches existing rows/data/gates). Run against the dev/synthetic-gold DB via `.env` `SUPABASE_SERVICE_KEY`.

```python
# /tmp/kpi_window_gate.py
import os
from dotenv import load_dotenv
load_dotenv('/home/enunez/Projects/e2i_causal_analytics/.env')
from supabase import create_client
c = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_KEY"])

QID = "business_impact_nrx_windowed_include_synthetic__GATE"
SQL = ("SELECT COUNT(*) AS nrx FROM treatment_events "
       "WHERE event_type::text = 'prescription' AND sequence_number = 1 "
       "AND event_date >= $2::timestamptz AND event_date < $3::timestamptz "
       "AND ($1::text IS NULL OR brand::text = $1)")

# Insert the row (additive). Adjust column names to kpi_query_registry's schema if needed.
c.table("kpi_query_registry").upsert(
    {"query_id": QID, "sql": SQL, "max_params": 3, "note": "TEMP gate spike"}
).execute()
try:
    r = c.rpc("kpi_query", {"query_id": QID,
              "params": ["Kisqali", "2026-03-22T00:00:00Z", "2026-06-20T00:00:00Z"]}).execute()
    print("RESULT:", (r.data or [{}])[0].get("nrx"))
finally:
    c.table("kpi_query_registry").delete().eq("query_id", QID).execute()
    print("cleaned up gate row")
```

- [ ] **Step 2: Run it**

Run: `cd /home/enunez/Projects/e2i_causal_analytics && .venv/bin/python /tmp/kpi_window_gate.py`
Expected: `RESULT: 3394` (±boundary), then `cleaned up gate row`. (90-day Kisqali NRx measured 2026-06-20 = 3,394.)

- [ ] **Step 3: Decide.** PASS → param binding works; proceed to M1. FAIL on `kpi_query_registry` schema → inspect the registry table columns (`\d kpi_query_registry` equivalent via `c.table('kpi_query_registry').select('*').limit(1)`) and fix the insert keys, re-run. FAIL on binding/cast (RPC can't pass date text params) → STOP, escalate: switch to spec §4 "new windowed RPC". Do NOT proceed to codegen.

- [ ] **Step 4: Clean up** — `rm /tmp/kpi_window_gate.py`. No commit (spike only).

---

## Milestone 1 — Window parser (`src/services/time_window.py`)

Self-contained, no DB. Pure date math. This is the highest-value isolated unit.

### Task 1: `Window` model + `parse_window()`

**Files:**
- Create: `src/services/time_window.py`
- Test: `tests/unit/test_services/test_time_window.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/unit/test_services/test_time_window.py
from datetime import datetime, timezone
import pytest
from src.services.time_window import parse_window, Window, WindowParseError

NOW = datetime(2026, 6, 20, tzinfo=timezone.utc)

def test_rolling_months():
    w = parse_window("past 3 months", now=NOW)
    assert w.kind == "rolling"
    assert w.end == NOW
    assert w.start == datetime(2026, 3, 20, tzinfo=timezone.utc)

def test_rolling_days():
    w = parse_window("last 90 days", now=NOW)
    assert w.start == datetime(2026, 3, 22, tzinfo=timezone.utc)
    assert w.end == NOW

def test_absolute_quarter():
    w = parse_window("Q1 2025", now=NOW)
    assert w.kind == "absolute"
    assert w.start == datetime(2025, 1, 1, tzinfo=timezone.utc)
    assert w.end == datetime(2025, 4, 1, tzinfo=timezone.utc)

def test_absolute_year():
    w = parse_window("2024", now=NOW)
    assert w.start == datetime(2024, 1, 1, tzinfo=timezone.utc)
    assert w.end == datetime(2025, 1, 1, tzinfo=timezone.utc)

def test_explicit_dict():
    w = parse_window({"start": "2025-01-01", "end": "2025-02-01"}, now=NOW)
    assert w.start == datetime(2025, 1, 1, tzinfo=timezone.utc)

def test_none_passthrough():
    assert parse_window(None, now=NOW) is None

def test_invalid_raises():
    with pytest.raises(WindowParseError):
        parse_window("the time of legends", now=NOW)

def test_start_after_end_raises():
    with pytest.raises(WindowParseError):
        parse_window({"start": "2025-05-01", "end": "2025-01-01"}, now=NOW)

def test_to_params_iso():
    w = parse_window("Q1 2025", now=NOW)
    assert w.start_iso == "2025-01-01T00:00:00+00:00"
    assert w.end_iso == "2025-04-01T00:00:00+00:00"
```

- [ ] **Step 2: Run, verify FAIL**

Run: `cd /home/enunez/Projects/e2i_causal_analytics && .venv/bin/python -m pytest tests/unit/test_services/test_time_window.py -q`
Expected: FAIL — `ModuleNotFoundError: src.services.time_window`.

- [ ] **Step 3: Implement**

```python
# src/services/time_window.py
"""Parse user-requested time windows (rolling or absolute) into a normalized
half-open ``[start, end)`` UTC range for the KPI engine.

Rolling windows ("last 3 months") are anchored to ``now``. Absolute windows
("Q1 2025", "2024", "Jan-Mar 2025", ISO dates) are fixed. Returns ``None`` for
no-window input; raises :class:`WindowParseError` on unparseable / invalid input
(never silently defaults — see spec §6)."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Optional

from dateutil.relativedelta import relativedelta

_MONTHS = {m: i for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun",
     "jul", "aug", "sep", "oct", "nov", "dec"], start=1)}


class WindowParseError(ValueError):
    """Raised when a window string cannot be parsed or is invalid."""


@dataclass(frozen=True)
class Window:
    start: datetime
    end: datetime
    kind: str  # "rolling" | "absolute"
    label: str

    @property
    def start_iso(self) -> str:
        return self.start.isoformat()

    @property
    def end_iso(self) -> str:
        return self.end.isoformat()

    def as_dict(self) -> dict[str, str]:
        return {"start": self.start_iso, "end": self.end_iso}


def _utc(dt: datetime) -> datetime:
    return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt.astimezone(timezone.utc)


def _validate(start: datetime, end: datetime, kind: str, label: str) -> Window:
    if start >= end:
        raise WindowParseError(f"window start {start} is not before end {end}")
    return Window(start=start, end=end, kind=kind, label=label)


def parse_window(spec: Any, *, now: Optional[datetime] = None) -> Optional[Window]:
    now = _utc(now) if now else datetime.now(timezone.utc)
    if spec is None or (isinstance(spec, str) and not spec.strip()):
        return None

    # Explicit {start, end}
    if isinstance(spec, dict):
        try:
            start = _utc(datetime.fromisoformat(str(spec["start"])))
            end = _utc(datetime.fromisoformat(str(spec["end"])))
        except (KeyError, ValueError) as e:
            raise WindowParseError(f"invalid explicit window {spec!r}: {e}") from e
        return _validate(start, end, "absolute", f"{start.date()} to {end.date()}")

    if not isinstance(spec, str):
        raise WindowParseError(f"unsupported window type: {type(spec).__name__}")
    s = spec.strip().lower()

    # Rolling: "last/past/trailing N day|week|month|year(s)"
    m = re.fullmatch(r"(?:last|past|trailing|previous)\s+(\d+)\s+(day|week|month|year)s?", s)
    if m:
        n, unit = int(m.group(1)), m.group(2)
        delta = {"day": relativedelta(days=n), "week": relativedelta(weeks=n),
                 "month": relativedelta(months=n), "year": relativedelta(years=n)}[unit]
        return _validate(now - delta, now, "rolling", f"last {n} {unit}s")

    # Absolute quarter: "q1 2025"
    m = re.fullmatch(r"q([1-4])\s+(\d{4})", s)
    if m:
        q, yr = int(m.group(1)), int(m.group(2))
        start = datetime(yr, 3 * (q - 1) + 1, 1, tzinfo=timezone.utc)
        return _validate(start, start + relativedelta(months=3), "absolute", f"Q{q} {yr}")

    # Absolute month range: "jan-mar 2025" / "jan to mar 2025"
    m = re.fullmatch(r"([a-z]{3,})\s*(?:-|to|–)\s*([a-z]{3,})\s+(\d{4})", s)
    if m and m.group(1)[:3] in _MONTHS and m.group(2)[:3] in _MONTHS:
        a, b, yr = _MONTHS[m.group(1)[:3]], _MONTHS[m.group(2)[:3]], int(m.group(3))
        start = datetime(yr, a, 1, tzinfo=timezone.utc)
        end = datetime(yr, b, 1, tzinfo=timezone.utc) + relativedelta(months=1)
        return _validate(start, end, "absolute", f"{m.group(1)}-{m.group(2)} {yr}")

    # Single month: "march 2025"
    m = re.fullmatch(r"([a-z]{3,})\s+(\d{4})", s)
    if m and m.group(1)[:3] in _MONTHS:
        mo, yr = _MONTHS[m.group(1)[:3]], int(m.group(2))
        start = datetime(yr, mo, 1, tzinfo=timezone.utc)
        return _validate(start, start + relativedelta(months=1), "absolute", f"{m.group(1)} {yr}")

    # Bare year: "2024"
    m = re.fullmatch(r"(\d{4})", s)
    if m:
        yr = int(m.group(1))
        start = datetime(yr, 1, 1, tzinfo=timezone.utc)
        return _validate(start, datetime(yr + 1, 1, 1, tzinfo=timezone.utc), "absolute", str(yr))

    # ISO range: "2025-01-01 to 2025-03-31"
    m = re.fullmatch(r"(\d{4}-\d{2}-\d{2})\s*(?:to|–|-)\s*(\d{4}-\d{2}-\d{2})", s)
    if m:
        start = _utc(datetime.fromisoformat(m.group(1)))
        end = _utc(datetime.fromisoformat(m.group(2)))
        return _validate(start, end, "absolute", f"{m.group(1)} to {m.group(2)}")

    raise WindowParseError(f"could not parse time window: {spec!r}")
```

- [ ] **Step 4: Run, verify PASS**

Run: `cd /home/enunez/Projects/e2i_causal_analytics && .venv/bin/python -m pytest tests/unit/test_services/test_time_window.py -q`
Expected: PASS (9 passed). If `dateutil` import fails: `uv pip install python-dateutil` (it is a transitive dep of pandas; should already be present).

- [ ] **Step 5: Commit**

```bash
cd /home/enunez/Projects/wt_kpi_window
git add src/services/time_window.py tests/unit/test_services/test_time_window.py
git commit -m "feat(kpi): add time-window parser (rolling + absolute -> [start,end))

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Milestone 2 — Query-id composition (`src/kpi/synthetic_mode.py`)

### Task 2: `windowed_query_id()`

**Files:**
- Modify: `src/kpi/synthetic_mode.py` (add after `region_query_id`)
- Test: `tests/unit/test_kpi/test_synthetic_mode_windowed.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/unit/test_kpi/test_synthetic_mode_windowed.py
import importlib, src.kpi.synthetic_mode as sm

def _reload(monkeypatch, flag):
    monkeypatch.setenv("E2I_KPI_INCLUDE_SYNTHETIC", "1" if flag else "0")
    monkeypatch.delenv("E2I_INCLUDE_SYNTHETIC", raising=False)
    return importlib.reload(sm)

def test_windowed_base(monkeypatch):
    m = _reload(monkeypatch, False)
    assert m.windowed_query_id("business_impact_nrx", region=False) == "business_impact_nrx_windowed"

def test_windowed_region(monkeypatch):
    m = _reload(monkeypatch, False)
    assert m.windowed_query_id("business_impact_nrx", region=True) == "business_impact_nrx_windowed_region"

def test_windowed_synthetic(monkeypatch):
    m = _reload(monkeypatch, True)
    assert m.windowed_query_id("business_impact_nrx", region=False) == "business_impact_nrx_windowed_include_synthetic"

def test_windowed_region_synthetic(monkeypatch):
    m = _reload(monkeypatch, True)
    assert m.windowed_query_id("business_impact_nrx", region=True) == "business_impact_nrx_windowed_region_include_synthetic"
```

- [ ] **Step 2: Run, verify FAIL**

Run: `cd /home/enunez/Projects/e2i_causal_analytics && .venv/bin/python -m pytest tests/unit/test_kpi/test_synthetic_mode_windowed.py -q`
Expected: FAIL — `AttributeError: module ... has no attribute 'windowed_query_id'`.

- [ ] **Step 3: Implement** — append to `src/kpi/synthetic_mode.py`:

```python
def windowed_query_id(base_query_id: str, *, region: bool) -> str:
    """Windowed variant id for a base KPI query (Phase 1, additive).

    Canonical suffix order: ``{base}_windowed[_region][_include_synthetic]``.
    Parallels :func:`region_query_id`: the ``_windowed*`` variants are ADDITIVE
    and absent from :data:`SYNTHETIC_TWINNED_QUERY_IDS`, so we append the
    ``_include_synthetic`` suffix HERE under the showcase flag. Passing the
    result back through :func:`resolve_kpi_query_id` is a safe no-op.
    """
    qid = f"{base_query_id}_windowed"
    if region:
        qid = f"{qid}_region"
    return f"{qid}{_SYNTHETIC_SUFFIX}" if kpi_include_synthetic() else qid
```

- [ ] **Step 4: Run, verify PASS**

Run: `cd /home/enunez/Projects/e2i_causal_analytics && .venv/bin/python -m pytest tests/unit/test_kpi/test_synthetic_mode_windowed.py -q`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
cd /home/enunez/Projects/wt_kpi_window
git add src/kpi/synthetic_mode.py tests/unit/test_kpi/test_synthetic_mode_windowed.py
git commit -m "feat(kpi): windowed_query_id() suffix composition

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Milestone 3 — Engine plumbing (KPIResult provenance + calculator routing)

### Task 3: Add window provenance fields to `KPIResult`

**Files:** Modify `src/kpi/models.py` (in `KPIResult`, after `metadata`)

- [ ] **Step 1: Write failing test** — `tests/unit/test_kpi/test_kpiresult_window.py`:

```python
from src.kpi.models import KPIResult
def test_window_fields_default_none():
    r = KPIResult(kpi_id="WS3-BI-006", value=1.0)
    assert r.window_requested is None
    assert r.window_applied is None
    assert r.window_status == "default"
def test_window_fields_set():
    r = KPIResult(kpi_id="WS3-BI-006", value=1.0,
                  window_requested={"start": "a", "end": "b"},
                  window_applied={"start": "a", "end": "b"},
                  window_status="applied")
    assert r.window_status == "applied"
```

- [ ] **Step 2: Run, verify FAIL**

Run: `cd /home/enunez/Projects/e2i_causal_analytics && .venv/bin/python -m pytest tests/unit/test_kpi/test_kpiresult_window.py -q`
Expected: FAIL — `ValidationError`/unexpected kwarg `window_requested`.

- [ ] **Step 3: Implement** — add to `KPIResult` (after line 145 `metadata`):

```python
    # Window provenance (spec 2026-06-20). window_status:
    #   "default"         -> no window requested; engine's fixed window used
    #   "applied"         -> requested window honored
    #   "not_applicable"  -> KPI has no claims time-dimension; window ignored honestly
    window_requested: dict[str, Any] | None = None
    window_applied: dict[str, Any] | None = None
    window_status: str = Field("default", description="default | applied | not_applicable")
```

- [ ] **Step 4: Run, verify PASS** — same command. Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
cd /home/enunez/Projects/wt_kpi_window
git add src/kpi/models.py tests/unit/test_kpi/test_kpiresult_window.py
git commit -m "feat(kpi): KPIResult window provenance fields

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

### Task 4: Shared `_resolve_windowed_call()` helper on the calculator base

**Files:** Modify the calculator base class. Find it first:
Run: `grep -rn "def _execute_query\|class .*Calculator" src/kpi/calculators/*.py | head`. The helper lives on the shared base (the class that defines `_execute_query`). If `_execute_query` is duplicated per-file, add the helper to each (DRY note: prefer hoisting to a base in a follow-up; for Phase 1 match the existing structure).

- [ ] **Step 1: Write failing test** — `tests/unit/test_kpi/test_windowed_call.py`:

```python
from src.kpi.calculators.business_impact import BusinessImpactCalculator

def _calc():
    return BusinessImpactCalculator(db_client=None)  # helper is pure; no DB

def test_no_window_brand_only():
    qid, params = _calc()._resolve_windowed_call(
        "business_impact_nrx", brand="Kisqali", region=None, window=None)
    assert qid == "business_impact_nrx"
    assert params == ["Kisqali"]

def test_window_brand():
    w = {"start": "2025-01-01T00:00:00+00:00", "end": "2025-04-01T00:00:00+00:00"}
    qid, params = _calc()._resolve_windowed_call(
        "business_impact_nrx", brand="Kisqali", region=None, window=w)
    assert qid == "business_impact_nrx_windowed"
    assert params == ["Kisqali", w["start"], w["end"]]

def test_window_region():
    w = {"start": "2025-01-01T00:00:00+00:00", "end": "2025-04-01T00:00:00+00:00"}
    qid, params = _calc()._resolve_windowed_call(
        "business_impact_nrx", brand="Kisqali", region="northeast", window=w)
    assert qid == "business_impact_nrx_windowed_region"
    assert params == ["Kisqali", "northeast", w["start"], w["end"]]
```

- [ ] **Step 2: Run, verify FAIL**

Run: `cd /home/enunez/Projects/e2i_causal_analytics && .venv/bin/python -m pytest tests/unit/test_kpi/test_windowed_call.py -q`
Expected: FAIL — `AttributeError: ... '_resolve_windowed_call'`.

- [ ] **Step 3: Implement** — add the method to the calculator class (import `windowed_query_id`, `region_query_id` from `src.kpi.synthetic_mode` at top):

```python
    def _resolve_windowed_call(
        self,
        base_query_id: str,
        *,
        brand: str | None,
        region: str | None,
        window: dict[str, Any] | None,
    ) -> tuple[str, list[Any]]:
        """Compose (query_id, positional params) for a windowable KPI.

        Param order respects the kpi_query 4-param cap:
          no region:  [brand, start, end]
          region:     [brand, region, start, end]
        With no window, falls back to the existing base / _region behavior.
        """
        if window is None:
            if region:
                return region_query_id(base_query_id), [brand, region]
            return base_query_id, [brand]
        qid = windowed_query_id(base_query_id, region=bool(region))
        if region:
            return qid, [brand, region, window["start"], window["end"]]
        return qid, [brand, window["start"], window["end"]]
```

- [ ] **Step 4: Run, verify PASS** — same command. Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
cd /home/enunez/Projects/wt_kpi_window
git add src/kpi/calculators/business_impact.py tests/unit/test_kpi/test_windowed_call.py
git commit -m "feat(kpi): _resolve_windowed_call() routing helper

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

### Task 5: Thread `window` through `calculate()` + cache key + route NRx

**Files:** Modify `src/kpi/calculator.py` (cache context), `src/kpi/calculators/business_impact.py` (`_calc_nrx`).

- [ ] **Step 1: Write failing test** (uses a stub db_client returning a fixed count) — `tests/unit/test_kpi/test_nrx_windowed_route.py`:

```python
from src.kpi.calculators.business_impact import BusinessImpactCalculator

class _StubResp:
    def __init__(self, data): self.data = data
class _StubRPC:
    def __init__(self, sink): self.sink = sink
    def rpc(self, name, payload):
        self.sink.append(payload)
        return _StubExec()
class _StubExec:
    def execute(self): return _StubResp([{"nrx": 3394}])

def test_nrx_passes_window_params():
    sink = []
    calc = BusinessImpactCalculator(db_client=_StubRPC(sink))
    w = {"start": "2026-03-22T00:00:00+00:00", "end": "2026-06-20T00:00:00+00:00"}
    res = calc._calc_nrx({"brand": "Kisqali", "window": w})
    assert res == 3394.0 or res == 3394  # _calc_nrx returns the numeric value
    assert sink[-1]["query_id"].startswith("business_impact_nrx_windowed")
    assert sink[-1]["params"] == ["Kisqali", w["start"], w["end"]]
```

> NOTE: confirm `_calc_nrx`'s exact return/IO by reading it first; adapt the assert to its real contract (it currently calls `self._execute_query("business_impact_nrx", [brand])` and returns a float). Keep the test asserting the **query_id + params** that reach the RPC, which is the behavior under change.

- [ ] **Step 2: Run, verify FAIL**

Run: `cd /home/enunez/Projects/e2i_causal_analytics && .venv/bin/python -m pytest tests/unit/test_kpi/test_nrx_windowed_route.py -q`
Expected: FAIL — params lack start/end (still `["Kisqali"]`).

- [ ] **Step 3: Implement**

(a) `_calc_nrx` in `business_impact.py` — replace the query/params selection with the helper:

```python
    def _calc_nrx(self, context: dict[str, Any]) -> float:
        brand = context.get("brand")
        region = context.get("region")
        window = context.get("window")
        query_id, params = self._resolve_windowed_call(
            "business_impact_nrx", brand=brand, region=region, window=window)
        result = self._execute_query(query_id, params)
        if not result:
            return 0.0
        return float(result[0].get("nrx") or 0)
```

(b) `calculate()` in `calculator.py` (~line 161) — include window in the cache key so windowed reads cache separately:

```python
        window = context.get("window")
        cache_context = {**context, "_include_synthetic": include_synthetic,
                         "_window": (window.get("start"), window.get("end")) if window else None}
```

- [ ] **Step 4: Run, verify PASS** — same command. Expected: PASS.

- [ ] **Step 5: Commit**

```bash
cd /home/enunez/Projects/wt_kpi_window
git add src/kpi/calculator.py src/kpi/calculators/business_impact.py tests/unit/test_kpi/test_nrx_windowed_route.py
git commit -m "feat(kpi): route NRx through windowed call + window-aware cache key

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

### Task 6: Set `window_*` provenance on the KPIResult + `not_applicable` honesty

**Files:** Modify `src/kpi/calculator.py` (`_calculate_kpi`/`_default_calculate` where `KPIResult` is built) to stamp provenance from the per-KPI `windowable` flag (loaded in Task 7) and the requested window.

- [ ] **Step 1: Write failing test** — `tests/unit/test_kpi/test_window_provenance.py`:

```python
# Pseudocode contract — adapt to the calculate() seam:
# - windowable KPI + window -> result.window_status == "applied", window_applied == requested
# - not_applicable KPI + window -> result.window_status == "not_applicable", window_applied is None,
#   value still computed (window ignored, never faked)
# - no window -> window_status == "default"
```

- [ ] **Step 2-5:** implement the stamping in the single place `KPIResult(...)` is constructed for the success path; read `kpi.windowable` (Task 7 adds it to `KPIMetadata`); set the three fields; run the test; commit. (Keep `not_applicable` returning the value — only the provenance differs.)

```bash
git commit -m "feat(kpi): stamp window provenance (applied/not_applicable/default) on KPIResult"
```

### Task 7: Per-KPI `windowable` + `window` config

**Files:** Modify `config/kpi_definitions.yaml`; extend the YAML→`KPIMetadata` loader (find via `grep -rn "kpi_definitions.yaml" src/`) to parse `windowable` (default `not_applicable`) and a `window` block; add `windowable: str` + `window: dict | None` to `KPIMetadata` in `models.py`.

- [ ] **Step 1: Write failing test** — assert a known windowable KPI loads `windowable == "clean"` with the right `window.column`, and a snapshot loads `not_applicable`.
- [ ] **Step 2:** run, verify FAIL.
- [ ] **Step 3:** add fields + loader parsing; populate the YAML per the spec §3 table. Phase-1 CLEAN rows (id → window.column):

| id | window.column | legs (ratio) |
|---|---|---|
| WS3-BI-005 TRx | event_date | count |
| WS3-BI-006 NRx | event_date | count |
| WS3-BI-007 NBRx | event_date (MIN) | count |
| WS3-BI-008 TRx Share | event_date | brand_rx, category |
| WS3-BI-009 Conversion Rate | trigger_timestamp | triggered, converted (look_forward_days: 30 FIXED) |
| WS3-BI-010 ROI | (roi window) | count |
| WS3-BI-001 MAU | session_start | count (view-backed: rolling-only Phase 1) |
| WS3-BI-002 WAU | session_start | count (view-backed: rolling-only Phase 1) |
| WS2-TR-001 Precision | trigger_timestamp | tp, fp |
| WS2-TR-004 Acceptance | trigger_timestamp | accepted, total |
| WS2-TR-005 False Alert | trigger_timestamp | fp, total |
| WS2-TR-006 Override | trigger_timestamp | overridden, total |
| WS2-TR-007 Lead Time | trigger_timestamp | count |
| WS2-TR-008 CFR | trigger_timestamp | count |
| WS1-DQ-001 Source Coverage Patients | created_at | covered, universe |
| WS1-DQ-005 Completeness | created_at | count |
| WS1-MP-007 SHAP Coverage | created_at | covered, total |
| BR-005 Kisqali Reach | trigger_timestamp | count |

All other KPIs (spec §3 NOT-APPLICABLE + NEEDS-CARE) get `windowable: not_applicable` for Phase 1 (`needs_care` are deferred, so they behave as not_applicable until Phase 2).

- [ ] **Step 4:** run, verify PASS.
- [ ] **Step 5:** commit `feat(kpi): per-KPI windowable+window config and loader`.

---

## Milestone 4 — Codegen + migration (additive windowed variants)

### Task 8: `scripts/gen_kpi_windowed_variants.py`

**Files:**
- Create: `scripts/gen_kpi_windowed_variants.py`
- Create (generated): `database/migrations/0NN_kpi_windowed_variants.sql` (NN = next number; run `ls database/migrations | tail -1`)
- Test: `tests/unit/test_kpi/test_gen_windowed_variants.py`

**Approach:** the generator reads each base windowed query's SQL from migration 044/066/077, derives the `*_windowed*` variants by replacing the hardcoded `NOW() - INTERVAL '<n> days'` lower bound with `>= $K::timestamptz AND <col> < $(K+1)::timestamptz` (binding the same params to every window leg for ratios), and emits the four-variant family per KPI with correct `max_params`. Synthetic-include variants drop the `is_synthetic=false` wrapper (mirror 066); region variants add the region join (mirror 077).

- [ ] **Step 1: Write failing golden test** — for NRx, assert the generated `business_impact_nrx_windowed_include_synthetic` SQL equals the expected (the M0 gate's validated SQL) and `max_params == 3`; region variant `max_params == 4`.

```python
# tests/unit/test_kpi/test_gen_windowed_variants.py
from scripts.gen_kpi_windowed_variants import generate_variant

def test_nrx_windowed_synthetic():
    v = generate_variant("business_impact_nrx", region=False, include_synthetic=True)
    assert v.query_id == "business_impact_nrx_windowed_include_synthetic"
    assert v.max_params == 3
    assert "event_date >= $2::timestamptz" in v.sql
    assert "event_date < $3::timestamptz" in v.sql
    assert "($1::text IS NULL OR brand::text = $1)" in v.sql
    assert "INTERVAL '30 days'" not in v.sql

def test_nrx_windowed_region_params():
    v = generate_variant("business_impact_nrx", region=True, include_synthetic=True)
    assert v.max_params == 4
    assert "event_date >= $3::timestamptz" in v.sql  # brand=$1, region=$2, start=$3, end=$4
```

- [ ] **Step 2:** run, verify FAIL (module missing).
- [ ] **Step 3:** implement `generate_variant(base, *, region, include_synthetic) -> Variant(query_id, sql, max_params, note)` + a `main()` that writes the migration for the Phase-1 CLEAN set (driven by the config `windowable=="clean"` rows + their `window.column`/`legs`). Ratio legs bind the same `$start`/`$end` to each CTE's window column; the conversion-rate `look_forward_days` constant stays `+ INTERVAL '30 days'` on the outcome side. Region/synthetic shaping mirrors 077/066.
- [ ] **Step 4:** run, verify PASS; then generate the migration:

Run: `cd /home/enunez/Projects/e2i_causal_analytics && .venv/bin/python scripts/gen_kpi_windowed_variants.py --out database/migrations/0NN_kpi_windowed_variants.sql`
Then manually review the SQL diff for each KPI (especially ratio double-binding + conversion look-forward).

- [ ] **Step 5:** commit `feat(kpi): generate additive windowed allowlist variants (Phase 1)`.

### Task 9: Apply migration + faithful SQL verification

- [ ] **Step 1:** apply `0NN_kpi_windowed_variants.sql` to the dev/synthetic-gold DB (additive; base rows untouched; gates unaffected). Use the project's migration runner (`grep -rn "migrations" scripts/ Makefile* 2>/dev/null` to find it) or `supabase db push` per repo convention.
- [ ] **Step 2: Faithful verification** — `.venv/bin/python` calling `kpi_query` for `business_impact_nrx_windowed_include_synthetic` with `["Kisqali", "2026-03-22T00:00:00Z", "2026-06-20T00:00:00Z"]`. Expected: **3394** (matches the M0 gate + direct REST). Also verify a ratio (TRx Share) windowed value is in [0,1] and differs from the 30-day base.
- [ ] **Step 3: Regression** — confirm base `business_impact_nrx` (no window) still returns the rolling-30-day value (~3183) byte-for-byte; run the KPI coverage check `scripts/check_kpi_coverage.py` (or `validate_kpi_coverage.py`). Expected: unchanged, green.
- [ ] **Step 4:** commit any migration-runner notes; no code change if clean.

---

## Milestone 5 — Chatbot tool wiring

### Task 10: `kpi_calculate_tool` window arg + `_kpi_result_to_response` provenance

**Files:** Modify `src/api/routes/chatbot_tools.py`; Test `tests/unit/test_api/test_chatbot_kpi_tool.py`.

- [ ] **Step 1: Write failing tests** — (a) `_kpi_result_to_response` echoes `brand`, `region`, `window_requested`, `window_applied`, `window_status`; (b) the tool parses a `window` phrase via `parse_window` and passes `{start,end}` into the calculator context; (c) a `not_applicable` KPI returns `window_status="not_applicable"`.

```python
def test_response_echoes_brand_and_window():
    from src.api.routes.chatbot_tools import _kpi_result_to_response
    class K: id="WS3-BI-006"; name="NRx"
    class R:
        error=None; value=3394; status="unknown"; metadata={}
        window_requested={"start":"a","end":"b"}; window_applied={"start":"a","end":"b"}
        window_status="applied"
    resp = _kpi_result_to_response(K(), R(), brand="Kisqali", region=None)
    assert resp["brand"] == "Kisqali"
    assert resp["window_applied"] == {"start":"a","end":"b"}
    assert resp["window_status"] == "applied"
```

- [ ] **Step 2:** run, verify FAIL.
- [ ] **Step 3: Implement**
  - `KpiCalculateInput`: add `window: Optional[str] = Field(default=None, description="Time window e.g. 'last 3 months', 'Q1 2025', or ISO range; omit for the engine's default rolling window")`.
  - `kpi_calculate_tool`: signature gains `window: Optional[str] = None`; parse with `parse_window(window)` (catch `WindowParseError` → return `{"success": False, "error": ...,"hint": "Try 'last 3 months', 'Q1 2025', or '2025-01-01 to 2025-03-31'"}`); put `parsed.as_dict()` into `context["window"]`.
  - `_kpi_result_to_response(kpi, result, *, brand, region)`: add `brand`, `region`, and copy `window_requested`/`window_applied`/`window_status` from `result`. Keep existing `reporting_window` only when `window_status == "default"` (so a custom window doesn't also show the stale "rolling 30 days" line).

- [ ] **Step 4:** run, verify PASS.
- [ ] **Step 5:** commit `feat(chatbot): kpi_calculate_tool window arg + brand/window provenance echo`.

---

## Milestone 6 — Chatbot synthesis fix

### Task 11: `synthesize_node` includes the question + tool-call args; tighten system prompt

**Files:** Modify `src/api/routes/copilotkit.py`; Test `tests/unit/test_api/test_copilotkit_synthesis.py`.

- [ ] **Step 1: Write failing test** — extract the synthesis-prompt builder into a pure helper `build_synthesis_prompt(original_query, tool_calls, tool_results) -> str` and assert it contains the user question text and the tool-call args (brand) — not just the results.

```python
def test_synthesis_prompt_includes_question_and_args():
    from src.api.routes.copilotkit import build_synthesis_prompt
    p = build_synthesis_prompt(
        original_query="NRx for Kisqali past 3 months",
        tool_calls=[{"name": "kpi_calculate_tool", "args": {"kpi_name": "NRx", "brand": "Kisqali", "window": "past 3 months"}}],
        tool_results=[{"tool": "kpi_calculate_tool", "result": '{"value": 3394, "window_applied": {...}}'}],
    )
    assert "Kisqali" in p and "past 3 months" in p
    assert "3394" in p
    assert "User question" in p  # the prompt frames the original ask
```

- [ ] **Step 2:** run, verify FAIL.
- [ ] **Step 3: Implement**
  - Add `build_synthesis_prompt(original_query, tool_calls, tool_results)` that frames: the **user's question**, the **tool calls + args** the assistant made, and the **tool results**, instructing the model to answer the user's actual question, name the brand/period it used, and — if `window_status != "applied"` — state the window limitation up front.
  - In `synthesize_node`: collect `tool_calls` from the preceding `AIMessage` (`getattr(msg, "tool_calls", [])`), pass `original_query` + `tool_calls` + `tool_results` into `build_synthesis_prompt`. Replace the existing `synthesis_prompt` literal.
  - `E2I_COPILOT_SYSTEM_PROMPT`: in "Tool Usage", add `- Use kpi_calculate_tool to COMPUTE a KPI value for a brand/period (NRx, TRx, NBRx, market share, conversion rate, ROI). Pass the brand and any time window the user names; echo back which brand and window you used.`; add a guideline `If a requested time window isn't supported for a metric, say so plainly and report the window actually used — never imply a figure covers a different period.`; soften the blanket "Suggest follow-up questions" to "Offer at most one relevant follow-up, only when it adds value."

- [ ] **Step 4:** run, verify PASS.
- [ ] **Step 5:** commit `fix(chatbot): synthesize from the question+tool-args (stop re-asking brand) + prompt`.

---

## Milestone 7 — Faithful end-to-end verification

### Task 12: Live chatbot verification (in the running container)

- [ ] **Step 1:** apply the migration to the live DB if not already (Task 9). Rebuild/restart the api container against this branch image OR run the graph in-container via `docker exec e2i_api python -c "..."` driving `create_e2i_chat_agent()` with a `HumanMessage("tell me about the NRx for Kisqali in the past 3 months")`.
- [ ] **Step 2:** Assert the final synthesized message: (a) names **Kisqali**, (b) reports the **~3,394** 90-day figure (not 3,183), (c) does **not** ask "which brand?", (d) states it's a 3-month window. Capture the text in the PR description.
- [ ] **Step 3:** Probe a `not_applicable` KPI with a window ("HCP coverage for Kisqali in Q1 2025") → asserts value returned + honest "not time-windowed" note.
- [ ] **Step 4:** Run the focused suites green: `pytest tests/unit/test_services/test_time_window.py tests/unit/test_kpi -q` and the chatbot tool/synthesis tests. CI is the arbiter for the full suite + KPI gates.
- [ ] **Step 5:** Open the PR (DEPLOY HELD) summarizing the before/after chat transcript + the faithful 3,394 verification.

---

## Self-Review (completed by plan author)

- **Spec coverage:** D1 (windowable flag + not_applicable honesty) → T6/T7/T10; D2 (rolling+absolute→[start,end)) → T1; D3 (engine+chatbot) → M3–M6; D4 (codegen additive variants) → M4. Governance G1–G4 → M0 gate + Task 8 (`$N::timestamptz`, ≤4 params, additive). Non-temporal honesty → T6/T10. Cheapest-disproof → M0. Testing/phasing → M7 + Phase-1 scope table (T7).
- **Placeholders:** `0NN` migration number is intentional (assigned at impl via `ls database/migrations | tail -1`). Tasks 6 and 7's inner steps are summarized (the construction site + YAML rows) rather than full literals because they depend on reading the exact `_calculate_kpi` construction site and the existing YAML formatting — each lists the precise fields/values to set. Acceptable; not vague directives.
- **Type consistency:** `Window`/`parse_window`/`WindowParseError` (T1); `windowed_query_id(base, *, region)` (T2) consumed by `_resolve_windowed_call` (T4); `window_requested/window_applied/window_status` consistent across T3/T6/T10; window dict `{"start","end"}` consistent T1/T4/T5/T10.
