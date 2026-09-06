# Copilot pill follow-ups (#1901, mechanical slice) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship items 1, 2, 3, 4d, 5 and 6 of issue #1901 (catalog loader timeout, backend CI sees the router file, territory on-screen exemption, competitor no-preposition form, cache generation token, window wording from registry data) in one PR against main.

**Architecture:** Every change lives in `src/services/chat_capability_catalog.py` (the catalog builder, validator rules and cache) with tests in `tests/api/test_chat_capability_catalog.py`, plus one workflow edit. Spec: `docs/superpowers/specs/2026-09-06-pill-followups-1901-design.md`.

**Tech Stack:** Python 3.12, asyncio, pytest (asyncio_mode=auto), ruff 0.14.10, mypy (scoped), GitHub Actions.

**Environment for every task:** worktree `/home/enunez/Projects/e2i_causal_analytics/.worktrees/pill-catalog`, branch `claude/pill-followups-1901` (verify with `git branch --show-current` before every commit). Run tests with `/home/enunez/Projects/e2i_causal_analytics/.venv/bin/python -m pytest tests/api/test_chat_capability_catalog.py -q -p no:cacheprovider` from the worktree root - always the MAIN checkout's venv (Python 3.12) by that absolute path; the worktree's own `.venv` is a stray Python 3.13 environment, never use it and never install into it. Never run whole-tree mypy or pytest. Never `git stash`, never amend, never squash. Commit trailers:

```
Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01QCr4zzEYNQDCrLaM4gFHGr
```

---

### Task 1: Loader timeout (item 1)

**Files:**
- Modify: `src/services/chat_capability_catalog.py` (constants after `DEGRADED_TTL_SECONDS`, and `build_capability_catalog`)
- Test: `tests/api/test_chat_capability_catalog.py` (after `test_empty_results_are_degraded_too`)

- [ ] **Step 1: Write the failing test**

Add `import time` next to `import asyncio` at the top of the test file, then after `test_empty_results_are_degraded_too`:

```python
async def test_stalled_loader_times_out_and_degrades(monkeypatch):
    """A loader that never answers is bounded by CATALOG_LOADER_TIMEOUT_SECONDS and
    marks its field degraded; the other loader still lands."""
    monkeypatch.setattr(cat, "CATALOG_LOADER_TIMEOUT_SECONDS", 0.05)

    async def stalled() -> List[Dict[str, Any]]:
        await asyncio.sleep(10)
        return await _coverage()

    started = time.monotonic()
    c = await make_catalog(coverage=stalled, outcomes=_outcomes)
    assert time.monotonic() - started < 2.0
    assert c.degraded == ("trend_coverage",)
    assert c.trend_kpi_ids == frozenset()
    assert c.causal_outcomes  # the outcomes loader was not affected
```

- [ ] **Step 2: Run it to verify it fails**

Run: `/home/enunez/Projects/e2i_causal_analytics/.venv/bin/python -m pytest tests/api/test_chat_capability_catalog.py -q -p no:cacheprovider -k stalled_loader`
Expected: FAIL with `AttributeError: ... has no attribute 'CATALOG_LOADER_TIMEOUT_SECONDS'`

- [ ] **Step 3: Implement**

After `DEGRADED_TTL_SECONDS = 60.0` add:

```python
# Each DB loader gets its own budget so a stalled connection cannot hold the
# refreshing request for the client's full connect+read timeouts (10 s + 30 s
# per loader). Measured 2026-09-06 on the droplet: 30-70 ms per query plus
# 130 ms client creation; one unreproduced 18 s cold read. A timeout is an
# ordinary exception below: the field is marked degraded, the last-good lists
# carry forward and the refresh is retried after DEGRADED_TTL_SECONDS. The two
# loaders stay sequential on purpose: one in-flight query at a time against a
# possibly stalled database, so a refresh costs at most twice this budget
# (#1901 item 1).
CATALOG_LOADER_TIMEOUT_SECONDS = 5.0
```

In `build_capability_catalog` replace the two loader awaits:

```python
    rows: List[Dict[str, Any]] = []
    try:
        rows = list(
            await asyncio.wait_for(
                (coverage_loader or _default_coverage_loader)(), CATALOG_LOADER_TIMEOUT_SECONDS
            )
        )
    except Exception as exc:  # noqa: BLE001 - degrade, never 502 the pills
        logger.warning(
            "capability catalog: trend coverage unavailable: %s: %s", type(exc).__name__, exc
        )
```

and

```python
    outcomes: List[str] = []
    try:
        outcomes = [
            str(o)
            for o in await asyncio.wait_for(
                (outcomes_loader or _default_outcomes_loader)(), CATALOG_LOADER_TIMEOUT_SECONDS
            )
            if o
        ]
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "capability catalog: causal outcomes unavailable: %s: %s", type(exc).__name__, exc
        )
```

(The `%s: %s` form is there because `asyncio.TimeoutError` stringifies to an empty string.)

- [ ] **Step 4: Run the file's tests**

Run: `/home/enunez/Projects/e2i_causal_analytics/.venv/bin/python -m pytest tests/api/test_chat_capability_catalog.py -q -p no:cacheprovider`
Expected: all PASS (previous count + 1).

- [ ] **Step 5: Lint and scoped type-check, then commit**

```bash
/home/enunez/Projects/e2i_causal_analytics/.venv/bin/ruff check src/services/chat_capability_catalog.py tests/api/test_chat_capability_catalog.py
/home/enunez/Projects/e2i_causal_analytics/.venv/bin/ruff format --check src/services/chat_capability_catalog.py tests/api/test_chat_capability_catalog.py
/home/enunez/Projects/e2i_causal_analytics/.venv/bin/mypy --config-file pyproject.toml src/services/chat_capability_catalog.py
git branch --show-current   # must print claude/pill-followups-1901
git add src/services/chat_capability_catalog.py tests/api/test_chat_capability_catalog.py
git commit -m "fix(chat): bound each catalog loader with its own timeout (#1901 item 1)"
```

---

### Task 2: Cache generation token (item 5)

**Files:**
- Modify: `src/services/chat_capability_catalog.py` (`_CatalogCache`)
- Test: `tests/api/test_chat_capability_catalog.py` (after `test_reset_mid_flight_discards_the_stale_build`)

- [ ] **Step 1: Write the failing test**

```python
async def test_eager_task_factory_publishes_and_clears():
    """Under asyncio.eager_task_factory a build whose loaders never suspend
    finishes inside ensure_future(); it must still publish and clear so an
    expired get() rebuilds instead of serving the first build forever."""
    cov, out = _Counting(_coverage), _Counting(_outcomes)
    c = cat._CatalogCache()
    loop = asyncio.get_running_loop()
    loop.set_task_factory(asyncio.eager_task_factory)
    try:
        first = await c.get(now=1000.0, coverage_loader=cov, outcomes_loader=out)
        assert c._catalog is first
        assert c._inflight is None
        second = await c.get(
            now=1000.0 + cat.CATALOG_TTL_SECONDS + 1, coverage_loader=cov, outcomes_loader=out
        )
    finally:
        loop.set_task_factory(None)
    assert second is not first
    assert c._catalog is second
    assert (cov.calls, out.calls) == (2, 2)
```

- [ ] **Step 2: Run it to verify it fails**

Run: `/home/enunez/Projects/e2i_causal_analytics/.venv/bin/python -m pytest tests/api/test_chat_capability_catalog.py -q -p no:cacheprovider -k eager_task_factory`
Expected: FAIL at `assert c._catalog is first` (the catalog is never published under the eager factory).

- [ ] **Step 3: Implement**

Replace the body of `_CatalogCache` from `__init__` through `reset` with:

```python
    def __init__(self) -> None:
        self._catalog: Optional[CapabilityCatalog] = None
        self._inflight: Optional["asyncio.Future[CapabilityCatalog]"] = None
        # The build allowed to publish, as a token rather than the task object:
        # under an eager task factory the build can finish inside
        # ensure_future(), before ``_inflight`` is even assigned, and a
        # task-identity check would then never publish nor clear (#1901 item 5).
        self._generation: Optional[object] = None

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
        if self._inflight is None:
            token = object()
            self._generation = token
            future = asyncio.ensure_future(
                self._refresh(token, cached, now, coverage_loader, outcomes_loader)
            )
            # An eager build that never suspended has already published and
            # cleared itself; storing its finished future would pin it forever.
            if not future.done():
                self._inflight = future
            return await asyncio.shield(future)
        return await asyncio.shield(self._inflight)

    async def _refresh(
        self,
        token: object,
        previous: Optional[CapabilityCatalog],
        now: Optional[float],
        coverage_loader: Optional[CoverageLoader],
        outcomes_loader: Optional[OutcomesLoader],
    ) -> CapabilityCatalog:
        try:
            fresh = await build_capability_catalog(
                coverage_loader=coverage_loader, outcomes_loader=outcomes_loader
            )
            fresh = _keep_last_good_fields(fresh, previous)
            if now is not None:
                fresh = dataclasses.replace(fresh, loaded_at=now)
            if self._generation is token:
                self._catalog = fresh
            return fresh
        finally:
            if self._generation is token:
                self._inflight = None
                self._generation = None

    def reset(self) -> None:
        self._catalog = None
        self._inflight = None
        self._generation = None
```

Update the class docstring's last sentence to: "A build that ``reset()`` orphaned mid-flight still serves its own waiters but neither writes the cache nor clears a newer build's future; the guard is a per-build token so an eager task factory (build finished inside ``ensure_future``) publishes too."

- [ ] **Step 4: Run the file's tests** (all cache tests must stay green)

Run: `/home/enunez/Projects/e2i_causal_analytics/.venv/bin/python -m pytest tests/api/test_chat_capability_catalog.py -q -p no:cacheprovider`
Expected: all PASS.

- [ ] **Step 5: Lint, scoped mypy, commit** (same commands as Task 1 Step 5)

```bash
git commit -m "fix(chat): catalog cache publishes by generation token, not task identity (#1901 item 5)"
```

---

### Task 3: Territory on-screen exemption (item 3)

**Files:**
- Modify: `src/services/chat_capability_catalog.py` (`_ON_SCREEN_ARTEFACT_RULES` and the comment above it)
- Test: `tests/api/test_chat_capability_catalog.py` (`KEEP_FIXTURES`, `DROP_FIXTURES`)

- [ ] **Step 1: Add the fixtures (failing first)**

Append to `KEEP_FIXTURES` (before the closing `]`):

```python
    (
        "Largest reallocation shown",
        "Which of the territories shown has the largest recommended reallocation?",
    ),
    (
        "On-screen territory table",
        "Read the on-screen territory table: which territory gains the most budget?",
    ),
```

Also (final review, commit `ed32bb5b6`): the deny-list `_EXTENDS_ON_SCREEN_RE` must name EVERY `AXIS_RULES` axis, and it omitted line of therapy, so once `territory_detail` joined the exemption "Break the on-screen territory allocation down by line of therapy." was kept (the same ask against SHAP/CATE was already kept). Add, directly after the existing `by (?:census |HCP )?(...)` alternation line, inside the same `re.compile`:

```python
    # "by" + up to two modifier words + an axis noun: "by patient severity tier",
    # "by US census region", "by prior therapy line", "by current line of therapy".
    r"\bby (?:[\w-]+ ){0,2}(?:region|territory|segment|tier|specialty|severity|biologic|IgE|cohort|subgroup)\w*|"
    r"\bby (?:[\w-]+ ){0,2}(?:lines?[- ]of[- ]therapy|therapy[- ]lines?|therapy_line|LoT)\b|"
```

(These two lines REPLACE the original `\bby (?:census |HCP )?(...)` line: codex iterations 2 and 3 found the hyphenated spelling and "by prior therapy line" kept, and measuring the class showed the fixed `census |HCP ` modifiers were the real gap — "by patient severity tier", "by US census region" and "by current line of therapy" were all kept inside the exemption. Up to two modifier words are tolerated; on-screen reads that use "by" without an axis noun ("by importance", "by far") are unaffected and no KEEP fixture changes — commits `0af5effbc`, `140ff6156`.)

and two more `DROP_FIXTURES` entries (last in the list):

```python
    (
        "territory_detail",
        "On-screen territories by therapy line",
        "Break the on-screen territory allocation down by line of therapy.",
    ),
    (
        "shap_or_feature_importance",
        "On-screen SHAP by therapy line",
        "Split the on-screen SHAP features by line of therapy.",
    ),
    (
        "territory_detail",
        "Displayed territories by line-of-therapy",
        "Break the displayed territory allocation down by line-of-therapy.",
    ),
    (
        "territory_detail",
        "Displayed territories by prior therapy line",
        "Break the displayed territory allocation down by prior therapy line.",
    ),
    (
        "shap_or_feature_importance",
        "On-screen SHAP by patient severity tier",
        "Split the on-screen SHAP features by patient severity tier.",
    ),
    (
        "territory_detail",
        "On-screen territories by US census region",
        "Break the on-screen territory allocation down by US census region.",
    ),
```

All six are red before the alternations (rule `None`); the hyphenated one is how this codebase spells the axis in prose, hence `[- ]` between the words. A plain catalog ask ("Compare TRx by line of therapy for Kisqali.") is unaffected: the deny-list is consulted only inside the exemption.

Append to `DROP_FIXTURES`:

```python
    (
        "territory_detail",
        "On-screen territories by region",
        "Break down the on-screen territory table by census region.",
    ),
    (
        "territory_detail",
        "Trend of territories shown",
        "Show the territory allocation trend over time for the territories shown.",
    ),
```

- [ ] **Step 2: Run to verify the two keeps fail**

Run: `/home/enunez/Projects/e2i_causal_analytics/.venv/bin/python -m pytest tests/api/test_chat_capability_catalog.py -q -p no:cacheprovider -k "supported_pills_are_kept or known_unsupported"`
Expected: 2 FAIL (the keeps are dropped as `territory_detail`), the drops PASS.

- [ ] **Step 3: Implement**

```python
_ON_SCREEN_ARTEFACT_RULES = frozenset(
    {
        "shap_or_feature_importance",
        "gap_recompute",
        "uplift_by_segment",
        "individual_prediction",
        "territory_detail",
    }
)
```

Replace the whole comment block above it (from "# Part C publishes the page summary" to the end of that block) with this text. It states the exemption's boundary rather than an absolute guarantee, refers to the rules by "listed below" instead of a count that goes stale, and says what the /resource-optimization summary does NOT carry:

```python
# Part C publishes the page summary to the assistant as a readable, so a pill
# MAY read, rank or compare SHAP, CATE, gap, prediction or optimizer territory
# values that are literally on screen (the pill prompt says so; the
# /resource-optimization summary publishes the allocation count, projected
# ROI and outcome, and the largest increase and decrease, not the territory
# table itself). The artefact rules listed below therefore yield when the
# question names the on-screen artefact AND asks for nothing that would
# extend it (another axis, a trend, a recomputation). The extends-list
# mirrors the pill prompt's own forbidden verbs (recompute, validate, extend,
# explain WHY) and the artefact rules' own trend/axis vocabulary, so the
# exemption drops the recompute, why, trend and by-axis shapes that list names
# (every AXIS_RULES axis plus territory, specialty, cohort and subgroup); an
# extension phrased outside that vocabulary is kept, and the exemption never
# proves the summary carries the row the pill names.
```

Also change the `match_unsupported_rule` docstring from "bypass the four artefact rules unless" to "bypass the artefact rules named in ``_ON_SCREEN_ARTEFACT_RULES`` unless", so the docstring refers to the set by name and the count cannot go stale.

- [ ] **Step 4: Run the file's tests** — all PASS.

- [ ] **Step 5: Lint, scoped mypy, commit**

```bash
git commit -m "fix(chat): on-screen territory reads survive the validator (#1901 item 3)"
```

---

### Task 4: Competitor no-preposition form (item 4d)

**Files:**
- Modify: `src/services/chat_capability_catalog.py` (`competitor_data` rule)
- Test: `tests/api/test_chat_capability_catalog.py`

- [ ] **Step 1: Add fixtures (failing first)**

`DROP_FIXTURES`:

```python
    (
        "competitor_data",
        "Outperforming the competition?",
        "Is Fabhalta outperforming the competition?",
    ),
    (
        "competitor_data",
        "Outperforming competitors on share",
        "Is Fabhalta outperforming competitors on market share?",
    ),
    (
        "competitor_data",
        "Beating the competition?",
        "Is Kisqali beating the competition?",
    ),
```

`KEEP_FIXTURES`:

```python
    (
        "MoA differences",
        "How does Fabhalta's mechanism of action differ from competitors'?",
    ),
    (
        "Competitor review",
        "Perform the competitor landscape review for Fabhalta's PNH indication.",
    ),
```

- [ ] **Step 2: Run** `-k "known_unsupported or supported_pills_are_kept"` — the "Outperforming …" drops FAIL (rule `None`), the "Beating …" drop FAILS too, and both keeps PASS on the old rule. (The "Competitor review" keep is the regression guard for the mandatory prefix below: with an optional `(?:out|under)?` it would be dropped, because bare "perform" before the noun is the transitive "perform the competitor review", a served section-E ask.)

- [ ] **Step 3: Implement** — add two alternations to the `competitor_data` pattern (after the existing third alternation, inside the same `re.compile`):

```python
            rf"|\b(?:out|under)perform\w*\s+{_COMPETITOR_NOUN}\b"
            rf"|\bbeat(?:s|ing|en)?\s+{_COMPETITOR_NOUN}\b",
```

(Keep `re.I`. The `out|under` prefix is MANDATORY: an optional prefix lets the bare transitive "perform the competitor landscape review" drop a served ask, and the bare-verb performance ask always carries a preposition that the second alternation already catches.) Extend the comment above `_COMPETITOR_DATA_WORDS` with these lines, after the three existing ones:

```python
# An out-/underperform or beat verb directly before the noun ("outperforming the
# competition") needs no data word: the ask is a commercial comparison the platform
# cannot make. Bare "perform" before the noun is the transitive "perform the
# competitor review" (served context), so the prefix is mandatory. A trial-endpoint
# ask in the verb-noun shape is an accepted false drop (it costs only a backfill
# pill); "compare versus competitors on <endpoint>" survives.
```

- [ ] **Step 4: Run the file's tests** — all PASS.

- [ ] **Step 5: Lint, scoped mypy, commit**

```bash
git commit -m "fix(chat): competitor rule catches 'outperforming the competition' (#1901 item 4d)"
```

---

### Task 5: Window wording from registry data (item 6)

**Files:**
- Modify: `src/services/chat_capability_catalog.py` (`CapabilityCatalog`, `build_capability_catalog`, `AXIS_RULES`, `render_catalog_block`)
- Test: `tests/api/test_chat_capability_catalog.py` (after `test_render_trend_and_axis_kpis_by_name`)

- [ ] **Step 1: Write the failing tests**

```python
async def test_windowable_kpis_come_from_the_registry():
    from src.kpi.registry import get_registry

    c = await make_catalog()
    expected = frozenset(
        k.id for k in get_registry().get_all() if k.windowable != "not_applicable"
    )
    assert expected, "the registry must declare windowable KPIs (TRx, share, triggers)"
    assert c.windowable_kpi_ids == expected
    assert "WS3-BI-005" in c.windowable_kpi_ids  # Total Prescriptions (TRx)
    assert "CM-002" not in c.windowable_kpi_ids  # Conditional ATE (CATE)


async def test_render_window_clause_names_only_windowable_kpis():
    c = await make_catalog()
    block = cat.render_catalog_block(c)
    clause = next((line for line in block.splitlines() if "applies ONLY to" in line), None)
    assert clause is not None, "section A must carry the window clause"
    for kpi_id in c.windowable_kpi_ids:
        assert c.kpi_name(kpi_id) in clause
    assert c.kpi_name("CM-002") not in clause
    assert "optionally over a time window" not in block
    # the axis rules keep the composition sentence but no longer promise a window
    assert "time window" not in cat.AXIS_RULES or "composes with" in cat.AXIS_RULES
```

- [ ] **Step 1b: Pin the composition sentence to the calculators (regression guard, DB-free)**

Add after `test_axis_vocabulary_matches_kpi_calculate_tool`. It passes on its own once Step 3 is in (it pins query-id routing, not prose), so it is a guard against the sentence going stale, not a red-first test. `calculate()` for WS3-BI-008/009 records the guard's RuntimeError in `result.error` instead of re-raising, and no query runs before the guard, so `db_client=None` is safe.

```python
async def test_axis_rules_window_composition_matches_calculators():
    """AXIS_RULES' composition sentence is prose; pin each clause to the
    calculators' query-id routing so the prompt cannot go stale against a
    migration again (the pre-#1900 wording had already gone stale against
    migration 120)."""
    from src.kpi.calculators.business_impact import BusinessImpactCalculator
    from src.kpi.calculators.trigger_performance import TriggerPerformanceCalculator
    from src.kpi.registry import get_registry
    from src.kpi.synthetic_mode import _SYNTHETIC_SUFFIX

    def suffix(query_id: str) -> str:
        # The showcase flag appends _include_synthetic to every additive
        # variant id; strip it so the assertions pin ROUTING, not the env.
        return (
            query_id[: -len(_SYNTHETIC_SUFFIX)]
            if query_id.endswith(_SYNTHETIC_SUFFIX)
            else query_id
        )

    class _NoQueries:
        """Sentinel client: any attribute access means a guard moved after its query."""

        def __getattr__(self, name: str) -> Any:
            raise AssertionError(f"routing test must not touch the DB client ({name})")

    window = {"start": "2025-01-01", "end": "2025-03-31"}
    # No query runs below: the routing helpers are pure and every guard raises
    # before _execute_query. The sentinel PROVES it - a guard that moved after
    # its query would reach this client and fail the test instead of the DB.
    calc = BusinessImpactCalculator(db_client=_NoQueries())

    # 1. Volume KPIs: a window composes with ANY ONE axis (migrations 084/105/108).
    region_qid, _ = calc._resolve_windowed_call(
        "business_impact_trx", brand="Kisqali", region="west", window=window, context={}
    )
    assert suffix(region_qid).endswith("_windowed_region")
    segment_qid, _ = calc._resolve_windowed_call(
        "business_impact_trx",
        brand="Kisqali",
        region=None,
        window=window,
        segment="high",
        context={},
    )
    assert suffix(segment_qid).endswith("_segment_windowed")
    # biologic is brand-gated to _BIOLOGIC_AXIS_BRANDS, so it needs Remibrutinib
    biologic_qid, _ = calc._resolve_windowed_call(
        "business_impact_trx",
        brand="Remibrutinib",
        region=None,
        window=window,
        biologic="naive",
        context={},
    )
    assert suffix(biologic_qid).endswith("_biologic_windowed")
    line_qid, _ = calc._resolve_windowed_call(
        "business_impact_trx",
        brand="Kisqali",
        region=None,
        window=window,
        therapy_line="1",
        context={},
    )
    assert suffix(line_qid).endswith("_line_windowed")
    # ige_tier is brand-gated like biologic
    ige_qid, _ = calc._resolve_windowed_call(
        "business_impact_trx",
        brand="Remibrutinib",
        region=None,
        window=window,
        ige_tier="high",
        context={},
    )
    assert suffix(ige_qid).endswith("_ige_tier_windowed")

    # 2. TRx Share / Conversion Rate: a window does NOT compose with region.
    for kpi_id in ("WS3-BI-008", "WS3-BI-009"):
        kpi = get_registry().get(kpi_id)
        assert kpi is not None, kpi_id
        context = {"brand": "Kisqali", "window": window, "region": "west"}
        guard = calc._calc_trx_share if kpi_id == "WS3-BI-008" else calc._calc_conversion_rate
        with pytest.raises(RuntimeError, match=kpi_id):
            guard(dict(context))
        # calculate() records the guard instead of re-raising, so the chat layer
        # reports the refusal rather than a fabricated number.
        result = calc.calculate(kpi, dict(context))
        assert result.value is None and kpi_id in (result.error or "")

    # 3. Trigger KPIs: a window composes with region (migration 120, #1388).
    trigger_qid, _ = TriggerPerformanceCalculator._effectiveness_scoped(
        "precision", {"brand": "Kisqali", "region": "west", "trigger_type": None, "window": window}
    )
    assert suffix(trigger_qid).endswith("_windowed_region")

    # The sentence's three families partition the windowable set exactly.
    c = await make_catalog()
    volume = {"WS3-BI-005", "WS3-BI-006", "WS3-BI-007"}
    share_conversion = {"WS3-BI-008", "WS3-BI-009"}
    assert volume <= c.windowable_kpi_ids
    assert share_conversion <= c.windowable_kpi_ids
    triggers = c.windowable_kpi_ids - volume - share_conversion
    assert triggers == {"WS2-TR-001", "WS2-TR-004", "WS2-TR-006", "WS2-TR-009"}
```

- [ ] **Step 2: Run** `-k "windowable or window_clause"` — both Step 1 tests FAIL (`AttributeError: windowable_kpi_ids`); the Step 1b test is not selected by that filter; run alone it fails the same way until Step 3 adds the field.

- [ ] **Step 3: Implement**

`CapabilityCatalog`: add after `axis_kpi_ids`:

```python
    windowable_kpi_ids: FrozenSet[str]  # registry windowable != "not_applicable" (code-derived)
```

Add a helper next to `_kpi_entries`:

```python
def _windowable_ids() -> FrozenSet[str]:
    """Registry KPIs the KPI tool can window; the rest answer window_status='not_applicable'."""
    return frozenset(k.id for k in get_registry().get_all() if k.windowable != "not_applicable")
```

`build_capability_catalog` return: add `windowable_kpi_ids=_windowable_ids(),` after `axis_kpi_ids=...`.

`AXIS_RULES`: replace the window sentence so the constant reads:

```python
AXIS_RULES = (
    "Breakdown axes, AT MOST ONE per ask: segment = patient severity tier (low/medium/high); "
    "therapy_line = line of therapy (0-3); region = US census region (northeast/south/midwest/west); "
    "and - Remibrutinib ONLY - biologic status (naive/experienced) or ige_tier (low/medium/high). "
    "The time window composes with any one axis for TRx, NRx and NBRx; only with segment/therapy_line "
    "for TRx Share and Conversion Rate; and only with region for Trigger Precision, Acceptance Rate, "
    "Override Rate and Trigger Funnel Conversion. "
    "TRx share is share of the tracked 3-brand portfolio, NOT share versus competitors."
)
```

(The composition sentence is per KPI family because the calculators differ: TRx/NRx/NBRx route a window with any one axis (`BusinessImpactCalculator._resolve_windowed_call`, migrations 084/105/108); TRx Share and Conversion Rate raise for window + region/biologic/ige_tier; the four windowable trigger KPIs route window + region to the migration-120 `_windowed_region` variant (#1388); they are named individually because "the trigger KPIs" would be ambiguous next to section A's nine trigger-performance KPIs and the Conversion Rate / Trigger Funnel Conversion name collision. A blanket "NOT with region/biologic/ige_tier" is true only for share/conversion, and the pre-#1900 wording was already stale for trigger + region. Step 1b pins each clause to the calculators.)

`render_catalog_block` section A: change the opening line and add the window clause before the axis rules:

```python
    lines.append("A. KPI values - the current value of any registry KPI, per brand. Registry KPIs by area:")
    for workstream, label in _WORKSTREAM_ORDER:
        ...  # unchanged
    window_names = _names(catalog, catalog.windowable_kpi_ids) or "none of the registry KPIs"
    lines.append(
        '   A time window ("last 3 months", "Q1 2025", "2025-01-01 to 2025-03-31") applies ONLY to '
        f"{window_names}; every other KPI answers its current value and reports that a window "
        "does not apply, so never promise a window for those."
    )
    lines.append("   " + AXIS_RULES)
```

Check `grep -n "CapabilityCatalog(" src/ tests/` still shows only the builder (no other constructor to update).

- [ ] **Step 4: Run the file's tests plus the suggestions tests**

Run: `/home/enunez/Projects/e2i_causal_analytics/.venv/bin/python -m pytest tests/api/test_chat_capability_catalog.py tests/api/test_chat_suggestions.py -n 0 -q -p no:cacheprovider`
Expected: all PASS (165 = 139 + 26 at the end of the branch).

- [ ] **Step 5: Lint, scoped mypy, commit**

```bash
git commit -m "feat(chat): section A names the windowable KPIs from the registry (#1901 item 6)"
```

---

### Task 6: Backend CI sees the router file (item 2)

**Files:**
- Modify: `.github/workflows/backend-tests.yml` (`push.paths` list and the `changes` job `PATTERN`)

- [ ] **Step 1: Edit both places**

In `push.paths` after `- 'tests/**'`:

```yaml
      # The catalog's route-hint coverage test reads the frontend router, so a
      # route add/rename must run backend CI on its own PR (#1901 item 2).
      - 'frontend/src/router/routes.tsx'
```

In the `changes` job:

```bash
          PATTERN='^(src/|tests/|frontend/src/router/routes\.tsx$|scripts/deploy_model\.py$|pyproject\.toml$|requirements\.txt$|\.github/workflows/backend-tests\.yml$)'
```

- [ ] **Step 2: Dry-run the pattern locally**

```bash
PATTERN='^(src/|tests/|frontend/src/router/routes\.tsx$|scripts/deploy_model\.py$|pyproject\.toml$|requirements\.txt$|\.github/workflows/backend-tests\.yml$)'
printf 'frontend/src/router/routes.tsx\n' | grep -qE "$PATTERN" && echo "router: runs backend CI"
printf 'frontend/src/pages/Overview.tsx\n' | grep -qE "$PATTERN" || echo "other frontend file: still skips"
```

Expected: both lines print.

- [ ] **Step 3: Commit**

```bash
git branch --show-current
git add .github/workflows/backend-tests.yml
git commit -m "ci: backend tests run when the frontend router changes (#1901 item 2)"
```

---

### Task 7: Ship

- [ ] **Step 1: Targeted tests and lint one more time** (`tests/api/test_chat_capability_catalog.py`, `tests/api/test_chat_suggestions.py`, `tests/api/test_copilotkit_readables_note.py`; ruff check + format --check on the two python files; scoped mypy).
- [ ] **Step 2: Codex audit** (`codex exec -m gpt-5.5 "<brief>" < /dev/null`, verdict only in the FINAL block; the brief carries the mandatory intent paragraph). Iterate until ACCEPT; each fix commits separately.
- [ ] **Step 3: Push and open the PR** against main, body ends with the required footer; `Refs #1901 (items 1, 2, 3, 4d, 5, 6)`; verify the body by read-back.
- [ ] **Step 4: CI green** (all required contexts; read the mypy report artifact if the gate reds).
- [ ] **Step 5: Merge only on the user's explicit go**, with `--merge`. Then: wait for the LAST deploy run on main to be terminal, verify container CONTENT markers (`CATALOG_LOADER_TIMEOUT_SECONDS`, `_generation`, `applies ONLY to` in the api container), rerun the probe (`probe_pills_post_merge.py` with a new output name), grade, and record the result; flip memory.
