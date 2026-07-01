# Treatment-effect Strategic Insight Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a 6th per-page strategic insight (`/insights/treatment-effect`) on the causal-analysis Treatment-effects tab, auto-generated when an estimate lands, and ship the accompanying client-timeout fix that lets the ~40s estimate reach the UI.

**Architecture:** Mirror the existing `causal-discovery` insight exactly — a `src/insights/treatment_effect.py` module (DSPy signature + `build_grounding` + `_fallback` + `generate_insight`) wired through a cached `POST /api/insights/treatment-effect` endpoint on the already-registered `insights_strategic_router`. Frontend adds an api fn + hook + a `StrategicInsightCard` on the Treatment-effects tab that auto-fetches on estimate completion (keyed on estimate identity via a ref).

**Tech Stack:** Python 3.12 / FastAPI / DSPy / redis (async); React 18 / TypeScript / @tanstack/react-query / axios / vitest.

**Branch:** `feat/causal-treatment-effect-insight` (already created; the timeout-fix code edits are already staged in the working tree — Task 1 tests + commits them).

---

## File structure

- **Modify** `frontend/src/lib/api-client.ts` — *(already edited)* `timeout?` on `ValidatedRequestConfig` + forward it in `get()`.
- **Modify** `frontend/src/api/causal.ts` — *(already edited)* `getTreatmentEffects` passes `{ timeout: 95000 }`; jsdoc copy.
- **Modify** `frontend/src/pages/CausalAnalysis.tsx` — *(copy already edited)* + insight card & auto-generate effect (Task 6).
- **Modify** `frontend/src/lib/api-client.test.ts` — timeout-forwarding test (Task 1).
- **Create** `src/insights/treatment_effect.py` — insight module (Task 2).
- **Modify** `src/api/routes/insights_strategic.py` — request model + endpoint + import (Task 3).
- **Modify** `tests/api/test_insights_strategic_routes.py` — route tests (Task 3).
- **Modify** `frontend/src/types/insights.ts` — request type (Task 4).
- **Modify** `frontend/src/api/insights.ts` — api fn (Task 4).
- **Modify** `frontend/src/api/insights.test.ts` — api test (Task 4).
- **Modify** `frontend/src/hooks/api/use-insights.ts` + `frontend/src/hooks/api/index.ts` — hook + export (Task 5).
- **Modify** `frontend/src/pages/CausalAnalysis.test.tsx` — auto-generate test (Task 6).

**No change needed** to `src/insights/__init__.py` (submodule imports work as-is) or `tests/unit/test_security/test_sentinel_external_unreachable.py` (the slim-app drift guard already includes `insights_strategic_router`; a new endpoint on it introduces no drift). The `"Strategic Insights"` openapi tag already exists.

---

## Task 1: Ship the client-timeout fix (already-staged edits)

**Files:**
- Modify (done): `frontend/src/lib/api-client.ts`, `frontend/src/api/causal.ts`, `frontend/src/pages/CausalAnalysis.tsx`
- Test: `frontend/src/lib/api-client.test.ts`

- [ ] **Step 1: Write the failing test** — add inside the `describe('get', ...)` block in `frontend/src/lib/api-client.test.ts`:

```typescript
    it('forwards a per-call timeout into the axios config', async () => {
      vi.mocked(apiClient.get).mockResolvedValueOnce({ data: { ok: true } } as never);
      await get('/heavy', { a: 1 }, { timeout: 95000 });
      expect(apiClient.get).toHaveBeenCalledWith('/heavy', {
        params: { a: 1 },
        timeout: 95000,
      });
    });

    it('omits timeout from the axios config when not provided', async () => {
      vi.mocked(apiClient.get).mockResolvedValueOnce({ data: { ok: true } } as never);
      await get('/light', { a: 1 });
      expect(apiClient.get).toHaveBeenCalledWith('/light', { params: { a: 1 } });
    });
```

- [ ] **Step 2: Run and verify the first test fails on current (unforwarded) code**

Run: `cd frontend && npx vitest run src/lib/api-client.test.ts -t "forwards a per-call timeout"`
Expected: If the `get()` forwarding edit were absent it FAILS (config lacks `timeout`). Since the edit is already in place, it PASSES — confirm both new tests PASS and the whole file is green.

Run: `cd frontend && npx vitest run src/lib/api-client.test.ts`
Expected: PASS (all).

- [ ] **Step 3: Verify the three already-made edits are present**

Run: `cd frontend && git diff --stat src/lib/api-client.ts src/api/causal.ts src/pages/CausalAnalysis.tsx`
Expected: all three listed as modified. Confirm `api-client.ts` has `timeout?: number` on `ValidatedRequestConfig` and the `...(options?.timeout !== undefined ? { timeout: options.timeout } : {})` spread in `get()`; `causal.ts` `getTreatmentEffects` passes `{ timeout: 95000 }`; both `~5-30s` copies now read `~10-90s`.

- [ ] **Step 4: Typecheck + lint the changed frontend files**

Run: `cd frontend && npx tsc --noEmit && npx eslint src/lib/api-client.ts src/api/causal.ts src/pages/CausalAnalysis.tsx src/lib/api-client.test.ts`
Expected: no errors.

- [ ] **Step 5: Commit**

```bash
git add frontend/src/lib/api-client.ts frontend/src/api/causal.ts frontend/src/pages/CausalAnalysis.tsx frontend/src/lib/api-client.test.ts
git commit -m "fix(causal): raise client timeout for the ~40s treatment-effect fit

The DoWhy+EconML treatment-effect fit takes ~40s (measured:
hcp_adoption/Remibrutinib, LinearDML, n=5000); the backend budgets 90s
and nginx allows 120-300s, but the axios client capped at 30s and aborted
first ('timeout of 30000ms exceeded'). Forward a per-call 95s timeout for
getTreatmentEffects and correct the '~5-30s' UI copy to '~10-90s'.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 2: Backend insight module `src/insights/treatment_effect.py`

**Files:**
- Create: `src/insights/treatment_effect.py`
- Test: `tests/api/test_insights_strategic_routes.py` (route-level, added in Task 3). This task adds the module; a quick import + unit smoke is done in Step 2 below.

- [ ] **Step 1: Write the module**

Create `src/insights/treatment_effect.py`:

```python
"""Treatment-effect strategic insight: interpret ONE de-confounded ATE cell."""

from __future__ import annotations

import logging
from typing import Any

from src.insights.common import normalize_list, run_signature

logger = logging.getLogger(__name__)

try:
    import dspy

    class TreatmentEffectInsightSignature(dspy.Signature):
        """Interpret ONE de-confounded average treatment effect for a brand analyst,
        STRICTLY grounded in the provided numbers. Use ONLY the ATE, CI, p-value, n,
        estimator, treatment/outcome, and confounders given; NEVER invent numbers or
        claim any refutation/robustness test was run. State the effect's magnitude and
        direction; judge ACTIONABILITY from whether the 95% CI excludes 0 (robust) or
        straddles 0 (not distinguishable from no effect), using p-value and n as
        supporting evidence; name the confounders adjusted for; and ALWAYS close with
        the caveat that this is a single model-based estimate whose robustness was NOT
        validated (refutation tests were not run)."""

        scope: str = dspy.InputField(desc="Cohort + brand for this estimate")
        estimate: str = dspy.InputField(
            desc="ATE [95% CI], p-value, n, estimator, and CI-vs-0 verdict"
        )
        design: str = dspy.InputField(
            desc="Treatment -> outcome and the confounders adjusted for"
        )

        interpretation: str = dspy.OutputField(
            desc="Grounded read of the effect, its actionability, and the robustness caveat"
        )
        key_takeaways: list = dspy.OutputField(desc="3-5 grounded, actionable takeaways")

    DSPY_AVAILABLE = True
except ImportError:
    DSPY_AVAILABLE = False
    TreatmentEffectInsightSignature = None  # type: ignore[assignment,misc]


def _fmt_num(v: Any, places: int = 4) -> str:
    try:
        return f"{float(v):+.{places}f}"
    except (TypeError, ValueError):
        return "—"


def _ci_str(lo: Any, hi: Any) -> str:
    if lo is None or hi is None:
        return "—"
    return f"[{_fmt_num(lo)}, {_fmt_num(hi)}]"


def _p_str(p: Any) -> str:
    if p is None:
        return "—"
    try:
        pv = float(p)
    except (TypeError, ValueError):
        return "—"
    return "< 0.001" if pv < 0.001 else f"{pv:.3f}"


def _ci_excludes_zero(lo: Any, hi: Any) -> bool | None:
    if lo is None or hi is None:
        return None
    try:
        return float(lo) > 0.0 or float(hi) < 0.0
    except (TypeError, ValueError):
        return None


def build_grounding(
    cohort: str,
    brand: str,
    treatment_var: str,
    outcome_var: str,
    confounders: list[str],
    ate: float,
    ci_lower: float | None,
    ci_upper: float | None,
    p_value: float | None,
    n: int,
    estimator: str | None,
) -> dict[str, Any]:
    excludes = _ci_excludes_zero(ci_lower, ci_upper)
    if excludes is None:
        verdict = "no CI available (single-estimator fallback)"
    elif excludes:
        verdict = "95% CI excludes 0 (distinguishable from no effect)"
    else:
        verdict = "95% CI straddles 0 (not distinguishable from no effect)"
    estimate = (
        f"ATE {_fmt_num(ate)} {_ci_str(ci_lower, ci_upper)}, "
        f"p={_p_str(p_value)}, n={n}, estimator={estimator or '—'}; {verdict}"
    )
    design = (
        f"{treatment_var} -> {outcome_var}; "
        f"adjusted for {', '.join(confounders) or 'none'}"
    )
    return {
        "scope": f"{cohort} / {brand}",
        "estimate": estimate,
        "design": design,
        "verdict": verdict,
        "grounding": [
            {"label": "ATE", "value": _fmt_num(ate)},
            {"label": "95% CI", "value": _ci_str(ci_lower, ci_upper)},
            {"label": "p", "value": _p_str(p_value)},
            {"label": "n", "value": str(n)},
        ],
    }


def _fallback(g: dict[str, Any]) -> dict[str, Any]:
    insight = (
        f"For {g['scope']}: {g['estimate']}. Design: {g['design']}. "
        "This is a single model-based estimate; its robustness was NOT validated "
        "(refutation tests were not run). "
        "(Factual summary — LLM interpretation unavailable.)"
    )
    return {
        "insight": insight,
        "key_takeaways": [g["verdict"], g["design"]],
        "grounding": g["grounding"],
        "is_fallback": True,
    }


def generate_insight(g: dict[str, Any]) -> dict[str, Any]:
    pred = run_signature(
        TreatmentEffectInsightSignature,
        scope=g["scope"],
        estimate=g["estimate"],
        design=g["design"],
    )
    if pred is None:
        return _fallback(g)
    interpretation = str(getattr(pred, "interpretation", "")).strip()
    if not interpretation:
        return _fallback(g)
    return {
        "insight": interpretation,
        "key_takeaways": normalize_list(getattr(pred, "key_takeaways", [])),
        "grounding": g["grounding"],
        "is_fallback": False,
    }
```

- [ ] **Step 2: Smoke-test the module in isolation (fallback path, no LLM)**

Run:
```bash
python -c "
from src.insights import treatment_effect as te
g = te.build_grounding('hcp_adoption','Remibrutinib','treatment_arm','adopted',
    ['peer_influence_score','influence_network_size'],0.1448,0.1426,0.1470,0.0004,5000,'linear_dml')
p = te._fallback(g)
assert p['is_fallback'] is True
assert 'refutation tests were not run' in p['insight']
assert p['grounding'][0]['label'] == 'ATE'
g2 = te.build_grounding('initiation','Fabhalta','treatment_arm','initiated_180d',['disease_severity'],0.01,-0.02,0.04,0.5,1200,'linear_dml')
assert 'not distinguishable from no effect' in te._fallback(g2)['insight']
print('OK')
"
```
Expected: prints `OK`.

- [ ] **Step 3: ruff format + check the new module**

Run: `ruff format src/insights/treatment_effect.py && ruff check src/insights/treatment_effect.py`
Expected: formatted, no lint errors.

- [ ] **Step 4: mypy the new module**

Run: `mypy src/insights/treatment_effect.py`
Expected: no errors. (Scoped to the changed file per the droplet policy — do NOT run whole-tree mypy here.)

- [ ] **Step 5: Commit**

```bash
git add src/insights/treatment_effect.py
git commit -m "feat(insights): treatment-effect insight module (signature + grounding + fallback)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 3: Backend endpoint + request model + route tests

**Files:**
- Modify: `src/api/routes/insights_strategic.py`
- Test: `tests/api/test_insights_strategic_routes.py`

- [ ] **Step 1: Write the failing route tests** — append to `tests/api/test_insights_strategic_routes.py`:

```python
def test_treatment_effect_insight_fallback(test_client):
    body = {
        "cohort": "hcp_adoption",
        "brand": "Remibrutinib",
        "treatment_var": "treatment_arm",
        "outcome_var": "adopted",
        "confounders": ["peer_influence_score", "influence_network_size"],
        "ate": 0.1448,
        "ci_lower": 0.1426,
        "ci_upper": 0.1470,
        "p_value": 0.0004,
        "n": 5000,
        "estimator": "linear_dml",
    }
    r = test_client.post("/api/insights/treatment-effect", json=body)
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["is_fallback"] is True
    assert "hcp_adoption" in data["insight"]
    assert "refutation tests were not run" in data["insight"]
    assert any(c["label"] == "ATE" for c in data["grounding"])
    assert data["provenance"]
    assert data["generated_at"]


def test_treatment_effect_insight_ci_straddles_zero(test_client):
    body = {
        "cohort": "initiation",
        "brand": "Fabhalta",
        "treatment_var": "treatment_arm",
        "outcome_var": "initiated_180d",
        "confounders": ["disease_severity"],
        "ate": 0.01,
        "ci_lower": -0.02,
        "ci_upper": 0.04,
        "p_value": 0.5,
        "n": 1200,
        "estimator": "linear_dml",
    }
    r = test_client.post("/api/insights/treatment-effect", json=body)
    assert r.status_code == 200, r.text
    data = r.json()
    assert "not distinguishable from no effect" in data["insight"]
```

- [ ] **Step 2: Run to verify they fail (404 — endpoint not yet defined)**

Run: `pytest tests/api/test_insights_strategic_routes.py -k treatment_effect -v`
Expected: FAIL — `assert 404 == 200` (route not registered yet).

- [ ] **Step 3: Add the import** — in `src/api/routes/insights_strategic.py`, extend the existing `from src.insights import (...)` block to include `treatment_effect` (keep alphabetical grouping consistent with ruff I-sort):

```python
from src.insights import (
    causal_discovery,
    knowledge_graph,
    model_performance,
    predictive_cohort,
    resource_optimization,
    treatment_effect,
)
```

- [ ] **Step 4: Add the request model** — after `class ResourceInsightRequest(...)` in the `# ---- Request models ----` section:

```python
class TreatmentEffectInsightRequest(BaseModel):
    cohort: str
    brand: str
    treatment_var: str
    outcome_var: str
    confounders: list[str] = Field(default_factory=list)
    ate: float
    ci_lower: float | None = None
    ci_upper: float | None = None
    p_value: float | None = None
    n: int
    estimator: str | None = None
```

- [ ] **Step 5: Add the endpoint** — after `causal_discovery_insight` (before `predictive_cohort_insight`):

```python
@router.post("/treatment-effect", response_model=StrategicInsightResponse)
async def treatment_effect_insight(
    req: TreatmentEffectInsightRequest, user: dict[str, Any] = Depends(require_analyst)
) -> StrategicInsightResponse:
    """Interpret a single de-confounded (cohort, brand) treatment-effect estimate."""
    g = treatment_effect.build_grounding(
        req.cohort,
        req.brand,
        req.treatment_var,
        req.outcome_var,
        list(req.confounders),
        req.ate,
        req.ci_lower,
        req.ci_upper,
        req.p_value,
        req.n,
        req.estimator,
    )
    key = cache_key(
        "treatment-effect",
        f"{req.cohort}/{req.brand}",
        {"ate": round(req.ate, 4), "n": req.n},
    )
    cached = await cache_get(key)
    if cached is not None:
        payload = cached
    else:
        payload = await asyncio.to_thread(treatment_effect.generate_insight, g)
        await cache_set(key, payload)
    return _finalize(payload, provenance="Live DoWhy+EconML treatment-effect fit")
```

- [ ] **Step 6: Run the route tests to verify they pass**

Run: `pytest tests/api/test_insights_strategic_routes.py -k treatment_effect -v`
Expected: PASS (both).

- [ ] **Step 7: Run the full insight-routes file (no regressions)**

Run: `pytest tests/api/test_insights_strategic_routes.py -v`
Expected: PASS (all).

- [ ] **Step 8: Confirm the slim-app drift guard still passes (no change expected)**

Run: `pytest tests/unit/test_security/test_sentinel_external_unreachable.py -v`
Expected: PASS — the endpoint is on the already-registered `insights_strategic_router`, so no drift.

- [ ] **Step 9: ruff (format + I-sort + check) and mypy on the changed file**

Run: `ruff format --check src/api/routes/insights_strategic.py && ruff check src/api/routes/insights_strategic.py && mypy src/api/routes/insights_strategic.py`
Expected: clean. (If `ruff format --check` reports a diff, run `ruff format src/api/routes/insights_strategic.py` and re-check.)

- [ ] **Step 10: Commit**

```bash
git add src/api/routes/insights_strategic.py tests/api/test_insights_strategic_routes.py
git commit -m "feat(insights): POST /insights/treatment-effect endpoint + route tests

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 4: Frontend api fn + request type + test

**Files:**
- Modify: `frontend/src/types/insights.ts`, `frontend/src/api/insights.ts`
- Test: `frontend/src/api/insights.test.ts`

- [ ] **Step 1: Add the request type** — append to `frontend/src/types/insights.ts`:

```typescript
export interface TreatmentEffectInsightRequest {
  cohort: string;
  brand: string;
  treatment_var: string;
  outcome_var: string;
  confounders: string[];
  ate: number;
  ci_lower?: number | null;
  ci_upper?: number | null;
  p_value?: number | null;
  n: number;
  estimator?: string | null;
}
```

- [ ] **Step 2: Write the failing api test** — append inside the `describe('insights api', ...)` block in `frontend/src/api/insights.test.ts`:

```typescript
  it('POSTs to /insights/treatment-effect and returns the response', async () => {
    const resp = {
      insight: 'x', key_takeaways: [], grounding: [], is_fallback: true,
      generated_at: 't', provenance: 'p',
    };
    const spy = vi.spyOn(apiClient, 'post').mockResolvedValue(resp as never);
    const out = await getTreatmentEffectInsight({
      cohort: 'hcp_adoption', brand: 'Remibrutinib', treatment_var: 'treatment_arm',
      outcome_var: 'adopted', confounders: [], ate: 0.14, n: 5000,
    });
    expect(spy).toHaveBeenCalledWith('/insights/treatment-effect', expect.any(Object));
    expect(out.is_fallback).toBe(true);
  });
```

Add the import at the top of the test file:

```typescript
import { getCausalDiscoveryInsight, getTreatmentEffectInsight } from './insights';
```
(replace the existing single-symbol import line).

- [ ] **Step 3: Run to verify it fails (symbol not exported)**

Run: `cd frontend && npx vitest run src/api/insights.test.ts -t "treatment-effect"`
Expected: FAIL — `getTreatmentEffectInsight` is not exported.

- [ ] **Step 4: Add the api fn** — in `frontend/src/api/insights.ts`, add `TreatmentEffectInsightRequest` to the type import block and append:

```typescript
export const getTreatmentEffectInsight = (r: TreatmentEffectInsightRequest) =>
  post<StrategicInsightResponse, TreatmentEffectInsightRequest>(`${BASE}/treatment-effect`, r);
```

- [ ] **Step 5: Run to verify it passes**

Run: `cd frontend && npx vitest run src/api/insights.test.ts`
Expected: PASS (all).

- [ ] **Step 6: Commit**

```bash
git add frontend/src/types/insights.ts frontend/src/api/insights.ts frontend/src/api/insights.test.ts
git commit -m "feat(insights): frontend treatment-effect insight api fn + request type

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 5: Frontend hook + index export

**Files:**
- Modify: `frontend/src/hooks/api/use-insights.ts`, `frontend/src/hooks/api/index.ts`

- [ ] **Step 1: Add the hook** — in `frontend/src/hooks/api/use-insights.ts`, add `getTreatmentEffectInsight` to the `@/api/insights` import, add `TreatmentEffectInsightRequest` to the `@/types/insights` import, and append:

```typescript
export const useTreatmentEffectInsight = () =>
  useMutation<StrategicInsightResponse, ApiError, TreatmentEffectInsightRequest>({
    mutationFn: getTreatmentEffectInsight,
  });
```

- [ ] **Step 2: Export it** — in `frontend/src/hooks/api/index.ts`, add `useTreatmentEffectInsight,` immediately after `useResourceOptimizationInsight,` (line ~186).

- [ ] **Step 3: Typecheck**

Run: `cd frontend && npx tsc --noEmit`
Expected: no errors.

- [ ] **Step 4: Commit**

```bash
git add frontend/src/hooks/api/use-insights.ts frontend/src/hooks/api/index.ts
git commit -m "feat(insights): useTreatmentEffectInsight hook + export

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 6: Page wiring — insight card + auto-generate on the Treatment-effects tab

**Files:**
- Modify: `frontend/src/pages/CausalAnalysis.tsx`
- Test: `frontend/src/pages/CausalAnalysis.test.tsx`

- [ ] **Step 1: Write the failing auto-generate test** — in `frontend/src/pages/CausalAnalysis.test.tsx`:

(a) add `useTreatmentEffectInsight: vi.fn(),` to the `vi.mock('@/hooks/api', () => ({ ... }))` object and to the subsequent `import { ... } from '@/hooks/api';` list;

(b) in `beforeEach`, after the `useTreatmentEffects` mock, add a default insight mock:

```typescript
    (useTreatmentEffectInsight as ReturnType<typeof vi.fn>).mockReturnValue({
      mutate: vi.fn(),
      isPending: false,
      error: null,
      data: undefined,
    });
```

(c) add the test (place near the other treatment-effects tests):

```typescript
  it('auto-generates the treatment-effect strategic insight once a result lands', async () => {
    const mutate = vi.fn();
    (useTreatmentEffectInsight as ReturnType<typeof vi.fn>).mockReturnValue({
      mutate,
      isPending: false,
      error: null,
      data: undefined,
    });
    (useTreatmentEffects as ReturnType<typeof vi.fn>).mockReturnValue({
      data: {
        cohort: 'hcp_adoption',
        brand: 'Remibrutinib',
        treatment_var: 'treatment_arm',
        outcome_var: 'adopted',
        confounders: ['peer_influence_score', 'influence_network_size'],
        ate: 0.1448,
        ci_lower: 0.1426,
        ci_upper: 0.147,
        p_value: 0.0004,
        std_error: 0.001,
        n: 5000,
        estimator: 'linear_dml',
        method: 'dowhy+econml sequential',
        confidence_level: 0.95,
        latency_ms: 40000,
        is_synthetic: true,
        warnings: ['robustness not validated'],
      },
      isFetching: false,
      isError: false,
      error: null,
    });
    render(<CausalAnalysis />, { wrapper: createWrapper() });
    await userEvent.click(screen.getByRole('tab', { name: /Treatment effects/i }));
    expect(mutate).toHaveBeenCalledTimes(1);
    expect(mutate).toHaveBeenCalledWith(
      expect.objectContaining({
        cohort: 'hcp_adoption',
        brand: 'Remibrutinib',
        ate: 0.1448,
        n: 5000,
      })
    );
  }, 20000);
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd frontend && npx vitest run src/pages/CausalAnalysis.test.tsx -t "auto-generates the treatment-effect"`
Expected: FAIL — `mutate` not called (wiring absent). (May also fail earlier if the `useTreatmentEffectInsight` mock isn't wired; that's expected pre-implementation.)

- [ ] **Step 3: Wire imports** — in `frontend/src/pages/CausalAnalysis.tsx`:
  - ensure the `from 'react'` import includes `useEffect` and `useRef` (add if missing);
  - add `useTreatmentEffectInsight` to the existing `from '@/hooks/api'` import that already lists `useCausalDiscoveryInsight` and `useTreatmentEffects`.

- [ ] **Step 4: Declare the hook + auto-generate effect** — immediately after the `useTreatmentEffects(...)` destructuring block (the `} = useTreatmentEffects(teCohort, teBrand, { enabled: teRun });` at ~line 317), add:

```typescript
  // Agentic strategic read of THIS estimate. Auto-generate once a fresh result
  // lands (keyed on the estimate identity so it fires once per distinct result,
  // not on every re-render). Manual re-generate stays available on the card.
  const teInsight = useTreatmentEffectInsight();
  const { mutate: generateTeInsight } = teInsight;
  const teInsightKeyRef = useRef<string | null>(null);
  useEffect(() => {
    if (!teData) return;
    const key = `${teData.cohort}-${teData.brand}-${teData.ate}`;
    if (teInsightKeyRef.current === key) return;
    teInsightKeyRef.current = key;
    generateTeInsight({
      cohort: teData.cohort,
      brand: teData.brand,
      treatment_var: teData.treatment_var,
      outcome_var: teData.outcome_var,
      confounders: teData.confounders,
      ate: teData.ate,
      ci_lower: teData.ci_lower ?? undefined,
      ci_upper: teData.ci_upper ?? undefined,
      p_value: teData.p_value ?? undefined,
      n: teData.n,
      estimator: teData.estimator ?? undefined,
    });
  }, [teData, generateTeInsight]);
```

- [ ] **Step 5: Render the card** — inside the result block `{!teFetching && !teIsError && teData && ( <div className="space-y-4"> ... )}`, immediately before that div's closing `</div>` (the one at ~line 1116, right after the `warnings` `{...}` block), insert:

```tsx
                  {/* Agentic strategic read of THIS estimate (auto-generated
                      when the result lands; grounded in the returned ATE/CI/p/n). */}
                  <StrategicInsightCard
                    title="Strategic insight"
                    description="Agentic interpretation of this treatment-effect estimate, grounded in the returned ATE, CI, p-value, and n."
                    isLoading={teInsight.isPending}
                    error={teInsight.error?.message ?? null}
                    insight={teInsight.data?.insight}
                    keyTakeaways={teInsight.data?.key_takeaways}
                    grounding={teInsight.data?.grounding}
                    isFallback={teInsight.data?.is_fallback}
                    provenance={teInsight.data?.provenance}
                    generatedAt={teInsight.data?.generated_at}
                    onGenerate={() =>
                      generateTeInsight({
                        cohort: teData.cohort,
                        brand: teData.brand,
                        treatment_var: teData.treatment_var,
                        outcome_var: teData.outcome_var,
                        confounders: teData.confounders,
                        ate: teData.ate,
                        ci_lower: teData.ci_lower ?? undefined,
                        ci_upper: teData.ci_upper ?? undefined,
                        p_value: teData.p_value ?? undefined,
                        n: teData.n,
                        estimator: teData.estimator ?? undefined,
                      })
                    }
                  />
```

(`StrategicInsightCard` is already imported at the top of the file; reuse it.)

- [ ] **Step 6: Run the page tests to verify green**

Run: `cd frontend && npx vitest run src/pages/CausalAnalysis.test.tsx`
Expected: PASS (all, including the new auto-generate test and the existing leaderboard-card test).

- [ ] **Step 7: Typecheck + lint**

Run: `cd frontend && npx tsc --noEmit && npx eslint src/pages/CausalAnalysis.tsx src/pages/CausalAnalysis.test.tsx`
Expected: no errors (esp. no `react-hooks/exhaustive-deps` warning — `generateTeInsight` is the stable react-query `mutate`).

- [ ] **Step 8: Commit**

```bash
git add frontend/src/pages/CausalAnalysis.tsx frontend/src/pages/CausalAnalysis.test.tsx
git commit -m "feat(causal): auto-generated strategic insight on the Treatment-effects tab

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 7: Full verification + PR

**Files:** none (verification + PR only)

- [ ] **Step 1: Backend — targeted tests + lint + type (changed files only)**

Run:
```bash
ruff format --check src/insights/treatment_effect.py src/api/routes/insights_strategic.py
ruff check src/insights/treatment_effect.py src/api/routes/insights_strategic.py
mypy src/insights/treatment_effect.py src/api/routes/insights_strategic.py
pytest tests/api/test_insights_strategic_routes.py tests/unit/test_security/test_sentinel_external_unreachable.py -v
```
Expected: all clean / PASS. (Scoped mypy/pytest per the droplet policy — CI runs the full suite.)

- [ ] **Step 2: Frontend — full typecheck, lint, and the touched test files**

Run:
```bash
cd frontend && npx tsc --noEmit && npx eslint src/ --max-warnings=0 && \
  npx vitest run src/lib/api-client.test.ts src/api/insights.test.ts src/pages/CausalAnalysis.test.tsx
```
Expected: no type/lint errors; tests PASS.

- [ ] **Step 3: Faithful end-to-end check on the droplet (real LLM path)**

The route tests only cover the deterministic fallback. Verify the live LLM path returns a grounded, non-fallback insight against the running `e2i_api` (auth via the same viewer/analyst path the UI uses). At minimum, confirm the endpoint responds 200 with grounding echoing the ATE for a known cell, e.g. by POSTing the `hcp_adoption`/`Remibrutinib` body used in the tests through an authenticated session. Record whether `is_fallback` is `false` (LLM configured) or `true` (no key) — both are honest; note which in the PR.

- [ ] **Step 4: Push the branch**

```bash
git config --global http.https://github.com.proxy ""
git push -u origin feat/causal-treatment-effect-insight
```

- [ ] **Step 5: Open the PR** (merge-commit policy — never squash)

```bash
gh pr create --title "feat(causal): treatment-effect strategic insight + client-timeout fix" --body "$(cat <<'EOF'
## What
On /causal-analysis → **Treatment effects**:
1. **Timeout fix** — the DoWhy+EconML fit legitimately takes ~40s (measured: hcp_adoption/Remibrutinib, LinearDML, n=5000); the backend budgets 90s and nginx allows 120–300s, but the axios client capped at 30s and aborted ("timeout of 30000ms exceeded"). Forward a per-call 95s timeout for `getTreatmentEffects`; correct the "~5-30s" copy to "~10-90s".
2. **Strategic insight** — a 6th per-page insight `POST /api/insights/treatment-effect`, mirroring `causal-discovery` (DSPy signature + grounded deterministic fallback + redis cache), rendered on the Treatment-effects tab and **auto-generated** when an estimate lands (grounded strictly in the returned ATE/CI/p/n, with a mandatory "robustness unvalidated — refutation not run" caveat).

## Testing
- Backend route tests (fallback path): grounded numbers echoed, `is_fallback=true`, CI-straddles-0 verdict; slim-app drift guard green.
- Frontend: api + hook + page auto-generate test; tsc + eslint clean.
- Droplet live-LLM check recorded in Step 3.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

---

## Self-review notes

- **Spec coverage:** dedicated module ✓ (T2); endpoint+model+cache+provenance ✓ (T3); frontend api/type/hook ✓ (T4/T5); card + auto-generate keyed on `cohort-brand-ate` + manual re-generate ✓ (T6); honesty (CI-vs-0 verdict + refutation caveat in signed & fallback text) ✓ (T2/T3 tests); timeout fix shipped together ✓ (T1); CI gotchas — ruff/mypy scoped ✓, drift guard verified no-op ✓, existing tag reused ✓.
- **Grain dropped** per the approved spec — `scope` is `cohort / brand` only.
- **Type consistency:** `build_grounding(cohort, brand, treatment_var, outcome_var, confounders, ate, ci_lower, ci_upper, p_value, n, estimator)` — identical positional order in the module (T2) and the endpoint call (T3). `TreatmentEffectInsightRequest` fields match backend↔frontend (T3↔T4). `generateTeInsight` is the destructured `mutate` used in both the effect and the card (T6).
