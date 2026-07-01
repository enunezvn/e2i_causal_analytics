# Strategic Insights (5 pages) + Brand Selector Fix — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the Home brand-selector overflow (remove the "Combined Portfolio" indication for the All option) and add a bespoke, LLM-grounded "Strategic Interpretation" card to five pages (knowledge-graph, causal-analysis, predictive-analytics, model-performance, resource-optimization).

**Architecture:** A shared frontend `StrategicInsightCard` renders a narrative + grounding chips. A new backend package `src/insights/` holds one bespoke DSPy signature per page plus a deterministic factual fallback; a new route module `src/api/routes/insights_strategic.py` exposes 5 POST endpoints (auth: `require_analyst`), grounds each insight in REAL data (server-side for the two always-on pages; from the backend-sourced run result for the three run pages), Redis-caches by input hash (~1h TTL), and returns a uniform `StrategicInsightResponse`. With no `OPENAI_API_KEY` (CI default) every endpoint returns the honest grounded fallback — never fabricated.

**Tech Stack:** FastAPI · DSPy (OpenAI via `ensure_dspy_configured`) · Redis · Supabase · React 18 + TypeScript · TanStack Query · shadcn/ui · pytest · Vitest.

---

## Conventions (apply to every task)

- **TDD rhythm per task:** write failing test → run it, confirm it FAILS for the expected reason → write minimal impl → run it, confirm PASS → commit.
- **Backend test cmd:** `python -m pytest <path>::<test> -v` (targeted; never whole-tree on the droplet).
- **Backend type check (changed file only):** `python -m mypy <changed_file.py>` (CI is the arbiter; do NOT run whole-tree mypy on the droplet — ~1.6 GiB spike).
- **Frontend test cmd:** `npm --prefix frontend run test -- run <path>`.
- **No CI per task.** CI runs once at the very end (Task 13). Commit locally after each task.
- **No mocking of insight values.** Backend tests run with `OPENAI_API_KEY` unset → they exercise the REAL deterministic fallback path (grounded in real inputs). The live LLM path is verified manually on the droplet (Task 12).
- **Commit message footer** (every commit):
  ```
  Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
  ```

## Shared Contracts (defined here; referenced by all tasks — keep names exact)

**Backend uniform response** (`src/insights/common.py`):
```python
# GroundingChip: {"label": str, "value": str}
# Insight payload dict returned by every generate_*_insight():
#   {"insight": str, "key_takeaways": list[str], "grounding": list[dict], "is_fallback": bool}
```
**API response model** (`src/api/routes/insights_strategic.py`):
```python
class GroundingChip(BaseModel):
    label: str
    value: str

class StrategicInsightResponse(BaseModel):
    insight: str
    key_takeaways: list[str] = Field(default_factory=list)
    grounding: list[GroundingChip] = Field(default_factory=list)
    is_fallback: bool
    generated_at: str
    provenance: str
```
**Endpoints** (router prefix `/insights`, mounted under `/api` → e.g. `POST /api/insights/knowledge-graph`):
`knowledge-graph`, `model-performance`, `causal-discovery`, `predictive-cohort`, `resource-optimization`.

**Frontend TS mirror** (`frontend/src/types/insights.ts`):
```typescript
export interface GroundingChip { label: string; value: string }
export interface StrategicInsightResponse {
  insight: string;
  key_takeaways: string[];
  grounding: GroundingChip[];
  is_fallback: boolean;
  generated_at: string;
  provenance: string;
}
```

---

## Task 1: Brand selector — remove "Combined Portfolio" + fix overflow

**Files:**
- Modify: `frontend/src/pages/Home.tsx` (BRANDS array ~line 115; SelectItem render ~lines 889-892)
- Test: `frontend/src/pages/Home.test.tsx` (indication test ~line 360)

- [ ] **Step 1 — Failing test.** Add to `Home.test.tsx`:
```tsx
it('shows "All Brands" without a Combined Portfolio indication', async () => {
  const user = userEvent.setup();
  renderHome(); // existing helper in this file
  await user.click(screen.getByRole('combobox', { name: /select brand/i }));
  // The All option renders its label but NO "(Combined Portfolio)" parenthetical
  const allOption = await screen.findByRole('option', { name: /all brands/i });
  expect(allOption).toBeInTheDocument();
  expect(allOption).not.toHaveTextContent(/combined portfolio/i);
});

it('still shows the indication for a specific brand', async () => {
  const user = userEvent.setup();
  renderHome();
  await user.click(screen.getByRole('combobox', { name: /select brand/i }));
  const remi = await screen.findByRole('option', { name: /remibrutinib/i });
  expect(remi).toHaveTextContent(/CSU/i);
});
```
  (If the file's existing render helper differs, use the one already in `Home.test.tsx`.)

- [ ] **Step 2 — Run, confirm FAIL.** `npm --prefix frontend run test -- run src/pages/Home.test.tsx` → the first test FAILS (option currently contains "Combined Portfolio").

- [ ] **Step 3 — Implement.** In `Home.tsx`:
  - Change the All entry (line ~115): `{ value: 'All', label: 'All Brands', indication: '', color: 'bg-slate-500' },`
  - In the SelectItem map (line ~889), guard the indication span and harden the row:
```tsx
<SelectItem key={brand.value} value={brand.value}>
  <div className="flex min-w-0 items-center gap-2">
    <div className={cn('w-2 h-2 rounded-full shrink-0', brand.color)} />
    <span className="truncate">{brand.label}</span>
    {brand.indication && (
      <span className="text-xs text-muted-foreground truncate">({brand.indication})</span>
    )}
  </div>
</SelectItem>
```

- [ ] **Step 4 — Run, confirm PASS.** `npm --prefix frontend run test -- run src/pages/Home.test.tsx` → both new tests pass; existing tests still green.

- [ ] **Step 5 — Commit.**
```bash
git add frontend/src/pages/Home.tsx frontend/src/pages/Home.test.tsx
git commit -m "fix(home): drop 'Combined Portfolio' label for All brands, fix selector overflow"
```

---

## Task 2: Backend insight scaffold (`src/insights/common.py`)

**Files:**
- Create: `src/insights/__init__.py`
- Create: `src/insights/common.py`
- Test: `tests/insights/__init__.py`, `tests/insights/test_common.py`

- [ ] **Step 1 — Failing test** (`tests/insights/test_common.py`):
```python
from src.insights.common import normalize_list, run_signature, cache_key


def test_normalize_list_from_str_splits_lines():
    assert normalize_list("- a\n- b\n- c") == ["a", "b", "c"]

def test_normalize_list_from_list_trims_and_caps():
    assert normalize_list([" x ", "", "y"]) == ["x", "y"]

def test_run_signature_returns_none_without_lm(monkeypatch):
    # No OPENAI_API_KEY -> ensure_dspy_configured() is False -> None (caller falls back)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert run_signature(object, foo="bar") is None

def test_cache_key_is_stable_and_input_sensitive():
    k1 = cache_key("knowledge-graph", "Kisqali", {"n": 10})
    k2 = cache_key("knowledge-graph", "Kisqali", {"n": 10})
    k3 = cache_key("knowledge-graph", "Kisqali", {"n": 11})
    assert k1 == k2 and k1 != k3
    assert k1.startswith("insight:knowledge-graph:")
```

- [ ] **Step 2 — Run, confirm FAIL.** `python -m pytest tests/insights/test_common.py -v` → ModuleNotFoundError.

- [ ] **Step 3 — Implement** (`src/insights/common.py`):
```python
"""Shared helpers for page-level strategic-insight generation.

Every insight is grounded in real, caller-provided numbers. When DSPy/the LM is
unavailable (e.g. no OPENAI_API_KEY in CI) run_signature returns None and the
caller renders a deterministic factual fallback — never fabricated content.
"""
from __future__ import annotations

import hashlib
import json
import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)


def normalize_list(value: Any, cap: int = 5) -> list[str]:
    """DSPy list outputs may arrive as a real list or a newline/`-`-delimited str."""
    if isinstance(value, str):
        items = [ln.strip(" -•\t") for ln in value.splitlines() if ln.strip()]
    elif isinstance(value, (list, tuple)):
        items = [str(i) for i in value]
    else:
        items = []
    return [s.strip() for s in items if s and s.strip()][:cap]


def run_signature(signature_cls: Any, **inputs: Any):
    """Run a DSPy ChainOfThought over `signature_cls`, or return None.

    Returns None when dspy is unavailable, no LM is configured (no API key), the
    signature is None, or the call raises — the caller then uses its factual
    fallback. BLOCKING: call from a worker thread (asyncio.to_thread).
    """
    if signature_cls is None:
        return None
    try:
        import dspy
    except ImportError:
        return None
    try:
        from src.optimization.dspy_lm import ensure_dspy_configured
        if not ensure_dspy_configured():
            logger.info("DSPy LM not configured (no API key); factual fallback")
            return None
        return dspy.ChainOfThought(signature_cls)(**inputs)
    except Exception as e:  # noqa: BLE001 — LLM failure must never break the request
        logger.warning("Strategic-insight LLM call failed (non-fatal): %s", e)
        return None


def cache_key(page: str, scope: str, inputs: dict[str, Any]) -> str:
    digest = hashlib.sha256(
        json.dumps(inputs, sort_keys=True, default=str).encode()
    ).hexdigest()[:16]
    return f"insight:{page}:{scope}:{digest}"


def cache_get(key: str) -> Optional[dict]:
    try:
        from src.memory.services.factories import get_redis_client
        raw = get_redis_client().get(key)
        return json.loads(raw) if raw else None
    except Exception as e:  # noqa: BLE001 — cache is best-effort
        logger.debug("insight cache_get miss/error: %s", e)
        return None


def cache_set(key: str, value: dict, ttl_seconds: int = 3600) -> None:
    try:
        from src.memory.services.factories import get_redis_client
        get_redis_client().setex(key, ttl_seconds, json.dumps(value, default=str))
    except Exception as e:  # noqa: BLE001 — cache is best-effort
        logger.debug("insight cache_set skipped: %s", e)
```
  Also create empty `src/insights/__init__.py` and `tests/insights/__init__.py`.

- [ ] **Step 4 — Run, confirm PASS.** `python -m pytest tests/insights/test_common.py -v`.

- [ ] **Step 5 — Commit.**
```bash
git add src/insights/__init__.py src/insights/common.py tests/insights/__init__.py tests/insights/test_common.py
git commit -m "feat(insights): shared scaffold (dspy runner, list normalize, redis cache)"
```

---

## Task 3: Knowledge-graph signature + grounding + fallback (`src/insights/knowledge_graph.py`)

**Files:**
- Create: `src/insights/knowledge_graph.py`
- Test: `tests/insights/test_knowledge_graph.py`

- [ ] **Step 1 — Failing test:**
```python
from src.insights.knowledge_graph import build_grounding, generate_insight

NODES = [
    {"id": "1", "name": "Adherence", "type": "Variable"},
    {"id": "2", "name": "NRx", "type": "KPI"},
    {"id": "3", "name": "Copay", "type": "Variable"},
]
RELS = [
    {"source_id": "3", "target_id": "1", "type": "CAUSES", "confidence": 0.82},
    {"source_id": "1", "target_id": "2", "type": "CAUSES", "confidence": 0.77},
]

def test_build_grounding_counts_and_chips():
    g = build_grounding("Kisqali", NODES, RELS, node_count=3, rel_count=2)
    assert "Variable" in g["node_summary"] and "KPI" in g["node_summary"]
    assert any(c["label"] == "Nodes" and c["value"] == "3" for c in g["grounding"])
    assert any(c["label"] == "Relationships" and c["value"] == "2" for c in g["grounding"])

def test_generate_insight_fallback_is_grounded_without_lm(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    g = build_grounding("Kisqali", NODES, RELS, node_count=3, rel_count=2)
    out = generate_insight(g)
    assert out["is_fallback"] is True
    assert "3" in out["insight"] and "Kisqali" in out["insight"]  # real numbers, no fabrication
    assert isinstance(out["key_takeaways"], list)
```

- [ ] **Step 2 — Run, confirm FAIL.** `python -m pytest tests/insights/test_knowledge_graph.py -v`.

- [ ] **Step 3 — Implement** (`src/insights/knowledge_graph.py`):
```python
"""Knowledge-graph strategic insight: interpret the curated KG for a brand."""
from __future__ import annotations

import logging
from collections import Counter
from typing import Any

from src.insights.common import normalize_list, run_signature

logger = logging.getLogger(__name__)

try:
    import dspy

    class KnowledgeGraphInsightSignature(dspy.Signature):
        """Interpret a curated pharmaceutical knowledge graph for a brand analyst,
        STRICTLY grounded in the provided counts and entity names. Use ONLY the
        numbers and names given; NEVER invent nodes, edges, or confidence values.
        Explain what the structure implies about causal drivers/levers; if the graph
        is sparse, say so plainly rather than over-reading it."""

        scope: str = dspy.InputField(desc="Brand/region scope of this graph view")
        node_summary: str = dspy.InputField(desc="Node counts by type and total")
        top_hubs: str = dspy.InputField(desc="Highest-degree entities: name, type, degree")
        key_paths: str = dspy.InputField(desc="Notable CAUSES/INFLUENCES chains + confidence")
        edge_summary: str = dspy.InputField(desc="Relationship counts by type + confidence range")

        interpretation: str = dspy.OutputField(
            desc="What the structure says about causal drivers/levers for this brand"
        )
        key_takeaways: list = dspy.OutputField(desc="3-5 specific, grounded takeaways")

    DSPY_AVAILABLE = True
except ImportError:
    DSPY_AVAILABLE = False
    KnowledgeGraphInsightSignature = None  # type: ignore[assignment,misc]


def build_grounding(
    scope: str,
    nodes: list[dict[str, Any]],
    relationships: list[dict[str, Any]],
    node_count: int,
    rel_count: int,
) -> dict[str, Any]:
    node_types = Counter(n.get("type", "Unknown") for n in nodes)
    node_summary = f"{node_count} nodes total: " + ", ".join(
        f"{t}={c}" for t, c in node_types.most_common()
    )
    degree: Counter = Counter()
    name_by_id = {n.get("id"): n.get("name", n.get("id")) for n in nodes}
    type_by_id = {n.get("id"): n.get("type", "Unknown") for n in nodes}
    for r in relationships:
        degree[r.get("source_id")] += 1
        degree[r.get("target_id")] += 1
    top_hubs = "; ".join(
        f"{name_by_id.get(nid, nid)} ({type_by_id.get(nid, '?')}, degree {d})"
        for nid, d in degree.most_common(5)
    ) or "none"
    key_paths = "; ".join(
        f"{name_by_id.get(r.get('source_id'), r.get('source_id'))} -{r.get('type')}-> "
        f"{name_by_id.get(r.get('target_id'), r.get('target_id'))} "
        f"(conf {float(r.get('confidence') or 0):.2f})"
        for r in relationships[:6]
    ) or "none"
    edge_types = Counter(r.get("type", "?") for r in relationships)
    confs = [float(r.get("confidence") or 0) for r in relationships if r.get("confidence")]
    edge_summary = f"{rel_count} relationships: " + ", ".join(
        f"{t}={c}" for t, c in edge_types.most_common()
    )
    if confs:
        edge_summary += f"; confidence {min(confs):.2f}-{max(confs):.2f}"
    return {
        "scope": scope,
        "node_summary": node_summary,
        "top_hubs": top_hubs,
        "key_paths": key_paths,
        "edge_summary": edge_summary,
        "grounding": [
            {"label": "Nodes", "value": str(node_count)},
            {"label": "Relationships", "value": str(rel_count)},
            {"label": "Node types", "value": str(len(node_types))},
        ],
    }


def _fallback(g: dict[str, Any]) -> dict[str, Any]:
    insight = (
        f"For {g['scope']}, the curated graph holds {g['node_summary']}. "
        f"Highest-connectivity entities: {g['top_hubs']}. "
        f"Key causal links: {g['key_paths']}. "
        f"Edge profile: {g['edge_summary']}. "
        "(Factual summary — LLM interpretation unavailable.)"
    )
    return {
        "insight": insight,
        "key_takeaways": [g["node_summary"], f"Top hubs: {g['top_hubs']}"],
        "grounding": g["grounding"],
        "is_fallback": True,
    }


def generate_insight(g: dict[str, Any]) -> dict[str, Any]:
    """LLM interpretation grounded in `g`, or a deterministic factual fallback."""
    pred = run_signature(
        KnowledgeGraphInsightSignature,
        scope=g["scope"],
        node_summary=g["node_summary"],
        top_hubs=g["top_hubs"],
        key_paths=g["key_paths"],
        edge_summary=g["edge_summary"],
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

- [ ] **Step 4 — Run, confirm PASS.** `python -m pytest tests/insights/test_knowledge_graph.py -v`.

- [ ] **Step 5 — Commit.**
```bash
git add src/insights/knowledge_graph.py tests/insights/test_knowledge_graph.py
git commit -m "feat(insights): knowledge-graph signature, grounding, factual fallback"
```

---

## Task 4: Model-performance signature + grounding + fallback (`src/insights/model_performance.py`)

**Files:**
- Create: `src/insights/model_performance.py`
- Test: `tests/insights/test_model_performance.py`

- [ ] **Step 1 — Failing test:**
```python
from src.insights.model_performance import build_grounding, generate_insight

def test_build_grounding_derives_prf_and_chips():
    g = build_grounding(
        model_version="csu_adherence_v3",
        current_accuracy=0.86, baseline_accuracy=0.81, trend="improving",
        confusion={"tn": 80, "fp": 10, "fn": 12, "tp": 98},
        auc=0.88,
        alerts=[{"metric_name": "precision", "severity": "warning"}],
    )
    assert any(c["label"] == "Accuracy" and c["value"].startswith("0.86") for c in g["grounding"])
    assert "precision" in g["confusion_summary"].lower()

def test_generate_insight_fallback_grounded(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    g = build_grounding("m1", 0.86, 0.81, "improving",
                        {"tn": 80, "fp": 10, "fn": 12, "tp": 98}, 0.88, [])
    out = generate_insight(g)
    assert out["is_fallback"] is True
    assert "0.86" in out["insight"] and "0.88" in out["insight"]
```

- [ ] **Step 2 — Run, confirm FAIL.** `python -m pytest tests/insights/test_model_performance.py -v`.

- [ ] **Step 3 — Implement** (`src/insights/model_performance.py`):
```python
"""Model-performance strategic insight: diagnose a model's health + next action."""
from __future__ import annotations

import logging
from typing import Any

from src.insights.common import normalize_list, run_signature

logger = logging.getLogger(__name__)

try:
    import dspy

    class ModelPerformanceInsightSignature(dspy.Signature):
        """Diagnose a deployed classifier's health for an ML/commercial analyst,
        STRICTLY grounded in the provided metrics. Use ONLY the numbers given; never
        invent metrics or thresholds. State whether the model is healthy vs degrading,
        what the confusion/ROC imply (e.g. precision vs recall trade-off), and the
        single most appropriate next action (monitor / retrain / investigate drift)."""

        model_version: str = dspy.InputField(desc="Model version/identifier")
        accuracy_summary: str = dspy.InputField(desc="Current vs baseline accuracy + trend")
        confusion_summary: str = dspy.InputField(desc="Precision, recall, specificity, F1 + counts")
        auc_summary: str = dspy.InputField(desc="ROC AUC")
        alerts_summary: str = dspy.InputField(desc="Active performance alerts (or none)")

        interpretation: str = dspy.OutputField(desc="Health diagnosis grounded in the metrics")
        key_takeaways: list = dspy.OutputField(desc="3-5 grounded takeaways incl. recommended action")

    DSPY_AVAILABLE = True
except ImportError:
    DSPY_AVAILABLE = False
    ModelPerformanceInsightSignature = None  # type: ignore[assignment,misc]


def _prf(cm: dict[str, Any]) -> dict[str, float]:
    tp, fp, fn, tn = (float(cm.get(k, 0)) for k in ("tp", "fp", "fn", "tn"))
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    spec = tn / (tn + fp) if (tn + fp) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return {"precision": prec, "recall": rec, "specificity": spec, "f1": f1}


def build_grounding(
    model_version: str,
    current_accuracy: float,
    baseline_accuracy: float,
    trend: str,
    confusion: dict[str, Any] | None,
    auc: float | None,
    alerts: list[dict[str, Any]] | None,
) -> dict[str, Any]:
    delta = float(current_accuracy) - float(baseline_accuracy)
    accuracy_summary = (
        f"accuracy {current_accuracy:.3f} vs baseline {baseline_accuracy:.3f} "
        f"(Δ{delta:+.3f}), trend {trend}"
    )
    chips = [
        {"label": "Accuracy", "value": f"{current_accuracy:.3f}"},
        {"label": "Baseline", "value": f"{baseline_accuracy:.3f}"},
        {"label": "Trend", "value": str(trend)},
    ]
    if confusion:
        m = _prf(confusion)
        confusion_summary = (
            f"precision {m['precision']:.2f}, recall {m['recall']:.2f}, "
            f"specificity {m['specificity']:.2f}, F1 {m['f1']:.2f} "
            f"(TP={confusion.get('tp')}, FP={confusion.get('fp')}, "
            f"FN={confusion.get('fn')}, TN={confusion.get('tn')})"
        )
        chips.append({"label": "F1", "value": f"{m['f1']:.2f}"})
    else:
        confusion_summary = "no confusion matrix available"
    auc_summary = f"ROC AUC {auc:.3f}" if auc is not None else "no ROC curve available"
    if auc is not None:
        chips.append({"label": "AUC", "value": f"{auc:.3f}"})
    alerts = alerts or []
    alerts_summary = (
        "; ".join(f"{a.get('metric_name')} ({a.get('severity')})" for a in alerts)
        if alerts else "no active alerts"
    )
    return {
        "model_version": model_version,
        "accuracy_summary": accuracy_summary,
        "confusion_summary": confusion_summary,
        "auc_summary": auc_summary,
        "alerts_summary": alerts_summary,
        "grounding": chips,
    }


def _fallback(g: dict[str, Any]) -> dict[str, Any]:
    insight = (
        f"Model {g['model_version']}: {g['accuracy_summary']}. "
        f"{g['confusion_summary']}. {g['auc_summary']}. Alerts: {g['alerts_summary']}. "
        "(Factual summary — LLM interpretation unavailable.)"
    )
    return {
        "insight": insight,
        "key_takeaways": [g["accuracy_summary"], g["confusion_summary"]],
        "grounding": g["grounding"],
        "is_fallback": True,
    }


def generate_insight(g: dict[str, Any]) -> dict[str, Any]:
    pred = run_signature(
        ModelPerformanceInsightSignature,
        model_version=g["model_version"],
        accuracy_summary=g["accuracy_summary"],
        confusion_summary=g["confusion_summary"],
        auc_summary=g["auc_summary"],
        alerts_summary=g["alerts_summary"],
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

- [ ] **Step 4 — Run, confirm PASS.** `python -m pytest tests/insights/test_model_performance.py -v`.

- [ ] **Step 5 — Commit.**
```bash
git add src/insights/model_performance.py tests/insights/test_model_performance.py
git commit -m "feat(insights): model-performance signature, P/R/F1 grounding, fallback"
```

---

## Task 5: Causal-discovery signature + grounding + fallback (`src/insights/causal_discovery.py`)

**Files:**
- Create: `src/insights/causal_discovery.py`
- Test: `tests/insights/test_causal_discovery.py`

- [ ] **Step 1 — Failing test:**
```python
from src.insights.causal_discovery import build_grounding, generate_insight

EFFECTS = [
    {"treatment": "copay_card", "outcome": "adherence_180d", "ate": 0.043,
     "ate_ci_lower": 0.02, "ate_ci_upper": 0.066, "status": "proceed",
     "selected_estimator": "CausalForestDML"},
    {"treatment": "nurse_call", "outcome": "adherence_180d", "ate": 0.011,
     "ate_ci_lower": -0.01, "ate_ci_upper": 0.03, "status": "review",
     "selected_estimator": "LinearDML"},
]

def test_build_grounding_ranks_and_counts_gates():
    g = build_grounding("Kisqali", "patient", EFFECTS)
    assert "proceed" in g["gate_summary"] and "review" in g["gate_summary"]
    assert any(c["label"] == "Effects" and c["value"] == "2" for c in g["grounding"])

def test_generate_insight_fallback_grounded(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    g = build_grounding("Kisqali", "patient", EFFECTS)
    out = generate_insight(g)
    assert out["is_fallback"] is True
    assert "copay_card" in out["insight"]
```

- [ ] **Step 2 — Run, confirm FAIL.** `python -m pytest tests/insights/test_causal_discovery.py -v`.

- [ ] **Step 3 — Implement** (`src/insights/causal_discovery.py`):
```python
"""Causal-discovery strategic insight: portfolio-level read of discovered effects."""
from __future__ import annotations

import logging
from collections import Counter
from typing import Any

from src.insights.common import normalize_list, run_signature

logger = logging.getLogger(__name__)

try:
    import dspy

    class CausalDiscoveryInsightSignature(dspy.Signature):
        """Interpret a leaderboard of agent-validated causal effects for a brand
        analyst, STRICTLY grounded in the provided effects. Use ONLY the treatments,
        outcomes, ATEs, CIs, gate statuses, and estimators given; NEVER invent effects
        or numbers. Emphasise which effects are robust and ACTIONABLE (gate=proceed,
        CI excludes 0) vs which need review; if none are robust, say so plainly."""

        scope: str = dspy.InputField(desc="Brand + analysis grain")
        effects_table: str = dspy.InputField(
            desc="Ranked effects: treatment->outcome, ATE [CI], gate, estimator"
        )
        gate_summary: str = dspy.InputField(desc="Counts by gate status")

        interpretation: str = dspy.OutputField(
            desc="Which effects to act on and why, grounded in ATE/CI/gate"
        )
        key_takeaways: list = dspy.OutputField(desc="3-5 grounded, actionable takeaways")

    DSPY_AVAILABLE = True
except ImportError:
    DSPY_AVAILABLE = False
    CausalDiscoveryInsightSignature = None  # type: ignore[assignment,misc]


def build_grounding(brand: str, grain: str, effects: list[dict[str, Any]]) -> dict[str, Any]:
    def _rank(e: dict[str, Any]) -> float:
        return abs(float(e.get("ate") or 0))
    ranked = sorted(effects, key=_rank, reverse=True)
    rows = []
    for e in ranked[:8]:
        rows.append(
            f"{e.get('treatment')}->{e.get('outcome')}: "
            f"ATE {float(e.get('ate') or 0):+.3f} "
            f"[{float(e.get('ate_ci_lower') or 0):+.3f}, {float(e.get('ate_ci_upper') or 0):+.3f}], "
            f"gate={e.get('status')}, est={e.get('selected_estimator')}"
        )
    effects_table = "\n".join(rows) or "no effects discovered"
    gates = Counter(e.get("status", "unknown") for e in effects)
    gate_summary = ", ".join(f"{g}={c}" for g, c in gates.most_common()) or "none"
    return {
        "scope": f"{brand} / {grain}",
        "effects_table": effects_table,
        "gate_summary": gate_summary,
        "grounding": [
            {"label": "Effects", "value": str(len(effects))},
            {"label": "Proceed", "value": str(gates.get("proceed", 0))},
            {"label": "Review", "value": str(gates.get("review", 0))},
        ],
    }


def _fallback(g: dict[str, Any]) -> dict[str, Any]:
    insight = (
        f"For {g['scope']}, discovered effects (by |ATE|):\n{g['effects_table']}\n"
        f"Gate distribution: {g['gate_summary']}. "
        "(Factual summary — LLM interpretation unavailable.)"
    )
    first_line = g["effects_table"].splitlines()[0] if g["effects_table"] else g["gate_summary"]
    return {
        "insight": insight,
        "key_takeaways": [f"Gates: {g['gate_summary']}", first_line],
        "grounding": g["grounding"],
        "is_fallback": True,
    }


def generate_insight(g: dict[str, Any]) -> dict[str, Any]:
    pred = run_signature(
        CausalDiscoveryInsightSignature,
        scope=g["scope"],
        effects_table=g["effects_table"],
        gate_summary=g["gate_summary"],
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

- [ ] **Step 4 — Run, confirm PASS.** `python -m pytest tests/insights/test_causal_discovery.py -v`.

- [ ] **Step 5 — Commit.**
```bash
git add src/insights/causal_discovery.py tests/insights/test_causal_discovery.py
git commit -m "feat(insights): causal-discovery leaderboard signature, grounding, fallback"
```

---

## Task 6: Predictive-cohort signature + grounding + fallback (`src/insights/predictive_cohort.py`)

**Files:**
- Create: `src/insights/predictive_cohort.py`
- Test: `tests/insights/test_predictive_cohort.py`

- [ ] **Step 1 — Failing test:**
```python
from src.insights.predictive_cohort import build_grounding, generate_insight

def test_build_grounding_summarizes_distribution_and_drivers():
    g = build_grounding(
        model_version="csu_adherence_v3", n_scored=250, mean_prob=0.34,
        top_targets=[{"entity_id": "HCP7", "probability": 0.91},
                     {"entity_id": "HCP3", "probability": 0.88}],
        top_drivers=[{"feature": "prior_adherence", "importance": 0.4},
                     {"feature": "copay", "importance": 0.25}],
    )
    assert any(c["label"] == "Scored" and c["value"] == "250" for c in g["grounding"])
    assert "prior_adherence" in g["drivers_summary"]

def test_generate_insight_fallback_grounded(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    g = build_grounding("m1", 250, 0.34,
                        [{"entity_id": "HCP7", "probability": 0.91}],
                        [{"feature": "prior_adherence", "importance": 0.4}])
    out = generate_insight(g)
    assert out["is_fallback"] is True
    assert "HCP7" in out["insight"] and "250" in out["insight"]
```

- [ ] **Step 2 — Run, confirm FAIL.** `python -m pytest tests/insights/test_predictive_cohort.py -v`.

- [ ] **Step 3 — Implement** (`src/insights/predictive_cohort.py`):
```python
"""Predictive-cohort strategic insight: who to target and why (cohort + SHAP)."""
from __future__ import annotations

import logging
from typing import Any

from src.insights.common import normalize_list, run_signature

logger = logging.getLogger(__name__)

try:
    import dspy

    class PredictiveCohortInsightSignature(dspy.Signature):
        """Turn a scored out-of-sample cohort into a targeting read for a commercial
        analyst, STRICTLY grounded in the provided numbers. Use ONLY the score
        distribution, named top targets, and SHAP driver importances given; NEVER
        invent entities, probabilities, or features. Say who to prioritise, what
        drives their scores, and how confident the ranking is."""

        model_version: str = dspy.InputField(desc="Scoring model version")
        distribution_summary: str = dspy.InputField(desc="n scored, mean probability")
        top_targets_summary: str = dspy.InputField(desc="Top-ranked entities with probabilities")
        drivers_summary: str = dspy.InputField(desc="Top SHAP feature drivers + importances")

        interpretation: str = dspy.OutputField(desc="Targeting read grounded in the numbers")
        key_takeaways: list = dspy.OutputField(desc="3-5 grounded targeting takeaways")

    DSPY_AVAILABLE = True
except ImportError:
    DSPY_AVAILABLE = False
    PredictiveCohortInsightSignature = None  # type: ignore[assignment,misc]


def build_grounding(
    model_version: str,
    n_scored: int,
    mean_prob: float,
    top_targets: list[dict[str, Any]],
    top_drivers: list[dict[str, Any]],
) -> dict[str, Any]:
    distribution_summary = f"{n_scored} entities scored, mean probability {mean_prob:.3f}"
    top_targets_summary = "; ".join(
        f"{t.get('entity_id')} ({float(t.get('probability') or 0):.2f})" for t in top_targets[:5]
    ) or "none"
    drivers_summary = "; ".join(
        f"{d.get('feature')} ({float(d.get('importance') or 0):.2f})" for d in top_drivers[:5]
    ) or "none"
    return {
        "model_version": model_version,
        "distribution_summary": distribution_summary,
        "top_targets_summary": top_targets_summary,
        "drivers_summary": drivers_summary,
        "grounding": [
            {"label": "Scored", "value": str(n_scored)},
            {"label": "Mean p", "value": f"{mean_prob:.3f}"},
            {"label": "Top targets", "value": str(min(len(top_targets), 5))},
        ],
    }


def _fallback(g: dict[str, Any]) -> dict[str, Any]:
    insight = (
        f"Model {g['model_version']}: {g['distribution_summary']}. "
        f"Highest-probability targets: {g['top_targets_summary']}. "
        f"Main drivers: {g['drivers_summary']}. "
        "(Factual summary — LLM interpretation unavailable.)"
    )
    return {
        "insight": insight,
        "key_takeaways": [g["distribution_summary"], f"Drivers: {g['drivers_summary']}"],
        "grounding": g["grounding"],
        "is_fallback": True,
    }


def generate_insight(g: dict[str, Any]) -> dict[str, Any]:
    pred = run_signature(
        PredictiveCohortInsightSignature,
        model_version=g["model_version"],
        distribution_summary=g["distribution_summary"],
        top_targets_summary=g["top_targets_summary"],
        drivers_summary=g["drivers_summary"],
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

- [ ] **Step 4 — Run, confirm PASS.** `python -m pytest tests/insights/test_predictive_cohort.py -v`.

- [ ] **Step 5 — Commit.**
```bash
git add src/insights/predictive_cohort.py tests/insights/test_predictive_cohort.py
git commit -m "feat(insights): predictive-cohort signature, grounding, fallback"
```

---

## Task 7: Resource-optimization insight adapter (`src/insights/resource_optimization.py`)

Resource-opt already produces `optimization_summary` (str) + `recommendations` (list[str]) via its existing `OptimizationSummarySignature`/`AllocationRecommendationSignature`. This adapter reshapes that EXISTING output into the uniform insight payload — no new signature, no re-generation.

**Files:**
- Create: `src/insights/resource_optimization.py`
- Test: `tests/insights/test_resource_optimization.py`

- [ ] **Step 1 — Failing test:**
```python
from src.insights.resource_optimization import to_insight

def test_to_insight_surfaces_existing_summary():
    out = to_insight(
        optimization_summary="Reallocating to high-ROI HCPs lifts projected outcome 6%.",
        recommendations=["Shift 12% budget to segment A", "Hold segment C"],
        projected_lift_pct=6.0,
        solver_status="optimal",
    )
    assert out["is_fallback"] is False
    assert "high-ROI" in out["insight"]
    assert out["key_takeaways"][0].startswith("Shift 12%")
    assert any(c["label"] == "Projected lift" for c in out["grounding"])

def test_to_insight_empty_summary_is_fallback():
    out = to_insight(optimization_summary="", recommendations=[],
                     projected_lift_pct=None, solver_status="infeasible")
    assert out["is_fallback"] is True
```

- [ ] **Step 2 — Run, confirm FAIL.** `python -m pytest tests/insights/test_resource_optimization.py -v`.

- [ ] **Step 3 — Implement** (`src/insights/resource_optimization.py`):
```python
"""Adapt the resource-optimizer's existing summary/recommendations to the uniform
strategic-insight payload (no new LLM call — surfaces what the agent already made)."""
from __future__ import annotations

from typing import Any

from src.insights.common import normalize_list


def to_insight(
    optimization_summary: str,
    recommendations: list[str] | None,
    projected_lift_pct: float | None,
    solver_status: str | None,
) -> dict[str, Any]:
    summary = (optimization_summary or "").strip()
    recs = normalize_list(recommendations or [])
    grounding = [{"label": "Solver", "value": str(solver_status or "unknown")}]
    if projected_lift_pct is not None:
        grounding.insert(0, {"label": "Projected lift", "value": f"{projected_lift_pct:+.1f}%"})
    if not summary:
        return {
            "insight": "No optimization narrative is available yet — run an optimization.",
            "key_takeaways": recs,
            "grounding": grounding,
            "is_fallback": True,
        }
    return {
        "insight": summary,
        "key_takeaways": recs,
        "grounding": grounding,
        "is_fallback": False,
    }
```

- [ ] **Step 4 — Run, confirm PASS.** `python -m pytest tests/insights/test_resource_optimization.py -v`.

- [ ] **Step 5 — Commit.**
```bash
git add src/insights/resource_optimization.py tests/insights/test_resource_optimization.py
git commit -m "feat(insights): resource-optimization adapter over existing summary"
```

---

## Task 8: Route module — 5 endpoints, grounding, caching, registration

**Files:**
- Create: `src/api/routes/insights_strategic.py`
- Modify: `src/api/main.py` (import ~line 79; `include_router` ~line 1103)
- Test: `tests/insights/test_routes.py`

**Grounding sources (real data):**
- knowledge-graph & model-performance: derived **server-side** — `SemanticMemory` (`get_semantic_memory()`: `list_nodes`, `count_nodes`, `list_relationships`, `count_relationships` — SYNC → wrap in `asyncio.to_thread`) and `PerformanceTracker` (`get_performance_tracker()`: `await get_performance_trend`, `await get_confusion_matrix`, `await get_roc_curve`).
- causal-discovery, predictive-cohort, resource-optimization: the client posts the run result it already received from the backend (real by construction); endpoint validates shape via the request model.

- [ ] **Step 1 — Failing test** (`tests/insights/test_routes.py`):
```python
import pytest
from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)

def _auth_headers():
    # Reuse the repo's existing test auth helper/fixture used by other route tests.
    # (See tests/api/ for the analyst-token pattern; import and reuse it here.)
    from tests.api.conftest import analyst_auth_headers  # adjust to real helper
    return analyst_auth_headers()

def test_causal_discovery_insight_fallback(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    body = {"brand": "Kisqali", "grain": "patient", "effects": [
        {"treatment": "copay_card", "outcome": "adherence_180d", "ate": 0.043,
         "ate_ci_lower": 0.02, "ate_ci_upper": 0.066, "status": "proceed",
         "selected_estimator": "CausalForestDML"}]}
    r = client.post("/api/insights/causal-discovery", json=body, headers=_auth_headers())
    assert r.status_code == 200
    data = r.json()
    assert data["is_fallback"] is True
    assert "copay_card" in data["insight"]
    assert {"label", "value"} <= set(data["grounding"][0].keys())
    assert data["provenance"]
```
  (If the repo has no `analyst_auth_headers` helper, use the same auth-override pattern other `tests/api/test_*.py` files use for `require_analyst`.)

- [ ] **Step 2 — Run, confirm FAIL.** `python -m pytest tests/insights/test_routes.py -v` → 404 (route not registered).

- [ ] **Step 3 — Implement route module** (`src/api/routes/insights_strategic.py`):
```python
"""Per-page strategic-insight endpoints. Each grounds an LLM interpretation in REAL
data with an honest deterministic fallback (no OPENAI_API_KEY -> is_fallback=True)."""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from src.api.dependencies.auth import require_analyst
from src.insights import (
    causal_discovery,
    knowledge_graph,
    model_performance,
    predictive_cohort,
    resource_optimization,
)
from src.insights.common import cache_get, cache_key, cache_set

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/insights", tags=["Strategic Insights"])


class GroundingChip(BaseModel):
    label: str
    value: str


class StrategicInsightResponse(BaseModel):
    insight: str
    key_takeaways: list[str] = Field(default_factory=list)
    grounding: list[GroundingChip] = Field(default_factory=list)
    is_fallback: bool
    generated_at: str
    provenance: str


def _finalize(payload: dict[str, Any], provenance: str) -> StrategicInsightResponse:
    return StrategicInsightResponse(
        insight=payload["insight"],
        key_takeaways=payload.get("key_takeaways", []),
        grounding=[GroundingChip(**c) for c in payload.get("grounding", [])],
        is_fallback=payload["is_fallback"],
        generated_at=datetime.now(timezone.utc).isoformat(),
        provenance=provenance,
    )


# ---- Request models for the run-pages (client posts backend-sourced results) ----
class KGInsightRequest(BaseModel):
    brand: str = "All"
    curated_only: bool = True


class ModelPerfInsightRequest(BaseModel):
    model_version: str


class CausalEffect(BaseModel):
    treatment: str
    outcome: str
    ate: float
    ate_ci_lower: float | None = None
    ate_ci_upper: float | None = None
    status: str | None = None
    selected_estimator: str | None = None


class CausalInsightRequest(BaseModel):
    brand: str
    grain: str
    effects: list[CausalEffect]


class TargetRow(BaseModel):
    entity_id: str
    probability: float


class DriverRow(BaseModel):
    feature: str
    importance: float


class PredictiveInsightRequest(BaseModel):
    model_version: str
    n_scored: int
    mean_prob: float
    top_targets: list[TargetRow] = Field(default_factory=list)
    top_drivers: list[DriverRow] = Field(default_factory=list)


class ResourceInsightRequest(BaseModel):
    optimization_summary: str = ""
    recommendations: list[str] = Field(default_factory=list)
    projected_lift_pct: float | None = None
    solver_status: str | None = None


@router.post("/knowledge-graph", response_model=StrategicInsightResponse)
async def knowledge_graph_insight(
    req: KGInsightRequest, user: dict[str, Any] = Depends(require_analyst)
) -> StrategicInsightResponse:
    from src.memory.services.factories import get_semantic_memory

    sm = get_semantic_memory()
    brand = None if req.brand == "All" else req.brand

    def _load() -> dict[str, Any]:
        nodes = sm.list_nodes(limit=500, curated_only=req.curated_only)
        rels = sm.list_relationships(limit=500, curated_only=req.curated_only)
        if brand:  # scope edges to the brand when the property is present
            rels = [r for r in rels if (r.get("properties") or {}).get("brand") in (None, brand)]
        return knowledge_graph.build_grounding(
            req.brand, nodes, rels,
            node_count=sm.count_nodes(curated_only=req.curated_only),
            rel_count=len(rels),
        )

    g = await asyncio.to_thread(_load)
    key = cache_key("knowledge-graph", req.brand, {"n": g["node_summary"], "e": g["edge_summary"]})
    cached = cache_get(key)
    payload = cached or await asyncio.to_thread(knowledge_graph.generate_insight, g)
    if not cached:
        cache_set(key, payload)
    return _finalize(payload, provenance="Curated knowledge graph (server-derived)")


@router.post("/model-performance", response_model=StrategicInsightResponse)
async def model_performance_insight(
    req: ModelPerfInsightRequest, user: dict[str, Any] = Depends(require_analyst)
) -> StrategicInsightResponse:
    from src.services.performance_tracking import get_performance_tracker

    tracker = get_performance_tracker()
    trend = await tracker.get_performance_trend(req.model_version, "accuracy")
    confusion = await tracker.get_confusion_matrix(req.model_version)
    roc = await tracker.get_roc_curve(req.model_version)
    g = model_performance.build_grounding(
        model_version=req.model_version,
        current_accuracy=float(getattr(trend, "current_value", 0.0) or 0.0),
        baseline_accuracy=float(getattr(trend, "baseline_value", 0.0) or 0.0),
        trend=str(getattr(trend, "trend", "stable")),
        confusion=confusion,
        auc=(float(roc["auc"]) if roc and roc.get("auc") is not None else None),
        alerts=list(getattr(trend, "alerts", []) or []),
    )
    key = cache_key("model-performance", req.model_version,
                    {"a": g["accuracy_summary"], "c": g["confusion_summary"]})
    cached = cache_get(key)
    payload = cached or await asyncio.to_thread(model_performance.generate_insight, g)
    if not cached:
        cache_set(key, payload)
    return _finalize(payload, provenance="Live model-performance metrics (server-derived)")


@router.post("/causal-discovery", response_model=StrategicInsightResponse)
async def causal_discovery_insight(
    req: CausalInsightRequest, user: dict[str, Any] = Depends(require_analyst)
) -> StrategicInsightResponse:
    g = causal_discovery.build_grounding(
        req.brand, req.grain, [e.model_dump() for e in req.effects]
    )
    key = cache_key("causal-discovery", req.brand, {"t": g["effects_table"]})
    cached = cache_get(key)
    payload = cached or await asyncio.to_thread(causal_discovery.generate_insight, g)
    if not cached:
        cache_set(key, payload)
    return _finalize(payload, provenance="Agent-validated discovered effects")


@router.post("/predictive-cohort", response_model=StrategicInsightResponse)
async def predictive_cohort_insight(
    req: PredictiveInsightRequest, user: dict[str, Any] = Depends(require_analyst)
) -> StrategicInsightResponse:
    g = predictive_cohort.build_grounding(
        req.model_version, req.n_scored, req.mean_prob,
        [t.model_dump() for t in req.top_targets],
        [d.model_dump() for d in req.top_drivers],
    )
    key = cache_key("predictive-cohort", req.model_version,
                    {"d": g["distribution_summary"], "t": g["top_targets_summary"]})
    cached = cache_get(key)
    payload = cached or await asyncio.to_thread(predictive_cohort.generate_insight, g)
    if not cached:
        cache_set(key, payload)
    return _finalize(payload, provenance="Out-of-sample scored cohort + SHAP")


@router.post("/resource-optimization", response_model=StrategicInsightResponse)
async def resource_optimization_insight(
    req: ResourceInsightRequest, user: dict[str, Any] = Depends(require_analyst)
) -> StrategicInsightResponse:
    payload = resource_optimization.to_insight(
        req.optimization_summary, req.recommendations, req.projected_lift_pct, req.solver_status
    )
    return _finalize(payload, provenance="Resource optimizer (existing agent output)")
```
  Then register in `src/api/main.py`: add import near line 79
  `from src.api.routes.insights_strategic import router as insights_strategic_router`
  and near line 1103 `app.include_router(insights_strategic_router, prefix="/api")`.

- [ ] **Step 4 — Run, confirm PASS.** `python -m pytest tests/insights/test_routes.py -v`. Then `python -m mypy src/api/routes/insights_strategic.py src/insights/`.

- [ ] **Step 5 — Commit.**
```bash
git add src/api/routes/insights_strategic.py src/api/main.py tests/insights/test_routes.py
git commit -m "feat(insights): 5 strategic-insight endpoints (grounding, cache, registration)"
```

---

## Task 9: Frontend — types, api client, hooks

**Files:**
- Create: `frontend/src/types/insights.ts`
- Create: `frontend/src/api/insights.ts`
- Create: `frontend/src/hooks/api/use-insights.ts`
- Modify: `frontend/src/hooks/api/index.ts` (add barrel exports)
- Test: `frontend/src/api/insights.test.ts`

- [ ] **Step 1 — Failing test** (`frontend/src/api/insights.test.ts`):
```typescript
import { describe, it, expect, vi, beforeEach } from 'vitest';
import * as apiClient from '@/lib/api-client';
import { getCausalDiscoveryInsight } from './insights';

describe('insights api', () => {
  beforeEach(() => vi.restoreAllMocks());
  it('POSTs to /insights/causal-discovery and returns the response', async () => {
    const resp = { insight: 'x', key_takeaways: [], grounding: [], is_fallback: true,
      generated_at: 't', provenance: 'p' };
    const spy = vi.spyOn(apiClient, 'post').mockResolvedValue(resp as never);
    const out = await getCausalDiscoveryInsight({ brand: 'Kisqali', grain: 'patient', effects: [] });
    expect(spy).toHaveBeenCalledWith('/insights/causal-discovery', expect.any(Object));
    expect(out.is_fallback).toBe(true);
  });
});
```

- [ ] **Step 2 — Run, confirm FAIL.** `npm --prefix frontend run test -- run src/api/insights.test.ts`.

- [ ] **Step 3 — Implement.**
  `frontend/src/types/insights.ts` — the `GroundingChip` + `StrategicInsightResponse` interfaces from **Shared Contracts** above, plus request types:
```typescript
export interface GroundingChip { label: string; value: string }
export interface StrategicInsightResponse {
  insight: string;
  key_takeaways: string[];
  grounding: GroundingChip[];
  is_fallback: boolean;
  generated_at: string;
  provenance: string;
}
export interface KGInsightRequest { brand: string; curated_only?: boolean }
export interface ModelPerfInsightRequest { model_version: string }
export interface CausalInsightRequest {
  brand: string; grain: string;
  effects: Array<{ treatment: string; outcome: string; ate: number;
    ate_ci_lower?: number; ate_ci_upper?: number; status?: string; selected_estimator?: string }>;
}
export interface PredictiveInsightRequest {
  model_version: string; n_scored: number; mean_prob: number;
  top_targets: Array<{ entity_id: string; probability: number }>;
  top_drivers: Array<{ feature: string; importance: number }>;
}
export interface ResourceInsightRequest {
  optimization_summary: string; recommendations: string[];
  projected_lift_pct?: number | null; solver_status?: string | null;
}
```
  `frontend/src/api/insights.ts`:
```typescript
import { post } from '@/lib/api-client';
import type {
  StrategicInsightResponse, KGInsightRequest, ModelPerfInsightRequest,
  CausalInsightRequest, PredictiveInsightRequest, ResourceInsightRequest,
} from '@/types/insights';

const BASE = '/insights';

export const getKnowledgeGraphInsight = (r: KGInsightRequest) =>
  post<StrategicInsightResponse, KGInsightRequest>(`${BASE}/knowledge-graph`, r);
export const getModelPerformanceInsight = (r: ModelPerfInsightRequest) =>
  post<StrategicInsightResponse, ModelPerfInsightRequest>(`${BASE}/model-performance`, r);
export const getCausalDiscoveryInsight = (r: CausalInsightRequest) =>
  post<StrategicInsightResponse, CausalInsightRequest>(`${BASE}/causal-discovery`, r);
export const getPredictiveCohortInsight = (r: PredictiveInsightRequest) =>
  post<StrategicInsightResponse, PredictiveInsightRequest>(`${BASE}/predictive-cohort`, r);
export const getResourceOptimizationInsight = (r: ResourceInsightRequest) =>
  post<StrategicInsightResponse, ResourceInsightRequest>(`${BASE}/resource-optimization`, r);
```
  `frontend/src/hooks/api/use-insights.ts` (one hook per page; button pages use `useMutation`):
```typescript
import { useMutation } from '@tanstack/react-query';
import { ApiError } from '@/lib/api-client';
import {
  getKnowledgeGraphInsight, getModelPerformanceInsight, getCausalDiscoveryInsight,
  getPredictiveCohortInsight, getResourceOptimizationInsight,
} from '@/api/insights';
import type {
  StrategicInsightResponse, KGInsightRequest, ModelPerfInsightRequest,
  CausalInsightRequest, PredictiveInsightRequest, ResourceInsightRequest,
} from '@/types/insights';

export const useKnowledgeGraphInsight = () =>
  useMutation<StrategicInsightResponse, ApiError, KGInsightRequest>({ mutationFn: getKnowledgeGraphInsight });
export const useModelPerformanceInsight = () =>
  useMutation<StrategicInsightResponse, ApiError, ModelPerfInsightRequest>({ mutationFn: getModelPerformanceInsight });
export const useCausalDiscoveryInsight = () =>
  useMutation<StrategicInsightResponse, ApiError, CausalInsightRequest>({ mutationFn: getCausalDiscoveryInsight });
export const usePredictiveCohortInsight = () =>
  useMutation<StrategicInsightResponse, ApiError, PredictiveInsightRequest>({ mutationFn: getPredictiveCohortInsight });
export const useResourceOptimizationInsight = () =>
  useMutation<StrategicInsightResponse, ApiError, ResourceInsightRequest>({ mutationFn: getResourceOptimizationInsight });
```
  Append to `frontend/src/hooks/api/index.ts`:
```typescript
export {
  useKnowledgeGraphInsight, useModelPerformanceInsight, useCausalDiscoveryInsight,
  usePredictiveCohortInsight, useResourceOptimizationInsight,
} from './use-insights';
```

- [ ] **Step 4 — Run, confirm PASS.** `npm --prefix frontend run test -- run src/api/insights.test.ts`.

- [ ] **Step 5 — Commit.**
```bash
git add frontend/src/types/insights.ts frontend/src/api/insights.ts frontend/src/hooks/api/use-insights.ts frontend/src/hooks/api/index.ts frontend/src/api/insights.test.ts
git commit -m "feat(insights): frontend types, api client, hooks for 5 insight endpoints"
```

---

## Task 10: Shared `StrategicInsightCard` component

**Files:**
- Create: `frontend/src/components/insights/StrategicInsightCard.tsx`
- Modify: `frontend/src/components/insights/index.ts` (barrel export)
- Test: `frontend/src/components/insights/StrategicInsightCard.test.tsx`

- [ ] **Step 1 — Failing test:**
```tsx
import { describe, it, expect, vi } from 'vitest';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { StrategicInsightCard } from './StrategicInsightCard';

describe('StrategicInsightCard', () => {
  it('shows a Generate button when empty and onGenerate is provided', async () => {
    const onGenerate = vi.fn();
    render(<StrategicInsightCard onGenerate={onGenerate} />);
    const btn = screen.getByRole('button', { name: /generate strategic insight/i });
    await userEvent.click(btn);
    expect(onGenerate).toHaveBeenCalledOnce();
  });
  it('renders the narrative, grounding chips, and a fallback badge', () => {
    render(<StrategicInsightCard insight="Adherence drives NRx." isFallback
      grounding={[{ label: 'Nodes', value: '10' }]} />);
    expect(screen.getByText(/adherence drives nrx/i)).toBeInTheDocument();
    expect(screen.getByText('Nodes')).toBeInTheDocument();
    expect(screen.getByText(/factual summary/i)).toBeInTheDocument();
  });
  it('shows a loading skeleton', () => {
    const { container } = render(<StrategicInsightCard isLoading />);
    expect(container.querySelector('.animate-pulse')).toBeTruthy();
  });
  it('shows an error message', () => {
    render(<StrategicInsightCard error="boom" />);
    expect(screen.getByText(/boom/i)).toBeInTheDocument();
  });
});
```

- [ ] **Step 2 — Run, confirm FAIL.** `npm --prefix frontend run test -- run src/components/insights/StrategicInsightCard.test.tsx`.

- [ ] **Step 3 — Implement** (`StrategicInsightCard.tsx`):
```tsx
import { Sparkles } from 'lucide-react';
import { Card, CardHeader, CardTitle, CardDescription, CardContent } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import type { GroundingChip } from '@/types/insights';

interface StrategicInsightCardProps {
  title?: string;
  description?: string;
  insight?: string;
  keyTakeaways?: string[];
  grounding?: GroundingChip[];
  isLoading?: boolean;
  error?: string | null;
  onGenerate?: () => void;
  isFallback?: boolean;
  provenance?: string;
  generatedAt?: string;
}

export function StrategicInsightCard({
  title = 'Strategic Interpretation',
  description = 'Agentic read of this view, grounded in the underlying numbers',
  insight, keyTakeaways = [], grounding = [], isLoading, error, onGenerate,
  isFallback, provenance, generatedAt,
}: StrategicInsightCardProps) {
  return (
    <Card className="border-primary/40">
      <CardHeader>
        <div className="flex items-center gap-2">
          <Sparkles className="h-4 w-4 text-primary" />
          <CardTitle>{title}</CardTitle>
        </div>
        <CardDescription>{description}</CardDescription>
      </CardHeader>
      <CardContent className="space-y-3">
        {isLoading && (
          <div className="space-y-2" aria-label="Generating insight">
            <div className="h-4 w-3/4 animate-pulse rounded bg-muted" />
            <div className="h-4 w-full animate-pulse rounded bg-muted" />
            <div className="h-4 w-5/6 animate-pulse rounded bg-muted" />
          </div>
        )}
        {!isLoading && error && (
          <p className="text-sm text-destructive">{error}</p>
        )}
        {!isLoading && !error && !insight && onGenerate && (
          <Button variant="outline" onClick={onGenerate}>
            <Sparkles className="mr-2 h-4 w-4" /> Generate strategic insight
          </Button>
        )}
        {!isLoading && !error && insight && (
          <>
            <p className="whitespace-pre-line leading-relaxed">{insight}</p>
            {keyTakeaways.length > 0 && (
              <ul className="list-disc space-y-1 pl-5 text-sm">
                {keyTakeaways.map((t, i) => <li key={i}>{t}</li>)}
              </ul>
            )}
            {grounding.length > 0 && (
              <div className="flex flex-wrap gap-2 pt-1">
                {grounding.map((c, i) => (
                  <span key={i} className="rounded-full border px-2 py-0.5 text-xs text-muted-foreground">
                    <span className="font-medium">{c.label}</span>: {c.value}
                  </span>
                ))}
              </div>
            )}
            <div className="flex items-center gap-2 pt-1 text-xs text-muted-foreground">
              {isFallback && (
                <span className="rounded bg-muted px-1.5 py-0.5">factual summary — LLM unavailable</span>
              )}
              {provenance && <span>{provenance}</span>}
              {generatedAt && <span>· {new Date(generatedAt).toLocaleString()}</span>}
            </div>
          </>
        )}
      </CardContent>
    </Card>
  );
}
```
  Append to `frontend/src/components/insights/index.ts`: `export { StrategicInsightCard } from './StrategicInsightCard';`

- [ ] **Step 4 — Run, confirm PASS.** `npm --prefix frontend run test -- run src/components/insights/StrategicInsightCard.test.tsx`.

- [ ] **Step 5 — Commit.**
```bash
git add frontend/src/components/insights/StrategicInsightCard.tsx frontend/src/components/insights/index.ts frontend/src/components/insights/StrategicInsightCard.test.tsx
git commit -m "feat(insights): shared StrategicInsightCard component"
```

---

## Task 11: Wire the card into all 5 pages

Each sub-task: import `StrategicInsightCard` + the page's hook, add state, place the card, and add a smoke test that the card renders. Button pages (KG, model-perf) render the card with `onGenerate` wired to the mutation. Run pages generate after their run completes (or via a button in the results area if no result yet).

- [ ] **11a — Knowledge-graph** (`frontend/src/pages/KnowledgeGraph.tsx`, place below the stats cards, above the graph card):
```tsx
// imports
import { StrategicInsightCard } from '@/components/insights';
import { useKnowledgeGraphInsight } from '@/hooks/api';
// inside component
const kgInsight = useKnowledgeGraphInsight();
// JSX (below stats cards):
<StrategicInsightCard
  isLoading={kgInsight.isPending}
  error={kgInsight.error?.message ?? null}
  insight={kgInsight.data?.insight}
  keyTakeaways={kgInsight.data?.key_takeaways}
  grounding={kgInsight.data?.grounding}
  isFallback={kgInsight.data?.is_fallback}
  provenance={kgInsight.data?.provenance}
  generatedAt={kgInsight.data?.generated_at}
  onGenerate={() => kgInsight.mutate({ brand: selectedBrand ?? 'All', curated_only: true })}
/>
```
  Test (`frontend/src/pages/KnowledgeGraph.test.tsx`, add): renders the page and asserts the "Generate strategic insight" button is present. Run: `npm --prefix frontend run test -- run src/pages/KnowledgeGraph.test.tsx`. Commit:
```bash
git add frontend/src/pages/KnowledgeGraph.tsx frontend/src/pages/KnowledgeGraph.test.tsx
git commit -m "feat(knowledge-graph): strategic insight card"
```

- [ ] **11b — Model-performance** (`frontend/src/pages/ModelPerformance.tsx`, above the metric KPI cards). Same pattern with `useModelPerformanceInsight`; `onGenerate={() => mpInsight.mutate({ model_version: selectedModel })}` (use the page's existing selected-model variable). Add smoke test + commit `feat(model-performance): strategic insight card`.

- [ ] **11c — Causal-analysis** (`frontend/src/pages/CausalAnalysis.tsx`, above the leaderboard table). Use `useCausalDiscoveryInsight`; after `useDiscoverEffects` returns, call `mutate({ brand, grain, effects: discovered.map(...) })`. Place card so it shows loading during generation and the narrative above the leaderboard. Add smoke test + commit `feat(causal-analysis): strategic insight card`.

- [ ] **11d — Predictive-analytics** (`frontend/src/pages/PredictiveAnalytics.tsx`, between the model summary card and the results grid). Use `usePredictiveCohortInsight`; after the cohort is scored, call `mutate({ model_version, n_scored, mean_prob, top_targets, top_drivers })` derived from the real scored-cohort result. Add smoke test + commit `feat(predictive-analytics): strategic insight card`.

- [ ] **11e — Resource-optimization** (`frontend/src/pages/ResourceOptimization.tsx`, in the results section). Use `useResourceOptimizationInsight`; after a run, call `mutate({ optimization_summary, recommendations, projected_lift_pct, solver_status })` from the existing `OptimizationResponse`. Render the card near the KPI summary so the existing summary/recommendations surface consistently. Add smoke test + commit `feat(resource-optimization): strategic insight card`.

For each of 11a–11e: **write the smoke test first, run it to FAIL, wire the card, run to PASS, commit.**

---

## Task 12: Manual LLM-path verification on the droplet (faithful env)

CI cannot exercise the live OpenAI path (no key, and we don't burn CI on OpenAI throughput). Verify the real LLM path once, locally on the droplet where `OPENAI_API_KEY` is set.

- [ ] **Step 1 — Confirm key present.** `python -c "import os; print(bool(os.getenv('OPENAI_API_KEY')))"` → `True`.
- [ ] **Step 2 — One faithful call per page** (example, causal-discovery):
```bash
python -c "
import asyncio
from src.insights import causal_discovery as cd
g = cd.build_grounding('Kisqali','patient',[{'treatment':'copay_card','outcome':'adherence_180d','ate':0.043,'ate_ci_lower':0.02,'ate_ci_upper':0.066,'status':'proceed','selected_estimator':'CausalForestDML'}])
out = cd.generate_insight(g)
print('is_fallback=', out['is_fallback']); print(out['insight'][:400])
"
```
  Expected: `is_fallback= False` and a grounded narrative citing copay_card / the ATE. Repeat for knowledge_graph, model_performance, predictive_cohort with small real inputs.
- [ ] **Step 3 — Record** the observed outputs in the PR description (evidence, not assertion). No commit (verification only).

---

## Task 13: Batched CI + PR (at the very end)

- [ ] **Step 1 — Local pre-flight (memory-safe, targeted):**
  - `python -m pytest tests/insights/ -v`
  - `python -m mypy src/insights/ src/api/routes/insights_strategic.py`
  - `npm --prefix frontend run test -- run src/components/insights src/api/insights.test.ts src/pages/Home.test.tsx`
  - `npm --prefix frontend run lint` (or the repo's FE lint script)
- [ ] **Step 2 — Push branch** (proxy bypass first):
```bash
git config --global http.https://github.com.proxy ""
git push -u origin worktree-feat+strategic-insights-brand-selector
```
- [ ] **Step 3 — Open PR** (this triggers CI — the single batched run):
```bash
gh pr create --title "feat: strategic insights on 5 pages + brand-selector fix" \
  --body "$(cat <<'EOF'
## Summary
- Remove "Combined Portfolio" indication for the All brand (fixes Home selector overflow; label-only, no behavior change).
- Add a bespoke, LLM-grounded StrategicInsightCard to knowledge-graph, causal-analysis, predictive-analytics, model-performance, resource-optimization.

## Backend
- New `src/insights/` package: 4 bespoke DSPy signatures (KG, model-perf, causal-discovery, predictive-cohort) + resource-opt adapter over the existing agent output; each with a deterministic factual fallback.
- New `POST /api/insights/*` endpoints (auth: require_analyst), server-derived grounding for always-on pages, Redis cache (~1h).

## Honesty / no-mock
- No OPENAI_API_KEY (CI) → every endpoint returns the grounded factual fallback (`is_fallback: true`); never fabricated.
- Live LLM path verified manually on the droplet (see Task 12 evidence below).

## Test
- pytest tests/insights/ green; Vitest for card + api + Home green; mypy on changed files clean.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```
- [ ] **Step 4 — Watch CI, converge to green.** `gh pr checks --watch`. Fix failures (ralph-loop + codex-rescue as needed), commit, push. Repeat until all checks pass. Merge with `--merge` or `--rebase` (NEVER `--squash`).

---

## Self-Review (checklist run against the spec)

**Spec coverage:**
- Brand fix → Task 1 ✓
- Shared `StrategicInsightCard` → Task 10 ✓
- 2 new signatures (KG, model-perf) → Tasks 3, 4 ✓
- causal + predictive page-level signatures → Tasks 5, 6 (upgraded from "adapt" to "new" — existing signatures are single-item, documented in spec §3.3/handoff) ✓
- resource-opt reuse (no new signature) → Task 7 ✓
- 5 endpoints + grounding + cache + registration → Task 8 ✓
- Frontend types/api/hooks → Task 9 ✓
- Per-page wiring + placement → Task 11a–e ✓
- No-mock testing (fallback in CI, live verified on droplet) → Tasks 2–8 tests + Task 12 ✓
- Redis cache ~1h → common.py + endpoints ✓
- Batched CI at the end, no-squash, proxy bypass → Task 13 ✓
- Memory-safe (targeted mypy/pytest, CI arbiter) → Conventions + Task 13 ✓

**Placeholder scan:** No "TBD/TODO/handle edge cases" — every code step has real code. The two auth-helper references in Tasks 8 (`analyst_auth_headers`) and the page smoke tests intentionally defer to the repo's existing test auth pattern (must be read at execution time — not inventable here).

**Type consistency:** `StrategicInsightResponse`/`GroundingChip` identical across `src/insights/common.py` docstring, `insights_strategic.py`, and `frontend/src/types/insights.ts`. Every `generate_insight(g)` returns `{insight, key_takeaways, grounding, is_fallback}`; `_finalize` adds `generated_at`+`provenance`. Hook names match barrel exports and api fn names.

**Known execution-time reads (not guesses to fix, but files to open):**
- The exact field names on `PerformanceTrend` (`current_value`/`baseline_value`/`trend`/`alerts`) — verify against `src/services/performance_tracking.py` when implementing Task 8; adjust `getattr` keys if they differ.
- `get_semantic_memory` / `get_performance_tracker` factory import paths — confirm in `src/memory/services/factories.py` and `src/services/performance_tracking.py`.
- Each page's existing selected-brand / selected-model variable names — read the page before wiring in Task 11.
