# Refusal Reason Codes Lane (#2021 + #2050 + #2020) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give every composable-tool refusal and failure a stable, closed-set `reason_code` plus a data-free canonical sentence, propagate it through the executor to the learning-loop recorder and the database, use it to keep library internals out of the user-facing fail-closed answer, and surface it on the admin observability page.

**Architecture:** One vocabulary module (`ReasonCode` + a canonical-sentence catalogue) is the single source of truth. Tool-authored refusals (`ToolRefusalError` / `ToolInputError`) carry a code at the raise site; the executor assigns a code to every non-refusal failure arm from the outcome class it already computes. The code and its canonical sentence (never the raw message) are what get persisted and displayed; the raw message keeps its two existing homes — the container log, and the fail-closed answer **only when the tool authored it**.

**Tech Stack:** Python 3.12, Pydantic v2, FastAPI, PostgreSQL (Supabase, `ml` migrations), pytest.

---

## Why these three issues are one lane

They are three views of one missing field.

- **#2021** — refusals carry free text only; the learning loop cannot aggregate "why tools fail".
- **#2050** — a refusal's reason is not persisted (two independent causes, both confirmed in source).
- **#2020** — the fail-closed answer concatenates raw exception text, so library internals reach the user alongside intentional refusals.

All three are unblocked by the same artefact: a closed reason-code set that is safe to persist and safe to display. Landing them separately would mean three passes over `tool_registrations.py`, the executor, and the recorder.

## Findings that change the issues as written

Verified 2026-09-12 against `origin/main` `0a16e9c18`. **Read this section before Task 1.**

1. **#2050's cause-1 fix would not compile as written.** The issue says "`StepResult.error_message` exists (`src/agents/tool_composer/schemas.py`) and the executor populates it on the failure path". It does not. `schemas.py:142`'s `error_message` belongs to **`CompositionResult`**. The `StepResult` the recorder actually receives is `src/agents/tool_composer/models/composition_models.py:319`, which has `output: ToolOutput`, `outcome_class`, `attempts`, `cache_hit`, `error_type` — and **no `error_message`**. The failure text lives at `result.output.error` (`executor.py:829`, `859`, `907`, `946`).
2. **#2050's cause-2 is confirmed verbatim.** `database/ml/041_composer_learning_loop_recording.sql:631` is a bare `NULL` in the `error_message` slot of the INSERT's SELECT, between the `status` CASE and `retry_count`.
3. **The `NULL` is very likely deliberate.** ml/041's header (line 27) states the recording contract: *"identifiers are positional, intents are normalized, and **no error text is stored**."* `RecentFailure.error_type` in `src/api/schemas/admin_tool_composer.py:73` documents the same rule: *"The exception class; never its message (spec §5.5)."* Refusal messages interpolate data — `{list(df.columns)!r}`, `{cohort_result!r}`, `{sorted(kwargs.keys())!r}` — so persisting them wholesale would put column names and value reprs into a table designed to hold none.
   **Owner decision (2026-09-12): persist the `reason_code` and a fixed catalogue sentence derived from it, plus bounded structured `details`. The raw message is never persisted.** That satisfies #2050's operator need ("an operator sees `refused` with no why") without breaking ml/041's invariant.
4. **`get_tool_reliability` reads `tool_performance`, not `composition_steps`** (ml/041:806). So a per-tool `most_common_refusal_reason` needs `reason_code` on **both** tables.
5. **Scope is 87 raise sites, all in one file.** Measured by `ast.walk` for `raise ToolRefusalError(...)` / `raise ToolInputError(...)` on the lane base `0a16e9c18`: `ToolRefusalError` 66 + `ToolInputError` 21 = 87. `src/tool_registry/` → 0.
   **This plan first said 94, which was wrong for two independent reasons** (caught by the Task 1/2 implementer, corrected 2026-09-12): (a) the figure came from `grep -c`, which counts LINES mentioning either name and dedupes multiple matches per line — 16 of the 103 matching lines are docstrings, comments and the import; (b) it was measured on `6c6a6a0ae`, three commits behind the lane base, before PR #2059 (`8bb85a772`, `772733dc2`, the #2022 sensitivity work) added 5 more `ToolRefusalError` sites. Method alone gives 82 on that stale base; base drift takes it to 87. **The AST test is the only count that governs — do not re-derive this number with grep.**
6. **96 existing assertions pin refusal prose** across 17 test files. Task 2 must not reword a single message.

## Owner decisions (2026-09-12)

| # | Decision |
|---|----------|
| D1 | `composition_steps.error_message` receives the **canonical catalogue sentence** derived from the code, never the tool's raw message. Structured `details` carry the specifics. |
| D2 | **All 87 raise sites** get a code in this lane, enforced by an AST test that fails if any raise site lacks one. |
| D3 | The admin observability route **does** surface the codes (per-tool `most_common_refusal_reason`, per-step `reason_code`). |

## File structure

| File | Responsibility | Task |
|---|---|---|
| `src/agents/tool_composer/reason_codes.py` *(new)* | `ReasonCode` closed set + canonical sentence catalogue + `canonical_sentence()` | 1 |
| `src/agents/tool_composer/errors.py` | `ToolRefusalError` / `ToolInputError` carry `reason_code` + `details` | 1 |
| `src/agents/tool_composer/tool_registrations.py` | 87 raise sites get codes | 2 |
| `src/agents/tool_composer/models/composition_models.py` | `StepResult.reason_code`, `StepResult.reason_details` | 3 |
| `src/agents/tool_composer/executor.py` | assign a code on every failure arm | 3 |
| `src/agents/tool_composer/composer.py` | fail-closed answer: verbatim for refusals, canonical for everything else | 4 |
| `src/agents/tool_composer/learning_recorder.py` | emit `reason_code` + canonical `error_message` | 5 |
| `database/ml/042_composer_refusal_reason_codes.sql` *(new)* | columns, RPC fix, reliability function | 6 |
| `database/ml/rollback_042.sql` *(new)* | reverse of 042 | 6 |
| `src/agents/tool_composer/reliability.py` | `most_common_refusal_reason` on `ToolReliability` | 7 |
| `src/api/schemas/admin_tool_composer.py` | wire fields | 7 |
| `src/services/tool_composer_observability_service.py` | map the new fields | 7 |

---

## Task 1: Reason-code vocabulary and error classes

**Files:**
- Create: `src/agents/tool_composer/reason_codes.py`
- Modify: `src/agents/tool_composer/errors.py`
- Test: `tests/unit/test_agents/test_tool_composer/test_reason_codes_2021.py`

The vocabulary is **closed and extensible only by editing this module**: every member must have a catalogue sentence, and a test enforces that. Task 2 may add members when a raise site genuinely has no home — adding a member means adding its sentence in the same edit.

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_agents/test_tool_composer/test_reason_codes_2021.py`:

```python
"""#2021: refusals carry a closed reason code and a data-free canonical sentence."""

import pytest

from src.agents.tool_composer.errors import ToolInputError, ToolRefusalError
from src.agents.tool_composer.reason_codes import (
    CANONICAL_SENTENCES,
    ReasonCode,
    canonical_sentence,
)


def test_every_code_has_a_canonical_sentence():
    """The catalogue is total: a code without a sentence cannot be displayed or stored."""
    missing = [c.value for c in ReasonCode if c not in CANONICAL_SENTENCES]
    assert missing == [], f"codes with no canonical sentence: {missing}"


def test_canonical_sentences_carry_no_interpolation():
    """The sentence is persisted and shown; it must never be a format string."""
    for code, sentence in CANONICAL_SENTENCES.items():
        assert "{" not in sentence and "}" not in sentence, code
        assert "%s" not in sentence, code
        assert sentence and sentence[0].islower(), code
        assert len(sentence) <= 160, code


def test_refusal_carries_code_and_details():
    err = ToolRefusalError(
        "carries 4 distinct non-null values, including [0, 1, 2, 3].",
        reason_code=ReasonCode.NON_BINARY_TREATMENT,
        details={"n_distinct": 4},
    )
    assert err.reason_code is ReasonCode.NON_BINARY_TREATMENT
    assert err.details == {"n_distinct": 4}
    assert err.canonical_sentence == canonical_sentence(ReasonCode.NON_BINARY_TREATMENT)
    assert "4 distinct" in str(err), "the human message is preserved unchanged"


def test_input_error_carries_code():
    err = ToolInputError(
        "expected_effect must not be None",
        reason_code=ReasonCode.MISSING_REQUIRED_INPUT,
    )
    assert err.reason_code is ReasonCode.MISSING_REQUIRED_INPUT
    assert isinstance(err, ValueError)


def test_code_is_required():
    """A refusal without a code is the defect #2021 exists to remove."""
    with pytest.raises(TypeError):
        ToolRefusalError("no code given")  # type: ignore[call-arg]


def test_details_must_be_structure_only():
    """details is persisted; it may not smuggle free text back in."""
    with pytest.raises(ValueError, match="details"):
        ToolRefusalError(
            "x",
            reason_code=ReasonCode.NO_USABLE_ROWS,
            details={"message": "a sentence that is really free text " * 5},
        )


def test_canonical_sentence_accepts_a_raw_string_code():
    """The recorder and composer read codes back off models as plain strings."""
    assert canonical_sentence("non_binary_treatment") == canonical_sentence(
        ReasonCode.NON_BINARY_TREATMENT
    )
    assert canonical_sentence("not_a_real_code") == canonical_sentence(ReasonCode.TOOL_ERROR)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /home/enunez/Projects/e2i_causal_analytics/.worktrees/lane-refusal-codes
/home/enunez/Projects/e2i_causal_analytics/.venv/bin/python -c "import src.agents.tool_composer.errors" 2>/dev/null || true
/home/enunez/Projects/e2i_causal_analytics/.venv/bin/pytest tests/unit/test_agents/test_tool_composer/test_reason_codes_2021.py -n 0 -p no:cacheprovider -q --timeout=120
```

Expected: collection error — `ModuleNotFoundError: No module named 'src.agents.tool_composer.reason_codes'`.

- [ ] **Step 3: Create `src/agents/tool_composer/reason_codes.py`**

```python
"""The closed reason-code vocabulary for composable-tool refusals and failures (#2021).

Why a code and not the message. A refusal's human message is written for the person
reading the answer and interpolates the data that caused it — column names, value
reprs, kwargs keys. That makes it useless as an aggregation key (every refusal is its
own bucket) and unsafe to persist: ml/041's recording contract stores structure only,
"no error text is stored" (041 header, §5.5). The code is the aggregation key; the
canonical sentence is the data-free rendering that is safe to store and to show.

Adding a member REQUIRES adding its canonical sentence in the same edit —
``test_every_code_has_a_canonical_sentence`` fails otherwise.
"""

from __future__ import annotations

from enum import Enum
from typing import Dict, Union


class ReasonCode(str, Enum):
    """Why a step did not produce a result. Closed set; stable wire values."""

    # --- Tool-authored refusals: the inputs are legal but cannot answer the question
    NON_BINARY_TREATMENT = "non_binary_treatment"
    SINGLE_CLASS_TREATMENT = "single_class_treatment"
    NON_BINARY_OUTCOME = "non_binary_outcome"
    NO_USABLE_ROWS = "no_usable_rows"
    NO_USABLE_COLUMNS = "no_usable_columns"
    NON_FINITE_INPUT = "non_finite_input"
    INSUFFICIENT_GROUPS = "insufficient_groups"
    INSUFFICIENT_SAMPLE = "insufficient_sample"
    COVERAGE_GAP = "coverage_gap"
    UNKNOWN_COLUMN = "unknown_column"
    MISSING_DATAFRAME = "missing_dataframe"
    CI_OUTSIDE_ESTIMATE = "ci_outside_estimate"
    POINT_ESTIMATE_ONLY = "point_estimate_only"
    UPSTREAM_STEP_FAILED = "upstream_step_failed"
    UNSUPPORTED_REQUEST = "unsupported_request"

    # --- Tool-authored input rejections: the value is not a legal input
    MISSING_REQUIRED_INPUT = "missing_required_input"
    INVALID_INPUT_TYPE = "invalid_input_type"
    INVALID_INPUT_VALUE = "invalid_input_value"

    # --- Executor-assigned: no tool authored these
    TOOL_ERROR = "tool_error"
    TOOL_TIMEOUT = "tool_timeout"
    PLAN_DEFECT = "plan_defect"
    REFERENCE_UNRESOLVABLE = "reference_unresolvable"
    DEPENDENCY_UNMET = "dependency_unmet"
    CIRCUIT_OPEN = "circuit_open"
    TOOL_NOT_REGISTERED = "tool_not_registered"


CANONICAL_SENTENCES: Dict[ReasonCode, str] = {
    ReasonCode.NON_BINARY_TREATMENT: "the treatment column is not a binary 0/1 indicator",
    ReasonCode.SINGLE_CLASS_TREATMENT: "the treatment column has only one class, so there is nothing to compare",
    ReasonCode.NON_BINARY_OUTCOME: "the outcome column is not a binary 0/1 indicator",
    ReasonCode.NO_USABLE_ROWS: "no rows remained that the analysis could use",
    ReasonCode.NO_USABLE_COLUMNS: "no usable columns of the required kind were present",
    ReasonCode.NON_FINITE_INPUT: "the inputs contained missing or non-finite values where finite numbers are required",
    ReasonCode.INSUFFICIENT_GROUPS: "fewer groups were present than the comparison requires",
    ReasonCode.INSUFFICIENT_SAMPLE: "too few observations were available to support an estimate",
    ReasonCode.COVERAGE_GAP: "the data does not cover everything the question asked about",
    ReasonCode.UNKNOWN_COLUMN: "a column the request named is not present in the data",
    ReasonCode.MISSING_DATAFRAME: "the real source data was not supplied to the tool",
    ReasonCode.CI_OUTSIDE_ESTIMATE: "the confidence interval is inconsistent with the point estimate",
    ReasonCode.POINT_ESTIMATE_ONLY: "no uncertainty could be computed for the estimate",
    ReasonCode.UPSTREAM_STEP_FAILED: "an earlier step this one depends on did not produce a result",
    ReasonCode.UNSUPPORTED_REQUEST: "the tool cannot answer a question of this shape",
    ReasonCode.MISSING_REQUIRED_INPUT: "a required input was not provided",
    ReasonCode.INVALID_INPUT_TYPE: "an input was of the wrong type",
    ReasonCode.INVALID_INPUT_VALUE: "an input value was outside what the tool accepts",
    ReasonCode.TOOL_ERROR: "the tool failed to complete",
    ReasonCode.TOOL_TIMEOUT: "the tool exceeded its time budget",
    ReasonCode.PLAN_DEFECT: "the plan called this tool incorrectly",
    ReasonCode.REFERENCE_UNRESOLVABLE: "the plan referred to a result that does not exist",
    ReasonCode.DEPENDENCY_UNMET: "a step this one depends on did not run",
    ReasonCode.CIRCUIT_OPEN: "the tool was temporarily withheld after repeated failures",
    ReasonCode.TOOL_NOT_REGISTERED: "the plan named a tool that is not registered",
}

# Bounds on the structured ``details`` payload. It is persisted, so it must stay
# structure: short scalar values, never a sentence smuggled back in as a string.
_MAX_DETAIL_KEYS = 8
_MAX_DETAIL_STR = 64


def canonical_sentence(code: Union[ReasonCode, str, None]) -> str:
    """The data-free sentence for ``code``; the generic one for anything unknown.

    Accepts a raw string because the recorder and the composer read codes back off
    Pydantic models, where the value has already been coerced to ``str``.
    """
    if isinstance(code, ReasonCode):
        return CANONICAL_SENTENCES[code]
    if isinstance(code, str):
        try:
            return CANONICAL_SENTENCES[ReasonCode(code)]
        except ValueError:
            pass
    return CANONICAL_SENTENCES[ReasonCode.TOOL_ERROR]


def validate_details(details: Dict[str, object]) -> Dict[str, object]:
    """Reject a ``details`` payload that is really free text. Returns it unchanged."""
    if len(details) > _MAX_DETAIL_KEYS:
        raise ValueError(f"details carries {len(details)} keys; at most {_MAX_DETAIL_KEYS}")
    for key, value in details.items():
        if isinstance(value, str) and len(value) > _MAX_DETAIL_STR:
            raise ValueError(
                f"details[{key!r}] is {len(value)} characters; details is structure, not prose "
                f"(at most {_MAX_DETAIL_STR})"
            )
    return details
```

- [ ] **Step 4: Modify `src/agents/tool_composer/errors.py`**

Add the import at the top of the module body (after `from __future__ import annotations`):

```python
from typing import Any, Dict, Optional, Union

from .reason_codes import ReasonCode, canonical_sentence, validate_details


class _CodedError(Exception):
    """Mixin: a deterministic tool objection that carries a closed reason code (#2021).

    The human ``message`` is preserved exactly — 96 assertions pin that prose, and
    #1574's ``estimation_data_scope`` disclosure rides it into the fail-closed answer.
    The code is what the learning loop aggregates and what the database stores; the
    canonical sentence is what is shown when the raw message is not safe to show.
    """

    def __init__(
        self,
        message: str,
        *,
        reason_code: Union[ReasonCode, str],
        details: Optional[Dict[str, Any]] = None,
    ):
        self.reason_code = ReasonCode(reason_code)
        self.details: Dict[str, Any] = validate_details(dict(details or {}))
        super().__init__(message)

    @property
    def canonical_sentence(self) -> str:
        return canonical_sentence(self.reason_code)
```

Then change the two classes to inherit it, keeping their existing docstrings verbatim and appending one paragraph each. `ToolInputError` must stay a `ValueError` and `ToolRefusalError` a `RuntimeError` — the MRO is `(_CodedError, ValueError)` and `(_CodedError, RuntimeError)`:

```python
class ToolInputError(_CodedError, ValueError):
    """<existing docstring, unchanged>

    Carries a :class:`~src.agents.tool_composer.reason_codes.ReasonCode` (#2021) so the
    learning loop can aggregate rejections by category rather than by prose.
    """


class ToolRefusalError(_CodedError, RuntimeError):
    """<existing docstring, unchanged>

    Carries a :class:`~src.agents.tool_composer.reason_codes.ReasonCode` (#2021) so the
    learning loop can aggregate refusals by category rather than by prose, and so the
    fail-closed answer can render a data-free sentence when the raw message is not
    safe to show (#2020).
    """
```

`ReferenceResolutionError` is **not** changed: it composes its own message from `reference` and `reason` and the executor assigns it `REFERENCE_UNRESOLVABLE` in Task 3.

- [ ] **Step 5: Run the test to verify it passes**

```bash
cd /home/enunez/Projects/e2i_causal_analytics/.worktrees/lane-refusal-codes
/home/enunez/Projects/e2i_causal_analytics/.venv/bin/pytest tests/unit/test_agents/test_tool_composer/test_reason_codes_2021.py -n 0 -p no:cacheprovider -q --timeout=120
```

Expected: `7 passed`.

- [ ] **Step 6: Confirm the 87 existing raise sites are now broken, and that this is the only breakage**

```bash
/home/enunez/Projects/e2i_causal_analytics/.venv/bin/pytest tests/unit/test_agents/test_tool_composer/test_nonretryable_refusals_1600.py -n 0 -p no:cacheprovider -q --timeout=300 2>&1 | tail -15
```

Expected: failures with `TypeError: __init__() missing 1 required keyword-only argument: 'reason_code'`. That is Task 2's work. **Do not commit Task 1 alone if the suite is red** — commit Task 1 and Task 2 together at the end of Task 2.

---

## Task 2: Give all 87 raise sites a code, and enforce it

**Files:**
- Modify: `src/agents/tool_composer/tool_registrations.py` (87 sites)
- Test: `tests/unit/test_agents/test_tool_composer/test_reason_code_coverage_2021.py`

**Hard rule: do not change a single character of any message string.** 96 assertions pin that prose (`grep -rn "ToolRefusalError\|ToolInputError" tests/ | wc -l` → 96), and #1574's scope disclosure rides the text. The only edit at each site is adding the `reason_code=` keyword (and `details=` where a scalar is already in hand).

- [ ] **Step 1: Write the failing enforcement test**

Create `tests/unit/test_agents/test_tool_composer/test_reason_code_coverage_2021.py`:

```python
"""#2021: every refusal raise site names a code. An AST check, so it cannot drift.

A grep would miss a multi-line call and match a comment. Walking the AST of the
module source finds every ``raise ToolRefusalError(...)`` / ``raise ToolInputError(...)``
however it is formatted.
"""

import ast
from pathlib import Path

import pytest

from src.agents.tool_composer.reason_codes import ReasonCode

_TARGETS = {"ToolRefusalError", "ToolInputError"}
_SOURCES = [
    Path("src/agents/tool_composer/tool_registrations.py"),
]


def _raise_sites(path: Path):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Raise) or not isinstance(node.exc, ast.Call):
            continue
        func = node.exc.func
        name = func.id if isinstance(func, ast.Name) else getattr(func, "attr", None)
        if name in _TARGETS:
            yield node.lineno, name, node.exc


@pytest.mark.parametrize("path", _SOURCES, ids=lambda p: p.name)
def test_every_raise_site_names_a_reason_code(path: Path):
    uncoded = []
    for lineno, name, call in _raise_sites(path):
        keywords = {kw.arg for kw in call.keywords}
        if "reason_code" not in keywords:
            uncoded.append(f"{path}:{lineno} {name}")
    assert uncoded == [], (
        "raise sites without a reason_code (#2021):\n  " + "\n  ".join(uncoded)
    )


@pytest.mark.parametrize("path", _SOURCES, ids=lambda p: p.name)
def test_every_reason_code_is_a_member_of_the_closed_set(path: Path):
    """A literal string or an unknown ReasonCode attribute is a drift vector."""
    members = {c.name for c in ReasonCode}
    bad = []
    for lineno, name, call in _raise_sites(path):
        for kw in call.keywords:
            if kw.arg != "reason_code":
                continue
            value = kw.value
            if isinstance(value, ast.Attribute) and isinstance(value.value, ast.Name):
                if value.value.id != "ReasonCode" or value.attr not in members:
                    bad.append(f"{path}:{lineno} {ast.unparse(value)}")
            else:
                bad.append(f"{path}:{lineno} {ast.unparse(value)} (not ReasonCode.MEMBER)")
    assert bad == [], "reason_code values outside the closed set:\n  " + "\n  ".join(bad)


def test_the_site_count_is_what_the_lane_measured():
    """A floor, not a ceiling: new sites are fine, a silent drop to zero is not."""
    total = sum(1 for path in _SOURCES for _ in _raise_sites(path))
    assert total >= 87, f"expected at least the 87 sites measured for #2021, found {total}"
```

- [ ] **Step 2: Run it to verify it fails**

```bash
cd /home/enunez/Projects/e2i_causal_analytics/.worktrees/lane-refusal-codes
/home/enunez/Projects/e2i_causal_analytics/.venv/bin/pytest tests/unit/test_agents/test_tool_composer/test_reason_code_coverage_2021.py -n 0 -p no:cacheprovider -q --timeout=120
```

Expected: FAIL listing ~87 uncoded sites.

- [ ] **Step 3: Enumerate the sites**

```bash
/home/enunez/Projects/e2i_causal_analytics/.venv/bin/python - <<'PY'
import ast
from pathlib import Path
p = Path("src/agents/tool_composer/tool_registrations.py")
tree = ast.parse(p.read_text())
lines = p.read_text().splitlines()
for node in ast.walk(tree):
    if isinstance(node, ast.Raise) and isinstance(node.exc, ast.Call):
        f = node.exc.func
        name = f.id if isinstance(f, ast.Name) else getattr(f, "attr", None)
        if name in {"ToolRefusalError", "ToolInputError"}:
            print(f"{node.lineno}\t{name}\t{lines[node.lineno-1].strip()[:110]}")
PY
```

- [ ] **Step 4: Assign a code at each site**

Add `ReasonCode` to the module's imports:

```python
from .reason_codes import ReasonCode
```

Then at each site add the keyword. Example shape (the message is untouched):

```python
raise ToolRefusalError(
    f"causal_effect_estimator: treatment column {treatment!r} carries "
    f"{n_distinct} distinct non-null values, including {sample!r}. Refusing to "
    f"report a difference between only the rows equal to 1 and those equal to 0.",
    reason_code=ReasonCode.NON_BINARY_TREATMENT,
    details={"n_distinct": n_distinct},
)
```

Mapping rule — read the message, then pick the member whose canonical sentence a pharma leader would accept as the reason:

| Message says | Code |
|---|---|
| treatment/outcome is not 0/1, more than two values | `NON_BINARY_TREATMENT` / `NON_BINARY_OUTCOME` |
| treatment has one class / one group only | `SINGLE_CLASS_TREATMENT` |
| no rows left after filtering/dropna, empty frame | `NO_USABLE_ROWS` |
| no usable numeric/feature columns | `NO_USABLE_COLUMNS` |
| NaN / inf / non-finite where finite required | `NON_FINITE_INPUT` |
| fewer than N groups/segments/arms for the comparison | `INSUFFICIENT_GROUPS` |
| n below a minimum for the estimate | `INSUFFICIENT_SAMPLE` |
| the frame covers only some requested entities/brands (#1574) | `COVERAGE_GAP` |
| a named column is absent from the frame | `UNKNOWN_COLUMN` |
| "requires a real DataFrame supplied via one of the kwargs keys", "does not fabricate" | `MISSING_DATAFRAME` |
| CI does not bracket / contradicts the point estimate | `CI_OUTSIDE_ESTIMATE` |
| point estimate with no uncertainty (#2014) | `POINT_ESTIMATE_ONLY` |
| the upstream result reports `status='failed'` / did not complete | `UPSTREAM_STEP_FAILED` |
| the tool cannot answer this question shape at all | `UNSUPPORTED_REQUEST` |
| a required kwarg is absent or `None` | `MISSING_REQUIRED_INPUT` |
| "must be a dict", "got {type(...).__name__}" | `INVALID_INPUT_TYPE` |
| a value is out of range / not one of the accepted values | `INVALID_INPUT_VALUE` |

If a site fits none of these, **add a member to `ReasonCode` and its sentence to `CANONICAL_SENTENCES` in the same edit** and note it in the commit message. Do not reach for `TOOL_ERROR` — that code means "no tool authored a reason".

Add `details` only where a scalar is already computed at the site (`n_distinct`, `n_rows`, `n_groups`). Never put a message, a column name list, or a value repr in `details`.

- [ ] **Step 5: Run the enforcement test and the prose-pinning suites**

```bash
cd /home/enunez/Projects/e2i_causal_analytics/.worktrees/lane-refusal-codes
/home/enunez/Projects/e2i_causal_analytics/.venv/bin/pytest tests/unit/test_agents/test_tool_composer/test_reason_code_coverage_2021.py \
  tests/unit/test_agents/test_tool_composer/test_reason_codes_2021.py -n 0 -p no:cacheprovider -q --timeout=300
/home/enunez/Projects/e2i_causal_analytics/.venv/bin/pytest tests/unit/test_agents/test_tool_composer/ -n 0 -p no:cacheprovider -q --timeout=900 2>&1 | tail -15
```

Expected: the two new files all pass; `tests/unit/test_agents/test_tool_composer/` is green with the same counts as before the lane (record the baseline first with `git stash`-free means: run it on `origin/main` in a scratch clone if the number is disputed).

- [ ] **Step 6: Verify no message text changed**

```bash
git diff -U0 src/agents/tool_composer/tool_registrations.py \
  | grep '^[-+]' | grep -v '^[-+][-+]' | grep -vE '^\+\s*(reason_code=|details=|from \.reason_codes)' \
  | grep '^-' | head -20
```

Expected: **no output.** Any removed line that is not a pure re-indent of an unchanged string is a message edit — revert it.

- [ ] **Step 7: Lint and commit Tasks 1 + 2 together**

```bash
/home/enunez/Projects/e2i_causal_analytics/.venv/bin/ruff check --no-cache src/agents/tool_composer/ tests/unit/test_agents/test_tool_composer/
/home/enunez/Projects/e2i_causal_analytics/.venv/bin/ruff format --check src/agents/tool_composer/reason_codes.py src/agents/tool_composer/errors.py
git add src/agents/tool_composer/reason_codes.py src/agents/tool_composer/errors.py \
        src/agents/tool_composer/tool_registrations.py \
        tests/unit/test_agents/test_tool_composer/test_reason_codes_2021.py \
        tests/unit/test_agents/test_tool_composer/test_reason_code_coverage_2021.py
git commit -m "feat(tool-composer): closed reason-code vocabulary on every tool refusal (#2021)

ToolRefusalError/ToolInputError now require a ReasonCode from a closed set with a
data-free canonical sentence. All 87 raise sites in tool_registrations.py carry one,
enforced by an AST test. No message text changed: 96 assertions pin that prose and
 #1574's estimation_data_scope disclosure rides it into the fail-closed answer.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_017AdJUYq6yCMahuwKuFfWJw"
```

---

## Task 3: Propagate the code through the executor

**Files:**
- Modify: `src/agents/tool_composer/models/composition_models.py:319-345`
- Modify: `src/agents/tool_composer/executor.py` (6 failure arms)
- Test: `tests/unit/test_agents/test_tool_composer/test_executor_reason_codes_2021.py`

The executor already computes `outcome_class` on every arm. The code is the finer-grained sibling: tool-authored where a coded exception was caught, executor-assigned otherwise.

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_agents/test_tool_composer/test_executor_reason_codes_2021.py`:

```python
"""#2021: the executor carries a reason code out of every failure arm."""

import asyncio

import pytest

from src.agents.tool_composer.errors import ToolInputError, ToolRefusalError
from src.agents.tool_composer.reason_codes import ReasonCode

pytestmark = pytest.mark.asyncio


async def test_refusal_code_reaches_the_step_result(executor_with_tool):
    """A tool-authored code is carried, not re-derived."""

    def refusing_tool(**kwargs):
        raise ToolRefusalError(
            "treatment column carries 4 distinct values",
            reason_code=ReasonCode.NON_BINARY_TREATMENT,
            details={"n_distinct": 4},
        )

    result = await executor_with_tool(refusing_tool)
    assert result.outcome_class == "refused"
    assert result.reason_code == "non_binary_treatment"
    assert result.reason_details == {"n_distinct": 4}


async def test_input_error_code_reaches_the_step_result(executor_with_tool):
    def rejecting_tool(**kwargs):
        raise ToolInputError(
            "expected_effect must not be None",
            reason_code=ReasonCode.MISSING_REQUIRED_INPUT,
        )

    result = await executor_with_tool(rejecting_tool)
    assert result.outcome_class == "input_rejected"
    assert result.reason_code == "missing_required_input"


async def test_generic_exception_gets_the_executor_assigned_code(executor_with_tool):
    """No tool authored a reason, so the executor supplies the generic one."""

    def exploding_tool(**kwargs):
        raise RuntimeError(
            "DoWhy estimate_effect failed for method_name='backdoor.linear_regression': "
            "Found array with 0 sample(s) (shape=(0,)) while a minimum of 1 is required."
        )

    result = await executor_with_tool(exploding_tool)
    assert result.outcome_class == "error"
    assert result.reason_code == "tool_error"
    assert result.reason_details == {}


async def test_timeout_gets_the_timeout_code(executor_with_tool):
    async def slow_tool(**kwargs):
        await asyncio.sleep(30)

    result = await executor_with_tool(slow_tool, timeout_seconds=0.05)
    assert result.outcome_class == "timeout"
    assert result.reason_code == "tool_timeout"


async def test_every_failed_step_result_has_a_code(executor_with_tool):
    """The invariant the recorder and the composer both rely on."""

    def refusing_tool(**kwargs):
        raise ToolRefusalError("x", reason_code=ReasonCode.NO_USABLE_ROWS)

    result = await executor_with_tool(refusing_tool)
    assert result.reason_code is not None
```

The `executor_with_tool` fixture belongs in this file. Build it by copying the executor construction already used in `tests/unit/test_agents/test_tool_composer/test_executor_outcome_classes.py` — read that file first and reuse its fixtures rather than inventing a second way to build a `ToolExecutor`.

- [ ] **Step 2: Run it to verify it fails**

```bash
/home/enunez/Projects/e2i_causal_analytics/.venv/bin/pytest tests/unit/test_agents/test_tool_composer/test_executor_reason_codes_2021.py -n 0 -p no:cacheprovider -q --timeout=300
```

Expected: FAIL — `AttributeError: 'StepResult' object has no attribute 'reason_code'`.

- [ ] **Step 3: Add the fields to `StepResult`**

In `src/agents/tool_composer/models/composition_models.py`, after `error_type: Optional[str] = None` (line ~345):

```python
    # #2021: the closed reason code for this step's failure. Tool-authored when a
    # ToolRefusalError/ToolInputError was caught, executor-assigned otherwise. This is
    # the aggregation key the learning loop stores — the raw message is never stored.
    reason_code: Optional[str] = None
    reason_details: Dict[str, Any] = Field(default_factory=dict)
```

- [ ] **Step 4: Set it on every failure arm of `executor.py`**

Add the import beside the existing errors import (line 30):

```python
from .reason_codes import ReasonCode
```

Then, at each `StepResult(...)` construction on a failure path, add the pair. The exact arms and their codes:

| `executor.py` anchor | `outcome_class` | add |
|---|---|---|
| line ~547 | `dependency_unmet` | `reason_code=ReasonCode.DEPENDENCY_UNMET.value` |
| line ~576 | `plan_defect` | `reason_code=ReasonCode.REFERENCE_UNRESOLVABLE.value` if the caught exception is a `ReferenceResolutionError`, else `ReasonCode.PLAN_DEFECT.value` |
| line ~664 | `circuit_open` | `reason_code=ReasonCode.CIRCUIT_OPEN.value` |
| line ~687 | `not_registered` | `reason_code=ReasonCode.TOOL_NOT_REGISTERED.value` |
| line ~764 (`PlanArgumentError`) | `plan_defect` | `reason_code=ReasonCode.PLAN_DEFECT.value` |
| line ~794 (`ToolInputError`/`ToolRefusalError`) | `input_rejected`/`refused` | `reason_code=e.reason_code.value`, `reason_details=e.details` |
| line ~903 (`SyncToolTimeout`) | `timeout` | `reason_code=ReasonCode.TOOL_TIMEOUT.value` |
| line ~946 (retries exhausted) | `timeout`/`error` | `reason_code=(ReasonCode.TOOL_TIMEOUT if isinstance(last_exc, (asyncio.TimeoutError, TimeoutError)) else ReasonCode.TOOL_ERROR).value` |

The last arm's `outcome_class` already branches on `isinstance(last_exc, ...)`; mirror that branch exactly so the class and the code can never disagree.

- [ ] **Step 5: Run the test to verify it passes**

```bash
/home/enunez/Projects/e2i_causal_analytics/.venv/bin/pytest tests/unit/test_agents/test_tool_composer/test_executor_reason_codes_2021.py \
  tests/unit/test_agents/test_tool_composer/test_executor_outcome_classes.py \
  tests/unit/test_agents/test_tool_composer/test_executor.py -n 0 -p no:cacheprovider -q --timeout=600
```

Expected: all pass.

- [ ] **Step 6: Commit**

```bash
/home/enunez/Projects/e2i_causal_analytics/.venv/bin/ruff check --no-cache src/agents/tool_composer/ tests/unit/test_agents/test_tool_composer/
git add src/agents/tool_composer/models/composition_models.py src/agents/tool_composer/executor.py \
        tests/unit/test_agents/test_tool_composer/test_executor_reason_codes_2021.py
git commit -m "feat(tool-composer): carry the reason code out of every executor failure arm (#2021)

StepResult gains reason_code + reason_details. Tool-authored codes come off the caught
ToolRefusalError/ToolInputError; the other six arms are executor-assigned from the same
branch that picks outcome_class, so the two can never disagree.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_017AdJUYq6yCMahuwKuFfWJw"
```

---

## Task 4: Keep library internals out of the fail-closed answer (#2020)

**Files:**
- Modify: `src/agents/tool_composer/composer.py:1217-1292` (`_create_total_failure_result`)
- Test: `tests/unit/test_agents/test_tool_composer/test_fail_closed_answer_sanitization_2020.py`

The rule, from #2020: **a tool-authored refusal reaches the answer verbatim; anything else is replaced by its canonical sentence, and the raw text goes to the log.** `outcome_class in ("refused", "input_rejected")` is exactly "a tool authored this" — it is set only in the arm that catches the coded exceptions.

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_agents/test_tool_composer/test_fail_closed_answer_sanitization_2020.py`:

```python
"""#2020: the fail-closed answer keeps honest refusals and drops library internals."""

import logging
from datetime import datetime, timezone

import pytest

from src.agents.tool_composer.models.composition_models import (
    ExecutionTrace,
    StepResult,
    ToolInput,
    ToolOutput,
)

_DOWHY_INTERNALS = (
    "DoWhy estimate_effect failed for method_name='backdoor.linear_regression': "
    "Found array with 0 sample(s) (shape=(0,)) while a minimum of 1 is required."
)
_REFUSAL = (
    "gap_calculator: the estimation data covers only brand 'Fabhalta'; a brand-vs-brand "
    "gap needs at least two brands. Refusing to report a gap against a single brand."
)


def _step(tool_name, error, outcome_class, reason_code):
    now = datetime.now(timezone.utc)
    return StepResult(
        step_id=f"s_{tool_name}",
        sub_question_id="Q1",
        tool_name=tool_name,
        input=ToolInput(tool_name=tool_name, parameters={}),
        output=ToolOutput(tool_name=tool_name, success=False, error=error),
        status="FAILED",
        started_at=now,
        completed_at=now,
        outcome_class=outcome_class,
        reason_code=reason_code,
        error_type="ToolRefusalError" if outcome_class == "refused" else "RuntimeError",
    )


def _trace(*steps):
    trace = ExecutionTrace(plan_id="p1")
    for step in steps:
        trace.add_result(step)
    return trace


def test_tool_authored_refusal_reaches_the_answer_verbatim(composer):
    trace = _trace(_step("gap_calculator", _REFUSAL, "refused", "coverage_gap"))
    result = composer._create_total_failure_result(
        query="q", decomposition=None, plan=None, execution_trace=trace,
        started_at=datetime.now(timezone.utc), phase_durations={},
    )
    assert "Refusing to report a gap against a single brand" in result.response.answer


def test_library_internals_do_not_reach_the_answer(composer, caplog):
    trace = _trace(_step("causal_effect_estimator", _DOWHY_INTERNALS, "error", "tool_error"))
    with caplog.at_level(logging.WARNING):
        result = composer._create_total_failure_result(
            query="q", decomposition=None, plan=None, execution_trace=trace,
            started_at=datetime.now(timezone.utc), phase_durations={},
        )
    answer = result.response.answer
    for leak in ("DoWhy", "backdoor.linear_regression", "shape=(0,)", "sklearn", "Traceback"):
        assert leak not in answer, f"{leak!r} leaked into the user-facing answer"
    assert "the tool failed to complete" in answer
    assert "tool_error" in answer
    assert _DOWHY_INTERNALS in caplog.text, "the raw text must still be logged"


def test_caveats_and_errors_are_sanitized_too(composer):
    """The answer is not the only field the caller renders."""
    trace = _trace(_step("causal_effect_estimator", _DOWHY_INTERNALS, "error", "tool_error"))
    result = composer._create_total_failure_result(
        query="q", decomposition=None, plan=None, execution_trace=trace,
        started_at=datetime.now(timezone.utc), phase_durations={},
    )
    assert not any("DoWhy" in c for c in result.response.caveats)
    assert not any("DoWhy" in e for e in result.errors)


def test_a_mixed_composition_keeps_the_refusal_and_drops_the_internals(composer):
    trace = _trace(
        _step("gap_calculator", _REFUSAL, "refused", "coverage_gap"),
        _step("causal_effect_estimator", _DOWHY_INTERNALS, "error", "tool_error"),
    )
    result = composer._create_total_failure_result(
        query="q", decomposition=None, plan=None, execution_trace=trace,
        started_at=datetime.now(timezone.utc), phase_durations={},
    )
    answer = result.response.answer
    assert "Refusing to report a gap against a single brand" in answer
    assert "DoWhy" not in answer


def test_a_step_with_no_code_is_sanitized_not_passed_through(composer):
    """Fail closed: an uncoded failure is treated as untrusted text."""
    trace = _trace(_step("mystery_tool", "some raw text", "error", None))
    result = composer._create_total_failure_result(
        query="q", decomposition=None, plan=None, execution_trace=trace,
        started_at=datetime.now(timezone.utc), phase_durations={},
    )
    assert "some raw text" not in result.response.answer
```

The `composer` fixture builds a `ToolComposer` without network dependencies — reuse the construction already used in the composer tests under `tests/unit/test_agents/test_tool_composer/`; read `test_executor_outcome_classes.py` and the existing composer tests first and follow whichever pattern is there. `_create_total_failure_result` is called directly, so no LLM client is exercised.

- [ ] **Step 2: Run it to verify it fails**

```bash
/home/enunez/Projects/e2i_causal_analytics/.venv/bin/pytest tests/unit/test_agents/test_tool_composer/test_fail_closed_answer_sanitization_2020.py -n 0 -p no:cacheprovider -q --timeout=300
```

Expected: the internals tests FAIL — `'DoWhy' leaked into the user-facing answer`.

- [ ] **Step 3: Implement the split in `composer.py`**

Add near the other imports at the top of the module:

```python
from .reason_codes import canonical_sentence
```

Replace the reason-collection loop inside `_create_total_failure_result` (currently lines ~1244-1254) with:

```python
        # #2020: what a failed step is allowed to say to the user.
        #
        # A tool-authored refusal (ToolRefusalError / ToolInputError — the only two
        # exceptions that produce these outcome classes) is an honest, user-meaningful
        # finding: #1574's gap_calculator states which entity groups the estimation data
        # actually covered, and a one-step plan fail-closes here, before synthesis, so
        # dropping it would leave the answer LESS informative. Those reach the user
        # verbatim, exactly as they did before this change.
        #
        # Every other failure carries library internals — DoWhy/sklearn shape errors, DB
        # driver messages, file paths, reprs of inputs. They are not meaningful to a
        # pharma leader, they read as a crash rather than a finding, and they expose
        # implementation detail. Those are replaced by the closed code's canonical
        # sentence; the raw text keeps its home in the log and the audit trail.
        _TOOL_AUTHORED = ("refused", "input_rejected")
        for step in getattr(execution_trace, "step_results", None) or []:
            output = getattr(step, "output", None)
            # ``is_success`` (success AND a result present) is the model's own
            # definition of a successful step — it is what ``ExecutionTrace``
            # counts and what ``get_all_outputs`` returns, so the failed-step
            # collector must agree with it or a ``success=True, result=None``
            # step would be counted failed and then listed nowhere.
            if getattr(output, "is_success", False):
                continue
            tool_name = str(
                getattr(step, "tool_name", None) or getattr(output, "tool_name", None) or "unknown"
            )
            failed_tools.append(tool_name)
            raw = str(getattr(output, "error", None) or "").strip()
            outcome_class = getattr(step, "outcome_class", None)
            reason_code = getattr(step, "reason_code", None)
            if outcome_class in _TOOL_AUTHORED and reason_code:
                if raw:
                    reasons.append(f"{tool_name}: {raw}")
                continue
            # Not tool-authored (or uncoded, which fails closed the same way): log the
            # raw text, tell the user the category.
            if raw:
                logger.warning(
                    "Step %s tool %r failed with non-user-facing text (reason_code=%s): %s",
                    getattr(step, "step_id", "?"),
                    tool_name,
                    reason_code,
                    raw,
                )
            reasons.append(f"{tool_name}: {canonical_sentence(reason_code)} [{reason_code or 'tool_error'}]")
```

`answer`, `caveats` and `errors` are all built from `reasons` below this loop and need no further change — sanitizing at the source covers all three.

- [ ] **Step 4: Run the test to verify it passes**

```bash
/home/enunez/Projects/e2i_causal_analytics/.venv/bin/pytest tests/unit/test_agents/test_tool_composer/test_fail_closed_answer_sanitization_2020.py -n 0 -p no:cacheprovider -q --timeout=300
```

Expected: `6 passed`.

- [ ] **Step 5: Confirm no existing fail-closed expectation regressed**

```bash
/home/enunez/Projects/e2i_causal_analytics/.venv/bin/pytest tests/unit/test_agents/test_tool_composer/ -n 0 -p no:cacheprovider -q --timeout=900 2>&1 | tail -12
grep -rln "Unable to complete analysis\|Reason(s):" tests/ | head
```

Run every file that grep names. A test that asserted raw exception text in the answer is asserting the defect — **update it and say so in the commit body**, do not weaken the new test.

- [ ] **Step 6: Commit**

```bash
/home/enunez/Projects/e2i_causal_analytics/.venv/bin/ruff check --no-cache src/agents/tool_composer/ tests/unit/test_agents/test_tool_composer/
git add src/agents/tool_composer/composer.py \
        tests/unit/test_agents/test_tool_composer/test_fail_closed_answer_sanitization_2020.py
git commit -m "fix(tool-composer): keep library internals out of the fail-closed answer (#2020)

Tool-authored refusals (outcome_class refused/input_rejected, the only classes the coded
exceptions produce) still reach the user verbatim — #1574's coverage disclosure depends on
it. Every other failure now renders its reason code's canonical sentence; the raw DoWhy /
sklearn / driver text goes to the log only.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_017AdJUYq6yCMahuwKuFfWJw"
```

---

## Task 5: The recorder emits the code and a canonical sentence (#2050 cause 1)

**Files:**
- Modify: `src/agents/tool_composer/learning_recorder.py:262-291` (`step_record`)
- Test: `tests/unit/test_agents/test_tool_composer/test_learning_recorder_reason_code_2050.py`

Note the finding: `StepResult` has **no** `error_message`. The canonical sentence is derived from `result.reason_code`, not read off the result.

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_agents/test_tool_composer/test_learning_recorder_reason_code_2050.py`:

```python
"""#2050 cause 1: the recorder's step row carries the reason, not just the class."""

from datetime import datetime, timezone

from src.agents.tool_composer.learning_recorder import step_record
from src.agents.tool_composer.models.composition_models import (
    StepResult,
    ToolInput,
    ToolOutput,
)
from src.agents.tool_composer.reason_codes import ReasonCode, canonical_sentence

_RAW = (
    "causal_effect_estimator: treatment column 'rep_detailing' carries 4 distinct "
    "non-null values, including [0, 1, 2, 3]."
)


def _refused_step():
    now = datetime.now(timezone.utc)
    return StepResult(
        step_id="s1",
        sub_question_id="Q1",
        tool_name="causal_effect_estimator",
        input=ToolInput(tool_name="causal_effect_estimator", parameters={}),
        output=ToolOutput(tool_name="causal_effect_estimator", success=False, error=_RAW),
        status="FAILED",
        started_at=now,
        completed_at=now,
        outcome_class="refused",
        attempts=1,
        error_type="ToolRefusalError",
        reason_code=ReasonCode.NON_BINARY_TREATMENT.value,
        reason_details={"n_distinct": 4},
    )


def test_step_row_carries_the_reason_code():
    row = step_record(1, _refused_step(), plan=None, allowlist=None)
    assert row["reason_code"] == "non_binary_treatment"


def test_step_row_error_message_is_the_canonical_sentence():
    row = step_record(1, _refused_step(), plan=None, allowlist=None)
    assert row["error_message"] == canonical_sentence(ReasonCode.NON_BINARY_TREATMENT)


def test_the_raw_message_is_never_emitted():
    """ml/041's contract: structure only, no error text (041 header, spec 5.5)."""
    row = step_record(1, _refused_step(), plan=None, allowlist=None)
    blob = repr(row)
    for fragment in ("rep_detailing", "[0, 1, 2, 3]", "distinct non-null"):
        assert fragment not in blob, f"{fragment!r} reached the recorded payload"


def test_reason_details_are_bounded_structure():
    row = step_record(1, _refused_step(), plan=None, allowlist=None)
    assert row["reason_details"] == {"n_distinct": 4}


def test_a_succeeded_step_carries_no_reason():
    now = datetime.now(timezone.utc)
    ok = StepResult(
        step_id="s1", sub_question_id="Q1", tool_name="causal_effect_estimator",
        input=ToolInput(tool_name="causal_effect_estimator", parameters={}),
        output=ToolOutput(tool_name="causal_effect_estimator", success=True, result={"ate": 0.1}),
        status="COMPLETED", started_at=now, completed_at=now,
        outcome_class="succeeded", attempts=1,
    )
    row = step_record(1, ok, plan=None, allowlist=None)
    assert row["reason_code"] is None
    assert row["error_message"] is None
```

Check `step_record`'s real signature before writing the calls — read `src/agents/tool_composer/learning_recorder.py:240-292` and match it exactly (it takes the step number, the result, the plan, and the allowlist; `plan=None` must be a supported path, and if it is not, build a minimal `ExecutionPlan` the way `test_learning_recorder_serializer.py` does).

- [ ] **Step 2: Run it to verify it fails**

```bash
/home/enunez/Projects/e2i_causal_analytics/.venv/bin/pytest tests/unit/test_agents/test_tool_composer/test_learning_recorder_reason_code_2050.py -n 0 -p no:cacheprovider -q --timeout=300
```

Expected: FAIL — `KeyError: 'reason_code'`.

- [ ] **Step 3: Add the three keys to the step-row dict**

In `step_record`, after `"error_type": result.error_type,`:

```python
        "error_type": result.error_type,
        # #2050: an operator seeing `refused` with no why is a missing explanation on a
        # decision-support surface. The CODE is the aggregation key; error_message is the
        # code's fixed catalogue sentence, never the tool's own message — that message
        # interpolates column names and value reprs, and ml/041's recording contract is
        # structure only ("no error text is stored", 041 header).
        "reason_code": result.reason_code,
        "reason_details": result.reason_details or {},
        "error_message": (canonical_sentence(result.reason_code) if result.reason_code else None),
```

and import at the top of the module:

```python
from .reason_codes import canonical_sentence
```

- [ ] **Step 4: Run the test to verify it passes**

```bash
/home/enunez/Projects/e2i_causal_analytics/.venv/bin/pytest tests/unit/test_agents/test_tool_composer/test_learning_recorder_reason_code_2050.py \
  tests/unit/test_agents/test_tool_composer/test_learning_recorder_serializer.py -n 0 -p no:cacheprovider -q --timeout=300
```

Expected: all pass.

- [ ] **Step 5: Commit**

```bash
/home/enunez/Projects/e2i_causal_analytics/.venv/bin/ruff check --no-cache src/agents/tool_composer/ tests/unit/test_agents/test_tool_composer/
git add src/agents/tool_composer/learning_recorder.py \
        tests/unit/test_agents/test_tool_composer/test_learning_recorder_reason_code_2050.py
git commit -m "fix(tool-composer): recorder emits reason_code and a canonical sentence (#2050 cause 1)

The issue said StepResult.error_message exists and the executor populates it; it does not
 — schemas.py's error_message is on CompositionResult, and the step's text lives on
output.error. The recorded row now carries the closed code plus the code's fixed catalogue
sentence, so ml/041's no-error-text contract holds while an operator can finally see why.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_017AdJUYq6yCMahuwKuFfWJw"
```

---

## Task 6: Migration ml/042 — the RPC drops the value on the floor (#2050 cause 2)

**Files:**
- Create: `database/ml/042_composer_refusal_reason_codes.sql`
- Create: `database/ml/rollback_042.sql`
- Test: `tests/unit/test_database/learning_loop/test_042_reason_codes.py`

**Before writing the file:** confirm `042` is still free — another lane may have taken it.

```bash
ls database/ml/ | sort
git fetch origin main --quiet && git ls-tree origin/main --name-only database/ml/ | sort
```

If `042` is taken, use the next free number consistently in the filename, the rollback, the ledger key (`ml/<file>`) and this plan's remaining steps.

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_database/learning_loop/test_042_reason_codes.py`. Read `tests/unit/test_database/learning_loop/test_041_recording.py` first and follow its harness exactly — it already knows how to apply a migration and call an RPC against the local database, including how it skips when no database is reachable.

```python
"""#2050 cause 2: the RPC must stop writing a hardcoded NULL into error_message.

The trap this test exists to avoid: a payload-level assertion passes while the column
stays NULL. Every assertion here reads the PERSISTED ROW.
"""

# Follow test_041_recording.py's fixtures for connection + migration application.


def test_ml_041_has_the_hardcoded_null_this_migration_removes():
    """A characterization test: prove the defect is where the issue says it is."""
    sql = Path("database/ml/041_composer_learning_loop_recording.sql").read_text()
    insert = sql.split("INSERT INTO composition_steps (")[1]
    assert "error_message, retry_count" in insert, "the column list changed; re-locate the slot"


def test_composition_steps_has_reason_code(db):
    cols = db.columns("composition_steps")
    assert "reason_code" in cols


def test_tool_performance_has_reason_code(db):
    cols = db.columns("tool_performance")
    assert "reason_code" in cols


def test_record_steps_persists_reason_code_and_error_message(db, seed):
    """The row, not the payload."""
    db.rpc(
        "composer_record_steps",
        p_seed=seed,
        p_steps=[{
            "step_number": 1,
            "tool_name": "causal_effect_estimator",
            "outcome_class": "refused",
            "attempts": 1,
            "cache_hit": False,
            "error_type": "ToolRefusalError",
            "reason_code": "non_binary_treatment",
            "error_message": "the treatment column is not a binary 0/1 indicator",
            "started_at": "2026-09-12T12:00:00Z",
            "completed_at": "2026-09-12T12:00:01Z",
            "latency_ms": 1000,
        }],
    )
    row = db.one("SELECT reason_code, error_message FROM composition_steps WHERE step_number = 1")
    assert row["reason_code"] == "non_binary_treatment"
    assert row["error_message"] == "the treatment column is not a binary 0/1 indicator"


def test_error_message_is_bounded(db, seed):
    db.rpc("composer_record_steps", p_seed=seed, p_steps=[{
        "step_number": 2, "tool_name": "causal_effect_estimator",
        "outcome_class": "refused", "attempts": 1, "cache_hit": False,
        "reason_code": "no_usable_rows", "error_message": "x" * 5000,
        "started_at": "2026-09-12T12:00:00Z", "completed_at": "2026-09-12T12:00:01Z",
    }])
    row = db.one("SELECT error_message FROM composition_steps WHERE step_number = 2")
    assert len(row["error_message"]) <= 200, "a long message must be truncated, not dropped"


def test_a_malformed_reason_code_is_rejected_not_stored(db, seed):
    """The format guard, mirroring how ml/041 guards error_type."""
    db.rpc("composer_record_steps", p_seed=seed, p_steps=[{
        "step_number": 3, "tool_name": "causal_effect_estimator",
        "outcome_class": "refused", "attempts": 1, "cache_hit": False,
        "reason_code": "DROP TABLE composition_steps; --",
        "started_at": "2026-09-12T12:00:00Z", "completed_at": "2026-09-12T12:00:01Z",
    }])
    row = db.one("SELECT reason_code FROM composition_steps WHERE step_number = 3")
    assert row["reason_code"] is None


def test_reliability_reports_the_most_common_refusal_reason(db, seed):
    """get_tool_reliability reads tool_performance, so the code must land there too."""
    for n, code in ((10, "non_binary_treatment"), (11, "non_binary_treatment"), (12, "no_usable_rows")):
        db.rpc("composer_record_steps", p_seed=seed, p_steps=[{
            "step_number": n, "tool_name": "causal_effect_estimator",
            "outcome_class": "refused", "attempts": 1, "cache_hit": False,
            "reason_code": code,
            "started_at": "2026-09-12T12:00:00Z", "completed_at": "2026-09-12T12:00:01Z",
        }])
    row = db.one(
        "SELECT most_common_refusal_reason FROM get_tool_reliability(30, true) "
        "WHERE tool_name = 'causal_effect_estimator'"
    )
    assert row["most_common_refusal_reason"] == "non_binary_treatment"
```

- [ ] **Step 2: Run it to verify it fails**

```bash
/home/enunez/Projects/e2i_causal_analytics/.venv/bin/pytest tests/unit/test_database/learning_loop/test_042_reason_codes.py -n 0 -p no:cacheprovider -q --timeout=600
```

Expected: FAIL on the column and persistence assertions. If every test SKIPS, the database is unreachable — fix that before continuing; a skipped migration test is not evidence.

- [ ] **Step 3: Write `database/ml/042_composer_refusal_reason_codes.sql`**

```sql
-- ============================================================================
-- ml/042 — refusal reason codes on the recording path (#2021, #2050)
--
-- WHY. A refusal's reason was not persisted, for TWO independent reasons:
--   1. the recorder never sent it (fixed in src/agents/tool_composer/learning_recorder.py);
--   2. this file's predecessor, ml/041, listed error_message in composer_record_steps'
--      INSERT column list but supplied a hardcoded NULL in that slot (041 line 631).
-- Either alone yields error_message IS NULL, so the table cannot tell them apart; only
-- reading the function body separates them. Both are fixed together or neither is.
--
-- WHAT IS STORED. ml/041's contract is structure only — "no error text is stored". That
-- is preserved: what lands in error_message is the CLOSED reason code's fixed catalogue
-- sentence (src/agents/tool_composer/reason_codes.py), never the tool's own message,
-- which interpolates column names and value reprs. reason_code is the aggregation key.
--
-- Applied by scripts/run_migrations.sh inside --single-transaction with its ledger row
-- (key: ml/042_composer_refusal_reason_codes.sql). Re-applying it is a no-op for data.
-- ============================================================================

-- ---------------------------------------------------------------------------
-- 1. Columns
-- ---------------------------------------------------------------------------
ALTER TABLE composition_steps ADD COLUMN IF NOT EXISTS reason_code text;
ALTER TABLE tool_performance  ADD COLUMN IF NOT EXISTS reason_code text;

-- Format guard only, mirroring how ml/041 guards error_type. The authoritative closed
-- set lives in Python (reason_codes.ReasonCode); duplicating the member list here would
-- drift the moment a member is added, and a migration is the worst place to learn that.
ALTER TABLE composition_steps DROP CONSTRAINT IF EXISTS composition_steps_reason_code_format;
ALTER TABLE composition_steps ADD CONSTRAINT composition_steps_reason_code_format
    CHECK (reason_code IS NULL OR reason_code ~ '^[a-z][a-z0-9_]{0,63}$');
ALTER TABLE tool_performance DROP CONSTRAINT IF EXISTS tool_performance_reason_code_format;
ALTER TABLE tool_performance ADD CONSTRAINT tool_performance_reason_code_format
    CHECK (reason_code IS NULL OR reason_code ~ '^[a-z][a-z0-9_]{0,63}$');

COMMENT ON COLUMN composition_steps.reason_code IS
    'Closed-set code for why the step did not produce a result (#2021). The aggregation key; the message is never stored.';
COMMENT ON COLUMN composition_steps.error_message IS
    'The reason code''s fixed catalogue sentence, at most 200 chars. NEVER the tool''s own message (#2050).';
COMMENT ON COLUMN tool_performance.reason_code IS
    'Mirror of composition_steps.reason_code, so get_tool_reliability can report the most common refusal reason.';

CREATE INDEX IF NOT EXISTS idx_composition_steps_reason_code
    ON composition_steps(reason_code) WHERE reason_code IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_tool_performance_reason_code
    ON tool_performance(reason_code) WHERE reason_code IS NOT NULL;
```

Then re-create `composer_record_steps` by copying the whole function from ml/041 (lines 540 to the end of that `$fn$;`) into this file **unchanged except for three edits**, so the diff between the two files is readable:

1. In the `composition_steps` INSERT column list, append `, reason_code` after `error_type`.
2. Replace the bare `NULL` in the SELECT (the slot between the `status` CASE and the `retry_count` `GREATEST(...)`) with:
   ```sql
               left(s->>'error_message', 200),
   ```
3. Append the matching value at the end of the SELECT list, after the `error_type` CASE:
   ```sql
               ,CASE WHEN (s->>'reason_code') ~ '^[a-z][a-z0-9_]{0,63}$' THEN s->>'reason_code' END
   ```
4. In the `tool_performance` INSERT, add `reason_code` to the column list and `cs.reason_code` to the SELECT.

Finally re-create the reliability function. Its return type changes, so `CREATE OR REPLACE` is not enough — the view must be dropped first, then the function:

```sql
DROP VIEW IF EXISTS v_tool_reliability;
DROP FUNCTION IF EXISTS get_tool_reliability(integer, boolean);
```

Then copy `get_tool_reliability` from ml/041 (lines ~770-836) with exactly two edits: add `most_common_refusal_reason text` as the last column of the `RETURNS TABLE`, add `tp.reason_code` to the `perf` CTE's select list, and add this as the last expression of the outer SELECT, immediately after the existing `most_common_health_error` expression:

```sql
        ,(mode() WITHIN GROUP (ORDER BY p.reason_code)
            FILTER (WHERE p.counted AND p.outcome_class IN ('refused', 'input_rejected')))::text
```

and re-create the view and its comment:

```sql
CREATE VIEW v_tool_reliability AS
SELECT * FROM get_tool_reliability(30, true);

COMMENT ON VIEW v_tool_reliability IS
    'Per-tool measured reliability over 30 days, synthetic rows included (get_tool_reliability(30, true)).';
```

- [ ] **Step 4: Write `database/ml/rollback_042.sql`**

Follow `database/ml/rollback_041.sql`'s shape. It must drop the two columns and their constraints and indexes, and restore ml/041's `composer_record_steps` and `get_tool_reliability` verbatim (including the hardcoded `NULL`, since that is what 041 shipped). State in a header comment that rolling back re-introduces #2050.

- [ ] **Step 5: Rehearse against the live database inside a transaction**

The rehearsal is the evidence. A unit test that skips proves nothing.

```bash
cd /home/enunez/Projects/e2i_causal_analytics/.worktrees/lane-refusal-codes
docker exec -i supabase_db_e2i psql -U postgres -d postgres <<'SQL'
BEGIN;
\i /dev/stdin
SQL
```

That form does not work for a file — use the project's established path instead: copy the migration into the container and run it inside an explicit transaction, and **pass `-i` to `docker exec`** (without `-i` a heredoc is silently discarded, psql runs nothing, prints nothing and exits 0 — indistinguishable from success):

```bash
docker cp database/ml/042_composer_refusal_reason_codes.sql supabase_db_e2i:/tmp/042.sql
docker exec -i supabase_db_e2i psql -U postgres -d postgres -v ON_ERROR_STOP=1 <<'SQL'
BEGIN;
\i /tmp/042.sql
-- End-state guard INSIDE the transaction: a no-op or partial apply must fail loudly.
DO $$
DECLARE n integer;
BEGIN
    SELECT count(*) INTO n FROM information_schema.columns
     WHERE table_name IN ('composition_steps', 'tool_performance') AND column_name = 'reason_code';
    IF n <> 2 THEN RAISE EXCEPTION 'reason_code present on % of 2 tables', n; END IF;
    IF (SELECT prosrc FROM pg_proc WHERE proname = 'composer_record_steps') LIKE '%retry_count,%NULL,%' THEN
        RAISE EXCEPTION 'composer_record_steps still writes a hardcoded NULL into error_message';
    END IF;
    PERFORM 1 FROM get_tool_reliability(30, true) LIMIT 1;
    RAISE NOTICE 'REHEARSAL OK';
END $$;
ROLLBACK;
SQL
```

Expected: `NOTICE:  REHEARSAL OK` followed by `ROLLBACK`. Save the full transcript to `docs/demos/results/2026-09-12_lane_refusal_codes_cert/migration_rehearsal.txt`. Re-read the AFTER state in a separate command — never claim an effect in the turn that requests it.

- [ ] **Step 6: Run the migration test**

```bash
/home/enunez/Projects/e2i_causal_analytics/.venv/bin/pytest tests/unit/test_database/learning_loop/test_042_reason_codes.py \
  tests/unit/test_database/learning_loop/test_041_recording.py -n 0 -p no:cacheprovider -q --timeout=900
```

Expected: all pass, none skipped.

- [ ] **Step 7: Commit**

```bash
git add database/ml/042_composer_refusal_reason_codes.sql database/ml/rollback_042.sql \
        tests/unit/test_database/learning_loop/test_042_reason_codes.py
git commit -m "fix(db): ml/042 persists the refusal reason the RPC was dropping (#2050 cause 2)

ml/041 listed error_message in composer_record_steps' INSERT but supplied a hardcoded NULL
in that slot (041:631), so even a recorder that sent the value had it dropped. 042 supplies
left(s->>'error_message', 200), adds reason_code to composition_steps and tool_performance,
and re-creates get_tool_reliability with most_common_refusal_reason. What is stored is the
closed code and its fixed catalogue sentence — ml/041's no-error-text contract is preserved.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_017AdJUYq6yCMahuwKuFfWJw"
```

---

## Task 7: Surface the codes on the admin observability page (#2021 D3)

**Files:**
- Modify: `src/agents/tool_composer/reliability.py:131,171`
- Modify: `src/api/schemas/admin_tool_composer.py`
- Modify: `src/services/tool_composer_observability_service.py:35,196,223-231`
- Test: `tests/unit/test_api/test_admin_tool_composer_reason_codes_2021.py`

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_api/test_admin_tool_composer_reason_codes_2021.py`. Read the existing observability tests first (`grep -rln "tool_composer_overview\|ToolComposerObservability" tests/`) and follow their fixtures.

```python
"""#2021 D3: the codes are written AND read — the page shows why a tool refuses."""

from src.agents.tool_composer.reliability import ToolReliability
from src.api.schemas.admin_tool_composer import RecentFailure, ToolReliabilityRow


def test_reliability_row_carries_the_most_common_refusal_reason():
    row = ToolReliability.from_row({
        "tool_name": "causal_effect_estimator",
        "n_invoked": 12, "n_succeeded": 4, "n_refused": 8,
        "n_health_failures": 0, "n_health": 4, "n_retried": 0, "n_synthetic": 0,
        "most_common_refusal_reason": "non_binary_treatment",
    })
    assert row.most_common_refusal_reason == "non_binary_treatment"


def test_a_row_without_the_field_is_none_not_an_error():
    """The API must survive a database that has not had ml/042 applied yet."""
    row = ToolReliability.from_row({
        "tool_name": "gap_calculator",
        "n_invoked": 1, "n_succeeded": 1, "n_refused": 0,
        "n_health_failures": 0, "n_health": 1, "n_retried": 0, "n_synthetic": 0,
    })
    assert row.most_common_refusal_reason is None


def test_wire_schema_accepts_the_new_fields():
    ToolReliabilityRow(
        tool_name="t", verdict="caveat", most_common_refusal_reason="coverage_gap"
    )
    RecentFailure(
        composition_id="c1",
        step_classes=[{"step_number": 1, "tool_name": "t",
                       "outcome_class": "refused", "reason_code": "coverage_gap"}],
    )


def test_service_maps_the_reason_onto_the_tool_row(observability_service, one_refusing_tool):
    rows = observability_service.overview(30, one_refusing_tool)["tools"]
    assert rows[0]["most_common_refusal_reason"] == "non_binary_treatment"


def test_service_maps_reason_code_onto_each_failed_step(observability_service_with_steps):
    failures = observability_service_with_steps.overview(30, {})["recent_failures"]
    assert failures[0]["step_classes"][0]["reason_code"] == "coverage_gap"
```

- [ ] **Step 2: Run it to verify it fails**

```bash
/home/enunez/Projects/e2i_causal_analytics/.venv/bin/pytest tests/unit/test_api/test_admin_tool_composer_reason_codes_2021.py -n 0 -p no:cacheprovider -q --timeout=300
```

Expected: FAIL — `TypeError: ToolReliability() got an unexpected keyword argument` / `Extra inputs are not permitted` (the schemas use `extra="forbid"`).

- [ ] **Step 3: Add the field to `ToolReliability`**

In `src/agents/tool_composer/reliability.py`, after `most_common_health_error: Optional[str] = None` (line 131):

```python
    # #2021: the most common CLOSED code among this tool's refusals in the window.
    # Its sibling above reports health failures by exception class; this reports
    # declines-to-answer by category, which is what the page could not show before.
    most_common_refusal_reason: Optional[str] = None
```

and in `from_row`, beside the existing `most_common_health_error=row.get(...)` (line 171):

```python
            most_common_refusal_reason=row.get("most_common_refusal_reason"),
```

`row.get` is why a pre-ml/042 database still works.

- [ ] **Step 4: Add the wire fields**

In `src/api/schemas/admin_tool_composer.py`, on `ToolReliabilityRow` after `most_common_health_error`:

```python
    most_common_refusal_reason: Optional[str] = Field(
        default=None,
        description="Most common closed reason code among this tool's refusals (#2021)",
    )
```

`RecentFailure.step_classes` is `List[Dict[str, Any]]`, so it needs no type change — only its description:

```python
    step_classes: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="The steps that did not succeed, with their classes and reason codes",
    )
```

- [ ] **Step 5: Map the fields in the service**

In `src/services/tool_composer_observability_service.py`:

Line 35 — add the column so the step rows carry it:

```python
_STEP_COLUMNS = "episode_id, step_number, tool_name, outcome_class, reason_code"
```

Line ~196 — beside `"most_common_health_error": tool.most_common_health_error,`:

```python
                    "most_common_refusal_reason": tool.most_common_refusal_reason,
```

Lines 223-231 — add the code to each step class:

```python
                    "step_classes": [
                        {
                            "step_number": s.get("step_number"),
                            "tool_name": s.get("tool_name"),
                            "outcome_class": s.get("outcome_class"),
                            "reason_code": s.get("reason_code"),
                        }
                        for s in steps
                        if s.get("outcome_class") not in ("succeeded", "cache_hit")
                    ],
```

- [ ] **Step 6: Run the test to verify it passes**

```bash
/home/enunez/Projects/e2i_causal_analytics/.venv/bin/pytest tests/unit/test_api/test_admin_tool_composer_reason_codes_2021.py -n 0 -p no:cacheprovider -q --timeout=300
/home/enunez/Projects/e2i_causal_analytics/.venv/bin/pytest tests/unit/test_api/ -k "observability or tool_composer" -n 0 -p no:cacheprovider -q --timeout=600 2>&1 | tail -10
```

- [ ] **Step 7: Regenerate the frontend types**

The response model changed, so `frontend/src/generated/api.ts` and the OpenAPI snapshot move. Find the generator and the verify-types gate first:

```bash
grep -rn "generate.*openapi\|openapi.*generate" package.json Makefile scripts/*.sh 2>/dev/null | head
ls frontend/src/generated/
```

Run whatever that names, then:

```bash
git diff --stat frontend/src/generated/
```

Expected: only `most_common_refusal_reason` (and the `step_classes` description) appear. A schema-name renumbering in the diff is the known nondeterminism — re-run once before investigating.

- [ ] **Step 8: Commit**

```bash
/home/enunez/Projects/e2i_causal_analytics/.venv/bin/ruff check --no-cache src/ tests/unit/test_api/
git add src/agents/tool_composer/reliability.py src/api/schemas/admin_tool_composer.py \
        src/services/tool_composer_observability_service.py \
        tests/unit/test_api/test_admin_tool_composer_reason_codes_2021.py \
        frontend/src/generated/
git commit -m "feat(admin): observability shows the most common refusal reason per tool (#2021)

The codes are now read, not only written: each tool row carries most_common_refusal_reason
from get_tool_reliability, and each recent-failure step class carries its reason_code.
from_row uses .get, so a database without ml/042 still serves the page with nulls.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_017AdJUYq6yCMahuwKuFfWJw"
```

---

## Task 8: Whole-lane verification, push, and the live certificate

**Files:**
- Create: `docs/demos/results/2026-09-12_lane_refusal_codes_cert/cert.md`

- [ ] **Step 1: Run the affected suites**

Every run `-n 0` — the repo's `addopts` is `-n 4`, and four ~800 MiB workers fill swap on this box.

```bash
cd /home/enunez/Projects/e2i_causal_analytics/.worktrees/lane-refusal-codes
/home/enunez/Projects/e2i_causal_analytics/.venv/bin/pytest tests/unit/test_agents/test_tool_composer/ -n 0 -p no:cacheprovider -q --timeout=1800 2>&1 | tail -20
/home/enunez/Projects/e2i_causal_analytics/.venv/bin/pytest tests/unit/test_database/learning_loop/ -n 0 -p no:cacheprovider -q --timeout=900 2>&1 | tail -20
/home/enunez/Projects/e2i_causal_analytics/.venv/bin/pytest tests/unit/test_api/ -n 0 -p no:cacheprovider -q --timeout=900 2>&1 | tail -20
```

Do **not** pipe a pytest run through a pager — `pytest | tail` returns the pager's exit code and a suite that died mid-run reads as green. Read the printed summary line.

- [ ] **Step 2: Lint**

```bash
/home/enunez/Projects/e2i_causal_analytics/.venv/bin/ruff check --no-cache src/ tests/
/home/enunez/Projects/e2i_causal_analytics/.venv/bin/ruff format --check src/agents/tool_composer/ src/services/ src/api/
```

`--no-cache` is not optional: ruff's cache can print "All checks passed!" on a file that fails on a fresh checkout, which is what CI does.

**Do not run mypy here.** CI's `Type Check (MyPy)` gate is the arbiter; the config follows imports, so even one file type-checks the whole closure (~1.5 GiB, ~6 min) and drives this box into swap.

- [ ] **Step 3: Codex review round**

```bash
codex exec -C /home/enunez/Projects/e2i_causal_analytics/.worktrees/lane-refusal-codes \
  --sandbox read-only "$(cat /tmp/claude-1000/.../brief.md)" < /dev/null
```

The brief must include, verbatim:

> If a recommendation solves a labeling problem instead of a functional problem, flag it as HIGH finding. If a recommendation preserves code without investigating intent (PR history, linked issues, user-requested functionality), flag it as HIGH finding. If a recommendation deletes code without verifying intent, flag it as HIGH finding. Audit the question being asked, not just the answer given.

Iterate to ACCEPT. Specifically ask it to audit: (a) whether storing the canonical sentence rather than the raw message actually answers #2050's operator need, or whether it is a labeling fix; (b) whether the refused/input_rejected passthrough in Task 4 can carry text no tool authored.

- [ ] **Step 4: One push, then the PR**

```bash
git branch --show-current   # must be claude/2021-2050-2020-refusal-reason-codes
git log --oneline origin/main..HEAD
git push -u origin claude/2021-2050-2020-refusal-reason-codes
gh pr create --title "fix(tool-composer): closed refusal reason codes, persisted and surfaced (#2021, #2050, #2020)" --body-file <(cat <<'BODY'
...
BODY
)
```

Verify the body landed — `gh pr edit --body-file` can silently fail. Re-read with `gh pr view <n> --json body`.

- [ ] **Step 5: Watch CI, then get owner go before merging**

`gh pr checks` 403s on this PAT. Use the actions runs API with the **full 40-char sha**:

```bash
SHA=$(git rev-parse HEAD)
gh api "repos/enunezvn/e2i_causal_analytics/actions/runs?head_sha=$SHA" \
  --jq '.workflow_runs[] | "\(.name) \(.status) \(.conclusion)"'
```

Merge with `--merge` (never `--squash`) **only after the owner says go**.

- [ ] **Step 6: Apply the migration and deploy**

Apply ml/042 on the droplet per `reference-supabase-droplet-migration-apply.md`. Before reading anything live, confirm the deploy gate:

```bash
gh api "repos/enunezvn/e2i_causal_analytics/actions/workflows/deploy.yml/runs?per_page=5" \
  --jq '[.workflow_runs[]|select(.status!="completed")]|length'   # must be 0
docker inspect e2i_api --format '{{.Config.Image}} {{.State.StartedAt}} {{.State.Health.Status}}'
```

The container tag must equal main HEAD. A deploy job can report FAILED while prod flipped correctly, and vice versa — trust the container's content, not the job conclusion.

- [ ] **Step 7: Live certificate**

Write `docs/demos/results/2026-09-12_lane_refusal_codes_cert/cert.md` with the verdict word first, then the numbers. It must contain:

1. **The container tag and StartedAt** before and after, plus the image sha.
2. **#2050 proof — the persisted row.** Force a refusal through a composed question (an all-null or 4-valued treatment column reproduces `non_binary_treatment`), then:
   ```sql
   SELECT step_number, tool_name, outcome_class, error_type, reason_code, error_message
   FROM composition_steps WHERE episode_id = '<the new episode>' ORDER BY step_number;
   ```
   PASS requires `reason_code` non-null **and** `error_message` equal to the catalogue sentence. The pre-fix state — the two rows from the 2026-09-12 tool-composer wave cert with `error_message IS NULL` — is the before-picture; cite it.
3. **#2020 proof.** Force an all-null treatment through a composed question so the DoWhy `RuntimeError` path fires, and show that the returned answer contains no `DoWhy` / `shape=(0,)` / `backdoor.` substring while the container log for the same composition does. Both must be shown; the log half is what proves the text was not merely lost.
4. **#2021 proof.** `GET /admin/observability/tool-composer` returning a non-null `most_common_refusal_reason` for the tool that refused.
5. **A negative control.** A composition whose refusal IS tool-authored, showing the reason still reaches the answer verbatim. Without it, a PASS on item 3 could just mean all reasons were suppressed.

- [ ] **Step 8: Close out**

```bash
gh issue comment 2021 --body "..."   # the vocabulary, the 87 sites, the observability surface
gh issue comment 2050 --body "..."   # both causes, and the correction to cause 1 as filed
gh issue comment 2020 --body "..."   # the split, with the cert's before/after
gh issue close 2021 2050 2020
git worktree remove .worktrees/lane-refusal-codes
git branch -d claude/2021-2050-2020-refusal-reason-codes
```

Then write the memory file and its `MEMORY.md` line.

---

## Self-review

**Spec coverage.** #2021 → Tasks 1, 2, 3, 7. #2050 cause 1 → Task 5; cause 2 → Task 6. #2020 → Task 4. The three owner decisions: D1 → Tasks 5 and 6; D2 → Task 2; D3 → Task 7.

**Type consistency.** `ReasonCode` members are referenced as `ReasonCode.MEMBER` in Python and as their `.value` lowercase strings on `StepResult`, in the recorder payload, and in the database — `canonical_sentence()` takes either, which is why Task 4 and Task 5 can pass a plain string read off a Pydantic model. `StepResult.reason_code` is `Optional[str]` (not the enum) because the recorder serializes it and the database stores text; Task 3's tests assert the string form deliberately.

**Known gaps, deliberate.** The database has a format CHECK, not a member list — the closed set lives in Python, and a duplicated list in SQL would drift silently. `ReferenceResolutionError` is not made a coded error; the executor assigns it `REFERENCE_UNRESOLVABLE` instead, so its message-composing constructor is untouched.
