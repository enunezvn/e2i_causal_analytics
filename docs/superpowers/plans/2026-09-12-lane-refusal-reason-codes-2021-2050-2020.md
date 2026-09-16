# Refusal Reason Codes Lane (#2021 + #2050 + #2020) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give every composable-tool refusal and failure a stable, closed-set `reason_code` plus a data-free canonical sentence, propagate it through the executor to the learning-loop recorder and the database, use it to keep library internals out of the user-facing fail-closed answer, and surface it on the admin observability page.

**Architecture:** One vocabulary module (`ReasonCode` + a canonical-sentence catalogue) is the single source of truth. Tool-authored refusals (`ToolRefusalError` / `ToolInputError`) carry a code at the raise site; the executor assigns a code to every non-refusal failure arm from the outcome class it already computes. The code and its numeric details are what get persisted; the canonical sentence is rendered at READ time from the one Python catalogue (owner decision D1′), so no text of any kind is written to the recording tables. The raw message keeps its two existing homes — the container log, and the fail-closed answer **only when the tool authored it**.

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
   **Owner decision D1 (2026-09-12): persist the `reason_code` and a fixed catalogue sentence derived from it, plus bounded structured `details`. The raw message is never persisted.**
   **Revised by D1′ (2026-09-12), after the ml/041 tests were read:** the RPC's hardcoded `NULL` is the database's deliberate SECOND guard — `test_041_recording.py` (lines 668–740) sends `error_message=SENTINEL` and asserts it reaches no stored row, line 441 asserts `error_message IS NULL` for all ten classes, and `test_learning_recorder_realdb.py:363` repeats it. Storing even a derived sentence from the payload would make the database trust caller text. So: **persist `reason_code` + numeric `reason_details`; `error_message` stays NULL; the sentence is rendered at read time.** #2050's operator need ("an operator sees `refused` with no why") is met on the admin page (Task 7), and the code itself is in the row.
4. **`get_tool_reliability` reads `tool_performance`, not `composition_steps`** (ml/041:806). So a per-tool `most_common_refusal_reason` needs `reason_code` on **both** tables.
5. **Scope is 87 raise sites, all in one file.** Measured by `ast.walk` for `raise ToolRefusalError(...)` / `raise ToolInputError(...)` on the lane base `0a16e9c18`: `ToolRefusalError` 66 + `ToolInputError` 21 = 87. `src/tool_registry/` → 0.
   **This plan first said 94, which was wrong for two independent reasons** (caught by the Task 1/2 implementer, corrected 2026-09-12): (a) the figure came from `grep -c`, which counts LINES mentioning either name and dedupes multiple matches per line — 16 of the 103 matching lines are docstrings, comments and the import; (b) it was measured on `6c6a6a0ae`, three commits behind the lane base, before PR #2059 (`8bb85a772`, `772733dc2`, the #2022 sensitivity work) added 5 more `ToolRefusalError` sites. Method alone gives 82 on that stale base; base drift takes it to 87. **The AST test is the only count that governs — do not re-derive this number with grep.**
6. **96 existing assertions pin refusal prose** across 17 test files. Task 2 must not reword a single message.

## Owner decisions (2026-09-12)

| # | Decision |
|---|----------|
| D1 | ~~`composition_steps.error_message` receives the canonical catalogue sentence~~ — **superseded by D1′**. |
| D1′ | **Render at read time.** Persist `reason_code` + numeric `reason_details` only. `error_message` stays NULL, so ml/041's no-error-text guard and the three tests that pin it are untouched. The admin page renders the sentence from `CANONICAL_SENTENCES`; adding a code never needs a migration. |
| D4 | **ml/043 is proven by a scratch rehearsal**, not an in-harness test: the shared `_pg` fixture skips whenever prod already holds any of `LANE_MIGRATIONS` (prod has 039–041, so every learning-loop DB test skips on this droplet), and an in-harness 043 test would run at most once, pre-deploy, anyway. Rehearse on a throwaway container cloned from prod's image and schema; commit only the static CI checks for 043; leave the shared harness untouched. |
| L1 | *(lead call, 2026-09-12)* `details` values are finite numbers or booleans only, keys are snake_case. Task 1 as first planned allowed strings up to 64 characters, which fits a column or brand name — exactly the data D1/D1′ keep out. All 15 detail values at the 7 current sites are integers, so tightening breaks nothing. Applied as a Task 1 fix after its spec review. |
| D2 | **All 87 raise sites** get a code in this lane, enforced by an AST test that fails if any raise site lacks one. |
| D3 | The admin observability route **does** surface the codes (per-tool `most_common_refusal_reason`, per-step `reason_code`). |
| D5 | *(owner, 2026-09-12)* **Fix library text at its source.** A coded refusal that wraps a caught exception drops the exception's text from its message, keeps its authored sentence, logs the original exception at the wrap site, and keeps `raise ... from exc`. An AST guard forbids a coded refusal (or an `EffectDataUnavailable`) from interpolating an `except ... as` name, except at an explicit allowlist of sites whose wrapped exception text is itself authored. Task 4b. |
| D6 | *(owner, 2026-09-12)* **Extend the lane to the two other surfaces that carry raw failure text:** the synthesis prompt (`synthesizer._format_results` writes raw `output.error` for a failed step on partial success) and the Digital Twin API's error fields. Larger blast radius — it changes what the synthesis LLM sees and a second API's error contract — accepted by the owner. Task 7b, with its own certificate items. |
| R1 | *(lead, after verifying the Task 1+2 spec review against source)* Recodes: `:3987` (twin simulation did not complete — it runs INSIDE this step, not upstream) → new `SIMULATION_INCOMPLETE`; `:2417` segment_ranker and `:3115` roi_estimator (executor F5 already short-circuits a failed upstream as `dependency_unmet`, so these fire on a wrong-shape or literal input) → `MISSING_REQUIRED_INPUT`; `:3741` `_load_cohort_provider` and `:3865` the `EffectDataUnavailable` wrap → new `EFFECT_NOT_ESTIMABLE`, because each covers several causes (missing columns, too few rows, no contrast, unestimable intervention, estimator failure) and per-cause coding would need `src/digital_twin` to carry a code — a cross-package contract, and importing `reason_codes` from there costs the ~564 MB package `__init__`; `:3224` `_power_number` splits into two raises with the IDENTICAL message: bool/non-number → `INVALID_INPUT_TYPE`, non-finite number → `NON_FINITE_INPUT`. `:2350` cate_analyzer stays `INSUFFICIENT_GROUPS` (zero measurable segments is fewer than the one required). |
| P1 | *(lead)* `_CodedError.__reduce__` is **kept**: a required keyword-only `reason_code` makes `BaseException.__reduce__` raise `TypeError` on pickle or `copy.deepcopy`, turning a refusal into a crash wherever an exception is copied. Trivial and tested. Its docstring's claim that these errors "crossed process boundaries before #2021" is unverified and is replaced by the invariant itself. |

## File structure

| File | Responsibility | Task |
|---|---|---|
| `src/agents/tool_composer/reason_codes.py` *(new)* | `ReasonCode` closed set + canonical sentence catalogue + `canonical_sentence()` | 1 |
| `src/agents/tool_composer/errors.py` | `ToolRefusalError` / `ToolInputError` carry `reason_code` + `details` | 1 |
| `src/agents/tool_composer/tool_registrations.py` | 87 raise sites get codes | 2 |
| `src/agents/tool_composer/models/composition_models.py` | `StepResult.reason_code`, `StepResult.reason_details` | 3 |
| `src/agents/tool_composer/executor.py` | assign a code on every failure arm | 3 |
| `src/agents/tool_composer/composer.py` | fail-closed answer: verbatim for refusals, canonical for everything else | 4 |
| `src/agents/tool_composer/learning_recorder.py` | emit `reason_code` + numeric `reason_details` — no text | 5 |
| `database/ml/043_composer_refusal_reason_codes.sql` *(new)* | reason columns; RPC carries code + details, `error_message` stays NULL; reliability function | 6 |
| `database/ml/rollback_043.sql` *(new)* | reverse of 043 | 6 |
| `src/agents/tool_composer/reliability.py` | `most_common_refusal_reason` on `ToolReliability` | 7 |
| `src/api/schemas/admin_tool_composer.py` | wire fields | 7 |
| `src/services/tool_composer_observability_service.py` | map the new fields | 7 |
| `src/digital_twin/effect/cohort_causal_estimator.py`, `src/digital_twin/effect/estimator.py` | exception text removed from wrapped estimator errors (D5) | 4b |
| `src/agents/tool_composer/synthesizer.py` | failed-step text in the synthesis prompt follows the #2020 rule (D6) | 7b |
| `src/api/routes/digital_twin.py` | error details carry no library text, where measured reachable (D6) | 7b |

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

> **Historical — Task 1 is done; do not copy this block.** The module on the branch supersedes it: `StrEnum`; 32 members; `UPSTREAM_STEP_FAILED` removed; `DEPENDENCY_UNMET` reads "did not produce a result"; `SIMULATION_INCOMPLETE`, `EFFECT_NOT_ESTIMABLE`, `AMBIGUOUS_COLUMN` / `AMBIGUOUS_GROUP_LABEL` and `NON_NUMERIC_COLUMN` added; `EXECUTOR_ASSIGNED`; and a `validate_details` that is numeric-only and fail-soft (Tasks 2b and 3 reviews). Read `src/agents/tool_composer/reason_codes.py` itself.

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
| ~~the upstream result reports `status='failed'` / did not complete~~ | ~~`UPSTREAM_STEP_FAILED`~~ — **removed 2026-09-12, before first deploy.** R1 recoded all three sites that used it (one failure ran inside its own step; the other two fired on wrong-shape inputs), and the executor's F5 short-circuit means no tool can receive a failed upstream result. The executor-assigned `DEPENDENCY_UNMET` covers the real case, and its sentence was corrected to "a step this one depends on did not produce a result" (the Task 1 vocabulary block above predates both changes). |
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

## Task 2b: Apply the Task 1+2 review — L1, R1, P1 and the attribute-call hole

> **Added 2026-09-12 after the Task 1+2 spec review (`spec-review-task12`), every finding re-verified against source by the lead.** Line numbers are at `956a57e3b`; re-derive them by AST before editing.

**Files:**
- Modify: `src/agents/tool_composer/reason_codes.py`, `src/agents/tool_composer/errors.py`, `src/agents/tool_composer/tool_registrations.py`
- Test: `tests/unit/test_agents/test_tool_composer/test_reason_codes_2021.py`, `test_reason_code_coverage_2021.py` (extend in place)

**Still the hard rule: no message text changes.** The one exception is structural — `_power_number`'s single raise becomes two raises carrying the byte-identical message.

- [ ] **Step 1: L1 — `details` is numbers and booleans under snake_case keys.** Red first. Rejected: any `str` value (drop `_MAX_DETAIL_STR`), `None`, a list or dict (the reviewer's `{"cols": ["free text" * 5]}` passes today), a `float` that is NaN or ±inf, and a key that does not match `^[a-z][a-z0-9_]*$`. Accepted: `bool`, `int`, a finite `float`. Keep `_MAX_DETAIL_KEYS = 8`. The existing `test_details_must_be_structure_only` must still pass unchanged. All 15 detail values at the 7 current sites are ints, so no site changes.

- [ ] **Step 2: R1 — recodes.** Add two members with sentences obeying `test_canonical_sentences_carry_no_interpolation`:
  - `SIMULATION_INCOMPLETE = "simulation_incomplete"` — `"the twin simulation did not complete"`
  - `EFFECT_NOT_ESTIMABLE = "effect_not_estimable"` — `"the cohort data cannot support a causal effect estimate for this intervention"`

  Then: `:3987` → `SIMULATION_INCOMPLETE`; `:2417` segment_ranker and `:3115` roi_estimator → `MISSING_REQUIRED_INPUT`; `:3741` and `:3865` → `EFFECT_NOT_ESTIMABLE`; `:3224` `_power_number` → two `raise` statements with the identical message, `isinstance(value, bool) or not isinstance(value, (int, float))` → `INVALID_INPUT_TYPE`, otherwise (a non-finite number) → `NON_FINITE_INPUT`. `:2350` is unchanged (R1 says why).

  Pin each recode with a test that fails on the old code. Where the function takes plain inputs, call it: `segment_ranker({})`, `roi_estimator({}, 1.0)`, `_power_number("x", "a")`, `_power_number("x", True)`, `_power_number("x", float("nan"))`. For `:3741`, `:3865` and `:3987`, find how their existing tests reach those branches (`grep -rn "_load_cohort_provider\|EffectDataUnavailable\|did not complete" tests/unit/test_agents/test_tool_composer/`) and reuse that seam. If a branch cannot be reached without a heavy fit, say so in the report and pin that site's code by AST inside its named function instead — never an invented seam.

- [ ] **Step 3: P1 — keep `__reduce__`, fix its docstring.** Replace the sentence claiming these errors "crossed process boundaries before #2021" with the invariant: a required keyword-only `reason_code` makes `BaseException.__reduce__` raise `TypeError` on `pickle` or `copy.deepcopy`. No code change.

- [ ] **Step 4: Close the attribute-call hole in the threaded-code check.** `test_a_threaded_reason_code_is_literal_at_every_call_site` only inspects `ast.Name` calls, so `tr._refuse_unless_binary_01(..., reason_code="oops")` passes every test today. Factor the check into a helper that takes a parsed tree, make it cover `ast.Attribute` calls too, and add a test that feeds the helper a synthetic source string with a bad attribute call and asserts it is flagged. That synthetic test is the proof of teeth; a green run on the real file is not.

- [ ] **Step 5: Verify.**
  - Every message string at every raise site is identical to `0a16e9c18`'s, compared by AST (the only allowed difference is `_power_number`'s message now appearing at two sites).
  - `test_reason_codes_2021.py` + `test_reason_code_coverage_2021.py`, then the whole `tests/unit/test_agents/test_tool_composer/` directory, `-n 0`, exit code read from `$?` (the directory took 150 s and gave 1227 passed at `dd6146b16`).
  - `ruff check --no-cache` and `ruff format --check --no-cache` on every changed file.

- [ ] **Step 6: Commit** — `fix(tool-composer): apply the #2021 code review — numeric details, recodes, attribute-call guard`, the body naming each recode and why.

---

## Task 3: Propagate the code through the executor

> **Corrected 2026-09-12, before dispatch.** The first draft of this task called an `executor_with_tool` fixture that does not exist, carried stale line anchors, and gave the plan-defect arm an `isinstance` branch it does not need — there are two separate plan-defect arms, one per exception. Everything below is verified against `executor.py` and `test_executor_outcome_classes.py` on the lane base `0a16e9c18`; Tasks 1+2 do not touch `executor.py`. Re-run `grep -n 'outcome_class=' src/agents/tool_composer/executor.py` before editing and trust it over the numbers here.

**Files:**
- Modify: `src/agents/tool_composer/models/composition_models.py` — `StepResult`, after `error_type`
- Modify: `src/agents/tool_composer/executor.py` — the failure arms in the table in Step 4
- Test: `tests/unit/test_agents/test_tool_composer/test_executor_outcome_classes.py` — extend in place

The executor already computes `outcome_class` on every exit of `_execute_step`, from the exception type it caught and never from error text. The reason code is the finer-grained sibling: taken off the exception where a coded `ToolRefusalError` / `ToolInputError` was caught, assigned by the executor on every other arm. Both are set in the same arm, so they cannot disagree, and the tests pin that.

**Why this test file.** It already has one test per exit of `_execute_step` — succeeded, cache_hit, refused, input_rejected (sync and async), timeout (sync and async), error, plan_defect, dependency_unmet, circuit_open, not_registered — built from `registry`, `_register`, `_executor`, `_one` and `_classes`. `asyncio_mode = "auto"` is set in `pyproject.toml`, so async tests need no marker, and `ReasonCode` is already imported. Forcing each arm a second time in a new file would duplicate that machinery; asserting the code beside the class that is already forced does not.

- [ ] **Step 1: Write the failing tests**

(a) Add a helper next to `_classes`:

```python
def _code(result: StepResult) -> Tuple[Optional[str], Dict[str, Any]]:
    return (result.reason_code, result.reason_details)
```

(b) In **every existing test in this file that asserts a `_classes(...)` tuple**, add one assertion on the same result, from this table. Do not change the existing `_classes` assertions.

| `outcome_class` asserted | add |
|---|---|
| `succeeded`, `cache_hit` | `assert _code(result) == (None, {})` |
| `refused`, `input_rejected` | `assert result.reason_code is not None and result.reason_code != ReasonCode.TOOL_ERROR.value` |
| `timeout` | `assert result.reason_code == ReasonCode.TOOL_TIMEOUT.value` |
| `error` | `assert _code(result) == (ReasonCode.TOOL_ERROR.value, {})` |
| `plan_defect` from an unresolvable reference | `assert result.reason_code == ReasonCode.REFERENCE_UNRESOLVABLE.value` |
| `plan_defect` from missing or misnamed arguments (`PlanArgumentError`) | `assert result.reason_code == ReasonCode.PLAN_DEFECT.value` |
| `dependency_unmet` | `assert result.reason_code == ReasonCode.DEPENDENCY_UNMET.value` |
| `circuit_open` | `assert result.reason_code == ReasonCode.CIRCUIT_OPEN.value` |
| `not_registered` | `assert result.reason_code == ReasonCode.TOOL_NOT_REGISTERED.value` |

The refused and input_rejected rows deliberately name no member. Those tests drive the REAL guards in `tool_registrations`, and which code each guard carries is Task 2's contract, pinned there; here the contract is only that the tool's code survives the executor. If this file has no existing test for one of the classes in the table, say so in your report — do not invent a way to force it.

(c) Append these tests. They pin what the table cannot: that the code and details come off the exception unchanged, and that on the retry arm the code follows the last exception exactly as the class does.

```python
# ---------------------------------------------------------------------------
# #2021: the reason code survives the executor
# ---------------------------------------------------------------------------


async def test_a_refusal_code_and_its_details_are_carried_not_rederived(registry):
    def refusing(**_: Any) -> Any:
        raise ToolRefusalError(
            "probe: treatment column carries 4 distinct non-null values.",
            reason_code=ReasonCode.NON_BINARY_TREATMENT,
            details={"n_distinct": 4},
        )

    _register(registry, "refusing_probe", refusing)
    trace = await _executor(registry).execute(_one("s", "refusing_probe"))
    result = trace.step_results[0]
    assert _classes(result) == ("refused", 1, False, "ToolRefusalError")
    assert _code(result) == ("non_binary_treatment", {"n_distinct": 4})


async def test_an_input_rejection_code_is_carried(registry):
    def rejecting(**_: Any) -> Any:
        raise ToolInputError(
            "probe: expected_effect must not be None",
            reason_code=ReasonCode.MISSING_REQUIRED_INPUT,
        )

    _register(registry, "rejecting_probe", rejecting)
    trace = await _executor(registry).execute(_one("s", "rejecting_probe"))
    result = trace.step_results[0]
    assert _classes(result) == ("input_rejected", 1, False, "ToolInputError")
    assert _code(result) == ("missing_required_input", {})


@pytest.mark.parametrize(
    "raised, expected_class, expected_code",
    [
        ([TimeoutError("first"), KeyError("last")], "error", "tool_error"),
        ([KeyError("first"), TimeoutError("last")], "timeout", "tool_timeout"),
    ],
)
async def test_the_retry_arm_code_follows_the_last_exception_like_the_class(
    registry, raised, expected_class, expected_code
):
    remaining = list(raised)

    def failing(**_: Any) -> Any:
        raise remaining.pop(0)

    _register(registry, "failing", failing)
    trace = await _executor(registry, max_retries=1).execute(_one("s", "failing"))
    result = trace.step_results[0]
    assert result.outcome_class == expected_class
    assert result.reason_code == expected_code
```

If `ToolRefusalError` / `ToolInputError` are not already imported in this file, add `from src.agents.tool_composer.errors import ToolInputError, ToolRefusalError`.

- [ ] **Step 2: Run the tests to verify they fail**

```bash
cd /home/enunez/Projects/e2i_causal_analytics/.worktrees/lane-refusal-codes && \
  /home/enunez/Projects/e2i_causal_analytics/.venv/bin/pytest \
  tests/unit/test_agents/test_tool_composer/test_executor_outcome_classes.py \
  -n 0 -p no:cacheprovider -q --timeout=600 > /tmp/t3_red.log 2>&1; echo "EXIT=$?"; tail -5 /tmp/t3_red.log
```

Expected: failures raising `AttributeError: 'StepResult' object has no attribute 'reason_code'`. **Read the failure text.** Any other exception is a broken test, not the red this step wants.

- [ ] **Step 3: Add the fields to `StepResult`**

In `src/agents/tool_composer/models/composition_models.py`, directly after `error_type: Optional[str] = None`:

```python
    # #2021: the closed reason code for this step's failure. Tool-authored when a
    # ToolRefusalError/ToolInputError was caught, executor-assigned on every other failure arm,
    # None on success. The aggregation key the learning loop stores; the raw message never is.
    reason_code: Optional[str] = None
    reason_details: Dict[str, Any] = Field(default_factory=dict)
```

Check `Dict`, `Any` and `Field` are imported in that module; add what is missing.

- [ ] **Step 4: Set the code on every failure arm of `executor.py`**

Import beside the existing `from .errors import ...`:

```python
from .reason_codes import ReasonCode
```

Verified anchors (line of the `outcome_class=` keyword on `0a16e9c18`):

| line | arm | add to that `StepResult(...)` |
|---|---|---|
| 576 | F5 dependency short-circuit, `dependency_unmet` | `reason_code=ReasonCode.DEPENDENCY_UNMET.value,` |
| 617 | inside `except ReferenceResolutionError` (604), `plan_defect` | `reason_code=ReasonCode.REFERENCE_UNRESOLVABLE.value,` |
| 689 | `cache_hit` | nothing — a success |
| 711 | `circuit_open` | `reason_code=ReasonCode.CIRCUIT_OPEN.value,` |
| 734 | `not_registered` | `reason_code=ReasonCode.TOOL_NOT_REGISTERED.value,` |
| 768 | inside `except PlanArgumentError` (756), `plan_defect` | `reason_code=ReasonCode.PLAN_DEFECT.value,` |
| 825 | `succeeded` | nothing — a success |
| 876 | inside `except (ToolInputError, ToolRefusalError)` (832) | `reason_code=e.reason_code.value,` and `reason_details=dict(e.details),` |
| 914 | inside `except SyncToolTimeout` (883), `timeout` | `reason_code=ReasonCode.TOOL_TIMEOUT.value,` |
| 946 | retries exhausted, `timeout`/`error` | see below |

Each plan-defect arm is unconditional: the arm IS the exception type, so no `isinstance` check is needed. The other `except ReferenceResolutionError` in this file (about line 1346) is the nested-reference resolver, which degrades to `None` and builds no `StepResult` — leave it alone.

On the retries-exhausted arm, compute the timeout test once and use it for BOTH fields, so the class and the code cannot drift apart:

```python
        last_was_timeout = isinstance(last_exc, (asyncio.TimeoutError, TimeoutError))
        return StepResult(
            ...
            # An async tool's wait_for timeout lands in this generic arm; only the LAST
            # attempt decides the class, and the code follows the class.
            outcome_class="timeout" if last_was_timeout else "error",
            reason_code=(
                ReasonCode.TOOL_TIMEOUT if last_was_timeout else ReasonCode.TOOL_ERROR
            ).value,
            ...
        )
```

Codes are stored as `.value` strings: `StepResult.reason_code` is `Optional[str]`, the recorder serializes it, and the database stores text.

- [ ] **Step 5: Run the tests to verify they pass**

```bash
cd /home/enunez/Projects/e2i_causal_analytics/.worktrees/lane-refusal-codes && \
  /home/enunez/Projects/e2i_causal_analytics/.venv/bin/pytest \
  tests/unit/test_agents/test_tool_composer/test_executor_outcome_classes.py \
  tests/unit/test_agents/test_tool_composer/test_executor.py \
  tests/unit/test_agents/test_tool_composer/test_nonretryable_refusals_1600.py \
  tests/unit/test_agents/test_tool_composer/test_plan_defect_taxonomy.py \
  tests/unit/test_agents/test_tool_composer/test_reason_codes_2021.py \
  -n 0 -p no:cacheprovider -q --timeout=600 > /tmp/t3_green.log 2>&1; echo "EXIT=$?"; tail -3 /tmp/t3_green.log
```

Expected: `EXIT=0` and a summary line with no failures.

- [ ] **Step 6: Commit**

```bash
cd /home/enunez/Projects/e2i_causal_analytics/.worktrees/lane-refusal-codes && \
  /home/enunez/Projects/e2i_causal_analytics/.venv/bin/ruff check --no-cache src/agents/tool_composer/ tests/unit/test_agents/test_tool_composer/ && \
  git add src/agents/tool_composer/models/composition_models.py src/agents/tool_composer/executor.py \
          tests/unit/test_agents/test_tool_composer/test_executor_outcome_classes.py && \
  git commit -m "feat(tool-composer): carry the reason code out of every executor failure arm (#2021)

StepResult gains reason_code + reason_details. Tool-authored codes come off the caught
ToolRefusalError/ToolInputError; every other failure arm is executor-assigned in the same arm
that picks outcome_class, and the retry arm derives both from one timeout test, so the class
and the code can never disagree.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01SnzgDeMLxZN48UsJXTaazb"
```

---

## Task 4: Keep library internals out of the fail-closed answer (#2020)

> **Corrected 2026-09-12, before dispatch.** The first draft built `StepResult(status="FAILED")` — `ExecutionStatus` values are lowercase, so that fails validation — and called `_create_total_failure_result` with `decomposition=None, plan=None`, which `CompositionResult` rejects because both fields are required. Either would make the red-first run fail for the wrong reason. The builders below follow `test_fail_closed_zero_tools_f6.py`, the existing test of this path. **This task depends on Task 3**: only after Task 3 does a real refused step carry a code.

> **Pre-dispatch findings (lead, 2026-09-12, verified against the tests after Task 3 landed).** Read these before Step 5; they decide which existing test changes are faithful and which would be weakening.
> 1. **`test_gap_comparability_1574.py`'s `_failed_trace` builds a `StepResult` with no `outcome_class` and no `reason_code`.** Since Task 3 the executor never produces that for a tool refusal: the real site (`tool_registrations.py` ~2648, `_gap_comparability_reason`) raises `ToolRefusalError(..., reason_code=ReasonCode.COVERAGE_GAP)`, so the executor sets `outcome_class="refused"`, `reason_code="coverage_gap"`, `error_type="ToolRefusalError"`. Give `_failed_trace` those as keyword defaults. The three scope and length tests that use it (`test_total_failure_result_preserves_the_tool_reason`, `test_total_failure_reason_is_length_bounded`, `test_pathological_gap_reason_still_carries_the_scope`) then keep EVERY assertion unchanged. That is the proof #1574's verbatim disclosure survives. Changing any of their assertions instead is weakening.
> 2. **`test_step_with_success_flag_but_no_result_is_reported_failed` asserts `"gap_calculator: empty result"` in the answer**, from a step with no code and no tool-authored class. That asserts the defect #2020 removes. Change only that one line to the canonical rendering, `"gap_calculator: the tool failed to complete [tool_error]"`, and leave its `failed_components` assertion (the point of the test) unchanged. Say so in the commit body.
> 3. **Empty-text rule (lead call).** `test_total_failure_result_without_a_reason_is_unchanged` pins that a failed step with NO error text adds no reason fragment. Keep that contract: a step with neither raw text nor a code contributes nothing to `reasons`, because there is nothing to withhold and nothing to say. The Step 3 loop below encodes it; add one new test for it to the Task 4 test file.
> 4. **The Step 5 grep finds three files.** `test_composer_recording_wiring.py` calls `_create_total_failure_result` and asserts only `composition_id`, so it is unaffected. `test_chatbot_tools.py` mocks the answer string, so it is also unaffected. `test_gap_comparability_1574.py` is items 1–2 above. Also run `test_fail_closed_zero_tools_f6.py` and `test_binary_treatment_guard_2016.py`; the latter drives real refusals through a real executor.

**Files:**
- Modify: `src/agents/tool_composer/composer.py` — one module-level constant, and the reason loop in `_create_total_failure_result` (the method starts at about line 1217)
- Test: `tests/unit/test_agents/test_tool_composer/test_fail_closed_answer_sanitization_2020.py` (new)

The rule, from #2020: **a tool-authored refusal reaches the answer verbatim; any other failure is replaced by its code's canonical sentence, and the raw text goes to the log.** `outcome_class in {"refused", "input_rejected"}` is exactly "a tool authored this": the executor sets those two classes only in the arm that catches `ToolRefusalError` / `ToolInputError`, and after Task 3 always with a code. So a step in one of those classes WITHOUT a code did not come from that arm, and its text is not trusted.

`answer`, `caveats` and `errors` are all built from the one `reasons` list, so sanitizing where `reasons` is filled covers all three.

- [ ] **Step 1: Write the test file**

```python
"""#2020: the fail-closed answer keeps honest refusals and drops library internals.

Calls ``_create_total_failure_result`` directly with REAL decomposition and plan objects (both
are required fields of ``CompositionResult``), built the way ``test_fail_closed_zero_tools_f6``
builds them. No LLM call is made on this path.
"""

import logging
from datetime import datetime, timezone
from typing import Optional

from src.agents.tool_composer.composer import ToolComposer
from src.agents.tool_composer.models.composition_models import (
    CompositionResult,
    DecompositionResult,
    ExecutionPlan,
    ExecutionStatus,
    ExecutionTrace,
    StepResult,
    ToolInput,
    ToolOutput,
)

_COMPOSER_LOGGER = "src.agents.tool_composer.composer"

# The shape #2020 measured on the deployed image: causal_effect_estimator's RuntimeError carrying
# the DoWhy pipeline's own error text for an all-null treatment column.
_DOWHY_INTERNALS = (
    "causal_effect_estimator: DoWhy pipeline failed errors=[{'library': 'dowhy', 'error': "
    "\"DoWhy estimate_effect failed for method_name='backdoor.linear_regression': Found array "
    "with 0 sample(s) (shape=(0,)) while a minimum of 1 is required.\"}]"
)
_LEAKS = ("DoWhy", "dowhy", "backdoor.linear_regression", "shape=(0,)", "array with 0 sample")
_REFUSAL = (
    "gap_calculator: the estimation data covers only brand 'Kisqali'; a brand-vs-brand gap "
    "needs at least two brands. Refusing to report a gap against a single brand."
)


def _step(tool: str, error: str, outcome_class: str, reason_code: Optional[str]) -> StepResult:
    now = datetime.now(timezone.utc)
    return StepResult(
        step_id=f"s_{tool}",
        sub_question_id="sq_1",
        tool_name=tool,
        input=ToolInput(tool_name=tool, parameters={}),
        output=ToolOutput(tool_name=tool, success=False, error=error),
        status=ExecutionStatus.FAILED,
        started_at=now,
        completed_at=now,
        outcome_class=outcome_class,
        attempts=1,
        error_type="ToolRefusalError" if outcome_class == "refused" else "RuntimeError",
        reason_code=reason_code,
    )


def _fail_closed(*steps: StepResult) -> CompositionResult:
    decomposition = DecompositionResult(
        original_query="q", sub_questions=[], decomposition_reasoning="r"
    )
    plan = ExecutionPlan(
        decomposition=decomposition, steps=[], tool_mappings=[], planning_reasoning="r"
    )
    trace = ExecutionTrace(plan_id=plan.plan_id)
    for step in steps:
        trace.add_result(step)
    composer = ToolComposer(llm_client=object(), enable_memory_contribution=False)
    return composer._create_total_failure_result(
        "q", decomposition, plan, trace, datetime.now(timezone.utc), {}
    )


def _user_visible_text(result: CompositionResult) -> str:
    return " || ".join([result.response.answer, *result.response.caveats, *result.errors])


def test_a_tool_authored_refusal_reaches_the_answer_verbatim():
    result = _fail_closed(_step("gap_calculator", _REFUSAL, "refused", "coverage_gap"))
    assert _REFUSAL in result.response.answer


def test_an_input_rejection_reaches_the_answer_verbatim():
    text = "input contract violation: counterfactual_simulator: expected_effect is None"
    result = _fail_closed(
        _step("counterfactual_simulator", text, "input_rejected", "missing_required_input")
    )
    assert text in result.response.answer


def test_library_internals_reach_no_user_visible_field_but_do_reach_the_log(caplog):
    with caplog.at_level(logging.WARNING, logger=_COMPOSER_LOGGER):
        result = _fail_closed(
            _step("causal_effect_estimator", _DOWHY_INTERNALS, "error", "tool_error")
        )
    visible = _user_visible_text(result)
    for leak in _LEAKS:
        assert leak not in visible, f"{leak!r} leaked into a user-visible field"
    assert (
        "causal_effect_estimator: the tool failed to complete [tool_error]"
        in result.response.answer
    )
    assert _DOWHY_INTERNALS in caplog.text, "the raw text must still be logged"


def test_a_timeout_renders_its_own_sentence():
    result = _fail_closed(
        _step("refutation_runner", "exceeded the 120s step budget", "timeout", "tool_timeout")
    )
    assert (
        "refutation_runner: the tool exceeded its time budget [tool_timeout]"
        in result.response.answer
    )
    assert "120s" not in result.response.answer


def test_a_mixed_composition_keeps_the_refusal_and_drops_the_internals():
    result = _fail_closed(
        _step("gap_calculator", _REFUSAL, "refused", "coverage_gap"),
        _step("causal_effect_estimator", _DOWHY_INTERNALS, "error", "tool_error"),
    )
    assert _REFUSAL in result.response.answer
    visible = _user_visible_text(result)
    for leak in _LEAKS:
        assert leak not in visible


def test_a_refused_step_without_a_code_fails_closed():
    """After Task 3 the refusal arm always sets a code, so a refused step without one did not
    come from that arm and its text is not trusted."""
    result = _fail_closed(_step("mystery_tool", "raw text from somewhere", "refused", None))
    assert "raw text from somewhere" not in _user_visible_text(result)
    assert "mystery_tool: the tool failed to complete [tool_error]" in result.response.answer


def test_the_failed_components_are_unchanged():
    result = _fail_closed(
        _step("gap_calculator", _REFUSAL, "refused", "coverage_gap"),
        _step("causal_effect_estimator", _DOWHY_INTERNALS, "error", "tool_error"),
    )
    assert result.response.failed_components == ["gap_calculator", "causal_effect_estimator"]
```

The two sentence literals (`the tool failed to complete`, `the tool exceeded its time budget`) are copied from `CANONICAL_SENTENCES`. If the Task 1+2 review changed either sentence, update the literal here to match.

- [ ] **Step 2: Run it to verify the right tests fail**

```bash
cd /home/enunez/Projects/e2i_causal_analytics/.worktrees/lane-refusal-codes && \
  /home/enunez/Projects/e2i_causal_analytics/.venv/bin/pytest \
  tests/unit/test_agents/test_tool_composer/test_fail_closed_answer_sanitization_2020.py \
  -n 0 -p no:cacheprovider -q --timeout=300 -rA > /tmp/t4_red.log 2>&1; echo "EXIT=$?"; grep -E "^(PASSED|FAILED)" /tmp/t4_red.log
```

Expected — and be exact about it:
- **FAIL:** `test_library_internals_reach_no_user_visible_field_but_do_reach_the_log`, `test_a_timeout_renders_its_own_sentence`, `test_a_mixed_composition_keeps_the_refusal_and_drops_the_internals`, `test_a_refused_step_without_a_code_fails_closed` — each on an assertion about leaked or missing text.
- **PASS already:** `test_a_tool_authored_refusal_reaches_the_answer_verbatim`, `test_an_input_rejection_reaches_the_answer_verbatim`, `test_the_failed_components_are_unchanged`. They are the regression guard for #1574's verbatim disclosure, not the red.

A `ValidationError` or `TypeError` anywhere means a builder is wrong. Fix the builder before going further.

- [ ] **Step 3: Implement the split in `composer.py`**

Add the import beside the module's other relative imports:

```python
from .reason_codes import canonical_sentence
```

Add the constant directly below `_TRUNCATION_MARKER`:

```python
# #2020: the only outcome classes a TOOL authors. The executor sets these two solely in the arm
# that catches ToolRefusalError / ToolInputError, and since #2021 always with a reason code.
_TOOL_AUTHORED_CLASSES = frozenset({"refused", "input_rejected"})
```

In `_create_total_failure_result`, replace the whole `for step in getattr(execution_trace, "step_results", None) or []:` loop — from that line through the end of its `if reason: reasons.append(...)` — with:

```python
        # #2020: what a failed step is allowed to say to the user.
        #
        # A tool-authored refusal is an honest, user-meaningful finding: #1574's gap_calculator
        # states which entity groups the estimation data actually covered, and a one-step plan
        # fail-closes here, before synthesis, so dropping it would leave the answer LESS
        # informative. Those reach the user verbatim, exactly as before.
        #
        # Every other failure carries library internals — DoWhy/sklearn shape errors, driver
        # messages, file paths, reprs of inputs. They mean nothing to a pharma leader, read as a
        # crash rather than a finding, and expose implementation detail. Those are replaced by
        # the closed code's canonical sentence, and the raw text goes to the log.
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
            reason_code = getattr(step, "reason_code", None)
            if not raw and not reason_code:
                # Nothing to withhold and nothing to say: no reason fragment, as before #2020.
                continue
            if getattr(step, "outcome_class", None) in _TOOL_AUTHORED_CLASSES and reason_code:
                if raw:
                    reasons.append(f"{tool_name}: {raw}")
                continue
            # Not tool-authored, or uncoded (which fails closed the same way).
            if raw:
                logger.warning(
                    "Step %s tool %r failed with non-user-facing text (reason_code=%s): %s",
                    getattr(step, "step_id", "?"),
                    tool_name,
                    reason_code,
                    raw,
                )
            code = reason_code or "tool_error"
            reasons.append(f"{tool_name}: {canonical_sentence(code)} [{code}]")
```

Keep the existing comment block that follows the loop (the one explaining that the answer is returned verbatim and bounded), but correct its first sentence so it no longer says the per-step reasons are carried unconditionally.

- [ ] **Step 4: Run it to verify it passes**

```bash
cd /home/enunez/Projects/e2i_causal_analytics/.worktrees/lane-refusal-codes && \
  /home/enunez/Projects/e2i_causal_analytics/.venv/bin/pytest \
  tests/unit/test_agents/test_tool_composer/test_fail_closed_answer_sanitization_2020.py \
  tests/unit/test_agents/test_tool_composer/test_fail_closed_zero_tools_f6.py \
  -n 0 -p no:cacheprovider -q --timeout=300 > /tmp/t4_green.log 2>&1; echo "EXIT=$?"; tail -3 /tmp/t4_green.log
```

Expected: `EXIT=0`, 7 new tests plus f6 passing.

- [ ] **Step 5: Confirm no existing fail-closed expectation regressed**

```bash
cd /home/enunez/Projects/e2i_causal_analytics/.worktrees/lane-refusal-codes && \
  grep -rln "Unable to complete analysis\|Reason(s):\|_create_total_failure_result" tests/
```

Run every file that grep names, `-n 0`, exit code captured. Several drive REAL refusals end to end — `test_binary_treatment_guard_2016.py` builds a `ToolComposer`, and the #1574 gap-coverage tests carry `estimation_data_scope` through this answer — and they must stay green: that is Task 3 and Task 4 working together.

A test that asserted RAW EXCEPTION TEXT (not a refusal) in the answer was asserting the defect. Update it, and say so in the commit body. Never weaken a new test to make an old one pass.

- [ ] **Step 6: Commit**

```bash
cd /home/enunez/Projects/e2i_causal_analytics/.worktrees/lane-refusal-codes && \
  /home/enunez/Projects/e2i_causal_analytics/.venv/bin/ruff check --no-cache src/agents/tool_composer/ tests/unit/test_agents/test_tool_composer/ && \
  git add src/agents/tool_composer/composer.py \
          tests/unit/test_agents/test_tool_composer/test_fail_closed_answer_sanitization_2020.py && \
  git commit -m "fix(tool-composer): keep library internals out of the fail-closed answer (#2020)

Tool-authored refusals (outcome_class refused/input_rejected with a reason code, which only the
coded-exception arm produces) still reach the user verbatim; #1574's coverage disclosure depends
on it. Every other failure, and any uncoded one, now renders its code's canonical sentence; the
raw DoWhy / sklearn / driver text goes to the log only.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01SnzgDeMLxZN48UsJXTaazb"
```

---

## Task 4b: Fix library text at its source (D5)

> **Added 2026-09-12 for owner decision D5.** Task 4 trusts a coded refusal's text. That trust is only sound if no coded refusal carries library text inside it — and eight sites do. Lines are at `956a57e3b`; re-derive them before editing.

**Files:**
- Modify: `src/agents/tool_composer/tool_registrations.py`, `src/digital_twin/effect/cohort_causal_estimator.py`, `src/digital_twin/effect/estimator.py`
- Test: `tests/unit/test_agents/test_tool_composer/test_refusal_text_is_authored_2020.py` (new), plus the existing estimator tests extended in place

**The rule at each fixed site:** remove the interpolated exception text from the message; keep every authored word (pinned substrings that must survive include `"not a DataFrame"` — `test_sensitivity_derives_inputs_2022.py:260` — and any column or region name the message already states, such as `"atlantis"` and `"west"`); log the original exception at the wrap site with the module logger (`exc_info=exc`, WARNING); keep `raise ... from exc` so the cause stays on the chain.

| Site | Wraps | Why it is library text |
|---|---|---|
| `tool_registrations.py:1669` refutation_runner `not a DataFrame ({exc})` | `except Exception` around `df.columns` | an `AttributeError` repr of whatever object was passed |
| `:1822` sensitivity_analyzer `not a DataFrame ({exc})` | same | same |
| `:1842` sensitivity_analyzer `could not be computed … : {exc}` | `except Exception` around `evalue.benchmark_inputs_from_frame` / `outcome_std_from_frame` | pandas / numpy errors from the frame |
| `:2292` cate_analyzer `is not numeric … : {exc}` | `except (TypeError, ValueError)` around pandas `mean()` | `"Could not convert string 'hi' to numeric"` — carries a data VALUE |
| `:2560` gap_calculator `is not numeric … : {exc}` | pandas `groupby().mean()` | `"agg function failed [how->mean,dtype->object]"` |
| `cohort_causal_estimator.py:234` `cohort causal estimation failed for …: {e}` | `except Exception` around econml `CausalForestDML` | econml / sklearn internals; reaches the user through `tool_registrations.py:3865` and, via `SimulationEngine`'s `"Effect estimation failed: {e}"`, `:3987` and the Digital Twin route |
| `cohort_causal_estimator.py:259` `target-region inference failed …: {e}` | econml `ate_interval` | same |
| `estimator.py:78` `uplift fit failed: {result.error_message}` | `UpliftRandomForest.estimate`'s result | **lead addition, not in D5's list:** `SimulationEngine` defaults to `TwinEffectEstimator` when no estimator is passed, which `experiment_designer/tools/simulate_intervention_tool.py:269` and `digital_twin/simulation_runner.py:96` do. **Lead-verified 2026-09-12: it is library text.** `UpliftRandomForest` inherits `estimate` from `src/causal_engine/uplift/base.py`, whose `except Exception as e` (:369) sets `error_message=str(e)` (:374), so causalml / sklearn text reaches this message. Fix it here by the rule above: keep `"TwinEffectEstimator: uplift fit failed"`, log `result.error_message`, and add no `from` (there is no exception object in hand). The AST guard does not see this site, because the text is a model attribute and not an `except` name, so pin it with a behavioral test. |

**Kept verbatim — the wrapped text is authored all the way down** (the allowlist; keyed by enclosing FUNCTION name, never by line):
- `:2014`, `:2070` sensitivity_analyzer — `evalue`'s own `raise ValueError(...)` messages are authored (lead-verified: `src/causal_engine/evalue.py` :100, :107, :166, :256, :654, :761, :814). **Each site catches ANY `ValueError`, and the lead has shown that every one of them is authored (2026-09-12), so both stay allowlisted.**
  - Both wrap calls on already-derived numbers only: `float(ate)`, the CI bounds, and `_SensitivityInputs` floats and ints. No frame reaches them.
  - `evalue`'s only `math` calls on inputs are safe:
    - `e_value_from_rr` runs `_orient(_finite(rr))` and returns early when `r <= 1` before `math.sqrt(r - 1.0)` (evalue.py:120–123), so there is no domain error.
    - `rr_from_smd`'s `math.exp` can only overflow, which raises `OverflowError`, not `ValueError`. That lands in the executor's generic arm, which Task 4 already renders as the canonical sentence.
  - The allowlist entry's one-line reason should cite this.
- No existing test pins text this task removes (lead grep, 2026-09-12). The only pinned phrase in scope is `"not a DataFrame"`, which survives. `"uplift fit failed"` in `test_executor_causalml.py:732` belongs to a different path (the causalml executor) and is unaffected.
- `:3489` power_calculator — **split this site (lead call, 2026-09-12, after the Task 4 quality review).** It is `except (PowerCalculationError, ArithmeticError) as exc`. The `PowerCalculationError` half is authored and pinned (`test_power_calculator_2015.py`, `match=reason`), so it stays verbatim. The `ArithmeticError` half carries Python's own text (`OverflowError` "math range error", `ZeroDivisionError` "float division by zero"), which is not authored. Split it into two `except` clauses:
  - `PowerCalculationError` keeps `f"power_calculator: {exc}"` and `INVALID_INPUT_VALUE`.
  - `ArithmeticError` raises the same class and code with an authored sentence (e.g. `"power_calculator: the requested design is outside the range the calculation can represent."`), logs `exc`, and keeps `from exc`.
  - The guard's allowlist names `power_calculator` for the `PowerCalculationError` clause only. The guard must still flag an `ArithmeticError` clause that interpolates its name, so key that allowlist entry by function AND the caught exception type.
  - Add a behavioral test: a design that overflows produces no Python arithmetic text in the refusal, and does produce it in `caplog`.
- `:3865` — `EffectDataUnavailable`, authored once `:234` / `:259` are fixed.
- `:3987` — `result.error_message`, which is `SimulationEngine`'s authored prefix plus an authored `EffectDataUnavailable` / `EstimationError` once the estimator sites are fixed. It is a model attribute, not an `except` name, so the guard below does not see it — pin it with a behavioral test instead.
- `src/digital_twin/effect/recommendation.py:98` — interpolates `control_outcome_sd`'s `EffectDataUnavailable`, which is authored.

> **Sites re-derived by AST at `4d1892dc7` (lead, 2026-09-12).** Line numbers below replace the table's, which predate Tasks 2b and 3. The search covered every `raise ToolRefusalError/ToolInputError/EffectDataUnavailable/EstimationError` inside an `except … as NAME` whose arguments reference `NAME`.
> - **Fix (7):** `tool_registrations.py` :1669 `_run_dowhy_refutation`, :1822 and :1842 `_derive_sensitivity_inputs`, :2292 `cate_analyzer`, :2560 `gap_calculator`; `cohort_causal_estimator.py` :234 and :259 `estimate_cohort_effect`.
> - **Split (1):** :3489 `power_calculator`. The `PowerCalculationError` clause stays verbatim (allowlisted); the `ArithmeticError` clause is fixed (see the allowlist bullet below).
> - **Allowlist (4 entries):** :2014 `sensitivity_analyzer`, :2070 `_point_only_sensitivity`, :3876 `_targeted_effect`, keyed by function; `power_calculator`, keyed by function AND caught type `PowerCalculationError`.
>   - **No function other than `power_calculator` contains both a site to fix and an allowlisted site.** Keying those three entries by function therefore cannot exempt a site that should be fixed.
>   - Assert this in the guard test: an allowlisted function must hold only allowlisted sites, with `power_calculator` qualified by exception type.
> - **Model-attribute interpolations, invisible to the AST guard, pinned behaviorally:** `estimator.py:78` (fix) and `tool_registrations.py:4000` `_simulation_results` (keep; authored once the estimator sites are fixed).
> - **Correction:** `recommendation.py:98` *returns* its text; it does not raise it. The guard cannot see it, and it does not belong on the allowlist.

> **Seams, lead-verified 2026-09-12. Reuse these; do not invent new ones.**
> - **econml sites** (`cohort_causal_estimator.py` :234, :259). No existing test patches econml. `CausalForestDML` is imported INSIDE `estimate_cohort_effect` (`from econml.dml import CausalForestDML`), so patch the attribute on `econml.dml` (`monkeypatch.setattr("econml.dml.CausalForestDML", ...)`). A fake whose `fit` raises `RuntimeError("LIBTEXT_SENTINEL")` covers :234. A fake whose `fit`/`effect` succeed and whose `ate_interval` raises on the second, target-region call covers :259. Take the cohort fixtures from `tests/unit/test_digital_twin/effect/test_cohort_causal_estimator.py`.
> - **`estimator.py:78`**: `tests/unit/test_digital_twin/test_error_recovery.py` (:111, :215, :608) and `test_engine_real_effect.py:90` already construct `TwinEffectEstimator`. Patch `UpliftRandomForest.estimate` to return an unsuccessful result whose `error_message` is the sentinel.
> - **evalue allowlist** (:2014, :2070). Both `except ValueError` blocks wrap calls on already-derived numbers only: `float(ate)`, the CI bounds, and `_SensitivityInputs` floats and ints (`tool_registrations.py:1722`). No frame reaches them, so pandas cannot raise inside. What remains to establish is that Python `math` inside `evalue` cannot raise its own `ValueError("math domain error")` before `evalue`'s authored guards run (see the note under the allowlist).
> - **Non-DataFrame sites** (:1669, :1822). `test_sensitivity_derives_inputs_2022.py:260` already asserts `match="not a DataFrame"` for sensitivity; add the sentinel assertions beside it. For refutation_runner, reuse the calling pattern from `test_refutation_estimate_id_optional_2014.py` / `test_real_tools_778.py`.

- [ ] **Step 1: Red — behavioral tests, one per fixed site.** Force each wrap with a sentinel: pass an object whose `.columns` raises `AttributeError("LIBTEXT_SENTINEL")`; monkeypatch the `evalue` helper, or the econml fit and `ate_interval`, to raise `RuntimeError("LIBTEXT_SENTINEL")`; use a non-numeric outcome for the pandas sites and assert pandas' own phrase is absent. For each, assert:
  - the sentinel (or pandas phrase) is NOT in `str(err)`
  - the authored words and pinned substrings ARE
  - the sentinel IS in `caplog.text`
  - `err.__cause__` is the original exception

  Reuse the seams existing tests already use for these functions (`grep -rn` them first); do not fit a real forest when a patch reaches the branch. Add one test for `:3987` end to end at the tool boundary: a failed engine result whose estimator raised sentinel text does not carry the sentinel into the refusal.
- [ ] **Step 2: Red — the AST guard,** `test_no_coded_refusal_interpolates_a_caught_exception`. Scope: `tool_registrations.py` and every `src/digital_twin/effect/*.py`. For every `except … as NAME` handler, any `raise` inside it of `ToolRefusalError`, `ToolInputError`, `EffectDataUnavailable` or `EstimationError` whose arguments reference `NAME` — f-string `FormattedValue`, `str(NAME)`, `repr(NAME)`, `%` formatting or `.format(NAME)` — is a violation unless the enclosing function is on the allowlist, each entry carrying a one-line reason. Prove it has teeth with synthetic-source tests covering each reference form; a green run on the real files proves nothing by itself.
- [ ] **Step 3: Green.** Fix the sites. Then find existing tests that asserted the removed exception text (`grep -rn "Could not convert\|agg function failed\|estimation failed for\|inference failed for\|uplift fit failed" tests/`). A test that asserted library text was asserting the defect — update it and say so in the commit body.
- [ ] **Step 4: Verify.** The new file, the estimator tests, `test_sensitivity_derives_inputs_2022.py`, `test_power_calculator_2015.py`, then all of `tests/unit/test_agents/test_tool_composer/` and `tests/unit/test_digital_twin/`, each `-n 0` with `$?` captured. `ruff check --no-cache` and `ruff format --check --no-cache` on every changed file.
- [ ] **Step 5: Commit** — `fix(tool-composer,digital-twin): keep library exception text out of authored refusals (#2020, D5)`.

---

## Task 4c: The same rule for a phase that throws (#2020, found by the Task 4 quality review)

> **Added 2026-09-12.** Task 4 sanitized the *total-failure* answer. The Task 4 quality review then found the same answer family reached by a second door. When a whole phase raises, `composer.py`'s four `_fail_closed` callers pass `f"Decomposition failed: {e}"`, `f"Planning failed: {e}"`, `f"Execution failed: {e}"` and `f"Unexpected error: {e}"` into `_create_error_result`. That builds `answer=f"Unable to complete analysis: {error}"` and `caveats=[error]`, so any library exception escaping a phase reaches the user verbatim. Separately, `agent.py`'s composer-exception path sets `ToolComposerOutput(error=str(e))`. This is squarely #2020's answer, not a new surface, so it stays in this lane. It is recorded as its own task so Task 4's fix round stays reviewable.

**Measure before changing, site by site** (the same discipline as Task 7b):
- Grep the tests that pin these strings, and read what each asserts.
- For `ToolComposerOutput.error`, find every consumer (orchestrator, API routes, frontend) and establish whether it reaches a user.
- For each phase, list what can actually raise inside it: an authored `DecompositionError` / `PlanningError` / `ExecutionError`, or anything at all.

**Lead classification of the phase-error raises (2026-09-12). This supersedes the "authored may keep its text" assumption below, because the phase classes are NOT authored all the way down.**

**Library text wrapped at source — fix D5-style:**
- `decomposer.py:141`: `except Exception as e` → `DecompositionError(f"Failed to decompose query: {e}")`
- `decomposer.py:174`: `except (json.JSONDecodeError, TypeError) as e` → `f"Invalid JSON in LLM response: {e}"`
- `planner.py:302`: `except Exception as e` → `PlanningError(f"Failed to create execution plan: {e}")`
- `planner.py:695`: `JSONDecodeError`/`TypeError` → `f"Invalid JSON in LLM response: {e}"`
- `executor.py:512`: `except Exception as e` → `ExecutionError(f"Plan execution failed: {e}")`

**Authored — keep (11):**
- `decomposer.py`: `:181` too few sub-questions, `:212` invalid dependency (ids from the decomposition), `:239` dependency cycle.
- `planner.py`:
  - `:440` no tools
  - `:698` names only `type(parsed).__name__`
  - `:744` unknown tool in plan (a planner-LLM tool name, not library text)
  - `:801` cannot map a sub-question
  - `:806` step references an unknown tool
  - `:813` step depends on an unknown step
  - `:906` unbound column: plan values plus the schema's own column list, which is intentionally user-meaningful
  - `:1058` cycle in the execution plan

The full repo-wide count is 16 = 5 fix + 11 keep; every raise of the three classes is classified above. Re-derive the line numbers by AST before editing.

**Trap at the three catch-alls (`decomposer.py:141`, `planner.py:302`, `executor.py:512`).** Each `except Exception` also catches its module's OWN authored error raised deeper in the same `try`, e.g. `:181`'s "Too few sub-questions", and re-wraps it with `{e}`. Merely dropping `{e}` would erase those useful authored messages from the user's view. Use the pattern `cohort_causal_estimator.py:231` already uses:
- Add `except DecompositionError: raise` (and the equivalent for the other two classes) BEFORE the generic `except Exception`, so authored errors pass through unchanged.
- The generic arm raises the class with a fixed sentence, logs `e`, and keeps `from e`.

**Follow-through:**
- Extend Task 4b's AST guard: add `DecompositionError`, `PlanningError` and `ExecutionError` to its target classes, and `decomposer.py`, `planner.py` and `executor.py` to its scope, so these wraps cannot regress.
- Enumerate every raise of the three classes by AST before editing. The repo-wide grep finds 16, and this list names 12.

**Existing tests this touches (lead-measured 2026-09-12; decides which changes are faithful).** No test pins the full text of the five wrap messages. The tests that catch the three classes and read the message split into two groups:

- **Survive unchanged.** They check a substring of an AUTHORED message, which the `except <OwnClass>: raise` pass-through keeps intact:
  - `test_decomposer.py` :121 `"Too few sub-questions"`, :174 `"cycle"`, :201 `"unknown"`, :259 `"JSON"`
  - `test_planner.py` :202 `"unknown"`, :249/:449 `"no tools"`, :306 `"cycle"`, :529 `"JSON"`
  - `test_planner_semantic_binding_f6b.py` :255 `"conversion_rate"` + `"unbound column"`, :312 `"dosage"`

  For the two `"JSON"` checks, keep the authored prefix `"Invalid JSON in LLM response"` in the fixed sentence.
- **Assert the defect; update them.** `test_decomposer.py:397` and `test_planner.py:560` set the LLM client to raise `Exception("LLM error")` and assert `"LLM error" in str(exc_info.value)`. That is raw client-exception text riding the phase error to the user, exactly what this task removes. Update each to assert:
  - the fixed sentence;
  - `"LLM error" in caplog.text`;
  - `isinstance(exc_info.value.__cause__, Exception)` with that text.

  Name both in the commit body. This is a faithful update, not a weakening.
- **Also run:** `test_executor_outcome_classes.py:605` (`pytest.raises(ExecutionError)`, no message check), `test_fail_closed_zero_tools_f6.py`, `test_execution_order_repair.py`, `test_planner_column_awareness.py`, `test_planner_token_budget_1365.py`, `tests/unit/test_api/test_chatbot_tools_composer_di_1557.py`, `test_plan_cache_eviction.py`, `test_composer.py`.

**Rule (revised):**
- Once the wraps above are fixed at source, the three phase classes carry only authored text, so the composer keeps `Decomposition failed: {e}` / `Planning failed: {e}` / `Execution failed: {e}` as they are.
- Only `Unexpected error: {e}` (`composer.py:593`) still carries arbitrary exception text; it becomes a fixed sentence, and its existing `logger.exception` keeps the raw text.
- The general rule for anything else stays:
- Anything else renders `"<Phase> failed: <fixed sentence>"`. Use a phase-level sentence, not a `ReasonCode`; these are not step failures.
- The raw exception goes to `logger.warning` / `logger.exception` with `exc_info`.
- `agent.py`'s `error=str(e)` follows the same rule when the error reaches a user. Report it and leave it unchanged when it provably does not.

**Tests:** red first. For each phase, force a library-style exception carrying a sentinel. Assert the sentinel is absent from `answer`, `caveats`, `errors` and `error`, and present in `caplog`. Add one test showing an authored phase error keeps its text.

**Stop condition:** if a test or a consumer pins the raw phase text as a contract, report `NEEDS_CONTEXT` before changing it.

**Leads the lead measured on 2026-09-12. They are starting points, not conclusions: trace each one to a user-visible surface.**
- **Phase sites:** `composer.py` :550 / :563 / :577 / :593 build the phase string, which reaches `answer` at :1195 and `caveats`.
- **Test pins:** only `test_plan_cache_eviction.py:348` pins these strings, with `"Execution failed" in (result.error or "")`. A `"<Phase> failed: <fixed sentence>"` rule keeps that green. `test_composer.py:163` only raises a `RuntimeError("Unexpected error")`, so read what it asserts before assuming it is unaffected.
- **Answer-path consumer:** `chatbot_tools.py:29` imports `compose_query` and returns `response.answer`, the path Task 4 already sanitizes.
- **Agent path:**
  - `ToolComposerAgent` is reached through the orchestrator (`_agent_method_map.py:87`) and `factory.py:150`.
  - `agent.py:457–463` returns `ToolComposerOutput(error=str(e))` and has already logged `exc_info=True` at :453.
  - Establish whether the orchestrator or any route renders that `error` to a user. `chatbot_graph.py:2915` / `:3011` read `result.get("error")` — check whether that `result` is this output.
  - If `error` does reach a user, it gets a fixed sentence (the log line already holds the raw text). If it does not, leave it and cite the evidence.

---

## Task 5: The recorder emits the code and its details — and no text (#2050 cause 1)

> **Rewritten 2026-09-12 for D1′, before dispatch.** The previous version emitted an `error_message` sentence from the recorder. Under D1′ the recorder sends NO text of any kind: the sentence is rendered at read time (Task 7), and ml/041's RPC keeps writing `NULL` into `error_message` as its second guard. Still true from the earlier correction: `step_record(step_number, result, plan, *, allowlist)` needs a REAL `ExecutionPlan`, `ExecutionStatus` values are lowercase, and `test_step_record_fields` pins the record by EXACT dict equality, so it is updated deliberately below.

Finding 1 still applies: `StepResult` has no `error_message`. Nothing in this task reads `result.output.error`.

**Serialization boundary (quality review M4, 2026-09-12).** `ToolRefusalError.details` is validated once, at construction, and stays a mutable dict — `err.details["s"] = "Kisqali"` goes through afterwards. (A `MappingProxyType` would break `__reduce__`, so the attribute is not frozen.) The recorder therefore re-runs `validate_details(dict(result.reason_details))` when it builds the record and does NOT trust the attribute. On a `ValueError` it sends `{}` and logs, the same fail-soft rule as the constructor. Add one test: a `StepResult` whose `reason_details` was mutated to carry a string is serialized with `{}`. Detail keys follow the convention `^(n|is|has|share)_[a-z0-9_]+$` from Task 2b's quality fix.

**Files:**
- Modify: `src/agents/tool_composer/learning_recorder.py` — `step_record` (starts at about line 242)
- Test: `tests/unit/test_agents/test_tool_composer/test_learning_recorder_serializer.py` — extend in place

- [ ] **Step 1: Write the failing tests**

(a) Extend the `_result` builder with two keyword arguments:

```python
def _result(
    step: ExecutionStep,
    *,
    outcome_class: str,
    result: Optional[Dict[str, Any]] = None,
    error: Optional[str] = None,
    error_type: Optional[str] = None,
    attempts: int = 1,
    reason_code: Optional[str] = None,
    reason_details: Optional[Dict[str, Any]] = None,
) -> StepResult:
```

and pass `reason_code=reason_code, reason_details=reason_details or {},` into its `StepResult(...)`.

(b) In `_sentinel_models`, give the three failed results codes: the `dag` error gets `reason_code="tool_error"`; the `rank` input rejection gets `reason_code="missing_required_input"`; the `gap` refusal gets `reason_code="coverage_gap", reason_details={"n_groups": 1}`. Their `error=` strings already embed `SENTINEL`, so `test_sentinel_absent_everywhere` becomes the proof that the raw message still never reaches the record even though the record now carries a reason.

(c) Update `test_step_record_fields`: pass `reason_code="tool_timeout"` to its `_result(...)`, and append these two entries to its expected dict literal, after the existing last key:

```python
        "reason_code": "tool_timeout",
        "reason_details": {},
```

Because that assertion is exact equality, it also proves no `error_message` key is emitted.

(d) Append:

```python
# ---------------------------------------------------------------------------
# #2050: the recorded step carries the reason as structure, never as text
# ---------------------------------------------------------------------------


def test_a_refused_step_records_its_code_and_numeric_details_and_no_text():
    d = _decomposition(["CAUSAL"])
    ate = _step("ate", "causal_effect_estimator", "sq_0", {})
    plan = _plan(d, [ate], [["ate"]])
    result = _result(
        ate,
        outcome_class="refused",
        error=f"treatment column {SENTINEL!r} carries 4 distinct non-null values",
        error_type="ToolRefusalError",
        reason_code="non_binary_treatment",
        reason_details={"n_distinct": 4},
    )
    record = step_record(0, result, plan, allowlist=None)
    assert record["reason_code"] == "non_binary_treatment"
    assert record["reason_details"] == {"n_distinct": 4}
    # D1′: the sentence is rendered at read time. The recorder sends no text, and the RPC's
    # NULL in error_message (ml/041) stays the second guard behind this one.
    assert "error_message" not in record
    assert SENTINEL not in json.dumps(record)


def test_a_succeeded_step_records_no_reason():
    d = _decomposition(["CAUSAL"])
    ate = _step("ate", "causal_effect_estimator", "sq_0", {})
    plan = _plan(d, [ate], [["ate"]])
    record = step_record(
        0, _result(ate, outcome_class="succeeded", result={"ate": 0.1}), plan, allowlist=None
    )
    assert (record["reason_code"], record["reason_details"]) == (None, {})


def test_every_failed_step_of_a_whole_record_carries_its_code():
    d, plan, trace = _sentinel_models()
    steps = to_record(decomposition=d, plan=plan, trace=trace, allowlist=CATALOG)["steps"]
    assert [s["reason_code"] for s in steps] == [
        None,
        "tool_error",
        "missing_required_input",
        "coverage_gap",
    ]
```

If `json` is not already imported in this file, add `import json`.

- [ ] **Step 2: Run to verify the right tests fail**

```bash
cd /home/enunez/Projects/e2i_causal_analytics/.worktrees/lane-refusal-codes && \
  /home/enunez/Projects/e2i_causal_analytics/.venv/bin/pytest \
  tests/unit/test_agents/test_tool_composer/test_learning_recorder_serializer.py \
  -n 0 -p no:cacheprovider -q --timeout=300 > /tmp/t5_red.log 2>&1; echo "EXIT=$?"; grep -E "Error|assert" /tmp/t5_red.log | head -12
```

Expected: the three new tests fail with `KeyError: 'reason_code'`, and `test_step_record_fields` fails on the dict comparison. `test_sentinel_absent_everywhere` still PASSES — it is a guard, not the red.

- [ ] **Step 3: Add the two keys to the step-row dict**

In `step_record`, directly after `"error_type": result.error_type,`:

```python
        # #2050: an operator seeing `refused` with no why is a missing explanation on a
        # decision-support surface. The CODE is the aggregation key and the details are numeric
        # structure. No text is sent: the canonical sentence is rendered at read time from
        # reason_codes.CANONICAL_SENTENCES, and ml/041's RPC still writes NULL into
        # error_message as the database's own guard ("no error text is stored", 041 header).
        "reason_code": result.reason_code,
        "reason_details": dict(result.reason_details or {}),
```

- [ ] **Step 4: Run to verify it passes**

```bash
cd /home/enunez/Projects/e2i_causal_analytics/.worktrees/lane-refusal-codes && \
  /home/enunez/Projects/e2i_causal_analytics/.venv/bin/pytest \
  tests/unit/test_agents/test_tool_composer/test_learning_recorder_serializer.py \
  tests/unit/test_agents/test_tool_composer/test_composer_recording_wiring.py \
  -n 0 -p no:cacheprovider -q --timeout=300 > /tmp/t5_green.log 2>&1; echo "EXIT=$?"; tail -3 /tmp/t5_green.log
```

Expected: `EXIT=0`. The real-database recorder test is opt-in and skips on this droplet (D4); a skip is not evidence. The persisted row is proven by Task 6's rehearsal.

- [ ] **Step 5: Commit**

```bash
cd /home/enunez/Projects/e2i_causal_analytics/.worktrees/lane-refusal-codes && \
  /home/enunez/Projects/e2i_causal_analytics/.venv/bin/ruff check --no-cache src/agents/tool_composer/ tests/unit/test_agents/test_tool_composer/ && \
  git add src/agents/tool_composer/learning_recorder.py \
          tests/unit/test_agents/test_tool_composer/test_learning_recorder_serializer.py && \
  git commit -m "fix(tool-composer): recorder emits the refusal reason code and details (#2050 cause 1)

The issue said StepResult.error_message exists and the executor populates it; it does not -
schemas.py's error_message belongs to CompositionResult, and the step's text lives on
output.error. The recorded step now carries the closed reason_code and its numeric details and
no text at all: the sentence is rendered at read time (owner decision D1'), so ml/041's RPC
keeps its NULL in error_message as the database's second guard. test_step_record_fields pins
the payload by exact equality and is updated for the two new keys.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01SnzgDeMLxZN48UsJXTaazb"
```

---

## Task 6: Migration ml/043 — the RPC carries the reason as structure (#2050 cause 2)

> **Renumbered 042 → 043 (lead, 2026-09-12 23:58Z).** `origin/main` (#2053, merged into this branch as `38d60e9fe`) took `database/ml/042_twin_simulations_estimate_scope.sql` + `rollback_042.sql`. That migration only alters `twin_simulations`, which does not overlap this task. `_pg.LANE_MIGRATIONS` still lists only 039–041. Step 0 still applies: re-confirm `043` is free before writing.

> **CI and harness facts (lead-verified 2026-09-12, before dispatch).**
> - **The real-DB suite never runs in CI.** `tests/unit/test_database/learning_loop/conftest.py:19` skips unless `E2I_DB_INTEGRATION=1`, and no workflow under `.github/` sets that variable. `backend-tests.yml` still *collects* `tests/unit/test_database/`.
> - **So the break in `test_migration_runner.py` is latent, not a red CI.** Once 043 exists, its assertions that the pending list is exactly `[ml/041…]` (:111) and that `Applying ` appears once (:117) are wrong. They only run with the gate on, and on this droplet they skip anyway, because prod holds 039–041. Disclose it in the PR body, as planned; do not edit the shared harness.
> - **Stall-watchdog guard.** `tests/unit/test_tests_meta/test_session_stall_watchdog_1655.py` statically scans every `@pytest.mark.timeout` the backend lane collects, gated or not, and requires the lane's 600 s window to be at least 2× the largest. The current maximum is 300 s, in this directory. **No new 043 test may carry a timeout marker above 300 s.**

> **Rewritten 2026-09-12 for D1′ and D4, before dispatch.** The first version stored `left(s->>'error_message', 200)` and proved it with an in-harness test. Both are reversed. ml/041's `NULL` in the `error_message` slot is the database's deliberate second guard — `test_041_recording.py:441`, `:668–740` and `test_learning_recorder_realdb.py:363` pin it — so **043 keeps that `NULL`**. The shared `_pg` fixture skips on this droplet because prod's ledger already holds 039–041 (`base_db` skips when any of `_pg.LANE_MIGRATIONS` is in prod), so **043 is proven by a scratch rehearsal** on a throwaway container cloned from prod, and only static checks are committed.

**Why #2050 cause 2 is still the thing fixed here.** Cause 2 was "the RPC drops the reason on the floor". Under D1′ the reason is the code and its numeric details, and 043 makes the RPC carry both into `composition_steps` and the code into `tool_performance`. The `NULL` it keeps is for *text*, which D1′ never sends.

**Files:**
- Create: `database/ml/043_composer_refusal_reason_codes.sql`
- Create: `database/ml/rollback_043.sql`
- Modify: `tests/unit/test_database/learning_loop/test_lane_migration_files.py` — static; runs in CI
- Create, **untracked** (like every prior cert): `docs/demos/results/2026-09-12_lane_refusal_codes_cert/rehearse_043.py` and its transcript

Facts this task relies on, all verified on `0a16e9c18`:
- `_pg.build_base` restores prod's full schema dump, so the throwaway base already contains ml/041's `composer_record_steps` and `get_tool_reliability`; only its ledger omits the lane keys. Apply 043 directly with `_pg.apply_migration` — do NOT call `_pg.migrate`.
- ml/041 revokes every function and view from `PUBLIC, anon, authenticated` and grants only `service_role`, and it sets no function comments. `DROP FUNCTION` discards an ACL, and a recreated function is executable by `PUBLIC` by default, so 043 must re-apply those grants.
- `scripts/run_migrations.sh` skips `rollback_*.sql` by glob and runs on every deploy (`deploy.yml`), so 043 is applied automatically, wrapped in one transaction with its ledger row.
- `test_lane_migration_files.py` forbids the runner's unwrap-trigger WORDS anywhere in a wrapped file, comments included. Do not write them in 043 or its rollback.

- [ ] **Step 0: Confirm `043` is still free**

```bash
cd /home/enunez/Projects/e2i_causal_analytics && git fetch origin main --quiet && \
  git ls-tree --name-only origin/main database/ml/ | grep -E '/04[2-9]_' ; \
  ls .worktrees/*/database/ml/04[2-9]_* 2>/dev/null; echo "(nothing above = free)"
```

If `043` is taken, use the next free number consistently in both filenames, the ledger key and every step below.

- [ ] **Step 1: Write the failing static tests**

In `tests/unit/test_database/learning_loop/test_lane_migration_files.py`:

(a) Add to the `test_runner_branch` parameter list:

```python
        ("ml/043_composer_refusal_reason_codes.sql", False),
```

(b) Change `test_no_script_level_transaction_control`'s parametrize to cover 043 explicitly:

```python
# ml/043 is deliberately NOT in _pg.LANE_MIGRATIONS: that tuple defines the shared fixture's
# "prod before this lane" base, which is 039-041's (D4, 2026-09-12).
_REASON_CODES_MIGRATION = "ml/043_composer_refusal_reason_codes.sql"


@pytest.mark.parametrize("key", [*_pg.LANE_MIGRATIONS, _REASON_CODES_MIGRATION])
def test_no_script_level_transaction_control(key):
```

(c) Add `_REASON_CODES_MIGRATION` to the `test_wrapped_files_never_mention_the_unwrap_triggers` list, and `"rollback_043.sql"` to the `test_rollbacks_are_never_auto_applied_and_hold_no_transaction_control` list.

(d) Append two tests that pin 043's two invariants where CI can see them (the real-DB tests never run in CI):

```python
def _select_expressions(sql: str, table: str) -> tuple:
    """The INSERT column list and the SELECT expressions feeding it, split at top-level commas."""
    head = sql.split(f"INSERT INTO {table} (", 1)[1]
    columns = [c.strip() for c in head.split(")", 1)[0].replace("\n", " ").split(",")]
    body = head.split("SELECT", 1)[1].split("FROM jsonb_array_elements(p_steps)", 1)[0]
    parts, depth, quoted, current = [], 0, False, []
    for ch in body:
        if ch == "'":
            quoted = not quoted
        elif not quoted and ch == "(":
            depth += 1
        elif not quoted and ch == ")":
            depth -= 1
        if ch == "," and depth == 0 and not quoted:
            parts.append("".join(current).strip())
            current = []
        else:
            current.append(ch)
    parts.append("".join(current).strip())
    return columns, [re.sub(r"--[^\n]*", "", p).strip() for p in parts]


def test_043_still_writes_no_text_into_error_message():
    """D1′: ml/041's NULL in the error_message slot is the database's second guard. 043 keeps it."""
    path = ML / _REASON_CODES_MIGRATION
    if not path.exists():
        pytest.fail(f"{_REASON_CODES_MIGRATION} is missing")
    columns, expressions = _select_expressions(path.read_text(), "composition_steps")
    assert len(columns) == len(expressions), (columns, expressions)
    assert expressions[columns.index("error_message")] == "NULL"
    assert "reason_code" in columns and "reason_details" in columns


def test_every_reason_code_passes_the_043_format_guard():
    """The SQL guard is a format, not a member list; every Python member must satisfy it.

    Read by AST, not imported: importing the tool_composer package costs ~564 MB here.
    """
    path = ML / _REASON_CODES_MIGRATION
    if not path.exists():
        pytest.fail(f"{_REASON_CODES_MIGRATION} is missing")
    guard = re.search(r"reason_code ~ '([^']+)'", path.read_text()).group(1)
    source = (_pg.REPO_ROOT / "src/agents/tool_composer/reason_codes.py").read_text()
    enum = next(
        n for n in ast.walk(ast.parse(source))
        if isinstance(n, ast.ClassDef) and n.name == "ReasonCode"
    )
    values = [
        n.value.value for n in enum.body
        if isinstance(n, ast.Assign) and isinstance(n.value, ast.Constant)
    ]
    assert len(values) >= 30
    assert [v for v in values if not re.fullmatch(guard, v)] == []
```

Add `import ast` to the file's imports.

- [ ] **Step 2: Run to verify they fail for the right reason**

```bash
cd /home/enunez/Projects/e2i_causal_analytics/.worktrees/lane-refusal-codes && \
  /home/enunez/Projects/e2i_causal_analytics/.venv/bin/pytest \
  tests/unit/test_database/learning_loop/test_lane_migration_files.py \
  -n 0 -p no:cacheprovider -q --timeout=300 -rf > /tmp/t6_red.log 2>&1; echo "EXIT=$?"; grep -E "^FAILED" /tmp/t6_red.log
```

Expected: every 043 and `rollback_043` parameter fails with `… is missing` or an `assert path.exists()`, and the two new tests fail with `… is missing`. Any `FileNotFoundError`, `IndexError` or `AttributeError` is a broken test, not the red.

- [ ] **Step 3: Write `database/ml/043_composer_refusal_reason_codes.sql`**

Sections 1, 2, 4 and 5 are given in full. Section 3 is a COPY of ml/041's function with four marked edits, and Step 4 proves the copy is otherwise exact.

```sql
-- ============================================================================
-- ml/043 - refusal reason codes on the recording path (#2021, #2050)
--
-- WHY. A refusal's reason was not persisted. ml/041's composer_record_steps recorded the
-- outcome class and exception type, but nothing a reader could aggregate "why tools fail" by.
-- 043 records the CLOSED reason code (src/agents/tool_composer/reason_codes.py) and its
-- numeric details, and mirrors the code into tool_performance so get_tool_reliability can
-- report the most common refusal reason per tool.
--
-- WHAT IS STORED - AND WHAT IS NOT. Structure only, as ml/041 established. reason_code must
-- match the snake_case format; reason_details keeps at most 8 snake_case keys whose values are
-- JSON numbers or booleans, and drops everything else. NO TEXT: the error_message slot of the
-- step INSERT stays NULL - that NULL is ml/041's guard against storing caller text, and three
-- tests pin it. The human sentence for a code is rendered at READ time from the Python
-- catalogue (owner decision D1', 2026-09-12), so adding a code never needs a migration.
--
-- Applied by scripts/run_migrations.sh inside one transaction with its ledger row
-- (key: ml/043_composer_refusal_reason_codes.sql). Re-applying it changes nothing.
-- ============================================================================

-- ---------------------------------------------------------------------------
-- 1. Columns
-- ---------------------------------------------------------------------------
ALTER TABLE composition_steps
    ADD COLUMN IF NOT EXISTS reason_code text,
    ADD COLUMN IF NOT EXISTS reason_details jsonb NOT NULL DEFAULT '{}'::jsonb;
ALTER TABLE tool_performance
    ADD COLUMN IF NOT EXISTS reason_code text;

-- A format guard, like ml/041's error_type guard. The closed member list lives in Python; a
-- copy here would drift the moment a member is added (test_lane_migration_files checks every
-- member passes this format).
ALTER TABLE composition_steps DROP CONSTRAINT IF EXISTS composition_steps_reason_code_format;
ALTER TABLE composition_steps ADD CONSTRAINT composition_steps_reason_code_format
    CHECK (reason_code IS NULL OR reason_code ~ '^[a-z][a-z0-9_]{0,63}$');
ALTER TABLE composition_steps DROP CONSTRAINT IF EXISTS composition_steps_reason_details_object;
ALTER TABLE composition_steps ADD CONSTRAINT composition_steps_reason_details_object
    CHECK (jsonb_typeof(reason_details) = 'object');
ALTER TABLE tool_performance DROP CONSTRAINT IF EXISTS tool_performance_reason_code_format;
ALTER TABLE tool_performance ADD CONSTRAINT tool_performance_reason_code_format
    CHECK (reason_code IS NULL OR reason_code ~ '^[a-z][a-z0-9_]{0,63}$');

COMMENT ON COLUMN composition_steps.reason_code IS
    'Closed-set code for why the step produced no result (#2021). Rendered to a sentence at read time; no message is stored.';
COMMENT ON COLUMN composition_steps.reason_details IS
    'Numbers and booleans only, at most 8 snake_case keys, reduced by composer_structure_reason_details.';
COMMENT ON COLUMN tool_performance.reason_code IS
    'Mirror of composition_steps.reason_code, for get_tool_reliability.most_common_refusal_reason.';

-- ---------------------------------------------------------------------------
-- 2. The details reducer (same shape as ml/041's composer_structure_numbers)
-- ---------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION composer_structure_reason_details(p_value jsonb)
RETURNS jsonb
LANGUAGE sql
IMMUTABLE
SECURITY INVOKER
SET search_path = public
AS $fn$
    SELECT COALESCE(
        (SELECT jsonb_object_agg(k.key, k.value)
         FROM (SELECT e.key, e.value
               FROM jsonb_each(CASE WHEN jsonb_typeof(p_value) = 'object'
                                    THEN p_value ELSE '{}'::jsonb END) AS e
               WHERE e.key ~ '^[a-z][a-z0-9_]{0,63}$'
                 AND jsonb_typeof(e.value) IN ('number', 'boolean')
               ORDER BY e.key
               LIMIT 8) AS k),
        '{}'::jsonb);
$fn$;

-- ---------------------------------------------------------------------------
-- 3. composer_record_steps: carry the code and the details. The error_message slot stays NULL.
--    Copied from ml/041 with four edits, each marked "-- 043". CREATE OR REPLACE with an
--    unchanged signature keeps the function's grants; section 5 re-asserts them anyway.
-- ---------------------------------------------------------------------------
<COPY HERE: ml/041's CREATE OR REPLACE FUNCTION composer_record_steps(p_seed jsonb, p_steps jsonb)
 through its closing $fn$; — with exactly edits E1-E4 below>

-- ---------------------------------------------------------------------------
-- 4. get_tool_reliability: add most_common_refusal_reason. The return type changes, so the view
--    that selects from it and the function itself are dropped and recreated, and the grants and
--    view comment that DROP discards are re-applied in section 5.
-- ---------------------------------------------------------------------------
DROP VIEW IF EXISTS v_tool_reliability;
DROP FUNCTION IF EXISTS get_tool_reliability(integer, boolean);

<COPY HERE: ml/041's CREATE OR REPLACE FUNCTION get_tool_reliability(...)
 through its closing $fn$; — with exactly edits R1-R3 below>

CREATE VIEW v_tool_reliability AS
SELECT * FROM get_tool_reliability(30, true);

COMMENT ON VIEW v_tool_reliability IS
    'Per-tool measured reliability over 30 days, synthetic rows included (get_tool_reliability(30, true)).';

-- ---------------------------------------------------------------------------
-- 5. Access: service_role only, as ml/041 section 6
-- ---------------------------------------------------------------------------
REVOKE ALL ON FUNCTION composer_structure_reason_details(jsonb) FROM PUBLIC, anon, authenticated;
REVOKE ALL ON FUNCTION composer_record_steps(jsonb, jsonb) FROM PUBLIC, anon, authenticated;
REVOKE ALL ON FUNCTION get_tool_reliability(integer, boolean) FROM PUBLIC, anon, authenticated;

GRANT EXECUTE ON FUNCTION
    composer_structure_reason_details(jsonb),
    composer_record_steps(jsonb, jsonb),
    get_tool_reliability(integer, boolean)
TO service_role;

REVOKE ALL ON v_tool_reliability FROM PUBLIC, anon, authenticated;
GRANT SELECT ON v_tool_reliability TO service_role;
```

The `<COPY HERE: …>` lines are instructions to you, not SQL. Replace each with the copied function.

**Edits to the copied `composer_record_steps`:**

- **E1** — in the `INSERT INTO composition_steps (` column list, change the last line `error_message, retry_count, outcome_class, attempts, cache_hit, error_type` to end `…, cache_hit, error_type, reason_code, reason_details  -- 043`.
- **E2** — the SELECT's last expression is the `CASE WHEN (s->>'error_type') ~ … END`. Add a comma after its `END`, then append:
  ```sql
              CASE WHEN (s->>'reason_code') ~ '^[a-z][a-z0-9_]{0,63}$' THEN s->>'reason_code' END,  -- 043
              composer_structure_reason_details(s->'reason_details')  -- 043
  ```
- **E3** — leave the `NULL,` in the `error_message` slot unchanged, and put this line directly above it: `            -- error_message: NULL by design (ml/041's guard; D1' sends no text)  -- 043`.
- **E4** — in the `INSERT INTO tool_performance (` column list append `, reason_code` after `executed_at`, and in its SELECT append `, cs.reason_code` after `COALESCE(cs.completed_at, now())`. Mark both lines `-- 043`.

**Edits to the copied `get_tool_reliability`:**

- **R1** — in `RETURNS TABLE (`, change `most_common_health_error text` to `most_common_health_error text,` and add the line `    most_common_refusal_reason text  -- 043`.
- **R2** — in the `perf` CTE, change `tp.error_type, tp.is_synthetic,` to `tp.error_type, tp.reason_code, tp.is_synthetic,  -- 043`.
- **R3** — after the `most_common_health_error` expression (`(mode() … FILTER (WHERE p.counted AND p.outcome_class IN ('timeout', 'error')))::text`), add:
  ```sql
          ,(mode() WITHIN GROUP (ORDER BY p.reason_code)  -- 043
              FILTER (WHERE p.counted AND p.outcome_class IN ('refused', 'input_rejected')))::text
  ```

- [ ] **Step 4: Prove each copy differs from ml/041 by exactly its edits**

```bash
cd /home/enunez/Projects/e2i_causal_analytics/.worktrees/lane-refusal-codes && \
fn() { awk -v n="$2" '$0 ~ "CREATE OR REPLACE FUNCTION "n"\\(" {f=1} f {print} f && /^\$fn\$;/ {exit}' "$1"; } && \
for f in composer_record_steps get_tool_reliability; do
  echo "=== $f: 041 -> 043 ==="
  diff <(fn database/ml/041_composer_learning_loop_recording.sql $f) <(fn database/ml/043_composer_refusal_reason_codes.sql $f)
done
```

Expected: `composer_record_steps` shows only the E1–E4 lines and `get_tool_reliability` only R1–R3. Every changed line on the 043 side carries `-- 043`, except the SQL continuation lines inside E2 and R3, and the `most_common_health_error text,` comma. Any other difference means the copy was edited: restore it from ml/041.

- [ ] **Step 5: Write `database/ml/rollback_043.sql`**

Follow `rollback_041.sql`'s shape.

```sql
-- ============================================================================
-- E2I Causal Analytics - ROLLBACK for ml/043_composer_refusal_reason_codes.sql
-- NOT a forward migration: scripts/run_migrations.sh skips rollback_*.sql. Apply by hand,
-- AFTER the code revert:
--
--   docker exec -i supabase-db psql -U postgres -d postgres -v ON_ERROR_STOP=1 \
--       --single-transaction < database/ml/rollback_043.sql
--
-- Returns composer_record_steps and get_tool_reliability (and v_tool_reliability) to their
-- ml/041 definitions, copied verbatim, re-applies ml/041's grants for them, and drops every
-- object ml/043 created. Reason codes and details recorded since 043 are lost. Deletes the
-- ml/043 ledger row, so re-deploying the code re-applies 043. Idempotent: every drop is
-- IF EXISTS and every recreation replaces, so a second run changes nothing.
-- ============================================================================

DELETE FROM public.schema_migrations WHERE filename = 'ml/043_composer_refusal_reason_codes.sql';

DROP VIEW IF EXISTS v_tool_reliability;
DROP FUNCTION IF EXISTS get_tool_reliability(integer, boolean);
```

Then, in this order:
1. ml/041's `CREATE OR REPLACE FUNCTION get_tool_reliability(…) … $fn$;` copied **verbatim**, followed by its `CREATE VIEW v_tool_reliability …` and `COMMENT ON VIEW v_tool_reliability …`, both copied from 041.
2. ml/041's `CREATE OR REPLACE FUNCTION composer_record_steps(…) … $fn$;` copied **verbatim**.
3. Then:

```sql
DROP FUNCTION IF EXISTS composer_structure_reason_details(jsonb);

ALTER TABLE tool_performance DROP CONSTRAINT IF EXISTS tool_performance_reason_code_format;
ALTER TABLE tool_performance DROP COLUMN IF EXISTS reason_code;

ALTER TABLE composition_steps DROP CONSTRAINT IF EXISTS composition_steps_reason_details_object;
ALTER TABLE composition_steps DROP CONSTRAINT IF EXISTS composition_steps_reason_code_format;
ALTER TABLE composition_steps
    DROP COLUMN IF EXISTS reason_details,
    DROP COLUMN IF EXISTS reason_code;

REVOKE ALL ON FUNCTION composer_record_steps(jsonb, jsonb) FROM PUBLIC, anon, authenticated;
REVOKE ALL ON FUNCTION get_tool_reliability(integer, boolean) FROM PUBLIC, anon, authenticated;
GRANT EXECUTE ON FUNCTION
    composer_record_steps(jsonb, jsonb),
    get_tool_reliability(integer, boolean)
TO service_role;
REVOKE ALL ON v_tool_reliability FROM PUBLIC, anon, authenticated;
GRANT SELECT ON v_tool_reliability TO service_role;
```

Prove the verbatim copies with the Step 4 `fn` helper: `diff <(fn …041… $f) <(fn database/ml/rollback_043.sql $f)` must print NOTHING for both functions.

- [ ] **Step 6: Run the static tests green**

```bash
cd /home/enunez/Projects/e2i_causal_analytics/.worktrees/lane-refusal-codes && \
  /home/enunez/Projects/e2i_causal_analytics/.venv/bin/pytest \
  tests/unit/test_database/learning_loop/test_lane_migration_files.py \
  -n 0 -p no:cacheprovider -q --timeout=300 > /tmp/t6_green.log 2>&1; echo "EXIT=$?"; tail -3 /tmp/t6_green.log
```

Expected: `EXIT=0`, no skips.

- [ ] **Step 7: Rehearse on a throwaway prod clone (D4) — the evidence**

Preconditions. The throwaway container is capped at 1 GiB, the base restore takes minutes, and this box runs production:

```bash
free -m | awk '/Mem/ {print "MemAvailable MB:", $7}'   # stop and report if below 2500
docker ps --filter name=e2i-learnloop-pg- --format '{{.Names}} {{.Status}}'   # another session's? leave it
mkdir -p /home/enunez/Projects/e2i_causal_analytics/.worktrees/lane-refusal-codes/docs/demos/results/2026-09-12_lane_refusal_codes_cert
```

Create `docs/demos/results/2026-09-12_lane_refusal_codes_cert/rehearse_043.py`:

```python
"""D4 rehearsal: ml/043 on a throwaway Postgres cloned from prod's image and schema.

Never writes supabase-db (ProdReadOnly runs only approved read statements). Calls the RPCs as
service_role through the same port the real-DB tests use, so a missing GRANT fails here and not
in prod. One line per check; exits non-zero at the first failure, so the transcript IS the
evidence.
"""

import asyncio
import json
from datetime import datetime, timedelta, timezone

from tests.unit.test_database.learning_loop import _pg

M043 = _pg.REPO_ROOT / "database/ml/043_composer_refusal_reason_codes.sql"
SENTINEL = "PT-SENTINEL-043"
MD5_STEPS = "select md5(prosrc) from pg_proc where proname = 'composer_record_steps'"
MD5_RELIABILITY = (
    "select md5(prosrc || pg_get_function_result(oid)) from pg_proc "
    "where proname = 'get_tool_reliability'"
)


def check(label, ok, detail=""):
    print(("PASS " if ok else "FAIL ") + label + (f" -- {detail}" if detail else ""), flush=True)
    if not ok:
        raise SystemExit(1)


def seed(cid):
    return {
        "composition_id": cid, "query_text": "rehearsal", "session_id": f"sess-{cid}",
        "user_id": "user-1", "entry_point": "chat_tool", "brand": "Kisqali", "region": "US",
        "audit_workflow_id": None, "is_synthetic": True,
    }


def step(n, tool, cls, **over):
    now = datetime.now(timezone.utc)
    return {
        "step_number": n, "tool_name": tool, "input_params": {},
        "output_keys": {"keys": [], "other_keys": 0}, "depends_on_steps": [],
        "serves_sub_question": "0",
        "started_at": (now - timedelta(seconds=2)).isoformat(),
        "completed_at": (now - timedelta(seconds=1)).isoformat(),
        "latency_ms": 120.0, "outcome_class": cls, "attempts": 1, "cache_hit": False,
        "error_type": None if cls == "succeeded" else "ToolRefusalError",
        **over,
    }


def record(conn, cid, steps):
    port = _pg.PsycopgRpcPort(conn, role="service_role")
    return asyncio.run(port.call("composer_record_steps", {"p_seed": seed(cid), "p_steps": steps}))


def rehearse(conn):
    before_steps, before_rel = conn.rows(MD5_STEPS), conn.rows(MD5_RELIABILITY)

    check("apply 043 (wrapped)", _pg.apply_migration(conn, M043) == "wrapped")
    _pg.apply_migration(conn, M043)
    check("re-apply 043 is a no-op", True)

    cols = conn.rows(
        "select table_name || '.' || column_name || ':' || data_type || ':' || is_nullable "
        "from information_schema.columns where table_schema = 'public' "
        "and column_name in ('reason_code', 'reason_details') order by 1"
    )
    check("columns", cols == [
        "composition_steps.reason_code:text:YES",
        "composition_steps.reason_details:jsonb:NO",
        "tool_performance.reason_code:text:YES",
    ], str(cols))

    for fn in ("composer_structure_reason_details(jsonb)", "composer_record_steps(jsonb,jsonb)",
               "get_tool_reliability(integer,boolean)"):
        got = conn.rows(
            f"select has_function_privilege('anon', 'public.{fn}', 'EXECUTE') || '|' "
            f"|| has_function_privilege('service_role', 'public.{fn}', 'EXECUTE')"
        )
        check(f"grants {fn}", got == ["false|true"], str(got))
    got = conn.rows(
        "select has_table_privilege('anon', 'public.v_tool_reliability', 'SELECT') || '|' "
        "|| has_table_privilege('service_role', 'public.v_tool_reliability', 'SELECT')"
    )
    check("grants v_tool_reliability", got == ["false|true"], str(got))

    record(conn, "rehearsal_a", [
        step(0, "gap_calculator", "refused", reason_code="coverage_gap",
             reason_details={"n_groups": 1, "label": SENTINEL, "Bad-Key": 3, "flag": True},
             error_message=SENTINEL),
        step(1, "causal_effect_estimator", "refused", reason_code="non_binary_treatment",
             reason_details={"n_distinct": 4}),
        step(2, "causal_effect_estimator", "refused", reason_code="DROP TABLE x; --"),
        step(3, "causal_effect_estimator", "succeeded"),
    ])
    rows = conn.rows(
        "select s.step_number || '|' || coalesce(s.reason_code, '-') || '|' || s.reason_details::text "
        "|| '|' || coalesce(s.error_message, '-') from composition_steps s "
        "join composer_episodes e using (episode_id) where e.composition_id = 'rehearsal_a' "
        "order by s.step_number"
    )
    check("persisted rows", rows == [
        '0|coverage_gap|{"flag": true, "n_groups": 1}|-',
        '1|non_binary_treatment|{"n_distinct": 4}|-',
        "2|-|{}|-",
        "3|-|{}|-",
    ], json.dumps(rows))

    perf = conn.rows(
        "select outcome_class || '|' || coalesce(reason_code, '-') from tool_performance "
        "where composition_id = 'rehearsal_a' order by 1"
    )
    check("tool_performance mirror", sorted(perf) == sorted([
        "refused|coverage_gap", "refused|non_binary_treatment", "refused|-", "succeeded|-",
    ]), str(perf))

    # mode() must ignore NULL codes: 3 uncoded refusals (the shape of every pre-043 row) against
    # 2 coded ones must still report the code, not NULL.
    record(conn, "rehearsal_b", [
        step(0, "causal_effect_estimator", "refused", reason_code="non_binary_treatment"),
        step(1, "causal_effect_estimator", "refused"),
        step(2, "causal_effect_estimator", "refused"),
    ])
    port = _pg.PsycopgRpcPort(conn, role="service_role")
    reliability = asyncio.run(
        port.call("get_tool_reliability", {"p_days": 30, "p_include_synthetic": True})
    )
    cee = [r for r in reliability if r.get("tool_name") == "causal_effect_estimator"]
    check("reliability reports the code despite NULLs",
          len(cee) == 1 and cee[0].get("most_common_refusal_reason") == "non_binary_treatment",
          json.dumps(cee, default=str))
    view = conn.rows(
        "select coalesce(most_common_refusal_reason, '-') from v_tool_reliability "
        "where tool_name = 'causal_effect_estimator'"
    )
    check("view exposes the column", view == ["non_binary_treatment"], str(view))

    blob = "\n".join(conn.rows(
        "select row_to_json(x)::text from ("
        " select to_jsonb(s) as r from composition_steps s join composer_episodes e using (episode_id)"
        "  where e.composition_id like 'rehearsal_%'"
        " union all select to_jsonb(p) from tool_performance p where composition_id like 'rehearsal_%'"
        " union all select to_jsonb(e) from composer_episodes e where composition_id like 'rehearsal_%'"
        ") x"
    ))
    check("sentinel stored nowhere", SENTINEL not in blob)

    for attempt in ("first", "second"):
        proc = _pg.apply_rollback(conn, "rollback_043.sql")
        check(f"rollback_043 {attempt} run", proc.returncode == 0, proc.stderr.decode()[-500:])
        check(f"rollback_043 {attempt}: record_steps body is ml/041's", conn.rows(MD5_STEPS) == before_steps)
        check(f"rollback_043 {attempt}: reliability is ml/041's", conn.rows(MD5_RELIABILITY) == before_rel)
    left = conn.rows(
        "select count(*) from information_schema.columns where table_schema = 'public' "
        "and column_name in ('reason_code', 'reason_details')"
    )
    check("rollback removed the columns", left == ["0"], str(left))
    got = conn.rows(
        "select has_function_privilege('anon', 'public.get_tool_reliability(integer,boolean)', 'EXECUTE') "
        "|| '|' || has_function_privilege('service_role', 'public.get_tool_reliability(integer,boolean)', 'EXECUTE')"
    )
    check("rollback restored grants", got == ["false|true"], str(got))
    gone = conn.rows("select count(*) from pg_proc where proname = 'composer_structure_reason_details'")
    check("rollback dropped the reducer", gone == ["0"], str(gone))

    check("re-apply 043 after rollback", _pg.apply_migration(conn, M043) == "wrapped")


def main():
    _pg.reap_orphans()
    prod = _pg.ProdReadOnly()
    pg = _pg.ThrowawayPg(image=prod.image())
    pg.start()
    try:
        log = _pg.build_base(pg, prod)
        check("base restored with no unexpected errors", not log.unexpected, str(log.unexpected[:5]))
        conn = _pg.clone(pg, "rehearsal_043")
        try:
            rehearse(conn)
        finally:
            _pg.drop(conn)
    finally:
        pg.stop()
    print("REHEARSAL OK", flush=True)


if __name__ == "__main__":
    main()
```

If a `_pg` call's real signature differs from what this script assumes — `PsycopgRpcPort.call`, `PgConn.rows`, `apply_rollback` — read `tests/unit/test_database/learning_loop/_pg.py`, adapt the call, and say so in your report. Never weaken a `check`.

Run it with the worktree as cwd, capturing the SCRIPT's exit code rather than `tee`'s:

```bash
cd /home/enunez/Projects/e2i_causal_analytics/.worktrees/lane-refusal-codes && \
  C=docs/demos/results/2026-09-12_lane_refusal_codes_cert && \
  /home/enunez/Projects/e2i_causal_analytics/.venv/bin/python $C/rehearse_043.py 2>&1 | tee $C/rehearsal_043_transcript.txt; \
  echo "REHEARSAL_EXIT=${PIPESTATUS[0]}" | tee -a $C/rehearsal_043_transcript.txt
```

Expected: every line `PASS`, then `REHEARSAL OK` and `REHEARSAL_EXIT=0`. A `FAIL` line is a real finding. Stop and report it with the transcript; do not edit a `check` to pass.

- [ ] **Step 8: Commit** (the cert directory stays untracked)

```bash
cd /home/enunez/Projects/e2i_causal_analytics/.worktrees/lane-refusal-codes && git branch --show-current && \
  git add database/ml/043_composer_refusal_reason_codes.sql database/ml/rollback_043.sql \
          tests/unit/test_database/learning_loop/test_lane_migration_files.py && \
  git commit -m "feat(db): ml/043 records refusal reason codes as structure (#2050 cause 2, #2021)

ml/041's composer_record_steps kept the outcome class and exception type but nothing a reader
could aggregate 'why tools fail' by. 043 carries the closed reason_code and numeric-only
reason_details into composition_steps, mirrors the code into tool_performance, and recreates
get_tool_reliability with most_common_refusal_reason, re-applying the grants DROP discards.
The error_message slot stays NULL: that is ml/041's guard against storing caller text, and the
sentence is rendered at read time (owner decision D1'). Proven by a rehearsal on a throwaway
prod clone (D4); static checks for 043 and rollback_043 run in CI.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01SnzgDeMLxZN48UsJXTaazb"
```

**Known consequence, not fixed here (D4: harness untouched).** `test_migration_runner.py` (opt-in, real DB) asserts the runner applies exactly three lane migrations. With 043 in `database/ml/`, a replay on a pre-lane base would apply four. That module already skips on this droplet and never runs in CI. Say so in the PR body.

---

## Task 7: Render the reason on the admin observability page (#2021 D3, D1′)

> **Read-side notes from the Task 3 quality review (2026-09-12).**
> - Episodes recorded before ml/043 have no `reason_code` and `reason_details`. Read both with `.get()`, and render a null code as not recorded (`reason: null`), never as a failure or as the tool-failure sentence. That is why this task uses `known_sentence`, not `canonical_sentence`.
> - A `refused` or `input_rejected` step carrying `tool_error` is not a contradiction. It means the tool raised with a code outside the closed set, and the `_CodedError` constructor failed soft to `TOOL_ERROR`. Render it as-is, and do not "correct" it on the page.

> **Rewritten 2026-09-12 for D1′, before dispatch.** The database now holds codes, not sentences. This task turns codes into sentences at read time, from the single catalogue, on the one surface an operator reads.

**Files:**
- Modify: `src/agents/tool_composer/reason_codes.py` — add `known_sentence()`
- Modify: `src/agents/tool_composer/reliability.py` — `ToolReliability.most_common_refusal_reason`
- Modify: `src/api/schemas/admin_tool_composer.py` — two `ToolReliabilityRow` fields; `RecentFailure.step_classes` description
- Modify: `src/services/tool_composer_observability_service.py` — `_STEP_COLUMNS`, the tool rows, the step classes
- Regenerate: `frontend/src/types/generated/api.ts` — never hand-edited
- Test: `tests/unit/test_agents/test_tool_composer/test_reason_codes_2021.py` (extend), `tests/unit/test_agents/test_tool_composer/test_reliability_refusal_reason_2021.py` (new), `tests/unit/test_services/test_tool_composer_observability_service.py` (extend)

**Two constraints, both measured:**

1. **Import weight.** Measured 2026-09-12 on this box: `import src.agents.tool_composer.reason_codes` costs **564 MB RSS and 17.8 s**, because it executes the package `__init__` (composer, planner, synthesizer, every tool registration). `import src.services.tool_composer_observability_service` costs **47 MB and 0.75 s**, and does not load that package. `src.api.routes.admin` imports the service at module top, so a module-level import in the service would add about half a gigabyte and 17 seconds to importing the API. **Import `known_sentence` function-locally**, as `admin.py:76` already does for the reliability reader. A test pins it.
2. **An unknown code gets no sentence.** `canonical_sentence()` falls back to the generic tool-failure sentence. That is right for the user answer (#2020: fail closed) and wrong on an operator page, where it would relabel a refusal this build cannot read as a tool failure. The page uses `known_sentence()`, which returns `None`, and then shows the bare code.

The planning prompt's `reliability_line` is NOT changed. Refusals never enter the caveat rule, and D3 is the admin page only.

- [ ] **Step 0: Baseline — prove local type generation reproduces the committed file BEFORE any edit**

The export imports the whole FastAPI app. Check memory first.

```bash
free -m | awk '/Mem/ {print "MemAvailable MB:", $7}'   # stop and report if below 2500
cd /home/enunez/Projects/e2i_causal_analytics/.worktrees/lane-refusal-codes && git status --porcelain | grep -v '^??' ; echo "(nothing above = clean tracked tree)" && \
  /usr/bin/time -v /home/enunez/Projects/e2i_causal_analytics/.venv/bin/python -m scripts.export_openapi --output /tmp/lane_openapi_base.json 2>&1 | grep -E "written|Maximum resident" && \
  cd frontend && /home/enunez/Projects/e2i_causal_analytics/frontend/node_modules/.bin/openapi-typescript /tmp/lane_openapi_base.json -o /tmp/lane_api_base.ts && \
  cmp src/types/generated/api.ts /tmp/lane_api_base.ts && echo "BASELINE REPRODUCES THE COMMITTED api.ts"
```

The worktree has no `frontend/node_modules`. The main checkout's `openapi-typescript` is 7.10.1, the version pinned in the lane's lockfile that CI's `npm ci` installs, and the two lockfiles are identical. If `cmp` reports a difference, the drift predates this lane: STOP and report it, because every later diff of `api.ts` would be uninterpretable.

- [ ] **Step 1: Write the failing tests**

(a) Append to `tests/unit/test_agents/test_tool_composer/test_reason_codes_2021.py`, and add `known_sentence` to its import from `reason_codes`:

```python
def test_known_sentence_renders_only_codes_this_build_knows():
    """Read-side rendering: an unknown code must not be relabelled as a tool failure."""
    assert known_sentence("non_binary_treatment") == CANONICAL_SENTENCES[
        ReasonCode.NON_BINARY_TREATMENT
    ]
    assert known_sentence(ReasonCode.COVERAGE_GAP) == CANONICAL_SENTENCES[ReasonCode.COVERAGE_GAP]
    assert known_sentence(None) is None
    assert known_sentence("a_code_this_build_does_not_know") is None
```

(b) Create `tests/unit/test_agents/test_tool_composer/test_reliability_refusal_reason_2021.py`:

```python
"""#2021 D3: get_tool_reliability's most common refusal reason survives ToolReliability.from_row."""

from src.agents.tool_composer.reliability import ToolReliability

_COUNTS = {
    "n_invoked": 12,
    "n_succeeded": 4,
    "n_refused": 8,
    "n_health_failures": 0,
    "n_health": 4,
    "n_retried": 0,
    "n_synthetic": 0,
}


def test_from_row_carries_the_most_common_refusal_reason():
    row = ToolReliability.from_row(
        {"tool_name": "causal_effect_estimator", **_COUNTS,
         "most_common_refusal_reason": "non_binary_treatment"}
    )
    assert row.most_common_refusal_reason == "non_binary_treatment"


def test_a_row_from_a_database_without_ml_043_reads_as_none():
    row = ToolReliability.from_row({"tool_name": "gap_calculator", **_COUNTS})
    assert row.most_common_refusal_reason is None
```

(c) Append to `tests/unit/test_services/test_tool_composer_observability_service.py`:

```python
# ---------------------------------------------------------------------------
# #2021 D3 / D1′: codes come from the database; sentences are rendered here
# ---------------------------------------------------------------------------


def _verdict(**over: Any) -> Any:
    from src.agents.tool_composer.reliability import ToolReliability

    row = {
        "tool_name": "causal_effect_estimator",
        "n_invoked": 12, "n_succeeded": 4, "n_refused": 8, "n_health_failures": 0,
        "n_health": 4, "n_retried": 0, "n_synthetic": 0,
    }
    row.update(over)
    return ToolReliability.from_row(row)


def test_a_tool_row_carries_the_refusal_code_and_its_rendered_sentence():
    verdicts = {
        "causal_effect_estimator": _verdict(most_common_refusal_reason="non_binary_treatment")
    }
    service = _service({"composer_episodes": [], "composition_steps": []})
    (tool,) = service.overview(30, verdicts)["tools"]
    assert tool["most_common_refusal_reason"] == "non_binary_treatment"
    assert tool["most_common_refusal_sentence"] == (
        "the treatment column is not a binary 0/1 indicator"
    )


def test_an_unknown_code_shows_the_code_and_no_invented_sentence():
    verdicts = {
        "causal_effect_estimator": _verdict(
            most_common_refusal_reason="a_code_this_build_does_not_know"
        )
    }
    service = _service({"composer_episodes": [], "composition_steps": []})
    (tool,) = service.overview(30, verdicts)["tools"]
    assert tool["most_common_refusal_reason"] == "a_code_this_build_does_not_know"
    assert tool["most_common_refusal_sentence"] is None


def test_a_failed_step_class_carries_its_code_and_rendered_reason():
    failed = _episode(status="FAILED", outcome="failed")
    steps = [
        {
            "episode_id": failed["episode_id"],
            "step_number": 0,
            "tool_name": "gap_calculator",
            "outcome_class": "refused",
            "reason_code": "coverage_gap",
        }
    ]
    service = _service({"composer_episodes": [failed], "composition_steps": steps})
    (row,) = service.overview(30)["recent_failures"]
    assert row["step_classes"] == [
        {
            "step_number": 0,
            "tool_name": "gap_calculator",
            "outcome_class": "refused",
            "reason_code": "coverage_gap",
            "reason": "the data does not cover everything the question asked about",
        }
    ]


def test_the_service_module_does_not_import_the_tool_composer_package():
    """Measured 2026-09-12: the package costs ~564 MB and ~17 s; this module alone ~47 MB."""
    import subprocess
    import sys
    from pathlib import Path

    code = (
        "import sys, src.services.tool_composer_observability_service; "
        "print('src.agents.tool_composer' in sys.modules)"
    )
    out = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True, text=True, check=True,
        cwd=Path(__file__).resolve().parents[3],
    )
    assert out.stdout.strip() == "False"
```

`FakeQuery` honours the `select` projection, so until `_STEP_COLUMNS` names `reason_code` the step row arrives without it. The step-class test is therefore red for the right reason. If an existing test in this file pins a tool row or step class by exact equality, update it deliberately for the new keys and say so.

- [ ] **Step 2: Run to verify the right tests fail**

```bash
cd /home/enunez/Projects/e2i_causal_analytics/.worktrees/lane-refusal-codes && \
  /home/enunez/Projects/e2i_causal_analytics/.venv/bin/pytest \
  tests/unit/test_agents/test_tool_composer/test_reason_codes_2021.py \
  tests/unit/test_agents/test_tool_composer/test_reliability_refusal_reason_2021.py \
  tests/unit/test_services/test_tool_composer_observability_service.py \
  -n 0 -p no:cacheprovider -q --timeout=300 -rf > /tmp/t7_red.log 2>&1; echo "EXIT=$?"; grep -E "^FAILED|^ERROR" /tmp/t7_red.log
```

Expected: `known_sentence` import error; `AttributeError` for `most_common_refusal_reason`; `KeyError` or equality failures for the service tests. `test_the_service_module_does_not_import_the_tool_composer_package` PASSES now — it is a guard that must stay green through Step 3.

- [ ] **Step 3: Implement**

`src/agents/tool_composer/reason_codes.py` — add, below `canonical_sentence`, and add `Optional` to the `typing` import:

```python
def known_sentence(code: Union[ReasonCode, str, None]) -> Optional[str]:
    """The sentence for a code this build knows, else ``None``. For read-side display.

    Unlike :func:`canonical_sentence`, an unknown code does NOT fall back to the generic
    tool-failure sentence: an operator page must not relabel a refusal it cannot read as a tool
    failure. The caller shows the bare code instead.
    """
    if isinstance(code, ReasonCode):
        return CANONICAL_SENTENCES[code]
    if isinstance(code, str):
        try:
            return CANONICAL_SENTENCES[ReasonCode(code)]
        except ValueError:
            return None
    return None
```

`src/agents/tool_composer/reliability.py` — after `most_common_health_error: Optional[str] = None`:

```python
    # #2021: the most common CLOSED code among this tool's refusals in the window. Its sibling
    # above names health failures by exception class; this names declines-to-answer by
    # category. A code, never a sentence: rendering happens at read time (D1′).
    most_common_refusal_reason: Optional[str] = None
```

and in `from_row`, beside `most_common_health_error=row.get("most_common_health_error"),`:

```python
            most_common_refusal_reason=row.get("most_common_refusal_reason"),
```

`row.get` is why a database without ml/043 still serves the page.

`src/api/schemas/admin_tool_composer.py` — on `ToolReliabilityRow`, after `most_common_health_error`:

```python
    most_common_refusal_reason: Optional[str] = Field(
        default=None,
        description="Most common closed reason code among this tool's refusals in the window (#2021)",
    )
    most_common_refusal_sentence: Optional[str] = Field(
        default=None,
        description="That code's catalogue sentence, rendered at read time; null for a code this build does not know",
    )
```

and on `RecentFailure.step_classes`, change only the description to `"The steps that did not succeed, with their classes, reason codes and rendered reasons"`.

`src/services/tool_composer_observability_service.py`:

```python
_STEP_COLUMNS = "episode_id, step_number, tool_name, outcome_class, reason_code"
```

In the method that builds tool rows from `verdicts` (it contains `"most_common_health_error": tool.most_common_health_error,`), add as its first statement:

```python
        # Function-local: src.api.routes.admin imports this module at API start, and the
        # tool_composer package's __init__ imports the composer, planner and every tool
        # registration (measured 2026-09-12: ~564 MB / ~17 s, against ~47 MB for this module).
        from src.agents.tool_composer.reason_codes import known_sentence
```

and beside the health-error key:

```python
                    "most_common_refusal_reason": tool.most_common_refusal_reason,
                    "most_common_refusal_sentence": known_sentence(tool.most_common_refusal_reason),
```

In `_recent_failures`, add the same function-local import as its first statement, and make each step class:

```python
                        {
                            "step_number": s.get("step_number"),
                            "tool_name": s.get("tool_name"),
                            "outcome_class": s.get("outcome_class"),
                            "reason_code": s.get("reason_code"),
                            "reason": known_sentence(s.get("reason_code")),
                        }
```

- [ ] **Step 4: Run to verify it passes**

```bash
cd /home/enunez/Projects/e2i_causal_analytics/.worktrees/lane-refusal-codes && \
  /home/enunez/Projects/e2i_causal_analytics/.venv/bin/pytest \
  tests/unit/test_agents/test_tool_composer/test_reason_codes_2021.py \
  tests/unit/test_agents/test_tool_composer/test_reliability_refusal_reason_2021.py \
  tests/unit/test_agents/test_tool_composer/test_reliability_rule.py \
  tests/unit/test_agents/test_tool_composer/test_planner_reliability_flag.py \
  tests/unit/test_services/test_tool_composer_observability_service.py \
  tests/unit/test_api/test_routes/test_admin_tool_composer.py \
  -n 0 -p no:cacheprovider -q --timeout=600 > /tmp/t7_green.log 2>&1; echo "EXIT=$?"; tail -3 /tmp/t7_green.log
```

Expected: `EXIT=0`.

- [ ] **Step 5: Regenerate the committed types, exactly as CI does**

```bash
free -m | awk '/Mem/ {print "MemAvailable MB:", $7}'   # stop and report if below 2500
cd /home/enunez/Projects/e2i_causal_analytics/.worktrees/lane-refusal-codes && \
  /home/enunez/Projects/e2i_causal_analytics/.venv/bin/python -m scripts.export_openapi --output /tmp/lane_openapi.json && \
  cd frontend && \
  /home/enunez/Projects/e2i_causal_analytics/frontend/node_modules/.bin/openapi-typescript /tmp/lane_openapi.json -o src/types/generated/api.ts && \
  /home/enunez/Projects/e2i_causal_analytics/frontend/node_modules/.bin/tsc --noEmit --strict --skipLibCheck src/types/generated/api.ts && \
  cd .. && git diff --stat -- frontend/ && git diff -- frontend/src/types/generated/api.ts
```

Expected: only `api.ts` changes, and its diff shows only `most_common_refusal_reason`, `most_common_refusal_sentence` and the `step_classes` description. Anything else (a renumbered schema name, an unrelated type) contradicts the Step 0 baseline: regenerate once and compare. If it persists, stop and report rather than committing churn.

- [ ] **Step 6: Commit**

```bash
cd /home/enunez/Projects/e2i_causal_analytics/.worktrees/lane-refusal-codes && git branch --show-current && \
  /home/enunez/Projects/e2i_causal_analytics/.venv/bin/ruff check --no-cache src/ tests/unit/test_services/ tests/unit/test_agents/test_tool_composer/ && \
  git add src/agents/tool_composer/reason_codes.py src/agents/tool_composer/reliability.py \
          src/api/schemas/admin_tool_composer.py src/services/tool_composer_observability_service.py \
          frontend/src/types/generated/api.ts \
          tests/unit/test_agents/test_tool_composer/test_reason_codes_2021.py \
          tests/unit/test_agents/test_tool_composer/test_reliability_refusal_reason_2021.py \
          tests/unit/test_services/test_tool_composer_observability_service.py && \
  git commit -m "feat(admin): observability renders why tools refuse (#2021 D3)

The codes are now read, not only written. Each tool row carries most_common_refusal_reason
from get_tool_reliability and its catalogue sentence; each recent-failure step class carries
its reason_code and rendered reason. Sentences are rendered at read time (owner decision D1'),
through known_sentence, which returns null for a code this build does not know rather than
relabelling it a tool failure. Imported function-locally: the tool_composer package costs
~564 MB to import, the service ~47 MB, and the admin route loads the service at API start.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01SnzgDeMLxZN48UsJXTaazb"
```

---

## Task 7b: The same rule on the synthesis prompt and the Digital Twin API (D6)

> **Added 2026-09-12 for owner decision D6.** Depends on Tasks 3, 4 and 4b. The owner accepted the larger blast radius: this changes what the synthesis LLM sees on partial success, and a second API's error contract.

**Files:**
- Modify: `src/agents/tool_composer/synthesizer.py` (`_format_results`), `src/agents/tool_composer/composer.py` (Task 4's loop, to share the helper), `src/api/routes/digital_twin.py`
- Create or modify: one shared helper for the rule — see Step 1
- Test: `tests/unit/test_agents/test_tool_composer/test_synthesis_failure_text_2020.py` (new); `tests/unit/test_api/test_routes/test_digital_twin*.py` (extend the file that already covers each route)

### (a) The synthesis prompt

`_format_results` writes `Error: {result.output.error}` for every failed step (about line 424). On a partial success, raw DoWhy / sklearn / driver text reaches the synthesis LLM, and from there it can reach the answer, which is exactly what Task 4 closed on the fail-closed path.

> **Extraction notes from the Task 4 quality review (M6).**
> - **Move `_TOOL_AUTHORED_CLASSES` out of `composer.py`.** `composer.py` imports `synthesizer.py`, so a helper that imported the constant from `composer.py` would be circular. Put it in `reason_codes.py` beside `EXECUTOR_ASSIGNED`. `ToolComposer._PLAN_DEFECT_CLASSES` may move with it.
> - **Keep the helper PURE.** Have it return `(fragment_or_None, withheld_raw_or_None)`, with no tool-name prefix and no logging; each caller prefixes and logs. If the helper logged, the synthesizer path would log the same text twice.
> - **Preserve Task 4's final rule exactly:** verbatim if trusted and non-empty, else the canonical sentence for a KNOWN code; nothing when there is no text and no code. Verbatim reasons are ordered before canonical fragments (I1).

- [ ] **Step 1: One rule, one helper.** Move Task 4's decision into a single function that both the composer and the synthesizer call, so the two surfaces cannot drift. The function takes `outcome_class`, `reason_code` and the raw text, and returns the user-safe text plus whether the raw text was withheld: verbatim when `outcome_class in {"refused", "input_rejected"}` and a code is present, otherwise `"<canonical sentence> [<code>]"` with `tool_error` as the fallback code. Keep it import-light — `reason_codes.py` is imported by `errors.py`, so the helper may live there only if it imports nothing new. The caller logs the withheld raw text, as Task 4 does.
- [ ] **Step 2: Red.** Build a partial-success trace — one succeeded step, one `error` step carrying `_DOWHY_INTERNALS` from Task 4's test, one coded `refused` step, one uncoded `refused` step — and assert on `_format_results`' output:
  - no `_LEAKS` substring appears
  - the refusal is present verbatim
  - both non-trusted steps render their canonical sentence and code
  - the raw text is in `caplog`

  Run it and see it fail.

  Build it through the seam that already exists (lead-verified): `test_synthesis_truncation_2019.py:192-222` renders through the REAL `_format_results`. It uses `SynthesisInput(original_query, decomposition, execution_trace)` — those three are the model's only fields — and `ResponseSynthesizer(llm_client=...)`, whose other constructor arguments all have defaults. Reuse that shape rather than inventing a builder. Note that `_format_results` reads `result.output.error` only on the failed branch (synthesizer.py about line 424).
- [ ] **Step 3: Green, then re-run** Task 4's test file, `test_synthesizer.py` and `test_synthesis_truncation_2019.py`, `-n 0`.

### (b) The Digital Twin API

**Measure before changing, site by site.** After Task 4b, all three production `SimulationEngine` constructions that pass `CohortCausalEstimator` produce authored `error_message` text. So the question at each site is whether library text can STILL arrive there, not whether a field is named "error". For each site, write down what reaches it and how you established that (source read, test, or a read-only prod query), then act only where library text can arrive. A site shown to be unreachable, or to carry only authored text, is left unchanged and reported with that evidence.

| Site | What to establish |
|---|---|
| `digital_twin.py:957`, `:1277` — 422 `detail=result.error_message` | After Task 4b: engine prefix plus authored estimator text. Expect keep-as-is. Confirm that no `except` in the engine path still interpolates library text. |
| `:985`, `:1302` — `error_message=`; `:977`, `:1294` — `recommendation_rationale=` | These are on the success response, and a FAILED result raises at `:957` / `:1277` first. Establish whether any non-FAILED status can carry an error message. |
| `:1441` — history `error_message=result.get("error_message")` | Failed results are not persisted (the N1 comment at `:951`). Check the repository for any other path that persists one, and count prod rows with a non-null `error_message`, read-only: `PGOPTIONS='-c default_transaction_read_only=on'`. |
| `:1011`, `:1355` — `except ValueError: detail=str(e)` → 400 | **The likeliest real leak.** Pydantic's `ValidationError` subclasses `ValueError`, and its text carries input values and a `pydantic.dev` URL. Enumerate every `ValueError` source inside each `try`. Grep the route tests and `frontend/src` for consumers of `detail`. |

- [ ] **Step 4: Report the findings table to the lead BEFORE changing any site** (status `NEEDS_CONTEXT` is correct here). Any change to an HTTP `detail` a test or the frontend pins is a contract change, and the lead confirms it with the owner. Expected shape of a fix where one is warranted: authored `ValueError`s the route itself raises keep their text; anything else gets a fixed sentence, and the raw text goes to `logger.warning`. The status code is unchanged.
- [ ] **Step 5: Red then green** for each site the lead confirms. Each test forces library text into the path (for example, a request that makes pydantic raise inside the `try`), then asserts the text is absent from the response body and present in the log.
- [ ] **Step 6: Verify and commit.** Run the route test files, the Task 4 and 7b(a) tests, and `ruff --no-cache`. Commit: `fix(tool-composer,api): no library text in the synthesis prompt or Digital Twin errors (#2020, D6)`.

**Certificate items this task adds to Task 8** (items 6–7 there).

---

## Task 8: Whole-lane verification, push, and the live certificate

> **Rewritten 2026-09-12 for D1′ and D4.** Deploy applies ml/043 automatically (`deploy.yml` runs `scripts/run_migrations.sh`); nothing is applied by hand. **Each outward action needs the owner's explicit go at the time: the push and PR, the merge, any PAID call in the live certificate, and every issue comment or close.**

**Files:**
- Create, untracked: `docs/demos/results/2026-09-12_lane_refusal_codes_cert/cert.md` (beside Task 6's rehearsal transcript)

- [ ] **Step 1: Run the affected suites**

Every run `-n 0`; redirect to a file and read `$?`, never a pager's exit code.

```bash
cd /home/enunez/Projects/e2i_causal_analytics/.worktrees/lane-refusal-codes && PYT=/home/enunez/Projects/e2i_causal_analytics/.venv/bin/pytest && \
  $PYT tests/unit/test_agents/test_tool_composer/ -n 0 -p no:cacheprovider -q --timeout=900 > /tmp/t8_tc.log 2>&1; echo "tool_composer EXIT=$?"; tail -2 /tmp/t8_tc.log; \
  $PYT tests/unit/test_services/test_tool_composer_observability_service.py tests/unit/test_api/test_routes/test_admin_tool_composer.py \
       tests/unit/test_database/learning_loop/test_lane_migration_files.py -n 0 -p no:cacheprovider -q --timeout=600 > /tmp/t8_rest.log 2>&1; echo "admin+static EXIT=$?"; tail -2 /tmp/t8_rest.log
```

The learning-loop real-DB modules skip on this droplet (D4). Say so in the report; Task 6's rehearsal transcript is their evidence.

- [ ] **Step 2: Lint** — `ruff check --no-cache src/ tests/` and `ruff format --check --no-cache` on every changed file. **No mypy on this box, not even one file**; CI's gate is the arbiter.

- [ ] **Step 3: Codex review to ACCEPT**

```bash
codex exec -C /home/enunez/Projects/e2i_causal_analytics/.worktrees/lane-refusal-codes \
  --sandbox read-only "$(cat /tmp/claude-1000/-home-enunez-Projects-e2i-causal-analytics/5e63ac9d-aac4-41c2-abc8-a2a1165854c4/scratchpad/codex_brief.md)" < /dev/null
```

Omit `-m` (the catalogue rotates). The brief carries the mandatory pushback paragraph verbatim. Iterate on REVISE findings until ACCEPT.

- [ ] **Step 4: Final whole-branch code review** by a fresh reviewer, against the full plan and `git diff 0a16e9c18..HEAD`.

- [ ] **Step 5: With the owner's go — ONE push, then the PR**

```bash
cd /home/enunez/Projects/e2i_causal_analytics/.worktrees/lane-refusal-codes && git branch --show-current && \
  git log --oneline origin/main..HEAD && git push -u origin claude/2021-2050-2020-refusal-reason-codes
```

Write the body to a file and create the PR with `--body-file`. Then re-read it with `gh pr view <n> --json body`: `--body-file` has silently failed before. The body must state:
- the two corrections to the issues as filed: #2050 cause 1's `StepResult.error_message` does not exist, and the plan's 94 raise sites are really 87
- owner decisions D1′, D4, D5 and D6, and lead calls L1, R1 and P1
- Task 7b's per-site findings for the Digital Twin API, including the sites left unchanged and the evidence for each
- the latent three-migration assertion in `test_migration_runner.py`
- `rollback_043.sql`'s runbook

End the body with the attribution block.

- [ ] **Step 6: CI** — `gh pr checks` 403s on this PAT. Use the actions runs API with the FULL 40-character sha:

```bash
SHA=$(git -C /home/enunez/Projects/e2i_causal_analytics/.worktrees/lane-refusal-codes rev-parse HEAD) && \
  gh api "repos/enunezvn/e2i_causal_analytics/actions/runs?head_sha=$SHA&per_page=50" \
  --jq '.workflow_runs[] | "\(.name)\t\(.status)\t\(.conclusion)"'
```

`Verify OpenAPI Types` must be green: it proves the regenerated `api.ts`. A red is investigated, never re-run blind.

- [ ] **Step 7: With the owner's go — merge** with `--merge`, never `--squash`.

- [ ] **Step 8: Deploy gate, before any live reading**

```bash
cd /home/enunez/Projects/e2i_causal_analytics && git fetch origin main --quiet && echo "main=$(git rev-parse origin/main)" && \
  gh api "repos/enunezvn/e2i_causal_analytics/actions/workflows/deploy.yml/runs?per_page=6" --jq '[.workflow_runs[]|select(.status!="completed")]|length' && \
  docker inspect e2i_api --format '{{.Config.Image}} {{.State.StartedAt}} {{.State.Health.Status}}' && \
  docker exec -e PGOPTIONS='-c default_transaction_read_only=on' supabase-db psql -U postgres -d postgres -X -tA \
    -c "select filename from public.schema_migrations where filename = 'ml/043_composer_refusal_reason_codes.sql'"
```

The gate passes only when: non-terminal deploys = 0, the container tag equals `main`'s full sha, it is healthy, and the ledger row is present. Trust the container and the ledger, not the deploy job's conclusion.

- [ ] **Step 9: Live certificate**

Before the FIRST paid call:
- run `pgrep -fa claude` and `ls -d docs/demos/results/2026-09-12*`, in case a peer session is already certifying this lane
- get the owner's authorization for the paid compositions
- build any scratch container's environment from `docker inspect e2i_api --format '{{range .Config.Env}}{{println .}}{{end}}'`, never from `--env-file .env`

Write `cert.md` verdict word first, then the numbers:

1. **Deploy:** the container tag and StartedAt before and after, and the `ml/043` ledger row.
2. **#2050 — the persisted row.** Force a refusal through a composed question: a 4-valued treatment column gives `non_binary_treatment`. Then:
   ```sql
   SELECT step_number, tool_name, outcome_class, error_type, reason_code, reason_details, error_message
   FROM composition_steps WHERE episode_id = '<the new episode>' ORDER BY step_number;
   ```
   PASS requires `reason_code` non-null, `reason_details` numeric, and **`error_message IS NULL`** (ml/041's guard intact). The before-picture is the two refused rows in `docs/demos/results/2026-09-12_tool_composer_wave_cert/`, which carry `error_type` only.
3. **#2021 — the admin page.** `GET /admin/observability/tool-composer` shows that tool's `most_common_refusal_reason` and `most_common_refusal_sentence`, and the failure's step class carries `reason_code` and `reason`.
4. **#2020 — the answer.** Force an all-null treatment column through a composed question. The returned answer contains no `DoWhy`, `shape=(0,)` or `backdoor.` substring, AND the container log for that same composition does contain the raw text. Both halves are required: the log half is what proves the text was withheld, not lost.
5. **Negative control.** A tool-authored refusal (a single-brand gap) reaches the answer verbatim. Without it, item 4 could pass because every reason was suppressed.
6. **D6(a) — the synthesis prompt, on the deployed image, free.** Inside `e2i_api`, build a partial-success trace (one succeeded step, one `error` step carrying DoWhy text, one coded refusal) and print `ResponseSynthesizer(llm_client=object())._format_results(...)`'s output. PASS requires no DoWhy substring, the refusal verbatim, and the canonical sentence plus code for the error step. No LLM call is made, so this item needs no paid authorization.
7. **D6(b) — the Digital Twin API.** For each site Task 7b changed, force its path on the live route and read the response body and the container log for that request. PASS requires the library text in the log and absent from the body, with the status code unchanged. For each site Task 7b left unchanged, cite its evidence rather than re-testing it.
8. **D5 — authored refusals carry no library text.** Force a non-numeric metric through a composed `gap_calculator` question. PASS requires the refusal to reach the answer verbatim (it is tool-authored) with no pandas `agg function failed` phrase in it, and that phrase present in the container log.

- [ ] **Step 10: With the owner's go — close out**

- Comment on #2021, #2050 (including the cause-1 correction and D1′) and #2020 (with the certificate's before and after).
- Close the three issues.
- Remove the worktree and delete the local branch.
- Write the lane's memory file and its `MEMORY.md` line.

---

## Self-review

**Spec coverage.**

| Requirement | Task |
|---|---|
| #2021: closed vocabulary with data-free sentences | 1 |
| #2021: every raise site coded, enforced | 2 |
| #2021: code carried through the executor | 3 |
| #2021: codes aggregated and shown per tool (D3) | 6, 7 |
| #2050 cause 1: the recorder never sent the reason | 5 |
| #2050 cause 2: the RPC dropped the reason | 6 |
| #2020: library internals kept out of the answer | 4 |
| D1′: code and details persisted, no text, sentence rendered at read time | 5, 6, 7 |
| D2: all sites | 2 |
| D4: rehearsal rather than an in-harness test | 6 |
| L1: numeric-only details | 2b |
| R1: recodes from the Task 1+2 review; P1: `__reduce__` kept | 2b |
| D5: library exception text removed at its source, AST-guarded | 4b |
| D6: synthesis prompt and Digital Twin API follow the same rule | 7b |

**Type consistency.**
- `StepResult.reason_code` is `Optional[str]` holding `ReasonCode` values, from Task 3 on.
- `StepResult.reason_details` is `Dict[str, number | bool]`.
- The recorder sends exactly these two keys (Task 5).
- The database stores `text` and `jsonb`, with a matching format guard and reducer (Task 6).
- `most_common_refusal_reason` is spelled identically in `get_tool_reliability` (Task 6), `ToolReliability` (Task 7), `ToolReliabilityRow` (Task 7) and the service (Task 7).
- `canonical_sentence` (fails closed to the tool-failure sentence) is used only for the user answer (Task 4). `known_sentence` (returns `None` when unknown) is used only for display (Task 7).

**Known gaps, deliberate.**
- The database guards the code's format, not its member list, because the list lives in Python and a copy would drift. A static test checks every member passes the format.
- The same split applies to `details`, except for the key rule. Python enforces the magnitude bound `|int| ≤ 2**53` and numpy normalization (Task 2b quality fix), which the database does not repeat. `composer_structure_reason_details` enforces number or boolean values, at most 8 keys, and — **reversed by lead decision 2026-09-13, Task 6 quality review** — the same key convention as Python's `_DETAIL_KEY`, `(n|is|has|share)_[a-z0-9_]{1,58}` (at most 64 characters; a static test pins the two equal). Plain snake_case let a data-keyed map (`{column_name: count}`, a lowercase brand name) through, which is the accidental shape a second guard exists for under D1′; a new prefix is rare enough to justify a migration. Task 6's rehearsal key `"flag": true` is therefore no longer persisted.
- Sentences are never stored, so ml/041's no-text guard stays exactly as built.
- ml/043 is proven by a rehearsal, not a permanent real-DB test: the shared fixture could only run such a test once, pre-deploy.
- **Added by lead decision 2026-09-13 (Task 7 quality review, D1):** `get_tool_reliability` also returns `n_most_common_refusal_reason`, the number of coded refusals carrying `most_common_refusal_reason`. `mode()` returned only the value, so a 1-1-1 tie (alphabetical winner) read exactly like a 3-of-3 majority: the same harm `n_refused_coded` exists to prevent. The code and its count now come from one `row_number()` ranking over exactly `n_refused_coded`'s rows. Ties resolve to the highest count, then the code ascending in `"C"` collation, so the result does not depend on the locale. A static test pins the expressions and the CTE filter; the rehearsal proves a tie and a majority.
- `test_migration_runner.py`'s three-migration assertion goes stale, and is disclosed rather than fixed, to leave the shared harness untouched.
- `ReferenceResolutionError` is not a coded error; the executor assigns its code.
