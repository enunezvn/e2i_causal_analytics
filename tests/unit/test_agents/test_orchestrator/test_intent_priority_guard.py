"""Import-time guard: ``INTENT_PATTERNS`` ⊆ ``INTENT_PRIORITY``.

Issue #266. A future PR adding a key to ``INTENT_PATTERNS`` but forgetting to
rank it in ``INTENT_PRIORITY`` silently regresses dict-iteration-order
tie-break (the exact root cause of issue #254, fixed by PR #247 commit
``bf27ff40``). The module-level structures themselves must enforce this
invariant at import time so the failure mode is loud, fast, and impossible
to miss in code review.

The fixture below exercises the import-time assertion (not a runtime check
*after* import) by either reloading the module or executing the module
source in a fresh namespace with a mutated ``INTENT_PATTERNS``.
"""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

import pytest


def test_module_imports_cleanly() -> None:
    """Positive case: today the module imports fine because
    ``set(INTENT_PATTERNS) <= set(INTENT_PRIORITY)`` holds. Reload to make
    sure the import-time logic runs even if pytest cached a prior import.
    """
    import src.agents.orchestrator.nodes.intent_classifier as ic

    reloaded = importlib.reload(ic)

    # Belt-and-suspenders: the contract must hold post-reload too.
    missing = set(reloaded.IntentClassifierNode.INTENT_PATTERNS) - set(reloaded.INTENT_PRIORITY)
    assert not missing, f"Drift detected post-reload: {sorted(missing)}"


def test_intent_patterns_subset_of_priority() -> None:
    """Negative case: a synthetic intent absent from ``INTENT_PRIORITY``
    must trip the import-time guard.

    We execute the module source in a fresh namespace with ``INTENT_PRIORITY``
    pre-populated to *omit* a fake intent, then mutate the class body's
    ``INTENT_PATTERNS`` to include the fake intent. The guard runs as part of
    module evaluation, so an ``AssertionError`` must propagate out of
    ``exec``. We cannot use ``importlib.reload`` here because the source we
    need to falsify is the module itself; instead we re-execute the file in
    a controlled namespace.

    Per issue #266 acceptance criterion 2.
    """
    src_path = Path("src/agents/orchestrator/nodes/intent_classifier.py").resolve()
    source = src_path.read_text(encoding="utf-8")

    # Inject a fake intent into the patterns dict *after* the class body
    # would have established the dict. We tail-append a mutation + re-run
    # the guard logic. Simpler: monkeypatch the class attribute and call a
    # helper that re-evaluates the guard. But the guard must be module-level
    # to satisfy AC2. So we capture the assertion text and grep for it
    # immediately, *and* we additionally exec a small synthetic copy of
    # the guard with adversarial inputs to verify the assertion is the one
    # that would actually trip in production.

    # 1. Static check: the guard literal is present in the module source.
    assert "set(IntentClassifierNode.INTENT_PATTERNS)" in source or (
        "INTENT_PATTERNS" in source and "INTENT_PRIORITY" in source and "assert" in source
    ), "Guard logic must reference both structures and assert"

    # 2. Dynamic check: exec a fresh namespace where INTENT_PRIORITY is
    # missing the synthetic intent, and confirm an AssertionError fires.
    # We simulate the guard in isolation to exercise the actual assertion
    # text the module uses.
    ns: dict[str, Any] = {
        "INTENT_PRIORITY": ("experiment_monitor", "general"),
        "FAKE_INTENT_PATTERNS": {
            "experiment_monitor": [r"x"],
            "synthetic_unknown_intent": [r"y"],
            "general": [r"z"],
        },
    }
    guard_snippet = (
        "_missing = set(FAKE_INTENT_PATTERNS) - set(INTENT_PRIORITY)\n"
        "assert not _missing, (\n"
        "    f'INTENT_PATTERNS contains intents missing from INTENT_PRIORITY: '\n"
        "    f'{sorted(_missing)}. Add them to INTENT_PRIORITY to preserve '\n"
        "    f'deterministic tie-break.'\n"
        ")\n"
    )
    with pytest.raises(AssertionError) as excinfo:
        exec(guard_snippet, ns)
    assert "synthetic_unknown_intent" in str(excinfo.value)

    # 3. End-to-end: monkeypatch ``IntentClassifierNode.INTENT_PATTERNS`` to
    # include the fake intent and reload — the import-time guard inside the
    # module must trip. We re-execute the module body in a controlled
    # namespace with a small instrumented preamble that mutates the patterns
    # dict before the guard runs.
    instrumented = source.replace(
        "    def __init__(self):",
        # Inject the fake key into the dict literal AFTER the class body has
        # set INTENT_PATTERNS as an attribute (right before __init__).
        "    INTENT_PATTERNS['__synthetic_test_intent__'] = [r'never-matches']\n\n    def __init__(self):",
        1,
    )
    assert "__synthetic_test_intent__" in instrumented, "instrumentation did not apply"
    # Provide a synthetic package context so relative imports inside the
    # module body resolve. We name the synthetic module __intent_guard_test__
    # but anchor __package__ on the real package so ``from ..state import …``
    # works in exec.
    exec_globals: dict[str, Any] = {
        "__name__": "src.agents.orchestrator.nodes.intent_classifier",
        "__package__": "src.agents.orchestrator.nodes",
        "__file__": str(src_path),
    }
    with pytest.raises(AssertionError) as excinfo:
        exec(compile(instrumented, str(src_path), "exec"), exec_globals)
    msg = str(excinfo.value)
    assert "__synthetic_test_intent__" in msg, (
        f"Guard must name the missing intent in its assertion message; got: {msg!r}"
    )


def test_module_docstring_documents_contract() -> None:
    """AC3: the module docstring must mention the contract so future authors
    of new intents see the invariant before adding to ``INTENT_PATTERNS``.
    """
    import src.agents.orchestrator.nodes.intent_classifier as ic

    doc = (ic.__doc__ or "").lower()
    # The contract phrasing can be flexible but the load-bearing words must
    # all appear together.
    assert "intent_patterns" in doc, "Docstring must mention INTENT_PATTERNS"
    assert "intent_priority" in doc, "Docstring must mention INTENT_PRIORITY"
    # 'must' / 'required' / 'contract' indicates this is binding, not advisory.
    assert any(token in doc for token in ("must", "required", "contract", "invariant")), (
        "Docstring must signal the contract is binding"
    )


def test_intent_priority_is_strict_superset_today() -> None:
    """Sanity: prove the invariant holds in the as-shipped main branch so the
    guard does not falsely block legitimate work. If this ever fails,
    something legitimate has drifted and the guard should be the one
    catching it — this test is the canary that something deeper changed.
    """
    from src.agents.orchestrator.nodes.intent_classifier import (
        INTENT_PRIORITY,
        IntentClassifierNode,
    )

    patterns = set(IntentClassifierNode.INTENT_PATTERNS)
    priority = set(INTENT_PRIORITY)
    assert patterns <= priority, (
        f"INTENT_PATTERNS includes intents missing from INTENT_PRIORITY: "
        f"{sorted(patterns - priority)}"
    )
