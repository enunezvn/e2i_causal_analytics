"""Module-size ratchet (#1991 debt 4): no file under src/ may exceed LIMIT lines unless it is
pinned here at its current size, and a pinned file may only shrink.

Why a ratchet and not a hard cap: 32 files already exceed the limit (measured 2026-09-13).
A hard cap would either fail forever or exempt them forever. src/api/routes/causal.py was
the one deliberate omission while it was being split; it is now the ``causal/`` package and
every module in it is under LIMIT, so nothing there needs a pin and the ratchet simply guards
against regrowth. Pins can only move DOWN:
the test fails if a pinned file grows past its pin AND if a pin is above the file's
actual size (so the number on record is always the real one). Delete a pin when the
file drops under LIMIT.
"""

from __future__ import annotations

from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
SRC = REPO / "src"
LIMIT = 1500

# path (relative to repo) -> pinned line count. Measured, not guessed. Only shrinks.
ALLOWLIST: dict[str, int] = {
    "src/agents/causal_impact/nodes/refutation.py": 2520,
    "src/agents/feedback_learner/dspy_integration.py": 1836,
    "src/agents/ml_foundation/data_preparer/nodes/adaptive_validity_check.py": 4238,
    "src/agents/ml_foundation/model_deployer/nodes/registry_manager.py": 1670,
    "src/agents/ml_foundation/model_trainer/nodes/evaluator.py": 4026,
    # 3710: MEASURED from the merged tree, not either side's pin. The lane's
    # value-lookup mask (#2114) and main's #2139 structural guard both run here,
    # and the guard now receives the structured brand (codex iter10). That costs
    # lines relative to the lane (3700) and saves them relative to main (3770),
    # so the pin still shrinks from main's.
    "src/agents/orchestrator/nodes/dispatcher.py": 3697,
    "src/agents/tool_composer/executor.py": 1627,
    "src/agents/tool_composer/tool_registrations.py": 4843,
    "src/api/main.py": 1692,
    "src/api/routes/chatbot_dspy.py": 4182,
    "src/api/routes/chatbot_graph.py": 3190,
    # 3054 -> 3029: #2150 replaced hand-kept capability literals with policy calls
    # (and then lazy accessors, which removed two more lines).
    "src/api/routes/chatbot_tools.py": 3029,
    "src/api/routes/copilotkit.py": 6142,
    "src/api/routes/digital_twin.py": 1963,
    "src/api/routes/experiments.py": 1595,
    "src/api/routes/explain.py": 2822,
    "src/api/routes/feedback.py": 2041,
    "src/api/routes/health_score.py": 2379,
    "src/api/routes/monitoring.py": 2154,
    "src/api/routes/predictions.py": 1519,
    "src/api/routes/resource_optimizer.py": 1549,
    "src/api/routes/segments.py": 2934,
    "src/api/schemas/causal.py": 2142,
    "src/causal_engine/energy_score/estimator_selector.py": 1922,
    "src/causal_engine/refutation_runner.py": 3166,
    "src/data/causal_role_classifier.py": 6525,
    "src/feature_store/feast_client.py": 1711,
    "src/memory/lifecycle/consolidator.py": 1954,
    "src/memory/semantic_memory.py": 1795,
    "src/ml/data_generator.py": 1759,
    "src/rag/cognitive_rag_dspy.py": 1587,
    "src/rag/evaluation.py": 1753,
}


def _lines(p: Path) -> int:
    with p.open("rb") as fh:
        return sum(1 for _ in fh)


def _all_py() -> list[Path]:
    assert SRC.is_dir(), SRC
    files = sorted(p for p in SRC.rglob("*.py") if "__pycache__" not in p.parts)
    # src/ holds ~1,000 .py files today (measured 2026-09-13). If REPO/SRC ever
    # resolves to the wrong place, rglob silently returns few or zero files and
    # the offenders test below passes vacuously with nothing scanned. This floor
    # is far below the real count but far above zero, so a broken path fails loud.
    assert len(files) >= 200, f"only {len(files)} .py files under {SRC} — path resolution broke"
    return files


def test_no_unpinned_file_exceeds_limit():
    offenders = []
    for p in _all_py():
        n = _lines(p)
        rel = p.relative_to(REPO).as_posix()
        if n > LIMIT and rel not in ALLOWLIST:
            offenders.append(f"{rel}: {n} > {LIMIT}")
    assert not offenders, "split these by concern or pin them (pins only shrink):\n" + "\n".join(
        offenders
    )


@pytest.mark.parametrize("rel,pin", sorted(ALLOWLIST.items()))
def test_pinned_file_has_not_grown_and_pin_is_current(rel, pin):
    assert rel.startswith("src/"), f"{rel}: pins must point under src/"
    p = REPO / rel
    assert p.exists(), f"{rel} is pinned but missing — delete its pin"
    n = _lines(p)
    assert n <= pin, f"{rel} grew to {n} lines (pin {pin}); shrink it, do not raise the pin"
    assert n > LIMIT, f"{rel} is under {LIMIT}; delete its pin"
    assert n == pin, f"{rel} is {n} lines but pinned at {pin}; lower the pin to {n}"
