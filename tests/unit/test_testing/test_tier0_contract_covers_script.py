"""Self-policing test for Tier0StateContract drift.

If ``scripts/run_tier0_test.py`` adds a state key that isn't in
``Tier0StateContract``, this test fails — preventing the schema-drift
class of bug that 1B-M7's strict validator (``_validate_contract``)
raises at runtime. The fix-up that introduced this test was prompted by
``f1_threshold_analysis`` and ``test_metrics_at_05`` being emitted by
the script but absent from the contract — fresh tier0 runs would crash
``Tier0OutputMapper(state)`` with ``TypeError`` on those keys.

What this test detects
----------------------
Direct writes of the form ``state["..."] = ...`` in
``scripts/run_tier0_test.py``, plus the keys initialised in the
top-level pipeline state dict literal (``experiment_id``, ``patient_df``).

Assumptions / known gaps
------------------------
1. ``scripts/run_tier0_test.py`` does NOT use ``state.update({...})``
   or ``state.setdefault(...)`` to introduce new top-level keys at the
   time of writing — verified by ``grep`` of the script. If the script
   ever starts using either pattern, this test will silently miss those
   keys; the regex would need to be widened. We trade some completeness
   for a low false-positive rate.
2. The initial state literal at the top of the pipeline (currently at
   the call site that builds ``state`` in the main runner) carries the
   gate keys ``experiment_id`` and ``patient_df``. These are listed
   explicitly below so the test does not have to parse arbitrary dict
   literals.
3. Nested writes such as ``state["scope_spec"]["cost_matrix"] = ...``
   mutate already-known keys and are intentionally NOT inspected — the
   contract keys themselves are the boundary the mapper enforces.
4. Other dicts named with the suffix ``_state`` (e.g. ``feature_state``,
   ``_rem_state``) are local to helpers and never reach the mapper, so
   the regex deliberately matches the bare identifier ``state``.
"""

import re
from pathlib import Path

import pytest

from src.testing.tier0_output_mapper import Tier0StateContract

PROJECT_ROOT = Path(__file__).resolve().parents[3]


@pytest.mark.unit
def test_contract_covers_all_script_emitted_keys() -> None:
    """Every ``state["..."] = ...`` in ``scripts/run_tier0_test.py`` must
    be declared in ``Tier0StateContract`` (Required or NotRequired).
    """
    script_path = PROJECT_ROOT / "scripts" / "run_tier0_test.py"
    script_text = script_path.read_text()

    # Match: state["KEY"] = ... — bare ``state`` identifier only, so we
    # do not pick up writes against ``feature_state`` / ``_rem_state``.
    # The (?<![A-Za-z0-9_]) lookbehind enforces the word boundary on the
    # left without consuming a character (so we can also match at start
    # of line).
    direct_writes = set(
        re.findall(r'(?<![A-Za-z0-9_])state\["([a-zA-Z_][a-zA-Z_0-9]*)"\]\s*=', script_text)
    )

    # Initial-state dict literal carries these gate keys (see the
    # ``state: dict[str, Any] = {...}`` block in the main runner).
    direct_writes |= {"experiment_id", "patient_df"}

    allowed = Tier0StateContract.__required_keys__ | Tier0StateContract.__optional_keys__
    missing = direct_writes - allowed
    assert not missing, (
        f"scripts/run_tier0_test.py emits keys not declared in Tier0StateContract: "
        f"{sorted(missing)}. Either add them to Tier0StateContract (most likely "
        f"NotRequired[...]) or remove the emit. The strict validator in "
        f"Tier0OutputMapper.__init__ would otherwise raise TypeError on a fresh "
        f"tier0 run."
    )
