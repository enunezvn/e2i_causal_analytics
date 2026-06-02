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
Top-level keys introduced via ``state["..."] = ...`` (direct write) and
``state.setdefault("...", ...)`` in ``scripts/run_tier0_test.py``, plus the
keys initialised in the top-level pipeline state dict literal
(``experiment_id``, ``patient_df``).

Assumptions / known gaps
------------------------
1. ``state.setdefault("KEY", ...)`` IS covered (the script uses it for
   ``leakage_diagnostics`` / ``halt_reason``). ``state.update({...})`` is
   NOT auto-parsed — its payload can't be regex-parsed reliably — but the
   script does not use it and a tripwire assertion fails loudly if that
   ever changes, so those keys cannot silently slip past the guard.
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

# Match: state["KEY"] = ... — bare ``state`` identifier only, so we do not pick
# up writes against ``feature_state`` / ``_rem_state``. The (?<![A-Za-z0-9_])
# lookbehind enforces the word boundary on the left without consuming a
# character (so we can also match at start of line).
_STATE_WRITE_RE = re.compile(r'(?<![A-Za-z0-9_])state\["([a-zA-Z_][a-zA-Z_0-9]*)"\]\s*=')

# Match: state.setdefault("KEY", ...) / state.setdefault('KEY', ...) — same
# bare-``state`` word boundary. setdefault introduces a top-level key exactly
# like a direct write (the script uses it for ``leakage_diagnostics`` and
# ``halt_reason``), so it must be covered or the guard silently misses an
# undeclared key (#619).
_STATE_SETDEFAULT_RE = re.compile(
    r"(?<![A-Za-z0-9_])state\.setdefault\(\s*[\"']([a-zA-Z_][a-zA-Z_0-9]*)[\"']"
)


def _emitted_state_keys(script_text: str) -> set[str]:
    """Top-level keys the script introduces on the pipeline ``state`` dict.

    Captures both direct writes ``state["KEY"] = ...`` and
    ``state.setdefault("KEY", ...)`` — both create a new top-level key the
    ``Tier0OutputMapper`` contract must declare.
    """
    return set(_STATE_WRITE_RE.findall(script_text)) | set(
        _STATE_SETDEFAULT_RE.findall(script_text)
    )


@pytest.mark.unit
def test_contract_covers_all_script_emitted_keys() -> None:
    """Every ``state["..."] = ...`` in ``scripts/run_tier0_test.py`` must
    be declared in ``Tier0StateContract`` (Required or NotRequired).
    """
    script_path = PROJECT_ROOT / "scripts" / "run_tier0_test.py"
    script_text = script_path.read_text()

    # ``state.update({...})`` would also introduce top-level keys, but its
    # payload can't be regex-parsed reliably. The script doesn't use it; trip
    # loudly if that changes so a contributor extends ``_emitted_state_keys``
    # rather than the guard silently missing the new keys (#619).
    assert not re.search(r"(?<![A-Za-z0-9_])state\.update\(", script_text), (
        "scripts/run_tier0_test.py now uses state.update(...); extend "
        "_emitted_state_keys to parse its keys so the contract guard does not "
        "silently miss them."
    )

    direct_writes = _emitted_state_keys(script_text)

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


@pytest.mark.unit
def test_extractor_captures_setdefault_keys() -> None:
    """``state.setdefault("KEY", ...)`` introduces a top-level key just like a
    direct write, so the contract-coverage extractor must capture it.

    ``scripts/run_tier0_test.py`` uses this pattern (``leakage_diagnostics``,
    ``halt_reason``). Before this, the extractor only matched ``state["KEY"] =``
    and silently missed setdefault keys — the exact drift the guard exists to
    catch (an undeclared setdefault key would crash ``Tier0OutputMapper(state)``
    on a fresh run). See #619.
    """
    sample = "diagnostics = state.setdefault(\"leak_diag\", {})\nstate.setdefault('halt', 'x')\n"
    assert _emitted_state_keys(sample) == {"leak_diag", "halt"}
