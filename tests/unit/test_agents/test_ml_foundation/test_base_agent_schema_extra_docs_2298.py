"""#2298: ml_foundation docs must not describe ``BaseAgentSchema`` as ``extra="allow"``.

PR #67 tightened ``extra="allow"`` -> ``extra="ignore"`` (see ``_pydantic_utils.py``'s
changelog), but eight docstrings/comments across five modules were never updated. This is
not ordinary doc drift: ``extra`` decides whether a node's returned key SURVIVES. Under
``ignore`` an undeclared key is silently dropped at validation, so a reader who believes
``allow`` will reasonably skip declaring a new state field -- which is #2288, where every
``ge_*`` key returned by ``run_ge_validation`` is undeclared and never reaches state.

Two of the eight were not merely stale but FALSE, measured 2026-09-24:
  * ``model_trainer/state.py`` claimed a runner-up ``AliasChoices`` key lands in
    ``model_extra``; measured ``model_extra is None`` -- it is dropped.
  * ``scope_definer/schemas.py`` claimed ``ScopeSpecSchema`` lets unknown keys pass
    through; measured ``extra="ignore"`` -- they are dropped. (Independently hit by the
    #2283 lane, which had to pin a key on the state because scope_spec swallowed it.)

Each case below pins the exact stale sentence rather than a loose pattern: a proximity
regex either misses the bullet-list form (the tokens are lines apart) or fires on
``_pydantic_utils.py``'s legitimate changelog and on real per-class overrides.
"""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import pytest

from src.agents.ml_foundation._pydantic_utils import BaseAgentSchema

SRC = Path(__file__).resolve().parents[4] / "src" / "agents" / "ml_foundation"

#: (file, the exact stale text, why it is wrong). All eight were present when #2298 was filed.
STALE_CLAIMS = [
    ("scope_definer/state.py", '- ``extra="allow"`` for forward-compat', "header: base is ignore"),
    ("data_preparer/state.py", '- ``extra="allow"`` for forward-compat', "header: base is ignore"),
    ("model_trainer/state.py", 'BaseAgentSchema`` (extra="allow"', "header: base is ignore"),
    (
        "model_trainer/state.py",
        'reserved-name rule (BaseAgentSchema\'s extra="allow" preserves them)',
        "true behaviour, wrong owner: SuccessCriteriaSchema overrides to allow",
    ),
    (
        "model_trainer/state.py",
        'with its discarded value (because ``extra="allow"``',
        "FALSE: measured model_extra is None; the runner-up is dropped",
    ),
    (
        "model_trainer/schemas.py",
        'OVERRIDE BaseAgentSchema\'s ``extra="allow"``',
        "the forbid override is real, but the baseline it names is ignore",
    ),
    (
        "scope_definer/schemas.py",
        '``extra="allow"`` (inherited from',
        "FALSE: ScopeSpecSchema is extra=ignore; unknown keys are dropped",
    ),
    (
        "_pydantic_utils.py",
        'keys to ``model_extra`` (preserved by ``extra="allow"``)',
        "__setitem__ writes model_extra explicitly, not via extra=allow",
    ),
]


def test_the_base_schema_is_ignore_not_allow():
    """The measured fact every doc must agree with."""
    assert BaseAgentSchema.model_config.get("extra") == "ignore"


@pytest.mark.parametrize(
    "relpath,stale,why", STALE_CLAIMS, ids=[f"{f}::{w[:28]}" for f, _, w in STALE_CLAIMS]
)
def test_the_stale_extra_allow_claim_is_gone(relpath: str, stale: str, why: str):
    text = (SRC / relpath).read_text()
    assert stale not in text, f"{relpath} still says {stale!r} — {why}"


def test_the_runner_up_alias_is_dropped_not_kept():
    """Pins the behaviour the FALSE comment in model_trainer/state.py described backwards."""
    from src.agents.ml_foundation.model_trainer.state import ModelTrainerState

    state = ModelTrainerState(
        audit_workflow_id=uuid4(),
        repeated_mode_fold_invocation=True,
        _repeated_mode_fold_invocation=False,
    )
    assert state.repeated_mode_fold_invocation is True  # canonical still wins
    assert not (state.model_extra or {})  # ...and the runner-up leaves NO residue


def test_scope_spec_drops_unknown_keys():
    """Pins the behaviour the FALSE comment in scope_definer/schemas.py described backwards."""
    from src.agents.ml_foundation.scope_definer.schemas import ScopeSpecSchema

    assert ScopeSpecSchema.model_config.get("extra") == "ignore"
    spec = ScopeSpecSchema(audit_workflow_id=uuid4(), unknown_future_key="dropped")
    assert "unknown_future_key" not in (spec.model_extra or {})


def test_setitem_still_routes_unknown_keys_to_model_extra():
    """The asymmetry that explains #2288: a dict-style WRITE keeps an unknown key, while a
    key merely RETURNED by a node goes through validation and is dropped. Both are true at
    once, which is exactly why the stale ``extra="allow"`` wording misleads."""
    from src.agents.ml_foundation.data_preparer.state import DataPreparerState

    state = DataPreparerState(audit_workflow_id=uuid4())
    state["some_unknown_key"] = "written"
    assert (state.model_extra or {}).get("some_unknown_key") == "written"


def test_a_legitimate_per_class_allow_override_is_untouched():
    """The guard must not forbid documenting a REAL override."""
    from src.agents.ml_foundation.scope_definer.schemas import SuccessCriteriaSchema

    assert SuccessCriteriaSchema.model_config.get("extra") == "allow"
    assert "_adaptive_skipped" in (SuccessCriteriaSchema(_adaptive_skipped=True).model_extra or {})
