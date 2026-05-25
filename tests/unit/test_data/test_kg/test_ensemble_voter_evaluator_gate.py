"""Issue #240 Stage 3 — env-gated audit-evaluator soft-gate on EnsembleVoter.

Design reference: ``docs/plans/240-audit-evaluator-gate-promotion.md`` §3 Stage 3
(Mechanism + Fail-open) and §4 R1, **reframed 2026-05-25** to the reachable
``info → moderate`` transition (see
``docs/plans/240-r1-reachability-investigation.md``).

Why the reframe matters for THESE tests: the original suite drove the gate via
an *invalid* LLM role ("not_a_real_role") so the voter sanitised the LLM to
None and fell through to the adversarial-moderate-alone branch while still
carrying the audit. A reachability audit proved that arrangement is impossible
in production — the loader (``classify_feature``) returns ``None`` for an
out-of-vocabulary role *before* the evaluator ever runs, so an audit-bearing
verdict ALWAYS has a valid role, and a valid role maps to severity ``high``
(leak) or ``info`` (accept) — never ``moderate``. So the audited candidate the
gate can actually see is ``info``, and the reachable escalation is
``info → moderate``.

Load-bearing invariants pinned here:

(reachability) **The gate precondition is reachable in production.** A VALID
    accept-role worker verdict carrying an evaluator audit, routed through the
    real ``vote``, yields candidate severity ``"info"`` — the exact state R1'
    triggers on. (The old suite could only manufacture the gate's precondition
    by bypassing the loader; this one does not.)

(a) **Default OFF ⇒ byte-identical.** Kill-switch unset ⇒ full ``vote`` output
    equals ``_vote_candidate`` field-for-field — the gate never fires.

(b) **flag=1 + R1 fires + candidate info ⇒ flip.** severity info→moderate,
    remediation→"review", decided_by="evaluator_gate", evidence tag appended,
    ``gate_rule_fired == "R1"``.

(c) **Fail-open.** flag=1 but no audit (evaluator disabled) OR ``satisfied is
    None`` (evaluator error) ⇒ no flip.

(d) **flag=1 but candidate severity != info ⇒ no flip** (high leak-role
    candidate; real moderate adversarial-alone candidate).
"""

from __future__ import annotations

import pytest

from src.data.kg.ensemble_voter import EVALUATOR_GATE_ENABLED_ENV, EnsembleVoter
from src.data.kg.types import LLMEvaluatorAudit, LLMVerdict

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _audit(
    *,
    satisfied,
    rationale_complete: bool = True,
    missed: tuple[str, ...] = (),
) -> LLMEvaluatorAudit:
    return LLMEvaluatorAudit(
        satisfied=satisfied,  # type: ignore[arg-type]
        rationale_complete=rationale_complete,
        missed_considerations=missed,
        notes="",
        evaluator_model="anthropic/claude-haiku-4-5-20251001",
    )


def _llm_with_audit(audit, *, role: str = "ancestor") -> LLMVerdict:
    """LLMVerdict carrying an evaluator audit.

    ``role`` defaults to ``"ancestor"`` — a *valid accept role*. The voter maps
    accept roles to ensemble severity ``"info"`` via ``_llm_severity``, which is
    R1's reachable precondition. This mirrors the real production object: the
    loader only ever emits audit-bearing verdicts with valid roles.
    """
    return LLMVerdict(
        causal_role=role,  # type: ignore[arg-type]
        mechanism="m",
        recommended_remediation="keep_with_caveat",
        evaluator_audit=audit,
    )


def _adv_moderate() -> dict:
    """Adversarial-moderate verdict dict (the only producer of a *moderate*
    ensemble candidate — and, in production, it carries NO LLM verdict/audit)."""
    return {
        "feature": "feat_x",
        "layer": "3",
        "severity": "moderate",
        "z_score": 4.0,
        "actual_auc": 0.7,
        "null_mean": 0.5,
        "null_std": 0.05,
        "p_value": 0.01,
        "n_permutations": 200,
        "remediation": "ambiguous",
        "evidence": "z=4.0",
        "contract_source": None,
        "contract_window_days": None,
    }


def _vote_info_candidate(voter: EnsembleVoter, audit, *, role: str = "ancestor") -> tuple:
    """Return (candidate, final) for the PRODUCTION-REACHABLE info path: a valid
    accept-role worker verdict carrying ``audit``, routed through the real voter.
    Candidate severity is ``"info"`` (R1's reframed precondition)."""
    llm = _llm_with_audit(audit, role=role)
    candidate = voter._vote_candidate("feat_x", llm_verdict=llm)
    final = voter.vote("feat_x", llm_verdict=llm)
    return candidate, final


@pytest.fixture
def voter() -> EnsembleVoter:
    return EnsembleVoter()


# ---------------------------------------------------------------------------
# (reachability) the gate precondition occurs on the real production path
# ---------------------------------------------------------------------------


def test_info_candidate_is_reachable_via_valid_accept_role(monkeypatch, voter):
    """The reframed gate fires on ``info``. Prove a VALID accept-role worker
    verdict (the only kind that carries an audit in production) yields a
    ``info`` candidate through the real voter — so the gate's precondition is
    genuinely reachable, unlike the pre-reframe ``moderate`` precondition which
    no production path could pair with an audit.

    FALSIFIABILITY: if ``_llm_severity`` ever maps accept roles away from
    ``info``, this trips and the flip tests below would be testing an
    unreachable state."""
    monkeypatch.delenv(EVALUATOR_GATE_ENABLED_ENV, raising=False)
    audit = _audit(satisfied=False, missed=("temporal_filter",))
    candidate, _final = _vote_info_candidate(voter, audit)
    assert candidate.severity == "info", (
        f"valid accept role must yield an info candidate; got {candidate.severity!r}"
    )
    assert candidate.llm_input is not None
    assert candidate.llm_input.evaluator_audit is audit  # audit really rides along
    assert candidate.gate_rule_fired is None


# ---------------------------------------------------------------------------
# (a) Default OFF ⇒ byte-identical
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "audit",
    [
        _audit(satisfied=False, missed=("temporal_filter",)),  # would fire if on
        _audit(satisfied=True),
        None,
    ],
)
def test_default_off_is_byte_identical(monkeypatch, voter, audit):
    """Flag unset ⇒ vote() == _vote_candidate() field-for-field. The gate
    never fires and never mutates anything (the ship-disabled guarantee)."""
    monkeypatch.delenv(EVALUATOR_GATE_ENABLED_ENV, raising=False)
    candidate, final = _vote_info_candidate(voter, audit)
    assert final.severity == candidate.severity
    assert final.remediation == candidate.remediation
    assert final.decided_by == candidate.decided_by
    assert final.evidence == candidate.evidence
    assert final.gate_rule_fired == candidate.gate_rule_fired
    assert final.confidence == candidate.confidence
    assert final.final_role == candidate.final_role
    # decided_by must never be the gate value when the flag is off.
    assert final.decided_by != "evaluator_gate"
    assert final.gate_rule_fired is None


@pytest.mark.parametrize("flag_value", ["0", "", "true", "yes", "2", "01"])
def test_non_one_flag_values_keep_gate_off(monkeypatch, voter, flag_value):
    """Only the exact string "1" enables the gate; everything else is OFF."""
    monkeypatch.setenv(EVALUATOR_GATE_ENABLED_ENV, flag_value)
    audit = _audit(satisfied=False, missed=("temporal_filter",))
    candidate, final = _vote_info_candidate(voter, audit)
    assert final.severity == candidate.severity == "info"
    assert final.gate_rule_fired is None
    assert final.decided_by != "evaluator_gate"


# ---------------------------------------------------------------------------
# (b) flag=1 + R1 fires + candidate info ⇒ flip info→moderate
# ---------------------------------------------------------------------------


def test_flag_on_r1_fires_flips_to_moderate(monkeypatch, voter):
    monkeypatch.setenv(EVALUATOR_GATE_ENABLED_ENV, "1")
    audit = _audit(satisfied=False, missed=("temporal_filter",))
    candidate, final = _vote_info_candidate(voter, audit)

    # Candidate was info (accept role); gate escalated it.
    assert candidate.severity == "info"
    assert final.severity == "moderate"
    assert final.remediation == "review"
    assert final.decided_by == "evaluator_gate"
    assert final.gate_rule_fired == "R1"
    # Structured evidence tag appended (and the candidate's evidence kept).
    assert "evaluator_gate:R1:info→moderate" in final.evidence
    for line in candidate.evidence:
        assert line in final.evidence
    # Non-mutated audit-trail inputs preserved.
    assert final.llm_input is candidate.llm_input
    assert final.confidence == candidate.confidence


def test_flag_on_input_verdict_not_mutated(monkeypatch, voter):
    """The gate returns a NEW frozen verdict; the candidate is untouched."""
    monkeypatch.setenv(EVALUATOR_GATE_ENABLED_ENV, "1")
    audit = _audit(satisfied=False, missed=("temporal_filter",))
    candidate, final = _vote_info_candidate(voter, audit)
    assert candidate.severity == "info"  # candidate object unchanged
    assert candidate.gate_rule_fired is None
    assert final is not candidate


# ---------------------------------------------------------------------------
# (c) Fail-open
# ---------------------------------------------------------------------------


def test_flag_on_fail_open_when_audit_none(monkeypatch, voter):
    """No evaluator_audit (evaluator disabled) ⇒ no flip even with flag on."""
    monkeypatch.setenv(EVALUATOR_GATE_ENABLED_ENV, "1")
    candidate, final = _vote_info_candidate(voter, None)
    assert candidate.severity == "info"
    assert final.severity == "info"
    assert final.gate_rule_fired is None
    assert final.decided_by != "evaluator_gate"


def test_flag_on_fail_open_when_evaluator_errored(monkeypatch, voter):
    """satisfied=None is the runner's signal for an evaluator exception ⇒
    fail-open: no flip."""
    monkeypatch.setenv(EVALUATOR_GATE_ENABLED_ENV, "1")
    audit = _audit(satisfied=None, missed=("temporal_filter",))
    candidate, final = _vote_info_candidate(voter, audit)
    assert candidate.severity == "info"
    assert final.severity == "info"
    assert final.gate_rule_fired is None


def test_flag_on_no_fire_when_satisfied_true(monkeypatch, voter):
    """Evaluator satisfied ⇒ R1 does not fire ⇒ no flip."""
    monkeypatch.setenv(EVALUATOR_GATE_ENABLED_ENV, "1")
    audit = _audit(satisfied=True, missed=())
    candidate, final = _vote_info_candidate(voter, audit)
    assert candidate.severity == "info"
    assert final.severity == "info"
    assert final.gate_rule_fired is None


def test_flag_on_no_fire_when_no_missed_considerations(monkeypatch, voter):
    """R1 requires >=1 missed consideration; zero ⇒ no fire."""
    monkeypatch.setenv(EVALUATOR_GATE_ENABLED_ENV, "1")
    audit = _audit(satisfied=False, missed=())
    candidate, final = _vote_info_candidate(voter, audit)
    assert candidate.severity == "info"
    assert final.severity == "info"
    assert final.gate_rule_fired is None


# ---------------------------------------------------------------------------
# (d) flag=1 but candidate severity != info ⇒ no flip
# ---------------------------------------------------------------------------


def test_flag_on_high_candidate_not_relabeled_as_gate(monkeypatch, voter):
    """A candidate the voter scored "high" (valid leak LLM role) is NOT
    relabeled decided_by="evaluator_gate" — the gate only marks itself when it
    actually flips an info candidate. Even with a dissatisfied evaluator."""
    monkeypatch.setenv(EVALUATOR_GATE_ENABLED_ENV, "1")
    audit = _audit(satisfied=False, missed=("temporal_filter",))
    # "mediator" is a valid leak role → voter scores severity="high".
    llm = _llm_with_audit(audit, role="mediator")
    candidate = voter._vote_candidate("feat_x", llm_verdict=llm)
    final = voter.vote("feat_x", llm_verdict=llm)
    assert candidate.severity == "high"
    assert final.severity == "high"
    assert final.decided_by == candidate.decided_by  # stays "llm"
    assert final.decided_by != "evaluator_gate"
    assert final.gate_rule_fired is None
    assert "evaluator_gate:R1:info→moderate" not in final.evidence


def test_flag_on_real_moderate_candidate_no_flip(monkeypatch, voter):
    """A genuine *moderate* candidate (adversarial-alone, no LLM verdict) is no
    longer the gate's precondition (which is now ``info``) — and in production
    it carries no audit anyway. Confirm the gate leaves it untouched: this is
    the exact state the pre-reframe gate wrongly targeted."""
    monkeypatch.setenv(EVALUATOR_GATE_ENABLED_ENV, "1")
    candidate = voter._vote_candidate("feat_x", adversarial_verdict=_adv_moderate())
    final = voter.vote("feat_x", adversarial_verdict=_adv_moderate())
    assert candidate.severity == "moderate"
    assert candidate.llm_input is None  # no audit on this path, by construction
    assert final.severity == "moderate"
    assert final.gate_rule_fired is None
    assert final.decided_by != "evaluator_gate"
