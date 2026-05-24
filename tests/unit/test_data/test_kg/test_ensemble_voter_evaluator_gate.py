"""Issue #240 Stage 3 — env-gated audit-evaluator soft-gate on EnsembleVoter.

Design reference: ``docs/plans/240-audit-evaluator-gate-promotion.md`` §3 Stage 3
(Mechanism + Fail-open) and §4 R1.

Load-bearing invariants pinned here:

(a) **Default OFF ⇒ byte-identical.** With the kill-switch env var unset, the
    full ``vote`` output (severity / remediation / decided_by / evidence /
    gate_rule_fired) is identical to ``_vote_candidate`` — the gate never
    fires and adds nothing. This is the central "ship the disabled mechanism
    only" guarantee.

(b) **flag=1 + R1 fires + candidate moderate ⇒ flip.** severity moderate→high,
    decided_by="evaluator_gate", the structured evidence tag is appended, and
    ``gate_rule_fired == "R1"``.

(c) **Fail-open.** flag=1 but the worker carried no evaluator_audit (evaluator
    disabled) ⇒ no flip. Same for an evaluator error (``satisfied is None``).

(d) **flag=1 but candidate severity != moderate ⇒ no flip.**

The voter never produces ``severity="moderate"`` from a *valid* LLM role
(``_llm_severity`` maps valid leak roles → "high", accept roles → "info"); the
moderate candidate the gate acts on is the **adversarial-moderate-alone** path
(``sanitised_llm is None`` because the LLM role was outside the supported
vocabulary, while ``llm_verdict`` — and its ``evaluator_audit`` — are still
carried in). These tests construct exactly that arrangement.
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

    ``role`` defaults to ``"ancestor"`` (a *valid* role). To reach the
    voter's adversarial-moderate candidate path we pass an INVALID role so
    the voter sanitises the LLM to None (skips the LLM path) while still
    carrying ``llm_input`` — and thus the evaluator audit — into the verdict.
    """
    return LLMVerdict(
        causal_role=role,  # type: ignore[arg-type]
        mechanism="m",
        recommended_remediation="keep_with_caveat",
        evaluator_audit=audit,
    )


def _adv_moderate() -> dict:
    """Adversarial-moderate verdict dict (mirror of _build_verdict shape)."""
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


def _vote_moderate_candidate(voter: EnsembleVoter, audit) -> tuple:
    """Return (candidate, final) verdicts for a moderate adversarial path
    carrying an LLM verdict with the given evaluator audit.

    The LLM role is INVALID ("collider_invalid") so ``sanitised_llm`` is None
    and the voter falls through to the adversarial-moderate-alone path
    (candidate severity="moderate") while still carrying ``llm_input``.
    """
    llm = _llm_with_audit(audit, role="not_a_real_role")
    candidate = voter._vote_candidate(
        "feat_x",
        adversarial_verdict=_adv_moderate(),
        llm_verdict=llm,
    )
    final = voter.vote(
        "feat_x",
        adversarial_verdict=_adv_moderate(),
        llm_verdict=llm,
    )
    return candidate, final


# ---------------------------------------------------------------------------
# (a) Default OFF ⇒ byte-identical
# ---------------------------------------------------------------------------


@pytest.fixture
def voter() -> EnsembleVoter:
    return EnsembleVoter()


def test_candidate_path_actually_reaches_moderate(monkeypatch, voter):
    """Sanity: the arrangement under test really yields a moderate candidate.

    If this regresses (the voter stops producing moderate here), the flip
    test below would pass vacuously, so pin it explicitly."""
    monkeypatch.delenv(EVALUATOR_GATE_ENABLED_ENV, raising=False)
    audit = _audit(satisfied=False, missed=("temporal_filter",))
    candidate, _final = _vote_moderate_candidate(voter, audit)
    assert candidate.severity == "moderate", (
        f"test setup must produce a moderate candidate; got {candidate.severity!r}"
    )
    assert candidate.gate_rule_fired is None


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
    candidate, final = _vote_moderate_candidate(voter, audit)
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
    candidate, final = _vote_moderate_candidate(voter, audit)
    assert final.severity == candidate.severity == "moderate"
    assert final.gate_rule_fired is None
    assert final.decided_by != "evaluator_gate"


# ---------------------------------------------------------------------------
# (b) flag=1 + R1 fires + candidate moderate ⇒ flip
# ---------------------------------------------------------------------------


def test_flag_on_r1_fires_flips_to_high(monkeypatch, voter):
    monkeypatch.setenv(EVALUATOR_GATE_ENABLED_ENV, "1")
    audit = _audit(satisfied=False, missed=("temporal_filter",))
    candidate, final = _vote_moderate_candidate(voter, audit)

    # Candidate was moderate; gate escalated it.
    assert candidate.severity == "moderate"
    assert final.severity == "high"
    assert final.remediation == "drop"
    assert final.decided_by == "evaluator_gate"
    assert final.gate_rule_fired == "R1"
    # Structured evidence tag appended (and the candidate's evidence kept).
    assert "evaluator_gate:R1:moderate→high" in final.evidence
    for line in candidate.evidence:
        assert line in final.evidence
    # Non-mutated audit-trail inputs preserved.
    assert final.llm_input is candidate.llm_input
    assert final.confidence == candidate.confidence


def test_flag_on_input_verdict_not_mutated(monkeypatch, voter):
    """The gate returns a NEW frozen verdict; the candidate is untouched."""
    monkeypatch.setenv(EVALUATOR_GATE_ENABLED_ENV, "1")
    audit = _audit(satisfied=False, missed=("temporal_filter",))
    candidate, final = _vote_moderate_candidate(voter, audit)
    assert candidate.severity == "moderate"  # candidate object unchanged
    assert candidate.gate_rule_fired is None
    assert final is not candidate


# ---------------------------------------------------------------------------
# (c) Fail-open
# ---------------------------------------------------------------------------


def test_flag_on_fail_open_when_audit_none(monkeypatch, voter):
    """No evaluator_audit (evaluator disabled) ⇒ no flip even with flag on."""
    monkeypatch.setenv(EVALUATOR_GATE_ENABLED_ENV, "1")
    candidate, final = _vote_moderate_candidate(voter, None)
    assert candidate.severity == "moderate"
    assert final.severity == "moderate"
    assert final.gate_rule_fired is None
    assert final.decided_by != "evaluator_gate"


def test_flag_on_fail_open_when_evaluator_errored(monkeypatch, voter):
    """satisfied=None is the runner's signal for an evaluator exception ⇒
    fail-open: no flip."""
    monkeypatch.setenv(EVALUATOR_GATE_ENABLED_ENV, "1")
    audit = _audit(satisfied=None, missed=("temporal_filter",))
    candidate, final = _vote_moderate_candidate(voter, audit)
    assert candidate.severity == "moderate"
    assert final.severity == "moderate"
    assert final.gate_rule_fired is None


def test_flag_on_no_fire_when_satisfied_true(monkeypatch, voter):
    """Evaluator satisfied ⇒ R1 does not fire ⇒ no flip."""
    monkeypatch.setenv(EVALUATOR_GATE_ENABLED_ENV, "1")
    audit = _audit(satisfied=True, missed=())
    candidate, final = _vote_moderate_candidate(voter, audit)
    assert candidate.severity == "moderate"
    assert final.severity == "moderate"
    assert final.gate_rule_fired is None


def test_flag_on_no_fire_when_no_missed_considerations(monkeypatch, voter):
    """R1 requires >=1 missed consideration; zero ⇒ no fire."""
    monkeypatch.setenv(EVALUATOR_GATE_ENABLED_ENV, "1")
    audit = _audit(satisfied=False, missed=())
    candidate, final = _vote_moderate_candidate(voter, audit)
    assert candidate.severity == "moderate"
    assert final.severity == "moderate"
    assert final.gate_rule_fired is None


# ---------------------------------------------------------------------------
# (d) flag=1 but candidate severity != moderate ⇒ no flip
# ---------------------------------------------------------------------------


def test_flag_on_high_candidate_not_relabeled_as_gate(monkeypatch, voter):
    """A candidate the voter independently scored "high" (valid leak LLM role)
    is NOT relabeled decided_by="evaluator_gate" — the gate only marks itself
    when it actually flips a moderate. Even with a dissatisfied evaluator."""
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
    assert "evaluator_gate:R1:moderate→high" not in final.evidence


def test_flag_on_info_candidate_no_flip(monkeypatch, voter):
    """Accept-role LLM → severity="info"; gate's moderate precondition fails."""
    monkeypatch.setenv(EVALUATOR_GATE_ENABLED_ENV, "1")
    audit = _audit(satisfied=False, missed=("temporal_filter",))
    llm = _llm_with_audit(audit, role="ancestor")  # accept role → info
    final = voter.vote("feat_x", llm_verdict=llm)
    assert final.severity == "info"
    assert final.gate_rule_fired is None
    assert final.decided_by != "evaluator_gate"
