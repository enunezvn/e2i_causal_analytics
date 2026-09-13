"""#1991 debt 4: gate fields are Literals of their own vocabulary."""

import pytest
from pydantic import ValidationError

from src.api.schemas.causal import RefutationSummary


def test_refutation_gate_rejects_discovery_tokens():
    with pytest.raises(ValidationError):
        RefutationSummary(gate_decision="reject")
    assert RefutationSummary(gate_decision="block").gate_decision == "block"


def test_expert_review_decision_is_its_own_vocabulary():
    with pytest.raises(ValidationError):
        RefutationSummary(expert_review_decision="review")
    ok = RefutationSummary(expert_review_decision="pending_review")
    assert ok.expert_review_decision == "pending_review"
