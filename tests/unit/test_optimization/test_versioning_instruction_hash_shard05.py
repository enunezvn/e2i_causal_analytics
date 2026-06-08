"""Shard 05: save_optimized_module must capture optimized instructions (dspy 3.x).

dspy 3.x predictors expose `signature.instructions` (not `extended_signature`),
so the old extraction produced an empty instruction set and every saved version
hashed to the empty-string digest (silently breaking dedup). Offline (no LM).
"""

from __future__ import annotations

import pytest

dspy = pytest.importorskip("dspy")

_EMPTY_HASH = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"


def test_instruction_hash_reflects_optimized_instructions(tmp_path):
    from src.agents.feedback_learner.dspy_integration import PatternDetectionSignature
    from src.optimization.gepa import save_optimized_module

    module = dspy.ChainOfThought(PatternDetectionSignature)
    # Simulate an optimized signature (what GEPA produces).
    pred = module.predictors()[0]
    pred.signature = pred.signature.with_instructions(
        "OPTIMIZED: lead with the highest-severity accuracy pattern."
    )

    info = save_optimized_module(module, agent_name="vh_test", output_dir=str(tmp_path))
    assert info["instruction_hash"] != _EMPTY_HASH

    # Two distinct instruction texts must hash differently (dedup works).
    pred.signature = pred.signature.with_instructions("DIFFERENT: be terse.")
    info2 = save_optimized_module(module, agent_name="vh_test", output_dir=str(tmp_path))
    assert info2["instruction_hash"] != info["instruction_hash"]
