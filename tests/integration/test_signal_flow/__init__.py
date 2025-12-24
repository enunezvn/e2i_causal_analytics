"""
E2I Signal Flow Integration Tests

This package contains integration tests for the DSPy signal flow between agents.

Test Batches:
1. test_sender_signals.py - Sender agent signal generation
2. test_signal_collection.py - Signal collection and batching
3. test_hub_coordination.py - Orchestrator/feedback_learner hub coordination
4. test_recipient_prompts.py - Recipient prompt distribution
5. test_e2e_signal_flow.py - End-to-end signal flow tests

Run individual batches:
    pytest tests/integration/test_signal_flow/test_sender_signals.py -v
    pytest tests/integration/test_signal_flow/test_signal_collection.py -v
    pytest tests/integration/test_signal_flow/test_hub_coordination.py -v
    pytest tests/integration/test_signal_flow/test_recipient_prompts.py -v
    pytest tests/integration/test_signal_flow/test_e2e_signal_flow.py -v

Run all signal flow tests:
    pytest tests/integration/test_signal_flow/ -v
"""
