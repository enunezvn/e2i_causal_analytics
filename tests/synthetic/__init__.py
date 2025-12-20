"""Synthetic Benchmark Suite for Causal Validation Protocol.

Version: 4.3
Purpose: Known ground-truth datasets for CI/CD regression testing

This module provides synthetic causal datasets with known true effects
to validate that the causal inference engine can correctly recover
causal relationships. These tests serve as regression tests in CI/CD.

Benchmark Scenarios:
    - simple_linear: Baseline sanity check (True ATE = +0.50, no confounding)
    - confounded_moderate: Adjustment recovery (True ATE = +0.30, 1 confounder)
    - heterogeneous_cate: CATE estimation (varies by segment, 2 confounders)

Reference: docs/E2I_Causal_Validation_Protocol.html
"""
