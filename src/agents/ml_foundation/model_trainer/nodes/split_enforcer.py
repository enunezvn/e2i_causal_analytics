"""Split enforcement for model_trainer.

This module validates ML split ratios and detects potential data leakage.
"""

from typing import Any, Dict


async def enforce_splits(state: Dict[str, Any]) -> Dict[str, Any]:
    """Validate ML split ratios and detect data leakage.

    CRITICAL: This enforces the E2I split policy:
    - Train: 60% ± 2%
    - Validation: 20% ± 2%
    - Test: 15% ± 2%
    - Holdout: 5% ± 2%

    This also checks for potential data leakage by ensuring splits are properly
    isolated.

    Args:
        state: ModelTrainerState with split data and ratios

    Returns:
        Dictionary with split_ratios_valid, split_validation_message,
        leakage_warnings

    Raises:
        No exceptions - returns error in state if validation fails
    """
    # Extract split ratios
    train_ratio = state.get("train_ratio", 0.0)
    validation_ratio = state.get("validation_ratio", 0.0)
    test_ratio = state.get("test_ratio", 0.0)
    holdout_ratio = state.get("holdout_ratio", 0.0)

    # Extract sample counts
    train_samples = state.get("train_samples", 0)
    validation_samples = state.get("validation_samples", 0)
    test_samples = state.get("test_samples", 0)
    holdout_samples = state.get("holdout_samples", 0)
    total_samples = state.get("total_samples", 0)

    # Expected ratios (E2I ML Foundation split policy)
    expected_ratios = {
        "train": 0.60,
        "validation": 0.20,
        "test": 0.15,
        "holdout": 0.05,
    }
    tolerance = 0.02  # ±2%
    # Add small epsilon for floating point comparison (0.62 - 0.60 may be 0.0200000001)
    epsilon = 1e-9

    # Validate each split ratio
    ratio_checks = []
    ratios_valid = True

    # Train split
    if abs(train_ratio - expected_ratios["train"]) > tolerance + epsilon:
        ratio_checks.append(
            f"Train split ratio {train_ratio:.2%} outside expected "
            f"{expected_ratios['train']:.0%} ± {tolerance:.0%}"
        )
        ratios_valid = False
    else:
        ratio_checks.append(f"Train split: {train_ratio:.2%} ✓")

    # Validation split
    if abs(validation_ratio - expected_ratios["validation"]) > tolerance + epsilon:
        ratio_checks.append(
            f"Validation split ratio {validation_ratio:.2%} outside expected "
            f"{expected_ratios['validation']:.0%} ± {tolerance:.0%}"
        )
        ratios_valid = False
    else:
        ratio_checks.append(f"Validation split: {validation_ratio:.2%} ✓")

    # Test split
    if abs(test_ratio - expected_ratios["test"]) > tolerance + epsilon:
        ratio_checks.append(
            f"Test split ratio {test_ratio:.2%} outside expected "
            f"{expected_ratios['test']:.0%} ± {tolerance:.0%}"
        )
        ratios_valid = False
    else:
        ratio_checks.append(f"Test split: {test_ratio:.2%} ✓")

    # Holdout split
    if abs(holdout_ratio - expected_ratios["holdout"]) > tolerance + epsilon:
        ratio_checks.append(
            f"Holdout split ratio {holdout_ratio:.2%} outside expected "
            f"{expected_ratios['holdout']:.0%} ± {tolerance:.0%}"
        )
        ratios_valid = False
    else:
        ratio_checks.append(f"Holdout split: {holdout_ratio:.2%} ✓")

    # Check for minimum sample sizes
    min_samples_per_split = 10  # Minimum viable samples
    leakage_warnings = []

    if train_samples < min_samples_per_split:
        leakage_warnings.append(
            f"Train split has only {train_samples} samples (minimum: {min_samples_per_split})"
        )
        ratios_valid = False

    if validation_samples < min_samples_per_split:
        leakage_warnings.append(
            f"Validation split has only {validation_samples} samples (minimum: {min_samples_per_split})"
        )
        ratios_valid = False

    if test_samples < min_samples_per_split:
        leakage_warnings.append(
            f"Test split has only {test_samples} samples (minimum: {min_samples_per_split})"
        )
        ratios_valid = False

    # Check holdout - warn if zero (holdout is optional but important for final validation)
    if holdout_samples == 0 and holdout_ratio > 0:
        leakage_warnings.append(f"Holdout split has 0 samples but ratio is {holdout_ratio:.2%}")
        ratios_valid = False
    elif holdout_samples == 0:
        leakage_warnings.append(
            "Holdout split has 0 samples - final model validation will be limited"
        )
        ratios_valid = False

    # Check if total adds up (should be close to 1.0)
    total_ratio = train_ratio + validation_ratio + test_ratio + holdout_ratio
    if abs(total_ratio - 1.0) > 0.01:
        leakage_warnings.append(
            f"Split ratios sum to {total_ratio:.4f}, expected 1.0 (possible overlap or missing data)"
        )
        ratios_valid = False

    # TODO: Advanced leakage detection
    # - Check for duplicate row indices across splits
    # - Validate temporal ordering (if time-series)
    # - Check for feature leakage (target in features)
    # This requires access to actual data, not just metadata

    # Construct validation message
    if ratios_valid:
        split_validation_message = (
            f"Split ratios validated: "
            f"{train_ratio:.1%}/{validation_ratio:.1%}/{test_ratio:.1%}/{holdout_ratio:.1%} "
            f"({total_samples:,} total samples)"
        )
    else:
        split_validation_message = (
            f"Split ratio validation FAILED: "
            f"{', '.join(ratio_checks[:2])}. "
            f"Warnings: {len(leakage_warnings)}"
        )

    result = {
        "split_ratios_valid": ratios_valid,
        "split_validation_message": split_validation_message,
        "split_ratio_checks": ratio_checks,
        "leakage_warnings": leakage_warnings if leakage_warnings else [],
    }

    # Set error when validation fails so agent can detect it
    if not ratios_valid:
        failed_checks = [c for c in ratio_checks if "✓" not in c]
        result["error"] = (
            f"Split validation failed: {split_validation_message}. "
            f"Failed checks: {', '.join(failed_checks[:3])}"
        )
        result["error_type"] = "split_validation_error"

    return result
