"""Class imbalance detection for model_trainer.

This module detects class imbalance in training data and selects an
optimal remediation strategy via a deterministic, YAML-configured
decision matrix. The previous LLM-based implementation has been
replaced (Tier-0 Block 6A, Findings #9 + #16) so the same input always
produces the same `(strategy, rationale)` output.

The strategy matrix is loaded from
``config/imbalance_strategy.yaml`` (relative to the project root).
Tests can override the path by calling
:func:`_load_imbalance_config(path=...)`.

Configuration contract (asserted at load time)
----------------------------------------------
* ``severity_bands`` must define ``none``, ``moderate``, ``severe`` and the
  values must be strictly descending and positive
  (``none > moderate > severe > 0``).
* Each non_tree branch in ``strategy_matrix`` may be a single
  ``{strategy, rationale}`` dict, OR a list of
  ``{min_minority_count, strategy, rationale}`` rules. List-form
  branches MUST include a ``min_minority_count: 0`` rule as the
  catch-all (otherwise low-count cases fall through silently). Every
  list-form rule must provide all three keys.
* Empty list-form branches are rejected.

Version: 2.0.0
"""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import yaml

from src.utils.project_root import find_project_root

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration loader
# ---------------------------------------------------------------------------

# Path to the canonical imbalance-strategy config relative to repo root.
# Resolved lazily so unit tests can override via `_load_imbalance_config`.
_DEFAULT_CONFIG_PATH = (
    find_project_root() / "config" / "imbalance_strategy.yaml"
)

# Type alias for a single decision-matrix leaf.
StrategyLeaf = Dict[str, Any]
StrategyEntry = Union[StrategyLeaf, List[StrategyLeaf]]

# Required keys for every list-form non_tree rule. Missing any of these
# raises ``ValueError`` at load time so misconfigurations surface before
# the resolver hits a query path.
_REQUIRED_RULE_KEYS: frozenset[str] = frozenset(
    {"min_minority_count", "strategy", "rationale"}
)

# Valid remediation strategies. Kept as a module-level frozenset so
# downstream consumers (e.g. apply_resampling) can validate against it
# without loading the YAML and so the set is unmistakably immutable.
VALID_STRATEGIES: frozenset[str] = frozenset(
    {
        "smote",  # Synthetic minority oversampling
        "random_oversample",  # Duplicate minority samples
        "random_undersample",  # Remove majority samples
        "smote_tomek",  # SMOTE + Tomek links cleaning
        "class_weight",  # Use class weights only (no resampling)
        "combined",  # Moderate resampling + class weights
        "none",  # No action needed
    }
)


def _normalize_non_tree_rules(
    entry: StrategyEntry, *, severity: str, branch_name: str
) -> StrategyEntry:
    """Sort and validate list-form rules.

    First-match-wins lookup at query time depends on rules being ordered
    from most-restrictive to least-restrictive. We sort defensively at
    load time so the YAML can be authored in any order.

    Validation enforced here (raised as ``ValueError``):

    * empty lists — list-form branches must declare at least one rule
      so the resolver does not blow up on ``entry[-1]``;
    * per-rule structure — every rule must declare every key in
      :data:`_REQUIRED_RULE_KEYS` so the resolver does not raise
      ``KeyError`` deep in a query path;
    * catch-all contract — at least one rule must declare
      ``min_minority_count: 0`` so the resolver always lands on a
      defined strategy regardless of the runtime minority count.
    """
    if not isinstance(entry, list):
        return entry

    location = f"strategy_matrix[{severity!r}][{branch_name!r}]"

    if not entry:
        raise ValueError(
            f"{location} is an empty list; list-form branches must "
            "declare at least one rule (and a min_minority_count: 0 "
            "catch-all)."
        )

    for index, rule in enumerate(entry):
        if not isinstance(rule, dict):
            raise ValueError(
                f"{location}[{index}] must be a mapping with keys "
                f"{sorted(_REQUIRED_RULE_KEYS)!r}; got "
                f"{type(rule).__name__}."
            )
        missing = _REQUIRED_RULE_KEYS - rule.keys()
        if missing:
            raise ValueError(
                f"{location}[{index}] is missing required key(s): "
                f"{sorted(missing)!r}. Every list-form rule must "
                f"provide {sorted(_REQUIRED_RULE_KEYS)!r}."
            )

    if not any(int(rule["min_minority_count"]) == 0 for rule in entry):
        raise ValueError(
            f"{location} has no catch-all rule; one of its rules must "
            "declare min_minority_count: 0 so low-count cases land on a "
            "defined strategy."
        )

    return sorted(
        entry,
        key=lambda rule: int(rule["min_minority_count"]),
        reverse=True,
    )


@lru_cache(maxsize=8)
def _read_yaml(path_str: str) -> Dict[str, Any]:
    """Cached YAML reader keyed by absolute path string.

    Cache key is a string so :class:`pathlib.Path` instances passed by
    callers normalize cleanly. ``functools.lru_cache`` does not accept
    unhashable defaults, so the public loader resolves paths before
    delegating here.

    Normalisation (sorting list-form rules, validating per-rule
    structure, enforcing the catch-all contract) is performed HERE so
    the cached payload is already-normalised — the public loader does
    not mutate the cached dict in place on subsequent calls.
    """
    with open(path_str, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ValueError(
            f"imbalance_strategy config at {path_str} must be a mapping, "
            f"got {type(data).__name__}"
        )

    matrix = data.get("strategy_matrix")
    if isinstance(matrix, dict):
        for severity, branches in matrix.items():
            if not isinstance(branches, dict):
                continue
            for branch_name, branch in branches.items():
                branches[branch_name] = _normalize_non_tree_rules(
                    branch, severity=severity, branch_name=branch_name
                )

    return data


def _load_imbalance_config(path: Optional[Path] = None) -> Dict[str, Any]:
    """Load and validate the imbalance-strategy configuration.

    Args:
        path: Optional override for the YAML config path. When None,
            the canonical path under ``config/imbalance_strategy.yaml``
            is used.

    Returns:
        Parsed config dict. Strategy-matrix list-form rules are
        already sorted (descending by ``min_minority_count``) and
        validated by ``_read_yaml``; this function adds top-level
        structural checks on top.

    Raises:
        FileNotFoundError: if the config file is missing.
        ValueError: if required keys are missing or malformed.
    """
    target = Path(path) if path is not None else _DEFAULT_CONFIG_PATH
    raw = _read_yaml(str(target.resolve()))

    # Validate top-level structure
    for required in ("severity_bands", "tree_models", "strategy_matrix"):
        if required not in raw:
            raise ValueError(
                f"imbalance_strategy config missing required key: {required!r}"
            )

    bands = raw["severity_bands"]
    if not all(k in bands for k in ("none", "moderate", "severe")):
        raise ValueError(
            "severity_bands must define 'none', 'moderate', and 'severe'"
        )

    none_band = float(bands["none"])
    moderate_band = float(bands["moderate"])
    severe_band = float(bands["severe"])
    if not (none_band > moderate_band > severe_band > 0.0):
        raise ValueError(
            "severity_bands must satisfy none > moderate > severe > 0; "
            f"got none={none_band!r}, moderate={moderate_band!r}, "
            f"severe={severe_band!r}."
        )

    return raw


def _resolve_strategy_leaf(
    entry: StrategyEntry,
    minority_count: int,
) -> StrategyLeaf:
    """Resolve a strategy entry to a single ``{strategy, rationale}`` leaf.

    For dict entries, the entry itself is the leaf. For list entries,
    rules are scanned in descending ``min_minority_count`` order and the
    first whose threshold is met (``minority_count >= min_minority_count``)
    wins.
    """
    if isinstance(entry, dict):
        return entry
    if isinstance(entry, list):
        for rule in entry:  # already sorted descending by loader
            if minority_count >= int(rule.get("min_minority_count", 0)):
                return rule
        # Fallback: last rule (lowest threshold). Should be unreachable
        # when YAML includes a `min_minority_count: 0` rule.
        return entry[-1]
    raise ValueError(
        f"strategy_matrix entry must be dict or list, got {type(entry).__name__}"
    )


# ---------------------------------------------------------------------------
# Severity + metric calculation
# ---------------------------------------------------------------------------


def _calculate_imbalance_metrics(
    y: np.ndarray,
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Calculate class imbalance metrics.

    Args:
        y: Target labels.
        config: Optional pre-loaded config (so tests can supply alternate
            severity bands without monkey-patching). When None, the
            canonical config is loaded once and cached.

    Returns:
        Dictionary with imbalance metrics. ``severity`` is bound to the
        bands declared in the active config.
    """
    if config is None:
        config = _load_imbalance_config()

    bands = config["severity_bands"]

    unique, counts = np.unique(y, return_counts=True)
    class_distribution = dict(
        zip(unique.astype(int).tolist(), counts.tolist(), strict=False)
    )

    total = len(y)
    minority_count = min(counts)
    majority_count = max(counts)
    minority_class = unique[np.argmin(counts)]
    majority_class = unique[np.argmax(counts)]

    minority_ratio = minority_count / total
    imbalance_ratio = majority_count / minority_count if minority_count > 0 else float("inf")

    # Determine severity from configured bands.
    if minority_ratio >= bands["none"]:
        severity = "none"
    elif minority_ratio >= bands["moderate"]:
        severity = "moderate"
    elif minority_ratio >= bands["severe"]:
        severity = "severe"
    else:
        severity = "extreme"

    return {
        "class_distribution": class_distribution,
        "minority_count": int(minority_count),
        "majority_count": int(majority_count),
        "minority_class": int(minority_class),
        "majority_class": int(majority_class),
        "minority_ratio": float(minority_ratio),
        "imbalance_ratio": float(imbalance_ratio),
        "severity": severity,
        "total_samples": total,
    }


# Backward-compat module-level dict for callers that still read the
# severity thresholds (e.g. tests/synthetic/test_adverse_regime.py
# references SEVERITY_THRESHOLDS in a comment). Populated lazily on
# first config load so import-time errors are deferred.
SEVERITY_THRESHOLDS: Dict[str, float] = {}

try:
    _bootstrap_cfg = _load_imbalance_config()
    SEVERITY_THRESHOLDS = {
        "none": float(_bootstrap_cfg["severity_bands"]["none"]),
        "moderate": float(_bootstrap_cfg["severity_bands"]["moderate"]),
        "severe": float(_bootstrap_cfg["severity_bands"]["severe"]),
        "extreme": 0.0,  # implicit lowest band; kept for backward-compat
    }
except Exception as _exc:  # pragma: no cover - bootstrap best-effort
    logger.warning(
        "Could not bootstrap SEVERITY_THRESHOLDS from imbalance_strategy.yaml: %s",
        _exc,
    )


# ---------------------------------------------------------------------------
# Strategy lookup
# ---------------------------------------------------------------------------


def _lookup_strategy(
    metrics: Dict[str, Any],
    algorithm_name: str,
    problem_type: str,
    config: Optional[Dict[str, Any]] = None,
) -> tuple[str, str]:
    """Deterministic strategy lookup against the YAML decision matrix.

    Args:
        metrics: Imbalance metrics from :func:`_calculate_imbalance_metrics`.
            Must include ``severity`` and ``minority_count``.
        algorithm_name: Algorithm being trained (e.g. "XGBoost").
        problem_type: Problem type. Today only ``"binary_classification"``
            is supported; the public node :func:`detect_class_imbalance`
            already short-circuits ``"regression"``/``"continuous"`` before
            reaching this lookup, so the check here is a safety net rather
            than a behaviour change.
        config: Optional pre-loaded config; defaults to the canonical
            file load. Tests pass an override here to exercise alternate
            matrices.

    Returns:
        Tuple of (strategy, rationale). ``strategy`` is guaranteed to be
        in :data:`VALID_STRATEGIES`.
    """
    if problem_type != "binary_classification":
        raise ValueError(
            f"_lookup_strategy only supports problem_type='binary_classification'; "
            f"got {problem_type!r}. The public detect_class_imbalance node "
            "is responsible for short-circuiting other problem types before "
            "reaching the matrix."
        )

    if config is None:
        config = _load_imbalance_config()

    severity = metrics["severity"]
    minority_count = int(metrics["minority_count"])

    matrix = config["strategy_matrix"]
    if severity not in matrix:
        raise ValueError(
            f"strategy_matrix has no entry for severity={severity!r}; "
            f"available: {sorted(matrix.keys())}"
        )

    branches = matrix[severity]
    tree_models = set(config.get("tree_models", []))
    is_tree_model = algorithm_name in tree_models

    # Branch resolution order:
    #   1. severity == "none" -> use 'default' branch (single global rule)
    #   2. otherwise -> 'tree' if tree model, 'non_tree' if not
    #   3. fallback to 'default' if the specific branch is absent
    if severity == "none":
        entry = branches.get("default") or branches.get("tree") or branches.get("non_tree")
    elif is_tree_model:
        entry = branches.get("tree") or branches.get("default") or branches.get("non_tree")
    else:
        entry = branches.get("non_tree") or branches.get("default") or branches.get("tree")

    if entry is None:
        raise ValueError(
            f"strategy_matrix[{severity!r}] has no usable branch for "
            f"algorithm={algorithm_name!r} (tree={is_tree_model})"
        )

    leaf = _resolve_strategy_leaf(entry, minority_count)

    strategy = str(leaf["strategy"]).strip().lower()
    rationale = str(leaf["rationale"]).strip()

    if strategy not in VALID_STRATEGIES:
        raise ValueError(
            f"strategy {strategy!r} from matrix is not in VALID_STRATEGIES "
            f"(must be one of {sorted(VALID_STRATEGIES)!r}); check "
            "config/imbalance_strategy.yaml."
        )

    return strategy, rationale


# ---------------------------------------------------------------------------
# LangGraph node entrypoint
# ---------------------------------------------------------------------------


async def detect_class_imbalance(state: Dict[str, Any]) -> Dict[str, Any]:
    """Detect class imbalance in training data and recommend remediation.

    This LangGraph node:

    1. Analyzes class distribution in the training labels.
    2. Classifies severity (none/moderate/severe/extreme) using the
       configurable bands from ``config/imbalance_strategy.yaml``.
    3. Picks a deterministic remediation strategy from the matrix in the
       same YAML file. **No LLM call** — replaced in Block 6A so two
       runs with identical inputs always produce identical outputs.
    4. Returns the chosen strategy for the downstream resampling node.

    The function is kept ``async`` to preserve LangGraph's node
    signature; no internal awaits remain after Block 6A.

    Args:
        state: ModelTrainerState with ``train_data``.

    Returns:
        Dictionary with imbalance_detected, imbalance_ratio, minority_ratio,
        imbalance_severity, class_distribution, recommended_strategy,
        strategy_rationale.
    """
    # Extract training labels
    train_data = state.get("train_data", {})
    y_train = train_data.get("y")
    algorithm_name = state.get("algorithm_name", "Unknown")
    problem_type = state.get("problem_type", "binary_classification")

    if y_train is None:
        logger.warning("No training labels available for imbalance detection")
        return {
            "imbalance_detected": False,
            "imbalance_ratio": 1.0,
            "minority_ratio": 0.5,
            "imbalance_severity": "unknown",
            "class_distribution": {},
            "recommended_strategy": "none",
            "strategy_rationale": "No training data available for imbalance detection",
        }

    # Convert to numpy if needed
    if hasattr(y_train, "values"):
        y_train = y_train.values
    y_train = np.asarray(y_train).flatten()

    # Check for regression problem (no class imbalance applicable)
    if problem_type in ["regression", "continuous"]:
        logger.info("Regression problem - class imbalance not applicable")
        return {
            "imbalance_detected": False,
            "imbalance_ratio": 1.0,
            "minority_ratio": 0.5,
            "imbalance_severity": "not_applicable",
            "class_distribution": {},
            "recommended_strategy": "none",
            "strategy_rationale": "Class imbalance detection not applicable for regression",
        }

    # Check for unique classes
    unique_classes = np.unique(y_train)
    if len(unique_classes) < 2:
        logger.warning(f"Only {len(unique_classes)} class(es) found in training data")
        return {
            "imbalance_detected": False,
            "imbalance_ratio": float("inf"),
            "minority_ratio": 0.0,
            "imbalance_severity": "degenerate",
            "class_distribution": {int(c): int(np.sum(y_train == c)) for c in unique_classes},
            "recommended_strategy": "none",
            "strategy_rationale": "Insufficient classes for imbalance analysis",
        }

    # Load config once per call so test overrides flow through.
    config = _load_imbalance_config()

    # Calculate imbalance metrics
    metrics = _calculate_imbalance_metrics(y_train, config=config)

    logger.info(
        f"Class imbalance analysis: severity={metrics['severity']}, "
        f"minority_ratio={metrics['minority_ratio']:.2%}, "
        f"imbalance_ratio={metrics['imbalance_ratio']:.1f}:1"
    )

    # Determine if imbalance is detected (anything beyond "none" severity)
    imbalance_detected = metrics["severity"] != "none"

    if not imbalance_detected:
        return {
            "imbalance_detected": False,
            "imbalance_ratio": metrics["imbalance_ratio"],
            "minority_ratio": metrics["minority_ratio"],
            "imbalance_severity": metrics["severity"],
            "class_distribution": metrics["class_distribution"],
            "recommended_strategy": "none",
            "strategy_rationale": "Class distribution is balanced - no remediation needed",
        }

    # Deterministic strategy lookup (no LLM, no async work).
    strategy, rationale = _lookup_strategy(
        metrics,
        algorithm_name,
        problem_type,
        config=config,
    )

    logger.info(f"Recommended strategy: {strategy} - {rationale}")

    return {
        "imbalance_detected": True,
        "imbalance_ratio": metrics["imbalance_ratio"],
        "minority_ratio": metrics["minority_ratio"],
        "imbalance_severity": metrics["severity"],
        "class_distribution": metrics["class_distribution"],
        "recommended_strategy": strategy,
        "strategy_rationale": rationale,
    }
