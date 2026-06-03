"""Leakage Remediation Node - LLM-Assisted Feature Remediation.

When the leakage detector flags CRITICAL or HIGH severity findings, this node
uses Claude to reason about WHY features leak, discover clean alternatives
from the available data, and apply remediation before training proceeds.

This follows the same pattern as qc_remediation.py:
  detect → LLM analysis → automatic fix → re-check loop

Gate N1 (plan v4 §2): each successful remediation pass produces a
``regulatory_adaptation_entry`` dict in the return payload. The entry's
shape matches the ``adaptation_history`` schema consumed by the model_deployer's
``RegulatoryEligibilityAudit`` — i.e. one row per adaptation event with the
keys ``commit_sha``, ``justification_doc``, ``gate_name``,
``before_threshold``, ``after_threshold``, ``timestamp``.

Handoff contract (codex-rescue HIGH-3 / N1-H3): the entry is emitted on the
top-level ``regulatory_adaptation_entry`` state channel AND mirrored into
``scope_spec["regulatory_adaptation_entry"]``. ``scope_spec`` is the carrier the
tier_0 pipeline threads to the model_deployer agent, which forwards it onto its
initial state (it does NOT splat arbitrary top-level keys). The deployer's
last-line-of-defense backstop
(``registry_manager._detect_leftover_adaptation_entries``) reads the entry off
EITHER carrier and FAILS CLOSED — ``regulatory_eligible`` cannot be granted —
whenever the entry has not been ingested into
``validation_metrics["regulatory_eligibility_audit"]["adaptation_history"]``.
The mirror is required because the prior contract assumed an orchestrator
``append_adaptation`` aggregation step that never existed in any production
path, leaving the backstop reading ``None`` and the attestation false.
"""

import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..state import DataPreparerState

logger = logging.getLogger(__name__)

# Maximum number of remediation attempts before giving up
MAX_LEAKAGE_REMEDIATION_ATTEMPTS = 5


# =============================================================================
# Gate N1 (plan v4 §2) — regulatory-eligibility adaptation hook.
# =============================================================================


def _build_regulatory_adaptation_entry(
    *,
    dropped: List[str],
    added: List[str],
    reasoning: str,
) -> Dict[str, Any]:
    """Build one ``adaptation_history`` entry for the model_deployer.

    Plan v4 §2 Gate N1: leakage_remediation drops features that would
    otherwise have been fed to the trainer. That is a pipeline-level
    adaptation — the success-criteria thresholds remain literature-
    anchored, but the feature set the model is allowed to use has been
    relaxed post-hoc to dodge a leakage detection. ANY such adaptation
    in the model's lifecycle disqualifies ``regulatory_eligible=True``
    (codex-rescue HIGH-3); the entry exists so the deployer can read it
    and downgrade the model to ``adapted_regulatory_candidate``.

    Returns a dict matching ``RegulatoryEligibilityAudit.append_adaptation``'s
    keyword args, except the deployer wraps the dict via
    ``adaptation_history.append(entry)`` so we surface it as a plain dict
    here. The ``commit_sha`` is read from ``GIT_COMMIT_SHA`` (CI-set) or
    falls back to ``"unknown"``; same for ``justification_doc`` (env var
    ``REMEDIATION_JUSTIFICATION_DOC`` if set, else a default placeholder).

    Args:
        dropped: list of features the remediation pass dropped.
        added: list of replacement features the remediation pass kept.
        reasoning: LLM/rule-based reasoning string from the analysis.

    Returns:
        A dict with the six adaptation-entry keys.
    """
    return {
        "commit_sha": os.environ.get("GIT_COMMIT_SHA", "unknown"),
        "justification_doc": os.environ.get(
            "REMEDIATION_JUSTIFICATION_DOC",
            "data_preparer/leakage_remediation (auto-generated; "
            "supply REMEDIATION_JUSTIFICATION_DOC env var to override)",
        ),
        "gate_name": "leakage_remediation_feature_drop",
        "before_threshold": {
            "leaked_features_count": len(dropped),
            "dropped_features": list(dropped),
        },
        "after_threshold": {
            "remediated_features_count": len(added),
            "added_features": list(added),
        },
        "timestamp": datetime.now(tz=None).isoformat(),
    }


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================


async def review_and_remediate_leakage(state: DataPreparerState) -> Dict[str, Any]:
    """Review leakage findings and attempt remediation using LLM analysis.

    This node:
    1. Analyzes structured leakage findings from detect_leakage
    2. Uses Claude to classify root causes and propose alternatives
    3. Drops leaked features and validates replacements
    4. Returns updated DataFrames with clean feature sets

    When severity is not CRITICAL/HIGH, this is a no-op pass-through.

    Args:
        state: Current agent state with leakage detection results

    Returns:
        Updated state with remediation results or pass-through
    """
    severity = state.get("leakage_severity", "none")
    attempts = state.get("leakage_remediation_attempts", 0)
    leaked = state.get("leaked_features", [])

    # No-op if leakage is not severe enough to warrant remediation
    if severity not in ("critical", "high"):
        logger.info(f"Leakage severity '{severity}' does not require remediation")
        return {"leakage_remediation_status": "not_needed"}

    # No-op if skip_leakage_check is set
    if state.get("skip_leakage_check", False):
        logger.warning("Leakage remediation skipped per configuration")
        return {"leakage_remediation_status": "not_needed"}

    # Guard against infinite loops — but preserve work from prior rounds
    if attempts >= MAX_LEAKAGE_REMEDIATION_ATTEMPTS:
        previous_features = state.get("leakage_remediated_features", [])
        if previous_features:
            logger.warning(
                f"Max remediation attempts ({MAX_LEAKAGE_REMEDIATION_ATTEMPTS}) reached. "
                f"Proceeding with {len(previous_features)} features from prior rounds."
            )
            return {
                "leakage_remediation_status": "max_attempts_reached",
                "leakage_remediation_attempts": attempts,
                "leakage_remediation_viable": True,
                "requires_leakage_revalidation": False,
                "leakage_remediation_reasoning": (
                    f"Exhausted {MAX_LEAKAGE_REMEDIATION_ATTEMPTS} remediation attempts. "
                    f"Proceeding with {len(previous_features)} remediated features "
                    f"from prior rounds: {', '.join(previous_features[:5])}"
                    f"{'...' if len(previous_features) > 5 else ''}."
                ),
            }
        else:
            logger.warning(
                f"Max remediation attempts ({MAX_LEAKAGE_REMEDIATION_ATTEMPTS}) reached "
                "with no viable features from any round — halting pipeline."
            )
            return {
                "leakage_remediation_status": "failed",
                "leakage_remediation_viable": False,
                "leakage_remediation_reasoning": (
                    f"Exhausted {MAX_LEAKAGE_REMEDIATION_ATTEMPTS} remediation attempts. "
                    f"Leaked features ({', '.join(leaked)}) could not be remediated."
                ),
            }

    logger.info(
        f"Leakage remediation: severity={severity}, "
        f"leaked_features={leaked}, attempt={attempts + 1}"
    )

    try:
        # Step 1: Gather context for LLM
        context = _gather_leakage_context(state)

        # Step 1.5: Deterministic pre-drop for unambiguous CRITICAL leakage
        auto_dropped, auto_classifications = _deterministic_pre_drop(context)
        if auto_dropped:
            logger.info(f"Deterministic pre-drop: {auto_dropped}")
            # Remove auto-dropped features from context so LLM only sees ambiguous cases
            context["leaked_features"] = [
                f for f in context["leaked_features"] if f not in auto_dropped
            ]
            context["leakage_findings"] = [
                f for f in context["leakage_findings"] if f.get("feature", "") not in auto_dropped
            ]

        # Step 2: Check cache, then analyze with LLM (with rule-based fallback)
        cache_key = _compute_cache_key(context)
        analysis = _load_cached_analysis(cache_key)
        if analysis is None:
            if context["leaked_features"]:
                analysis = await _analyze_leakage_with_llm(context)
            else:
                # All leaked features were auto-dropped, no LLM needed
                analysis = {
                    "leakage_classifications": {},
                    "features_to_drop": [],
                    "replacement_candidates": [],
                    "recommended_feature_set": [
                        p["name"]
                        for p in context["column_profiles"]
                        if p["name"] != context["target_variable"]
                        and p["name"] not in auto_dropped
                        and p["null_pct"] <= 50
                        and p["nunique"] > 1
                    ],
                    "viable": True,
                    "confidence": "high",
                    "reasoning": "All leaked features had unambiguous CRITICAL leakage and were auto-dropped.",
                }
            _save_cached_analysis(cache_key, analysis)

        # Merge auto-drop results back into the analysis
        if auto_dropped:
            analysis["leakage_classifications"].update(auto_classifications)
            for feat in auto_dropped:
                if feat not in analysis["features_to_drop"]:
                    analysis["features_to_drop"].append(feat)
            # Remove auto-dropped features from recommended set
            analysis["recommended_feature_set"] = [
                f for f in analysis["recommended_feature_set"] if f not in auto_dropped
            ]

        # Step 2.6 — Declared-safe immunity companion. The adaptive_validity_check
        # node already exempts manifest-declared-safe (pre-index) features from
        # leakage and strips them from leaked_features, but the LLM remediator
        # reasons over ALL columns and can narratively add a contract-certified
        # feature to its drop list anyway. Full manifest immunity (user decision,
        # 2026-06-03) means such a feature is NEVER dropped — strip declared-safe
        # features from the LLM drop list. Statistical governance still applies to
        # un-contracted features. No-op when no manifest source resolved.
        manifest_source = (state.get("scope_spec") or {}).get("feature_manifest_source")
        if manifest_source:
            from .adaptive_validity_check import _declared_safe_immune_features

            immune = _declared_safe_immune_features(
                set(analysis.get("features_to_drop", [])), manifest_source
            )
            if immune:
                analysis["features_to_drop"] = [
                    f for f in analysis["features_to_drop"] if f not in immune
                ]
                recommended = analysis.get("recommended_feature_set", [])
                for feat in sorted(immune):
                    if feat not in recommended:
                        recommended.append(feat)
                analysis["recommended_feature_set"] = recommended
                logger.warning(
                    "Declared-safe immunity (remediation): kept %d manifest pre-index "
                    "feature(s) off the LLM drop list: %s",
                    len(immune),
                    sorted(immune),
                )

        # Step 3: Apply remediation to DataFrames
        result = _apply_leakage_remediation(state, analysis)

        if result["success"]:
            logger.info(
                f"Leakage remediation applied: dropped={result['dropped']}, "
                f"final_features={result['final_features']}"
            )
            # Gate N1 (plan v4 §2 codex-rescue HIGH-3): record this
            # remediation pass as an adaptation entry. Downstream the
            # model_deployer reads the cumulative ``adaptation_history``
            # from ``validation_metrics["regulatory_eligibility_audit"]``
            # and disqualifies ``regulatory_eligible=True`` if ANY entry
            # exists. The deployer downgrades the model to
            # ``adapted_regulatory_candidate=True`` if absolute
            # thresholds still clear at promotion time.
            adaptation_entry = _build_regulatory_adaptation_entry(
                dropped=result["dropped"],
                added=result["added"],
                reasoning=analysis.get("reasoning", ""),
            )
            logger.info(
                "Gate N1: emitted regulatory_adaptation_entry "
                f"(dropped={len(result['dropped'])}, "
                f"added={len(result['added'])}, "
                f"gate={adaptation_entry['gate_name']})"
            )
            # Mirror the entry into scope_spec so it survives the agent
            # boundary to the deployer (the deployer agent forwards scope_spec
            # onto its initial state; see module docstring's handoff contract).
            # Merge into the existing scope_spec rather than replace it so
            # cohort identity (feature_manifest_source, prediction_target, ...)
            # is preserved on the channel write.
            updated_scope_spec = dict(state.get("scope_spec") or {})
            updated_scope_spec["regulatory_adaptation_entry"] = adaptation_entry
            return {
                "leakage_remediation_status": "applied",
                # Gate N1: mirror onto scope_spec carrier (see above).
                "scope_spec": updated_scope_spec,
                "leakage_remediation_attempts": attempts + 1,
                "leakage_remediated_features": result["final_features"],
                "leakage_dropped_features": result["dropped"],
                "leakage_added_features": result["added"],
                "leakage_remediation_reasoning": analysis.get("reasoning", ""),
                "leakage_remediation_viable": result["viable"],
                "requires_leakage_revalidation": True,
                # Gate N1 (plan v4 §2): one adaptation entry per remediation
                # pass; the deployer aggregates these across all data_preparer
                # invocations during a model's lifecycle.
                "regulatory_adaptation_entry": adaptation_entry,
                # Updated DataFrames with leaked columns removed
                "train_df": result["train_df"],
                "validation_df": result["validation_df"],
                "test_df": result["test_df"],
                "holdout_df": result["holdout_df"],
                # Clear previous leakage state so re-check starts fresh.
                # Asymmetry note (backlog #11.d): we clear `leakage_findings`
                # here so detect_leakage's next pass starts from scratch, but
                # the Layer 5 `adaptive_verdicts` produced by the next
                # adaptive_validity_check invocation are CUMULATIVE across
                # re-entries (see adaptive_validity_check.py merge block).
                # An audit-trail reader correlating the two streams must
                # account for the fact that a feature in `adaptive_verdicts`
                # from the first invocation may be absent from
                # `leakage_findings` after this clear.
                "leakage_detected": False,
                "leakage_issues": [],
                "leakage_findings": [],
                "leakage_severity": "none",
                "leaked_features": [],
                # Update blocking_issues: remove leakage-related entries
                "blocking_issues": [
                    issue
                    for issue in (state.get("blocking_issues") or [])
                    if not any(lf in issue for lf in leaked)
                ],
            }
        else:
            logger.warning(f"Leakage remediation failed: {result.get('reason')}")
            return {
                "leakage_remediation_status": "failed",
                "leakage_remediation_attempts": attempts + 1,
                "leakage_remediation_viable": False,
                "leakage_remediation_reasoning": result.get("reason", "Unknown failure"),
            }

    except Exception as e:
        logger.error(f"Leakage remediation error: {e}", exc_info=True)
        return {
            "leakage_remediation_status": "error",
            "leakage_remediation_attempts": attempts + 1,
            "leakage_remediation_viable": False,
            "leakage_remediation_reasoning": f"Remediation error: {e}",
        }


# =============================================================================
# CONTEXT GATHERING
# =============================================================================


def _gather_leakage_context(state: DataPreparerState) -> Dict[str, Any]:
    """Assemble context for LLM analysis.

    Args:
        state: Current agent state

    Returns:
        Context dictionary with leakage findings, available columns, and metadata
    """
    train_df = state.get("train_df")
    scope_spec = state.get("scope_spec", {})
    target_variable = scope_spec.get("prediction_target", "")

    # Build column profiles from training data
    column_profiles: List[Dict[str, Any]] = []
    if train_df is not None:
        for col in train_df.columns:
            try:
                nunique = int(train_df[col].nunique())
            except TypeError:
                # Columns with unhashable types (lists, dicts) can't compute nunique
                nunique = -1
            profile: Dict[str, Any] = {
                "name": col,
                "dtype": str(train_df[col].dtype),
                "dtype_kind": train_df[col].dtype.kind,
                "null_pct": round(train_df[col].isna().mean() * 100, 1),
                "nunique": nunique,
                "n_rows": len(train_df),
            }
            if train_df[col].dtype.kind in "iufb":
                all_null = train_df[col].isna().all()
                profile["min"] = float(train_df[col].min()) if not all_null else None
                profile["max"] = float(train_df[col].max()) if not all_null else None
                profile["mean"] = round(float(train_df[col].mean()), 4) if not all_null else None
                profile["std"] = round(float(train_df[col].std()), 4) if not all_null else None
            elif train_df[col].dtype == object and nunique > 0 and nunique <= 20:
                profile["top_values"] = train_df[col].value_counts(dropna=False).head(5).to_dict()
            column_profiles.append(profile)

    # Target distribution
    target_dist = {}
    if train_df is not None and target_variable and target_variable in train_df.columns:
        target_dist = train_df[target_variable].value_counts().to_dict()
        # Convert numpy types to native Python for JSON serialization
        target_dist = {str(k): int(v) for k, v in target_dist.items()}

    return {
        "leaked_features": state.get("leaked_features", []),
        "leakage_findings": state.get("leakage_findings", []),
        "leakage_severity": state.get("leakage_severity", "none"),
        "column_profiles": column_profiles,
        "target_variable": target_variable,
        "target_distribution": target_dist,
        "problem_type": scope_spec.get("problem_type", "unknown"),
        "current_feature_list": scope_spec.get("required_features", []),
        "experiment_id": state.get("experiment_id", "unknown"),
    }


# =============================================================================
# DETERMINISTIC PRE-DROP
# =============================================================================


def _deterministic_pre_drop(context: Dict[str, Any]) -> Tuple[List[str], Dict[str, str]]:
    """Auto-drop features with unambiguous CRITICAL leakage.

    Handles cases where no LLM judgment is needed:
    - logical_dependency with CRITICAL severity (logically equivalent to target)
    - perfect_class_separation with overlap == 0.0

    Args:
        context: Leakage context from _gather_leakage_context

    Returns:
        Tuple of (auto_dropped_features, classifications_dict)
    """
    auto_drop: List[str] = []
    classifications: Dict[str, str] = {}

    for finding in context.get("leakage_findings", []):
        feat = finding.get("feature", "")
        check = finding.get("check_name", "")
        severity = finding.get("severity", "").lower()
        evidence = finding.get("evidence", {})

        if not feat:
            continue

        # Case 1: logical_dependency with CRITICAL severity
        if check == "logical_dependency" and severity == "critical":
            if feat not in auto_drop:
                auto_drop.append(feat)
                classifications[feat] = (
                    "tautological — logically equivalent to target (auto-dropped)"
                )
                logger.info(f"Auto-dropping '{feat}': logical_dependency CRITICAL")

        # Case 2: perfect_class_separation with zero overlap
        elif check == "perfect_class_separation" and severity == "critical":
            overlap = evidence.get("overlap", None)
            if overlap is not None and float(overlap) == 0.0:
                if feat not in auto_drop:
                    auto_drop.append(feat)
                    classifications[feat] = (
                        "tautological — perfect class separation with zero overlap (auto-dropped)"
                    )
                    logger.info(f"Auto-dropping '{feat}': perfect_class_separation overlap=0.0")

    return auto_drop, classifications


# =============================================================================
# REMEDIATION CACHE
# =============================================================================


def _compute_cache_key(context: Dict[str, Any]) -> str:
    """Compute a SHA-256 cache key from the leakage context.

    Args:
        context: Leakage context

    Returns:
        Hex digest string
    """
    import hashlib
    import json

    key_data = {
        "leaked_features": sorted(context.get("leaked_features", [])),
        "target_variable": context.get("target_variable", ""),
        "column_names": sorted([p["name"] for p in context.get("column_profiles", [])]),
        "finding_signatures": sorted(
            [
                f"{f.get('check_name', '')}:{f.get('feature', '')}:{f.get('severity', '')}"
                for f in context.get("leakage_findings", [])
            ]
        ),
    }
    key_str = json.dumps(key_data, sort_keys=True)
    return hashlib.sha256(key_str.encode()).hexdigest()


def _get_cache_dir() -> Path:
    """Get the cache directory for leakage remediation."""
    from pathlib import Path

    cache_dir = Path(".cache") / "leakage_remediation"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def _load_cached_analysis(key: str) -> Optional[Dict[str, Any]]:
    """Load a cached remediation analysis if it exists.

    Args:
        key: SHA-256 cache key

    Returns:
        Cached analysis dict or None
    """
    import json

    cache_file = _get_cache_dir() / f"{key}.json"
    if cache_file.exists():
        try:
            with open(cache_file) as f:
                cached = json.load(f)
            logger.info(f"Loaded cached leakage remediation: {cache_file.name}")
            return cached
        except Exception as e:
            logger.warning(f"Failed to load cache {cache_file}: {e}")
    return None


def _save_cached_analysis(key: str, analysis: Dict[str, Any]) -> None:
    """Save a remediation analysis to cache.

    Only stores JSON-serializable fields.

    Args:
        key: SHA-256 cache key
        analysis: Analysis dict to cache
    """
    import json

    cacheable_fields = {
        "leakage_classifications",
        "features_to_drop",
        "replacement_candidates",
        "recommended_feature_set",
        "viable",
        "confidence",
        "reasoning",
    }
    cache_data = {k: v for k, v in analysis.items() if k in cacheable_fields}

    cache_file = _get_cache_dir() / f"{key}.json"
    try:
        with open(cache_file, "w") as f:
            json.dump(cache_data, f, indent=2)
        logger.info(f"Cached leakage remediation: {cache_file.name}")
    except Exception as e:
        logger.warning(f"Failed to save cache {cache_file}: {e}")


# =============================================================================
# LLM ANALYSIS
# =============================================================================


async def _analyze_leakage_with_llm(context: Dict[str, Any]) -> Dict[str, Any]:
    """Use Claude to analyze leakage findings and propose remediation.

    Falls back to rule-based analysis if the API is unavailable.

    Args:
        context: Leakage context from _gather_leakage_context

    Returns:
        Parsed analysis with classifications, recommendations, and reasoning
    """
    try:
        import anthropic

        client = anthropic.AsyncAnthropic()
        prompt = _build_leakage_analysis_prompt(context)

        response = await client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=2048,
            temperature=0.0,
            messages=[{"role": "user", "content": prompt}],
        )

        content_block = response.content[0]
        if not hasattr(content_block, "text"):
            logger.warning(f"Unexpected response type: {type(content_block)}")
            return _rule_based_leakage_analysis(context)

        return _parse_leakage_analysis(content_block.text, context)

    except Exception as e:
        logger.warning(f"LLM leakage analysis failed ({e}), using rule-based fallback")
        return _rule_based_leakage_analysis(context)


def _build_leakage_analysis_prompt(context: Dict[str, Any]) -> str:
    """Build the analysis prompt for Claude.

    Args:
        context: Leakage context

    Returns:
        Formatted prompt string
    """
    leaked = context["leaked_features"]
    findings = context["leakage_findings"]
    profiles = context["column_profiles"]
    target = context["target_variable"]
    target_dist = context["target_distribution"]
    problem_type = context["problem_type"]

    # Format findings
    findings_text = ""
    for f in findings:
        sev = f.get("severity", "unknown").upper()
        findings_text += (
            f"  - [{sev}] {f.get('check_name', '?')}: feature='{f.get('feature', '?')}'\n"
            f"    Description: {f.get('description', 'N/A')}\n"
            f"    Evidence: {f.get('evidence', {})}\n"
            f"    Recommendation: {f.get('recommendation', 'N/A')}\n"
        )

    # Format column profiles (excluding target and already-leaked features)
    available_text = ""
    for p in profiles:
        name = p["name"]
        if name == target or name in leaked:
            continue
        dtype = p["dtype"]
        null_pct = p["null_pct"]
        nunique = p["nunique"]
        stats = ""
        if "mean" in p and p["mean"] is not None:
            stats = f" | mean={p['mean']}, std={p['std']}, min={p['min']}, max={p['max']}"
        elif "top_values" in p:
            top = list(p["top_values"].items())[:3]
            stats = f" | top values: {top}"
        available_text += (
            f"  - {name} (dtype={dtype}, null={null_pct}%, nunique={nunique}{stats})\n"
        )

    return f"""You are a data leakage expert analyzing an ML pipeline that has detected critical data leakage.

## Detected Leakage (overall severity: {context["leakage_severity"]})

Leaked features: {leaked}

Findings:
{findings_text}

## Target Variable
Name: {target}
Problem type: {problem_type}
Distribution: {target_dist}

## Available Columns (excluding target and leaked features)
{available_text}

## Task

Analyze the leakage and propose remediation. For each section below, output the section header exactly as shown, followed by your analysis.

LEAKAGE_CLASSIFICATION:
For each leaked feature, classify WHY it leaks. Use one line per feature in the format:
- feature_name: classification (tautological/temporal/proxy/contamination), brief explanation

FEATURES_TO_DROP:
List all features that must be removed. One per line:
- feature_name

REPLACEMENT_CANDIDATES:
From the available columns, identify potential replacement features. Consider:
- Is the feature legitimately available at prediction time?
- Could it have its own leakage issues (e.g., derived from the target)?
- Is it informative enough to contribute to the model?
One per line:
- column_name: risk_level (clean/borderline/risky), brief rationale

RECOMMENDED_FEATURE_SET:
List the final recommended features for training. One per line:
- column_name

VIABLE: yes or no — is the recommended feature set sufficient for a useful model?
CONFIDENCE: high, medium, or low
REASONING: One paragraph summarizing your remediation logic — why features leak, what alternatives exist, and whether the remediated model is worth training."""


def _parse_leakage_analysis(text: str, context: Dict[str, Any]) -> Dict[str, Any]:
    """Parse the structured LLM response.

    Args:
        text: Raw LLM response text
        context: Original context for fallback

    Returns:
        Structured analysis dict
    """
    analysis: Dict[str, Any] = {
        "leakage_classifications": {},
        "features_to_drop": [],
        "replacement_candidates": [],
        "recommended_feature_set": [],
        "viable": False,
        "confidence": "low",
        "reasoning": "",
    }

    current_section: Optional[str] = None

    for line in text.split("\n"):
        stripped = line.strip()
        if not stripped:
            continue

        # Once in REASONING, accumulate all remaining text without
        # re-parsing section headers (the LLM may echo them in prose).
        if current_section == "reasoning":
            if analysis["reasoning"]:
                analysis["reasoning"] += " " + stripped
            else:
                analysis["reasoning"] = stripped
            continue

        # Section headers (only parsed when NOT in reasoning section)
        if stripped.startswith("LEAKAGE_CLASSIFICATION"):
            current_section = "classifications"
            continue
        elif stripped.startswith("FEATURES_TO_DROP"):
            current_section = "drop"
            continue
        elif stripped.startswith("REPLACEMENT_CANDIDATES"):
            current_section = "candidates"
            continue
        elif stripped.startswith("RECOMMENDED_FEATURE_SET"):
            current_section = "recommended"
            continue
        elif stripped.startswith("VIABLE:"):
            value = stripped.replace("VIABLE:", "").strip().lower()
            analysis["viable"] = value in ("yes", "true", "1")
            current_section = None
            continue
        elif stripped.startswith("CONFIDENCE:"):
            analysis["confidence"] = stripped.replace("CONFIDENCE:", "").strip().lower()
            current_section = None
            continue
        elif stripped.startswith("REASONING:"):
            analysis["reasoning"] = stripped.replace("REASONING:", "").strip()
            current_section = "reasoning"
            continue

        # Parse items within current section
        if current_section == "classifications" and stripped.startswith("- "):
            item = stripped[2:]
            if ":" in item:
                feat, rest = item.split(":", 1)
                analysis["leakage_classifications"][feat.strip()] = rest.strip()
        elif current_section == "drop" and stripped.startswith("- "):
            feat = stripped[2:].strip()
            if feat:
                analysis["features_to_drop"].append(feat)
        elif current_section == "candidates" and stripped.startswith("- "):
            item = stripped[2:]
            if ":" in item:
                col, rest = item.split(":", 1)
                analysis["replacement_candidates"].append(
                    {
                        "column": col.strip(),
                        "assessment": rest.strip(),
                    }
                )
        elif current_section == "recommended" and stripped.startswith("- "):
            feat = stripped[2:].strip()
            if feat:
                analysis["recommended_feature_set"].append(feat)

    # Ensure features_to_drop includes all leaked features even if LLM missed some
    for feat in context.get("leaked_features", []):
        if feat not in analysis["features_to_drop"]:
            analysis["features_to_drop"].append(feat)

    logger.info(
        f"Parsed LLM leakage analysis: "
        f"drop={analysis['features_to_drop']}, "
        f"recommended={analysis['recommended_feature_set']}, "
        f"viable={analysis['viable']}"
    )

    return analysis


# =============================================================================
# RULE-BASED FALLBACK
# =============================================================================


def _rule_based_leakage_analysis(context: Dict[str, Any]) -> Dict[str, Any]:
    """Deterministic fallback when Claude API is unavailable.

    Applies conservative rules:
    - Drop all CRITICAL/HIGH features
    - Accept numeric columns with >5% class overlap and <0.95 abs correlation
    - Report viability based on >=2 clean features remaining

    Args:
        context: Leakage context

    Returns:
        Analysis dict matching the LLM output structure
    """

    leaked = set(context.get("leaked_features", []))
    target = context["target_variable"]
    profiles = context["column_profiles"]
    findings = context.get("leakage_findings", [])

    # Classify leaked features from findings
    classifications = {}
    for f in findings:
        feat = f.get("feature", "")
        check = f.get("check_name", "")
        sev = f.get("severity", "")
        if feat in leaked:
            if check == "logical_dependency":
                classifications[feat] = "tautological — logically equivalent to target"
            elif check == "perfect_class_separation":
                classifications[feat] = "tautological — perfectly separates target classes"
            elif check == "zero_variance_within_class":
                classifications[feat] = "tautological — constant in one class"
            elif check == "mutual_information":
                classifications[feat] = "proxy — implausibly high mutual information"
            elif check == "target_correlation":
                classifications[feat] = "proxy — near-perfect correlation with target"
            elif check == "single_feature_auc":
                classifications[feat] = "proxy — single feature achieves high AUC against target"
            elif check == "categorical_class_separation":
                classifications[feat] = (
                    "proxy — categorical feature strongly associated with target"
                )
            else:
                classifications[feat] = f"{check} — {sev} severity"

    # Discover candidate replacements from available columns (numeric + low-cardinality categorical)
    candidates = []
    recommended = []

    for p in profiles:
        name = p["name"]
        if name == target or name in leaked:
            continue
        if p["null_pct"] > 50:
            continue
        if p["nunique"] <= 1:
            continue

        dtype_kind = p.get("dtype_kind", "")
        is_numeric = dtype_kind in "iufb"
        is_categorical = not is_numeric and p["nunique"] <= 50

        if not is_numeric and not is_categorical:
            continue

        # Run leakage checks on this candidate if we have train_df info
        risk = "clean"
        col_type = "numeric" if is_numeric else f"categorical ({p['nunique']} categories)"
        rationale = f"{col_type}, {p['null_pct']}% null, {p['nunique']} unique values"

        candidates.append({"column": name, "assessment": f"{risk}, {rationale}"})
        recommended.append(name)

    viable = len(recommended) >= 2
    reasoning = (
        f"Dropped {len(leaked)} leaked features ({', '.join(sorted(leaked))}). "
        f"Found {len(recommended)} candidate replacement features from available data. "
        f"{'Feature set is viable for training.' if viable else 'Insufficient features remaining — enrichment needed.'}"
    )

    return {
        "leakage_classifications": classifications,
        "features_to_drop": sorted(leaked),
        "replacement_candidates": candidates,
        "recommended_feature_set": recommended,
        "viable": viable,
        "confidence": "medium" if viable else "low",
        "reasoning": reasoning,
    }


# =============================================================================
# APPLY REMEDIATION
# =============================================================================


def _apply_leakage_remediation(
    state: DataPreparerState,
    analysis: Dict[str, Any],
) -> Dict[str, Any]:
    """Apply the remediation plan to the DataFrames.

    Drops leaked features and validates that recommended replacements exist
    and pass structural leakage checks.

    Args:
        state: Current agent state with DataFrames
        analysis: Parsed LLM analysis

    Returns:
        Result dict with updated DataFrames and metadata
    """
    from .leakage_detector import (
        _aggregate_severity,
        check_categorical_class_separation,
        check_feature_target_logical_dependency,
        check_mutual_information,
        check_perfect_class_separation,
        check_single_feature_auc,
        check_zero_variance_within_class,
    )

    train_df = state.get("train_df")
    validation_df = state.get("validation_df")
    test_df = state.get("test_df")
    holdout_df = state.get("holdout_df")
    scope_spec = state.get("scope_spec", {})
    target_variable = scope_spec.get("prediction_target", "")

    if train_df is None:
        return {"success": False, "reason": "No training data available"}

    features_to_drop = analysis.get("features_to_drop", [])
    recommended = analysis.get("recommended_feature_set", [])

    # Validate recommended features exist in the data and are ML-compatible
    # Reject columns with unhashable types (lists, dicts) or pure strings
    available_cols = set(train_df.columns)
    valid_recommended = []
    for f in recommended:
        if f not in available_cols or f == target_variable:
            continue
        col = train_df[f]
        # Accept numeric/boolean columns
        if col.dtype.kind in "iufb":
            valid_recommended.append(f)
            continue
        # Reject object columns that contain unhashable types (lists, dicts)
        try:
            col.nunique()
            valid_recommended.append(f)
        except TypeError:
            logger.info(f"Rejecting '{f}': contains unhashable types (lists/dicts)")
            continue

    if not valid_recommended:
        available_sample = sorted(available_cols)[:10]
        return {
            "success": False,
            "viable": False,
            "reason": (
                f"No valid replacement features found. "
                f"Recommended {recommended} but available columns (first 10): {available_sample}"
            ),
        }

    # Run structural leakage checks on each recommended feature
    verified_features: List[str] = []
    rejected_features: List[str] = []

    for feat in valid_recommended:
        if feat not in train_df.columns or target_variable not in train_df.columns:
            rejected_features.append(feat)
            continue

        combined = train_df[[feat, target_variable]].dropna()
        if len(combined) < 30:
            verified_features.append(feat)  # Too few rows to check, accept
            continue

        if combined[feat].dtype.kind not in "iufb":
            # Non-numeric: run categorical class separation check
            try:
                cat_findings = check_categorical_class_separation(combined, target_variable, [feat])
                cat_severity = _aggregate_severity(cat_findings)
                if cat_severity in ("critical", "high"):
                    logger.info(
                        f"Categorical candidate '{feat}' has {cat_severity} association with target — rejecting"
                    )
                    rejected_features.append(feat)
                else:
                    verified_features.append(feat)
            except Exception as e:
                logger.warning(f"Categorical check failed for '{feat}': {e}")
                verified_features.append(feat)  # Accept on check failure
            continue

        # Run the 5 structural checks (4 original + single-feature AUC)
        check_findings = []
        numeric_feats = [feat]
        try:
            check_findings.extend(
                check_perfect_class_separation(combined, target_variable, numeric_feats)
            )
            check_findings.extend(
                check_zero_variance_within_class(combined, target_variable, numeric_feats)
            )
            check_findings.extend(
                check_mutual_information(combined, target_variable, numeric_feats)
            )
            check_findings.extend(
                check_feature_target_logical_dependency(combined, target_variable, numeric_feats)
            )
            check_findings.extend(
                check_single_feature_auc(combined, target_variable, numeric_feats)
            )
        except Exception as e:
            logger.warning(f"Leakage check failed for '{feat}': {e}")
            verified_features.append(feat)  # Accept on check failure
            continue

        severity = _aggregate_severity(check_findings)
        if severity in ("critical", "high"):
            logger.info(f"Replacement candidate '{feat}' also has {severity} leakage — rejecting")
            rejected_features.append(feat)
        else:
            verified_features.append(feat)

    if len(verified_features) < 2:
        return {
            "success": False,
            "viable": False,
            "reason": (
                f"Only {len(verified_features)} clean features after validation "
                f"(need >= 2). Verified: {verified_features}, "
                f"rejected: {rejected_features}"
            ),
        }

    # Combined-feature AUC sanity check (cross-validated): if all verified
    # features together still produce AUC > 0.95, try backward elimination
    if len(verified_features) >= 2 and target_variable in train_df.columns:
        try:
            from sklearn.linear_model import LogisticRegression
            from sklearn.metrics import roc_auc_score
            from sklearn.model_selection import StratifiedKFold, cross_val_predict
            from sklearn.preprocessing import StandardScaler

            numeric_verified = [
                f
                for f in verified_features
                if f in train_df.columns and train_df[f].dtype.kind in "iufb"
            ]
            if len(numeric_verified) >= 2:
                combined_check = train_df[numeric_verified + [target_variable]].dropna()
                if len(combined_check) >= 50:
                    X_check = combined_check[numeric_verified].values
                    y_check = combined_check[target_variable].values.astype(float)

                    def _cv_auc(X: "np.ndarray", y: "np.ndarray") -> float:
                        """Cross-validated AUC using out-of-fold predictions."""
                        minority_count = int(min(y.sum(), len(y) - y.sum()))
                        n_splits = min(5, max(2, minority_count))
                        scaler = StandardScaler()
                        X_scaled = scaler.fit_transform(X)
                        lr = LogisticRegression(max_iter=200, solver="lbfgs", random_state=42)
                        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
                        y_prob = cross_val_predict(lr, X_scaled, y, cv=cv, method="predict_proba")[
                            :, 1
                        ]
                        return roc_auc_score(y, y_prob)

                    combined_auc = _cv_auc(X_check, y_check)
                    if combined_auc > 0.95:
                        logger.warning(
                            f"Combined verified features achieve CV AUC={combined_auc:.3f} "
                            f"(threshold 0.95) — attempting backward elimination"
                        )
                        # Backward elimination: iteratively remove the most
                        # suspicious feature (highest individual AUC) until
                        # combined AUC drops below threshold
                        remaining = list(numeric_verified)
                        for _elim_round in range(min(10, len(remaining) - 2)):
                            # Compute individual CV AUC for each remaining feature
                            individual_aucs = {}
                            for feat in remaining:
                                X_single = combined_check[[feat]].values
                                y_single = y_check
                                try:
                                    individual_aucs[feat] = _cv_auc(X_single, y_single)
                                except Exception:
                                    individual_aucs[feat] = 0.5

                            worst_feat = max(individual_aucs, key=individual_aucs.get)
                            worst_auc = individual_aucs[worst_feat]
                            logger.info(
                                f"Elimination round {_elim_round + 1}: removing "
                                f"'{worst_feat}' (individual AUC={worst_auc:.3f})"
                            )
                            remaining.remove(worst_feat)
                            rejected_features.append(worst_feat)

                            if len(remaining) < 2:
                                break

                            X_reduced = combined_check[remaining].values
                            combined_auc = _cv_auc(X_reduced, y_check)
                            logger.info(f"  Remaining CV AUC={combined_auc:.3f} with {remaining}")

                            if combined_auc <= 0.95:
                                break

                        # Update verified_features to reflect eliminations
                        verified_features = [f for f in verified_features if f in remaining]

                        if combined_auc > 0.95 or len(verified_features) < 2:
                            return {
                                "success": False,
                                "viable": False,
                                "reason": (
                                    f"After backward elimination, remaining features "
                                    f"still achieve CV AUC={combined_auc:.3f} (> 0.95) "
                                    f"or too few remain ({len(verified_features)}). "
                                    f"Remaining: {verified_features}, "
                                    f"eliminated: {rejected_features}"
                                ),
                            }
                        logger.info(
                            f"Backward elimination succeeded: CV AUC={combined_auc:.3f} "
                            f"with {len(verified_features)} features: {verified_features}"
                        )
                    else:
                        logger.info(
                            f"Combined-feature CV AUC sanity check passed: AUC={combined_auc:.3f}"
                        )
        except Exception as e:
            logger.warning(f"Combined AUC sanity check failed (non-fatal): {e}")

    # Determine which columns to keep in the DataFrames
    # Keep: target, verified features, and non-feature columns (IDs, metadata)
    # We only drop the explicitly leaked features — other columns stay for downstream use
    cols_to_drop = [c for c in features_to_drop if c in available_cols and c != target_variable]
    # Also drop any recommended features that were rejected by our checks
    cols_to_drop.extend([c for c in rejected_features if c in available_cols])
    cols_to_drop = list(set(cols_to_drop))

    # Apply drops to all splits
    updated_train = train_df.drop(columns=cols_to_drop, errors="ignore")
    updated_val = (
        validation_df.drop(columns=cols_to_drop, errors="ignore")
        if validation_df is not None
        else None
    )
    updated_test = (
        test_df.drop(columns=cols_to_drop, errors="ignore") if test_df is not None else None
    )
    updated_holdout = (
        holdout_df.drop(columns=cols_to_drop, errors="ignore") if holdout_df is not None else None
    )

    # Determine which features were added (not in original leaked set)
    original_leaked = set(state.get("leaked_features", []))
    added = [f for f in verified_features if f not in original_leaked]

    logger.info(
        f"Leakage remediation applied: "
        f"dropped={cols_to_drop}, verified={verified_features}, "
        f"rejected={rejected_features}, added={added}"
    )

    return {
        "success": True,
        "viable": True,
        "dropped": cols_to_drop,
        "final_features": verified_features,
        "added": added,
        "rejected": rejected_features,
        "train_df": updated_train,
        "validation_df": updated_val,
        "test_df": updated_test,
        "holdout_df": updated_holdout,
    }
