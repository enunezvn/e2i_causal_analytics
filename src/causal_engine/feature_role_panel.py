"""Feature-role panel: the four feature-role voters, run over a CAUSAL frame.

Lane E of the real-data causal estimation program
(``docs/superpowers/specs/2026-09-22-real-data-causal-estimation-design.md``,
§3 "Lane E", items 1–3(d)).

The ML data-preparer node ``adaptive_validity_check`` already runs four
independent processes that assign each feature a causal role — Layer 1 the
declarative manifest contract, Layer 3 the adversarial permutation probe,
Layer 2 the knowledge-graph cache, Layer 4 the compiled DSPy classifier — and
combines them with ``EnsembleVoter``. Until this lane none of that reached the
causal agent: ``graph_builder`` adjusted for every registry covariate blind.

This module REUSES that node (it is called, not re-implemented) over a causal
frame's covariates for a ``(manifest source, T, Y)`` and returns, per feature,
what every layer said plus the ensemble verdict, in a serialisable shape that
Lane B's structural author consumes and that is recorded as evidence.

Three design points that matter for causal (not predictive) use:

* **Only two things are a leak verdict.** A Layer 1 ``post_index`` contract
  (the column is knowable after the index) or a Layer 3 ``high`` on a feature
  the manifest does NOT declare pre-index (declared-safe immunity is the
  node's own rule: the contract is the temporal arbiter). A strong pre-index
  predictor of Y is kept — in causal terms it is a confounder candidate, and
  the prediction-era instinct to drop it is exactly what this panel must not
  import. The Layer 3 statistic is still recorded as evidence.
* **Layer 2 informs, it does not decide.** The KG runs in ``shadow``: a
  ``treats`` edge between the treatment drug and a pre-index comorbidity is
  indication evidence for the treatment CHOICE (a confounder), and the voter's
  prediction-era vocabulary still calls the signal ``leak_drug_treats_disease``;
  the panel records the signal WITH its supporting edges and leaves the call to
  the author and the reviewer.
* **The causal activation profile is per run.** ``CAUSAL_ACTIVATION_PROFILE``
  turns Layer 4 and the structural decider on for THIS node call only, through
  the node's existing per-run state keys. No global flag is flipped and
  ``ADAPTIVE_LAYER4_LLM_DECIDES`` stays unset, so Layer 4 is audit-only: under
  this design the voters inform, the author and the human decide.

Layer 4 calls a paid LLM. Tests run it on a ``DummyLM``; the real run is an
owner decision (``scripts/measure_feature_role_panel.py --layer4 real``).
"""

from __future__ import annotations

import asyncio
import logging
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import pandas as pd

logger = logging.getLogger(__name__)

#: The cohort-scoped causal activation profile (spec Lane E item 1). Applied to
#: the node's PER-RUN state / scope_spec by :func:`build_feature_role_panel`;
#: nothing here is a process-wide flag.
CAUSAL_ACTIVATION_PROFILE: Dict[str, Any] = {
    "adaptive_layer4_enabled": True,
    "adaptive_structural_decider_enabled": True,
    "kg_mode": "shadow",
}

LEAK_SOURCE_LAYER_1 = "layer_1_post_index"
LEAK_SOURCE_LAYER_3 = "layer_3_high"
_LEAK_SOURCES = (LEAK_SOURCE_LAYER_1, LEAK_SOURCE_LAYER_3)

#: Serialised-panel schema version. Bump when the record shape changes; the
#: causal-agent route refuses a panel whose version it does not know.
PANEL_SCHEMA_VERSION = "1"

#: What the voter's prediction-era KG signal means in the CAUSAL contrast: a
#: treatment drug approved for a PRE-index condition is indication evidence
#: (the condition drives the treatment choice — a confounder candidate), not
#: evidence that the condition descends from the treatment.
KG_CAUSAL_INTERPRETATION_INDICATION = "indication_evidence"


@dataclass(frozen=True)
class FeatureRoleRecord:
    """What every layer said about ONE covariate, plus the ensemble verdict.

    ``leak_verdict`` is True only for the two causes documented in the module
    docstring; ``leak_source`` names which (``layer_1_post_index`` /
    ``layer_3_high``) or is None.
    """

    feature: str
    layer_1: Dict[str, Any]
    layer_2: Dict[str, Any]
    layer_3: Dict[str, Any]
    layer_4: Dict[str, Any]
    ensemble: Dict[str, Any]
    leak_verdict: bool
    leak_source: Optional[str] = None
    #: True when the leak verdict rests on Layer 3 alone for a column with NO
    #: contract: Layer 3 measures predictiveness of Y, not timing, so the
    #: column's temporal status is unknown. It is excluded from adjustment per
    #: spec 3(b) but must be presented as a review item, never as proven
    #: leakage (codex r1). A post-index contract IS proven: no review flag.
    review_required: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "FeatureRoleRecord":
        return cls(
            feature=str(payload["feature"]),
            layer_1=dict(payload.get("layer_1") or {}),
            layer_2=dict(payload.get("layer_2") or {}),
            layer_3=dict(payload.get("layer_3") or {}),
            layer_4=dict(payload.get("layer_4") or {}),
            ensemble=dict(payload.get("ensemble") or {}),
            leak_verdict=bool(payload.get("leak_verdict", False)),
            leak_source=payload.get("leak_source"),
            review_required=bool(payload.get("review_required", False)),
        )


@dataclass(frozen=True)
class FeatureRolePanel:
    """The panel for one ``(manifest_source, treatment, outcome)`` over a frame."""

    manifest_source: str
    treatment: str
    outcome: str
    n_rows: int
    features: Tuple[str, ...]
    records: Dict[str, FeatureRoleRecord]
    layer_activity: Dict[str, Any]
    activation_profile: Dict[str, Any]
    leakage_fdr: Dict[str, Any] = field(default_factory=dict)
    promotion_eligibility: Dict[str, Any] = field(default_factory=dict)
    built_at: str = ""
    schema_version: str = PANEL_SCHEMA_VERSION

    def leak_features(self) -> List[str]:
        return sorted(name for name, rec in self.records.items() if rec.leak_verdict)

    def validate_strict(self) -> None:
        """Raise ``ValueError`` naming the first invariant a panel violates.

        ``from_dict`` is a structural parse; this is the consistency check the
        causal-agent route runs at submit (codex r3): the panel must be
        internally coherent and every leak verdict must be backed by the
        evidence it claims. It establishes CONSISTENCY, not authenticity — a
        caller-supplied panel stays labelled as such downstream.
        """
        from src.data.manifests import MANIFEST_SOURCES

        if self.schema_version != PANEL_SCHEMA_VERSION:
            raise ValueError(
                f"schema_version {self.schema_version!r} is not the supported "
                f"{PANEL_SCHEMA_VERSION!r}"
            )
        if self.manifest_source not in MANIFEST_SOURCES:
            raise ValueError(
                f"manifest_source {self.manifest_source!r} is not a registered manifest"
            )
        if not self.treatment or not self.outcome or self.treatment == self.outcome:
            raise ValueError("treatment and outcome must be distinct non-empty column names")
        if not isinstance(self.n_rows, int) or self.n_rows <= 0:
            raise ValueError(f"n_rows must be a positive integer, got {self.n_rows!r}")
        if not self.features:
            raise ValueError("features is empty")
        if len(set(self.features)) != len(self.features):
            raise ValueError("features contains duplicates")
        if set(self.records) != set(self.features):
            raise ValueError(
                "records keys must equal features: "
                f"missing={sorted(set(self.features) - set(self.records))} "
                f"extra={sorted(set(self.records) - set(self.features))}"
            )
        for name, rec in self.records.items():
            if rec.feature != name:
                raise ValueError(f"record {name!r} carries feature={rec.feature!r}")
            if not isinstance(rec.leak_verdict, bool) or not isinstance(rec.review_required, bool):
                raise ValueError(f"record {name!r}: leak_verdict / review_required must be bool")
            if rec.leak_verdict:
                if rec.leak_source not in _LEAK_SOURCES:
                    raise ValueError(
                        f"record {name!r}: leak_verdict=True needs leak_source in {_LEAK_SOURCES}, "
                        f"got {rec.leak_source!r}"
                    )
                if (
                    rec.leak_source == LEAK_SOURCE_LAYER_1
                    and rec.layer_1.get("verdict") != "post_index"
                ):
                    raise ValueError(
                        f"record {name!r}: leak_source layer_1_post_index requires a post_index "
                        f"Layer-1 verdict, got {rec.layer_1.get('verdict')!r}"
                    )
                if rec.leak_source == LEAK_SOURCE_LAYER_3 and (
                    not rec.layer_3.get("ran") or rec.layer_1.get("verdict") == "post_index"
                ):
                    raise ValueError(
                        f"record {name!r}: leak_source layer_3_high requires layer_3.ran and a "
                        "non-post_index Layer-1 verdict"
                    )
            elif rec.leak_source is not None:
                raise ValueError(f"record {name!r}: leak_source set without leak_verdict")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "manifest_source": self.manifest_source,
            "treatment": self.treatment,
            "outcome": self.outcome,
            "n_rows": self.n_rows,
            "features": list(self.features),
            "records": {name: rec.to_dict() for name, rec in self.records.items()},
            "layer_activity": self.layer_activity,
            "activation_profile": self.activation_profile,
            "leakage_fdr": self.leakage_fdr,
            "promotion_eligibility": self.promotion_eligibility,
            "built_at": self.built_at,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "FeatureRolePanel":
        return cls(
            manifest_source=str(payload["manifest_source"]),
            treatment=str(payload["treatment"]),
            outcome=str(payload["outcome"]),
            n_rows=int(payload.get("n_rows", 0)),
            features=tuple(str(f) for f in payload.get("features") or []),
            records={
                str(name): FeatureRoleRecord.from_dict(rec)
                for name, rec in (payload.get("records") or {}).items()
            },
            layer_activity=dict(payload.get("layer_activity") or {}),
            activation_profile=dict(payload.get("activation_profile") or {}),
            leakage_fdr=dict(payload.get("leakage_fdr") or {}),
            promotion_eligibility=dict(payload.get("promotion_eligibility") or {}),
            built_at=str(payload.get("built_at") or ""),
            schema_version=str(payload.get("schema_version", PANEL_SCHEMA_VERSION)),
        )


PanelLike = Union[FeatureRolePanel, Mapping[str, Any]]


def _resolve_covariates(
    frame: pd.DataFrame,
    treatment: str,
    outcome: str,
    covariates: Optional[Sequence[str]],
) -> List[str]:
    if treatment not in frame.columns:
        raise ValueError(f"treatment column {treatment!r} is not in the frame")
    if outcome not in frame.columns:
        raise ValueError(f"outcome column {outcome!r} is not in the frame")
    if treatment == outcome:
        raise ValueError("treatment and outcome must be different columns")
    if covariates is None:
        return [str(c) for c in frame.columns if c not in (treatment, outcome)]
    resolved: List[str] = []
    for c in covariates:
        if c not in frame.columns:
            raise ValueError(f"covariate {c!r} is not in the frame")
        if c in (treatment, outcome) or c in resolved:
            continue
        resolved.append(str(c))
    if not resolved:
        raise ValueError("no covariates left after removing treatment/outcome")
    return resolved


async def build_feature_role_panel(
    frame: pd.DataFrame,
    *,
    manifest_source: str,
    treatment: str,
    outcome: str,
    covariates: Optional[Sequence[str]] = None,
    activation_profile: Optional[Mapping[str, Any]] = None,
    n_permutations: Optional[int] = None,
    seed: int = 7,
    lm_label: Optional[str] = None,
) -> FeatureRolePanel:
    """Run the node's four layers over ``frame[covariates]`` against ``outcome``.

    Args:
        frame: The causal frame (T, Y and the covariates as columns).
        manifest_source: A registered ``MANIFEST_SOURCES`` key (``optum_mart``
            for the real cohort). Drives Layer 1 and the KG activation.
        treatment: T. Excluded from the panel's features and passed to Layer 4
            through ``scope_spec["causal_treatment"]`` so the classifier's role
            question is asked relative to it.
        outcome: Y — the node's ``prediction_target`` (Layer 3 scores each
            covariate against it). Must be binary {0, 1} (the node's rule).
        covariates: Explicit subset; default every other column.
        activation_profile: Per-run flags; default :data:`CAUSAL_ACTIVATION_PROFILE`.
        n_permutations: Layer 3 permutation budget override (the node's FDR
            sizing applies when None).
        seed: Layer 3 permutation seed.
        lm_label: Free-text label recorded under ``layer_activity["layer_4"]["lm"]``
            (``"fake"`` / ``"real"`` / ``"off"``) so the evidence says what
            answered Layer 4.
    """
    from src.agents.ml_foundation.data_preparer.nodes.adaptive_validity_check import (
        _load_kg_cache,
        _parse_target_entity_codes,
        _resolve_kg_mode,
        _try_load_layer_4_classifier,
        adaptive_validity_check,
        compute_promotion_eligibility,
    )
    from src.data.kg.cache import _kg_edge_to_json
    from src.data.kg.ensemble_voter import classify_kg_signal
    from src.data.manifests import MANIFEST_SOURCES, lookup_feature_contract

    if manifest_source not in MANIFEST_SOURCES:
        raise ValueError(
            f"manifest_source {manifest_source!r} is not a registered manifest "
            f"({sorted(MANIFEST_SOURCES)})"
        )
    profile: Dict[str, Any] = dict(
        CAUSAL_ACTIVATION_PROFILE if activation_profile is None else activation_profile
    )
    covs = _resolve_covariates(frame, treatment, outcome, covariates)

    train_df = frame.loc[:, [*covs, outcome]]
    scope_spec: Dict[str, Any] = {
        "prediction_target": outcome,
        "required_features": list(covs),
        "excluded_features": [],
        "feature_manifest_source": manifest_source,
        "kg_mode": profile.get("kg_mode", "shadow"),
        # Read by ``_build_layer_4_inputs`` so the dataset_context names T.
        "causal_treatment": treatment,
    }
    state: Dict[str, Any] = {
        "experiment_id": f"feature-role-panel:{manifest_source}:{treatment}->{outcome}",
        "train_df": train_df,
        "validation_df": None,
        "test_df": None,
        "scope_spec": scope_spec,
        "leakage_findings": [],
        "leaked_features": [],
        "adaptive_layer4_enabled": bool(profile.get("adaptive_layer4_enabled", False)),
        "adaptive_structural_decider_enabled": bool(
            profile.get("adaptive_structural_decider_enabled", False)
        ),
        "adaptive_seed": int(seed),
    }
    if n_permutations is not None:
        state["adaptive_n_permutations"] = int(n_permutations)

    result = await adaptive_validity_check(state)

    verdicts: Dict[str, Dict[str, Any]] = {
        str(v.get("feature")): v
        for v in (result.get("adaptive_verdicts") or [])
        if isinstance(v, dict)
    }
    leaked = {str(f) for f in (result.get("leaked_features") or [])}
    fdr = dict(result.get("leakage_fdr") or {})
    confident = {str(f) for f in (fdr.get("confident_features") or [])}

    # Layer 2 inputs: the node bound the cache onto scope_spec (apply_kg_activation
    # mutates in place); read the same cache and classify with the same function
    # so the supporting edges can be reported (the legacy verdict dict carries
    # only the signal name).
    kg_mode = _resolve_kg_mode(scope_spec.get("kg_mode"))
    kg_cache = _load_kg_cache(scope_spec)
    target_ids = _parse_target_entity_codes(scope_spec.get("target_entity_codes") or [])

    layer4_enabled = bool(state["adaptive_layer4_enabled"])
    classifier_loaded = bool(layer4_enabled and _try_load_layer_4_classifier() is not None)

    records: Dict[str, FeatureRoleRecord] = {}
    for feat in covs:
        contract = lookup_feature_contract(feat, data_source=manifest_source)
        v = verdicts.get(feat)

        if contract is None:
            declared_safe = False
            layer_1: Dict[str, Any] = {
                "contract_present": False,
                "verdict": "no_contract",
                "temporal_status": "unknown",
                "declared_safe": False,
                "knowable_at": None,
                "source": None,
                "window_days": None,
                "entity_codes": [],
            }
            feat_ids: Tuple[str, ...] = ()
        else:
            declared_safe = bool(contract.knowable_at.is_pre_or_at_index())
            feat_ids = tuple(code for _system, code in contract.kg_entity_codes)
            layer_1 = {
                "contract_present": True,
                "verdict": "pre_index" if declared_safe else "post_index",
                "temporal_status": "pre_index" if declared_safe else "post_index",
                "declared_safe": declared_safe,
                "knowable_at": str(contract.knowable_at),
                "source": contract.source,
                "window_days": contract.window_days,
                "entity_codes": [list(t) for t in contract.kg_entity_codes],
            }

        cached_edges = list((kg_cache or {}).get(feat, []))
        if cached_edges:
            signal, considered = classify_kg_signal(
                tuple(cached_edges), feature_entity_ids=feat_ids, target_entity_ids=target_ids
            )
        else:
            signal, considered = "no_signal", ()
        kg_causal = (
            KG_CAUSAL_INTERPRETATION_INDICATION
            if (signal == "leak_drug_treats_disease" and contract is not None and declared_safe)
            else None
        )
        layer_2 = {
            "mode": kg_mode,
            "cache_bound": kg_cache is not None,
            "signal": signal,
            # The voter's vocabulary is prediction-era; this is what the signal
            # means for the causal contrast (None when there is no signal or the
            # column is not a declared pre-index covariate).
            "causal_interpretation": kg_causal,
            "edges": [_kg_edge_to_json(e) for e in considered],
            "n_edges_cached": len(cached_edges),
            "feature_entity_ids": list(feat_ids),
            "target_entity_ids": list(target_ids),
        }

        ran = bool(v is not None and v.get("z_score") is not None)
        severity = v.get("severity") if v else None
        layer_3 = {
            "ran": ran,
            "z_score": v.get("z_score") if ran else None,
            "actual_auc": v.get("actual_auc") if ran else None,
            "null_mean": v.get("null_mean") if ran else None,
            "null_std": v.get("null_std") if ran else None,
            "p_value": v.get("p_value") if ran else None,
            "n_permutations": v.get("n_permutations") if ran else None,
            "delta_auc": v.get("delta_auc") if ran else None,
            "delta_auc_below_floor": bool(v.get("delta_auc_below_floor")) if ran else None,
            "severity_pre_joint_check": v.get("severity_pre_joint_check") if ran else None,
            "ablation_severity": v.get("ablation_severity") if ran else None,
            "fdr_confident": feat in confident,
            # The node's declared-safe immunity stripped a high finding: recorded
            # so the evidence shows Layer 3 fired and the contract overruled it.
            "declared_safe_immunity_applied": bool(
                ran and severity == "high" and declared_safe and feat not in leaked
            ),
        }

        llm_role = v.get("llm_role") if v else None
        layer_4 = {
            "fired": llm_role is not None,
            "role": llm_role,
            "mechanism": v.get("llm_mechanism") if v else None,
            "remediation": v.get("llm_remediation") if v else None,
            "cited_pmids": list((v.get("cited_pmids") if v else None) or []),
            "citations": {
                "checked": v.get("citations_checked") if v else None,
                "verified": v.get("citations_verified") if v else None,
                "unverified": v.get("citations_unverified") if v else None,
                "verified_ids": list((v.get("verified_citation_ids") if v else None) or []),
                "verdicts": list((v.get("citation_verdicts") if v else None) or []),
            },
            "evaluator": {
                key: (v.get(f"evaluator_{key}") if v else None)
                for key in (
                    "satisfied",
                    "rationale_complete",
                    "missed_considerations",
                    "notes",
                    "model",
                )
            },
        }

        if v is None:
            ensemble: Dict[str, Any] = {
                "decided_by": None,
                "final_role": None,
                "confidence": None,
                "severity": None,
                "remediation": None,
                "layer": None,
                "disagreements": [],
                "evidence": (
                    "no verdict: the node evaluates non-numeric columns only through a "
                    "manifest contract, and this column has none"
                ),
                "kg_signal": signal,
                "structural_role": None,
                "structural_unclassifiable": None,
                "kg_predictive_role": None,
                "final_role_note": None,
            }
        else:
            kg_decided = v.get("decided_by") == "kg"
            ensemble = {
                "decided_by": v.get("decided_by"),
                # A KG-decided verdict carries the prediction-era conclusion
                # ("descendant": the condition leaks the target). In the causal
                # contrast that is indication evidence, so the causal role is
                # WITHHELD for the author and the reviewer; the raw predictive
                # role stays under its own name (codex r3).
                "final_role": None if kg_decided else v.get("final_role"),
                "kg_predictive_role": v.get("final_role") if kg_decided else None,
                "final_role_note": (
                    "withheld: the KG's predictive verdict is indication evidence in the "
                    "causal contrast (treatment drug approved for this pre-index condition); "
                    "the structural author and the reviewer decide the role"
                    if kg_decided
                    else None
                ),
                "confidence": v.get("confidence"),
                "severity": severity,
                "remediation": v.get("remediation"),
                "layer": v.get("layer"),
                "disagreements": list(v.get("disagreements") or []),
                "evidence": v.get("evidence"),
                "kg_signal": v.get("kg_signal"),
                "structural_role": v.get("structural_role"),
                "structural_unclassifiable": v.get("structural_unclassifiable"),
            }

        leak_source: Optional[str] = None
        if feat in leaked:
            leak_source = (
                LEAK_SOURCE_LAYER_1
                if (v is not None and v.get("layer") == "1")
                else LEAK_SOURCE_LAYER_3
            )
        records[feat] = FeatureRoleRecord(
            feature=feat,
            layer_1=layer_1,
            layer_2=layer_2,
            layer_3=layer_3,
            layer_4=layer_4,
            ensemble=ensemble,
            leak_verdict=leak_source is not None,
            leak_source=leak_source,
            review_required=(leak_source == LEAK_SOURCE_LAYER_3 and contract is None),
        )

    layer_activity = _summarise(
        records,
        kg_mode=kg_mode,
        cache_bound=kg_cache is not None,
        fdr=fdr,
        layer4_enabled=layer4_enabled,
        classifier_loaded=classifier_loaded,
        lm_label=lm_label,
    )
    promotion = compute_promotion_eligibility(verdicts.values(), n_patients=int(len(frame)))

    return FeatureRolePanel(
        manifest_source=manifest_source,
        treatment=treatment,
        outcome=outcome,
        n_rows=int(len(frame)),
        features=tuple(covs),
        records=records,
        layer_activity=layer_activity,
        activation_profile=profile,
        leakage_fdr=fdr,
        promotion_eligibility=promotion,
        built_at=datetime.now(timezone.utc).isoformat(),
    )


def build_feature_role_panel_sync(frame: pd.DataFrame, **kwargs: Any) -> FeatureRolePanel:
    """Synchronous wrapper for scripts and Lane B's author (no running loop)."""
    return asyncio.run(build_feature_role_panel(frame, **kwargs))


def _summarise(
    records: Mapping[str, FeatureRoleRecord],
    *,
    kg_mode: str,
    cache_bound: bool,
    fdr: Mapping[str, Any],
    layer4_enabled: bool,
    classifier_loaded: bool,
    lm_label: Optional[str],
) -> Dict[str, Any]:
    """Which layers fired, how many features each decided, and the abstain rate."""
    n = len(records)
    recs = list(records.values())
    decided_by = Counter(str(r.ensemble.get("decided_by")) for r in recs)
    n_abstain = sum(1 for r in recs if r.ensemble.get("decided_by") in (None, "abstain"))
    return {
        "layer_1": {
            "consulted": n,
            "contracted": sum(1 for r in recs if r.layer_1.get("contract_present")),
            "declared_safe": sum(1 for r in recs if r.layer_1.get("declared_safe")),
            "post_index": sum(1 for r in recs if r.layer_1.get("verdict") == "post_index"),
        },
        "layer_2": {
            "mode": kg_mode,
            "cache_bound": cache_bound,
            "with_cached_edges": sum(1 for r in recs if r.layer_2.get("n_edges_cached")),
            "signalled": sum(1 for r in recs if r.layer_2.get("signal") != "no_signal"),
            "signals": dict(Counter(str(r.layer_2.get("signal")) for r in recs)),
        },
        "layer_3": {
            "scored": sum(1 for r in recs if r.layer_3.get("ran")),
            "severity_pre_joint_check": dict(
                Counter(
                    str(r.layer_3.get("severity_pre_joint_check"))
                    for r in recs
                    if r.layer_3.get("ran")
                )
            ),
            "fdr_confident": sum(1 for r in recs if r.layer_3.get("fdr_confident")),
            "declared_safe_immunity_applied": sum(
                1 for r in recs if r.layer_3.get("declared_safe_immunity_applied")
            ),
            "fdr": dict(fdr),
        },
        "layer_4": {
            "enabled": layer4_enabled,
            "classifier_loaded": classifier_loaded,
            "lm": lm_label,
            "fired": sum(1 for r in recs if r.layer_4.get("fired")),
            "roles": dict(
                Counter(str(r.layer_4.get("role")) for r in recs if r.layer_4.get("fired"))
            ),
        },
        "ensemble": {
            "decided_by": dict(decided_by),
            "final_roles": dict(
                Counter(
                    str(r.ensemble.get("final_role")) for r in recs if r.ensemble.get("final_role")
                )
            ),
            "abstain_rate": (n_abstain / n) if n else 0.0,
            "leak_verdicts": sum(1 for r in recs if r.leak_verdict),
            "leak_sources": dict(Counter(str(r.leak_source) for r in recs if r.leak_verdict)),
            "review_required": sum(1 for r in recs if r.review_required),
        },
    }


# ---------------------------------------------------------------------------
# Item 3(d): panel -> the causal agent's confounder channels
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ConfounderChannels:
    """What the causal agent should adjust for, given the panel.

    ``anchored_confounders`` is ``None`` when no approved structure was given
    (the state's channel is left as the caller set it); otherwise exactly the
    approved confounders that carry no leak verdict. ``instruments`` carries
    the approved instruments (never anchored: graph_builder forces
    ``conf -> outcome`` for anchored confounders and an instrument must not
    have that edge — the state has its own ``instruments`` channel).
    """

    modeled_confounders: List[str]
    anchored_confounders: Optional[List[str]]
    instruments: List[str]
    #: ``(feature, why)`` for every declared covariate taken OUT of the modeled
    #: set: a leak source (``layer_1_post_index`` / ``layer_3_high``) or an
    #: approved non-confounder role (``approved_mediator`` / ``approved_collider``
    #: / ``approved_descendant`` / ``approved_instrument``).
    removed: List[Tuple[str, str]]
    warnings: List[str]
    #: Removed covariates whose exclusion rests on Layer 3 alone (temporal
    #: status unknown): excluded per spec 3(b), but a human must confirm.
    review_required: List[str] = field(default_factory=list)


#: Approved roles that are NOT backdoor variables and leave the modeled set.
_NON_ADJUSTMENT_ROLES = ("mediator", "collider", "descendant", "instrument")


def _leak_map(panel: PanelLike) -> Tuple[Dict[str, Tuple[Optional[str], bool]], int]:
    """feature -> (leak_source or None, review_required) for panel features, + size."""
    if isinstance(panel, FeatureRolePanel):
        return (
            {
                name: ((rec.leak_source if rec.leak_verdict else None), bool(rec.review_required))
                for name, rec in panel.records.items()
            },
            len(panel.records),
        )
    records = panel.get("records") or {}
    out: Dict[str, Tuple[Optional[str], bool]] = {}
    for name, rec in records.items():
        rec = rec or {}
        leak = bool(rec.get("leak_verdict"))
        source = (rec.get("leak_source") or LEAK_SOURCE_LAYER_3) if leak else None
        out[str(name)] = (source, bool(rec.get("review_required")))
    return out, len(records)


def derive_confounder_channels(
    panel: PanelLike,
    *,
    declared_covariates: Sequence[str],
    approved_structure_roles: Optional[Mapping[str, str]] = None,
) -> ConfounderChannels:
    """Apply the panel to the agent's confounder channels (spec item 3(d)).

    * ``modeled_confounders`` = the declared covariates minus those carrying a
      leak verdict and minus those whose APPROVED role is not a backdoor
      variable (mediator / collider / descendant / instrument); each removal is
      a NAMED warning (the response's only prose channel), never a silent drop.
    * ``anchored_confounders`` = approved ``confounder`` features with no leak
      verdict (None when ``approved_structure_roles`` is None — Lane B's seam).
    * ``instruments`` = approved instruments with no leak verdict, for the
      state's ``instruments`` channel. They are NOT adjusted for and NOT
      anchored: graph_builder forces ``conf -> outcome`` for every anchored
      confounder and an instrument must not have that edge.
    * A Layer-3-only exclusion of an uncontracted column is listed in
      ``review_required``: Layer 3 measures predictiveness of Y, not timing, so
      the warning says the temporal status is unknown (codex r1).
    * A declared covariate the panel never saw is kept and named: the panel
      cannot vouch for a column it did not evaluate.
    """
    leaks, n_panel = _leak_map(panel)
    approved: Dict[str, str] = (
        {str(k): str(v) for k, v in approved_structure_roles.items()}
        if approved_structure_roles is not None
        else {}
    )
    modeled: List[str] = []
    removed: List[Tuple[str, str]] = []
    warnings: List[str] = []
    review: List[str] = []
    for cov in declared_covariates:
        name = str(cov)
        # The agent route one-hot expands categoricals into the loader's stable
        # ``<col>=<level>`` dummies (src/api/routes/causal/loaders.py::
        # _one_hot_categoricals) BEFORE graph_builder runs. A dummy inherits its
        # source column's verdict, or the expansion would launder a leak back
        # into the adjustment set (codex r2).
        key = name
        via_dummy = False
        if name not in leaks and "=" in name and name.split("=", 1)[0] in leaks:
            key = name.split("=", 1)[0]
            via_dummy = True
        if key not in leaks:
            modeled.append(name)
            warnings.append(
                f"feature_role_panel: '{name}' is not in the panel ({n_panel} covariates "
                "evaluated); kept in modeled_confounders unvetted"
            )
            continue
        source, _needs_review = leaks[key]
        suffix = f" (one-hot dummy of '{key}', which carries the verdict)" if via_dummy else ""
        if source is not None:
            removed.append((name, source))
            if source == LEAK_SOURCE_LAYER_1:
                warnings.append(
                    f"feature_role_panel: '{name}' removed from modeled_confounders "
                    f"(leak verdict: {source}){suffix}; a post-index column is not a "
                    "backdoor variable"
                )
            else:
                review.append(name)
                warnings.append(
                    f"feature_role_panel: '{name}' removed from modeled_confounders "
                    f"(leak verdict: {source}){suffix}; it confidently predicts the outcome "
                    "and has no manifest contract, so its temporal status unknown — excluded "
                    "per spec 3(b) pending temporal review, not proven leakage"
                )
            continue
        role = approved.get(key)
        if role in _NON_ADJUSTMENT_ROLES:
            removed.append((name, f"approved_{role}"))
            warnings.append(
                f"feature_role_panel: '{name}' removed from modeled_confounders "
                f"(approved structure derives {role}); a {role} is not a backdoor variable"
                + (
                    " — routed to the instruments channel, not adjusted for and not anchored"
                    if role == "instrument"
                    else ""
                )
            )
            continue
        modeled.append(name)

    # An approved role names the LOGICAL feature; the frame may carry it as
    # one-hot dummies (``<col>=<level>``). Anchoring and the instruments channel
    # must name the columns the DAG actually has (codex r2).
    declared_by_key: Dict[str, List[str]] = {}
    for cov in declared_covariates:
        name = str(cov)
        key = name.split("=", 1)[0] if ("=" in name and name not in leaks) else name
        declared_by_key.setdefault(key, []).append(name)

    anchored: Optional[List[str]] = None
    instruments: List[str] = []
    if approved_structure_roles is not None:
        anchored = []
        for name, role in approved.items():
            source, _needs_review = leaks.get(name, (None, False))
            columns = declared_by_key.get(name, [name])
            if role == "confounder":
                if source is None:
                    anchored.extend(c for c in columns if c not in anchored)
                else:
                    warnings.append(
                        f"feature_role_panel: approved confounder '{name}' not anchored "
                        f"(leak verdict: {source})"
                    )
            elif role == "instrument":
                if source is None:
                    instruments.extend(c for c in columns if c not in instruments)
                else:
                    warnings.append(
                        f"feature_role_panel: approved instrument '{name}' not used "
                        f"(leak verdict: {source})"
                    )
            # mediator / collider / descendant / ancestor: handled in the loop
            # above (never adjustment inputs; an ancestor of T alone is neither
            # anchored nor removed).
    return ConfounderChannels(
        modeled_confounders=modeled,
        anchored_confounders=anchored,
        instruments=instruments,
        removed=removed,
        warnings=warnings,
        review_required=review,
    )


__all__ = [
    "CAUSAL_ACTIVATION_PROFILE",
    "LEAK_SOURCE_LAYER_1",
    "LEAK_SOURCE_LAYER_3",
    "ConfounderChannels",
    "FeatureRolePanel",
    "FeatureRoleRecord",
    "build_feature_role_panel",
    "build_feature_role_panel_sync",
    "derive_confounder_channels",
]
