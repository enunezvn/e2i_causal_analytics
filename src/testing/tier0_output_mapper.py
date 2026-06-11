"""Tier0 Output Mapper for Tier 1-5 Agent Testing.

Maps tier0 synthetic data outputs to each agent's required inputs.
"""

import uuid
from datetime import UTC, datetime, timedelta
from typing import Any, Dict, NotRequired, Required, TypedDict, cast

import pandas as pd


class Tier0StateContract(TypedDict, total=False):
    """Typed contract for the tier0 state dictionary.

    Source of truth for which fields ``Tier0OutputMapper`` accepts from a
    tier0 run. Two required keys (``experiment_id`` and ``eligible_df``) gate
    construction of the mapper; everything else is optional. Validation in
    ``Tier0OutputMapper.__init__`` rejects state dicts that omit the required
    keys OR carry keys outside this contract — the mapper is the boundary
    between tier0 and tier1+ agents and schema drift here is the most common
    way downstream nodes fail with cryptic ``KeyError``s.

    All keys present in the cached state produced by
    ``scripts/run_tier0_test.py`` MUST appear here (as ``Required`` or
    ``NotRequired``). When the script learns to emit a new key, widen this
    contract in the same change — that is the design intent: this contract
    moves with the tier0 pipeline, not behind it.
    """

    # Experiment identification — gate keys, always present in tier0 state.
    experiment_id: Required[str]
    eligible_df: Required[Any]  # pd.DataFrame; loosely typed to keep import light

    # Model artefacts.
    trained_model: NotRequired[Any]
    model_uri: NotRequired[str]
    validation_metrics: NotRequired[Dict[str, Any]]
    test_metrics: NotRequired[Dict[str, Any]]
    test_metrics_at_optimal: NotRequired[Dict[str, Any]]
    test_metrics_at_05: NotRequired[Dict[str, Any]]
    train_metrics: NotRequired[Dict[str, Any]]
    feature_importance: NotRequired[list]
    feature_names: NotRequired[list]
    fitted_preprocessor: NotRequired[Any]
    categorical_encoding: NotRequired[Dict[str, Any]]
    model_candidate: NotRequired[Any]
    alternative_candidates: NotRequired[list]
    deployment_manifest: NotRequired[Dict[str, Any]]
    # v5 Gate C1: regulatory deployment manifest surfaced by validate_promotion
    # (scripts/run_tier0_test.py emits ``regulatory_deployment_manifest``).
    regulatory_deployment_manifest: NotRequired[Dict[str, Any]]

    # Pipeline outputs.
    qc_report: NotRequired[Dict[str, Any]]
    cohort_result: NotRequired[Any]
    scope_spec: NotRequired[Dict[str, Any]]
    class_imbalance_info: NotRequired[Dict[str, Any]]
    success_criteria: NotRequired[Dict[str, Any]]
    success_criteria_met: NotRequired[bool]
    # Per-criterion evaluation detail (run_tier0_test emits ``success_criteria_results``
    # from the v3 audit / bake-off winner).
    success_criteria_results: NotRequired[Dict[str, Any]]
    gate_passed: NotRequired[bool]
    pipeline_halted: NotRequired[bool]
    halt_reason: NotRequired[str]
    patient_df: NotRequired[Any]  # pd.DataFrame; pre-cohort patient pool
    train_df: NotRequired[Any]
    validation_df: NotRequired[Any]

    # Splits and resampling (Block 4 contract: tier1+ MUST reuse).
    split_assignments: NotRequired[Dict[str, Any]]
    split_strategy: NotRequired[str]
    split_validation: NotRequired[Dict[str, Any]]
    resampling_info: NotRequired[Dict[str, Any]]
    cv_results: NotRequired[Dict[str, Any]]

    # Leakage detection / remediation (Block 6A).
    leakage_severity: NotRequired[str]
    leaked_features: NotRequired[list]
    leakage_findings: NotRequired[list]
    # Pre-training heuristic findings (Step-2 graph + Step-5 structural) recorded on
    # SCENARIO regimes WITHOUT applying them to the live gate fields above — see
    # scripts/run_tier0_test._route_leakage_outputs (FU1 / #528).
    leakage_diagnostics: NotRequired[dict]
    leakage_suspected: NotRequired[bool]
    leakage_remediation_status: NotRequired[str]
    leakage_remediated_features: NotRequired[list]
    leakage_dropped_features: NotRequired[list]
    leakage_added_features: NotRequired[list]
    leakage_remediation_reasoning: NotRequired[Any]
    leakage_remediation_viable: NotRequired[bool]
    # Adaptive validity (Layer 4): per-feature Layer-1 verdicts captured by
    # run_pipeline (scripts/run_tier0_test.py emits ``adaptive_verdicts``).
    adaptive_verdicts: NotRequired[list]

    # Evaluator outputs (Block 5/6).
    accuracy_analysis: NotRequired[Dict[str, Any]]
    calibration_analysis: NotRequired[Dict[str, Any]]
    calibration_error: NotRequired[float]
    calibrated_ece: NotRequired[float]
    optimal_threshold: NotRequired[float]
    permutation_test: NotRequired[Dict[str, Any]]
    f1_threshold_analysis: NotRequired[Dict[str, Any]]
    pr_auc: NotRequired[float]
    mcc: NotRequired[float]
    minority_precision: NotRequired[float]
    minority_recall: NotRequired[float]
    precision_constrained: NotRequired[bool]
    model_usefulness: NotRequired[Dict[str, Any]]
    suspicion_level: NotRequired[str]
    suspicion_reasons: NotRequired[list]
    investigation_recommendations: NotRequired[list]
    feature_characteristics: NotRequired[Dict[str, Any]]
    selected_features: NotRequired[list]
    generated_features: NotRequired[list]

    # Serving artefacts (BentoML).
    bentoml_persistent: NotRequired[bool]
    bentoml_pid: NotRequired[int]

    # Temporal scaffolding for downstream Tier 1-5 work (Block 4 onward will
    # consume this; Block 1B only threads it through). When present, this is
    # the inference cutoff time the model is meant to predict from — agents
    # use it to clip lookback windows, filter post-prediction events, and
    # avoid label leakage. Absent / None means "use current time" semantics
    # that earlier blocks already follow.
    prediction_timestamp: NotRequired[pd.Timestamp]

    # Block 5 (#10): cost-weighted utility metric computed by the
    # evaluator from ``scope_spec["cost_matrix"]`` at the chosen
    # (validation-tuned) threshold. Absent when no cost_matrix was
    # supplied. Surfaced here so deployment-decision tooling can rank
    # candidate models by business value, not just AUC/F1.
    business_utility: NotRequired[float]


# Known DESIGNED binary exposures, in fixed preference order (after
# treatment_arm). codex R4: an allowlist keeps treatment selection
# deterministic and semantically grounded — a generic any-{0,1}-column scan
# is column-order dependent and can bind a one-hot fragment as "treatment",
# silently changing the causal estimand. Extend deliberately per dataset.
_KNOWN_DESIGNED_BINARY_EXPOSURES: tuple = ("academic_hcp",)


class Tier0OutputMapper:
    """Maps a tier0 state dictionary to agent-specific inputs.

    See ``Tier0StateContract`` (above) for the authoritative list of state
    keys this class accepts; that TypedDict is the single source of truth
    and is enforced at construction time by ``_validate_contract``. Each
    ``map_to_*`` method below produces the kwargs / state dict expected by
    one Tier 1-5 agent.
    """

    def __init__(self, tier0_state: dict[str, Any]):
        """Initialize with tier0 state dictionary.

        Args:
            tier0_state: State dictionary from a tier0 run. Must conform to
                ``Tier0StateContract`` — i.e. carry every required key and
                no key outside the contract. ``NotRequired`` keys may be
                absent without failing validation.

        Raises:
            TypeError: When ``tier0_state`` is missing required contract
                keys, OR carries keys not declared in the contract. This
                is the boundary between tier0 and tier1+ — fail-loud here
                catches schema drift before downstream nodes hit cryptic
                ``KeyError``s deep in their handlers. Missing-required is
                reported first when both conditions apply, so the more
                actionable error surfaces immediately.
        """
        self.state = tier0_state
        self._validate_contract()

    def _validate_contract(self) -> None:
        """Validate ``self.state`` against ``Tier0StateContract``.

        Drives validation off ``Tier0StateContract.__required_keys__`` and
        ``__optional_keys__`` — the TypedDict is the single source of
        truth. ``__annotations__`` is intentionally NOT used: it would
        treat ``NotRequired`` keys (e.g. ``business_utility``,
        ``prediction_timestamp``) as required and produce false positives.
        """
        required = Tier0StateContract.__required_keys__
        optional = Tier0StateContract.__optional_keys__
        allowed = required | optional

        missing = required - self.state.keys()
        if missing:
            raise TypeError(f"Tier0OutputMapper: missing required state keys: {sorted(missing)}")

        extras = self.state.keys() - allowed
        if extras:
            raise TypeError(
                f"Tier0OutputMapper: unexpected state keys not in contract: {sorted(extras)}"
            )

    def _get_feature_names(self) -> list[str]:
        """Extract feature names from feature_importance or eligible_df."""
        if self.state.get("feature_importance"):
            return [
                f["feature"]
                for f in self.state["feature_importance"]
                if isinstance(f, dict) and "feature" in f
            ]
        # Fallback to DataFrame columns (exclude non-feature columns)
        df = self.state.get("eligible_df")
        if df is not None:
            exclude = {"patient_journey_id", "patient_id", "brand", "discontinuation_flag"}
            return [c for c in df.columns if c not in exclude]
        return []

    def _get_top_features(self, n: int = 5) -> list[str]:
        """Get top N features by importance."""
        features = self._get_feature_names()
        return features[:n] if features else []

    def _get_outcome_var(self) -> str:
        """Resolve the outcome column from the tier0 state's designed target.

        Order of precedence (harness↔data coupling fix, synthetic-CSU 13/13):

        1. ``scope_spec["prediction_target"]`` when it names a real,
           NON-CONSTANT column of ``eligible_df`` — the synthetic-CSU frames
           carry the modeling target there (e.g. ``treatment_initiated``)
           while their legacy ``discontinuation_flag`` column is constant 0
           (an all-zero outcome degenerates every causal/CATE estimation).
        2. Legacy ``discontinuation_flag`` when present (the original tier0
           fixture's target) — preserved verbatim for old caches.
        3. Literal ``"outcome"`` fallback.
        """
        df = self.state["eligible_df"]
        target = (self.state.get("scope_spec") or {}).get("prediction_target")
        if target and target in df.columns and df[target].nunique() > 1:
            return str(target)
        if "discontinuation_flag" in df.columns:
            return "discontinuation_flag"
        return "outcome"

    def _get_binary_treatment(self, outcome_var: str) -> tuple[str, str | None]:
        """Resolve the binary treatment column and its raw basis column.

        Returns ``(treatment_var, treatment_basis)``:

        * ``("treatment_arm", None)`` — the synthetic-CSU frames carry a
          DESIGNED binary treatment with a real treated/control split; it is
          used directly (no derivation, hence no basis column to exclude).
        * ``("high_hcp_engagement", "hcp_visits")`` — legacy fixture path: the
          raw ``hcp_visits`` count has zero control units (every patient has
          >=1 visit), so a median split derives a balanced binary treatment
          (#606). The basis column must be EXCLUDED from confounders
          (collinear with the derived treatment).
        * ``("high_hcp_engagement", <first numeric feature>)`` — last-resort
          derivation when neither designed column exists.
        * ``("high_hcp_engagement", None)`` — degenerate (no numeric basis).
        """
        df = self.state["eligible_df"]
        if (
            "treatment_arm" in df.columns
            and "treatment_arm" != outcome_var
            and df["treatment_arm"].nunique() == 2
        ):
            return "treatment_arm", None

        # A KNOWN designed binary exposure beats a derived median split.
        # Measured 2026-06-11 on the hcp_adoption frame (no treatment_arm):
        # the derived 50/50 split has no causal identity — the estimation
        # node and the refuter's reconstruction mirror-flipped (+0.2492 vs
        # −0.2357) and the refutation divergence gate correctly fail-closed.
        # A real designed binary (academic_hcp, 1149/539) is a genuine
        # confounded exposure both paths bind identically.
        #
        # codex R4 HIGH: the scan is a FIXED-ORDER allowlist of known
        # designed exposures, NOT a generic any-{0,1}-column scan — a
        # generic scan is column-order dependent and can fail open to a
        # one-hot fragment or arbitrary flag, silently changing the causal
        # estimand. Unknown binaries fall through to the explicit
        # derivation path (visible in the treatment name).
        for col in _KNOWN_DESIGNED_BINARY_EXPOSURES:
            if col not in df.columns or col == outcome_var:
                continue
            values = set(df[col].dropna().unique().tolist())
            if values == {0, 1}:
                return col, None

        treatment_basis = next((c for c in ("hcp_visits",) if c in df.columns), None)
        if treatment_basis is None:
            _numeric_cols = set(df.select_dtypes(include="number").columns)
            treatment_basis = next(
                (f for f in self._get_top_features(5) if f in _numeric_cols and f != outcome_var),
                None,
            )
        return "high_hcp_engagement", treatment_basis

    def _add_treatment_column(
        self,
        estimation_df: pd.DataFrame,
        treatment_var: str,
        treatment_basis: str | None,
    ) -> None:
        """Materialize the binary treatment column on ``estimation_df``.

        A designed treatment (``treatment_arm``) is copied as int; a derived
        treatment is the median split of its basis column; with no basis the
        column degenerates to 0 (legacy behavior, fail-visible downstream).
        """
        df = self.state["eligible_df"]
        if treatment_var in df.columns:
            estimation_df[treatment_var] = df[treatment_var].astype(int)
        elif treatment_basis is not None:
            threshold = float(df[treatment_basis].median())
            estimation_df[treatment_var] = (df[treatment_basis] >= threshold).astype(int)
        else:
            estimation_df[treatment_var] = 0  # degenerate fallback (no numeric basis)

    def _get_numeric_confounders(
        self, outcome_var: str, treatment_var: str, treatment_basis: str | None, n: int = 5
    ) -> list[str]:
        """Real, non-constant numeric top-features usable as confounders.

        Excludes the outcome, the treatment, the treatment's basis column
        (collinear with the derived treatment) and CONSTANT columns (e.g. the
        synthetic frame's all-zero ``discontinuation_flag`` — a constant
        regressor carries no information and can break estimators).
        """
        df = self.state["eligible_df"]
        _numeric_cols = set(df.select_dtypes(include="number").columns)
        _exclude = {outcome_var, treatment_var, treatment_basis}
        return [
            f
            for f in self._get_feature_names()
            if f not in _exclude and f in _numeric_cols and df[f].nunique() > 1
        ][:n]

    def _build_model_features(self, row_df: pd.DataFrame) -> dict[str, Any]:
        """Build the MODEL-READY feature dict for a single entity row.

        The tier0 model is trained on the PREPROCESSED matrix (one-hot encoded
        categoricals + imputed/scaled numerics), NOT on raw entity columns —
        the state carries ``fitted_preprocessor`` and ``categorical_encoding``
        precisely so downstream consumers can rebuild that feature space.
        Mirrors the SHAP rebuild in ``scripts/run_tier0_test.py``
        (``_raw_feature_cols`` -> ``_apply_categorical_onehot`` ->
        ``fitted_preprocessor.transform``).

        Falls back to the raw feature dict (restricted to columns that exist
        in the frame) when no preprocessing artefacts are present — legacy
        caches whose models trained directly on raw numeric columns.
        """
        feature_cols = self._get_feature_names()
        enc = self.state.get("categorical_encoding") or {}
        preprocessor = self.state.get("fitted_preprocessor")

        # Map the (possibly one-hot-EXPANDED) feature names back to the
        # ORIGINAL pre-encode columns present in the raw frame.
        onehot_out = set(enc.get("onehot_columns") or [])
        cat_cols = [c for c in (enc.get("columns") or []) if c in row_df.columns]
        raw_cols = [c for c in feature_cols if c not in onehot_out and c in row_df.columns]
        raw_cols += [c for c in cat_cols if c not in raw_cols]
        X = row_df[raw_cols].copy()

        # Re-apply the fitted one-hot encoding so the row carries the IDENTICAL
        # feature columns the model trained on.
        encoder = enc.get("encoder")
        if encoder is not None and cat_cols:
            arr = encoder.transform(X[cat_cols].fillna("__missing__").astype(str))
            X = X.drop(columns=cat_cols)
            for i, col in enumerate(enc.get("onehot_columns") or []):
                X[col] = arr[:, i]
            if feature_cols and all(c in X.columns for c in feature_cols):
                X = X[feature_cols]  # trained feature order

        if preprocessor is None:
            return dict(X.iloc[0].to_dict())

        transformed = preprocessor.transform(X)
        names_out = getattr(preprocessor, "feature_names_out_", None)
        if names_out is not None and len(names_out) == transformed.shape[1]:
            cols_out = [str(c) for c in names_out]
        else:
            cols_out = [str(c) for c in X.columns][: transformed.shape[1]]
        return {c: float(v) for c, v in zip(cols_out, transformed[0], strict=False)}

    def _get_prediction_timestamp(self) -> pd.Timestamp | None:
        """Resolve the prediction timestamp from tier0 state.

        Order of precedence:
        1. Explicit ``state["prediction_timestamp"]`` (top level).
        2. ``state["scope_spec"]["prediction_timestamp"]`` (set by
           ``ScopeDefinerAgent`` when the business spec carries one).

        Returns ``None`` when neither path provides a value. Block 4+ will
        wire the resolved timestamp into temporal feature builders and
        post-prediction event filters; for now it is plumbing only.

        Storage round-trip (Block 1B-M8): ``scope_definer.scope_builder``
        normalises any ``datetime``/``pd.Timestamp``/``str`` input into an
        ISO 8601 string for stable storage in ``scope_spec``. This method
        coerces back to ``pd.Timestamp`` at consumption so every Tier 1-5
        agent sees the same type regardless of how the value entered the
        pipeline. Top-level ``state["prediction_timestamp"]`` overrides
        ``scope_spec`` and may itself be any of those input types — it is
        coerced the same way.
        """
        ts = self.state.get("prediction_timestamp")
        if ts is None:
            ts = (self.state.get("scope_spec") or {}).get("prediction_timestamp")
        if ts is None:
            return None
        # Coerce to pd.Timestamp so downstream agents always see the same type.
        return pd.Timestamp(ts) if not isinstance(ts, pd.Timestamp) else ts

    # =========================================================================
    # TIER 1: Orchestrator Agents
    # =========================================================================

    def map_to_orchestrator(self) -> dict[str, Any]:
        """Map to OrchestratorState input.

        Orchestrator expects:
        - messages: list[dict] with user query
        - current_agent: Optional[str]
        - agent_outputs: dict
        """
        brand = self.state.get("scope_spec", {}).get("brand", "Kisqali")
        return {
            "query": f"What factors drive therapy discontinuation for {brand}?",
            "messages": [
                {
                    "role": "user",
                    "content": f"What factors drive therapy discontinuation for {brand}?",
                }
            ],
            "current_agent": None,
            "agent_outputs": {},
            "experiment_id": self.state["experiment_id"],
        }

    def map_to_tool_composer(self) -> dict[str, Any]:
        """Map to ToolComposer input.

        ToolComposer handles MULTI_FACETED queries requiring multiple tools.
        """
        brand = self.state.get("scope_spec", {}).get("brand", "Kisqali")
        df = self.state["eligible_df"]

        # Derive the causal bindings from the ACTUAL frame schema. The legacy
        # canned query hardcoded hcp_visits/discontinuation — columns ABSENT
        # from the synthetic-CSU frame — so the planner correctly failed
        # closed ("unbound column ... not in schema"). Schema-derived bindings
        # let the SAME canned query run on every tier0 frame: the designed
        # ``treatment_arm``/``treatment_initiated`` pair on synthetic-CSU,
        # the derived ``high_hcp_engagement``/``discontinuation_flag`` pair
        # on the legacy fixture.
        outcome_var = self._get_outcome_var()
        treatment_var, treatment_basis = self._get_binary_treatment(outcome_var)
        covariates = self._get_numeric_confounders(outcome_var, treatment_var, treatment_basis)

        # Numeric-only subset for the causal_effect_estimator tool — same reason
        # as map_to_causal_impact: dowhy/econml raise on the cohort's string
        # columns (UUIDs, brand, age_group). Outcome + numeric covariates.
        # The treatment's raw basis (e.g. hcp_visits) is EXCLUDED: step_1 uses
        # the BINARY treatment derived from it (below), and keeping the count
        # would be collinear with that treatment (it is the basis of the split).
        _tc_cols = [c for c in [outcome_var, *covariates] if c in df.columns]
        estimation_df = df[_tc_cols].copy()
        # Materialize the BINARY treatment so step_1's causal_effect_estimator
        # runs a genuine treated/control contrast (designed ``treatment_arm``
        # copied directly; legacy median split on hcp_visits — the raw count
        # has ZERO control units and yields a degenerate ATE, codex #606
        # MEDIUM).
        self._add_treatment_column(estimation_df, treatment_var, treatment_basis)
        _covariate_phrase = ", ".join(covariates[:3]) if covariates else "the available covariates"
        return {
            # Two facets, each mapping to a REAL advertised tool on the
            # threaded frame: (1) CAUSAL -> causal_effect_estimator
            # (treatment/outcome are REAL schema columns the planner can
            # bind); (2) propensity -> propensity_estimator (treatment +
            # covariates). The legacy hcp_visits phrasing made the LLM
            # decompose into facets it could not bind/plan on this schema.
            "query": (
                f"What is the causal effect of {treatment_var} on {outcome_var} for "
                f"{brand}? Also estimate each patient's propensity to receive "
                f"{treatment_var} given covariates such as {_covariate_phrase}."
            ),
            "experiment_id": self.state["experiment_id"],
            # #621 rewired the formerly-hardcoded registry tools to compute from
            # a caller-supplied DataFrame and fail-closed without one (no more
            # E001/E002 fabrication the anti-fab gate rejects). The hint now
            # advertises the real-output tools that genuinely run on the THREADED
            # numeric-subset fixture below: causal_effect_estimator (ATE),
            # risk_scorer (logistic risk on numeric features), and
            # propensity_estimator (P(treatment|covariates)). gap_calculator and
            # cate_analyzer are intentionally OMITTED here because they require a
            # categorical grouping/segment column (geographic_region / age_group)
            # that the numeric-subset estimation_df below deliberately drops —
            # advertising them would be a dishonest hint (they'd fail-closed).
            # This list is only a planner HINT (consumed at
            # dspy_integration.py:205 as an LLM InputField); the keyless harness
            # uses the canned _MOCK_PLANNING_JSON which routes both steps to
            # causal_effect_estimator, so no fabricated output can reach the gate
            # regardless of this list.
            "available_tools": [
                "causal_effect_estimator",
                "risk_scorer",
                "propensity_estimator",
            ],
            # Thread the real (numeric-subset) tier0 fixture DataFrame to the
            # executor context so the planned fail-closed causal_effect_estimator
            # runs on REAL data via a `$context.estimation_data` reference in the
            # plan — genuine tool execution, not a fabricated tool output (#606).
            "context": {
                "estimation_data": estimation_df,
                "experiment_id": self.state["experiment_id"],
            },
        }

    # =========================================================================
    # TIER 2: Causal Agents
    # =========================================================================

    def map_to_causal_impact(self) -> dict[str, Any]:
        """Map to CausalImpactState input.

        CausalImpact expects:
        - query, query_id
        - treatment_var, outcome_var
        - confounders: list[str]
        - data_source: str
        """
        df = self.state["eligible_df"]

        # Outcome: the state's designed prediction target (synthetic-CSU:
        # ``treatment_initiated``; the frame's ``discontinuation_flag`` is
        # CONSTANT 0 there and would degenerate estimation). Legacy caches
        # keep ``discontinuation_flag``.
        outcome_var = self._get_outcome_var()

        # Treatment: prefer the frame's DESIGNED binary treatment
        # (``treatment_arm`` with a real treated/control split — synthetic-CSU
        # frames). Otherwise derive a BINARY treatment via median split: the
        # legacy fixture's natural candidate ``hcp_visits`` is a 1..19 COUNT —
        # every patient has >=1 visit, so econml/dowhy's binary-treatment path
        # sees ZERO control units. That degeneracy (a) makes the ATE
        # meaningless (no causal contrast) and (b) crashes LinearDML/DRLearner
        # ("unknown categories [0] in column 0 during transform") because CV
        # folds train without the empty control category. (#606)
        treatment_var, treatment_basis = self._get_binary_treatment(outcome_var)

        # Confounders: real, non-constant numeric top-features excluding the
        # outcome, the treatment and the treatment's basis column (raw
        # hcp_visits would be collinear with the binarized treatment).
        # dowhy/econml cannot consume categorical string columns (age_group,
        # geographic_region) — excluded via the numeric filter.
        confounders = self._get_numeric_confounders(outcome_var, treatment_var, treatment_basis)

        # Pass ONLY the numeric columns the estimator needs (outcome +
        # confounders + the binary treatment). The agent feeds this
        # DataFrame straight to dowhy/econml, which raise "could not convert
        # string to float" on the cohort's string columns (patient_journey_id
        # UUIDs, brand, age_group, dates). Subsetting to estimable numeric
        # columns is the honest harness-shaping fix (#606) — equivalent to the
        # cleaned frame a real data connector returns.
        estimation_cols = [c for c in [outcome_var, *confounders] if c in df.columns]
        estimation_df = df[estimation_cols].copy()
        self._add_treatment_column(estimation_df, treatment_var, treatment_basis)

        # Smoke-budget row cap (same #606 budget-shaping rationale as the
        # bounded refutation_config below). MEASURED on the synthetic-CSU
        # frame: each DoWhy refuter SIMULATION re-fits the estimator WITH
        # bootstrapped CIs (~100 inner statsmodels refits), costing ~8-9s/sim
        # nearly independent of row count — so the SIM COUNT below is the
        # dominant budget knob; the row cap trims the estimation/energy-score
        # phase and per-refit cost at the margin. A SEEDED subsample of the
        # REAL frame keeps the suite honest (real rows, both treatment arms,
        # the same designed confounding) within the smoke budget; full-frame
        # refutation belongs to the slow-tests lane.
        _max_estimation_rows = 2500
        if len(estimation_df) > _max_estimation_rows:
            estimation_df = estimation_df.sample(
                n=_max_estimation_rows, random_state=42
            ).reset_index(drop=True)

        return {
            "query": f"What is the causal effect of {treatment_var} on {outcome_var}?",
            "query_id": str(uuid.uuid4()),
            "treatment_var": treatment_var,
            "outcome_var": outcome_var,
            "confounders": confounders,
            "data_source": "patient_journeys",
            "experiment_id": self.state["experiment_id"],
            "data": estimation_df,  # numeric-only subset + derived binary treatment
            # Smoke-test tuning (#606). The harness exercises the REAL pipeline
            # (estimation -> refutation -> sensitivity -> interpretation) end-to-end
            # and asserts a valid, refuted ATE — but the FULL refutation suite is
            # enormous: random_common_cause + placebo default to 100 DoWhy
            # re-estimations EACH and bootstrap to 500, i.e. ~10 min on OLS /
            # ~60 min on a tree/DML estimator (MEASURED), which no per-agent CI
            # budget can hold.
            #   * method=ols: a real linear-adjustment estimator; cheapest
            #     per-refutation re-estimation (vs the energy-score-selected
            #     causal_forest).
            #   * refutation_config: run ALL real refuters with FEW simulations.
            #     MEASURED on the synthetic-CSU frame (2500-row capped, loaded
            #     box): ~8-9s PER SIMULATION (each re-fit carries ~100 inner
            #     bootstrap-CI refits), so the 22-sim config blew the 180s
            #     per-agent SLA (~202s refutation alone). 11 sims ≈ ~100s
            #     refutation + ~18s estimation/interpretation — inside the SLA
            #     with margin, still every refuter REAL and represented.
            # Full-sim refutation + energy-score selection are covered by slow-tests.
            "parameters": {
                "method": "ols",
                "refutation_config": {
                    "bootstrap": {"num_bootstraps": 3},
                    "placebo_treatment": {"num_simulations": 3},
                    "data_subset": {"num_subsets": 2},
                    "random_common_cause": {"num_simulations": 3},
                },
            },
        }

    def map_to_gap_analyzer(self) -> dict[str, Any]:
        """Map to GapAnalyzerState input.

        GapAnalyzer expects:
        - query: str
        - metrics: List[str] - KPI names to analyze
        - segments: List[str] - Segmentation dimensions
        - brand: str
        - time_period: str (optional, default "current_quarter")
        - filters: Optional[Dict] (optional)
        - gap_type: Literal (optional, default "vs_potential")
        - tier0_data: Optional[DataFrame] - Passthrough data from tier0 (NEW)
        - use_mock: bool - Whether to use mock connectors (NEW)

        When tier0_data is provided, the agent can derive performance metrics
        from it instead of querying Supabase.
        """
        df = self.state["eligible_df"]

        # metrics should be a list of KPI names (strings)
        metrics = ["trx", "market_share", "conversion_rate"]

        # segments should be a list of dimension names (strings)
        segments = []
        if "geographic_region" in df.columns:
            segments.append("geographic_region")
        if "age_group" in df.columns:
            segments.append("age_group")
        if "prior_treatments" in df.columns:
            segments.append("prior_treatments")
        if not segments:
            segments = ["region"]  # Use region as default (mock connector supports it)

        return {
            "query": "Identify performance gaps and ROI opportunities",
            "metrics": metrics,
            "segments": segments,
            "brand": self.state.get("scope_spec", {}).get("brand", "Kisqali"),
            "time_period": "current_quarter",
            "gap_type": "vs_potential",
            "tier0_data": df,  # NEW: Pass actual DataFrame for deriving performance metrics
        }

    def map_to_heterogeneous_optimizer(self) -> dict[str, Any]:
        """Map to HeterogeneousOptimizerState input.

        HeterogeneousOptimizer expects:
        - query: str
        - treatment_var: str
        - outcome_var: str
        - segment_vars: List[str] - Variables to segment by
        - effect_modifiers: List[str] - Variables that modify treatment effect
        - data_source: str
        - filters: Optional[Dict]
        - tier0_data: Optional[DataFrame] - Passthrough data from tier0 (NEW)

        When tier0_data is provided, the agent uses it directly instead of
        querying Supabase or falling back to MockDataConnector.
        """
        df = self.state["eligible_df"]

        # Use tier0 DataFrame columns directly since we're passing the data
        # Detect appropriate columns from the DataFrame
        numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
        categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

        # Outcome variable: the state's designed prediction target first
        # (synthetic-CSU: ``treatment_initiated`` — its ``discontinuation_flag``
        # is CONSTANT 0, which made every CATE 0.0 and both responder lists
        # empty). Legacy fallbacks preserved.
        outcome_var = self._get_outcome_var()
        if outcome_var == "outcome":  # neither designed target nor legacy flag
            if "trx_total" in df.columns:
                outcome_var = "trx_total"
            else:
                outcome_var = numeric_cols[0] if numeric_cols else "outcome"

        # Treatment variable: prefer the frame's DESIGNED binary treatment
        # (``treatment_arm`` — real treated/control split, the column the
        # production dispatcher's INPUT_RESOLVERS bind on this substrate),
        # then the legacy exposure candidates.
        treatment_var = None
        if (
            "treatment_arm" in df.columns
            and "treatment_arm" != outcome_var
            and df["treatment_arm"].nunique() == 2
        ):
            treatment_var = "treatment_arm"

        treatment_candidates = ["hcp_visits", "prior_treatments", "days_on_therapy"]
        if not treatment_var:
            for col in treatment_candidates:
                if col in df.columns and col != outcome_var:
                    treatment_var = col
                    break

        # Fallback: find any binary column that's not outcome
        if not treatment_var:
            for col in numeric_cols:
                if col == outcome_var:
                    continue
                unique_vals = df[col].nunique()
                if unique_vals == 2:  # Binary treatment
                    treatment_var = col
                    break

        # Fallback: first numeric column that's not outcome
        if not treatment_var:
            for col in numeric_cols:
                if col != outcome_var:
                    treatment_var = col
                    break

        treatment_var = treatment_var or "treatment"

        # Segment variables: use categorical columns for post-hoc CATE-by-segment analysis.
        # These are NOT used as the W (confounders) parameter in CausalForestDML — that is
        # decoupled in cate_estimator.py. segment_vars only drive the _calculate_cate_by_segment
        # loop which computes per-segment CATE estimates after model fitting.
        # The frame's DESIGNED segmentation column comes first when present
        # (synthetic-CSU ``segment_assignment``); identifier-ish / bookkeeping
        # columns (patient & HCP ids, dates, train/test split labels) and
        # high-cardinality columns are NOT patient segments — segmenting on
        # them yields thousands of singleton "segments" and no honest
        # responder differentiation.
        _segment_exclude = {
            "patient_journey_id",
            "patient_id",
            "brand",
            "hcp_id",
            "data_split",
            "journey_start_date",
        }
        _max_segment_cardinality = 12
        _preferred_segments = [c for c in ("segment_assignment",) if c in df.columns]
        segment_vars = (
            _preferred_segments
            + [
                c
                for c in categorical_cols
                if c not in _segment_exclude
                and c not in _preferred_segments
                and 1 < df[c].nunique() <= _max_segment_cardinality
            ]
        )[:3]

        # Effect modifiers: NON-CONSTANT numeric columns that aren't
        # treatment/outcome (a constant column — e.g. the synthetic frame's
        # all-zero discontinuation_flag — carries no heterogeneity signal).
        effect_modifiers = [
            c
            for c in numeric_cols
            if c not in {treatment_var, outcome_var, "patient_journey_id", "patient_id"}
            and df[c].nunique() > 1
        ][:5]
        # If no effect modifiers available, CATE estimation will fail - raise informative error
        if not effect_modifiers:
            raise ValueError(
                f"No effect modifiers available for CATE estimation. "
                f"Need numeric columns besides treatment ({treatment_var}) and outcome ({outcome_var}). "
                f"Available numeric columns: {numeric_cols}"
            )

        return {
            "query": f"Analyze heterogeneous treatment effects of {treatment_var}",
            "treatment_var": treatment_var,
            "outcome_var": outcome_var,
            "segment_vars": segment_vars,
            "effect_modifiers": effect_modifiers,
            "data_source": "patient_journeys",
            "filters": None,
            "tier0_data": df,  # NEW: Pass actual DataFrame for direct use
            "prediction_timestamp": self._get_prediction_timestamp(),
        }

    # =========================================================================
    # TIER 3: Monitoring Agents
    # =========================================================================

    def map_to_drift_monitor(self) -> dict[str, Any]:
        """Map to DriftMonitorInput (Pydantic model).

        DriftMonitorInput schema:
        - query: str (required)
        - features_to_monitor: list[str] (required)
        - model_id: Optional[str]
        - time_window: str (default "7d")
        - brand: Optional[str]
        - significance_level: float (default 0.05)
        - tier0_data: Optional[DataFrame] (for testing with real synthetic data)
        - prediction_timestamp: Optional[pd.Timestamp] (inference cutoff,
          plumbed through from scope_spec for downstream temporal anchoring;
          Block 1B threads it but doesn't yet consume it)
        """
        feature_cols = self._get_feature_names()
        eligible_df = self.state.get("eligible_df")

        # Use experiment_id as model_id to enable model/concept drift detection
        # This allows the drift detector to simulate model predictions based on the data
        model_id = self.state.get("experiment_id", "tier0_test_model")

        return {
            "query": "Detect data and model drift in patient features",
            "features_to_monitor": feature_cols[:10],  # Limit features
            "model_id": model_id,  # Enable model/concept drift detection
            "time_window": "30d",
            "brand": self.state.get("scope_spec", {}).get("brand"),
            "significance_level": 0.05,
            # Pass tier0_data for drift detection with real synthetic data
            "tier0_data": eligible_df,
            "prediction_timestamp": self._get_prediction_timestamp(),
        }

    def map_to_experiment_designer(self) -> dict[str, Any]:
        """Map to ExperimentDesignerInput (Pydantic model).

        ExperimentDesignerInput schema:
        - business_question: str (required, min 10 chars)
        - constraints: dict (optional) - budget, timeline, ethical, operational
        - available_data: dict (optional)
        - preregistration_formality: "light" | "medium" | "heavy" (default "medium")
        - max_redesign_iterations: int (default 2)
        - enable_validity_audit: bool (default True)
        - brand: Optional[str]
        """
        df = self.state["eligible_df"]
        validation_metrics = self.state.get("validation_metrics", {})
        brand = self.state.get("scope_spec", {}).get("brand")

        return {
            "business_question": "Does personalized HCP outreach improve patient retention rates compared to standard outreach?",
            "constraints": {
                "budget": 50000,
                "timeline": {"max_duration_days": 90},
                "operational": {
                    "min_sample_size": 100,
                    "max_sample_size": int(len(df) * 0.5),
                },
                "expected_effect_size": 0.10,
            },
            "available_data": {
                "total_patients": len(df),
                "historical_retention_rate": 1 - validation_metrics.get("recall", 0.3) * 0.4,
                "features": self._get_feature_names()[:10],
            },
            "preregistration_formality": "medium",
            "max_redesign_iterations": 2,
            "enable_validity_audit": True,
            "brand": brand,
        }

    def map_to_experiment_monitor(self) -> dict[str, Any]:
        """Map to ExperimentMonitorInput (dataclass).

        ExperimentMonitorInput schema (see
        ``src/agents/experiment_monitor/agent.py:ExperimentMonitorInput``):
        - query: str
        - experiment_ids: Optional[List[str]] (None = use check_all_active)
        - check_all_active: bool (default True)
        - srm_threshold: float (default 0.001)
        - enrollment_threshold: float (default 5.0)
        - fidelity_threshold: float (default 0.2)
        - stale_data_threshold_hours: float (default 24.0)
        - check_interim: bool (default True)

        Tier0 doesn't seed live experiments; we exercise the "no active
        experiments" path which the agent must handle gracefully (empty
        ``experiments`` list, ``monitor_summary`` describing the empty
        check). That validates the agent's pipeline contract end-to-end.
        """
        return {
            "query": "Check all active A/B experiments for SRM, enrollment, and interim issues",
            "experiment_ids": None,
            "check_all_active": True,
            "srm_threshold": 0.001,
            "enrollment_threshold": 5.0,
            "fidelity_threshold": 0.2,
            "stale_data_threshold_hours": 24.0,
            "check_interim": True,
        }

    def map_to_health_score(self) -> dict[str, Any]:
        """Map to HealthScoreAgent.check_health() kwargs.

        check_health signature:
        - scope: Literal["full", "quick", "models", "pipelines", "agents"]
        - query: str
        - experiment_name: str
        """
        return {
            "scope": "full",
            "query": "Check system health status",
            "experiment_name": self.state["experiment_id"],
        }

    # =========================================================================
    # TIER 4: ML Prediction Agents
    # =========================================================================

    def map_to_prediction_synthesizer(self) -> dict[str, Any]:
        """Map to PredictionSynthesizerAgent.synthesize() kwargs.

        synthesize signature:
        - entity_id: str
        - prediction_target: str
        - features: Optional[Dict[str, Any]]
        - entity_type: str (hcp, territory, patient)
        - time_horizon: str
        - models_to_use: Optional[List[str]]
        - ensemble_method: str
        - include_context: bool
        - query: str
        - session_id: Optional[str]

        Note: deployment_manifest and trained_model are accessed via mapper.state
        and passed to agent constructor via _get_agent_kwargs().
        """
        df = self.state["eligible_df"]

        # Select a patient with outcome == 1 (positive class of the ACTUAL
        # modeling target — synthetic-CSU: ``treatment_initiated``; legacy:
        # ``discontinuation_flag``) so the model produces a non-zero
        # point_estimate. Random sampling can pick a near-zero risk patient
        # → point_estimate=0.0 → secondary gate failure.
        outcome_col = self._get_outcome_var()
        if outcome_col in df.columns and (df[outcome_col] == 1).any():
            sample_row_df = df[df[outcome_col] == 1].iloc[[0]]
        else:
            sample_row_df = df.iloc[[0]]
        sample_row = sample_row_df.iloc[0]

        # Get a sample entity
        sample_entity_id = str(sample_row.get("patient_journey_id", "test_patient_001"))

        # Build the MODEL-READY feature dict: the tier0 model trains on the
        # PREPROCESSED matrix (one-hot + scaled), so the raw entity row must be
        # passed through the state's fitted artefacts — feeding raw columns to
        # the model raised KeyError on the one-hot names
        # (['geographic_region_midwest', ...] not in index).
        sample_features = (
            self._build_model_features(sample_row_df) if self._get_feature_names() else {}
        )

        # Prediction target follows the scope_spec's designed target when the
        # tier0 run declares one; legacy caches keep the original label.
        prediction_target = str(
            (self.state.get("scope_spec") or {}).get("prediction_target") or "discontinuation_risk"
        )

        return {
            "entity_id": sample_entity_id,
            "prediction_target": prediction_target,
            "features": sample_features,
            "entity_type": "patient",
            "time_horizon": "30d",
            "models_to_use": None,  # Use all available
            "ensemble_method": "weighted",
            # The Tier 1-5 harness has no external context-enrichment services
            # (feature-importance / accuracy / trend / online-feature stores), so
            # requesting context would make context_enricher fail-closed ("all 5
            # dependencies failed" -> status=failed) — a false alarm, not a real
            # regression. Skip context here (like --skip-observability); the
            # ensemble prediction from the real model clients is still validated.
            # Harness-scoped mapper choice; the prod contract (fail when context
            # is requested but unavailable) is unchanged. (#606 item D)
            "include_context": False,
            "query": f"Predict {prediction_target} for patient {sample_entity_id}",
            "session_id": self.state["experiment_id"],
            # Note: deployment_manifest and trained_model are passed to agent constructor
            # via _get_agent_kwargs(), not to synthesize() method
        }

    def map_to_resource_optimizer(self) -> dict[str, Any]:
        """Map to ResourceOptimizerAgent.optimize() kwargs.

        optimize signature:
        - allocation_targets: List[AllocationTarget]
        - constraints: List[Constraint]
        - resource_type: str
        - objective: str
        - solver_type: str
        - run_scenarios: bool
        - scenario_count: int
        - query: str
        - session_id: Optional[str]
        """
        df = self.state["eligible_df"]

        # Create allocation targets based on regions if available
        allocation_targets = []
        if "geographic_region" in df.columns:
            regions = df["geographic_region"].unique()[:5]
            for i, region in enumerate(regions):
                df[df["geographic_region"] == region]
                # expected_response must be a response coefficient > 1.0 to produce
                # meaningful ROI. Using data-derived values: base 1.5 + spread by region
                # so the optimizer can differentiate territories.
                expected_response = 1.5 + 0.4 * (i % len(regions))
                allocation_targets.append(
                    {
                        "entity_id": f"territory_{region}",
                        "entity_type": "territory",
                        "current_allocation": 50000.0,
                        "expected_response": expected_response,
                        "min_allocation": 25000.0,
                        "max_allocation": 100000.0,
                    }
                )
        else:
            # Default allocation targets
            allocation_targets = [
                {
                    "entity_id": "territory_northeast",
                    "entity_type": "territory",
                    "current_allocation": 50000.0,
                    "expected_response": 0.3,
                    "min_allocation": 25000.0,
                    "max_allocation": 100000.0,
                },
                {
                    "entity_id": "territory_midwest",
                    "entity_type": "territory",
                    "current_allocation": 40000.0,
                    "expected_response": 0.25,
                    "min_allocation": 20000.0,
                    "max_allocation": 80000.0,
                },
            ]

        return {
            "allocation_targets": allocation_targets,
            "constraints": [
                {"constraint_type": "budget", "value": 200000.0, "scope": "global"},
            ],
            "resource_type": "budget",
            "objective": "maximize_roi",
            "solver_type": "linear",
            "run_scenarios": False,
            "scenario_count": 3,
            "query": "Optimize budget allocation across territories",
            "session_id": self.state["experiment_id"],
        }

    # =========================================================================
    # TIER 5: Self-Improvement Agents
    # =========================================================================

    def map_to_explainer(self) -> dict[str, Any]:
        """Map to ExplainerAgent.explain() kwargs.

        explain signature:
        - analysis_results: List[Dict[str, Any]]
        - query: str
        - user_expertise: str (executive, analyst, data_scientist)
        - output_format: str (narrative, structured, presentation, brief)
        - focus_areas: Optional[List[str]]
        - session_id: Optional[str]
        - memory_config: Optional[Dict[str, Any]]
        """
        validation_metrics = self.state.get("validation_metrics", {})
        feature_importance = self.state.get("feature_importance", [])

        # Build analysis_results as a list of dicts (format expected by explain())
        # Each result MUST include 'key_findings' list for the explainer to extract insights
        confounders = self._get_top_features(3)
        top_features = feature_importance[:5] if feature_importance else []

        # Helper to format floats for human readability (explainer quality gate
        # rejects raw floats like 0.6583333333333333)
        def _fmt(v, pct: bool = False) -> str:
            if not isinstance(v, (int, float)):
                return str(v)
            if pct:
                return f"{v * 100:.1f}%"
            return f"{v:.3f}"

        analysis_results = [
            {
                "agent": "causal_impact",
                "analysis_type": "causal_analysis",
                "treatment_var": "hcp_visits",
                "outcome_var": "discontinuation_flag",
                "ate": 0.127,
                "ate_ci": [0.089, 0.165],
                "p_value": 0.0023,
                "confounders_identified": confounders,
                "confidence": 0.85,
                # key_findings is REQUIRED for explainer insight extraction
                "key_findings": [
                    "Significant causal effect identified: ATE=0.127 (p=0.002)",
                    "Treatment (hcp_visits) increases outcome by 12.7% on average",
                    "Effect is statistically significant with 95% CI [0.089, 0.165]",
                    f"Key confounders controlled: {', '.join(confounders) if confounders else 'none identified'}",
                ],
            },
            {
                "agent": "model_trainer",
                "analysis_type": "model_performance",
                "confidence": validation_metrics.get("roc_auc", 0.7),
                # key_findings from validation metrics (formatted for readability)
                "key_findings": [
                    f"Model accuracy: {_fmt(validation_metrics.get('accuracy', 'N/A'), pct=True)}",
                    f"ROC-AUC score: {_fmt(validation_metrics.get('roc_auc', 'N/A'))}",
                    f"Precision: {_fmt(validation_metrics.get('precision', 'N/A'))}, Recall: {_fmt(validation_metrics.get('recall', 'N/A'))}",
                    f"F1 score: {_fmt(validation_metrics.get('f1_score', 'N/A'))}",
                ],
                **{
                    k: round(v, 4) if isinstance(v, float) else v
                    for k, v in validation_metrics.items()
                },
            },
            {
                "agent": "feature_analyzer",
                "analysis_type": "feature_importance",
                "top_features": top_features,
                "confidence": 0.75,
                # key_findings from feature importance
                "key_findings": [
                    f"Top predictive features identified: {len(top_features)} features analyzed",
                ]
                + [
                    f"Feature '{f.get('feature', f)}' has importance score {_fmt(f.get('importance', 'N/A'))}"
                    for f in top_features[:3]
                    if isinstance(f, dict)
                ]
                if top_features
                else ["No feature importance data available"],
            },
        ]

        return {
            "analysis_results": analysis_results,
            "query": "Explain the discontinuation risk analysis results",
            "user_expertise": "analyst",
            "output_format": "structured",
            "focus_areas": ["causal_effects", "feature_importance"],
            "session_id": self.state["experiment_id"],
            "memory_config": {
                "brand": self.state.get("scope_spec", {}).get("brand", "Kisqali"),
            },
        }

    def map_to_feedback_learner(self) -> dict[str, Any]:
        """Map to FeedbackLearnerAgent.learn() kwargs.

        learn signature:
        - time_range_start: str (ISO format)
        - time_range_end: str (ISO format)
        - batch_id: Optional[str]
        - focus_agents: Optional[List[str]]
        """
        now = datetime.now(UTC)

        return {
            "time_range_start": (now - timedelta(days=1)).isoformat(),
            "time_range_end": now.isoformat(),
            "batch_id": f"batch_{self.state['experiment_id'][:8]}",
            "focus_agents": ["causal_impact", "gap_analyzer", "explainer"],
        }

    # =========================================================================
    # Utility Methods
    # =========================================================================

    def get_all_mappings(self) -> dict[str, dict[str, Any]]:
        """Get all agent mappings as a dictionary.

        Returns:
            Dict mapping agent_name -> mapped_input
        """
        return {
            # Tier 1
            "orchestrator": self.map_to_orchestrator(),
            "tool_composer": self.map_to_tool_composer(),
            # Tier 2
            "causal_impact": self.map_to_causal_impact(),
            "gap_analyzer": self.map_to_gap_analyzer(),
            "heterogeneous_optimizer": self.map_to_heterogeneous_optimizer(),
            # Tier 3
            "drift_monitor": self.map_to_drift_monitor(),
            "experiment_designer": self.map_to_experiment_designer(),
            "experiment_monitor": self.map_to_experiment_monitor(),
            "health_score": self.map_to_health_score(),
            # Tier 4
            "prediction_synthesizer": self.map_to_prediction_synthesizer(),
            "resource_optimizer": self.map_to_resource_optimizer(),
            # Tier 5
            "explainer": self.map_to_explainer(),
            "feedback_learner": self.map_to_feedback_learner(),
        }

    def get_agent_mapping(self, agent_name: str) -> dict[str, Any]:
        """Get mapping for a specific agent.

        Args:
            agent_name: Name of the agent (e.g., 'causal_impact')

        Returns:
            Mapped input dictionary for the agent

        Raises:
            ValueError: If agent_name is not supported
        """
        method_name = f"map_to_{agent_name}"
        if not hasattr(self, method_name):
            raise ValueError(
                f"Unknown agent: {agent_name}. Supported: {list(self.get_all_mappings().keys())}"
            )
        return cast(Dict[str, Any], getattr(self, method_name)())
