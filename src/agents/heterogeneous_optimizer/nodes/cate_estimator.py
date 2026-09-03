"""CATE Estimator Node for Heterogeneous Optimizer Agent.

This node estimates Conditional Average Treatment Effects using EconML's CausalForestDML.
Core computational node with minimal LLM usage.
"""

import asyncio
import logging
import os
import time
import traceback
from typing import Any, Dict, List, cast

import numpy as np
import pandas as pd

from src.causal.stats import z_score_for_alpha
from src.utils.supabase_env import resolve_supabase_service_key

from ..design import binarize_treatment, sanitize_effect_modifiers
from ..state import CATEResult, HeterogeneousOptimizerState

logger = logging.getLogger(__name__)


_KNOWN_DEV_ENVIRONMENTS = {"development", "dev", "test", "testing", "local"}


def _mock_connector_allowed() -> bool:
    """Return True iff MockDataConnector fallback is permitted.

    F-013 (#431): cate_estimator previously silently fell through to
    ``MockDataConnector`` whenever Supabase env vars were absent OR the
    real connector failed to initialize. Downstream consumers had no signal
    that they were receiving synthetic ``np.random.seed(42)`` data.

    Policy (closed-by-default — codex iter-1 H1 fix):

    * ``E2I_ALLOW_MOCK_CONNECTOR=1`` (truthy) → mock allowed.
    * ``E2I_ALLOW_MOCK_CONNECTOR=0`` (falsy) → mock forbidden.
    * Unset → only an EXPLICIT dev ``ENVIRONMENT`` value
      (``development``/``dev``/``test``/``testing``/``local``) permits the
      mock connector. Unset/misspelled/``production``/anything else =>
      raise RuntimeError. Missing metadata MUST NOT enable fabricated data.
    """
    raw = os.environ.get("E2I_ALLOW_MOCK_CONNECTOR", "").strip().lower()
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    env = os.environ.get("ENVIRONMENT", "").strip().lower()
    return env in _KNOWN_DEV_ENVIRONMENTS


def _get_default_data_connector():
    """Get the default data connector based on environment.

    Uses HeterogeneousOptimizerDataConnector when Supabase credentials are
    available. When credentials are absent or the real connector fails to
    initialize, the function honors ``_mock_connector_allowed()``:

    * Mock allowed → return ``MockDataConnector`` (offline dev).
    * Mock forbidden → raise ``RuntimeError`` so the caller sees an explicit
      configuration error instead of synthetic data (F-013, issue #431).
    """
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = resolve_supabase_service_key()

    if supabase_url and supabase_key:
        try:
            from ..connectors import HeterogeneousOptimizerDataConnector

            logger.info("Using HeterogeneousOptimizerDataConnector (Supabase)")
            return HeterogeneousOptimizerDataConnector()
        except Exception as e:
            logger.warning(f"Failed to initialize Supabase connector: {e}")
            if not _mock_connector_allowed():
                raise RuntimeError(
                    "Failed to initialize Supabase data connector and "
                    "MockDataConnector fallback is disabled "
                    "(E2I_ALLOW_MOCK_CONNECTOR=1 or ENVIRONMENT in "
                    "{development,dev,test,testing,local} required for mock "
                    f"fallback). Original error: {e}"
                ) from e

    # Supabase env vars absent OR real connector init failed under
    # mock-allowed conditions: decide whether to fall through to mock.
    if not _mock_connector_allowed():
        env_summary = (
            f"SUPABASE_URL={'set' if supabase_url else 'unset'}, "
            f"SUPABASE_*_KEY={'set' if supabase_key else 'unset'}, "
            f"ENVIRONMENT={os.getenv('ENVIRONMENT', 'development')!r}"
        )
        raise RuntimeError(
            "MockDataConnector fallback is disabled "
            "(E2I_ALLOW_MOCK_CONNECTOR=1 or ENVIRONMENT in "
            "{development,dev,test,testing,local} required). "
            "Configure Supabase credentials (SUPABASE_URL and "
            "SUPABASE_SERVICE_ROLE_KEY or SUPABASE_ANON_KEY) to enable the "
            f"real data connector. Current env: {env_summary}."
        )

    # Fallback to mock connector (explicit dev/test opt-in)
    from ..connectors import MockDataConnector

    logger.info("Using MockDataConnector (development/testing mode)")
    return MockDataConnector()


class CATEEstimatorNode:
    """Estimate Conditional Average Treatment Effects using EconML.

    This node uses CausalForestDML to estimate treatment effect heterogeneity
    across segments.
    """

    def __init__(self, data_connector=None, require_real_data: bool = False):
        """Initialize CATE estimator node.

        Args:
            data_connector: Data connector for fetching analysis data.
                           If None, uses default based on environment.
            require_real_data: If True, raises ValueError if only mock data
                              is available. Used in testing to ensure real
                              Supabase data is used.
        """
        self.require_real_data = require_real_data
        self.data_connector = data_connector or _get_default_data_connector()
        self.timeout_seconds = 180

        # Validate real data requirement
        if self.require_real_data:
            connector_type = type(self.data_connector).__name__
            if "Mock" in connector_type:
                raise ValueError(
                    f"require_real_data=True but data connector is {connector_type}. "
                    "Set SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY environment variables "
                    "to use real Supabase data."
                )

    async def execute(self, state: HeterogeneousOptimizerState) -> HeterogeneousOptimizerState:
        """Execute CATE estimation."""
        start_time = time.time()
        logger.info(
            "Starting CATE estimation",
            extra={
                "node": "cate_estimator",
                "treatment_var": state.get("treatment_var"),
                "outcome_var": state.get("outcome_var"),
                "effect_modifiers": state.get("effect_modifiers", []),
                "n_estimators": state.get("n_estimators", 100),
            },
        )

        try:
            from econml.dml import CausalForestDML
            from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

            # Label-gater (opt-in, fail-open): augment segment_vars with the brand's
            # label-relevant categorical columns so on/off-label bands form (codex#3).
            # Done BEFORE the load so the columns are fetched; _calculate_cate_by_segment
            # guards any column missing from the frame.
            if state.get("label_segmentation") and state.get("brand"):
                try:
                    from src.services.clinical_context.label_criteria_provider import (
                        label_segment_columns,
                    )

                    extra = label_segment_columns(state["brand"], state.get("indication"))
                    seg_vars = list(state.get("segment_vars") or [])
                    for col in extra:
                        if col not in seg_vars:
                            seg_vars.append(col)
                    if seg_vars != (state.get("segment_vars") or []):
                        state = {**state, "segment_vars": seg_vars}
                except Exception as exc:  # noqa: BLE001 — fail-open
                    logger.warning("label-gater: segment_vars augmentation skipped: %s", exc)

            # Fetch data
            df = await self._fetch_data(state)

            if df is None or len(df) < 100:
                return {
                    **state,
                    "errors": [
                        {"node": "cate_estimator", "error": "Insufficient data (need >= 100 rows)"}
                    ],
                    "status": "failed",
                }

            # Label-gater: resolve the indication from the frame's diagnosis distribution
            # (codex#2 — no silent default) and carry it forward for the policy_learner gate.
            if (
                state.get("label_segmentation")
                and state.get("brand")
                and not state.get("indication")
                and "diagnosis_code" in df.columns
            ):
                try:
                    from src.services.clinical_context.label_criteria_provider import (
                        resolve_indication,
                    )

                    resolved = resolve_indication(state["brand"], df["diagnosis_code"].tolist())
                    if resolved:
                        state = {**state, "indication": resolved}
                except Exception as exc:  # noqa: BLE001 — fail-open
                    logger.warning("label-gater: indication resolution skipped: %s", exc)

            # Prepare data
            #
            # Fail-honest null/empty guard (#30): coerce the outcome and
            # treatment columns to numeric and DROP rows where either is
            # null/NaN BEFORE any median/unique computation. Without this a
            # future all-null (or partially-null) column would either crash
            # np.median/np.unique with a TypeError ("'<' not supported between
            # NoneType and float") or, worse, silently binarize NaN-as-False
            # into a degenerate all-control treatment that fabricates a
            # plausible-looking ATE. We mirror the existing :161-168
            # "Insufficient data" fail shape: return status="failed" with a
            # clear error, never a fabricated value.
            outcome_var = state["outcome_var"]
            treatment_var = state["treatment_var"]
            Y_series = pd.to_numeric(df[outcome_var], errors="coerce")
            T_series = pd.to_numeric(df[treatment_var], errors="coerce")
            finite_mask = (Y_series.notna() & T_series.notna()).to_numpy()
            n_finite = int(finite_mask.sum())

            if n_finite < 100:
                if n_finite == 0:
                    detail = (
                        f"treatment '{treatment_var}' or outcome "
                        f"'{outcome_var}' is entirely null/non-numeric "
                        f"after coercion (0 usable rows from {len(df)})"
                    )
                else:
                    detail = (
                        f"only {n_finite} rows have a non-null numeric "
                        f"treatment '{treatment_var}' AND outcome "
                        f"'{outcome_var}' (need >= 100, from {len(df)} fetched)"
                    )
                logger.error(
                    "CATE estimation aborted: insufficient usable rows after null/numeric coercion",
                    extra={
                        "node": "cate_estimator",
                        "treatment_var": treatment_var,
                        "outcome_var": outcome_var,
                        "rows_fetched": len(df),
                        "rows_usable": n_finite,
                    },
                )
                return {
                    **state,
                    "errors": [
                        {
                            "node": "cate_estimator",
                            "error": f"Insufficient usable data: {detail}",
                        }
                    ],
                    "status": "failed",
                }

            # Apply the finite mask consistently to df, Y and T so the forest,
            # the per-segment CATE design matrix, and the segment masks all see
            # the SAME aligned rows (df is reused downstream by
            # _calculate_cate_by_segment).
            if n_finite < len(df):
                df = df.loc[finite_mask].reset_index(drop=True)
                Y_series = Y_series.loc[finite_mask].reset_index(drop=True)
                T_series = T_series.loc[finite_mask].reset_index(drop=True)
                logger.info(
                    "Dropped rows with null/non-numeric treatment or outcome "
                    "before CATE estimation",
                    extra={
                        "node": "cate_estimator",
                        "rows_dropped": len(finite_mask) - n_finite,
                        "rows_remaining": n_finite,
                    },
                )

            Y = Y_series.to_numpy()
            T_raw = T_series.to_numpy()

            # Binarize continuous treatment at median (consistent with causal_impact agent)
            # This ensures comparable results between agents and better CATE estimation.
            # ONE shared rule (design.binarize_treatment) so EconML here and CausalML in
            # the uplift node estimate the SAME contrast.
            T, binarized = binarize_treatment(T_raw)
            if binarized is not None:
                logger.info(
                    f"Binarized continuous treatment at median={binarized['median_threshold']:.2f}",
                    extra={
                        "node": "cate_estimator",
                        "treatment_var": state["treatment_var"],
                        **binarized,
                    },
                )

            # Diagnostic logging for debugging ATE=0 issue
            logger.info(
                "CATE data prepared",
                extra={
                    "node": "cate_estimator",
                    "n_rows": len(df),
                    "treatment_var": state["treatment_var"],
                    "outcome_var": state["outcome_var"],
                    "T_mean": float(np.mean(T)),
                    "T_std": float(np.std(T)),
                    "T_min": float(np.min(T)),
                    "T_max": float(np.max(T)),
                    "T_unique": int(len(np.unique(T))),
                    "Y_mean": float(np.mean(Y)),
                    "Y_std": float(np.std(Y)),
                    "Y_unique": int(len(np.unique(Y))),
                    "correlation_T_Y": float(np.corrcoef(T, Y)[0, 1])
                    if len(np.unique(T)) > 1
                    else 0.0,
                },
            )

            # Encode effect modifiers (handle categorical).
            # Shard 07 C2: a provenance column (is_synthetic) must NEVER enter
            # the CATE design matrix as an effect modifier, even if a caller
            # passed it explicitly. Neither may the treatment or the outcome
            # (wave 53): with T inside X the propensity model is perfect and the
            # DML residual is zero — live ATE -0.514 on a 0/1 outcome.
            effect_modifiers, dropped_modifiers = sanitize_effect_modifiers(state)
            if dropped_modifiers:
                logger.warning(
                    "Dropped effect modifiers that must not enter the CATE design matrix",
                    extra={
                        "node": "cate_estimator",
                        "dropped": dropped_modifiers,
                        "treatment_var": state["treatment_var"],
                        "outcome_var": state["outcome_var"],
                    },
                )
            X_df = df[effect_modifiers].copy()
            X = self._encode_features(X_df)

            # Phase 3 (Issue #237): route confounders into CausalForestDML's
            # nuisance-model ``W`` parameter when the caller (or upstream
            # data_preparer via role_attributions) has identified them.
            #
            # Precedence (high → low):
            #   1. Explicit ``state["confounders"]`` (caller override).
            #   2. Derived from ``state["role_attributions"]`` filtered by
            #      the C1 trust-gate (manifest|kg unconditional, llm only
            #      when ``evaluator_satisfied=True``).
            #   3. None — preserves the pre-#237 baseline behavior (W=None).
            #
            # segment_vars are NOT considered: they are for post-hoc
            # CATE-by-segment analysis in ``_calculate_cate_by_segment()``.
            # Using segment_vars as W conflates segmentation with confounding
            # and can produce ATE=0 when segment categories absorb treatment
            # variation (the original rationale for the unconditional
            # W=None default).
            confounders = self._resolve_confounders(state, list(df.columns))
            if confounders:
                W_df = df[confounders].copy()
                W = self._encode_features(W_df)
                logger.info(
                    "CATE nuisance controls routed",
                    extra={
                        "node": "cate_estimator",
                        "confounders": confounders,
                        "confounder_count": len(confounders),
                    },
                )
            else:
                W = None

            # Fit Causal Forest
            is_binary_treatment = self._is_binary(T)

            # EconML's CausalForestDML requires n_estimators to be divisible by subforest_size
            # Default subforest_size is 4, so adjust n_estimators to be divisible by 4
            subforest_size = 4
            raw_n_estimators = state.get("n_estimators", 100)
            # Round up to nearest multiple of subforest_size
            n_estimators = (
                (raw_n_estimators + subforest_size - 1) // subforest_size
            ) * subforest_size

            if n_estimators != raw_n_estimators:
                logger.info(
                    f"Adjusted n_estimators from {raw_n_estimators} to {n_estimators} "
                    f"(must be divisible by subforest_size={subforest_size})",
                    extra={"node": "cate_estimator"},
                )

            cf = CausalForestDML(
                model_y=RandomForestRegressor(
                    n_estimators=50,
                    min_samples_leaf=5,
                    min_impurity_decrease=1e-7,
                    random_state=42,
                ),
                model_t=(
                    RandomForestClassifier(
                        n_estimators=50,
                        min_samples_leaf=5,
                        min_impurity_decrease=1e-7,
                        random_state=42,
                    )
                    if is_binary_treatment
                    else RandomForestRegressor(
                        n_estimators=50,
                        min_samples_leaf=5,
                        min_impurity_decrease=1e-7,
                        random_state=42,
                    )
                ),
                discrete_treatment=is_binary_treatment,
                n_estimators=n_estimators,
                subforest_size=subforest_size,
                min_samples_leaf=state.get("min_samples_leaf", 10),
                min_impurity_decrease=1e-7,
                random_state=42,
            )

            # Fit with timeout. cache_values=True keeps the cross-fitted DML
            # residuals (Y_res, T_res) on the estimator — required by the
            # honest segment-mean inference in _calculate_cate_by_segment.
            await asyncio.wait_for(
                asyncio.to_thread(cf.fit, Y, T, X=X, W=W, cache_values=True),
                timeout=self.timeout_seconds,
            )

            # Get overall ATE
            ate = cf.ate(X)

            # Get individual treatment effects
            cate_individual = cf.effect(X)

            # Diagnostic logging for debugging ATE=0 issue
            logger.info(
                "CausalForestDML results",
                extra={
                    "node": "cate_estimator",
                    "ate_raw": ate,
                    "ate_type": type(ate).__name__,
                    "cate_mean": float(np.mean(cate_individual)),
                    "cate_std": float(np.std(cate_individual)),
                    "cate_min": float(np.min(cate_individual)),
                    "cate_max": float(np.max(cate_individual)),
                    "is_binary_treatment": is_binary_treatment,
                },
            )

            # Calculate heterogeneity score
            heterogeneity = self._calculate_heterogeneity(cate_individual, ate)

            # Get feature importance. Use the sanitized ``effect_modifiers``
            # (PROVENANCE_DROP_COLS stripped) so the importance keys line up 1:1
            # with the columns the forest was actually trained on (cf was fit on
            # the sanitized X above) — Shard 07 C2.
            feature_importance = dict(
                zip(
                    effect_modifiers,
                    (
                        cf.feature_importances_.tolist()
                        if hasattr(cf, "feature_importances_")
                        else [0] * len(effect_modifiers)
                    ),
                    strict=False,
                )
            )

            # Calculate CATE by segment. Pass the sanitized effect_modifiers so
            # the per-segment design matrix matches the trained forest and never
            # includes a provenance column (Shard 07 C2).
            cate_by_segment = await self._calculate_cate_by_segment(
                df,
                cf,
                state["segment_vars"],
                effect_modifiers,
                state.get("significance_level", 0.05),
                T,
            )

            estimation_time = int((time.time() - start_time) * 1000)

            logger.info(
                "CATE estimation complete",
                extra={
                    "node": "cate_estimator",
                    "overall_ate": float(ate),
                    "heterogeneity_score": heterogeneity,
                    "segment_count": len(cate_by_segment),
                    "latency_ms": estimation_time,
                },
            )

            return {
                **state,
                "overall_ate": float(ate),
                "heterogeneity_score": heterogeneity,
                "feature_importance": feature_importance,
                "cate_by_segment": cate_by_segment,
                "estimation_latency_ms": estimation_time,
                # Library provenance for the Library Validation card; the
                # uplift node extends this with "causalml" when it fits one.
                "libraries_executed": ["econml"],
                "status": "analyzing",
            }

        except asyncio.TimeoutError:
            logger.error(
                "CATE estimation timed out",
                extra={"node": "cate_estimator", "timeout_seconds": self.timeout_seconds},
            )
            return {
                **state,
                "errors": [
                    {"node": "cate_estimator", "error": f"Timed out after {self.timeout_seconds}s"}
                ],
                "status": "failed",
            }
        except Exception as e:
            logger.error(
                "CATE estimation failed",
                extra={"node": "cate_estimator", "error": str(e)},
                exc_info=True,
            )
            return {
                **state,
                "errors": [
                    {"node": "cate_estimator", "error": str(e), "traceback": traceback.format_exc()}
                ],
                "status": "failed",
            }

    async def _fetch_data(self, state: HeterogeneousOptimizerState) -> pd.DataFrame:
        """Fetch data for CATE estimation.

        Data source priority:
        1. tier0_data passthrough (from tier0 testing framework)
        2. Primary data connector (Supabase)
        3. Raises ValueError if insufficient data (NO mock fallback)

        Args:
            state: HeterogeneousOptimizerState with data configuration

        Returns:
            DataFrame with required columns for CATE estimation

        Raises:
            ValueError: If insufficient data available
        """
        required_columns = (
            [state["treatment_var"], state["outcome_var"]]
            + state["effect_modifiers"]
            + state["segment_vars"]
            # Confounders (issue #237) are residualized as the DML W; they must
            # be fetched too, or _resolve_confounders silently drops them as
            # "absent from available_columns" and the CATE stays confounded.
            + list(state.get("confounders") or [])
        )
        # is_synthetic must never be an effect modifier / segment var (Shard 07 C2).
        from src.repositories.provenance import PROVENANCE_DROP_COLS

        required_columns = [c for c in required_columns if c not in PROVENANCE_DROP_COLS]

        # Priority 1: Use tier0 passthrough data if available. #1734: the frame
        # rides the process-local frame registry (state carries only the
        # tier0_frame_ref handle — a frame in state would re-stream to the chat
        # client via every on_chain_* event); direct node callers may still hand
        # an in-dict frame, which can never reach a compiled graph.
        from src.utils.frame_registry import resolve_state_frame

        tier0_data = resolve_state_frame(state)
        if tier0_data is not None and len(tier0_data) >= 100:
            # Validate required columns exist in tier0 data
            missing_cols = [c for c in required_columns if c not in tier0_data.columns]
            if not missing_cols:
                logger.info(
                    f"Using tier0 passthrough data ({len(tier0_data)} rows)",
                    extra={
                        "node": "cate_estimator",
                        "data_source": "tier0_passthrough",
                        "row_count": len(tier0_data),
                    },
                )
                return tier0_data
            else:
                logger.warning(
                    f"Tier0 data missing columns {missing_cols}, trying primary connector",
                    extra={"node": "cate_estimator", "missing_columns": missing_cols},
                )

        # Priority 2: Fetch from primary data connector (Supabase)
        df = await self.data_connector.query(
            source=state["data_source"],
            columns=list(set(required_columns)),
            filters=state.get("filters"),
        )

        # Validate we have sufficient data
        if df is None or len(df) < 100:
            row_count = len(df) if df is not None else 0
            raise ValueError(
                f"Insufficient data for CATE estimation ({row_count} rows, need >= 100). "
                f"Either pass tier0_data with required columns or configure Supabase. "
                f"Required columns: {required_columns}"
            )

        logger.info(
            f"Using primary connector data ({len(df)} rows)",
            extra={
                "node": "cate_estimator",
                "data_source": "primary_connector",
                "row_count": len(df),
            },
        )
        return df

    def _resolve_confounders(
        self,
        state: HeterogeneousOptimizerState,
        available_columns: List[str],
    ) -> List[str]:
        """Determine the confounder column list to route into ``W``.

        Phase 3 (Issue #237) — see ``execute()`` for the precedence rules.

        Args:
            state: Current HeterogeneousOptimizerState. Reads optional
                ``confounders`` and ``role_attributions`` keys.
            available_columns: Columns present on the fetched DataFrame.
                Confounders the caller / role_attributions reference that
                are absent from ``available_columns`` are silently dropped
                with a warning (the alternative — raising — would break
                callers that mix declared confounders across multiple
                data sources).

        Returns:
            Ordered, de-duplicated list of column names to use as ``W``.
            Empty list (NOT ``None``) when no confounders should be routed;
            the caller branches on truthiness.
        """
        # Source-1: explicit caller override. ``state.get`` is used (not
        # subscripting) because ``confounders`` is ``NotRequired``.
        explicit = state.get("confounders")
        if explicit:
            resolved: List[str] = list(explicit)
        else:
            # Source-2: derived from role_attributions. Only attributions
            # whose causal_role == "confounder" AND pass the C1 trust-gate
            # (manifest|kg unconditional, llm gated on evaluator_satisfied)
            # contribute. ``should_act`` is the single source of truth for
            # the gate so any future policy change (e.g. ADVISORY vs
            # STRICT) lands in one place.
            from src.data.role_attribution import RoleAttribution, should_act

            role_attrs = state.get("role_attributions") or []
            resolved = []
            for attr in role_attrs:
                # Defensive: state typing allows ``List[Dict[str, Any]]``
                # so we shape-check before consuming.
                if not isinstance(attr, dict):
                    continue
                feature = attr.get("feature")
                role = attr.get("causal_role")
                if not isinstance(feature, str) or role != "confounder":
                    continue
                # Cast through RoleAttribution to exercise the same gate
                # contract Phase 2 will use; should_act tolerates the
                # dict shape because RoleAttribution is a TypedDict.
                if not should_act(cast("RoleAttribution", attr)):
                    continue
                resolved.append(feature)

        # De-duplicate while preserving first-seen order. ``dict.fromkeys``
        # is the canonical Py3.7+ idiom for order-preserving uniqueness.
        # Shard 07 C2: a provenance column (is_synthetic) must NEVER be routed
        # into the nuisance ``W`` matrix as a confounder, even if a caller
        # passed it explicitly. Strip PROVENANCE_DROP_COLS here.
        from src.repositories.provenance import PROVENANCE_DROP_COLS

        deduped = [c for c in dict.fromkeys(resolved) if c not in PROVENANCE_DROP_COLS]

        # Drop any column not present on the fetched DataFrame. This
        # tolerates schema drift between the manifest / role-attribution
        # producer and the connector's row payload without failing the
        # whole estimation.
        available = set(available_columns)
        present = [c for c in deduped if c in available]
        missing = [c for c in deduped if c not in available]
        if missing:
            logger.warning(
                "Confounders referenced but not present in dataframe",
                extra={
                    "node": "cate_estimator",
                    "missing_confounders": missing,
                    "available_count": len(available_columns),
                },
            )
        return present

    def _is_binary(self, T: np.ndarray) -> bool:
        """Check if treatment is binary."""
        unique_vals = np.unique(T)
        return len(unique_vals) == 2

    def _encode_features(self, df: pd.DataFrame) -> np.ndarray:
        """Encode features, handling categorical columns.

        Uses label encoding for categorical columns.

        Args:
            df: DataFrame with features

        Returns:
            Numpy array with encoded features
        """
        result = df.copy()

        for col in result.columns:
            if result[col].dtype == "object" or str(result[col].dtype) == "category":
                # Label encode categorical columns
                categories = result[col].unique()
                cat_to_int = {cat: i for i, cat in enumerate(categories)}
                result[col] = result[col].map(cat_to_int).astype(float)

        return cast("np.ndarray[Any, Any]", result.values)

    def _calculate_heterogeneity(self, cate_individual: np.ndarray, ate: float) -> float:
        """Calculate heterogeneity score (coefficient of variation).

        Returns 0-1 score where higher = more heterogeneity.
        """
        std = np.std(cate_individual)
        if ate == 0:
            return 0.0
        cv = std / abs(ate)
        # Normalize to 0-1 scale (CV/2, capped at 1.0). float() because cv is
        # numpy.float64 and min() preserves it — a numpy scalar in the output
        # kills orchestrator checkpoint serialization (#1732).
        return float(min(cv / 2, 1.0))

    @staticmethod
    def _extract_dml_residuals(cf, n_rows: int) -> "tuple[np.ndarray, np.ndarray] | None":
        """Return the cross-fitted DML residuals ``(y_res, t_res)`` as 1-D arrays.

        Available when the forest was fit with ``cache_values=True``. Returns
        ``None`` (caller falls back to the forest's per-point intervals) when:

        * the estimator does not expose ``residuals_`` (e.g. test doubles or a
          fit without cached values),
        * the treatment residual has more than one column (multi-valued
          discrete treatment — the single-θ GATE moment below does not apply;
          note the node binarizes continuous treatments at the median upstream,
          so this is a defensive guard rather than a live path),
        * the residual length does not match the estimation frame (alignment
          guard: the segment masks index ``df`` positionally).
        """
        try:
            y_res, t_res, _, _ = cf.residuals_
        except Exception:  # noqa: BLE001 — absence of cached residuals is expected on fallbacks
            logger.warning(
                "DML residuals unavailable; segment CIs fall back to per-point intervals",
                extra={"node": "cate_estimator"},
            )
            return None
        y_arr = np.asarray(y_res, dtype=float)
        t_arr = np.asarray(t_res, dtype=float)
        if t_arr.ndim > 1:
            if t_arr.shape[1] != 1:
                logger.warning(
                    "Multi-column treatment residuals (%s); segment CIs fall back "
                    "to per-point intervals",
                    t_arr.shape,
                    extra={"node": "cate_estimator"},
                )
                return None
            t_arr = t_arr.ravel()
        if y_arr.ndim > 1:
            y_arr = y_arr.ravel()
        if len(y_arr) != n_rows or len(t_arr) != n_rows:
            logger.warning(
                "Residual length mismatch (y=%d, t=%d, rows=%d); segment CIs fall "
                "back to per-point intervals",
                len(y_arr),
                len(t_arr),
                n_rows,
                extra={"node": "cate_estimator"},
            )
            return None
        return y_arr, t_arr

    @staticmethod
    def _gate_interval(
        y_res: np.ndarray, t_res: np.ndarray, alpha: float
    ) -> "tuple[float, float, float] | None":
        """Segment-mean effect + CI from the partially-linear DML moment (GATE).

        Within a segment S, the group average treatment effect is estimated by
        the residual-on-residual projection

            θ_S = Σ_{i∈S} T̃_i Ỹ_i / Σ_{i∈S} T̃_i²

        with the heteroskedasticity-robust standard error from the orthogonal
        score ψ_i = T̃_i (Ỹ_i − T̃_i θ_S):  se = sqrt(Σ ψ_i²) / Σ T̃_i².
        This is the standard DoubleML GATE for the partially linear model and
        shrinks ~1/√n, unlike the forest's per-point prediction intervals.

        Returns ``None`` for degenerate segments (no treatment-residual
        variance / non-finite inputs) so the caller can fall back.
        """
        denom = float(np.sum(t_res * t_res))
        if not np.isfinite(denom) or denom <= 1e-12:
            return None
        theta = float(np.sum(t_res * y_res) / denom)
        psi = t_res * (y_res - t_res * theta)
        se = float(np.sqrt(np.sum(psi * psi)) / denom)
        if not (np.isfinite(theta) and np.isfinite(se)):
            return None
        z = z_score_for_alpha(alpha)
        return theta, theta - z * se, theta + z * se

    async def _calculate_cate_by_segment(
        self,
        df: pd.DataFrame,
        cf,
        segment_vars: List[str],
        effect_modifiers: List[str],
        alpha: float,
        T: np.ndarray,
    ) -> Dict[str, List[CATEResult]]:
        """Calculate the segment-mean CATE + CI for each segment value.

        Honest-inference contract (2026-07-05): the point estimate and CI come
        from the residual-based GATE (``_gate_interval``) computed on the DML
        cross-fitted residuals, NOT from averaging the forest's per-individual
        ``effect_interval`` bounds. Averaging individual bounds produces an
        individual-level prediction interval whose width never shrinks with
        segment size — on the live cohort every segment CI was ±17.7pp
        regardless of n (1.4k–5.9k), so a +11pp segment-mean effect could
        never test significant (the /ai-insights "0/14 significant" incident).
        The forest still provides individual CATEs, heterogeneity, feature
        importances and downstream personalization; the GATE answers the
        narrower question "is this segment's AVERAGE effect nonzero?".

        The forest per-point path is retained ONLY as a fallback (estimators
        without cached residuals, multi-valued treatments, degenerate
        segments) and is honestly conservative there.
        """

        cate_by_segment = {}

        residuals = self._extract_dml_residuals(cf, len(df))

        # Encode the effect modifiers ONCE over the FULL frame, identical to how the
        # forest's training matrix was built (``self._encode_features(df[effect_modifiers])``
        # at fit time). The fitted CausalForest only ever saw those deterministic codes,
        # so the per-segment design matrix MUST use the same encoding — feeding the raw
        # string columns to ``cf.effect`` otherwise raises "could not convert string to
        # float" for any categorical effect modifier (e.g. the conversion-KPI substrate's
        # trigger_type/priority). We encode the full frame and POSITIONALLY mask per
        # segment; the segment mask stays on the RAW ``segment_var`` so categorical
        # segment values still match.
        X_all = self._encode_features(df[effect_modifiers])

        for segment_var in segment_vars:
            # Guard: a label-augmented segment column may be absent from the frame
            # (e.g. not loaded for this data_source); skip rather than KeyError.
            if segment_var not in df.columns:
                logger.warning(
                    "Segment column %r absent from frame; skipping",
                    segment_var,
                    extra={"node": "cate_estimator"},
                )
                continue
            segment_results = []

            for segment_value in df[segment_var].unique():
                mask = (df[segment_var] == segment_value).to_numpy()
                segment_df = df[mask]

                if len(segment_df) < 10:
                    continue

                gate = (
                    self._gate_interval(residuals[0][mask], residuals[1][mask], alpha)
                    if residuals is not None
                    else None
                )
                if gate is not None:
                    # Primary path: residual-based GATE (segment-mean effect
                    # with an honest ~1/sqrt(n) CI). See method docstring.
                    cate_mean, ci_lower, ci_upper = gate
                else:
                    # Fallback path ONLY (no cached residuals / multi-valued
                    # treatment / degenerate segment): the forest's per-point
                    # intervals, averaged. This is an INDIVIDUAL-level interval
                    # applied to a group mean — conservative by construction.
                    X_segment = X_all[mask]
                    cate = cf.effect(X_segment)
                    cate_mean = float(np.mean(cate))
                    try:
                        cate_interval = cf.effect_interval(X_segment, alpha=alpha)
                        ci_lower = float(np.mean(cate_interval[0]))
                        ci_upper = float(np.mean(cate_interval[1]))
                    except Exception:
                        # #27: derive the fallback z-score from the requested
                        # significance level (alpha) instead of a hardcoded 1.96,
                        # so the fallback CI is at the SAME level as the primary
                        # ``effect_interval(alpha=...)`` path. At the default
                        # alpha=0.05 this is ~1.96 (legacy behavior unchanged); a
                        # 90% request (alpha=0.10) now correctly yields ~1.645*sigma.
                        z = z_score_for_alpha(alpha)
                        ci_lower = cate_mean - z * float(np.std(cate))
                        ci_upper = cate_mean + z * float(np.std(cate))

                # Determine statistical significance
                significant = (ci_lower > 0) or (ci_upper < 0)

                segment_results.append(
                    CATEResult(
                        segment_name=segment_var,
                        segment_value=str(segment_value),
                        cate_estimate=cate_mean,
                        cate_ci_lower=ci_lower,
                        cate_ci_upper=ci_upper,
                        sample_size=len(segment_df),
                        statistical_significance=significant,
                        # Observed treated share in this segment, from the SAME
                        # T the forest was fit on (positionally aligned with df
                        # via the finite-mask reset above). Replaces the former
                        # policy_learner assumption of a flat 50% baseline. For
                        # a binarized continuous treatment this is the share
                        # above the cohort median.
                        treatment_rate=float(np.mean(T[mask])),
                    )
                )

            # Sort by CATE estimate
            segment_results.sort(key=lambda x: x["cate_estimate"], reverse=True)
            cate_by_segment[segment_var] = segment_results

        return cate_by_segment
