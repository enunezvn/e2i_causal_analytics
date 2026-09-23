"""Data loader node for data_preparer agent.

Supports three ingestion paths:
  - Supabase tables (default; ``data_source`` is a table name string, OR a table
    dict ``{"type": "table", "table": ..., "filters": {...}, "columns": [...]}`` —
    see ``_load_from_supabase``)
  - Sample data generator (``use_sample_data=True``)
  - Local files (``data_source`` is a dict — see ``_load_from_files``)

Split policy for tables (#2207 split contract, owner decision 2026-09-23): a table
that carries a ``data_split`` column declares its own split and it is honoured
verbatim — exactly as ``_load_from_files`` does via ``_split_from_column``. A table
without one keeps the temporal ``val_days`` / ``test_days`` split (no holdout).
"""

import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import numpy as np
import pandas as pd

# Use direct module imports to avoid circular import with src.repositories
from src.repositories.data_splitter import SplitConfig, get_data_splitter
from src.repositories.ml_data_loader import get_ml_data_loader
from src.repositories.sample_data import SampleDataGenerator

from ..ingestion import FileIngestor, IngestionError
from ..state import DataPreparerState

logger = logging.getLogger(__name__)

# Hard cap on a table cohort load. PostgREST on this deployment has no
# PGRST_DB_MAX_ROWS (docker inspect supabase-rest, 2026-09-23), so the loader's
# ``.limit()`` is the only cap — and a frame that reaches it may be TRUNCATED. A
# silently truncated cohort is a plausible-wrong model, so ``_load_precomputed_split``
# fails loud at ``len(df) >= _TABLE_ROW_LIMIT`` instead of training on a prefix.
_TABLE_ROW_LIMIT = 100_000

# Labels ``_split_from_column`` understands (converter / migration vocabulary).
_SPLIT_LABELS = ("train", "validation", "val", "test", "holdout")


def _legacy_split_config() -> SplitConfig:
    """Build the 60/20/10/10 holdout-bearing config the model_trainer
    ``split_enforcer`` single-mode contract expects (#44 holdout enlargement
    2026-07-21: test 0.15→0.10, holdout 0.05→0.10, lockstep with the enforcer's
    expected_ratios and the seed quota in generators/base.py).

    The splitter's generic defaults are 60/20/20 with a 0-sample holdout,
    which the enforcer hard-fails (``ratios_valid=False``) both on the
    empty-holdout check and on the test-ratio drift (20% vs 10%). The
    ``random_split`` and ``entity_split`` fallbacks are reconciled here: both
    cleanly honour these config ratios (entity_split cuts cumulative hash
    bands on them, mirroring SplitAwareRepository.assign_split). The temporal
    fallback is intentionally left unchanged — it splits on day windows
    (val_days/test_days), takes no ratio config, and has no holdout concept;
    grafting one on is separate surgery.
    """
    return SplitConfig(
        train_ratio=0.60,
        val_ratio=0.20,
        test_ratio=0.10,
        holdout_ratio=0.10,
    )


async def load_data(state: DataPreparerState) -> Dict[str, Any]:
    """Load and split data for ML training.

    This node:
    1. Extracts data source configuration from scope_spec
    2. Loads data from Supabase using MLDataLoader
    3. Applies appropriate splitting strategy (temporal, entity, or combined)
    4. Populates train_df, validation_df, test_df, holdout_df in state

    ``data_source`` shapes (routed below):
      - ``str`` — a Supabase table name (temporal split, or the table's own
        ``data_split`` column when it carries one).
      - ``{"type": "table", "table": <name>, "filters": {...}, "columns": [...]}`` —
        the retrain cohort contract (#2207): ``filters`` are merged OVER
        ``scope_spec.filters`` (the contract wins; ``scope_builder`` never emits
        filters, so this is the only way a brand partition reaches the query); an
        explicit ``is_synthetic`` filter forces ``include_synthetic=True`` so the
        provenance default-exclude cannot contradict the contract; ``columns``
        scopes the SELECT (a whole-table load of ``patient_journeys`` would learn
        its own label from ``days_to_treatment`` — see
        ``gold_standard_eval.feature_builder.LEAKAGE_DENYLIST``) and MUST contain
        ``scope_spec.prediction_target`` (fail loud otherwise — this also catches
        the scope_definer's ``adopt* -> will_adopt`` target rewrite).
      - ``{"type": "file_dir" | "files", ...}`` — local files (``_load_from_files``).

    Args:
        state: Current agent state

    Returns:
        Updated state with loaded data splits
    """
    start_time = datetime.now()
    experiment_id = state.get("experiment_id", "unknown")
    logger.info(f"Loading data for experiment {experiment_id}")

    try:
        scope_spec = state.get("scope_spec", {})
        data_source = state.get("data_source") or scope_spec.get("data_source", "business_metrics")

        # Extract configuration from scope_spec
        filters = scope_spec.get("filters", {})
        date_column = scope_spec.get("date_column", "created_at")
        entity_column = scope_spec.get("entity_column")
        split_date = scope_spec.get("split_date")
        val_days = scope_spec.get("val_days", 30)
        test_days = scope_spec.get("test_days", 30)
        use_sample_data = scope_spec.get("use_sample_data", False)

        # Route on data_source shape:
        #   - dict with "type" in {"file_dir", "files"} → local file ingestion
        #   - dict with "type" == "table" → Supabase table cohort contract
        #   - any other dict → fail loud (never str() a dict into a table name)
        #   - use_sample_data=True → synthetic generator
        #   - otherwise → Supabase table name
        if isinstance(data_source, dict) and data_source.get("type") in (
            "file_dir",
            "files",
        ):
            logger.info("Loading from files: %s", data_source.get("type"))
            dataset = _load_from_files(
                data_source=data_source,
                entity_column=entity_column,
                date_column=date_column,
            )
        elif isinstance(data_source, dict) and data_source.get("type") == "table":
            table = data_source.get("table")
            if not table or not isinstance(table, str):
                raise ValueError("data_source.table required for type='table'")
            contract_filters = data_source.get("filters") or {}
            merged_filters = {**filters, **contract_filters}
            raw_columns = data_source.get("columns")
            columns = [str(c) for c in raw_columns] if raw_columns is not None else None
            _require_target_in_columns(columns, scope_spec.get("prediction_target"))
            logger.info(
                "Loading table cohort contract: table=%s filters=%s columns=%s",
                table,
                merged_filters,
                columns,
            )
            dataset = await _load_from_supabase(
                data_source=table,
                filters=merged_filters,
                date_column=date_column,
                entity_column=entity_column,
                split_date=split_date,
                val_days=val_days,
                test_days=test_days,
                columns=columns,
                include_synthetic="is_synthetic" in merged_filters,
            )
            if all(len(dataset[k]) == 0 for k in ("train", "val", "test")):
                raise ValueError(
                    f"table cohort {table!r} with filters {merged_filters!r} loaded an "
                    "empty frame (no rows match, or the loader logged a query error)"
                )
        elif isinstance(data_source, dict):
            raise ValueError(
                f"Unknown data_source type: {data_source.get('type')!r} "
                "(expected 'table', 'file_dir' or 'files')"
            )
        elif use_sample_data:
            # Narrow to str for the table-name branches (dict paths handled above).
            ds_str = data_source if isinstance(data_source, str) else str(data_source)
            logger.info("Using sample data generator")
            dataset = await _load_sample_data(
                data_source=ds_str,
                n_samples=scope_spec.get("sample_size", 1000),
                entity_column=entity_column,
                date_column=date_column,
            )
        else:
            ds_str = data_source if isinstance(data_source, str) else str(data_source)
            dataset = await _load_from_supabase(
                data_source=ds_str,
                filters=filters,
                date_column=date_column,
                entity_column=entity_column,
                split_date=split_date,
                val_days=val_days,
                test_days=test_days,
                include_synthetic="is_synthetic" in filters,
            )

        # Calculate loading duration
        load_duration = (datetime.now() - start_time).total_seconds()

        # Prepare update
        updates = {
            "train_df": dataset["train"],
            "validation_df": dataset["val"],
            "test_df": dataset["test"],
            "holdout_df": dataset.get("holdout"),
        }

        logger.info(
            f"Data loaded successfully: "
            f"train={len(dataset['train'])}, "
            f"val={len(dataset['val'])}, "
            f"test={len(dataset['test'])}, "
            f"duration={load_duration:.2f}s"
        )

        return updates

    except Exception as e:
        logger.error(f"Data loading failed: {e}", exc_info=True)
        return {
            "error": str(e),
            "error_type": "data_loading_error",
            "blocking_issues": [f"Data loading failed: {str(e)}"],
        }


def _require_target_in_columns(columns: Optional[list[str]], target: Any) -> None:
    """A column-scoped table contract must select its own prediction target."""
    if columns is None or not isinstance(target, str) or not target:
        return
    if target not in columns:
        raise ValueError(
            f"contract columns omit the target column {target!r}: columns={columns!r}. "
            "A table cohort contract must select scope_spec.prediction_target (note the "
            "scope_definer rewrites some requested targets, e.g. 'adopt*' -> 'will_adopt')."
        )


async def _table_has_data_split(
    loader: Any, table: str, filters: Dict[str, Any], include_synthetic: bool
) -> bool:
    """1-row presence probe for a precomputed ``data_split`` column.

    Selects ``*`` on one row under the SAME filters / provenance mode as the real
    load, so a table without the column never raises a PostgREST 42703 (which the
    loader would log at ERROR and swallow into an empty frame). An empty cohort reads
    as ``False``; the caller's route then reports it.
    """
    probe = await loader.load_table_sample(
        table, filters=filters, limit=1, include_synthetic=include_synthetic
    )
    return "data_split" in probe.columns


async def _load_precomputed_split(
    loader: Any,
    table: str,
    filters: Dict[str, Any],
    columns: Optional[list[str]],
    include_synthetic: bool,
) -> Dict[str, Any]:
    """Load a table that carries ``data_split`` and partition on it verbatim."""
    select: Optional[list[str]] = None
    if columns:
        select = list(columns) + ([] if "data_split" in columns else ["data_split"])
    df = await loader.load_table_sample(
        table,
        filters=filters,
        limit=_TABLE_ROW_LIMIT,
        columns=select,
        include_synthetic=include_synthetic,
    )
    if df.empty:
        raise ValueError(
            f"table cohort {table!r} with filters {filters!r} loaded an empty frame "
            f"(no rows match, or the loader logged a query error — e.g. a column in "
            f"{select!r} does not exist)"
        )
    if len(df) >= _TABLE_ROW_LIMIT:
        raise ValueError(
            f"table cohort {table!r} with filters {filters!r} hit the "
            f"{_TABLE_ROW_LIMIT}-row load cap: the frame may be truncated, refusing to "
            "train on a prefix of the cohort"
        )
    if "data_split" not in df.columns:
        raise ValueError(
            f"table {table!r} probed with a data_split column but the full load has none"
        )
    logger.info(
        "Using precomputed 'data_split' column from table %s (%d rows, %d cols)",
        table,
        len(df),
        len(df.columns),
    )
    return _split_from_column(df)


async def _load_from_supabase(
    data_source: str,
    filters: Dict[str, Any],
    date_column: str,
    entity_column: Optional[str],
    split_date: Optional[str],
    val_days: int,
    test_days: int,
    columns: Optional[list[str]] = None,
    include_synthetic: bool = False,
) -> Dict[str, Any]:
    """Load data from Supabase and split.

    Precomputed split FIRST: if the table carries a ``data_split`` column it is
    honoured verbatim (``_load_precomputed_split`` -> ``_split_from_column``), which
    is the only way a table-sourced retrain reaches the trainer's ``split_enforcer``
    with the non-empty holdout and 60/20/10/10 ratios it requires. Otherwise the
    temporal path below runs unchanged (``MLDataset`` has no holdout and
    ``combined_split`` yields none either — measured 2026-09-23).

    Args:
        data_source: Table name
        filters: Query filters
        date_column: Date column for temporal splits
        entity_column: Entity column for entity-level splits
        split_date: Reference date for temporal split
        val_days: Days for validation set
        test_days: Days for test set
        columns: Columns to SELECT (None = all); the split column is added on the
            precomputed path
        include_synthetic: opt out of the provenance default-exclude (the caller
            sets it when the filters name ``is_synthetic`` explicitly)

    Returns:
        Dict with train, val, test, holdout DataFrames
    """
    loader = get_ml_data_loader()

    if await _table_has_data_split(loader, data_source, filters, include_synthetic):
        return await _load_precomputed_split(
            loader, data_source, filters, columns, include_synthetic
        )

    # Load with temporal split
    dataset = await loader.load_for_training(
        table=data_source,
        filters=filters,
        date_column=date_column,
        split_date=split_date,
        val_days=val_days,
        test_days=test_days,
        columns=columns,
        include_synthetic=include_synthetic,
    )

    result = {
        "train": dataset.train,
        "val": dataset.val,
        "test": dataset.test,
        "holdout": None,
    }

    # If entity column specified, apply entity-level split to ensure no leakage
    if entity_column and entity_column in dataset.train.columns:
        logger.info(f"Applying entity-level split on column: {entity_column}")
        splitter = get_data_splitter()

        # Combined temporal + entity split
        # This is already temporal, now ensure entity integrity
        combined_result = splitter.combined_split(
            # F16: DataFrame.append was removed in pandas 2.x — combine the temporal
            # splits with pd.concat before the entity-level re-split (the downstream
            # split keys on date/entity COLUMNS, so a fresh index is safe).
            pd.concat([dataset.train, dataset.val, dataset.test], ignore_index=True),
            date_column=date_column,
            entity_column=entity_column,
            split_date=split_date,
            val_days=val_days,
            test_days=test_days,
        )

        result = {
            "train": combined_result.train,
            "val": combined_result.val,
            "test": combined_result.test,
            "holdout": combined_result.holdout,
        }

    return result


def _drop_unhashable_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Drop object-dtype columns containing any unhashable cell.

    pandas marks any column with a list/dict/set/tuple cell as object dtype
    but cannot compute set-based aggregations like ``nunique()`` /
    ``value_counts()`` because those types are unhashable. CSU's
    ``patient_journeys.json`` carries list-typed metadata cols
    (``comorbidities``, ``secondary_diagnosis_codes``, ``data_sources_matched``)
    that crash multiple downstream nodes (``leakage_detector``,
    ``data_transformer``, ``baseline_computer``).

    The same logical "list-of-strings" cell can land as a Python ``list``
    (JSON → pandas roundtrip) OR as ``numpy.ndarray`` (Parquet → pandas
    roundtrip via pyarrow's ``ListArray`` decode). PR #105 caught the JSON
    case but not the Parquet case; iter-5 audit (2026-05-09) surfaced the
    Optum-init e2e crashing at ``baseline_computer.py:75`` with
    ``TypeError: unhashable type: 'numpy.ndarray'`` because the Optum
    converter writes Parquet, not JSON.

    Scans every non-null cell — sampling only ``iloc[0]`` would let a column
    whose first row is a scalar but later rows contain lists silently
    survive the filter (codex review HIGH-B on PR #105). ``Series.map`` with
    a short-circuiting ``.any()`` is O(n) but cheap relative to the
    downstream encoding it would otherwise crash; pandas does not guarantee
    type uniformity within an object-dtype column.

    Empty / all-null columns are left in place; they are benign for
    downstream nunique calls and may carry intentional contract semantics.

    Args:
        df: Raw DataFrame from file ingestion.

    Returns:
        DataFrame with unhashable-cell columns removed; warns naming them.
    """
    drop_cols: list[str] = []
    for col in df.columns:
        if not pd.api.types.is_object_dtype(df[col].dtype):
            continue
        non_null = df[col].dropna()
        if non_null.empty:
            continue
        if non_null.map(
            lambda v: isinstance(v, (list, dict, set, frozenset, tuple, np.ndarray))
        ).any():
            drop_cols.append(col)

    if not drop_cols:
        return df

    logger.warning(
        "Dropping %d non-encodable column(s) with unhashable cells: %s",
        len(drop_cols),
        drop_cols,
    )
    return df.drop(columns=drop_cols, errors="ignore")


def _load_from_files(
    data_source: Dict[str, Any],
    entity_column: Optional[str],
    date_column: str,
) -> Dict[str, Any]:
    """Load patient-journey data from local files.

    Accepted ``data_source`` shapes:
      - ``{"type": "file_dir", "path": "/path/to/dir"}`` — reads canonical
        ``e2i_ml_v3_*`` files from the directory.
      - ``{"type": "files", "paths": {"patient_journeys": "/...", ...}}`` —
        reads each file from an explicit mapping.

    Splitting policy:
      - If ``patient_journeys`` has a ``data_split`` column (produced
        upstream by a converter's chronological splitter), use it verbatim.
      - Otherwise fall back to ``get_data_splitter()`` — prefer entity
        split when ``entity_column`` exists, then temporal on ``date_column``,
        then random.

    Defensive cleanup: object-dtype columns whose cells are unhashable
    (list/dict/set/tuple) are dropped post-read via
    ``_drop_unhashable_columns``. This protects downstream nodes
    (``leakage_detector``, ``data_transformer``, ``baseline_computer``)
    that call ``nunique()`` / ``value_counts()`` indiscriminately on
    object cols.
    """
    ingestor = FileIngestor()

    if data_source["type"] == "file_dir":
        path = data_source.get("path")
        if not path:
            raise IngestionError("data_source.path required for type='file_dir'")
        frames = ingestor.ingest_directory(Path(path))
    elif data_source["type"] == "files":
        paths = data_source.get("paths")
        if not paths or not isinstance(paths, Mapping):
            raise IngestionError("data_source.paths required for type='files' (got: %r)" % (paths,))
        frames = ingestor.ingest_mapping(paths)
    else:
        raise IngestionError(f"Unknown file data_source type: {data_source.get('type')!r}")

    if "patient_journeys" not in frames:
        raise IngestionError(
            "File ingestion produced no 'patient_journeys' DataFrame — "
            "downstream nodes require it as the primary table"
        )

    df = _drop_unhashable_columns(frames["patient_journeys"])
    logger.info("Loaded patient_journeys: %d rows, %d cols", len(df), len(df.columns))

    # Prefer precomputed split from converter.
    if "data_split" in df.columns:
        logger.info("Using precomputed 'data_split' column from ingested data")
        return _split_from_column(df)

    # Fallback: standard splitter.
    splitter = get_data_splitter()
    if entity_column and entity_column in df.columns:
        logger.info("Applying entity split on '%s'", entity_column)
        # 60/20/10/10 hash bands — config-less entity_split falls back to the
        # generic 60/20/20/0 defaults, which the split_enforcer hard-fails.
        result = splitter.entity_split(
            df, entity_column=entity_column, config=_legacy_split_config()
        )
    elif date_column in df.columns:
        logger.info("Applying temporal split on '%s'", date_column)
        result = splitter.temporal_split(
            df,
            date_column=date_column,
            val_days=30,
            test_days=30,
        )
    else:
        logger.info("No entity or date column — applying random split")
        # Request the 60/20/10/10 holdout-bearing contract so the downstream
        # split_enforcer does not hard-fail on a 0-sample holdout.
        result = splitter.random_split(df, config=_legacy_split_config())

    return {
        "train": result.train,
        "val": result.val,
        "test": result.test,
        "holdout": result.holdout,
    }


def _split_from_column(df: pd.DataFrame) -> Dict[str, Any]:
    """Partition DataFrame by its ``data_split`` column.

    Accepts converter-produced labels: 'train', 'validation'/'val', 'test',
    'holdout'. Empty splits are returned as empty DataFrames matching the
    original schema. Rows whose label is none of those are DROPPED — with a
    warning naming the count and the stray labels (they used to vanish silently).
    """
    split_col = df["data_split"]
    train = df[split_col == "train"].reset_index(drop=True)
    val = df[split_col.isin(["validation", "val"])].reset_index(drop=True)
    test = df[split_col == "test"].reset_index(drop=True)
    holdout = df[split_col == "holdout"].reset_index(drop=True)
    dropped = len(df) - (len(train) + len(val) + len(test) + len(holdout))
    if dropped:
        stray = split_col[~split_col.isin(_SPLIT_LABELS)].astype(str).unique().tolist()
        logger.warning(
            "Dropped %d row(s) whose data_split label is not one of %s (stray labels: %s)",
            dropped,
            list(_SPLIT_LABELS),
            stray[:10],
        )
    logger.info(
        "Split from 'data_split' column: train=%d, val=%d, test=%d, holdout=%d",
        len(train),
        len(val),
        len(test),
        len(holdout),
    )
    return {
        "train": train,
        "val": val,
        "test": test,
        "holdout": holdout if len(holdout) > 0 else None,
    }


async def _load_sample_data(
    data_source: str,
    n_samples: int,
    entity_column: Optional[str],
    date_column: str,
) -> Dict[str, Any]:
    """Load sample data for testing/development.

    Args:
        data_source: Table name to emulate
        n_samples: Number of samples to generate
        entity_column: Entity column for entity-level splits
        date_column: Date column for temporal splits

    Returns:
        Dict with train, val, test DataFrames
    """
    generator = SampleDataGenerator(seed=42)
    splitter = get_data_splitter(random_seed=42)

    # Generate sample data based on table type
    if data_source == "business_metrics":
        df = generator.business_metrics(n_samples=n_samples)
    elif data_source == "predictions":
        df = generator.predictions(n_samples=n_samples)
    elif data_source == "triggers":
        df = generator.triggers(n_samples=n_samples)
    elif data_source == "patient_journeys":
        # Use ml_patients() for ML-ready patient data with discontinuation_flag
        # Use fresh date range (last 90 days) to pass timeliness checks
        # while leaving room for temporal splitting (val_days=30, test_days=30)
        end_date = datetime.now().isoformat()
        start_date = (datetime.now() - timedelta(days=90)).isoformat()
        df = generator.ml_patients(
            n_patients=n_samples,
            start_date=start_date,
            end_date=end_date,
        )
    elif data_source == "agent_activities":
        df = generator.agent_activities(n_samples=n_samples)
    elif data_source == "causal_paths":
        df = generator.causal_paths(n_samples=n_samples)
    else:
        # Default to business metrics
        df = generator.business_metrics(n_samples=n_samples)

    # Apply temporal split if date column exists
    if date_column in df.columns:
        result = splitter.temporal_split(
            df,
            date_column=date_column,
            val_days=30,
            test_days=30,
        )
    elif entity_column and entity_column in df.columns:
        # 60/20/10/10 hash bands — see _load_from_files' entity branch.
        result = splitter.entity_split(
            df, entity_column=entity_column, config=_legacy_split_config()
        )
    else:
        # Request the 60/20/10/10 holdout-bearing contract so the downstream
        # split_enforcer does not hard-fail on a 0-sample holdout.
        result = splitter.random_split(df, config=_legacy_split_config())

    return {
        "train": result.train,
        "val": result.val,
        "test": result.test,
        "holdout": result.holdout,
    }
