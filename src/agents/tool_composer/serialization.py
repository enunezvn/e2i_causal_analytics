"""JSON-safe serialization of composition results for memory contribution.

Why this exists (#1583)
-----------------------
``ToolComposer._contribute_to_memory`` dumps the ``CompositionResult`` with
``model_dump(mode="json")``. The result's ``Dict[str, Any]`` members carry
whatever the tools produced *and* whatever the executor threaded in: for every
step, ``_maybe_autopopulate_dataframe`` puts the real cohort
``pandas.DataFrame`` into ``ToolInput.parameters['estimation_data']`` and it is
also present in the shared ``ToolInput.context``. pydantic's JSON serializer
has no encoder for a DataFrame, so the dump raised
``PydanticSerializationError: Unable to serialize unknown type:
<class 'pandas.core.frame.DataFrame'>`` *before* ``contribute_to_memory`` ran —
and the catch-and-warn one frame up dropped the ENTIRE contribution (working
cache + episodic + procedural). Measured in the 2026-08-13 forced q08 replay.

Representation: summarize containers, never dump their contents
---------------------------------------------------------------
Production frames run to ~37.5k rows and the same frame is attached to every
step, so serializing contents would put megabytes into each contribution. It
would also buy nothing: the memory consumers
(``memory_hooks.store_composition`` → counts + tool sequence,
``store_composition_pattern`` → decomposition reasoning + plan steps,
``cache_composition_result`` → the whole dict, ``json.dumps(default=str)``-ed
into Redis) read none of the step payload's frame members. So:

* numpy **scalars** -> the native python value (lossless, tiny). Mirrors the
  ``src/api/routes/segments.py`` ``_to_native`` precedent for ``numpy.int64``.
* array-likes and frames (DataFrame / Series / Index / ndarray / pandas
  extension arrays) -> a compact structured summary: type, shape or length,
  columns, dtypes (positionally aligned with ``columns``) — enough to tell a
  reader *what* the step carried, marked ``__summarized__`` so nobody mistakes
  it for the data.
* anything else -> a type marker only, plus an honest WARNING naming the type.
  The ``repr`` is deliberately NOT included: contributions are persisted to
  Redis and Supabase, and an arbitrary object's repr can carry credentials or
  patient-level detail.

``dump_json_safe`` passes pydantic's ``fallback=`` hook, which fires only for
types the serializer does not recognise — so a result with no unserializable
member serializes exactly as it did before this module existed.
"""

import logging
from typing import Any, Dict, cast

from pydantic import BaseModel

logger = logging.getLogger(__name__)

#: Cap on how many column names / dtypes a frame summary carries. Wide feature
#: matrices exist; the summary is a descriptor, not an inventory.
MAX_SUMMARIZED_COLUMNS = 50


def _type_name(obj: Any) -> str:
    """Fully-qualified type name, matching the pydantic error text's class path."""
    cls = type(obj)
    module = getattr(cls, "__module__", "") or ""
    qualname = getattr(cls, "__qualname__", cls.__name__)
    return f"{module}.{qualname}" if module else qualname


def _summarize_dataframe(frame: Any) -> Dict[str, Any]:
    """Compact descriptor for a pandas DataFrame — shape, columns, dtypes.

    ``dtypes`` is positionally aligned with ``columns`` rather than keyed by
    column name: a frame can carry duplicate labels (a sloppy merge) or
    MultiIndex columns that stringify to the same key, and a dict would
    silently drop one of them.
    """
    described = [(str(column), str(dtype)) for column, dtype in frame.dtypes.items()]
    kept = described[:MAX_SUMMARIZED_COLUMNS]
    summary: Dict[str, Any] = {
        "__type__": _type_name(frame),
        "__summarized__": True,
        "shape": [int(frame.shape[0]), int(frame.shape[1])],
        "columns": [column for column, _ in kept],
        "dtypes": [dtype for _, dtype in kept],
    }
    if len(described) > len(kept):
        summary["columns_truncated"] = True
    return summary


def _summarize_sequence(values: Any) -> Dict[str, Any]:
    """Compact descriptor for a Series / Index / extension array — length, dtype."""
    summary: Dict[str, Any] = {
        "__type__": _type_name(values),
        "__summarized__": True,
        "length": int(len(values)),
        "dtype": str(getattr(values, "dtype", "unknown")),
    }
    name = getattr(values, "name", None)
    if name is not None:
        summary["name"] = str(name)
    return summary


def _summarize_ndarray(array: Any) -> Dict[str, Any]:
    """Compact descriptor for a numpy array — shape, dtype, size."""
    return {
        "__type__": _type_name(array),
        "__summarized__": True,
        "shape": [int(dim) for dim in array.shape],
        "dtype": str(array.dtype),
        "size": int(array.size),
    }


def json_safe_fallback(obj: Any) -> Any:
    """Replace an object pydantic's JSON serializer cannot encode.

    Invoked by ``model_dump(mode="json", fallback=...)`` for unknown types only.
    Never raises: an unrecognised object degrades to a type marker so that one
    stray member cannot drop the whole memory contribution.
    """
    # Imported lazily: this only runs once a numpy/pandas object has already
    # been produced, so the modules are necessarily loaded (segments.py
    # ``_to_native`` precedent). The guard keeps the helper usable in a slim
    # environment where neither is installed.
    try:
        import numpy as np
    except ImportError:  # pragma: no cover - numpy is a hard dependency here
        np = None  # type: ignore[assignment]
    try:
        import pandas as pd
    except ImportError:  # pragma: no cover - pandas is a hard dependency here
        pd = None  # type: ignore[assignment]

    if np is not None:
        if isinstance(obj, np.ndarray):
            return _summarize_ndarray(obj)
        if isinstance(obj, np.generic):
            # numpy scalar (int64 / bool_ / ...): lossless native value.
            return obj.item()
        if isinstance(obj, np.dtype):
            return str(obj)

    if pd is not None:
        if isinstance(obj, pd.DataFrame):
            return _summarize_dataframe(obj)
        if isinstance(obj, (pd.Series, pd.Index)):
            return _summarize_sequence(obj)
        if isinstance(obj, pd.api.extensions.ExtensionArray):
            return _summarize_sequence(obj)

    type_name = _type_name(obj)
    logger.warning(
        "Memory contribution: replaced unserializable member of type %s with a "
        "type marker (#1583); the rest of the contribution is unaffected.",
        type_name,
    )
    return {"__type__": type_name, "__unserializable__": True}


def dump_json_safe(model: BaseModel) -> Dict[str, Any]:
    """Dump ``model`` to JSON-safe primitives, summarizing what cannot be encoded.

    Identical to ``model.model_dump(mode="json")`` for any model whose members
    are all JSON-safe — ``fallback`` is consulted only for types pydantic's
    serializer does not recognise.

    Not everything is recoverable: a *known* type with a broken encoder (e.g.
    ``pandas.NaT``, a ``datetime`` subclass that raises inside pydantic's
    datetime encoder) never reaches ``fallback`` and still raises. Callers keep
    their honest catch-and-warn for that remainder.
    """
    return cast(Dict[str, Any], model.model_dump(mode="json", fallback=json_safe_fallback))
