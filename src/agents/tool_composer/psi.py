"""Population Stability Index, the numeric core of the ``psi_calculator`` tool.

Moved unchanged out of ``tool_registrations.py``, a file the module-size ratchet pins
(#1991 debt 4). The tool keeps its input checks and refusals; this is the arithmetic.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple


def population_stability_index(
    baseline: Any, current: Any, *, bins: int = 10
) -> Tuple[float, List[Dict[str, Any]]]:
    """Population Stability Index between two 1-D numeric arrays.

    Bins by ``baseline`` deciles; ``PSI = sum((c_pct - b_pct) * ln(c_pct/b_pct))``
    with percentages floored at 1e-6 to avoid log(0). Returns ``(psi, buckets)``.
    """
    import numpy as np

    b = np.asarray(baseline, dtype=float)
    c = np.asarray(current, dtype=float)
    edges = np.quantile(b, np.linspace(0, 1, bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    edges = np.unique(edges)
    b_counts = np.histogram(b, bins=edges)[0].astype(float)
    c_counts = np.histogram(c, bins=edges)[0].astype(float)
    b_pct = np.clip(b_counts / b_counts.sum(), 1e-6, None)
    c_pct = np.clip(c_counts / c_counts.sum(), 1e-6, None)
    psi = float(np.sum((c_pct - b_pct) * np.log(c_pct / b_pct)))
    buckets = [
        {
            "range": f"{edges[i]:.4g}-{edges[i + 1]:.4g}",
            "baseline_pct": float(b_pct[i]),
            "current_pct": float(c_pct[i]),
        }
        for i in range(len(b_pct))
    ]
    return psi, buckets
