"""Uplift-quantile segmentation -> top/bottom responding segments (REFINE input).

v1 segments the scored twin population by uplift quantile. A richer
covariate-conditioned CATE drill-down (causal_engine/hierarchical/segment_cate)
is a v2 enhancement.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class SegmentEffect:
    name: str
    size: int
    mean_uplift: float
    profile: dict  # mean of each covariate in the segment


def segment_by_uplift_quantiles(
    population: pd.DataFrame, uplift: np.ndarray, top_frac: float = 0.2
) -> list[SegmentEffect]:
    scores = np.asarray(uplift, dtype=float).ravel()
    n = scores.shape[0]
    if n == 0:
        return []
    k = max(1, int(round(top_frac * n)))
    order = np.argsort(scores)
    bottom_idx, top_idx = order[:k], order[-k:]

    def _segment(name: str, idx: np.ndarray) -> SegmentEffect:
        return SegmentEffect(
            name=name,
            size=int(idx.shape[0]),
            mean_uplift=float(np.mean(scores[idx])),
            profile={c: float(population.iloc[idx][c].mean()) for c in population.columns},
        )

    return [_segment("top_responders", top_idx), _segment("bottom_responders", bottom_idx)]
