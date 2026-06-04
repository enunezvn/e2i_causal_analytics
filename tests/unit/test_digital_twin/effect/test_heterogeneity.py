import numpy as np
import pandas as pd

from src.digital_twin.effect.heterogeneity import SegmentEffect, segment_by_uplift_quantiles


def test_segments_split_population_by_uplift_quantiles():
    n = 100
    population = pd.DataFrame({"decile": np.arange(n) % 10 + 1})
    uplift = np.linspace(-0.05, 0.25, n)  # monotonic so top != bottom

    segments = segment_by_uplift_quantiles(population, uplift, top_frac=0.2)

    assert all(isinstance(s, SegmentEffect) for s in segments)
    names = {s.name for s in segments}
    assert names == {"top_responders", "bottom_responders"}
    top = next(s for s in segments if s.name == "top_responders")
    bottom = next(s for s in segments if s.name == "bottom_responders")
    assert top.mean_uplift > bottom.mean_uplift
    assert top.size == 20  # 20% of 100
