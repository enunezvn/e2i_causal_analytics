"""Task 03.2 — per-brand-scaled CATE map (Kisqali != Remibrutinib) + segment assignment."""
import numpy as np

from src.ml.synthetic.config import DGP_CONFIGS, Brand, DGPType
from src.ml.synthetic.dgp.treatment_arm import (
    SEGMENT_HIGH,
    SEGMENT_LOW,
    SEGMENT_MEDIUM,
    assign_segment,
    brand_scaled_cate,
)


def test_segment_assignment_thresholds():
    sev = np.array([8.0, 5.0, 2.0])
    seg = assign_segment(sev)
    assert list(seg) == [SEGMENT_HIGH, SEGMENT_MEDIUM, SEGMENT_LOW]


def test_base_map_matches_config():
    base = brand_scaled_cate(Brand.REMIBRUTINIB)  # scale 1.0 == base
    assert base == DGP_CONFIGS[DGPType.HETEROGENEOUS].cate_by_segment


def test_each_brand_distinct_but_ordered():
    remi = brand_scaled_cate(Brand.REMIBRUTINIB)
    kisq = brand_scaled_cate(Brand.KISQALI)
    fabh = brand_scaled_cate(Brand.FABHALTA)
    # distinct per brand (gate 6: Kisqali must differ from Remibrutinib)
    assert kisq != remi and fabh != remi and kisq != fabh
    # ordering preserved within every brand
    for m in (remi, kisq, fabh):
        assert m[SEGMENT_HIGH] > m[SEGMENT_MEDIUM] > m[SEGMENT_LOW] > 0
