"""FeatureStoreSeeder — deterministic (uuid5) group/feature ids.

The seeder's catalog is a fixed constant set (4 groups / 15 features), so its ids
must be a pure function of the natural key. Random (uuid4) minting broke
frontier-append (2026-07-15, feature_values 0/340 FK 23503): each weekly cohort
seeds its own FeatureStoreSeeder, build_frontier_datasets keeps only the first
cohort's features frame, and the loader's #852 reconcile can only remap feature_ids
it finds in that frame — every other cohort's feature_values carried orphaned ids.

Deterministic ids make every mint identical, so multi-cohort merges are FK-coherent
by construction. The #852 reconcile stays: DBs seeded before this change hold
legacy random canonical ids the generated ids must still be remapped onto.
"""

import uuid

import pandas as pd

from src.ml.synthetic.generators import FeatureStoreSeeder, GeneratorConfig


def test_ids_identical_across_seeder_instances():
    """Two independent seeders (as frontier_append creates per weekly cohort)
    must mint byte-identical frames, ids included."""
    groups_a, features_a = FeatureStoreSeeder(GeneratorConfig(seed=1)).seed()
    groups_b, features_b = FeatureStoreSeeder(GeneratorConfig(seed=2)).seed()
    pd.testing.assert_frame_equal(groups_a, groups_b)
    pd.testing.assert_frame_equal(features_a, features_b)


def test_ids_are_valid_uuids_and_unique():
    groups, features = FeatureStoreSeeder(GeneratorConfig()).seed()
    all_ids = list(groups["id"]) + list(features["id"])
    for value in all_ids:
        uuid.UUID(value)  # raises on malformed ids
    assert len(set(all_ids)) == len(all_ids)


def test_features_reference_their_groups():
    groups, features = FeatureStoreSeeder(GeneratorConfig()).seed()
    assert set(features["feature_group_id"]) <= set(groups["id"])


def test_get_feature_ids_matches_seeded_frame():
    """get_feature_ids() re-seeds internally; with deterministic ids the mapping
    must agree with a frame seeded by a different instance."""
    seeder = FeatureStoreSeeder(GeneratorConfig())
    _, features = FeatureStoreSeeder(GeneratorConfig()).seed()
    id_map = seeder.get_feature_ids()
    expected = dict(zip(features["name"], features["id"], strict=False))
    assert id_map == expected
