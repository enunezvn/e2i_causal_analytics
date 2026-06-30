"""Issue #852 — feature-store tables fail to load (feature_groups/features/
feature_values: 0 loaded).

ROOT CAUSE (proven against the live docker supabase-db): the generic
``BatchLoader._load_batch`` calls ``.upsert(records)`` with NO ``on_conflict``,
so PostgREST defaults the conflict target to the PRIMARY KEY (``id``). The
``FeatureStoreSeeder`` mints fresh random UUIDs every run, and the four canonical
group names (``hcp_demographics`` / ``patient_features`` / ``brand_performance`` /
``causal_features``) already exist in the DB (registered 2026-01-26) with different
ids. The upsert-by-id therefore INSERTs and collides with the *separate*
``feature_groups_name_key`` UNIQUE constraint on ``name`` -> 23505 -> whole batch
fails -> 0 loaded -> ``features``/``feature_values`` orphaned -> 0/0.

The DB schema is correct (the UNIQUE-on-natural-key constraints are intentional,
mirrored by the production ``src/feature_store/client.py``). This is CODE drift
(#825/#842 precedent): the loader must upsert on the NATURAL KEY, and the FK chain
must reconcile to the existing parent ids.

These unit tests lock the loader contract WITHOUT a DB (the faithful >0 proof lives
in tests/integration/test_issue_852_feature_store_load_realdb.py).
"""

import json

import pandas as pd

from src.ml.synthetic.loaders import BatchLoader, LoaderConfig
from src.ml.synthetic.loaders.batch_loader import TABLE_ON_CONFLICT


class TestFeatureStoreOnConflictMap:
    """The loader must declare natural-key conflict targets for the 3 feature-store
    tables so the upsert is idempotent against the existing rows instead of colliding
    on the secondary UNIQUE(name)/UNIQUE(feature_group_id,name) constraints."""

    def test_feature_groups_conflicts_on_name(self):
        assert TABLE_ON_CONFLICT.get("feature_groups") == "name"

    def test_features_conflicts_on_group_and_name(self):
        assert TABLE_ON_CONFLICT.get("features") == "feature_group_id,name"

    def test_feature_values_conflicts_on_entity_timestamp(self):
        # Mirrors feature_entity_timestamp_unique (feature_id, entity_values, event_timestamp)
        assert TABLE_ON_CONFLICT.get("feature_values") == "feature_id,entity_values,event_timestamp"


class _RecordingClient:
    """Minimal fake supabase client that records the on_conflict passed to upsert and
    raises a 23505 when a natural-key-collision row is upserted with the WRONG conflict
    target (i.e. conflict-on-id), reproducing the live 0-load failure."""

    def __init__(self, existing_names):
        self._existing_names = set(existing_names)
        self.upsert_calls = []  # list of (table, on_conflict, n_records)

    def table(self, name):
        return _RecordingTable(self, name)


class _RecordingTable:
    def __init__(self, client, name):
        self._client = client
        self._name = name
        self._records = None
        self._on_conflict = None

    def upsert(self, records, on_conflict=None):
        self._records = records
        self._on_conflict = on_conflict
        return self

    def execute(self):
        self._client.upsert_calls.append((self._name, self._on_conflict, len(self._records)))
        # Reproduce the live bug: feature_groups rows whose name already exists in the
        # DB collide on the UNIQUE(name) constraint UNLESS the upsert conflicts on name.
        if self._name == "feature_groups" and self._on_conflict != "name":
            for rec in self._records:
                if rec.get("name") in self._client._existing_names:
                    raise RuntimeError(
                        "duplicate key value violates unique constraint "
                        '"feature_groups_name_key" (code 23505)'
                    )

        class _Resp:
            data = []

        return _Resp()


class TestLoaderPassesOnConflict:
    """The loader must thread TABLE_ON_CONFLICT through to .upsert(on_conflict=...).
    Without it, a feature_groups frame whose canonical names already exist 0-loads."""

    def _loader_with_client(self, existing_names):
        loader = BatchLoader(LoaderConfig(batch_size=100, dry_run=False, max_retries=1))
        loader._client = _RecordingClient(existing_names)
        return loader

    def test_feature_groups_load_succeeds_with_preexisting_names(self):
        """RED before fix: conflict defaults to id -> 23505 -> records_failed==4.
        GREEN after fix: on_conflict='name' -> upsert reconciles -> records_loaded==4."""
        df = pd.DataFrame(
            {
                "id": ["new-uuid-1", "new-uuid-2", "new-uuid-3", "new-uuid-4"],
                "name": [
                    "hcp_demographics",
                    "patient_features",
                    "brand_performance",
                    "causal_features",
                ],
                "is_synthetic": [True, True, True, True],
            }
        )
        loader = self._loader_with_client(
            existing_names={
                "hcp_demographics",
                "patient_features",
                "brand_performance",
                "causal_features",
            }
        )
        result = loader.load_table("feature_groups", df)
        assert result.records_loaded == 4, f"expected idempotent load, got {result.errors}"
        assert result.records_failed == 0
        # And the on_conflict that reached the client was the natural key.
        assert loader._client.upsert_calls[0][1] == "name"

    def test_features_upsert_uses_group_name_conflict(self):
        df = pd.DataFrame(
            {
                "id": ["f1"],
                "feature_group_id": ["g1"],
                "name": ["specialty_encoded"],
                "is_synthetic": [True],
            }
        )
        loader = self._loader_with_client(existing_names=set())
        loader.load_table("features", df)
        assert loader._client.upsert_calls[0][1] == "feature_group_id,name"

    def test_feature_values_upsert_uses_entity_timestamp_conflict(self):
        df = pd.DataFrame(
            {
                "id": ["v1"],
                "feature_id": ["f1"],
                "entity_values": [{"hcp_id": "hcp_1"}],
                "value": [{"value": 1}],
                "event_timestamp": ["2026-01-01T00:00:00"],
                "freshness_status": ["fresh"],
                "is_synthetic": [True],
            }
        )
        loader = self._loader_with_client(existing_names=set())
        loader.load_table("feature_values", df)
        assert loader._client.upsert_calls[0][1] == "feature_id,entity_values,event_timestamp"

    def test_non_feature_store_table_keeps_default_conflict(self):
        """Other tables must NOT get a spurious on_conflict (regression guard)."""
        df = pd.DataFrame({"hcp_id": ["hcp_1"], "is_synthetic": [True]})
        loader = self._loader_with_client(existing_names=set())
        loader.load_table("hcp_profiles", df)
        assert loader._client.upsert_calls[0][1] is None


class _ReadClient:
    """Fake client whose .select().execute() returns canned feature-store rows so
    reconcile_feature_store_ids can be unit-tested without a DB. ``raise_on`` names a
    table whose read should raise, to exercise the fail-closed path."""

    def __init__(self, groups_rows, features_rows, raise_on=None):
        self._rows = {"feature_groups": groups_rows, "features": features_rows}
        self._raise_on = raise_on

    def table(self, name):
        return _ReadTable(self._rows.get(name, []), raise_=(name == self._raise_on))


class _ReadTable:
    def __init__(self, rows, raise_=False):
        self._rows = rows
        self._raise = raise_

    def select(self, *_cols):
        return self

    def execute(self):
        if self._raise:
            raise RuntimeError("simulated read failure")

        class _Resp:
            pass

        r = _Resp()
        r.data = self._rows
        return r


class TestReconcileFeatureStoreIds:
    """The reconcile pass must remap generated ids onto existing DB ids by natural key,
    cascading group->feature->value, so the upsert never rewrites a referenced PK."""

    def _loader(self, groups_rows, features_rows, raise_on=None):
        loader = BatchLoader(LoaderConfig(dry_run=False, max_retries=1))
        loader._client = _ReadClient(groups_rows, features_rows, raise_on=raise_on)
        return loader

    def test_group_id_remapped_to_existing(self):
        datasets = {
            "feature_groups": pd.DataFrame(
                {"id": ["fresh-g"], "name": ["brand_performance"], "is_synthetic": [True]}
            )
        }
        loader = self._loader(
            groups_rows=[{"id": "real-g", "name": "brand_performance"}],
            features_rows=[],
        )
        loader.reconcile_feature_store_ids(datasets)
        assert datasets["feature_groups"].iloc[0]["id"] == "real-g"

    def test_feature_fk_and_id_cascade(self):
        datasets = {
            "feature_groups": pd.DataFrame(
                {"id": ["fresh-g"], "name": ["brand_performance"], "is_synthetic": [True]}
            ),
            "features": pd.DataFrame(
                {
                    "id": ["fresh-f"],
                    "feature_group_id": ["fresh-g"],
                    "name": ["trx_30d"],
                    "is_synthetic": [True],
                }
            ),
            "feature_values": pd.DataFrame(
                {
                    "id": ["v1"],
                    "feature_id": ["fresh-f"],
                    "entity_values": [{"brand": "Kisqali"}],
                    "value": [{"value": 1}],
                    "event_timestamp": ["2026-01-01T00:00:00"],
                    "freshness_status": ["fresh"],
                    "is_synthetic": [True],
                }
            ),
        }
        loader = self._loader(
            groups_rows=[{"id": "real-g", "name": "brand_performance"}],
            features_rows=[{"id": "real-f", "feature_group_id": "real-g", "name": "trx_30d"}],
        )
        loader.reconcile_feature_store_ids(datasets)
        # group reconciled
        assert datasets["feature_groups"].iloc[0]["id"] == "real-g"
        # feature FK repointed to reconciled group, id reconciled to existing
        assert datasets["features"].iloc[0]["feature_group_id"] == "real-g"
        assert datasets["features"].iloc[0]["id"] == "real-f"
        # value FK repointed to reconciled feature id
        assert datasets["feature_values"].iloc[0]["feature_id"] == "real-f"

    def test_new_rows_keep_fresh_ids(self):
        """A group/feature not present in the DB keeps its generated id (no spurious remap)."""
        datasets = {
            "feature_groups": pd.DataFrame(
                {"id": ["fresh-g"], "name": ["brand_new_group"], "is_synthetic": [True]}
            )
        }
        loader = self._loader(groups_rows=[], features_rows=[])
        loader.reconcile_feature_store_ids(datasets)
        assert datasets["feature_groups"].iloc[0]["id"] == "fresh-g"

    def test_dry_run_is_noop(self):
        datasets = {
            "feature_groups": pd.DataFrame(
                {"id": ["fresh-g"], "name": ["brand_performance"], "is_synthetic": [True]}
            )
        }
        loader = BatchLoader(LoaderConfig(dry_run=True))
        # No client; must not raise and must not mutate.
        loader.reconcile_feature_store_ids(datasets)
        assert datasets["feature_groups"].iloc[0]["id"] == "fresh-g"

    def test_features_read_failure_is_fail_closed(self):
        """If the features lookup fails AFTER the group remap, reconcile must RAISE
        (not silently fall back to fresh-feature-id-on-natural-key, which 23503/23505
        -> 0 loaded). Fail loud, do not fabricate a partial FK-incoherent load."""
        import pytest

        datasets = {
            "feature_groups": pd.DataFrame(
                {"id": ["fresh-g"], "name": ["brand_performance"], "is_synthetic": [True]}
            ),
            "features": pd.DataFrame(
                {
                    "id": ["fresh-f"],
                    "feature_group_id": ["fresh-g"],
                    "name": ["trx_30d"],
                    "is_synthetic": [True],
                }
            ),
        }
        loader = self._loader(
            groups_rows=[{"id": "real-g", "name": "brand_performance"}],
            features_rows=[],
            raise_on="features",
        )
        with pytest.raises(RuntimeError, match="simulated read failure"):
            loader.reconcile_feature_store_ids(datasets)


class _DupDetectingClient:
    """Fake client that reproduces PostgREST's 21000 (cardinality_violation): an
    ``INSERT ... ON CONFLICT (...) DO UPDATE`` raises when a single batch contains two
    rows with the SAME conflict-target key ("cannot affect row a second time"). This is
    the live feature_values failure mode (low-cardinality entity_values collide on the
    (feature_id, entity_values, event_timestamp) natural key within a batch)."""

    def __init__(self):
        self.upsert_calls = []  # list of (table, on_conflict, n_records)

    def table(self, name):
        return _DupDetectingTable(self, name)


class _DupDetectingTable:
    def __init__(self, client, name):
        self._client = client
        self._name = name
        self._records = None
        self._on_conflict = None

    def upsert(self, records, on_conflict=None):
        self._records = records
        self._on_conflict = on_conflict
        return self

    def execute(self):
        if self._on_conflict:
            keys = self._on_conflict.split(",")
            seen = set()
            for rec in self._records:
                k = tuple(json.dumps(rec.get(c), sort_keys=True, default=str) for c in keys)
                if k in seen:
                    raise RuntimeError(
                        "ON CONFLICT DO UPDATE command cannot affect row a second time (code 21000)"
                    )
                seen.add(k)
        self._client.upsert_calls.append((self._name, self._on_conflict, len(self._records)))

        class _Resp:
            data = []

        return _Resp()


class TestIntraBatchConflictKeyDedup:
    """For natural-key-conflict tables, the loader must drop intra-frame duplicate
    conflict keys BEFORE batching, or PostgREST raises 21000 and the whole batch (and
    its ~500 good rows) fails. Mirrors the live feature_values ~9% loss."""

    def _loader(self):
        loader = BatchLoader(LoaderConfig(batch_size=100, dry_run=False, max_retries=1))
        loader._client = _DupDetectingClient()
        return loader

    def test_feature_values_duplicate_natural_keys_are_deduped(self):
        """v1 and v2 share (feature_id, entity_values, event_timestamp). Without dedup
        the upsert raises 21000 -> batch fails. With dedup -> last wins, 2 rows load."""
        df = pd.DataFrame(
            {
                "id": ["v1", "v2", "v3"],
                "feature_id": ["f1", "f1", "f1"],
                "entity_values": [
                    {"brand": "Kisqali"},
                    {"brand": "Kisqali"},
                    {"brand": "Fabhalta"},
                ],
                "value": [{"value": 1}, {"value": 2}, {"value": 3}],
                "event_timestamp": [
                    "2026-01-01T00:00:00",
                    "2026-01-01T00:00:00",
                    "2026-01-01T00:00:00",
                ],
                "freshness_status": ["fresh", "fresh", "fresh"],
                "is_synthetic": [True, True, True],
            }
        )
        loader = self._loader()
        result = loader.load_table("feature_values", df)
        assert result.records_failed == 0, f"dup natural keys must be deduped, got {result.errors}"
        assert result.records_loaded == 2  # one of the two colliding rows dropped
        assert loader._client.upsert_calls[0][2] == 2  # the batch the client saw was deduped

    def test_jsonb_key_order_does_not_defeat_dedup(self):
        """entity_values is jsonb (order-independent equality in PG). Two dicts with the
        same content but different key order must be treated as ONE conflict key."""
        df = pd.DataFrame(
            {
                "id": ["v1", "v2"],
                "feature_id": ["f1", "f1"],
                "entity_values": [
                    {"brand": "Kisqali", "region": "west"},
                    {"region": "west", "brand": "Kisqali"},
                ],
                "value": [{"value": 1}, {"value": 2}],
                "event_timestamp": ["2026-01-01T00:00:00", "2026-01-01T00:00:00"],
                "freshness_status": ["fresh", "fresh"],
                "is_synthetic": [True, True],
            }
        )
        loader = self._loader()
        result = loader.load_table("feature_values", df)
        assert result.records_failed == 0
        assert result.records_loaded == 1

    def test_non_conflict_table_is_not_deduped(self):
        """Tables with no on_conflict target must NOT be de-duplicated (no behavior
        change for the bulk fact tables)."""
        df = pd.DataFrame(
            {
                "hcp_id": ["hcp_1", "hcp_1"],  # same id twice; PK upsert handles it
                "is_synthetic": [True, True],
            }
        )
        loader = self._loader()
        result = loader.load_table("hcp_profiles", df)
        assert result.records_loaded == 2  # both rows kept (no dedup)
        assert loader._client.upsert_calls[0][2] == 2
