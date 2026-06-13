"""Unit tests for scripts/seed_falkordb_init.py (#890).

After #749 the deployed semantic graph is ``e2i_causal``; nothing at HEAD
reads ``e2i_semantic``. The init seeder must no longer probe or seed
``e2i_semantic`` -- even its read-only count probe re-creates the empty
graph shell (FalkorDB creates a graph key on any GRAPH.QUERY), which is
exactly the shell issue #890 cleans up.
"""

from __future__ import annotations

from unittest.mock import patch

import scripts.seed_falkordb_init as mod


class TestSemanticSeedingRetired:
    def test_module_has_no_semantic_seed_step(self) -> None:
        assert not hasattr(mod, "seed_semantic_graph"), (
            "seed_semantic_graph step must be retired: it (re-)creates and "
            "populates the orphan e2i_semantic graph nothing reads (#749, #890)"
        )

    def test_main_seeds_only_causal_graph(self) -> None:
        with (
            patch.object(mod, "seed_causal_graph", return_value=True) as causal,
            patch.object(mod, "FALKORDB_PASSWORD", "pw"),
        ):
            rc = mod.main()
        assert rc == 0
        causal.assert_called_once()

    def test_main_fails_when_causal_seeding_fails(self) -> None:
        with (
            patch.object(mod, "seed_causal_graph", return_value=False),
            patch.object(mod, "FALKORDB_PASSWORD", "pw"),
        ):
            assert mod.main() == 1
