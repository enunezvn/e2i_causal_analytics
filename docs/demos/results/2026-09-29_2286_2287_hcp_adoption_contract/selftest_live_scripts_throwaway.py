"""Prod-free self-test of the three live proofs: their run() on a throwaway Postgres
(prod's image) + PostgREST (prod's tag), seeded from the committed generators.

Checks each script reaches the verdict it should in every state it can meet live:
  * frame / walk before migration 162 -> NOT-RUNNABLE;
  * registry before 163 -> PASS; after 163 -> PASS (already applied);
  * registry with a drifted label on one row -> FAIL (negative control);
  * frame / walk after 162 -> PASS.
Writes selftest_live_scripts_throwaway.out beside itself.
"""

from __future__ import annotations

import asyncio
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import _live_common as C  # noqa: E402


async def main() -> None:
    out = C.tee_to_out(__file__)
    C.header("self-test of the live proofs on a throwaway Postgres + PostgREST (prod-free)")
    import live_frame_equivalence_2287 as frame
    import live_preparer_walk_2286_2287 as walk
    import live_registry_rows_2286_2287 as registry

    from tests.unit.test_database._hcp_adoption_pg import (
        M162_PATH,
        PostgrestServer,
        SupabaseShim,
        apply_file,
        build_base,
        prod_image,
        seed,
    )
    from tests.unit.test_database.learning_loop._pg import PgConn, ThrowawayPg

    pg = ThrowawayPg(image=prod_image("supabase-db"))
    pg.start()
    server = None
    results = {}
    try:
        pg.rows("postgres", "CREATE DATABASE selftest OWNER postgres")
        conn = PgConn(pg, "selftest")
        build_base(conn)
        # the registry columns proof 1 selects, beyond the harness stub
        conn.execute(
            "ALTER TABLE ml_model_registry ADD COLUMN model_version TEXT, "
            "ADD COLUMN is_champion BOOLEAN DEFAULT false, ADD COLUMN hyperparameters JSONB",
            user="postgres",
        )
        rows = ",".join(
            f"('{C.model_name(b)}', 'v1', 'production', true, false, 'adopted', "
            '\'{"calibration_method": "sigmoid"}\')'
            for b in C.BRANDS
        )
        conn.execute(
            "INSERT INTO ml_model_registry (model_name, model_version, stage, is_champion, "
            f"is_synthetic, cohort_target_outcome, hyperparameters) VALUES {rows}, "
            f"('{C.model_name('Kisqali')}', 'v0', 'archived', false, false, 'adopted', NULL)",
            user="postgres",
        )
        print(f"seeded: {seed(conn)}")
        server = PostgrestServer(image=prod_image("supabase-rest"), pg=pg, db="selftest")
        server.start()
        sync = SupabaseShim(server.client("service_role"))
        asy = SupabaseShim(server.client("service_role", is_async=True))

        print("\n##### frame + walk BEFORE 162")
        results["frame_before_162"] = await frame.run(sync, asy)
        results["walk_before_162"] = await walk.run(sync, asy)
        print("\n##### registry BEFORE 163")
        results["registry_before_163"] = registry.run(sync)

        apply_file(conn, M162_PATH)
        server.reload_schema()
        print("\n##### frame AFTER 162")
        results["frame_after_162"] = await frame.run(sync, asy)
        print("\n##### walk AFTER 162")
        results["walk_after_162"] = await walk.run(sync, asy)

        apply_file(conn, C.MIGRATION_163)
        print("\n##### registry AFTER 163")
        results["registry_after_163"] = registry.run(sync)

        conn.execute(
            "UPDATE ml_model_registry SET cohort_data_source = NULL, "
            "cohort_feature_manifest_source = NULL, cohort_target_outcome = 'will_adopt' "
            f"WHERE model_name = '{C.model_name('Fabhalta')}'",
            user="postgres",
        )
        print("\n##### registry NEGATIVE CONTROL (Fabhalta label drifted)")
        results["registry_negative_control"] = registry.run(sync)
    finally:
        if server is not None:
            server.stop()
        pg.stop()

    expected = {
        "frame_before_162": "NOT-RUNNABLE",
        "walk_before_162": "NOT-RUNNABLE",
        "registry_before_163": "PASS",
        "frame_after_162": "PASS",
        "walk_after_162": "PASS",
        "registry_after_163": "PASS",
        "registry_negative_control": "FAIL",
    }
    print("\nSELF-TEST")
    bad = []
    for key, want in expected.items():
        got = results.get(key, "<missing>")
        ok = got.startswith(want)
        print(f"  {key}: expected {want} -> got {got!r} [{'OK' if ok else 'MISMATCH'}]")
        if not ok:
            bad.append(key)
    print(f"\n(output also written to {out})")
    print(f"VERDICT: {'PASS' if not bad else 'FAIL: ' + ', '.join(bad)}")


if __name__ == "__main__":
    asyncio.run(main())
