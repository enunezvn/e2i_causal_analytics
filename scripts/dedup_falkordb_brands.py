"""Dedup + canonicalise Brand nodes in the FalkorDB knowledge graph.

Different writers minted duplicate Brand nodes for the same brand under
different identity schemes and casing:

* the demo seed (``scripts/seed_falkordb.py``) creates a name-keyed proper-case
  ``(:Brand {name:'Remibrutinib', indication, launch_year})`` (no ``id`` prop);
* the cohort-constructor agent (``agents/cohort_constructor/memory_hooks.py``,
  before the canonicalise fix) created an id-keyed lowercase
  ``(:Brand {id:'brand:remibrutinib'})`` — a SECOND node for the same brand.

Because one MERGEs by ``name`` and the other by ``id``, they never converge.
This migration collapses, per canonical brand, every case-insensitive variant
node onto ONE canonical node ``(:Brand {id:'brand:<CanonicalName>',
name:'<CanonicalName>'})`` — re-pointing every relationship and filling any
missing node properties from the variants — then normalises the ``brand``
property on ``CAUSES`` edges to canonical casing.

Idempotent: re-running after convergence is a no-op. Dry-run by DEFAULT; pass
``--apply`` to write. Setting the canonical ``id`` on the surviving node means
the (now casing-fixed) agent writes MERGE onto it instead of forking again.

Env (read from the process; e.g. ``FALKORDB_HOST=127.0.0.1 FALKORDB_PORT=6381
FALKORDB_PASSWORD=changeme``): FALKORDB_HOST/PORT/PASSWORD/GRAPH_NAME.

NOTE: ``scripts/seed_falkordb.py`` still creates name-keyed Brand nodes, so a
re-seed re-introduces a variant; re-run this migration afterwards (it is safe to
re-run). Giving the seed canonical ids is a deliberately-separate follow-up.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from typing import Any, Dict, List, Tuple

# Canonical, proper-case brand names (the single source of truth).
from src.memory.episodic_memory import E2IBrand, canonical_brand

_REL_TYPE_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
# Node props that identify/seed the canonical node — never copied FROM a variant
# over the primary (the primary owns its identity + freshness stamp).
_IDENTITY_KEYS = {"id", "name", "updated_at"}


def _canonical_brands() -> List[str]:
    return [b.value for b in E2IBrand if b.value.lower() != "all"]


def _rows(result: Any) -> List[List[Any]]:
    return list(getattr(result, "result_set", []) or [])


def _find_brand_variants(g: Any, canonical_name: str) -> List[Dict[str, Any]]:
    """All Brand nodes that are (case-insensitively) this brand: by name, or by an
    id of the form ``brand:<name>``. Returns internal id, degree, and props."""
    res = g.query(
        """
        MATCH (b:Brand)
        WHERE toLower(b.name) = toLower($cn)
           OR toLower(coalesce(b.id, '')) = toLower($cid)
        OPTIONAL MATCH (b)-[r]-()
        RETURN id(b) AS nid, properties(b) AS props, count(r) AS degree
        """,
        {"cn": canonical_name, "cid": f"brand:{canonical_name}"},
    )
    out = []
    for nid, props, degree in _rows(res):
        out.append({"nid": nid, "props": dict(props or {}), "degree": int(degree)})
    return out


def _pick_primary(variants: List[Dict[str, Any]], canonical_id: str) -> Dict[str, Any]:
    """Choose the node to KEEP: an already-canonical id wins, else the richest /
    most-connected (the seed node, which carries indication/launch_year)."""

    def score(v: Dict[str, Any]) -> Tuple[int, int, int]:
        is_canon = 1 if v["props"].get("id") == canonical_id else 0
        has_indication = 1 if "indication" in v["props"] else 0
        return (is_canon, has_indication, v["degree"])

    return max(variants, key=score)


def _edges_of(g: Any, nid: int) -> Dict[str, List[Tuple[str, int, Dict[str, Any]]]]:
    out_res = g.query(
        "MATCH (v)-[r]->(x) WHERE id(v) = $nid RETURN type(r), id(x), properties(r)",
        {"nid": nid},
    )
    in_res = g.query(
        "MATCH (x)-[r]->(v) WHERE id(v) = $nid RETURN type(r), id(x), properties(r)",
        {"nid": nid},
    )

    def parse(rows: List[List[Any]]) -> List[Tuple[str, int, Dict[str, Any]]]:
        return [(t, int(xid), dict(p or {})) for t, xid, p in rows]

    return {"out": parse(_rows(out_res)), "in": parse(_rows(in_res))}


def _repoint_and_delete(
    g: Any, primary_nid: int, variant: Dict[str, Any], apply: bool
) -> Tuple[int, int]:
    """Move every edge of ``variant`` onto the primary, fill missing primary props
    from the variant, then DETACH DELETE the variant. Returns (edges_moved, deleted)."""
    vid = variant["nid"]
    edges = _edges_of(g, vid)
    moved = 0
    for direction, items in edges.items():
        for rel_type, other_nid, props in items:
            if not _REL_TYPE_RE.match(rel_type):
                print(f"    !! skipping edge with non-identifier type {rel_type!r}")
                continue
            if other_nid == vid:
                continue  # self-loop on the variant — drop with the node
            moved += 1
            if not apply:
                continue
            if direction == "out":
                pattern = f"MERGE (p)-[r2:{rel_type}]->(x)"
            else:
                pattern = f"MERGE (x)-[r2:{rel_type}]->(p)"
            g.query(
                f"MATCH (p), (x) WHERE id(p) = $pid AND id(x) = $xid {pattern} SET r2 += $props",
                {"pid": primary_nid, "xid": other_nid, "props": props},
            )

    # Fill any node properties the primary is missing (e.g. indication from seed).
    fill = {k: v for k, v in variant["props"].items() if k not in _IDENTITY_KEYS}
    if apply and fill:
        # COALESCE per key so the primary's existing values always win.
        sets = ", ".join(f"p.{k} = coalesce(p.{k}, ${k})" for k in fill)
        g.query(
            f"MATCH (p) WHERE id(p) = $pid SET {sets}",
            {"pid": primary_nid, **fill},
        )

    if apply:
        g.query("MATCH (v) WHERE id(v) = $vid DETACH DELETE v", {"vid": vid})
    return moved, 1


def _normalise_causes_brands(g: Any, apply: bool) -> Dict[str, int]:
    """Rewrite the ``brand`` property of CAUSES edges to canonical casing; report
    edges whose brand is null/empty (left as-is — they belong to no brand)."""
    res = g.query("MATCH ()-[r:CAUSES]->() RETURN r.brand AS brand, count(r) AS c")
    fixed = 0
    null_brand = 0
    by_value: Dict[str, int] = {}
    for brand, c in _rows(res):
        c = int(c)
        if brand is None or str(brand).strip() == "":
            null_brand += c
            continue
        canon = canonical_brand(str(brand))
        by_value[str(brand)] = c
        if canon != brand:
            fixed += c
            if apply:
                g.query(
                    "MATCH ()-[r:CAUSES]->() WHERE r.brand = $old SET r.brand = $new",
                    {"old": brand, "new": canon},
                )
    return {"fixed": fixed, "null_brand": null_brand, "distinct": len(by_value)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--apply", action="store_true", help="Write to FalkorDB (default: dry-run)")
    args = ap.parse_args()

    from falkordb import FalkorDB  # lazy: dry-run still needs the live graph to plan

    host = os.environ.get("FALKORDB_HOST", "localhost")
    port = int(os.environ.get("FALKORDB_PORT", "6379"))
    password = os.environ.get("FALKORDB_PASSWORD") or None
    graph_name = os.environ.get("FALKORDB_GRAPH_NAME", "e2i_causal")
    g = FalkorDB(host=host, port=port, password=password).select_graph(graph_name)

    mode = "APPLY" if args.apply else "DRY-RUN"
    print(f"=== Brand dedup [{mode}] on '{graph_name}' @ {host}:{port} ===\n")

    total_merged = 0
    total_edges = 0
    for cn in _canonical_brands():
        canonical_id = f"brand:{cn}"
        variants = _find_brand_variants(g, cn)
        if not variants:
            print(f"{cn}: no Brand node found — skipped")
            continue
        primary = _pick_primary(variants, canonical_id)
        others = [v for v in variants if v["nid"] != primary["nid"]]
        print(
            f"{cn}: {len(variants)} node(s) "
            f"(primary id={primary['props'].get('id')!r} name={primary['props'].get('name')!r} "
            f"deg={primary['degree']}); folding {len(others)} variant(s)"
        )
        # Stamp the canonical identity on the surviving node.
        if args.apply:
            g.query(
                "MATCH (p) WHERE id(p) = $pid SET p.id = $cid, p.name = $cn",
                {"pid": primary["nid"], "cid": canonical_id, "cn": cn},
            )
        for v in others:
            moved, deleted = _repoint_and_delete(g, primary["nid"], v, args.apply)
            total_merged += deleted
            total_edges += moved
            print(
                f"    folded id={v['props'].get('id')!r} name={v['props'].get('name')!r} "
                f"deg={v['degree']} -> moved {moved} edge(s)"
            )

    causes = _normalise_causes_brands(g, args.apply)
    print(
        f"\nCAUSES edges: {causes['fixed']} re-cased to canonical, "
        f"{causes['null_brand']} with null/empty brand (left as-is, belong to no brand)"
    )
    print(f"Brand nodes folded: {total_merged} (edges moved: {total_edges})")
    if not args.apply:
        print("\nDRY-RUN — no writes. Re-run with --apply to execute.")
    else:
        print("\nAPPLIED.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
