"""Cheap disproof for codex r2 MED 1 ("the full candidate set can fail while a
subset is admissible"). Enumerates every union of 2–3 fragments drawn from a
vocabulary of fragment shapes over {feature, T, Y, U_1, U_2} (latents shared
by name), keeps only fragments the extractor classifies (as the author path
does), takes the assembler's candidates (roles confounder / ancestor), and
reports every union where the full candidate set fails the backdoor
criterion while some proper subset passes. Run from the worktree root."""

from __future__ import annotations

import itertools
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
import networkx as nx  # noqa: E402

import src  # noqa: E402
from src.ml.causal_role_dgp.backdoor import satisfies_backdoor_criterion  # noqa: E402
from src.ml.causal_role_dgp.extractor import extract_role  # noqa: E402

assert ".worktrees/lane-b-structural-author-scaffold" in src.__file__

LATENTS = ("U_1", "U_2")


# Every directed edge a fragment may contain (guide §1): feature <-> T/Y/U, U -> T/Y, U -> U.
def _vocab(f):
    others = ["T", "Y", *LATENTS]
    edges = []
    for o in others:
        edges.append((f, o))
        edges.append((o, f))
    for u in LATENTS:
        edges += [(u, "T"), (u, "Y")]
    edges.append(("U_1", "U_2"))
    edges.append(("U_2", "U_1"))
    return edges


def _fragments(f, max_edges=4):
    vocab = _vocab(f)
    out = []
    for k in range(1, max_edges + 1):
        for combo in itertools.combinations(vocab, k):
            edges = list(combo) + [("T", "Y")]
            g = nx.DiGraph(edges)
            if not nx.is_directed_acyclic_graph(g) or f not in g:
                continue
            try:
                role = extract_role(f, "T", "Y", g)
            except ValueError:
                continue
            if role in ("confounder", "ancestor"):
                out.append((tuple(edges), role))
    return out


def main() -> int:
    feats = ["A", "B", "C"]
    found = 0
    unions = 0
    # 2 features with fragments of up to 3 authored edges; 3 features with up
    # to 2 (the product is otherwise ~10^7 unions).
    for n, max_edges in ((2, 3), (3, 2)):
        frags = {f: _fragments(f, max_edges=max_edges) for f in feats[:n]}
        print(
            f"n={n} max_edges={max_edges} classified fragments per feature:",
            {f: len(v) for f, v in frags.items()},
            flush=True,
        )
        for chosen in itertools.product(*[frags[f] for f in feats[:n]]):
            unions += 1
            g = nx.DiGraph()
            for edges, _ in chosen:
                g.add_edges_from(edges)
            if not nx.is_directed_acyclic_graph(g):
                continue
            cands = []
            for f, (_, _) in zip(feats[:n], chosen, strict=True):
                try:
                    r = extract_role(f, "T", "Y", g)
                except ValueError:
                    continue
                if r in ("confounder", "ancestor"):
                    cands.append(f)
            if not cands:
                continue
            if satisfies_backdoor_criterion(g, cands, "T", "Y"):
                continue
            for k in range(0, len(cands)):
                for sub in itertools.combinations(cands, k):
                    if satisfies_backdoor_criterion(g, list(sub), "T", "Y"):
                        found += 1
                        if found <= 5:
                            print(
                                "NON-MONOTONE:",
                                sorted(g.edges()),
                                "full",
                                cands,
                                "subset",
                                list(sub),
                            )
                        break
                else:
                    continue
                break
    print(f"unions checked: {unions}; full-fails-but-subset-passes cases: {found}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
