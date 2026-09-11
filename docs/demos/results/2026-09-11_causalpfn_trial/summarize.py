"""Summarise #1990 trial jsonl files against spec §11 pass criteria."""
import json, sys, statistics as st
from collections import defaultdict
from pathlib import Path
D = Path(sys.argv[1])
def load(tag):
    p = D / f"{tag}.jsonl"
    return [json.loads(l) for l in p.read_text().splitlines()] if p.exists() else []
def q(v): return f"{v:.3f}"
for tag in sys.argv[2:]:
    rows = load(tag)
    if not rows: print(f"## {tag}: no rows"); continue
    errs = [r for r in rows if "error" in r]
    ok = [r for r in rows if "error" not in r]
    print(f"## {tag}: {len(rows)} frames, {len(errs)} errors")
    for r in errs: print("  ERROR", r["name"], r["error"])
    by = defaultdict(list)
    for r in ok: by[r["brand"]].append(r)
    if any("true_ate" in r for r in ok):
        print("| brand | seeds | max abs err (pfn) | max abs err (dml) | order ok | pfn coverage | dml coverage | median CI width pfn / dml | median t_total s | max t_total s | max ru_maxrss MB |")
        print("|---|---|---|---|---|---|---|---|---|---|---|")
        for b, rs in sorted(by.items()):
            rs = [r for r in rs if "true_ate" in r]
            w_p = st.median([r["pfn_ci_hi"]-r["pfn_ci_lo"] for r in rs]); w_d = st.median([r["dml_ci_hi"]-r["dml_ci_lo"] for r in rs if "dml_ci_hi" in r] or [float("nan")])
            print(f"| {b} | {len(rs)} | {q(max(r['abs_err'] for r in rs))} | {q(max(r.get('dml_abs_err',float('nan')) for r in rs))} | {sum(r['order_ok'] for r in rs)}/{len(rs)} | {sum(r['covered'] for r in rs)}/{len(rs)} | {sum(r.get('dml_covered',False) for r in rs)}/{len(rs)} | {q(w_p)} / {q(w_d)} | {st.median(r['t_total_s'] for r in rs):.0f} | {max(r['t_total_s'] for r in rs):.0f} | {max(r['ru_maxrss_mb'] for r in rs)} |")
        allr = [r for r in ok if "true_ate" in r]
        print(f"ALL: abs_err<0.15 on {sum(r['abs_err']<0.15 for r in allr)}/{len(allr)}; order_ok {sum(r['order_ok'] for r in allr)}/{len(allr)}; pfn covered {sum(r['covered'] for r in allr)}/{len(allr)}; dml covered {sum(r.get('dml_covered',False) for r in allr)}/{len(allr)}; t<10s on {sum(r['t_total_s']<10 for r in allr)}/{len(allr)}; ru_maxrss<1024MB on {sum(r['ru_maxrss_mb']<1024 for r in allr)}/{len(allr)}; temperatures {sorted(set(round(r['temperature'],4) for r in allr))[:5]}")
        # signed bias
        bias = [r["pfn_ate"]-r["true_ate"] for r in allr]; print(f"pfn signed bias: mean {st.mean(bias):+.4f}, sd {st.pstdev(bias):.4f}; dml: mean {st.mean([r['dml_ate']-r['true_ate'] for r in allr if 'dml_ate' in r]):+.4f}")
    for r in ok:
        if r["kind"] == "live":
            print(f"LIVE {r['name']}: pfn ate {q(r['pfn_ate'])} CI [{q(r['pfn_ci_lo'])}, {q(r['pfn_ci_hi'])}] T={r['temperature']:.4g}; dml ate {q(r.get('dml_ate',float('nan')))} CI [{q(r.get('dml_ci_lo',float('nan')))}, {q(r.get('dml_ci_hi',float('nan')))}]; spec §2 reported CI {r.get('reported_ci_spec_s2')}; t_total {r['t_total_s']}s rss {r['ru_maxrss_mb']}MB n_treated {r['n_treated']}")
    print()
