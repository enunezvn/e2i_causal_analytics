"""Summarise refit / placebo / seed CSVs into summary.txt."""
import numpy as np, pandas as pd
pd.set_option("display.width", 220)
r = pd.read_csv("refit_augmented_w.csv")
print("== null channel (rep_training_score), DR ATE and CI by W spec")
print(r[r.channel == "rep_training_score"][["brand", "spec", "ate", "lo", "hi", "se", "excl0"]].round(4).to_string(index=False))
print("\n== planted channels: error (ate - planted) by W spec")
p = r[r.channel != "rep_training_score"].pivot_table(index=["brand", "channel"], columns="spec", values="err").round(4)
print(p[["default", "+other7_tbins", "+dgp_drivers", "+both"]].to_string())
print("\n  mean |err| by spec:", r[r.channel != "rep_training_score"].groupby("spec")["err"].apply(lambda e: round(e.abs().mean(), 4)).to_dict())
print("  mean err by spec:", r[r.channel != "rep_training_score"].groupby("spec")["err"].mean().round(4).to_dict())
print("  coverage of planted by spec:", r[r.channel != "rep_training_score"].groupby("spec")["covers_planted"].mean().round(3).to_dict())
print("  mean SE by spec:", r.groupby("spec")["se"].mean().round(4).to_dict())
pl = pd.read_csv("placebo_within_region.csv")
print("\n== placebo (rep_training permuted within region)")
for b, g in pl.groupby("brand"):
    print(f"  {b:13s} K={len(g)} excl0={int(g.excl0.sum())} ({g.excl0.mean():.3f}) mean ate {g.ate.mean():+.4f} "
          f"empSD {g.ate.std():.4f} mean SE {g.se.mean():.4f} SE/empSD {g.se.mean()/g.ate.std():.2f}")
s = pd.read_csv("seed_null_remibrutinib.csv")
live = s[s.seed == 427].iloc[0]; o = s[s.seed != 427]
print("\n== Remibrutinib null under fresh derive seeds (design fixed; arm+noise+coin flips re-drawn)")
print(f"  seed427 ate {live.ate:+.4f} (CI {live.lo:+.4f},{live.hi:+.4f})")
print(f"  other seeds n={len(o)} mean {o.ate.mean():+.4f} (SE of mean {o.ate.std()/np.sqrt(len(o)):.4f}) empSD {o.ate.std():.4f} "
      f"mean SE {o.se.mean():.4f} excl0 {int(o.excl0.sum())}/{len(o)}")
print(f"  seeds with ate >= seed427: {int((o.ate >= live.ate).sum())}/{len(o)}; |ate| >= : {int((o.ate.abs() >= abs(live.ate)).sum())}/{len(o)}; "
      f"z vs seed distribution {(live.ate - o.ate.mean())/o.ate.std():.2f}")
