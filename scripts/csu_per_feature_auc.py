"""Per-feature single-variable AUC sweep across lookback regimes.

Spec from `02_csu_under_masking_rca.md` §B.2 (with codex I1 fix):

For each (regime, feature) pair, fit a 1-feature `LogisticRegression()` on
a 90/10 train/hold random split (seed=42) and compute
`roc_auc_score(y_test, y_pred_proba)`. Use `max(auc, 1-auc)` to handle
inverted single-feature predictors. The regime axis is the unmasked
baseline (OFF) plus four masked variants at lookback windows
{30, 90, 180, 365}. The verdict per feature follows §B.2-D1:

- TARGET-EQUIVALENT: min(AUC across all 4 windows) >= 0.85
- LOOKBACK-FIXABLE: AUC monotonically decreases as window shrinks
  AND AUC at the 30d window <= 0.55
- PARTIAL-COLLAPSE: otherwise

Output: a markdown table at `--out` with rows = features, columns = regimes.

Inputs are directories produced by `scripts/convert_csu_rwd.py`; we read
each one's `e2i_ml_v3_patient_journeys.json` and coerce to DataFrame.

This script is single-purpose and does not modify any existing repo state.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split


REGIME_ORDER = ["OFF", "ON_30", "ON_90", "ON_180", "ON_365"]


def _load_journeys(directory: Path) -> pd.DataFrame:
    """Load `e2i_ml_v3_patient_journeys.json` from a converted-output dir."""
    path = directory / "e2i_ml_v3_patient_journeys.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing patient_journeys at {path}")
    with open(path, encoding="utf-8") as f:
        records: list[dict[str, Any]] = json.load(f)
    return pd.DataFrame.from_records(records)


def _single_feature_auc(
    df: pd.DataFrame, feature: str, target: str
) -> tuple[float | None, int, int]:
    """Fit 1-feature LogisticRegression on 90/10 split, return AUC.

    Returns ``(auc, n_class_0, n_class_1)``. If skip-conditions fire
    (target has <2 classes, feature has <2 unique values, fewer than 30
    rows, target value_counts min < 30, or train/test target degenerate
    after split), returns ``(None, n_c0, n_c1)``.
    """
    if feature not in df.columns or target not in df.columns:
        return None, 0, 0

    sub = df[[feature, target]].dropna().copy()
    sub[target] = pd.to_numeric(sub[target], errors="coerce")
    sub[feature] = pd.to_numeric(sub[feature], errors="coerce")
    sub = sub.dropna()

    n_c0 = int((sub[target] == 0).sum())
    n_c1 = int((sub[target] == 1).sum())

    if sub[target].nunique() < 2:
        return None, n_c0, n_c1
    if sub[feature].nunique() < 2:
        return None, n_c0, n_c1
    if len(sub) < 30:
        return None, n_c0, n_c1
    vc = sub[target].value_counts()
    if int(vc.min()) < 30:
        return None, n_c0, n_c1

    X = sub[[feature]].values
    y = sub[target].astype(int).values

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.10, random_state=42, stratify=y
        )
    except ValueError:
        return None, n_c0, n_c1

    if len(set(y_train)) < 2 or len(set(y_test)) < 2:
        return None, n_c0, n_c1

    clf = LogisticRegression(max_iter=200)
    try:
        clf.fit(X_train, y_train)
    except Exception:
        return None, n_c0, n_c1

    y_pred = clf.predict_proba(X_test)[:, 1]
    try:
        auc = float(roc_auc_score(y_test, y_pred))
    except ValueError:
        return None, n_c0, n_c1
    auc = max(auc, 1.0 - auc)
    return auc, n_c0, n_c1


def _verdict(aucs: dict[str, float | None]) -> str:
    """Apply the §B.2-D1 verdict logic across the 4 masked regimes."""
    masked = ["ON_30", "ON_90", "ON_180", "ON_365"]
    vals = [aucs.get(k) for k in masked]
    if any(v is None for v in vals):
        return "INSUFFICIENT-DATA"

    finite_vals: list[float] = [float(v) for v in vals if v is not None]

    if min(finite_vals) >= 0.85:
        return "TARGET-EQUIVALENT"

    # Monotonic-decrease check from largest window (ON_365) to smallest (ON_30)
    # i.e. ON_365 > ON_180 > ON_90 > ON_30 (non-strict, so >=).
    # Equivalent: AUC[365] >= AUC[180] >= AUC[90] >= AUC[30].
    auc_30 = aucs["ON_30"]
    auc_90 = aucs["ON_90"]
    auc_180 = aucs["ON_180"]
    auc_365 = aucs["ON_365"]
    assert auc_30 is not None and auc_90 is not None
    assert auc_180 is not None and auc_365 is not None

    monotone_decreasing = (
        auc_365 >= auc_180 >= auc_90 >= auc_30
    )
    if monotone_decreasing and auc_30 <= 0.55:
        return "LOOKBACK-FIXABLE"

    return "PARTIAL-COLLAPSE"


def _format_auc(v: float | None) -> str:
    if v is None:
        return "—"
    return f"{v:.3f}"


def run(
    off_dir: Path,
    on_dirs: dict[str, Path],
    features: list[str],
    target: str,
    out_path: Path,
) -> None:
    regimes: dict[str, pd.DataFrame] = {"OFF": _load_journeys(off_dir)}
    for label, p in on_dirs.items():
        regimes[label] = _load_journeys(p)

    # Per-regime row counts and target class_1 counts (for the sample-size row)
    n_total: dict[str, int] = {}
    n_class1: dict[str, int] = {}
    for label, df in regimes.items():
        n_total[label] = len(df)
        if target in df.columns:
            n_class1[label] = int(pd.to_numeric(df[target], errors="coerce").fillna(-1).eq(1).sum())
        else:
            n_class1[label] = 0

    # Per-(feature, regime) AUC
    auc_table: dict[str, dict[str, float | None]] = {}
    n_c0_table: dict[str, dict[str, int]] = {}
    n_c1_table: dict[str, dict[str, int]] = {}
    for feat in features:
        auc_table[feat] = {}
        n_c0_table[feat] = {}
        n_c1_table[feat] = {}
        for label in REGIME_ORDER:
            df = regimes[label]
            auc, n0, n1 = _single_feature_auc(df, feat, target)
            auc_table[feat][label] = auc
            n_c0_table[feat][label] = n0
            n_c1_table[feat][label] = n1

    # ------------------------------------------------------------------
    # Markdown rendering
    # ------------------------------------------------------------------
    lines: list[str] = []
    lines.append("# CSU Per-Feature AUC Sweep (n=9,607)")
    lines.append("")
    lines.append(
        "Single-feature `LogisticRegression()` on a 90/10 random split "
        "(seed=42), `max(auc, 1-auc)` for orientation invariance. Regimes: "
        "OFF (no masking) and ON with `--lookback-days` set to "
        "{30, 90, 180, 365}."
    )
    lines.append("")
    lines.append(f"- Target: `{target}`")
    lines.append(f"- OFF dir: `{off_dir}`")
    for label, p in on_dirs.items():
        lines.append(f"- {label} dir: `{p}`")
    lines.append("")
    lines.append("## Cohort sizes")
    lines.append("")
    header = "| metric | " + " | ".join(REGIME_ORDER) + " |"
    sep = "|" + "---|" * (len(REGIME_ORDER) + 1)
    lines.append(header)
    lines.append(sep)
    lines.append(
        "| n_journeys | " + " | ".join(str(n_total[r]) for r in REGIME_ORDER) + " |"
    )
    lines.append(
        "| n_class_1 (target=1) | "
        + " | ".join(str(n_class1[r]) for r in REGIME_ORDER)
        + " |"
    )
    lines.append("")

    lines.append("## Per-feature AUC across regimes")
    lines.append("")
    header2 = (
        "| feature | "
        + " | ".join(f"AUC_{r}" for r in REGIME_ORDER)
        + " | verdict |"
    )
    sep2 = "|" + "---|" * (len(REGIME_ORDER) + 2)
    lines.append(header2)
    lines.append(sep2)

    feature_verdicts: dict[str, str] = {}
    for feat in features:
        row_aucs = auc_table[feat]
        verdict = _verdict(row_aucs)
        feature_verdicts[feat] = verdict
        cells = [_format_auc(row_aucs[r]) for r in REGIME_ORDER]
        lines.append(f"| `{feat}` | " + " | ".join(cells) + f" | {verdict} |")
    lines.append("")

    lines.append("## Per-feature class counts (n_class_0 / n_class_1)")
    lines.append("")
    header3 = "| feature | " + " | ".join(REGIME_ORDER) + " |"
    sep3 = "|" + "---|" * (len(REGIME_ORDER) + 1)
    lines.append(header3)
    lines.append(sep3)
    for feat in features:
        cells = [
            f"{n_c0_table[feat][r]} / {n_c1_table[feat][r]}" for r in REGIME_ORDER
        ]
        lines.append(f"| `{feat}` | " + " | ".join(cells) + " |")
    lines.append("")

    lines.append("## Verdict logic (per §B.2-D1 with codex I1 fix)")
    lines.append("")
    lines.append(
        "- **TARGET-EQUIVALENT**: `min(AUC across {ON_30,ON_90,ON_180,ON_365}) >= 0.85`. "
        "The leakage cannot be removed by shrinking the lookback window — the feature "
        "is structurally tautological with the target."
    )
    lines.append(
        "- **LOOKBACK-FIXABLE**: AUC monotonically decreases as the window shrinks "
        "(AUC_365 >= AUC_180 >= AUC_90 >= AUC_30) **and** AUC_30 <= 0.55. The "
        "leakage attenuates with shorter lookback and is empirically eliminable."
    )
    lines.append(
        "- **PARTIAL-COLLAPSE**: anything else — leakage attenuates but doesn't fully "
        "collapse, or non-monotone, or stays in [0.55, 0.85]."
    )
    lines.append(
        "- **INSUFFICIENT-DATA**: at least one masked regime returned `None` "
        "(e.g. <30 rows in a class, single-class target, or constant feature)."
    )
    lines.append("")

    lines.append("## Recommended verdict")
    lines.append("")
    target_eq = [f for f, v in feature_verdicts.items() if v == "TARGET-EQUIVALENT"]
    fixable = [f for f, v in feature_verdicts.items() if v == "LOOKBACK-FIXABLE"]
    partial = [f for f, v in feature_verdicts.items() if v == "PARTIAL-COLLAPSE"]
    insufficient = [f for f, v in feature_verdicts.items() if v == "INSUFFICIENT-DATA"]
    if target_eq:
        rec = "Path B (structurally not remediable)"
        why = (
            f"{len(target_eq)} feature(s) "
            f"({', '.join(target_eq)}) stay >= 0.85 across all 4 lookback windows; "
            "lookback masking cannot remediate them."
        )
    elif partial:
        rec = "Path B-leaning (partial-collapse blocks Path A acceptance)"
        why = (
            f"{len(partial)} feature(s) "
            f"({', '.join(partial)}) attenuate but do not fully collapse to AUC<=0.55 at 30d. "
            "Path A's '5-features fixable' acceptance criterion not met empirically."
        )
    elif fixable:
        rec = "Path A (5-features fixable)"
        why = (
            f"{len(fixable)} feature(s) "
            f"({', '.join(fixable)}) fully collapse to AUC<=0.55 with monotone decrease "
            "across windows; PR #40's lookback masking is valid at scale."
        )
    else:
        rec = "INSUFFICIENT-DATA — cannot recommend"
        why = "All features returned INSUFFICIENT-DATA; investigate cohort sizes."

    lines.append(f"**{rec}** — {why}")
    if insufficient:
        lines.append("")
        lines.append(
            f"Note: insufficient-data features were skipped from the verdict logic: "
            f"{', '.join(insufficient)}"
        )
    lines.append("")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote: {out_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--off", type=Path, required=True)
    parser.add_argument("--on-30", type=Path, required=True)
    parser.add_argument("--on-90", type=Path, required=True)
    parser.add_argument("--on-180", type=Path, required=True)
    parser.add_argument("--on-365", type=Path, required=True)
    parser.add_argument(
        "--features",
        type=str,
        required=True,
        help="comma-separated feature column names",
    )
    parser.add_argument("--target", type=str, default="treatment_initiated")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    features = [f.strip() for f in args.features.split(",") if f.strip()]
    on_dirs = {
        "ON_30": args.on_30,
        "ON_90": args.on_90,
        "ON_180": args.on_180,
        "ON_365": args.on_365,
    }
    run(
        off_dir=args.off,
        on_dirs=on_dirs,
        features=features,
        target=args.target,
        out_path=args.out,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
