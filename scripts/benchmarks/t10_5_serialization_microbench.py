"""T10.5 droplet serialization micro-benchmark for shard 21 §F.

Measures the per-fold serialization+manifest+reload+bootstrap-CI lifecycle on
the droplet, repeated 10×, to replace ungrounded "+5-10s/fold" and "~200 MB
peak" guesses with empirical droplet values.

Design (codex cycle 3, Q-W3-3 RESOLVED 2026-05-01):
  - per-run `tempfile.TemporaryDirectory`
  - atomic-rename ({name}.tmp → os.fsync → os.replace)
  - manifest-MD5 commit (manifest written LAST, contains model_md5 +
    predictions_md5 + idx hashes + numpy/sklearn versions + created_utc)
  - explicit `mlflow.log_artifact(fold_path)` — STUBBED here (no MLflow server)
  - `del fold_state, fold_predictions; gc.collect()` between iterations
  - per-fold telemetry: serialization_wall_clock_ms, manifest_write_wall_clock_ms,
    reload_wall_clock_ms, bootstrap_ci_wall_clock_ms, rss_before_write_mb,
    rss_after_write_mb, rss_after_bootstrap_mb, rss_after_del_mb

Surrogates (documented limitations):
  - "Smallest scenario" — W1 synthetic generator v2 not yet shipped. Use
    sklearn.datasets.make_classification with N=500, prevalence ≈ 0.10,
    n_features=30 → represents the small-N / low-prevalence end of the
    Phase 1 portfolio. NOT the diagnostic / treatment-decision scenario
    that may have larger N + larger model.
  - W2 NGBoost not yet shipped. Use LightGBM frozen-config (100 trees,
    num_leaves=31) — the current Phase 0 estimator. NGBoost serialization
    weight is generally larger (per-tree distributional fit + base
    learners) — re-bench at W2 commit boundary.

The mechanics being validated (TempDir, atomic-rename, manifest-MD5 verify,
dump→bootstrap→reload cycle, del+gc between iterations) do NOT depend on
either W1 or W2 shipping; the absolute timings are lower-bound estimates.

Usage:
    python scripts/benchmarks/t10_5_serialization_microbench.py [--n-iters 10] \
        [--n-rows 500] [--prevalence 0.10] [--seed 42] [--out PATH]

Output: JSON dump with per-iteration records + aggregate summary, plus a
text block formatted for direct paste into shard 21 §F.
"""
from __future__ import annotations

import argparse
import contextlib
import gc
import hashlib
import json
import os
import platform
import sys
import tempfile
import time
import tracemalloc
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import lightgbm as lgb
import numpy as np
import psutil
import sklearn
from sklearn.datasets import make_classification
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedShuffleSplit


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def md5_of_bytes(data: bytes) -> str:
    return hashlib.md5(data).hexdigest()


def md5_of_file(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def md5_of_indices(idx: np.ndarray) -> str:
    # Stable serialization for deterministic hash
    return md5_of_bytes(idx.astype(np.int64, copy=False).tobytes())


def rss_mb() -> float:
    return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)


def atomic_write_bytes(target: Path, payload: bytes) -> None:
    """Write payload atomically: tmp → fsync → replace."""
    tmp = target.with_suffix(target.suffix + ".tmp")
    with tmp.open("wb") as f:
        f.write(payload)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, target)


def atomic_dump_joblib(target: Path, obj: Any) -> int:
    """joblib.dump via a tmp path + atomic rename. Returns bytes written."""
    tmp = target.with_suffix(target.suffix + ".tmp")
    joblib.dump(obj, tmp)
    # Force flush to disk before rename (joblib closes the file on return,
    # but the directory entry is what os.replace commits)
    with tmp.open("rb") as f:
        os.fsync(f.fileno())
    size = tmp.stat().st_size
    os.replace(tmp, target)
    return size


def percentile_ci(values: List[float], confidence: float = 0.95) -> Tuple[float, float]:
    if not values:
        return (float("nan"), float("nan"))
    alpha = (1 - confidence) / 2
    return (
        float(np.percentile(values, alpha * 100)),
        float(np.percentile(values, (1 - alpha) * 100)),
    )


def bootstrap_ci_replica(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba_pos: np.ndarray,
    n_bootstrap: int = 1000,
    rng: np.random.Generator | None = None,
) -> Dict[str, Tuple[float, float]]:
    """Replicates evaluator._compute_bootstrap_ci for binary classification.

    Mirrors src/agents/ml_foundation/model_trainer/nodes/evaluator.py:1455-1551
    (1000-iter percentile-CI on AUC + accuracy + precision + recall) so the
    benchmark exercises an equivalent memory + CPU footprint.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    n = len(y_true)
    auc_buf: List[float] = []
    acc_buf: List[float] = []
    prec_buf: List[float] = []
    rec_buf: List[float] = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        yt = y_true[idx]
        yp = y_pred[idx]
        yprob = y_proba_pos[idx]
        acc_buf.append(accuracy_score(yt, yp))
        prec_buf.append(precision_score(yt, yp, zero_division=0))
        rec_buf.append(recall_score(yt, yp, zero_division=0))
        try:
            auc_buf.append(roc_auc_score(yt, yprob))
        except ValueError:
            pass
    return {
        "auc": percentile_ci(auc_buf),
        "accuracy": percentile_ci(acc_buf),
        "precision": percentile_ci(prec_buf),
        "recall": percentile_ci(rec_buf),
    }


# ---------------------------------------------------------------------------
# One iteration of the per-fold lifecycle
# ---------------------------------------------------------------------------


def run_one_iteration(
    iter_idx: int,
    seed_base: int,
    X: np.ndarray,
    y: np.ndarray,
    n_bootstrap: int,
) -> Dict[str, Any]:
    """One fit → dump → bootstrap CI → reload → del+gc cycle.

    Returns telemetry dict matching shard 21 §M.1 + the file-size additions.
    """
    derived_seed = int(
        np.random.SeedSequence((iter_idx, seed_base)).generate_state(1)[0]
    )

    # 70/15/15 stratified split
    sss_outer = StratifiedShuffleSplit(
        n_splits=1, test_size=0.30, random_state=derived_seed
    )
    train_idx, rest_idx = next(sss_outer.split(X, y))
    sss_inner = StratifiedShuffleSplit(
        n_splits=1, test_size=0.50, random_state=derived_seed + 1
    )
    val_rel, test_rel = next(sss_inner.split(X[rest_idx], y[rest_idx]))
    val_idx = rest_idx[val_rel]
    test_idx = rest_idx[test_rel]

    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    rss_before_fit = rss_mb()

    # Fit (frozen-config LightGBM surrogate)
    fit_t0 = time.perf_counter()
    model = lgb.LGBMClassifier(
        n_estimators=100,
        num_leaves=31,
        max_depth=-1,
        random_state=derived_seed & 0xFFFFFFFF,
        verbose=-1,
    )
    model.fit(X_train, y_train)
    fit_wall_clock_ms = (time.perf_counter() - fit_t0) * 1000

    # Predict
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    y_proba_pos = y_proba[:, 1]

    rss_before_write = rss_mb()

    with tempfile.TemporaryDirectory(prefix=f"t10_5_fold_{iter_idx:02d}_") as tmpdir:
        fold_dir = Path(tmpdir)
        model_path = fold_dir / "model.joblib"
        preds_path = fold_dir / "predictions.npz"
        manifest_path = fold_dir / "manifest.json"

        # Atomic dumps (tmp → fsync → rename)
        ser_t0 = time.perf_counter()
        model_size = atomic_dump_joblib(model_path, model)
        # predictions.npz via in-memory bytes + atomic write (savez_compressed
        # would be a kwarg; we use savez to match the per-fold raw-array
        # contract — compression decision is what T10.5 informs)
        preds_buf = tempfile.NamedTemporaryFile(delete=False, suffix=".npz")
        try:
            np.savez(
                preds_buf,
                y_true=y_test,
                y_pred=y_pred,
                y_proba=y_proba,
                train_idx=train_idx,
                val_idx=val_idx,
                test_idx=test_idx,
            )
            preds_buf.flush()
            os.fsync(preds_buf.fileno())
            preds_buf.close()
            os.replace(preds_buf.name, preds_path)
        finally:
            with contextlib.suppress(FileNotFoundError):
                os.unlink(preds_buf.name)
        preds_size = preds_path.stat().st_size
        serialization_wall_clock_ms = (time.perf_counter() - ser_t0) * 1000

        rss_after_write = rss_mb()

        # Manifest written LAST as commit signal
        man_t0 = time.perf_counter()
        manifest = {
            "schema_version": "adaptive_criteria_v3.phase1.1",
            "iter_idx": iter_idx,
            "seed_base": seed_base,
            "derived_seed": derived_seed,
            "model_md5": md5_of_file(model_path),
            "predictions_md5": md5_of_file(preds_path),
            "train_idx_hash": md5_of_indices(train_idx),
            "val_idx_hash": md5_of_indices(val_idx),
            "test_idx_hash": md5_of_indices(test_idx),
            "numpy_version": np.__version__,
            "sklearn_version": sklearn.__version__,
            "lightgbm_version": lgb.__version__,
            "joblib_version": joblib.__version__,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "model_bytes": model_size,
            "predictions_bytes": preds_size,
            "n_train": int(len(train_idx)),
            "n_val": int(len(val_idx)),
            "n_test": int(len(test_idx)),
            "prevalence_train": float(np.mean(y_train)),
            "prevalence_test": float(np.mean(y_test)),
        }
        atomic_write_bytes(manifest_path, json.dumps(manifest, indent=2).encode())
        manifest_size = manifest_path.stat().st_size
        manifest_write_wall_clock_ms = (time.perf_counter() - man_t0) * 1000

        # Bootstrap CI in between dump and reload (per §F spec)
        boot_t0 = time.perf_counter()
        boot_rng = np.random.default_rng(derived_seed)
        cis = bootstrap_ci_replica(y_test, y_pred, y_proba_pos, n_bootstrap, boot_rng)
        bootstrap_ci_wall_clock_ms = (time.perf_counter() - boot_t0) * 1000

        rss_after_bootstrap = rss_mb()

        # Reload — verify manifest MD5 first, then joblib.load
        rel_t0 = time.perf_counter()
        with manifest_path.open("rb") as f:
            manifest_loaded = json.loads(f.read())
        observed_model_md5 = md5_of_file(model_path)
        observed_preds_md5 = md5_of_file(preds_path)
        if observed_model_md5 != manifest_loaded["model_md5"]:
            raise RuntimeError(
                f"manifest MD5 mismatch (model): "
                f"expected {manifest_loaded['model_md5']} got {observed_model_md5}"
            )
        if observed_preds_md5 != manifest_loaded["predictions_md5"]:
            raise RuntimeError(
                f"manifest MD5 mismatch (preds): "
                f"expected {manifest_loaded['predictions_md5']} got {observed_preds_md5}"
            )
        model_reloaded = joblib.load(model_path)
        preds_reloaded = np.load(preds_path)
        # Sanity: re-score AUC and require identity to original
        y_proba_pos_reloaded = model_reloaded.predict_proba(X_test)[:, 1]
        auc_orig = roc_auc_score(y_test, y_proba_pos)
        auc_reloaded = roc_auc_score(preds_reloaded["y_true"], y_proba_pos_reloaded)
        reload_wall_clock_ms = (time.perf_counter() - rel_t0) * 1000

        if not np.isclose(auc_orig, auc_reloaded, atol=1e-12):
            raise RuntimeError(
                f"AUC drift after reload: orig={auc_orig} reloaded={auc_reloaded}"
            )

        # Capture telemetry before del
        record = {
            "iter_idx": iter_idx,
            "derived_seed": derived_seed,
            "fit_wall_clock_ms": fit_wall_clock_ms,
            "serialization_wall_clock_ms": serialization_wall_clock_ms,
            "manifest_write_wall_clock_ms": manifest_write_wall_clock_ms,
            "bootstrap_ci_wall_clock_ms": bootstrap_ci_wall_clock_ms,
            "reload_wall_clock_ms": reload_wall_clock_ms,
            "rss_before_fit_mb": rss_before_fit,
            "rss_before_write_mb": rss_before_write,
            "rss_after_write_mb": rss_after_write,
            "rss_after_bootstrap_mb": rss_after_bootstrap,
            "model_bytes": model_size,
            "predictions_bytes": preds_size,
            "manifest_bytes": manifest_size,
            "n_train": int(len(train_idx)),
            "n_val": int(len(val_idx)),
            "n_test": int(len(test_idx)),
            "auc_test": float(auc_orig),
            "ci_auc": list(cis["auc"]),
        }

        # Cleanup happens via TemporaryDirectory exit; capture rss after del
        del model, model_reloaded, preds_reloaded, y_pred, y_proba, y_proba_pos
        del cis

    # TempDir destroyed; collect Python-side garbage
    gc.collect()
    record["rss_after_del_mb"] = rss_mb()
    return record


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def aggregate(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Median, p95, max, min for each numeric telemetry field."""
    fields = [
        "fit_wall_clock_ms",
        "serialization_wall_clock_ms",
        "manifest_write_wall_clock_ms",
        "bootstrap_ci_wall_clock_ms",
        "reload_wall_clock_ms",
        "rss_before_write_mb",
        "rss_after_write_mb",
        "rss_after_bootstrap_mb",
        "rss_after_del_mb",
        "model_bytes",
        "predictions_bytes",
        "manifest_bytes",
    ]
    summary: Dict[str, Any] = {"n_iters": len(records)}
    for field in fields:
        values = [r[field] for r in records]
        summary[field] = {
            "min": float(np.min(values)),
            "median": float(np.median(values)),
            "mean": float(np.mean(values)),
            "p95": float(np.percentile(values, 95)),
            "max": float(np.max(values)),
        }
    return summary


def format_for_shard_21(summary: Dict[str, Any], cfg: Dict[str, Any]) -> str:
    """Format the §F results block for direct paste."""
    s = summary
    lines = [
        f"**T10.5 droplet measurements ({cfg['datestamp']})** — N={cfg['n_rows']}, "
        f"prevalence≈{cfg['prevalence']:.2f}, n_features={cfg['n_features']}, "
        f"surrogate estimator: LightGBM frozen-config (100 trees, num_leaves=31), "
        f"surrogate scenario: smallest configurable today (W1 generator v2 not yet shipped); "
        f"hardware: {cfg['hardware']}; Python {cfg['python']}; "
        f"numpy {cfg['numpy']}; sklearn {cfg['sklearn']}; lightgbm {cfg['lightgbm']}; "
        f"joblib {cfg['joblib']}; n_iters={cfg['n_iters']}; n_bootstrap={cfg['n_bootstrap']}.",
        "",
        "| Stage (per fold) | Median | p95 | Max |",
        "|---|---|---|---|",
        f"| Fit (frozen-config LightGBM) | {s['fit_wall_clock_ms']['median']:.0f} ms | {s['fit_wall_clock_ms']['p95']:.0f} ms | {s['fit_wall_clock_ms']['max']:.0f} ms |",
        f"| Serialization (joblib.dump model + savez preds, atomic-rename) | {s['serialization_wall_clock_ms']['median']:.0f} ms | {s['serialization_wall_clock_ms']['p95']:.0f} ms | {s['serialization_wall_clock_ms']['max']:.0f} ms |",
        f"| Manifest write (atomic-rename) | {s['manifest_write_wall_clock_ms']['median']:.1f} ms | {s['manifest_write_wall_clock_ms']['p95']:.1f} ms | {s['manifest_write_wall_clock_ms']['max']:.1f} ms |",
        f"| Bootstrap CI (1000 iter on AUC/acc/prec/rec) | {s['bootstrap_ci_wall_clock_ms']['median']:.0f} ms | {s['bootstrap_ci_wall_clock_ms']['p95']:.0f} ms | {s['bootstrap_ci_wall_clock_ms']['max']:.0f} ms |",
        f"| Reload (manifest-MD5 verify + joblib.load + AUC reidentity) | {s['reload_wall_clock_ms']['median']:.0f} ms | {s['reload_wall_clock_ms']['p95']:.0f} ms | {s['reload_wall_clock_ms']['max']:.0f} ms |",
        "",
        "| Memory / size | Median | p95 | Max |",
        "|---|---|---|---|",
        f"| RSS before write (post-fit, post-predict) | {s['rss_before_write_mb']['median']:.0f} MB | {s['rss_before_write_mb']['p95']:.0f} MB | {s['rss_before_write_mb']['max']:.0f} MB |",
        f"| RSS after write | {s['rss_after_write_mb']['median']:.0f} MB | {s['rss_after_write_mb']['p95']:.0f} MB | {s['rss_after_write_mb']['max']:.0f} MB |",
        f"| RSS after bootstrap CI | {s['rss_after_bootstrap_mb']['median']:.0f} MB | {s['rss_after_bootstrap_mb']['p95']:.0f} MB | {s['rss_after_bootstrap_mb']['max']:.0f} MB |",
        f"| RSS after del + gc.collect (next-iter floor) | {s['rss_after_del_mb']['median']:.0f} MB | {s['rss_after_del_mb']['p95']:.0f} MB | {s['rss_after_del_mb']['max']:.0f} MB |",
        f"| Model bytes (joblib.dump, no compression) | {s['model_bytes']['median']/1024:.1f} KB | {s['model_bytes']['p95']/1024:.1f} KB | {s['model_bytes']['max']/1024:.1f} KB |",
        f"| Predictions bytes (np.savez, no compression) | {s['predictions_bytes']['median']/1024:.1f} KB | {s['predictions_bytes']['p95']/1024:.1f} KB | {s['predictions_bytes']['max']/1024:.1f} KB |",
        f"| Manifest bytes | {s['manifest_bytes']['median']:.0f} B | {s['manifest_bytes']['p95']:.0f} B | {s['manifest_bytes']['max']:.0f} B |",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="T10.5 droplet serialization micro-benchmark"
    )
    parser.add_argument("--n-iters", type=int, default=10)
    parser.add_argument("--n-rows", type=int, default=500)
    parser.add_argument("--prevalence", type=float, default=0.10)
    parser.add_argument("--n-features", type=int, default=30)
    parser.add_argument("--n-informative", type=int, default=10)
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("scripts/benchmarks/t10_5_results.json"),
    )
    args = parser.parse_args()

    np.random.seed(args.seed)

    print(
        f"[T10.5] Generating data: N={args.n_rows}, prevalence≈{args.prevalence:.2f}, "
        f"n_features={args.n_features}",
        flush=True,
    )
    X, y = make_classification(
        n_samples=args.n_rows,
        n_features=args.n_features,
        n_informative=args.n_informative,
        n_redundant=5,
        n_classes=2,
        weights=[1 - args.prevalence, args.prevalence],
        flip_y=0.02,
        class_sep=0.7,
        random_state=args.seed,
    )

    print(
        f"[T10.5] Running {args.n_iters} iterations "
        f"(fit → dump+manifest → bootstrap CI ({args.n_bootstrap} iter) → reload+verify → del+gc)…",
        flush=True,
    )

    records: List[Dict[str, Any]] = []
    overall_t0 = time.perf_counter()
    tracemalloc.start()
    for i in range(args.n_iters):
        rec = run_one_iteration(i, args.seed, X, y, args.n_bootstrap)
        records.append(rec)
        peak_python_mb = tracemalloc.get_traced_memory()[1] / (1024 * 1024)
        rec["python_traced_peak_mb"] = peak_python_mb
        print(
            f"  iter {i:02d}: ser={rec['serialization_wall_clock_ms']:.0f}ms "
            f"man={rec['manifest_write_wall_clock_ms']:.1f}ms "
            f"boot={rec['bootstrap_ci_wall_clock_ms']:.0f}ms "
            f"reload={rec['reload_wall_clock_ms']:.0f}ms "
            f"rss_after_del={rec['rss_after_del_mb']:.0f}MB "
            f"model={rec['model_bytes']/1024:.1f}KB",
            flush=True,
        )
    tracemalloc.stop()
    overall_wall = time.perf_counter() - overall_t0

    summary = aggregate(records)
    cfg = {
        "datestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "n_iters": args.n_iters,
        "n_rows": args.n_rows,
        "n_features": args.n_features,
        "prevalence": args.prevalence,
        "n_bootstrap": args.n_bootstrap,
        "hardware": "8 vCPU / 16 GB RAM droplet (Ubuntu 24.04)",
        "python": platform.python_version(),
        "numpy": np.__version__,
        "sklearn": sklearn.__version__,
        "lightgbm": lgb.__version__,
        "joblib": joblib.__version__,
        "overall_wall_clock_s": overall_wall,
    }
    out_payload = {"config": cfg, "summary": summary, "records": records}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out_payload, indent=2))

    print(f"\n[T10.5] Wrote {args.out} (overall wall: {overall_wall:.1f}s).\n")
    print("=" * 72)
    print("§F PASTE BLOCK BELOW")
    print("=" * 72)
    print(format_for_shard_21(summary, cfg))
    return 0


if __name__ == "__main__":
    sys.exit(main())
