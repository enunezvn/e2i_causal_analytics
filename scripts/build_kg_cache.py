"""Phase 2.9 Stage 2 KG cache builder.

Reads a manifest module (e.g. ``src.data.manifests.optum_feature_manifest``),
queries KG for every feature with ``kg_entity_codes`` set, and writes a
cache file at::

    {out_dir}/{manifest_sha8}__{target_sha8}.json

Plus a companion summary report at::

    {out_dir}/{manifest_sha8}__{target_sha8}.summary.md

The cache filename has no cohort name — two cohorts with identical
(manifest, target_entity_codes) legitimately share a cache file.

Usage::

    python scripts/build_kg_cache.py \
        --manifest-module src.data.manifests.optum_feature_manifest \
        --features-attr OPTUM_FEATURES \
        --target-entity-codes RXNORM:479158,RXNORM:1011295 \
        --out data/kg_cache

Reference: docs/superpowers/specs/2026-05-08-phase29-stage2-entity-mapping-design.md
"""

from __future__ import annotations

import argparse
import importlib
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, Optional

# Allow `python scripts/build_kg_cache.py` invocation without `python -m`.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.feature_contract import FeatureContract  # noqa: E402
from src.data.kg.cache import (  # noqa: E402
    CacheRecord,
    compose_cache_filename,
    compute_manifest_fingerprint,
    compute_target_codes_fingerprint,
    save_cache,
)

if TYPE_CHECKING:
    from src.data.kg.open_targets import OpenTargetsClient
    from src.data.kg.umls_uts import UMLSClient

logger = logging.getLogger(__name__)


def _parse_target_codes(arg: str) -> list[tuple[str, str]]:
    """Parse ``RXNORM:479158,RXNORM:1011295`` into ``[(RXNORM, 479158), ...]``.

    Empty / whitespace-only input parses to an empty list. Each piece must
    contain exactly one colon; both system and code must be non-empty
    after stripping (target codes bypass FeatureContract's ``__post_init__``
    code-validation, so the parser is the only gate against malformed
    input entering the cache fingerprint).
    """
    if not arg or not arg.strip():
        return []
    out: list[tuple[str, str]] = []
    for piece in arg.split(","):
        piece = piece.strip()
        if not piece:
            continue
        if piece.count(":") != 1:
            raise ValueError(
                f"Bad target code {piece!r}; expected exactly SYSTEM:code with one colon"
            )
        system, code = (part.strip() for part in piece.split(":", 1))
        if not system or not code:
            raise ValueError(
                f"Bad target code {piece!r}; expected non-empty SYSTEM:code"
            )
        out.append((system, code))
    return out


def build_cache_for_manifest(
    *,
    features: Iterable[FeatureContract],
    target_entity_codes: list[tuple[str, str]],
    out_dir: Path,
    umls_client: Optional["UMLSClient"] = None,
    open_targets_client: Optional["OpenTargetsClient"] = None,
) -> Path:
    """Build the cache file for a manifest's entity-bearing features.

    Records are emitted only for features with non-empty
    ``kg_entity_codes``. The skeleton in this PR records a
    ``queried_no_edges`` status with empty edge list when no clients
    are supplied (the no-op path used by CI smoke tests). Live KG
    querying via ``UMLSClient`` + ``OpenTargetsClient`` is wired in
    PR-D; supplying a non-None client here raises NotImplementedError
    so callers don't get silently-empty caches that look successful.

    ``sources_attempted`` reflects what was *actually* attempted, not
    what could be — passing both clients as None records an empty
    tuple, which lets downstream audit logic distinguish "no source
    available" from "source returned no edges."

    Returns the path of the written cache file.
    """
    if umls_client is not None or open_targets_client is not None:
        raise NotImplementedError(
            "Live KG cache querying lands in PR-D; pass umls_client=None and "
            "open_targets_client=None for the schema-and-IO smoke path"
        )

    features = list(features)
    manifest_fp = compute_manifest_fingerprint(features)
    target_fp = compute_target_codes_fingerprint(target_entity_codes)
    cache_path = out_dir / compose_cache_filename(manifest_fp, target_fp)

    sources_attempted = tuple(
        source
        for source, client in (
            ("umls_uts", umls_client),
            ("open_targets", open_targets_client),
        )
        if client is not None
    )

    records: list[CacheRecord] = []
    for fc in features:
        if not fc.kg_entity_codes:
            continue
        records.append(
            CacheRecord(
                feature_name=fc.name,
                manifest_fingerprint_sha8=manifest_fp,
                target_codes_fingerprint_sha8=target_fp,
                queried_at=datetime.now(timezone.utc),
                feature_entity_codes=tuple((t[0], t[1]) for t in fc.kg_entity_codes),
                target_entity_codes=tuple((t[0], t[1]) for t in target_entity_codes),
                sources_attempted=sources_attempted,
                status="queried_no_edges",
                edges=(),
                errors=(),
            )
        )

    save_cache(records, cache_path)

    summary_path = cache_path.with_suffix(".summary.md")
    _write_summary_report(summary_path, records, manifest_fp, target_fp)

    return cache_path


def _write_summary_report(
    path: Path, records: list[CacheRecord], manifest_fp: str, target_fp: str
) -> None:
    lines: list[str] = [
        "# KG Cache Summary",
        "",
        f"**Manifest fingerprint:** `{manifest_fp}`",
        f"**Target codes fingerprint:** `{target_fp}`",
        f"**Generated:** {datetime.now(timezone.utc).isoformat()}",
        f"**Records:** {len(records)}",
        "",
        "| Feature | Status | Edges | Sources |",
        "|---|---|---|---|",
    ]
    for r in sorted(records, key=lambda r: r.feature_name):
        lines.append(
            f"| `{r.feature_name}` | {r.status} | {len(r.edges)} | "
            f"{', '.join(r.sources_attempted) or '—'} |"
        )
    path.write_text("\n".join(lines) + "\n")


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest-module",
        required=True,
        help="Dotted module path containing the FEATURES list "
        "(e.g. src.data.manifests.optum_feature_manifest)",
    )
    parser.add_argument(
        "--features-attr",
        default="OPTUM_FEATURES",
        help="Attribute on the manifest module (default: OPTUM_FEATURES)",
    )
    parser.add_argument(
        "--target-entity-codes",
        default="",
        help="Comma-separated SYSTEM:code list, e.g. 'RXNORM:479158,RXNORM:1011295'",
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output directory for the cache file",
    )
    args = parser.parse_args(argv)

    module = importlib.import_module(args.manifest_module)
    features: Any = getattr(module, args.features_attr)
    target_codes = _parse_target_codes(args.target_entity_codes)

    cache_path = build_cache_for_manifest(
        features=features,
        target_entity_codes=target_codes,
        out_dir=args.out,
    )
    print(f"Wrote cache to {cache_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
