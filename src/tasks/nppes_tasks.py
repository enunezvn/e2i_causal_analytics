"""NPPES NPI taxonomy cache refresh tasks (issue #154).

Maintains the ``npi_taxonomy`` Postgres table (migration 034) by ingesting
the monthly CMS NPPES bulk dump from
https://download.cms.gov/nppes/NPI_Files.html. The dump is ~10 GB CSV +
auxiliary "other name" / endpoint files; the monthly cadence comes from CMS.

Architecture
------------
* ``refresh_npi_taxonomy_cache`` — Celery task fired by beat once a month
  (or manually). Downloads the bulk ZIP, streams + transforms each row, and
  UPSERTs into ``npi_taxonomy``.
* ``ingest_bulk_dump_csv`` — pure-Python helper that takes an opened CSV
  iterable and yields normalized rows. Decoupled so unit tests can drive
  it with a tiny fixture without needing a 10 GB download.
* ``upsert_npi_records`` — DB writer; wrapped so tests can inject a fake
  connection.

Production setup
----------------
1. Run ``database/migrations/034_npi_taxonomy_cache.sql`` on the target DB.
2. Set ``NPPES_BULK_DUMP_URL`` env var (defaults to the public CMS URL).
3. Register the cache loader at converter startup:

       from scripts.rwd_common import set_npi_cache_loader
       from src.tasks.nppes_tasks import postgres_cache_loader_factory
       set_npi_cache_loader(postgres_cache_loader_factory(db_url))

4. The beat schedule (``src/workers/celery_app.py``) calls this task on
   the first of every month.

The bulk dump is intentionally NOT downloaded in CI / test environments —
the task is a no-op stub that emits an informational log when invoked
without ``NPPES_BULK_DUMP_PATH`` set, so unit tests can exercise the code
path without network access or 10 GB of disk.
"""

from __future__ import annotations

import csv
import logging
import os
from datetime import date, datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Iterable, Iterator, Mapping, Optional

from src.workers.celery_app import celery_app

if TYPE_CHECKING:
    # ``scripts/`` is intentionally NOT copied into the runtime container image
    # (it ships only ``src/``). Importing ``scripts.rwd_common`` at module load
    # crashed celery autodiscover at boot (prod incident 2026-05-26), so the
    # NPPES schema symbols are imported lazily inside the functions that build
    # records (below); here they exist for type-checkers only.
    from scripts.rwd_common import NppesAddress, NppesRecord, NppesTaxonomy

logger = logging.getLogger(__name__)

NPPES_BULK_DUMP_URL_DEFAULT = "https://download.cms.gov/nppes/NPI_Files.html"

# A subset of the ~330 columns in the NPPES bulk dump. We only need the
# fields that feed our NppesRecord schema; the rest are dropped at ingest
# time to keep the cache table compact.
BULK_DUMP_FIELDS_REQUIRED = (
    "NPI",
    "Entity Type Code",
    "Provider Enumeration Date",
    "Last Update Date",
    "Provider Organization Name (Legal Business Name)",
    "Parent Organization LBN",
    "Is Sole Proprietor",
    "Provider First Name",
    "Provider Last Name (Legal Name)",
    "Provider First Line Business Practice Location Address",
    "Provider Second Line Business Practice Location Address",
    "Provider Business Practice Location Address City Name",
    "Provider Business Practice Location Address State Name",
    "Provider Business Practice Location Address Postal Code",
    "Provider Business Practice Location Address Country Code (If outside U.S.)",
)


def _extract_bulk_taxonomies(row: Mapping[str, str]) -> tuple[NppesTaxonomy, ...]:
    """Pull up to 15 taxonomy slots out of a bulk-dump row.

    The bulk schema repeats taxonomy fields as
    ``Healthcare Provider Taxonomy Code_N`` (N=1..15) plus a parallel
    ``Healthcare Provider Primary Taxonomy Switch_N`` flag.
    """
    from scripts.rwd_common import NppesTaxonomy

    out: list[NppesTaxonomy] = []
    for n in range(1, 16):
        code = (row.get(f"Healthcare Provider Taxonomy Code_{n}") or "").strip()
        if not code:
            continue
        primary_flag = (
            (row.get(f"Healthcare Provider Primary Taxonomy Switch_{n}") or "").strip().upper()
        )
        license_val = (
            row.get(f"Provider License Number_{n}") or row.get(f"License Number_{n}") or ""
        ).strip() or None
        state_val = (
            row.get(f"Provider License Number State Code_{n}")
            or row.get(f"License Number State Code_{n}")
            or ""
        ).strip() or None
        out.append(
            NppesTaxonomy(
                code=code,
                desc=None,  # bulk dump does not carry desc; resolved via NUCC table separately
                primary=(primary_flag == "Y"),
                license=license_val,
                state=state_val,
            )
        )
    return tuple(out)


def _row_to_record(row: Mapping[str, str]) -> NppesRecord | None:
    """Translate one bulk-dump row into an ``NppesRecord``. Returns ``None``
    on missing/invalid NPI."""
    from scripts.rwd_common import NppesAddress, NppesRecord, _parse_nppes_date

    npi = (row.get("NPI") or "").strip()
    if not (len(npi) == 10 and npi.isdigit()):
        return None

    entity_type = (row.get("Entity Type Code") or "").strip() or None

    sole_prop_raw = (row.get("Is Sole Proprietor") or "").strip().upper()
    if sole_prop_raw == "Y":
        sole_prop: bool | None = True
    elif sole_prop_raw == "N":
        sole_prop = False
    else:
        sole_prop = None

    address = NppesAddress(
        address_1=(row.get("Provider First Line Business Practice Location Address") or "").strip()
        or None,
        address_2=(row.get("Provider Second Line Business Practice Location Address") or "").strip()
        or None,
        city=(row.get("Provider Business Practice Location Address City Name") or "").strip()
        or None,
        state=(row.get("Provider Business Practice Location Address State Name") or "").strip()
        or None,
        postal_code=(
            row.get("Provider Business Practice Location Address Postal Code") or ""
        ).strip()
        or None,
        country_code=(
            row.get("Provider Business Practice Location Address Country Code (If outside U.S.)")
            or ""
        ).strip()
        or None,
    )

    return NppesRecord(
        npi=npi,
        entity_type=entity_type,
        enumeration_date=_parse_nppes_date(row.get("Provider Enumeration Date")),
        last_updated_nppes=_parse_nppes_date(row.get("Last Update Date")),
        taxonomies=_extract_bulk_taxonomies(row),
        practice_address=address
        if any(v for v in (address.address_1, address.city, address.state, address.postal_code))
        else None,
        parent_organization_legal_name=(row.get("Parent Organization LBN") or "").strip() or None,
        organization_legal_name=(
            row.get("Provider Organization Name (Legal Business Name)") or ""
        ).strip()
        or None,
        sole_proprietor=sole_prop,
        first_name=(row.get("Provider First Name") or "").strip() or None,
        last_name=(row.get("Provider Last Name (Legal Name)") or "").strip() or None,
        source="bulk_dump",
    )


def ingest_bulk_dump_csv(reader: Iterable[Mapping[str, str]]) -> Iterator[NppesRecord]:
    """Stream a CSV iterable into ``NppesRecord``s.

    Public so tests can drive it with a small fixture without exercising
    Celery or the network. Skips rows with missing/invalid NPI.
    """
    for row in reader:
        rec = _row_to_record(row)
        if rec is None:
            continue
        yield rec


def _record_to_upsert_params(rec: NppesRecord) -> dict[str, Any]:
    """Flatten an NppesRecord to the param dict for the UPSERT query."""
    addr = rec.practice_address
    taxonomies_json = [
        {
            "code": t.code,
            "desc": t.desc,
            "primary": t.primary,
            "license": t.license,
            "state": t.state,
        }
        for t in rec.taxonomies
    ]
    address_json: dict[str, Any] | None = None
    if addr is not None:
        address_json = {
            "address_1": addr.address_1,
            "address_2": addr.address_2,
            "city": addr.city,
            "state": addr.state,
            "postal_code": addr.postal_code,
            "country_code": addr.country_code,
        }
    return {
        "npi": rec.npi,
        "entity_type": rec.entity_type,
        "enumeration_date": rec.enumeration_date,
        "last_updated_nppes": rec.last_updated_nppes,
        "taxonomies": taxonomies_json,
        "practice_address": address_json,
        "parent_organization_legal_name": rec.parent_organization_legal_name,
        "organization_legal_name": rec.organization_legal_name,
        "sole_proprietor": rec.sole_proprietor,
        "first_name": rec.first_name,
        "last_name": rec.last_name,
        "source": rec.source,
    }


UPSERT_SQL = """
INSERT INTO npi_taxonomy (
    npi, entity_type, enumeration_date, last_updated_nppes, taxonomies,
    practice_address, parent_organization_legal_name, organization_legal_name,
    sole_proprietor, first_name, last_name, source, cached_at
) VALUES (
    %(npi)s, %(entity_type)s, %(enumeration_date)s, %(last_updated_nppes)s,
    %(taxonomies)s, %(practice_address)s, %(parent_organization_legal_name)s,
    %(organization_legal_name)s, %(sole_proprietor)s, %(first_name)s,
    %(last_name)s, %(source)s, NOW()
)
ON CONFLICT (npi) DO UPDATE SET
    entity_type = EXCLUDED.entity_type,
    enumeration_date = EXCLUDED.enumeration_date,
    last_updated_nppes = EXCLUDED.last_updated_nppes,
    taxonomies = EXCLUDED.taxonomies,
    practice_address = EXCLUDED.practice_address,
    parent_organization_legal_name = EXCLUDED.parent_organization_legal_name,
    organization_legal_name = EXCLUDED.organization_legal_name,
    sole_proprietor = EXCLUDED.sole_proprietor,
    first_name = EXCLUDED.first_name,
    last_name = EXCLUDED.last_name,
    source = EXCLUDED.source,
    cached_at = NOW()
"""


def _chunked(iterable: Iterable[Any], chunk_size: int) -> Iterator[list[Any]]:
    """Yield successive ``chunk_size``-row lists drawn from ``iterable``.

    Used by ``refresh_npi_taxonomy_cache`` so the monthly NPPES ingest commits
    its transaction at chunk boundaries instead of all-or-nothing at the end.
    A worker death between commits loses at most one chunk of progress; the
    NPI UPSERT is idempotent on the PK so a retry safely re-applies any
    already-committed range.
    """
    if chunk_size < 1:
        raise ValueError(f"chunk_size must be >= 1, got {chunk_size}")
    chunk: list[Any] = []
    for item in iterable:
        chunk.append(item)
        if len(chunk) >= chunk_size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


def upsert_npi_records(
    records: Iterable[NppesRecord],
    *,
    cursor: Any,
    batch_size: int = 1000,
) -> int:
    """UPSERT a stream of records via the supplied DB cursor.

    Returns the number of rows submitted (not necessarily the number of
    inserted-vs-updated rows; Postgres doesn't surface that distinction
    without RETURNING). The cursor must support ``executemany`` with named
    params and a JSON adapter capable of serializing dict/list values
    (psycopg/psycopg2 do; tests inject a fake).
    """
    import json as _json

    batch: list[dict[str, Any]] = []
    total = 0

    def _flush() -> None:
        nonlocal total
        if not batch:
            return
        # Pre-serialize JSON columns so the test fake doesn't have to know
        # about psycopg's adapter chain.
        for p in batch:
            p["taxonomies"] = _json.dumps(p["taxonomies"])
            if p["practice_address"] is not None:
                p["practice_address"] = _json.dumps(p["practice_address"])
        cursor.executemany(UPSERT_SQL, batch)
        total += len(batch)
        batch.clear()

    for rec in records:
        batch.append(_record_to_upsert_params(rec))
        if len(batch) >= batch_size:
            _flush()
    _flush()
    return total


def postgres_cache_loader_factory(db_url: str) -> Callable[[str], Optional[NppesRecord]]:
    """Build a cache loader bound to a Postgres connection URL.

    Imported lazily so callers (and tests) that never construct one don't
    have to install psycopg.
    """

    def _loader(npi: str) -> Optional[NppesRecord]:
        try:
            import psycopg  # type: ignore[import-untyped]
        except ImportError:  # pragma: no cover
            logger.warning("psycopg not installed; cache loader cannot run")
            return None

        with psycopg.connect(db_url) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT npi, entity_type, enumeration_date, last_updated_nppes,
                           taxonomies, practice_address,
                           parent_organization_legal_name, organization_legal_name,
                           sole_proprietor, first_name, last_name, source
                    FROM npi_taxonomy
                    WHERE npi = %s
                    """,
                    (npi,),
                )
                row = cur.fetchone()
        if row is None:
            return None
        return _row_tuple_to_record(row)

    return _loader


def _row_tuple_to_record(row: tuple[Any, ...]) -> NppesRecord:
    """Reconstruct ``NppesRecord`` from a DB row tuple. Centralised so the
    psycopg loader + any future SQLAlchemy loader stay consistent."""
    from scripts.rwd_common import NppesAddress, NppesRecord, NppesTaxonomy

    (
        npi,
        entity_type,
        enumeration_date,
        last_updated_nppes,
        taxonomies_json,
        address_json,
        parent_org,
        org_legal,
        sole_prop,
        first_name,
        last_name,
        source,
    ) = row

    taxonomies: list[NppesTaxonomy] = []
    if isinstance(taxonomies_json, list):
        for t in taxonomies_json:
            if not isinstance(t, Mapping):
                continue
            taxonomies.append(
                NppesTaxonomy(
                    code=str(t.get("code", "")),
                    desc=t.get("desc"),
                    primary=bool(t.get("primary", False)),
                    license=t.get("license"),
                    state=t.get("state"),
                )
            )

    address: NppesAddress | None = None
    if isinstance(address_json, Mapping):
        address = NppesAddress(
            address_1=address_json.get("address_1"),
            address_2=address_json.get("address_2"),
            city=address_json.get("city"),
            state=address_json.get("state"),
            postal_code=address_json.get("postal_code"),
            country_code=address_json.get("country_code"),
        )

    return NppesRecord(
        npi=str(npi),
        entity_type=entity_type,
        enumeration_date=enumeration_date if isinstance(enumeration_date, date) else None,
        last_updated_nppes=last_updated_nppes if isinstance(last_updated_nppes, date) else None,
        taxonomies=tuple(taxonomies),
        practice_address=address,
        parent_organization_legal_name=parent_org,
        organization_legal_name=org_legal,
        sole_proprietor=sole_prop,
        first_name=first_name,
        last_name=last_name,
        source=source or "bulk_dump",
    )


@celery_app.task(bind=True, name="src.tasks.refresh_npi_taxonomy_cache")
def refresh_npi_taxonomy_cache(
    self,
    *,
    bulk_dump_path: Optional[str] = None,
    db_url: Optional[str] = None,
) -> dict[str, Any]:
    """Refresh the NPPES local cache from the monthly bulk dump.

    Parameters
    ----------
    bulk_dump_path
        Filesystem path to a downloaded NPPES bulk dump CSV (uncompressed
        or .gz). When ``None``, read from ``NPPES_BULK_DUMP_PATH`` env var.
        If still unset, the task is a no-op stub that logs an informational
        message — production deployments must set the env var.
    db_url
        Postgres URL. When ``None``, read from ``NPPES_DB_URL`` /
        ``DATABASE_URL`` env vars.

    Returns a result dict with row counts + duration metadata so Celery
    flower / monitoring can surface the refresh.
    """
    bulk_dump_path = bulk_dump_path or os.environ.get("NPPES_BULK_DUMP_PATH")
    db_url = db_url or os.environ.get("NPPES_DB_URL") or os.environ.get("DATABASE_URL")
    start = datetime.now(timezone.utc)

    # Validate env-var settings FIRST so malformed values fail fast — before
    # any file is opened or any DB / module imported.
    commit_chunk_raw = os.environ.get("NPPES_REFRESH_COMMIT_CHUNK", "10000")
    try:
        commit_chunk_size = int(commit_chunk_raw)
    except ValueError:
        logger.error(
            "NPPES_REFRESH_COMMIT_CHUNK=%r is not an integer; skipping refresh",
            commit_chunk_raw,
        )
        return {"status": "error", "reason": "invalid_commit_chunk", "rows_upserted": 0}
    if commit_chunk_size < 1:
        logger.error(
            "NPPES_REFRESH_COMMIT_CHUNK=%d must be >= 1; skipping refresh",
            commit_chunk_size,
        )
        return {"status": "error", "reason": "invalid_commit_chunk", "rows_upserted": 0}

    if not bulk_dump_path:
        msg = (
            "NPPES_BULK_DUMP_PATH not set; skipping refresh. Download the monthly "
            f"dump from {NPPES_BULK_DUMP_URL_DEFAULT} and set NPPES_BULK_DUMP_PATH."
        )
        logger.info(msg)
        return {"status": "skipped", "reason": "no_bulk_dump_path", "rows_upserted": 0}

    if not db_url:
        logger.warning("NPPES_DB_URL/DATABASE_URL not set; skipping DB writes")
        return {"status": "skipped", "reason": "no_db_url", "rows_upserted": 0}

    path = Path(bulk_dump_path)
    if not path.exists():
        logger.error("NPPES bulk dump not found at %s", path)
        return {"status": "error", "reason": "bulk_dump_missing", "rows_upserted": 0}

    try:
        import psycopg  # type: ignore[import-untyped]
    except ImportError:
        logger.error("psycopg not installed; cannot refresh NPPES cache")
        return {"status": "error", "reason": "psycopg_missing", "rows_upserted": 0}

    # Use gzip transparently if the path ends in .gz, else plain text.
    if path.suffix == ".gz":
        import gzip

        fh = gzip.open(path, "rt", newline="", encoding="utf-8", errors="replace")
    else:
        fh = open(path, "r", newline="", encoding="utf-8", errors="replace")

    rows_upserted = 0
    # Commit every commit_chunk_size rows so worker death / Celery hard-time-
    # limit hit doesn't discard the entire ~10 GB ingest. UPSERT is idempotent
    # on the `npi` PK, so a fresh retry safely re-runs the already-committed
    # range — no checkpoint table required.
    try:
        reader = csv.DictReader(fh)
        records_iter = ingest_bulk_dump_csv(reader)
        with psycopg.connect(db_url) as conn:
            for chunk in _chunked(records_iter, commit_chunk_size):
                with conn.cursor() as cur:
                    rows_upserted += upsert_npi_records(chunk, cursor=cur)
                conn.commit()
                logger.info(
                    "NPPES cache refresh progress: %d rows committed",
                    rows_upserted,
                )
    finally:
        fh.close()

    elapsed = (datetime.now(timezone.utc) - start).total_seconds()
    logger.info("NPPES cache refresh complete: %d rows upserted in %.1fs", rows_upserted, elapsed)
    return {
        "status": "ok",
        "rows_upserted": rows_upserted,
        "elapsed_seconds": elapsed,
        "bulk_dump_path": str(path),
    }
