"""Crossref REST client for DOI → metadata + abstract resolution.

Phase 2.6 ``CitationResolver`` uses this client to retrieve the abstract
for a DOI cited in evidence. Crossref is zero-auth but has a "polite pool"
convention: clients that supply a contact email in the ``User-Agent``
header receive better service prioritization. We expose the email as a
constructor argument and default to the project name + a public placeholder.

Endpoints:
    - ``GET /works/{doi}`` returns full metadata, including ``abstract``
      (when the publisher deposits one — coverage is uneven).

References:
    - API docs: https://api.crossref.org/swagger-ui/
    - Etiquette: https://www.crossref.org/documentation/retrieve-metadata/rest-api/tips-for-using-the-crossref-rest-api/
    - License: CC0 metadata (https://www.crossref.org/license/)
"""

from __future__ import annotations

import html
import logging
import re
from functools import lru_cache
from typing import Any, Optional
from urllib.parse import quote

import httpx

from src.data.kg.types import AbstractRecord

logger = logging.getLogger(__name__)

CROSSREF_BASE = "https://api.crossref.org"
DEFAULT_TIMEOUT = 15.0
_LRU_MAXSIZE = 4096

# Many publishers deposit JATS-XML wrapped abstracts; some publishers also
# include nested non-JATS markup (XHTML, MathML, plain XML). Strip ALL
# tags, not just the JATS-namespaced ones, then unescape HTML entities so
# CitationResolver's substring matcher sees clean text.
#
# Codex review MEDIUM (2026-05-08): the prior ``</?jats:[^>]+>`` only
# matched ``jats:`` tags, leaking ``<p>``, ``<i>``, ``<sub>``, ``<mml:math>``,
# and ``&amp;`` / ``&#x2014;`` artifacts into the abstract text. Those
# leaks would prevent entity matches and confuse downstream consumers.
_ALL_TAGS = re.compile(r"<[^>]+>")

# Closing punctuation that should never be preceded by whitespace once tags
# have been replaced with spaces (see ``_fetch_doi_uncached``).
#
# ``%`` is deliberately EXCLUDED (codex review, #1608): "50 %" is a legitimate
# house style in published abstracts, not a tag artifact, and rewriting it to
# "50%" would alter source text this substitution has no business touching.
# Only marks that a stripped tag can realistically sit in front of are listed.
_SPACE_BEFORE_PUNCT = re.compile(r"\s+([,.;:!?)\]])")


class CrossrefError(Exception):
    """Crossref request failed."""


class CrossrefClient:
    """Synchronous Crossref REST client."""

    def __init__(
        self,
        *,
        contact_email: str = "noreply@example.com",
        base: str = CROSSREF_BASE,
        timeout: float = DEFAULT_TIMEOUT,
        client: Optional[httpx.Client] = None,
    ) -> None:
        self._base = base.rstrip("/")
        self._user_agent = f"e2i-causal-analytics/0.1 (mailto:{contact_email})"
        self._client = client if client is not None else httpx.Client(timeout=timeout)
        self._owns_client = client is None

    def __enter__(self) -> "CrossrefClient":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def fetch_doi_metadata(self, doi: str) -> Optional[AbstractRecord]:
        """Fetch metadata + abstract for a DOI. Returns None when not found."""
        return _fetch_doi_cached(self, doi)

    def _fetch_doi_uncached(self, doi: str) -> Optional[AbstractRecord]:
        if not doi:
            return None
        # Crossref's ``/works/{doi}`` accepts the canonical DOI form, but DOIs
        # legitimately contain ``?``, ``#``, ``%``, spaces, and other
        # reserved URL characters that ``httpx`` would interpret as
        # query-string or fragment delimiters. URL-encode everything except
        # ``/`` (which Crossref expects unencoded as the registrant separator).
        encoded_doi = quote(doi, safe="/")
        try:
            response = self._client.get(
                f"{self._base}/works/{encoded_doi}",
                headers={"User-Agent": self._user_agent},
            )
        except httpx.HTTPError as exc:
            raise CrossrefError(f"Crossref transport error: {exc}") from exc
        if response.status_code == 404:
            return None
        if response.status_code >= 400:
            raise CrossrefError(f"Crossref HTTP {response.status_code}: {response.text[:200]!r}")
        try:
            payload: dict[str, Any] = response.json()
        except ValueError as exc:
            raise CrossrefError(f"Crossref non-JSON body: {response.text[:200]!r}") from exc
        message = payload.get("message")
        if not isinstance(message, dict):
            return None
        abstract_raw = message.get("abstract")
        # Crossref ``abstract`` is JATS-XML wrapped (sometimes nested with
        # XHTML or MathML). Strip ALL tags + unescape HTML entities so a
        # simple substring matcher can find entity names. Multiple
        # consecutive whitespace runs (left over after tag removal) are
        # collapsed to a single space so word-boundary matching works.
        # Substitute a SPACE, not the empty string: tags separate adjacent text
        # nodes, and deleting them fuses them. ``<jats:title>Background
        # </jats:title><jats:p>Breast cancer ...`` collapsed to
        # "BackgroundBreast cancer", and since ``_first_match`` matches on WORD
        # BOUNDARIES the fused term could never match — making the first entity
        # after every section heading invisible to verification. JATS abstracts
        # are almost always structured, so this systematically produced false
        # "unverified" verdicts (#1608; measured against the live record for
        # 10.1186/s13058-023-01623-6). The whitespace collapse below removes the
        # extra spaces this introduces.
        abstract_stripped = _ALL_TAGS.sub(" ", abstract_raw or "")
        abstract_unescaped = html.unescape(abstract_stripped)
        abstract = re.sub(r"\s+", " ", abstract_unescaped).strip()
        # The tag->space substitution above leaves a space wherever a tag sat
        # directly before punctuation (``<p>vivo</p>.`` -> ``vivo .``). Re-join
        # so the text reads naturally and terms ending at a punctuation mark
        # (e.g. "C-reactive protein (CRP)") keep their expected shape.
        abstract = _SPACE_BEFORE_PUNCT.sub(r"\1", abstract)
        if not abstract:
            # Many publishers don't deposit abstracts; degrade to None so
            # CitationResolver records "abstract not retrieved".
            return None
        title_list = message.get("title")
        title = str(title_list[0]) if isinstance(title_list, list) and title_list else ""
        container_list = message.get("container-title")
        journal = (
            str(container_list[0]) if isinstance(container_list, list) and container_list else None
        )
        year = _extract_year(message)
        return AbstractRecord(
            identifier=str(doi),
            identifier_kind="doi",
            title=title,
            abstract=abstract,
            source="crossref",
            journal=journal,
            year=year,
            raw=message,
        )


def _extract_year(message: dict[str, Any]) -> Optional[int]:
    """Pull the publication year out of Crossref's nested date arrays.

    Crossref records the publication year inside ``issued.date-parts[0][0]``
    (``date-parts`` is a list of [year, month, day] triples). Falls back to
    ``published-print``/``published-online`` if ``issued`` is absent.
    """
    for key in ("issued", "published-print", "published-online", "published"):
        block = message.get(key)
        if not isinstance(block, dict):
            continue
        date_parts = block.get("date-parts")
        if (
            isinstance(date_parts, list)
            and date_parts
            and isinstance(date_parts[0], list)
            and date_parts[0]
        ):
            year_raw = date_parts[0][0]
            if isinstance(year_raw, int):
                return year_raw
    return None


@lru_cache(maxsize=_LRU_MAXSIZE)
def _fetch_doi_cached(client: CrossrefClient, doi: str) -> Optional[AbstractRecord]:
    return client._fetch_doi_uncached(doi)


def reset_caches() -> None:
    """Clear Crossref caches (useful in tests)."""
    _fetch_doi_cached.cache_clear()
