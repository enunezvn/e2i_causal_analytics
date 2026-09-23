"""Classify a red nightly Job A: upstream provider outage vs real failure.

#1804/#1813/#2173: live provider suites deliberately hit real services so an outage
goes RED instead of silently skipping (#1612/#1629). Transient provider 500s
used to file the same red nightly alarm as a real regression. The recorded fix:
classify the failure in the REPORTER — never skip/xfail in the tests, never
retry 5xx in the client.

Reads the junit XML Job A wrote and prints a GITHUB_OUTPUT payload on stdout:

    classification=upstream-transient|real
    detail=<one line of evidence>

The verdict is ``upstream-transient`` ONLY when every failed/errored test
1. is in an explicitly recognized live-provider family, and
2. carries hard upstream evidence (HTTP 5xx / timeout / connect error) in its
   failure text or captured output, or is a recognized fallback ECHO of one
   (the ``static_fallback`` degradation assertions), and
3. at least one test carries the HARD evidence — all-echo is what a client
   parsing bug produces and must stay ``real``.

Anything else — a failure outside the family, an unrecognized error inside it
(e.g. the #1766 arity TypeError), a missing/unparseable junit, zero recorded
failures (infra died before pytest reported) — fails SAFE to ``real``.

The per-test derivation is printed to stderr (wave-27: a guard must print what
it computed), keeping stdout a clean GITHUB_OUTPUT payload for ``tee``.
Stdlib-only and exit-0 always: this runs in an ``if: failure()`` step where a
crash would only blank the output — the reporter treats blank as ``real``.
"""

from __future__ import annotations

import re
import sys

# The junit XML is written by pytest in this same job seconds earlier (embedded
# text is XML-escaped by pytest), so XXE does not apply — and this must stay
# stdlib-only: a missing dep in this `if: failure()` step would blank the
# verdict silently.
import xml.etree.ElementTree as ET  # nosemgrep: python.lang.security.use-defused-xml.use-defused-xml
from pathlib import Path

# Both id styles: junit classname dots and file paths. Keep the UMLS entry an
# exact module/file boundary: a similarly named generic KG test mentioning an
# HTTP 500 can still be a code defect.
_CLINICAL_CLASSNAME_PREFIX = "tests.integration.test_clinical_context"
_CLINICAL_FILE_PREFIX = "tests/integration/test_clinical_context"
_UMLS_CLASSNAME = "tests.integration.test_kg.test_umls_uts_live"
_UMLS_FILE = "tests/integration/test_kg/test_umls_uts_live.py"
# #2267: the RxNav brand-alias live test. Its outage evidence is the logged
# RxNav reason it puts in its own assertion message (a transport error names
# the httpx class, a 5xx says "RxNav HTTP 5xx").
_BRAND_ALIASES_CLASSNAME = "tests.integration.test_rag.test_brand_aliases_live"
_BRAND_ALIASES_FILE = "tests/integration/test_rag/test_brand_aliases_live.py"

# Hard evidence: the provider itself misbehaved on the wire. Sources: the
# 08-24/08-25 outages ("ChEMBL HTTP 500", "HTTP 500 Internal Server Error"
# assertion reprs), the 08-21 15:51 UTC flavour (read timeout), httpx's error
# taxonomy, and pytest-timeout's kill message for a hung upstream request.
_HARD = re.compile(
    r"HTTP[ /]5\d\d"
    r"|status[=:]\s*5\d\d"
    r"|Server error '5\d\d"
    r"|\b5\d\d (?:Internal Server Error|Bad Gateway|Service Unavailable|Gateway Time-?out)"
    r"|ReadTimeout|ConnectTimeout|PoolTimeout|WriteTimeout|TimeoutException"
    r"|ConnectError|ReadError|RemoteProtocolError"
    r"|timed out|Timeout >\d",
    re.IGNORECASE,
)

# Not upstream, whatever tokens follow: brand_aliases logs these only when the
# failure did NOT come through RxNavClient's RxNavError wrapper (an escaped
# httpx error, a malformed payload, a constructor/close() crash) — client
# defects, even when the escaped class is ConnectError (#2267).
_NOT_UPSTREAM = re.compile(
    r"brand_aliases: (?:unexpected \w+ while resolving|RxNav client raised \w+ outside the round)"
)

# Echo evidence: the fan-out degradation assertions report the provider fell
# back — true during an outage, but ALSO true under a client parsing bug, so an
# echo never counts as hard evidence on its own.
_ECHO = re.compile(r"static_fallback", re.IGNORECASE)

# An exception headline outside this explicit provider/transport set is a real
# failure even when its traceback/request context also contains a 500.  #1766
# was a TypeError; an open-ended code-defect denylist would just move the hole
# to the next exception class.  AssertionError is allowed because live-contract
# assertions expose the provider's actual 500 in their failure representation.
_EXCEPTION_HEADLINE = re.compile(r"^(?:E\s+)?(?:[A-Za-z_]\w*\.)*([A-Za-z_]\w*(?:Error|Exception)):")
_UPSTREAM_EXCEPTION_TYPES = frozenset(
    {
        "AssertionError",
        "ChEMBLError",
        "UMLSError",
        "HTTPError",
        "HTTPStatusError",
        "ReadTimeout",
        "ConnectTimeout",
        "PoolTimeout",
        "WriteTimeout",
        "TimeoutError",
        "TimeoutException",
        "ConnectionError",
        "ConnectError",
        "ReadError",
        "RemoteProtocolError",
    }
)

_DETAIL_CAP = 900


def _test_id(testcase: ET.Element) -> str:
    return f"{testcase.get('classname', '')}.{testcase.get('name', '')}"


def _in_family(testcase: ET.Element) -> bool:
    classname = testcase.get("classname") or ""
    file_attr = testcase.get("file") or ""
    return (
        classname == _CLINICAL_CLASSNAME_PREFIX
        or classname.startswith(f"{_CLINICAL_CLASSNAME_PREFIX}.")
        or classname == _UMLS_CLASSNAME
        or classname.startswith(f"{_UMLS_CLASSNAME}.")
        or file_attr == _CLINICAL_FILE_PREFIX
        or file_attr.startswith(f"{_CLINICAL_FILE_PREFIX}/")
        or file_attr == _UMLS_FILE
        or classname == _BRAND_ALIASES_CLASSNAME
        or file_attr == _BRAND_ALIASES_FILE
    )


def _node_text(testcase: ET.Element, tags: tuple[str, ...]) -> str:
    parts: list[str] = []
    for tag in tags:
        for node in testcase.findall(tag):
            parts.append(node.get("message") or "")
            parts.append(node.text or "")
    return "\n".join(parts)


def _failure_exception_type(testcase: ET.Element) -> str | None:
    """Return the exception type from the failure/error headline, if any."""
    for tag in ("failure", "error"):
        for node in testcase.findall(tag):
            text = node.get("message") or node.text or ""
            first_line = text.lstrip().splitlines()[0] if text.strip() else ""
            match = _EXCEPTION_HEADLINE.match(first_line)
            if match:
                return match.group(1)
    return None


def classify(junit_path: Path) -> tuple[str, str]:
    """Return (classification, one-line detail)."""
    if not junit_path.exists():
        return "real", f"no junit at {junit_path} — Job A died before pytest reported (fail-safe)"
    try:
        root = ET.parse(junit_path).getroot()  # nosemgrep — see import comment
    except ET.ParseError as exc:
        return "real", f"junit unparseable ({exc}) — treating as real (fail-safe)"

    failed = [
        tc
        for tc in root.iter("testcase")
        if tc.find("failure") is not None or tc.find("error") is not None
    ]
    if not failed:
        return "real", "junit records 0 failures/errors — the red was infra, not a test (fail-safe)"

    # Keep one verdict per failure node, not per display id.  JUnit may contain
    # the same classname/name more than once (for example after a rerun or when
    # suites are combined), and a later outage must not overwrite an earlier
    # real failure with the same id.
    verdicts: list[tuple[str, str]] = []
    for tc in failed:
        failure_text = _node_text(tc, ("failure", "error"))
        captured_text = _node_text(tc, ("system-out", "system-err"))
        exception_type = _failure_exception_type(tc)
        if not _in_family(tc):
            verdict = "foreign"
        elif exception_type is not None and exception_type not in _UPSTREAM_EXCEPTION_TYPES:
            verdict = "unrecognized"
        elif _NOT_UPSTREAM.search(failure_text):
            verdict = "unrecognized"
        elif _HARD.search(failure_text):
            verdict = "hard"
        elif _ECHO.search(failure_text):
            # Captured output corroborates a recognized fail-open echo.  It
            # cannot by itself reclassify an unrelated assertion/exception:
            # a test can log an upstream 500 and then fail on a code defect.
            verdict = "hard" if _HARD.search(captured_text) else "echo"
        else:
            verdict = "unrecognized"
        verdicts.append((_test_id(tc), verdict))
        print(f"  {verdict:<12} {_test_id(tc)}", file=sys.stderr)

    counts = {
        v: sum(1 for _, verdict in verdicts if verdict == v)
        for v in ("foreign", "hard", "echo", "unrecognized")
    }
    print(f"derived: {len(failed)} failed -> {counts}", file=sys.stderr)

    if counts["foreign"]:
        return "real", (
            f"{counts['foreign']}/{len(failed)} failures outside recognized live-provider families"
        )
    if counts["unrecognized"]:
        return "real", (
            f"{counts['unrecognized']}/{len(failed)} live-provider failures carry no upstream "
            "5xx/timeout evidence and are not fallback echoes"
        )
    if not counts["hard"]:
        return "real", (
            f"all {len(failed)} failures are fallback echoes (static_fallback) with no hard "
            "5xx/timeout evidence anywhere — a client bug produces exactly this shape"
        )
    names = ", ".join(sorted(test_id.rsplit(".", 1)[-1] for test_id, _ in verdicts))
    return "upstream-transient", (
        f"{len(failed)}/{len(failed)} failures in recognized live-provider suites "
        f"(hard 5xx/timeout evidence: {counts['hard']}, fallback echoes: {counts['echo']}) — {names}"
    )


def main(argv: list[str]) -> int:
    junit = Path(argv[1]) if len(argv) > 1 else Path("/tmp/slow-tests-junit.xml")
    try:
        classification, detail = classify(junit)
    except Exception as exc:  # noqa: BLE001 — any crash must still emit a fail-safe verdict
        classification, detail = (
            "real",
            f"classifier crashed ({exc!r}) — treating as real (fail-safe)",
        )
    detail = " ".join(detail.split())[:_DETAIL_CAP]
    print(f"classification={classification}")
    print(f"detail={detail}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
