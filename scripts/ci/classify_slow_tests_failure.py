"""Classify a red nightly Job A: upstream provider outage vs real failure.

#1804/#1813: the clinical-context live suite (tests/integration/
test_clinical_context/) deliberately hits real providers so an outage goes RED
instead of silently skipping (#1612). Three transient EMBL-EBI HTTP 500s
(08-21/08-24/08-25) each filed the same red nightly alarm a real regression
files, costing a triage session apiece. The recorded fix: classify the failure
in the REPORTER — never skip/xfail in the tests, never retry 5xx in the client.

Reads the junit XML Job A wrote and prints a GITHUB_OUTPUT payload on stdout:

    classification=upstream-transient|real
    detail=<one line of evidence>

The verdict is ``upstream-transient`` ONLY when every failed/errored test
1. is in the clinical-context live family, and
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

# Both id styles: junit classname dots and file-path prefixes.
_FAMILY_PREFIXES = (
    "tests.integration.test_clinical_context",
    "tests/integration/test_clinical_context",
)

# Hard evidence: the provider itself misbehaved on the wire. Sources: the
# 08-24/08-25 outages ("ChEMBL HTTP 500", "HTTP 500 Internal Server Error"
# assertion reprs), the 08-21 15:51 UTC flavour (read timeout), httpx's error
# taxonomy, and pytest-timeout's kill message for a hung upstream request.
_HARD = re.compile(
    r"HTTP[ /]5\d\d"
    r"|Server error '5\d\d"
    r"|\b5\d\d (?:Internal Server Error|Bad Gateway|Service Unavailable|Gateway Time-?out)"
    r"|ReadTimeout|ConnectTimeout|PoolTimeout|WriteTimeout|TimeoutException"
    r"|ConnectError|ReadError|RemoteProtocolError"
    r"|timed out|Timeout >\d",
    re.IGNORECASE,
)

# Echo evidence: the fan-out degradation assertions report the provider fell
# back — true during an outage, but ALSO true under a client parsing bug, so an
# echo never counts as hard evidence on its own.
_ECHO = re.compile(r"static_fallback", re.IGNORECASE)

_DETAIL_CAP = 900


def _test_id(testcase: ET.Element) -> str:
    return f"{testcase.get('classname', '')}.{testcase.get('name', '')}"


def _in_family(testcase: ET.Element) -> bool:
    classname = testcase.get("classname") or ""
    file_attr = testcase.get("file") or ""
    return classname.startswith(_FAMILY_PREFIXES[0]) or file_attr.startswith(_FAMILY_PREFIXES[1])


def _evidence_text(testcase: ET.Element) -> str:
    """Everything the XML holds for this test: failure/error nodes plus any
    captured output (present once Job A passes ``-o junit_logging=all``)."""
    parts: list[str] = []
    for tag in ("failure", "error", "system-out", "system-err"):
        for node in testcase.findall(tag):
            parts.append(node.get("message") or "")
            parts.append(node.text or "")
    return "\n".join(parts)


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

    verdicts: dict[str, str] = {}
    for tc in failed:
        text = _evidence_text(tc)
        if not _in_family(tc):
            verdict = "foreign"
        elif _HARD.search(text):
            verdict = "hard"
        elif _ECHO.search(text):
            verdict = "echo"
        else:
            verdict = "unrecognized"
        verdicts[_test_id(tc)] = verdict
        print(f"  {verdict:<12} {_test_id(tc)}", file=sys.stderr)

    counts = {
        v: sum(1 for x in verdicts.values() if x == v)
        for v in ("foreign", "hard", "echo", "unrecognized")
    }
    print(f"derived: {len(failed)} failed -> {counts}", file=sys.stderr)

    if counts["foreign"]:
        return "real", (
            f"{counts['foreign']}/{len(failed)} failures outside the clinical-context live family"
        )
    if counts["unrecognized"]:
        return "real", (
            f"{counts['unrecognized']}/{len(failed)} clinical-context failures carry no upstream "
            "5xx/timeout evidence and are not fallback echoes"
        )
    if not counts["hard"]:
        return "real", (
            f"all {len(failed)} failures are fallback echoes (static_fallback) with no hard "
            "5xx/timeout evidence anywhere — a client bug produces exactly this shape"
        )
    names = ", ".join(sorted(t.rsplit(".", 1)[-1] for t in verdicts))
    return "upstream-transient", (
        f"{len(failed)}/{len(failed)} failures in the clinical-context live suite "
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
