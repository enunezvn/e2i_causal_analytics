"""Contract tests: nightly Job A reds caused by an UPSTREAM provider outage are
routed to a distinct low-priority outcome, not the red nightly alarm (#1804/#1813).

The clinical-context live suite deliberately hits real providers (ChEMBL et al.)
so that a provider outage goes RED instead of silently skipping (#1612). That
signal fired correctly on 08-21/08-24/08-25 — three transient EMBL-EBI HTTP 500s
— but each red filed the same "Nightly slow-tests failed" alarm that a real
regression files, costing a full triage session each time. #1804 recorded the
fix: classify "only the clinical-context live suite failed, on upstream
5xx/timeout" in slow-tests.yml's REPORTER — never skip/xfail in the tests
(that reintroduces the #1612 blind spot) and never retry 5xx in the client
(outages outlast backoff).

These tests pin three layers:
1. the classifier script's verdicts, on the REAL failure text from run
   32822893174 (the 2026-08-25 outage) plus the real-defect shapes it must NOT
   absorb (the #1766 arity TypeError; a parsing bug that only echoes
   static_fallback);
2. the workflow wiring: Job A must actually run the classifier on the junit it
   wrote and publish the verdict, or the reporter silently degrades to
   always-red;
3. the reporter routing, by executing the REAL reporter bash with `gh` shimmed:
   upstream-transient files the low-priority rolling issue; every other shape
   (no verdict, real verdict, a second red lane) still files the red alarm.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[2]
WORKFLOW_PATH = REPO / ".github" / "workflows" / "slow-tests.yml"
CLASSIFIER = REPO / "scripts" / "ci" / "classify_slow_tests_failure.py"

JUNIT_PATH_IN_JOB = "/tmp/slow-tests-junit.xml"
RED_LABEL = "nightly-slow-tests-failure"
UPSTREAM_LABEL = "nightly-upstream-transient"
UPSTREAM_TITLE_PREFIX = "[upstream-transient] "

# ─────────────────────────────────────────────────────────────────────────────
# junit fixtures. Failure text is VERBATIM from run 32822893174 (2026-08-25
# nightly, job "Slow Tests (tracked)") so the classifier is proven against the
# exact artifact shape it will parse, not a paraphrase.
# ─────────────────────────────────────────────────────────────────────────────

_CC_LIVE = "tests.integration.test_clinical_context.test_live_contracts"
_CC_FANOUT = "tests.integration.test_clinical_context.test_fan_out_degradation_signal"

HARD_HTTP_500_ASSERT = (
    "AssertionError: ChEMBL molecule search HTTP 500\n"
    "assert 500 == 200\n"
    " +  where 500 = <Response [500 Internal Server Error]>.status_code"
)
HARD_CHEMBL_ERROR = (
    "src.data.kg.chembl.ChEMBLError: ChEMBL HTTP 500: '<!doctype html>\\n"
    '<html lang="en" class="vf-no-js">\\n  <head>\\n    <script>\\n'
    "// Detect if JS is on and swap vf-no-js for vf-js on the html element"
)
ECHO_FAN_OUT = (
    "AssertionError: clinical-context fan-out is degraded for Kisqali — these "
    "providers fell back instead of returning live data: "
    "{'mechanism': 'static_fallback'}. Expected {'mechanism': 'chembl', "
    "'endpoints': 'clinicaltrials.gov', 'citation': 'pubmed', "
    "'indications': 'openfda'}."
)
ECHO_PROVENANCE = (
    "AssertionError: assert 'static_fallback' == 'chembl'\n\n- chembl\n+ static_fallback"
)
# The #1766 class of REAL defect this classifier must never absorb: a seam
# arity change that only the nightly exercised.
REAL_ARITY_TYPEERROR = "TypeError: fetch_citations() takes 4 positional arguments but 5 were given"
HARD_READ_TIMEOUT = "httpx.ReadTimeout: The read operation timed out"


def _junit(
    cases: list[tuple[str, str, str | None]], system_out: dict[str, str] | None = None
) -> str:
    """Build a junit XML string: (classname, name, failure_text|None for pass)."""
    from xml.sax.saxutils import escape, quoteattr

    rows = []
    n_fail = sum(1 for _, _, f in cases if f is not None)
    for classname, name, failure in cases:
        body = ""
        if failure is not None:
            first_line = failure.splitlines()[0]
            body += f"<failure message={quoteattr(first_line)}>{escape(failure)}</failure>"
        out = (system_out or {}).get(name)
        if out is not None:
            body += f"<system-out>{escape(out)}</system-out>"
        rows.append(
            f'<testcase classname={quoteattr(classname)} name={quoteattr(name)} time="1.0">{body}</testcase>'
        )
    return (
        '<?xml version="1.0" encoding="utf-8"?><testsuites name="pytest tests">'
        f'<testsuite name="pytest" errors="0" failures="{n_fail}" skipped="0" '
        f'tests="{len(cases)}" time="60.0">{"".join(rows)}</testsuite></testsuites>'
    )


REAL_0825_CASES: list[tuple[str, str, str | None]] = [
    (_CC_LIVE, "test_chembl_wire_shape_molecule_and_mechanism", HARD_HTTP_500_ASSERT),
    (_CC_LIVE, "test_chembl_mechanism_of_action_parsed_contract", HARD_CHEMBL_ERROR),
    (_CC_FANOUT, "test_fully_live_fan_out_is_reachable_for_kisqali", ECHO_FAN_OUT),
    (_CC_FANOUT, "test_get_context_payload_carries_live_provenance", ECHO_PROVENANCE),
    (
        "tests.integration.test_adversarial_synthetic.TestMeasurementError",
        "test_measurement_error_detected",
        None,
    ),
]


def _classify(
    tmp_path: Path, junit_xml: str | None
) -> tuple[subprocess.CompletedProcess, dict[str, str]]:
    """Run the real classifier CLI; parse its stdout as GITHUB_OUTPUT lines."""
    junit = tmp_path / "junit.xml"
    if junit_xml is not None:
        junit.write_text(junit_xml, encoding="utf-8")
    result = subprocess.run(
        [sys.executable, str(CLASSIFIER), str(junit)],
        capture_output=True,
        text=True,
        timeout=60,
    )
    outputs: dict[str, str] = {}
    for line in result.stdout.splitlines():
        key, sep, value = line.partition("=")
        if sep:
            outputs[key] = value
    return result, outputs


# ─────────────────────────────────────────────────────────────────────────────
# 1. Classifier verdicts
# ─────────────────────────────────────────────────────────────────────────────


def test_classifier_exists() -> None:
    assert CLASSIFIER.exists(), f"missing {CLASSIFIER}"


def test_real_0825_outage_classifies_upstream_transient(tmp_path: Path) -> None:
    """The exact 2026-08-25 shape: 2 hard HTTP-500 failures + 2 static_fallback
    echoes, all in the clinical-context family → upstream-transient."""
    result, outputs = _classify(tmp_path, _junit(REAL_0825_CASES))
    assert result.returncode == 0, result.stderr
    assert outputs.get("classification") == "upstream-transient", (
        f"outputs={outputs}\nstderr={result.stderr}"
    )
    assert "detail" in outputs and "\n" not in outputs["detail"]


def test_stdout_is_a_clean_github_output_payload(tmp_path: Path) -> None:
    """stdout is tee'd into $GITHUB_OUTPUT, so it must be ONLY key=value lines;
    the per-test derivation (wave-27: print what you computed) goes to stderr."""
    result, outputs = _classify(tmp_path, _junit(REAL_0825_CASES))
    lines = [line for line in result.stdout.splitlines() if line]
    assert [line.split("=", 1)[0] for line in lines] == ["classification", "detail"], lines
    # The derivation must exist and be on stderr, per failed test.
    assert "test_chembl_wire_shape_molecule_and_mechanism" in result.stderr


def test_echo_only_failures_are_real(tmp_path: Path) -> None:
    """All-static_fallback with NO hard 5xx/timeout evidence anywhere is what a
    client parsing bug produces — that must alarm as real, not be absorbed."""
    cases = [
        (_CC_FANOUT, "test_fully_live_fan_out_is_reachable_for_kisqali", ECHO_FAN_OUT),
        (_CC_FANOUT, "test_get_context_payload_carries_live_provenance", ECHO_PROVENANCE),
    ]
    _, outputs = _classify(tmp_path, _junit(cases))
    assert outputs.get("classification") == "real", outputs


def test_a_failure_outside_the_family_forces_real(tmp_path: Path) -> None:
    cases = REAL_0825_CASES[:4] + [
        (
            "tests.integration.test_adversarial_synthetic.TestMeasurementError",
            "test_measurement_error_detected",
            "AssertionError: measurement error not detected",
        )
    ]
    _, outputs = _classify(tmp_path, _junit(cases))
    assert outputs.get("classification") == "real", outputs


def test_the_1766_arity_defect_shape_is_real(tmp_path: Path) -> None:
    """A TypeError inside the family (the #1766 citation-seam arity change) has
    no upstream evidence and must stay a red alarm, even next to a true 500."""
    cases = [
        (_CC_LIVE, "test_chembl_wire_shape_molecule_and_mechanism", HARD_HTTP_500_ASSERT),
        (_CC_FANOUT, "test_get_context_payload_carries_live_provenance", REAL_ARITY_TYPEERROR),
    ]
    _, outputs = _classify(tmp_path, _junit(cases))
    assert outputs.get("classification") == "real", outputs


def test_timeout_flavoured_outage_is_upstream_transient(tmp_path: Path) -> None:
    """The 08-21 15:51 UTC outage flavour: read timeout instead of a 500."""
    cases = [(_CC_LIVE, "test_chembl_wire_shape_molecule_and_mechanism", HARD_READ_TIMEOUT)]
    _, outputs = _classify(tmp_path, _junit(cases))
    assert outputs.get("classification") == "upstream-transient", outputs


def test_system_out_evidence_promotes_an_echo_to_hard(tmp_path: Path) -> None:
    """With `junit_logging=all` the fan-out echoes carry their captured WARNING
    ('ChEMBL MoA lookup failed ... HTTP 500') in <system-out>; that alone must
    satisfy the hard-evidence requirement."""
    warning = (
        "WARNING  src.services.clinical_context.providers:providers.py:144 "
        "clinical-context: ChEMBL MoA lookup failed for ribociclib: ChEMBL HTTP 500: '<!doctype html>"
    )
    cases = [(_CC_FANOUT, "test_get_context_payload_carries_live_provenance", ECHO_PROVENANCE)]
    _, outputs = _classify(
        tmp_path,
        _junit(cases, system_out={"test_get_context_payload_carries_live_provenance": warning}),
    )
    assert outputs.get("classification") == "upstream-transient", outputs


def test_missing_junit_is_real(tmp_path: Path) -> None:
    """Job A can red before pytest writes any junit (pip install, mlflow boot);
    the classifier must fail SAFE to the red alarm, exit 0, and say why."""
    result, outputs = _classify(tmp_path, None)
    assert result.returncode == 0, result.stderr
    assert outputs.get("classification") == "real", outputs


def test_unparseable_junit_is_real(tmp_path: Path) -> None:
    result, outputs = _classify(tmp_path, "<testsuites><truncated")
    assert result.returncode == 0, result.stderr
    assert outputs.get("classification") == "real", outputs


def test_zero_recorded_failures_is_real(tmp_path: Path) -> None:
    """A red job whose junit shows 0 failures means pytest itself (or the
    harness) died — infra, not upstream."""
    cases = [(_CC_LIVE, "test_chembl_wire_shape_molecule_and_mechanism", None)]
    _, outputs = _classify(tmp_path, _junit(cases))
    assert outputs.get("classification") == "real", outputs


# ─────────────────────────────────────────────────────────────────────────────
# 2. Workflow wiring
# ─────────────────────────────────────────────────────────────────────────────


def _load_workflow() -> dict:
    with WORKFLOW_PATH.open() as fh:
        return yaml.safe_load(fh)


def _job_a(workflow: dict) -> dict:
    return workflow["jobs"]["slow-tests"]


def _report_job(workflow: dict) -> dict:
    return workflow["jobs"]["report-failure"]


def _classify_step(job: dict) -> dict:
    return next(
        s for s in job["steps"] if "classify_slow_tests_failure.py" in str(s.get("run", ""))
    )


def test_job_a_pytest_step_captures_logs_into_the_junit() -> None:
    """Without `junit_logging=all` the fan-out echoes carry no HTTP evidence in
    the XML (measured on run 32822893174: no <system-out> at all), so a pure
    fan-out outage would mis-classify as real."""
    job = _job_a(_load_workflow())
    pytest_step = next(s for s in job["steps"] if "--junitxml" in str(s.get("run", "")))
    run = str(pytest_step["run"])
    assert f"--junitxml={JUNIT_PATH_IN_JOB}" in run
    assert "-o junit_logging=all" in run


def test_job_a_classifies_its_own_failure_and_publishes_the_verdict() -> None:
    job = _job_a(_load_workflow())
    step = _classify_step(job)
    assert step.get("if") == "failure()", "must run exactly when the job has failed"
    run = str(step["run"])
    assert JUNIT_PATH_IN_JOB in run, "must classify the junit the pytest step wrote"
    assert "GITHUB_OUTPUT" in run and "tee" in run, (
        "stdout must land in GITHUB_OUTPUT and STAY VISIBLE in the run log (tee)"
    )
    outputs = job.get("outputs") or {}
    assert "steps.classify.outputs.classification" in str(
        outputs.get("outage_classification", "")
    ), outputs
    assert step.get("id") == "classify"


def test_reporter_receives_the_verdict_from_job_a() -> None:
    step = _reporter_step(_report_job(_load_workflow()))
    env = step.get("env") or {}
    assert "needs.slow-tests.outputs.outage_classification" in str(
        env.get("JOB_A_CLASSIFICATION", "")
    ), env


# ─────────────────────────────────────────────────────────────────────────────
# 3. Reporter routing (the REAL bash, gh shimmed)
# ─────────────────────────────────────────────────────────────────────────────

_GH_SHIM = r"""#!/usr/bin/env bash
set -u
STATE="$GH_SHIM_STATE"
printf '%s\n' "$*" >> "$STATE/calls.log"
touch "$STATE/labels" "$STATE/issues"
JQ=""; LABELS=(); TITLE=""; SEARCH=""; DESC=""; POS=()
while [ $# -gt 0 ]; do
  case "$1" in
    --jq) JQ="$2"; shift 2 ;;
    --label) LABELS+=("$2"); shift 2 ;;
    --title) TITLE="$2"; shift 2 ;;
    --search) SEARCH="$2"; shift 2 ;;
    --description) DESC="$2"; shift 2 ;;
    --color|--body|--state|--limit|--json) shift 2 ;;
    *) POS+=("$1"); shift ;;
  esac
done
apply_jq() { if [ -n "$JQ" ]; then jq -r "$JQ"; else cat; fi; }
case "${POS[0]} ${POS[1]}" in
  "label create")
    name="${POS[2]}"
    if [ "${GH_SHIM_LABEL_CREATE:-allow}" = "forbid" ]; then
      echo "HTTP 403: Resource not accessible by integration (https://api.github.com/repos/o/r/labels)" >&2
      exit 1
    fi
    if [ "${#DESC}" -gt 100 ]; then
      echo "HTTP 422: Validation Failed (https://api.github.com/repos/o/r/labels)" >&2
      echo "description is too long (maximum is 100 characters)" >&2
      exit 1
    fi
    if grep -qxF "$name" "$STATE/labels"; then
      echo "HTTP 422: Validation Failed (Label already exists)" >&2; exit 1
    fi
    echo "$name" >> "$STATE/labels"; exit 0 ;;
  "label list")
    grep -F -- "$SEARCH" "$STATE/labels" | jq -R '{name: .}' | jq -s . | apply_jq; exit 0 ;;
  "issue list")
    want="${LABELS[0]:-}"
    awk -F'\t' -v want="$want" '($3==want || want=="") {print}' "$STATE/issues" \
      | jq -R 'split("\t") | {number: (.[0]|tonumber), title: .[1], labels: [{name: .[2]}]}' \
      | jq -s . | apply_jq; exit 0 ;;
  "issue create")
    for l in "${LABELS[@]:-}"; do
      [ -z "$l" ] && continue
      grep -qxF "$l" "$STATE/labels" || { echo "could not add label: '$l' not found" >&2; exit 1; }
    done
    printf '%s\t%s\t%s\n' 999 "$TITLE" "${LABELS[0]:-}" >> "$STATE/issues"
    echo "https://github.com/o/r/issues/999"; exit 0 ;;
  "issue comment")
    exit 0 ;;
esac
echo "gh shim: unmodelled command: ${POS[*]}" >&2; exit 64
"""


def _reporter_step(job: dict) -> dict:
    return next(s for s in job["steps"] if "gh issue create" in str(s.get("run", "")))


def _run_reporter(
    tmp_path: Path,
    *,
    results: dict[str, str],
    classification: str,
    detail: str = "4/4 clinical-context failures (hard=2, echo=2)",
    labels: tuple[str, ...] = (),
    open_issues: tuple[tuple[int, str, str], ...] = (),
    label_create: str = "allow",
) -> tuple[subprocess.CompletedProcess, list[str], Path]:
    step = _reporter_step(_report_job(_load_workflow()))
    bindir = tmp_path / "bin"
    bindir.mkdir(parents=True)
    gh = bindir / "gh"
    gh.write_text(_GH_SHIM)
    gh.chmod(0o755)
    state = tmp_path / "state"
    state.mkdir()
    (state / "labels").write_text("".join(f"{name}\n" for name in labels))
    (state / "issues").write_text(
        "".join(f"{n}\t{title}\t{label}\n" for n, title, label in open_issues)
    )
    env = {
        **os.environ,
        "PATH": f"{bindir}:{os.environ['PATH']}",
        "GH_SHIM_STATE": str(state),
        "GH_SHIM_LABEL_CREATE": label_create,
        "GH_TOKEN": "shim",
        "GH_REPO": "o/r",
        "RUN_URL": "https://github.com/o/r/actions/runs/1",
        "GIT_SHA": "8ccb83951badc0de8ccb83951badc0de8ccb8395",
        "JOB_A_RESULT": results.get("A", "success"),
        "JOB_C_RESULT": results.get("C", "success"),
        "JOB_D_RESULT": results.get("D", "success"),
        "JOB_B_RESULT": results.get("B", "success"),
        "JOB_E_RESULT": results.get("E", "success"),
        "JOB_A_CLASSIFICATION": classification,
        "JOB_A_CLASSIFY_DETAIL": detail,
    }
    result = subprocess.run(
        ["bash", "-c", str(step["run"])], env=env, capture_output=True, text=True, timeout=60
    )
    calls = (state / "calls.log").read_text().splitlines() if (state / "calls.log").exists() else []
    return result, calls, state


def _issue_creates(calls: list[str]) -> list[str]:
    return [c for c in calls if c.startswith("issue create ")]


def _issue_comments(calls: list[str]) -> list[str]:
    return [c for c in calls if c.startswith("issue comment ")]


def test_upstream_transient_files_the_rolling_issue_not_the_red_alarm(tmp_path: Path) -> None:
    result, calls, _ = _run_reporter(
        tmp_path, results={"A": "failure"}, classification="upstream-transient"
    )
    assert result.returncode == 0, f"stdout={result.stdout}\nstderr={result.stderr}"
    creates = _issue_creates(calls)
    assert len(creates) == 1, calls
    assert UPSTREAM_TITLE_PREFIX in creates[0], creates
    assert "Nightly slow-tests failed" not in creates[0], (
        "an upstream-transient must not file the red alarm title"
    )
    # The routing derivation must be printed, not just decided (wave-27).
    assert "upstream_transient=true" in result.stdout, result.stdout


def test_no_verdict_still_files_the_red_alarm(tmp_path: Path) -> None:
    """An empty classification (classify step never ran, crashed, or the output
    plumbing broke) must fail SAFE to the red alarm."""
    result, calls, _ = _run_reporter(
        tmp_path, results={"A": "failure"}, classification="", labels=(RED_LABEL,)
    )
    assert result.returncode == 0, f"stdout={result.stdout}\nstderr={result.stderr}"
    creates = _issue_creates(calls)
    assert len(creates) == 1 and "Nightly slow-tests failed" in creates[0], calls


def test_real_verdict_files_the_red_alarm(tmp_path: Path) -> None:
    result, calls, _ = _run_reporter(
        tmp_path, results={"A": "failure"}, classification="real", labels=(RED_LABEL,)
    )
    assert result.returncode == 0, f"stdout={result.stdout}\nstderr={result.stderr}"
    creates = _issue_creates(calls)
    assert len(creates) == 1 and "Nightly slow-tests failed" in creates[0], calls


def test_a_second_red_lane_overrides_the_upstream_verdict(tmp_path: Path) -> None:
    """Job B (blocking heavies) red + Job A upstream-transient is NOT a quiet
    night — the red alarm must fire for the B failure."""
    result, calls, _ = _run_reporter(
        tmp_path,
        results={"A": "failure", "B": "failure"},
        classification="upstream-transient",
        labels=(RED_LABEL,),
    )
    assert result.returncode == 0, f"stdout={result.stdout}\nstderr={result.stderr}"
    creates = _issue_creates(calls)
    assert len(creates) == 1 and "Nightly slow-tests failed" in creates[0], calls
    assert not any(UPSTREAM_TITLE_PREFIX in c for c in creates), calls


def test_upstream_rolling_issue_dedups_on_title_prefix_without_the_label(tmp_path: Path) -> None:
    """#1807/D3 lesson: `gh issue list --label <missing>` is silently empty, so
    dedup must key on the title prefix the workflow controls. With label create
    forbidden AND an existing open rolling issue, the reporter must comment on
    it — not file a duplicate, and not crash on the label."""
    first, _, state = _run_reporter(
        tmp_path / "first",
        results={"A": "failure"},
        classification="upstream-transient",
        label_create="forbid",
    )
    assert first.returncode == 0, f"stdout={first.stdout}\nstderr={first.stderr}"
    rows = [r.split("\t") for r in (state / "issues").read_text().splitlines() if r]
    assert len(rows) == 1, rows
    existing_title = rows[0][1]
    result, calls, _ = _run_reporter(
        tmp_path / "second",
        results={"A": "failure"},
        classification="upstream-transient",
        label_create="forbid",
        open_issues=((42, existing_title, ""),),
    )
    assert result.returncode == 0, f"stdout={result.stdout}\nstderr={result.stderr}"
    assert _issue_creates(calls) == [], calls
    comments = _issue_comments(calls)
    assert len(comments) == 1 and comments[0].startswith("issue comment 42 "), calls


def test_upstream_label_description_respects_the_100_char_limit(tmp_path: Path) -> None:
    """The exact #1807 D3 defect: a >100-char label description 422s, and a
    hidden 422 muted the alarm. The shim enforces the measured limit, so a too
    -long description would fail label create; the issue must still land WITH
    the reason visible."""
    result, calls, _ = _run_reporter(
        tmp_path, results={"A": "failure"}, classification="upstream-transient"
    )
    assert result.returncode == 0, f"stdout={result.stdout}\nstderr={result.stderr}"
    label_creates = [c for c in calls if c.startswith(f"label create {UPSTREAM_LABEL}")]
    assert label_creates, f"expected the upstream label to be self-healed: {calls}"
    creates = _issue_creates(calls)
    assert len(creates) == 1 and f"--label {UPSTREAM_LABEL}" in creates[0], calls


@pytest.mark.parametrize("lane", ["C", "D", "E"])
def test_other_lane_failures_alone_still_file_the_red_alarm(tmp_path: Path, lane: str) -> None:
    result, calls, _ = _run_reporter(
        tmp_path, results={lane: "failure"}, classification="", labels=(RED_LABEL,)
    )
    assert result.returncode == 0, f"stdout={result.stdout}\nstderr={result.stderr}"
    creates = _issue_creates(calls)
    assert len(creates) == 1 and "Nightly slow-tests failed" in creates[0], calls
