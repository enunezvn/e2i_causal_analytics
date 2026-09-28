"""The tag ``build_bento`` returns must be a valid ``name:version`` (Part of #2157).

Live, 2026-09-28 (retrain job 073c38eb, worker_medium, bentoml 1.4.39):

    Built Bento: tag="exp_kisq_al_20260923184034_e9f834_deployment:v3
    Deployment failed: bento_validation_error: Bento validation failed:
        ['Bento not found: Invalid Tag tag="exp_kisq_al_20260923184034_e9f834_deployment:v3']

bentoml 1.4.x prints ``Successfully built Bento(tag="name:version").``. The old parser
took everything inside ``Bento(...)`` and stripped quotes off the ENDS only, which left
``tag="name:version``. The fixture below is the verbatim line of a real build. Capture:

    cd <dir with service.py + bentofile.yaml> && BENTOML_HOME=<scratch> \
        bentoml build . --name exp_kisq_al_20260923184034_e9f834_deployment --version v3

``test_real_build_*`` repeats that build through ``build_bento`` into a tmp_path store
and hands the tag to bentoml itself, so a future bentoml output change is caught too.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from src.mlops.bentoml_packaging import _parse_built_bento_tag

pytestmark = pytest.mark.unit

NAME = "exp_kisq_al_20260923184034_e9f834_deployment"

# Verbatim, bentoml 1.4.39 (the version pinned in requirements.txt and in the worker).
REAL_1_4_39_LINE = f'Successfully built Bento(tag="{NAME}:v3").'


@pytest.mark.parametrize(
    "line",
    [
        REAL_1_4_39_LINE,
        # The formats the earlier parser was written for (commit c32e9f50e).
        f"Successfully built Bento({NAME}:v3).",
        f"Successfully built Bento('{NAME}:v3').",
        f"Successfully built Bento(tag='{NAME}:v3').",
        f"Successfully built Bento {NAME}:v3.",
    ],
)
def test_successfully_built_line_yields_name_colon_version(line: str) -> None:
    stdout = f"INFO: Adding BentoML requirement to the image: bentoml==1.4.39.\n{line}\n"
    assert _parse_built_bento_tag(stdout) == f"{NAME}:v3"


def test_no_built_line_yields_none() -> None:
    assert _parse_built_bento_tag("INFO: nothing built here\n") is None


def test_parsed_real_line_is_a_tag_bentoml_accepts() -> None:
    bentoml = pytest.importorskip("bentoml")
    tag = _parse_built_bento_tag(REAL_1_4_39_LINE)
    parsed = bentoml.Tag.from_taglike(tag)
    assert (parsed.name, parsed.version) == (NAME, "v3")


_SERVICE = """import bentoml


@bentoml.service
class Echo:
    @bentoml.api
    def echo(self, x: int) -> int:
        return x
"""


def test_real_build_returns_a_tag_the_store_resolves(tmp_path: Path, monkeypatch) -> None:
    """A real ``bentoml build`` into a throwaway store; never the default store."""
    pytest.importorskip("bentoml")
    from src.mlops.bentoml_packaging import _get_bentoml_executable, build_bento

    home = tmp_path / "bentoml_home"
    monkeypatch.setenv("BENTOML_HOME", str(home))
    monkeypatch.setenv("BENTOML_DO_NOT_TRACK", "true")
    svc = tmp_path / "svc"
    svc.mkdir()
    (svc / "service.py").write_text(_SERVICE)
    (svc / "bentofile.yaml").write_text('service: "service:Echo"\ninclude:\n  - "service.py"\n')
    monkeypatch.chdir(svc)  # bentoml 1.4 refuses a bentofile outside the cwd

    tag = build_bento(service_dir=svc, bento_name=NAME, version="v3")

    assert tag == f"{NAME}:v3"
    got = subprocess.run(
        [_get_bentoml_executable(), "get", tag, "-o", "json"],
        capture_output=True,
        text=True,
        env={**os.environ, "BENTOML_HOME": str(home)},
    )
    assert got.returncode == 0, got.stderr
    assert (home / "bentos" / NAME / "v3").is_dir()
