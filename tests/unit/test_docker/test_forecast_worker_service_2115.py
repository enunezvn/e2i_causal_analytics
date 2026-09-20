"""The ``forecast`` Celery worker, as declared (#2115, Lane B).

TimesFM 2.5 is a 1.4 GB peak in the prod api image (measured 2026-09-20: 3 GB cap,
2 CPU, load 0.6 s warm, 0.37 s per forecast). The owner's decision was that it gets its
OWN worker rather than a slot on an existing tier, and each clause of that decision is a
guard here because each one has a failure mode on this box:

* not inside ``e2i_api`` — a 1.4 GB spike in the tier that serves every chat turn, on a
  16 GB box whose swap is already half used, is how the app tier dies;
* not in ``e2i_bentoml`` — no torch, a 512 MB limit, and a failed start there rolls back
  the app tier;
* concurrency 1 — two prefork children would each hold their own copy of the weights;
* ``HF_HOME`` on a NAMED VOLUME — otherwise every container restart re-downloads 200M
  parameters, and a container that cannot reach huggingface.co starts broken;
* ``replicas: 0`` by default — the same headroom rule that keeps ``worker_heavy`` dark.
  The forecast tool degrades to Holt-Winters when no worker answers, so shipping this
  service dark costs a model and never an answer.

Declared config only; no Docker daemon needed.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
BASE_COMPOSE = REPO_ROOT / "docker" / "docker-compose.yml"
SERVICE = "worker_forecast"


@pytest.fixture(scope="module")
def compose():
    return yaml.safe_load(BASE_COMPOSE.read_text())


@pytest.fixture(scope="module")
def forecast(compose):
    assert SERVICE in compose["services"], f"{SERVICE} is not declared"
    return compose["services"][SERVICE]


def test_the_service_exists_and_runs_the_shared_api_image(forecast):
    assert "e2i-api" in forecast["image"], (
        "TimesFM needs torch + transformers, so it is the api image"
    )


def test_it_consumes_the_forecast_queue_and_only_that_queue(forecast):
    command = " ".join(str(forecast["command"]).split())
    assert "--queues=forecast" in command
    for other in ("shap", "causal", "twins", "analytics", "default", "quick"):
        assert f"--queues={other}" not in command


def test_the_queue_it_consumes_is_the_one_the_dispatcher_sends_to(forecast):
    """A worker consuming a queue nobody publishes to is a worker that never runs."""
    from src.kpi.forecast.timesfm import FORECAST_QUEUE

    assert f"--queues={FORECAST_QUEUE}" in " ".join(str(forecast["command"]).split())


def test_concurrency_is_one_so_only_one_copy_of_the_weights_is_resident(forecast):
    assert "--concurrency=1" in " ".join(str(forecast["command"]).split())


def test_the_memory_limit_clears_the_measured_peak_without_starving_the_box(forecast):
    """Measured peak is 1394 MB; the limit has to clear it and stay modest on 16 GB."""
    limit = forecast["deploy"]["resources"]["limits"]["memory"]
    assert limit.endswith("G")
    gigabytes = float(limit[:-1])
    assert 2.0 <= gigabytes <= 3.0, f"{limit} is outside the measured envelope"


def test_the_hf_cache_is_a_named_volume_so_the_weights_survive_a_restart(forecast, compose):
    mounts = {
        str(v).split(":")[0]: str(v).split(":")[1] for v in forecast["volumes"] if ":" in str(v)
    }
    hf_home = forecast["environment"]["HF_HOME"]
    assert hf_home in mounts.values(), f"HF_HOME {hf_home} is not a mounted volume"
    volume_name = next(name for name, path in mounts.items() if path == hf_home)
    assert volume_name in compose["volumes"], f"{volume_name} is not declared as a volume"


def test_the_model_is_pinned_so_a_silent_upstream_change_cannot_reach_prod(forecast):
    from src.kpi.forecast.timesfm import TIMESFM_MODEL_ID

    assert forecast["environment"]["E2I_TIMESFM_MODEL"] == TIMESFM_MODEL_ID


def test_torch_thread_counts_match_the_cpu_limit_so_it_cannot_oversubscribe(forecast):
    cpus = float(str(forecast["deploy"]["resources"]["limits"]["cpus"]).strip("'\""))
    env = forecast["environment"]
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        assert int(env[key]) <= cpus, f"{key}={env[key]} oversubscribes cpus={cpus}"


def test_it_ships_dark_like_worker_heavy_because_the_box_has_no_headroom(forecast):
    assert forecast["deploy"]["replicas"] == 0


def test_timesfm_is_not_loaded_inside_the_api_or_the_bentoml_service(compose):
    """The two places the owner's decision explicitly forbids."""
    for service in ("api", "bentoml"):
        declared = compose["services"].get(service, {})
        command = " ".join(str(declared.get("command", "")).split())
        assert "--queues=forecast" not in command
        assert "HF_HOME" not in (declared.get("environment") or {}), (
            f"{service} must not carry a TimesFM weight cache"
        )


def test_the_healthcheck_asks_about_THIS_node_not_whether_any_worker_is_alive(forecast):
    """A bare `celery inspect ping` is a proxy, and on this box a false one.

    With no destination it broadcasts to every worker on the broker and exits 0 if ANY
    answers -- so alongside worker_light and worker_medium it would report this
    container healthy while it was dead. MEASURED 2026-09-20 in the prod api image:
    `-d worker_forecast@$HOSTNAME` exits 0 for a live node and 69 for an absent one,
    while the bare broadcast exits 0 for both.
    """
    assert "healthcheck" in forecast
    test = forecast["healthcheck"]["test"]
    assert test[0] == "CMD-SHELL", "$HOSTNAME has to be expanded by a shell"
    command = " ".join(str(x) for x in test[1:])
    assert "inspect ping" in command
    assert "-d worker_forecast@" in command, "the check must name this node"
    assert "$$HOSTNAME" in command or "$HOSTNAME" in command


def test_the_healthcheck_names_the_same_node_the_worker_registers_as(forecast):
    """A destination that does not match --hostname would fail forever."""
    command = " ".join(str(x) for x in forecast["healthcheck"]["test"][1:])
    assert "--hostname=worker_forecast@" in " ".join(str(forecast["command"]).split())
    assert "worker_forecast@" in command
