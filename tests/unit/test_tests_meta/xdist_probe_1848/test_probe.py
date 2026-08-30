"""Two trivial tests: the workload the #1848 probe must (fail to) run.

Both pass. In the nested crash arm neither is ever executed -- which is the
point: HEAD exited 0 anyway.
"""


def test_probe_one() -> None:
    assert True


def test_probe_two() -> None:
    assert True
