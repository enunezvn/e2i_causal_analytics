"""#1991 debt 4: the discovery gate enum has its own name; the refutation gate keeps GateDecision."""


def test_discovery_enum_is_named_distinctly():
    from src.causal_engine.discovery.base import DiscoveryGateDecision
    from src.causal_engine.refutation_runner import GateDecision

    assert {m.value for m in DiscoveryGateDecision} == {"accept", "review", "reject", "augment"}
    assert {m.value for m in GateDecision} == {"proceed", "review", "block"}
    assert DiscoveryGateDecision.__name__ != GateDecision.__name__


def test_old_name_is_gone_from_discovery():
    import src.causal_engine.discovery.base as base

    assert not hasattr(base, "GateDecision")
