"""RxNav-backed brand aliases (lane 2 of the 2026-09-22 public-APIs design).

The chat extractor's curated alias table missed "iptacopan" and "Rhapsido"
(measured 2026-09-22). RxNav knows every ingredient<->brand pair; these tests
pin how that knowledge is fetched, cached, and — above all — how it degrades.
"""

from __future__ import annotations

import pytest

from src.data.kg.rxnav import RxCUIMatch, RxNavError
from src.rag import brand_aliases
from src.rag.brand_aliases import rxnav_brand_aliases

_KNOWN = {
    "Kisqali": ("1873984", ["ribociclib", "Kisqali"]),
    "Fabhalta": ("2671075", ["iptacopan", "Fabhalta"]),
    "Remibrutinib": ("2724365", ["remibrutinib", "Rhapsido"]),
}


class _FakeRxNav:
    def __init__(self, *, known=None, boom_on=None, approximate_for=()):
        self._known = known if known is not None else _KNOWN
        self._boom_on = boom_on
        self._approximate_for = set(approximate_for)
        self.calls: list[str] = []
        self.related_calls: list[str] = []
        self.closed = False

    def rxcui_for_name(self, name):
        self.calls.append(name)
        if self._boom_on == name:
            raise RxNavError("simulated outage")
        hit = self._known.get(name)
        if not hit:
            return None
        return RxCUIMatch(rxcui=hit[0], approximate=name in self._approximate_for)

    def related_names(self, rxcui, *, ttys=("IN", "BN", "PIN")):
        self.related_calls.append(rxcui)
        for _brand, (cui, names) in self._known.items():
            if cui == rxcui:
                return list(names)
        return []

    def close(self):
        self.closed = True


@pytest.fixture(autouse=True)
def _fresh_cache(monkeypatch):
    brand_aliases.reset_cache()
    monkeypatch.setenv("RXNAV_BRAND_ALIASES", "1")


def test_maps_ingredient_and_marketed_names_to_each_canonical_brand():
    out = rxnav_brand_aliases(["Kisqali", "Fabhalta", "Remibrutinib"], client=_FakeRxNav())
    assert out == {
        "Kisqali": ["ribociclib"],
        "Fabhalta": ["iptacopan"],
        "Remibrutinib": ["rhapsido"],
    }


def test_the_brands_own_name_and_short_tokens_are_dropped():
    fake = _FakeRxNav(known={"Kisqali": ("1", ["KISQALI", "ribociclib", "rib", ""])})
    assert rxnav_brand_aliases(["Kisqali"], client=fake) == {"Kisqali": ["ribociclib"]}


def test_an_unknown_brand_contributes_nothing():
    assert rxnav_brand_aliases(["Nonesuch"], client=_FakeRxNav()) == {}


def test_a_failure_returns_what_was_gathered_and_stops_the_round():
    fake = _FakeRxNav(boom_on="Fabhalta")
    out = rxnav_brand_aliases(["Kisqali", "Fabhalta", "Remibrutinib"], client=fake)
    assert out == {"Kisqali": ["ribociclib"]}
    # Remibrutinib was never asked: one outage must not cost N timeouts.
    assert fake.calls == ["Kisqali", "Fabhalta"]


def test_success_is_cached_for_the_process(monkeypatch):
    fake = _FakeRxNav()
    rxnav_brand_aliases(["Kisqali"], client=fake)
    rxnav_brand_aliases(["Kisqali"], client=fake)
    assert fake.calls == ["Kisqali"]


def test_a_failure_is_retried_only_after_the_negative_ttl(monkeypatch):
    now = {"t": 1000.0}
    monkeypatch.setattr(brand_aliases, "_now", lambda: now["t"])
    fake = _FakeRxNav(boom_on="Kisqali")
    assert rxnav_brand_aliases(["Kisqali"], client=fake) == {}
    assert rxnav_brand_aliases(["Kisqali"], client=fake) == {}
    assert fake.calls == ["Kisqali"]  # within the negative TTL: no retry
    now["t"] += brand_aliases.NEGATIVE_TTL_S + 1
    rxnav_brand_aliases(["Kisqali"], client=fake)
    assert fake.calls == ["Kisqali", "Kisqali"]


def test_the_kill_switch_makes_no_calls(monkeypatch):
    monkeypatch.setenv("RXNAV_BRAND_ALIASES", "0")
    fake = _FakeRxNav()
    assert rxnav_brand_aliases(["Kisqali"], client=fake) == {}
    assert fake.calls == []


def test_the_cache_key_is_the_brand_set_not_the_client(monkeypatch):
    a = _FakeRxNav()
    rxnav_brand_aliases(["Kisqali", "Fabhalta"], client=a)
    b = _FakeRxNav()
    assert rxnav_brand_aliases(["Fabhalta", "Kisqali"], client=b)["Fabhalta"] == ["iptacopan"]
    assert b.calls == []


def test_an_approximate_match_contributes_nothing_and_is_not_expanded():
    # RxNav's typo-corrected fallback (search=2) can land on a DIFFERENT drug;
    # importing that drug's names as our aliases would mislabel chat queries.
    fake = _FakeRxNav(approximate_for={"Fabhalta"})
    out = rxnav_brand_aliases(["Kisqali", "Fabhalta"], client=fake)
    assert out == {"Kisqali": ["ribociclib"]}
    assert fake.related_calls == [_KNOWN["Kisqali"][0]]


def test_a_stale_negative_never_overwrites_a_fresh_positive(monkeypatch):
    now = {"t": 1000.0}
    monkeypatch.setattr(brand_aliases, "_now", lambda: now["t"])
    good = _FakeRxNav()
    assert rxnav_brand_aliases(["Kisqali"], client=good) == {"Kisqali": ["ribociclib"]}
    # A late-arriving failed round (e.g. a concurrent builder whose RxNav call
    # timed out) must not replace the unexpired positive entry.
    brand_aliases._remember(("Kisqali",), {}, complete=False)
    late = _FakeRxNav(boom_on="Kisqali")
    assert rxnav_brand_aliases(["Kisqali"], client=late) == {"Kisqali": ["ribociclib"]}
    assert late.calls == []
    # Once the positive entry has expired, a failed round IS remembered.
    now["t"] += brand_aliases.TTL_S + 1
    assert rxnav_brand_aliases(["Kisqali"], client=late) == {}
    assert late.calls == ["Kisqali"]
    assert rxnav_brand_aliases(["Kisqali"], client=late) == {}
    assert late.calls == ["Kisqali"]  # negative TTL now in force


def test_a_non_rxnav_exception_is_a_failed_round_and_is_remembered():
    # e.g. a schema-malformed payload surfacing as AttributeError/ValueError:
    # gathered aliases are kept, the round stops, and the negative TTL applies so
    # every extractor build does not retry immediately.
    class _Boom(_FakeRxNav):
        def rxcui_for_name(self, name):
            if name == "Fabhalta":
                self.calls.append(name)
                raise ValueError("malformed payload")
            return super().rxcui_for_name(name)

    fake = _Boom()
    out = rxnav_brand_aliases(["Kisqali", "Fabhalta", "Remibrutinib"], client=fake)
    assert out == {"Kisqali": ["ribociclib"]}
    assert fake.calls == ["Kisqali", "Fabhalta"]
    assert rxnav_brand_aliases(["Kisqali", "Fabhalta", "Remibrutinib"], client=fake) == {
        "Kisqali": ["ribociclib"]
    }
    assert fake.calls == ["Kisqali", "Fabhalta"]  # remembered: no retry


def test_concurrent_builders_share_one_round():
    import threading
    import time

    class _Slow(_FakeRxNav):
        def rxcui_for_name(self, name):
            time.sleep(0.2)
            return super().rxcui_for_name(name)

    fake = _Slow()
    results: list[dict] = []

    def _run():
        results.append(rxnav_brand_aliases(["Kisqali"], client=fake))

    workers = [threading.Thread(target=_run) for _ in range(2)]
    for w in workers:
        w.start()
    for w in workers:
        w.join(timeout=5)
    assert fake.calls == ["Kisqali"], "two first-builders must not each run the RxNav round"
    assert results == [{"Kisqali": ["ribociclib"]}, {"Kisqali": ["ribociclib"]}]


def test_a_client_that_cannot_be_built_or_closed_is_a_failed_round_and_is_remembered(monkeypatch):
    # Codex r4 MED: RxNavClient(...) (httpx reads proxy env at construction) and
    # close() sat outside the failed-round handling, so such an exception escaped
    # the documented {}-when-down contract and was never negative-cached.
    built = {"n": 0}

    class _Unbuildable:
        def __init__(self, *, timeout):
            built["n"] += 1
            raise ValueError("bad proxy url")

    monkeypatch.setattr(brand_aliases, "RxNavClient", _Unbuildable)
    assert rxnav_brand_aliases(["Kisqali"]) == {}
    assert rxnav_brand_aliases(["Kisqali"]) == {}
    assert built["n"] == 1, "a failed construction must be remembered, not retried per build"

    brand_aliases.reset_cache()
    closed = {"n": 0}

    class _CloseBoom(_FakeRxNav):
        def close(self):
            closed["n"] += 1
            raise OSError("socket already gone")

    monkeypatch.setattr(brand_aliases, "RxNavClient", lambda *, timeout: _CloseBoom())
    out = rxnav_brand_aliases(["Kisqali"])
    assert out == {"Kisqali": ["ribociclib"]}, (
        "a close() failure after a full round keeps its aliases"
    )
    assert rxnav_brand_aliases(["Kisqali"]) == {"Kisqali": ["ribociclib"]}
    assert closed["n"] == 1, "the full round is cached; close() is not attempted again"
