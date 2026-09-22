# Lane 2: RxNav-backed brand aliases for chat entity extraction — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A typed ingredient or marketed name ("iptacopan", "Rhapsido") resolves to its canonical brand in chat entity extraction, sourced live from RxNav, degrading to today's curated alias table when RxNav is unavailable.

**Architecture:** One new client method (`RxNavClient.related_names`), one new focused module (`src/rag/brand_aliases.py`) holding the RxNav→alias logic and its process-level TTL cache, and a two-line merge in `EntityVocabulary.from_default()`. Curated aliases always win on conflict. An env kill-switch (`RXNAV_BRAND_ALIASES=0`) keeps the unit suite offline and gives ops a rollback lever.

**Tech Stack:** httpx (`MockTransport` in tests), functools.lru_cache, threading.Lock, pytest.

**Spec:** `docs/superpowers/specs/2026-09-22-public-apis-live-path-design.md` (Lane 2).

**Measured premise (2026-09-22, inside `e2i_api`):** `EntityExtractor().extract("iptacopan TRx in the northeast").brands == []`, `("Rhapsido NBRx last quarter") == []`, `("ribociclib share trend") == ["Kisqali"]`. RxNav: Kisqali→1873984 / ribociclib→1873916; Fabhalta→2671075 / iptacopan→2671061; Rhapsido→2724393 / remibrutinib→2724365. `related.json?tty=BN` for ribociclib → `['Kisqali']`, iptacopan → `['Fabhalta']`, remibrutinib → `['Rhapsido']`.

---

## Worktree

```bash
git -C /home/enunez/Projects/e2i_causal_analytics worktree add \
  /home/enunez/Projects/e2i_causal_analytics/.worktrees/lane2-rxnav-aliases \
  -b claude/lane2-rxnav-aliases origin/main
cd /home/enunez/Projects/e2i_causal_analytics/.worktrees/lane2-rxnav-aliases
git branch --show-current   # claude/lane2-rxnav-aliases
```

(Branch from `origin/main` once lane 1 has merged; the spec is then on main.) Every `pytest` runs with `-n 0` from this directory; assert `src.__file__` starts with this path in every python invocation.

---

### Task 1: `RxNavClient.related_names`

**Files:**
- Modify: `src/data/kg/rxnav.py` (add a method after `properties`, a cache helper after `_properties_cached`, and a line in `reset_caches`)
- Test: `tests/unit/test_data/test_kg/test_rxnav.py` (append)

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/test_data/test_kg/test_rxnav.py`:

```python
# --------------------------------------------------------------- related_names


def _related_transport(payload: dict, *, status: int = 200) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith("/rxcui/1873916/related.json")
        assert request.url.params.get("tty") == "IN BN PIN"
        return httpx.Response(status, json=payload)

    return httpx.MockTransport(handler)


_RELATED_PAYLOAD = {
    "relatedGroup": {
        "rxcui": "1873916",
        "conceptGroup": [
            {"tty": "BN", "conceptProperties": [{"rxcui": "1873984", "name": "Kisqali", "tty": "BN"}]},
            {"tty": "IN", "conceptProperties": [{"rxcui": "1873916", "name": "ribociclib", "tty": "IN"}]},
            {"tty": "PIN"},  # RxNav omits conceptProperties when a group is empty
        ],
    }
}


def test_related_names_returns_every_name_across_the_requested_ttys() -> None:
    reset_caches()
    client = RxNavClient(client=httpx.Client(transport=_related_transport(_RELATED_PAYLOAD)))
    assert client.related_names("1873916") == ["Kisqali", "ribociclib"]


def test_related_names_is_empty_for_an_empty_rxcui_without_a_request() -> None:
    reset_caches()

    def boom(request: httpx.Request) -> httpx.Response:
        raise AssertionError("no request expected")

    client = RxNavClient(client=httpx.Client(transport=httpx.MockTransport(boom)))
    assert client.related_names("") == []


def test_related_names_raises_rxnav_error_on_http_failure() -> None:
    reset_caches()
    client = RxNavClient(client=httpx.Client(transport=_related_transport({}, status=503)))
    with pytest.raises(RxNavError):
        client.related_names("1873916")


def test_related_names_is_cached_per_client_and_cleared_by_reset() -> None:
    reset_caches()
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(200, json=_RELATED_PAYLOAD)

    client = RxNavClient(client=httpx.Client(transport=httpx.MockTransport(handler)))
    client.related_names("1873916")
    client.related_names("1873916")
    assert calls["n"] == 1
    reset_caches()
    client.related_names("1873916")
    assert calls["n"] == 2
```

Check the file's existing imports include `httpx`, `pytest`, `RxNavClient`, `RxNavError`, `reset_caches`; add any that are missing.

- [ ] **Step 2: Run to verify they fail**

Run: `pytest -n 0 tests/unit/test_data/test_kg/test_rxnav.py -k related_names -v`
Expected: 4 FAIL with `AttributeError: 'RxNavClient' object has no attribute 'related_names'`.

- [ ] **Step 3: Implement**

In `src/data/kg/rxnav.py`, after `_properties_uncached`:

```python
    def related_names(
        self, rxcui: str, *, ttys: tuple[str, ...] = ("IN", "BN", "PIN")
    ) -> list[str]:
        """Names of the concepts related to ``rxcui`` with the given term types.

        ``IN`` (ingredient), ``BN`` (brand name) and ``PIN`` (precise ingredient)
        together give every way a clinician writes one drug: Kisqali <-> ribociclib,
        Fabhalta <-> iptacopan, Rhapsido <-> remibrutinib (measured 2026-09-22).
        Order follows RxNav's concept groups; duplicates are dropped. Raises
        ``RxNavError`` on transport/HTTP failure, like every other method here.
        """
        return list(_related_names_cached(self, rxcui, ttys))

    def _related_names_uncached(self, rxcui: str, ttys: tuple[str, ...]) -> tuple[str, ...]:
        if not rxcui:
            return ()
        payload = self._get(f"/rxcui/{rxcui}/related.json", {"tty": " ".join(ttys)})
        groups = payload.get("relatedGroup", {}).get("conceptGroup") or []
        names: list[str] = []
        for group in groups:
            for concept in group.get("conceptProperties") or []:
                name = concept.get("name")
                if isinstance(name, str) and name and name not in names:
                    names.append(name)
        return tuple(names)
```

After `_properties_cached`:

```python
@lru_cache(maxsize=_LRU_MAXSIZE)
def _related_names_cached(
    client: RxNavClient, rxcui: str, ttys: tuple[str, ...]
) -> tuple[str, ...]:
    return client._related_names_uncached(rxcui, ttys)
```

In `reset_caches()` add `_related_names_cached.cache_clear()`.

- [ ] **Step 4: Run the whole RxNav file**

Run: `pytest -n 0 tests/unit/test_data/test_kg/test_rxnav.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/data/kg/rxnav.py tests/unit/test_data/test_kg/test_rxnav.py
git commit -m "feat(rxnav): related_names — ingredient/brand names for an RxCUI

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 2: `src/rag/brand_aliases.py` — RxNav aliases with a TTL cache and honest degrade

**Files:**
- Create: `src/rag/brand_aliases.py`
- Test: `tests/unit/test_rag/test_brand_aliases.py`

- [ ] **Step 1: Write the failing tests**

```python
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
    def __init__(self, *, known=None, boom_on=None):
        self._known = known if known is not None else _KNOWN
        self._boom_on = boom_on
        self.calls: list[str] = []

    def rxcui_for_name(self, name):
        self.calls.append(name)
        if self._boom_on == name:
            raise RxNavError("simulated outage")
        hit = self._known.get(name)
        return RxCUIMatch(rxcui=hit[0], approximate=False) if hit else None

    def related_names(self, rxcui, *, ttys=("IN", "BN", "PIN")):
        for _brand, (cui, names) in self._known.items():
            if cui == rxcui:
                return list(names)
        return []

    def close(self):
        pass


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
```

- [ ] **Step 2: Run to verify they fail**

Run: `pytest -n 0 tests/unit/test_rag/test_brand_aliases.py -v`
Expected: collection error `ModuleNotFoundError: No module named 'src.rag.brand_aliases'`.

- [ ] **Step 3: Implement**

Create `src/rag/brand_aliases.py`:

```python
"""RxNav-backed brand aliases for chat entity extraction.

Lane 2 of docs/superpowers/specs/2026-09-22-public-apis-live-path-design.md.

``EntityVocabulary.from_default`` ships a hand-curated alias table. Measured on
2026-09-22 it resolved "ribociclib" but not "iptacopan" or "Rhapsido". RxNav's
``related.json`` knows every ingredient <-> brand pair, so this module asks it
once per process and hands back lowercase aliases per canonical brand.

Degrade rules (the point of this module):
- any ``RxNavError`` returns what was gathered so far and STOPS the round, so an
  outage costs one timeout, not one per brand;
- a failed round is remembered for ``NEGATIVE_TTL_S`` before RxNav is asked
  again; a successful round for ``TTL_S``;
- ``RXNAV_BRAND_ALIASES=0`` disables the lookup entirely (offline unit runs,
  operator rollback). The curated table is untouched either way.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from typing import Dict, Iterable, List, Optional, Protocol

from src.data.kg.rxnav import RxCUIMatch, RxNavClient, RxNavError

logger = logging.getLogger(__name__)

__all__ = ["NEGATIVE_TTL_S", "TTL_S", "reset_cache", "rxnav_brand_aliases"]

TTL_S = 24 * 60 * 60
NEGATIVE_TTL_S = 10 * 60
CLIENT_TIMEOUT_S = 2.0
MIN_ALIAS_LEN = 4
ENV_SWITCH = "RXNAV_BRAND_ALIASES"


class _RxNavLike(Protocol):
    def rxcui_for_name(self, name: str) -> Optional[RxCUIMatch]: ...

    def related_names(self, rxcui: str, *, ttys: tuple[str, ...] = ...) -> list[str]: ...


_lock = threading.Lock()
# key -> (expires_at, aliases). A failed round stores {} with the shorter TTL.
_cache: Dict[tuple[str, ...], tuple[float, Dict[str, List[str]]]] = {}


def _now() -> float:  # patched in tests
    return time.monotonic()


def reset_cache() -> None:
    with _lock:
        _cache.clear()


def _enabled() -> bool:
    return os.environ.get(ENV_SWITCH, "1").strip().lower() not in ("0", "false", "no", "off")


def _aliases_for(brand: str, client: _RxNavLike) -> List[str]:
    match = client.rxcui_for_name(brand)
    if match is None:
        return []
    own = brand.strip().lower()
    out: List[str] = []
    for name in client.related_names(match.rxcui):
        alias = name.strip().lower()
        if len(alias) < MIN_ALIAS_LEN or alias == own or alias in out:
            continue
        out.append(alias)
    return out


def _fetch_round(brands: tuple[str, ...], client: _RxNavLike) -> tuple[Dict[str, List[str]], bool]:
    """Returns (aliases, complete). ``complete`` is False when RxNav raised."""
    gathered: Dict[str, List[str]] = {}
    for brand in brands:
        try:
            aliases = _aliases_for(brand, client)
        except RxNavError as exc:
            logger.warning(
                "brand_aliases: RxNav unavailable while resolving %r (%s); "
                "keeping curated aliases only for the remaining brands",
                brand,
                exc,
            )
            return gathered, False
        if aliases:
            gathered[brand] = aliases
    return gathered, True


def rxnav_brand_aliases(
    brands: Iterable[str], *, client: Optional[_RxNavLike] = None
) -> Dict[str, List[str]]:
    """Lowercase RxNav aliases per canonical brand; ``{}`` when disabled or down."""
    if not _enabled():
        return {}
    key = tuple(sorted(set(brands)))
    if not key:
        return {}
    with _lock:
        hit = _cache.get(key)
        if hit is not None and hit[0] > _now():
            return {b: list(a) for b, a in hit[1].items()}
    owns_client = client is None
    rx: _RxNavLike = client if client is not None else RxNavClient(timeout=CLIENT_TIMEOUT_S)
    try:
        gathered, complete = _fetch_round(key, rx)
    finally:
        if owns_client:
            rx.close()  # type: ignore[attr-defined]
    ttl = TTL_S if complete else NEGATIVE_TTL_S
    with _lock:
        _cache[key] = (_now() + ttl, gathered)
    return {b: list(a) for b, a in gathered.items()}
```

- [ ] **Step 4: Run to verify they pass**

Run: `pytest -n 0 tests/unit/test_rag/test_brand_aliases.py -v`
Expected: 8 PASS.

- [ ] **Step 5: Commit**

```bash
git add src/rag/brand_aliases.py tests/unit/test_rag/test_brand_aliases.py
git commit -m "feat(rag): RxNav-backed brand aliases with TTL cache and honest degrade

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 3: Keep the unit suite offline

**Files:**
- Modify: `tests/conftest.py` (top-level, after imports)

- [ ] **Step 1: Set the kill-switch for every unit run**

Add near the top of `tests/conftest.py`, after the imports:

```python
# Lane 2 (2026-09-22): EntityVocabulary.from_default() asks RxNav for brand
# aliases at first build. Unit tests must never reach the network, so the
# lookup is off for the whole suite; the live integration test opts back in.
os.environ.setdefault("RXNAV_BRAND_ALIASES", "0")
```

Ensure `import os` exists in that file.

- [ ] **Step 2: Prove the switch is honoured by the existing extractor suite without network**

Run: `pytest -n 0 tests/unit/test_rag/test_entity_extractor.py -v -p no:cacheprovider`
Expected: all PASS (no change yet; this is the baseline for Task 4).

- [ ] **Step 3: Commit**

```bash
git add tests/conftest.py
git commit -m "test: keep RxNav brand-alias lookup off in the unit suite

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 4: Merge RxNav aliases into `EntityVocabulary.from_default()`

**Files:**
- Modify: `src/rag/entity_extractor.py` (the `brand_aliases` block inside `from_default`, lines ~98-106)
- Test: `tests/unit/test_rag/test_entity_extractor.py` (append)

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/test_rag/test_entity_extractor.py`:

```python
# ============================================================================
# RxNav-backed aliases (lane 2, 2026-09-22)
# ============================================================================


class TestRxNavAliases:
    def test_rxnav_aliases_are_merged_after_curated_ones(self, monkeypatch):
        monkeypatch.setattr(
            "src.rag.entity_extractor.rxnav_brand_aliases",
            lambda brands: {"Fabhalta": ["iptacopan"], "Remibrutinib": ["rhapsido"]},
        )
        extractor = EntityExtractor()
        assert extractor.extract("iptacopan TRx in the northeast").brands == ["Fabhalta"]
        assert extractor.extract("Rhapsido NBRx last quarter").brands == ["Remibrutinib"]
        # curated entries are still first and intact
        assert extractor.vocabulary.brands["Fabhalta"][:3] == ["fabhalta", "factor b", "factor b inhibitor"]

    def test_without_rxnav_the_vocabulary_is_identical_to_the_curated_table(self, monkeypatch):
        monkeypatch.setattr("src.rag.entity_extractor.rxnav_brand_aliases", lambda brands: {})
        with_none = EntityExtractor().vocabulary.brands
        monkeypatch.setattr(
            "src.rag.entity_extractor.rxnav_brand_aliases",
            lambda brands: (_ for _ in ()).throw(RuntimeError("must not be raised by design")),
        )
        # The function contract is "never raises"; if it ever did, from_default must
        # still build the curated table.
        with_raise = EntityExtractor().vocabulary.brands
        assert with_none == with_raise
        assert "iptacopan" not in with_none["Fabhalta"]

    def test_curated_wins_on_conflict(self, monkeypatch):
        # RxNav (hypothetically) returning another brand's curated alias must not steal it.
        monkeypatch.setattr(
            "src.rag.entity_extractor.rxnav_brand_aliases",
            lambda brands: {"Fabhalta": ["ribociclib", "iptacopan"]},
        )
        extractor = EntityExtractor()
        assert extractor.extract("ribociclib share trend").brands == ["Kisqali"]
        assert "ribociclib" not in extractor.vocabulary.brands["Fabhalta"]

    def test_from_default_asks_for_exactly_the_canonical_brands(self, monkeypatch):
        asked = {}
        monkeypatch.setattr(
            "src.rag.entity_extractor.rxnav_brand_aliases",
            lambda brands: asked.setdefault("brands", sorted(brands)) and {},
        )
        vocab = EntityVocabulary.from_default()
        assert asked["brands"] == sorted(vocab.brands)
```

- [ ] **Step 2: Run to verify they fail**

Run: `pytest -n 0 tests/unit/test_rag/test_entity_extractor.py -k RxNav -v`
Expected: 4 FAIL with `AttributeError: module 'src.rag.entity_extractor' has no attribute 'rxnav_brand_aliases'`.

- [ ] **Step 3: Implement**

In `src/rag/entity_extractor.py` add the import after `from src.rag.exceptions import EntityExtractionError`:

```python
from src.rag.brand_aliases import rxnav_brand_aliases
```

Replace the block

```python
        # Ensure all canonical brands are included
        brands = {}
        for brand in canonical_brands:
            brands[brand] = brand_aliases.get(brand, [brand.lower()])
```

with

```python
        # Ensure all canonical brands are included
        brands: Dict[str, List[str]] = {}
        for brand in canonical_brands:
            brands[brand] = list(brand_aliases.get(brand, [brand.lower()]))

        # RxNav-backed aliases (lane 2, 2026-09-22): ingredient and marketed names
        # the curated table does not carry ("iptacopan", "Rhapsido"). Curated
        # entries stay first and win any conflict; the lookup never raises and is
        # empty when RxNav is down or RXNAV_BRAND_ALIASES=0.
        try:
            live = rxnav_brand_aliases(canonical_brands)
        except Exception as exc:  # noqa: BLE001 — belt and braces over a never-raises contract
            logger.warning("RxNav brand aliases unavailable: %s", exc)
            live = {}
        claimed = {alias for aliases in brands.values() for alias in aliases}
        for brand, aliases in live.items():
            if brand not in brands:
                continue
            for alias in aliases:
                if alias in claimed:
                    continue
                brands[brand].append(alias)
                claimed.add(alias)
```

- [ ] **Step 4: Run the whole extractor suite plus the alias module**

Run: `pytest -n 0 tests/unit/test_rag/test_entity_extractor.py tests/unit/test_rag/test_brand_aliases.py -v`
Expected: all PASS, including `test_no_brand_found`.

- [ ] **Step 5: Commit**

```bash
git add src/rag/entity_extractor.py tests/unit/test_rag/test_entity_extractor.py
git commit -m "feat(rag): merge RxNav ingredient/brand names into the chat brand vocabulary

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 5: Live integration test

**Files:**
- Create: `tests/integration/test_rag/test_brand_aliases_live.py` (create `tests/integration/test_rag/__init__.py` if the directory is new)

- [ ] **Step 1: Write the test**

```python
"""Live RxNav: the three real ingredient<->brand pairs (measured 2026-09-22)."""

from __future__ import annotations

import socket

import pytest

from src.rag import brand_aliases
from src.rag.brand_aliases import rxnav_brand_aliases


def _network_available() -> bool:
    try:
        socket.create_connection(("rxnav.nlm.nih.gov", 443), timeout=3).close()
        return True
    except OSError:
        return False


pytestmark = [
    pytest.mark.integration,
    pytest.mark.slow,
    pytest.mark.skipif(not _network_available(), reason="No outbound network to RxNav."),
]


def test_real_rxnav_resolves_the_three_pairs(monkeypatch):
    monkeypatch.setenv("RXNAV_BRAND_ALIASES", "1")
    brand_aliases.reset_cache()
    out = rxnav_brand_aliases(["Kisqali", "Fabhalta", "Remibrutinib"])
    assert "ribociclib" in out["Kisqali"]
    assert "iptacopan" in out["Fabhalta"]
    assert "rhapsido" in out["Remibrutinib"]
```

- [ ] **Step 2: Run it (network is available on the droplet)**

Run: `pytest -n 0 tests/integration/test_rag/test_brand_aliases_live.py -v`
Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add tests/integration/test_rag/
git commit -m "test(rag): live RxNav pairs for the three brands

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 6: Gates, PR, review, merge, deploy, cert

- [ ] **Step 1: Lint whole-tree, CI's commands**

```bash
ruff check --no-cache src/ tests/ && ruff format --check --no-cache src/ tests/
```
Expected: clean. If `ruff format` complains about the new files, run `ruff format src/rag/brand_aliases.py tests/unit/test_rag/test_brand_aliases.py tests/integration/test_rag/` and re-check. No mypy on the droplet.

- [ ] **Step 2: Targeted regression run**

```bash
pytest -n 0 tests/unit/test_rag tests/unit/test_data/test_kg/test_rxnav.py -q
```
Expected: all PASS.

- [ ] **Step 3: Push, PR, CI, codex review (brief must include the verbatim design-pushback paragraph from lane 1's plan), merge with `--merge`.** PR body must state the measured premise (two queries resolve to no brand today), the degrade rules, and the kill-switch.

- [ ] **Step 4: Live cert after deploy** (record before control from 2026-09-22: both queries → `[]`):

```bash
docker inspect -f '{{.State.StartedAt}}' e2i_api
docker exec -i -e PYTHONPATH=/app e2i_api python - <<'EOF' | tee docs/demos/results/2026-09-22_public_apis_live_path/lane2_rxnav_aliases_cert.txt
import os; print("RXNAV_BRAND_ALIASES =", os.environ.get("RXNAV_BRAND_ALIASES", "<unset → on>"))
from src.rag.entity_extractor import EntityExtractor
ex = EntityExtractor()
for q in ("iptacopan TRx in the northeast", "Rhapsido NBRx last quarter", "ribociclib share trend", "Show overall market trends"):
    print(f"{q!r:40s} -> {ex.extract(q).brands}")
print("Fabhalta aliases:", ex.vocabulary.brands["Fabhalta"])
EOF
```
Expected: `['Fabhalta']`, `['Remibrutinib']`, `['Kisqali']`, `[]`; Fabhalta aliases end with `iptacopan`. Commit the cert file on a docs branch.
