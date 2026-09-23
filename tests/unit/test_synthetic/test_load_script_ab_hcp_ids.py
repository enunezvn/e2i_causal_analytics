"""--refresh-ab must source the A/B unit universe from hcp_profiles (Part of #2207).

The weekly ``--refresh-ab`` path generates NO hcp_profiles frame, so the
namespaced HCP ids that ``ABExperimentGenerator`` samples its panels from
have to be READ from the database (read-only, PK-ordered, paged to
exhaustion: PostgREST pages at 1,000 by default and the universe is 5,000).
An empty or short read must fail LOUD — fabricating ids would silently
recreate the un-joinable ``hcp_NNNNN`` substrate this lane removes.
"""

import importlib

import pytest

load_mod = importlib.import_module("scripts.load_synthetic_data")

_TINY_SIZES = {"trigger": 200}  # -> max(10, 2) = 10 experiments per brand


class _Resp:
    def __init__(self, data, count):
        self.data = data
        self.count = count


class _FakeQuery:
    """Minimal PostgREST builder: records the filters, serves .range() pages."""

    def __init__(self, ids, log, report_count=None):
        self._ids = ids
        self._log = log
        self._report_count = report_count
        self._count = len(ids) if report_count is None else report_count
        self._range = None

    def select(self, *a, **k):
        self._log.append(("select", (a, k)))
        return self

    def eq(self, *a, **k):
        self._log.append(("eq", a))
        return self

    def like(self, column, pattern):
        # Faithful SQL LIKE: '%' = any run, '_' = any ONE char (codex r2 MED —
        # 'scvhcp_%' also matches 'scvhcpX00001').
        import re

        assert column == "hcp_id", column
        self._log.append(("like", (column, pattern)))
        rx = (
            "^"
            + "".join(".*" if c == "%" else "." if c == "_" else re.escape(c) for c in pattern)
            + "$"
        )
        self._ids = [h for h in self._ids if re.match(rx, h)]
        if self._report_count is None:
            self._count = len(self._ids)
        return self

    def order(self, *a, **k):
        self._log.append(("order", a))
        return self

    def range(self, start, end):
        self._log.append(("range", (start, end)))
        self._range = (start, end)
        return self

    def execute(self):
        start, end = self._range
        page = [{"hcp_id": h} for h in self._ids[start : end + 1]]
        return _Resp(page, self._count)


class _FakeClient:
    def __init__(self, ids, report_count=None):
        self.ids = ids
        self.log = []
        self._report_count = report_count

    def table(self, name):
        assert name == "hcp_profiles", name
        return _FakeQuery(self.ids, self.log, self._report_count)


def test_fetch_reads_all_5000_ids_across_pages_pk_ordered():
    ids = [f"scvhcp_{i:05d}" for i in range(5000)]
    client = _FakeClient(ids)
    got = load_mod.fetch_synthetic_hcp_ids(client, page_size=1000)
    assert got == ids
    # read-only, synthetic-only, PK-ordered, paged to an empty terminator
    assert ("eq", ("is_synthetic", True)) in client.log
    assert ("like", ("hcp_id", "scvhcp_%")) in client.log  # scoped to the --tag namespace
    assert ("order", ("hcp_id",)) in client.log
    ranges = [r for kind, r in client.log if kind == "range"]
    assert ranges == [
        (0, 999),
        (1000, 1999),
        (2000, 2999),
        (3000, 3999),
        (4000, 4999),
        (5000, 5999),
    ]
    assert ("select", (("hcp_id",), {"count": "exact"})) in client.log
    assert {kind for kind, _ in client.log} == {"select", "eq", "like", "order", "range"}


def test_fetch_is_cap_agnostic_advances_by_rows_returned():
    """A PostgREST db-max-rows cap below page_size must not drop the tail."""
    ids = [f"scvhcp_{i:05d}" for i in range(2500)]

    class _Capped(_FakeQuery):
        def execute(self):
            start, end = self._range
            page = [{"hcp_id": h} for h in self._ids[start : min(end + 1, start + 700)]]
            return _Resp(page, self._count)

    class _CappedClient(_FakeClient):
        def table(self, name):
            return _Capped(self.ids, self.log)

    assert load_mod.fetch_synthetic_hcp_ids(_CappedClient(ids), page_size=1000) == ids


def test_fetch_fails_loud_on_empty_or_short_universe():
    with pytest.raises(ValueError, match="hcp_profiles"):
        load_mod.fetch_synthetic_hcp_ids(_FakeClient([]))
    # the paged read disagrees with the server's exact count -> refuse
    ids = [f"scvhcp_{i:05d}" for i in range(1500)]
    with pytest.raises(ValueError, match="5000"):
        load_mod.fetch_synthetic_hcp_ids(_FakeClient(ids, report_count=5000))


def test_fetch_fails_loud_on_duplicate_ids():
    ids = [f"scvhcp_{i:05d}" for i in range(100)] + ["scvhcp_00000"]
    with pytest.raises(ValueError, match="duplicate"):
        load_mod.fetch_synthetic_hcp_ids(_FakeClient(ids))


def test_build_ab_refresh_datasets_requires_hcp_ids_and_uses_them():
    with pytest.raises(TypeError):
        load_mod.build_ab_refresh_datasets(_TINY_SIZES, seed=42, id_prefix="scv")
    ids = [f"scvhcp_{i:05d}" for i in range(5000)]
    datasets = load_mod.build_ab_refresh_datasets(
        _TINY_SIZES, seed=42, id_prefix="scv", hcp_ids=ids
    )
    asn = datasets["ab_experiment_assignments"]
    assert len(datasets["ml_experiments"]) == 30
    assert asn["unit_id"].isin(ids).all()
    assert asn["unit_id"].str.startswith("scvhcp_").all()
    assert asn["is_synthetic"].all()


def test_refresh_ab_cli_sources_ids_from_the_db_before_generating(monkeypatch):
    """The CLI path wires fetch -> build. Under --dry-run nothing is written
    (load_to_supabase is replaced here; its real dry-run guard is pinned by the
    loader's own tests)."""
    ids = [f"scvhcp_{i:05d}" for i in range(5000)]
    client = _FakeClient(ids)
    captured = {}
    monkeypatch.setattr(load_mod, "_read_only_supabase_client", lambda: client)
    monkeypatch.setattr(load_mod, "FULL_SIZES", _TINY_SIZES)

    def _fake_load(datasets, dry_run=False, verbose=False):
        captured["datasets"] = datasets
        captured["dry_run"] = dry_run
        return {}

    monkeypatch.setattr(load_mod, "load_to_supabase", _fake_load)
    monkeypatch.setattr("sys.argv", ["load_synthetic_data.py", "--refresh-ab", "--dry-run"])
    load_mod.main()
    assert captured["dry_run"] is True
    asn = captured["datasets"]["ab_experiment_assignments"]
    assert asn["unit_id"].isin(ids).all()
    assert [r for kind, r in client.log if kind == "range"][0] == (0, 999)


def test_refresh_ab_cli_fails_loud_when_universe_unreadable(monkeypatch):
    def _boom():
        raise RuntimeError("no SUPABASE_URL")

    monkeypatch.setattr(load_mod, "_read_only_supabase_client", _boom)
    called = []
    monkeypatch.setattr(load_mod, "load_to_supabase", lambda *a, **k: called.append(1) or {})
    monkeypatch.setattr("sys.argv", ["load_synthetic_data.py", "--refresh-ab", "--dry-run"])
    assert load_mod.main() == 1  # main() logs the exception and returns 1
    assert not called, "must not reach the sink with fabricated ids"


def test_fetch_is_scoped_to_the_tag_namespace_when_namespaces_coexist():
    """codex r1 MED: the full load samples from THIS run's tagged hcp_profiles
    frame; the refresh must sample from the same namespace, not every synthetic
    HCP that happens to exist. A coexisting 'xyz' namespace is excluded, and
    the exact-count check is evaluated on the scoped universe."""
    scv = [f"scvhcp_{i:05d}" for i in range(2000)]
    xyz = [f"xyzhcp_{i:05d}" for i in range(2000)]
    # codex r2 MED: LIKE's '_' is a one-char wildcard, so a server-side
    # 'scvhcp_%' ALSO returns this adjacent shape — it must not reach the panel.
    adjacent = [f"scvhcpX{i:05d}" for i in range(50)]
    client = _FakeClient(sorted(scv + xyz + adjacent))
    got = load_mod.fetch_synthetic_hcp_ids(client, id_prefix="scv", page_size=1000)
    assert got == scv
    with pytest.raises(ValueError, match="hcp_profiles"):
        load_mod.fetch_synthetic_hcp_ids(_FakeClient(xyz), id_prefix="scv")


@pytest.mark.parametrize("tag", ["s%v", "s_v", "s*v", "s\\v", ""])
def test_fetch_rejects_wildcard_bearing_or_empty_tags(tag):
    """A --tag carrying a LIKE metacharacter would widen the server-side read;
    refuse it rather than escape-and-hope (codex r2 MED)."""
    with pytest.raises(ValueError, match="tag"):
        load_mod.fetch_synthetic_hcp_ids(
            _FakeClient([f"scvhcp_{i:05d}" for i in range(5)]), id_prefix=tag
        )


def test_refresh_ab_cli_passes_the_tag_to_the_fetch(monkeypatch):
    seen = {}

    def _fake_fetch(client, id_prefix="scv", **kw):
        seen["prefix"] = id_prefix
        return [f"{id_prefix}hcp_{i:05d}" for i in range(5000)]

    monkeypatch.setattr(load_mod, "_read_only_supabase_client", lambda: object())
    monkeypatch.setattr(load_mod, "fetch_synthetic_hcp_ids", _fake_fetch)
    monkeypatch.setattr(load_mod, "FULL_SIZES", _TINY_SIZES)
    captured = {}
    monkeypatch.setattr(
        load_mod,
        "load_to_supabase",
        lambda d, dry_run=False, verbose=False: captured.update(d) or {},
    )
    monkeypatch.setattr(
        "sys.argv", ["load_synthetic_data.py", "--refresh-ab", "--dry-run", "--tag", "abc"]
    )
    load_mod.main()
    assert seen["prefix"] == "abc"
    assert captured["ab_experiment_assignments"]["unit_id"].str.startswith("abchcp_").all()
