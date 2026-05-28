"""Pin the CSU edge-augmented golden fixture (Layer-4 Track-2B-v3 Phase 1).

``tests/fixtures/causal_role_golden_set_csu_edges.json`` carries, for the 31
``CSU_remibrutinib`` literature features, a domain-authored DAG (``edges`` +
``feature_node``/``treatment_node``/``outcome_node``) authored BLIND to the
independent literature ``ground_truth_role`` (which is preserved verbatim for
scoring). These tests pin the fixture's shape AND the leak-vs-accept decision
gate it was built to demonstrate: ``missed_leaks == 0`` — no LEAK feature
(mediator/collider/descendant) is ever placed in ACCEPT
(ancestor/confounder/instrument) by the deterministic ``extract_role`` decider.

Non-circularity is made MECHANICALLY AUDITABLE (not merely asserted): the
scoring fixture is REBUILT here from a committed LABEL-FREE / ROLE-FREE blind
authoring source (``causal_role_csu_blind_authored_edges.json``) joined with the
INDEPENDENT labels in ``causal_role_golden_set.json`` via
``build_edge_augmented_fixture``. The labels therefore enter only from the
independent golden set, never from the authored payload — a p-hacked edge file
that encoded the labels would be caught by the label-free-source guard
(``test_blind_source_is_label_free``) and the byte-identical rebuild
(``test_scoring_fixture_is_rebuilt_from_blind_source``).

The decider remains DARK in production; this fixture is the labeled CSU-cohort
acceptance evidence that gates a future production-authoring phase.
"""

import importlib.util
import json
import sys
from pathlib import Path

import networkx as nx

from src.ml.causal_role_dgp.extractor import extract_role

_REPO = Path(__file__).resolve().parents[3]
_FIXTURE = _REPO / "tests" / "fixtures" / "causal_role_golden_set_csu_edges.json"
_BLIND_SOURCE = _REPO / "tests" / "fixtures" / "causal_role_csu_blind_authored_edges.json"
_GOLDEN = _REPO / "tests" / "fixtures" / "causal_role_golden_set.json"

_VALID_ROLES = frozenset(
    {"ancestor", "confounder", "instrument", "mediator", "collider", "descendant"}
)
# Fields whose presence in the BLIND authoring source would mean the edges could
# have been reverse-engineered from the answer (circularity / p-hacking).
_LABEL_REVEALING_FIELDS = (
    "ground_truth_role",
    "true_role",
    "expected_role",
    "leak_bucket",
    "leak_bucket_true",
    "provenance",
)


def _load_fixture() -> dict:
    data: dict = json.loads(_FIXTURE.read_text())
    return data


def _load_trackb_module():
    spec = importlib.util.spec_from_file_location(
        "layer4_trackb_authoring", _REPO / "scripts" / "layer4_trackb_authoring.py"
    )
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _load_measure_module():
    spec = importlib.util.spec_from_file_location(
        "measure_layer4_precision", _REPO / "scripts" / "measure_layer4_precision.py"
    )
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    # Register before exec: module-level @dataclass resolves itself in sys.modules.
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_fixture_has_31_csu_entries():
    data = _load_fixture()
    entries = data["entries"]
    assert data["total_entries"] == 31
    assert len(entries) == 31
    assert all(e["cohort"] == "CSU_remibrutinib" for e in entries)


def test_every_entry_has_structural_keys_and_anchors():
    entries = _load_fixture()["entries"]
    for e in entries:
        # required structural keys for --decider structural scoring
        assert e["feature_node"] == e["feature_name"], e["feature_name"]
        assert e["treatment_node"] == "T", e["feature_name"]
        assert e["outcome_node"] == "Y", e["feature_name"]
        assert e["edges"], f"empty edges: {e['feature_name']}"
        # both T and Y must appear as nodes (the cohort frame anchors); otherwise
        # extract_role raises on nx.descendants(graph, missing_node).
        nodes = {n for edge in e["edges"] for n in edge}
        assert "T" in nodes and "Y" in nodes, e["feature_name"]
        # independent literature label preserved + valid
        assert e["ground_truth_role"] in _VALID_ROLES, e["feature_name"]


def test_no_prose_leaks_into_fixture():
    # Non-circularity hygiene: the blind authoring rationale/provenance must NOT
    # be carried into the scoring fixture (only the independent label + edges).
    entries = _load_fixture()["entries"]
    for e in entries:
        assert "rationale" not in e, e["feature_name"]
        assert "provenance" not in e, e["feature_name"]


def test_label_distribution_matches_literature_cohort():
    # The 31-entry CSU literature cohort distribution (independent labels).
    from collections import Counter

    entries = _load_fixture()["entries"]
    dist = Counter(e["ground_truth_role"] for e in entries)
    assert dist == {
        "confounder": 6,
        "instrument": 6,
        "ancestor": 5,
        "collider": 5,
        "mediator": 5,
        "descendant": 4,
    }


def test_every_dag_is_classifiable_by_extract_role():
    # Each authored DAG must yield a valid role (no unclassifiable/raise) via the
    # SAME path the eval uses (_structural_predict).
    mod = _load_measure_module()
    entries = _load_fixture()["entries"]
    for e in entries:
        role = mod._structural_predict(e)
        assert role in _VALID_ROLES, f"{e['feature_name']} -> {role}"


def test_leak_decision_gate_zero_missed_leaks():
    """THE GATE: scoring the CSU fixture yields zero missed leaks.

    A missed leak = a feature whose independent label is a LEAK role
    (mediator/collider/descendant) but whose authored DAG derives an ACCEPT role
    (ancestor/confounder/instrument). This is the safety-critical failure mode the
    structural decider must never exhibit before activation.
    """
    mod = _load_measure_module()
    entries = _load_fixture()["entries"]

    cm = mod.CohortMetrics(cohort="CSU_remibrutinib", gate="ungated")
    for e in entries:
        predicted = mod._structural_predict(e)
        truth = str(e["ground_truth_role"])
        cm.confusion.setdefault(truth, {})
        cm.confusion[truth][predicted] = cm.confusion[truth].get(predicted, 0) + 1

    metrics = mod._leak_decision_metrics({("CSU_remibrutinib", "ungated"): cm})
    assert metrics["missed_leaks"] == 0, metrics
    assert metrics["false_alarms"] == 0, metrics
    assert metrics["decided"] == 31, metrics
    assert metrics["leak_decision_accuracy"] == 1.0, metrics


# --- Non-circularity made mechanically auditable (codex iter-0 HIGH) ---


def test_blind_source_is_label_free():
    """The blind authoring source must carry NO label/role-revealing field.

    If the edges had been reverse-engineered from ``ground_truth_role`` (the
    p-hacking / circularity failure mode this acceptance gate exists to rule
    out), the source would need to know the answer. This guard fails if any
    label/role field appears in the source — at the key OR value level.
    """
    src = json.loads(_BLIND_SOURCE.read_text())
    assert src["total_entries"] == 31
    # Scope the guard to the ENTRY payloads (the actual circularity surface) — not
    # the document's own ``description``/``protocol`` prose, which legitimately
    # NAMES the label fields it excludes when describing the non-circular method.
    entries_blob = json.dumps(src["entries"])
    for field in _LABEL_REVEALING_FIELDS:
        assert field not in entries_blob, f"blind source entries leak label field {field!r}"
    # No role-name STRING may appear anywhere in the entry payloads (a derived/true
    # role smuggled into a key, value, or prose) — the rebuild source must not know
    # the answer. (The six role names are the answer space.)
    for role in _VALID_ROLES:
        assert role not in entries_blob, f"blind source entries leak role string {role!r}"
    for e in src["entries"]:
        assert e["feature_node"] == e["feature_name"]
        assert e["treatment_node"] == "T" and e["outcome_node"] == "Y"
        assert e["edges"], e["feature_name"]
        for field in _LABEL_REVEALING_FIELDS:
            assert field not in e, f"{e['feature_name']} leaks {field!r}"


def test_scoring_fixture_is_rebuilt_from_blind_source():
    """The scoring fixture is the label-free blind source JOINED with the
    INDEPENDENT golden-set labels — rebuilt here and required byte-identical.

    ``build_edge_augmented_fixture`` pulls ``ground_truth_role``/``cohort`` from
    the golden set, never from the authored payload, so the labels enter only
    from the independent source. A byte-identical rebuild proves the committed
    fixture is exactly (blind edges) + (independent labels) with nothing in
    between — non-circular by construction, not by claim.
    """
    trackb = _load_trackb_module()
    blind = json.loads(_BLIND_SOURCE.read_text())["entries"]
    golden = json.loads(_GOLDEN.read_text())["entries"]
    csu_golden = [e for e in golden if e.get("cohort") == "CSU_remibrutinib"]

    authored = {
        e["feature_name"]: {
            k: e[k] for k in ("feature_node", "treatment_node", "outcome_node", "edges")
        }
        for e in blind
    }
    rebuilt = trackb.build_edge_augmented_fixture(
        csu_golden, authored, fixture_kind="csu_remibrutinib_heldout_edges"
    )
    rebuilt_text = json.dumps(rebuilt, indent=2, sort_keys=True) + "\n"
    assert rebuilt_text == _FIXTURE.read_text(), (
        "committed scoring fixture is NOT a byte-identical rebuild of "
        "(blind authored edges + independent golden labels) — the fixture may "
        "have been hand-edited away from the auditable join."
    )


def test_fixture_labels_match_independent_golden_set_exactly():
    # Every label in the scoring fixture is byte-identical to the ORIGINAL
    # literature golden set (no label tampering during edge augmentation).
    golden = {
        e["feature_name"]: e["ground_truth_role"]
        for e in json.loads(_GOLDEN.read_text())["entries"]
        if e.get("cohort") == "CSU_remibrutinib"
    }
    fixture = {e["feature_name"]: e["ground_truth_role"] for e in _load_fixture()["entries"]}
    assert set(fixture) == set(golden)
    assert all(fixture[f] == golden[f] for f in golden), "label drift vs golden set"


# --- T->Y is a load-bearing causal assumption, not just an anchor (codex iter-0 MEDIUM) ---


def test_t_to_y_edge_present_in_every_dag():
    # Every authored DAG asserts the true treatment-effect edge T->Y (remibrutinib
    # does reduce UAS7 — the studied effect). It is a required causal assumption,
    # not merely a node-presence workaround.
    for e in _load_fixture()["entries"]:
        edge_set = {tuple(edge) for edge in e["edges"]}
        assert ("T", "Y") in edge_set, f"{e['feature_name']} missing the T->Y effect edge"


def test_t_to_y_is_load_bearing_for_outcome_echo_descendants():
    """For pure ``Y -> feature`` outcome echoes (DLQI, UCT), the T->Y edge is
    REQUIRED to classify them ``descendant``; without it (T isolated) the node is
    unrelated to T and ``extract_role`` raises (-> route to review).

    This documents, as a regression, that T->Y is a genuine causal assumption
    these two leak classifications depend on — not a no-op anchor. (Removing it
    is conservative: descendant/LEAK -> review, never a missed leak.)
    """
    pure_echoes = ["post_index_dlqi_score_150_180d", "uct_score_150_180d_post_index"]
    by_name = {e["feature_name"]: e for e in _load_fixture()["entries"]}
    for fname in pure_echoes:
        e = by_name[fname]
        # with T->Y (as authored): descendant
        g_with = nx.DiGraph([tuple(edge) for edge in e["edges"]])
        assert extract_role(fname, "T", "Y", g_with) == "descendant"
        # without T->Y but T kept as an isolated node: unclassifiable (raises)
        g_without = nx.DiGraph([tuple(edge) for edge in e["edges"] if tuple(edge) != ("T", "Y")])
        g_without.add_node("T")
        raised = False
        try:
            extract_role(fname, "T", "Y", g_without)
        except ValueError:
            raised = True
        assert raised, f"{fname}: expected unclassifiable without T->Y"
