"""Pin the CSU edge-augmented golden fixture (Layer-4 Track-2B-v3 Phase 1).

``tests/fixtures/causal_role_golden_set_csu_edges.json`` carries, for the 31
``CSU_remibrutinib`` literature features, a domain-authored DAG (``edges`` +
``feature_node``/``treatment_node``/``outcome_node``) authored BLIND to the
independent literature ``ground_truth_role`` (which is preserved verbatim for
scoring). These tests pin the fixture's shape AND the leak-vs-accept decision
gate it was built to demonstrate: ``missed_leaks == 0`` — no LEAK feature
(mediator/collider/descendant) is ever placed in ACCEPT
(ancestor/confounder/instrument) by the deterministic ``extract_role`` decider.

Non-circularity rests on TWO committed provenance artifacts plus the
author-once/score-once protocol — NOT on artifact tests alone:

1. ``causal_role_csu_blind_briefs.json`` — the EXACT label-free inputs the DAG
   author was given (feature_name + derivation_pseudocode + dataset_context),
   verified to match the golden set's inputs and to contain no label/role string
   (``test_blind_briefs_*``).
2. ``causal_role_csu_blind_authored_edges.json`` — the STRUCTURAL-ONLY authored
   edges (no role string, no label, no rationale; ``test_blind_source_is_label_free``).
The scoring fixture is the byte-identical join of (2) with the INDEPENDENT golden
labels via ``build_edge_augmented_fixture`` (``test_scoring_fixture_is_rebuilt_
from_blind_source``), so labels enter only from the independent golden set.

HONEST EPISTEMIC BOUNDARY (codex iter-1, accepted): the artifact tests are
NECESSARY, NOT SUFFICIENT. They prove (a) the authoring inputs were label-free,
(b) the edge source carries no role/label string, and (c) the labels were joined
independently. They CANNOT, by inspecting the resulting JSON, prove the edge
*topology* was not reverse-engineered from the labels — a label-derived author
could emit the canonical shapes (feature->Y for ancestor, feature->T + feature->Y
for confounder, ...) using no forbidden string. ``test_negative_control_label_
derived_source_still_passes_string_guards`` documents exactly that gap. The
guarantee that the topology is genuinely blind rests on the committed blind-input
provenance (1) + the author-once/score-once protocol, which this artifact set
makes auditable but does not, on its own, mechanically enforce.

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
_BLIND_BRIEFS = _REPO / "tests" / "fixtures" / "causal_role_csu_blind_briefs.json"
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


# --- Authoring-process provenance + honest negative control (codex iter-1 HIGH) ---


def test_blind_briefs_are_label_free_and_match_golden_inputs():
    """The committed blind brief is EXACTLY the label-free inputs the author saw.

    Provenance of the authoring PROCESS (not just the resulting JSON): the brief
    carries only feature_name + derivation_pseudocode + dataset_context, contains
    no label/role string, and its per-feature inputs are byte-identical to the
    golden set's — i.e. the author worked from the real mechanism metadata, never
    the answer.
    """
    doc = json.loads(_BLIND_BRIEFS.read_text())
    briefs = {b["feature_name"]: b for b in doc["briefs"]}
    assert doc["total_entries"] == 31 and len(briefs) == 31

    blob = json.dumps(doc["briefs"])
    for field in _LABEL_REVEALING_FIELDS + ("rationale",):
        assert field not in blob, f"blind brief leaks {field!r}"
    for role in _VALID_ROLES:
        assert role not in blob, f"blind brief leaks role string {role!r}"

    golden = {
        e["feature_name"]: e
        for e in json.loads(_GOLDEN.read_text())["entries"]
        if e.get("cohort") == "CSU_remibrutinib"
    }
    assert set(briefs) == set(golden)
    for f, g in golden.items():
        assert set(briefs[f]) == {"feature_name", "derivation_pseudocode", "dataset_context"}
        assert briefs[f]["derivation_pseudocode"] == g["derivation_pseudocode"], f
        assert briefs[f]["dataset_context"] == g["dataset_context"], f


def test_blind_source_features_match_briefs():
    # The authored edge source covers exactly the features in the blind brief —
    # the author authored the briefed set, not a hand-picked subset.
    briefs = {b["feature_name"] for b in json.loads(_BLIND_BRIEFS.read_text())["briefs"]}
    src = {e["feature_name"] for e in json.loads(_BLIND_SOURCE.read_text())["entries"]}
    assert src == briefs


def test_negative_control_label_derived_source_still_passes_string_guards():
    """HONEST NEGATIVE CONTROL (codex iter-1): the string/rebuild guards are
    necessary but NOT sufficient to prove blind authorship.

    A source authored *from the labels* — emitting each role's canonical topology
    — uses none of the forbidden role/label strings, so it passes
    ``test_blind_source_is_label_free`` and would rebuild a valid scoring fixture.
    This test asserts that gap explicitly so no reviewer mistakes the artifact
    tests for a proof of blindness. The blindness guarantee rests on the committed
    blind-brief provenance + the author-once/score-once protocol, not on these
    guards.
    """
    canonical = {  # role -> canonical edge topology (the label IS the topology)
        "ancestor": [["f", "Y"], ["T", "Y"]],
        "confounder": [["f", "T"], ["f", "Y"], ["T", "Y"]],
        "instrument": [["f", "T"], ["T", "Y"]],
        "mediator": [["T", "f"], ["f", "Y"], ["T", "Y"]],
        "collider": [["T", "f"], ["Y", "f"], ["T", "Y"]],
        "descendant": [["Y", "f"], ["T", "Y"]],
    }
    # NEUTRAL feature names decoupled from the role (do NOT name a feature after
    # its role, or the name itself would leak it — the point is a source that
    # encodes the answer ONLY in the topology, not in any string).
    fnames = {role: f"nc_feature_{i}" for i, role in enumerate(canonical)}
    label_derived_entries = [
        {
            "feature_name": fnames[role],
            "feature_node": fnames[role],
            "treatment_node": "T",
            "outcome_node": "Y",
            "edges": [
                [a if a != "f" else fnames[role], b if b != "f" else fnames[role]] for a, b in edges
            ],
            "ambiguous": False,
        }
        for role, edges in canonical.items()
    ]
    entries_blob = json.dumps(label_derived_entries)
    # It passes the same string guards the real blind source passes:
    for field in _LABEL_REVEALING_FIELDS:
        assert field not in entries_blob
    for role in _VALID_ROLES:
        assert role not in entries_blob
    # ...yet each topology trivially re-derives its source role — i.e. the label
    # is encoded in the shape, which the string guards cannot detect. This is the
    # documented necessary-not-sufficient boundary; provenance covers it.
    for role, edges in canonical.items():
        fnode = fnames[role]
        norm = [[a if a != "f" else fnode, b if b != "f" else fnode] for a, b in edges]
        derived = extract_role(fnode, "T", "Y", nx.DiGraph([tuple(e) for e in norm]))
        assert derived == role, f"canonical topology for {role} -> {derived}"
