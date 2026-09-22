"""Structural author: a DSPy program that draws one feature's DAG fragment.

Lane B of the real-data causal estimation program (spec
``docs/superpowers/specs/2026-09-22-real-data-causal-estimation-design.md``
§3 Lane B item 1). The program's instructions are sections 0–6 of the authoring
guide ``docs/layer4/structural_attestation_authoring.md`` VERBATIM
(``src.data.kg._structural_author_guide``, pinned to the guide by a test).

Per feature the author receives the brief the Layer-4 classifier already gets
(``derivation_pseudocode`` + ``dataset_context`` from
``_build_layer_4_inputs``, with the causal treatment appended) plus the Lane E
feature-role panel record for that feature, and returns edges over
``{feature, T, Y, U_*}``, a cited rationale per non-obvious edge, ``ambiguous``
and the role it expects. Post-processing (all deterministic, LLM-free):

* :func:`parse_author_output` validates the fragment (allowed nodes, no cycles,
  the ``T -> Y`` estimand edge present, the feature present);
* :func:`extract_role` derives the role — the extractor is authoritative, the
  author's ``expected_role`` is a cross-check only;
* :func:`grade_edges` sends every citation through
  ``CitationResolver.verify_citation`` and grades each non-estimand edge
  ``direct`` / ``family`` / ``unsupported``;
* Lane E hard constraints (spec §3 Lane E item 3): a Layer-1 post-index
  verdict forbids ``feature -> T`` (violation → review); the derived role is
  cross-checked against the panel's ensemble ``final_role`` (disagreement →
  ``ambiguous``); a leak verdict is carried on the record so the assembler
  keeps the feature out of every adjustment set;
* every record is stamped with the model id, the prompt hash and the guide
  hash, and carries ``provenance="machine"`` — audit-only until a human
  approves it in the expert-review queue (``CausalStructureAttestation``).

Author failures (LM error, unparseable output, unclassifiable DAG) route the
feature to review with the reason recorded; nothing is fabricated (spec §5).
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal, Mapping, Optional, Sequence

import networkx as nx

from src.data.feature_contract import CausalStructureAttestation
from src.data.kg._structural_author_guide import (
    GUIDE_FIRST_HEADING,
    GUIDE_HASH,
    GUIDE_PATH,
    GUIDE_SECTIONS_0_TO_6,
)
from src.ml.causal_role_dgp.extractor import extract_role

logger = logging.getLogger(__name__)

__all__ = [
    "AUTHOR_SCHEMA_VERSION",
    "GUIDE_FIRST_HEADING",
    "GUIDE_HASH",
    "GUIDE_PATH",
    "GUIDE_SECTIONS_0_TO_6",
    "EDGE_GRADES",
    "ROLES",
    "TREATMENT_NODE",
    "OUTCOME_NODE",
    "AuthoringBrief",
    "AuthoredAttestation",
    "EdgeProvenance",
    "PanelRecordView",
    "ParsedFragment",
    "StructuralAuthor",
    "StructuralAuthorError",
    "author_feature",
    "build_brief",
    "grade_edges",
    "parse_author_output",
    "prompt_hash",
]

AUTHOR_SCHEMA_VERSION = "1"

#: Node labels the guide prescribes (§1): the treatment and outcome anchors.
TREATMENT_NODE = "T"
OUTCOME_NODE = "Y"
#: Latent nodes are ``U_<something>`` (§1, convention 2).
_LATENT_RE = re.compile(r"^U_[A-Za-z0-9_]+$")

ROLES: tuple[str, ...] = (
    "ancestor",
    "confounder",
    "instrument",
    "mediator",
    "collider",
    "descendant",
)
EDGE_GRADES: tuple[str, ...] = ("direct", "family", "unsupported")

# Citation-verdict thresholds, read from the resolver's own weights so the grade
# and the resolver cannot drift apart (``CitationResolver.verify_citation``:
# 0.5 both entities, +0.3 causal cue, +0.2 co-occurrence).
_PMID_RE = re.compile(r"^(?:pmid:?\s*)?(\d{1,9})$", re.IGNORECASE)
_DOI_RE = re.compile(r"^(?:doi:?\s*)?(10\.\d{4,9}/\S+)$", re.IGNORECASE)
_PUBMED_URL_RE = re.compile(r"pubmed\.ncbi\.nlm\.nih\.gov/(\d{1,9})", re.IGNORECASE)
_DOI_URL_RE = re.compile(r"doi\.org/(10\.\d{4,9}/\S+)$", re.IGNORECASE)


class StructuralAuthorError(ValueError):
    """The author's output cannot be turned into a fragment (routes to review)."""


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PanelRecordView:
    """The slice of a Lane E ``FeatureRoleRecord`` the author and its post-
    processing consume. Field names follow ``src.causal_engine.feature_role_panel``
    (Lane E, PR #2226) so the real record's ``to_dict()`` loads unchanged; when
    that branch is not on the tree this adapter is the typed contract.
    """

    feature: str
    layer_1_verdict: Optional[str] = None  # pre_index | post_index | no_contract
    layer_1_temporal_status: Optional[str] = None
    layer_3_ran: bool = False
    layer_3_severity: Optional[str] = None
    layer_3_z_score: Optional[float] = None
    layer_2_signal: Optional[str] = None
    layer_4_role: Optional[str] = None
    layer_4_mechanism: Optional[str] = None
    ensemble_decided_by: Optional[str] = None
    ensemble_final_role: Optional[str] = None
    ensemble_confidence: Optional[float] = None
    leak_verdict: bool = False
    leak_source: Optional[str] = None
    review_required: bool = False

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> "PanelRecordView":
        l1 = dict(record.get("layer_1") or {})
        l2 = dict(record.get("layer_2") or {})
        l3 = dict(record.get("layer_3") or {})
        l4 = dict(record.get("layer_4") or {})
        ens = dict(record.get("ensemble") or {})
        signal = l2.get("signal")
        return cls(
            feature=str(record["feature"]),
            layer_1_verdict=l1.get("verdict"),
            layer_1_temporal_status=l1.get("temporal_status"),
            layer_3_ran=bool(l3.get("ran", False)),
            layer_3_severity=l3.get("severity_pre_joint_check"),
            layer_3_z_score=l3.get("z_score"),
            layer_2_signal=str(signal) if signal is not None else None,
            layer_4_role=l4.get("role"),
            layer_4_mechanism=l4.get("mechanism"),
            ensemble_decided_by=ens.get("decided_by"),
            ensemble_final_role=ens.get("final_role"),
            ensemble_confidence=ens.get("confidence"),
            leak_verdict=bool(record.get("leak_verdict", False)),
            leak_source=record.get("leak_source"),
            review_required=bool(record.get("review_required", False)),
        )

    def brief_text(self) -> str:
        """The panel as the author reads it (one line per layer; ``None`` spelt out)."""
        lines = [
            f"layer_1: verdict={self.layer_1_verdict}; temporal_status={self.layer_1_temporal_status}",
            f"layer_2: kg_signal={self.layer_2_signal}",
            (
                f"layer_3: ran={self.layer_3_ran}; severity={self.layer_3_severity}; "
                f"z_score={self.layer_3_z_score}"
            ),
            f"layer_4: role={self.layer_4_role}; mechanism={self.layer_4_mechanism}",
            (
                f"ensemble: decided_by={self.ensemble_decided_by}; "
                f"final_role={self.ensemble_final_role}; confidence={self.ensemble_confidence}"
            ),
            (
                f"leak_verdict={self.leak_verdict}; leak_source={self.leak_source}; "
                f"review_required={self.review_required}"
            ),
        ]
        return "\n".join(lines)


@dataclass(frozen=True)
class AuthoringBrief:
    """Everything the author sees for one feature (label-free by construction)."""

    feature_name: str
    derivation_pseudocode: str
    dataset_context: str
    treatment_label: str
    outcome_label: str
    panel: Optional[PanelRecordView] = None

    def panel_text(self) -> str:
        if self.panel is None:
            return "no feature-role panel record for this feature"
        return self.panel.brief_text()


def build_brief(
    feature_name: str,
    *,
    derivation_pseudocode: str,
    dataset_context: str,
    treatment_label: str,
    outcome_label: str,
    panel_record: Optional[Mapping[str, Any]] = None,
) -> AuthoringBrief:
    """Assemble the brief from the Layer-4 inputs and an optional panel record.

    The ``(derivation_pseudocode, dataset_context)`` pair is the one
    ``adaptive_validity_check._build_layer_4_inputs`` builds (callers pass it
    through; this module does not import the node so its import stays light).
    """
    panel = PanelRecordView.from_record(panel_record) if panel_record is not None else None
    if panel is not None and panel.feature != feature_name:
        raise StructuralAuthorError(
            f"panel record is for {panel.feature!r}, brief is for {feature_name!r}"
        )
    return AuthoringBrief(
        feature_name=feature_name,
        derivation_pseudocode=derivation_pseudocode,
        dataset_context=dataset_context,
        treatment_label=treatment_label,
        outcome_label=outcome_label,
        panel=panel,
    )


# ---------------------------------------------------------------------------
# The DSPy program
# ---------------------------------------------------------------------------

try:  # dspy is a heavy optional import; the parser/grader work without it.
    import dspy
    from pydantic import BaseModel as _PydanticBaseModel

    class EdgeRationale(_PydanticBaseModel):
        """One authored edge with its cited rationale (guide §6)."""

        from_node: str
        to_node: str
        rationale: str
        citations: list[str] = []

    class StructuralAttestationSignature(dspy.Signature):
        __doc__ = GUIDE_SECTIONS_0_TO_6

        feature_name: str = dspy.InputField(desc="The feature node you are classifying.")
        derivation_pseudocode: str = dspy.InputField(
            desc="How the feature is derived (source, inputs, aggregation, window, knowable_at)."
        )
        dataset_context: str = dspy.InputField(
            desc="cohort; target; prediction anchor; treatment; the causal question."
        )
        treatment_label: str = dspy.InputField(desc="What the node T stands for.")
        outcome_label: str = dspy.InputField(desc="What the node Y stands for.")
        feature_role_panel: str = dspy.InputField(
            desc=(
                "What the four feature-role voters (Layer 1 contract, Layer 2 knowledge "
                "graph, Layer 3 adversarial probe, Layer 4 LLM) and their ensemble said "
                "about this feature. Evidence to weigh, not a role to target."
            )
        )
        edges: list[list[str]] = dspy.OutputField(
            desc=(
                'The arrows you drew as [from, to] pairs over the nodes "T", "Y", the '
                'feature name and optional latents "U_<name>". Always include ["T", "Y"].'
            )
        )
        edge_rationales: list[EdgeRationale] = dspy.OutputField(
            desc=(
                "One entry per NON-OBVIOUS edge: from_node, to_node, rationale, and "
                "citations as PMIDs, DOIs or NCT ids."
            )
        )
        entity_names: dict[str, str] = dspy.OutputField(
            desc=(
                "Clinical entity name for each node you used (e.g. the feature's "
                "concept, the latent's concept), used to verify the citations."
            )
        )
        expected_role: Literal[
            "ancestor", "confounder", "instrument", "mediator", "collider", "descendant"
        ] = dspy.OutputField(desc="The role you expect the extractor to derive (cross-check).")
        ambiguous: bool = dspy.OutputField(
            desc="true when a second equally-defensible mechanism gives a different role."
        )

    class StructuralAuthor(dspy.Module):
        """Chain-of-thought author over :class:`StructuralAttestationSignature`."""

        def __init__(self) -> None:
            super().__init__()
            self.author = dspy.ChainOfThought(StructuralAttestationSignature)

        def forward(self, brief: AuthoringBrief) -> "dspy.Prediction":
            return self.author(
                feature_name=brief.feature_name,
                derivation_pseudocode=brief.derivation_pseudocode,
                dataset_context=brief.dataset_context,
                treatment_label=brief.treatment_label,
                outcome_label=brief.outcome_label,
                feature_role_panel=brief.panel_text(),
            )

    _DSPY_AVAILABLE = True
except ImportError:  # pragma: no cover - dspy is installed on this platform
    _DSPY_AVAILABLE = False
    StructuralAuthor = None  # type: ignore[assignment,misc]
    StructuralAttestationSignature = None  # type: ignore[assignment,misc]


def prompt_hash() -> str:
    """sha256 over the signature's instructions and its field names/descriptions.

    Stamped on every record so a change to the prompt (guide text, a field
    description, the field set) is visible on the attestations it produced.
    """
    if not _DSPY_AVAILABLE:
        raise StructuralAuthorError("dspy is not installed; cannot hash the prompt")
    sig = StructuralAttestationSignature
    fields = []
    for name, f in sig.input_fields.items():
        fields.append(["input", name, str(f.json_schema_extra.get("desc", ""))])
    for name, f in sig.output_fields.items():
        fields.append(["output", name, str(f.json_schema_extra.get("desc", ""))])
    payload = json.dumps({"instructions": sig.instructions, "fields": fields}, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Parsing (deterministic, strict)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Citation:
    raw: str
    identifier: str
    kind: str  # pmid | doi | nct | url | unknown


@dataclass(frozen=True)
class ParsedFragment:
    feature_node: str
    edges: tuple[tuple[str, str], ...]
    rationales: dict[tuple[str, str], str]
    citations: dict[tuple[str, str], tuple[Citation, ...]]
    entity_names: dict[str, str]
    expected_role: Optional[str]
    ambiguous: bool
    latents: tuple[str, ...]


def parse_citation(raw: str) -> Citation:
    text = str(raw).strip()
    m = _PUBMED_URL_RE.search(text)
    if m:
        return Citation(raw=text, identifier=m.group(1), kind="pmid")
    m = _DOI_URL_RE.search(text)
    if m:
        return Citation(raw=text, identifier=m.group(1), kind="doi")
    m = _PMID_RE.match(text)
    if m:
        return Citation(raw=text, identifier=m.group(1), kind="pmid")
    m = _DOI_RE.match(text)
    if m:
        return Citation(raw=text, identifier=m.group(1), kind="doi")
    if re.match(r"^NCT\d{8}$", text, re.IGNORECASE):
        return Citation(raw=text, identifier=text.upper(), kind="nct")
    if text.lower().startswith(("http://", "https://")):
        return Citation(raw=text, identifier=text, kind="url")
    return Citation(raw=text, identifier=text, kind="unknown")


def _as_edge(item: Any) -> tuple[str, str]:
    if isinstance(item, Mapping):
        src, dst = item.get("from_node", item.get("from")), item.get("to_node", item.get("to"))
    else:
        try:
            src, dst = item
        except (TypeError, ValueError) as exc:
            raise StructuralAuthorError(f"edge {item!r} is not a [from, to] pair") from exc
    if not isinstance(src, str) or not isinstance(dst, str) or not src or not dst:
        raise StructuralAuthorError(f"edge {item!r} has a non-string endpoint")
    return src.strip(), dst.strip()


def parse_author_output(
    raw: Mapping[str, Any],
    *,
    feature_name: str,
    treatment_node: str = TREATMENT_NODE,
    outcome_node: str = OUTCOME_NODE,
) -> ParsedFragment:
    """Validate the author's fields into a :class:`ParsedFragment`.

    Raises :class:`StructuralAuthorError` (→ review) when the fragment is not a
    DAG over the allowed nodes, omits the ``T -> Y`` estimand edge (guide §1
    convention 6), or leaves the feature out of the diagram.
    """
    edges_raw = raw.get("edges")
    if isinstance(edges_raw, str):
        try:
            edges_raw = json.loads(edges_raw)
        except json.JSONDecodeError as exc:
            raise StructuralAuthorError(f"edges is not JSON: {exc}") from exc
    if not isinstance(edges_raw, Sequence) or isinstance(edges_raw, (str, bytes)):
        raise StructuralAuthorError("edges must be a list of [from, to] pairs")
    if not edges_raw:
        raise StructuralAuthorError("edges is empty")

    edges: list[tuple[str, str]] = []
    for item in edges_raw:
        e = _as_edge(item)
        if e[0] == e[1]:
            raise StructuralAuthorError(f"self-loop {e!r}")
        if e not in edges:
            edges.append(e)

    allowed_fixed = {feature_name, treatment_node, outcome_node}
    latents: list[str] = []
    for src, dst in edges:
        for node in (src, dst):
            if node in allowed_fixed:
                continue
            if _LATENT_RE.match(node):
                if node not in latents:
                    latents.append(node)
                continue
            raise StructuralAuthorError(
                f"node {node!r} is not the feature, {treatment_node!r}, {outcome_node!r} "
                f"or a latent 'U_<name>'"
            )
    if (treatment_node, outcome_node) not in edges:
        raise StructuralAuthorError(
            f"the estimand edge [{treatment_node!r}, {outcome_node!r}] is missing (guide §1.6)"
        )
    if not any(feature_name in e for e in edges):
        raise StructuralAuthorError(f"the feature {feature_name!r} appears in no edge")
    graph = nx.DiGraph(edges)
    if not nx.is_directed_acyclic_graph(graph):
        cycle = nx.find_cycle(graph)
        raise StructuralAuthorError(f"the fragment has a cycle: {cycle}")

    rationales: dict[tuple[str, str], str] = {}
    citations: dict[tuple[str, str], tuple[Citation, ...]] = {}
    rat_raw = raw.get("edge_rationales") or []
    if isinstance(rat_raw, str):
        try:
            rat_raw = json.loads(rat_raw)
        except json.JSONDecodeError as exc:
            raise StructuralAuthorError(f"edge_rationales is not JSON: {exc}") from exc
    for item in rat_raw:
        if hasattr(item, "model_dump"):
            item = item.model_dump()
        if not isinstance(item, Mapping):
            raise StructuralAuthorError(f"edge rationale {item!r} is not an object")
        e = _as_edge(item)
        if e not in edges:
            raise StructuralAuthorError(f"rationale cites an edge not in edges: {e!r}")
        rationales[e] = str(item.get("rationale", "")).strip()
        cits = item.get("citations") or []
        if isinstance(cits, str):
            cits = [c for c in re.split(r"[;,\s]+", cits) if c]
        citations[e] = tuple(parse_citation(c) for c in cits if str(c).strip())

    names_raw = raw.get("entity_names") or {}
    if isinstance(names_raw, str):
        try:
            names_raw = json.loads(names_raw)
        except json.JSONDecodeError:
            names_raw = {}
    entity_names = {str(k): str(v) for k, v in dict(names_raw).items() if str(v).strip()}

    expected = raw.get("expected_role")
    expected_role = str(expected).strip().lower() if expected is not None else None
    if expected_role is not None and expected_role not in ROLES:
        expected_role = None

    amb_raw = raw.get("ambiguous", False)
    if isinstance(amb_raw, str):
        ambiguous = amb_raw.strip().lower() in ("true", "1", "yes")
    else:
        ambiguous = bool(amb_raw)

    return ParsedFragment(
        feature_node=feature_name,
        edges=tuple(edges),
        rationales=rationales,
        citations=citations,
        entity_names=entity_names,
        expected_role=expected_role,
        ambiguous=ambiguous,
        latents=tuple(latents),
    )


# ---------------------------------------------------------------------------
# Grading (citations through CitationResolver.verify_citation)
# ---------------------------------------------------------------------------


@dataclass
class EdgeProvenance:
    """Per-edge provenance carried into the assembled DAG and the review."""

    from_node: str
    to_node: str
    grade: str  # direct | family | unsupported | estimand
    rationale: str = ""
    citations: list[dict[str, Any]] = field(default_factory=list)
    # Lane E constraint that voids the edge in the assembled structure, if any.
    constraint_violation: Optional[str] = None


def _verdict_to_dict(verdict: Any, *, raw: str, kind: str) -> dict[str, Any]:
    return {
        "raw": raw,
        "identifier": getattr(verdict, "identifier", None),
        "identifier_kind": getattr(verdict, "identifier_kind", kind),
        "abstract_resolved": bool(getattr(verdict, "abstract_resolved", False)),
        "entities_found": list(getattr(verdict, "entities_found", ()) or ()),
        "causal_cue_found": getattr(verdict, "causal_cue_found", None),
        "overall_confidence": float(getattr(verdict, "overall_confidence", 0.0) or 0.0),
        "error": getattr(verdict, "error", None),
    }


def _grade_from_verdicts(verdicts: Sequence[Mapping[str, Any]]) -> str:
    """direct: an abstract co-mentions both endpoint entities WITH a causal cue;
    family: both entities co-mentioned, no causal cue (association-level
    support); unsupported: no citation, unresolved, or an entity missing."""
    best = "unsupported"
    for v in verdicts:
        if not v.get("abstract_resolved"):
            continue
        found = len(v.get("entities_found") or [])
        if found >= 2 and v.get("causal_cue_found"):
            return "direct"
        if found >= 2:
            best = "family"
    return best


def grade_edges(
    fragment: ParsedFragment,
    *,
    resolver: Any,
    treatment_label: str,
    outcome_label: str,
    treatment_node: str = TREATMENT_NODE,
    outcome_node: str = OUTCOME_NODE,
) -> list[EdgeProvenance]:
    """Grade every authored edge.

    ``resolver`` is a ``CitationResolver`` (or anything with its
    ``verify_citation(identifier, *, identifier_kind, subject_name, object_name)``).
    The ``T -> Y`` edge is the estimand assumption (guide §1.6), graded
    ``estimand`` and never sent to the resolver. NCT ids and bare URLs cannot be
    verified by the resolver (PMID/DOI only) and count as ``unsupported`` with
    the reason recorded.
    """
    names = dict(fragment.entity_names)
    names.setdefault(treatment_node, treatment_label)
    names.setdefault(outcome_node, outcome_label)
    names.setdefault(fragment.feature_node, fragment.feature_node)
    out: list[EdgeProvenance] = []
    for src, dst in fragment.edges:
        rationale = fragment.rationales.get((src, dst), "")
        if (src, dst) == (treatment_node, outcome_node):
            out.append(
                EdgeProvenance(
                    from_node=src,
                    to_node=dst,
                    grade="estimand",
                    rationale=rationale or "treatment-effect edge (guide §1, convention 6)",
                )
            )
            continue
        verdicts: list[dict[str, Any]] = []
        for cit in fragment.citations.get((src, dst), ()):
            if cit.kind not in ("pmid", "doi"):
                verdicts.append(
                    {
                        "raw": cit.raw,
                        "identifier": cit.identifier,
                        "identifier_kind": cit.kind,
                        "abstract_resolved": False,
                        "entities_found": [],
                        "causal_cue_found": None,
                        "overall_confidence": 0.0,
                        "error": f"unverifiable citation kind {cit.kind!r} (resolver takes pmid/doi)",
                    }
                )
                continue
            try:
                verdict = resolver.verify_citation(
                    cit.identifier,
                    identifier_kind=cit.kind,
                    subject_name=names.get(src, src),
                    object_name=names.get(dst, dst),
                )
            except Exception as exc:  # noqa: BLE001 — a resolver outage is a recorded verdict
                verdicts.append(
                    {
                        "raw": cit.raw,
                        "identifier": cit.identifier,
                        "identifier_kind": cit.kind,
                        "abstract_resolved": False,
                        "entities_found": [],
                        "causal_cue_found": None,
                        "overall_confidence": 0.0,
                        "error": f"resolver raised: {exc}",
                    }
                )
                continue
            verdicts.append(_verdict_to_dict(verdict, raw=cit.raw, kind=cit.kind))
        out.append(
            EdgeProvenance(
                from_node=src,
                to_node=dst,
                grade=_grade_from_verdicts(verdicts),
                rationale=rationale,
                citations=verdicts,
            )
        )
    return out


# ---------------------------------------------------------------------------
# The authored record
# ---------------------------------------------------------------------------

_LAYER_1_POST_INDEX = "post_index"
_CONSTRAINT_L1 = "layer_1_post_index_forbids_feature_to_T"


@dataclass
class AuthoredAttestation:
    """One feature's authored fragment plus everything the review needs."""

    feature_name: str
    treatment_node: str
    outcome_node: str
    feature_node: str
    edges: list[list[str]]
    edge_provenance: list[EdgeProvenance]
    derived_role: Optional[str]
    expected_role: Optional[str]
    ambiguous: bool
    review_required: bool
    review_reasons: list[str]
    cross_check: dict[str, Any]
    constraint_violations: list[str]
    panel_summary: dict[str, Any]
    latents: list[str]
    provenance: str
    model_id: str
    prompt_hash: str
    guide_hash: str
    authored_at: str
    author_schema_version: str = AUTHOR_SCHEMA_VERSION
    reasoning: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        return d

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "AuthoredAttestation":
        data = dict(payload)
        data["edge_provenance"] = [
            EdgeProvenance(**dict(e)) if not isinstance(e, EdgeProvenance) else e
            for e in data.get("edge_provenance", [])
        ]
        data["edges"] = [list(e) for e in data.get("edges", [])]
        return cls(**data)

    def to_attestation(self) -> Optional[CausalStructureAttestation]:
        """The ``FeatureContract`` attestation shape (``provenance`` carried).

        ``None`` when the author produced no usable fragment (review only).
        """
        if not self.edges:
            return None
        return CausalStructureAttestation(
            treatment_node=self.treatment_node,
            outcome_node=self.outcome_node,
            feature_node=self.feature_node,
            edges=tuple((e[0], e[1]) for e in self.edges),
            provenance=self.provenance,
        )


def _model_id(lm: Any) -> str:
    if lm is None:
        return "none"
    model = getattr(lm, "model", None)
    if model:
        return str(model)
    return type(lm).__name__


def _panel_summary(panel: Optional[PanelRecordView]) -> dict[str, Any]:
    if panel is None:
        return {"present": False}
    return {
        "present": True,
        "layer_1_verdict": panel.layer_1_verdict,
        "layer_3_severity": panel.layer_3_severity,
        "layer_2_signal": panel.layer_2_signal,
        "layer_4_role": panel.layer_4_role,
        "ensemble_decided_by": panel.ensemble_decided_by,
        "ensemble_final_role": panel.ensemble_final_role,
        "ensemble_confidence": panel.ensemble_confidence,
        "leak_verdict": panel.leak_verdict,
        "leak_source": panel.leak_source,
        "review_required": panel.review_required,
    }


def _review_record(
    brief: AuthoringBrief,
    *,
    reasons: list[str],
    lm: Any,
    reasoning: Optional[str] = None,
    treatment_node: str = TREATMENT_NODE,
    outcome_node: str = OUTCOME_NODE,
) -> AuthoredAttestation:
    """An author failure: no edges, review required, reason recorded."""
    return AuthoredAttestation(
        feature_name=brief.feature_name,
        treatment_node=treatment_node,
        outcome_node=outcome_node,
        feature_node=brief.feature_name,
        edges=[],
        edge_provenance=[],
        derived_role=None,
        expected_role=None,
        ambiguous=True,
        review_required=True,
        review_reasons=list(reasons),
        cross_check={"derived_role": None, "panel_final_role": None, "agrees": None},
        constraint_violations=[],
        panel_summary=_panel_summary(brief.panel),
        latents=[],
        provenance="machine",
        model_id=_model_id(lm),
        prompt_hash=prompt_hash() if _DSPY_AVAILABLE else "",
        guide_hash=GUIDE_HASH,
        authored_at=datetime.now(timezone.utc).isoformat(),
        reasoning=reasoning,
    )


def postprocess_fragment(
    brief: AuthoringBrief,
    fragment: ParsedFragment,
    *,
    resolver: Any,
    lm: Any,
    reasoning: Optional[str] = None,
    treatment_node: str = TREATMENT_NODE,
    outcome_node: str = OUTCOME_NODE,
) -> AuthoredAttestation:
    """Derive the role, apply the Lane E constraints, grade the edges, stamp."""
    reasons: list[str] = []
    violations: list[str] = []
    graph = nx.DiGraph(list(fragment.edges))
    derived: Optional[str]
    try:
        derived = extract_role(fragment.feature_node, treatment_node, outcome_node, graph)
    except ValueError as exc:
        derived = None
        reasons.append(f"unclassifiable fragment: {exc}")

    panel = brief.panel
    # Lane E 3(b): a Layer-1 post-index verdict forbids feature -> T.
    if (
        panel is not None
        and panel.layer_1_verdict == _LAYER_1_POST_INDEX
        and (fragment.feature_node, treatment_node) in fragment.edges
    ):
        violations.append(_CONSTRAINT_L1)
        reasons.append(
            "Layer 1 says the feature is post-index, but the author drew feature -> T "
            "(a post-index quantity cannot cause the treatment decision)"
        )
    # Lane E 3(c): derived role vs the ensemble's final role.
    panel_role = panel.ensemble_final_role if panel is not None else None
    agrees: Optional[bool] = None
    if derived is not None and panel_role is not None:
        agrees = derived == panel_role
    cross_check = {"derived_role": derived, "panel_final_role": panel_role, "agrees": agrees}
    ambiguous = bool(fragment.ambiguous) or agrees is False
    if agrees is False:
        reasons.append(f"derived role {derived!r} disagrees with the panel ensemble {panel_role!r}")
    if fragment.expected_role is not None and derived is not None:
        if fragment.expected_role != derived:
            reasons.append(
                f"author expected {fragment.expected_role!r}, extractor derived {derived!r}"
            )
    if panel is not None and panel.leak_verdict:
        reasons.append(
            f"panel leak verdict ({panel.leak_source}): excluded from every adjustment set "
            "whatever the authored edges say"
        )

    provenance = grade_edges(
        fragment,
        resolver=resolver,
        treatment_label=brief.treatment_label,
        outcome_label=brief.outcome_label,
        treatment_node=treatment_node,
        outcome_node=outcome_node,
    )
    if _CONSTRAINT_L1 in violations:
        for ep in provenance:
            if (ep.from_node, ep.to_node) == (fragment.feature_node, treatment_node):
                ep.constraint_violation = _CONSTRAINT_L1
    unsupported = [
        f"{ep.from_node}->{ep.to_node}" for ep in provenance if ep.grade == "unsupported"
    ]
    if unsupported:
        reasons.append(f"unsupported edges (no verified citation): {', '.join(unsupported)}")

    review_required = (
        derived is None
        or bool(violations)
        or agrees is False
        or (panel is not None and panel.review_required)
    )
    return AuthoredAttestation(
        feature_name=brief.feature_name,
        treatment_node=treatment_node,
        outcome_node=outcome_node,
        feature_node=fragment.feature_node,
        edges=[list(e) for e in fragment.edges],
        edge_provenance=provenance,
        derived_role=derived,
        expected_role=fragment.expected_role,
        ambiguous=ambiguous,
        review_required=review_required,
        review_reasons=reasons,
        cross_check=cross_check,
        constraint_violations=violations,
        panel_summary=_panel_summary(panel),
        latents=list(fragment.latents),
        provenance="machine",
        model_id=_model_id(lm),
        prompt_hash=prompt_hash() if _DSPY_AVAILABLE else "",
        guide_hash=GUIDE_HASH,
        authored_at=datetime.now(timezone.utc).isoformat(),
        reasoning=reasoning,
    )


def author_feature(
    brief: AuthoringBrief,
    *,
    resolver: Any,
    lm: Any = None,
    program: Any = None,
) -> AuthoredAttestation:
    """Author one feature end to end.

    ``lm`` is the DSPy LM to run under (``dspy.context(lm=lm)``); when ``None``
    the globally configured LM is used (``ensure_dspy_configured``). ``program``
    lets a caller reuse one :class:`StructuralAuthor` across features. Any LM or
    parsing failure yields a review-only record, never an exception.
    """
    if not _DSPY_AVAILABLE:
        return _review_record(brief, reasons=["dspy is not installed"], lm=lm)
    prog = program or StructuralAuthor()
    active_lm = lm if lm is not None else getattr(dspy.settings, "lm", None)
    if active_lm is None:
        return _review_record(brief, reasons=["no DSPy LM configured"], lm=None)
    try:
        if lm is not None:
            with dspy.context(lm=lm):
                pred = prog(brief)
        else:
            pred = prog(brief)
    except Exception as exc:  # noqa: BLE001 — an LM failure routes the feature to review
        logger.warning("structural author failed for %s: %s", brief.feature_name, exc)
        return _review_record(brief, reasons=[f"LM call failed: {exc}"], lm=active_lm)
    raw = {
        "edges": getattr(pred, "edges", None),
        "edge_rationales": getattr(pred, "edge_rationales", None),
        "entity_names": getattr(pred, "entity_names", None),
        "expected_role": getattr(pred, "expected_role", None),
        "ambiguous": getattr(pred, "ambiguous", False),
    }
    reasoning = getattr(pred, "reasoning", None)
    try:
        fragment = parse_author_output(raw, feature_name=brief.feature_name)
    except StructuralAuthorError as exc:
        logger.warning("structural author output rejected for %s: %s", brief.feature_name, exc)
        return _review_record(
            brief,
            reasons=[f"output rejected: {exc}"],
            lm=active_lm,
            reasoning=str(reasoning) if reasoning is not None else None,
        )
    return postprocess_fragment(
        brief,
        fragment,
        resolver=resolver,
        lm=active_lm,
        reasoning=str(reasoning) if reasoning is not None else None,
    )
