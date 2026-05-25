import math

from tests.benchmarks.substrate.embedder import EMBED_DIM, embed_text, to_pgvector_literal


def _cos(a, b):
    return sum(x * y for x, y in zip(a, b))


def test_dimension_and_unit_norm():
    v = embed_text("Kisqali TRx growth West region")
    assert len(v) == EMBED_DIM
    assert math.isclose(_cos(v, v), 1.0, abs_tol=1e-6)


def test_identical_text_identical_vector():
    assert embed_text("fabhalta PNH discontinuation") == embed_text("fabhalta PNH discontinuation")


def test_token_overlap_gives_high_cosine():
    q = embed_text("kisqali trx growth west region q3")
    doc = embed_text("kisqali trx growth west region q3 confidence score high")
    assert _cos(q, doc) > 0.5  # must clear hybrid_vector_search's hardcoded 0.5 floor


def test_disjoint_tokens_near_zero():
    a = embed_text("alpha beta gamma delta")
    b = embed_text("xenon yttrium zirconium niobium")
    assert _cos(a, b) < 0.1


def test_empty_text_is_deterministic_unit_vector():
    v = embed_text("!!! ???")
    assert math.isclose(_cos(v, v), 1.0, abs_tol=1e-6)


def test_pgvector_literal_format():
    lit = to_pgvector_literal([0.0, 1.0, -0.5])
    assert lit == "[0.000000,1.000000,-0.500000]"
