"""#891: non-finite-float sanitization for JSONB-bound memory payloads.

Postgres jsonb rejects the bare ``NaN``/``Infinity`` tokens Python's default
``json.dumps`` emits, and supabase-py's strict JSON encoder refuses to emit
them at all (``ValueError: Out of range float values are not JSON
compliant``) — so a single non-finite float anywhere in a JSONB-bound payload
makes the whole episodic insert raise BEFORE the request is sent, and the
agent memory hooks swallow that into a silent drop (probe-verified live
2026-06-12, issue #891; the historical variant of this bug class produced the
137 NaN-bearing string-scalar rows migration 073 had to skip).

``sanitize_jsonb_payload`` maps non-finite floats to ``None`` (JSON null)
recursively while leaving every other value — crucially string values that
merely CONTAIN the text ``NaN`` — byte-identical. The codex-R2 corruption
payload (``"threshold: NaN means missing, Infinity capped"``) is pinned here
and in tests/integration/test_episodic_jsonb_shape_883c.py.
"""

import json
import math

from src.memory.jsonb_sanitize import sanitize_jsonb_payload

# The exact string value codex R2 flagged: any quote-UNaware NaN rewriter
# would corrupt it. The sanitizer must never touch string content.
CODEX_R2_NOTE = "threshold: NaN means missing, Infinity capped"


class TestNonFiniteFloatMapping:
    def test_nan_maps_to_none(self):
        assert sanitize_jsonb_payload(float("nan")) is None

    def test_positive_infinity_maps_to_none(self):
        assert sanitize_jsonb_payload(float("inf")) is None

    def test_negative_infinity_maps_to_none(self):
        assert sanitize_jsonb_payload(float("-inf")) is None

    def test_finite_floats_pass_through(self):
        assert sanitize_jsonb_payload(0.81) == 0.81
        assert sanitize_jsonb_payload(-0.0) == -0.0
        assert sanitize_jsonb_payload(1e308) == 1e308


class TestStructureRecursion:
    def test_nested_dicts_lists_tuples(self):
        payload = {
            "test_metrics": {"auc_roc": 0.81, "rmse": float("nan")},
            "folds": [1.0, float("inf"), {"brier": float("-inf")}],
            "pair": (float("nan"), 2.0),
        }
        out = sanitize_jsonb_payload(payload)
        assert out == {
            "test_metrics": {"auc_roc": 0.81, "rmse": None},
            "folds": [1.0, None, {"brier": None}],
            "pair": [None, 2.0],  # tuples normalize to lists (JSON arrays)
        }

    def test_scalars_pass_through_unchanged(self):
        for v in (None, True, False, 0, 42, "x", ""):
            assert sanitize_jsonb_payload(v) == v

    def test_bools_are_not_treated_as_floats(self):
        # bool subclasses int, not float — but pin it explicitly.
        out = sanitize_jsonb_payload({"met": True, "failed": False})
        assert out == {"met": True, "failed": False}


class TestStringContentIsNeverTouched:
    def test_codex_r2_corruption_payload_survives_verbatim(self):
        out = sanitize_jsonb_payload({"note": CODEX_R2_NOTE, "auc": float("nan")})
        assert out["note"] == CODEX_R2_NOTE
        assert out["note"].encode() == CODEX_R2_NOTE.encode()  # byte-identical
        assert out["auc"] is None

    def test_keys_containing_nan_text_survive(self):
        out = sanitize_jsonb_payload({"NaN_count": 3, "Infinity cap": "NaN"})
        assert out == {"NaN_count": 3, "Infinity cap": "NaN"}


class TestStrictJsonGuarantee:
    def test_output_always_strict_json_serializable(self):
        """The whole point: after sanitization, json.dumps(allow_nan=False)
        (supabase-py's encoder semantics) must never raise."""
        payload = {
            "m": {"a": float("nan"), "b": [float("inf"), {"c": float("-inf")}]},
            "note": CODEX_R2_NOTE,
        }
        text = json.dumps(sanitize_jsonb_payload(payload), allow_nan=False)
        assert "NaN" in text  # only inside the string VALUE
        assert json.loads(text)["note"] == CODEX_R2_NOTE

    def test_idempotent(self):
        payload = {"a": float("nan"), "b": [1.0, float("inf")], "note": CODEX_R2_NOTE}
        once = sanitize_jsonb_payload(payload)
        assert sanitize_jsonb_payload(once) == once

    def test_numpy_float_subclasses_are_handled(self):
        """model_trainer metrics frequently arrive as np.float64 (a float
        subclass) — e.g. float('nan') from stacking/learning_curve nodes."""
        np = __import__("numpy")
        out = sanitize_jsonb_payload({"m": np.float64("nan"), "k": np.float64(0.5)})
        assert out["m"] is None
        assert math.isclose(float(out["k"]), 0.5)

    def test_numpy_non_float_subclass_scalars_are_handled(self):
        """codex iter-1 MEDIUM: np.float32 (and friends) are NOT float
        subclasses — stdlib json rejects even FINITE ones (TypeError), so a
        float32-typed metric recreates the silent-drop class. The sanitizer
        must normalize numpy scalars to Python types and then apply the
        non-finite mapping."""
        np = __import__("numpy")
        out = sanitize_jsonb_payload(
            {
                "nan32": np.float32("nan"),
                "inf16": np.float16("inf"),
                "fin32": np.float32(0.25),
                "i32": np.int32(7),
                "b": np.bool_(True),
            }
        )
        assert out["nan32"] is None
        assert out["inf16"] is None
        assert math.isclose(float(out["fin32"]), 0.25)
        assert out["i32"] == 7
        assert out["b"] is True
        json.dumps(out, allow_nan=False)  # must be strict-JSON clean

    def test_non_finite_float_dict_keys_are_sanitized(self):
        """codex iter-1 MEDIUM: a non-finite float KEY also crashes the strict
        encoder (ValueError) — the silent-drop class through the key position.
        Map it to None, which json renders as the "null" key."""
        out = sanitize_jsonb_payload({float("nan"): "x", 0.5: "y", "name": "z"})
        assert out == {None: "x", 0.5: "y", "name": "z"}
        text = json.dumps(out, allow_nan=False)
        assert json.loads(text) == {"null": "x", "0.5": "y", "name": "z"}

    def test_numpy_scalar_subclasses_are_handled(self):
        """codex iter-2 MEDIUM: a user-defined subclass of a numpy scalar type
        carries its DEFINING module as type(obj).__module__ (not 'numpy'), so
        an exact-module check skips it; np.float32 subclasses are not float
        subclasses either, and strict json raises TypeError — the silent-drop
        class again. Numpy ancestry must be detected through the MRO."""
        np = __import__("numpy")

        class SubF32(np.float32):
            pass

        out = sanitize_jsonb_payload(
            {"nan": SubF32("nan"), "fin": SubF32(0.5), SubF32("inf"): "via-key"}
        )
        assert out["nan"] is None
        assert math.isclose(float(out["fin"]), 0.5)
        assert out[None] == "via-key"  # non-finite subclass KEY also sanitized
        json.dumps(out, allow_nan=False)  # must be strict-JSON clean
