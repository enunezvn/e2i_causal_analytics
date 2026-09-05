"""Column display labels: ONE leaf SSOT for every surface (2026-09-05, #1895).

The curated labels used to live only in src/api/routes/causal.py, so the
backend insight builders (src/insights/*.py) could not use them without an
insights -> api.routes import inversion, and rendered raw column names into
user-facing prose ("For treatment_arm -> persistent_180d ...") under headers
that already read "Treatment arm -> Persistent at 180d".

NOTE: CI's unit job does NOT run tests/insights/ — run scoped locally when
touching src/insights/column_labels.py.
"""

import subprocess
import sys

from src.insights import column_labels


def test_leaf_module_imports_without_the_api_package():
    # The whole point of the move: src/insights must be able to import the
    # labels without pulling FastAPI routes (and their heavy imports) in.
    code = (
        "import sys; import src.insights.column_labels as m; "
        "assert not any(k == 'src.api' or k.startswith('src.api.') for k in sys.modules), "
        "[k for k in sys.modules if k.startswith('src.api')]; "
        "print(m.column_label('sample_dropped'))"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True, timeout=120
    )
    assert out.stdout.strip() == "Product samples provided (rep sample drop)"


def test_route_module_re_exports_the_same_objects():
    # Six existing test modules import the SSOT through the route; the move
    # must not fork the dictionaries.
    from src.api.routes.causal import _COLUMN_DEFINITIONS, _COLUMN_LABELS, _column_label

    assert _COLUMN_LABELS is column_labels.COLUMN_LABELS
    assert _COLUMN_DEFINITIONS is column_labels.COLUMN_DEFINITIONS
    assert _column_label is column_labels.column_label


def test_column_label_curated_else_capitalize_fallback():
    assert (
        column_labels.column_label("sample_dropped") == "Product samples provided (rep sample drop)"
    )
    assert column_labels.column_label("trigger_accepted") == "NBA trigger accepted"
    assert column_labels.column_label("conversion_flag") == "Conversion flag"
    # Parity with frontend/src/lib/column-labels.ts (same inputs pinned there).
    assert column_labels.column_label("geographic_region=West") == "Geographic region=west"
    assert column_labels.column_label("uas7_HIGH") == "Uas7 high"
