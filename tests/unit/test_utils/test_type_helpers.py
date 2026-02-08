"""Tests for src.utils.type_helpers."""

import pytest

from src.utils.type_helpers import parse_supabase_row, parse_supabase_rows, safe_get


class TestParseSupabaseRow:
    def test_dict_passthrough(self):
        row = {"id": 1, "name": "test"}
        result = parse_supabase_row(row)
        assert result == {"id": 1, "name": "test"}

    def test_returns_new_dict(self):
        row = {"id": 1}
        result = parse_supabase_row(row)
        assert result is not row

    def test_mapping_subclass(self):
        from collections import OrderedDict

        row = OrderedDict([("a", 1), ("b", 2)])
        result = parse_supabase_row(row)
        assert result == {"a": 1, "b": 2}
        assert isinstance(result, dict)

    def test_raises_on_string(self):
        with pytest.raises(TypeError, match="Expected dict row.*got str"):
            parse_supabase_row("not a dict")

    def test_raises_on_int(self):
        with pytest.raises(TypeError, match="Expected dict row.*got int"):
            parse_supabase_row(42)

    def test_raises_on_list(self):
        with pytest.raises(TypeError, match="Expected dict row.*got list"):
            parse_supabase_row([1, 2, 3])

    def test_raises_on_none(self):
        with pytest.raises(TypeError, match="Expected dict row.*got NoneType"):
            parse_supabase_row(None)

    def test_raises_on_bool(self):
        with pytest.raises(TypeError, match="Expected dict row.*got bool"):
            parse_supabase_row(True)

    def test_empty_dict(self):
        assert parse_supabase_row({}) == {}

    def test_nested_values(self):
        row = {"data": {"nested": [1, 2, 3]}, "count": 5}
        result = parse_supabase_row(row)
        assert result["data"] == {"nested": [1, 2, 3]}


class TestParseSupabaseRows:
    def test_list_of_dicts(self):
        rows = [{"id": 1}, {"id": 2}]
        result = parse_supabase_rows(rows)
        assert result == [{"id": 1}, {"id": 2}]

    def test_empty_list(self):
        assert parse_supabase_rows([]) == []

    def test_tuple_input(self):
        rows = ({"a": 1}, {"b": 2})
        result = parse_supabase_rows(rows)
        assert result == [{"a": 1}, {"b": 2}]

    def test_raises_on_non_list(self):
        with pytest.raises(TypeError, match="Expected list of rows.*got str"):
            parse_supabase_rows("not a list")

    def test_raises_on_bad_row(self):
        with pytest.raises(TypeError, match="Expected dict row.*got int"):
            parse_supabase_rows([{"ok": True}, 42])


class TestSafeGet:
    def test_present_correct_type(self):
        assert safe_get({"x": 42}, "x", int, 0) == 42

    def test_missing_key(self):
        assert safe_get({}, "x", int, 0) == 0

    def test_wrong_type(self):
        assert safe_get({"x": "not_int"}, "x", int, 0) == 0

    def test_none_value(self):
        assert safe_get({"x": None}, "x", str, "default") == "default"

    def test_string_type(self):
        assert safe_get({"name": "alice"}, "name", str, "") == "alice"

    def test_list_type(self):
        assert safe_get({"items": [1, 2]}, "items", list, []) == [1, 2]

    def test_bool_type(self):
        assert safe_get({"flag": True}, "flag", bool, False) is True
