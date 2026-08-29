import os

import pytest

from app.credentials import (
    ROBOFLOW_API_KEY_ENV,
    has_api_key,
    load_api_key,
    load_env_file,
    parse_env_file,
)


@pytest.fixture(autouse=True)
def _clear_env(monkeypatch):
    monkeypatch.delenv(ROBOFLOW_API_KEY_ENV, raising=False)


# --- parse_env_file ------------------------------------------------------


def test_parses_simple_pairs():
    assert parse_env_file("A=1\nB=two\n") == {"A": "1", "B": "two"}


def test_skips_blank_lines_and_comments():
    assert parse_env_file("\n# a comment\n\nA=1\n  # indented\n") == {"A": "1"}


def test_strips_export_prefix():
    assert parse_env_file("export A=1\n") == {"A": "1"}


def test_strips_matched_quotes():
    assert parse_env_file("A='1'\nB=\"2\"\n") == {"A": "1", "B": "2"}


def test_keeps_unmatched_quotes():
    assert parse_env_file("A='1\n") == {"A": "'1"}


def test_value_may_contain_equals():
    assert parse_env_file("A=b=c\n") == {"A": "b=c"}


def test_skips_lines_without_equals():
    assert parse_env_file("nonsense\nA=1\n") == {"A": "1"}


def test_skips_empty_key():
    assert parse_env_file("=value\nA=1\n") == {"A": "1"}


def test_empty_value_is_kept_as_empty_string():
    assert parse_env_file("A=\n") == {"A": ""}


# --- load_env_file -------------------------------------------------------


def test_missing_file_yields_empty_dict(tmp_path):
    assert load_env_file(str(tmp_path / "nope.env")) == {}


def test_reads_a_real_file(tmp_path):
    p = tmp_path / ".env"
    p.write_text(f"{ROBOFLOW_API_KEY_ENV}=abc123\n", encoding="utf-8")
    assert load_env_file(str(p)) == {ROBOFLOW_API_KEY_ENV: "abc123"}


def test_directory_path_does_not_raise(tmp_path):
    assert load_env_file(str(tmp_path)) == {}


# --- load_api_key --------------------------------------------------------


def test_returns_none_when_nothing_is_set(tmp_path):
    assert load_api_key(str(tmp_path / ".env")) is None


def test_reads_from_file(tmp_path):
    p = tmp_path / ".env"
    p.write_text(f"{ROBOFLOW_API_KEY_ENV}=from_file\n", encoding="utf-8")
    assert load_api_key(str(p)) == "from_file"


def test_environment_wins_over_file(tmp_path, monkeypatch):
    p = tmp_path / ".env"
    p.write_text(f"{ROBOFLOW_API_KEY_ENV}=from_file\n", encoding="utf-8")
    monkeypatch.setenv(ROBOFLOW_API_KEY_ENV, "from_env")
    assert load_api_key(str(p)) == "from_env"


def test_blank_environment_value_falls_back_to_file(tmp_path, monkeypatch):
    p = tmp_path / ".env"
    p.write_text(f"{ROBOFLOW_API_KEY_ENV}=from_file\n", encoding="utf-8")
    monkeypatch.setenv(ROBOFLOW_API_KEY_ENV, "   ")
    assert load_api_key(str(p)) == "from_file"


def test_blank_file_value_is_none(tmp_path):
    p = tmp_path / ".env"
    p.write_text(f"{ROBOFLOW_API_KEY_ENV}=\n", encoding="utf-8")
    assert load_api_key(str(p)) is None


def test_surrounding_whitespace_is_stripped(tmp_path, monkeypatch):
    monkeypatch.setenv(ROBOFLOW_API_KEY_ENV, "  spaced  ")
    assert load_api_key(str(tmp_path / ".env")) == "spaced"


# --- has_api_key ---------------------------------------------------------


def test_has_api_key_reflects_presence(tmp_path, monkeypatch):
    p = tmp_path / ".env"
    assert has_api_key(str(p)) is False
    monkeypatch.setenv(ROBOFLOW_API_KEY_ENV, "k")
    assert has_api_key(str(p)) is True
