import subprocess
import sys

import pytest


def run_cli(*args):
    return subprocess.run(
        [sys.executable, "-m", "lfu_tree_self_play", *map(str, args)],
        capture_output=True,
        text=True,
        check=False,
    )


def test_help_lists_config_command():
    result = run_cli("--help")

    assert result.returncode == 0
    assert "check-config" in result.stdout
    assert result.stderr == ""


def test_check_config_accepts_valid_file(tmp_path):
    path = tmp_path / "experiment.toml"
    path.write_text('name = "smoke"\n', encoding="utf-8")

    result = run_cli("check-config", path)

    assert result.returncode == 0
    assert str(path) in result.stdout
    assert result.stderr == ""


@pytest.mark.parametrize("kind", ["missing", "invalid", "directory", "non-utf8"])
def test_check_config_reports_read_errors_without_traceback(tmp_path, kind):
    path = tmp_path / "bad.toml"
    if kind == "invalid":
        path.write_text("[unfinished", encoding="utf-8")
    elif kind == "directory":
        path.mkdir()
    elif kind == "non-utf8":
        path.write_bytes(b"\xff")

    result = run_cli("check-config", path)

    assert result.returncode == 2
    assert str(path) in result.stderr
    assert "Traceback" not in result.stderr
    assert result.stdout == ""


@pytest.mark.parametrize("args", [[], ["check-config"], ["unknown"]])
def test_cli_requires_a_known_command_and_path(args):
    result = run_cli(*args)

    assert result.returncode == 2
    assert "usage:" in result.stderr
    assert "Traceback" not in result.stderr
