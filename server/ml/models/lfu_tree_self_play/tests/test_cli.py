import subprocess
import sys
from pathlib import Path

import pytest


@pytest.fixture(params=["module", "console"])
def run_cli(request, tmp_path):
    if request.param == "module":
        command = [sys.executable, "-m", "lfu_tree_self_play"]
    else:
        executable = "lfu-tree-self-play" + (".exe" if sys.platform == "win32" else "")
        command = [str(Path(sys.executable).with_name(executable))]

    def run(*args):
        return subprocess.run(
            [*command, *map(str, args)],
            cwd=tmp_path,
            capture_output=True,
            text=True,
            check=False,
        )

    return run


def test_help_lists_config_command(run_cli):
    result = run_cli("--help")

    assert result.returncode == 0
    assert "check-config" in result.stdout
    assert result.stderr == ""


def test_check_config_accepts_valid_file(run_cli, tmp_path):
    path = tmp_path / "experiment.toml"
    path.write_text('name = "smoke"\n', encoding="utf-8")

    result = run_cli("check-config", path)

    assert result.returncode == 0
    assert str(path) in result.stdout
    assert result.stderr == ""


@pytest.mark.parametrize("kind", ["missing", "invalid", "directory", "non-utf8"])
def test_check_config_reports_read_errors_without_traceback(run_cli, tmp_path, kind):
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
def test_cli_requires_a_known_command_and_path(run_cli, args):
    result = run_cli(*args)

    assert result.returncode == 2
    assert "usage:" in result.stderr
    assert "Traceback" not in result.stderr
