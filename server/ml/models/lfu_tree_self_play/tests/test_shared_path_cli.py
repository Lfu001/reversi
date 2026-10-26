"""The H1 command reports identity failures without losing its evidence."""

import json

import pytest

from lfu_tree_self_play import cli


@pytest.mark.parametrize("passed", [True, False])
def test_shared_path_command_saves_report_and_propagates_status(
    tmp_path, monkeypatch, capsys, passed
):
    report = {"passed": passed, "measurements": [], "errors": []}
    calls = []

    def verify(*, device):
        calls.append(device)
        return report

    monkeypatch.setattr(cli, "verify_shared_paths", verify, raising=False)
    output = tmp_path / "h1.json"
    result = cli.main(
        ["verify-shared-paths", "--output", str(output), "--device", "cpu"]
    )

    assert result == (0 if passed else 1)
    assert json.loads(output.read_text()) == report
    assert calls == ["cpu"]
    assert ("PASS" if passed else "FAIL") in capsys.readouterr().out


def test_shared_path_command_requires_explicit_supported_device(tmp_path):
    with pytest.raises(SystemExit) as failure:
        cli.main(["verify-shared-paths", "--output", str(tmp_path / "h1.json")])
    assert failure.value.code == 2


def test_shared_path_command_reports_output_write_failure(
    tmp_path, monkeypatch, capsys
):
    monkeypatch.setattr(
        cli, "verify_shared_paths", lambda **_: {"passed": True}, raising=False
    )
    with pytest.raises(SystemExit) as failure:
        cli.main(["verify-shared-paths", "--output", str(tmp_path), "--device", "mps"])
    assert failure.value.code == 2
    assert "Traceback" not in capsys.readouterr().err
