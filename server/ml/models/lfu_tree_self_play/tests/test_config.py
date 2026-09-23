import tomllib

import pytest


def test_load_config_preserves_nested_toml_values(tmp_path):
    from lfu_tree_self_play.config import load_config

    path = tmp_path / "experiment.toml"
    path.write_text(
        'name = "試験"\n[training]\nlr = 0.001\nseeds = [1, 2]\nenabled = true\n',
        encoding="utf-8",
    )

    assert load_config(path) == {
        "name": "試験",
        "training": {"lr": 0.001, "seeds": [1, 2], "enabled": True},
    }


def test_load_config_rejects_invalid_toml(tmp_path):
    from lfu_tree_self_play.config import load_config

    path = tmp_path / "invalid.toml"
    path.write_text("[unfinished", encoding="utf-8")

    with pytest.raises(tomllib.TOMLDecodeError):
        load_config(path)


def test_load_config_reports_missing_file(tmp_path):
    from lfu_tree_self_play.config import load_config

    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "missing.toml")
