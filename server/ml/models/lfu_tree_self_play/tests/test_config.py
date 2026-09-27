import re
import tomllib

import pytest
from omegaconf.errors import ReadonlyConfigError

VALID_TOML = """[experiment]
name = "smoke"
[seeds]
training = [7, 2]
tuning = [11]
final_evaluation = [19]
"""


def write_config(tmp_path, content=VALID_TOML):
    path = tmp_path / "experiment.toml"
    path.write_text(content, encoding="utf-8")
    return path


def test_load_config_returns_readonly_structured_config(tmp_path):
    from lfu_tree_self_play.config import load_config

    config = load_config(write_config(tmp_path))

    assert type(config).__name__ == "DictConfig"
    assert config.experiment.name == "smoke"
    assert list(config.seeds.training) == [2, 7]
    with pytest.raises(ReadonlyConfigError):
        config.experiment.name = "changed"


@pytest.mark.parametrize(
    "content",
    [
        "[experiment]\nname = 'smoke'\n",
        "[experiment]\nname = 'smoke'\n[seeds]\ntraining = [1]\ntuning = [2]\n",
        VALID_TOML + "[unexpected]\nvalue = 1\n",
        VALID_TOML.replace('name = "smoke"', "name = 42"),
        VALID_TOML.replace('name = "smoke"', 'name = "   "'),
        VALID_TOML.replace("training = [7, 2]", "training = []"),
        VALID_TOML.replace("training = [7, 2]", "training = [7, 7]"),
        VALID_TOML.replace("training = [7, 2]", "training = [-1, 2]"),
        VALID_TOML.replace("training = [7, 2]", "training = [true, 2]"),
        VALID_TOML.replace("tuning = [11]", "tuning = [7]"),
        VALID_TOML.replace('name = "smoke"', 'name = "${seeds.training}"'),
    ],
)
def test_load_config_rejects_invalid_schema_or_seeds(tmp_path, content):
    from lfu_tree_self_play.config import ConfigError, load_config

    with pytest.raises(ConfigError):
        load_config(write_config(tmp_path, content))


def test_experiment_id_ignores_toml_layout_and_seed_order(tmp_path):
    from lfu_tree_self_play.config import experiment_id, load_config

    first = load_config(write_config(tmp_path, VALID_TOML))
    reordered = load_config(
        write_config(
            tmp_path,
            "[seeds]\nfinal_evaluation=[19]\ntuning=[11]\ntraining=[2,7]\n"
            '[experiment]\nname="smoke"\n',
        )
    )

    assert experiment_id(first) == experiment_id(reordered)
    assert re.fullmatch(r"[0-9a-f]{64}", experiment_id(first))


@pytest.mark.parametrize(
    "changed",
    [
        VALID_TOML.replace('name = "smoke"', 'name = "other"'),
        VALID_TOML.replace("training = [7, 2]", "training = [7, 3]"),
    ],
)
def test_experiment_id_changes_with_validated_setting(tmp_path, changed):
    from lfu_tree_self_play.config import experiment_id, load_config

    original = load_config(write_config(tmp_path, VALID_TOML))
    modified = load_config(write_config(tmp_path, changed))

    assert experiment_id(original) != experiment_id(modified)


def test_load_config_rejects_invalid_toml(tmp_path):
    from lfu_tree_self_play.config import load_config

    with pytest.raises(tomllib.TOMLDecodeError):
        load_config(write_config(tmp_path, "[unfinished"))


def test_load_config_reports_missing_file(tmp_path):
    from lfu_tree_self_play.config import load_config

    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "missing.toml")
