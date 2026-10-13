"""Fixed opening sets reject leakage and replay their saved provenance."""

import json
from dataclasses import replace

import pytest

from lfu_tree_self_play.config import load_config


def _config(tmp_path, seeds=("[1, 2]", "[101]", "[201]")):
    path = tmp_path / "config.toml"
    path.write_text(
        '[experiment]\nname = "roots"\n[seeds]\n'
        f"training = {seeds[0]}\ntuning = {seeds[1]}\n"
        f"final_evaluation = {seeds[2]}\n"
    )
    return load_config(path)


def test_registry_fills_each_seed_with_disjoint_reproducible_roots(tmp_path):
    from lfu_tree_self_play.roots import generate_root_registry

    config = _config(tmp_path)
    first = generate_root_registry(config, roots_per_seed=20)
    second = generate_root_registry(config, roots_per_seed=20)

    assert first == second
    assert len(first.positions("training")) == 40
    assert len(first.positions("tuning")) == 20
    assert len(first.positions("final_evaluation")) == 20
    keys = [entry.root.key for entry in first.entries]
    assert len(set(keys)) == 80
    assert {entry.run_seed for entry in first.entries} == {1, 2, 101, 201}
    assert any(entry.attempt > 0 for entry in first.entries)
    assert first.registry_id == second.registry_id
    with pytest.raises(ValueError):
        first.positions("unknown")


def test_registry_is_independent_of_seed_list_order(tmp_path):
    from lfu_tree_self_play.roots import generate_root_registry

    first = generate_root_registry(_config(tmp_path))
    second = generate_root_registry(_config(tmp_path, ("[2, 1]", "[101]", "[201]")))
    assert first.registry_id == second.registry_id


@pytest.mark.parametrize("within_split", [False, True])
def test_registry_rejects_duplicate_positions_even_with_distinct_seeds(
    tmp_path, within_split
):
    from lfu_tree_self_play.roots import generate_root_registry

    registry = generate_root_registry(_config(tmp_path))
    entries = list(registry.entries)
    target = 1 if within_split else 2
    entries[target] = replace(entries[target], root=entries[0].root)

    with pytest.raises(ValueError, match="duplicate root"):
        replace(registry, entries=tuple(entries))


def test_registry_rejects_distinct_one_move_roots_related_by_symmetry(tmp_path):
    from lfu_tree_self_play.roots import generate_root_position, generate_root_registry

    registry = generate_root_registry(_config(tmp_path))
    entries = list(registry.entries)
    first = generate_root_position(1, 1)
    other = next(
        root
        for seed in range(2, 100)
        if (root := generate_root_position(seed, 1)).actions != first.actions
    )
    assert first.state != other.state
    entries[0] = replace(entries[0], root=first)
    entries[2] = replace(entries[2], root=other)
    with pytest.raises(ValueError, match="duplicate root"):
        replace(registry, entries=tuple(entries))


def test_registry_rejects_duplicate_seed_across_splits(tmp_path):
    from lfu_tree_self_play.roots import generate_root_registry

    registry = generate_root_registry(_config(tmp_path))
    entries = list(registry.entries)
    entries[2] = replace(entries[2], run_seed=1)
    with pytest.raises(ValueError, match="seed.*split"):
        replace(registry, entries=tuple(entries))


def test_registry_round_trip_replays_roots_and_refuses_overwrite(tmp_path):
    from lfu_tree_self_play.roots import RootRegistry, generate_root_registry

    config = _config(tmp_path)
    original = generate_root_registry(config, roots_per_seed=4)
    path = tmp_path / "roots.json"
    original.save(path)
    loaded = RootRegistry.load(path, config)

    assert loaded == original
    assert loaded.registry_id == original.registry_id
    assert loaded.positions("final_evaluation") == original.positions(
        "final_evaluation"
    )
    with pytest.raises(FileExistsError):
        original.save(path)
    assert RootRegistry.load(path, config) == original


@pytest.mark.parametrize(
    "corruption", ["version", "actions", "key", "seed", "missing", "extra", "count"]
)
def test_registry_load_rejects_corrupt_or_incomplete_records(tmp_path, corruption):
    from lfu_tree_self_play.roots import RootRegistry, generate_root_registry

    config = _config(tmp_path)
    registry = generate_root_registry(config, roots_per_seed=3)
    path = tmp_path / "roots.json"
    registry.save(path)
    document = json.loads(path.read_text())
    if corruption == "version":
        document["version"] = True
    elif corruption == "actions":
        document["entries"][0]["actions"] = [64]
    elif corruption == "key":
        document["entries"][0]["key"] = "0" * 64
    elif corruption == "seed":
        document["entries"][0]["run_seed"] = 999
    elif corruption == "missing":
        document["entries"] = [e for e in document["entries"] if e["run_seed"] != 201]
    elif corruption == "extra":
        document["entries"][0]["unknown"] = 1
    else:
        document["roots_per_seed"] = True
    path.write_text(json.dumps(document))

    with pytest.raises(ValueError):
        RootRegistry.load(path, config)


def test_registry_load_rejects_different_experiment(tmp_path):
    from lfu_tree_self_play.roots import RootRegistry, generate_root_registry

    original_config = _config(tmp_path)
    path = tmp_path / "roots.json"
    generate_root_registry(original_config).save(path)
    changed_config = _config(tmp_path, ("[1, 2]", "[102]", "[201]"))

    with pytest.raises(ValueError, match="experiment"):
        RootRegistry.load(path, changed_config)


def test_registry_load_rejects_duplicate_json_fields(tmp_path):
    from lfu_tree_self_play.roots import RootRegistry, generate_root_registry

    config = _config(tmp_path)
    path = tmp_path / "roots.json"
    generate_root_registry(config).save(path)
    path.write_text(path.read_text().replace('"version":1', '"version":1,"version":1'))
    with pytest.raises(ValueError, match="duplicate"):
        RootRegistry.load(path, config)


@pytest.mark.parametrize("argument", ["roots_per_seed", "max_attempts"])
@pytest.mark.parametrize("value", [0, -1, True, 1.5])
def test_registry_rejects_invalid_generation_limits(tmp_path, argument, value):
    from lfu_tree_self_play.roots import generate_root_registry

    with pytest.raises(ValueError):
        generate_root_registry(_config(tmp_path), **{argument: value})


def test_registry_stops_when_attempt_budget_cannot_fill_requested_roots(tmp_path):
    from lfu_tree_self_play.roots import generate_root_registry

    with pytest.raises(ValueError, match="attempt"):
        generate_root_registry(_config(tmp_path), roots_per_seed=100, max_attempts=1)
