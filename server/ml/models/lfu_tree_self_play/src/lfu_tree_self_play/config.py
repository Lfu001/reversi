"""Typed TOML experiment settings and stable experiment identifiers."""

import hashlib
import json
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from omegaconf import MISSING, DictConfig, OmegaConf
from omegaconf.errors import OmegaConfBaseException


class ConfigError(ValueError):
    """The TOML document does not satisfy the experiment schema."""


@dataclass
class ExperimentSettings:
    name: str = MISSING


@dataclass
class SeedSets:
    training: list[int] = MISSING
    tuning: list[int] = MISSING
    final_evaluation: list[int] = MISSING


@dataclass
class ExperimentConfig:
    experiment: ExperimentSettings = field(default_factory=ExperimentSettings)
    seeds: SeedSets = field(default_factory=SeedSets)


_SEED_FIELDS = ("training", "tuning", "final_evaluation")


def _validate_raw_values(data: dict) -> None:
    experiment = data.get("experiment")
    if not isinstance(experiment, dict):
        raise ConfigError("experiment must be a table")
    name = experiment.get("name")
    if type(name) is not str or not name.strip() or "${" in name:
        raise ConfigError("experiment.name must be a nonempty plain string")

    seeds = data.get("seeds")
    if not isinstance(seeds, dict):
        raise ConfigError("seeds must be a table")
    seen: set[int] = set()
    for field_name in _SEED_FIELDS:
        values = seeds.get(field_name)
        if not isinstance(values, list) or not values:
            raise ConfigError(f"seeds.{field_name} must be a nonempty integer list")
        if any(type(seed) is not int or seed < 0 for seed in values):
            raise ConfigError(f"seeds.{field_name} must contain nonnegative integers")
        if len(set(values)) != len(values):
            raise ConfigError(f"seeds.{field_name} contains a duplicate seed")
        if seen.intersection(values):
            raise ConfigError(f"seeds.{field_name} overlaps another seed set")
        seen.update(values)


def load_config(path: str | Path) -> DictConfig:
    """Read TOML into a validated, normalized, read-only OmegaConf config."""
    with Path(path).open("rb") as file:
        data = tomllib.load(file)

    _validate_raw_values(data)
    try:
        config = OmegaConf.merge(OmegaConf.structured(ExperimentConfig), data)
        OmegaConf.to_container(config, resolve=True, throw_on_missing=True)
        for field_name in _SEED_FIELDS:
            config.seeds[field_name] = sorted(config.seeds[field_name])
        OmegaConf.set_readonly(config, True)
    except OmegaConfBaseException as error:
        raise ConfigError(str(error)) from error
    return config


def experiment_id(config: DictConfig) -> str:
    """Hash every validated setting in a deterministic JSON representation."""
    values = OmegaConf.to_container(config, resolve=True, throw_on_missing=True)
    canonical = json.dumps(
        values, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
