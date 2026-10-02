"""Minimal command-line checks for experiment configuration files."""

import argparse
import json
import tomllib
from pathlib import Path

from lfu_tree_self_play.config import ConfigError, experiment_id, load_config
from lfu_tree_self_play.g0 import run_g0_verification


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="lfu-tree-self-play")
    commands = parser.add_subparsers(dest="command", required=True)
    check_config = commands.add_parser(
        "check-config", help="Validate a TOML experiment configuration"
    )
    check_config.add_argument("path", type=Path, help="Path to a TOML file")
    verify_g0 = commands.add_parser(
        "verify-g0", help="Save the G0 mathematical verification report"
    )
    verify_g0.add_argument("--output", required=True, type=Path)
    verify_g0.add_argument("--pairs", type=int, default=3000)
    verify_g0.add_argument("--seed", type=int, default=314159)
    args = parser.parse_args(argv)

    if args.command == "verify-g0":
        try:
            report = run_g0_verification(seed=args.seed, pairs=args.pairs)
            args.output.write_text(
                json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
        except (ValueError, OSError) as error:
            parser.error(f"Cannot verify G0 or save report: {error}")
        print(f"G0 mathematical checks: {'PASS' if report['passed'] else 'FAIL'}")
        print(f"Report: {args.output}")
        return 0 if report["passed"] else 1

    try:
        config = load_config(args.path)
    except (OSError, UnicodeError, tomllib.TOMLDecodeError, ConfigError) as error:
        parser.error(f"Cannot load configuration {args.path}: {error}")

    print(f"Configuration loaded: {args.path}")
    print(f"Experiment ID: {experiment_id(config)}")
    return 0
