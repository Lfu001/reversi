"""Minimal command-line checks for experiment configuration files."""

import argparse
import json
import tomllib
from pathlib import Path

from lfu_tree_self_play.config import ConfigError, experiment_id, load_config
from lfu_tree_self_play.verification import verify_policy_gradient


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="lfu-tree-self-play")
    commands = parser.add_subparsers(dest="command", required=True)
    check_config = commands.add_parser(
        "check-config", help="Validate a TOML experiment configuration"
    )
    check_config.add_argument("path", type=Path, help="Path to a TOML file")
    verification_command = commands.add_parser(
        "verify-policy-gradient",
        help="Verify expected returns and conditional policy gradients",
    )
    verification_command.add_argument("--output", required=True, type=Path)
    verification_command.add_argument("--pairs", type=int, default=3000)
    verification_command.add_argument("--seed", type=int, default=314159)
    args = parser.parse_args(argv)

    if args.command == "verify-policy-gradient":
        try:
            report = verify_policy_gradient(seed=args.seed, pairs=args.pairs)
            args.output.write_text(
                json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
        except (ValueError, OSError) as error:
            parser.error(f"Cannot verify policy gradients or save report: {error}")
        print(f"Policy gradient verification: {'PASS' if report['passed'] else 'FAIL'}")
        print(f"Report: {args.output}")
        return 0 if report["passed"] else 1

    try:
        config = load_config(args.path)
    except (OSError, UnicodeError, tomllib.TOMLDecodeError, ConfigError) as error:
        parser.error(f"Cannot load configuration {args.path}: {error}")

    print(f"Configuration loaded: {args.path}")
    print(f"Experiment ID: {experiment_id(config)}")
    return 0
