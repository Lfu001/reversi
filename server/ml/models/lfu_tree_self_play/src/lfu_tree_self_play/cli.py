"""Minimal command-line checks for experiment configuration files."""

import argparse
import tomllib
from pathlib import Path

from lfu_tree_self_play.config import load_config


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="lfu-tree-self-play")
    commands = parser.add_subparsers(dest="command", required=True)
    check_config = commands.add_parser(
        "check-config", help="Check that a TOML configuration can be read"
    )
    check_config.add_argument("path", type=Path, help="Path to a TOML file")
    args = parser.parse_args(argv)

    try:
        load_config(args.path)
    except (OSError, UnicodeError, tomllib.TOMLDecodeError) as error:
        parser.error(f"Cannot load configuration {args.path}: {error}")

    print(f"Configuration loaded: {args.path}")
    return 0
