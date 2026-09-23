#!/usr/bin/env bash
set -euo pipefail

package_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
smoke_dir="$(mktemp -d)"
trap 'rm -rf "$smoke_dir"' EXIT

uv build --wheel --out-dir "$smoke_dir/dist" "$package_dir"
uv venv --python 3.14 "$smoke_dir/venv"
uv pip install --python "$smoke_dir/venv/bin/python" "$smoke_dir"/dist/*.whl
cp "$package_dir/configs/example.toml" "$smoke_dir/example.toml"
cd "$smoke_dir"
unset PYTHONPATH PYTHONHOME
"$smoke_dir/venv/bin/python" -I -c 'import lfu_tree_self_play'
"$smoke_dir/venv/bin/python" -I -m lfu_tree_self_play check-config example.toml
"$smoke_dir/venv/bin/lfu-tree-self-play" check-config example.toml
