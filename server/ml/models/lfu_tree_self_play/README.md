# lfu_tree_self_play

木状自己対戦学習の実験用 Python パッケージです。
研究計画と開発タスクは [docs](docs/README.md) を参照してください。

## セットアップと実行

このディレクトリで実行します。通常版 CPython 3.14 と uv を使用します。

```sh
uv sync --locked
uv run --locked lfu-tree-self-play check-config configs/example.toml
uv run --locked python -m lfu_tree_self_play check-config configs/example.toml
```

`check-config` は TOML を読み込めれば終了コード 0 で成功を表示します。
ファイル不在、読込失敗、文字コード・構文エラーは標準エラー出力に表示し、
終了コード 2 で終了します。設定項目の意味や学習実行の可否は検証しません。

## テスト

```sh
uv run --locked pytest -q
```
