# lfu_tree_self_play

木状自己対戦学習の実験用 Python パッケージです。

## クイックスタート

通常版 CPython 3.14 と uv を用意し、このディレクトリで実行します。

```sh
uv sync --locked
uv run --locked lfu-tree-self-play check-config configs/example.toml
uv run --locked pytest -q
```

設定の検証が成功すると experiment ID が表示されます。

## 詳細

- [設定と乱数の管理](docs/configuration.md)
- [期待リターンと方策勾配の検証](docs/verification.md)
- [Policy/value モデル](docs/model.md)
- [収集 record と共通検証器](docs/records.md)
- [研究計画と開発タスク](docs/README.md)
