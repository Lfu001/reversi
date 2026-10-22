# lfu_tree_self_play

オセロの自己対戦データを検証し、学習用のリターンを計算して、データとモデル状態を保存する Python パッケージです。独立対局と、途中で複数の継続試行に分岐する木を扱います。

CLI は設定検証と方策勾配の数値検証を提供します。収集から学習・対戦評価までを実行するコマンドはありません。

## クイックスタート

通常版 CPython 3.14、uv、Rust toolchain を用意し、このディレクトリで実行します。ローカル依存の `reversi-py` は Rust 拡張をビルドします。データとチェックポイントの保存には POSIX のローカルファイルシステムを使います。

```sh
uv sync --locked
uv run --locked lfu-tree-self-play check-config configs/example.toml
```

設定例は [configs/example.toml](configs/example.toml) にあります。検証成功時に表示される experiment ID は正規化した設定のハッシュです。設定を変えた実験には別の保存先を使います。

利用・復旧の手順と研究計画は [ドキュメント一覧](docs/README.md) を参照してください。

## 開発

```sh
uv run --locked pytest -q
```

利用ガイドの Python コードもテストで直接実行します。

文書には、利用・復旧の手順と、設計判断や実験条件の理由を残します。API は [ソース](src/lfu_tree_self_play) を参照し、引数・フィールドの一覧、コードの言い換え、作業報告は文書に追加しません。必要な説明がなければ項目や文書を作りません。
