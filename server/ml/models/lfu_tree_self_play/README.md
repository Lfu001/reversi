# lfu_tree_self_play

オセロの自己対戦データを収集・検証し、学習用のリターンを計算して、データとモデル状態を保存する Python パッケージです。独立対局と、途中で複数の継続試行に分岐する木を Python から生成できます。

CLI は設定検証、方策勾配の数値検証、共有経路の同一性・生成コストの比較を提供します。収集から学習・対戦評価までを実行するコマンドはありません。

## クイックスタート

通常版 CPython 3.14、uv、Rust toolchain を用意し、このディレクトリで実行します。ローカル依存の `reversi-py` は Rust 拡張をビルドします。データとチェックポイントの保存には POSIX のローカルファイルシステムを使います。

```sh
uv sync --locked
uv run --locked lfu-tree-self-play check-config configs/example.toml
```

設定例は [configs/example.toml](configs/example.toml) にあります。検証成功時に表示される experiment ID は正規化した設定のハッシュです。設定を変えた実験には別の保存先を使います。

## 目的から探す

| 目的 | 文書 |
| --- | --- |
| 小さな自己対戦木を収集し、データと checkpoint を保存する | [自己対戦データの収集と保存](docs/guides/collection.md) |
| 複数の対局・木をまとめて推論し、処理量を計測する | [batch 推論による収集](docs/guides/batch-collection.md) |
| 同じ木の共有方式と再計算方式で生成コストを比較する | [共有経路の同一性と生成コスト](docs/guides/shared-paths.md) |
| 保存済みのデータとモデル状態を復元する | [停止後の復旧と再試行](docs/guides/recovery.md) |
| 期待リターンと方策勾配の正しさを確認する | [数値検証の実行](docs/guides/verification.md) |
| 研究の仮説、数理、比較条件を理解する | [研究計画](docs/research/README.md) |
| 開発タスクと次工程へ進む条件を確認する | [開発計画](docs/development/README.md) |

## 開発

```sh
uv run --locked pytest -q
```

利用ガイドの Python コードもテストで直接実行します。文書の例だけを確認する場合は `uv run --locked pytest tests/test_documentation.py -q` を使います。

文書には、利用・復旧の手順と、設計判断や実験条件の理由を残します。API は [ソース](src/lfu_tree_self_play) を参照し、引数・フィールドの一覧、コードの言い換え、作業報告は文書に追加しません。必要な説明がなければ項目や文書を作りません。
