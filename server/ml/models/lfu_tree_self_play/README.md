# lfu_tree_self_play

オセロの自己対戦データを検証し、学習用のリターンを計算して、データとモデル状態を保存する Python パッケージです。独立対局と、途中で複数の継続試行に分岐する木を扱います。

ゲーム操作、policy/value モデル、設定・乱数管理、データ保存、チェックポイント復元を Python API で利用できます。CLI は設定検証と方策勾配の数値検証を提供します。自己対戦の収集から学習・対戦評価までを実行するコマンドはありません。

## クイックスタート

通常版 CPython 3.14、uv、Rust toolchain を用意し、このディレクトリで実行します。ローカル依存の `reversi-py` は Rust 拡張をビルドします。データとチェックポイントの保存には POSIX のローカルファイルシステムを使います。

```sh
uv sync --locked
uv run --locked lfu-tree-self-play check-config configs/example.toml
```

設定の検証が成功すると experiment ID が表示されます。これは正規化した設定のハッシュで、保存先に異なる実験のデータを混在させないために使います。

## 利用ガイド

- [設定と乱数の管理](docs/configuration.md)
- [根局面の生成と用途別集合の固定](docs/roots.md)
- [分岐位置の抽選と実行記録](docs/schedules.md)
- [期待リターンと方策勾配の検証](docs/verification.md)
- [Policy/value モデル](docs/model.md)
- [固定方策からの行動抽出](docs/sampling.md)
- [B/C 共通の木状データ収集](docs/tree-collector.md)
- [収集 record と共通検証器](docs/records.md)
- [終局リターンと経路重み](docs/returns.md)
- [自己対戦データの保存と読込](docs/dataset-store.md)
- [チェックポイントと消費位置の保存](docs/checkpoints.md)
- [停止後の復旧と再試行](docs/recovery.md)

API の使い分けは [ドキュメント一覧](docs/README.md)、仮説・比較条件・今後の開発項目は [研究計画](docs/research/README.md) を参照してください。
