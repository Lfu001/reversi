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

`check-config` は TOML の型と seed 集合を検証し、experiment ID を表示します。
ファイル不在、読込失敗、文字コード・構文エラーは標準エラー出力に表示し、
終了コード 2 で終了します。設定エラーも同じ終了コードです。

設定は `[experiment]` の `name` と、`[seeds]` の `training`、`tuning`、
`final_evaluation` を必須とします。各 seed 集合は空にできず、0 以上の整数を
重複なく指定します。学習・調整・最終評価の集合間でも seed を共有できません。
読み込んだ設定は読み取り専用の OmegaConf `DictConfig` として扱います。
seed 集合の記載順と TOML の整形は experiment ID に影響しません。

`create_rng_streams(run_seed)` は位置抽選、行動抽出、兄弟の継続試行、評価用の
独立した `random.Random` を返します。別の RNG 実装が必要な場合は
`derive_seed(run_seed, purpose)` で同じ用途別 seed を取得できます。

## Policy/value モデル

`PolicyValueModel` は `GameState.observation` と同じ4面を `(batch, 4, 8, 8)` の
float32 tensor で受け取り、`policy_logits` `(batch, 64)` と着手者視点の
`value` `(batch,)` を返します。残差ブロックは Transformers の
`ResNetBasicLayer` を4個使い、盤面の8×8を維持します。

合法手確率が必要なときは、観測の第4面を `(batch, 64)` の bool mask に変換し、
`masked_policy_logits(policy_logits, legal_mask).softmax(dim=-1)` を使います。
終局局面には合法手がないため、この関数は `ValueError` を返します。
モデル重みの保存と読込には `save_model(model, path)` と `load_model(path)` を使います。
読込後のモデルは CPU 上に置かれます。

## テスト

```sh
uv run --locked pytest -q
```
