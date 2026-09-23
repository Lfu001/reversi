# lfu_tree_self_play

木状自己対戦実験の独立した Python パッケージです。現在は T01 の基盤として、
TOML 設定の読込、最小 CLI、高速な単体テストを提供します。
研究計画は [docs](docs/README.md)、合格条件は [T01](docs/milestones/00-foundation.md#t01-新パッケージとテスト基盤) を参照してください。

## セットアップと実行

このディレクトリで実行します。通常版 CPython 3.14 と uv を使用します。
`.python-version` は 3.14 系を指定し、依存パッケージは `uv.lock` で固定しています。
旧実装、PyTorch、GPU、Rust 拡張は必要ありません。

```sh
uv sync --locked
uv run --locked python -c "import lfu_tree_self_play"
uv run --locked lfu-tree-self-play check-config configs/example.toml
uv run --locked python -m lfu_tree_self_play check-config configs/example.toml
```

`check-config` は TOML を読み込めれば終了コード 0 で成功を表示します。
ファイル不在、読込失敗、文字コード・構文エラーは標準エラー出力に表示し、
終了コード 2 で終了します。設定項目の意味や学習実行の可否は検証しません。

## テストと配布物の検証

```sh
uv run --locked pytest --collect-only -q
uv run --locked pytest -q
bash scripts/check-wheel.sh
```

`check-wheel.sh` は wheel をビルドし、一時ディレクトリの新規仮想環境へ
インストールします。ソースツリー外で import、サンプル設定の読込、両方の
CLI 入口を確認し、終了時に一時ディレクトリを削除します。
同じ検証を GitHub Actions の `test (models)` で実行します。

## 設定の責務

`lfu_tree_self_play.config.load_config(path)` は TOML の値を辞書として返します。
ファイル・文字コード・構文の例外は呼出側へ伝えます。
設定内容の型検証、既定値や上書き、設定ハッシュ、seed 分離は T02 の責務です。
OmegaConf は今回の依存に含めません。後続で導入する場合も、この辞書を
`OmegaConf.create()` に渡せるため、TOML 形式を維持できます。
