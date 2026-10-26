# 期待リターンと方策勾配の検証

列挙可能な終盤局面で、固定方策の正確な期待リターンと Monte Carlo 推定、黒・白それぞれの条件付き方策勾配を照合します。
importance ratio、leave-one-out advantage、共有経路と再計算の生成データ・重み・policy/value loss も検査します。

## 実行と結果の保存

[セットアップ](../../README.md#クイックスタート) 後、パッケージのディレクトリで実行します。

```sh
mkdir -p artifacts
uv run --locked lfu-tree-self-play verify-policy-gradient --output artifacts/policy-gradient-verification.json
```

成功すると `Policy gradient verification: PASS` が表示され、`artifacts/policy-gradient-verification.json` に推定値、標準誤差、許容誤差、各条件の合否を保存します。`FAIL` の場合も JSON の各条件を確認してください。
検証成功は終了コード 0、検証不合格は 1、引数や保存先のエラーは 2 です。

レポートは実行ごとの生成物です。`artifacts/` は Git 管理から除外します。
同じコード・依存環境・seed・試行数で結果を再生成できます。試行数と seed は `--pairs`・`--seed` で指定します。
検証結果を記録する際は、コミット、実行コマンド、seed、試行数と合否を残します。

このコマンドは列挙可能な局面での数値検証です。実際の収集・学習・対戦評価の一巡や研究仮説の成立を検証するものではありません。次工程への接続条件は [検証ゲート](../development/verification-gates.md) を参照してください。

## 利用ガイドの例を検証する

```sh
uv run --locked pytest tests/test_documentation.py -q
```

収集、batch 推論、復元の Python 例を一時ディレクトリで実行し、保存データ、計測結果と復元した状態を確認します。成功すると pytest が `1 passed` と表示します。

## 保存基盤の復旧を検証する

パッケージのディレクトリで実行します。

```sh
uv run --locked pytest tests/test_recovery.py -q
```

復旧テストは別プロセスを強制終了し、再開後の dataset、モデル、optimizer、scheduler、RNG、確定消費・評価を、停止しない場合と比較します。pytest が失敗なしで終了すれば成功です。プロセス停止からの復旧を検査し、実際の自己対戦収集・学習ループ・対戦評価や電源断の動作は検査しません。

これらの成功だけで [G1](../development/verification-gates.md#g1データ生成と永続化) 全体の通過とはみなしません。収集データの意味検証や処理ループとの接続も確認する必要があります。全テストの実行方法は [README の開発手順](../../README.md#開発) を参照してください。
