# 期待リターンと方策勾配の検証

列挙可能な終盤局面で、固定方策の正確な期待リターンと Monte Carlo 推定、黒・白それぞれの条件付き方策勾配を照合します。
importance ratio、leave-one-out advantage、共有経路と再計算の生成データ・重み・policy/value loss も検査します。

## 実行と結果の保存

パッケージのディレクトリで実行します。

```sh
mkdir -p artifacts
uv run --locked lfu-tree-self-play verify-policy-gradient --output artifacts/policy-gradient-verification.json
```

JSON レポートには推定値、標準誤差、許容誤差、各条件の合否を保存します。
検証成功は終了コード 0、検証不合格は 1、引数や保存先のエラーは 2 です。

レポートは実行ごとの生成物です。`artifacts/` は Git 管理から除外します。
同じコード・依存環境・seed・試行数で結果を再生成できます。試行数と seed は `--pairs`・`--seed` で指定します。
検証結果を記録する際は、コミット、実行コマンド、seed、試行数と合否を残します。

## パッケージのテスト

パッケージのディレクトリで実行します。

```sh
uv run --locked pytest tests/test_recovery.py -q
```

復旧テストは別プロセスを強制終了し、再開後の dataset、モデル、optimizer、scheduler、RNG、確定消費・評価を、停止しない場合と比較します。プロセス停止からの復旧を検査し、実際の自己対戦収集・学習ループ・対戦評価や電源断の動作は検査しません。
