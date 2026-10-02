# 期待リターンと方策勾配の検証

列挙可能な終盤局面で、固定方策の正確な期待リターンと Monte Carlo 推定、黒・白それぞれの条件付き方策勾配を照合します。
importance ratio、leave-one-out advantage、共有経路と再計算の生成データ・重み・policy/value loss も検査します。
これは研究計画の [G0：ゲームと数理の意味検証](verification-gates.md#g0ゲームと数理の意味検証) に対応する数理検証です。
G0 はこの計画内のゲート番号です。

## 実行と結果の保存

パッケージのディレクトリで実行します。

```sh
mkdir -p artifacts
uv run --locked lfu-tree-self-play verify-policy-gradient --output artifacts/policy-gradient-verification.json
```

Python API は `lfu_tree_self_play.verification.verify_policy_gradient` です。
既定では各色 3,000 組の独立した兄弟試行を使い、`--pairs` と `--seed` で変更できます。
JSON レポートには推定値、標準誤差、許容誤差、各条件の合否を保存します。
検証成功は終了コード 0、検証不合格は 1、引数や保存先のエラーは 2 です。

レポートは実行ごとの生成物です。`artifacts/` は Git 管理から除外します。
CI は毎回レポートを生成し、`policy-gradient-verification` という Actions artifact として保存します。
検証結果を共有する際は、対応する CI run と artifact へのリンクを記録します。

ゲーム規則の条件は `uv run --locked pytest -q` のゲーム契約テストで検証します。
