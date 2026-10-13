# 固定方策からの行動抽出

[← ドキュメント一覧](README.md)

`FrozenPolicySampler` は収集ラウンド開始時の `PolicyValueModel` をコピーし、評価モードと勾配なしで推論します。元のモデルの重みや BatchNorm の統計を後から更新しても、sampler が使う π_old は変わりません。元のモデルの訓練モードや勾配設定も変更しません。新しいラウンドでは、新しい sampler を作ります。

```python
from lfu_tree_self_play.game import ReversiPyGame
from lfu_tree_self_play.model import PolicyValueModel
from lfu_tree_self_play.rng import create_rng_streams
from lfu_tree_self_play.sampler import FrozenPolicySampler

game = ReversiPyGame()
state = game.initial_state()
streams = create_rng_streams(83)
sampler = FrozenPolicySampler(PolicyValueModel())
draws = sampler.sample(state, streams.action_sampling, count=4)

for draw in draws:
    action = draw.action
    old_log_probability = draw.old_log_probability
    legal_mask = draw.legal_mask
    child = game.step(state, action)
```

同じ局面の合法手 logits に温度1の softmax を適用し、`count` 回の復元抽出を行います。戻り値は抽出順の `tuple[ActionSample, ...]` です。同じ手が何度出ても別の標本として保持します。各標本の `legal_mask` は64要素の読み取り専用 bool tuple、`old_log_probability` は抽出分布に対応する Python float です。旧確率には勾配を流しません。

抽出には呼出側が渡す `random.Random` だけを使います。位置抽選には `streams.branch_position`、通常の行動抽出には `streams.action_sampling`、兄弟の継続試行には各標本専用の RNG を使い分けてください。途中から再開する場合は RNG の `getstate()` / `setstate()` を使います。sampler 自体は stream ID、draw index、continuation seed、モデル世代を管理しません。

## 推論済み logits を使う

`sample_policy(policy_logits, legal_mask, rng, count=1)` は一局面の `(64,)` の浮動小数点 logits と `(64,)` の bool mask を受け取り、同じ `ActionSample` の tuple を返します。推論をまとめる呼出側は、固定 π_old の batch 出力を局面ごとに渡せます。推論モデルの固定は呼出側が担います。

合法手 logits は有限値である必要があります。非合法手 logits はマスクで除外します。確率と log probability は logits を detach して CPU の float64 に変換し、既存の `masked_policy_logits` と `log_softmax` で計算します。極端に小さい確率は浮動小数点の範囲外で0になる場合があります。温度の変更、探索ノイズ、非復元抽出は提供しません。

終局局面、合法手のない mask、不正な入力形状・型、非有限の合法手 logits、正の整数以外の `count` は `ValueError` になります。これらの拒否で行動抽出 RNG は進みません。強制パスはゲームアダプターで解決した局面を渡し、pass action は抽出しません。

## record に保存する

`ActionSample.action` と `old_log_probability` はそれぞれ `EdgeRecord.action` と `EdgeRecord.old_log_probability` に保存します。使用した mask は親の `NodeRecord.state.observation` の第4面として保存済みなので、`ActionSample.legal_mask` と一致する親局面を記録してください。既存の record schema は変更しません。標本ごとに別の辺・子ノードを作り、重複した行動の兄弟をまとめません。

方策比を計算する際は、保存した mask と温度1を新方策にも適用します。更新前の同じ logits から計算した新方策確率と、保存した旧 log probability の比は1になります。

## 検証する

パッケージのディレクトリで実行します。

```sh
uv run --locked pytest tests/test_sampler.py
```

確率 `1/4`・`3/4` の合法手を20,000回抽出する固定seedの頻度試験、非合法手の排除、重複保持、log probability と方策比、RNG の分離・復元、モデル固定、終局・不正入力の拒否を検証します。
