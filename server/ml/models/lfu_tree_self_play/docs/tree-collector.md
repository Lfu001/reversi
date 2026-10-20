# B/C 共通の木状データ収集

[← ドキュメント一覧](README.md)

`TreeCollector` と `collect_tree` は、収集ラウンド開始時に固定した `PolicyValueModel`（$\pi_{\text{old}}$）から幅 4・最大 2 段の浅い分岐木を生成し、全枝を実際の終局まで進めて検証済みの完了単位を作ります。比較群 B と C は同じ木構造を共有し、学習時の advantage 計算だけを分けます。

```python
from lfu_tree_self_play.game import ReversiPyGame
from lfu_tree_self_play.model import PolicyValueModel
from lfu_tree_self_play.rng import create_rng_streams
from lfu_tree_self_play.roots import generate_root_position
from lfu_tree_self_play.tree_collector import TreeCollector

game = ReversiPyGame()
streams = create_rng_streams(87)
collector = TreeCollector(
    PolicyValueModel(),
    experiment_id="exp-87",
    round_id="round-0",
    model_generation=0,
    run_seed=87,
    streams=streams,
    levels=2,
    strategy="uniform",
    game=game,
)

root = generate_root_position(87)
tree = collector.collect_tree(root, unit_id="tree-0")
print(tree.planned_branch_count, tree.executed_branch_count, tree.terminal_leaf_count)
```

## 固定方策と乱数系列の分離

`TreeCollector` は初期化時に `PolicyValueModel` をコピーし、評価モード・勾配なしで保持します。収集途中に元のモデルを更新しても $\pi_{\text{old}}$ は変化せず、元のモデルの訓練モードやグローバル RNG も変更しません。

乱数系列は用途ごとに分離します。

| 用途 | 使用する stream | 役割 |
| --- | --- | --- |
| 分岐位置抽選 | `streams.branch_position` | 木の生成前に `sample_branch_plan` で予定位置を 1 回だけ抽選 |
| 根経路・初回分岐の行動抽出 | `streams.action_sampling` | 根からの共通経路と最初の分岐での温度 1 復元抽出 |
| 兄弟標本の継続試行 | `streams.sibling_rollout` | 各兄弟標本の `continuation_seed` を発行し、標本ごとの独立 RNG で終局まで継続 |

同じラウンドで複数の木を収集する場合は、同じ `TreeCollector`（または共有した `RNGStreams`）を使って `collect_tree` を順に呼び出します。不正な入力は RNG を進める前に `ValueError` で拒否します。

## 分岐・同一手標本・予定と実行の記録

各非終局ノードでは、強制パスと終局を解決した `GameState` に対して 1 回の推論で着手者視点の `old_value` と `policy_logits` を計算します。

- 石を置く直前の `placement_depth` が予定位置に達していれば、同じ親局面から幅 4 の行動標本を復元抽出します。
- 同じ手が複数回抽出された兄弟標本も 1 本にまとめず、別々の `EdgeRecord`・子 `NodeRecord`・`sample_index`・`continuation_seed` を持つ独立した終局試行として保持します。実際に異なる行動の数は `SiblingGroupRecord.distinct_action_count` に記録します。
- 予定位置より前に終局した経路では未到達の分岐を実行せず、位置の移動・補充・再抽選もしません。$H=0$（終局根）では推論も分岐も行わず、単一終局ノードの木を返します。

戻り値の `CollectedTree` は以下を保持します。

| 属性 / プロパティ | 内容 |
| --- | --- |
| `record` | `validate_record` を通過した `mode="tree"` の `CollectionRecord` |
| `targets` | `aggregate_returns` で計算した終局リターンと経路重み (`ReturnTargets`) |
| `branch_execution` | `summarize_branch_execution` による予定位置・予定分岐群数・実行分岐群数 (`BranchExecution`) |
| `planned_branch_count` | 全経路が予定位置へ到達した場合の予定分岐群数（2 段なら最大 `1 + 4 = 5`） |
| `executed_branch_count` | 実際に生成された兄弟群数 (`len(record.sibling_groups)`) |
| `terminal_leaf_count` | 完了した木の終局葉ノード数（2 段なら最大 16） |

完了した木をそのまま `DatasetStore` へ atomic に公開する場合は `collector.collect_and_publish(root, store, unit_id="...")` を使います。

## 検証

パッケージのディレクトリで実行します。

```sh
uv run --locked pytest tests/test_tree_collector.py
```

要件の正本は [T15](research/milestones/02-collection.md#t15-bc共通の木状collector) です。
