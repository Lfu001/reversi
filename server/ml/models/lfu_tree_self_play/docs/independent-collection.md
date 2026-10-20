# 独立自己対戦の収集

[← ドキュメント一覧](README.md)

`collector.py` は比較群 A の独立自己対戦を終局まで生成し、B/C と同じ形式の `CollectionRecord`（`mode="independent"`）、`ReturnTargets`、同一根からの複数経路に対する `1/K` 重み付け、および収集コストを提供します。

## 独立対局を1局収集する

```python
from lfu_tree_self_play.collector import IndependentCollector
from lfu_tree_self_play.config import experiment_id, load_config
from lfu_tree_self_play.game import ReversiPyGame
from lfu_tree_self_play.model import PolicyValueModel
from lfu_tree_self_play.roots import generate_root_position

config = load_config("configs/example.toml")
game = ReversiPyGame()
root = generate_root_position(seed=42)
collector = IndependentCollector(
    PolicyValueModel(),
    game,
    experiment_id=experiment_id(config),
    round_id="round-0",
    model_generation=0,
    run_seed=config.seeds.training[0],
)
unit = collector.collect_game(root, unit_id="game-0")
record = unit.record
targets = unit.targets
```

`IndependentCollector` は初期化時に `FrozenPolicySampler` でラウンド開始時の π_old を固定します。収集途中で元の `PolicyValueModel` を更新しても、collector が参照する π_old は変化しません。根には `GameState`、`RootPosition`、`RootEntry` のいずれも渡せます。

各非終局局面では固定モデルを1回だけ評価し、着手者視点の `old_value` と合法手マスク付き温度1方策からの `ActionSample`（`action`、`old_log_probability`）を同時に取得します。強制パスは `game.step` が解決し、全経路を実際の終局まで進めます。`H=0` の終局根では推論や環境遷移を行わず、終局ノード1個・辺0本の単位を返します。

行動抽出には `action_sampling` RNG ストリームだけを使い、`draw_index` を単調増加させます。`branch_position`、`sibling_rollout`、`evaluation` ストリームや global RNG は消費しません。入力エラーや実行中の例外が発生した場合は、`action_sampling` の RNG 状態と `draw_index` を呼出し前の状態に戻します。

## 同じ根から複数経路を収集して 1/K で重み付けする

同じ根局面から `K` 本の独立終局経路を生成する場合、各経路を完了単位として保持しつつ、根ごとの学習重みを `1/K` に揃えます。

```python
from pathlib import Path

from lfu_tree_self_play.collector import weight_independent_records
from lfu_tree_self_play.dataset_store import DatasetStore

collection = collector.collect_paths(
    root,
    paths_per_root=4,
    unit_id_prefix="root-0",
)
store = DatasetStore(
    Path("artifacts/dataset"),
    experiment_id=experiment_id(config),
    game=game,
)
identities = collection.publish_all(store)

loaded_records = [store.load(identity.unit_id).record for identity in identities]
weighted_targets = weight_independent_records(loaded_records, game)
```

- `unit.targets` は `DatasetStore.publish` が要求する単一単位の `ReturnTargets`（経路内の重み `1.0`）です。
- `unit.weighted_targets`（および `collection.weighted_targets`）は、同じ根から生成した `K` 本の経路に対して `node_weights` と `edge_weights` を `1/K` にスケールした `ReturnTargets` です。リターンと手番視点教師（`node_black_returns`、`edge_black_returns`、`node_player_targets`、`edge_parent_targets`）は変わりません。
- 保存済みの独立対局群を `DatasetStore` から読み戻した後は、`weight_independent_records(records, game)` で同じ `1/K` 重み付きターゲットを再構成できます。異なる根局面、異なるモデル世代・ラウンド、重複した `unit_id` や同一 draw イベントを含む集合、または `tree` モードの record は `ValueError` で拒否します。

## 計測されるコスト

各 `CollectionRecord.cost` には収集時の実測値を記録します。永続化や学習の時間は含みません。

| field | 内容 |
| --- | --- |
| `environment_transitions` | `game.step` を実行した回数（辺の数） |
| `evaluated_positions` | π_old で推論した非終局局面数 |
| `inference_batches` | π_old の推論 batch 実行回数 |
| `collection_seconds` | 単位の収集にかかった経過秒 |
| `inference_seconds` | 推論と行動抽出にかかった経過秒 |
| `transition_seconds` | 環境遷移にかかった経過秒 |
| `management_seconds` | record 構築・管理にかかった経過秒 |
| `peak_memory_bytes` | プロセス RSS（および利用時はデバイス割当）のピーク bytes |

## 検証する

パッケージのディレクトリで実行します。

```sh
uv run --locked pytest tests/test_independent_collector.py
```

終局までの完全生成、`validate_record` と `DatasetStore` の保存・読込 round-trip、`H=0` と強制パス、同一根からの複数経路生成と `1/K` 重み付け、収集中のモデル不変性、RNG ストリーム分離を検証します。要件の正本は [T14](research/milestones/02-collection.md#t14-aの独立自己対戦collector) です。
