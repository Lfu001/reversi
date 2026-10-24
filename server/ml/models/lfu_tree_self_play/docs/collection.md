# 自己対戦データの収集と保存

セットアップ後、パッケージのディレクトリで実行する例です。空の `artifacts/` を使います。小型の未学習モデルで、根局面の固定、木の収集、dataset の公開、checkpoint の保存までを行います。学習・対戦評価は含みません。

```python
from pathlib import Path

from torch.optim import AdamW

from lfu_tree_self_play.checkpoints import CheckpointStore, Progress
from lfu_tree_self_play.config import experiment_id, load_config
from lfu_tree_self_play.dataset_store import DatasetStore
from lfu_tree_self_play.game import ReversiPyGame
from lfu_tree_self_play.model import PolicyValueConfig, PolicyValueModel
from lfu_tree_self_play.rng import create_rng_streams
from lfu_tree_self_play.roots import generate_root_registry
from lfu_tree_self_play.tree_collector import TreeCollector

config = load_config("configs/example.toml")
run_seed = config.seeds.training[0]
game = ReversiPyGame()
model = PolicyValueModel(PolicyValueConfig(width=4, num_blocks=1))
optimizer = AdamW(model.parameters())
streams = create_rng_streams(run_seed)

Path("artifacts").mkdir(exist_ok=True)
registry = generate_root_registry(config, roots_per_seed=1)
registry.save("artifacts/roots.json")
root = registry.positions("training")[0]
dataset = DatasetStore(
    Path("artifacts/dataset"), experiment_id=experiment_id(config), game=game
)
collector = TreeCollector(
    model,
    experiment_id=experiment_id(config),
    round_id="round-0",
    model_generation=0,
    run_seed=run_seed,
    streams=streams,
    game=game,
)
tree = collector.collect_tree(root, unit_id="tree-0")
identity = dataset.publish(tree.record, tree.targets)

store = CheckpointStore(
    Path("artifacts/checkpoints"), config=config, dataset_store=dataset
)
snapshot = store.commit(
    "collected-0",
    model=model,
    optimizer=optimizer,
    scheduler=None,
    rng=streams,
    progress=Progress(
        model_generation=0,
        stage="collection",
        cumulative_cost=tree.record.cost.collection_seconds,
    ),
    inputs=store.pin_inputs(0),
    expected_parent=None,
)
```

根集合は比較条件の一部として保存し、A/B/Cで共有します。重複を除外した集合では採用される手数・盤面の分布が一様にならず、根の分離だけで学習中に到達する全局面との非重複は保証できません。

この checkpoint は収集後のモデルと乱数、固定入力を保存したもので、入力はまだ未消費です。学習を接続するときは、更新済みのモデルとその入力の消費を同じ checkpoint で確定します。学習への接続条件は [検証ゲート](research/verification-gates.md#g1データ生成と永続化) を参照してください。

ラウンド中は同じ collector を使い、新しいラウンドでは更新後のモデルから作り直します。再開時は [復旧手順](recovery.md) に従います。

## 複数の対局・木をまとめて推論する

A の独立対局と B/C の木は、同じ凍結モデルと scheduler を共有できます。乱数を持つ collector ごとに lane を作り、lane 内の対局・木の順序を保って、異なる lane の次の局面をまとめます。同じ collector を複数 lane へ登録すると、乱数順が曖昧になるため拒否します。比較群には同じ batch 上限、同時進行単位数、メモリ制限を与えます。

```python
import json
from dataclasses import asdict

from lfu_tree_self_play.batch_inference import FrozenBatchEvaluator
from lfu_tree_self_play.batch_scheduler import BatchScheduler, SchedulerLimits
from lfu_tree_self_play.collection_session import CollectionLane
from lfu_tree_self_play.collector import IndependentCollector

batch_evaluator = FrozenBatchEvaluator(model, model_generation=0)
independent_lane = IndependentCollector(
    batch_evaluator,
    game,
    experiment_id=experiment_id(config),
    round_id="round-0",
    model_generation=0,
    run_seed=10,
)
tree_lane = TreeCollector(
    batch_evaluator,
    game=game,
    experiment_id=experiment_id(config),
    round_id="round-0",
    model_generation=0,
    run_seed=11,
    levels=1,
)


def tree_sessions():
    yield tree_lane.create_session(root, unit_id="batched-tree")


batch_store = DatasetStore(
    Path("artifacts/batch-dataset"), experiment_id=experiment_id(config), game=game
)
batch_metrics = BatchScheduler(
    batch_evaluator,
    limits=SchedulerLimits(
        max_active_units=2,
        max_batch_size=2,
        rss_limit_bytes=2 * 1024**3,
    ),
).run(
    [
        CollectionLane(
            independent_lane,
            independent_lane.iter_sessions(
                root,
                paths_per_root=2,
                unit_ids=["batched-game-0", "batched-game-1"],
            ),
        ),
        CollectionLane(tree_lane, tree_sessions()),
    ],
    on_complete=lambda unit: batch_store.publish(unit.record, unit.targets),
)
Path("artifacts/batch-metrics.json").write_text(json.dumps(asdict(batch_metrics)))
```

この例の上限値は実行環境に応じて調整します。accelerator では `device_limit_bytes` も指定します。RSS は現在のプロセス使用量、device 側は生存 tensor の割当量を別々に確認します。上限超過・計測失敗時は実行を止め、未完成単位を破棄してその乱数状態を開始時点へ戻します。公開済みの完成単位は残ります。判定は admission・推論・公開の境界で行うため、処理途中の一時ピークを抑える OS の hard cap ではありません。

batch サイズ分布、各要求の待機時間、物理的な推論回数、RSS/device の観測ピークは別の計測ファイルへ保存し、既存 dataset の schema は変えません。RSS の取得は psutil、accelerator の割当量・ピーク・同期は PyTorch の統一 API に任せます。`device_peak_bytes` はプロセスのピーク counter で、CPU 実行時は `null` です。単位の `inference_batches` は参加した batch 数なので、複数単位の合計を物理的な推論回数として使わず、`batch_metrics.inference_batches` を使います。推論時間は batch 内の行数で等分し、待機時間を含めません。

同じ根から得た A の K 経路は `unit.weighted_targets` で各経路を `1/K` に重み付けできます。上の公開例は既存と同じ単位内の `targets` を保存するため、根ごとの経路重みは利用側で別に適用します。途中で永続化に失敗した完成単位の再公開は、同じ完成値を保持した利用側が既存の保存 API で行います。
