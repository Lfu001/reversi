# 複数の対局・木をまとめて推論する

[収集と保存の手順](collection.md) を先に実行し、`artifacts/roots.json` を用意してください。同じパッケージのディレクトリで、未使用の `artifacts/batch-dataset/` と `artifacts/batch-metrics.json` を保存先にします。

この例は CPU 上で新しい小型の未学習モデルを作り、保存済みの根から独立対局を2件、木を1件生成します。前のガイドの Python 変数は使わないため、新しいプロセスで実行できます。

## batch 推論で収集する

A の独立対局と B/C の木は、同じ凍結モデルと scheduler を共有できます。乱数を持つ collector ごとに lane を作り、lane 内の対局・木の順序を保って、異なる lane の次の局面をまとめます。同じ collector を複数 lane へ登録すると、乱数順が曖昧になるため拒否します。比較群には同じ batch 上限、同時進行単位数、メモリ制限を与えます。

学習直後のモデルを渡す場合も、呼び出し側で `.eval()` する必要はありません。`FrozenBatchEvaluator` は評価モードのコピーを作り、元の学習モデルを保持します。共通推論関数 `evaluate_model()` を直接使う場合も、推論中だけ評価モードに切り替え、例外発生時を含めて各層の元のモードを復元します。推論によって BatchNorm の統計や学習済みの重み・勾配は更新しません。

```python
import json
from dataclasses import asdict
from pathlib import Path

from lfu_tree_self_play.batch_inference import FrozenBatchEvaluator
from lfu_tree_self_play.batch_scheduler import BatchScheduler, SchedulerLimits
from lfu_tree_self_play.collection_session import CollectionLane
from lfu_tree_self_play.collector import IndependentCollector
from lfu_tree_self_play.config import experiment_id, load_config
from lfu_tree_self_play.dataset_store import DatasetStore
from lfu_tree_self_play.game import ReversiPyGame
from lfu_tree_self_play.model import PolicyValueConfig, PolicyValueModel
from lfu_tree_self_play.roots import RootRegistry
from lfu_tree_self_play.tree_collector import TreeCollector

config = load_config("configs/example.toml")
game = ReversiPyGame()
model = PolicyValueModel(PolicyValueConfig(width=4, num_blocks=1))
registry = RootRegistry.load("artifacts/roots.json", config)
root = registry.positions("training")[0]
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
print(f"Stop reason: {batch_metrics.stop_reason}")
print(f"Completed units: {batch_metrics.completed_units}")
```

上のブロック全体を `collect_batch.py` として保存し、実行します。

```sh
uv run --locked python collect_batch.py
```

成功すると `Stop reason: complete` と `Completed units: 3` が表示されます。`artifacts/batch-dataset/` に `batched-game-0`、`batched-game-1`、`batched-tree` が公開され、計測結果が `artifacts/batch-metrics.json` に保存されます。この例は checkpoint の保存と学習を行いません。

## メモリ上限と停止時の扱い

この例の上限値は実行環境に応じて調整します。accelerator では `device_limit_bytes` も指定します。RSS は現在のプロセス使用量、device 側は生存 tensor の割当量を別々に確認します。上限超過・計測失敗時は実行を止め、未完成単位を破棄してその乱数状態を開始時点へ戻します。公開済みの完成単位は残ります。判定は admission・推論・公開の境界で行うため、処理途中の一時ピークを抑える OS の hard cap ではありません。

## 計測結果と学習用の重み

batch サイズ分布、各要求の待機時間、物理的な推論回数、RSS/device の観測ピークは別の計測ファイルへ保存し、既存 dataset の schema は変えません。RSS の取得は psutil、accelerator の割当量・ピーク・同期は PyTorch の統一 API に任せます。`device_peak_bytes` はプロセスのピーク counter で、CPU 実行時は `null` です。単位の `inference_batches` は参加した batch 数なので、複数単位の合計を物理的な推論回数として使わず、`batch_metrics.inference_batches` を使います。推論時間は batch 内の行数で等分し、待機時間を含めません。

同じ根から得た A の K 経路は `unit.weighted_targets` で各経路を `1/K` に重み付けできます。上の公開例は既存と同じ単位内の `targets` を保存するため、根ごとの経路重みは利用側で別に適用します。途中で永続化に失敗した完成単位の再公開は、同じ完成値を保持した利用側が既存の保存 API で行います。
