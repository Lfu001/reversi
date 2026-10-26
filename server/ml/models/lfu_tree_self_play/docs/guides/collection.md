# 自己対戦データの収集と保存

[セットアップ](../../README.md#クイックスタート) 後、小型の未学習モデルで自己対戦木を一つ収集し、根局面、dataset、checkpoint を保存します。学習・対戦評価は含みません。

## 収集して保存する

パッケージのディレクトリで、未使用の `artifacts/` を保存先にします。以下を `collect.py` として保存してください。

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
print(f"Published unit: {identity.unit_id}")
print(f"Checkpoint: {snapshot.commit_id}")
```

```sh
uv run --locked python collect.py
```

成功すると次の表示が得られます。

```text
Published unit: tree-0
Checkpoint: collected-0
```

`artifacts/roots.json` に固定した根集合、`artifacts/dataset/` に公開済みの木、`artifacts/checkpoints/` にモデルと実行状態が保存されます。保存済みの内容を読み直すには [復旧手順](recovery.md) を使います。

## 保存したデータを使うとき

根集合は比較条件の一部として保存し、A/B/Cで共有します。重複を除外した集合では採用される手数・盤面の分布が一様にならず、根の分離だけで学習中に到達する全局面との非重複は保証できません。

この checkpoint は収集後のモデルと乱数、固定入力を保存したもので、入力はまだ未消費です。学習を接続するときは、更新済みのモデルとその入力の消費を同じ checkpoint で確定します。学習への接続条件は [検証ゲート](../development/verification-gates.md#g1データ生成と永続化) を参照してください。

ラウンド中は同じ collector を使い、新しいラウンドでは更新後のモデルから作り直します。再開時は [復旧手順](recovery.md) に従います。

複数の対局・木をまとめて収集する場合は、保存した根集合を使って [batch 推論による収集](batch-collection.md) へ進めます。
