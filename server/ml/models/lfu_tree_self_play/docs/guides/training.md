# 保存済みの収集世代を1 epoch学習する

[収集と保存](collection.md) の `artifacts/` を使い、固定された入力を一度だけ学習します。モデル、optimizer、scheduler、乱数、入力の消費記録、更新指標を同じ checkpoint に確定します。

## 学習と再開

パッケージのディレクトリで、以下を `train.py` として保存してください。optimizer の種類は収集時と同じものを指定します。保存済みの学習率や状態は checkpoint から復元されます。

```python
from pathlib import Path

from accelerate import Accelerator
from torch.optim import AdamW

from lfu_tree_self_play.checkpoints import CheckpointStore
from lfu_tree_self_play.config import experiment_id, load_config
from lfu_tree_self_play.dataset_store import DatasetStore
from lfu_tree_self_play.game import ReversiPyGame
from lfu_tree_self_play.trainer import TrainingOptions, train_one_epoch

config = load_config("configs/example.toml")
dataset = DatasetStore(
    Path("artifacts/dataset"), experiment_id=experiment_id(config), game=ReversiPyGame()
)
store = CheckpointStore(
    Path("artifacts/checkpoints"), config=config, dataset_store=dataset
)
result = train_one_epoch(
    store,
    optimizer_factory=lambda model: AdamW(model.parameters()),
    options=TrainingOptions(advantage="standard"),
    accelerator=Accelerator(cpu=True),
)
print(f"Generation: {result.manifest.progress.model_generation}")
print(f"Consumed: {result.manifest.progress.consumed_unit_ids}")
print(result.report["metrics"]["after"])
```

```sh
uv run --locked python train.py
```

成功すると世代が `1`、消費済み入力が `('tree-0',)` になります。更新前後の指標は checkpoint 内の `training.json` に保存され、その checksum も manifest の検証対象です。同じコマンドを再実行すると、確定済みの結果を返します。

中断した場合も同じコマンドで再開します。学習開始時に入力と条件を固定した checkpoint を作り、更新の確定前ならそこから1 epochをやり直します。確定後なら再更新しません。再開時に advantage、損失係数、根のグループ、デバイス、精度を変更すると拒否されます。実測時間は再試行ごとに変わります。

次の世代は、更新済みモデルを [復元](recovery.md) して収集し、新しい入力を `pin_inputs` で固定した checkpoint を保存してから学習します。古い世代の manifest は再消費できません。

## 更新単位と指標の読み方

入力範囲の各 unit を一度だけ逆伝播し、勾配を合計して optimizer を1回更新します。損失は固定定数60と根数で正規化します。共有辺の重複展開は行わず、BatchNorm は収集時と同じ eval mode に保ちます。scheduler を使う場合は、引数なしの `step()` に対応するものを指定し、更新後に1回進めます。

各木と独立対局は既定で1根です。同じ根から収集したK本の独立経路は `root_groups=(("path-0", "path-1"), ...)` と明示すると、各経路の重みが1/Kになります。グループは manifest の入力を過不足なく一度ずつ含める必要があります。同一盤面という理由だけでは自動的にまとめません。

`before` と `after` は更新前後の値です。KL は更新前の全合法手分布から現在の分布への `KL(old || current)`、entropy は現在の合法手分布のエントロピー、value誤差は教師との二乗誤差です。いずれも非終局ノードの経路重みで平均します。clip率は保存された行動の確率比が `[1−ε, 1＋ε]` の外に出た割合を辺の経路重みで平均します。`gradient_norm` は更新前の合計勾配のL2 normです。

現在の確定境界は単一プロセスです。デバイス配置、autocast、逆伝播を Accelerate に委ね、FP32 と bfloat16 に対応します。分散実行と FP16 は、rank別の乱数や scaler の確定保存が未対応のため開始時に拒否します。停止・再開で同じ更新を得るには、同じ実行環境と決定的な演算が必要です。CPU FP32 の実プロセス停止・再開は `tests/test_trainer.py` で検証します。
