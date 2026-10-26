# 停止後の復旧と再試行

dataset の公開と checkpoint の確定には atomic rename を使います。停止前に計算を終えていても、rename 前なら未確定です。公開済みデータと検証済み checkpoint を基準に再開します。

## 復元する

[収集と保存の例](collection.md) で保存した成果物を開き直します。[セットアップ](../../README.md#クイックスタート) 済みのパッケージのディレクトリで実行してください。保存時と同じ設定、dataset、optimizer・scheduler の種類と構成が必要です。この例では scheduler を使っていません。

以下を `recover.py` として保存します。新しいプロセスで実行でき、収集例の Python 変数は使いません。

```python
from pathlib import Path

from torch.optim import AdamW

from lfu_tree_self_play.checkpoints import CheckpointStore
from lfu_tree_self_play.config import experiment_id, load_config
from lfu_tree_self_play.dataset_store import DatasetStore
from lfu_tree_self_play.game import ReversiPyGame

config = load_config("configs/example.toml")
dataset = DatasetStore(
    Path("artifacts/dataset"),
    experiment_id=experiment_id(config),
    game=ReversiPyGame(),
)
dataset.quarantine_incomplete()
store = CheckpointStore(
    Path("artifacts/checkpoints"), config=config, dataset_store=dataset
)
verified = store.recover()
if verified is None:
    raise RuntimeError("確定済み checkpoint がありません。")
runtime = store.restore(
    verified,
    optimizer_factory=lambda model: AdamW(model.parameters()),
    scheduler_factory=lambda optimizer: None,
)
loaded = dataset.load("tree-0")
print(f"Checkpoint: {runtime.manifest.commit_id}")
print(f"Stage: {runtime.manifest.progress.stage}")
print(f"Loaded unit: {loaded.identity.unit_id}")
```

```sh
uv run --locked python recover.py
```

収集例の直後なら `Checkpoint: collected-0`、`Stage: collection`、`Loaded unit: tree-0` が表示されます。収集例は学習を行わないため、checkpoint の固定入力はまだ未消費です。復元された `runtime.manifest.inputs` と `runtime.manifest.progress` で、入力と確定済みの消費位置を確認できます。

`runtime` のモデル、optimizer、乱数、進捗を使って再開します。乱数を seed から作り直したり、モデルを初期化し直したりすると保存時点の続きになりません。

## 再開する範囲

以下は保存基盤で扱う確定境界です。学習・評価を含む処理ループ全体の再開は [開発計画の実行モデル](../development/execution-model.md) に定める計画であり、この復元例だけでは検証できません。

| 停止した処理 | 再開時の扱い |
| --- | --- |
| 収集中・dataset 公開前 | staging を隔離し、未完了の対局・木を再収集する |
| dataset 公開後 | 公開済み単位を使う。同じ ID・内容の再公開も可能 |
| 学習後・checkpoint 確定前 | 直前の確定モデルを復元し、未消費入力から更新し直す |
| checkpoint 確定後 | 更新済みモデルと確定済み消費を引き継ぐ |
| 評価結果の確定前 | 未確定の評価だけをやり直す |
| 評価結果の確定後 | 保存済み原結果を使う |

入力と消費位置は checkpoint の manifest で確認します。dataset の列挙順から消費済み範囲を推測しないでください。同じ ID で内容を変えた再公開・再確定は拒否されます。外部評価サービスへの呼出し自体を一度だけにする保証はありません。

## 復旧できない場合

破損した最新 checkpoint が使えない場合は、検証済みの前の snapshot に戻ります。復旧した commit ID と進捗を確認してください。履歴があるのに有効な snapshot がなければ失敗します。確定履歴は自動削除されません。

設定不一致や、CUDA device 数・MPS 利用可否の違いは自動補正しません。元の設定・環境を用意してください。manifest version 2・runtime JSON version 1 が対象で、`runtime.pt` を使う旧形式の自動変換はありません。

保証は、協調する writer が `flock`・atomic rename・`fsync` を使える POSIX ローカルファイルシステムを前提とします。network filesystem、外部からのファイル変更、物理故障や電源断への耐性は保証しません。
