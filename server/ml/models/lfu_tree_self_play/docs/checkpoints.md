# Checkpoint・消費位置の整合保存

モデル、optimizer、scheduler、乱数状態、入力の消費位置、評価結果を一つの snapshot として保存します。モデル更新と入力消費を同時に確定するため、停止後に同じ入力を二重に学習することを防げます。

## 初期状態を保存する

パッケージのディレクトリで実行する例です。設定を変える場合は別の保存先を使ってください。

```python
from pathlib import Path

from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR

from lfu_tree_self_play.checkpoints import CheckpointStore, Progress
from lfu_tree_self_play.config import experiment_id, load_config
from lfu_tree_self_play.dataset_store import DatasetStore
from lfu_tree_self_play.game import ReversiPyGame
from lfu_tree_self_play.model import PolicyValueModel
from lfu_tree_self_play.rng import create_rng_streams

config = load_config(Path("configs/example.toml"))
dataset = DatasetStore(
    Path("artifacts/dataset"),
    experiment_id=experiment_id(config),
    game=ReversiPyGame(),
)
store = CheckpointStore(
    Path("artifacts/checkpoints"), config=config, dataset_store=dataset
)
model = PolicyValueModel()
optimizer = Adam(model.parameters())
scheduler = StepLR(optimizer, step_size=1)
streams = create_rng_streams(config.seeds.training[0])

initial = store.commit(
    "genesis",
    model=model,
    optimizer=optimizer,
    scheduler=scheduler,
    rng=streams,
    progress=Progress(model_generation=0, stage="collection", cumulative_cost=0.0),
    inputs=store.pin_inputs(0),
    expected_parent=None,
)
```

初回の `expected_parent` は `None` です。以後は直前の commit ID を指定します。別 writer が先に commit した場合は stale parent conflict になり、最新状態の確認が必要です。

commit ID は `[A-Za-z0-9][A-Za-z0-9_.-]{0,127}` に一致する文字列を使います。保存先は設定に固定され、異なる設定では開けません。再開する場合は初期モデルを再生成せず、[recover と restore](recovery.md#モデル状態を復元する) を使います。

## 学習更新を確定する

学習前に `inputs = store.pin_inputs(generation, start=0, stop=None)` で対象を固定し、`inputs.selected` の単位を読み込みます。その入力で更新したモデル・optimizer・scheduler と、更新後の乱数状態を `commit` に渡します。以下は更新済みの状態を受け取る関数で、学習計算自体は呼出側で行います。選択入力が空の場合は学習 commit を作れません。

```python
def commit_update(
    store, previous, inputs, model, optimizer, scheduler, streams, cumulative_cost
):
    generation = previous.progress.model_generation + 1
    selected_ids = tuple(unit.unit_id for unit in inputs.selected)
    return store.commit(
        f"update-{generation}",
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        rng=streams,
        progress=Progress(
            model_generation=generation,
            stage="learning",
            cumulative_cost=cumulative_cost,
            consumed_unit_ids=previous.progress.consumed_unit_ids + selected_ids,
            evaluations=previous.progress.evaluations,
        ),
        inputs=inputs,
        expected_parent=previous.commit_id,
    )
```

## 入力・進捗・評価の契約

`InputManifest(model_generation, units, start, stop)` は generation 内のソート済み
`UnitIdentity` 全体と half-open 範囲を固定します。`selected` はこの固定 tuple の slice。
後から辞書順で前に入る unit が公開されても入力範囲は移動しません。各 unit の ID、
experiment ID、generation、checksum を読込時に再検証します。

`Progress(model_generation, stage, cumulative_cost, consumed_unit_ids=(), evaluations=())`
は確定済み消費 ID の順序付き tuple と原評価結果を保持します。stage は `collection`、
`learning`、`evaluation`。初回 checkpoint は消費・評価なし、入力と同じ generation。
学習 commit は直前 generation から 1 だけ進み、直前の消費 tuple に選択入力をその順序で
追加します。入力 generation は直前モデルの generation、再消費は拒否します。
同じ generation の commit は消費と model/optimizer/scheduler および module ごとの
training mode を変更できません。コストは非負有限値で単調増加。

`EvaluationResult(checkpoint_id, opponent, start_pair, seed, result_json)` の最初の四項が
再実行 identity。`result_json` は有限値の JSON 原結果です。新しい評価結果は同じ generation の確定した祖先 checkpoint を参照します。
途中の評価結果だけを別 commit で確定しても、評価対象 checkpoint の identity は固定できます。同じ identity の重複と既存結果の変更・削除を拒否します。
未完了評価は manifest に追加せず、再開時に未完了 identity だけを試行します。

## 保存される状態

`CheckpointManifest` は commit ID、parent ID、sequence、`experiment_id(config)`、
正規化設定 JSON、入力 manifest、Progress、全 artifact checksum を結合します。
モデルは Transformers の `save_pretrained(safe_serialization=True)`、追加状態の
tensor は `runtime.safetensors`、構造と非 tensor 値は型情報付き `runtime.json` に
保存します。両 runtime artifact をモデルの二 artifact とともに checksum で検証し、
不足・追加 artifact と未参照・重複 tensor を拒否します。pickle による保存・読込はありません。
model の全 named module の training mode（root と mixed train/eval mode を含む）、
optimizer、任意 scheduler、四つの `RNGStreams`、global Python/NumPy/Torch CPU RNG、
利用可能 CUDA 全 device と MPS RNG を含みます。device RNG の利用環境が変わった場合は
再開を拒否します。seed からの再生成ではなく状態を復元します。

## 形式と互換性

runtime JSON の形式 version は 1、checkpoint manifest の version は 2 です。
manifest version 1 と `runtime.pt` を使う形式は読込を拒否し、
自動変換や pickle fallback は行いません。experiment root の設定 binding は version 1 のままです。
runtime は None、bool、int、str、float（scheduler の inf/-inf/nan を含む）、
str/int key の dict、list、tuple、torch.device を扱います。tensor の dtype は bool、
符号付き/符号なし 8/16/32/64 bit 整数、float16/bfloat16/float32/float64 に限定します。
NumPy ndarray は native byte order の同じ数値/bool dtype（bfloat16 を除く）を扱います。
tuple と list、int key と str key は区別します。
tensor は独立した CPU contiguous clone として保存し、値・dtype・shape を復元します。
dict は key の型と値の順に正規化し、挿入順に依存せず同じ artifact を生成します。
list と tuple の要素順は保持します。
元の device、stride、共有 storage の identity は復元しません。疎 tensor、meta tensor、
量子化 tensor、任意 object、循環参照は拒否します。metadata の未知 tag、型・shape・dtype の
不一致も拒否し、tree の深さは 128、node 数は 100,000 までに制限します。

## 保存形式と確定境界

cooperating POSIX ローカル filesystem を前提とします。writer は flock 内で root の設定
binding と expected_parent を検証し、初回 binding も同じ lock 内で公開します。
一時 directory 内の全 artifact と manifest、directory を fsync した後、directory rename
で `committed/{sequence}-{commit_id}` へ公開し、公開先と staging directory を fsync。
この rename が可視性・確定の境界です。途中の optimizer 更新や staging の消費は確定扱いに
なりません。rename 後の durability 処理に失敗したとき、同じ内容・commit ID の retry は
確定データを再検証して必要な directory fsync を完了します。異なる内容の retry は拒否。
未使用 staging は不可視のまま残ります。

復元の手順、破損時の fallback、再試行時の扱いは [停止後の復旧と再試行](recovery.md) を参照してください。`recover()` は確定状態がまだない場合に `None` を返します。同じ commit ID・同じ内容の再試行は既存 manifest を返し、異なる内容なら拒否します。入力・消費・世代・設定・artifact の不整合は `ValueError` として報告します。ファイルへのアクセスや同期の失敗は `OSError` です。

artifact は信頼するローカル実験出力であり、第三者の checkpoint を受け入れる import API は
含みません。network filesystem、分散 writer、device 間移行、外部評価サービスの実行保証
は対象外です。原結果の確定単位として開始局面ペアを使います。
