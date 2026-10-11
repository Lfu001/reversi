# 自己対戦データの保存と読込

`DatasetStore` は、完了した独立対局と全葉が終局した木を保存します。一つの対局・木を「単位」と呼び、record と教師を一緒に公開します。reader に見える状態は、単位が存在しないか、完全な単位が存在するかのいずれかです。

## 保存する

以下の関数は、呼出側が作成した [CollectionRecord](records.md) と設定ファイルを受け取ります。record の `experiment_id` は設定から得た ID と一致する必要があります。

```python
from pathlib import Path

from lfu_tree_self_play.config import experiment_id, load_config
from lfu_tree_self_play.dataset_store import DatasetStore
from lfu_tree_self_play.game import ReversiPyGame
from lfu_tree_self_play.records import CollectionRecord
from lfu_tree_self_play.returns import aggregate_returns


def save_record(record: CollectionRecord, config_path: Path, root: Path):
    config = load_config(config_path)
    game = ReversiPyGame()
    store = DatasetStore(root, experiment_id=experiment_id(config), game=game)
    targets = aggregate_returns(record, game)
    identity = store.publish(record, targets)
    return store, identity
```

`publish` は record の整合性と、六つの target mapping が再計算した値に一致することを検証します。未完了の葉、不正なゲーム遷移、世代混在、非有限値を含む単位は保存できません。

保存先は最初の公開時に一つの experiment ID に固定されます。設定を変えた実験には別の保存先を使ってください。単位の ID は `[A-Za-z0-9][A-Za-z0-9_.-]{0,127}` に一致する必要があります。

## 読み込む・列挙する

`store` は上の関数で返されたストア、`identity` は公開済み単位の識別情報です。

```python
unit = store.load(identity.unit_id, model_generation=identity.model_generation)
record = unit.record
targets = unit.targets
identities = store.enumerate_units(model_generation=identity.model_generation)
```

`DatasetUnit` は `record`、`targets`、不変の `UnitIdentity(unit_id, experiment_id, model_generation, checksum)` を持ちます。`load` の generation 指定は、不一致を拒否するために使います。`enumerate_units` は指定した generation の identity を ID の辞書順で返します。指定を省略すると全世代が対象です。

読込・列挙では checksum と record・ゲーム・target の意味を再検証します。列挙は世代で絞る前に全公開単位を検証するため、対象外の世代に破損があっても失敗します。同時に公開された単位は次回の列挙で現れる場合があります。学習入力の固定には [CheckpointStore.pin_inputs](checkpoints.md) を使ってください。

## エラーと再試行

| 状況 | 扱い |
| --- | --- |
| 同じ ID・同じ正規化内容を再公開 | 元の identity を返す |
| 同じ ID・異なる内容を公開 | `ValueError`。既存データは上書きしない |
| experiment ID・指定 generation の不一致 | `ValueError`。保存先・入力設定を確認する |
| 不正な record・target・保存形式、checksum 不一致 | `ValueError` の派生を含む検証エラー。受理しない |
| 存在しない ID の読込 | `FileNotFoundError` |
| ファイルへのアクセス・同期の失敗 | `OSError`。原因を解消し、同じ ID・内容で再試行する |

round metadata の変更も異なる内容とみなします。公開中に停止した場合、同じ ID と同じ内容で再試行できます。未公開ファイルの隔離と再収集は [復旧ガイド](recovery.md) を参照してください。

## 保存形式

保存先は次の構成です。公開済みファイルは直接編集しないでください。

```text
experiment.json          実験 ID の固定
writer.lock              writer 間の排他
published/<unit_id>.json 公開済み単位
staging/                 公開前の一時ファイル
quarantine/              隔離した未公開ファイル
```

各 JSON は version 1 の envelope で、field は `payload` と `checksum` です。payload の field は `version`、`record`、`targets` です。checksum は次の正規化 bytes に対する小文字の SHA256 です。

```python
json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
```

UTF-8、既定の `ensure_ascii=True` を使います。record の配列順は保持します。局面は `(4, 8, 8)` の四つの plane として保存し、float32 の `GameState` へ損失なく復元します。tuple field も復元します。

ファイルでは未知 field・version、JSON キー重複、bool を数値として使うこと、不正な array/object を拒否します。checksum を再計算したファイルにも意味検証を行います。checksum は破損検出用で、改ざんに対する認証ではありません。

ファイル以外との受渡しには `lfu_tree_self_play.dataset_codec` の `encode_unit(record, targets, game) -> bytes` と `decode_unit(data, game) -> DatasetUnit` を使えます。encode は record・局面・target の独立した snapshot を作り、検証した内容を保存します。

## 保存保証の範囲

協調する POSIX writer が、一つのローカルファイルシステムで `flock`、atomic rename、`fsync` を利用することが前提です。writer は排他中に一時ファイルを書き、同期後に rename で公開します。成功を返すまでにファイルと関連 directory の同期を完了します。reader は writer lock を取りません。

network filesystem、外部からのファイル変更、storage の物理故障は対象外です。プロセス停止からの復旧保証を、ハードウェアの電源断耐性として解釈しないでください。古いモデル世代のデータを学習に使う判断は呼出側が行います。
