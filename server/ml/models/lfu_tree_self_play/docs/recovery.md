# 停止後の復旧と再試行

dataset の公開と checkpoint の確定には、それぞれ atomic rename を使います。停止前に計算を終えていても、rename 前ならその結果は未確定です。再開時は公開済みデータと検証済み checkpoint を基準に処理を組み立てます。

## 未公開データを隔離する

[DatasetStore](dataset-store.md) の保存先を開き、`dataset.quarantine_incomplete()` を呼びます。writer lock 内で `staging/` の残存 entry を `quarantine/` へ移動し、移動先の path の tuple を返します。

隔離したファイルは公開データへ昇格しません。未完了の対局・木を再収集し、同じ unit ID で公開してください。既に公開済みなら、同じ正規化内容の再公開は成功し、異なる内容の再公開は拒否されます。公開済みデータの破損や設定不一致は、この隔離処理では解消しません。

## モデル状態を復元する

`store` は、保存時と同じ設定と dataset を渡して開いた [CheckpointStore](checkpoints.md) です。optimizer と scheduler の factory には、保存時と同じ種類・構成のものを指定します。

```python
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR

verified = store.recover()
if verified is None:
    raise RuntimeError("確定済み checkpoint がありません。初期状態を保存してください。")

runtime = store.restore(
    verified,
    optimizer_factory=lambda model: Adam(model.parameters()),
    scheduler_factory=lambda optimizer: StepLR(optimizer, step_size=1),
)
model = runtime.model
optimizer = runtime.optimizer
scheduler = runtime.scheduler
streams = runtime.rng
progress = runtime.manifest.progress
inputs = runtime.manifest.inputs
```

scheduler を保存していない場合は `scheduler_factory=lambda optimizer: None` を渡します。保存している場合は対応する scheduler を作る factory が必要です。モデルの各 module の train/eval mode も復元されるため、再開直後に一律の `model.train()` を呼ぶと混在 mode が変わります。

`restore` は新しい model・optimizer・scheduler・用途別 RNG を作成します。全状態の読込に成功した後、global Python/NumPy/Torch RNG を復元します。factory 失敗時は global RNG を元に戻し、呼出側の既存 model・optimizer を変更しません。

## 再開時に処理する範囲

| 停止した処理 | 再開時の扱い |
| --- | --- |
| 収集中・dataset 公開前 | 未公開ファイルを隔離し、単位を再収集する |
| dataset 公開後 | 公開済みの単位を読む。同じ ID・内容の再公開も可能 |
| 学習計算後・checkpoint 確定前 | 直前の確定モデルを復元し、未消費の入力から更新をやり直す |
| checkpoint 確定後 | 更新済みモデルと消費位置を引き継ぐ |
| 評価計算後・評価結果の確定前 | 未確定の評価だけをやり直す |
| 評価結果の確定後 | 保存済みの原結果を使う |

入力は checkpoint の `InputManifest`、確定済み消費 ID は `Progress.consumed_unit_ids` を参照します。公開順や directory の再列挙だけから消費位置を推測しないでください。

評価は `(checkpoint_id, opponent, start_pair, seed)` で識別します。`Progress.evaluations` に記録済みの identity は再確定せず、原結果を保持します。未確定の計算は再実行される場合があります。保存 API は外部評価サービスの呼出し自体を一度だけにする保証は提供しません。

## 破損・互換性エラー

`recover()` は履歴の checksum、固定入力、親関係、世代、消費位置を検証し、最新の検証済み snapshot を返します。不完全・破損候補を採用できない場合は、検証済みの前の snapshot を使います。復旧した commit ID と進捗を確認してから、未確定の処理を再開してください。確定履歴は自動削除されません。

- 履歴が空なら `None` を返します。
- 履歴があっても有効な状態がなければ `ValueError` で失敗します。
- 設定不一致は古い snapshot への fallback で隠さず拒否します。元の設定を用意するか、別の保存先を使ってください。
- CUDA device 数や MPS の利用可否が保存時と異なる場合は、乱数状態を復元できないため拒否します。
- manifest version 2 と runtime JSON version 1 が対象です。`runtime.pt` を使う旧形式の自動変換はありません。

復旧保証は [dataset](dataset-store.md#保存保証の範囲) と [checkpoint](checkpoints.md#保存形式と確定境界) のローカルファイルシステム契約を前提とします。
