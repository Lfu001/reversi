# Checkpoint・消費位置の整合保存

T09 の API は `lfu_tree_self_play.checkpoints`。実データを trainer へ接続する機能は含みません。

```python
store = CheckpointStore(root, config=config, dataset_store=dataset)
inputs = store.pin_inputs(model_generation, start=0, stop=None)
manifest = store.commit(
    "update-1", model=model, optimizer=optimizer, scheduler=scheduler,
    rng=streams, progress=progress, inputs=inputs, expected_parent="genesis",
)
verified = store.recover()
restored = store.restore(
    verified,
    optimizer_factory=lambda model: Adam(model.parameters()),
    scheduler_factory=lambda optimizer: StepLR(optimizer, step_size=1),
)
```

`InputManifest(model_generation, units, start, stop)` は generation 内のソート済み
`UnitIdentity` 全体と half-open 範囲を固定します。`selected` はこの固定 tuple の slice。
後から辞書順で前に入る unit が公開されても入力範囲は移動しません。各 unit の ID、
experiment ID、generation、checksum を読込時に再検証します。

`Progress(model_generation, stage, cumulative_cost, consumed_unit_ids=(), evaluations=())`
は確定済み消費 ID の順序付き tuple と原評価結果を保持します。stage は `collection`、
`learning`、`evaluation`。初回 checkpoint は消費・評価なし、入力と同じ generation。
学習 commit は直前 generation から 1 だけ進み、直前の消費 tuple に選択入力をその順序で
追加します。入力 generation は直前モデルの generation、再消費は拒否します。
同じ generation の commit は消費と model/optimizer/scheduler を変更できません。コストは非負有限値で単調増加。

`EvaluationResult(checkpoint_id, opponent, start_pair, seed, result_json)` の最初の四項が
再実行 identity。`result_json` は有限値の JSON 原結果です。新しい評価結果は同じ generation の確定した祖先 checkpoint を参照します。
途中の評価結果だけを別 commit で確定しても、評価対象 checkpoint の identity は固定できます。同じ identity の重複と既存結果の変更・削除を拒否します。
未完了評価は manifest に追加せず、再開時に未完了 identity だけを試行します。

`CheckpointManifest` は commit ID、parent ID、sequence、T02 の `experiment_id(config)`、
正規化設定 JSON、入力 manifest、Progress、全 artifact checksum を結合します。
モデルは Transformers の `save_pretrained`、追加状態は `runtime.pt` に保存します。
optimizer、任意 scheduler、四つの `RNGStreams`、global Python/NumPy/Torch CPU RNG、
利用可能 CUDA 全 device と MPS RNG を含みます。device RNG の利用環境が変わった場合は
再開を拒否します。seed からの再生成ではなく状態を復元します。

## 確定と再開

cooperating POSIX ローカル filesystem を前提とします。writer は flock 内で root の設定
binding と expected_parent を検証し、初回 binding も同じ lock 内で公開します。
一時 directory 内の全 artifact と manifest、directory を fsync した後、directory rename
で `committed/{sequence}-{commit_id}` へ公開し、公開先と staging directory を fsync。
この rename が可視性・確定の境界です。途中の optimizer 更新や staging の消費は確定扱いに
なりません。rename 後の durability 処理に失敗したとき、同じ内容・commit ID の retry は
確定データを再検証して必要な directory fsync を完了します。異なる内容の retry は拒否。
未使用 staging は不可視のまま残ります。

`recover()` は確定履歴を順に検証し、checksum、入力、親・generation・消費関係を含む
最新の検証済み snapshot を返します。不完全・破損候補は直前 snapshot に戻ります。
設定不一致は fallback で隠さず実行を拒否します。有効な確定状態が一つもない場合も拒否します。確定履歴は削除しません。
`restore()` は最新の検証済み snapshot に対し新しい model/optimizer/scheduler/RNG を作り、
全状態の読込に成功した後に global RNG を復元します。factory 失敗時は global RNG を
元に戻し、呼出側の既存 model/optimizer を変更しません。

artifact は信頼するローカル実験出力です。`torch.load(weights_only=False)` による
Python/NumPy RNG の読込を含むため、第三者の checkpoint を受け入れる import API では
ありません。network filesystem、分散 writer、device 間移行、外部評価サービスの実行保証
は対象外です。原結果の確定単位として開始局面ペアを使います。

## 検証

`tests/test_checkpoints.py` は populated optimizer/scheduler/model と次の RNG 出力、
入力範囲と後続公開、設定・generation・checksum・消費不一致、replay/conflict、
rename 前停止と rename 後 fsync retry、破損 fallback、評価結果保存、factory 失敗時の
rollback を検証します。これは T09 の成果であり G1 全体の完了ではありません。
