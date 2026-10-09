# 完了単位を原子的に保存する dataset store（T08）

`DatasetStore` は、完了した独立対局と全葉が終局した木を永続化します。
保存データを trainer へ接続する機能は含みません。G1 は別のゲートとして残します。

```python
store = DatasetStore(root, experiment_id=experiment_id(config), game=game)
identity = store.publish(record, aggregate_returns(record, game))
unit = store.load(record.unit_id, model_generation=record.model_generation)
identities = store.enumerate_units(model_generation=record.model_generation)
quarantined_paths = store.quarantine_incomplete()
```

store は `lfu_tree_self_play.dataset_store` から、明示的な codec API
`encode_unit(record, targets, game) -> bytes` / `decode_unit(data, game) -> DatasetUnit`
は `lfu_tree_self_play.dataset_codec` から import します。`DatasetUnit` は `record`、
`targets`、不変の `UnitIdentity(unit_id, experiment_id, model_generation,
checksum)` を保持します。T09 は自身の入力 manifest でこれらの identity を固定します。
この store には独立した消費位置や、変更可能な identity index はありません。

設定用の別の hash は作らず、既存の T02 `experiment_id(config)` を渡します。
初回公開時に writer lock 内で `experiment.json` を原子的に固定し、以後の追加でも
書き込み前に root の不変な設定 binding を再確認します。異なる設定の二つの初回 writer が
両方とも追加することはできません。binding のない root に既存 unit がある場合は、
binding の作成前にそれらを検証します。binding は設定 metadata であり、追加の identity や
消費 index ではありません。公開・読込・列挙・隔離は、binding と異なる experiment ID を拒否します。
generation は record の不変な metadata です。`load(..., model_generation=N)` は
不一致を拒否し、この引数付きの列挙は検証済みの過去 unit を generation N に絞ります。
絞り込み前に公開済みの全エントリーを検証し、破損があれば処理を拒否します。
列挙は ID の辞書順に並んだ snapshot を返します。同時に公開された unit は次回の呼出しで
現れることがあります。呼出側の manifest は返された identity を固定する必要があり、
directory の列挙自体を入力 transaction とみなすことはできません。

## 保存形式と完全性の契約

各 `published/<unit_id>.json` ファイルは version 1 の envelope 一つで、field は
`payload` と `checksum` のみです。payload の field は `version`、`record`、`targets` のみです。
checksum は `json.dumps(payload, sort_keys=True,
separators=(",", ":"), allow_nan=False).encode()` が生成する bytes に対する小文字の SHA256 です
（UTF-8、`ensure_ascii=True` は既定値）。envelope の checksum 自体はこの bytes に含めません。
配列は record の順序を維持し、target 辞書のキーは JSON の正規化でソートします。

局面は明示的な 4 plane の `(4, 8, 8)` 数値配列で保存し、float32 の `GameState`
インスタンスへ損失なく復元します。record の tuple field も明示的に再構築します。
未知の field・version、JSON キーの重複、数値 field の bool、非有限値、不正なゲーム遷移、
未完了の葉、世代混在、`aggregate_returns` と一致しない target は拒否します。
明示的な再構築前に、各 target mapping は JSON object、各 tuple field は JSON array
である必要があります。キーと値の組を並べた配列や、任意の iterable からの型変換は拒否します。
checksum は破損を検出します。破損した payload の checksum が再計算されていても、
意味検証を行います。これは破損検出であり、ファイルを書き換える攻撃者に対する認証ではありません。

encode では、渡された六つの target mapping をすべて通常の辞書へコピーします。
キーは Python の組込み文字列型そのもの、値は bool を除く有限な Python の int / float
そのものであることを要求し、各辞書を新たに検証・集約したリターンと比較します。
最初に基本の record schema に従う snapshot を作り、コピーした数値観測から
`GameState` 型そのもののインスタンスを再構築します。不正な plane や、損失を伴う
float32 変換は拒否します。集約とシリアライズは同じ snapshot を使うため、独自の record
serializer や state property が永続化されるゲームの意味を変えることはできません。
公開時の bytes と identity は、この同じ検証済み payload から取得します。
新たに生成した bytes を公開処理内で decode し直しません。既存ファイルの読込・列挙・replay では、
引き続き checksum、record、ゲーム、target の全検証を行います。

unit ID は `[A-Za-z0-9][A-Za-z0-9_.-]{0,127}` に一致する必要があります。
同じ ID を同じ正規化 payload で繰り返すと、元の identity を返します。
payload が異なる場合は replay conflict となり、元のデータを上書きしません。
round metadata の変更も conflict に含みます。既存データが破損している場合は、
置き換えずに失敗として拒否します。

## 公開と復旧

協調する writer は `writer.lock` に対して `fcntl.flock` を使います。
検証は staging より先に行います。writer は一意な staging ファイルを作り、
書き込み・flush・fsync を行い、staging directory を fsync した後、published へ原子的に
rename し、両方の directory を fsync します。同一内容の replay でも、成功を返す前に
published と staging directory を fsync し、rename 後に中断された永続性の処理を完了します。
新しい root を作成する場合も、新たに作成した各 directory entry を親 directory の sync で
確定します。reader は公開済みファイルだけを使い、writer lock を取得しません。
データより先に metadata が可視になるような、複数ファイルにまたがる unit はありません。

rename 前にプロセスが停止すると、不可視の staging ファイルが残ります。
明示的な `quarantine_incomplete()` は writer lock を保持し、放置された staging entry を
一意な quarantine path へ移動して directory を sync します。これらを公開済みデータへ
昇格することはありません。収集を再実行すると、同じ ID を公開できます。
rename 後にプロセスが停止した場合は、完全な unit が可視のまま残り、replay は成功します。
隔離処理は、公開済みデータの破損や互換性のない設定を、暗黙に dataset の外へ移動しません。

この保証は、flock・atomic rename・fsync が機能する一つのローカル filesystem 上で
協調する POSIX writer に適用します。network filesystem、悪意ある外部のファイル変更、
物理的に故障した storage は契約の対象外です。公開が成功した時点で fsync は完了していますが、
プロセス停止の tests はハードウェアの電源断時の動作を保証するものではありません。
過去の世代が特定の学習更新に適しているかどうかを判断する方針も含みません。

## 検証

`PYTHONPATH=src python -m pytest tests/test_dataset_store.py -q` は、厳密な round-trip
（不均等な木、重複標本、H=0/H=1、独立対局の強制パス）、checksum と checksum を再計算した
意味上の破損、replay/conflict、設定・世代の不一致拒否、安全でない ID、放置された staging の
隔離、rename の直前・直後での子プロセス停止を検証します。子 writer が一時停止している間も
writer は flock を保持し、親は繰り返し読込を行います。親から見えるのは、unit が存在しない状態か、
全検証を通過した完全な unit です。
