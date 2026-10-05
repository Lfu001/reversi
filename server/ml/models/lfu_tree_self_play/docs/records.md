# 収集 record と共通検証器

T06 は、A の完了した独立対局と B/C の全枝が完了した木を同じ意味モデルで表します。収集方式は `independent` / `tree` で、B/C の学習方式は record に持ち込みません。[T06 正本](milestones/01-data-storage.md#t06-木ノード辺兄弟群コストのrecord-schema) と [承認済み設計](../../../../../docs/superpowers/specs/2026-10-04-self-play-record-schema-design.md) が契約の根拠です。

## 型と単位

型は [records.py](../src/lfu_tree_self_play/records.py) にあります。`schema_version` は現在 `1` のみです。ID は空白だけではない非空文字列で、node / edge / sibling group / RNG stream の各集合内で一意です。`unit_id` は experiment / round の文脈で呼出側が発行し、dataset 全体の重複防止や時刻順は T06 では定義しません。

| 型 | 情報と意味 |
| --- | --- |
| `CollectionRecord` | unit / experiment / round ID、固定 `model_generation`、収集方式、root ID、分岐予定、乱数、nodes / edges / sibling groups、コスト |
| `NodeRecord` | node ID、根のみ `None` の incoming edge ID、既存 `GameState`、根からの `placement_depth`、モデル世代、非終局の `old_value`、終局の `black_result` |
| `EdgeRecord` | edge ID、親子 node ID、`action`（0〜63 の盤面 index）、`old_log_probability`、モデル世代、sampling、兄弟群 ID と sample index（通常辺は両方 `None`） |
| `SiblingGroupRecord` | group ID、親 node ID、順序付き `edge_ids`、実際に異なる action 数 `distinct_action_count` |
| `BranchPlan` | `width >= 2`、昇順かつ重複なしの予定着手位置 `positions` |
| `RNGRecord` | `algorithm="python.random.MT19937-v1"`、非負 `run_seed`、用途別 stream tuple |
| `RNGStreamRecord` | stream ID、用途 `branch_position` / `action_sampling` / `sibling_rollout`、非負 seed |
| `SamplingRecord` | stream ID、stream 内の非負 `draw_index`、分岐標本の `continuation_seed`（通常辺は `None`） |
| `CostRecord` | 下表の収集時に計測したコスト |

`old_value` はそのノードの着手者 (`state.to_play`) 視点で `[-1, 1]` です。辺の `old_log_probability` は親局面で合法手マスク・温度を適用して抽出した時点の有限値で `<= 0`。初期実験は温度 1 を使います。終局結果は黒視点の整数で、黒勝ち `+1`、引き分け `0`、黒負け `-1` です。終局ノードでは `old_value=None`、非終局では `black_result=None` とします。

| コスト field | 単位 |
| --- | --- |
| `environment_transitions` | 実際の環境遷移回数 |
| `evaluated_positions` | モデルで評価した局面数 |
| `inference_batches` | 推論 batch 数 |
| `collection_seconds` | 収集経過秒 |
| `inference_seconds` | 推論秒 |
| `transition_seconds` | 環境遷移秒 |
| `management_seconds` | 木管理秒 |
| `peak_memory_bytes` | ピークメモリ bytes |

計数と bytes は非負整数、時間は有限の非負実数です。共有・キャッシュ・再計算の方式によって実測値は変わるため、辺数・ノード数との等値や各時間の和は要求しません。コストは collector が計測する metadata で、検証器は計測の正しさを再構成しません。保存時間・学習時間はこの収集コストに含めません。

## 局面・分岐・標本

手番・合法手・終局の正本は既存 `GameState` です。観測は黒石・白石・手番・合法手の 4 plane。`placement_depth` は根で 0、辺を進むごとに 1 増えます。強制パスは `Game.step` が解決するため、パス辺やパスの着手数は追加しません。親子で同じ手番になることがあり、深さの偶奇から value の符号を推測できません。

予定位置は根から数えた着手前の深さで、根の空きマス数を H として `0 <= position < H` です。`tree` はその位置に達した非終局ノードで幅 K の兄弟群を持ち、それ以外の非終局は出辺 1 本です。早期終局した経路には未到達の分岐を要求しません。幅や段数を 4 / 2 に固定せず、分岐なし・1 段・2 段などを表せます。`independent` は全非終局の出辺が 1 本、群なし、予定位置は空 tuple です。H=0 の終局根はノード 1 個・辺 0 本で表せます。

同じ action を抽出した兄弟は、直後の局面が等しくても別の edge ID / child node ID / sample index / continuation seed を持ちます。統合せず、各標本の終局結果を保持します。群の `edge_ids` の順序は `sample_index=0..K-1` と対応し、親の全出辺をちょうど覆います。各非根ノードの親は 1 個で、全ノードが根から到達可能、全葉が終局です。

主要 RNG stream ID は `branch` / `action` / `sibling` で、用途と seed を既存 `derive_seed(run_seed, purpose)` と照合します。追加 stream は `sibling_rollout` 用で、兄弟標本の continuation seed と一致する seed を持ちます。同一群の continuation seed は重複できません。辺は `action_sampling` または `sibling_rollout` の stream を参照し、同じ `(stream_id, draw_index)` を複数の行動抽出に使えません。位置抽選 stream の行動抽出への流用も拒否します。

## API とエラー境界

```python
from lfu_tree_self_play.game import ReversiPyGame
from lfu_tree_self_play.records import CollectionRecord
from lfu_tree_self_play.record_validation import RecordValidationError, validate_record


def accept_complete_unit(record: CollectionRecord) -> None:
    validate_record(record, ReversiPyGame())
```

[validate_record](../src/lfu_tree_self_play/record_validation.py) は成功時 `None` を返し、不正な完了単位は `RecordValidationError(ValueError)` で拒否します。呼出側はこの例外を捕捉して単位を受理しない判断ができます。個々のモデルの構築時に型・範囲に違反すると Pydantic の `ValidationError` になります。検証器の入口では `CollectionRecord.model_validate(record)` によりネストされた既存インスタンスも再検証し、Pydantic エラーを `RecordValidationError` に包みます。`model_copy(update=...)` は更新値を検証しないので、コピー後もこの入口を通します。

全型は `strict=True`、`frozen=True`、`extra="forbid"`、`allow_inf_nan=False`、`validate_default=True`、`revalidate_instances="always"` を共有します。整数 field に bool や数値文字列は使えません。実数 field は Python の int / float を受け付けますが、bool・文字列・NaN / Inf は拒否します。未知 field、list から tuple への暗黙変換も拒否し、nodes / edges / sibling groups / streams / positions / edge IDs は tuple です。

`frozen` は属性の再代入を防ぐもので、既存 `GameState` の内部 NumPy 配列まで深い不変性を保証しません。`state` は `InstanceOf[GameState]` として保持し、配列変換や dataclass の再構築をしません。局面の公開操作にはコピーを返す既存 `GameState.observation` を使います。検証器は入力を補正・並び替え・書換えず、型のエラー詳細または最初の意味違反を報告します。

検証器は ID と親子関係、観測の shape / 有限性 / binary plane / 石の重複 / 占有マス上の合法手 / 均一な手番 plane、全辺の合法 action と `game.step` による子局面の完全一致、深さ、世代、終局結果、分岐構成、乱数参照を確認します。

## 検証の限界と後続タスク

根の合法手 mask 自体の規則再計算は既存 `Game` インターフェースではできません。根は規則に従ってパスを解決した局面を生成する検証済み game adapter から取得する、という収集側の契約です。観測の表現チェックと全保存遷移の replay は行いますが、根 mask の規則上の正しさは adapter を信頼します。

乱数参照と seed の検証は追跡契約で、PRNG の抽出値 replay や統計的独立性の証明ではありません。途中再開の完全な RNG state は T09 の責任です。検証器は当時のモデルを実行しないため、旧 value / log probability が当時のモデル出力だったことも証明しません。

`InstanceOf[GameState]` を含む本 API は JSON 永続化を提供しません。保存形式・round-trip・atomic store は T08、終局リターン・経路重みは T07、checkpoint は T09、停止・再開は T10 に残します。collector / trainer は追加していません。T06 の検証は G1 の完了を意味せず、G1 通過前に保存データを trainer へ接続しません。

## T06 の検証結果

2026-10-05 に package で以下を実行しました。実装前 baseline は 84 tests でした。T06 の型 30 cases と共通検証器 94 cases を追加し、pytest の失敗はありません。変更していない重い G0 Monte Carlo 実験は再実行していません。

| 検証 | 結果 |
| --- | --- |
| `uv sync --locked` | 成功（65 packages resolved、45 packages checked） |
| `uv run --locked pytest -q` | 208 passed in 63.94s（import 順序修正後） |
| Ruff 0.16.8 `check --target-version py312 src tests` | All checks passed! |
| Ruff 0.16.8 `format --check --target-version py312 src tests` | 22 files already formatted |
| repository `git diff --check` | 成功 |

[型の tests](../tests/test_record_types.py) は strict / frozen / tuple / 既存局面の保持を確認します。[共通検証器の tests](../tests/test_records.py) は独立対局と木、同一 action の兄弟、幅 2/3/4、終局根、H=1、強制パス、早期終局、子孫数の違い、非破壊検証と実測コストを受理します。同じ suite で親子関係・世代混在・局面改変・未完成葉・結果不一致・分岐と兄弟群の不整合・標本統合・RNG 参照や抽出イベントの重複・未検証コピーの型違反を拒否します。
